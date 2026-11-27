/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PublicBatchWrapper89`, split out of the generated module
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

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem publicBatchWrapper89_f8237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6501)) := by
  have c53928 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.1
  have c53929 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.1
  have c53930 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53931 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53932 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53933 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_12
  simp only [← c53928, ← c53929, ← c53930] at e_865_12
  have e_865_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_13
  simp only [← c53931, ← c53932, k2, ← c53933] at e_865_13
  have hr := e_865_13
  simp only [e_865_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6502)) := by
  have c53934 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53935 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53936 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53937 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53938 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53939 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_14
  simp only [← c53934, ← c53935, ← c53936] at e_865_14
  have e_865_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_15
  simp only [← c53937, ← c53938, k2, ← c53939] at e_865_15
  have hr := e_865_15
  simp only [e_865_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6503)) := by
  have c53940 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53941 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53942 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53943 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53944 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53945 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_16
  simp only [← c53940, ← c53941, ← c53942] at e_865_16
  have e_865_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_17
  simp only [← c53943, ← c53944, k2, ← c53945] at e_865_17
  have hr := e_865_17
  simp only [e_865_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 865 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6504)) := by
  have c53946 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53947 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53948 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53949 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53950 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53951 := (publicBatchWrapper89_copies1685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_865_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_18
  simp only [← c53946, ← c53947, ← c53948] at e_865_18
  have e_865_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_865_19
  simp only [← c53949, ← c53950, k2, ← c53951] at e_865_19
  have hr := e_865_19
  simp only [e_865_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6505)) := by
  have c53952 := (publicBatchWrapper89_copies1686 a h).1
  have c53953 := (publicBatchWrapper89_copies1686 a h).2.1
  have c53954 := (publicBatchWrapper89_copies1686 a h).2.2.1
  have c53955 := (publicBatchWrapper89_copies1686 a h).2.2.2.1
  have c53956 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.1
  have c53957 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_0
  simp only [← c53952, ← c53953, ← c53954] at e_866_0
  have e_866_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_1
  simp only [← c53955, ← c53956, k2, ← c53957] at e_866_1
  have hr := e_866_1
  simp only [e_866_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6506)) := by
  have c53958 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.1
  have c53959 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.1
  have c53960 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.1
  have c53961 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.1
  have c53962 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53963 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_2
  simp only [← c53958, ← c53959, ← c53960] at e_866_2
  have e_866_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_3
  simp only [← c53961, ← c53962, k2, ← c53963] at e_866_3
  have hr := e_866_3
  simp only [e_866_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6507)) := by
  have c53964 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53965 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53966 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53967 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53968 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53969 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_4
  simp only [← c53964, ← c53965, ← c53966] at e_866_4
  have e_866_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_5
  simp only [← c53967, ← c53968, k2, ← c53969] at e_866_5
  have hr := e_866_5
  simp only [e_866_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6508)) := by
  have c53970 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53971 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53972 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53973 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53974 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53975 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_6
  simp only [← c53970, ← c53971, ← c53972] at e_866_6
  have e_866_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_7
  simp only [← c53973, ← c53974, k2, ← c53975] at e_866_7
  have hr := e_866_7
  simp only [e_866_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6509)) := by
  have c53976 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53977 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53978 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53979 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53980 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53981 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_8
  simp only [← c53976, ← c53977, ← c53978] at e_866_8
  have e_866_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_9
  simp only [← c53979, ← c53980, k2, ← c53981] at e_866_9
  have hr := e_866_9
  simp only [e_866_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6510)) := by
  have c53982 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53983 := (publicBatchWrapper89_copies1686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c53984 := (publicBatchWrapper89_copies1687 a h).1
  have c53985 := (publicBatchWrapper89_copies1687 a h).2.1
  have c53986 := (publicBatchWrapper89_copies1687 a h).2.2.1
  have c53987 := (publicBatchWrapper89_copies1687 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_10
  simp only [← c53982, ← c53983, ← c53984] at e_866_10
  have e_866_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_11
  simp only [← c53985, ← c53986, k2, ← c53987] at e_866_11
  have hr := e_866_11
  simp only [e_866_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6511)) := by
  have c53988 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.1
  have c53989 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.1
  have c53990 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.1
  have c53991 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.1
  have c53992 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.1
  have c53993 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_12
  simp only [← c53988, ← c53989, ← c53990] at e_866_12
  have e_866_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_13
  simp only [← c53991, ← c53992, k2, ← c53993] at e_866_13
  have hr := e_866_13
  simp only [e_866_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6512)) := by
  have c53994 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.1
  have c53995 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c53996 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53997 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53998 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53999 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_14
  simp only [← c53994, ← c53995, ← c53996] at e_866_14
  have e_866_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_15
  simp only [← c53997, ← c53998, k2, ← c53999] at e_866_15
  have hr := e_866_15
  simp only [e_866_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6513)) := by
  have c54000 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54001 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54002 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54003 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54004 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54005 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_16
  simp only [← c54000, ← c54001, ← c54002] at e_866_16
  have e_866_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_17
  simp only [← c54003, ← c54004, k2, ← c54005] at e_866_17
  have hr := e_866_17
  simp only [e_866_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 866 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6514)) := by
  have c54006 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54007 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54008 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54009 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54010 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54011 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_866_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_18
  simp only [← c54006, ← c54007, ← c54008] at e_866_18
  have e_866_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_866_19
  simp only [← c54009, ← c54010, k2, ← c54011] at e_866_19
  have hr := e_866_19
  simp only [e_866_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6515)) := by
  have c54012 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54013 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54014 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54015 := (publicBatchWrapper89_copies1687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54016 := (publicBatchWrapper89_copies1688 a h).1
  have c54017 := (publicBatchWrapper89_copies1688 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_0
  simp only [← c54012, ← c54013, ← c54014] at e_867_0
  have e_867_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_1
  simp only [← c54015, ← c54016, k2, ← c54017] at e_867_1
  have hr := e_867_1
  simp only [e_867_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6516)) := by
  have c54018 := (publicBatchWrapper89_copies1688 a h).2.2.1
  have c54019 := (publicBatchWrapper89_copies1688 a h).2.2.2.1
  have c54020 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.1
  have c54021 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.1
  have c54022 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.1
  have c54023 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_2
  simp only [← c54018, ← c54019, ← c54020] at e_867_2
  have e_867_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_3
  simp only [← c54021, ← c54022, k2, ← c54023] at e_867_3
  have hr := e_867_3
  simp only [e_867_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6517)) := by
  have c54024 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.1
  have c54025 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.1
  have c54026 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54027 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54028 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54029 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_4
  simp only [← c54024, ← c54025, ← c54026] at e_867_4
  have e_867_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_5
  simp only [← c54027, ← c54028, k2, ← c54029] at e_867_5
  have hr := e_867_5
  simp only [e_867_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6518)) := by
  have c54030 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54031 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54032 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54033 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54034 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54035 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_6
  simp only [← c54030, ← c54031, ← c54032] at e_867_6
  have e_867_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_7
  simp only [← c54033, ← c54034, k2, ← c54035] at e_867_7
  have hr := e_867_7
  simp only [e_867_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6519)) := by
  have c54036 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54037 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54038 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54039 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54040 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54041 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_8
  simp only [← c54036, ← c54037, ← c54038] at e_867_8
  have e_867_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_9
  simp only [← c54039, ← c54040, k2, ← c54041] at e_867_9
  have hr := e_867_9
  simp only [e_867_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6520)) := by
  have c54042 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54043 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54044 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54045 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54046 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54047 := (publicBatchWrapper89_copies1688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_10
  simp only [← c54042, ← c54043, ← c54044] at e_867_10
  have e_867_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_11
  simp only [← c54045, ← c54046, k2, ← c54047] at e_867_11
  have hr := e_867_11
  simp only [e_867_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6521)) := by
  have c54048 := (publicBatchWrapper89_copies1689 a h).1
  have c54049 := (publicBatchWrapper89_copies1689 a h).2.1
  have c54050 := (publicBatchWrapper89_copies1689 a h).2.2.1
  have c54051 := (publicBatchWrapper89_copies1689 a h).2.2.2.1
  have c54052 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.1
  have c54053 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_12
  simp only [← c54048, ← c54049, ← c54050] at e_867_12
  have e_867_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_13
  simp only [← c54051, ← c54052, k2, ← c54053] at e_867_13
  have hr := e_867_13
  simp only [e_867_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6522)) := by
  have c54054 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.1
  have c54055 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.1
  have c54056 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.1
  have c54057 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.1
  have c54058 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54059 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_14
  simp only [← c54054, ← c54055, ← c54056] at e_867_14
  have e_867_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_15
  simp only [← c54057, ← c54058, k2, ← c54059] at e_867_15
  have hr := e_867_15
  simp only [e_867_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6523)) := by
  have c54060 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54061 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54062 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54063 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54064 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54065 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_16
  simp only [← c54060, ← c54061, ← c54062] at e_867_16
  have e_867_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_17
  simp only [← c54063, ← c54064, k2, ← c54065] at e_867_17
  have hr := e_867_17
  simp only [e_867_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 867 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6524)) := by
  have c54066 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54067 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54068 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54069 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54070 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54071 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_867_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_18
  simp only [← c54066, ← c54067, ← c54068] at e_867_18
  have e_867_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_867_19
  simp only [← c54069, ← c54070, k2, ← c54071] at e_867_19
  have hr := e_867_19
  simp only [e_867_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6525)) := by
  have c54072 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54073 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54074 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54075 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54076 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54077 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_0
  simp only [← c54072, ← c54073, ← c54074] at e_868_0
  have e_868_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_1
  simp only [← c54075, ← c54076, k2, ← c54077] at e_868_1
  have hr := e_868_1
  simp only [e_868_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6526)) := by
  have c54078 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54079 := (publicBatchWrapper89_copies1689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54080 := (publicBatchWrapper89_copies1690 a h).1
  have c54081 := (publicBatchWrapper89_copies1690 a h).2.1
  have c54082 := (publicBatchWrapper89_copies1690 a h).2.2.1
  have c54083 := (publicBatchWrapper89_copies1690 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_2
  simp only [← c54078, ← c54079, ← c54080] at e_868_2
  have e_868_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_3
  simp only [← c54081, ← c54082, k2, ← c54083] at e_868_3
  have hr := e_868_3
  simp only [e_868_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6527)) := by
  have c54084 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.1
  have c54085 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.1
  have c54086 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.1
  have c54087 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.1
  have c54088 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.1
  have c54089 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_4
  simp only [← c54084, ← c54085, ← c54086] at e_868_4
  have e_868_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_5
  simp only [← c54087, ← c54088, k2, ← c54089] at e_868_5
  have hr := e_868_5
  simp only [e_868_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6528)) := by
  have c54090 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54091 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54092 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54093 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54094 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54095 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_6
  simp only [← c54090, ← c54091, ← c54092] at e_868_6
  have e_868_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_7
  simp only [← c54093, ← c54094, k2, ← c54095] at e_868_7
  have hr := e_868_7
  simp only [e_868_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6529)) := by
  have c54096 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54097 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54098 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54099 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54100 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54101 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_8
  simp only [← c54096, ← c54097, ← c54098] at e_868_8
  have e_868_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_9
  simp only [← c54099, ← c54100, k2, ← c54101] at e_868_9
  have hr := e_868_9
  simp only [e_868_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6530)) := by
  have c54102 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54103 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54104 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54105 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54106 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54107 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_10
  simp only [← c54102, ← c54103, ← c54104] at e_868_10
  have e_868_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_11
  simp only [← c54105, ← c54106, k2, ← c54107] at e_868_11
  have hr := e_868_11
  simp only [e_868_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6531)) := by
  have c54108 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54109 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54110 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54111 := (publicBatchWrapper89_copies1690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54112 := (publicBatchWrapper89_copies1691 a h).1
  have c54113 := (publicBatchWrapper89_copies1691 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_12
  simp only [← c54108, ← c54109, ← c54110] at e_868_12
  have e_868_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_13
  simp only [← c54111, ← c54112, k2, ← c54113] at e_868_13
  have hr := e_868_13
  simp only [e_868_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6532)) := by
  have c54114 := (publicBatchWrapper89_copies1691 a h).2.2.1
  have c54115 := (publicBatchWrapper89_copies1691 a h).2.2.2.1
  have c54116 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.1
  have c54117 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.1
  have c54118 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.1
  have c54119 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_14
  simp only [← c54114, ← c54115, ← c54116] at e_868_14
  have e_868_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_15
  simp only [← c54117, ← c54118, k2, ← c54119] at e_868_15
  have hr := e_868_15
  simp only [e_868_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6533)) := by
  have c54120 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.1
  have c54121 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.1
  have c54122 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54123 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54124 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54125 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_16
  simp only [← c54120, ← c54121, ← c54122] at e_868_16
  have e_868_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_17
  simp only [← c54123, ← c54124, k2, ← c54125] at e_868_17
  have hr := e_868_17
  simp only [e_868_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 868 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6534)) := by
  have c54126 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54127 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54128 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54129 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54130 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54131 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_868_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_18
  simp only [← c54126, ← c54127, ← c54128] at e_868_18
  have e_868_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_868_19
  simp only [← c54129, ← c54130, k2, ← c54131] at e_868_19
  have hr := e_868_19
  simp only [e_868_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6535)) := by
  have c54132 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54133 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54134 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54135 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54136 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54137 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_0
  simp only [← c54132, ← c54133, ← c54134] at e_869_0
  have e_869_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_1
  simp only [← c54135, ← c54136, k2, ← c54137] at e_869_1
  have hr := e_869_1
  simp only [e_869_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6536)) := by
  have c54138 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54139 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54140 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54141 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54142 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54143 := (publicBatchWrapper89_copies1691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_2
  simp only [← c54138, ← c54139, ← c54140] at e_869_2
  have e_869_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_3
  simp only [← c54141, ← c54142, k2, ← c54143] at e_869_3
  have hr := e_869_3
  simp only [e_869_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6537)) := by
  have c54144 := (publicBatchWrapper89_copies1692 a h).1
  have c54145 := (publicBatchWrapper89_copies1692 a h).2.1
  have c54146 := (publicBatchWrapper89_copies1692 a h).2.2.1
  have c54147 := (publicBatchWrapper89_copies1692 a h).2.2.2.1
  have c54148 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.1
  have c54149 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_4
  simp only [← c54144, ← c54145, ← c54146] at e_869_4
  have e_869_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_5
  simp only [← c54147, ← c54148, k2, ← c54149] at e_869_5
  have hr := e_869_5
  simp only [e_869_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6538)) := by
  have c54150 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.1
  have c54151 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.1
  have c54152 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.1
  have c54153 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.1
  have c54154 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54155 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_6
  simp only [← c54150, ← c54151, ← c54152] at e_869_6
  have e_869_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_7
  simp only [← c54153, ← c54154, k2, ← c54155] at e_869_7
  have hr := e_869_7
  simp only [e_869_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6539)) := by
  have c54156 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54157 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54158 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54159 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54160 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54161 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_8
  simp only [← c54156, ← c54157, ← c54158] at e_869_8
  have e_869_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_9
  simp only [← c54159, ← c54160, k2, ← c54161] at e_869_9
  have hr := e_869_9
  simp only [e_869_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6540)) := by
  have c54162 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54163 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54164 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54165 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54166 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54167 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_10
  simp only [← c54162, ← c54163, ← c54164] at e_869_10
  have e_869_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_11
  simp only [← c54165, ← c54166, k2, ← c54167] at e_869_11
  have hr := e_869_11
  simp only [e_869_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6541)) := by
  have c54168 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54169 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54170 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54171 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54172 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54173 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_12
  simp only [← c54168, ← c54169, ← c54170] at e_869_12
  have e_869_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_13
  simp only [← c54171, ← c54172, k2, ← c54173] at e_869_13
  have hr := e_869_13
  simp only [e_869_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6542)) := by
  have c54174 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54175 := (publicBatchWrapper89_copies1692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54176 := (publicBatchWrapper89_copies1693 a h).1
  have c54177 := (publicBatchWrapper89_copies1693 a h).2.1
  have c54178 := (publicBatchWrapper89_copies1693 a h).2.2.1
  have c54179 := (publicBatchWrapper89_copies1693 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_14
  simp only [← c54174, ← c54175, ← c54176] at e_869_14
  have e_869_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_15
  simp only [← c54177, ← c54178, k2, ← c54179] at e_869_15
  have hr := e_869_15
  simp only [e_869_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6543)) := by
  have c54180 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.1
  have c54181 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.1
  have c54182 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.1
  have c54183 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.1
  have c54184 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.1
  have c54185 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_16
  simp only [← c54180, ← c54181, ← c54182] at e_869_16
  have e_869_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_17
  simp only [← c54183, ← c54184, k2, ← c54185] at e_869_17
  have hr := e_869_17
  simp only [e_869_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 869 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6544)) := by
  have c54186 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54187 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54188 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54189 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54190 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54191 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_869_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_18
  simp only [← c54186, ← c54187, ← c54188] at e_869_18
  have e_869_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_869_19
  simp only [← c54189, ← c54190, k2, ← c54191] at e_869_19
  have hr := e_869_19
  simp only [e_869_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6545)) := by
  have c54192 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54193 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54194 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54195 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54196 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54197 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_0
  simp only [← c54192, ← c54193, ← c54194] at e_870_0
  have e_870_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_1
  simp only [← c54195, ← c54196, k2, ← c54197] at e_870_1
  have hr := e_870_1
  simp only [e_870_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6546)) := by
  have c54198 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54199 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54200 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54201 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54202 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54203 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_2
  simp only [← c54198, ← c54199, ← c54200] at e_870_2
  have e_870_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_3
  simp only [← c54201, ← c54202, k2, ← c54203] at e_870_3
  have hr := e_870_3
  simp only [e_870_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6547)) := by
  have c54204 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54205 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54206 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54207 := (publicBatchWrapper89_copies1693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54208 := (publicBatchWrapper89_copies1694 a h).1
  have c54209 := (publicBatchWrapper89_copies1694 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_4
  simp only [← c54204, ← c54205, ← c54206] at e_870_4
  have e_870_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_5
  simp only [← c54207, ← c54208, k2, ← c54209] at e_870_5
  have hr := e_870_5
  simp only [e_870_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6548)) := by
  have c54210 := (publicBatchWrapper89_copies1694 a h).2.2.1
  have c54211 := (publicBatchWrapper89_copies1694 a h).2.2.2.1
  have c54212 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.1
  have c54213 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.1
  have c54214 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.1
  have c54215 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_6
  simp only [← c54210, ← c54211, ← c54212] at e_870_6
  have e_870_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_7
  simp only [← c54213, ← c54214, k2, ← c54215] at e_870_7
  have hr := e_870_7
  simp only [e_870_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6549)) := by
  have c54216 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.1
  have c54217 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.1
  have c54218 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54219 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54220 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54221 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_8
  simp only [← c54216, ← c54217, ← c54218] at e_870_8
  have e_870_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_9
  simp only [← c54219, ← c54220, k2, ← c54221] at e_870_9
  have hr := e_870_9
  simp only [e_870_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6550)) := by
  have c54222 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54223 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54224 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54225 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54226 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54227 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_10
  simp only [← c54222, ← c54223, ← c54224] at e_870_10
  have e_870_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_11
  simp only [← c54225, ← c54226, k2, ← c54227] at e_870_11
  have hr := e_870_11
  simp only [e_870_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6551)) := by
  have c54228 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54229 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54230 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54231 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54232 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54233 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_12
  simp only [← c54228, ← c54229, ← c54230] at e_870_12
  have e_870_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_13
  simp only [← c54231, ← c54232, k2, ← c54233] at e_870_13
  have hr := e_870_13
  simp only [e_870_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6552)) := by
  have c54234 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54235 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54236 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54237 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54238 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54239 := (publicBatchWrapper89_copies1694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_14
  simp only [← c54234, ← c54235, ← c54236] at e_870_14
  have e_870_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_15
  simp only [← c54237, ← c54238, k2, ← c54239] at e_870_15
  have hr := e_870_15
  simp only [e_870_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6553)) := by
  have c54240 := (publicBatchWrapper89_copies1695 a h).1
  have c54241 := (publicBatchWrapper89_copies1695 a h).2.1
  have c54242 := (publicBatchWrapper89_copies1695 a h).2.2.1
  have c54243 := (publicBatchWrapper89_copies1695 a h).2.2.2.1
  have c54244 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.1
  have c54245 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_16
  simp only [← c54240, ← c54241, ← c54242] at e_870_16
  have e_870_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_17
  simp only [← c54243, ← c54244, k2, ← c54245] at e_870_17
  have hr := e_870_17
  simp only [e_870_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 870 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6554)) := by
  have c54246 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.1
  have c54247 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.1
  have c54248 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.1
  have c54249 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.1
  have c54250 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54251 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_870_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_18
  simp only [← c54246, ← c54247, ← c54248] at e_870_18
  have e_870_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_870_19
  simp only [← c54249, ← c54250, k2, ← c54251] at e_870_19
  have hr := e_870_19
  simp only [e_870_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6555)) := by
  have c54252 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54253 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54254 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54255 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54256 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54257 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_0
  simp only [← c54252, ← c54253, ← c54254] at e_871_0
  have e_871_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_1
  simp only [← c54255, ← c54256, k2, ← c54257] at e_871_1
  have hr := e_871_1
  simp only [e_871_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6556)) := by
  have c54258 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54259 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54260 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54261 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54262 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54263 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_2
  simp only [← c54258, ← c54259, ← c54260] at e_871_2
  have e_871_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_3
  simp only [← c54261, ← c54262, k2, ← c54263] at e_871_3
  have hr := e_871_3
  simp only [e_871_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6557)) := by
  have c54264 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54265 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54266 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54267 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54268 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54269 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_4
  simp only [← c54264, ← c54265, ← c54266] at e_871_4
  have e_871_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_5
  simp only [← c54267, ← c54268, k2, ← c54269] at e_871_5
  have hr := e_871_5
  simp only [e_871_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6558)) := by
  have c54270 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54271 := (publicBatchWrapper89_copies1695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54272 := (publicBatchWrapper89_copies1696 a h).1
  have c54273 := (publicBatchWrapper89_copies1696 a h).2.1
  have c54274 := (publicBatchWrapper89_copies1696 a h).2.2.1
  have c54275 := (publicBatchWrapper89_copies1696 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_6
  simp only [← c54270, ← c54271, ← c54272] at e_871_6
  have e_871_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_7
  simp only [← c54273, ← c54274, k2, ← c54275] at e_871_7
  have hr := e_871_7
  simp only [e_871_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6559)) := by
  have c54276 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.1
  have c54277 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.1
  have c54278 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.1
  have c54279 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.1
  have c54280 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.1
  have c54281 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_8
  simp only [← c54276, ← c54277, ← c54278] at e_871_8
  have e_871_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_9
  simp only [← c54279, ← c54280, k2, ← c54281] at e_871_9
  have hr := e_871_9
  simp only [e_871_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6560)) := by
  have c54282 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54283 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54284 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54285 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54286 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54287 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_10
  simp only [← c54282, ← c54283, ← c54284] at e_871_10
  have e_871_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_11
  simp only [← c54285, ← c54286, k2, ← c54287] at e_871_11
  have hr := e_871_11
  simp only [e_871_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6561)) := by
  have c54288 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54289 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54290 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54291 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54292 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54293 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_12
  simp only [← c54288, ← c54289, ← c54290] at e_871_12
  have e_871_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_13
  simp only [← c54291, ← c54292, k2, ← c54293] at e_871_13
  have hr := e_871_13
  simp only [e_871_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6562)) := by
  have c54294 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54295 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54296 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54297 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54298 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54299 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_14
  simp only [← c54294, ← c54295, ← c54296] at e_871_14
  have e_871_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_15
  simp only [← c54297, ← c54298, k2, ← c54299] at e_871_15
  have hr := e_871_15
  simp only [e_871_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6563)) := by
  have c54300 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54301 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54302 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54303 := (publicBatchWrapper89_copies1696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54304 := (publicBatchWrapper89_copies1697 a h).1
  have c54305 := (publicBatchWrapper89_copies1697 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_16
  simp only [← c54300, ← c54301, ← c54302] at e_871_16
  have e_871_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_17
  simp only [← c54303, ← c54304, k2, ← c54305] at e_871_17
  have hr := e_871_17
  simp only [e_871_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 871 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6564)) := by
  have c54306 := (publicBatchWrapper89_copies1697 a h).2.2.1
  have c54307 := (publicBatchWrapper89_copies1697 a h).2.2.2.1
  have c54308 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.1
  have c54309 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.1
  have c54310 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.1
  have c54311 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_871_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_18
  simp only [← c54306, ← c54307, ← c54308] at e_871_18
  have e_871_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_871_19
  simp only [← c54309, ← c54310, k2, ← c54311] at e_871_19
  have hr := e_871_19
  simp only [e_871_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6565)) := by
  have c54312 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.1
  have c54313 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.1
  have c54314 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54315 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54316 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54317 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_0
  simp only [← c54312, ← c54313, ← c54314] at e_872_0
  have e_872_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_1
  simp only [← c54315, ← c54316, k2, ← c54317] at e_872_1
  have hr := e_872_1
  simp only [e_872_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6566)) := by
  have c54318 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54319 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54320 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54321 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54322 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54323 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_2
  simp only [← c54318, ← c54319, ← c54320] at e_872_2
  have e_872_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_3
  simp only [← c54321, ← c54322, k2, ← c54323] at e_872_3
  have hr := e_872_3
  simp only [e_872_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6567)) := by
  have c54324 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54325 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54326 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54327 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54328 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54329 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_4
  simp only [← c54324, ← c54325, ← c54326] at e_872_4
  have e_872_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_5
  simp only [← c54327, ← c54328, k2, ← c54329] at e_872_5
  have hr := e_872_5
  simp only [e_872_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6568)) := by
  have c54330 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54331 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54332 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54333 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54334 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54335 := (publicBatchWrapper89_copies1697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_6
  simp only [← c54330, ← c54331, ← c54332] at e_872_6
  have e_872_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_7
  simp only [← c54333, ← c54334, k2, ← c54335] at e_872_7
  have hr := e_872_7
  simp only [e_872_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6569)) := by
  have c54336 := (publicBatchWrapper89_copies1698 a h).1
  have c54337 := (publicBatchWrapper89_copies1698 a h).2.1
  have c54338 := (publicBatchWrapper89_copies1698 a h).2.2.1
  have c54339 := (publicBatchWrapper89_copies1698 a h).2.2.2.1
  have c54340 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.1
  have c54341 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_8
  simp only [← c54336, ← c54337, ← c54338] at e_872_8
  have e_872_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_9
  simp only [← c54339, ← c54340, k2, ← c54341] at e_872_9
  have hr := e_872_9
  simp only [e_872_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6610)) := by
  have c54342 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.1
  have c54343 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.1
  have c54344 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.1
  have c54345 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.1
  have c54346 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54347 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_10
  simp only [← c54342, ← c54343, ← c54344] at e_872_10
  have e_872_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_11
  simp only [← c54345, ← c54346, k2, ← c54347] at e_872_11
  have hr := e_872_11
  simp only [e_872_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6611)) := by
  have c54348 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54349 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54350 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54351 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54352 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54353 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_12
  simp only [← c54348, ← c54349, ← c54350] at e_872_12
  have e_872_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_13
  simp only [← c54351, ← c54352, k2, ← c54353] at e_872_13
  have hr := e_872_13
  simp only [e_872_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6612)) := by
  have c54354 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54355 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54356 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54357 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54358 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54359 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_14
  simp only [← c54354, ← c54355, ← c54356] at e_872_14
  have e_872_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_15
  simp only [← c54357, ← c54358, k2, ← c54359] at e_872_15
  have hr := e_872_15
  simp only [e_872_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6613)) := by
  have c54360 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54361 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54362 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54363 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54364 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54365 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_16
  simp only [← c54360, ← c54361, ← c54362] at e_872_16
  have e_872_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_17
  simp only [← c54363, ← c54364, k2, ← c54365] at e_872_17
  have hr := e_872_17
  simp only [e_872_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 872 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6614)) := by
  have c54366 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54367 := (publicBatchWrapper89_copies1698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54368 := (publicBatchWrapper89_copies1699 a h).1
  have c54369 := (publicBatchWrapper89_copies1699 a h).2.1
  have c54370 := (publicBatchWrapper89_copies1699 a h).2.2.1
  have c54371 := (publicBatchWrapper89_copies1699 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_872_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_18
  simp only [← c54366, ← c54367, ← c54368] at e_872_18
  have e_872_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_872_19
  simp only [← c54369, ← c54370, k2, ← c54371] at e_872_19
  have hr := e_872_19
  simp only [e_872_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6615)) := by
  have c54372 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.1
  have c54373 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.1
  have c54374 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.1
  have c54375 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.1
  have c54376 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.1
  have c54377 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_0
  simp only [← c54372, ← c54373, ← c54374] at e_873_0
  have e_873_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_1
  simp only [← c54375, ← c54376, k2, ← c54377] at e_873_1
  have hr := e_873_1
  simp only [e_873_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6616)) := by
  have c54378 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54379 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54380 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54381 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54382 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54383 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_2
  simp only [← c54378, ← c54379, ← c54380] at e_873_2
  have e_873_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_3
  simp only [← c54381, ← c54382, k2, ← c54383] at e_873_3
  have hr := e_873_3
  simp only [e_873_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6617)) := by
  have c54384 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54385 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54386 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54387 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54388 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54389 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_4
  simp only [← c54384, ← c54385, ← c54386] at e_873_4
  have e_873_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_5
  simp only [← c54387, ← c54388, k2, ← c54389] at e_873_5
  have hr := e_873_5
  simp only [e_873_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6618)) := by
  have c54390 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54391 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54392 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54393 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54394 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54395 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_6
  simp only [← c54390, ← c54391, ← c54392] at e_873_6
  have e_873_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_7
  simp only [← c54393, ← c54394, k2, ← c54395] at e_873_7
  have hr := e_873_7
  simp only [e_873_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6619)) := by
  have c54396 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54397 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54398 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54399 := (publicBatchWrapper89_copies1699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54400 := (publicBatchWrapper89_copies1700 a h).1
  have c54401 := (publicBatchWrapper89_copies1700 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_8
  simp only [← c54396, ← c54397, ← c54398] at e_873_8
  have e_873_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_9
  simp only [← c54399, ← c54400, k2, ← c54401] at e_873_9
  have hr := e_873_9
  simp only [e_873_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6620)) := by
  have c54402 := (publicBatchWrapper89_copies1700 a h).2.2.1
  have c54403 := (publicBatchWrapper89_copies1700 a h).2.2.2.1
  have c54404 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.1
  have c54405 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.1
  have c54406 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.1
  have c54407 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_10
  simp only [← c54402, ← c54403, ← c54404] at e_873_10
  have e_873_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_11
  simp only [← c54405, ← c54406, k2, ← c54407] at e_873_11
  have hr := e_873_11
  simp only [e_873_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6621)) := by
  have c54408 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.1
  have c54409 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.1
  have c54410 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54411 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54412 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54413 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_12
  simp only [← c54408, ← c54409, ← c54410] at e_873_12
  have e_873_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_13
  simp only [← c54411, ← c54412, k2, ← c54413] at e_873_13
  have hr := e_873_13
  simp only [e_873_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6622)) := by
  have c54414 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54415 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54416 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54417 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54418 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54419 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_14
  simp only [← c54414, ← c54415, ← c54416] at e_873_14
  have e_873_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_15
  simp only [← c54417, ← c54418, k2, ← c54419] at e_873_15
  have hr := e_873_15
  simp only [e_873_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6623)) := by
  have c54420 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54421 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54422 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54423 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54424 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54425 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_16
  simp only [← c54420, ← c54421, ← c54422] at e_873_16
  have e_873_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_17
  simp only [← c54423, ← c54424, k2, ← c54425] at e_873_17
  have hr := e_873_17
  simp only [e_873_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 873 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6624)) := by
  have c54426 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54427 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54428 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54429 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54430 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54431 := (publicBatchWrapper89_copies1700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_873_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_18
  simp only [← c54426, ← c54427, ← c54428] at e_873_18
  have e_873_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_873_19
  simp only [← c54429, ← c54430, k2, ← c54431] at e_873_19
  have hr := e_873_19
  simp only [e_873_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6625)) := by
  have c54432 := (publicBatchWrapper89_copies1701 a h).1
  have c54433 := (publicBatchWrapper89_copies1701 a h).2.1
  have c54434 := (publicBatchWrapper89_copies1701 a h).2.2.1
  have c54435 := (publicBatchWrapper89_copies1701 a h).2.2.2.1
  have c54436 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.1
  have c54437 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_0
  simp only [← c54432, ← c54433, ← c54434] at e_874_0
  have e_874_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_1
  simp only [← c54435, ← c54436, k2, ← c54437] at e_874_1
  have hr := e_874_1
  simp only [e_874_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6626)) := by
  have c54438 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.1
  have c54439 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.1
  have c54440 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.1
  have c54441 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.1
  have c54442 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54443 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_2
  simp only [← c54438, ← c54439, ← c54440] at e_874_2
  have e_874_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_3
  simp only [← c54441, ← c54442, k2, ← c54443] at e_874_3
  have hr := e_874_3
  simp only [e_874_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6627)) := by
  have c54444 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54445 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54446 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54447 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54448 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54449 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_4
  simp only [← c54444, ← c54445, ← c54446] at e_874_4
  have e_874_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_5
  simp only [← c54447, ← c54448, k2, ← c54449] at e_874_5
  have hr := e_874_5
  simp only [e_874_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6628)) := by
  have c54450 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54451 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54452 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54453 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54454 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54455 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_6
  simp only [← c54450, ← c54451, ← c54452] at e_874_6
  have e_874_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_7
  simp only [← c54453, ← c54454, k2, ← c54455] at e_874_7
  have hr := e_874_7
  simp only [e_874_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6629)) := by
  have c54456 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54457 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54458 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54459 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54460 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54461 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_8
  simp only [← c54456, ← c54457, ← c54458] at e_874_8
  have e_874_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_9
  simp only [← c54459, ← c54460, k2, ← c54461] at e_874_9
  have hr := e_874_9
  simp only [e_874_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6630)) := by
  have c54462 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54463 := (publicBatchWrapper89_copies1701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54464 := (publicBatchWrapper89_copies1702 a h).1
  have c54465 := (publicBatchWrapper89_copies1702 a h).2.1
  have c54466 := (publicBatchWrapper89_copies1702 a h).2.2.1
  have c54467 := (publicBatchWrapper89_copies1702 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_10
  simp only [← c54462, ← c54463, ← c54464] at e_874_10
  have e_874_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_11
  simp only [← c54465, ← c54466, k2, ← c54467] at e_874_11
  have hr := e_874_11
  simp only [e_874_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6631)) := by
  have c54468 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.1
  have c54469 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.1
  have c54470 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.1
  have c54471 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.1
  have c54472 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.1
  have c54473 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_12
  simp only [← c54468, ← c54469, ← c54470] at e_874_12
  have e_874_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_13
  simp only [← c54471, ← c54472, k2, ← c54473] at e_874_13
  have hr := e_874_13
  simp only [e_874_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6632)) := by
  have c54474 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54475 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54476 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54477 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54478 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54479 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_14
  simp only [← c54474, ← c54475, ← c54476] at e_874_14
  have e_874_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_15
  simp only [← c54477, ← c54478, k2, ← c54479] at e_874_15
  have hr := e_874_15
  simp only [e_874_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6633)) := by
  have c54480 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54481 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54482 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54483 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54484 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54485 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_16
  simp only [← c54480, ← c54481, ← c54482] at e_874_16
  have e_874_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_17
  simp only [← c54483, ← c54484, k2, ← c54485] at e_874_17
  have hr := e_874_17
  simp only [e_874_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 874 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6634)) := by
  have c54486 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54487 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54488 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54489 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54490 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54491 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_874_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_18
  simp only [← c54486, ← c54487, ← c54488] at e_874_18
  have e_874_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_874_19
  simp only [← c54489, ← c54490, k2, ← c54491] at e_874_19
  have hr := e_874_19
  simp only [e_874_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6635)) := by
  have c54492 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54493 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54494 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54495 := (publicBatchWrapper89_copies1702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54496 := (publicBatchWrapper89_copies1703 a h).1
  have c54497 := (publicBatchWrapper89_copies1703 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_0
  simp only [← c54492, ← c54493, ← c54494] at e_875_0
  have e_875_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_1
  simp only [← c54495, ← c54496, k2, ← c54497] at e_875_1
  have hr := e_875_1
  simp only [e_875_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6636)) := by
  have c54498 := (publicBatchWrapper89_copies1703 a h).2.2.1
  have c54499 := (publicBatchWrapper89_copies1703 a h).2.2.2.1
  have c54500 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.1
  have c54501 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.1
  have c54502 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.1
  have c54503 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_2
  simp only [← c54498, ← c54499, ← c54500] at e_875_2
  have e_875_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_3
  simp only [← c54501, ← c54502, k2, ← c54503] at e_875_3
  have hr := e_875_3
  simp only [e_875_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6637)) := by
  have c54504 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.1
  have c54505 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.1
  have c54506 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54507 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54508 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54509 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_4
  simp only [← c54504, ← c54505, ← c54506] at e_875_4
  have e_875_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_5
  simp only [← c54507, ← c54508, k2, ← c54509] at e_875_5
  have hr := e_875_5
  simp only [e_875_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6638)) := by
  have c54510 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54511 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54512 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54513 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54514 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54515 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_6
  simp only [← c54510, ← c54511, ← c54512] at e_875_6
  have e_875_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_7
  simp only [← c54513, ← c54514, k2, ← c54515] at e_875_7
  have hr := e_875_7
  simp only [e_875_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6639)) := by
  have c54516 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54517 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54518 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54519 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54520 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54521 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_8
  simp only [← c54516, ← c54517, ← c54518] at e_875_8
  have e_875_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_9
  simp only [← c54519, ← c54520, k2, ← c54521] at e_875_9
  have hr := e_875_9
  simp only [e_875_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6640)) := by
  have c54522 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54523 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54524 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54525 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54526 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54527 := (publicBatchWrapper89_copies1703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_10
  simp only [← c54522, ← c54523, ← c54524] at e_875_10
  have e_875_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_11
  simp only [← c54525, ← c54526, k2, ← c54527] at e_875_11
  have hr := e_875_11
  simp only [e_875_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6641)) := by
  have c54528 := (publicBatchWrapper89_copies1704 a h).1
  have c54529 := (publicBatchWrapper89_copies1704 a h).2.1
  have c54530 := (publicBatchWrapper89_copies1704 a h).2.2.1
  have c54531 := (publicBatchWrapper89_copies1704 a h).2.2.2.1
  have c54532 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.1
  have c54533 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_12
  simp only [← c54528, ← c54529, ← c54530] at e_875_12
  have e_875_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_13
  simp only [← c54531, ← c54532, k2, ← c54533] at e_875_13
  have hr := e_875_13
  simp only [e_875_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6642)) := by
  have c54534 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.1
  have c54535 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.1
  have c54536 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.1
  have c54537 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.1
  have c54538 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54539 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_14
  simp only [← c54534, ← c54535, ← c54536] at e_875_14
  have e_875_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_15
  simp only [← c54537, ← c54538, k2, ← c54539] at e_875_15
  have hr := e_875_15
  simp only [e_875_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6643)) := by
  have c54540 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54541 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54542 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54543 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54544 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54545 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_16
  simp only [← c54540, ← c54541, ← c54542] at e_875_16
  have e_875_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_17
  simp only [← c54543, ← c54544, k2, ← c54545] at e_875_17
  have hr := e_875_17
  simp only [e_875_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 875 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6644)) := by
  have c54546 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54547 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54548 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54549 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54550 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54551 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_875_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_18
  simp only [← c54546, ← c54547, ← c54548] at e_875_18
  have e_875_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_875_19
  simp only [← c54549, ← c54550, k2, ← c54551] at e_875_19
  have hr := e_875_19
  simp only [e_875_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6645)) := by
  have c54552 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54553 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54554 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54555 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54556 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54557 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_0
  simp only [← c54552, ← c54553, ← c54554] at e_876_0
  have e_876_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_1
  simp only [← c54555, ← c54556, k2, ← c54557] at e_876_1
  have hr := e_876_1
  simp only [e_876_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6646)) := by
  have c54558 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54559 := (publicBatchWrapper89_copies1704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54560 := (publicBatchWrapper89_copies1705 a h).1
  have c54561 := (publicBatchWrapper89_copies1705 a h).2.1
  have c54562 := (publicBatchWrapper89_copies1705 a h).2.2.1
  have c54563 := (publicBatchWrapper89_copies1705 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_2
  simp only [← c54558, ← c54559, ← c54560] at e_876_2
  have e_876_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_3
  simp only [← c54561, ← c54562, k2, ← c54563] at e_876_3
  have hr := e_876_3
  simp only [e_876_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6647)) := by
  have c54564 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.1
  have c54565 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.1
  have c54566 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.1
  have c54567 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.1
  have c54568 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.1
  have c54569 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_4
  simp only [← c54564, ← c54565, ← c54566] at e_876_4
  have e_876_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_5
  simp only [← c54567, ← c54568, k2, ← c54569] at e_876_5
  have hr := e_876_5
  simp only [e_876_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6648)) := by
  have c54570 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54571 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54572 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54573 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54574 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54575 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_6
  simp only [← c54570, ← c54571, ← c54572] at e_876_6
  have e_876_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_7
  simp only [← c54573, ← c54574, k2, ← c54575] at e_876_7
  have hr := e_876_7
  simp only [e_876_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6649)) := by
  have c54576 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54577 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54578 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54579 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54580 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54581 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_8
  simp only [← c54576, ← c54577, ← c54578] at e_876_8
  have e_876_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_9
  simp only [← c54579, ← c54580, k2, ← c54581] at e_876_9
  have hr := e_876_9
  simp only [e_876_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6650)) := by
  have c54582 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54583 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54584 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54585 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54586 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54587 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_10
  simp only [← c54582, ← c54583, ← c54584] at e_876_10
  have e_876_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_11
  simp only [← c54585, ← c54586, k2, ← c54587] at e_876_11
  have hr := e_876_11
  simp only [e_876_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6651)) := by
  have c54588 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54589 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54590 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54591 := (publicBatchWrapper89_copies1705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54592 := (publicBatchWrapper89_copies1706 a h).1
  have c54593 := (publicBatchWrapper89_copies1706 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_12
  simp only [← c54588, ← c54589, ← c54590] at e_876_12
  have e_876_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_13
  simp only [← c54591, ← c54592, k2, ← c54593] at e_876_13
  have hr := e_876_13
  simp only [e_876_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6652)) := by
  have c54594 := (publicBatchWrapper89_copies1706 a h).2.2.1
  have c54595 := (publicBatchWrapper89_copies1706 a h).2.2.2.1
  have c54596 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.1
  have c54597 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.1
  have c54598 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.1
  have c54599 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_14
  simp only [← c54594, ← c54595, ← c54596] at e_876_14
  have e_876_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_15
  simp only [← c54597, ← c54598, k2, ← c54599] at e_876_15
  have hr := e_876_15
  simp only [e_876_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6653)) := by
  have c54600 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.1
  have c54601 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.1
  have c54602 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54603 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54604 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54605 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_16
  simp only [← c54600, ← c54601, ← c54602] at e_876_16
  have e_876_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_17
  simp only [← c54603, ← c54604, k2, ← c54605] at e_876_17
  have hr := e_876_17
  simp only [e_876_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 876 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6654)) := by
  have c54606 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54607 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54608 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54609 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54610 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54611 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_876_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_18
  simp only [← c54606, ← c54607, ← c54608] at e_876_18
  have e_876_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_876_19
  simp only [← c54609, ← c54610, k2, ← c54611] at e_876_19
  have hr := e_876_19
  simp only [e_876_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6655)) := by
  have c54612 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54613 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54614 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54615 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54616 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54617 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_0
  simp only [← c54612, ← c54613, ← c54614] at e_877_0
  have e_877_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_1
  simp only [← c54615, ← c54616, k2, ← c54617] at e_877_1
  have hr := e_877_1
  simp only [e_877_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6656)) := by
  have c54618 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54619 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54620 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54621 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54622 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54623 := (publicBatchWrapper89_copies1706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_2
  simp only [← c54618, ← c54619, ← c54620] at e_877_2
  have e_877_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_3
  simp only [← c54621, ← c54622, k2, ← c54623] at e_877_3
  have hr := e_877_3
  simp only [e_877_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6657)) := by
  have c54624 := (publicBatchWrapper89_copies1707 a h).1
  have c54625 := (publicBatchWrapper89_copies1707 a h).2.1
  have c54626 := (publicBatchWrapper89_copies1707 a h).2.2.1
  have c54627 := (publicBatchWrapper89_copies1707 a h).2.2.2.1
  have c54628 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.1
  have c54629 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_4
  simp only [← c54624, ← c54625, ← c54626] at e_877_4
  have e_877_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_5
  simp only [← c54627, ← c54628, k2, ← c54629] at e_877_5
  have hr := e_877_5
  simp only [e_877_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6658)) := by
  have c54630 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.1
  have c54631 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.1
  have c54632 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.1
  have c54633 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.1
  have c54634 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54635 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_6
  simp only [← c54630, ← c54631, ← c54632] at e_877_6
  have e_877_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_7
  simp only [← c54633, ← c54634, k2, ← c54635] at e_877_7
  have hr := e_877_7
  simp only [e_877_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6659)) := by
  have c54636 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54637 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54638 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54639 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54640 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54641 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_8
  simp only [← c54636, ← c54637, ← c54638] at e_877_8
  have e_877_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_9
  simp only [← c54639, ← c54640, k2, ← c54641] at e_877_9
  have hr := e_877_9
  simp only [e_877_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6660)) := by
  have c54642 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54643 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54644 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54645 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54646 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54647 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_10
  simp only [← c54642, ← c54643, ← c54644] at e_877_10
  have e_877_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_11
  simp only [← c54645, ← c54646, k2, ← c54647] at e_877_11
  have hr := e_877_11
  simp only [e_877_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6661)) := by
  have c54648 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54649 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54650 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54651 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54652 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54653 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_12
  simp only [← c54648, ← c54649, ← c54650] at e_877_12
  have e_877_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_13
  simp only [← c54651, ← c54652, k2, ← c54653] at e_877_13
  have hr := e_877_13
  simp only [e_877_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6662)) := by
  have c54654 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54655 := (publicBatchWrapper89_copies1707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54656 := (publicBatchWrapper89_copies1708 a h).1
  have c54657 := (publicBatchWrapper89_copies1708 a h).2.1
  have c54658 := (publicBatchWrapper89_copies1708 a h).2.2.1
  have c54659 := (publicBatchWrapper89_copies1708 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_14
  simp only [← c54654, ← c54655, ← c54656] at e_877_14
  have e_877_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_15
  simp only [← c54657, ← c54658, k2, ← c54659] at e_877_15
  have hr := e_877_15
  simp only [e_877_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6663)) := by
  have c54660 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.1
  have c54661 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.1
  have c54662 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.1
  have c54663 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.1
  have c54664 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.1
  have c54665 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_16
  simp only [← c54660, ← c54661, ← c54662] at e_877_16
  have e_877_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_17
  simp only [← c54663, ← c54664, k2, ← c54665] at e_877_17
  have hr := e_877_17
  simp only [e_877_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 877 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6664)) := by
  have c54666 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54667 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54668 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54669 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54670 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54671 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_877_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_18
  simp only [← c54666, ← c54667, ← c54668] at e_877_18
  have e_877_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_877_19
  simp only [← c54669, ← c54670, k2, ← c54671] at e_877_19
  have hr := e_877_19
  simp only [e_877_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6665)) := by
  have c54672 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54673 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54674 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54675 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54676 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54677 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_0
  simp only [← c54672, ← c54673, ← c54674] at e_878_0
  have e_878_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_1
  simp only [← c54675, ← c54676, k2, ← c54677] at e_878_1
  have hr := e_878_1
  simp only [e_878_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6666)) := by
  have c54678 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54679 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54680 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54681 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54682 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54683 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_2
  simp only [← c54678, ← c54679, ← c54680] at e_878_2
  have e_878_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_3
  simp only [← c54681, ← c54682, k2, ← c54683] at e_878_3
  have hr := e_878_3
  simp only [e_878_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6667)) := by
  have c54684 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54685 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54686 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54687 := (publicBatchWrapper89_copies1708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54688 := (publicBatchWrapper89_copies1709 a h).1
  have c54689 := (publicBatchWrapper89_copies1709 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_4
  simp only [← c54684, ← c54685, ← c54686] at e_878_4
  have e_878_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_5
  simp only [← c54687, ← c54688, k2, ← c54689] at e_878_5
  have hr := e_878_5
  simp only [e_878_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6668)) := by
  have c54690 := (publicBatchWrapper89_copies1709 a h).2.2.1
  have c54691 := (publicBatchWrapper89_copies1709 a h).2.2.2.1
  have c54692 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.1
  have c54693 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.1
  have c54694 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.1
  have c54695 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_6
  simp only [← c54690, ← c54691, ← c54692] at e_878_6
  have e_878_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_7
  simp only [← c54693, ← c54694, k2, ← c54695] at e_878_7
  have hr := e_878_7
  simp only [e_878_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6669)) := by
  have c54696 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.1
  have c54697 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.1
  have c54698 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54699 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54700 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54701 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_8
  simp only [← c54696, ← c54697, ← c54698] at e_878_8
  have e_878_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_9
  simp only [← c54699, ← c54700, k2, ← c54701] at e_878_9
  have hr := e_878_9
  simp only [e_878_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6670)) := by
  have c54702 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54703 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54704 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54705 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54706 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54707 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_10
  simp only [← c54702, ← c54703, ← c54704] at e_878_10
  have e_878_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_11
  simp only [← c54705, ← c54706, k2, ← c54707] at e_878_11
  have hr := e_878_11
  simp only [e_878_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6671)) := by
  have c54708 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54709 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54710 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54711 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54712 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54713 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_12
  simp only [← c54708, ← c54709, ← c54710] at e_878_12
  have e_878_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_13
  simp only [← c54711, ← c54712, k2, ← c54713] at e_878_13
  have hr := e_878_13
  simp only [e_878_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6672)) := by
  have c54714 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54715 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54716 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54717 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54718 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54719 := (publicBatchWrapper89_copies1709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_14
  simp only [← c54714, ← c54715, ← c54716] at e_878_14
  have e_878_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_15
  simp only [← c54717, ← c54718, k2, ← c54719] at e_878_15
  have hr := e_878_15
  simp only [e_878_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6673)) := by
  have c54720 := (publicBatchWrapper89_copies1710 a h).1
  have c54721 := (publicBatchWrapper89_copies1710 a h).2.1
  have c54722 := (publicBatchWrapper89_copies1710 a h).2.2.1
  have c54723 := (publicBatchWrapper89_copies1710 a h).2.2.2.1
  have c54724 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.1
  have c54725 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_16
  simp only [← c54720, ← c54721, ← c54722] at e_878_16
  have e_878_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_17
  simp only [← c54723, ← c54724, k2, ← c54725] at e_878_17
  have hr := e_878_17
  simp only [e_878_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 878 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6674)) := by
  have c54726 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.1
  have c54727 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.1
  have c54728 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.1
  have c54729 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.1
  have c54730 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54731 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_878_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_18
  simp only [← c54726, ← c54727, ← c54728] at e_878_18
  have e_878_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_878_19
  simp only [← c54729, ← c54730, k2, ← c54731] at e_878_19
  have hr := e_878_19
  simp only [e_878_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6675)) := by
  have c54732 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54733 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54734 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54735 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54736 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54737 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_0
  simp only [← c54732, ← c54733, ← c54734] at e_879_0
  have e_879_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_1
  simp only [← c54735, ← c54736, k2, ← c54737] at e_879_1
  have hr := e_879_1
  simp only [e_879_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6676)) := by
  have c54738 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54739 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54740 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54741 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54742 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54743 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_2
  simp only [← c54738, ← c54739, ← c54740] at e_879_2
  have e_879_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_3
  simp only [← c54741, ← c54742, k2, ← c54743] at e_879_3
  have hr := e_879_3
  simp only [e_879_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6677)) := by
  have c54744 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54745 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54746 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54747 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54748 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54749 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_4
  simp only [← c54744, ← c54745, ← c54746] at e_879_4
  have e_879_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_5
  simp only [← c54747, ← c54748, k2, ← c54749] at e_879_5
  have hr := e_879_5
  simp only [e_879_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6678)) := by
  have c54750 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54751 := (publicBatchWrapper89_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54752 := (publicBatchWrapper89_copies1711 a h).1
  have c54753 := (publicBatchWrapper89_copies1711 a h).2.1
  have c54754 := (publicBatchWrapper89_copies1711 a h).2.2.1
  have c54755 := (publicBatchWrapper89_copies1711 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_6
  simp only [← c54750, ← c54751, ← c54752] at e_879_6
  have e_879_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_7
  simp only [← c54753, ← c54754, k2, ← c54755] at e_879_7
  have hr := e_879_7
  simp only [e_879_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6679)) := by
  have c54756 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.1
  have c54757 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.1
  have c54758 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.1
  have c54759 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.1
  have c54760 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.1
  have c54761 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_8
  simp only [← c54756, ← c54757, ← c54758] at e_879_8
  have e_879_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_9
  simp only [← c54759, ← c54760, k2, ← c54761] at e_879_9
  have hr := e_879_9
  simp only [e_879_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6680)) := by
  have c54762 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54763 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54764 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54765 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54766 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54767 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_10
  simp only [← c54762, ← c54763, ← c54764] at e_879_10
  have e_879_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_11
  simp only [← c54765, ← c54766, k2, ← c54767] at e_879_11
  have hr := e_879_11
  simp only [e_879_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6681)) := by
  have c54768 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54769 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54770 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54771 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54772 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54773 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_12
  simp only [← c54768, ← c54769, ← c54770] at e_879_12
  have e_879_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_13
  simp only [← c54771, ← c54772, k2, ← c54773] at e_879_13
  have hr := e_879_13
  simp only [e_879_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6682)) := by
  have c54774 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54775 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54776 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54777 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54778 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54779 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_14
  simp only [← c54774, ← c54775, ← c54776] at e_879_14
  have e_879_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_15
  simp only [← c54777, ← c54778, k2, ← c54779] at e_879_15
  have hr := e_879_15
  simp only [e_879_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6683)) := by
  have c54780 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54781 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54782 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54783 := (publicBatchWrapper89_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54784 := (publicBatchWrapper89_copies1712 a h).1
  have c54785 := (publicBatchWrapper89_copies1712 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_16
  simp only [← c54780, ← c54781, ← c54782] at e_879_16
  have e_879_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_17
  simp only [← c54783, ← c54784, k2, ← c54785] at e_879_17
  have hr := e_879_17
  simp only [e_879_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 879 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6684)) := by
  have c54786 := (publicBatchWrapper89_copies1712 a h).2.2.1
  have c54787 := (publicBatchWrapper89_copies1712 a h).2.2.2.1
  have c54788 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.1
  have c54789 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.1
  have c54790 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.1
  have c54791 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_879_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_18
  simp only [← c54786, ← c54787, ← c54788] at e_879_18
  have e_879_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_879_19
  simp only [← c54789, ← c54790, k2, ← c54791] at e_879_19
  have hr := e_879_19
  simp only [e_879_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6685)) := by
  have c54792 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.1
  have c54793 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.1
  have c54794 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54795 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54796 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54797 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_0
  simp only [← c54792, ← c54793, ← c54794] at e_880_0
  have e_880_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_1
  simp only [← c54795, ← c54796, k2, ← c54797] at e_880_1
  have hr := e_880_1
  simp only [e_880_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6686)) := by
  have c54798 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54799 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54800 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54801 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54802 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54803 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_2
  simp only [← c54798, ← c54799, ← c54800] at e_880_2
  have e_880_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_3
  simp only [← c54801, ← c54802, k2, ← c54803] at e_880_3
  have hr := e_880_3
  simp only [e_880_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6687)) := by
  have c54804 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54805 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54806 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54807 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54808 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54809 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_4
  simp only [← c54804, ← c54805, ← c54806] at e_880_4
  have e_880_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_5
  simp only [← c54807, ← c54808, k2, ← c54809] at e_880_5
  have hr := e_880_5
  simp only [e_880_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6688)) := by
  have c54810 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54811 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54812 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54813 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54814 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54815 := (publicBatchWrapper89_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_6
  simp only [← c54810, ← c54811, ← c54812] at e_880_6
  have e_880_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_7
  simp only [← c54813, ← c54814, k2, ← c54815] at e_880_7
  have hr := e_880_7
  simp only [e_880_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6689)) := by
  have c54816 := (publicBatchWrapper89_copies1713 a h).1
  have c54817 := (publicBatchWrapper89_copies1713 a h).2.1
  have c54818 := (publicBatchWrapper89_copies1713 a h).2.2.1
  have c54819 := (publicBatchWrapper89_copies1713 a h).2.2.2.1
  have c54820 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.1
  have c54821 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_8
  simp only [← c54816, ← c54817, ← c54818] at e_880_8
  have e_880_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_9
  simp only [← c54819, ← c54820, k2, ← c54821] at e_880_9
  have hr := e_880_9
  simp only [e_880_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6690)) := by
  have c54822 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.1
  have c54823 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.1
  have c54824 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.1
  have c54825 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.1
  have c54826 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54827 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_10
  simp only [← c54822, ← c54823, ← c54824] at e_880_10
  have e_880_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_11
  simp only [← c54825, ← c54826, k2, ← c54827] at e_880_11
  have hr := e_880_11
  simp only [e_880_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6691)) := by
  have c54828 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54829 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54830 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54831 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54832 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54833 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_12
  simp only [← c54828, ← c54829, ← c54830] at e_880_12
  have e_880_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_13
  simp only [← c54831, ← c54832, k2, ← c54833] at e_880_13
  have hr := e_880_13
  simp only [e_880_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6692)) := by
  have c54834 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54835 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54836 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54837 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54838 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54839 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_14
  simp only [← c54834, ← c54835, ← c54836] at e_880_14
  have e_880_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_15
  simp only [← c54837, ← c54838, k2, ← c54839] at e_880_15
  have hr := e_880_15
  simp only [e_880_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6693)) := by
  have c54840 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54841 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54842 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54843 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54844 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54845 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_16
  simp only [← c54840, ← c54841, ← c54842] at e_880_16
  have e_880_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_17
  simp only [← c54843, ← c54844, k2, ← c54845] at e_880_17
  have hr := e_880_17
  simp only [e_880_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 880 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6694)) := by
  have c54846 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54847 := (publicBatchWrapper89_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54848 := (publicBatchWrapper89_copies1714 a h).1
  have c54849 := (publicBatchWrapper89_copies1714 a h).2.1
  have c54850 := (publicBatchWrapper89_copies1714 a h).2.2.1
  have c54851 := (publicBatchWrapper89_copies1714 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_880_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_18
  simp only [← c54846, ← c54847, ← c54848] at e_880_18
  have e_880_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_19
  simp only [← c54849, ← c54850, k2, ← c54851] at e_880_19
  have hr := e_880_19
  simp only [e_880_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6695)) := by
  have c54852 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.1
  have c54853 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.1
  have c54854 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.1
  have c54855 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.1
  have c54856 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.1
  have c54857 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_0
  simp only [← c54852, ← c54853, ← c54854] at e_881_0
  have e_881_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_1
  simp only [← c54855, ← c54856, k2, ← c54857] at e_881_1
  have hr := e_881_1
  simp only [e_881_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6696)) := by
  have c54858 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54859 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54860 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54861 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54862 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54863 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_2
  simp only [← c54858, ← c54859, ← c54860] at e_881_2
  have e_881_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_3
  simp only [← c54861, ← c54862, k2, ← c54863] at e_881_3
  have hr := e_881_3
  simp only [e_881_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6697)) := by
  have c54864 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54865 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54866 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54867 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54868 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54869 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_4
  simp only [← c54864, ← c54865, ← c54866] at e_881_4
  have e_881_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_5
  simp only [← c54867, ← c54868, k2, ← c54869] at e_881_5
  have hr := e_881_5
  simp only [e_881_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6698)) := by
  have c54870 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54871 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54872 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54873 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54874 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54875 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_6
  simp only [← c54870, ← c54871, ← c54872] at e_881_6
  have e_881_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_7
  simp only [← c54873, ← c54874, k2, ← c54875] at e_881_7
  have hr := e_881_7
  simp only [e_881_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6699)) := by
  have c54876 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54877 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54878 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54879 := (publicBatchWrapper89_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54880 := (publicBatchWrapper89_copies1715 a h).1
  have c54881 := (publicBatchWrapper89_copies1715 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_8
  simp only [← c54876, ← c54877, ← c54878] at e_881_8
  have e_881_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_9
  simp only [← c54879, ← c54880, k2, ← c54881] at e_881_9
  have hr := e_881_9
  simp only [e_881_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6700)) := by
  have c54882 := (publicBatchWrapper89_copies1715 a h).2.2.1
  have c54883 := (publicBatchWrapper89_copies1715 a h).2.2.2.1
  have c54884 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.1
  have c54885 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.1
  have c54886 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.1
  have c54887 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_10
  simp only [← c54882, ← c54883, ← c54884] at e_881_10
  have e_881_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_11
  simp only [← c54885, ← c54886, k2, ← c54887] at e_881_11
  have hr := e_881_11
  simp only [e_881_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6701)) := by
  have c54888 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.1
  have c54889 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.1
  have c54890 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54891 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54892 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54893 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_12
  simp only [← c54888, ← c54889, ← c54890] at e_881_12
  have e_881_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_13
  simp only [← c54891, ← c54892, k2, ← c54893] at e_881_13
  have hr := e_881_13
  simp only [e_881_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6702)) := by
  have c54894 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54895 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54896 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54897 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54898 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54899 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_14
  simp only [← c54894, ← c54895, ← c54896] at e_881_14
  have e_881_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_15
  simp only [← c54897, ← c54898, k2, ← c54899] at e_881_15
  have hr := e_881_15
  simp only [e_881_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6703)) := by
  have c54900 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54901 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54902 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54903 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54904 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54905 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_16
  simp only [← c54900, ← c54901, ← c54902] at e_881_16
  have e_881_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_17
  simp only [← c54903, ← c54904, k2, ← c54905] at e_881_17
  have hr := e_881_17
  simp only [e_881_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 881 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6704)) := by
  have c54906 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54907 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54908 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54909 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54910 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54911 := (publicBatchWrapper89_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_881_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_18
  simp only [← c54906, ← c54907, ← c54908] at e_881_18
  have e_881_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_19
  simp only [← c54909, ← c54910, k2, ← c54911] at e_881_19
  have hr := e_881_19
  simp only [e_881_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6705)) := by
  have c54912 := (publicBatchWrapper89_copies1716 a h).1
  have c54913 := (publicBatchWrapper89_copies1716 a h).2.1
  have c54914 := (publicBatchWrapper89_copies1716 a h).2.2.1
  have c54915 := (publicBatchWrapper89_copies1716 a h).2.2.2.1
  have c54916 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.1
  have c54917 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_0
  simp only [← c54912, ← c54913, ← c54914] at e_882_0
  have e_882_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_1
  simp only [← c54915, ← c54916, k2, ← c54917] at e_882_1
  have hr := e_882_1
  simp only [e_882_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6706)) := by
  have c54918 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.1
  have c54919 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.1
  have c54920 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.1
  have c54921 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.1
  have c54922 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54923 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_2
  simp only [← c54918, ← c54919, ← c54920] at e_882_2
  have e_882_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_3
  simp only [← c54921, ← c54922, k2, ← c54923] at e_882_3
  have hr := e_882_3
  simp only [e_882_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6707)) := by
  have c54924 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54925 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54926 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54927 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54928 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54929 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_4
  simp only [← c54924, ← c54925, ← c54926] at e_882_4
  have e_882_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_5
  simp only [← c54927, ← c54928, k2, ← c54929] at e_882_5
  have hr := e_882_5
  simp only [e_882_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6708)) := by
  have c54930 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54931 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54932 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54933 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54934 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54935 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_6
  simp only [← c54930, ← c54931, ← c54932] at e_882_6
  have e_882_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_7
  simp only [← c54933, ← c54934, k2, ← c54935] at e_882_7
  have hr := e_882_7
  simp only [e_882_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6709)) := by
  have c54936 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54937 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54938 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54939 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54940 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54941 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_8
  simp only [← c54936, ← c54937, ← c54938] at e_882_8
  have e_882_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_9
  simp only [← c54939, ← c54940, k2, ← c54941] at e_882_9
  have hr := e_882_9
  simp only [e_882_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6710)) := by
  have c54942 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54943 := (publicBatchWrapper89_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54944 := (publicBatchWrapper89_copies1717 a h).1
  have c54945 := (publicBatchWrapper89_copies1717 a h).2.1
  have c54946 := (publicBatchWrapper89_copies1717 a h).2.2.1
  have c54947 := (publicBatchWrapper89_copies1717 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_10
  simp only [← c54942, ← c54943, ← c54944] at e_882_10
  have e_882_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_11
  simp only [← c54945, ← c54946, k2, ← c54947] at e_882_11
  have hr := e_882_11
  simp only [e_882_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6711)) := by
  have c54948 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.1
  have c54949 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.1
  have c54950 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.1
  have c54951 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.1
  have c54952 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.1
  have c54953 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_12
  simp only [← c54948, ← c54949, ← c54950] at e_882_12
  have e_882_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_13
  simp only [← c54951, ← c54952, k2, ← c54953] at e_882_13
  have hr := e_882_13
  simp only [e_882_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6712)) := by
  have c54954 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54955 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54956 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54957 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54958 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54959 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_14
  simp only [← c54954, ← c54955, ← c54956] at e_882_14
  have e_882_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_15
  simp only [← c54957, ← c54958, k2, ← c54959] at e_882_15
  have hr := e_882_15
  simp only [e_882_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6713)) := by
  have c54960 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54961 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54962 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54963 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54964 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54965 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_16
  simp only [← c54960, ← c54961, ← c54962] at e_882_16
  have e_882_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_17
  simp only [← c54963, ← c54964, k2, ← c54965] at e_882_17
  have hr := e_882_17
  simp only [e_882_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 882 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6714)) := by
  have c54966 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54967 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54968 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54969 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54970 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54971 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_882_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_18
  simp only [← c54966, ← c54967, ← c54968] at e_882_18
  have e_882_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_19
  simp only [← c54969, ← c54970, k2, ← c54971] at e_882_19
  have hr := e_882_19
  simp only [e_882_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6715)) := by
  have c54972 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54973 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54974 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54975 := (publicBatchWrapper89_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54976 := (publicBatchWrapper89_copies1718 a h).1
  have c54977 := (publicBatchWrapper89_copies1718 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_0
  simp only [← c54972, ← c54973, ← c54974] at e_883_0
  have e_883_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_1
  simp only [← c54975, ← c54976, k2, ← c54977] at e_883_1
  have hr := e_883_1
  simp only [e_883_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6716)) := by
  have c54978 := (publicBatchWrapper89_copies1718 a h).2.2.1
  have c54979 := (publicBatchWrapper89_copies1718 a h).2.2.2.1
  have c54980 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.1
  have c54981 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.1
  have c54982 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.1
  have c54983 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_2
  simp only [← c54978, ← c54979, ← c54980] at e_883_2
  have e_883_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_3
  simp only [← c54981, ← c54982, k2, ← c54983] at e_883_3
  have hr := e_883_3
  simp only [e_883_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6717)) := by
  have c54984 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.1
  have c54985 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.1
  have c54986 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54987 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54988 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54989 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_4
  simp only [← c54984, ← c54985, ← c54986] at e_883_4
  have e_883_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_5
  simp only [← c54987, ← c54988, k2, ← c54989] at e_883_5
  have hr := e_883_5
  simp only [e_883_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6718)) := by
  have c54990 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54991 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54992 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54993 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54994 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54995 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_6
  simp only [← c54990, ← c54991, ← c54992] at e_883_6
  have e_883_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_7
  simp only [← c54993, ← c54994, k2, ← c54995] at e_883_7
  have hr := e_883_7
  simp only [e_883_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6719)) := by
  have c54996 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54997 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54998 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54999 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55000 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55001 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_8
  simp only [← c54996, ← c54997, ← c54998] at e_883_8
  have e_883_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_9
  simp only [← c54999, ← c55000, k2, ← c55001] at e_883_9
  have hr := e_883_9
  simp only [e_883_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6720)) := by
  have c55002 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55003 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55004 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55005 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55006 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55007 := (publicBatchWrapper89_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_10
  simp only [← c55002, ← c55003, ← c55004] at e_883_10
  have e_883_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_11
  simp only [← c55005, ← c55006, k2, ← c55007] at e_883_11
  have hr := e_883_11
  simp only [e_883_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6721)) := by
  have c55008 := (publicBatchWrapper89_copies1719 a h).1
  have c55009 := (publicBatchWrapper89_copies1719 a h).2.1
  have c55010 := (publicBatchWrapper89_copies1719 a h).2.2.1
  have c55011 := (publicBatchWrapper89_copies1719 a h).2.2.2.1
  have c55012 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.1
  have c55013 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_12
  simp only [← c55008, ← c55009, ← c55010] at e_883_12
  have e_883_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_13
  simp only [← c55011, ← c55012, k2, ← c55013] at e_883_13
  have hr := e_883_13
  simp only [e_883_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6722)) := by
  have c55014 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.1
  have c55015 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.1
  have c55016 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.1
  have c55017 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.1
  have c55018 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55019 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_14
  simp only [← c55014, ← c55015, ← c55016] at e_883_14
  have e_883_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_15
  simp only [← c55017, ← c55018, k2, ← c55019] at e_883_15
  have hr := e_883_15
  simp only [e_883_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6723)) := by
  have c55020 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55021 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55022 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55023 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55024 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55025 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_16
  simp only [← c55020, ← c55021, ← c55022] at e_883_16
  have e_883_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_17
  simp only [← c55023, ← c55024, k2, ← c55025] at e_883_17
  have hr := e_883_17
  simp only [e_883_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 883 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6724)) := by
  have c55026 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55027 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55028 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55029 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55030 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55031 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_883_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_18
  simp only [← c55026, ← c55027, ← c55028] at e_883_18
  have e_883_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_19
  simp only [← c55029, ← c55030, k2, ← c55031] at e_883_19
  have hr := e_883_19
  simp only [e_883_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6725)) := by
  have c55032 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55033 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55034 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55035 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55036 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55037 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_0
  simp only [← c55032, ← c55033, ← c55034] at e_884_0
  have e_884_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_1
  simp only [← c55035, ← c55036, k2, ← c55037] at e_884_1
  have hr := e_884_1
  simp only [e_884_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6726)) := by
  have c55038 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55039 := (publicBatchWrapper89_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55040 := (publicBatchWrapper89_copies1720 a h).1
  have c55041 := (publicBatchWrapper89_copies1720 a h).2.1
  have c55042 := (publicBatchWrapper89_copies1720 a h).2.2.1
  have c55043 := (publicBatchWrapper89_copies1720 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_2
  simp only [← c55038, ← c55039, ← c55040] at e_884_2
  have e_884_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_3
  simp only [← c55041, ← c55042, k2, ← c55043] at e_884_3
  have hr := e_884_3
  simp only [e_884_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6727)) := by
  have c55044 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.1
  have c55045 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.1
  have c55046 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.1
  have c55047 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.1
  have c55048 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.1
  have c55049 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_4
  simp only [← c55044, ← c55045, ← c55046] at e_884_4
  have e_884_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_5
  simp only [← c55047, ← c55048, k2, ← c55049] at e_884_5
  have hr := e_884_5
  simp only [e_884_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6728)) := by
  have c55050 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55051 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55052 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55053 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55054 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55055 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_6
  simp only [← c55050, ← c55051, ← c55052] at e_884_6
  have e_884_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_7
  simp only [← c55053, ← c55054, k2, ← c55055] at e_884_7
  have hr := e_884_7
  simp only [e_884_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6729)) := by
  have c55056 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55057 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55058 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55059 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55060 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55061 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_8
  simp only [← c55056, ← c55057, ← c55058] at e_884_8
  have e_884_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_9
  simp only [← c55059, ← c55060, k2, ← c55061] at e_884_9
  have hr := e_884_9
  simp only [e_884_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6730)) := by
  have c55062 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55063 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55064 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55065 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55066 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55067 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_10
  simp only [← c55062, ← c55063, ← c55064] at e_884_10
  have e_884_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_11
  simp only [← c55065, ← c55066, k2, ← c55067] at e_884_11
  have hr := e_884_11
  simp only [e_884_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6731)) := by
  have c55068 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55069 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55070 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55071 := (publicBatchWrapper89_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55072 := (publicBatchWrapper89_copies1721 a h).1
  have c55073 := (publicBatchWrapper89_copies1721 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_12
  simp only [← c55068, ← c55069, ← c55070] at e_884_12
  have e_884_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_13
  simp only [← c55071, ← c55072, k2, ← c55073] at e_884_13
  have hr := e_884_13
  simp only [e_884_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6732)) := by
  have c55074 := (publicBatchWrapper89_copies1721 a h).2.2.1
  have c55075 := (publicBatchWrapper89_copies1721 a h).2.2.2.1
  have c55076 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.1
  have c55077 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.1
  have c55078 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.1
  have c55079 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_14
  simp only [← c55074, ← c55075, ← c55076] at e_884_14
  have e_884_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_15
  simp only [← c55077, ← c55078, k2, ← c55079] at e_884_15
  have hr := e_884_15
  simp only [e_884_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6733)) := by
  have c55080 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.1
  have c55081 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.1
  have c55082 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55083 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55084 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55085 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_16
  simp only [← c55080, ← c55081, ← c55082] at e_884_16
  have e_884_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_17
  simp only [← c55083, ← c55084, k2, ← c55085] at e_884_17
  have hr := e_884_17
  simp only [e_884_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 884 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6734)) := by
  have c55086 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55087 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55088 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55089 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55090 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55091 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_884_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_18
  simp only [← c55086, ← c55087, ← c55088] at e_884_18
  have e_884_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_19
  simp only [← c55089, ← c55090, k2, ← c55091] at e_884_19
  have hr := e_884_19
  simp only [e_884_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6735)) := by
  have c55092 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55093 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55094 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55095 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55096 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55097 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_0
  simp only [← c55092, ← c55093, ← c55094] at e_885_0
  have e_885_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_1
  simp only [← c55095, ← c55096, k2, ← c55097] at e_885_1
  have hr := e_885_1
  simp only [e_885_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6736)) := by
  have c55098 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55099 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55100 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55101 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55102 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55103 := (publicBatchWrapper89_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_2
  simp only [← c55098, ← c55099, ← c55100] at e_885_2
  have e_885_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_3
  simp only [← c55101, ← c55102, k2, ← c55103] at e_885_3
  have hr := e_885_3
  simp only [e_885_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6737)) := by
  have c55104 := (publicBatchWrapper89_copies1722 a h).1
  have c55105 := (publicBatchWrapper89_copies1722 a h).2.1
  have c55106 := (publicBatchWrapper89_copies1722 a h).2.2.1
  have c55107 := (publicBatchWrapper89_copies1722 a h).2.2.2.1
  have c55108 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.1
  have c55109 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_4
  simp only [← c55104, ← c55105, ← c55106] at e_885_4
  have e_885_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_5
  simp only [← c55107, ← c55108, k2, ← c55109] at e_885_5
  have hr := e_885_5
  simp only [e_885_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6738)) := by
  have c55110 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.1
  have c55111 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.1
  have c55112 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.1
  have c55113 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.1
  have c55114 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55115 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_6
  simp only [← c55110, ← c55111, ← c55112] at e_885_6
  have e_885_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_7
  simp only [← c55113, ← c55114, k2, ← c55115] at e_885_7
  have hr := e_885_7
  simp only [e_885_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6739)) := by
  have c55116 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55117 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55118 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55119 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55120 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55121 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_8
  simp only [← c55116, ← c55117, ← c55118] at e_885_8
  have e_885_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_9
  simp only [← c55119, ← c55120, k2, ← c55121] at e_885_9
  have hr := e_885_9
  simp only [e_885_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6740)) := by
  have c55122 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55123 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55124 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55125 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55126 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55127 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_10
  simp only [← c55122, ← c55123, ← c55124] at e_885_10
  have e_885_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_11
  simp only [← c55125, ← c55126, k2, ← c55127] at e_885_11
  have hr := e_885_11
  simp only [e_885_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6741)) := by
  have c55128 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55129 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55130 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55131 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55132 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55133 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_12
  simp only [← c55128, ← c55129, ← c55130] at e_885_12
  have e_885_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_13
  simp only [← c55131, ← c55132, k2, ← c55133] at e_885_13
  have hr := e_885_13
  simp only [e_885_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6742)) := by
  have c55134 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55135 := (publicBatchWrapper89_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55136 := (publicBatchWrapper89_copies1723 a h).1
  have c55137 := (publicBatchWrapper89_copies1723 a h).2.1
  have c55138 := (publicBatchWrapper89_copies1723 a h).2.2.1
  have c55139 := (publicBatchWrapper89_copies1723 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_14
  simp only [← c55134, ← c55135, ← c55136] at e_885_14
  have e_885_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_15
  simp only [← c55137, ← c55138, k2, ← c55139] at e_885_15
  have hr := e_885_15
  simp only [e_885_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6743)) := by
  have c55140 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.1
  have c55141 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.1
  have c55142 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.1
  have c55143 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.1
  have c55144 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.1
  have c55145 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_16
  simp only [← c55140, ← c55141, ← c55142] at e_885_16
  have e_885_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_17
  simp only [← c55143, ← c55144, k2, ← c55145] at e_885_17
  have hr := e_885_17
  simp only [e_885_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 885 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6744)) := by
  have c55146 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55147 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55148 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55149 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55150 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55151 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_885_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_18
  simp only [← c55146, ← c55147, ← c55148] at e_885_18
  have e_885_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_19
  simp only [← c55149, ← c55150, k2, ← c55151] at e_885_19
  have hr := e_885_19
  simp only [e_885_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6745)) := by
  have c55152 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55153 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55154 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55155 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55156 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55157 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_0
  simp only [← c55152, ← c55153, ← c55154] at e_886_0
  have e_886_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_1
  simp only [← c55155, ← c55156, k2, ← c55157] at e_886_1
  have hr := e_886_1
  simp only [e_886_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6746)) := by
  have c55158 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55159 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55160 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55161 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55162 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55163 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_2
  simp only [← c55158, ← c55159, ← c55160] at e_886_2
  have e_886_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_3
  simp only [← c55161, ← c55162, k2, ← c55163] at e_886_3
  have hr := e_886_3
  simp only [e_886_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6747)) := by
  have c55164 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55165 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55166 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55167 := (publicBatchWrapper89_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55168 := (publicBatchWrapper89_copies1724 a h).1
  have c55169 := (publicBatchWrapper89_copies1724 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_4
  simp only [← c55164, ← c55165, ← c55166] at e_886_4
  have e_886_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_5
  simp only [← c55167, ← c55168, k2, ← c55169] at e_886_5
  have hr := e_886_5
  simp only [e_886_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6748)) := by
  have c55170 := (publicBatchWrapper89_copies1724 a h).2.2.1
  have c55171 := (publicBatchWrapper89_copies1724 a h).2.2.2.1
  have c55172 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.1
  have c55173 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.1
  have c55174 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.1
  have c55175 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_6
  simp only [← c55170, ← c55171, ← c55172] at e_886_6
  have e_886_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_7
  simp only [← c55173, ← c55174, k2, ← c55175] at e_886_7
  have hr := e_886_7
  simp only [e_886_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6749)) := by
  have c55176 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.1
  have c55177 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.1
  have c55178 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55179 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55180 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55181 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_8
  simp only [← c55176, ← c55177, ← c55178] at e_886_8
  have e_886_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_9
  simp only [← c55179, ← c55180, k2, ← c55181] at e_886_9
  have hr := e_886_9
  simp only [e_886_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6750)) := by
  have c55182 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55183 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55184 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55185 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55186 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55187 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_10
  simp only [← c55182, ← c55183, ← c55184] at e_886_10
  have e_886_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_11
  simp only [← c55185, ← c55186, k2, ← c55187] at e_886_11
  have hr := e_886_11
  simp only [e_886_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6751)) := by
  have c55188 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55189 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55190 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55191 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55192 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55193 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_12
  simp only [← c55188, ← c55189, ← c55190] at e_886_12
  have e_886_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_13
  simp only [← c55191, ← c55192, k2, ← c55193] at e_886_13
  have hr := e_886_13
  simp only [e_886_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6752)) := by
  have c55194 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55195 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55196 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55197 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55198 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55199 := (publicBatchWrapper89_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_14
  simp only [← c55194, ← c55195, ← c55196] at e_886_14
  have e_886_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_15
  simp only [← c55197, ← c55198, k2, ← c55199] at e_886_15
  have hr := e_886_15
  simp only [e_886_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6753)) := by
  have c55200 := (publicBatchWrapper89_copies1725 a h).1
  have c55201 := (publicBatchWrapper89_copies1725 a h).2.1
  have c55202 := (publicBatchWrapper89_copies1725 a h).2.2.1
  have c55203 := (publicBatchWrapper89_copies1725 a h).2.2.2.1
  have c55204 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.1
  have c55205 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_16
  simp only [← c55200, ← c55201, ← c55202] at e_886_16
  have e_886_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_17
  simp only [← c55203, ← c55204, k2, ← c55205] at e_886_17
  have hr := e_886_17
  simp only [e_886_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 886 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6754)) := by
  have c55206 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.1
  have c55207 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.1
  have c55208 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.1
  have c55209 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.1
  have c55210 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55211 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_886_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_18
  simp only [← c55206, ← c55207, ← c55208] at e_886_18
  have e_886_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_19
  simp only [← c55209, ← c55210, k2, ← c55211] at e_886_19
  have hr := e_886_19
  simp only [e_886_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6755)) := by
  have c55212 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55213 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55214 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55215 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55216 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55217 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_0
  simp only [← c55212, ← c55213, ← c55214] at e_887_0
  have e_887_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_1
  simp only [← c55215, ← c55216, k2, ← c55217] at e_887_1
  have hr := e_887_1
  simp only [e_887_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6756)) := by
  have c55218 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55219 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55220 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55221 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55222 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55223 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_2
  simp only [← c55218, ← c55219, ← c55220] at e_887_2
  have e_887_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_3
  simp only [← c55221, ← c55222, k2, ← c55223] at e_887_3
  have hr := e_887_3
  simp only [e_887_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6757)) := by
  have c55224 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55225 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55226 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55227 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55228 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55229 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_4
  simp only [← c55224, ← c55225, ← c55226] at e_887_4
  have e_887_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_5
  simp only [← c55227, ← c55228, k2, ← c55229] at e_887_5
  have hr := e_887_5
  simp only [e_887_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6758)) := by
  have c55230 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55231 := (publicBatchWrapper89_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55232 := (publicBatchWrapper89_copies1726 a h).1
  have c55233 := (publicBatchWrapper89_copies1726 a h).2.1
  have c55234 := (publicBatchWrapper89_copies1726 a h).2.2.1
  have c55235 := (publicBatchWrapper89_copies1726 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_6
  simp only [← c55230, ← c55231, ← c55232] at e_887_6
  have e_887_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_7
  simp only [← c55233, ← c55234, k2, ← c55235] at e_887_7
  have hr := e_887_7
  simp only [e_887_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6759)) := by
  have c55236 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.1
  have c55237 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.1
  have c55238 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.1
  have c55239 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.1
  have c55240 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.1
  have c55241 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_8
  simp only [← c55236, ← c55237, ← c55238] at e_887_8
  have e_887_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_9
  simp only [← c55239, ← c55240, k2, ← c55241] at e_887_9
  have hr := e_887_9
  simp only [e_887_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6760)) := by
  have c55242 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55243 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55244 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55245 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55246 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55247 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_10
  simp only [← c55242, ← c55243, ← c55244] at e_887_10
  have e_887_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_11
  simp only [← c55245, ← c55246, k2, ← c55247] at e_887_11
  have hr := e_887_11
  simp only [e_887_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6761)) := by
  have c55248 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55249 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55250 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55251 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55252 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55253 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_12
  simp only [← c55248, ← c55249, ← c55250] at e_887_12
  have e_887_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_13
  simp only [← c55251, ← c55252, k2, ← c55253] at e_887_13
  have hr := e_887_13
  simp only [e_887_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6762)) := by
  have c55254 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55255 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55256 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55257 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55258 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55259 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_14
  simp only [← c55254, ← c55255, ← c55256] at e_887_14
  have e_887_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_15
  simp only [← c55257, ← c55258, k2, ← c55259] at e_887_15
  have hr := e_887_15
  simp only [e_887_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6763)) := by
  have c55260 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55261 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55262 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55263 := (publicBatchWrapper89_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55264 := (publicBatchWrapper89_copies1727 a h).1
  have c55265 := (publicBatchWrapper89_copies1727 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_16
  simp only [← c55260, ← c55261, ← c55262] at e_887_16
  have e_887_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_17
  simp only [← c55263, ← c55264, k2, ← c55265] at e_887_17
  have hr := e_887_17
  simp only [e_887_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 887 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6764)) := by
  have c55266 := (publicBatchWrapper89_copies1727 a h).2.2.1
  have c55267 := (publicBatchWrapper89_copies1727 a h).2.2.2.1
  have c55268 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.1
  have c55269 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.1
  have c55270 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.1
  have c55271 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_887_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_18
  simp only [← c55266, ← c55267, ← c55268] at e_887_18
  have e_887_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_19
  simp only [← c55269, ← c55270, k2, ← c55271] at e_887_19
  have hr := e_887_19
  simp only [e_887_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6765)) := by
  have c55272 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.1
  have c55273 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.1
  have c55274 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55275 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55276 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55277 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_0
  simp only [← c55272, ← c55273, ← c55274] at e_888_0
  have e_888_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_1
  simp only [← c55275, ← c55276, k2, ← c55277] at e_888_1
  have hr := e_888_1
  simp only [e_888_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6766)) := by
  have c55278 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55279 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55280 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55281 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55282 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55283 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_2
  simp only [← c55278, ← c55279, ← c55280] at e_888_2
  have e_888_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_3
  simp only [← c55281, ← c55282, k2, ← c55283] at e_888_3
  have hr := e_888_3
  simp only [e_888_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6767)) := by
  have c55284 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55285 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55286 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55287 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55288 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55289 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_4
  simp only [← c55284, ← c55285, ← c55286] at e_888_4
  have e_888_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_5
  simp only [← c55287, ← c55288, k2, ← c55289] at e_888_5
  have hr := e_888_5
  simp only [e_888_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6768)) := by
  have c55290 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55291 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55292 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55293 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55294 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55295 := (publicBatchWrapper89_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_6
  simp only [← c55290, ← c55291, ← c55292] at e_888_6
  have e_888_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_7
  simp only [← c55293, ← c55294, k2, ← c55295] at e_888_7
  have hr := e_888_7
  simp only [e_888_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6769)) := by
  have c55296 := (publicBatchWrapper89_copies1728 a h).1
  have c55297 := (publicBatchWrapper89_copies1728 a h).2.1
  have c55298 := (publicBatchWrapper89_copies1728 a h).2.2.1
  have c55299 := (publicBatchWrapper89_copies1728 a h).2.2.2.1
  have c55300 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.1
  have c55301 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_8
  simp only [← c55296, ← c55297, ← c55298] at e_888_8
  have e_888_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_9
  simp only [← c55299, ← c55300, k2, ← c55301] at e_888_9
  have hr := e_888_9
  simp only [e_888_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6810)) := by
  have c55302 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.1
  have c55303 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.1
  have c55304 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.1
  have c55305 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.1
  have c55306 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55307 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_10
  simp only [← c55302, ← c55303, ← c55304] at e_888_10
  have e_888_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_11
  simp only [← c55305, ← c55306, k2, ← c55307] at e_888_11
  have hr := e_888_11
  simp only [e_888_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6811)) := by
  have c55308 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55309 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55310 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55311 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55312 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55313 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_12
  simp only [← c55308, ← c55309, ← c55310] at e_888_12
  have e_888_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_13
  simp only [← c55311, ← c55312, k2, ← c55313] at e_888_13
  have hr := e_888_13
  simp only [e_888_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6812)) := by
  have c55314 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55315 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55316 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55317 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55318 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55319 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_14
  simp only [← c55314, ← c55315, ← c55316] at e_888_14
  have e_888_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_15
  simp only [← c55317, ← c55318, k2, ← c55319] at e_888_15
  have hr := e_888_15
  simp only [e_888_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6813)) := by
  have c55320 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55321 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55322 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55323 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55324 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55325 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_16
  simp only [← c55320, ← c55321, ← c55322] at e_888_16
  have e_888_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_17
  simp only [← c55323, ← c55324, k2, ← c55325] at e_888_17
  have hr := e_888_17
  simp only [e_888_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 888 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6814)) := by
  have c55326 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55327 := (publicBatchWrapper89_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55328 := (publicBatchWrapper89_copies1729 a h).1
  have c55329 := (publicBatchWrapper89_copies1729 a h).2.1
  have c55330 := (publicBatchWrapper89_copies1729 a h).2.2.1
  have c55331 := (publicBatchWrapper89_copies1729 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_888_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_18
  simp only [← c55326, ← c55327, ← c55328] at e_888_18
  have e_888_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_19
  simp only [← c55329, ← c55330, k2, ← c55331] at e_888_19
  have hr := e_888_19
  simp only [e_888_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6815)) := by
  have c55332 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.1
  have c55333 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.1
  have c55334 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.1
  have c55335 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.1
  have c55336 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.1
  have c55337 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_0
  simp only [← c55332, ← c55333, ← c55334] at e_889_0
  have e_889_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_1
  simp only [← c55335, ← c55336, k2, ← c55337] at e_889_1
  have hr := e_889_1
  simp only [e_889_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6816)) := by
  have c55338 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55339 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55340 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55341 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55342 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55343 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_2
  simp only [← c55338, ← c55339, ← c55340] at e_889_2
  have e_889_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_3
  simp only [← c55341, ← c55342, k2, ← c55343] at e_889_3
  have hr := e_889_3
  simp only [e_889_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6817)) := by
  have c55344 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55345 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55346 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55347 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55348 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55349 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_4
  simp only [← c55344, ← c55345, ← c55346] at e_889_4
  have e_889_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_5
  simp only [← c55347, ← c55348, k2, ← c55349] at e_889_5
  have hr := e_889_5
  simp only [e_889_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6818)) := by
  have c55350 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55351 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55352 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55353 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55354 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55355 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_6
  simp only [← c55350, ← c55351, ← c55352] at e_889_6
  have e_889_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_7
  simp only [← c55353, ← c55354, k2, ← c55355] at e_889_7
  have hr := e_889_7
  simp only [e_889_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6819)) := by
  have c55356 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55357 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55358 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55359 := (publicBatchWrapper89_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55360 := (publicBatchWrapper89_copies1730 a h).1
  have c55361 := (publicBatchWrapper89_copies1730 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_8
  simp only [← c55356, ← c55357, ← c55358] at e_889_8
  have e_889_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_9
  simp only [← c55359, ← c55360, k2, ← c55361] at e_889_9
  have hr := e_889_9
  simp only [e_889_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6820)) := by
  have c55362 := (publicBatchWrapper89_copies1730 a h).2.2.1
  have c55363 := (publicBatchWrapper89_copies1730 a h).2.2.2.1
  have c55364 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.1
  have c55365 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.1
  have c55366 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.1
  have c55367 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_10
  simp only [← c55362, ← c55363, ← c55364] at e_889_10
  have e_889_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_11
  simp only [← c55365, ← c55366, k2, ← c55367] at e_889_11
  have hr := e_889_11
  simp only [e_889_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6821)) := by
  have c55368 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.1
  have c55369 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.1
  have c55370 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55371 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55372 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55373 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_12
  simp only [← c55368, ← c55369, ← c55370] at e_889_12
  have e_889_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_13
  simp only [← c55371, ← c55372, k2, ← c55373] at e_889_13
  have hr := e_889_13
  simp only [e_889_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6822)) := by
  have c55374 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55375 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55376 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55377 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55378 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55379 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_14
  simp only [← c55374, ← c55375, ← c55376] at e_889_14
  have e_889_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_15
  simp only [← c55377, ← c55378, k2, ← c55379] at e_889_15
  have hr := e_889_15
  simp only [e_889_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6823)) := by
  have c55380 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55381 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55382 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55383 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55384 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55385 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_16
  simp only [← c55380, ← c55381, ← c55382] at e_889_16
  have e_889_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_17
  simp only [← c55383, ← c55384, k2, ← c55385] at e_889_17
  have hr := e_889_17
  simp only [e_889_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 889 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6824)) := by
  have c55386 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55387 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55388 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55389 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55390 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55391 := (publicBatchWrapper89_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_889_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_18
  simp only [← c55386, ← c55387, ← c55388] at e_889_18
  have e_889_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_19
  simp only [← c55389, ← c55390, k2, ← c55391] at e_889_19
  have hr := e_889_19
  simp only [e_889_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6825)) := by
  have c55392 := (publicBatchWrapper89_copies1731 a h).1
  have c55393 := (publicBatchWrapper89_copies1731 a h).2.1
  have c55394 := (publicBatchWrapper89_copies1731 a h).2.2.1
  have c55395 := (publicBatchWrapper89_copies1731 a h).2.2.2.1
  have c55396 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.1
  have c55397 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_0
  simp only [← c55392, ← c55393, ← c55394] at e_890_0
  have e_890_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_1
  simp only [← c55395, ← c55396, k2, ← c55397] at e_890_1
  have hr := e_890_1
  simp only [e_890_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6826)) := by
  have c55398 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.1
  have c55399 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.1
  have c55400 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.1
  have c55401 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.1
  have c55402 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55403 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_2
  simp only [← c55398, ← c55399, ← c55400] at e_890_2
  have e_890_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_3
  simp only [← c55401, ← c55402, k2, ← c55403] at e_890_3
  have hr := e_890_3
  simp only [e_890_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6827)) := by
  have c55404 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55405 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55406 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55407 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55408 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55409 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_4
  simp only [← c55404, ← c55405, ← c55406] at e_890_4
  have e_890_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_5
  simp only [← c55407, ← c55408, k2, ← c55409] at e_890_5
  have hr := e_890_5
  simp only [e_890_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6828)) := by
  have c55410 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55411 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55412 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55413 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55414 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55415 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_6
  simp only [← c55410, ← c55411, ← c55412] at e_890_6
  have e_890_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_7
  simp only [← c55413, ← c55414, k2, ← c55415] at e_890_7
  have hr := e_890_7
  simp only [e_890_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6829)) := by
  have c55416 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55417 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55418 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55419 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55420 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55421 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_8
  simp only [← c55416, ← c55417, ← c55418] at e_890_8
  have e_890_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_9
  simp only [← c55419, ← c55420, k2, ← c55421] at e_890_9
  have hr := e_890_9
  simp only [e_890_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6830)) := by
  have c55422 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55423 := (publicBatchWrapper89_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55424 := (publicBatchWrapper89_copies1732 a h).1
  have c55425 := (publicBatchWrapper89_copies1732 a h).2.1
  have c55426 := (publicBatchWrapper89_copies1732 a h).2.2.1
  have c55427 := (publicBatchWrapper89_copies1732 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_10
  simp only [← c55422, ← c55423, ← c55424] at e_890_10
  have e_890_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_11
  simp only [← c55425, ← c55426, k2, ← c55427] at e_890_11
  have hr := e_890_11
  simp only [e_890_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6831)) := by
  have c55428 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.1
  have c55429 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.1
  have c55430 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.1
  have c55431 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.1
  have c55432 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.1
  have c55433 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_12
  simp only [← c55428, ← c55429, ← c55430] at e_890_12
  have e_890_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_13
  simp only [← c55431, ← c55432, k2, ← c55433] at e_890_13
  have hr := e_890_13
  simp only [e_890_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6832)) := by
  have c55434 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55435 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55436 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55437 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55438 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55439 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_14
  simp only [← c55434, ← c55435, ← c55436] at e_890_14
  have e_890_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_15
  simp only [← c55437, ← c55438, k2, ← c55439] at e_890_15
  have hr := e_890_15
  simp only [e_890_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6833)) := by
  have c55440 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55441 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55442 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55443 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55444 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55445 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_16
  simp only [← c55440, ← c55441, ← c55442] at e_890_16
  have e_890_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_17
  simp only [← c55443, ← c55444, k2, ← c55445] at e_890_17
  have hr := e_890_17
  simp only [e_890_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 890 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6834)) := by
  have c55446 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55447 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55448 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55449 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55450 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55451 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_890_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_18
  simp only [← c55446, ← c55447, ← c55448] at e_890_18
  have e_890_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_19
  simp only [← c55449, ← c55450, k2, ← c55451] at e_890_19
  have hr := e_890_19
  simp only [e_890_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6835)) := by
  have c55452 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55453 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55454 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55455 := (publicBatchWrapper89_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55456 := (publicBatchWrapper89_copies1733 a h).1
  have c55457 := (publicBatchWrapper89_copies1733 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_0
  simp only [← c55452, ← c55453, ← c55454] at e_891_0
  have e_891_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_1
  simp only [← c55455, ← c55456, k2, ← c55457] at e_891_1
  have hr := e_891_1
  simp only [e_891_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6836)) := by
  have c55458 := (publicBatchWrapper89_copies1733 a h).2.2.1
  have c55459 := (publicBatchWrapper89_copies1733 a h).2.2.2.1
  have c55460 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.1
  have c55461 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.1
  have c55462 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.1
  have c55463 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_2
  simp only [← c55458, ← c55459, ← c55460] at e_891_2
  have e_891_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_3
  simp only [← c55461, ← c55462, k2, ← c55463] at e_891_3
  have hr := e_891_3
  simp only [e_891_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6837)) := by
  have c55464 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.1
  have c55465 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.1
  have c55466 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55467 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55468 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55469 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_4
  simp only [← c55464, ← c55465, ← c55466] at e_891_4
  have e_891_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_5
  simp only [← c55467, ← c55468, k2, ← c55469] at e_891_5
  have hr := e_891_5
  simp only [e_891_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6838)) := by
  have c55470 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55471 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55472 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55473 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55474 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55475 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_6
  simp only [← c55470, ← c55471, ← c55472] at e_891_6
  have e_891_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_7
  simp only [← c55473, ← c55474, k2, ← c55475] at e_891_7
  have hr := e_891_7
  simp only [e_891_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6839)) := by
  have c55476 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55477 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55478 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55479 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55480 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55481 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_8
  simp only [← c55476, ← c55477, ← c55478] at e_891_8
  have e_891_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_9
  simp only [← c55479, ← c55480, k2, ← c55481] at e_891_9
  have hr := e_891_9
  simp only [e_891_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6840)) := by
  have c55482 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55483 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55484 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55485 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55486 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55487 := (publicBatchWrapper89_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_10
  simp only [← c55482, ← c55483, ← c55484] at e_891_10
  have e_891_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_11
  simp only [← c55485, ← c55486, k2, ← c55487] at e_891_11
  have hr := e_891_11
  simp only [e_891_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6841)) := by
  have c55488 := (publicBatchWrapper89_copies1734 a h).1
  have c55489 := (publicBatchWrapper89_copies1734 a h).2.1
  have c55490 := (publicBatchWrapper89_copies1734 a h).2.2.1
  have c55491 := (publicBatchWrapper89_copies1734 a h).2.2.2.1
  have c55492 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.1
  have c55493 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_12
  simp only [← c55488, ← c55489, ← c55490] at e_891_12
  have e_891_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_13
  simp only [← c55491, ← c55492, k2, ← c55493] at e_891_13
  have hr := e_891_13
  simp only [e_891_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6842)) := by
  have c55494 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.1
  have c55495 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.1
  have c55496 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.1
  have c55497 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.1
  have c55498 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55499 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_14
  simp only [← c55494, ← c55495, ← c55496] at e_891_14
  have e_891_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_15
  simp only [← c55497, ← c55498, k2, ← c55499] at e_891_15
  have hr := e_891_15
  simp only [e_891_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6843)) := by
  have c55500 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55501 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55502 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55503 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55504 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55505 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_16
  simp only [← c55500, ← c55501, ← c55502] at e_891_16
  have e_891_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_17
  simp only [← c55503, ← c55504, k2, ← c55505] at e_891_17
  have hr := e_891_17
  simp only [e_891_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 891 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6844)) := by
  have c55506 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55507 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55508 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55509 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55510 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55511 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_891_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_18
  simp only [← c55506, ← c55507, ← c55508] at e_891_18
  have e_891_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_19
  simp only [← c55509, ← c55510, k2, ← c55511] at e_891_19
  have hr := e_891_19
  simp only [e_891_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6845)) := by
  have c55512 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55513 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55514 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55515 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55516 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55517 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_0
  simp only [← c55512, ← c55513, ← c55514] at e_892_0
  have e_892_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_1
  simp only [← c55515, ← c55516, k2, ← c55517] at e_892_1
  have hr := e_892_1
  simp only [e_892_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6846)) := by
  have c55518 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55519 := (publicBatchWrapper89_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55520 := (publicBatchWrapper89_copies1735 a h).1
  have c55521 := (publicBatchWrapper89_copies1735 a h).2.1
  have c55522 := (publicBatchWrapper89_copies1735 a h).2.2.1
  have c55523 := (publicBatchWrapper89_copies1735 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_2
  simp only [← c55518, ← c55519, ← c55520] at e_892_2
  have e_892_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_3
  simp only [← c55521, ← c55522, k2, ← c55523] at e_892_3
  have hr := e_892_3
  simp only [e_892_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6847)) := by
  have c55524 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.1
  have c55525 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.1
  have c55526 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.1
  have c55527 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.1
  have c55528 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.1
  have c55529 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_4
  simp only [← c55524, ← c55525, ← c55526] at e_892_4
  have e_892_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_5
  simp only [← c55527, ← c55528, k2, ← c55529] at e_892_5
  have hr := e_892_5
  simp only [e_892_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6848)) := by
  have c55530 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55531 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55532 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55533 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55534 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55535 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_6
  simp only [← c55530, ← c55531, ← c55532] at e_892_6
  have e_892_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_7
  simp only [← c55533, ← c55534, k2, ← c55535] at e_892_7
  have hr := e_892_7
  simp only [e_892_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6849)) := by
  have c55536 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55537 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55538 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55539 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55540 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55541 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_8
  simp only [← c55536, ← c55537, ← c55538] at e_892_8
  have e_892_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_9
  simp only [← c55539, ← c55540, k2, ← c55541] at e_892_9
  have hr := e_892_9
  simp only [e_892_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6850)) := by
  have c55542 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55543 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55544 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55545 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55546 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55547 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_10
  simp only [← c55542, ← c55543, ← c55544] at e_892_10
  have e_892_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_11
  simp only [← c55545, ← c55546, k2, ← c55547] at e_892_11
  have hr := e_892_11
  simp only [e_892_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6851)) := by
  have c55548 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55549 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55550 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55551 := (publicBatchWrapper89_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55552 := (publicBatchWrapper89_copies1736 a h).1
  have c55553 := (publicBatchWrapper89_copies1736 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_12
  simp only [← c55548, ← c55549, ← c55550] at e_892_12
  have e_892_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_13
  simp only [← c55551, ← c55552, k2, ← c55553] at e_892_13
  have hr := e_892_13
  simp only [e_892_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6852)) := by
  have c55554 := (publicBatchWrapper89_copies1736 a h).2.2.1
  have c55555 := (publicBatchWrapper89_copies1736 a h).2.2.2.1
  have c55556 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.1
  have c55557 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.1
  have c55558 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.1
  have c55559 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_14
  simp only [← c55554, ← c55555, ← c55556] at e_892_14
  have e_892_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_15
  simp only [← c55557, ← c55558, k2, ← c55559] at e_892_15
  have hr := e_892_15
  simp only [e_892_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6853)) := by
  have c55560 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.1
  have c55561 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.1
  have c55562 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55563 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55564 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55565 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_16
  simp only [← c55560, ← c55561, ← c55562] at e_892_16
  have e_892_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_17
  simp only [← c55563, ← c55564, k2, ← c55565] at e_892_17
  have hr := e_892_17
  simp only [e_892_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 892 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6854)) := by
  have c55566 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55567 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55568 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55569 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55570 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55571 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_892_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_18
  simp only [← c55566, ← c55567, ← c55568] at e_892_18
  have e_892_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_19
  simp only [← c55569, ← c55570, k2, ← c55571] at e_892_19
  have hr := e_892_19
  simp only [e_892_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6855)) := by
  have c55572 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55573 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55574 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55575 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55576 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55577 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_0
  simp only [← c55572, ← c55573, ← c55574] at e_893_0
  have e_893_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_1
  simp only [← c55575, ← c55576, k2, ← c55577] at e_893_1
  have hr := e_893_1
  simp only [e_893_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6856)) := by
  have c55578 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55579 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55580 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55581 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55582 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55583 := (publicBatchWrapper89_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_2
  simp only [← c55578, ← c55579, ← c55580] at e_893_2
  have e_893_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_3
  simp only [← c55581, ← c55582, k2, ← c55583] at e_893_3
  have hr := e_893_3
  simp only [e_893_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6857)) := by
  have c55584 := (publicBatchWrapper89_copies1737 a h).1
  have c55585 := (publicBatchWrapper89_copies1737 a h).2.1
  have c55586 := (publicBatchWrapper89_copies1737 a h).2.2.1
  have c55587 := (publicBatchWrapper89_copies1737 a h).2.2.2.1
  have c55588 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.1
  have c55589 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_4
  simp only [← c55584, ← c55585, ← c55586] at e_893_4
  have e_893_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_5
  simp only [← c55587, ← c55588, k2, ← c55589] at e_893_5
  have hr := e_893_5
  simp only [e_893_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6858)) := by
  have c55590 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.1
  have c55591 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.1
  have c55592 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.1
  have c55593 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.1
  have c55594 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55595 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_6
  simp only [← c55590, ← c55591, ← c55592] at e_893_6
  have e_893_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_7
  simp only [← c55593, ← c55594, k2, ← c55595] at e_893_7
  have hr := e_893_7
  simp only [e_893_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6859)) := by
  have c55596 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55597 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55598 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55599 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55600 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55601 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_8
  simp only [← c55596, ← c55597, ← c55598] at e_893_8
  have e_893_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_9
  simp only [← c55599, ← c55600, k2, ← c55601] at e_893_9
  have hr := e_893_9
  simp only [e_893_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6860)) := by
  have c55602 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55603 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55604 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55605 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55606 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55607 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_10
  simp only [← c55602, ← c55603, ← c55604] at e_893_10
  have e_893_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_11
  simp only [← c55605, ← c55606, k2, ← c55607] at e_893_11
  have hr := e_893_11
  simp only [e_893_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6861)) := by
  have c55608 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55609 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55610 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55611 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55612 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55613 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_12
  simp only [← c55608, ← c55609, ← c55610] at e_893_12
  have e_893_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_13
  simp only [← c55611, ← c55612, k2, ← c55613] at e_893_13
  have hr := e_893_13
  simp only [e_893_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6862)) := by
  have c55614 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55615 := (publicBatchWrapper89_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55616 := (publicBatchWrapper89_copies1738 a h).1
  have c55617 := (publicBatchWrapper89_copies1738 a h).2.1
  have c55618 := (publicBatchWrapper89_copies1738 a h).2.2.1
  have c55619 := (publicBatchWrapper89_copies1738 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_14
  simp only [← c55614, ← c55615, ← c55616] at e_893_14
  have e_893_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_15
  simp only [← c55617, ← c55618, k2, ← c55619] at e_893_15
  have hr := e_893_15
  simp only [e_893_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6863)) := by
  have c55620 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.1
  have c55621 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.1
  have c55622 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.1
  have c55623 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.1
  have c55624 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.1
  have c55625 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_16
  simp only [← c55620, ← c55621, ← c55622] at e_893_16
  have e_893_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_17
  simp only [← c55623, ← c55624, k2, ← c55625] at e_893_17
  have hr := e_893_17
  simp only [e_893_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 893 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6864)) := by
  have c55626 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55627 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55628 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55629 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55630 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55631 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_893_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_18
  simp only [← c55626, ← c55627, ← c55628] at e_893_18
  have e_893_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_19
  simp only [← c55629, ← c55630, k2, ← c55631] at e_893_19
  have hr := e_893_19
  simp only [e_893_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6865)) := by
  have c55632 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55633 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55634 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55635 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55636 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55637 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_0
  simp only [← c55632, ← c55633, ← c55634] at e_894_0
  have e_894_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_1
  simp only [← c55635, ← c55636, k2, ← c55637] at e_894_1
  have hr := e_894_1
  simp only [e_894_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6866)) := by
  have c55638 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55639 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55640 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55641 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55642 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55643 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_2
  simp only [← c55638, ← c55639, ← c55640] at e_894_2
  have e_894_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_3
  simp only [← c55641, ← c55642, k2, ← c55643] at e_894_3
  have hr := e_894_3
  simp only [e_894_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6867)) := by
  have c55644 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55645 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55646 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55647 := (publicBatchWrapper89_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55648 := (publicBatchWrapper89_copies1739 a h).1
  have c55649 := (publicBatchWrapper89_copies1739 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_4
  simp only [← c55644, ← c55645, ← c55646] at e_894_4
  have e_894_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_5
  simp only [← c55647, ← c55648, k2, ← c55649] at e_894_5
  have hr := e_894_5
  simp only [e_894_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6868)) := by
  have c55650 := (publicBatchWrapper89_copies1739 a h).2.2.1
  have c55651 := (publicBatchWrapper89_copies1739 a h).2.2.2.1
  have c55652 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.1
  have c55653 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.1
  have c55654 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.1
  have c55655 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_6
  simp only [← c55650, ← c55651, ← c55652] at e_894_6
  have e_894_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_7
  simp only [← c55653, ← c55654, k2, ← c55655] at e_894_7
  have hr := e_894_7
  simp only [e_894_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6869)) := by
  have c55656 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.1
  have c55657 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.1
  have c55658 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55659 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55660 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55661 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_8
  simp only [← c55656, ← c55657, ← c55658] at e_894_8
  have e_894_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_9
  simp only [← c55659, ← c55660, k2, ← c55661] at e_894_9
  have hr := e_894_9
  simp only [e_894_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6870)) := by
  have c55662 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55663 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55664 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55665 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55666 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55667 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_10
  simp only [← c55662, ← c55663, ← c55664] at e_894_10
  have e_894_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_11
  simp only [← c55665, ← c55666, k2, ← c55667] at e_894_11
  have hr := e_894_11
  simp only [e_894_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6871)) := by
  have c55668 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55669 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55670 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55671 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55672 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55673 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_12
  simp only [← c55668, ← c55669, ← c55670] at e_894_12
  have e_894_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_13
  simp only [← c55671, ← c55672, k2, ← c55673] at e_894_13
  have hr := e_894_13
  simp only [e_894_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6872)) := by
  have c55674 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55675 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55676 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55677 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55678 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55679 := (publicBatchWrapper89_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_14
  simp only [← c55674, ← c55675, ← c55676] at e_894_14
  have e_894_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_15
  simp only [← c55677, ← c55678, k2, ← c55679] at e_894_15
  have hr := e_894_15
  simp only [e_894_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6873)) := by
  have c55680 := (publicBatchWrapper89_copies1740 a h).1
  have c55681 := (publicBatchWrapper89_copies1740 a h).2.1
  have c55682 := (publicBatchWrapper89_copies1740 a h).2.2.1
  have c55683 := (publicBatchWrapper89_copies1740 a h).2.2.2.1
  have c55684 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.1
  have c55685 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_16
  simp only [← c55680, ← c55681, ← c55682] at e_894_16
  have e_894_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_17
  simp only [← c55683, ← c55684, k2, ← c55685] at e_894_17
  have hr := e_894_17
  simp only [e_894_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 894 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6874)) := by
  have c55686 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.1
  have c55687 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.1
  have c55688 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.1
  have c55689 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.1
  have c55690 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55691 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_894_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_18
  simp only [← c55686, ← c55687, ← c55688] at e_894_18
  have e_894_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_19
  simp only [← c55689, ← c55690, k2, ← c55691] at e_894_19
  have hr := e_894_19
  simp only [e_894_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6875)) := by
  have c55692 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55693 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55694 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55695 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55696 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55697 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_0
  simp only [← c55692, ← c55693, ← c55694] at e_895_0
  have e_895_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_1
  simp only [← c55695, ← c55696, k2, ← c55697] at e_895_1
  have hr := e_895_1
  simp only [e_895_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6876)) := by
  have c55698 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55699 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55700 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55701 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55702 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55703 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_2
  simp only [← c55698, ← c55699, ← c55700] at e_895_2
  have e_895_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_3
  simp only [← c55701, ← c55702, k2, ← c55703] at e_895_3
  have hr := e_895_3
  simp only [e_895_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6877)) := by
  have c55704 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55705 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55706 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55707 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55708 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55709 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_4
  simp only [← c55704, ← c55705, ← c55706] at e_895_4
  have e_895_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_5
  simp only [← c55707, ← c55708, k2, ← c55709] at e_895_5
  have hr := e_895_5
  simp only [e_895_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6878)) := by
  have c55710 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55711 := (publicBatchWrapper89_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55712 := (publicBatchWrapper89_copies1741 a h).1
  have c55713 := (publicBatchWrapper89_copies1741 a h).2.1
  have c55714 := (publicBatchWrapper89_copies1741 a h).2.2.1
  have c55715 := (publicBatchWrapper89_copies1741 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_6
  simp only [← c55710, ← c55711, ← c55712] at e_895_6
  have e_895_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_7
  simp only [← c55713, ← c55714, k2, ← c55715] at e_895_7
  have hr := e_895_7
  simp only [e_895_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6879)) := by
  have c55716 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.1
  have c55717 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.1
  have c55718 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.1
  have c55719 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.1
  have c55720 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.1
  have c55721 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_8
  simp only [← c55716, ← c55717, ← c55718] at e_895_8
  have e_895_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_9
  simp only [← c55719, ← c55720, k2, ← c55721] at e_895_9
  have hr := e_895_9
  simp only [e_895_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6880)) := by
  have c55722 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55723 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55724 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55725 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55726 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55727 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_10
  simp only [← c55722, ← c55723, ← c55724] at e_895_10
  have e_895_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_11
  simp only [← c55725, ← c55726, k2, ← c55727] at e_895_11
  have hr := e_895_11
  simp only [e_895_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6881)) := by
  have c55728 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55729 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55730 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55731 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55732 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55733 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_12
  simp only [← c55728, ← c55729, ← c55730] at e_895_12
  have e_895_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_13
  simp only [← c55731, ← c55732, k2, ← c55733] at e_895_13
  have hr := e_895_13
  simp only [e_895_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6882)) := by
  have c55734 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55735 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55736 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55737 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55738 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55739 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_14
  simp only [← c55734, ← c55735, ← c55736] at e_895_14
  have e_895_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_15
  simp only [← c55737, ← c55738, k2, ← c55739] at e_895_15
  have hr := e_895_15
  simp only [e_895_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6883)) := by
  have c55740 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55741 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55742 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55743 := (publicBatchWrapper89_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55744 := (publicBatchWrapper89_copies1742 a h).1
  have c55745 := (publicBatchWrapper89_copies1742 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_16
  simp only [← c55740, ← c55741, ← c55742] at e_895_16
  have e_895_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_17
  simp only [← c55743, ← c55744, k2, ← c55745] at e_895_17
  have hr := e_895_17
  simp only [e_895_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 895 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6884)) := by
  have c55746 := (publicBatchWrapper89_copies1742 a h).2.2.1
  have c55747 := (publicBatchWrapper89_copies1742 a h).2.2.2.1
  have c55748 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.1
  have c55749 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.1
  have c55750 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.1
  have c55751 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_895_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_18
  simp only [← c55746, ← c55747, ← c55748] at e_895_18
  have e_895_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_19
  simp only [← c55749, ← c55750, k2, ← c55751] at e_895_19
  have hr := e_895_19
  simp only [e_895_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6885)) := by
  have c55752 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.1
  have c55753 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.1
  have c55754 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55755 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55756 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55757 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_0
  simp only [← c55752, ← c55753, ← c55754] at e_896_0
  have e_896_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_1
  simp only [← c55755, ← c55756, k2, ← c55757] at e_896_1
  have hr := e_896_1
  simp only [e_896_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6886)) := by
  have c55758 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55759 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55760 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55761 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55762 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55763 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_2
  simp only [← c55758, ← c55759, ← c55760] at e_896_2
  have e_896_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_3
  simp only [← c55761, ← c55762, k2, ← c55763] at e_896_3
  have hr := e_896_3
  simp only [e_896_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6887)) := by
  have c55764 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55765 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55766 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55767 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55768 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55769 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_4
  simp only [← c55764, ← c55765, ← c55766] at e_896_4
  have e_896_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_5
  simp only [← c55767, ← c55768, k2, ← c55769] at e_896_5
  have hr := e_896_5
  simp only [e_896_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6888)) := by
  have c55770 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55771 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55772 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55773 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55774 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55775 := (publicBatchWrapper89_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_6
  simp only [← c55770, ← c55771, ← c55772] at e_896_6
  have e_896_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_7
  simp only [← c55773, ← c55774, k2, ← c55775] at e_896_7
  have hr := e_896_7
  simp only [e_896_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6889)) := by
  have c55776 := (publicBatchWrapper89_copies1743 a h).1
  have c55777 := (publicBatchWrapper89_copies1743 a h).2.1
  have c55778 := (publicBatchWrapper89_copies1743 a h).2.2.1
  have c55779 := (publicBatchWrapper89_copies1743 a h).2.2.2.1
  have c55780 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.1
  have c55781 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_8
  simp only [← c55776, ← c55777, ← c55778] at e_896_8
  have e_896_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_9
  simp only [← c55779, ← c55780, k2, ← c55781] at e_896_9
  have hr := e_896_9
  simp only [e_896_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6890)) := by
  have c55782 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.1
  have c55783 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.1
  have c55784 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.1
  have c55785 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.1
  have c55786 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55787 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_10
  simp only [← c55782, ← c55783, ← c55784] at e_896_10
  have e_896_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_11
  simp only [← c55785, ← c55786, k2, ← c55787] at e_896_11
  have hr := e_896_11
  simp only [e_896_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6891)) := by
  have c55788 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55789 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55790 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55791 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55792 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55793 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_12
  simp only [← c55788, ← c55789, ← c55790] at e_896_12
  have e_896_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_13
  simp only [← c55791, ← c55792, k2, ← c55793] at e_896_13
  have hr := e_896_13
  simp only [e_896_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6892)) := by
  have c55794 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55795 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55796 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55797 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55798 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55799 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_14
  simp only [← c55794, ← c55795, ← c55796] at e_896_14
  have e_896_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_15
  simp only [← c55797, ← c55798, k2, ← c55799] at e_896_15
  have hr := e_896_15
  simp only [e_896_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6893)) := by
  have c55800 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55801 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55802 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55803 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55804 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55805 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_16
  simp only [← c55800, ← c55801, ← c55802] at e_896_16
  have e_896_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_17
  simp only [← c55803, ← c55804, k2, ← c55805] at e_896_17
  have hr := e_896_17
  simp only [e_896_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 896 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6894)) := by
  have c55806 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55807 := (publicBatchWrapper89_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55808 := (publicBatchWrapper89_copies1744 a h).1
  have c55809 := (publicBatchWrapper89_copies1744 a h).2.1
  have c55810 := (publicBatchWrapper89_copies1744 a h).2.2.1
  have c55811 := (publicBatchWrapper89_copies1744 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_896_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_18
  simp only [← c55806, ← c55807, ← c55808] at e_896_18
  have e_896_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_19
  simp only [← c55809, ← c55810, k2, ← c55811] at e_896_19
  have hr := e_896_19
  simp only [e_896_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6895)) := by
  have c55812 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.1
  have c55813 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.1
  have c55814 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.1
  have c55815 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.1
  have c55816 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.1
  have c55817 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_0
  simp only [← c55812, ← c55813, ← c55814] at e_897_0
  have e_897_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_1
  simp only [← c55815, ← c55816, k2, ← c55817] at e_897_1
  have hr := e_897_1
  simp only [e_897_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6896)) := by
  have c55818 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55819 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55820 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55821 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55822 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55823 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_2
  simp only [← c55818, ← c55819, ← c55820] at e_897_2
  have e_897_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_3
  simp only [← c55821, ← c55822, k2, ← c55823] at e_897_3
  have hr := e_897_3
  simp only [e_897_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6897)) := by
  have c55824 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55825 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55826 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55827 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55828 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55829 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_4
  simp only [← c55824, ← c55825, ← c55826] at e_897_4
  have e_897_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_5
  simp only [← c55827, ← c55828, k2, ← c55829] at e_897_5
  have hr := e_897_5
  simp only [e_897_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6898)) := by
  have c55830 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55831 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55832 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55833 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55834 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55835 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_6
  simp only [← c55830, ← c55831, ← c55832] at e_897_6
  have e_897_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_7
  simp only [← c55833, ← c55834, k2, ← c55835] at e_897_7
  have hr := e_897_7
  simp only [e_897_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6899)) := by
  have c55836 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55837 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55838 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55839 := (publicBatchWrapper89_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55840 := (publicBatchWrapper89_copies1745 a h).1
  have c55841 := (publicBatchWrapper89_copies1745 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_8
  simp only [← c55836, ← c55837, ← c55838] at e_897_8
  have e_897_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_9
  simp only [← c55839, ← c55840, k2, ← c55841] at e_897_9
  have hr := e_897_9
  simp only [e_897_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6900)) := by
  have c55842 := (publicBatchWrapper89_copies1745 a h).2.2.1
  have c55843 := (publicBatchWrapper89_copies1745 a h).2.2.2.1
  have c55844 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.1
  have c55845 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.1
  have c55846 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.1
  have c55847 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_10
  simp only [← c55842, ← c55843, ← c55844] at e_897_10
  have e_897_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_11
  simp only [← c55845, ← c55846, k2, ← c55847] at e_897_11
  have hr := e_897_11
  simp only [e_897_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6901)) := by
  have c55848 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.1
  have c55849 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.1
  have c55850 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55851 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55852 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55853 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_12
  simp only [← c55848, ← c55849, ← c55850] at e_897_12
  have e_897_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_13
  simp only [← c55851, ← c55852, k2, ← c55853] at e_897_13
  have hr := e_897_13
  simp only [e_897_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6902)) := by
  have c55854 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55855 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55856 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55857 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55858 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55859 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_14
  simp only [← c55854, ← c55855, ← c55856] at e_897_14
  have e_897_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_15
  simp only [← c55857, ← c55858, k2, ← c55859] at e_897_15
  have hr := e_897_15
  simp only [e_897_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6903)) := by
  have c55860 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55861 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55862 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55863 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55864 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55865 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_16
  simp only [← c55860, ← c55861, ← c55862] at e_897_16
  have e_897_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_17
  simp only [← c55863, ← c55864, k2, ← c55865] at e_897_17
  have hr := e_897_17
  simp only [e_897_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 897 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6904)) := by
  have c55866 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55867 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55868 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55869 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55870 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55871 := (publicBatchWrapper89_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_897_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_18
  simp only [← c55866, ← c55867, ← c55868] at e_897_18
  have e_897_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_19
  simp only [← c55869, ← c55870, k2, ← c55871] at e_897_19
  have hr := e_897_19
  simp only [e_897_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6905)) := by
  have c55872 := (publicBatchWrapper89_copies1746 a h).1
  have c55873 := (publicBatchWrapper89_copies1746 a h).2.1
  have c55874 := (publicBatchWrapper89_copies1746 a h).2.2.1
  have c55875 := (publicBatchWrapper89_copies1746 a h).2.2.2.1
  have c55876 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.1
  have c55877 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_0
  simp only [← c55872, ← c55873, ← c55874] at e_898_0
  have e_898_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_1
  simp only [← c55875, ← c55876, k2, ← c55877] at e_898_1
  have hr := e_898_1
  simp only [e_898_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6906)) := by
  have c55878 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.1
  have c55879 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.1
  have c55880 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.1
  have c55881 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.1
  have c55882 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55883 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_2
  simp only [← c55878, ← c55879, ← c55880] at e_898_2
  have e_898_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_3
  simp only [← c55881, ← c55882, k2, ← c55883] at e_898_3
  have hr := e_898_3
  simp only [e_898_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6907)) := by
  have c55884 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55885 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55886 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55887 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55888 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55889 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_4
  simp only [← c55884, ← c55885, ← c55886] at e_898_4
  have e_898_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_5
  simp only [← c55887, ← c55888, k2, ← c55889] at e_898_5
  have hr := e_898_5
  simp only [e_898_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6908)) := by
  have c55890 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55891 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55892 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55893 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55894 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55895 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_6
  simp only [← c55890, ← c55891, ← c55892] at e_898_6
  have e_898_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_7
  simp only [← c55893, ← c55894, k2, ← c55895] at e_898_7
  have hr := e_898_7
  simp only [e_898_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6909)) := by
  have c55896 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55897 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55898 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55899 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55900 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55901 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_8
  simp only [← c55896, ← c55897, ← c55898] at e_898_8
  have e_898_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_9
  simp only [← c55899, ← c55900, k2, ← c55901] at e_898_9
  have hr := e_898_9
  simp only [e_898_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6910)) := by
  have c55902 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55903 := (publicBatchWrapper89_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55904 := (publicBatchWrapper89_copies1747 a h).1
  have c55905 := (publicBatchWrapper89_copies1747 a h).2.1
  have c55906 := (publicBatchWrapper89_copies1747 a h).2.2.1
  have c55907 := (publicBatchWrapper89_copies1747 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_10
  simp only [← c55902, ← c55903, ← c55904] at e_898_10
  have e_898_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_11
  simp only [← c55905, ← c55906, k2, ← c55907] at e_898_11
  have hr := e_898_11
  simp only [e_898_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6911)) := by
  have c55908 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.1
  have c55909 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.1
  have c55910 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.1
  have c55911 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.1
  have c55912 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.1
  have c55913 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_12
  simp only [← c55908, ← c55909, ← c55910] at e_898_12
  have e_898_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_13
  simp only [← c55911, ← c55912, k2, ← c55913] at e_898_13
  have hr := e_898_13
  simp only [e_898_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6912)) := by
  have c55914 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55915 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55916 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55917 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55918 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55919 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_14
  simp only [← c55914, ← c55915, ← c55916] at e_898_14
  have e_898_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_15
  simp only [← c55917, ← c55918, k2, ← c55919] at e_898_15
  have hr := e_898_15
  simp only [e_898_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6913)) := by
  have c55920 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55921 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55922 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55923 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55924 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55925 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_16
  simp only [← c55920, ← c55921, ← c55922] at e_898_16
  have e_898_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_17
  simp only [← c55923, ← c55924, k2, ← c55925] at e_898_17
  have hr := e_898_17
  simp only [e_898_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 898 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6914)) := by
  have c55926 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55927 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55928 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55929 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55930 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55931 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_898_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_18
  simp only [← c55926, ← c55927, ← c55928] at e_898_18
  have e_898_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_19
  simp only [← c55929, ← c55930, k2, ← c55931] at e_898_19
  have hr := e_898_19
  simp only [e_898_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6915)) := by
  have c55932 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55933 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55934 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55935 := (publicBatchWrapper89_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55936 := (publicBatchWrapper89_copies1748 a h).1
  have c55937 := (publicBatchWrapper89_copies1748 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_0
  simp only [← c55932, ← c55933, ← c55934] at e_899_0
  have e_899_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_1
  simp only [← c55935, ← c55936, k2, ← c55937] at e_899_1
  have hr := e_899_1
  simp only [e_899_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6916)) := by
  have c55938 := (publicBatchWrapper89_copies1748 a h).2.2.1
  have c55939 := (publicBatchWrapper89_copies1748 a h).2.2.2.1
  have c55940 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.1
  have c55941 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.1
  have c55942 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.1
  have c55943 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_2
  simp only [← c55938, ← c55939, ← c55940] at e_899_2
  have e_899_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_3
  simp only [← c55941, ← c55942, k2, ← c55943] at e_899_3
  have hr := e_899_3
  simp only [e_899_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6917)) := by
  have c55944 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.1
  have c55945 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.1
  have c55946 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55947 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55948 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55949 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_4
  simp only [← c55944, ← c55945, ← c55946] at e_899_4
  have e_899_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_5
  simp only [← c55947, ← c55948, k2, ← c55949] at e_899_5
  have hr := e_899_5
  simp only [e_899_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6918)) := by
  have c55950 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55951 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55952 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55953 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55954 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55955 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_6
  simp only [← c55950, ← c55951, ← c55952] at e_899_6
  have e_899_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_7
  simp only [← c55953, ← c55954, k2, ← c55955] at e_899_7
  have hr := e_899_7
  simp only [e_899_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6919)) := by
  have c55956 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55957 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55958 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55959 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55960 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55961 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_8
  simp only [← c55956, ← c55957, ← c55958] at e_899_8
  have e_899_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_9
  simp only [← c55959, ← c55960, k2, ← c55961] at e_899_9
  have hr := e_899_9
  simp only [e_899_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6920)) := by
  have c55962 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55963 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55964 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55965 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55966 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55967 := (publicBatchWrapper89_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_10
  simp only [← c55962, ← c55963, ← c55964] at e_899_10
  have e_899_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_11
  simp only [← c55965, ← c55966, k2, ← c55967] at e_899_11
  have hr := e_899_11
  simp only [e_899_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6921)) := by
  have c55968 := (publicBatchWrapper89_copies1749 a h).1
  have c55969 := (publicBatchWrapper89_copies1749 a h).2.1
  have c55970 := (publicBatchWrapper89_copies1749 a h).2.2.1
  have c55971 := (publicBatchWrapper89_copies1749 a h).2.2.2.1
  have c55972 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.1
  have c55973 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_12
  simp only [← c55968, ← c55969, ← c55970] at e_899_12
  have e_899_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_13
  simp only [← c55971, ← c55972, k2, ← c55973] at e_899_13
  have hr := e_899_13
  simp only [e_899_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6922)) := by
  have c55974 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.1
  have c55975 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.1
  have c55976 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.1
  have c55977 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.1
  have c55978 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55979 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_14
  simp only [← c55974, ← c55975, ← c55976] at e_899_14
  have e_899_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_15
  simp only [← c55977, ← c55978, k2, ← c55979] at e_899_15
  have hr := e_899_15
  simp only [e_899_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6923)) := by
  have c55980 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55981 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55982 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55983 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55984 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55985 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_16
  simp only [← c55980, ← c55981, ← c55982] at e_899_16
  have e_899_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_17
  simp only [← c55983, ← c55984, k2, ← c55985] at e_899_17
  have hr := e_899_17
  simp only [e_899_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 899 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6924)) := by
  have c55986 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55987 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55988 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55989 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55990 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55991 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_899_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_18
  simp only [← c55986, ← c55987, ← c55988] at e_899_18
  have e_899_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_19
  simp only [← c55989, ← c55990, k2, ← c55991] at e_899_19
  have hr := e_899_19
  simp only [e_899_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6925)) := by
  have c55992 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55993 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55994 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55995 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55996 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55997 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_0
  simp only [← c55992, ← c55993, ← c55994] at e_900_0
  have e_900_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_1
  simp only [← c55995, ← c55996, k2, ← c55997] at e_900_1
  have hr := e_900_1
  simp only [e_900_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6926)) := by
  have c55998 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55999 := (publicBatchWrapper89_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56000 := (publicBatchWrapper89_copies1750 a h).1
  have c56001 := (publicBatchWrapper89_copies1750 a h).2.1
  have c56002 := (publicBatchWrapper89_copies1750 a h).2.2.1
  have c56003 := (publicBatchWrapper89_copies1750 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_2
  simp only [← c55998, ← c55999, ← c56000] at e_900_2
  have e_900_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_3
  simp only [← c56001, ← c56002, k2, ← c56003] at e_900_3
  have hr := e_900_3
  simp only [e_900_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6927)) := by
  have c56004 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.1
  have c56005 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.1
  have c56006 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.1
  have c56007 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.1
  have c56008 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.1
  have c56009 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_4
  simp only [← c56004, ← c56005, ← c56006] at e_900_4
  have e_900_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_5
  simp only [← c56007, ← c56008, k2, ← c56009] at e_900_5
  have hr := e_900_5
  simp only [e_900_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6928)) := by
  have c56010 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56011 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56012 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56013 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56014 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56015 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_6
  simp only [← c56010, ← c56011, ← c56012] at e_900_6
  have e_900_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_7
  simp only [← c56013, ← c56014, k2, ← c56015] at e_900_7
  have hr := e_900_7
  simp only [e_900_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6929)) := by
  have c56016 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56017 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56018 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56019 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56020 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56021 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_8
  simp only [← c56016, ← c56017, ← c56018] at e_900_8
  have e_900_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_9
  simp only [← c56019, ← c56020, k2, ← c56021] at e_900_9
  have hr := e_900_9
  simp only [e_900_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6930)) := by
  have c56022 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56023 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56024 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56025 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56026 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56027 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_10
  simp only [← c56022, ← c56023, ← c56024] at e_900_10
  have e_900_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_11
  simp only [← c56025, ← c56026, k2, ← c56027] at e_900_11
  have hr := e_900_11
  simp only [e_900_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6931)) := by
  have c56028 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56029 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56030 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56031 := (publicBatchWrapper89_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56032 := (publicBatchWrapper89_copies1751 a h).1
  have c56033 := (publicBatchWrapper89_copies1751 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_12
  simp only [← c56028, ← c56029, ← c56030] at e_900_12
  have e_900_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_13
  simp only [← c56031, ← c56032, k2, ← c56033] at e_900_13
  have hr := e_900_13
  simp only [e_900_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6932)) := by
  have c56034 := (publicBatchWrapper89_copies1751 a h).2.2.1
  have c56035 := (publicBatchWrapper89_copies1751 a h).2.2.2.1
  have c56036 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.1
  have c56037 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.1
  have c56038 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.1
  have c56039 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_14
  simp only [← c56034, ← c56035, ← c56036] at e_900_14
  have e_900_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_15
  simp only [← c56037, ← c56038, k2, ← c56039] at e_900_15
  have hr := e_900_15
  simp only [e_900_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6933)) := by
  have c56040 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.1
  have c56041 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.1
  have c56042 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56043 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56044 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56045 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_16
  simp only [← c56040, ← c56041, ← c56042] at e_900_16
  have e_900_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_17
  simp only [← c56043, ← c56044, k2, ← c56045] at e_900_17
  have hr := e_900_17
  simp only [e_900_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 900 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6934)) := by
  have c56046 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56047 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56048 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56049 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56050 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56051 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_900_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_18
  simp only [← c56046, ← c56047, ← c56048] at e_900_18
  have e_900_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_19
  simp only [← c56049, ← c56050, k2, ← c56051] at e_900_19
  have hr := e_900_19
  simp only [e_900_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6935)) := by
  have c56052 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56053 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56054 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56055 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56056 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56057 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_0
  simp only [← c56052, ← c56053, ← c56054] at e_901_0
  have e_901_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_1
  simp only [← c56055, ← c56056, k2, ← c56057] at e_901_1
  have hr := e_901_1
  simp only [e_901_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6936)) := by
  have c56058 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56059 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56060 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56061 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56062 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56063 := (publicBatchWrapper89_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_2
  simp only [← c56058, ← c56059, ← c56060] at e_901_2
  have e_901_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_3
  simp only [← c56061, ← c56062, k2, ← c56063] at e_901_3
  have hr := e_901_3
  simp only [e_901_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6937)) := by
  have c56064 := (publicBatchWrapper89_copies1752 a h).1
  have c56065 := (publicBatchWrapper89_copies1752 a h).2.1
  have c56066 := (publicBatchWrapper89_copies1752 a h).2.2.1
  have c56067 := (publicBatchWrapper89_copies1752 a h).2.2.2.1
  have c56068 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.1
  have c56069 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_4
  simp only [← c56064, ← c56065, ← c56066] at e_901_4
  have e_901_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_5
  simp only [← c56067, ← c56068, k2, ← c56069] at e_901_5
  have hr := e_901_5
  simp only [e_901_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6938)) := by
  have c56070 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.1
  have c56071 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.1
  have c56072 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.1
  have c56073 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.1
  have c56074 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56075 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_6
  simp only [← c56070, ← c56071, ← c56072] at e_901_6
  have e_901_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_7
  simp only [← c56073, ← c56074, k2, ← c56075] at e_901_7
  have hr := e_901_7
  simp only [e_901_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6939)) := by
  have c56076 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56077 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56078 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56079 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56080 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56081 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_8
  simp only [← c56076, ← c56077, ← c56078] at e_901_8
  have e_901_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_9
  simp only [← c56079, ← c56080, k2, ← c56081] at e_901_9
  have hr := e_901_9
  simp only [e_901_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6940)) := by
  have c56082 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56083 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56084 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56085 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56086 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56087 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_10
  simp only [← c56082, ← c56083, ← c56084] at e_901_10
  have e_901_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_11
  simp only [← c56085, ← c56086, k2, ← c56087] at e_901_11
  have hr := e_901_11
  simp only [e_901_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6941)) := by
  have c56088 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56089 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56090 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56091 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56092 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56093 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_12
  simp only [← c56088, ← c56089, ← c56090] at e_901_12
  have e_901_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_13
  simp only [← c56091, ← c56092, k2, ← c56093] at e_901_13
  have hr := e_901_13
  simp only [e_901_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6942)) := by
  have c56094 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56095 := (publicBatchWrapper89_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56096 := (publicBatchWrapper89_copies1753 a h).1
  have c56097 := (publicBatchWrapper89_copies1753 a h).2.1
  have c56098 := (publicBatchWrapper89_copies1753 a h).2.2.1
  have c56099 := (publicBatchWrapper89_copies1753 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_14
  simp only [← c56094, ← c56095, ← c56096] at e_901_14
  have e_901_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_15
  simp only [← c56097, ← c56098, k2, ← c56099] at e_901_15
  have hr := e_901_15
  simp only [e_901_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6943)) := by
  have c56100 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.1
  have c56101 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.1
  have c56102 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.1
  have c56103 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.1
  have c56104 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.1
  have c56105 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_16
  simp only [← c56100, ← c56101, ← c56102] at e_901_16
  have e_901_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_17
  simp only [← c56103, ← c56104, k2, ← c56105] at e_901_17
  have hr := e_901_17
  simp only [e_901_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 901 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6944)) := by
  have c56106 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56107 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56108 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56109 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56110 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56111 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_901_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_18
  simp only [← c56106, ← c56107, ← c56108] at e_901_18
  have e_901_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_19
  simp only [← c56109, ← c56110, k2, ← c56111] at e_901_19
  have hr := e_901_19
  simp only [e_901_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6945)) := by
  have c56112 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56113 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56114 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56115 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56116 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56117 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_0
  simp only [← c56112, ← c56113, ← c56114] at e_902_0
  have e_902_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_1
  simp only [← c56115, ← c56116, k2, ← c56117] at e_902_1
  have hr := e_902_1
  simp only [e_902_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6946)) := by
  have c56118 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56119 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56120 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56121 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56122 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56123 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_2
  simp only [← c56118, ← c56119, ← c56120] at e_902_2
  have e_902_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_3
  simp only [← c56121, ← c56122, k2, ← c56123] at e_902_3
  have hr := e_902_3
  simp only [e_902_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6947)) := by
  have c56124 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56125 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56126 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56127 := (publicBatchWrapper89_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56128 := (publicBatchWrapper89_copies1754 a h).1
  have c56129 := (publicBatchWrapper89_copies1754 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_4
  simp only [← c56124, ← c56125, ← c56126] at e_902_4
  have e_902_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_5
  simp only [← c56127, ← c56128, k2, ← c56129] at e_902_5
  have hr := e_902_5
  simp only [e_902_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6948)) := by
  have c56130 := (publicBatchWrapper89_copies1754 a h).2.2.1
  have c56131 := (publicBatchWrapper89_copies1754 a h).2.2.2.1
  have c56132 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.1
  have c56133 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.1
  have c56134 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.1
  have c56135 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_6
  simp only [← c56130, ← c56131, ← c56132] at e_902_6
  have e_902_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_7
  simp only [← c56133, ← c56134, k2, ← c56135] at e_902_7
  have hr := e_902_7
  simp only [e_902_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6949)) := by
  have c56136 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.1
  have c56137 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.1
  have c56138 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56139 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56140 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56141 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_8
  simp only [← c56136, ← c56137, ← c56138] at e_902_8
  have e_902_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_9
  simp only [← c56139, ← c56140, k2, ← c56141] at e_902_9
  have hr := e_902_9
  simp only [e_902_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6950)) := by
  have c56142 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56143 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56144 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56145 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56146 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56147 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_10
  simp only [← c56142, ← c56143, ← c56144] at e_902_10
  have e_902_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_11
  simp only [← c56145, ← c56146, k2, ← c56147] at e_902_11
  have hr := e_902_11
  simp only [e_902_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6951)) := by
  have c56148 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56149 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56150 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56151 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56152 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56153 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_12
  simp only [← c56148, ← c56149, ← c56150] at e_902_12
  have e_902_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_13
  simp only [← c56151, ← c56152, k2, ← c56153] at e_902_13
  have hr := e_902_13
  simp only [e_902_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6952)) := by
  have c56154 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56155 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56156 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56157 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56158 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56159 := (publicBatchWrapper89_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_14
  simp only [← c56154, ← c56155, ← c56156] at e_902_14
  have e_902_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_15
  simp only [← c56157, ← c56158, k2, ← c56159] at e_902_15
  have hr := e_902_15
  simp only [e_902_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6953)) := by
  have c56160 := (publicBatchWrapper89_copies1755 a h).1
  have c56161 := (publicBatchWrapper89_copies1755 a h).2.1
  have c56162 := (publicBatchWrapper89_copies1755 a h).2.2.1
  have c56163 := (publicBatchWrapper89_copies1755 a h).2.2.2.1
  have c56164 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.1
  have c56165 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_16
  simp only [← c56160, ← c56161, ← c56162] at e_902_16
  have e_902_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_17
  simp only [← c56163, ← c56164, k2, ← c56165] at e_902_17
  have hr := e_902_17
  simp only [e_902_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 902 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6954)) := by
  have c56166 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.1
  have c56167 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.1
  have c56168 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.1
  have c56169 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.1
  have c56170 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56171 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_902_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_18
  simp only [← c56166, ← c56167, ← c56168] at e_902_18
  have e_902_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_19
  simp only [← c56169, ← c56170, k2, ← c56171] at e_902_19
  have hr := e_902_19
  simp only [e_902_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6955)) := by
  have c56172 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56173 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56174 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56175 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56176 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56177 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_0
  simp only [← c56172, ← c56173, ← c56174] at e_903_0
  have e_903_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_1
  simp only [← c56175, ← c56176, k2, ← c56177] at e_903_1
  have hr := e_903_1
  simp only [e_903_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6956)) := by
  have c56178 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56179 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56180 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56181 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56182 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56183 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_2
  simp only [← c56178, ← c56179, ← c56180] at e_903_2
  have e_903_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_3
  simp only [← c56181, ← c56182, k2, ← c56183] at e_903_3
  have hr := e_903_3
  simp only [e_903_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6957)) := by
  have c56184 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56185 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56186 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56187 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56188 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56189 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_4
  simp only [← c56184, ← c56185, ← c56186] at e_903_4
  have e_903_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_5
  simp only [← c56187, ← c56188, k2, ← c56189] at e_903_5
  have hr := e_903_5
  simp only [e_903_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6958)) := by
  have c56190 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56191 := (publicBatchWrapper89_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56192 := (publicBatchWrapper89_copies1756 a h).1
  have c56193 := (publicBatchWrapper89_copies1756 a h).2.1
  have c56194 := (publicBatchWrapper89_copies1756 a h).2.2.1
  have c56195 := (publicBatchWrapper89_copies1756 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_6
  simp only [← c56190, ← c56191, ← c56192] at e_903_6
  have e_903_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_7
  simp only [← c56193, ← c56194, k2, ← c56195] at e_903_7
  have hr := e_903_7
  simp only [e_903_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6959)) := by
  have c56196 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.1
  have c56197 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.1
  have c56198 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.1
  have c56199 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.1
  have c56200 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.1
  have c56201 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_8
  simp only [← c56196, ← c56197, ← c56198] at e_903_8
  have e_903_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_9
  simp only [← c56199, ← c56200, k2, ← c56201] at e_903_9
  have hr := e_903_9
  simp only [e_903_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6960)) := by
  have c56202 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56203 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56204 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56205 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56206 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56207 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_10
  simp only [← c56202, ← c56203, ← c56204] at e_903_10
  have e_903_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_11
  simp only [← c56205, ← c56206, k2, ← c56207] at e_903_11
  have hr := e_903_11
  simp only [e_903_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6961)) := by
  have c56208 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56209 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56210 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56211 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56212 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56213 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_12
  simp only [← c56208, ← c56209, ← c56210] at e_903_12
  have e_903_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_13
  simp only [← c56211, ← c56212, k2, ← c56213] at e_903_13
  have hr := e_903_13
  simp only [e_903_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6962)) := by
  have c56214 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56215 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56216 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56217 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56218 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56219 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_14
  simp only [← c56214, ← c56215, ← c56216] at e_903_14
  have e_903_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_15
  simp only [← c56217, ← c56218, k2, ← c56219] at e_903_15
  have hr := e_903_15
  simp only [e_903_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6963)) := by
  have c56220 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56221 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56222 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56223 := (publicBatchWrapper89_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56224 := (publicBatchWrapper89_copies1757 a h).1
  have c56225 := (publicBatchWrapper89_copies1757 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_16
  simp only [← c56220, ← c56221, ← c56222] at e_903_16
  have e_903_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_17
  simp only [← c56223, ← c56224, k2, ← c56225] at e_903_17
  have hr := e_903_17
  simp only [e_903_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 903 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6964)) := by
  have c56226 := (publicBatchWrapper89_copies1757 a h).2.2.1
  have c56227 := (publicBatchWrapper89_copies1757 a h).2.2.2.1
  have c56228 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.1
  have c56229 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.1
  have c56230 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.1
  have c56231 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_903_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_18
  simp only [← c56226, ← c56227, ← c56228] at e_903_18
  have e_903_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_19
  simp only [← c56229, ← c56230, k2, ← c56231] at e_903_19
  have hr := e_903_19
  simp only [e_903_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6965)) := by
  have c56232 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.1
  have c56233 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.1
  have c56234 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56235 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56236 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56237 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_0
  simp only [← c56232, ← c56233, ← c56234] at e_904_0
  have e_904_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_1
  simp only [← c56235, ← c56236, k2, ← c56237] at e_904_1
  have hr := e_904_1
  simp only [e_904_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6966)) := by
  have c56238 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56239 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56240 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56241 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56242 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56243 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_2
  simp only [← c56238, ← c56239, ← c56240] at e_904_2
  have e_904_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_3
  simp only [← c56241, ← c56242, k2, ← c56243] at e_904_3
  have hr := e_904_3
  simp only [e_904_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6967)) := by
  have c56244 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56245 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56246 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56247 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56248 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56249 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_4
  simp only [← c56244, ← c56245, ← c56246] at e_904_4
  have e_904_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_5
  simp only [← c56247, ← c56248, k2, ← c56249] at e_904_5
  have hr := e_904_5
  simp only [e_904_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6968)) := by
  have c56250 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56251 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56252 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56253 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56254 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56255 := (publicBatchWrapper89_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_6
  simp only [← c56250, ← c56251, ← c56252] at e_904_6
  have e_904_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_7
  simp only [← c56253, ← c56254, k2, ← c56255] at e_904_7
  have hr := e_904_7
  simp only [e_904_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6969)) := by
  have c56256 := (publicBatchWrapper89_copies1758 a h).1
  have c56257 := (publicBatchWrapper89_copies1758 a h).2.1
  have c56258 := (publicBatchWrapper89_copies1758 a h).2.2.1
  have c56259 := (publicBatchWrapper89_copies1758 a h).2.2.2.1
  have c56260 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.1
  have c56261 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_8
  simp only [← c56256, ← c56257, ← c56258] at e_904_8
  have e_904_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_9
  simp only [← c56259, ← c56260, k2, ← c56261] at e_904_9
  have hr := e_904_9
  simp only [e_904_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7010)) := by
  have c56262 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.1
  have c56263 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.1
  have c56264 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.1
  have c56265 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.1
  have c56266 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56267 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_10
  simp only [← c56262, ← c56263, ← c56264] at e_904_10
  have e_904_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_11
  simp only [← c56265, ← c56266, k2, ← c56267] at e_904_11
  have hr := e_904_11
  simp only [e_904_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7011)) := by
  have c56268 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56269 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56270 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56271 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56272 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56273 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_12
  simp only [← c56268, ← c56269, ← c56270] at e_904_12
  have e_904_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_13
  simp only [← c56271, ← c56272, k2, ← c56273] at e_904_13
  have hr := e_904_13
  simp only [e_904_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7012)) := by
  have c56274 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56275 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56276 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56277 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56278 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56279 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_14
  simp only [← c56274, ← c56275, ← c56276] at e_904_14
  have e_904_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_15
  simp only [← c56277, ← c56278, k2, ← c56279] at e_904_15
  have hr := e_904_15
  simp only [e_904_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7013)) := by
  have c56280 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56281 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56282 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56283 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56284 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56285 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_16
  simp only [← c56280, ← c56281, ← c56282] at e_904_16
  have e_904_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_17
  simp only [← c56283, ← c56284, k2, ← c56285] at e_904_17
  have hr := e_904_17
  simp only [e_904_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 904 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7014)) := by
  have c56286 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56287 := (publicBatchWrapper89_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56288 := (publicBatchWrapper89_copies1759 a h).1
  have c56289 := (publicBatchWrapper89_copies1759 a h).2.1
  have c56290 := (publicBatchWrapper89_copies1759 a h).2.2.1
  have c56291 := (publicBatchWrapper89_copies1759 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_904_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_18
  simp only [← c56286, ← c56287, ← c56288] at e_904_18
  have e_904_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_19
  simp only [← c56289, ← c56290, k2, ← c56291] at e_904_19
  have hr := e_904_19
  simp only [e_904_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7015)) := by
  have c56292 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.1
  have c56293 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.1
  have c56294 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.1
  have c56295 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.1
  have c56296 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.1
  have c56297 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_0
  simp only [← c56292, ← c56293, ← c56294] at e_905_0
  have e_905_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_1
  simp only [← c56295, ← c56296, k2, ← c56297] at e_905_1
  have hr := e_905_1
  simp only [e_905_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7016)) := by
  have c56298 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56299 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56300 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56301 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56302 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56303 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_2
  simp only [← c56298, ← c56299, ← c56300] at e_905_2
  have e_905_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_3
  simp only [← c56301, ← c56302, k2, ← c56303] at e_905_3
  have hr := e_905_3
  simp only [e_905_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7017)) := by
  have c56304 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56305 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56306 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56307 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56308 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56309 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_4
  simp only [← c56304, ← c56305, ← c56306] at e_905_4
  have e_905_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_5
  simp only [← c56307, ← c56308, k2, ← c56309] at e_905_5
  have hr := e_905_5
  simp only [e_905_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7018)) := by
  have c56310 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56311 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56312 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56313 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56314 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56315 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_6
  simp only [← c56310, ← c56311, ← c56312] at e_905_6
  have e_905_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_7
  simp only [← c56313, ← c56314, k2, ← c56315] at e_905_7
  have hr := e_905_7
  simp only [e_905_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7019)) := by
  have c56316 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56317 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56318 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56319 := (publicBatchWrapper89_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56320 := (publicBatchWrapper89_copies1760 a h).1
  have c56321 := (publicBatchWrapper89_copies1760 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_8
  simp only [← c56316, ← c56317, ← c56318] at e_905_8
  have e_905_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_9
  simp only [← c56319, ← c56320, k2, ← c56321] at e_905_9
  have hr := e_905_9
  simp only [e_905_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7020)) := by
  have c56322 := (publicBatchWrapper89_copies1760 a h).2.2.1
  have c56323 := (publicBatchWrapper89_copies1760 a h).2.2.2.1
  have c56324 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.1
  have c56325 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.1
  have c56326 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.1
  have c56327 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_10
  simp only [← c56322, ← c56323, ← c56324] at e_905_10
  have e_905_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_11
  simp only [← c56325, ← c56326, k2, ← c56327] at e_905_11
  have hr := e_905_11
  simp only [e_905_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7021)) := by
  have c56328 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.1
  have c56329 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.1
  have c56330 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56331 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56332 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56333 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_12
  simp only [← c56328, ← c56329, ← c56330] at e_905_12
  have e_905_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_13
  simp only [← c56331, ← c56332, k2, ← c56333] at e_905_13
  have hr := e_905_13
  simp only [e_905_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7022)) := by
  have c56334 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56335 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56336 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56337 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56338 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56339 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_14
  simp only [← c56334, ← c56335, ← c56336] at e_905_14
  have e_905_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_15
  simp only [← c56337, ← c56338, k2, ← c56339] at e_905_15
  have hr := e_905_15
  simp only [e_905_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7023)) := by
  have c56340 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56341 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56342 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56343 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56344 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56345 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_16
  simp only [← c56340, ← c56341, ← c56342] at e_905_16
  have e_905_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_17
  simp only [← c56343, ← c56344, k2, ← c56345] at e_905_17
  have hr := e_905_17
  simp only [e_905_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 905 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7024)) := by
  have c56346 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56347 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56348 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56349 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56350 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56351 := (publicBatchWrapper89_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_905_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_18
  simp only [← c56346, ← c56347, ← c56348] at e_905_18
  have e_905_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_19
  simp only [← c56349, ← c56350, k2, ← c56351] at e_905_19
  have hr := e_905_19
  simp only [e_905_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7025)) := by
  have c56352 := (publicBatchWrapper89_copies1761 a h).1
  have c56353 := (publicBatchWrapper89_copies1761 a h).2.1
  have c56354 := (publicBatchWrapper89_copies1761 a h).2.2.1
  have c56355 := (publicBatchWrapper89_copies1761 a h).2.2.2.1
  have c56356 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.1
  have c56357 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_0
  simp only [← c56352, ← c56353, ← c56354] at e_906_0
  have e_906_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_1
  simp only [← c56355, ← c56356, k2, ← c56357] at e_906_1
  have hr := e_906_1
  simp only [e_906_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7026)) := by
  have c56358 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.1
  have c56359 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.1
  have c56360 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.1
  have c56361 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.1
  have c56362 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56363 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_2
  simp only [← c56358, ← c56359, ← c56360] at e_906_2
  have e_906_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_3
  simp only [← c56361, ← c56362, k2, ← c56363] at e_906_3
  have hr := e_906_3
  simp only [e_906_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7027)) := by
  have c56364 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56365 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56366 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56367 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56368 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56369 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_4
  simp only [← c56364, ← c56365, ← c56366] at e_906_4
  have e_906_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_5
  simp only [← c56367, ← c56368, k2, ← c56369] at e_906_5
  have hr := e_906_5
  simp only [e_906_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7028)) := by
  have c56370 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56371 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56372 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56373 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56374 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56375 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_6
  simp only [← c56370, ← c56371, ← c56372] at e_906_6
  have e_906_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_7
  simp only [← c56373, ← c56374, k2, ← c56375] at e_906_7
  have hr := e_906_7
  simp only [e_906_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7029)) := by
  have c56376 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56377 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56378 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56379 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56380 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56381 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_8
  simp only [← c56376, ← c56377, ← c56378] at e_906_8
  have e_906_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_9
  simp only [← c56379, ← c56380, k2, ← c56381] at e_906_9
  have hr := e_906_9
  simp only [e_906_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7030)) := by
  have c56382 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56383 := (publicBatchWrapper89_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56384 := (publicBatchWrapper89_copies1762 a h).1
  have c56385 := (publicBatchWrapper89_copies1762 a h).2.1
  have c56386 := (publicBatchWrapper89_copies1762 a h).2.2.1
  have c56387 := (publicBatchWrapper89_copies1762 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_10
  simp only [← c56382, ← c56383, ← c56384] at e_906_10
  have e_906_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_11
  simp only [← c56385, ← c56386, k2, ← c56387] at e_906_11
  have hr := e_906_11
  simp only [e_906_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7031)) := by
  have c56388 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.1
  have c56389 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.1
  have c56390 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.1
  have c56391 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.1
  have c56392 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.1
  have c56393 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_12
  simp only [← c56388, ← c56389, ← c56390] at e_906_12
  have e_906_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_13
  simp only [← c56391, ← c56392, k2, ← c56393] at e_906_13
  have hr := e_906_13
  simp only [e_906_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7032)) := by
  have c56394 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56395 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56396 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56397 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56398 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56399 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_14
  simp only [← c56394, ← c56395, ← c56396] at e_906_14
  have e_906_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_15
  simp only [← c56397, ← c56398, k2, ← c56399] at e_906_15
  have hr := e_906_15
  simp only [e_906_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7033)) := by
  have c56400 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56401 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56402 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56403 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56404 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56405 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_16
  simp only [← c56400, ← c56401, ← c56402] at e_906_16
  have e_906_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_17
  simp only [← c56403, ← c56404, k2, ← c56405] at e_906_17
  have hr := e_906_17
  simp only [e_906_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 906 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7034)) := by
  have c56406 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56407 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56408 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56409 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56410 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56411 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_906_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_18
  simp only [← c56406, ← c56407, ← c56408] at e_906_18
  have e_906_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_19
  simp only [← c56409, ← c56410, k2, ← c56411] at e_906_19
  have hr := e_906_19
  simp only [e_906_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7035)) := by
  have c56412 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56413 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56414 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56415 := (publicBatchWrapper89_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56416 := (publicBatchWrapper89_copies1763 a h).1
  have c56417 := (publicBatchWrapper89_copies1763 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_0
  simp only [← c56412, ← c56413, ← c56414] at e_907_0
  have e_907_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_1
  simp only [← c56415, ← c56416, k2, ← c56417] at e_907_1
  have hr := e_907_1
  simp only [e_907_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7036)) := by
  have c56418 := (publicBatchWrapper89_copies1763 a h).2.2.1
  have c56419 := (publicBatchWrapper89_copies1763 a h).2.2.2.1
  have c56420 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.1
  have c56421 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.1
  have c56422 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.1
  have c56423 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_2
  simp only [← c56418, ← c56419, ← c56420] at e_907_2
  have e_907_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_3
  simp only [← c56421, ← c56422, k2, ← c56423] at e_907_3
  have hr := e_907_3
  simp only [e_907_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7037)) := by
  have c56424 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.1
  have c56425 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.1
  have c56426 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56427 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56428 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56429 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_4
  simp only [← c56424, ← c56425, ← c56426] at e_907_4
  have e_907_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_5
  simp only [← c56427, ← c56428, k2, ← c56429] at e_907_5
  have hr := e_907_5
  simp only [e_907_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7038)) := by
  have c56430 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56431 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56432 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56433 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56434 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56435 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_6
  simp only [← c56430, ← c56431, ← c56432] at e_907_6
  have e_907_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_7
  simp only [← c56433, ← c56434, k2, ← c56435] at e_907_7
  have hr := e_907_7
  simp only [e_907_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7039)) := by
  have c56436 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56437 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56438 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56439 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56440 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56441 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_8
  simp only [← c56436, ← c56437, ← c56438] at e_907_8
  have e_907_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_9
  simp only [← c56439, ← c56440, k2, ← c56441] at e_907_9
  have hr := e_907_9
  simp only [e_907_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7040)) := by
  have c56442 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56443 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56444 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56445 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56446 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56447 := (publicBatchWrapper89_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_10
  simp only [← c56442, ← c56443, ← c56444] at e_907_10
  have e_907_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_11
  simp only [← c56445, ← c56446, k2, ← c56447] at e_907_11
  have hr := e_907_11
  simp only [e_907_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7041)) := by
  have c56448 := (publicBatchWrapper89_copies1764 a h).1
  have c56449 := (publicBatchWrapper89_copies1764 a h).2.1
  have c56450 := (publicBatchWrapper89_copies1764 a h).2.2.1
  have c56451 := (publicBatchWrapper89_copies1764 a h).2.2.2.1
  have c56452 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.1
  have c56453 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_12
  simp only [← c56448, ← c56449, ← c56450] at e_907_12
  have e_907_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_13
  simp only [← c56451, ← c56452, k2, ← c56453] at e_907_13
  have hr := e_907_13
  simp only [e_907_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7042)) := by
  have c56454 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.1
  have c56455 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.1
  have c56456 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.1
  have c56457 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.1
  have c56458 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56459 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_14
  simp only [← c56454, ← c56455, ← c56456] at e_907_14
  have e_907_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_15
  simp only [← c56457, ← c56458, k2, ← c56459] at e_907_15
  have hr := e_907_15
  simp only [e_907_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7043)) := by
  have c56460 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56461 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56462 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56463 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56464 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56465 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_16
  simp only [← c56460, ← c56461, ← c56462] at e_907_16
  have e_907_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_17
  simp only [← c56463, ← c56464, k2, ← c56465] at e_907_17
  have hr := e_907_17
  simp only [e_907_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 907 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7044)) := by
  have c56466 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56467 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56468 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56469 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56470 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56471 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_907_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_18
  simp only [← c56466, ← c56467, ← c56468] at e_907_18
  have e_907_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_19
  simp only [← c56469, ← c56470, k2, ← c56471] at e_907_19
  have hr := e_907_19
  simp only [e_907_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7045)) := by
  have c56472 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56473 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56474 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56475 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56476 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56477 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_0
  simp only [← c56472, ← c56473, ← c56474] at e_908_0
  have e_908_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_1
  simp only [← c56475, ← c56476, k2, ← c56477] at e_908_1
  have hr := e_908_1
  simp only [e_908_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7046)) := by
  have c56478 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56479 := (publicBatchWrapper89_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56480 := (publicBatchWrapper89_copies1765 a h).1
  have c56481 := (publicBatchWrapper89_copies1765 a h).2.1
  have c56482 := (publicBatchWrapper89_copies1765 a h).2.2.1
  have c56483 := (publicBatchWrapper89_copies1765 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_2
  simp only [← c56478, ← c56479, ← c56480] at e_908_2
  have e_908_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_3
  simp only [← c56481, ← c56482, k2, ← c56483] at e_908_3
  have hr := e_908_3
  simp only [e_908_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7047)) := by
  have c56484 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.1
  have c56485 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.1
  have c56486 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.1
  have c56487 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.1
  have c56488 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.1
  have c56489 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_4
  simp only [← c56484, ← c56485, ← c56486] at e_908_4
  have e_908_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_5
  simp only [← c56487, ← c56488, k2, ← c56489] at e_908_5
  have hr := e_908_5
  simp only [e_908_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7048)) := by
  have c56490 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56491 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56492 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56493 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56494 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56495 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_6
  simp only [← c56490, ← c56491, ← c56492] at e_908_6
  have e_908_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_7
  simp only [← c56493, ← c56494, k2, ← c56495] at e_908_7
  have hr := e_908_7
  simp only [e_908_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7049)) := by
  have c56496 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56497 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56498 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56499 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56500 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56501 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_8
  simp only [← c56496, ← c56497, ← c56498] at e_908_8
  have e_908_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_9
  simp only [← c56499, ← c56500, k2, ← c56501] at e_908_9
  have hr := e_908_9
  simp only [e_908_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7050)) := by
  have c56502 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56503 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56504 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56505 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56506 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56507 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_10
  simp only [← c56502, ← c56503, ← c56504] at e_908_10
  have e_908_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_11
  simp only [← c56505, ← c56506, k2, ← c56507] at e_908_11
  have hr := e_908_11
  simp only [e_908_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7051)) := by
  have c56508 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56509 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56510 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56511 := (publicBatchWrapper89_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56512 := (publicBatchWrapper89_copies1766 a h).1
  have c56513 := (publicBatchWrapper89_copies1766 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_12
  simp only [← c56508, ← c56509, ← c56510] at e_908_12
  have e_908_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_13
  simp only [← c56511, ← c56512, k2, ← c56513] at e_908_13
  have hr := e_908_13
  simp only [e_908_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7052)) := by
  have c56514 := (publicBatchWrapper89_copies1766 a h).2.2.1
  have c56515 := (publicBatchWrapper89_copies1766 a h).2.2.2.1
  have c56516 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.1
  have c56517 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.1
  have c56518 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.1
  have c56519 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_14
  simp only [← c56514, ← c56515, ← c56516] at e_908_14
  have e_908_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_15
  simp only [← c56517, ← c56518, k2, ← c56519] at e_908_15
  have hr := e_908_15
  simp only [e_908_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7053)) := by
  have c56520 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.1
  have c56521 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.1
  have c56522 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56523 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56524 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56525 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_16
  simp only [← c56520, ← c56521, ← c56522] at e_908_16
  have e_908_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_17
  simp only [← c56523, ← c56524, k2, ← c56525] at e_908_17
  have hr := e_908_17
  simp only [e_908_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 908 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7054)) := by
  have c56526 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56527 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56528 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56529 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56530 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56531 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_908_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_18
  simp only [← c56526, ← c56527, ← c56528] at e_908_18
  have e_908_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_19
  simp only [← c56529, ← c56530, k2, ← c56531] at e_908_19
  have hr := e_908_19
  simp only [e_908_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7055)) := by
  have c56532 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56533 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56534 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56535 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56536 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56537 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_0
  simp only [← c56532, ← c56533, ← c56534] at e_909_0
  have e_909_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_1
  simp only [← c56535, ← c56536, k2, ← c56537] at e_909_1
  have hr := e_909_1
  simp only [e_909_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7056)) := by
  have c56538 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56539 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56540 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56541 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56542 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56543 := (publicBatchWrapper89_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_2
  simp only [← c56538, ← c56539, ← c56540] at e_909_2
  have e_909_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_3
  simp only [← c56541, ← c56542, k2, ← c56543] at e_909_3
  have hr := e_909_3
  simp only [e_909_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7057)) := by
  have c56544 := (publicBatchWrapper89_copies1767 a h).1
  have c56545 := (publicBatchWrapper89_copies1767 a h).2.1
  have c56546 := (publicBatchWrapper89_copies1767 a h).2.2.1
  have c56547 := (publicBatchWrapper89_copies1767 a h).2.2.2.1
  have c56548 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.1
  have c56549 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_4
  simp only [← c56544, ← c56545, ← c56546] at e_909_4
  have e_909_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_5
  simp only [← c56547, ← c56548, k2, ← c56549] at e_909_5
  have hr := e_909_5
  simp only [e_909_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7058)) := by
  have c56550 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.1
  have c56551 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.1
  have c56552 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.1
  have c56553 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.1
  have c56554 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56555 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_6
  simp only [← c56550, ← c56551, ← c56552] at e_909_6
  have e_909_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_7
  simp only [← c56553, ← c56554, k2, ← c56555] at e_909_7
  have hr := e_909_7
  simp only [e_909_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7059)) := by
  have c56556 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56557 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56558 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56559 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56560 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56561 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_8
  simp only [← c56556, ← c56557, ← c56558] at e_909_8
  have e_909_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_9
  simp only [← c56559, ← c56560, k2, ← c56561] at e_909_9
  have hr := e_909_9
  simp only [e_909_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7060)) := by
  have c56562 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56563 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56564 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56565 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56566 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56567 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_10
  simp only [← c56562, ← c56563, ← c56564] at e_909_10
  have e_909_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_11
  simp only [← c56565, ← c56566, k2, ← c56567] at e_909_11
  have hr := e_909_11
  simp only [e_909_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7061)) := by
  have c56568 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56569 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56570 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56571 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56572 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56573 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_12
  simp only [← c56568, ← c56569, ← c56570] at e_909_12
  have e_909_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_13
  simp only [← c56571, ← c56572, k2, ← c56573] at e_909_13
  have hr := e_909_13
  simp only [e_909_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7062)) := by
  have c56574 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56575 := (publicBatchWrapper89_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56576 := (publicBatchWrapper89_copies1768 a h).1
  have c56577 := (publicBatchWrapper89_copies1768 a h).2.1
  have c56578 := (publicBatchWrapper89_copies1768 a h).2.2.1
  have c56579 := (publicBatchWrapper89_copies1768 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_14
  simp only [← c56574, ← c56575, ← c56576] at e_909_14
  have e_909_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_15
  simp only [← c56577, ← c56578, k2, ← c56579] at e_909_15
  have hr := e_909_15
  simp only [e_909_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7063)) := by
  have c56580 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.1
  have c56581 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.1
  have c56582 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.1
  have c56583 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.1
  have c56584 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.1
  have c56585 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_16
  simp only [← c56580, ← c56581, ← c56582] at e_909_16
  have e_909_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_17
  simp only [← c56583, ← c56584, k2, ← c56585] at e_909_17
  have hr := e_909_17
  simp only [e_909_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 909 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7064)) := by
  have c56586 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56587 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56588 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56589 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56590 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56591 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_909_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_18
  simp only [← c56586, ← c56587, ← c56588] at e_909_18
  have e_909_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_19
  simp only [← c56589, ← c56590, k2, ← c56591] at e_909_19
  have hr := e_909_19
  simp only [e_909_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7065)) := by
  have c56592 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56593 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56594 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56595 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56596 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56597 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_0
  simp only [← c56592, ← c56593, ← c56594] at e_910_0
  have e_910_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_1
  simp only [← c56595, ← c56596, k2, ← c56597] at e_910_1
  have hr := e_910_1
  simp only [e_910_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7066)) := by
  have c56598 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56599 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56600 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56601 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56602 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56603 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_2
  simp only [← c56598, ← c56599, ← c56600] at e_910_2
  have e_910_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_3
  simp only [← c56601, ← c56602, k2, ← c56603] at e_910_3
  have hr := e_910_3
  simp only [e_910_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7067)) := by
  have c56604 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56605 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56606 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56607 := (publicBatchWrapper89_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56608 := (publicBatchWrapper89_copies1769 a h).1
  have c56609 := (publicBatchWrapper89_copies1769 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_4
  simp only [← c56604, ← c56605, ← c56606] at e_910_4
  have e_910_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_5
  simp only [← c56607, ← c56608, k2, ← c56609] at e_910_5
  have hr := e_910_5
  simp only [e_910_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7068)) := by
  have c56610 := (publicBatchWrapper89_copies1769 a h).2.2.1
  have c56611 := (publicBatchWrapper89_copies1769 a h).2.2.2.1
  have c56612 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.1
  have c56613 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.1
  have c56614 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.1
  have c56615 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_6
  simp only [← c56610, ← c56611, ← c56612] at e_910_6
  have e_910_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_7
  simp only [← c56613, ← c56614, k2, ← c56615] at e_910_7
  have hr := e_910_7
  simp only [e_910_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7069)) := by
  have c56616 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.1
  have c56617 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.1
  have c56618 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56619 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56620 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56621 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_8
  simp only [← c56616, ← c56617, ← c56618] at e_910_8
  have e_910_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_9
  simp only [← c56619, ← c56620, k2, ← c56621] at e_910_9
  have hr := e_910_9
  simp only [e_910_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7070)) := by
  have c56622 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56623 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56624 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56625 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56626 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56627 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_10
  simp only [← c56622, ← c56623, ← c56624] at e_910_10
  have e_910_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_11
  simp only [← c56625, ← c56626, k2, ← c56627] at e_910_11
  have hr := e_910_11
  simp only [e_910_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7071)) := by
  have c56628 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56629 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56630 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56631 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56632 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56633 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_12
  simp only [← c56628, ← c56629, ← c56630] at e_910_12
  have e_910_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_13
  simp only [← c56631, ← c56632, k2, ← c56633] at e_910_13
  have hr := e_910_13
  simp only [e_910_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7072)) := by
  have c56634 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56635 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56636 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56637 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56638 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56639 := (publicBatchWrapper89_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_14
  simp only [← c56634, ← c56635, ← c56636] at e_910_14
  have e_910_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_15
  simp only [← c56637, ← c56638, k2, ← c56639] at e_910_15
  have hr := e_910_15
  simp only [e_910_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7073)) := by
  have c56640 := (publicBatchWrapper89_copies1770 a h).1
  have c56641 := (publicBatchWrapper89_copies1770 a h).2.1
  have c56642 := (publicBatchWrapper89_copies1770 a h).2.2.1
  have c56643 := (publicBatchWrapper89_copies1770 a h).2.2.2.1
  have c56644 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.1
  have c56645 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_16
  simp only [← c56640, ← c56641, ← c56642] at e_910_16
  have e_910_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_17
  simp only [← c56643, ← c56644, k2, ← c56645] at e_910_17
  have hr := e_910_17
  simp only [e_910_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 910 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7074)) := by
  have c56646 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.1
  have c56647 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.1
  have c56648 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.1
  have c56649 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.1
  have c56650 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56651 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_910_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_18
  simp only [← c56646, ← c56647, ← c56648] at e_910_18
  have e_910_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_19
  simp only [← c56649, ← c56650, k2, ← c56651] at e_910_19
  have hr := e_910_19
  simp only [e_910_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7075)) := by
  have c56652 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56653 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56654 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56655 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56656 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56657 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_0
  simp only [← c56652, ← c56653, ← c56654] at e_911_0
  have e_911_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_1
  simp only [← c56655, ← c56656, k2, ← c56657] at e_911_1
  have hr := e_911_1
  simp only [e_911_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7076)) := by
  have c56658 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56659 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56660 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56661 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56662 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56663 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_2
  simp only [← c56658, ← c56659, ← c56660] at e_911_2
  have e_911_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_3
  simp only [← c56661, ← c56662, k2, ← c56663] at e_911_3
  have hr := e_911_3
  simp only [e_911_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7077)) := by
  have c56664 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56665 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56666 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56667 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56668 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56669 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_4
  simp only [← c56664, ← c56665, ← c56666] at e_911_4
  have e_911_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_5
  simp only [← c56667, ← c56668, k2, ← c56669] at e_911_5
  have hr := e_911_5
  simp only [e_911_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7078)) := by
  have c56670 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56671 := (publicBatchWrapper89_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56672 := (publicBatchWrapper89_copies1771 a h).1
  have c56673 := (publicBatchWrapper89_copies1771 a h).2.1
  have c56674 := (publicBatchWrapper89_copies1771 a h).2.2.1
  have c56675 := (publicBatchWrapper89_copies1771 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_6
  simp only [← c56670, ← c56671, ← c56672] at e_911_6
  have e_911_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_7
  simp only [← c56673, ← c56674, k2, ← c56675] at e_911_7
  have hr := e_911_7
  simp only [e_911_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7079)) := by
  have c56676 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.1
  have c56677 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.1
  have c56678 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.1
  have c56679 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.1
  have c56680 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.1
  have c56681 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_8
  simp only [← c56676, ← c56677, ← c56678] at e_911_8
  have e_911_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_9
  simp only [← c56679, ← c56680, k2, ← c56681] at e_911_9
  have hr := e_911_9
  simp only [e_911_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7080)) := by
  have c56682 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56683 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56684 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56685 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56686 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56687 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_10
  simp only [← c56682, ← c56683, ← c56684] at e_911_10
  have e_911_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_11
  simp only [← c56685, ← c56686, k2, ← c56687] at e_911_11
  have hr := e_911_11
  simp only [e_911_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7081)) := by
  have c56688 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56689 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56690 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56691 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56692 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56693 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_12
  simp only [← c56688, ← c56689, ← c56690] at e_911_12
  have e_911_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_13
  simp only [← c56691, ← c56692, k2, ← c56693] at e_911_13
  have hr := e_911_13
  simp only [e_911_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7082)) := by
  have c56694 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56695 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56696 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56697 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56698 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56699 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_14
  simp only [← c56694, ← c56695, ← c56696] at e_911_14
  have e_911_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_15
  simp only [← c56697, ← c56698, k2, ← c56699] at e_911_15
  have hr := e_911_15
  simp only [e_911_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7083)) := by
  have c56700 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56701 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56702 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56703 := (publicBatchWrapper89_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56704 := (publicBatchWrapper89_copies1772 a h).1
  have c56705 := (publicBatchWrapper89_copies1772 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_16
  simp only [← c56700, ← c56701, ← c56702] at e_911_16
  have e_911_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_17
  simp only [← c56703, ← c56704, k2, ← c56705] at e_911_17
  have hr := e_911_17
  simp only [e_911_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 911 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7084)) := by
  have c56706 := (publicBatchWrapper89_copies1772 a h).2.2.1
  have c56707 := (publicBatchWrapper89_copies1772 a h).2.2.2.1
  have c56708 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.1
  have c56709 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.1
  have c56710 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.1
  have c56711 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_911_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_18
  simp only [← c56706, ← c56707, ← c56708] at e_911_18
  have e_911_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_19
  simp only [← c56709, ← c56710, k2, ← c56711] at e_911_19
  have hr := e_911_19
  simp only [e_911_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7085)) := by
  have c56712 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.1
  have c56713 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.1
  have c56714 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56715 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56716 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56717 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_0
  simp only [← c56712, ← c56713, ← c56714] at e_912_0
  have e_912_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_1
  simp only [← c56715, ← c56716, k2, ← c56717] at e_912_1
  have hr := e_912_1
  simp only [e_912_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7086)) := by
  have c56718 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56719 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56720 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56721 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56722 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56723 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_2
  simp only [← c56718, ← c56719, ← c56720] at e_912_2
  have e_912_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_3
  simp only [← c56721, ← c56722, k2, ← c56723] at e_912_3
  have hr := e_912_3
  simp only [e_912_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7087)) := by
  have c56724 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56725 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56726 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56727 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56728 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56729 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_4
  simp only [← c56724, ← c56725, ← c56726] at e_912_4
  have e_912_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_5
  simp only [← c56727, ← c56728, k2, ← c56729] at e_912_5
  have hr := e_912_5
  simp only [e_912_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7088)) := by
  have c56730 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56731 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56732 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56733 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56734 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56735 := (publicBatchWrapper89_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_6
  simp only [← c56730, ← c56731, ← c56732] at e_912_6
  have e_912_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_7
  simp only [← c56733, ← c56734, k2, ← c56735] at e_912_7
  have hr := e_912_7
  simp only [e_912_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7089)) := by
  have c56736 := (publicBatchWrapper89_copies1773 a h).1
  have c56737 := (publicBatchWrapper89_copies1773 a h).2.1
  have c56738 := (publicBatchWrapper89_copies1773 a h).2.2.1
  have c56739 := (publicBatchWrapper89_copies1773 a h).2.2.2.1
  have c56740 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.1
  have c56741 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_8
  simp only [← c56736, ← c56737, ← c56738] at e_912_8
  have e_912_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_9
  simp only [← c56739, ← c56740, k2, ← c56741] at e_912_9
  have hr := e_912_9
  simp only [e_912_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7090)) := by
  have c56742 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.1
  have c56743 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.1
  have c56744 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.1
  have c56745 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.1
  have c56746 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56747 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_10
  simp only [← c56742, ← c56743, ← c56744] at e_912_10
  have e_912_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_11
  simp only [← c56745, ← c56746, k2, ← c56747] at e_912_11
  have hr := e_912_11
  simp only [e_912_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7091)) := by
  have c56748 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56749 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56750 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56751 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56752 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56753 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_12
  simp only [← c56748, ← c56749, ← c56750] at e_912_12
  have e_912_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_13
  simp only [← c56751, ← c56752, k2, ← c56753] at e_912_13
  have hr := e_912_13
  simp only [e_912_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7092)) := by
  have c56754 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56755 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56756 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56757 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56758 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56759 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_14
  simp only [← c56754, ← c56755, ← c56756] at e_912_14
  have e_912_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_15
  simp only [← c56757, ← c56758, k2, ← c56759] at e_912_15
  have hr := e_912_15
  simp only [e_912_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7093)) := by
  have c56760 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56761 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56762 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56763 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56764 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56765 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_16
  simp only [← c56760, ← c56761, ← c56762] at e_912_16
  have e_912_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_17
  simp only [← c56763, ← c56764, k2, ← c56765] at e_912_17
  have hr := e_912_17
  simp only [e_912_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 912 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7094)) := by
  have c56766 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56767 := (publicBatchWrapper89_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56768 := (publicBatchWrapper89_copies1774 a h).1
  have c56769 := (publicBatchWrapper89_copies1774 a h).2.1
  have c56770 := (publicBatchWrapper89_copies1774 a h).2.2.1
  have c56771 := (publicBatchWrapper89_copies1774 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_912_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_18
  simp only [← c56766, ← c56767, ← c56768] at e_912_18
  have e_912_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_19
  simp only [← c56769, ← c56770, k2, ← c56771] at e_912_19
  have hr := e_912_19
  simp only [e_912_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7095)) := by
  have c56772 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.1
  have c56773 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.1
  have c56774 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.1
  have c56775 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.1
  have c56776 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.1
  have c56777 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_0
  simp only [← c56772, ← c56773, ← c56774] at e_913_0
  have e_913_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_1
  simp only [← c56775, ← c56776, k2, ← c56777] at e_913_1
  have hr := e_913_1
  simp only [e_913_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7096)) := by
  have c56778 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56779 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56780 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56781 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56782 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56783 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_2
  simp only [← c56778, ← c56779, ← c56780] at e_913_2
  have e_913_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_3
  simp only [← c56781, ← c56782, k2, ← c56783] at e_913_3
  have hr := e_913_3
  simp only [e_913_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7097)) := by
  have c56784 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56785 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56786 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56787 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56788 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56789 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_4
  simp only [← c56784, ← c56785, ← c56786] at e_913_4
  have e_913_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_5
  simp only [← c56787, ← c56788, k2, ← c56789] at e_913_5
  have hr := e_913_5
  simp only [e_913_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7098)) := by
  have c56790 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56791 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56792 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56793 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56794 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56795 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_6
  simp only [← c56790, ← c56791, ← c56792] at e_913_6
  have e_913_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_7
  simp only [← c56793, ← c56794, k2, ← c56795] at e_913_7
  have hr := e_913_7
  simp only [e_913_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7099)) := by
  have c56796 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56797 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56798 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56799 := (publicBatchWrapper89_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56800 := (publicBatchWrapper89_copies1775 a h).1
  have c56801 := (publicBatchWrapper89_copies1775 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_8
  simp only [← c56796, ← c56797, ← c56798] at e_913_8
  have e_913_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_9
  simp only [← c56799, ← c56800, k2, ← c56801] at e_913_9
  have hr := e_913_9
  simp only [e_913_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7100)) := by
  have c56802 := (publicBatchWrapper89_copies1775 a h).2.2.1
  have c56803 := (publicBatchWrapper89_copies1775 a h).2.2.2.1
  have c56804 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.1
  have c56805 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.1
  have c56806 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.1
  have c56807 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_10
  simp only [← c56802, ← c56803, ← c56804] at e_913_10
  have e_913_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_11
  simp only [← c56805, ← c56806, k2, ← c56807] at e_913_11
  have hr := e_913_11
  simp only [e_913_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7101)) := by
  have c56808 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.1
  have c56809 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.1
  have c56810 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56811 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56812 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56813 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_12
  simp only [← c56808, ← c56809, ← c56810] at e_913_12
  have e_913_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_13
  simp only [← c56811, ← c56812, k2, ← c56813] at e_913_13
  have hr := e_913_13
  simp only [e_913_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7102)) := by
  have c56814 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56815 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56816 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56817 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56818 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56819 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_14
  simp only [← c56814, ← c56815, ← c56816] at e_913_14
  have e_913_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_15
  simp only [← c56817, ← c56818, k2, ← c56819] at e_913_15
  have hr := e_913_15
  simp only [e_913_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7103)) := by
  have c56820 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56821 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56822 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56823 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56824 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56825 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_16
  simp only [← c56820, ← c56821, ← c56822] at e_913_16
  have e_913_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_17
  simp only [← c56823, ← c56824, k2, ← c56825] at e_913_17
  have hr := e_913_17
  simp only [e_913_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 913 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7104)) := by
  have c56826 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56827 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56828 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56829 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56830 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56831 := (publicBatchWrapper89_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_913_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_18
  simp only [← c56826, ← c56827, ← c56828] at e_913_18
  have e_913_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_19
  simp only [← c56829, ← c56830, k2, ← c56831] at e_913_19
  have hr := e_913_19
  simp only [e_913_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7105)) := by
  have c56832 := (publicBatchWrapper89_copies1776 a h).1
  have c56833 := (publicBatchWrapper89_copies1776 a h).2.1
  have c56834 := (publicBatchWrapper89_copies1776 a h).2.2.1
  have c56835 := (publicBatchWrapper89_copies1776 a h).2.2.2.1
  have c56836 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.1
  have c56837 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_0
  simp only [← c56832, ← c56833, ← c56834] at e_914_0
  have e_914_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_1
  simp only [← c56835, ← c56836, k2, ← c56837] at e_914_1
  have hr := e_914_1
  simp only [e_914_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7106)) := by
  have c56838 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.1
  have c56839 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.1
  have c56840 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.1
  have c56841 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.1
  have c56842 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56843 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_2
  simp only [← c56838, ← c56839, ← c56840] at e_914_2
  have e_914_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_3
  simp only [← c56841, ← c56842, k2, ← c56843] at e_914_3
  have hr := e_914_3
  simp only [e_914_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7107)) := by
  have c56844 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56845 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56846 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56847 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56848 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56849 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_4
  simp only [← c56844, ← c56845, ← c56846] at e_914_4
  have e_914_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_5
  simp only [← c56847, ← c56848, k2, ← c56849] at e_914_5
  have hr := e_914_5
  simp only [e_914_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7108)) := by
  have c56850 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56851 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56852 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56853 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56854 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56855 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_6
  simp only [← c56850, ← c56851, ← c56852] at e_914_6
  have e_914_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_7
  simp only [← c56853, ← c56854, k2, ← c56855] at e_914_7
  have hr := e_914_7
  simp only [e_914_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7109)) := by
  have c56856 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56857 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56858 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56859 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56860 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56861 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_8
  simp only [← c56856, ← c56857, ← c56858] at e_914_8
  have e_914_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_9
  simp only [← c56859, ← c56860, k2, ← c56861] at e_914_9
  have hr := e_914_9
  simp only [e_914_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7110)) := by
  have c56862 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56863 := (publicBatchWrapper89_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56864 := (publicBatchWrapper89_copies1777 a h).1
  have c56865 := (publicBatchWrapper89_copies1777 a h).2.1
  have c56866 := (publicBatchWrapper89_copies1777 a h).2.2.1
  have c56867 := (publicBatchWrapper89_copies1777 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_10
  simp only [← c56862, ← c56863, ← c56864] at e_914_10
  have e_914_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_11
  simp only [← c56865, ← c56866, k2, ← c56867] at e_914_11
  have hr := e_914_11
  simp only [e_914_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7111)) := by
  have c56868 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.1
  have c56869 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.1
  have c56870 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.1
  have c56871 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.1
  have c56872 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.1
  have c56873 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_12
  simp only [← c56868, ← c56869, ← c56870] at e_914_12
  have e_914_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_13
  simp only [← c56871, ← c56872, k2, ← c56873] at e_914_13
  have hr := e_914_13
  simp only [e_914_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7112)) := by
  have c56874 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56875 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56876 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56877 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56878 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56879 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_14
  simp only [← c56874, ← c56875, ← c56876] at e_914_14
  have e_914_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_15
  simp only [← c56877, ← c56878, k2, ← c56879] at e_914_15
  have hr := e_914_15
  simp only [e_914_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7113)) := by
  have c56880 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56881 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56882 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56883 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56884 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56885 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_16
  simp only [← c56880, ← c56881, ← c56882] at e_914_16
  have e_914_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_17
  simp only [← c56883, ← c56884, k2, ← c56885] at e_914_17
  have hr := e_914_17
  simp only [e_914_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 914 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7114)) := by
  have c56886 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56887 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56888 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56889 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56890 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56891 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_914_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_18
  simp only [← c56886, ← c56887, ← c56888] at e_914_18
  have e_914_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_19
  simp only [← c56889, ← c56890, k2, ← c56891] at e_914_19
  have hr := e_914_19
  simp only [e_914_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7115)) := by
  have c56892 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56893 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56894 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56895 := (publicBatchWrapper89_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56896 := (publicBatchWrapper89_copies1778 a h).1
  have c56897 := (publicBatchWrapper89_copies1778 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_0
  simp only [← c56892, ← c56893, ← c56894] at e_915_0
  have e_915_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_1
  simp only [← c56895, ← c56896, k2, ← c56897] at e_915_1
  have hr := e_915_1
  simp only [e_915_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7116)) := by
  have c56898 := (publicBatchWrapper89_copies1778 a h).2.2.1
  have c56899 := (publicBatchWrapper89_copies1778 a h).2.2.2.1
  have c56900 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.1
  have c56901 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.1
  have c56902 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.1
  have c56903 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_2
  simp only [← c56898, ← c56899, ← c56900] at e_915_2
  have e_915_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_3
  simp only [← c56901, ← c56902, k2, ← c56903] at e_915_3
  have hr := e_915_3
  simp only [e_915_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7117)) := by
  have c56904 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.1
  have c56905 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.1
  have c56906 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56907 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56908 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56909 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_4
  simp only [← c56904, ← c56905, ← c56906] at e_915_4
  have e_915_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_5
  simp only [← c56907, ← c56908, k2, ← c56909] at e_915_5
  have hr := e_915_5
  simp only [e_915_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7118)) := by
  have c56910 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56911 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56912 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56913 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56914 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56915 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_6
  simp only [← c56910, ← c56911, ← c56912] at e_915_6
  have e_915_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_7
  simp only [← c56913, ← c56914, k2, ← c56915] at e_915_7
  have hr := e_915_7
  simp only [e_915_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7119)) := by
  have c56916 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56917 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56918 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56919 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56920 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56921 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_8
  simp only [← c56916, ← c56917, ← c56918] at e_915_8
  have e_915_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_9
  simp only [← c56919, ← c56920, k2, ← c56921] at e_915_9
  have hr := e_915_9
  simp only [e_915_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7120)) := by
  have c56922 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56923 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56924 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56925 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56926 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56927 := (publicBatchWrapper89_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_10
  simp only [← c56922, ← c56923, ← c56924] at e_915_10
  have e_915_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_11
  simp only [← c56925, ← c56926, k2, ← c56927] at e_915_11
  have hr := e_915_11
  simp only [e_915_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7121)) := by
  have c56928 := (publicBatchWrapper89_copies1779 a h).1
  have c56929 := (publicBatchWrapper89_copies1779 a h).2.1
  have c56930 := (publicBatchWrapper89_copies1779 a h).2.2.1
  have c56931 := (publicBatchWrapper89_copies1779 a h).2.2.2.1
  have c56932 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.1
  have c56933 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_12
  simp only [← c56928, ← c56929, ← c56930] at e_915_12
  have e_915_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_13
  simp only [← c56931, ← c56932, k2, ← c56933] at e_915_13
  have hr := e_915_13
  simp only [e_915_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7122)) := by
  have c56934 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.1
  have c56935 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.1
  have c56936 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.1
  have c56937 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.1
  have c56938 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56939 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_14
  simp only [← c56934, ← c56935, ← c56936] at e_915_14
  have e_915_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_15
  simp only [← c56937, ← c56938, k2, ← c56939] at e_915_15
  have hr := e_915_15
  simp only [e_915_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7123)) := by
  have c56940 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56941 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56942 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56943 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56944 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56945 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_16
  simp only [← c56940, ← c56941, ← c56942] at e_915_16
  have e_915_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_17
  simp only [← c56943, ← c56944, k2, ← c56945] at e_915_17
  have hr := e_915_17
  simp only [e_915_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 915 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7124)) := by
  have c56946 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56947 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56948 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56949 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56950 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56951 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_915_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_18
  simp only [← c56946, ← c56947, ← c56948] at e_915_18
  have e_915_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_19
  simp only [← c56949, ← c56950, k2, ← c56951] at e_915_19
  have hr := e_915_19
  simp only [e_915_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7125)) := by
  have c56952 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56953 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56954 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56955 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56956 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56957 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_0
  simp only [← c56952, ← c56953, ← c56954] at e_916_0
  have e_916_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_1
  simp only [← c56955, ← c56956, k2, ← c56957] at e_916_1
  have hr := e_916_1
  simp only [e_916_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7126)) := by
  have c56958 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56959 := (publicBatchWrapper89_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56960 := (publicBatchWrapper89_copies1780 a h).1
  have c56961 := (publicBatchWrapper89_copies1780 a h).2.1
  have c56962 := (publicBatchWrapper89_copies1780 a h).2.2.1
  have c56963 := (publicBatchWrapper89_copies1780 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_2
  simp only [← c56958, ← c56959, ← c56960] at e_916_2
  have e_916_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_3
  simp only [← c56961, ← c56962, k2, ← c56963] at e_916_3
  have hr := e_916_3
  simp only [e_916_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7127)) := by
  have c56964 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.1
  have c56965 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.1
  have c56966 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.1
  have c56967 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.1
  have c56968 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.1
  have c56969 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_4
  simp only [← c56964, ← c56965, ← c56966] at e_916_4
  have e_916_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_5
  simp only [← c56967, ← c56968, k2, ← c56969] at e_916_5
  have hr := e_916_5
  simp only [e_916_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7128)) := by
  have c56970 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56971 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56972 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56973 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56974 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56975 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_6
  simp only [← c56970, ← c56971, ← c56972] at e_916_6
  have e_916_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_7
  simp only [← c56973, ← c56974, k2, ← c56975] at e_916_7
  have hr := e_916_7
  simp only [e_916_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7129)) := by
  have c56976 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56977 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56978 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56979 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56980 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56981 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_8
  simp only [← c56976, ← c56977, ← c56978] at e_916_8
  have e_916_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_9
  simp only [← c56979, ← c56980, k2, ← c56981] at e_916_9
  have hr := e_916_9
  simp only [e_916_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7130)) := by
  have c56982 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56983 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56984 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56985 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56986 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56987 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_10
  simp only [← c56982, ← c56983, ← c56984] at e_916_10
  have e_916_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_11
  simp only [← c56985, ← c56986, k2, ← c56987] at e_916_11
  have hr := e_916_11
  simp only [e_916_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7131)) := by
  have c56988 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56989 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56990 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56991 := (publicBatchWrapper89_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56992 := (publicBatchWrapper89_copies1781 a h).1
  have c56993 := (publicBatchWrapper89_copies1781 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_12
  simp only [← c56988, ← c56989, ← c56990] at e_916_12
  have e_916_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_13
  simp only [← c56991, ← c56992, k2, ← c56993] at e_916_13
  have hr := e_916_13
  simp only [e_916_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7132)) := by
  have c56994 := (publicBatchWrapper89_copies1781 a h).2.2.1
  have c56995 := (publicBatchWrapper89_copies1781 a h).2.2.2.1
  have c56996 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.1
  have c56997 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.1
  have c56998 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.1
  have c56999 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_14
  simp only [← c56994, ← c56995, ← c56996] at e_916_14
  have e_916_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_15
  simp only [← c56997, ← c56998, k2, ← c56999] at e_916_15
  have hr := e_916_15
  simp only [e_916_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7133)) := by
  have c57000 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.1
  have c57001 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.1
  have c57002 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57003 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57004 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57005 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_16
  simp only [← c57000, ← c57001, ← c57002] at e_916_16
  have e_916_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_17
  simp only [← c57003, ← c57004, k2, ← c57005] at e_916_17
  have hr := e_916_17
  simp only [e_916_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 916 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7134)) := by
  have c57006 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57007 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57008 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57009 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57010 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57011 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_916_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_18
  simp only [← c57006, ← c57007, ← c57008] at e_916_18
  have e_916_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_19
  simp only [← c57009, ← c57010, k2, ← c57011] at e_916_19
  have hr := e_916_19
  simp only [e_916_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7135)) := by
  have c57012 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57013 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57014 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57015 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57016 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57017 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_0
  simp only [← c57012, ← c57013, ← c57014] at e_917_0
  have e_917_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_1
  simp only [← c57015, ← c57016, k2, ← c57017] at e_917_1
  have hr := e_917_1
  simp only [e_917_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7136)) := by
  have c57018 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57019 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57020 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57021 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57022 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57023 := (publicBatchWrapper89_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_2
  simp only [← c57018, ← c57019, ← c57020] at e_917_2
  have e_917_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_3
  simp only [← c57021, ← c57022, k2, ← c57023] at e_917_3
  have hr := e_917_3
  simp only [e_917_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7137)) := by
  have c57024 := (publicBatchWrapper89_copies1782 a h).1
  have c57025 := (publicBatchWrapper89_copies1782 a h).2.1
  have c57026 := (publicBatchWrapper89_copies1782 a h).2.2.1
  have c57027 := (publicBatchWrapper89_copies1782 a h).2.2.2.1
  have c57028 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.1
  have c57029 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_4
  simp only [← c57024, ← c57025, ← c57026] at e_917_4
  have e_917_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_5
  simp only [← c57027, ← c57028, k2, ← c57029] at e_917_5
  have hr := e_917_5
  simp only [e_917_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7138)) := by
  have c57030 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.1
  have c57031 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.1
  have c57032 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.1
  have c57033 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.1
  have c57034 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57035 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_6
  simp only [← c57030, ← c57031, ← c57032] at e_917_6
  have e_917_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_7
  simp only [← c57033, ← c57034, k2, ← c57035] at e_917_7
  have hr := e_917_7
  simp only [e_917_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7139)) := by
  have c57036 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57037 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57038 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57039 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57040 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57041 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_8
  simp only [← c57036, ← c57037, ← c57038] at e_917_8
  have e_917_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_9
  simp only [← c57039, ← c57040, k2, ← c57041] at e_917_9
  have hr := e_917_9
  simp only [e_917_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7140)) := by
  have c57042 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57043 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57044 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57045 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57046 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57047 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_10
  simp only [← c57042, ← c57043, ← c57044] at e_917_10
  have e_917_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_11
  simp only [← c57045, ← c57046, k2, ← c57047] at e_917_11
  have hr := e_917_11
  simp only [e_917_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7141)) := by
  have c57048 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57049 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57050 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57051 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57052 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57053 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_12
  simp only [← c57048, ← c57049, ← c57050] at e_917_12
  have e_917_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_13
  simp only [← c57051, ← c57052, k2, ← c57053] at e_917_13
  have hr := e_917_13
  simp only [e_917_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7142)) := by
  have c57054 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57055 := (publicBatchWrapper89_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57056 := (publicBatchWrapper89_copies1783 a h).1
  have c57057 := (publicBatchWrapper89_copies1783 a h).2.1
  have c57058 := (publicBatchWrapper89_copies1783 a h).2.2.1
  have c57059 := (publicBatchWrapper89_copies1783 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_14
  simp only [← c57054, ← c57055, ← c57056] at e_917_14
  have e_917_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_15
  simp only [← c57057, ← c57058, k2, ← c57059] at e_917_15
  have hr := e_917_15
  simp only [e_917_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7143)) := by
  have c57060 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.1
  have c57061 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.1
  have c57062 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.1
  have c57063 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.1
  have c57064 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.1
  have c57065 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_16
  simp only [← c57060, ← c57061, ← c57062] at e_917_16
  have e_917_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_17
  simp only [← c57063, ← c57064, k2, ← c57065] at e_917_17
  have hr := e_917_17
  simp only [e_917_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 917 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7144)) := by
  have c57066 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57067 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57068 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57069 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57070 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57071 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_917_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_18
  simp only [← c57066, ← c57067, ← c57068] at e_917_18
  have e_917_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_19
  simp only [← c57069, ← c57070, k2, ← c57071] at e_917_19
  have hr := e_917_19
  simp only [e_917_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7145)) := by
  have c57072 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57073 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57074 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57075 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57076 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57077 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_0
  simp only [← c57072, ← c57073, ← c57074] at e_918_0
  have e_918_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_1
  simp only [← c57075, ← c57076, k2, ← c57077] at e_918_1
  have hr := e_918_1
  simp only [e_918_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7146)) := by
  have c57078 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57079 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57080 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57081 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57082 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57083 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_2
  simp only [← c57078, ← c57079, ← c57080] at e_918_2
  have e_918_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_3
  simp only [← c57081, ← c57082, k2, ← c57083] at e_918_3
  have hr := e_918_3
  simp only [e_918_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7147)) := by
  have c57084 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57085 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57086 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57087 := (publicBatchWrapper89_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57088 := (publicBatchWrapper89_copies1784 a h).1
  have c57089 := (publicBatchWrapper89_copies1784 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_4
  simp only [← c57084, ← c57085, ← c57086] at e_918_4
  have e_918_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_5
  simp only [← c57087, ← c57088, k2, ← c57089] at e_918_5
  have hr := e_918_5
  simp only [e_918_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7148)) := by
  have c57090 := (publicBatchWrapper89_copies1784 a h).2.2.1
  have c57091 := (publicBatchWrapper89_copies1784 a h).2.2.2.1
  have c57092 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.1
  have c57093 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.1
  have c57094 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.1
  have c57095 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_6
  simp only [← c57090, ← c57091, ← c57092] at e_918_6
  have e_918_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_7
  simp only [← c57093, ← c57094, k2, ← c57095] at e_918_7
  have hr := e_918_7
  simp only [e_918_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7149)) := by
  have c57096 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.1
  have c57097 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.1
  have c57098 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57099 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57100 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57101 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_8
  simp only [← c57096, ← c57097, ← c57098] at e_918_8
  have e_918_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_9
  simp only [← c57099, ← c57100, k2, ← c57101] at e_918_9
  have hr := e_918_9
  simp only [e_918_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7150)) := by
  have c57102 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57103 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57104 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57105 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57106 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57107 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_10
  simp only [← c57102, ← c57103, ← c57104] at e_918_10
  have e_918_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_11
  simp only [← c57105, ← c57106, k2, ← c57107] at e_918_11
  have hr := e_918_11
  simp only [e_918_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7151)) := by
  have c57108 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57109 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57110 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57111 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57112 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57113 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_12
  simp only [← c57108, ← c57109, ← c57110] at e_918_12
  have e_918_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_13
  simp only [← c57111, ← c57112, k2, ← c57113] at e_918_13
  have hr := e_918_13
  simp only [e_918_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7152)) := by
  have c57114 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57115 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57116 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57117 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57118 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57119 := (publicBatchWrapper89_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_14
  simp only [← c57114, ← c57115, ← c57116] at e_918_14
  have e_918_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_15
  simp only [← c57117, ← c57118, k2, ← c57119] at e_918_15
  have hr := e_918_15
  simp only [e_918_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7153)) := by
  have c57120 := (publicBatchWrapper89_copies1785 a h).1
  have c57121 := (publicBatchWrapper89_copies1785 a h).2.1
  have c57122 := (publicBatchWrapper89_copies1785 a h).2.2.1
  have c57123 := (publicBatchWrapper89_copies1785 a h).2.2.2.1
  have c57124 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.1
  have c57125 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_16
  simp only [← c57120, ← c57121, ← c57122] at e_918_16
  have e_918_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_17
  simp only [← c57123, ← c57124, k2, ← c57125] at e_918_17
  have hr := e_918_17
  simp only [e_918_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 918 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7154)) := by
  have c57126 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.1
  have c57127 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.1
  have c57128 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.1
  have c57129 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.1
  have c57130 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57131 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_918_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_18
  simp only [← c57126, ← c57127, ← c57128] at e_918_18
  have e_918_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_19
  simp only [← c57129, ← c57130, k2, ← c57131] at e_918_19
  have hr := e_918_19
  simp only [e_918_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7155)) := by
  have c57132 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57133 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57134 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57135 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57136 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57137 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_0
  simp only [← c57132, ← c57133, ← c57134] at e_919_0
  have e_919_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_1
  simp only [← c57135, ← c57136, k2, ← c57137] at e_919_1
  have hr := e_919_1
  simp only [e_919_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7156)) := by
  have c57138 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57139 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57140 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57141 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57142 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57143 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_2
  simp only [← c57138, ← c57139, ← c57140] at e_919_2
  have e_919_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_3
  simp only [← c57141, ← c57142, k2, ← c57143] at e_919_3
  have hr := e_919_3
  simp only [e_919_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7157)) := by
  have c57144 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57145 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57146 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57147 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57148 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57149 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_4
  simp only [← c57144, ← c57145, ← c57146] at e_919_4
  have e_919_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_5
  simp only [← c57147, ← c57148, k2, ← c57149] at e_919_5
  have hr := e_919_5
  simp only [e_919_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7158)) := by
  have c57150 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57151 := (publicBatchWrapper89_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57152 := (publicBatchWrapper89_copies1786 a h).1
  have c57153 := (publicBatchWrapper89_copies1786 a h).2.1
  have c57154 := (publicBatchWrapper89_copies1786 a h).2.2.1
  have c57155 := (publicBatchWrapper89_copies1786 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_6
  simp only [← c57150, ← c57151, ← c57152] at e_919_6
  have e_919_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_7
  simp only [← c57153, ← c57154, k2, ← c57155] at e_919_7
  have hr := e_919_7
  simp only [e_919_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7159)) := by
  have c57156 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.1
  have c57157 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.1
  have c57158 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.1
  have c57159 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.1
  have c57160 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.1
  have c57161 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_8
  simp only [← c57156, ← c57157, ← c57158] at e_919_8
  have e_919_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_9
  simp only [← c57159, ← c57160, k2, ← c57161] at e_919_9
  have hr := e_919_9
  simp only [e_919_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7160)) := by
  have c57162 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57163 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57164 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57165 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57166 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57167 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_10
  simp only [← c57162, ← c57163, ← c57164] at e_919_10
  have e_919_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_11
  simp only [← c57165, ← c57166, k2, ← c57167] at e_919_11
  have hr := e_919_11
  simp only [e_919_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7161)) := by
  have c57168 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57169 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57170 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57171 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57172 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57173 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_12
  simp only [← c57168, ← c57169, ← c57170] at e_919_12
  have e_919_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_13
  simp only [← c57171, ← c57172, k2, ← c57173] at e_919_13
  have hr := e_919_13
  simp only [e_919_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7162)) := by
  have c57174 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57175 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57176 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57177 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57178 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57179 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_14
  simp only [← c57174, ← c57175, ← c57176] at e_919_14
  have e_919_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_15
  simp only [← c57177, ← c57178, k2, ← c57179] at e_919_15
  have hr := e_919_15
  simp only [e_919_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7163)) := by
  have c57180 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57181 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57182 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57183 := (publicBatchWrapper89_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57184 := (publicBatchWrapper89_copies1787 a h).1
  have c57185 := (publicBatchWrapper89_copies1787 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_16
  simp only [← c57180, ← c57181, ← c57182] at e_919_16
  have e_919_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_17
  simp only [← c57183, ← c57184, k2, ← c57185] at e_919_17
  have hr := e_919_17
  simp only [e_919_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 919 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7164)) := by
  have c57186 := (publicBatchWrapper89_copies1787 a h).2.2.1
  have c57187 := (publicBatchWrapper89_copies1787 a h).2.2.2.1
  have c57188 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.1
  have c57189 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.1
  have c57190 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.1
  have c57191 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_919_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_18
  simp only [← c57186, ← c57187, ← c57188] at e_919_18
  have e_919_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_19
  simp only [← c57189, ← c57190, k2, ← c57191] at e_919_19
  have hr := e_919_19
  simp only [e_919_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7165)) := by
  have c57192 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.1
  have c57193 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.1
  have c57194 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57195 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57196 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57197 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_0
  simp only [← c57192, ← c57193, ← c57194] at e_920_0
  have e_920_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_1
  simp only [← c57195, ← c57196, k2, ← c57197] at e_920_1
  have hr := e_920_1
  simp only [e_920_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7166)) := by
  have c57198 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57199 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57200 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57201 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57202 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57203 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_2
  simp only [← c57198, ← c57199, ← c57200] at e_920_2
  have e_920_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_3
  simp only [← c57201, ← c57202, k2, ← c57203] at e_920_3
  have hr := e_920_3
  simp only [e_920_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7167)) := by
  have c57204 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57205 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57206 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57207 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57208 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57209 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_4
  simp only [← c57204, ← c57205, ← c57206] at e_920_4
  have e_920_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_5
  simp only [← c57207, ← c57208, k2, ← c57209] at e_920_5
  have hr := e_920_5
  simp only [e_920_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7168)) := by
  have c57210 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57211 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57212 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57213 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57214 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57215 := (publicBatchWrapper89_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_6
  simp only [← c57210, ← c57211, ← c57212] at e_920_6
  have e_920_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_7
  simp only [← c57213, ← c57214, k2, ← c57215] at e_920_7
  have hr := e_920_7
  simp only [e_920_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7169)) := by
  have c57216 := (publicBatchWrapper89_copies1788 a h).1
  have c57217 := (publicBatchWrapper89_copies1788 a h).2.1
  have c57218 := (publicBatchWrapper89_copies1788 a h).2.2.1
  have c57219 := (publicBatchWrapper89_copies1788 a h).2.2.2.1
  have c57220 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.1
  have c57221 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_8
  simp only [← c57216, ← c57217, ← c57218] at e_920_8
  have e_920_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_9
  simp only [← c57219, ← c57220, k2, ← c57221] at e_920_9
  have hr := e_920_9
  simp only [e_920_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7210)) := by
  have c57222 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.1
  have c57223 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.1
  have c57224 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.1
  have c57225 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.1
  have c57226 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57227 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_10
  simp only [← c57222, ← c57223, ← c57224] at e_920_10
  have e_920_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_11
  simp only [← c57225, ← c57226, k2, ← c57227] at e_920_11
  have hr := e_920_11
  simp only [e_920_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7211)) := by
  have c57228 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57229 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57230 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57231 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57232 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57233 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_12
  simp only [← c57228, ← c57229, ← c57230] at e_920_12
  have e_920_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_13
  simp only [← c57231, ← c57232, k2, ← c57233] at e_920_13
  have hr := e_920_13
  simp only [e_920_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7212)) := by
  have c57234 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57235 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57236 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57237 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57238 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57239 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_14
  simp only [← c57234, ← c57235, ← c57236] at e_920_14
  have e_920_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_15
  simp only [← c57237, ← c57238, k2, ← c57239] at e_920_15
  have hr := e_920_15
  simp only [e_920_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7213)) := by
  have c57240 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57241 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57242 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57243 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57244 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57245 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_16
  simp only [← c57240, ← c57241, ← c57242] at e_920_16
  have e_920_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_17
  simp only [← c57243, ← c57244, k2, ← c57245] at e_920_17
  have hr := e_920_17
  simp only [e_920_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 920 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7214)) := by
  have c57246 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57247 := (publicBatchWrapper89_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57248 := (publicBatchWrapper89_copies1789 a h).1
  have c57249 := (publicBatchWrapper89_copies1789 a h).2.1
  have c57250 := (publicBatchWrapper89_copies1789 a h).2.2.1
  have c57251 := (publicBatchWrapper89_copies1789 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_920_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_18
  simp only [← c57246, ← c57247, ← c57248] at e_920_18
  have e_920_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_19
  simp only [← c57249, ← c57250, k2, ← c57251] at e_920_19
  have hr := e_920_19
  simp only [e_920_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7215)) := by
  have c57252 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.1
  have c57253 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.1
  have c57254 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.1
  have c57255 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.1
  have c57256 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.1
  have c57257 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_0
  simp only [← c57252, ← c57253, ← c57254] at e_921_0
  have e_921_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_1
  simp only [← c57255, ← c57256, k2, ← c57257] at e_921_1
  have hr := e_921_1
  simp only [e_921_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7216)) := by
  have c57258 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57259 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57260 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57261 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57262 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57263 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_2
  simp only [← c57258, ← c57259, ← c57260] at e_921_2
  have e_921_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_3
  simp only [← c57261, ← c57262, k2, ← c57263] at e_921_3
  have hr := e_921_3
  simp only [e_921_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7217)) := by
  have c57264 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57265 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57266 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57267 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57268 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57269 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_4
  simp only [← c57264, ← c57265, ← c57266] at e_921_4
  have e_921_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_5
  simp only [← c57267, ← c57268, k2, ← c57269] at e_921_5
  have hr := e_921_5
  simp only [e_921_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7218)) := by
  have c57270 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57271 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57272 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57273 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57274 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57275 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_6
  simp only [← c57270, ← c57271, ← c57272] at e_921_6
  have e_921_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_7
  simp only [← c57273, ← c57274, k2, ← c57275] at e_921_7
  have hr := e_921_7
  simp only [e_921_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7219)) := by
  have c57276 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57277 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57278 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57279 := (publicBatchWrapper89_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57280 := (publicBatchWrapper89_copies1790 a h).1
  have c57281 := (publicBatchWrapper89_copies1790 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_8
  simp only [← c57276, ← c57277, ← c57278] at e_921_8
  have e_921_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_9
  simp only [← c57279, ← c57280, k2, ← c57281] at e_921_9
  have hr := e_921_9
  simp only [e_921_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7220)) := by
  have c57282 := (publicBatchWrapper89_copies1790 a h).2.2.1
  have c57283 := (publicBatchWrapper89_copies1790 a h).2.2.2.1
  have c57284 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.1
  have c57285 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.1
  have c57286 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.1
  have c57287 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_10
  simp only [← c57282, ← c57283, ← c57284] at e_921_10
  have e_921_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_11
  simp only [← c57285, ← c57286, k2, ← c57287] at e_921_11
  have hr := e_921_11
  simp only [e_921_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7221)) := by
  have c57288 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.1
  have c57289 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.1
  have c57290 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57291 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57292 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57293 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_12
  simp only [← c57288, ← c57289, ← c57290] at e_921_12
  have e_921_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_13
  simp only [← c57291, ← c57292, k2, ← c57293] at e_921_13
  have hr := e_921_13
  simp only [e_921_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7222)) := by
  have c57294 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57295 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57296 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57297 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57298 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57299 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_14
  simp only [← c57294, ← c57295, ← c57296] at e_921_14
  have e_921_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_15
  simp only [← c57297, ← c57298, k2, ← c57299] at e_921_15
  have hr := e_921_15
  simp only [e_921_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7223)) := by
  have c57300 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57301 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57302 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57303 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57304 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57305 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_16
  simp only [← c57300, ← c57301, ← c57302] at e_921_16
  have e_921_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_17
  simp only [← c57303, ← c57304, k2, ← c57305] at e_921_17
  have hr := e_921_17
  simp only [e_921_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 921 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7224)) := by
  have c57306 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57307 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57308 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57309 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57310 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57311 := (publicBatchWrapper89_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_921_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_18
  simp only [← c57306, ← c57307, ← c57308] at e_921_18
  have e_921_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_19
  simp only [← c57309, ← c57310, k2, ← c57311] at e_921_19
  have hr := e_921_19
  simp only [e_921_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7225)) := by
  have c57312 := (publicBatchWrapper89_copies1791 a h).1
  have c57313 := (publicBatchWrapper89_copies1791 a h).2.1
  have c57314 := (publicBatchWrapper89_copies1791 a h).2.2.1
  have c57315 := (publicBatchWrapper89_copies1791 a h).2.2.2.1
  have c57316 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.1
  have c57317 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_0
  simp only [← c57312, ← c57313, ← c57314] at e_922_0
  have e_922_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_1
  simp only [← c57315, ← c57316, k2, ← c57317] at e_922_1
  have hr := e_922_1
  simp only [e_922_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7226)) := by
  have c57318 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.1
  have c57319 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.1
  have c57320 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.1
  have c57321 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.1
  have c57322 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57323 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_2
  simp only [← c57318, ← c57319, ← c57320] at e_922_2
  have e_922_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_3
  simp only [← c57321, ← c57322, k2, ← c57323] at e_922_3
  have hr := e_922_3
  simp only [e_922_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7227)) := by
  have c57324 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57325 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57326 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57327 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57328 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57329 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_4
  simp only [← c57324, ← c57325, ← c57326] at e_922_4
  have e_922_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_5
  simp only [← c57327, ← c57328, k2, ← c57329] at e_922_5
  have hr := e_922_5
  simp only [e_922_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7228)) := by
  have c57330 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57331 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57332 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57333 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57334 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57335 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_6
  simp only [← c57330, ← c57331, ← c57332] at e_922_6
  have e_922_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_7
  simp only [← c57333, ← c57334, k2, ← c57335] at e_922_7
  have hr := e_922_7
  simp only [e_922_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7229)) := by
  have c57336 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57337 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57338 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57339 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57340 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57341 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_8
  simp only [← c57336, ← c57337, ← c57338] at e_922_8
  have e_922_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_9
  simp only [← c57339, ← c57340, k2, ← c57341] at e_922_9
  have hr := e_922_9
  simp only [e_922_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7230)) := by
  have c57342 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57343 := (publicBatchWrapper89_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57344 := (publicBatchWrapper89_copies1792 a h).1
  have c57345 := (publicBatchWrapper89_copies1792 a h).2.1
  have c57346 := (publicBatchWrapper89_copies1792 a h).2.2.1
  have c57347 := (publicBatchWrapper89_copies1792 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_10
  simp only [← c57342, ← c57343, ← c57344] at e_922_10
  have e_922_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_11
  simp only [← c57345, ← c57346, k2, ← c57347] at e_922_11
  have hr := e_922_11
  simp only [e_922_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7231)) := by
  have c57348 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.1
  have c57349 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.1
  have c57350 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.1
  have c57351 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.1
  have c57352 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.1
  have c57353 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_12
  simp only [← c57348, ← c57349, ← c57350] at e_922_12
  have e_922_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_13
  simp only [← c57351, ← c57352, k2, ← c57353] at e_922_13
  have hr := e_922_13
  simp only [e_922_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7232)) := by
  have c57354 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57355 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57356 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57357 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57358 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57359 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_14
  simp only [← c57354, ← c57355, ← c57356] at e_922_14
  have e_922_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_15
  simp only [← c57357, ← c57358, k2, ← c57359] at e_922_15
  have hr := e_922_15
  simp only [e_922_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7233)) := by
  have c57360 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57361 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57362 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57363 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57364 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57365 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_16
  simp only [← c57360, ← c57361, ← c57362] at e_922_16
  have e_922_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_17
  simp only [← c57363, ← c57364, k2, ← c57365] at e_922_17
  have hr := e_922_17
  simp only [e_922_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 922 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7234)) := by
  have c57366 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57367 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57368 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57369 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57370 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57371 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_922_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_18
  simp only [← c57366, ← c57367, ← c57368] at e_922_18
  have e_922_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_19
  simp only [← c57369, ← c57370, k2, ← c57371] at e_922_19
  have hr := e_922_19
  simp only [e_922_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7235)) := by
  have c57372 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57373 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57374 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57375 := (publicBatchWrapper89_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57376 := (publicBatchWrapper89_copies1793 a h).1
  have c57377 := (publicBatchWrapper89_copies1793 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_0
  simp only [← c57372, ← c57373, ← c57374] at e_923_0
  have e_923_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_1
  simp only [← c57375, ← c57376, k2, ← c57377] at e_923_1
  have hr := e_923_1
  simp only [e_923_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7236)) := by
  have c57378 := (publicBatchWrapper89_copies1793 a h).2.2.1
  have c57379 := (publicBatchWrapper89_copies1793 a h).2.2.2.1
  have c57380 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.1
  have c57381 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.1
  have c57382 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.1
  have c57383 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_2
  simp only [← c57378, ← c57379, ← c57380] at e_923_2
  have e_923_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_3
  simp only [← c57381, ← c57382, k2, ← c57383] at e_923_3
  have hr := e_923_3
  simp only [e_923_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7237)) := by
  have c57384 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.1
  have c57385 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.1
  have c57386 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57387 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57388 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57389 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_4
  simp only [← c57384, ← c57385, ← c57386] at e_923_4
  have e_923_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_5
  simp only [← c57387, ← c57388, k2, ← c57389] at e_923_5
  have hr := e_923_5
  simp only [e_923_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7238)) := by
  have c57390 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57391 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57392 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57393 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57394 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57395 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_6
  simp only [← c57390, ← c57391, ← c57392] at e_923_6
  have e_923_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_7
  simp only [← c57393, ← c57394, k2, ← c57395] at e_923_7
  have hr := e_923_7
  simp only [e_923_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7239)) := by
  have c57396 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57397 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57398 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57399 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57400 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57401 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_8
  simp only [← c57396, ← c57397, ← c57398] at e_923_8
  have e_923_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_9
  simp only [← c57399, ← c57400, k2, ← c57401] at e_923_9
  have hr := e_923_9
  simp only [e_923_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7240)) := by
  have c57402 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57403 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57404 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57405 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57406 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57407 := (publicBatchWrapper89_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_10
  simp only [← c57402, ← c57403, ← c57404] at e_923_10
  have e_923_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_11
  simp only [← c57405, ← c57406, k2, ← c57407] at e_923_11
  have hr := e_923_11
  simp only [e_923_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7241)) := by
  have c57408 := (publicBatchWrapper89_copies1794 a h).1
  have c57409 := (publicBatchWrapper89_copies1794 a h).2.1
  have c57410 := (publicBatchWrapper89_copies1794 a h).2.2.1
  have c57411 := (publicBatchWrapper89_copies1794 a h).2.2.2.1
  have c57412 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.1
  have c57413 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_12
  simp only [← c57408, ← c57409, ← c57410] at e_923_12
  have e_923_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_13
  simp only [← c57411, ← c57412, k2, ← c57413] at e_923_13
  have hr := e_923_13
  simp only [e_923_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7242)) := by
  have c57414 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.1
  have c57415 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.1
  have c57416 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.1
  have c57417 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.1
  have c57418 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57419 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_14
  simp only [← c57414, ← c57415, ← c57416] at e_923_14
  have e_923_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_15
  simp only [← c57417, ← c57418, k2, ← c57419] at e_923_15
  have hr := e_923_15
  simp only [e_923_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7243)) := by
  have c57420 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57421 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57422 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57423 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57424 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57425 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_16
  simp only [← c57420, ← c57421, ← c57422] at e_923_16
  have e_923_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_17
  simp only [← c57423, ← c57424, k2, ← c57425] at e_923_17
  have hr := e_923_17
  simp only [e_923_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 923 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7244)) := by
  have c57426 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57427 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57428 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57429 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57430 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57431 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_923_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_18
  simp only [← c57426, ← c57427, ← c57428] at e_923_18
  have e_923_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_19
  simp only [← c57429, ← c57430, k2, ← c57431] at e_923_19
  have hr := e_923_19
  simp only [e_923_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7245)) := by
  have c57432 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57433 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57434 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57435 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57436 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57437 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_0
  simp only [← c57432, ← c57433, ← c57434] at e_924_0
  have e_924_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_1
  simp only [← c57435, ← c57436, k2, ← c57437] at e_924_1
  have hr := e_924_1
  simp only [e_924_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7246)) := by
  have c57438 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57439 := (publicBatchWrapper89_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57440 := (publicBatchWrapper89_copies1795 a h).1
  have c57441 := (publicBatchWrapper89_copies1795 a h).2.1
  have c57442 := (publicBatchWrapper89_copies1795 a h).2.2.1
  have c57443 := (publicBatchWrapper89_copies1795 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_2
  simp only [← c57438, ← c57439, ← c57440] at e_924_2
  have e_924_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_3
  simp only [← c57441, ← c57442, k2, ← c57443] at e_924_3
  have hr := e_924_3
  simp only [e_924_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7247)) := by
  have c57444 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.1
  have c57445 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.1
  have c57446 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.1
  have c57447 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.1
  have c57448 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.1
  have c57449 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_4
  simp only [← c57444, ← c57445, ← c57446] at e_924_4
  have e_924_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_5
  simp only [← c57447, ← c57448, k2, ← c57449] at e_924_5
  have hr := e_924_5
  simp only [e_924_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7248)) := by
  have c57450 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57451 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57452 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57453 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57454 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57455 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_6
  simp only [← c57450, ← c57451, ← c57452] at e_924_6
  have e_924_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_7
  simp only [← c57453, ← c57454, k2, ← c57455] at e_924_7
  have hr := e_924_7
  simp only [e_924_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7249)) := by
  have c57456 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57457 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57458 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57459 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57460 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57461 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_8
  simp only [← c57456, ← c57457, ← c57458] at e_924_8
  have e_924_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_9
  simp only [← c57459, ← c57460, k2, ← c57461] at e_924_9
  have hr := e_924_9
  simp only [e_924_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7250)) := by
  have c57462 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57463 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57464 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57465 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57466 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57467 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_10
  simp only [← c57462, ← c57463, ← c57464] at e_924_10
  have e_924_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_11
  simp only [← c57465, ← c57466, k2, ← c57467] at e_924_11
  have hr := e_924_11
  simp only [e_924_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7251)) := by
  have c57468 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57469 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57470 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57471 := (publicBatchWrapper89_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57472 := (publicBatchWrapper89_copies1796 a h).1
  have c57473 := (publicBatchWrapper89_copies1796 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_12
  simp only [← c57468, ← c57469, ← c57470] at e_924_12
  have e_924_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_13
  simp only [← c57471, ← c57472, k2, ← c57473] at e_924_13
  have hr := e_924_13
  simp only [e_924_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7252)) := by
  have c57474 := (publicBatchWrapper89_copies1796 a h).2.2.1
  have c57475 := (publicBatchWrapper89_copies1796 a h).2.2.2.1
  have c57476 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.1
  have c57477 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.1
  have c57478 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.1
  have c57479 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_14
  simp only [← c57474, ← c57475, ← c57476] at e_924_14
  have e_924_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_15
  simp only [← c57477, ← c57478, k2, ← c57479] at e_924_15
  have hr := e_924_15
  simp only [e_924_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7253)) := by
  have c57480 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.1
  have c57481 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.1
  have c57482 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57483 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57484 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57485 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_16
  simp only [← c57480, ← c57481, ← c57482] at e_924_16
  have e_924_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_17
  simp only [← c57483, ← c57484, k2, ← c57485] at e_924_17
  have hr := e_924_17
  simp only [e_924_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 924 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7254)) := by
  have c57486 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57487 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57488 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57489 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57490 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57491 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_924_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_18
  simp only [← c57486, ← c57487, ← c57488] at e_924_18
  have e_924_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_19
  simp only [← c57489, ← c57490, k2, ← c57491] at e_924_19
  have hr := e_924_19
  simp only [e_924_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7255)) := by
  have c57492 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57493 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57494 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57495 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57496 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57497 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_0
  simp only [← c57492, ← c57493, ← c57494] at e_925_0
  have e_925_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_1
  simp only [← c57495, ← c57496, k2, ← c57497] at e_925_1
  have hr := e_925_1
  simp only [e_925_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7256)) := by
  have c57498 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57499 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57500 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57501 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57502 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57503 := (publicBatchWrapper89_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_2
  simp only [← c57498, ← c57499, ← c57500] at e_925_2
  have e_925_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_3
  simp only [← c57501, ← c57502, k2, ← c57503] at e_925_3
  have hr := e_925_3
  simp only [e_925_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7257)) := by
  have c57504 := (publicBatchWrapper89_copies1797 a h).1
  have c57505 := (publicBatchWrapper89_copies1797 a h).2.1
  have c57506 := (publicBatchWrapper89_copies1797 a h).2.2.1
  have c57507 := (publicBatchWrapper89_copies1797 a h).2.2.2.1
  have c57508 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.1
  have c57509 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_4
  simp only [← c57504, ← c57505, ← c57506] at e_925_4
  have e_925_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_5
  simp only [← c57507, ← c57508, k2, ← c57509] at e_925_5
  have hr := e_925_5
  simp only [e_925_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7258)) := by
  have c57510 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.1
  have c57511 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.1
  have c57512 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.1
  have c57513 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.1
  have c57514 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57515 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_6
  simp only [← c57510, ← c57511, ← c57512] at e_925_6
  have e_925_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_7
  simp only [← c57513, ← c57514, k2, ← c57515] at e_925_7
  have hr := e_925_7
  simp only [e_925_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7259)) := by
  have c57516 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57517 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57518 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57519 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57520 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57521 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_8
  simp only [← c57516, ← c57517, ← c57518] at e_925_8
  have e_925_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_9
  simp only [← c57519, ← c57520, k2, ← c57521] at e_925_9
  have hr := e_925_9
  simp only [e_925_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7260)) := by
  have c57522 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57523 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57524 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57525 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57526 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57527 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_10
  simp only [← c57522, ← c57523, ← c57524] at e_925_10
  have e_925_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_11
  simp only [← c57525, ← c57526, k2, ← c57527] at e_925_11
  have hr := e_925_11
  simp only [e_925_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7261)) := by
  have c57528 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57529 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57530 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57531 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57532 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57533 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_12
  simp only [← c57528, ← c57529, ← c57530] at e_925_12
  have e_925_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_13
  simp only [← c57531, ← c57532, k2, ← c57533] at e_925_13
  have hr := e_925_13
  simp only [e_925_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7262)) := by
  have c57534 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57535 := (publicBatchWrapper89_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57536 := (publicBatchWrapper89_copies1798 a h).1
  have c57537 := (publicBatchWrapper89_copies1798 a h).2.1
  have c57538 := (publicBatchWrapper89_copies1798 a h).2.2.1
  have c57539 := (publicBatchWrapper89_copies1798 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_14
  simp only [← c57534, ← c57535, ← c57536] at e_925_14
  have e_925_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_15
  simp only [← c57537, ← c57538, k2, ← c57539] at e_925_15
  have hr := e_925_15
  simp only [e_925_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7263)) := by
  have c57540 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.1
  have c57541 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.1
  have c57542 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.1
  have c57543 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.1
  have c57544 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.1
  have c57545 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_16
  simp only [← c57540, ← c57541, ← c57542] at e_925_16
  have e_925_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_17
  simp only [← c57543, ← c57544, k2, ← c57545] at e_925_17
  have hr := e_925_17
  simp only [e_925_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 925 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7264)) := by
  have c57546 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57547 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57548 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57549 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57550 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57551 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_925_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_18
  simp only [← c57546, ← c57547, ← c57548] at e_925_18
  have e_925_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_19
  simp only [← c57549, ← c57550, k2, ← c57551] at e_925_19
  have hr := e_925_19
  simp only [e_925_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7265)) := by
  have c57552 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57553 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57554 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57555 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57556 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57557 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_0
  simp only [← c57552, ← c57553, ← c57554] at e_926_0
  have e_926_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_1
  simp only [← c57555, ← c57556, k2, ← c57557] at e_926_1
  have hr := e_926_1
  simp only [e_926_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7266)) := by
  have c57558 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57559 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57560 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57561 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57562 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57563 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_2
  simp only [← c57558, ← c57559, ← c57560] at e_926_2
  have e_926_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_3
  simp only [← c57561, ← c57562, k2, ← c57563] at e_926_3
  have hr := e_926_3
  simp only [e_926_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7267)) := by
  have c57564 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57565 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57566 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57567 := (publicBatchWrapper89_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57568 := (publicBatchWrapper89_copies1799 a h).1
  have c57569 := (publicBatchWrapper89_copies1799 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_4
  simp only [← c57564, ← c57565, ← c57566] at e_926_4
  have e_926_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_5
  simp only [← c57567, ← c57568, k2, ← c57569] at e_926_5
  have hr := e_926_5
  simp only [e_926_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7268)) := by
  have c57570 := (publicBatchWrapper89_copies1799 a h).2.2.1
  have c57571 := (publicBatchWrapper89_copies1799 a h).2.2.2.1
  have c57572 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.1
  have c57573 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.1
  have c57574 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.1
  have c57575 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_6
  simp only [← c57570, ← c57571, ← c57572] at e_926_6
  have e_926_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_7
  simp only [← c57573, ← c57574, k2, ← c57575] at e_926_7
  have hr := e_926_7
  simp only [e_926_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7269)) := by
  have c57576 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.1
  have c57577 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.1
  have c57578 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57579 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57580 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57581 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_8
  simp only [← c57576, ← c57577, ← c57578] at e_926_8
  have e_926_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_9
  simp only [← c57579, ← c57580, k2, ← c57581] at e_926_9
  have hr := e_926_9
  simp only [e_926_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7270)) := by
  have c57582 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57583 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57584 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57585 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57586 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57587 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_10
  simp only [← c57582, ← c57583, ← c57584] at e_926_10
  have e_926_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_11
  simp only [← c57585, ← c57586, k2, ← c57587] at e_926_11
  have hr := e_926_11
  simp only [e_926_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7271)) := by
  have c57588 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57589 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57590 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57591 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57592 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57593 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_12
  simp only [← c57588, ← c57589, ← c57590] at e_926_12
  have e_926_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_13
  simp only [← c57591, ← c57592, k2, ← c57593] at e_926_13
  have hr := e_926_13
  simp only [e_926_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7272)) := by
  have c57594 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57595 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57596 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57597 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57598 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57599 := (publicBatchWrapper89_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_14
  simp only [← c57594, ← c57595, ← c57596] at e_926_14
  have e_926_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_15
  simp only [← c57597, ← c57598, k2, ← c57599] at e_926_15
  have hr := e_926_15
  simp only [e_926_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7273)) := by
  have c57600 := (publicBatchWrapper89_copies1800 a h).1
  have c57601 := (publicBatchWrapper89_copies1800 a h).2.1
  have c57602 := (publicBatchWrapper89_copies1800 a h).2.2.1
  have c57603 := (publicBatchWrapper89_copies1800 a h).2.2.2.1
  have c57604 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.1
  have c57605 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_16
  simp only [← c57600, ← c57601, ← c57602] at e_926_16
  have e_926_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_17
  simp only [← c57603, ← c57604, k2, ← c57605] at e_926_17
  have hr := e_926_17
  simp only [e_926_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 926 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7274)) := by
  have c57606 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.1
  have c57607 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.1
  have c57608 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.1
  have c57609 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.1
  have c57610 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57611 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_926_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_18
  simp only [← c57606, ← c57607, ← c57608] at e_926_18
  have e_926_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_19
  simp only [← c57609, ← c57610, k2, ← c57611] at e_926_19
  have hr := e_926_19
  simp only [e_926_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7275)) := by
  have c57612 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57613 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57614 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57615 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57616 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57617 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_0
  simp only [← c57612, ← c57613, ← c57614] at e_927_0
  have e_927_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_1
  simp only [← c57615, ← c57616, k2, ← c57617] at e_927_1
  have hr := e_927_1
  simp only [e_927_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7276)) := by
  have c57618 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57619 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57620 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57621 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57622 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57623 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_2
  simp only [← c57618, ← c57619, ← c57620] at e_927_2
  have e_927_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_3
  simp only [← c57621, ← c57622, k2, ← c57623] at e_927_3
  have hr := e_927_3
  simp only [e_927_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7277)) := by
  have c57624 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57625 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57626 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57627 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57628 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57629 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_4
  simp only [← c57624, ← c57625, ← c57626] at e_927_4
  have e_927_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_5
  simp only [← c57627, ← c57628, k2, ← c57629] at e_927_5
  have hr := e_927_5
  simp only [e_927_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7278)) := by
  have c57630 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57631 := (publicBatchWrapper89_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57632 := (publicBatchWrapper89_copies1801 a h).1
  have c57633 := (publicBatchWrapper89_copies1801 a h).2.1
  have c57634 := (publicBatchWrapper89_copies1801 a h).2.2.1
  have c57635 := (publicBatchWrapper89_copies1801 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_6
  simp only [← c57630, ← c57631, ← c57632] at e_927_6
  have e_927_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_7
  simp only [← c57633, ← c57634, k2, ← c57635] at e_927_7
  have hr := e_927_7
  simp only [e_927_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7279)) := by
  have c57636 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.1
  have c57637 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.1
  have c57638 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.1
  have c57639 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.1
  have c57640 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.1
  have c57641 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_8
  simp only [← c57636, ← c57637, ← c57638] at e_927_8
  have e_927_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_9
  simp only [← c57639, ← c57640, k2, ← c57641] at e_927_9
  have hr := e_927_9
  simp only [e_927_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7280)) := by
  have c57642 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57643 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57644 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57645 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57646 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57647 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_10
  simp only [← c57642, ← c57643, ← c57644] at e_927_10
  have e_927_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_11
  simp only [← c57645, ← c57646, k2, ← c57647] at e_927_11
  have hr := e_927_11
  simp only [e_927_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7281)) := by
  have c57648 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57649 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57650 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57651 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57652 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57653 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_12
  simp only [← c57648, ← c57649, ← c57650] at e_927_12
  have e_927_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_13
  simp only [← c57651, ← c57652, k2, ← c57653] at e_927_13
  have hr := e_927_13
  simp only [e_927_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7282)) := by
  have c57654 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57655 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57656 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57657 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57658 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57659 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_14
  simp only [← c57654, ← c57655, ← c57656] at e_927_14
  have e_927_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_15
  simp only [← c57657, ← c57658, k2, ← c57659] at e_927_15
  have hr := e_927_15
  simp only [e_927_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7283)) := by
  have c57660 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57661 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57662 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57663 := (publicBatchWrapper89_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57664 := (publicBatchWrapper89_copies1802 a h).1
  have c57665 := (publicBatchWrapper89_copies1802 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_16
  simp only [← c57660, ← c57661, ← c57662] at e_927_16
  have e_927_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_17
  simp only [← c57663, ← c57664, k2, ← c57665] at e_927_17
  have hr := e_927_17
  simp only [e_927_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 927 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7284)) := by
  have c57666 := (publicBatchWrapper89_copies1802 a h).2.2.1
  have c57667 := (publicBatchWrapper89_copies1802 a h).2.2.2.1
  have c57668 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.1
  have c57669 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.1
  have c57670 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.1
  have c57671 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_927_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_18
  simp only [← c57666, ← c57667, ← c57668] at e_927_18
  have e_927_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_19
  simp only [← c57669, ← c57670, k2, ← c57671] at e_927_19
  have hr := e_927_19
  simp only [e_927_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7285)) := by
  have c57672 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.1
  have c57673 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.1
  have c57674 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57675 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57676 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57677 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_0
  simp only [← c57672, ← c57673, ← c57674] at e_928_0
  have e_928_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_1
  simp only [← c57675, ← c57676, k2, ← c57677] at e_928_1
  have hr := e_928_1
  simp only [e_928_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7286)) := by
  have c57678 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57679 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57680 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57681 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57682 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57683 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_2
  simp only [← c57678, ← c57679, ← c57680] at e_928_2
  have e_928_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_3
  simp only [← c57681, ← c57682, k2, ← c57683] at e_928_3
  have hr := e_928_3
  simp only [e_928_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7287)) := by
  have c57684 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57685 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57686 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57687 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57688 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57689 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_4
  simp only [← c57684, ← c57685, ← c57686] at e_928_4
  have e_928_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_5
  simp only [← c57687, ← c57688, k2, ← c57689] at e_928_5
  have hr := e_928_5
  simp only [e_928_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7288)) := by
  have c57690 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57691 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57692 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57693 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57694 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57695 := (publicBatchWrapper89_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_6
  simp only [← c57690, ← c57691, ← c57692] at e_928_6
  have e_928_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_7
  simp only [← c57693, ← c57694, k2, ← c57695] at e_928_7
  have hr := e_928_7
  simp only [e_928_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7289)) := by
  have c57696 := (publicBatchWrapper89_copies1803 a h).1
  have c57697 := (publicBatchWrapper89_copies1803 a h).2.1
  have c57698 := (publicBatchWrapper89_copies1803 a h).2.2.1
  have c57699 := (publicBatchWrapper89_copies1803 a h).2.2.2.1
  have c57700 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.1
  have c57701 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_8
  simp only [← c57696, ← c57697, ← c57698] at e_928_8
  have e_928_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_9
  simp only [← c57699, ← c57700, k2, ← c57701] at e_928_9
  have hr := e_928_9
  simp only [e_928_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7290)) := by
  have c57702 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.1
  have c57703 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.1
  have c57704 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.1
  have c57705 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.1
  have c57706 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57707 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_10
  simp only [← c57702, ← c57703, ← c57704] at e_928_10
  have e_928_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_11
  simp only [← c57705, ← c57706, k2, ← c57707] at e_928_11
  have hr := e_928_11
  simp only [e_928_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7291)) := by
  have c57708 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57709 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57710 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57711 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57712 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57713 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_12
  simp only [← c57708, ← c57709, ← c57710] at e_928_12
  have e_928_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_13
  simp only [← c57711, ← c57712, k2, ← c57713] at e_928_13
  have hr := e_928_13
  simp only [e_928_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7292)) := by
  have c57714 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57715 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57716 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57717 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57718 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57719 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_14
  simp only [← c57714, ← c57715, ← c57716] at e_928_14
  have e_928_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_15
  simp only [← c57717, ← c57718, k2, ← c57719] at e_928_15
  have hr := e_928_15
  simp only [e_928_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7293)) := by
  have c57720 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57721 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57722 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57723 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57724 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57725 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_16
  simp only [← c57720, ← c57721, ← c57722] at e_928_16
  have e_928_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_17
  simp only [← c57723, ← c57724, k2, ← c57725] at e_928_17
  have hr := e_928_17
  simp only [e_928_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 928 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7294)) := by
  have c57726 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57727 := (publicBatchWrapper89_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57728 := (publicBatchWrapper89_copies1804 a h).1
  have c57729 := (publicBatchWrapper89_copies1804 a h).2.1
  have c57730 := (publicBatchWrapper89_copies1804 a h).2.2.1
  have c57731 := (publicBatchWrapper89_copies1804 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_928_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_18
  simp only [← c57726, ← c57727, ← c57728] at e_928_18
  have e_928_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_19
  simp only [← c57729, ← c57730, k2, ← c57731] at e_928_19
  have hr := e_928_19
  simp only [e_928_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7295)) := by
  have c57732 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.1
  have c57733 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.1
  have c57734 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.1
  have c57735 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.1
  have c57736 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.1
  have c57737 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_0
  simp only [← c57732, ← c57733, ← c57734] at e_929_0
  have e_929_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_1
  simp only [← c57735, ← c57736, k2, ← c57737] at e_929_1
  have hr := e_929_1
  simp only [e_929_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7296)) := by
  have c57738 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57739 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57740 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57741 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57742 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57743 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_2
  simp only [← c57738, ← c57739, ← c57740] at e_929_2
  have e_929_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_3
  simp only [← c57741, ← c57742, k2, ← c57743] at e_929_3
  have hr := e_929_3
  simp only [e_929_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7297)) := by
  have c57744 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57745 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57746 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57747 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57748 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57749 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_4
  simp only [← c57744, ← c57745, ← c57746] at e_929_4
  have e_929_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_5
  simp only [← c57747, ← c57748, k2, ← c57749] at e_929_5
  have hr := e_929_5
  simp only [e_929_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7298)) := by
  have c57750 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57751 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57752 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57753 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57754 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57755 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_6
  simp only [← c57750, ← c57751, ← c57752] at e_929_6
  have e_929_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_7
  simp only [← c57753, ← c57754, k2, ← c57755] at e_929_7
  have hr := e_929_7
  simp only [e_929_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7299)) := by
  have c57756 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57757 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57758 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57759 := (publicBatchWrapper89_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57760 := (publicBatchWrapper89_copies1805 a h).1
  have c57761 := (publicBatchWrapper89_copies1805 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_8
  simp only [← c57756, ← c57757, ← c57758] at e_929_8
  have e_929_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_9
  simp only [← c57759, ← c57760, k2, ← c57761] at e_929_9
  have hr := e_929_9
  simp only [e_929_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7300)) := by
  have c57762 := (publicBatchWrapper89_copies1805 a h).2.2.1
  have c57763 := (publicBatchWrapper89_copies1805 a h).2.2.2.1
  have c57764 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.1
  have c57765 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.1
  have c57766 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.1
  have c57767 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_10
  simp only [← c57762, ← c57763, ← c57764] at e_929_10
  have e_929_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_11
  simp only [← c57765, ← c57766, k2, ← c57767] at e_929_11
  have hr := e_929_11
  simp only [e_929_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7301)) := by
  have c57768 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.1
  have c57769 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.1
  have c57770 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57771 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57772 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57773 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_12
  simp only [← c57768, ← c57769, ← c57770] at e_929_12
  have e_929_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_13
  simp only [← c57771, ← c57772, k2, ← c57773] at e_929_13
  have hr := e_929_13
  simp only [e_929_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7302)) := by
  have c57774 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57775 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57776 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57777 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57778 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57779 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_14
  simp only [← c57774, ← c57775, ← c57776] at e_929_14
  have e_929_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_15
  simp only [← c57777, ← c57778, k2, ← c57779] at e_929_15
  have hr := e_929_15
  simp only [e_929_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7303)) := by
  have c57780 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57781 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57782 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57783 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57784 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57785 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_16
  simp only [← c57780, ← c57781, ← c57782] at e_929_16
  have e_929_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_17
  simp only [← c57783, ← c57784, k2, ← c57785] at e_929_17
  have hr := e_929_17
  simp only [e_929_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 929 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7304)) := by
  have c57786 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57787 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57788 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57789 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57790 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57791 := (publicBatchWrapper89_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_929_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_18
  simp only [← c57786, ← c57787, ← c57788] at e_929_18
  have e_929_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_19
  simp only [← c57789, ← c57790, k2, ← c57791] at e_929_19
  have hr := e_929_19
  simp only [e_929_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7305)) := by
  have c57792 := (publicBatchWrapper89_copies1806 a h).1
  have c57793 := (publicBatchWrapper89_copies1806 a h).2.1
  have c57794 := (publicBatchWrapper89_copies1806 a h).2.2.1
  have c57795 := (publicBatchWrapper89_copies1806 a h).2.2.2.1
  have c57796 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.1
  have c57797 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_0
  simp only [← c57792, ← c57793, ← c57794] at e_930_0
  have e_930_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_1
  simp only [← c57795, ← c57796, k2, ← c57797] at e_930_1
  have hr := e_930_1
  simp only [e_930_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7306)) := by
  have c57798 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.1
  have c57799 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.1
  have c57800 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.1
  have c57801 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.1
  have c57802 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57803 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_2
  simp only [← c57798, ← c57799, ← c57800] at e_930_2
  have e_930_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_3
  simp only [← c57801, ← c57802, k2, ← c57803] at e_930_3
  have hr := e_930_3
  simp only [e_930_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7307)) := by
  have c57804 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57805 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57806 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57807 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57808 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57809 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_4
  simp only [← c57804, ← c57805, ← c57806] at e_930_4
  have e_930_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_5
  simp only [← c57807, ← c57808, k2, ← c57809] at e_930_5
  have hr := e_930_5
  simp only [e_930_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7308)) := by
  have c57810 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57811 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57812 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57813 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57814 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57815 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_6
  simp only [← c57810, ← c57811, ← c57812] at e_930_6
  have e_930_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_7
  simp only [← c57813, ← c57814, k2, ← c57815] at e_930_7
  have hr := e_930_7
  simp only [e_930_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7309)) := by
  have c57816 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57817 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57818 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57819 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57820 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57821 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_8
  simp only [← c57816, ← c57817, ← c57818] at e_930_8
  have e_930_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_9
  simp only [← c57819, ← c57820, k2, ← c57821] at e_930_9
  have hr := e_930_9
  simp only [e_930_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7310)) := by
  have c57822 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57823 := (publicBatchWrapper89_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57824 := (publicBatchWrapper89_copies1807 a h).1
  have c57825 := (publicBatchWrapper89_copies1807 a h).2.1
  have c57826 := (publicBatchWrapper89_copies1807 a h).2.2.1
  have c57827 := (publicBatchWrapper89_copies1807 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_10
  simp only [← c57822, ← c57823, ← c57824] at e_930_10
  have e_930_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_11
  simp only [← c57825, ← c57826, k2, ← c57827] at e_930_11
  have hr := e_930_11
  simp only [e_930_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7311)) := by
  have c57828 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.1
  have c57829 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.1
  have c57830 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.1
  have c57831 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.1
  have c57832 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.1
  have c57833 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_12
  simp only [← c57828, ← c57829, ← c57830] at e_930_12
  have e_930_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_13
  simp only [← c57831, ← c57832, k2, ← c57833] at e_930_13
  have hr := e_930_13
  simp only [e_930_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7312)) := by
  have c57834 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57835 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57836 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57837 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57838 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57839 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_14
  simp only [← c57834, ← c57835, ← c57836] at e_930_14
  have e_930_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_15
  simp only [← c57837, ← c57838, k2, ← c57839] at e_930_15
  have hr := e_930_15
  simp only [e_930_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7313)) := by
  have c57840 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57841 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57842 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57843 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57844 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57845 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_16
  simp only [← c57840, ← c57841, ← c57842] at e_930_16
  have e_930_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_17
  simp only [← c57843, ← c57844, k2, ← c57845] at e_930_17
  have hr := e_930_17
  simp only [e_930_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 930 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7314)) := by
  have c57846 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57847 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57848 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57849 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57850 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57851 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_930_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_18
  simp only [← c57846, ← c57847, ← c57848] at e_930_18
  have e_930_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_19
  simp only [← c57849, ← c57850, k2, ← c57851] at e_930_19
  have hr := e_930_19
  simp only [e_930_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7315)) := by
  have c57852 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57853 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57854 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57855 := (publicBatchWrapper89_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57856 := (publicBatchWrapper89_copies1808 a h).1
  have c57857 := (publicBatchWrapper89_copies1808 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_0
  simp only [← c57852, ← c57853, ← c57854] at e_931_0
  have e_931_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_1
  simp only [← c57855, ← c57856, k2, ← c57857] at e_931_1
  have hr := e_931_1
  simp only [e_931_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7316)) := by
  have c57858 := (publicBatchWrapper89_copies1808 a h).2.2.1
  have c57859 := (publicBatchWrapper89_copies1808 a h).2.2.2.1
  have c57860 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.1
  have c57861 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.1
  have c57862 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.1
  have c57863 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_2
  simp only [← c57858, ← c57859, ← c57860] at e_931_2
  have e_931_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_3
  simp only [← c57861, ← c57862, k2, ← c57863] at e_931_3
  have hr := e_931_3
  simp only [e_931_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7317)) := by
  have c57864 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.1
  have c57865 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.1
  have c57866 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57867 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57868 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57869 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_4
  simp only [← c57864, ← c57865, ← c57866] at e_931_4
  have e_931_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_5
  simp only [← c57867, ← c57868, k2, ← c57869] at e_931_5
  have hr := e_931_5
  simp only [e_931_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7318)) := by
  have c57870 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57871 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57872 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57873 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57874 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57875 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_6
  simp only [← c57870, ← c57871, ← c57872] at e_931_6
  have e_931_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_7
  simp only [← c57873, ← c57874, k2, ← c57875] at e_931_7
  have hr := e_931_7
  simp only [e_931_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7319)) := by
  have c57876 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57877 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57878 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57879 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57880 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57881 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_8
  simp only [← c57876, ← c57877, ← c57878] at e_931_8
  have e_931_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_9
  simp only [← c57879, ← c57880, k2, ← c57881] at e_931_9
  have hr := e_931_9
  simp only [e_931_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7320)) := by
  have c57882 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57883 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57884 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57885 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57886 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57887 := (publicBatchWrapper89_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_10
  simp only [← c57882, ← c57883, ← c57884] at e_931_10
  have e_931_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_11
  simp only [← c57885, ← c57886, k2, ← c57887] at e_931_11
  have hr := e_931_11
  simp only [e_931_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7321)) := by
  have c57888 := (publicBatchWrapper89_copies1809 a h).1
  have c57889 := (publicBatchWrapper89_copies1809 a h).2.1
  have c57890 := (publicBatchWrapper89_copies1809 a h).2.2.1
  have c57891 := (publicBatchWrapper89_copies1809 a h).2.2.2.1
  have c57892 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.1
  have c57893 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_12
  simp only [← c57888, ← c57889, ← c57890] at e_931_12
  have e_931_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_13
  simp only [← c57891, ← c57892, k2, ← c57893] at e_931_13
  have hr := e_931_13
  simp only [e_931_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7322)) := by
  have c57894 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.1
  have c57895 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.1
  have c57896 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.1
  have c57897 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.1
  have c57898 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57899 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_14
  simp only [← c57894, ← c57895, ← c57896] at e_931_14
  have e_931_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_15
  simp only [← c57897, ← c57898, k2, ← c57899] at e_931_15
  have hr := e_931_15
  simp only [e_931_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7323)) := by
  have c57900 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57901 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57902 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57903 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57904 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57905 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_16
  simp only [← c57900, ← c57901, ← c57902] at e_931_16
  have e_931_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_17
  simp only [← c57903, ← c57904, k2, ← c57905] at e_931_17
  have hr := e_931_17
  simp only [e_931_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 931 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7324)) := by
  have c57906 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57907 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57908 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57909 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57910 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57911 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_931_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_18
  simp only [← c57906, ← c57907, ← c57908] at e_931_18
  have e_931_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_19
  simp only [← c57909, ← c57910, k2, ← c57911] at e_931_19
  have hr := e_931_19
  simp only [e_931_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7325)) := by
  have c57912 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57913 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57914 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57915 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57916 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57917 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_0
  simp only [← c57912, ← c57913, ← c57914] at e_932_0
  have e_932_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_1
  simp only [← c57915, ← c57916, k2, ← c57917] at e_932_1
  have hr := e_932_1
  simp only [e_932_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7326)) := by
  have c57918 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57919 := (publicBatchWrapper89_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57920 := (publicBatchWrapper89_copies1810 a h).1
  have c57921 := (publicBatchWrapper89_copies1810 a h).2.1
  have c57922 := (publicBatchWrapper89_copies1810 a h).2.2.1
  have c57923 := (publicBatchWrapper89_copies1810 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_2
  simp only [← c57918, ← c57919, ← c57920] at e_932_2
  have e_932_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_3
  simp only [← c57921, ← c57922, k2, ← c57923] at e_932_3
  have hr := e_932_3
  simp only [e_932_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7327)) := by
  have c57924 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.1
  have c57925 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.1
  have c57926 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.1
  have c57927 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.1
  have c57928 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.1
  have c57929 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_4
  simp only [← c57924, ← c57925, ← c57926] at e_932_4
  have e_932_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_5
  simp only [← c57927, ← c57928, k2, ← c57929] at e_932_5
  have hr := e_932_5
  simp only [e_932_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7328)) := by
  have c57930 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57931 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57932 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57933 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57934 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57935 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_6
  simp only [← c57930, ← c57931, ← c57932] at e_932_6
  have e_932_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_7
  simp only [← c57933, ← c57934, k2, ← c57935] at e_932_7
  have hr := e_932_7
  simp only [e_932_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7329)) := by
  have c57936 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57937 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57938 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57939 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57940 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57941 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_8
  simp only [← c57936, ← c57937, ← c57938] at e_932_8
  have e_932_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_9
  simp only [← c57939, ← c57940, k2, ← c57941] at e_932_9
  have hr := e_932_9
  simp only [e_932_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7330)) := by
  have c57942 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57943 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57944 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57945 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57946 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57947 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_10
  simp only [← c57942, ← c57943, ← c57944] at e_932_10
  have e_932_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_11
  simp only [← c57945, ← c57946, k2, ← c57947] at e_932_11
  have hr := e_932_11
  simp only [e_932_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7331)) := by
  have c57948 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57949 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57950 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57951 := (publicBatchWrapper89_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57952 := (publicBatchWrapper89_copies1811 a h).1
  have c57953 := (publicBatchWrapper89_copies1811 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_12
  simp only [← c57948, ← c57949, ← c57950] at e_932_12
  have e_932_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_13
  simp only [← c57951, ← c57952, k2, ← c57953] at e_932_13
  have hr := e_932_13
  simp only [e_932_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7332)) := by
  have c57954 := (publicBatchWrapper89_copies1811 a h).2.2.1
  have c57955 := (publicBatchWrapper89_copies1811 a h).2.2.2.1
  have c57956 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.1
  have c57957 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.1
  have c57958 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.1
  have c57959 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_14
  simp only [← c57954, ← c57955, ← c57956] at e_932_14
  have e_932_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_15
  simp only [← c57957, ← c57958, k2, ← c57959] at e_932_15
  have hr := e_932_15
  simp only [e_932_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7333)) := by
  have c57960 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.1
  have c57961 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.1
  have c57962 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57963 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57964 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57965 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_16
  simp only [← c57960, ← c57961, ← c57962] at e_932_16
  have e_932_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_17
  simp only [← c57963, ← c57964, k2, ← c57965] at e_932_17
  have hr := e_932_17
  simp only [e_932_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 932 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7334)) := by
  have c57966 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57967 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57968 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57969 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57970 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57971 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_932_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_18
  simp only [← c57966, ← c57967, ← c57968] at e_932_18
  have e_932_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_19
  simp only [← c57969, ← c57970, k2, ← c57971] at e_932_19
  have hr := e_932_19
  simp only [e_932_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7335)) := by
  have c57972 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57973 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57974 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57975 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57976 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57977 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_0
  simp only [← c57972, ← c57973, ← c57974] at e_933_0
  have e_933_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_1
  simp only [← c57975, ← c57976, k2, ← c57977] at e_933_1
  have hr := e_933_1
  simp only [e_933_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7336)) := by
  have c57978 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57979 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57980 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57981 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57982 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57983 := (publicBatchWrapper89_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_2
  simp only [← c57978, ← c57979, ← c57980] at e_933_2
  have e_933_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_3
  simp only [← c57981, ← c57982, k2, ← c57983] at e_933_3
  have hr := e_933_3
  simp only [e_933_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7337)) := by
  have c57984 := (publicBatchWrapper89_copies1812 a h).1
  have c57985 := (publicBatchWrapper89_copies1812 a h).2.1
  have c57986 := (publicBatchWrapper89_copies1812 a h).2.2.1
  have c57987 := (publicBatchWrapper89_copies1812 a h).2.2.2.1
  have c57988 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.1
  have c57989 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_4
  simp only [← c57984, ← c57985, ← c57986] at e_933_4
  have e_933_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_5
  simp only [← c57987, ← c57988, k2, ← c57989] at e_933_5
  have hr := e_933_5
  simp only [e_933_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7338)) := by
  have c57990 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.1
  have c57991 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.1
  have c57992 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.1
  have c57993 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.1
  have c57994 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57995 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_6
  simp only [← c57990, ← c57991, ← c57992] at e_933_6
  have e_933_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_7
  simp only [← c57993, ← c57994, k2, ← c57995] at e_933_7
  have hr := e_933_7
  simp only [e_933_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7339)) := by
  have c57996 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57997 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57998 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57999 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58000 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58001 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_8
  simp only [← c57996, ← c57997, ← c57998] at e_933_8
  have e_933_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_9
  simp only [← c57999, ← c58000, k2, ← c58001] at e_933_9
  have hr := e_933_9
  simp only [e_933_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7340)) := by
  have c58002 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58003 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58004 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58005 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58006 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58007 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_10
  simp only [← c58002, ← c58003, ← c58004] at e_933_10
  have e_933_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_11
  simp only [← c58005, ← c58006, k2, ← c58007] at e_933_11
  have hr := e_933_11
  simp only [e_933_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7341)) := by
  have c58008 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58009 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58010 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58011 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58012 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58013 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_12
  simp only [← c58008, ← c58009, ← c58010] at e_933_12
  have e_933_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_13
  simp only [← c58011, ← c58012, k2, ← c58013] at e_933_13
  have hr := e_933_13
  simp only [e_933_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7342)) := by
  have c58014 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58015 := (publicBatchWrapper89_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58016 := (publicBatchWrapper89_copies1813 a h).1
  have c58017 := (publicBatchWrapper89_copies1813 a h).2.1
  have c58018 := (publicBatchWrapper89_copies1813 a h).2.2.1
  have c58019 := (publicBatchWrapper89_copies1813 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_14
  simp only [← c58014, ← c58015, ← c58016] at e_933_14
  have e_933_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_15
  simp only [← c58017, ← c58018, k2, ← c58019] at e_933_15
  have hr := e_933_15
  simp only [e_933_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7343)) := by
  have c58020 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.1
  have c58021 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.1
  have c58022 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.1
  have c58023 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.1
  have c58024 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.1
  have c58025 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_16
  simp only [← c58020, ← c58021, ← c58022] at e_933_16
  have e_933_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_17
  simp only [← c58023, ← c58024, k2, ← c58025] at e_933_17
  have hr := e_933_17
  simp only [e_933_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 933 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7344)) := by
  have c58026 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58027 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58028 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58029 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58030 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58031 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_933_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_18
  simp only [← c58026, ← c58027, ← c58028] at e_933_18
  have e_933_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_19
  simp only [← c58029, ← c58030, k2, ← c58031] at e_933_19
  have hr := e_933_19
  simp only [e_933_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7345)) := by
  have c58032 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58033 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58034 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58035 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58036 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58037 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_0
  simp only [← c58032, ← c58033, ← c58034] at e_934_0
  have e_934_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_1
  simp only [← c58035, ← c58036, k2, ← c58037] at e_934_1
  have hr := e_934_1
  simp only [e_934_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7346)) := by
  have c58038 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58039 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58040 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58041 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58042 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58043 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_2
  simp only [← c58038, ← c58039, ← c58040] at e_934_2
  have e_934_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_3
  simp only [← c58041, ← c58042, k2, ← c58043] at e_934_3
  have hr := e_934_3
  simp only [e_934_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7347)) := by
  have c58044 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58045 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58046 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58047 := (publicBatchWrapper89_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58048 := (publicBatchWrapper89_copies1814 a h).1
  have c58049 := (publicBatchWrapper89_copies1814 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_4
  simp only [← c58044, ← c58045, ← c58046] at e_934_4
  have e_934_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_5
  simp only [← c58047, ← c58048, k2, ← c58049] at e_934_5
  have hr := e_934_5
  simp only [e_934_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7348)) := by
  have c58050 := (publicBatchWrapper89_copies1814 a h).2.2.1
  have c58051 := (publicBatchWrapper89_copies1814 a h).2.2.2.1
  have c58052 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.1
  have c58053 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.1
  have c58054 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.1
  have c58055 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_6
  simp only [← c58050, ← c58051, ← c58052] at e_934_6
  have e_934_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_7
  simp only [← c58053, ← c58054, k2, ← c58055] at e_934_7
  have hr := e_934_7
  simp only [e_934_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7349)) := by
  have c58056 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.1
  have c58057 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.1
  have c58058 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58059 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58060 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58061 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_8
  simp only [← c58056, ← c58057, ← c58058] at e_934_8
  have e_934_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_9
  simp only [← c58059, ← c58060, k2, ← c58061] at e_934_9
  have hr := e_934_9
  simp only [e_934_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7350)) := by
  have c58062 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58063 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58064 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58065 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58066 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58067 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_10
  simp only [← c58062, ← c58063, ← c58064] at e_934_10
  have e_934_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_11
  simp only [← c58065, ← c58066, k2, ← c58067] at e_934_11
  have hr := e_934_11
  simp only [e_934_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7351)) := by
  have c58068 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58069 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58070 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58071 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58072 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58073 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_12
  simp only [← c58068, ← c58069, ← c58070] at e_934_12
  have e_934_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_13
  simp only [← c58071, ← c58072, k2, ← c58073] at e_934_13
  have hr := e_934_13
  simp only [e_934_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7352)) := by
  have c58074 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58075 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58076 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58077 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58078 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58079 := (publicBatchWrapper89_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_14
  simp only [← c58074, ← c58075, ← c58076] at e_934_14
  have e_934_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_15
  simp only [← c58077, ← c58078, k2, ← c58079] at e_934_15
  have hr := e_934_15
  simp only [e_934_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7353)) := by
  have c58080 := (publicBatchWrapper89_copies1815 a h).1
  have c58081 := (publicBatchWrapper89_copies1815 a h).2.1
  have c58082 := (publicBatchWrapper89_copies1815 a h).2.2.1
  have c58083 := (publicBatchWrapper89_copies1815 a h).2.2.2.1
  have c58084 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.1
  have c58085 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_16
  simp only [← c58080, ← c58081, ← c58082] at e_934_16
  have e_934_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_17
  simp only [← c58083, ← c58084, k2, ← c58085] at e_934_17
  have hr := e_934_17
  simp only [e_934_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 934 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7354)) := by
  have c58086 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.1
  have c58087 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.1
  have c58088 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.1
  have c58089 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.1
  have c58090 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58091 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_934_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_18
  simp only [← c58086, ← c58087, ← c58088] at e_934_18
  have e_934_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_19
  simp only [← c58089, ← c58090, k2, ← c58091] at e_934_19
  have hr := e_934_19
  simp only [e_934_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7355)) := by
  have c58092 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58093 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58094 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58095 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58096 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58097 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_0
  simp only [← c58092, ← c58093, ← c58094] at e_935_0
  have e_935_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_1
  simp only [← c58095, ← c58096, k2, ← c58097] at e_935_1
  have hr := e_935_1
  simp only [e_935_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7356)) := by
  have c58098 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58099 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58100 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58101 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58102 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58103 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_2
  simp only [← c58098, ← c58099, ← c58100] at e_935_2
  have e_935_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_3
  simp only [← c58101, ← c58102, k2, ← c58103] at e_935_3
  have hr := e_935_3
  simp only [e_935_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7357)) := by
  have c58104 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58105 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58106 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58107 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58108 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58109 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_4
  simp only [← c58104, ← c58105, ← c58106] at e_935_4
  have e_935_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_5
  simp only [← c58107, ← c58108, k2, ← c58109] at e_935_5
  have hr := e_935_5
  simp only [e_935_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7358)) := by
  have c58110 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58111 := (publicBatchWrapper89_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58112 := (publicBatchWrapper89_copies1816 a h).1
  have c58113 := (publicBatchWrapper89_copies1816 a h).2.1
  have c58114 := (publicBatchWrapper89_copies1816 a h).2.2.1
  have c58115 := (publicBatchWrapper89_copies1816 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_6
  simp only [← c58110, ← c58111, ← c58112] at e_935_6
  have e_935_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_7
  simp only [← c58113, ← c58114, k2, ← c58115] at e_935_7
  have hr := e_935_7
  simp only [e_935_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7359)) := by
  have c58116 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.1
  have c58117 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.1
  have c58118 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.1
  have c58119 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.1
  have c58120 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.1
  have c58121 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_8
  simp only [← c58116, ← c58117, ← c58118] at e_935_8
  have e_935_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_9
  simp only [← c58119, ← c58120, k2, ← c58121] at e_935_9
  have hr := e_935_9
  simp only [e_935_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7360)) := by
  have c58122 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58123 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58124 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58125 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58126 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58127 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_10
  simp only [← c58122, ← c58123, ← c58124] at e_935_10
  have e_935_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_11
  simp only [← c58125, ← c58126, k2, ← c58127] at e_935_11
  have hr := e_935_11
  simp only [e_935_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7361)) := by
  have c58128 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58129 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58130 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58131 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58132 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58133 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_12
  simp only [← c58128, ← c58129, ← c58130] at e_935_12
  have e_935_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_13
  simp only [← c58131, ← c58132, k2, ← c58133] at e_935_13
  have hr := e_935_13
  simp only [e_935_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7362)) := by
  have c58134 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58135 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58136 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58137 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58138 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58139 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_14
  simp only [← c58134, ← c58135, ← c58136] at e_935_14
  have e_935_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_15
  simp only [← c58137, ← c58138, k2, ← c58139] at e_935_15
  have hr := e_935_15
  simp only [e_935_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7363)) := by
  have c58140 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58141 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58142 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58143 := (publicBatchWrapper89_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58144 := (publicBatchWrapper89_copies1817 a h).1
  have c58145 := (publicBatchWrapper89_copies1817 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_16
  simp only [← c58140, ← c58141, ← c58142] at e_935_16
  have e_935_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_17
  simp only [← c58143, ← c58144, k2, ← c58145] at e_935_17
  have hr := e_935_17
  simp only [e_935_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 935 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7364)) := by
  have c58146 := (publicBatchWrapper89_copies1817 a h).2.2.1
  have c58147 := (publicBatchWrapper89_copies1817 a h).2.2.2.1
  have c58148 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.1
  have c58149 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.1
  have c58150 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.1
  have c58151 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_935_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_18
  simp only [← c58146, ← c58147, ← c58148] at e_935_18
  have e_935_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_19
  simp only [← c58149, ← c58150, k2, ← c58151] at e_935_19
  have hr := e_935_19
  simp only [e_935_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7365)) := by
  have c58152 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.1
  have c58153 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.1
  have c58154 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58155 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58156 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58157 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_0
  simp only [← c58152, ← c58153, ← c58154] at e_936_0
  have e_936_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_1
  simp only [← c58155, ← c58156, k2, ← c58157] at e_936_1
  have hr := e_936_1
  simp only [e_936_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7366)) := by
  have c58158 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58159 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58160 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58161 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58162 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58163 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_2
  simp only [← c58158, ← c58159, ← c58160] at e_936_2
  have e_936_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_3
  simp only [← c58161, ← c58162, k2, ← c58163] at e_936_3
  have hr := e_936_3
  simp only [e_936_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7367)) := by
  have c58164 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58165 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58166 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58167 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58168 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58169 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_4
  simp only [← c58164, ← c58165, ← c58166] at e_936_4
  have e_936_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_5
  simp only [← c58167, ← c58168, k2, ← c58169] at e_936_5
  have hr := e_936_5
  simp only [e_936_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7368)) := by
  have c58170 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58171 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58172 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58173 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58174 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58175 := (publicBatchWrapper89_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_6
  simp only [← c58170, ← c58171, ← c58172] at e_936_6
  have e_936_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_7
  simp only [← c58173, ← c58174, k2, ← c58175] at e_936_7
  have hr := e_936_7
  simp only [e_936_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7369)) := by
  have c58176 := (publicBatchWrapper89_copies1818 a h).1
  have c58177 := (publicBatchWrapper89_copies1818 a h).2.1
  have c58178 := (publicBatchWrapper89_copies1818 a h).2.2.1
  have c58179 := (publicBatchWrapper89_copies1818 a h).2.2.2.1
  have c58180 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.1
  have c58181 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_8
  simp only [← c58176, ← c58177, ← c58178] at e_936_8
  have e_936_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_9
  simp only [← c58179, ← c58180, k2, ← c58181] at e_936_9
  have hr := e_936_9
  simp only [e_936_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7410)) := by
  have c58182 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.1
  have c58183 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.1
  have c58184 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.1
  have c58185 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.1
  have c58186 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58187 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_10
  simp only [← c58182, ← c58183, ← c58184] at e_936_10
  have e_936_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_11
  simp only [← c58185, ← c58186, k2, ← c58187] at e_936_11
  have hr := e_936_11
  simp only [e_936_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7411)) := by
  have c58188 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58189 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58190 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58191 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58192 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58193 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_12
  simp only [← c58188, ← c58189, ← c58190] at e_936_12
  have e_936_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_13
  simp only [← c58191, ← c58192, k2, ← c58193] at e_936_13
  have hr := e_936_13
  simp only [e_936_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7412)) := by
  have c58194 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58195 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58196 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58197 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58198 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58199 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_14
  simp only [← c58194, ← c58195, ← c58196] at e_936_14
  have e_936_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_15
  simp only [← c58197, ← c58198, k2, ← c58199] at e_936_15
  have hr := e_936_15
  simp only [e_936_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7413)) := by
  have c58200 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58201 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58202 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58203 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58204 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58205 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_16
  simp only [← c58200, ← c58201, ← c58202] at e_936_16
  have e_936_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_17
  simp only [← c58203, ← c58204, k2, ← c58205] at e_936_17
  have hr := e_936_17
  simp only [e_936_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 936 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7414)) := by
  have c58206 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58207 := (publicBatchWrapper89_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58208 := (publicBatchWrapper89_copies1819 a h).1
  have c58209 := (publicBatchWrapper89_copies1819 a h).2.1
  have c58210 := (publicBatchWrapper89_copies1819 a h).2.2.1
  have c58211 := (publicBatchWrapper89_copies1819 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_936_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_18
  simp only [← c58206, ← c58207, ← c58208] at e_936_18
  have e_936_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_19
  simp only [← c58209, ← c58210, k2, ← c58211] at e_936_19
  have hr := e_936_19
  simp only [e_936_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7415)) := by
  have c58212 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.1
  have c58213 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.1
  have c58214 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.1
  have c58215 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.1
  have c58216 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.1
  have c58217 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_0
  simp only [← c58212, ← c58213, ← c58214] at e_937_0
  have e_937_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_1
  simp only [← c58215, ← c58216, k2, ← c58217] at e_937_1
  have hr := e_937_1
  simp only [e_937_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7416)) := by
  have c58218 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58219 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58220 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58221 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58222 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58223 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_2
  simp only [← c58218, ← c58219, ← c58220] at e_937_2
  have e_937_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_3
  simp only [← c58221, ← c58222, k2, ← c58223] at e_937_3
  have hr := e_937_3
  simp only [e_937_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7417)) := by
  have c58224 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58225 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58226 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58227 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58228 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58229 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_4
  simp only [← c58224, ← c58225, ← c58226] at e_937_4
  have e_937_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_5
  simp only [← c58227, ← c58228, k2, ← c58229] at e_937_5
  have hr := e_937_5
  simp only [e_937_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7418)) := by
  have c58230 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58231 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58232 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58233 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58234 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58235 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_6
  simp only [← c58230, ← c58231, ← c58232] at e_937_6
  have e_937_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_7
  simp only [← c58233, ← c58234, k2, ← c58235] at e_937_7
  have hr := e_937_7
  simp only [e_937_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7419)) := by
  have c58236 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58237 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58238 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58239 := (publicBatchWrapper89_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58240 := (publicBatchWrapper89_copies1820 a h).1
  have c58241 := (publicBatchWrapper89_copies1820 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_8
  simp only [← c58236, ← c58237, ← c58238] at e_937_8
  have e_937_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_9
  simp only [← c58239, ← c58240, k2, ← c58241] at e_937_9
  have hr := e_937_9
  simp only [e_937_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7420)) := by
  have c58242 := (publicBatchWrapper89_copies1820 a h).2.2.1
  have c58243 := (publicBatchWrapper89_copies1820 a h).2.2.2.1
  have c58244 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.1
  have c58245 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.1
  have c58246 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.1
  have c58247 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_10
  simp only [← c58242, ← c58243, ← c58244] at e_937_10
  have e_937_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_11
  simp only [← c58245, ← c58246, k2, ← c58247] at e_937_11
  have hr := e_937_11
  simp only [e_937_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7421)) := by
  have c58248 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.1
  have c58249 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.1
  have c58250 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58251 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58252 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58253 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_12
  simp only [← c58248, ← c58249, ← c58250] at e_937_12
  have e_937_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_13
  simp only [← c58251, ← c58252, k2, ← c58253] at e_937_13
  have hr := e_937_13
  simp only [e_937_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7422)) := by
  have c58254 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58255 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58256 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58257 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58258 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58259 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_14
  simp only [← c58254, ← c58255, ← c58256] at e_937_14
  have e_937_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_15
  simp only [← c58257, ← c58258, k2, ← c58259] at e_937_15
  have hr := e_937_15
  simp only [e_937_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7423)) := by
  have c58260 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58261 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58262 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58263 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58264 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58265 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_16
  simp only [← c58260, ← c58261, ← c58262] at e_937_16
  have e_937_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_17
  simp only [← c58263, ← c58264, k2, ← c58265] at e_937_17
  have hr := e_937_17
  simp only [e_937_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 937 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7424)) := by
  have c58266 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58267 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58268 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58269 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58270 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58271 := (publicBatchWrapper89_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_937_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_18
  simp only [← c58266, ← c58267, ← c58268] at e_937_18
  have e_937_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_19
  simp only [← c58269, ← c58270, k2, ← c58271] at e_937_19
  have hr := e_937_19
  simp only [e_937_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7425)) := by
  have c58272 := (publicBatchWrapper89_copies1821 a h).1
  have c58273 := (publicBatchWrapper89_copies1821 a h).2.1
  have c58274 := (publicBatchWrapper89_copies1821 a h).2.2.1
  have c58275 := (publicBatchWrapper89_copies1821 a h).2.2.2.1
  have c58276 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.1
  have c58277 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_0
  simp only [← c58272, ← c58273, ← c58274] at e_938_0
  have e_938_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_1
  simp only [← c58275, ← c58276, k2, ← c58277] at e_938_1
  have hr := e_938_1
  simp only [e_938_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7426)) := by
  have c58278 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.1
  have c58279 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.1
  have c58280 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.1
  have c58281 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.1
  have c58282 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58283 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_2
  simp only [← c58278, ← c58279, ← c58280] at e_938_2
  have e_938_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_3
  simp only [← c58281, ← c58282, k2, ← c58283] at e_938_3
  have hr := e_938_3
  simp only [e_938_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7427)) := by
  have c58284 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58285 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58286 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58287 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58288 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58289 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_4
  simp only [← c58284, ← c58285, ← c58286] at e_938_4
  have e_938_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_5
  simp only [← c58287, ← c58288, k2, ← c58289] at e_938_5
  have hr := e_938_5
  simp only [e_938_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7428)) := by
  have c58290 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58291 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58292 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58293 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58294 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58295 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_6
  simp only [← c58290, ← c58291, ← c58292] at e_938_6
  have e_938_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_7
  simp only [← c58293, ← c58294, k2, ← c58295] at e_938_7
  have hr := e_938_7
  simp only [e_938_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7429)) := by
  have c58296 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58297 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58298 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58299 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58300 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58301 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_8
  simp only [← c58296, ← c58297, ← c58298] at e_938_8
  have e_938_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_9
  simp only [← c58299, ← c58300, k2, ← c58301] at e_938_9
  have hr := e_938_9
  simp only [e_938_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7430)) := by
  have c58302 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58303 := (publicBatchWrapper89_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58304 := (publicBatchWrapper89_copies1822 a h).1
  have c58305 := (publicBatchWrapper89_copies1822 a h).2.1
  have c58306 := (publicBatchWrapper89_copies1822 a h).2.2.1
  have c58307 := (publicBatchWrapper89_copies1822 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_10
  simp only [← c58302, ← c58303, ← c58304] at e_938_10
  have e_938_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_11
  simp only [← c58305, ← c58306, k2, ← c58307] at e_938_11
  have hr := e_938_11
  simp only [e_938_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7431)) := by
  have c58308 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.1
  have c58309 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.1
  have c58310 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.1
  have c58311 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.1
  have c58312 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.1
  have c58313 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_12
  simp only [← c58308, ← c58309, ← c58310] at e_938_12
  have e_938_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_13
  simp only [← c58311, ← c58312, k2, ← c58313] at e_938_13
  have hr := e_938_13
  simp only [e_938_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7432)) := by
  have c58314 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58315 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58316 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58317 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58318 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58319 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_14
  simp only [← c58314, ← c58315, ← c58316] at e_938_14
  have e_938_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_15
  simp only [← c58317, ← c58318, k2, ← c58319] at e_938_15
  have hr := e_938_15
  simp only [e_938_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7433)) := by
  have c58320 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58321 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58322 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58323 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58324 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58325 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_16
  simp only [← c58320, ← c58321, ← c58322] at e_938_16
  have e_938_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_17
  simp only [← c58323, ← c58324, k2, ← c58325] at e_938_17
  have hr := e_938_17
  simp only [e_938_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 938 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7434)) := by
  have c58326 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58327 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58328 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58329 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58330 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58331 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_938_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_18
  simp only [← c58326, ← c58327, ← c58328] at e_938_18
  have e_938_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_19
  simp only [← c58329, ← c58330, k2, ← c58331] at e_938_19
  have hr := e_938_19
  simp only [e_938_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7435)) := by
  have c58332 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58333 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58334 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58335 := (publicBatchWrapper89_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58336 := (publicBatchWrapper89_copies1823 a h).1
  have c58337 := (publicBatchWrapper89_copies1823 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_0
  simp only [← c58332, ← c58333, ← c58334] at e_939_0
  have e_939_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_1
  simp only [← c58335, ← c58336, k2, ← c58337] at e_939_1
  have hr := e_939_1
  simp only [e_939_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7436)) := by
  have c58338 := (publicBatchWrapper89_copies1823 a h).2.2.1
  have c58339 := (publicBatchWrapper89_copies1823 a h).2.2.2.1
  have c58340 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.1
  have c58341 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.1
  have c58342 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.1
  have c58343 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_2
  simp only [← c58338, ← c58339, ← c58340] at e_939_2
  have e_939_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_3
  simp only [← c58341, ← c58342, k2, ← c58343] at e_939_3
  have hr := e_939_3
  simp only [e_939_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7437)) := by
  have c58344 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.1
  have c58345 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.1
  have c58346 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58347 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58348 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58349 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_4
  simp only [← c58344, ← c58345, ← c58346] at e_939_4
  have e_939_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_5
  simp only [← c58347, ← c58348, k2, ← c58349] at e_939_5
  have hr := e_939_5
  simp only [e_939_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7438)) := by
  have c58350 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58351 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58352 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58353 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58354 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58355 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_6
  simp only [← c58350, ← c58351, ← c58352] at e_939_6
  have e_939_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_7
  simp only [← c58353, ← c58354, k2, ← c58355] at e_939_7
  have hr := e_939_7
  simp only [e_939_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7439)) := by
  have c58356 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58357 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58358 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58359 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58360 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58361 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_8
  simp only [← c58356, ← c58357, ← c58358] at e_939_8
  have e_939_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_9
  simp only [← c58359, ← c58360, k2, ← c58361] at e_939_9
  have hr := e_939_9
  simp only [e_939_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7440)) := by
  have c58362 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58363 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58364 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58365 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58366 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58367 := (publicBatchWrapper89_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_10
  simp only [← c58362, ← c58363, ← c58364] at e_939_10
  have e_939_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_11
  simp only [← c58365, ← c58366, k2, ← c58367] at e_939_11
  have hr := e_939_11
  simp only [e_939_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7441)) := by
  have c58368 := (publicBatchWrapper89_copies1824 a h).1
  have c58369 := (publicBatchWrapper89_copies1824 a h).2.1
  have c58370 := (publicBatchWrapper89_copies1824 a h).2.2.1
  have c58371 := (publicBatchWrapper89_copies1824 a h).2.2.2.1
  have c58372 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.1
  have c58373 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_12
  simp only [← c58368, ← c58369, ← c58370] at e_939_12
  have e_939_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_13
  simp only [← c58371, ← c58372, k2, ← c58373] at e_939_13
  have hr := e_939_13
  simp only [e_939_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7442)) := by
  have c58374 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.1
  have c58375 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.1
  have c58376 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.1
  have c58377 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.1
  have c58378 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58379 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_14
  simp only [← c58374, ← c58375, ← c58376] at e_939_14
  have e_939_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_15
  simp only [← c58377, ← c58378, k2, ← c58379] at e_939_15
  have hr := e_939_15
  simp only [e_939_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7443)) := by
  have c58380 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58381 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58382 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58383 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58384 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58385 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_16
  simp only [← c58380, ← c58381, ← c58382] at e_939_16
  have e_939_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_17
  simp only [← c58383, ← c58384, k2, ← c58385] at e_939_17
  have hr := e_939_17
  simp only [e_939_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 939 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7444)) := by
  have c58386 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58387 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58388 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58389 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58390 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58391 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_939_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_18
  simp only [← c58386, ← c58387, ← c58388] at e_939_18
  have e_939_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_19
  simp only [← c58389, ← c58390, k2, ← c58391] at e_939_19
  have hr := e_939_19
  simp only [e_939_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7445)) := by
  have c58392 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58393 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58394 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58395 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58396 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58397 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_0
  simp only [← c58392, ← c58393, ← c58394] at e_940_0
  have e_940_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_1
  simp only [← c58395, ← c58396, k2, ← c58397] at e_940_1
  have hr := e_940_1
  simp only [e_940_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7446)) := by
  have c58398 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58399 := (publicBatchWrapper89_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58400 := (publicBatchWrapper89_copies1825 a h).1
  have c58401 := (publicBatchWrapper89_copies1825 a h).2.1
  have c58402 := (publicBatchWrapper89_copies1825 a h).2.2.1
  have c58403 := (publicBatchWrapper89_copies1825 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_2
  simp only [← c58398, ← c58399, ← c58400] at e_940_2
  have e_940_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_3
  simp only [← c58401, ← c58402, k2, ← c58403] at e_940_3
  have hr := e_940_3
  simp only [e_940_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7447)) := by
  have c58404 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.1
  have c58405 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.1
  have c58406 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.1
  have c58407 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.1
  have c58408 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.1
  have c58409 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_4
  simp only [← c58404, ← c58405, ← c58406] at e_940_4
  have e_940_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_5
  simp only [← c58407, ← c58408, k2, ← c58409] at e_940_5
  have hr := e_940_5
  simp only [e_940_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7448)) := by
  have c58410 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58411 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58412 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58413 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58414 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58415 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_6
  simp only [← c58410, ← c58411, ← c58412] at e_940_6
  have e_940_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_7
  simp only [← c58413, ← c58414, k2, ← c58415] at e_940_7
  have hr := e_940_7
  simp only [e_940_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7449)) := by
  have c58416 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58417 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58418 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58419 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58420 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58421 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_8
  simp only [← c58416, ← c58417, ← c58418] at e_940_8
  have e_940_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_9
  simp only [← c58419, ← c58420, k2, ← c58421] at e_940_9
  have hr := e_940_9
  simp only [e_940_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7450)) := by
  have c58422 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58423 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58424 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58425 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58426 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58427 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_10
  simp only [← c58422, ← c58423, ← c58424] at e_940_10
  have e_940_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_11
  simp only [← c58425, ← c58426, k2, ← c58427] at e_940_11
  have hr := e_940_11
  simp only [e_940_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7451)) := by
  have c58428 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58429 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58430 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58431 := (publicBatchWrapper89_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58432 := (publicBatchWrapper89_copies1826 a h).1
  have c58433 := (publicBatchWrapper89_copies1826 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_12
  simp only [← c58428, ← c58429, ← c58430] at e_940_12
  have e_940_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_13
  simp only [← c58431, ← c58432, k2, ← c58433] at e_940_13
  have hr := e_940_13
  simp only [e_940_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7452)) := by
  have c58434 := (publicBatchWrapper89_copies1826 a h).2.2.1
  have c58435 := (publicBatchWrapper89_copies1826 a h).2.2.2.1
  have c58436 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.1
  have c58437 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.1
  have c58438 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.1
  have c58439 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_14
  simp only [← c58434, ← c58435, ← c58436] at e_940_14
  have e_940_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_15
  simp only [← c58437, ← c58438, k2, ← c58439] at e_940_15
  have hr := e_940_15
  simp only [e_940_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7453)) := by
  have c58440 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.1
  have c58441 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.1
  have c58442 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58443 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58444 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58445 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_16
  simp only [← c58440, ← c58441, ← c58442] at e_940_16
  have e_940_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_17
  simp only [← c58443, ← c58444, k2, ← c58445] at e_940_17
  have hr := e_940_17
  simp only [e_940_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 940 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7454)) := by
  have c58446 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58447 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58448 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58449 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58450 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58451 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_940_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_18
  simp only [← c58446, ← c58447, ← c58448] at e_940_18
  have e_940_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_19
  simp only [← c58449, ← c58450, k2, ← c58451] at e_940_19
  have hr := e_940_19
  simp only [e_940_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7455)) := by
  have c58452 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58453 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58454 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58455 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58456 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58457 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_0
  simp only [← c58452, ← c58453, ← c58454] at e_941_0
  have e_941_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_1
  simp only [← c58455, ← c58456, k2, ← c58457] at e_941_1
  have hr := e_941_1
  simp only [e_941_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7456)) := by
  have c58458 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58459 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58460 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58461 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58462 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58463 := (publicBatchWrapper89_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_2
  simp only [← c58458, ← c58459, ← c58460] at e_941_2
  have e_941_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_3
  simp only [← c58461, ← c58462, k2, ← c58463] at e_941_3
  have hr := e_941_3
  simp only [e_941_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7457)) := by
  have c58464 := (publicBatchWrapper89_copies1827 a h).1
  have c58465 := (publicBatchWrapper89_copies1827 a h).2.1
  have c58466 := (publicBatchWrapper89_copies1827 a h).2.2.1
  have c58467 := (publicBatchWrapper89_copies1827 a h).2.2.2.1
  have c58468 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.1
  have c58469 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_4
  simp only [← c58464, ← c58465, ← c58466] at e_941_4
  have e_941_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_5
  simp only [← c58467, ← c58468, k2, ← c58469] at e_941_5
  have hr := e_941_5
  simp only [e_941_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7458)) := by
  have c58470 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.1
  have c58471 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.1
  have c58472 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.1
  have c58473 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.1
  have c58474 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58475 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_6
  simp only [← c58470, ← c58471, ← c58472] at e_941_6
  have e_941_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_7
  simp only [← c58473, ← c58474, k2, ← c58475] at e_941_7
  have hr := e_941_7
  simp only [e_941_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7459)) := by
  have c58476 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58477 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58478 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58479 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58480 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58481 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_8
  simp only [← c58476, ← c58477, ← c58478] at e_941_8
  have e_941_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_9
  simp only [← c58479, ← c58480, k2, ← c58481] at e_941_9
  have hr := e_941_9
  simp only [e_941_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7460)) := by
  have c58482 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58483 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58484 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58485 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58486 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58487 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_10
  simp only [← c58482, ← c58483, ← c58484] at e_941_10
  have e_941_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_11
  simp only [← c58485, ← c58486, k2, ← c58487] at e_941_11
  have hr := e_941_11
  simp only [e_941_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7461)) := by
  have c58488 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58489 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58490 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58491 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58492 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58493 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_12
  simp only [← c58488, ← c58489, ← c58490] at e_941_12
  have e_941_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_13
  simp only [← c58491, ← c58492, k2, ← c58493] at e_941_13
  have hr := e_941_13
  simp only [e_941_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7462)) := by
  have c58494 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58495 := (publicBatchWrapper89_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58496 := (publicBatchWrapper89_copies1828 a h).1
  have c58497 := (publicBatchWrapper89_copies1828 a h).2.1
  have c58498 := (publicBatchWrapper89_copies1828 a h).2.2.1
  have c58499 := (publicBatchWrapper89_copies1828 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_14
  simp only [← c58494, ← c58495, ← c58496] at e_941_14
  have e_941_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_15
  simp only [← c58497, ← c58498, k2, ← c58499] at e_941_15
  have hr := e_941_15
  simp only [e_941_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f8999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7463)) := by
  have c58500 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.1
  have c58501 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.1
  have c58502 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.1
  have c58503 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.1
  have c58504 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.1
  have c58505 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_16
  simp only [← c58500, ← c58501, ← c58502] at e_941_16
  have e_941_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_17
  simp only [← c58503, ← c58504, k2, ← c58505] at e_941_17
  have hr := e_941_17
  simp only [e_941_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 941 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7464)) := by
  have c58506 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58507 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58508 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58509 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58510 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58511 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_941_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_18
  simp only [← c58506, ← c58507, ← c58508] at e_941_18
  have e_941_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_19
  simp only [← c58509, ← c58510, k2, ← c58511] at e_941_19
  have hr := e_941_19
  simp only [e_941_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7465)) := by
  have c58512 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58513 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58514 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58515 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58516 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58517 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_0
  simp only [← c58512, ← c58513, ← c58514] at e_942_0
  have e_942_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_1
  simp only [← c58515, ← c58516, k2, ← c58517] at e_942_1
  have hr := e_942_1
  simp only [e_942_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7466)) := by
  have c58518 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58519 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58520 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58521 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58522 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58523 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_2
  simp only [← c58518, ← c58519, ← c58520] at e_942_2
  have e_942_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_3
  simp only [← c58521, ← c58522, k2, ← c58523] at e_942_3
  have hr := e_942_3
  simp only [e_942_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7467)) := by
  have c58524 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58525 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58526 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58527 := (publicBatchWrapper89_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58528 := (publicBatchWrapper89_copies1829 a h).1
  have c58529 := (publicBatchWrapper89_copies1829 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_4
  simp only [← c58524, ← c58525, ← c58526] at e_942_4
  have e_942_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_5
  simp only [← c58527, ← c58528, k2, ← c58529] at e_942_5
  have hr := e_942_5
  simp only [e_942_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7468)) := by
  have c58530 := (publicBatchWrapper89_copies1829 a h).2.2.1
  have c58531 := (publicBatchWrapper89_copies1829 a h).2.2.2.1
  have c58532 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.1
  have c58533 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.1
  have c58534 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.1
  have c58535 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_6
  simp only [← c58530, ← c58531, ← c58532] at e_942_6
  have e_942_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_7
  simp only [← c58533, ← c58534, k2, ← c58535] at e_942_7
  have hr := e_942_7
  simp only [e_942_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7469)) := by
  have c58536 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.1
  have c58537 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.1
  have c58538 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58539 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58540 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58541 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_8
  simp only [← c58536, ← c58537, ← c58538] at e_942_8
  have e_942_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_9
  simp only [← c58539, ← c58540, k2, ← c58541] at e_942_9
  have hr := e_942_9
  simp only [e_942_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7470)) := by
  have c58542 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58543 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58544 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58545 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58546 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58547 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_10
  simp only [← c58542, ← c58543, ← c58544] at e_942_10
  have e_942_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_11
  simp only [← c58545, ← c58546, k2, ← c58547] at e_942_11
  have hr := e_942_11
  simp only [e_942_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7471)) := by
  have c58548 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58549 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58550 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58551 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58552 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58553 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_12
  simp only [← c58548, ← c58549, ← c58550] at e_942_12
  have e_942_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_13
  simp only [← c58551, ← c58552, k2, ← c58553] at e_942_13
  have hr := e_942_13
  simp only [e_942_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7472)) := by
  have c58554 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58555 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58556 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58557 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58558 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58559 := (publicBatchWrapper89_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_14
  simp only [← c58554, ← c58555, ← c58556] at e_942_14
  have e_942_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_15
  simp only [← c58557, ← c58558, k2, ← c58559] at e_942_15
  have hr := e_942_15
  simp only [e_942_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7473)) := by
  have c58560 := (publicBatchWrapper89_copies1830 a h).1
  have c58561 := (publicBatchWrapper89_copies1830 a h).2.1
  have c58562 := (publicBatchWrapper89_copies1830 a h).2.2.1
  have c58563 := (publicBatchWrapper89_copies1830 a h).2.2.2.1
  have c58564 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.1
  have c58565 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_16
  simp only [← c58560, ← c58561, ← c58562] at e_942_16
  have e_942_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_17
  simp only [← c58563, ← c58564, k2, ← c58565] at e_942_17
  have hr := e_942_17
  simp only [e_942_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 942 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7474)) := by
  have c58566 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.1
  have c58567 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.1
  have c58568 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.1
  have c58569 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.1
  have c58570 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58571 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_942_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_18
  simp only [← c58566, ← c58567, ← c58568] at e_942_18
  have e_942_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_19
  simp only [← c58569, ← c58570, k2, ← c58571] at e_942_19
  have hr := e_942_19
  simp only [e_942_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 943 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7475)) := by
  have c58572 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58573 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58574 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58575 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58576 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58577 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_943_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_0
  simp only [← c58572, ← c58573, ← c58574] at e_943_0
  have e_943_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_1
  simp only [← c58575, ← c58576, k2, ← c58577] at e_943_1
  have hr := e_943_1
  simp only [e_943_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 943 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7476)) := by
  have c58578 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58579 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58580 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58581 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58582 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58583 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_943_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_2
  simp only [← c58578, ← c58579, ← c58580] at e_943_2
  have e_943_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_3
  simp only [← c58581, ← c58582, k2, ← c58583] at e_943_3
  have hr := e_943_3
  simp only [e_943_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 943 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7477)) := by
  have c58584 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58585 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58586 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58587 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58588 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58589 := (publicBatchWrapper89_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_943_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_4
  simp only [← c58584, ← c58585, ← c58586] at e_943_4
  have e_943_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_5
  simp only [← c58587, ← c58588, k2, ← c58589] at e_943_5
  have hr := e_943_5
  simp only [e_943_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
