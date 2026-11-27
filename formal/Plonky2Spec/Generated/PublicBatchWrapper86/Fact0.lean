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

theorem publicBatchWrapper86_f0 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 0) = a (.virt 17206) := by
  have c0 := (publicBatchWrapper86_copies0 a h).1
  exact c0

theorem publicBatchWrapper86_f1 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 202) = a (.virt 17206) := by
  have c1 := (publicBatchWrapper86_copies0 a h).2.1
  exact c1

theorem publicBatchWrapper86_f2 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 402) = a (.virt 17206) := by
  have c2 := (publicBatchWrapper86_copies0 a h).2.2.1
  exact c2

theorem publicBatchWrapper86_f3 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 602) = a (.virt 17206) := by
  have c3 := (publicBatchWrapper86_copies0 a h).2.2.2.1
  exact c3

theorem publicBatchWrapper86_f4 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 802) = a (.virt 17206) := by
  have c4 := (publicBatchWrapper86_copies0 a h).2.2.2.2.1
  exact c4

theorem publicBatchWrapper86_f5 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 1002) = a (.virt 17206) := by
  have c5 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.1
  exact c5

theorem publicBatchWrapper86_f6 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 1202) = a (.virt 17206) := by
  have c6 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.1
  exact c6

theorem publicBatchWrapper86_f7 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 1402) = a (.virt 17206) := by
  have c7 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.1
  exact c7

theorem publicBatchWrapper86_f8 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 1602) = a (.virt 17206) := by
  have c8 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.1
  exact c8

theorem publicBatchWrapper86_f9 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 1802) = a (.virt 17206) := by
  have c9 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.1
  exact c9

theorem publicBatchWrapper86_f10 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 2002) = a (.virt 17206) := by
  have c10 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c10

theorem publicBatchWrapper86_f11 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 2202) = a (.virt 17206) := by
  have c11 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c11

theorem publicBatchWrapper86_f12 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 2402) = a (.virt 17206) := by
  have c12 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12

theorem publicBatchWrapper86_f13 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 2602) = a (.virt 17206) := by
  have c13 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13

theorem publicBatchWrapper86_f14 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 2802) = a (.virt 17206) := by
  have c14 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14

theorem publicBatchWrapper86_f15 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 3002) = a (.virt 17206) := by
  have c15 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15

theorem publicBatchWrapper86_f16 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 3202) = a (.virt 17206) := by
  have c16 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16

theorem publicBatchWrapper86_f17 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 3402) = a (.virt 17206) := by
  have c17 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17

theorem publicBatchWrapper86_f18 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 3602) = a (.virt 17206) := by
  have c18 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18

theorem publicBatchWrapper86_f19 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 3802) = a (.virt 17206) := by
  have c19 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19

theorem publicBatchWrapper86_f20 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 4002) = a (.virt 17206) := by
  have c20 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20

theorem publicBatchWrapper86_f21 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 4202) = a (.virt 17206) := by
  have c21 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21

theorem publicBatchWrapper86_f22 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 4402) = a (.virt 17206) := by
  have c22 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22

theorem publicBatchWrapper86_f23 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 4602) = a (.virt 17206) := by
  have c23 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c23

theorem publicBatchWrapper86_f24 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 4802) = a (.virt 17206) := by
  have c24 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c24

theorem publicBatchWrapper86_f25 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 5002) = a (.virt 17206) := by
  have c25 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c25

theorem publicBatchWrapper86_f26 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 5202) = a (.virt 17206) := by
  have c26 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c26

theorem publicBatchWrapper86_f27 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 5402) = a (.virt 17206) := by
  have c27 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c27

theorem publicBatchWrapper86_f28 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 5602) = a (.virt 17206) := by
  have c28 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c28

theorem publicBatchWrapper86_f29 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 5802) = a (.virt 17206) := by
  have c29 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c29

theorem publicBatchWrapper86_f30 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 6002) = a (.virt 17206) := by
  have c30 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c30

theorem publicBatchWrapper86_f31 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 6202) = a (.virt 17206) := by
  have c31 := (publicBatchWrapper86_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c31

theorem publicBatchWrapper86_f32 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 6402) = a (.virt 17206) := by
  have c32 := (publicBatchWrapper86_copies1 a h).1
  exact c32

theorem publicBatchWrapper86_f33 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 6602) = a (.virt 17206) := by
  have c33 := (publicBatchWrapper86_copies1 a h).2.1
  exact c33

theorem publicBatchWrapper86_f34 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 6802) = a (.virt 17206) := by
  have c34 := (publicBatchWrapper86_copies1 a h).2.2.1
  exact c34

theorem publicBatchWrapper86_f35 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 7002) = a (.virt 17206) := by
  have c35 := (publicBatchWrapper86_copies1 a h).2.2.2.1
  exact c35

theorem publicBatchWrapper86_f36 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 7202) = a (.virt 17206) := by
  have c36 := (publicBatchWrapper86_copies1 a h).2.2.2.2.1
  exact c36

theorem publicBatchWrapper86_f37 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 7402) = a (.virt 17206) := by
  have c37 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.1
  exact c37

theorem publicBatchWrapper86_f38 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 7602) = a (.virt 17206) := by
  have c38 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.1
  exact c38

theorem publicBatchWrapper86_f39 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 7802) = a (.virt 17206) := by
  have c39 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.1
  exact c39

theorem publicBatchWrapper86_f40 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 8002) = a (.virt 17206) := by
  have c40 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.1
  exact c40

theorem publicBatchWrapper86_f41 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 8202) = a (.virt 17206) := by
  have c41 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.1
  exact c41

theorem publicBatchWrapper86_f42 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 8402) = a (.virt 17206) := by
  have c42 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c42

theorem publicBatchWrapper86_f43 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 8602) = a (.virt 17206) := by
  have c43 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c43

theorem publicBatchWrapper86_f44 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 8802) = a (.virt 17206) := by
  have c44 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c44

theorem publicBatchWrapper86_f45 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 9002) = a (.virt 17206) := by
  have c45 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c45

theorem publicBatchWrapper86_f46 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 9202) = a (.virt 17206) := by
  have c46 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c46

theorem publicBatchWrapper86_f47 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 9402) = a (.virt 17206) := by
  have c47 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c47

theorem publicBatchWrapper86_f48 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 9602) = a (.virt 17206) := by
  have c48 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c48

theorem publicBatchWrapper86_f49 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 9802) = a (.virt 17206) := by
  have c49 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c49

theorem publicBatchWrapper86_f50 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 10002) = a (.virt 17206) := by
  have c50 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c50

theorem publicBatchWrapper86_f51 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 10202) = a (.virt 17206) := by
  have c51 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c51

theorem publicBatchWrapper86_f52 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 10402) = a (.virt 17206) := by
  have c52 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c52

theorem publicBatchWrapper86_f53 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 10602) = a (.virt 17206) := by
  have c53 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c53

theorem publicBatchWrapper86_f54 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 10802) = a (.virt 17206) := by
  have c54 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c54

theorem publicBatchWrapper86_f55 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 11002) = a (.virt 17206) := by
  have c55 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c55

theorem publicBatchWrapper86_f56 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 11202) = a (.virt 17206) := by
  have c56 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c56

theorem publicBatchWrapper86_f57 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 11402) = a (.virt 17206) := by
  have c57 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c57

theorem publicBatchWrapper86_f58 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 11602) = a (.virt 17206) := by
  have c58 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c58

theorem publicBatchWrapper86_f59 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 11802) = a (.virt 17206) := by
  have c59 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c59

theorem publicBatchWrapper86_f60 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 12002) = a (.virt 17206) := by
  have c60 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c60

theorem publicBatchWrapper86_f61 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 12202) = a (.virt 17206) := by
  have c61 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c61

theorem publicBatchWrapper86_f62 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 12402) = a (.virt 17206) := by
  have c62 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c62

theorem publicBatchWrapper86_f63 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 12602) = a (.virt 17206) := by
  have c63 := (publicBatchWrapper86_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c63

theorem publicBatchWrapper86_f64 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 12802) = a (.virt 17206) := by
  have c64 := (publicBatchWrapper86_copies2 a h).1
  exact c64

theorem publicBatchWrapper86_f65 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 13002) = a (.virt 17206) := by
  have c65 := (publicBatchWrapper86_copies2 a h).2.1
  exact c65

theorem publicBatchWrapper86_f66 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 13202) = a (.virt 17206) := by
  have c66 := (publicBatchWrapper86_copies2 a h).2.2.1
  exact c66

theorem publicBatchWrapper86_f67 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 13402) = a (.virt 17206) := by
  have c67 := (publicBatchWrapper86_copies2 a h).2.2.2.1
  exact c67

theorem publicBatchWrapper86_f68 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 13602) = a (.virt 17206) := by
  have c68 := (publicBatchWrapper86_copies2 a h).2.2.2.2.1
  exact c68

theorem publicBatchWrapper86_f69 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 13802) = a (.virt 17206) := by
  have c69 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.1
  exact c69

theorem publicBatchWrapper86_f70 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 14002) = a (.virt 17206) := by
  have c70 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.1
  exact c70

theorem publicBatchWrapper86_f71 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 14202) = a (.virt 17206) := by
  have c71 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.1
  exact c71

theorem publicBatchWrapper86_f72 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 14402) = a (.virt 17206) := by
  have c72 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.1
  exact c72

theorem publicBatchWrapper86_f73 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 14602) = a (.virt 17206) := by
  have c73 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.1
  exact c73

theorem publicBatchWrapper86_f74 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 14802) = a (.virt 17206) := by
  have c74 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c74

theorem publicBatchWrapper86_f75 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 15002) = a (.virt 17206) := by
  have c75 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c75

theorem publicBatchWrapper86_f76 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 15202) = a (.virt 17206) := by
  have c76 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c76

theorem publicBatchWrapper86_f77 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 15402) = a (.virt 17206) := by
  have c77 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c77

theorem publicBatchWrapper86_f78 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 15602) = a (.virt 17206) := by
  have c78 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c78

theorem publicBatchWrapper86_f79 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 15802) = a (.virt 17206) := by
  have c79 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c79

theorem publicBatchWrapper86_f80 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 16002) = a (.virt 17206) := by
  have c80 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c80

theorem publicBatchWrapper86_f81 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 16202) = a (.virt 17206) := by
  have c81 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c81

theorem publicBatchWrapper86_f82 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 16402) = a (.virt 17206) := by
  have c82 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c82

theorem publicBatchWrapper86_f83 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 16602) = a (.virt 17206) := by
  have c83 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c83

theorem publicBatchWrapper86_f84 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 16802) = a (.virt 17206) := by
  have c84 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c84

theorem publicBatchWrapper86_f85 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.virt 17002) = a (.virt 17206) := by
  have c85 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c85

theorem publicBatchWrapper86_f86 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 200)) (a (.virt 17207)) (a (.virt 17208)) := by
  have c86 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (publicBatchWrapper86_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (publicBatchWrapper86_copies3 a h).1
  have c97 := (publicBatchWrapper86_copies3 a h).2.1
  have c98 := (publicBatchWrapper86_copies3 a h).2.2.1
  have c99 := (publicBatchWrapper86_copies3 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_0_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_16
  simp only [← c89, ← c90, ← c91] at e_0_16
  have e_0_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c92, ← c93, ← c94] at e_0_17
  have e_2_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c86, k3, ← c87, k3, ← c88] at e_2_0
  have e_2_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c95, ← c96, k3, ← c97] at e_2_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_0_16
    linear_combination c98.trans k2 - hc
  · have hc := e_2_1
    simp only [e_2_0, e_0_17] at hc
    linear_combination c99.trans k2 - hc

theorem publicBatchWrapper86_f87 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 200)) (a (.virt 17209)) (a (.virt 17210)) := by
  have c100 := (publicBatchWrapper86_copies3 a h).2.2.2.2.1
  have c101 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.1
  have c102 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.1
  have c103 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_0_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c103, ← c104, ← c105] at e_0_18
  have e_0_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row0 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c106, ← c107, ← c108] at e_0_19
  have e_2_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c100, k3, ← c101, k3, ← c102] at e_2_2
  have e_2_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c109, ← c110, k3, ← c111] at e_2_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_0_18
    linear_combination c112.trans k2 - hc
  · have hc := e_2_3
    simp only [e_2_2, e_0_19] at hc
    linear_combination c113.trans k2 - hc

theorem publicBatchWrapper86_f88 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5)) (a (.virt 200)) (a (.virt 17211)) (a (.virt 17212)) := by
  have c114 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c127 := (publicBatchWrapper86_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c114, k3, ← c115, k3, ← c116] at e_2_4
  have e_2_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c123, ← c124, k3, ← c125] at e_2_5
  have e_3_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c117, ← c118, ← c119] at e_3_0
  have e_3_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c120, ← c121, ← c122] at e_3_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_0
    linear_combination c126.trans k2 - hc
  · have hc := e_2_5
    simp only [e_2_4, e_3_1] at hc
    linear_combination c127.trans k2 - hc

theorem publicBatchWrapper86_f89 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6)) (a (.virt 200)) (a (.virt 17213)) (a (.virt 17214)) := by
  have c128 := (publicBatchWrapper86_copies4 a h).1
  have c129 := (publicBatchWrapper86_copies4 a h).2.1
  have c130 := (publicBatchWrapper86_copies4 a h).2.2.1
  have c131 := (publicBatchWrapper86_copies4 a h).2.2.2.1
  have c132 := (publicBatchWrapper86_copies4 a h).2.2.2.2.1
  have c133 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.1
  have c134 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.1
  have c135 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c128, k3, ← c129, k3, ← c130] at e_2_6
  have e_2_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c137, ← c138, k3, ← c139] at e_2_7
  have e_3_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c131, ← c132, ← c133] at e_3_2
  have e_3_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c134, ← c135, ← c136] at e_3_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_2
    linear_combination c140.trans k2 - hc
  · have hc := e_2_7
    simp only [e_2_6, e_3_3] at hc
    linear_combination c141.trans k2 - hc

theorem publicBatchWrapper86_f90 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 3 19) = band (a (.virt 17207)) (a (.virt 17209)) := by
  have c142 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c144 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c142, ← c143, ← c144] at e_3_4
  have hr := e_3_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f91 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 3 23) = band (a (.virt 17211)) (a (.virt 17213)) := by
  have c145 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c145, ← c146, ← c147] at e_3_5
  have hr := e_3_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f92 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 3 27) = band (a (.wire 3 19)) (a (.wire 3 23)) := by
  have c148 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c148, ← c149, ← c150] at e_3_6
  have hr := e_3_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f93 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 205)) (a (.virt 200)) (a (.virt 17215)) (a (.virt 17216)) := by
  have c151 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (publicBatchWrapper86_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (publicBatchWrapper86_copies5 a h).1
  have c161 := (publicBatchWrapper86_copies5 a h).2.1
  have c162 := (publicBatchWrapper86_copies5 a h).2.2.1
  have c163 := (publicBatchWrapper86_copies5 a h).2.2.2.1
  have c164 := (publicBatchWrapper86_copies5 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c151, k3, ← c152, k3, ← c153] at e_2_8
  have e_2_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c160, ← c161, k3, ← c162] at e_2_9
  have e_3_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c154, ← c155, ← c156] at e_3_7
  have e_3_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c157, ← c158, ← c159] at e_3_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_7
    linear_combination c163.trans k2 - hc
  · have hc := e_2_9
    simp only [e_2_8, e_3_8] at hc
    linear_combination c164.trans k2 - hc

theorem publicBatchWrapper86_f94 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 206)) (a (.virt 200)) (a (.virt 17217)) (a (.virt 17218)) := by
  have c165 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.1
  have c166 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.1
  have c167 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c165, k3, ← c166, k3, ← c167] at e_2_10
  have e_2_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c174, ← c175, k3, ← c176] at e_2_11
  have e_3_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c168, ← c169, ← c170] at e_3_9
  have e_3_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c171, ← c172, ← c173] at e_3_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_9
    linear_combination c177.trans k2 - hc
  · have hc := e_2_11
    simp only [e_2_10, e_3_10] at hc
    linear_combination c178.trans k2 - hc

theorem publicBatchWrapper86_f95 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 207)) (a (.virt 200)) (a (.virt 17219)) (a (.virt 17220)) := by
  have c179 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (publicBatchWrapper86_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (publicBatchWrapper86_copies6 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c179, k3, ← c180, k3, ← c181] at e_2_12
  have e_2_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c188, ← c189, k3, ← c190] at e_2_13
  have e_3_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c182, ← c183, ← c184] at e_3_11
  have e_3_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c185, ← c186, ← c187] at e_3_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_11
    linear_combination c191.trans k2 - hc
  · have hc := e_2_13
    simp only [e_2_12, e_3_12] at hc
    linear_combination c192.trans k2 - hc

theorem publicBatchWrapper86_f96 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 208)) (a (.virt 200)) (a (.virt 17221)) (a (.virt 17222)) := by
  have c193 := (publicBatchWrapper86_copies6 a h).2.1
  have c194 := (publicBatchWrapper86_copies6 a h).2.2.1
  have c195 := (publicBatchWrapper86_copies6 a h).2.2.2.1
  have c196 := (publicBatchWrapper86_copies6 a h).2.2.2.2.1
  have c197 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.1
  have c198 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.1
  have c199 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c193, k3, ← c194, k3, ← c195] at e_2_14
  have e_2_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c202, ← c203, k3, ← c204] at e_2_15
  have e_3_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c196, ← c197, ← c198] at e_3_13
  have e_3_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c199, ← c200, ← c201] at e_3_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_13
    linear_combination c205.trans k2 - hc
  · have hc := e_2_15
    simp only [e_2_14, e_3_14] at hc
    linear_combination c206.trans k2 - hc

theorem publicBatchWrapper86_f97 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 3 63) = band (a (.virt 17215)) (a (.virt 17217)) := by
  have c207 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c207, ← c208, ← c209] at e_3_15
  have hr := e_3_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f98 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 3 67) = band (a (.virt 17219)) (a (.virt 17221)) := by
  have c210 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c210, ← c211, ← c212] at e_3_16
  have hr := e_3_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f99 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 3 71) = band (a (.wire 3 63)) (a (.wire 3 67)) := by
  have c213 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c213, ← c214, ← c215] at e_3_17
  have hr := e_3_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 405)) (a (.virt 200)) (a (.virt 17223)) (a (.virt 17224)) := by
  have c216 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (publicBatchWrapper86_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (publicBatchWrapper86_copies7 a h).1
  have c225 := (publicBatchWrapper86_copies7 a h).2.1
  have c226 := (publicBatchWrapper86_copies7 a h).2.2.1
  have c227 := (publicBatchWrapper86_copies7 a h).2.2.2.1
  have c228 := (publicBatchWrapper86_copies7 a h).2.2.2.2.1
  have c229 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c216, k3, ← c217, k3, ← c218] at e_2_16
  have e_2_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c225, ← c226, k3, ← c227] at e_2_17
  have e_3_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c219, ← c220, ← c221] at e_3_18
  have e_3_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row3 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c222, ← c223, ← c224] at e_3_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_18
    linear_combination c228.trans k2 - hc
  · have hc := e_2_17
    simp only [e_2_16, e_3_19] at hc
    linear_combination c229.trans k2 - hc

theorem publicBatchWrapper86_f101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 406)) (a (.virt 200)) (a (.virt 17225)) (a (.virt 17226)) := by
  have c230 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.1
  have c231 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_2_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c230, k3, ← c231, k3, ← c232] at e_2_18
  have e_2_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row2 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c239, ← c240, k3, ← c241] at e_2_19
  have e_4_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c233, ← c234, ← c235] at e_4_0
  have e_4_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c236, ← c237, ← c238] at e_4_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_0
    linear_combination c242.trans k2 - hc
  · have hc := e_2_19
    simp only [e_2_18, e_4_1] at hc
    linear_combination c243.trans k2 - hc

theorem publicBatchWrapper86_f102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 407)) (a (.virt 200)) (a (.virt 17227)) (a (.virt 17228)) := by
  have c244 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (publicBatchWrapper86_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (publicBatchWrapper86_copies8 a h).1
  have c257 := (publicBatchWrapper86_copies8 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_4_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c247, ← c248, ← c249] at e_4_2
  have e_4_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c250, ← c251, ← c252] at e_4_3
  have e_5_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c244, k3, ← c245, k3, ← c246] at e_5_0
  have e_5_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c253, ← c254, k3, ← c255] at e_5_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_2
    linear_combination c256.trans k2 - hc
  · have hc := e_5_1
    simp only [e_5_0, e_4_3] at hc
    linear_combination c257.trans k2 - hc

theorem publicBatchWrapper86_f103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 408)) (a (.virt 200)) (a (.virt 17229)) (a (.virt 17230)) := by
  have c258 := (publicBatchWrapper86_copies8 a h).2.2.1
  have c259 := (publicBatchWrapper86_copies8 a h).2.2.2.1
  have c260 := (publicBatchWrapper86_copies8 a h).2.2.2.2.1
  have c261 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.1
  have c262 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.1
  have c263 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_4_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c261, ← c262, ← c263] at e_4_4
  have e_4_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c264, ← c265, ← c266] at e_4_5
  have e_5_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c258, k3, ← c259, k3, ← c260] at e_5_2
  have e_5_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c267, ← c268, k3, ← c269] at e_5_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_4
    linear_combination c270.trans k2 - hc
  · have hc := e_5_3
    simp only [e_5_2, e_4_5] at hc
    linear_combination c271.trans k2 - hc

theorem publicBatchWrapper86_f104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 4 27) = band (a (.virt 17223)) (a (.virt 17225)) := by
  have c272 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c272, ← c273, ← c274] at e_4_6
  have hr := e_4_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 4 31) = band (a (.virt 17227)) (a (.virt 17229)) := by
  have c275 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c275, ← c276, ← c277] at e_4_7
  have hr := e_4_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 4 35) = band (a (.wire 4 27)) (a (.wire 4 31)) := by
  have c278 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c278, ← c279, ← c280] at e_4_8
  have hr := e_4_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 605)) (a (.virt 200)) (a (.virt 17231)) (a (.virt 17232)) := by
  have c281 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (publicBatchWrapper86_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (publicBatchWrapper86_copies9 a h).1
  have c289 := (publicBatchWrapper86_copies9 a h).2.1
  have c290 := (publicBatchWrapper86_copies9 a h).2.2.1
  have c291 := (publicBatchWrapper86_copies9 a h).2.2.2.1
  have c292 := (publicBatchWrapper86_copies9 a h).2.2.2.2.1
  have c293 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.1
  have c294 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_4_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c284, ← c285, ← c286] at e_4_9
  have e_4_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c287, ← c288, ← c289] at e_4_10
  have e_5_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c281, k3, ← c282, k3, ← c283] at e_5_4
  have e_5_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c290, ← c291, k3, ← c292] at e_5_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_9
    linear_combination c293.trans k2 - hc
  · have hc := e_5_5
    simp only [e_5_4, e_4_10] at hc
    linear_combination c294.trans k2 - hc

theorem publicBatchWrapper86_f108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 606)) (a (.virt 200)) (a (.virt 17233)) (a (.virt 17234)) := by
  have c295 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_4_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c298, ← c299, ← c300] at e_4_11
  have e_4_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c301, ← c302, ← c303] at e_4_12
  have e_5_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c295, k3, ← c296, k3, ← c297] at e_5_6
  have e_5_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c304, ← c305, k3, ← c306] at e_5_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_11
    linear_combination c307.trans k2 - hc
  · have hc := e_5_7
    simp only [e_5_6, e_4_12] at hc
    linear_combination c308.trans k2 - hc

theorem publicBatchWrapper86_f109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 607)) (a (.virt 200)) (a (.virt 17235)) (a (.virt 17236)) := by
  have c309 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (publicBatchWrapper86_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (publicBatchWrapper86_copies10 a h).1
  have c321 := (publicBatchWrapper86_copies10 a h).2.1
  have c322 := (publicBatchWrapper86_copies10 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_4_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c312, ← c313, ← c314] at e_4_13
  have e_4_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c315, ← c316, ← c317] at e_4_14
  have e_5_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c309, k3, ← c310, k3, ← c311] at e_5_8
  have e_5_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c318, ← c319, k3, ← c320] at e_5_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_13
    linear_combination c321.trans k2 - hc
  · have hc := e_5_9
    simp only [e_5_8, e_4_14] at hc
    linear_combination c322.trans k2 - hc

theorem publicBatchWrapper86_f110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 608)) (a (.virt 200)) (a (.virt 17237)) (a (.virt 17238)) := by
  have c323 := (publicBatchWrapper86_copies10 a h).2.2.2.1
  have c324 := (publicBatchWrapper86_copies10 a h).2.2.2.2.1
  have c325 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.1
  have c326 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.1
  have c327 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_4_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_15
  simp only [← c326, ← c327, ← c328] at e_4_15
  have e_4_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_16
  simp only [← c329, ← c330, ← c331] at e_4_16
  have e_5_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c323, k3, ← c324, k3, ← c325] at e_5_10
  have e_5_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c332, ← c333, k3, ← c334] at e_5_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_4_15
    linear_combination c335.trans k2 - hc
  · have hc := e_5_11
    simp only [e_5_10, e_4_16] at hc
    linear_combination c336.trans k2 - hc

theorem publicBatchWrapper86_f111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 4 71) = band (a (.virt 17231)) (a (.virt 17233)) := by
  have c337 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_17
  simp only [← c337, ← c338, ← c339] at e_4_17
  have hr := e_4_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 4 75) = band (a (.virt 17235)) (a (.virt 17237)) := by
  have c340 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_18
  simp only [← c340, ← c341, ← c342] at e_4_18
  have hr := e_4_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 4 79) = band (a (.wire 4 71)) (a (.wire 4 75)) := by
  have c343 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row4 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_19
  simp only [← c343, ← c344, ← c345] at e_4_19
  have hr := e_4_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 805)) (a (.virt 200)) (a (.virt 17239)) (a (.virt 17240)) := by
  have c346 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (publicBatchWrapper86_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (publicBatchWrapper86_copies11 a h).1
  have c353 := (publicBatchWrapper86_copies11 a h).2.1
  have c354 := (publicBatchWrapper86_copies11 a h).2.2.1
  have c355 := (publicBatchWrapper86_copies11 a h).2.2.2.1
  have c356 := (publicBatchWrapper86_copies11 a h).2.2.2.2.1
  have c357 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.1
  have c358 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.1
  have c359 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_5_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c346, k3, ← c347, k3, ← c348] at e_5_12
  have e_5_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c355, ← c356, k3, ← c357] at e_5_13
  have e_6_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c349, ← c350, ← c351] at e_6_0
  have e_6_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c352, ← c353, ← c354] at e_6_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_0
    linear_combination c358.trans k2 - hc
  · have hc := e_5_13
    simp only [e_5_12, e_6_1] at hc
    linear_combination c359.trans k2 - hc

theorem publicBatchWrapper86_f115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 806)) (a (.virt 200)) (a (.virt 17241)) (a (.virt 17242)) := by
  have c360 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have c362 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_5_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c360, k3, ← c361, k3, ← c362] at e_5_14
  have e_5_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_15
  simp only [← c369, ← c370, k3, ← c371] at e_5_15
  have e_6_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c363, ← c364, ← c365] at e_6_2
  have e_6_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c366, ← c367, ← c368] at e_6_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_2
    linear_combination c372.trans k2 - hc
  · have hc := e_5_15
    simp only [e_5_14, e_6_3] at hc
    linear_combination c373.trans k2 - hc

theorem publicBatchWrapper86_f116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 807)) (a (.virt 200)) (a (.virt 17243)) (a (.virt 17244)) := by
  have c374 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (publicBatchWrapper86_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (publicBatchWrapper86_copies12 a h).1
  have c385 := (publicBatchWrapper86_copies12 a h).2.1
  have c386 := (publicBatchWrapper86_copies12 a h).2.2.1
  have c387 := (publicBatchWrapper86_copies12 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_5_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_16
  simp only [← c374, k3, ← c375, k3, ← c376] at e_5_16
  have e_5_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_17
  simp only [← c383, ← c384, k3, ← c385] at e_5_17
  have e_6_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c377, ← c378, ← c379] at e_6_4
  have e_6_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c380, ← c381, ← c382] at e_6_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_4
    linear_combination c386.trans k2 - hc
  · have hc := e_5_17
    simp only [e_5_16, e_6_5] at hc
    linear_combination c387.trans k2 - hc

theorem publicBatchWrapper86_f117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 808)) (a (.virt 200)) (a (.virt 17245)) (a (.virt 17246)) := by
  have c388 := (publicBatchWrapper86_copies12 a h).2.2.2.2.1
  have c389 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.1
  have c390 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.1
  have c391 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_5_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_18
  simp only [← c388, k3, ← c389, k3, ← c390] at e_5_18
  have e_5_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row5 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_19
  simp only [← c397, ← c398, k3, ← c399] at e_5_19
  have e_6_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c391, ← c392, ← c393] at e_6_6
  have e_6_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c394, ← c395, ← c396] at e_6_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_6
    linear_combination c400.trans k2 - hc
  · have hc := e_5_19
    simp only [e_5_18, e_6_7] at hc
    linear_combination c401.trans k2 - hc

theorem publicBatchWrapper86_f118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 6 35) = band (a (.virt 17239)) (a (.virt 17241)) := by
  have c402 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c402, ← c403, ← c404] at e_6_8
  have hr := e_6_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 6 39) = band (a (.virt 17243)) (a (.virt 17245)) := by
  have c405 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c405, ← c406, ← c407] at e_6_9
  have hr := e_6_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 6 43) = band (a (.wire 6 35)) (a (.wire 6 39)) := by
  have c408 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c408, ← c409, ← c410] at e_6_10
  have hr := e_6_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1005)) (a (.virt 200)) (a (.virt 17247)) (a (.virt 17248)) := by
  have c411 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (publicBatchWrapper86_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (publicBatchWrapper86_copies13 a h).1
  have c417 := (publicBatchWrapper86_copies13 a h).2.1
  have c418 := (publicBatchWrapper86_copies13 a h).2.2.1
  have c419 := (publicBatchWrapper86_copies13 a h).2.2.2.1
  have c420 := (publicBatchWrapper86_copies13 a h).2.2.2.2.1
  have c421 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.1
  have c422 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.1
  have c423 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_6_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c414, ← c415, ← c416] at e_6_11
  have e_6_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c417, ← c418, ← c419] at e_6_12
  have e_7_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c411, k3, ← c412, k3, ← c413] at e_7_0
  have e_7_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c420, ← c421, k3, ← c422] at e_7_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_11
    linear_combination c423.trans k2 - hc
  · have hc := e_7_1
    simp only [e_7_0, e_6_12] at hc
    linear_combination c424.trans k2 - hc

theorem publicBatchWrapper86_f122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1006)) (a (.virt 200)) (a (.virt 17249)) (a (.virt 17250)) := by
  have c425 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_6_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c428, ← c429, ← c430] at e_6_13
  have e_6_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c431, ← c432, ← c433] at e_6_14
  have e_7_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c425, k3, ← c426, k3, ← c427] at e_7_2
  have e_7_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c434, ← c435, k3, ← c436] at e_7_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_13
    linear_combination c437.trans k2 - hc
  · have hc := e_7_3
    simp only [e_7_2, e_6_14] at hc
    linear_combination c438.trans k2 - hc

theorem publicBatchWrapper86_f123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1007)) (a (.virt 200)) (a (.virt 17251)) (a (.virt 17252)) := by
  have c439 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (publicBatchWrapper86_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (publicBatchWrapper86_copies14 a h).1
  have c449 := (publicBatchWrapper86_copies14 a h).2.1
  have c450 := (publicBatchWrapper86_copies14 a h).2.2.1
  have c451 := (publicBatchWrapper86_copies14 a h).2.2.2.1
  have c452 := (publicBatchWrapper86_copies14 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_6_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_15
  simp only [← c442, ← c443, ← c444] at e_6_15
  have e_6_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_16
  simp only [← c445, ← c446, ← c447] at e_6_16
  have e_7_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c439, k3, ← c440, k3, ← c441] at e_7_4
  have e_7_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c448, ← c449, k3, ← c450] at e_7_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_15
    linear_combination c451.trans k2 - hc
  · have hc := e_7_5
    simp only [e_7_4, e_6_16] at hc
    linear_combination c452.trans k2 - hc

theorem publicBatchWrapper86_f124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1008)) (a (.virt 200)) (a (.virt 17253)) (a (.virt 17254)) := by
  have c453 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.1
  have c454 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.1
  have c455 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_6_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_17
  simp only [← c456, ← c457, ← c458] at e_6_17
  have e_6_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_18
  simp only [← c459, ← c460, ← c461] at e_6_18
  have e_7_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c453, k3, ← c454, k3, ← c455] at e_7_6
  have e_7_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c462, ← c463, k3, ← c464] at e_7_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_6_17
    linear_combination c465.trans k2 - hc
  · have hc := e_7_7
    simp only [e_7_6, e_6_18] at hc
    linear_combination c466.trans k2 - hc

theorem publicBatchWrapper86_f125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 6 79) = band (a (.virt 17247)) (a (.virt 17249)) := by
  have c467 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row6 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_19
  simp only [← c467, ← c468, ← c469] at e_6_19
  have hr := e_6_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 8 3) = band (a (.virt 17251)) (a (.virt 17253)) := by
  have c470 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c470, ← c471, ← c472] at e_8_0
  have hr := e_8_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 8 7) = band (a (.wire 6 79)) (a (.wire 8 3)) := by
  have c473 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c473, ← c474, ← c475] at e_8_1
  have hr := e_8_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1205)) (a (.virt 200)) (a (.virt 17255)) (a (.virt 17256)) := by
  have c476 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (publicBatchWrapper86_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (publicBatchWrapper86_copies15 a h).1
  have c481 := (publicBatchWrapper86_copies15 a h).2.1
  have c482 := (publicBatchWrapper86_copies15 a h).2.2.1
  have c483 := (publicBatchWrapper86_copies15 a h).2.2.2.1
  have c484 := (publicBatchWrapper86_copies15 a h).2.2.2.2.1
  have c485 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.1
  have c486 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.1
  have c487 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_7_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c476, k3, ← c477, k3, ← c478] at e_7_8
  have e_7_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c485, ← c486, k3, ← c487] at e_7_9
  have e_8_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c479, ← c480, ← c481] at e_8_2
  have e_8_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c482, ← c483, ← c484] at e_8_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_2
    linear_combination c488.trans k2 - hc
  · have hc := e_7_9
    simp only [e_7_8, e_8_3] at hc
    linear_combination c489.trans k2 - hc

theorem publicBatchWrapper86_f129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1206)) (a (.virt 200)) (a (.virt 17257)) (a (.virt 17258)) := by
  have c490 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_7_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c490, k3, ← c491, k3, ← c492] at e_7_10
  have e_7_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c499, ← c500, k3, ← c501] at e_7_11
  have e_8_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c493, ← c494, ← c495] at e_8_4
  have e_8_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c496, ← c497, ← c498] at e_8_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_4
    linear_combination c502.trans k2 - hc
  · have hc := e_7_11
    simp only [e_7_10, e_8_5] at hc
    linear_combination c503.trans k2 - hc

theorem publicBatchWrapper86_f130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1207)) (a (.virt 200)) (a (.virt 17259)) (a (.virt 17260)) := by
  have c504 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (publicBatchWrapper86_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (publicBatchWrapper86_copies16 a h).1
  have c513 := (publicBatchWrapper86_copies16 a h).2.1
  have c514 := (publicBatchWrapper86_copies16 a h).2.2.1
  have c515 := (publicBatchWrapper86_copies16 a h).2.2.2.1
  have c516 := (publicBatchWrapper86_copies16 a h).2.2.2.2.1
  have c517 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_7_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c504, k3, ← c505, k3, ← c506] at e_7_12
  have e_7_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c513, ← c514, k3, ← c515] at e_7_13
  have e_8_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c507, ← c508, ← c509] at e_8_6
  have e_8_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c510, ← c511, ← c512] at e_8_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_6
    linear_combination c516.trans k2 - hc
  · have hc := e_7_13
    simp only [e_7_12, e_8_7] at hc
    linear_combination c517.trans k2 - hc

theorem publicBatchWrapper86_f131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1208)) (a (.virt 200)) (a (.virt 17261)) (a (.virt 17262)) := by
  have c518 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.1
  have c519 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_7_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c518, k3, ← c519, k3, ← c520] at e_7_14
  have e_7_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_15
  simp only [← c527, ← c528, k3, ← c529] at e_7_15
  have e_8_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c521, ← c522, ← c523] at e_8_8
  have e_8_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c524, ← c525, ← c526] at e_8_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_8
    linear_combination c530.trans k2 - hc
  · have hc := e_7_15
    simp only [e_7_14, e_8_9] at hc
    linear_combination c531.trans k2 - hc

theorem publicBatchWrapper86_f132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 8 43) = band (a (.virt 17255)) (a (.virt 17257)) := by
  have c532 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c532, ← c533, ← c534] at e_8_10
  have hr := e_8_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 8 47) = band (a (.virt 17259)) (a (.virt 17261)) := by
  have c535 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c535, ← c536, ← c537] at e_8_11
  have hr := e_8_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 8 51) = band (a (.wire 8 43)) (a (.wire 8 47)) := by
  have c538 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c538, ← c539, ← c540] at e_8_12
  have hr := e_8_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1405)) (a (.virt 200)) (a (.virt 17263)) (a (.virt 17264)) := by
  have c541 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (publicBatchWrapper86_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (publicBatchWrapper86_copies17 a h).1
  have c545 := (publicBatchWrapper86_copies17 a h).2.1
  have c546 := (publicBatchWrapper86_copies17 a h).2.2.1
  have c547 := (publicBatchWrapper86_copies17 a h).2.2.2.1
  have c548 := (publicBatchWrapper86_copies17 a h).2.2.2.2.1
  have c549 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.1
  have c550 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.1
  have c551 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_7_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_16
  simp only [← c541, k3, ← c542, k3, ← c543] at e_7_16
  have e_7_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_17
  simp only [← c550, ← c551, k3, ← c552] at e_7_17
  have e_8_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c544, ← c545, ← c546] at e_8_13
  have e_8_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c547, ← c548, ← c549] at e_8_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_13
    linear_combination c553.trans k2 - hc
  · have hc := e_7_17
    simp only [e_7_16, e_8_14] at hc
    linear_combination c554.trans k2 - hc

theorem publicBatchWrapper86_f136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1406)) (a (.virt 200)) (a (.virt 17265)) (a (.virt 17266)) := by
  have c555 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_7_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_18
  simp only [← c555, k3, ← c556, k3, ← c557] at e_7_18
  have e_7_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row7 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_19
  simp only [← c564, ← c565, k3, ← c566] at e_7_19
  have e_8_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_15
  simp only [← c558, ← c559, ← c560] at e_8_15
  have e_8_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_16
  simp only [← c561, ← c562, ← c563] at e_8_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_15
    linear_combination c567.trans k2 - hc
  · have hc := e_7_19
    simp only [e_7_18, e_8_16] at hc
    linear_combination c568.trans k2 - hc

theorem publicBatchWrapper86_f137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1407)) (a (.virt 200)) (a (.virt 17267)) (a (.virt 17268)) := by
  have c569 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (publicBatchWrapper86_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (publicBatchWrapper86_copies18 a h).1
  have c577 := (publicBatchWrapper86_copies18 a h).2.1
  have c578 := (publicBatchWrapper86_copies18 a h).2.2.1
  have c579 := (publicBatchWrapper86_copies18 a h).2.2.2.1
  have c580 := (publicBatchWrapper86_copies18 a h).2.2.2.2.1
  have c581 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.1
  have c582 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_8_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_17
  simp only [← c572, ← c573, ← c574] at e_8_17
  have e_8_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_18
  simp only [← c575, ← c576, ← c577] at e_8_18
  have e_9_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c569, k3, ← c570, k3, ← c571] at e_9_0
  have e_9_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c578, ← c579, k3, ← c580] at e_9_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_17
    linear_combination c581.trans k2 - hc
  · have hc := e_9_1
    simp only [e_9_0, e_8_18] at hc
    linear_combination c582.trans k2 - hc

theorem publicBatchWrapper86_f138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1408)) (a (.virt 200)) (a (.virt 17269)) (a (.virt 17270)) := by
  have c583 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_8_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row8 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_19
  simp only [← c586, ← c587, ← c588] at e_8_19
  have e_9_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c583, k3, ← c584, k3, ← c585] at e_9_2
  have e_9_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c592, ← c593, k3, ← c594] at e_9_3
  have e_10_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c589, ← c590, ← c591] at e_10_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_19
    linear_combination c595.trans k2 - hc
  · have hc := e_9_3
    simp only [e_9_2, e_10_0] at hc
    linear_combination c596.trans k2 - hc

theorem publicBatchWrapper86_f139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 10 7) = band (a (.virt 17263)) (a (.virt 17265)) := by
  have c597 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c597, ← c598, ← c599] at e_10_1
  have hr := e_10_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 10 11) = band (a (.virt 17267)) (a (.virt 17269)) := by
  have c600 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c600, ← c601, ← c602] at e_10_2
  have hr := e_10_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 10 15) = band (a (.wire 10 7)) (a (.wire 10 11)) := by
  have c603 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c603, ← c604, ← c605] at e_10_3
  have hr := e_10_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1605)) (a (.virt 200)) (a (.virt 17271)) (a (.virt 17272)) := by
  have c606 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (publicBatchWrapper86_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (publicBatchWrapper86_copies19 a h).1
  have c609 := (publicBatchWrapper86_copies19 a h).2.1
  have c610 := (publicBatchWrapper86_copies19 a h).2.2.1
  have c611 := (publicBatchWrapper86_copies19 a h).2.2.2.1
  have c612 := (publicBatchWrapper86_copies19 a h).2.2.2.2.1
  have c613 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.1
  have c614 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.1
  have c615 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c606, k3, ← c607, k3, ← c608] at e_9_4
  have e_9_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c615, ← c616, k3, ← c617] at e_9_5
  have e_10_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c609, ← c610, ← c611] at e_10_4
  have e_10_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c612, ← c613, ← c614] at e_10_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_4
    linear_combination c618.trans k2 - hc
  · have hc := e_9_5
    simp only [e_9_4, e_10_5] at hc
    linear_combination c619.trans k2 - hc

theorem publicBatchWrapper86_f143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1606)) (a (.virt 200)) (a (.virt 17273)) (a (.virt 17274)) := by
  have c620 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c620, k3, ← c621, k3, ← c622] at e_9_6
  have e_9_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c629, ← c630, k3, ← c631] at e_9_7
  have e_10_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c623, ← c624, ← c625] at e_10_6
  have e_10_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c626, ← c627, ← c628] at e_10_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_6
    linear_combination c632.trans k2 - hc
  · have hc := e_9_7
    simp only [e_9_6, e_10_7] at hc
    linear_combination c633.trans k2 - hc

theorem publicBatchWrapper86_f144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1607)) (a (.virt 200)) (a (.virt 17275)) (a (.virt 17276)) := by
  have c634 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (publicBatchWrapper86_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (publicBatchWrapper86_copies20 a h).1
  have c641 := (publicBatchWrapper86_copies20 a h).2.1
  have c642 := (publicBatchWrapper86_copies20 a h).2.2.1
  have c643 := (publicBatchWrapper86_copies20 a h).2.2.2.1
  have c644 := (publicBatchWrapper86_copies20 a h).2.2.2.2.1
  have c645 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.1
  have c646 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.1
  have c647 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c634, k3, ← c635, k3, ← c636] at e_9_8
  have e_9_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c643, ← c644, k3, ← c645] at e_9_9
  have e_10_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c637, ← c638, ← c639] at e_10_8
  have e_10_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c640, ← c641, ← c642] at e_10_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_8
    linear_combination c646.trans k2 - hc
  · have hc := e_9_9
    simp only [e_9_8, e_10_9] at hc
    linear_combination c647.trans k2 - hc

theorem publicBatchWrapper86_f145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1608)) (a (.virt 200)) (a (.virt 17277)) (a (.virt 17278)) := by
  have c648 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c648, k3, ← c649, k3, ← c650] at e_9_10
  have e_9_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c657, ← c658, k3, ← c659] at e_9_11
  have e_10_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c651, ← c652, ← c653] at e_10_10
  have e_10_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c654, ← c655, ← c656] at e_10_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_10
    linear_combination c660.trans k2 - hc
  · have hc := e_9_11
    simp only [e_9_10, e_10_11] at hc
    linear_combination c661.trans k2 - hc

theorem publicBatchWrapper86_f146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 10 51) = band (a (.virt 17271)) (a (.virt 17273)) := by
  have c662 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c662, ← c663, ← c664] at e_10_12
  have hr := e_10_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 10 55) = band (a (.virt 17275)) (a (.virt 17277)) := by
  have c665 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c665, ← c666, ← c667] at e_10_13
  have hr := e_10_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 10 59) = band (a (.wire 10 51)) (a (.wire 10 55)) := by
  have c668 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c668, ← c669, ← c670] at e_10_14
  have hr := e_10_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1805)) (a (.virt 200)) (a (.virt 17279)) (a (.virt 17280)) := by
  have c671 := (publicBatchWrapper86_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (publicBatchWrapper86_copies21 a h).1
  have c673 := (publicBatchWrapper86_copies21 a h).2.1
  have c674 := (publicBatchWrapper86_copies21 a h).2.2.1
  have c675 := (publicBatchWrapper86_copies21 a h).2.2.2.1
  have c676 := (publicBatchWrapper86_copies21 a h).2.2.2.2.1
  have c677 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.1
  have c678 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.1
  have c679 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c671, k3, ← c672, k3, ← c673] at e_9_12
  have e_9_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c680, ← c681, k3, ← c682] at e_9_13
  have e_10_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_15
  simp only [← c674, ← c675, ← c676] at e_10_15
  have e_10_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_16
  simp only [← c677, ← c678, ← c679] at e_10_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_15
    linear_combination c683.trans k2 - hc
  · have hc := e_9_13
    simp only [e_9_12, e_10_16] at hc
    linear_combination c684.trans k2 - hc

theorem publicBatchWrapper86_f150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1806)) (a (.virt 200)) (a (.virt 17281)) (a (.virt 17282)) := by
  have c685 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c685, k3, ← c686, k3, ← c687] at e_9_14
  have e_9_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_15
  simp only [← c694, ← c695, k3, ← c696] at e_9_15
  have e_10_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_17
  simp only [← c688, ← c689, ← c690] at e_10_17
  have e_10_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_18
  simp only [← c691, ← c692, ← c693] at e_10_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_17
    linear_combination c697.trans k2 - hc
  · have hc := e_9_15
    simp only [e_9_14, e_10_18] at hc
    linear_combination c698.trans k2 - hc

theorem publicBatchWrapper86_f151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1807)) (a (.virt 200)) (a (.virt 17283)) (a (.virt 17284)) := by
  have c699 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (publicBatchWrapper86_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (publicBatchWrapper86_copies22 a h).1
  have c705 := (publicBatchWrapper86_copies22 a h).2.1
  have c706 := (publicBatchWrapper86_copies22 a h).2.2.1
  have c707 := (publicBatchWrapper86_copies22 a h).2.2.2.1
  have c708 := (publicBatchWrapper86_copies22 a h).2.2.2.2.1
  have c709 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.1
  have c710 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.1
  have c711 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_16
  simp only [← c699, k3, ← c700, k3, ← c701] at e_9_16
  have e_9_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_17
  simp only [← c708, ← c709, k3, ← c710] at e_9_17
  have e_10_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row10 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_19
  simp only [← c702, ← c703, ← c704] at e_10_19
  have e_11_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c705, ← c706, ← c707] at e_11_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_19
    linear_combination c711.trans k2 - hc
  · have hc := e_9_17
    simp only [e_9_16, e_11_0] at hc
    linear_combination c712.trans k2 - hc

theorem publicBatchWrapper86_f152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 1808)) (a (.virt 200)) (a (.virt 17285)) (a (.virt 17286)) := by
  have c713 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_9_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_18
  simp only [← c713, k3, ← c714, k3, ← c715] at e_9_18
  have e_9_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row9 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_19
  simp only [← c722, ← c723, k3, ← c724] at e_9_19
  have e_11_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c716, ← c717, ← c718] at e_11_1
  have e_11_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c719, ← c720, ← c721] at e_11_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_1
    linear_combination c725.trans k2 - hc
  · have hc := e_9_19
    simp only [e_9_18, e_11_2] at hc
    linear_combination c726.trans k2 - hc

theorem publicBatchWrapper86_f153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 11 15) = band (a (.virt 17279)) (a (.virt 17281)) := by
  have c727 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c727, ← c728, ← c729] at e_11_3
  have hr := e_11_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 11 19) = band (a (.virt 17283)) (a (.virt 17285)) := by
  have c730 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c730, ← c731, ← c732] at e_11_4
  have hr := e_11_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 11 23) = band (a (.wire 11 15)) (a (.wire 11 19)) := by
  have c733 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (publicBatchWrapper86_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_11_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c733, ← c734, ← c735] at e_11_5
  have hr := e_11_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2005)) (a (.virt 200)) (a (.virt 17287)) (a (.virt 17288)) := by
  have c736 := (publicBatchWrapper86_copies23 a h).1
  have c737 := (publicBatchWrapper86_copies23 a h).2.1
  have c738 := (publicBatchWrapper86_copies23 a h).2.2.1
  have c739 := (publicBatchWrapper86_copies23 a h).2.2.2.1
  have c740 := (publicBatchWrapper86_copies23 a h).2.2.2.2.1
  have c741 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.1
  have c742 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.1
  have c743 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.1
  have c745 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_11_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c739, ← c740, ← c741] at e_11_6
  have e_11_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c742, ← c743, ← c744] at e_11_7
  have e_12_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c736, k3, ← c737, k3, ← c738] at e_12_0
  have e_12_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c745, ← c746, k3, ← c747] at e_12_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_6
    linear_combination c748.trans k2 - hc
  · have hc := e_12_1
    simp only [e_12_0, e_11_7] at hc
    linear_combination c749.trans k2 - hc

theorem publicBatchWrapper86_f157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2006)) (a (.virt 200)) (a (.virt 17289)) (a (.virt 17290)) := by
  have c750 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c751 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c753 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c759 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_11_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c753, ← c754, ← c755] at e_11_8
  have e_11_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c756, ← c757, ← c758] at e_11_9
  have e_12_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c750, k3, ← c751, k3, ← c752] at e_12_2
  have e_12_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c759, ← c760, k3, ← c761] at e_12_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_8
    linear_combination c762.trans k2 - hc
  · have hc := e_12_3
    simp only [e_12_2, e_11_9] at hc
    linear_combination c763.trans k2 - hc

theorem publicBatchWrapper86_f158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2007)) (a (.virt 200)) (a (.virt 17291)) (a (.virt 17292)) := by
  have c764 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (publicBatchWrapper86_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (publicBatchWrapper86_copies24 a h).1
  have c769 := (publicBatchWrapper86_copies24 a h).2.1
  have c770 := (publicBatchWrapper86_copies24 a h).2.2.1
  have c771 := (publicBatchWrapper86_copies24 a h).2.2.2.1
  have c772 := (publicBatchWrapper86_copies24 a h).2.2.2.2.1
  have c773 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.1
  have c774 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.1
  have c775 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_11_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c767, ← c768, ← c769] at e_11_10
  have e_11_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c770, ← c771, ← c772] at e_11_11
  have e_12_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c764, k3, ← c765, k3, ← c766] at e_12_4
  have e_12_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c773, ← c774, k3, ← c775] at e_12_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_10
    linear_combination c776.trans k2 - hc
  · have hc := e_12_5
    simp only [e_12_4, e_11_11] at hc
    linear_combination c777.trans k2 - hc

theorem publicBatchWrapper86_f159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2008)) (a (.virt 200)) (a (.virt 17293)) (a (.virt 17294)) := by
  have c778 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c781 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_11_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c781, ← c782, ← c783] at e_11_12
  have e_11_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c784, ← c785, ← c786] at e_11_13
  have e_12_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c778, k3, ← c779, k3, ← c780] at e_12_6
  have e_12_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c787, ← c788, k3, ← c789] at e_12_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_12
    linear_combination c790.trans k2 - hc
  · have hc := e_12_7
    simp only [e_12_6, e_11_13] at hc
    linear_combination c791.trans k2 - hc

theorem publicBatchWrapper86_f160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 11 59) = band (a (.virt 17287)) (a (.virt 17289)) := by
  have c792 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c792, ← c793, ← c794] at e_11_14
  have hr := e_11_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 11 63) = band (a (.virt 17291)) (a (.virt 17293)) := by
  have c795 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c795, ← c796, ← c797] at e_11_15
  have hr := e_11_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 11 67) = band (a (.wire 11 59)) (a (.wire 11 63)) := by
  have c798 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (publicBatchWrapper86_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (publicBatchWrapper86_copies25 a h).1
  have e_11_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c798, ← c799, ← c800] at e_11_16
  have hr := e_11_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2205)) (a (.virt 200)) (a (.virt 17295)) (a (.virt 17296)) := by
  have c801 := (publicBatchWrapper86_copies25 a h).2.1
  have c802 := (publicBatchWrapper86_copies25 a h).2.2.1
  have c803 := (publicBatchWrapper86_copies25 a h).2.2.2.1
  have c804 := (publicBatchWrapper86_copies25 a h).2.2.2.2.1
  have c805 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.1
  have c806 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.1
  have c807 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c814 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_11_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c804, ← c805, ← c806] at e_11_17
  have e_11_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c807, ← c808, ← c809] at e_11_18
  have e_12_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c801, k3, ← c802, k3, ← c803] at e_12_8
  have e_12_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c810, ← c811, k3, ← c812] at e_12_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_17
    linear_combination c813.trans k2 - hc
  · have hc := e_12_9
    simp only [e_12_8, e_11_18] at hc
    linear_combination c814.trans k2 - hc

theorem publicBatchWrapper86_f164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2206)) (a (.virt 200)) (a (.virt 17297)) (a (.virt 17298)) := by
  have c815 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_11_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row11 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c818, ← c819, ← c820] at e_11_19
  have e_12_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c815, k3, ← c816, k3, ← c817] at e_12_10
  have e_12_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c824, ← c825, k3, ← c826] at e_12_11
  have e_13_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c821, ← c822, ← c823] at e_13_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_11_19
    linear_combination c827.trans k2 - hc
  · have hc := e_12_11
    simp only [e_12_10, e_13_0] at hc
    linear_combination c828.trans k2 - hc

theorem publicBatchWrapper86_f165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2207)) (a (.virt 200)) (a (.virt 17299)) (a (.virt 17300)) := by
  have c829 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (publicBatchWrapper86_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c832 := (publicBatchWrapper86_copies26 a h).1
  have c833 := (publicBatchWrapper86_copies26 a h).2.1
  have c834 := (publicBatchWrapper86_copies26 a h).2.2.1
  have c835 := (publicBatchWrapper86_copies26 a h).2.2.2.1
  have c836 := (publicBatchWrapper86_copies26 a h).2.2.2.2.1
  have c837 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.1
  have c838 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.1
  have c839 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.1
  have c841 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have c842 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_12_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c829, k3, ← c830, k3, ← c831] at e_12_12
  have e_12_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c838, ← c839, k3, ← c840] at e_12_13
  have e_13_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c832, ← c833, ← c834] at e_13_1
  have e_13_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c835, ← c836, ← c837] at e_13_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_1
    linear_combination c841.trans k2 - hc
  · have hc := e_12_13
    simp only [e_12_12, e_13_2] at hc
    linear_combination c842.trans k2 - hc

theorem publicBatchWrapper86_f166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2208)) (a (.virt 200)) (a (.virt 17301)) (a (.virt 17302)) := by
  have c843 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c844 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c845 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c847 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c853 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_12_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c843, k3, ← c844, k3, ← c845] at e_12_14
  have e_12_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c852, ← c853, k3, ← c854] at e_12_15
  have e_13_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c846, ← c847, ← c848] at e_13_3
  have e_13_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c849, ← c850, ← c851] at e_13_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_3
    linear_combination c855.trans k2 - hc
  · have hc := e_12_15
    simp only [e_12_14, e_13_4] at hc
    linear_combination c856.trans k2 - hc

theorem publicBatchWrapper86_f167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 13 23) = band (a (.virt 17295)) (a (.virt 17297)) := by
  have c857 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c857, ← c858, ← c859] at e_13_5
  have hr := e_13_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 13 27) = band (a (.virt 17299)) (a (.virt 17301)) := by
  have c860 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c860, ← c861, ← c862] at e_13_6
  have hr := e_13_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 13 31) = band (a (.wire 13 23)) (a (.wire 13 27)) := by
  have c863 := (publicBatchWrapper86_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (publicBatchWrapper86_copies27 a h).1
  have c865 := (publicBatchWrapper86_copies27 a h).2.1
  have e_13_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c863, ← c864, ← c865] at e_13_7
  have hr := e_13_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2405)) (a (.virt 200)) (a (.virt 17303)) (a (.virt 17304)) := by
  have c866 := (publicBatchWrapper86_copies27 a h).2.2.1
  have c867 := (publicBatchWrapper86_copies27 a h).2.2.2.1
  have c868 := (publicBatchWrapper86_copies27 a h).2.2.2.2.1
  have c869 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.1
  have c870 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.1
  have c871 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.1
  have c873 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c876 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_12_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c866, k3, ← c867, k3, ← c868] at e_12_16
  have e_12_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c875, ← c876, k3, ← c877] at e_12_17
  have e_13_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c869, ← c870, ← c871] at e_13_8
  have e_13_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c872, ← c873, ← c874] at e_13_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_8
    linear_combination c878.trans k2 - hc
  · have hc := e_12_17
    simp only [e_12_16, e_13_9] at hc
    linear_combination c879.trans k2 - hc

theorem publicBatchWrapper86_f171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2406)) (a (.virt 200)) (a (.virt 17305)) (a (.virt 17306)) := by
  have c880 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c883 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_12_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c880, k3, ← c881, k3, ← c882] at e_12_18
  have e_12_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row12 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c889, ← c890, k3, ← c891] at e_12_19
  have e_13_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c883, ← c884, ← c885] at e_13_10
  have e_13_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c886, ← c887, ← c888] at e_13_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_10
    linear_combination c892.trans k2 - hc
  · have hc := e_12_19
    simp only [e_12_18, e_13_11] at hc
    linear_combination c893.trans k2 - hc

theorem publicBatchWrapper86_f172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2407)) (a (.virt 200)) (a (.virt 17307)) (a (.virt 17308)) := by
  have c894 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (publicBatchWrapper86_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (publicBatchWrapper86_copies28 a h).1
  have c897 := (publicBatchWrapper86_copies28 a h).2.1
  have c898 := (publicBatchWrapper86_copies28 a h).2.2.1
  have c899 := (publicBatchWrapper86_copies28 a h).2.2.2.1
  have c900 := (publicBatchWrapper86_copies28 a h).2.2.2.2.1
  have c901 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.1
  have c902 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.1
  have c903 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.1
  have c904 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have c907 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_13_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c897, ← c898, ← c899] at e_13_12
  have e_13_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c900, ← c901, ← c902] at e_13_13
  have e_14_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c894, k3, ← c895, k3, ← c896] at e_14_0
  have e_14_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c903, ← c904, k3, ← c905] at e_14_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_12
    linear_combination c906.trans k2 - hc
  · have hc := e_14_1
    simp only [e_14_0, e_13_13] at hc
    linear_combination c907.trans k2 - hc

theorem publicBatchWrapper86_f173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2408)) (a (.virt 200)) (a (.virt 17309)) (a (.virt 17310)) := by
  have c908 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_13_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c911, ← c912, ← c913] at e_13_14
  have e_13_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c914, ← c915, ← c916] at e_13_15
  have e_14_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c908, k3, ← c909, k3, ← c910] at e_14_2
  have e_14_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c917, ← c918, k3, ← c919] at e_14_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_14
    linear_combination c920.trans k2 - hc
  · have hc := e_14_3
    simp only [e_14_2, e_13_15] at hc
    linear_combination c921.trans k2 - hc

theorem publicBatchWrapper86_f174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 13 67) = band (a (.virt 17303)) (a (.virt 17305)) := by
  have c922 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c922, ← c923, ← c924] at e_13_16
  have hr := e_13_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 13 71) = band (a (.virt 17307)) (a (.virt 17309)) := by
  have c925 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c927 := (publicBatchWrapper86_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_13_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c925, ← c926, ← c927] at e_13_17
  have hr := e_13_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 13 75) = band (a (.wire 13 67)) (a (.wire 13 71)) := by
  have c928 := (publicBatchWrapper86_copies29 a h).1
  have c929 := (publicBatchWrapper86_copies29 a h).2.1
  have c930 := (publicBatchWrapper86_copies29 a h).2.2.1
  have e_13_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c928, ← c929, ← c930] at e_13_18
  have hr := e_13_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2605)) (a (.virt 200)) (a (.virt 17311)) (a (.virt 17312)) := by
  have c931 := (publicBatchWrapper86_copies29 a h).2.2.2.1
  have c932 := (publicBatchWrapper86_copies29 a h).2.2.2.2.1
  have c933 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.1
  have c934 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.1
  have c935 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.1
  have c937 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_13_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row13 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c934, ← c935, ← c936] at e_13_19
  have e_14_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c931, k3, ← c932, k3, ← c933] at e_14_4
  have e_14_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c940, ← c941, k3, ← c942] at e_14_5
  have e_15_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c937, ← c938, ← c939] at e_15_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_13_19
    linear_combination c943.trans k2 - hc
  · have hc := e_14_5
    simp only [e_14_4, e_15_0] at hc
    linear_combination c944.trans k2 - hc

theorem publicBatchWrapper86_f178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2606)) (a (.virt 200)) (a (.virt 17313)) (a (.virt 17314)) := by
  have c945 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c947 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c953 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c956 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c945, k3, ← c946, k3, ← c947] at e_14_6
  have e_14_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c954, ← c955, k3, ← c956] at e_14_7
  have e_15_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c948, ← c949, ← c950] at e_15_1
  have e_15_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c951, ← c952, ← c953] at e_15_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_1
    linear_combination c957.trans k2 - hc
  · have hc := e_14_7
    simp only [e_14_6, e_15_2] at hc
    linear_combination c958.trans k2 - hc

theorem publicBatchWrapper86_f179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2607)) (a (.virt 200)) (a (.virt 17315)) (a (.virt 17316)) := by
  have c959 := (publicBatchWrapper86_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (publicBatchWrapper86_copies30 a h).1
  have c961 := (publicBatchWrapper86_copies30 a h).2.1
  have c962 := (publicBatchWrapper86_copies30 a h).2.2.1
  have c963 := (publicBatchWrapper86_copies30 a h).2.2.2.1
  have c964 := (publicBatchWrapper86_copies30 a h).2.2.2.2.1
  have c965 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.1
  have c966 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.1
  have c967 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  have c971 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c959, k3, ← c960, k3, ← c961] at e_14_8
  have e_14_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c968, ← c969, k3, ← c970] at e_14_9
  have e_15_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c962, ← c963, ← c964] at e_15_3
  have e_15_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c965, ← c966, ← c967] at e_15_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_3
    linear_combination c971.trans k2 - hc
  · have hc := e_14_9
    simp only [e_14_8, e_15_4] at hc
    linear_combination c972.trans k2 - hc

theorem publicBatchWrapper86_f180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2608)) (a (.virt 200)) (a (.virt 17317)) (a (.virt 17318)) := by
  have c973 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c977 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c973, k3, ← c974, k3, ← c975] at e_14_10
  have e_14_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c982, ← c983, k3, ← c984] at e_14_11
  have e_15_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c976, ← c977, ← c978] at e_15_5
  have e_15_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c979, ← c980, ← c981] at e_15_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_5
    linear_combination c985.trans k2 - hc
  · have hc := e_14_11
    simp only [e_14_10, e_15_6] at hc
    linear_combination c986.trans k2 - hc

theorem publicBatchWrapper86_f181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 15 31) = band (a (.virt 17311)) (a (.virt 17313)) := by
  have c987 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c989 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_7
  simp only [← c987, ← c988, ← c989] at e_15_7
  have hr := e_15_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 15 35) = band (a (.virt 17315)) (a (.virt 17317)) := by
  have c990 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (publicBatchWrapper86_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (publicBatchWrapper86_copies31 a h).1
  have e_15_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_8
  simp only [← c990, ← c991, ← c992] at e_15_8
  have hr := e_15_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 15 39) = band (a (.wire 15 31)) (a (.wire 15 35)) := by
  have c993 := (publicBatchWrapper86_copies31 a h).2.1
  have c994 := (publicBatchWrapper86_copies31 a h).2.2.1
  have c995 := (publicBatchWrapper86_copies31 a h).2.2.2.1
  have e_15_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_9
  simp only [← c993, ← c994, ← c995] at e_15_9
  have hr := e_15_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2805)) (a (.virt 200)) (a (.virt 17319)) (a (.virt 17320)) := by
  have c996 := (publicBatchWrapper86_copies31 a h).2.2.2.2.1
  have c997 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.1
  have c998 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.1
  have c999 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have c1002 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1005 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1008 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c996, k3, ← c997, k3, ← c998] at e_14_12
  have e_14_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c1005, ← c1006, k3, ← c1007] at e_14_13
  have e_15_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_10
  simp only [← c999, ← c1000, ← c1001] at e_15_10
  have e_15_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_11
  simp only [← c1002, ← c1003, ← c1004] at e_15_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_10
    linear_combination c1008.trans k2 - hc
  · have hc := e_14_13
    simp only [e_14_12, e_15_11] at hc
    linear_combination c1009.trans k2 - hc

theorem publicBatchWrapper86_f185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2806)) (a (.virt 200)) (a (.virt 17321)) (a (.virt 17322)) := by
  have c1010 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1011 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1012 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1017 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1019 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1022 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1023 := (publicBatchWrapper86_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c1010, k3, ← c1011, k3, ← c1012] at e_14_14
  have e_14_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_15
  simp only [← c1019, ← c1020, k3, ← c1021] at e_14_15
  have e_15_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_12
  simp only [← c1013, ← c1014, ← c1015] at e_15_12
  have e_15_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_13
  simp only [← c1016, ← c1017, ← c1018] at e_15_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_12
    linear_combination c1022.trans k2 - hc
  · have hc := e_14_15
    simp only [e_14_14, e_15_13] at hc
    linear_combination c1023.trans k2 - hc

theorem publicBatchWrapper86_f186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2807)) (a (.virt 200)) (a (.virt 17323)) (a (.virt 17324)) := by
  have c1024 := (publicBatchWrapper86_copies32 a h).1
  have c1025 := (publicBatchWrapper86_copies32 a h).2.1
  have c1026 := (publicBatchWrapper86_copies32 a h).2.2.1
  have c1027 := (publicBatchWrapper86_copies32 a h).2.2.2.1
  have c1028 := (publicBatchWrapper86_copies32 a h).2.2.2.2.1
  have c1029 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.1
  have c1030 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_16
  simp only [← c1024, k3, ← c1025, k3, ← c1026] at e_14_16
  have e_14_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_17
  simp only [← c1033, ← c1034, k3, ← c1035] at e_14_17
  have e_15_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_14
  simp only [← c1027, ← c1028, ← c1029] at e_15_14
  have e_15_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_15
  simp only [← c1030, ← c1031, ← c1032] at e_15_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_14
    linear_combination c1036.trans k2 - hc
  · have hc := e_14_17
    simp only [e_14_16, e_15_15] at hc
    linear_combination c1037.trans k2 - hc

theorem publicBatchWrapper86_f187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 2808)) (a (.virt 200)) (a (.virt 17325)) (a (.virt 17326)) := by
  have c1038 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1041 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_14_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_18
  simp only [← c1038, k3, ← c1039, k3, ← c1040] at e_14_18
  have e_14_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row14 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_19
  simp only [← c1047, ← c1048, k3, ← c1049] at e_14_19
  have e_15_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_16
  simp only [← c1041, ← c1042, ← c1043] at e_15_16
  have e_15_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_17
  simp only [← c1044, ← c1045, ← c1046] at e_15_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_16
    linear_combination c1050.trans k2 - hc
  · have hc := e_14_19
    simp only [e_14_18, e_15_17] at hc
    linear_combination c1051.trans k2 - hc

theorem publicBatchWrapper86_f188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 15 75) = band (a (.virt 17319)) (a (.virt 17321)) := by
  have c1052 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_18
  simp only [← c1052, ← c1053, ← c1054] at e_15_18
  have hr := e_15_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 15 79) = band (a (.virt 17323)) (a (.virt 17325)) := by
  have c1055 := (publicBatchWrapper86_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (publicBatchWrapper86_copies33 a h).1
  have c1057 := (publicBatchWrapper86_copies33 a h).2.1
  have e_15_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row15 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_19
  simp only [← c1055, ← c1056, ← c1057] at e_15_19
  have hr := e_15_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 16 3) = band (a (.wire 15 75)) (a (.wire 15 79)) := by
  have c1058 := (publicBatchWrapper86_copies33 a h).2.2.1
  have c1059 := (publicBatchWrapper86_copies33 a h).2.2.2.1
  have c1060 := (publicBatchWrapper86_copies33 a h).2.2.2.2.1
  have e_16_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c1058, ← c1059, ← c1060] at e_16_0
  have hr := e_16_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3005)) (a (.virt 200)) (a (.virt 17327)) (a (.virt 17328)) := by
  have c1061 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.1
  have c1062 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c1064, ← c1065, ← c1066] at e_16_1
  have e_16_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c1067, ← c1068, ← c1069] at e_16_2
  have e_17_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c1061, k3, ← c1062, k3, ← c1063] at e_17_0
  have e_17_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c1070, ← c1071, k3, ← c1072] at e_17_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_1
    linear_combination c1073.trans k2 - hc
  · have hc := e_17_1
    simp only [e_17_0, e_16_2] at hc
    linear_combination c1074.trans k2 - hc

theorem publicBatchWrapper86_f192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3006)) (a (.virt 200)) (a (.virt 17329)) (a (.virt 17330)) := by
  have c1075 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1078 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (publicBatchWrapper86_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (publicBatchWrapper86_copies34 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c1078, ← c1079, ← c1080] at e_16_3
  have e_16_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c1081, ← c1082, ← c1083] at e_16_4
  have e_17_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c1075, k3, ← c1076, k3, ← c1077] at e_17_2
  have e_17_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c1084, ← c1085, k3, ← c1086] at e_17_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_3
    linear_combination c1087.trans k2 - hc
  · have hc := e_17_3
    simp only [e_17_2, e_16_4] at hc
    linear_combination c1088.trans k2 - hc

theorem publicBatchWrapper86_f193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3007)) (a (.virt 200)) (a (.virt 17331)) (a (.virt 17332)) := by
  have c1089 := (publicBatchWrapper86_copies34 a h).2.1
  have c1090 := (publicBatchWrapper86_copies34 a h).2.2.1
  have c1091 := (publicBatchWrapper86_copies34 a h).2.2.2.1
  have c1092 := (publicBatchWrapper86_copies34 a h).2.2.2.2.1
  have c1093 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.1
  have c1094 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.1
  have c1096 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1099 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1102 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c1092, ← c1093, ← c1094] at e_16_5
  have e_16_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c1095, ← c1096, ← c1097] at e_16_6
  have e_17_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c1089, k3, ← c1090, k3, ← c1091] at e_17_4
  have e_17_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c1098, ← c1099, k3, ← c1100] at e_17_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_5
    linear_combination c1101.trans k2 - hc
  · have hc := e_17_5
    simp only [e_17_4, e_16_6] at hc
    linear_combination c1102.trans k2 - hc

theorem publicBatchWrapper86_f194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3008)) (a (.virt 200)) (a (.virt 17333)) (a (.virt 17334)) := by
  have c1103 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1106 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1107 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1108 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c1106, ← c1107, ← c1108] at e_16_7
  have e_16_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c1109, ← c1110, ← c1111] at e_16_8
  have e_17_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c1103, k3, ← c1104, k3, ← c1105] at e_17_6
  have e_17_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c1112, ← c1113, k3, ← c1114] at e_17_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_7
    linear_combination c1115.trans k2 - hc
  · have hc := e_17_7
    simp only [e_17_6, e_16_8] at hc
    linear_combination c1116.trans k2 - hc

theorem publicBatchWrapper86_f195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 16 39) = band (a (.virt 17327)) (a (.virt 17329)) := by
  have c1117 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (publicBatchWrapper86_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_16_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c1117, ← c1118, ← c1119] at e_16_9
  have hr := e_16_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 16 43) = band (a (.virt 17331)) (a (.virt 17333)) := by
  have c1120 := (publicBatchWrapper86_copies35 a h).1
  have c1121 := (publicBatchWrapper86_copies35 a h).2.1
  have c1122 := (publicBatchWrapper86_copies35 a h).2.2.1
  have e_16_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c1120, ← c1121, ← c1122] at e_16_10
  have hr := e_16_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 16 47) = band (a (.wire 16 39)) (a (.wire 16 43)) := by
  have c1123 := (publicBatchWrapper86_copies35 a h).2.2.2.1
  have c1124 := (publicBatchWrapper86_copies35 a h).2.2.2.2.1
  have c1125 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.1
  have e_16_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1123, ← c1124, ← c1125] at e_16_11
  have hr := e_16_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3205)) (a (.virt 200)) (a (.virt 17335)) (a (.virt 17336)) := by
  have c1126 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1129, ← c1130, ← c1131] at e_16_12
  have e_16_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1132, ← c1133, ← c1134] at e_16_13
  have e_17_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c1126, k3, ← c1127, k3, ← c1128] at e_17_8
  have e_17_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c1135, ← c1136, k3, ← c1137] at e_17_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_12
    linear_combination c1138.trans k2 - hc
  · have hc := e_17_9
    simp only [e_17_8, e_16_13] at hc
    linear_combination c1139.trans k2 - hc

theorem publicBatchWrapper86_f199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3206)) (a (.virt 200)) (a (.virt 17337)) (a (.virt 17338)) := by
  have c1140 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1143 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1150 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (publicBatchWrapper86_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (publicBatchWrapper86_copies36 a h).1
  have c1153 := (publicBatchWrapper86_copies36 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1143, ← c1144, ← c1145] at e_16_14
  have e_16_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_15
  simp only [← c1146, ← c1147, ← c1148] at e_16_15
  have e_17_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c1140, k3, ← c1141, k3, ← c1142] at e_17_10
  have e_17_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c1149, ← c1150, k3, ← c1151] at e_17_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_14
    linear_combination c1152.trans k2 - hc
  · have hc := e_17_11
    simp only [e_17_10, e_16_15] at hc
    linear_combination c1153.trans k2 - hc

theorem publicBatchWrapper86_f200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3207)) (a (.virt 200)) (a (.virt 17339)) (a (.virt 17340)) := by
  have c1154 := (publicBatchWrapper86_copies36 a h).2.2.1
  have c1155 := (publicBatchWrapper86_copies36 a h).2.2.2.1
  have c1156 := (publicBatchWrapper86_copies36 a h).2.2.2.2.1
  have c1157 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.1
  have c1158 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.1
  have c1159 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have c1162 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_16
  simp only [← c1157, ← c1158, ← c1159] at e_16_16
  have e_16_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_17
  simp only [← c1160, ← c1161, ← c1162] at e_16_17
  have e_17_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c1154, k3, ← c1155, k3, ← c1156] at e_17_12
  have e_17_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c1163, ← c1164, k3, ← c1165] at e_17_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_16
    linear_combination c1166.trans k2 - hc
  · have hc := e_17_13
    simp only [e_17_12, e_16_17] at hc
    linear_combination c1167.trans k2 - hc

theorem publicBatchWrapper86_f201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3208)) (a (.virt 200)) (a (.virt 17341)) (a (.virt 17342)) := by
  have c1168 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1174 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1177 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1179 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1180 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_16_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_18
  simp only [← c1171, ← c1172, ← c1173] at e_16_18
  have e_16_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row16 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_19
  simp only [← c1174, ← c1175, ← c1176] at e_16_19
  have e_17_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c1168, k3, ← c1169, k3, ← c1170] at e_17_14
  have e_17_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_15
  simp only [← c1177, ← c1178, k3, ← c1179] at e_17_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_16_18
    linear_combination c1180.trans k2 - hc
  · have hc := e_17_15
    simp only [e_17_14, e_16_19] at hc
    linear_combination c1181.trans k2 - hc

theorem publicBatchWrapper86_f202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 18 3) = band (a (.virt 17335)) (a (.virt 17337)) := by
  have c1182 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1183 := (publicBatchWrapper86_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (publicBatchWrapper86_copies37 a h).1
  have e_18_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c1182, ← c1183, ← c1184] at e_18_0
  have hr := e_18_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 18 7) = band (a (.virt 17339)) (a (.virt 17341)) := by
  have c1185 := (publicBatchWrapper86_copies37 a h).2.1
  have c1186 := (publicBatchWrapper86_copies37 a h).2.2.1
  have c1187 := (publicBatchWrapper86_copies37 a h).2.2.2.1
  have e_18_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c1185, ← c1186, ← c1187] at e_18_1
  have hr := e_18_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 18 11) = band (a (.wire 18 3)) (a (.wire 18 7)) := by
  have c1188 := (publicBatchWrapper86_copies37 a h).2.2.2.2.1
  have c1189 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.1
  have c1190 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.1
  have e_18_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c1188, ← c1189, ← c1190] at e_18_2
  have hr := e_18_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3405)) (a (.virt 200)) (a (.virt 17343)) (a (.virt 17344)) := by
  have c1191 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_17_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_16
  simp only [← c1191, k3, ← c1192, k3, ← c1193] at e_17_16
  have e_17_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_17
  simp only [← c1200, ← c1201, k3, ← c1202] at e_17_17
  have e_18_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c1194, ← c1195, ← c1196] at e_18_3
  have e_18_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c1197, ← c1198, ← c1199] at e_18_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_3
    linear_combination c1203.trans k2 - hc
  · have hc := e_17_17
    simp only [e_17_16, e_18_4] at hc
    linear_combination c1204.trans k2 - hc

theorem publicBatchWrapper86_f206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3406)) (a (.virt 200)) (a (.virt 17345)) (a (.virt 17346)) := by
  have c1205 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1207 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (publicBatchWrapper86_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (publicBatchWrapper86_copies38 a h).1
  have c1217 := (publicBatchWrapper86_copies38 a h).2.1
  have c1218 := (publicBatchWrapper86_copies38 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_17_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_18
  simp only [← c1205, k3, ← c1206, k3, ← c1207] at e_17_18
  have e_17_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row17 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_19
  simp only [← c1214, ← c1215, k3, ← c1216] at e_17_19
  have e_18_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c1208, ← c1209, ← c1210] at e_18_5
  have e_18_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c1211, ← c1212, ← c1213] at e_18_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_5
    linear_combination c1217.trans k2 - hc
  · have hc := e_17_19
    simp only [e_17_18, e_18_6] at hc
    linear_combination c1218.trans k2 - hc

theorem publicBatchWrapper86_f207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3407)) (a (.virt 200)) (a (.virt 17347)) (a (.virt 17348)) := by
  have c1219 := (publicBatchWrapper86_copies38 a h).2.2.2.1
  have c1220 := (publicBatchWrapper86_copies38 a h).2.2.2.2.1
  have c1221 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.1
  have c1222 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.1
  have c1224 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1227 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_18_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c1222, ← c1223, ← c1224] at e_18_7
  have e_18_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c1225, ← c1226, ← c1227] at e_18_8
  have e_19_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c1219, k3, ← c1220, k3, ← c1221] at e_19_0
  have e_19_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c1228, ← c1229, k3, ← c1230] at e_19_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_7
    linear_combination c1231.trans k2 - hc
  · have hc := e_19_1
    simp only [e_19_0, e_18_8] at hc
    linear_combination c1232.trans k2 - hc

theorem publicBatchWrapper86_f208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3408)) (a (.virt 200)) (a (.virt 17349)) (a (.virt 17350)) := by
  have c1233 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1236 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1239 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1241 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1245 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_18_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c1236, ← c1237, ← c1238] at e_18_9
  have e_18_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c1239, ← c1240, ← c1241] at e_18_10
  have e_19_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c1233, k3, ← c1234, k3, ← c1235] at e_19_2
  have e_19_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c1242, ← c1243, k3, ← c1244] at e_19_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_9
    linear_combination c1245.trans k2 - hc
  · have hc := e_19_3
    simp only [e_19_2, e_18_10] at hc
    linear_combination c1246.trans k2 - hc

theorem publicBatchWrapper86_f209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 18 47) = band (a (.virt 17343)) (a (.virt 17345)) := by
  have c1247 := (publicBatchWrapper86_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (publicBatchWrapper86_copies39 a h).1
  have c1249 := (publicBatchWrapper86_copies39 a h).2.1
  have e_18_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c1247, ← c1248, ← c1249] at e_18_11
  have hr := e_18_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 18 51) = band (a (.virt 17347)) (a (.virt 17349)) := by
  have c1250 := (publicBatchWrapper86_copies39 a h).2.2.1
  have c1251 := (publicBatchWrapper86_copies39 a h).2.2.2.1
  have c1252 := (publicBatchWrapper86_copies39 a h).2.2.2.2.1
  have e_18_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c1250, ← c1251, ← c1252] at e_18_12
  have hr := e_18_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 18 55) = band (a (.wire 18 47)) (a (.wire 18 51)) := by
  have c1253 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.1
  have c1254 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.1
  have e_18_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c1253, ← c1254, ← c1255] at e_18_13
  have hr := e_18_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3605)) (a (.virt 200)) (a (.virt 17351)) (a (.virt 17352)) := by
  have c1256 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have c1258 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1261 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1264 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1267 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_18_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c1259, ← c1260, ← c1261] at e_18_14
  have e_18_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_15
  simp only [← c1262, ← c1263, ← c1264] at e_18_15
  have e_19_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c1256, k3, ← c1257, k3, ← c1258] at e_19_4
  have e_19_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c1265, ← c1266, k3, ← c1267] at e_19_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_14
    linear_combination c1268.trans k2 - hc
  · have hc := e_19_5
    simp only [e_19_4, e_18_15] at hc
    linear_combination c1269.trans k2 - hc

theorem publicBatchWrapper86_f213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3606)) (a (.virt 200)) (a (.virt 17353)) (a (.virt 17354)) := by
  have c1270 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1271 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1274 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (publicBatchWrapper86_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (publicBatchWrapper86_copies40 a h).1
  have c1281 := (publicBatchWrapper86_copies40 a h).2.1
  have c1282 := (publicBatchWrapper86_copies40 a h).2.2.1
  have c1283 := (publicBatchWrapper86_copies40 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_18_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_16
  simp only [← c1273, ← c1274, ← c1275] at e_18_16
  have e_18_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_17
  simp only [← c1276, ← c1277, ← c1278] at e_18_17
  have e_19_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c1270, k3, ← c1271, k3, ← c1272] at e_19_6
  have e_19_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c1279, ← c1280, k3, ← c1281] at e_19_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_16
    linear_combination c1282.trans k2 - hc
  · have hc := e_19_7
    simp only [e_19_6, e_18_17] at hc
    linear_combination c1283.trans k2 - hc

theorem publicBatchWrapper86_f214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3607)) (a (.virt 200)) (a (.virt 17355)) (a (.virt 17356)) := by
  have c1284 := (publicBatchWrapper86_copies40 a h).2.2.2.2.1
  have c1285 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.1
  have c1286 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.1
  have c1287 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1291 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_18_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_18
  simp only [← c1287, ← c1288, ← c1289] at e_18_18
  have e_18_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row18 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_19
  simp only [← c1290, ← c1291, ← c1292] at e_18_19
  have e_19_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c1284, k3, ← c1285, k3, ← c1286] at e_19_8
  have e_19_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c1293, ← c1294, k3, ← c1295] at e_19_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_18_18
    linear_combination c1296.trans k2 - hc
  · have hc := e_19_9
    simp only [e_19_8, e_18_19] at hc
    linear_combination c1297.trans k2 - hc

theorem publicBatchWrapper86_f215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3608)) (a (.virt 200)) (a (.virt 17357)) (a (.virt 17358)) := by
  have c1298 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1308 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (publicBatchWrapper86_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_19_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c1298, k3, ← c1299, k3, ← c1300] at e_19_10
  have e_19_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c1307, ← c1308, k3, ← c1309] at e_19_11
  have e_20_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_0
  simp only [← c1301, ← c1302, ← c1303] at e_20_0
  have e_20_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_1
  simp only [← c1304, ← c1305, ← c1306] at e_20_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_0
    linear_combination c1310.trans k2 - hc
  · have hc := e_19_11
    simp only [e_19_10, e_20_1] at hc
    linear_combination c1311.trans k2 - hc

theorem publicBatchWrapper86_f216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 20 11) = band (a (.virt 17351)) (a (.virt 17353)) := by
  have c1312 := (publicBatchWrapper86_copies41 a h).1
  have c1313 := (publicBatchWrapper86_copies41 a h).2.1
  have c1314 := (publicBatchWrapper86_copies41 a h).2.2.1
  have e_20_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_2
  simp only [← c1312, ← c1313, ← c1314] at e_20_2
  have hr := e_20_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 20 15) = band (a (.virt 17355)) (a (.virt 17357)) := by
  have c1315 := (publicBatchWrapper86_copies41 a h).2.2.2.1
  have c1316 := (publicBatchWrapper86_copies41 a h).2.2.2.2.1
  have c1317 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.1
  have e_20_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_3
  simp only [← c1315, ← c1316, ← c1317] at e_20_3
  have hr := e_20_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 20 19) = band (a (.wire 20 11)) (a (.wire 20 15)) := by
  have c1318 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.1
  have c1319 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.1
  have e_20_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_4
  simp only [← c1318, ← c1319, ← c1320] at e_20_4
  have hr := e_20_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3805)) (a (.virt 200)) (a (.virt 17359)) (a (.virt 17360)) := by
  have c1321 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have c1322 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1325 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1329 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_19_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c1321, k3, ← c1322, k3, ← c1323] at e_19_12
  have e_19_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c1330, ← c1331, k3, ← c1332] at e_19_13
  have e_20_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_5
  simp only [← c1324, ← c1325, ← c1326] at e_20_5
  have e_20_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_6
  simp only [← c1327, ← c1328, ← c1329] at e_20_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_5
    linear_combination c1333.trans k2 - hc
  · have hc := e_19_13
    simp only [e_19_12, e_20_6] at hc
    linear_combination c1334.trans k2 - hc

theorem publicBatchWrapper86_f220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3806)) (a (.virt 200)) (a (.virt 17361)) (a (.virt 17362)) := by
  have c1335 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1342 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (publicBatchWrapper86_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (publicBatchWrapper86_copies42 a h).1
  have c1345 := (publicBatchWrapper86_copies42 a h).2.1
  have c1346 := (publicBatchWrapper86_copies42 a h).2.2.1
  have c1347 := (publicBatchWrapper86_copies42 a h).2.2.2.1
  have c1348 := (publicBatchWrapper86_copies42 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_19_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c1335, k3, ← c1336, k3, ← c1337] at e_19_14
  have e_19_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_15
  simp only [← c1344, ← c1345, k3, ← c1346] at e_19_15
  have e_20_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c1338, ← c1339, ← c1340] at e_20_7
  have e_20_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_8
  simp only [← c1341, ← c1342, ← c1343] at e_20_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_7
    linear_combination c1347.trans k2 - hc
  · have hc := e_19_15
    simp only [e_19_14, e_20_8] at hc
    linear_combination c1348.trans k2 - hc

theorem publicBatchWrapper86_f221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3807)) (a (.virt 200)) (a (.virt 17363)) (a (.virt 17364)) := by
  have c1349 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.1
  have c1350 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.1
  have c1351 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.1
  have c1352 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have c1354 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1355 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1358 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1361 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1362 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_19_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_16
  simp only [← c1349, k3, ← c1350, k3, ← c1351] at e_19_16
  have e_19_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_17
  simp only [← c1358, ← c1359, k3, ← c1360] at e_19_17
  have e_20_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_9
  simp only [← c1352, ← c1353, ← c1354] at e_20_9
  have e_20_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_10
  simp only [← c1355, ← c1356, ← c1357] at e_20_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_9
    linear_combination c1361.trans k2 - hc
  · have hc := e_19_17
    simp only [e_19_16, e_20_10] at hc
    linear_combination c1362.trans k2 - hc

theorem publicBatchWrapper86_f222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 3808)) (a (.virt 200)) (a (.virt 17365)) (a (.virt 17366)) := by
  have c1363 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1375 := (publicBatchWrapper86_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (publicBatchWrapper86_copies43 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_19_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_18
  simp only [← c1363, k3, ← c1364, k3, ← c1365] at e_19_18
  have e_19_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row19 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_19
  simp only [← c1372, ← c1373, k3, ← c1374] at e_19_19
  have e_20_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_11
  simp only [← c1366, ← c1367, ← c1368] at e_20_11
  have e_20_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_12
  simp only [← c1369, ← c1370, ← c1371] at e_20_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_11
    linear_combination c1375.trans k2 - hc
  · have hc := e_19_19
    simp only [e_19_18, e_20_12] at hc
    linear_combination c1376.trans k2 - hc

theorem publicBatchWrapper86_f223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 20 55) = band (a (.virt 17359)) (a (.virt 17361)) := by
  have c1377 := (publicBatchWrapper86_copies43 a h).2.1
  have c1378 := (publicBatchWrapper86_copies43 a h).2.2.1
  have c1379 := (publicBatchWrapper86_copies43 a h).2.2.2.1
  have e_20_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_13
  simp only [← c1377, ← c1378, ← c1379] at e_20_13
  have hr := e_20_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 20 59) = band (a (.virt 17363)) (a (.virt 17365)) := by
  have c1380 := (publicBatchWrapper86_copies43 a h).2.2.2.2.1
  have c1381 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.1
  have c1382 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.1
  have e_20_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_14
  simp only [← c1380, ← c1381, ← c1382] at e_20_14
  have hr := e_20_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 20 63) = band (a (.wire 20 55)) (a (.wire 20 59)) := by
  have c1383 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have e_20_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_15
  simp only [← c1383, ← c1384, ← c1385] at e_20_15
  have hr := e_20_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4005)) (a (.virt 200)) (a (.virt 17367)) (a (.virt 17368)) := by
  have c1386 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1389 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1392 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_20_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_16
  simp only [← c1389, ← c1390, ← c1391] at e_20_16
  have e_20_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_17
  simp only [← c1392, ← c1393, ← c1394] at e_20_17
  have e_21_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c1386, k3, ← c1387, k3, ← c1388] at e_21_0
  have e_21_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c1395, ← c1396, k3, ← c1397] at e_21_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_16
    linear_combination c1398.trans k2 - hc
  · have hc := e_21_1
    simp only [e_21_0, e_20_17] at hc
    linear_combination c1399.trans k2 - hc

theorem publicBatchWrapper86_f227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4006)) (a (.virt 200)) (a (.virt 17369)) (a (.virt 17370)) := by
  have c1400 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (publicBatchWrapper86_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (publicBatchWrapper86_copies44 a h).1
  have c1409 := (publicBatchWrapper86_copies44 a h).2.1
  have c1410 := (publicBatchWrapper86_copies44 a h).2.2.1
  have c1411 := (publicBatchWrapper86_copies44 a h).2.2.2.1
  have c1412 := (publicBatchWrapper86_copies44 a h).2.2.2.2.1
  have c1413 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_20_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_18
  simp only [← c1403, ← c1404, ← c1405] at e_20_18
  have e_20_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row20 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_19
  simp only [← c1406, ← c1407, ← c1408] at e_20_19
  have e_21_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c1400, k3, ← c1401, k3, ← c1402] at e_21_2
  have e_21_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c1409, ← c1410, k3, ← c1411] at e_21_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_18
    linear_combination c1412.trans k2 - hc
  · have hc := e_21_3
    simp only [e_21_2, e_20_19] at hc
    linear_combination c1413.trans k2 - hc

theorem publicBatchWrapper86_f228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4007)) (a (.virt 200)) (a (.virt 17371)) (a (.virt 17372)) := by
  have c1414 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.1
  have c1417 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1423 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1425 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1426 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c1414, k3, ← c1415, k3, ← c1416] at e_21_4
  have e_21_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c1423, ← c1424, k3, ← c1425] at e_21_5
  have e_22_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_0
  simp only [← c1417, ← c1418, ← c1419] at e_22_0
  have e_22_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_1
  simp only [← c1420, ← c1421, ← c1422] at e_22_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_0
    linear_combination c1426.trans k2 - hc
  · have hc := e_21_5
    simp only [e_21_4, e_22_1] at hc
    linear_combination c1427.trans k2 - hc

theorem publicBatchWrapper86_f229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4008)) (a (.virt 200)) (a (.virt 17373)) (a (.virt 17374)) := by
  have c1428 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1429 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1431 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1432 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1434 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1435 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1437 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1438 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1439 := (publicBatchWrapper86_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1440 := (publicBatchWrapper86_copies45 a h).1
  have c1441 := (publicBatchWrapper86_copies45 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c1428, k3, ← c1429, k3, ← c1430] at e_21_6
  have e_21_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c1437, ← c1438, k3, ← c1439] at e_21_7
  have e_22_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_2
  simp only [← c1431, ← c1432, ← c1433] at e_22_2
  have e_22_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_3
  simp only [← c1434, ← c1435, ← c1436] at e_22_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_2
    linear_combination c1440.trans k2 - hc
  · have hc := e_21_7
    simp only [e_21_6, e_22_3] at hc
    linear_combination c1441.trans k2 - hc

theorem publicBatchWrapper86_f230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 22 19) = band (a (.virt 17367)) (a (.virt 17369)) := by
  have c1442 := (publicBatchWrapper86_copies45 a h).2.2.1
  have c1443 := (publicBatchWrapper86_copies45 a h).2.2.2.1
  have c1444 := (publicBatchWrapper86_copies45 a h).2.2.2.2.1
  have e_22_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_4
  simp only [← c1442, ← c1443, ← c1444] at e_22_4
  have hr := e_22_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 22 23) = band (a (.virt 17371)) (a (.virt 17373)) := by
  have c1445 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.1
  have c1446 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.1
  have c1447 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.1
  have e_22_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_5
  simp only [← c1445, ← c1446, ← c1447] at e_22_5
  have hr := e_22_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 22 27) = band (a (.wire 22 19)) (a (.wire 22 23)) := by
  have c1448 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_22_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_6
  simp only [← c1448, ← c1449, ← c1450] at e_22_6
  have hr := e_22_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4205)) (a (.virt 200)) (a (.virt 17375)) (a (.virt 17376)) := by
  have c1451 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1454 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1456 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1459 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1451, k3, ← c1452, k3, ← c1453] at e_21_8
  have e_21_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c1460, ← c1461, k3, ← c1462] at e_21_9
  have e_22_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_7
  simp only [← c1454, ← c1455, ← c1456] at e_22_7
  have e_22_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_8
  simp only [← c1457, ← c1458, ← c1459] at e_22_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_7
    linear_combination c1463.trans k2 - hc
  · have hc := e_21_9
    simp only [e_21_8, e_22_8] at hc
    linear_combination c1464.trans k2 - hc

theorem publicBatchWrapper86_f234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4206)) (a (.virt 200)) (a (.virt 17377)) (a (.virt 17378)) := by
  have c1465 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1468 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1469 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (publicBatchWrapper86_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (publicBatchWrapper86_copies46 a h).1
  have c1473 := (publicBatchWrapper86_copies46 a h).2.1
  have c1474 := (publicBatchWrapper86_copies46 a h).2.2.1
  have c1475 := (publicBatchWrapper86_copies46 a h).2.2.2.1
  have c1476 := (publicBatchWrapper86_copies46 a h).2.2.2.2.1
  have c1477 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.1
  have c1478 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c1465, k3, ← c1466, k3, ← c1467] at e_21_10
  have e_21_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c1474, ← c1475, k3, ← c1476] at e_21_11
  have e_22_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_9
  simp only [← c1468, ← c1469, ← c1470] at e_22_9
  have e_22_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_10
  simp only [← c1471, ← c1472, ← c1473] at e_22_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_9
    linear_combination c1477.trans k2 - hc
  · have hc := e_21_11
    simp only [e_21_10, e_22_10] at hc
    linear_combination c1478.trans k2 - hc

theorem publicBatchWrapper86_f235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4207)) (a (.virt 200)) (a (.virt 17379)) (a (.virt 17380)) := by
  have c1479 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.1
  have c1482 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1486 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c1479, k3, ← c1480, k3, ← c1481] at e_21_12
  have e_21_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c1488, ← c1489, k3, ← c1490] at e_21_13
  have e_22_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c1482, ← c1483, ← c1484] at e_22_11
  have e_22_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_12
  simp only [← c1485, ← c1486, ← c1487] at e_22_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_11
    linear_combination c1491.trans k2 - hc
  · have hc := e_21_13
    simp only [e_21_12, e_22_12] at hc
    linear_combination c1492.trans k2 - hc

theorem publicBatchWrapper86_f236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4208)) (a (.virt 200)) (a (.virt 17381)) (a (.virt 17382)) := by
  have c1493 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1496 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1499 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1502 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1503 := (publicBatchWrapper86_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (publicBatchWrapper86_copies47 a h).1
  have c1505 := (publicBatchWrapper86_copies47 a h).2.1
  have c1506 := (publicBatchWrapper86_copies47 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c1493, k3, ← c1494, k3, ← c1495] at e_21_14
  have e_21_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_15
  simp only [← c1502, ← c1503, k3, ← c1504] at e_21_15
  have e_22_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_13
  simp only [← c1496, ← c1497, ← c1498] at e_22_13
  have e_22_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1499, ← c1500, ← c1501] at e_22_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_13
    linear_combination c1505.trans k2 - hc
  · have hc := e_21_15
    simp only [e_21_14, e_22_14] at hc
    linear_combination c1506.trans k2 - hc

theorem publicBatchWrapper86_f237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 22 63) = band (a (.virt 17375)) (a (.virt 17377)) := by
  have c1507 := (publicBatchWrapper86_copies47 a h).2.2.2.1
  have c1508 := (publicBatchWrapper86_copies47 a h).2.2.2.2.1
  have c1509 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.1
  have e_22_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_15
  simp only [← c1507, ← c1508, ← c1509] at e_22_15
  have hr := e_22_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 22 67) = band (a (.virt 17379)) (a (.virt 17381)) := by
  have c1510 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.1
  have e_22_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_16
  simp only [← c1510, ← c1511, ← c1512] at e_22_16
  have hr := e_22_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 22 71) = band (a (.wire 22 63)) (a (.wire 22 67)) := by
  have c1513 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have c1514 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_22_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_17
  simp only [← c1513, ← c1514, ← c1515] at e_22_17
  have hr := e_22_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4405)) (a (.virt 200)) (a (.virt 17383)) (a (.virt 17384)) := by
  have c1516 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1519 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1520 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1522 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1523 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1526 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_16
  simp only [← c1516, k3, ← c1517, k3, ← c1518] at e_21_16
  have e_21_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_17
  simp only [← c1525, ← c1526, k3, ← c1527] at e_21_17
  have e_22_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_18
  simp only [← c1519, ← c1520, ← c1521] at e_22_18
  have e_22_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row22 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_19
  simp only [← c1522, ← c1523, ← c1524] at e_22_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_18
    linear_combination c1528.trans k2 - hc
  · have hc := e_21_17
    simp only [e_21_16, e_22_19] at hc
    linear_combination c1529.trans k2 - hc

theorem publicBatchWrapper86_f241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4406)) (a (.virt 200)) (a (.virt 17385)) (a (.virt 17386)) := by
  have c1530 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1531 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1532 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1533 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1535 := (publicBatchWrapper86_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (publicBatchWrapper86_copies48 a h).1
  have c1537 := (publicBatchWrapper86_copies48 a h).2.1
  have c1538 := (publicBatchWrapper86_copies48 a h).2.2.1
  have c1539 := (publicBatchWrapper86_copies48 a h).2.2.2.1
  have c1540 := (publicBatchWrapper86_copies48 a h).2.2.2.2.1
  have c1541 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.1
  have c1542 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.1
  have c1543 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_21_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_18
  simp only [← c1530, k3, ← c1531, k3, ← c1532] at e_21_18
  have e_21_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row21 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_19
  simp only [← c1539, ← c1540, k3, ← c1541] at e_21_19
  have e_23_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_0
  simp only [← c1533, ← c1534, ← c1535] at e_23_0
  have e_23_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_1
  simp only [← c1536, ← c1537, ← c1538] at e_23_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_0
    linear_combination c1542.trans k2 - hc
  · have hc := e_21_19
    simp only [e_21_18, e_23_1] at hc
    linear_combination c1543.trans k2 - hc

theorem publicBatchWrapper86_f242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4407)) (a (.virt 200)) (a (.virt 17387)) (a (.virt 17388)) := by
  have c1544 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.1
  have c1545 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have c1546 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1547 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1548 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1553 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1554 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1556 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1557 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_23_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_2
  simp only [← c1547, ← c1548, ← c1549] at e_23_2
  have e_23_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_3
  simp only [← c1550, ← c1551, ← c1552] at e_23_3
  have e_24_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_0
  simp only [← c1544, k3, ← c1545, k3, ← c1546] at e_24_0
  have e_24_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_1
  simp only [← c1553, ← c1554, k3, ← c1555] at e_24_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_2
    linear_combination c1556.trans k2 - hc
  · have hc := e_24_1
    simp only [e_24_0, e_23_3] at hc
    linear_combination c1557.trans k2 - hc

theorem publicBatchWrapper86_f243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4408)) (a (.virt 200)) (a (.virt 17389)) (a (.virt 17390)) := by
  have c1558 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1559 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1561 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1562 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (publicBatchWrapper86_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (publicBatchWrapper86_copies49 a h).1
  have c1569 := (publicBatchWrapper86_copies49 a h).2.1
  have c1570 := (publicBatchWrapper86_copies49 a h).2.2.1
  have c1571 := (publicBatchWrapper86_copies49 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_23_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_4
  simp only [← c1561, ← c1562, ← c1563] at e_23_4
  have e_23_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_5
  simp only [← c1564, ← c1565, ← c1566] at e_23_5
  have e_24_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c1558, k3, ← c1559, k3, ← c1560] at e_24_2
  have e_24_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_3
  simp only [← c1567, ← c1568, k3, ← c1569] at e_24_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_4
    linear_combination c1570.trans k2 - hc
  · have hc := e_24_3
    simp only [e_24_2, e_23_5] at hc
    linear_combination c1571.trans k2 - hc

theorem publicBatchWrapper86_f244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 23 27) = band (a (.virt 17383)) (a (.virt 17385)) := by
  have c1572 := (publicBatchWrapper86_copies49 a h).2.2.2.2.1
  have c1573 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.1
  have c1574 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.1
  have e_23_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_6
  simp only [← c1572, ← c1573, ← c1574] at e_23_6
  have hr := e_23_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 23 31) = band (a (.virt 17387)) (a (.virt 17389)) := by
  have c1575 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have e_23_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_7
  simp only [← c1575, ← c1576, ← c1577] at e_23_7
  have hr := e_23_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 23 35) = band (a (.wire 23 27)) (a (.wire 23 31)) := by
  have c1578 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1580 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_23_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_8
  simp only [← c1578, ← c1579, ← c1580] at e_23_8
  have hr := e_23_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4605)) (a (.virt 200)) (a (.virt 17391)) (a (.virt 17392)) := by
  have c1581 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1583 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1584 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1586 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1590 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1592 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1593 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_23_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_9
  simp only [← c1584, ← c1585, ← c1586] at e_23_9
  have e_23_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_10
  simp only [← c1587, ← c1588, ← c1589] at e_23_10
  have e_24_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_4
  simp only [← c1581, k3, ← c1582, k3, ← c1583] at e_24_4
  have e_24_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c1590, ← c1591, k3, ← c1592] at e_24_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_9
    linear_combination c1593.trans k2 - hc
  · have hc := e_24_5
    simp only [e_24_4, e_23_10] at hc
    linear_combination c1594.trans k2 - hc

theorem publicBatchWrapper86_f248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4606)) (a (.virt 200)) (a (.virt 17393)) (a (.virt 17394)) := by
  have c1595 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1597 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1598 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1599 := (publicBatchWrapper86_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (publicBatchWrapper86_copies50 a h).1
  have c1601 := (publicBatchWrapper86_copies50 a h).2.1
  have c1602 := (publicBatchWrapper86_copies50 a h).2.2.1
  have c1603 := (publicBatchWrapper86_copies50 a h).2.2.2.1
  have c1604 := (publicBatchWrapper86_copies50 a h).2.2.2.2.1
  have c1605 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.1
  have c1606 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.1
  have c1607 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_23_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_11
  simp only [← c1598, ← c1599, ← c1600] at e_23_11
  have e_23_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_12
  simp only [← c1601, ← c1602, ← c1603] at e_23_12
  have e_24_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_6
  simp only [← c1595, k3, ← c1596, k3, ← c1597] at e_24_6
  have e_24_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_7
  simp only [← c1604, ← c1605, k3, ← c1606] at e_24_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_11
    linear_combination c1607.trans k2 - hc
  · have hc := e_24_7
    simp only [e_24_6, e_23_12] at hc
    linear_combination c1608.trans k2 - hc

theorem publicBatchWrapper86_f249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4607)) (a (.virt 200)) (a (.virt 17395)) (a (.virt 17396)) := by
  have c1609 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have c1610 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1612 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1613 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1614 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1616 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1617 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1619 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1620 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1622 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_23_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_13
  simp only [← c1612, ← c1613, ← c1614] at e_23_13
  have e_23_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_14
  simp only [← c1615, ← c1616, ← c1617] at e_23_14
  have e_24_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_8
  simp only [← c1609, k3, ← c1610, k3, ← c1611] at e_24_8
  have e_24_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_9
  simp only [← c1618, ← c1619, k3, ← c1620] at e_24_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_13
    linear_combination c1621.trans k2 - hc
  · have hc := e_24_9
    simp only [e_24_8, e_23_14] at hc
    linear_combination c1622.trans k2 - hc

theorem publicBatchWrapper86_f250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4608)) (a (.virt 200)) (a (.virt 17397)) (a (.virt 17398)) := by
  have c1623 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1626 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1628 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1629 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1630 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1631 := (publicBatchWrapper86_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (publicBatchWrapper86_copies51 a h).1
  have c1633 := (publicBatchWrapper86_copies51 a h).2.1
  have c1634 := (publicBatchWrapper86_copies51 a h).2.2.1
  have c1635 := (publicBatchWrapper86_copies51 a h).2.2.2.1
  have c1636 := (publicBatchWrapper86_copies51 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_23_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_15
  simp only [← c1626, ← c1627, ← c1628] at e_23_15
  have e_23_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_16
  simp only [← c1629, ← c1630, ← c1631] at e_23_16
  have e_24_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_10
  simp only [← c1623, k3, ← c1624, k3, ← c1625] at e_24_10
  have e_24_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_11
  simp only [← c1632, ← c1633, k3, ← c1634] at e_24_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_23_15
    linear_combination c1635.trans k2 - hc
  · have hc := e_24_11
    simp only [e_24_10, e_23_16] at hc
    linear_combination c1636.trans k2 - hc

theorem publicBatchWrapper86_f251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 23 71) = band (a (.virt 17391)) (a (.virt 17393)) := by
  have c1637 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.1
  have c1638 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.1
  have e_23_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_17
  simp only [← c1637, ← c1638, ← c1639] at e_23_17
  have hr := e_23_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 23 75) = band (a (.virt 17395)) (a (.virt 17397)) := by
  have c1640 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_23_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_18
  simp only [← c1640, ← c1641, ← c1642] at e_23_18
  have hr := e_23_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 23 79) = band (a (.wire 23 71)) (a (.wire 23 75)) := by
  have c1643 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_23_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row23 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_19
  simp only [← c1643, ← c1644, ← c1645] at e_23_19
  have hr := e_23_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4805)) (a (.virt 200)) (a (.virt 17399)) (a (.virt 17400)) := by
  have c1646 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1649 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1655 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1657 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1658 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_24_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_12
  simp only [← c1646, k3, ← c1647, k3, ← c1648] at e_24_12
  have e_24_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_13
  simp only [← c1655, ← c1656, k3, ← c1657] at e_24_13
  have e_25_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_0
  simp only [← c1649, ← c1650, ← c1651] at e_25_0
  have e_25_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_1
  simp only [← c1652, ← c1653, ← c1654] at e_25_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_0
    linear_combination c1658.trans k2 - hc
  · have hc := e_24_13
    simp only [e_24_12, e_25_1] at hc
    linear_combination c1659.trans k2 - hc

theorem publicBatchWrapper86_f255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4806)) (a (.virt 200)) (a (.virt 17401)) (a (.virt 17402)) := by
  have c1660 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1661 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1663 := (publicBatchWrapper86_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (publicBatchWrapper86_copies52 a h).1
  have c1665 := (publicBatchWrapper86_copies52 a h).2.1
  have c1666 := (publicBatchWrapper86_copies52 a h).2.2.1
  have c1667 := (publicBatchWrapper86_copies52 a h).2.2.2.1
  have c1668 := (publicBatchWrapper86_copies52 a h).2.2.2.2.1
  have c1669 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.1
  have c1670 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.1
  have c1671 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_24_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_14
  simp only [← c1660, k3, ← c1661, k3, ← c1662] at e_24_14
  have e_24_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_15
  simp only [← c1669, ← c1670, k3, ← c1671] at e_24_15
  have e_25_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_2
  simp only [← c1663, ← c1664, ← c1665] at e_25_2
  have e_25_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_3
  simp only [← c1666, ← c1667, ← c1668] at e_25_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_2
    linear_combination c1672.trans k2 - hc
  · have hc := e_24_15
    simp only [e_24_14, e_25_3] at hc
    linear_combination c1673.trans k2 - hc

theorem publicBatchWrapper86_f256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4807)) (a (.virt 200)) (a (.virt 17403)) (a (.virt 17404)) := by
  have c1674 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1677 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1679 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1685 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_24_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_16
  simp only [← c1674, k3, ← c1675, k3, ← c1676] at e_24_16
  have e_24_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_17
  simp only [← c1683, ← c1684, k3, ← c1685] at e_24_17
  have e_25_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_4
  simp only [← c1677, ← c1678, ← c1679] at e_25_4
  have e_25_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_5
  simp only [← c1680, ← c1681, ← c1682] at e_25_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_4
    linear_combination c1686.trans k2 - hc
  · have hc := e_24_17
    simp only [e_24_16, e_25_5] at hc
    linear_combination c1687.trans k2 - hc

theorem publicBatchWrapper86_f257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 4808)) (a (.virt 200)) (a (.virt 17405)) (a (.virt 17406)) := by
  have c1688 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1689 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1691 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1692 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1694 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (publicBatchWrapper86_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (publicBatchWrapper86_copies53 a h).1
  have c1697 := (publicBatchWrapper86_copies53 a h).2.1
  have c1698 := (publicBatchWrapper86_copies53 a h).2.2.1
  have c1699 := (publicBatchWrapper86_copies53 a h).2.2.2.1
  have c1700 := (publicBatchWrapper86_copies53 a h).2.2.2.2.1
  have c1701 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_24_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_18
  simp only [← c1688, k3, ← c1689, k3, ← c1690] at e_24_18
  have e_24_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row24 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_19
  simp only [← c1697, ← c1698, k3, ← c1699] at e_24_19
  have e_25_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_6
  simp only [← c1691, ← c1692, ← c1693] at e_25_6
  have e_25_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_7
  simp only [← c1694, ← c1695, ← c1696] at e_25_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_6
    linear_combination c1700.trans k2 - hc
  · have hc := e_24_19
    simp only [e_24_18, e_25_7] at hc
    linear_combination c1701.trans k2 - hc

theorem publicBatchWrapper86_f258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 25 35) = band (a (.virt 17399)) (a (.virt 17401)) := by
  have c1702 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.1
  have c1703 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.1
  have e_25_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_8
  simp only [← c1702, ← c1703, ← c1704] at e_25_8
  have hr := e_25_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 25 39) = band (a (.virt 17403)) (a (.virt 17405)) := by
  have c1705 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_25_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_9
  simp only [← c1705, ← c1706, ← c1707] at e_25_9
  have hr := e_25_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 25 43) = band (a (.wire 25 35)) (a (.wire 25 39)) := by
  have c1708 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1709 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_25_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_10
  simp only [← c1708, ← c1709, ← c1710] at e_25_10
  have hr := e_25_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5005)) (a (.virt 200)) (a (.virt 17407)) (a (.virt 17408)) := by
  have c1711 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1714 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1715 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1716 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1717 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1718 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1719 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1720 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1721 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1722 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1723 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1724 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_25_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_11
  simp only [← c1714, ← c1715, ← c1716] at e_25_11
  have e_25_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_12
  simp only [← c1717, ← c1718, ← c1719] at e_25_12
  have e_26_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1711, k3, ← c1712, k3, ← c1713] at e_26_0
  have e_26_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1720, ← c1721, k3, ← c1722] at e_26_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_11
    linear_combination c1723.trans k2 - hc
  · have hc := e_26_1
    simp only [e_26_0, e_25_12] at hc
    linear_combination c1724.trans k2 - hc

theorem publicBatchWrapper86_f262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5006)) (a (.virt 200)) (a (.virt 17409)) (a (.virt 17410)) := by
  have c1725 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1726 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1727 := (publicBatchWrapper86_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1728 := (publicBatchWrapper86_copies54 a h).1
  have c1729 := (publicBatchWrapper86_copies54 a h).2.1
  have c1730 := (publicBatchWrapper86_copies54 a h).2.2.1
  have c1731 := (publicBatchWrapper86_copies54 a h).2.2.2.1
  have c1732 := (publicBatchWrapper86_copies54 a h).2.2.2.2.1
  have c1733 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.1
  have c1734 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.1
  have c1735 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.1
  have c1736 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.1
  have c1737 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.1
  have c1738 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_25_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_13
  simp only [← c1728, ← c1729, ← c1730] at e_25_13
  have e_25_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_14
  simp only [← c1731, ← c1732, ← c1733] at e_25_14
  have e_26_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1725, k3, ← c1726, k3, ← c1727] at e_26_2
  have e_26_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1734, ← c1735, k3, ← c1736] at e_26_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_13
    linear_combination c1737.trans k2 - hc
  · have hc := e_26_3
    simp only [e_26_2, e_25_14] at hc
    linear_combination c1738.trans k2 - hc

theorem publicBatchWrapper86_f263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5007)) (a (.virt 200)) (a (.virt 17411)) (a (.virt 17412)) := by
  have c1739 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1740 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1741 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1742 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1743 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1744 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1745 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1746 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1747 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1748 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1749 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1750 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1751 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1752 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_25_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_15
  simp only [← c1742, ← c1743, ← c1744] at e_25_15
  have e_25_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_16
  simp only [← c1745, ← c1746, ← c1747] at e_25_16
  have e_26_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1739, k3, ← c1740, k3, ← c1741] at e_26_4
  have e_26_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1748, ← c1749, k3, ← c1750] at e_26_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_15
    linear_combination c1751.trans k2 - hc
  · have hc := e_26_5
    simp only [e_26_4, e_25_16] at hc
    linear_combination c1752.trans k2 - hc

theorem publicBatchWrapper86_f264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5008)) (a (.virt 200)) (a (.virt 17413)) (a (.virt 17414)) := by
  have c1753 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1754 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1755 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1756 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1757 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1758 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1759 := (publicBatchWrapper86_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1760 := (publicBatchWrapper86_copies55 a h).1
  have c1761 := (publicBatchWrapper86_copies55 a h).2.1
  have c1762 := (publicBatchWrapper86_copies55 a h).2.2.1
  have c1763 := (publicBatchWrapper86_copies55 a h).2.2.2.1
  have c1764 := (publicBatchWrapper86_copies55 a h).2.2.2.2.1
  have c1765 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.1
  have c1766 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_25_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_17
  simp only [← c1756, ← c1757, ← c1758] at e_25_17
  have e_25_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_18
  simp only [← c1759, ← c1760, ← c1761] at e_25_18
  have e_26_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1753, k3, ← c1754, k3, ← c1755] at e_26_6
  have e_26_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1762, ← c1763, k3, ← c1764] at e_26_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_25_17
    linear_combination c1765.trans k2 - hc
  · have hc := e_26_7
    simp only [e_26_6, e_25_18] at hc
    linear_combination c1766.trans k2 - hc

theorem publicBatchWrapper86_f265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 25 79) = band (a (.virt 17407)) (a (.virt 17409)) := by
  have c1767 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.1
  have c1768 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.1
  have c1769 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.1
  have e_25_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row25 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_19
  simp only [← c1767, ← c1768, ← c1769] at e_25_19
  have hr := e_25_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 27 3) = band (a (.virt 17411)) (a (.virt 17413)) := by
  have c1770 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1771 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1772 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1770, ← c1771, ← c1772] at e_27_0
  have hr := e_27_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 27 7) = band (a (.wire 25 79)) (a (.wire 27 3)) := by
  have c1773 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1774 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1775 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1773, ← c1774, ← c1775] at e_27_1
  have hr := e_27_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5205)) (a (.virt 200)) (a (.virt 17415)) (a (.virt 17416)) := by
  have c1776 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1777 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1778 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1779 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1780 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1781 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1782 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1783 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1784 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1785 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1786 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1787 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1788 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1789 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_26_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1776, k3, ← c1777, k3, ← c1778] at e_26_8
  have e_26_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1785, ← c1786, k3, ← c1787] at e_26_9
  have e_27_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1779, ← c1780, ← c1781] at e_27_2
  have e_27_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1782, ← c1783, ← c1784] at e_27_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_2
    linear_combination c1788.trans k2 - hc
  · have hc := e_26_9
    simp only [e_26_8, e_27_3] at hc
    linear_combination c1789.trans k2 - hc

theorem publicBatchWrapper86_f269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5206)) (a (.virt 200)) (a (.virt 17417)) (a (.virt 17418)) := by
  have c1790 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1791 := (publicBatchWrapper86_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1792 := (publicBatchWrapper86_copies56 a h).1
  have c1793 := (publicBatchWrapper86_copies56 a h).2.1
  have c1794 := (publicBatchWrapper86_copies56 a h).2.2.1
  have c1795 := (publicBatchWrapper86_copies56 a h).2.2.2.1
  have c1796 := (publicBatchWrapper86_copies56 a h).2.2.2.2.1
  have c1797 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.1
  have c1798 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.1
  have c1799 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.1
  have c1800 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.1
  have c1801 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.1
  have c1802 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1803 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_26_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1790, k3, ← c1791, k3, ← c1792] at e_26_10
  have e_26_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1799, ← c1800, k3, ← c1801] at e_26_11
  have e_27_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_4
  simp only [← c1793, ← c1794, ← c1795] at e_27_4
  have e_27_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_5
  simp only [← c1796, ← c1797, ← c1798] at e_27_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_4
    linear_combination c1802.trans k2 - hc
  · have hc := e_26_11
    simp only [e_26_10, e_27_5] at hc
    linear_combination c1803.trans k2 - hc

theorem publicBatchWrapper86_f270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5207)) (a (.virt 200)) (a (.virt 17419)) (a (.virt 17420)) := by
  have c1804 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1805 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1806 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1807 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1808 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1809 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1810 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1811 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1812 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1813 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1814 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1815 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1816 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1817 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_26_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1804, k3, ← c1805, k3, ← c1806] at e_26_12
  have e_26_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1813, ← c1814, k3, ← c1815] at e_26_13
  have e_27_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_6
  simp only [← c1807, ← c1808, ← c1809] at e_27_6
  have e_27_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_7
  simp only [← c1810, ← c1811, ← c1812] at e_27_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_6
    linear_combination c1816.trans k2 - hc
  · have hc := e_26_13
    simp only [e_26_12, e_27_7] at hc
    linear_combination c1817.trans k2 - hc

theorem publicBatchWrapper86_f271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5208)) (a (.virt 200)) (a (.virt 17421)) (a (.virt 17422)) := by
  have c1818 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1819 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1820 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1821 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1822 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1823 := (publicBatchWrapper86_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1824 := (publicBatchWrapper86_copies57 a h).1
  have c1825 := (publicBatchWrapper86_copies57 a h).2.1
  have c1826 := (publicBatchWrapper86_copies57 a h).2.2.1
  have c1827 := (publicBatchWrapper86_copies57 a h).2.2.2.1
  have c1828 := (publicBatchWrapper86_copies57 a h).2.2.2.2.1
  have c1829 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.1
  have c1830 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.1
  have c1831 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_26_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1818, k3, ← c1819, k3, ← c1820] at e_26_14
  have e_26_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_15
  simp only [← c1827, ← c1828, k3, ← c1829] at e_26_15
  have e_27_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_8
  simp only [← c1821, ← c1822, ← c1823] at e_27_8
  have e_27_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_9
  simp only [← c1824, ← c1825, ← c1826] at e_27_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_8
    linear_combination c1830.trans k2 - hc
  · have hc := e_26_15
    simp only [e_26_14, e_27_9] at hc
    linear_combination c1831.trans k2 - hc

theorem publicBatchWrapper86_f272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 27 43) = band (a (.virt 17415)) (a (.virt 17417)) := by
  have c1832 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.1
  have c1833 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.1
  have c1834 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_27_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_10
  simp only [← c1832, ← c1833, ← c1834] at e_27_10
  have hr := e_27_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 27 47) = band (a (.virt 17419)) (a (.virt 17421)) := by
  have c1835 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1836 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1837 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_11
  simp only [← c1835, ← c1836, ← c1837] at e_27_11
  have hr := e_27_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 27 51) = band (a (.wire 27 43)) (a (.wire 27 47)) := by
  have c1838 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1839 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1840 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_12
  simp only [← c1838, ← c1839, ← c1840] at e_27_12
  have hr := e_27_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5405)) (a (.virt 200)) (a (.virt 17423)) (a (.virt 17424)) := by
  have c1841 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1842 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1843 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1844 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1845 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1846 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1847 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1848 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1849 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1850 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1851 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1852 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1853 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1854 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_26_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_16
  simp only [← c1841, k3, ← c1842, k3, ← c1843] at e_26_16
  have e_26_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_17
  simp only [← c1850, ← c1851, k3, ← c1852] at e_26_17
  have e_27_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_13
  simp only [← c1844, ← c1845, ← c1846] at e_27_13
  have e_27_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_14
  simp only [← c1847, ← c1848, ← c1849] at e_27_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_13
    linear_combination c1853.trans k2 - hc
  · have hc := e_26_17
    simp only [e_26_16, e_27_14] at hc
    linear_combination c1854.trans k2 - hc

theorem publicBatchWrapper86_f276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5406)) (a (.virt 200)) (a (.virt 17425)) (a (.virt 17426)) := by
  have c1855 := (publicBatchWrapper86_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1856 := (publicBatchWrapper86_copies58 a h).1
  have c1857 := (publicBatchWrapper86_copies58 a h).2.1
  have c1858 := (publicBatchWrapper86_copies58 a h).2.2.1
  have c1859 := (publicBatchWrapper86_copies58 a h).2.2.2.1
  have c1860 := (publicBatchWrapper86_copies58 a h).2.2.2.2.1
  have c1861 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.1
  have c1862 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.1
  have c1863 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.1
  have c1864 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.1
  have c1865 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.1
  have c1866 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1867 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1868 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_26_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_18
  simp only [← c1855, k3, ← c1856, k3, ← c1857] at e_26_18
  have e_26_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row26 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_19
  simp only [← c1864, ← c1865, k3, ← c1866] at e_26_19
  have e_27_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_15
  simp only [← c1858, ← c1859, ← c1860] at e_27_15
  have e_27_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_16
  simp only [← c1861, ← c1862, ← c1863] at e_27_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_15
    linear_combination c1867.trans k2 - hc
  · have hc := e_26_19
    simp only [e_26_18, e_27_16] at hc
    linear_combination c1868.trans k2 - hc

theorem publicBatchWrapper86_f277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5407)) (a (.virt 200)) (a (.virt 17427)) (a (.virt 17428)) := by
  have c1869 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1870 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1871 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1872 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1873 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1874 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1875 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1876 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1877 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1878 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1879 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1880 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1881 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1882 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_27_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_17
  simp only [← c1872, ← c1873, ← c1874] at e_27_17
  have e_27_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_18
  simp only [← c1875, ← c1876, ← c1877] at e_27_18
  have e_28_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c1869, k3, ← c1870, k3, ← c1871] at e_28_0
  have e_28_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c1878, ← c1879, k3, ← c1880] at e_28_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_17
    linear_combination c1881.trans k2 - hc
  · have hc := e_28_1
    simp only [e_28_0, e_27_18] at hc
    linear_combination c1882.trans k2 - hc

theorem publicBatchWrapper86_f278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5408)) (a (.virt 200)) (a (.virt 17429)) (a (.virt 17430)) := by
  have c1883 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1884 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1885 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1886 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1887 := (publicBatchWrapper86_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1888 := (publicBatchWrapper86_copies59 a h).1
  have c1889 := (publicBatchWrapper86_copies59 a h).2.1
  have c1890 := (publicBatchWrapper86_copies59 a h).2.2.1
  have c1891 := (publicBatchWrapper86_copies59 a h).2.2.2.1
  have c1892 := (publicBatchWrapper86_copies59 a h).2.2.2.2.1
  have c1893 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.1
  have c1894 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.1
  have c1895 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.1
  have c1896 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_27_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row27 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_19
  simp only [← c1886, ← c1887, ← c1888] at e_27_19
  have e_28_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c1883, k3, ← c1884, k3, ← c1885] at e_28_2
  have e_28_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c1892, ← c1893, k3, ← c1894] at e_28_3
  have e_29_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1889, ← c1890, ← c1891] at e_29_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_19
    linear_combination c1895.trans k2 - hc
  · have hc := e_28_3
    simp only [e_28_2, e_29_0] at hc
    linear_combination c1896.trans k2 - hc

theorem publicBatchWrapper86_f279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 29 7) = band (a (.virt 17423)) (a (.virt 17425)) := by
  have c1897 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.1
  have c1898 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1899 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1897, ← c1898, ← c1899] at e_29_1
  have hr := e_29_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 29 11) = band (a (.virt 17427)) (a (.virt 17429)) := by
  have c1900 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1901 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1902 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1900, ← c1901, ← c1902] at e_29_2
  have hr := e_29_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 29 15) = band (a (.wire 29 7)) (a (.wire 29 11)) := by
  have c1903 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1904 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1905 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1903, ← c1904, ← c1905] at e_29_3
  have hr := e_29_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5605)) (a (.virt 200)) (a (.virt 17431)) (a (.virt 17432)) := by
  have c1906 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1907 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1908 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1909 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1910 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1911 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1912 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1913 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1914 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1915 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1916 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1917 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1918 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1919 := (publicBatchWrapper86_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c1906, k3, ← c1907, k3, ← c1908] at e_28_4
  have e_28_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1915, ← c1916, k3, ← c1917] at e_28_5
  have e_29_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1909, ← c1910, ← c1911] at e_29_4
  have e_29_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1912, ← c1913, ← c1914] at e_29_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_4
    linear_combination c1918.trans k2 - hc
  · have hc := e_28_5
    simp only [e_28_4, e_29_5] at hc
    linear_combination c1919.trans k2 - hc

theorem publicBatchWrapper86_f283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5606)) (a (.virt 200)) (a (.virt 17433)) (a (.virt 17434)) := by
  have c1920 := (publicBatchWrapper86_copies60 a h).1
  have c1921 := (publicBatchWrapper86_copies60 a h).2.1
  have c1922 := (publicBatchWrapper86_copies60 a h).2.2.1
  have c1923 := (publicBatchWrapper86_copies60 a h).2.2.2.1
  have c1924 := (publicBatchWrapper86_copies60 a h).2.2.2.2.1
  have c1925 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.1
  have c1926 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.1
  have c1927 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.1
  have c1928 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.1
  have c1929 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.1
  have c1930 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1931 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1932 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1933 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1920, k3, ← c1921, k3, ← c1922] at e_28_6
  have e_28_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1929, ← c1930, k3, ← c1931] at e_28_7
  have e_29_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1923, ← c1924, ← c1925] at e_29_6
  have e_29_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c1926, ← c1927, ← c1928] at e_29_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_6
    linear_combination c1932.trans k2 - hc
  · have hc := e_28_7
    simp only [e_28_6, e_29_7] at hc
    linear_combination c1933.trans k2 - hc

theorem publicBatchWrapper86_f284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5607)) (a (.virt 200)) (a (.virt 17435)) (a (.virt 17436)) := by
  have c1934 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1935 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1936 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1937 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1938 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1939 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1940 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1941 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1942 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1943 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1944 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1945 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1946 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1947 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1934, k3, ← c1935, k3, ← c1936] at e_28_8
  have e_28_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1943, ← c1944, k3, ← c1945] at e_28_9
  have e_29_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c1937, ← c1938, ← c1939] at e_29_8
  have e_29_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c1940, ← c1941, ← c1942] at e_29_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_8
    linear_combination c1946.trans k2 - hc
  · have hc := e_28_9
    simp only [e_28_8, e_29_9] at hc
    linear_combination c1947.trans k2 - hc

theorem publicBatchWrapper86_f285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5608)) (a (.virt 200)) (a (.virt 17437)) (a (.virt 17438)) := by
  have c1948 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1949 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1950 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1951 := (publicBatchWrapper86_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1952 := (publicBatchWrapper86_copies61 a h).1
  have c1953 := (publicBatchWrapper86_copies61 a h).2.1
  have c1954 := (publicBatchWrapper86_copies61 a h).2.2.1
  have c1955 := (publicBatchWrapper86_copies61 a h).2.2.2.1
  have c1956 := (publicBatchWrapper86_copies61 a h).2.2.2.2.1
  have c1957 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.1
  have c1958 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.1
  have c1959 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.1
  have c1960 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.1
  have c1961 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1948, k3, ← c1949, k3, ← c1950] at e_28_10
  have e_28_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1957, ← c1958, k3, ← c1959] at e_28_11
  have e_29_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c1951, ← c1952, ← c1953] at e_29_10
  have e_29_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c1954, ← c1955, ← c1956] at e_29_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_10
    linear_combination c1960.trans k2 - hc
  · have hc := e_28_11
    simp only [e_28_10, e_29_11] at hc
    linear_combination c1961.trans k2 - hc

theorem publicBatchWrapper86_f286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 29 51) = band (a (.virt 17431)) (a (.virt 17433)) := by
  have c1962 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1963 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1964 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c1962, ← c1963, ← c1964] at e_29_12
  have hr := e_29_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 29 55) = band (a (.virt 17435)) (a (.virt 17437)) := by
  have c1965 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1966 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1967 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c1965, ← c1966, ← c1967] at e_29_13
  have hr := e_29_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 29 59) = band (a (.wire 29 51)) (a (.wire 29 55)) := by
  have c1968 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1969 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1970 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c1968, ← c1969, ← c1970] at e_29_14
  have hr := e_29_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5805)) (a (.virt 200)) (a (.virt 17439)) (a (.virt 17440)) := by
  have c1971 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1972 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1973 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1974 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1975 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1976 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1977 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1978 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1979 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1980 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1981 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1982 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1983 := (publicBatchWrapper86_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1984 := (publicBatchWrapper86_copies62 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1971, k3, ← c1972, k3, ← c1973] at e_28_12
  have e_28_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1980, ← c1981, k3, ← c1982] at e_28_13
  have e_29_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_15
  simp only [← c1974, ← c1975, ← c1976] at e_29_15
  have e_29_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_16
  simp only [← c1977, ← c1978, ← c1979] at e_29_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_15
    linear_combination c1983.trans k2 - hc
  · have hc := e_28_13
    simp only [e_28_12, e_29_16] at hc
    linear_combination c1984.trans k2 - hc

theorem publicBatchWrapper86_f290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5806)) (a (.virt 200)) (a (.virt 17441)) (a (.virt 17442)) := by
  have c1985 := (publicBatchWrapper86_copies62 a h).2.1
  have c1986 := (publicBatchWrapper86_copies62 a h).2.2.1
  have c1987 := (publicBatchWrapper86_copies62 a h).2.2.2.1
  have c1988 := (publicBatchWrapper86_copies62 a h).2.2.2.2.1
  have c1989 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.1
  have c1990 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.1
  have c1991 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.1
  have c1992 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.1
  have c1993 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.1
  have c1994 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1995 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1996 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1997 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1998 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1985, k3, ← c1986, k3, ← c1987] at e_28_14
  have e_28_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_15
  simp only [← c1994, ← c1995, k3, ← c1996] at e_28_15
  have e_29_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_17
  simp only [← c1988, ← c1989, ← c1990] at e_29_17
  have e_29_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_18
  simp only [← c1991, ← c1992, ← c1993] at e_29_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_17
    linear_combination c1997.trans k2 - hc
  · have hc := e_28_15
    simp only [e_28_14, e_29_18] at hc
    linear_combination c1998.trans k2 - hc

theorem publicBatchWrapper86_f291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5807)) (a (.virt 200)) (a (.virt 17443)) (a (.virt 17444)) := by
  have c1999 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2000 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2001 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2002 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2003 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2004 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2005 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2006 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2007 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2008 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2009 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2010 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2011 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2012 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_16
  simp only [← c1999, k3, ← c2000, k3, ← c2001] at e_28_16
  have e_28_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_17
  simp only [← c2008, ← c2009, k3, ← c2010] at e_28_17
  have e_29_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row29 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_19
  simp only [← c2002, ← c2003, ← c2004] at e_29_19
  have e_30_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c2005, ← c2006, ← c2007] at e_30_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_19
    linear_combination c2011.trans k2 - hc
  · have hc := e_28_17
    simp only [e_28_16, e_30_0] at hc
    linear_combination c2012.trans k2 - hc

theorem publicBatchWrapper86_f292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 5808)) (a (.virt 200)) (a (.virt 17445)) (a (.virt 17446)) := by
  have c2013 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2014 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2015 := (publicBatchWrapper86_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2016 := (publicBatchWrapper86_copies63 a h).1
  have c2017 := (publicBatchWrapper86_copies63 a h).2.1
  have c2018 := (publicBatchWrapper86_copies63 a h).2.2.1
  have c2019 := (publicBatchWrapper86_copies63 a h).2.2.2.1
  have c2020 := (publicBatchWrapper86_copies63 a h).2.2.2.2.1
  have c2021 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.1
  have c2022 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.1
  have c2023 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.1
  have c2024 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.1
  have c2025 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.1
  have c2026 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_28_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_18
  simp only [← c2013, k3, ← c2014, k3, ← c2015] at e_28_18
  have e_28_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row28 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_19
  simp only [← c2022, ← c2023, k3, ← c2024] at e_28_19
  have e_30_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c2016, ← c2017, ← c2018] at e_30_1
  have e_30_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c2019, ← c2020, ← c2021] at e_30_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_1
    linear_combination c2025.trans k2 - hc
  · have hc := e_28_19
    simp only [e_28_18, e_30_2] at hc
    linear_combination c2026.trans k2 - hc

theorem publicBatchWrapper86_f293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 30 15) = band (a (.virt 17439)) (a (.virt 17441)) := by
  have c2027 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2028 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2029 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c2027, ← c2028, ← c2029] at e_30_3
  have hr := e_30_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 30 19) = band (a (.virt 17443)) (a (.virt 17445)) := by
  have c2030 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2031 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2032 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c2030, ← c2031, ← c2032] at e_30_4
  have hr := e_30_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 30 23) = band (a (.wire 30 15)) (a (.wire 30 19)) := by
  have c2033 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2034 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2035 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c2033, ← c2034, ← c2035] at e_30_5
  have hr := e_30_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6005)) (a (.virt 200)) (a (.virt 17447)) (a (.virt 17448)) := by
  have c2036 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2037 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2038 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2039 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2040 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2046 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2047 := (publicBatchWrapper86_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2048 := (publicBatchWrapper86_copies64 a h).1
  have c2049 := (publicBatchWrapper86_copies64 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_30_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c2039, ← c2040, ← c2041] at e_30_6
  have e_30_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c2042, ← c2043, ← c2044] at e_30_7
  have e_31_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c2036, k3, ← c2037, k3, ← c2038] at e_31_0
  have e_31_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c2045, ← c2046, k3, ← c2047] at e_31_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_6
    linear_combination c2048.trans k2 - hc
  · have hc := e_31_1
    simp only [e_31_0, e_30_7] at hc
    linear_combination c2049.trans k2 - hc

theorem publicBatchWrapper86_f297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6006)) (a (.virt 200)) (a (.virt 17449)) (a (.virt 17450)) := by
  have c2050 := (publicBatchWrapper86_copies64 a h).2.2.1
  have c2051 := (publicBatchWrapper86_copies64 a h).2.2.2.1
  have c2052 := (publicBatchWrapper86_copies64 a h).2.2.2.2.1
  have c2053 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.1
  have c2054 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.1
  have c2055 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.1
  have c2056 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.1
  have c2057 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.1
  have c2058 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2059 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2060 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2061 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2062 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2063 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_30_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c2053, ← c2054, ← c2055] at e_30_8
  have e_30_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c2056, ← c2057, ← c2058] at e_30_9
  have e_31_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c2050, k3, ← c2051, k3, ← c2052] at e_31_2
  have e_31_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c2059, ← c2060, k3, ← c2061] at e_31_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_8
    linear_combination c2062.trans k2 - hc
  · have hc := e_31_3
    simp only [e_31_2, e_30_9] at hc
    linear_combination c2063.trans k2 - hc

theorem publicBatchWrapper86_f298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6007)) (a (.virt 200)) (a (.virt 17451)) (a (.virt 17452)) := by
  have c2064 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2065 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2066 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2067 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2068 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2069 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2070 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2071 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2072 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2073 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2074 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2075 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2076 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2077 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_30_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c2067, ← c2068, ← c2069] at e_30_10
  have e_30_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c2070, ← c2071, ← c2072] at e_30_11
  have e_31_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c2064, k3, ← c2065, k3, ← c2066] at e_31_4
  have e_31_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c2073, ← c2074, k3, ← c2075] at e_31_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_10
    linear_combination c2076.trans k2 - hc
  · have hc := e_31_5
    simp only [e_31_4, e_30_11] at hc
    linear_combination c2077.trans k2 - hc

theorem publicBatchWrapper86_f299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6008)) (a (.virt 200)) (a (.virt 17453)) (a (.virt 17454)) := by
  have c2078 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2079 := (publicBatchWrapper86_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2080 := (publicBatchWrapper86_copies65 a h).1
  have c2081 := (publicBatchWrapper86_copies65 a h).2.1
  have c2082 := (publicBatchWrapper86_copies65 a h).2.2.1
  have c2083 := (publicBatchWrapper86_copies65 a h).2.2.2.1
  have c2084 := (publicBatchWrapper86_copies65 a h).2.2.2.2.1
  have c2085 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.1
  have c2086 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.1
  have c2087 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.1
  have c2088 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.1
  have c2089 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.1
  have c2090 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2091 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_30_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c2081, ← c2082, ← c2083] at e_30_12
  have e_30_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c2084, ← c2085, ← c2086] at e_30_13
  have e_31_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c2078, k3, ← c2079, k3, ← c2080] at e_31_6
  have e_31_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c2087, ← c2088, k3, ← c2089] at e_31_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_12
    linear_combination c2090.trans k2 - hc
  · have hc := e_31_7
    simp only [e_31_6, e_30_13] at hc
    linear_combination c2091.trans k2 - hc

theorem publicBatchWrapper86_f300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 30 59) = band (a (.virt 17447)) (a (.virt 17449)) := by
  have c2092 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2093 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2094 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c2092, ← c2093, ← c2094] at e_30_14
  have hr := e_30_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 30 63) = band (a (.virt 17451)) (a (.virt 17453)) := by
  have c2095 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2096 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2097 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_15
  simp only [← c2095, ← c2096, ← c2097] at e_30_15
  have hr := e_30_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 30 67) = band (a (.wire 30 59)) (a (.wire 30 63)) := by
  have c2098 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2099 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2100 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_16
  simp only [← c2098, ← c2099, ← c2100] at e_30_16
  have hr := e_30_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6205)) (a (.virt 200)) (a (.virt 17455)) (a (.virt 17456)) := by
  have c2101 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2102 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2103 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2104 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2105 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2106 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2107 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2108 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2109 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2110 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2111 := (publicBatchWrapper86_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2112 := (publicBatchWrapper86_copies66 a h).1
  have c2113 := (publicBatchWrapper86_copies66 a h).2.1
  have c2114 := (publicBatchWrapper86_copies66 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_30_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_17
  simp only [← c2104, ← c2105, ← c2106] at e_30_17
  have e_30_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_18
  simp only [← c2107, ← c2108, ← c2109] at e_30_18
  have e_31_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c2101, k3, ← c2102, k3, ← c2103] at e_31_8
  have e_31_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c2110, ← c2111, k3, ← c2112] at e_31_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_17
    linear_combination c2113.trans k2 - hc
  · have hc := e_31_9
    simp only [e_31_8, e_30_18] at hc
    linear_combination c2114.trans k2 - hc

theorem publicBatchWrapper86_f304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6206)) (a (.virt 200)) (a (.virt 17457)) (a (.virt 17458)) := by
  have c2115 := (publicBatchWrapper86_copies66 a h).2.2.2.1
  have c2116 := (publicBatchWrapper86_copies66 a h).2.2.2.2.1
  have c2117 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.1
  have c2118 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.1
  have c2119 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.1
  have c2120 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.1
  have c2121 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.1
  have c2122 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2123 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2124 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2125 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2126 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2127 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2128 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_30_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row30 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_19
  simp only [← c2118, ← c2119, ← c2120] at e_30_19
  have e_31_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c2115, k3, ← c2116, k3, ← c2117] at e_31_10
  have e_31_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c2124, ← c2125, k3, ← c2126] at e_31_11
  have e_32_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_0
  simp only [← c2121, ← c2122, ← c2123] at e_32_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_30_19
    linear_combination c2127.trans k2 - hc
  · have hc := e_31_11
    simp only [e_31_10, e_32_0] at hc
    linear_combination c2128.trans k2 - hc

theorem publicBatchWrapper86_f305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6207)) (a (.virt 200)) (a (.virt 17459)) (a (.virt 17460)) := by
  have c2129 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2130 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2131 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2132 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2133 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2134 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2135 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2136 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2137 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2138 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2139 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2140 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2141 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2142 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_31_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c2129, k3, ← c2130, k3, ← c2131] at e_31_12
  have e_31_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c2138, ← c2139, k3, ← c2140] at e_31_13
  have e_32_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_1
  simp only [← c2132, ← c2133, ← c2134] at e_32_1
  have e_32_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_2
  simp only [← c2135, ← c2136, ← c2137] at e_32_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_1
    linear_combination c2141.trans k2 - hc
  · have hc := e_31_13
    simp only [e_31_12, e_32_2] at hc
    linear_combination c2142.trans k2 - hc

theorem publicBatchWrapper86_f306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6208)) (a (.virt 200)) (a (.virt 17461)) (a (.virt 17462)) := by
  have c2143 := (publicBatchWrapper86_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2144 := (publicBatchWrapper86_copies67 a h).1
  have c2145 := (publicBatchWrapper86_copies67 a h).2.1
  have c2146 := (publicBatchWrapper86_copies67 a h).2.2.1
  have c2147 := (publicBatchWrapper86_copies67 a h).2.2.2.1
  have c2148 := (publicBatchWrapper86_copies67 a h).2.2.2.2.1
  have c2149 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.1
  have c2150 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.1
  have c2151 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.1
  have c2152 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.1
  have c2153 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.1
  have c2154 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2155 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2156 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_31_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c2143, k3, ← c2144, k3, ← c2145] at e_31_14
  have e_31_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_15
  simp only [← c2152, ← c2153, k3, ← c2154] at e_31_15
  have e_32_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_3
  simp only [← c2146, ← c2147, ← c2148] at e_32_3
  have e_32_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_4
  simp only [← c2149, ← c2150, ← c2151] at e_32_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_3
    linear_combination c2155.trans k2 - hc
  · have hc := e_31_15
    simp only [e_31_14, e_32_4] at hc
    linear_combination c2156.trans k2 - hc

theorem publicBatchWrapper86_f307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 32 23) = band (a (.virt 17455)) (a (.virt 17457)) := by
  have c2157 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2158 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2159 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_5
  simp only [← c2157, ← c2158, ← c2159] at e_32_5
  have hr := e_32_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 32 27) = band (a (.virt 17459)) (a (.virt 17461)) := by
  have c2160 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2161 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2162 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_6
  simp only [← c2160, ← c2161, ← c2162] at e_32_6
  have hr := e_32_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 32 31) = band (a (.wire 32 23)) (a (.wire 32 27)) := by
  have c2163 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2164 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2165 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_7
  simp only [← c2163, ← c2164, ← c2165] at e_32_7
  have hr := e_32_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6405)) (a (.virt 200)) (a (.virt 17463)) (a (.virt 17464)) := by
  have c2166 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2167 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2168 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2169 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2170 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2171 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2172 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2173 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2174 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2175 := (publicBatchWrapper86_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2176 := (publicBatchWrapper86_copies68 a h).1
  have c2177 := (publicBatchWrapper86_copies68 a h).2.1
  have c2178 := (publicBatchWrapper86_copies68 a h).2.2.1
  have c2179 := (publicBatchWrapper86_copies68 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_31_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_16
  simp only [← c2166, k3, ← c2167, k3, ← c2168] at e_31_16
  have e_31_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_17
  simp only [← c2175, ← c2176, k3, ← c2177] at e_31_17
  have e_32_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_8
  simp only [← c2169, ← c2170, ← c2171] at e_32_8
  have e_32_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_9
  simp only [← c2172, ← c2173, ← c2174] at e_32_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_8
    linear_combination c2178.trans k2 - hc
  · have hc := e_31_17
    simp only [e_31_16, e_32_9] at hc
    linear_combination c2179.trans k2 - hc

theorem publicBatchWrapper86_f311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6406)) (a (.virt 200)) (a (.virt 17465)) (a (.virt 17466)) := by
  have c2180 := (publicBatchWrapper86_copies68 a h).2.2.2.2.1
  have c2181 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.1
  have c2182 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.1
  have c2183 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.1
  have c2184 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.1
  have c2185 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.1
  have c2186 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2187 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2188 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2189 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2190 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2191 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2192 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2193 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_31_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_18
  simp only [← c2180, k3, ← c2181, k3, ← c2182] at e_31_18
  have e_31_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row31 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_19
  simp only [← c2189, ← c2190, k3, ← c2191] at e_31_19
  have e_32_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_10
  simp only [← c2183, ← c2184, ← c2185] at e_32_10
  have e_32_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_11
  simp only [← c2186, ← c2187, ← c2188] at e_32_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_10
    linear_combination c2192.trans k2 - hc
  · have hc := e_31_19
    simp only [e_31_18, e_32_11] at hc
    linear_combination c2193.trans k2 - hc

theorem publicBatchWrapper86_f312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6407)) (a (.virt 200)) (a (.virt 17467)) (a (.virt 17468)) := by
  have c2194 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2195 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2196 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2197 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2198 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2199 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2200 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2201 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2202 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2203 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2204 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2205 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2206 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2207 := (publicBatchWrapper86_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_32_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_12
  simp only [← c2197, ← c2198, ← c2199] at e_32_12
  have e_32_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_13
  simp only [← c2200, ← c2201, ← c2202] at e_32_13
  have e_33_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c2194, k3, ← c2195, k3, ← c2196] at e_33_0
  have e_33_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c2203, ← c2204, k3, ← c2205] at e_33_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_12
    linear_combination c2206.trans k2 - hc
  · have hc := e_33_1
    simp only [e_33_0, e_32_13] at hc
    linear_combination c2207.trans k2 - hc

theorem publicBatchWrapper86_f313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6408)) (a (.virt 200)) (a (.virt 17469)) (a (.virt 17470)) := by
  have c2208 := (publicBatchWrapper86_copies69 a h).1
  have c2209 := (publicBatchWrapper86_copies69 a h).2.1
  have c2210 := (publicBatchWrapper86_copies69 a h).2.2.1
  have c2211 := (publicBatchWrapper86_copies69 a h).2.2.2.1
  have c2212 := (publicBatchWrapper86_copies69 a h).2.2.2.2.1
  have c2213 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.1
  have c2214 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.1
  have c2215 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.1
  have c2216 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.1
  have c2217 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.1
  have c2218 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2219 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2220 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2221 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_32_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_14
  simp only [← c2211, ← c2212, ← c2213] at e_32_14
  have e_32_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_15
  simp only [← c2214, ← c2215, ← c2216] at e_32_15
  have e_33_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c2208, k3, ← c2209, k3, ← c2210] at e_33_2
  have e_33_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c2217, ← c2218, k3, ← c2219] at e_33_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_14
    linear_combination c2220.trans k2 - hc
  · have hc := e_33_3
    simp only [e_33_2, e_32_15] at hc
    linear_combination c2221.trans k2 - hc

theorem publicBatchWrapper86_f314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 32 67) = band (a (.virt 17463)) (a (.virt 17465)) := by
  have c2222 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2223 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2224 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_16
  simp only [← c2222, ← c2223, ← c2224] at e_32_16
  have hr := e_32_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 32 71) = band (a (.virt 17467)) (a (.virt 17469)) := by
  have c2225 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2226 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2227 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_17
  simp only [← c2225, ← c2226, ← c2227] at e_32_17
  have hr := e_32_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 32 75) = band (a (.wire 32 67)) (a (.wire 32 71)) := by
  have c2228 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2229 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2230 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_18
  simp only [← c2228, ← c2229, ← c2230] at e_32_18
  have hr := e_32_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6605)) (a (.virt 200)) (a (.virt 17471)) (a (.virt 17472)) := by
  have c2231 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2232 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2233 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2234 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2235 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2236 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2237 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2238 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2239 := (publicBatchWrapper86_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2240 := (publicBatchWrapper86_copies70 a h).1
  have c2241 := (publicBatchWrapper86_copies70 a h).2.1
  have c2242 := (publicBatchWrapper86_copies70 a h).2.2.1
  have c2243 := (publicBatchWrapper86_copies70 a h).2.2.2.1
  have c2244 := (publicBatchWrapper86_copies70 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_32_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row32 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_19
  simp only [← c2234, ← c2235, ← c2236] at e_32_19
  have e_33_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c2231, k3, ← c2232, k3, ← c2233] at e_33_4
  have e_33_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c2240, ← c2241, k3, ← c2242] at e_33_5
  have e_34_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c2237, ← c2238, ← c2239] at e_34_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_32_19
    linear_combination c2243.trans k2 - hc
  · have hc := e_33_5
    simp only [e_33_4, e_34_0] at hc
    linear_combination c2244.trans k2 - hc

theorem publicBatchWrapper86_f318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6606)) (a (.virt 200)) (a (.virt 17473)) (a (.virt 17474)) := by
  have c2245 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.1
  have c2246 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.1
  have c2247 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.1
  have c2248 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.1
  have c2249 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.1
  have c2250 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2251 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2252 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2253 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2254 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2255 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2256 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2257 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2258 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c2245, k3, ← c2246, k3, ← c2247] at e_33_6
  have e_33_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c2254, ← c2255, k3, ← c2256] at e_33_7
  have e_34_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_1
  simp only [← c2248, ← c2249, ← c2250] at e_34_1
  have e_34_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_2
  simp only [← c2251, ← c2252, ← c2253] at e_34_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_1
    linear_combination c2257.trans k2 - hc
  · have hc := e_33_7
    simp only [e_33_6, e_34_2] at hc
    linear_combination c2258.trans k2 - hc

theorem publicBatchWrapper86_f319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6607)) (a (.virt 200)) (a (.virt 17475)) (a (.virt 17476)) := by
  have c2259 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2260 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2261 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2262 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2263 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2264 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2265 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2266 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2267 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2268 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2269 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2270 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2271 := (publicBatchWrapper86_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2272 := (publicBatchWrapper86_copies71 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c2259, k3, ← c2260, k3, ← c2261] at e_33_8
  have e_33_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c2268, ← c2269, k3, ← c2270] at e_33_9
  have e_34_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c2262, ← c2263, ← c2264] at e_34_3
  have e_34_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_4
  simp only [← c2265, ← c2266, ← c2267] at e_34_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_3
    linear_combination c2271.trans k2 - hc
  · have hc := e_33_9
    simp only [e_33_8, e_34_4] at hc
    linear_combination c2272.trans k2 - hc

theorem publicBatchWrapper86_f320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6608)) (a (.virt 200)) (a (.virt 17477)) (a (.virt 17478)) := by
  have c2273 := (publicBatchWrapper86_copies71 a h).2.1
  have c2274 := (publicBatchWrapper86_copies71 a h).2.2.1
  have c2275 := (publicBatchWrapper86_copies71 a h).2.2.2.1
  have c2276 := (publicBatchWrapper86_copies71 a h).2.2.2.2.1
  have c2277 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.1
  have c2278 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.1
  have c2279 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.1
  have c2280 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.1
  have c2281 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.1
  have c2282 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2283 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2284 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2285 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2286 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c2273, k3, ← c2274, k3, ← c2275] at e_33_10
  have e_33_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c2282, ← c2283, k3, ← c2284] at e_33_11
  have e_34_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_5
  simp only [← c2276, ← c2277, ← c2278] at e_34_5
  have e_34_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_6
  simp only [← c2279, ← c2280, ← c2281] at e_34_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_5
    linear_combination c2285.trans k2 - hc
  · have hc := e_33_11
    simp only [e_33_10, e_34_6] at hc
    linear_combination c2286.trans k2 - hc

theorem publicBatchWrapper86_f321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 34 31) = band (a (.virt 17471)) (a (.virt 17473)) := by
  have c2287 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2288 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2289 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_7
  simp only [← c2287, ← c2288, ← c2289] at e_34_7
  have hr := e_34_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 34 35) = band (a (.virt 17475)) (a (.virt 17477)) := by
  have c2290 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2291 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2292 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_8
  simp only [← c2290, ← c2291, ← c2292] at e_34_8
  have hr := e_34_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 34 39) = band (a (.wire 34 31)) (a (.wire 34 35)) := by
  have c2293 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2294 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2295 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_9
  simp only [← c2293, ← c2294, ← c2295] at e_34_9
  have hr := e_34_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6805)) (a (.virt 200)) (a (.virt 17479)) (a (.virt 17480)) := by
  have c2296 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2297 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2298 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2299 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2300 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2301 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2302 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2303 := (publicBatchWrapper86_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2304 := (publicBatchWrapper86_copies72 a h).1
  have c2305 := (publicBatchWrapper86_copies72 a h).2.1
  have c2306 := (publicBatchWrapper86_copies72 a h).2.2.1
  have c2307 := (publicBatchWrapper86_copies72 a h).2.2.2.1
  have c2308 := (publicBatchWrapper86_copies72 a h).2.2.2.2.1
  have c2309 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c2296, k3, ← c2297, k3, ← c2298] at e_33_12
  have e_33_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c2305, ← c2306, k3, ← c2307] at e_33_13
  have e_34_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_10
  simp only [← c2299, ← c2300, ← c2301] at e_34_10
  have e_34_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_11
  simp only [← c2302, ← c2303, ← c2304] at e_34_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_10
    linear_combination c2308.trans k2 - hc
  · have hc := e_33_13
    simp only [e_33_12, e_34_11] at hc
    linear_combination c2309.trans k2 - hc

theorem publicBatchWrapper86_f325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6806)) (a (.virt 200)) (a (.virt 17481)) (a (.virt 17482)) := by
  have c2310 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.1
  have c2311 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.1
  have c2312 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.1
  have c2313 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.1
  have c2314 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2315 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2316 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2317 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2318 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2319 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2320 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2321 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2322 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2323 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c2310, k3, ← c2311, k3, ← c2312] at e_33_14
  have e_33_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_15
  simp only [← c2319, ← c2320, k3, ← c2321] at e_33_15
  have e_34_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_12
  simp only [← c2313, ← c2314, ← c2315] at e_34_12
  have e_34_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_13
  simp only [← c2316, ← c2317, ← c2318] at e_34_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_12
    linear_combination c2322.trans k2 - hc
  · have hc := e_33_15
    simp only [e_33_14, e_34_13] at hc
    linear_combination c2323.trans k2 - hc

theorem publicBatchWrapper86_f326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6807)) (a (.virt 200)) (a (.virt 17483)) (a (.virt 17484)) := by
  have c2324 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2325 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2326 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2327 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2328 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2329 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2330 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2331 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2332 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2333 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2334 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2335 := (publicBatchWrapper86_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2336 := (publicBatchWrapper86_copies73 a h).1
  have c2337 := (publicBatchWrapper86_copies73 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_16
  simp only [← c2324, k3, ← c2325, k3, ← c2326] at e_33_16
  have e_33_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_17
  simp only [← c2333, ← c2334, k3, ← c2335] at e_33_17
  have e_34_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_14
  simp only [← c2327, ← c2328, ← c2329] at e_34_14
  have e_34_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_15
  simp only [← c2330, ← c2331, ← c2332] at e_34_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_14
    linear_combination c2336.trans k2 - hc
  · have hc := e_33_17
    simp only [e_33_16, e_34_15] at hc
    linear_combination c2337.trans k2 - hc

theorem publicBatchWrapper86_f327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 6808)) (a (.virt 200)) (a (.virt 17485)) (a (.virt 17486)) := by
  have c2338 := (publicBatchWrapper86_copies73 a h).2.2.1
  have c2339 := (publicBatchWrapper86_copies73 a h).2.2.2.1
  have c2340 := (publicBatchWrapper86_copies73 a h).2.2.2.2.1
  have c2341 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.1
  have c2342 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.1
  have c2343 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.1
  have c2344 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.1
  have c2345 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.1
  have c2346 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2347 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2348 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2349 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2350 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2351 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_33_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_18
  simp only [← c2338, k3, ← c2339, k3, ← c2340] at e_33_18
  have e_33_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row33 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_19
  simp only [← c2347, ← c2348, k3, ← c2349] at e_33_19
  have e_34_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_16
  simp only [← c2341, ← c2342, ← c2343] at e_34_16
  have e_34_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_17
  simp only [← c2344, ← c2345, ← c2346] at e_34_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_16
    linear_combination c2350.trans k2 - hc
  · have hc := e_33_19
    simp only [e_33_18, e_34_17] at hc
    linear_combination c2351.trans k2 - hc

theorem publicBatchWrapper86_f328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 34 75) = band (a (.virt 17479)) (a (.virt 17481)) := by
  have c2352 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2353 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2354 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_18
  simp only [← c2352, ← c2353, ← c2354] at e_34_18
  have hr := e_34_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 34 79) = band (a (.virt 17483)) (a (.virt 17485)) := by
  have c2355 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2356 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2357 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row34 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_19
  simp only [← c2355, ← c2356, ← c2357] at e_34_19
  have hr := e_34_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 35 3) = band (a (.wire 34 75)) (a (.wire 34 79)) := by
  have c2358 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2359 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2360 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_0
  simp only [← c2358, ← c2359, ← c2360] at e_35_0
  have hr := e_35_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7005)) (a (.virt 200)) (a (.virt 17487)) (a (.virt 17488)) := by
  have c2361 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2362 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2363 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2364 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2365 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2366 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2367 := (publicBatchWrapper86_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2368 := (publicBatchWrapper86_copies74 a h).1
  have c2369 := (publicBatchWrapper86_copies74 a h).2.1
  have c2370 := (publicBatchWrapper86_copies74 a h).2.2.1
  have c2371 := (publicBatchWrapper86_copies74 a h).2.2.2.1
  have c2372 := (publicBatchWrapper86_copies74 a h).2.2.2.2.1
  have c2373 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.1
  have c2374 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_1
  simp only [← c2364, ← c2365, ← c2366] at e_35_1
  have e_35_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_2
  simp only [← c2367, ← c2368, ← c2369] at e_35_2
  have e_36_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_0
  simp only [← c2361, k3, ← c2362, k3, ← c2363] at e_36_0
  have e_36_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_1
  simp only [← c2370, ← c2371, k3, ← c2372] at e_36_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_1
    linear_combination c2373.trans k2 - hc
  · have hc := e_36_1
    simp only [e_36_0, e_35_2] at hc
    linear_combination c2374.trans k2 - hc

theorem publicBatchWrapper86_f332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7006)) (a (.virt 200)) (a (.virt 17489)) (a (.virt 17490)) := by
  have c2375 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.1
  have c2376 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.1
  have c2377 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.1
  have c2378 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2379 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2380 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2381 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2382 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2383 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2384 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2385 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2386 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2387 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2388 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_3
  simp only [← c2378, ← c2379, ← c2380] at e_35_3
  have e_35_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_4
  simp only [← c2381, ← c2382, ← c2383] at e_35_4
  have e_36_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_2
  simp only [← c2375, k3, ← c2376, k3, ← c2377] at e_36_2
  have e_36_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_3
  simp only [← c2384, ← c2385, k3, ← c2386] at e_36_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_3
    linear_combination c2387.trans k2 - hc
  · have hc := e_36_3
    simp only [e_36_2, e_35_4] at hc
    linear_combination c2388.trans k2 - hc

theorem publicBatchWrapper86_f333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7007)) (a (.virt 200)) (a (.virt 17491)) (a (.virt 17492)) := by
  have c2389 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2390 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2391 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2392 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2393 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2394 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2395 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2396 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2397 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2398 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2399 := (publicBatchWrapper86_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2400 := (publicBatchWrapper86_copies75 a h).1
  have c2401 := (publicBatchWrapper86_copies75 a h).2.1
  have c2402 := (publicBatchWrapper86_copies75 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_5
  simp only [← c2392, ← c2393, ← c2394] at e_35_5
  have e_35_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_6
  simp only [← c2395, ← c2396, ← c2397] at e_35_6
  have e_36_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_4
  simp only [← c2389, k3, ← c2390, k3, ← c2391] at e_36_4
  have e_36_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_5
  simp only [← c2398, ← c2399, k3, ← c2400] at e_36_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_5
    linear_combination c2401.trans k2 - hc
  · have hc := e_36_5
    simp only [e_36_4, e_35_6] at hc
    linear_combination c2402.trans k2 - hc

theorem publicBatchWrapper86_f334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7008)) (a (.virt 200)) (a (.virt 17493)) (a (.virt 17494)) := by
  have c2403 := (publicBatchWrapper86_copies75 a h).2.2.2.1
  have c2404 := (publicBatchWrapper86_copies75 a h).2.2.2.2.1
  have c2405 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.1
  have c2406 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.1
  have c2407 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.1
  have c2408 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.1
  have c2409 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.1
  have c2410 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2411 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2412 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2413 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2414 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2415 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2416 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_7
  simp only [← c2406, ← c2407, ← c2408] at e_35_7
  have e_35_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_8
  simp only [← c2409, ← c2410, ← c2411] at e_35_8
  have e_36_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_6
  simp only [← c2403, k3, ← c2404, k3, ← c2405] at e_36_6
  have e_36_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_7
  simp only [← c2412, ← c2413, k3, ← c2414] at e_36_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_7
    linear_combination c2415.trans k2 - hc
  · have hc := e_36_7
    simp only [e_36_6, e_35_8] at hc
    linear_combination c2416.trans k2 - hc

theorem publicBatchWrapper86_f335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 35 39) = band (a (.virt 17487)) (a (.virt 17489)) := by
  have c2417 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2418 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2419 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_9
  simp only [← c2417, ← c2418, ← c2419] at e_35_9
  have hr := e_35_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 35 43) = band (a (.virt 17491)) (a (.virt 17493)) := by
  have c2420 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2421 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2422 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_10
  simp only [← c2420, ← c2421, ← c2422] at e_35_10
  have hr := e_35_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 35 47) = band (a (.wire 35 39)) (a (.wire 35 43)) := by
  have c2423 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2424 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2425 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_11
  simp only [← c2423, ← c2424, ← c2425] at e_35_11
  have hr := e_35_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7205)) (a (.virt 200)) (a (.virt 17495)) (a (.virt 17496)) := by
  have c2426 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2427 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2428 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2429 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2430 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2431 := (publicBatchWrapper86_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2432 := (publicBatchWrapper86_copies76 a h).1
  have c2433 := (publicBatchWrapper86_copies76 a h).2.1
  have c2434 := (publicBatchWrapper86_copies76 a h).2.2.1
  have c2435 := (publicBatchWrapper86_copies76 a h).2.2.2.1
  have c2436 := (publicBatchWrapper86_copies76 a h).2.2.2.2.1
  have c2437 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.1
  have c2438 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.1
  have c2439 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_12
  simp only [← c2429, ← c2430, ← c2431] at e_35_12
  have e_35_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_13
  simp only [← c2432, ← c2433, ← c2434] at e_35_13
  have e_36_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_8
  simp only [← c2426, k3, ← c2427, k3, ← c2428] at e_36_8
  have e_36_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_9
  simp only [← c2435, ← c2436, k3, ← c2437] at e_36_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_12
    linear_combination c2438.trans k2 - hc
  · have hc := e_36_9
    simp only [e_36_8, e_35_13] at hc
    linear_combination c2439.trans k2 - hc

theorem publicBatchWrapper86_f339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7206)) (a (.virt 200)) (a (.virt 17497)) (a (.virt 17498)) := by
  have c2440 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.1
  have c2441 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.1
  have c2442 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2443 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2444 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2445 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2446 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2447 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2448 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2449 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2450 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2451 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2452 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2453 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_14
  simp only [← c2443, ← c2444, ← c2445] at e_35_14
  have e_35_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_15
  simp only [← c2446, ← c2447, ← c2448] at e_35_15
  have e_36_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_10
  simp only [← c2440, k3, ← c2441, k3, ← c2442] at e_36_10
  have e_36_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_11
  simp only [← c2449, ← c2450, k3, ← c2451] at e_36_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_14
    linear_combination c2452.trans k2 - hc
  · have hc := e_36_11
    simp only [e_36_10, e_35_15] at hc
    linear_combination c2453.trans k2 - hc

theorem publicBatchWrapper86_f340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7207)) (a (.virt 200)) (a (.virt 17499)) (a (.virt 17500)) := by
  have c2454 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2455 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2456 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2457 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2458 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2459 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2460 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2461 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2462 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2463 := (publicBatchWrapper86_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2464 := (publicBatchWrapper86_copies77 a h).1
  have c2465 := (publicBatchWrapper86_copies77 a h).2.1
  have c2466 := (publicBatchWrapper86_copies77 a h).2.2.1
  have c2467 := (publicBatchWrapper86_copies77 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_16
  simp only [← c2457, ← c2458, ← c2459] at e_35_16
  have e_35_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_17
  simp only [← c2460, ← c2461, ← c2462] at e_35_17
  have e_36_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_12
  simp only [← c2454, k3, ← c2455, k3, ← c2456] at e_36_12
  have e_36_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_13
  simp only [← c2463, ← c2464, k3, ← c2465] at e_36_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_16
    linear_combination c2466.trans k2 - hc
  · have hc := e_36_13
    simp only [e_36_12, e_35_17] at hc
    linear_combination c2467.trans k2 - hc

theorem publicBatchWrapper86_f341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7208)) (a (.virt 200)) (a (.virt 17501)) (a (.virt 17502)) := by
  have c2468 := (publicBatchWrapper86_copies77 a h).2.2.2.2.1
  have c2469 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.1
  have c2470 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.1
  have c2471 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.1
  have c2472 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.1
  have c2473 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.1
  have c2474 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2475 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2476 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2477 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2478 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2479 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2480 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2481 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_35_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_18
  simp only [← c2471, ← c2472, ← c2473] at e_35_18
  have e_35_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row35 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_19
  simp only [← c2474, ← c2475, ← c2476] at e_35_19
  have e_36_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_14
  simp only [← c2468, k3, ← c2469, k3, ← c2470] at e_36_14
  have e_36_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_15
  simp only [← c2477, ← c2478, k3, ← c2479] at e_36_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_35_18
    linear_combination c2480.trans k2 - hc
  · have hc := e_36_15
    simp only [e_36_14, e_35_19] at hc
    linear_combination c2481.trans k2 - hc

theorem publicBatchWrapper86_f342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 37 3) = band (a (.virt 17495)) (a (.virt 17497)) := by
  have c2482 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2483 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2484 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_0
  simp only [← c2482, ← c2483, ← c2484] at e_37_0
  have hr := e_37_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 37 7) = band (a (.virt 17499)) (a (.virt 17501)) := by
  have c2485 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2486 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2487 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_1
  simp only [← c2485, ← c2486, ← c2487] at e_37_1
  have hr := e_37_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 37 11) = band (a (.wire 37 3)) (a (.wire 37 7)) := by
  have c2488 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2489 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2490 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_2
  simp only [← c2488, ← c2489, ← c2490] at e_37_2
  have hr := e_37_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7405)) (a (.virt 200)) (a (.virt 17503)) (a (.virt 17504)) := by
  have c2491 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2492 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2493 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2494 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2495 := (publicBatchWrapper86_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2496 := (publicBatchWrapper86_copies78 a h).1
  have c2497 := (publicBatchWrapper86_copies78 a h).2.1
  have c2498 := (publicBatchWrapper86_copies78 a h).2.2.1
  have c2499 := (publicBatchWrapper86_copies78 a h).2.2.2.1
  have c2500 := (publicBatchWrapper86_copies78 a h).2.2.2.2.1
  have c2501 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.1
  have c2502 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.1
  have c2503 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.1
  have c2504 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_36_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_16
  simp only [← c2491, k3, ← c2492, k3, ← c2493] at e_36_16
  have e_36_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_17
  simp only [← c2500, ← c2501, k3, ← c2502] at e_36_17
  have e_37_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_3
  simp only [← c2494, ← c2495, ← c2496] at e_37_3
  have e_37_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_4
  simp only [← c2497, ← c2498, ← c2499] at e_37_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_3
    linear_combination c2503.trans k2 - hc
  · have hc := e_36_17
    simp only [e_36_16, e_37_4] at hc
    linear_combination c2504.trans k2 - hc

theorem publicBatchWrapper86_f346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7406)) (a (.virt 200)) (a (.virt 17505)) (a (.virt 17506)) := by
  have c2505 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.1
  have c2506 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2507 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2508 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2509 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2510 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2511 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2512 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2513 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2514 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2515 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2516 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2517 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2518 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_36_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_18
  simp only [← c2505, k3, ← c2506, k3, ← c2507] at e_36_18
  have e_36_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row36 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_19
  simp only [← c2514, ← c2515, k3, ← c2516] at e_36_19
  have e_37_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_5
  simp only [← c2508, ← c2509, ← c2510] at e_37_5
  have e_37_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_6
  simp only [← c2511, ← c2512, ← c2513] at e_37_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_5
    linear_combination c2517.trans k2 - hc
  · have hc := e_36_19
    simp only [e_36_18, e_37_6] at hc
    linear_combination c2518.trans k2 - hc

theorem publicBatchWrapper86_f347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7407)) (a (.virt 200)) (a (.virt 17507)) (a (.virt 17508)) := by
  have c2519 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2522 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2524 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2525 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2526 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2527 := (publicBatchWrapper86_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2528 := (publicBatchWrapper86_copies79 a h).1
  have c2529 := (publicBatchWrapper86_copies79 a h).2.1
  have c2530 := (publicBatchWrapper86_copies79 a h).2.2.1
  have c2531 := (publicBatchWrapper86_copies79 a h).2.2.2.1
  have c2532 := (publicBatchWrapper86_copies79 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_37_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_7
  simp only [← c2522, ← c2523, ← c2524] at e_37_7
  have e_37_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_8
  simp only [← c2525, ← c2526, ← c2527] at e_37_8
  have e_38_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_0
  simp only [← c2519, k3, ← c2520, k3, ← c2521] at e_38_0
  have e_38_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_1
  simp only [← c2528, ← c2529, k3, ← c2530] at e_38_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_7
    linear_combination c2531.trans k2 - hc
  · have hc := e_38_1
    simp only [e_38_0, e_37_8] at hc
    linear_combination c2532.trans k2 - hc

theorem publicBatchWrapper86_f348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7408)) (a (.virt 200)) (a (.virt 17509)) (a (.virt 17510)) := by
  have c2533 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.1
  have c2534 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.1
  have c2535 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.1
  have c2536 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.1
  have c2537 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.1
  have c2538 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2539 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2540 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2541 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2542 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2543 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2544 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2545 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2546 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_37_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_9
  simp only [← c2536, ← c2537, ← c2538] at e_37_9
  have e_37_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_10
  simp only [← c2539, ← c2540, ← c2541] at e_37_10
  have e_38_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_2
  simp only [← c2533, k3, ← c2534, k3, ← c2535] at e_38_2
  have e_38_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_3
  simp only [← c2542, ← c2543, k3, ← c2544] at e_38_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_9
    linear_combination c2545.trans k2 - hc
  · have hc := e_38_3
    simp only [e_38_2, e_37_10] at hc
    linear_combination c2546.trans k2 - hc

theorem publicBatchWrapper86_f349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 37 47) = band (a (.virt 17503)) (a (.virt 17505)) := by
  have c2547 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2548 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2549 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_11
  simp only [← c2547, ← c2548, ← c2549] at e_37_11
  have hr := e_37_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 37 51) = band (a (.virt 17507)) (a (.virt 17509)) := by
  have c2550 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2551 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2552 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_12
  simp only [← c2550, ← c2551, ← c2552] at e_37_12
  have hr := e_37_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 37 55) = band (a (.wire 37 47)) (a (.wire 37 51)) := by
  have c2553 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2554 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2555 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_13
  simp only [← c2553, ← c2554, ← c2555] at e_37_13
  have hr := e_37_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7605)) (a (.virt 200)) (a (.virt 17511)) (a (.virt 17512)) := by
  have c2556 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2557 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2558 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2559 := (publicBatchWrapper86_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2560 := (publicBatchWrapper86_copies80 a h).1
  have c2561 := (publicBatchWrapper86_copies80 a h).2.1
  have c2562 := (publicBatchWrapper86_copies80 a h).2.2.1
  have c2563 := (publicBatchWrapper86_copies80 a h).2.2.2.1
  have c2564 := (publicBatchWrapper86_copies80 a h).2.2.2.2.1
  have c2565 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.1
  have c2566 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.1
  have c2567 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.1
  have c2568 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.1
  have c2569 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_37_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_14
  simp only [← c2559, ← c2560, ← c2561] at e_37_14
  have e_37_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_15
  simp only [← c2562, ← c2563, ← c2564] at e_37_15
  have e_38_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_4
  simp only [← c2556, k3, ← c2557, k3, ← c2558] at e_38_4
  have e_38_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_5
  simp only [← c2565, ← c2566, k3, ← c2567] at e_38_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_14
    linear_combination c2568.trans k2 - hc
  · have hc := e_38_5
    simp only [e_38_4, e_37_15] at hc
    linear_combination c2569.trans k2 - hc

theorem publicBatchWrapper86_f353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7606)) (a (.virt 200)) (a (.virt 17513)) (a (.virt 17514)) := by
  have c2570 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2571 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2572 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2573 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2574 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2575 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2576 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2577 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2578 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2579 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2580 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2581 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2582 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2583 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_37_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_16
  simp only [← c2573, ← c2574, ← c2575] at e_37_16
  have e_37_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_17
  simp only [← c2576, ← c2577, ← c2578] at e_37_17
  have e_38_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_6
  simp only [← c2570, k3, ← c2571, k3, ← c2572] at e_38_6
  have e_38_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_7
  simp only [← c2579, ← c2580, k3, ← c2581] at e_38_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_16
    linear_combination c2582.trans k2 - hc
  · have hc := e_38_7
    simp only [e_38_6, e_37_17] at hc
    linear_combination c2583.trans k2 - hc

theorem publicBatchWrapper86_f354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7607)) (a (.virt 200)) (a (.virt 17515)) (a (.virt 17516)) := by
  have c2584 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2585 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2586 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2587 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2588 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2589 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2590 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2591 := (publicBatchWrapper86_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2592 := (publicBatchWrapper86_copies81 a h).1
  have c2593 := (publicBatchWrapper86_copies81 a h).2.1
  have c2594 := (publicBatchWrapper86_copies81 a h).2.2.1
  have c2595 := (publicBatchWrapper86_copies81 a h).2.2.2.1
  have c2596 := (publicBatchWrapper86_copies81 a h).2.2.2.2.1
  have c2597 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_37_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_18
  simp only [← c2587, ← c2588, ← c2589] at e_37_18
  have e_37_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row37 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_19
  simp only [← c2590, ← c2591, ← c2592] at e_37_19
  have e_38_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_8
  simp only [← c2584, k3, ← c2585, k3, ← c2586] at e_38_8
  have e_38_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_9
  simp only [← c2593, ← c2594, k3, ← c2595] at e_38_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_37_18
    linear_combination c2596.trans k2 - hc
  · have hc := e_38_9
    simp only [e_38_8, e_37_19] at hc
    linear_combination c2597.trans k2 - hc

theorem publicBatchWrapper86_f355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7608)) (a (.virt 200)) (a (.virt 17517)) (a (.virt 17518)) := by
  have c2598 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.1
  have c2599 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.1
  have c2600 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.1
  have c2601 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.1
  have c2602 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2603 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2604 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2605 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2606 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2607 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2608 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2609 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2610 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2611 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_38_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_10
  simp only [← c2598, k3, ← c2599, k3, ← c2600] at e_38_10
  have e_38_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_11
  simp only [← c2607, ← c2608, k3, ← c2609] at e_38_11
  have e_39_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_0
  simp only [← c2601, ← c2602, ← c2603] at e_39_0
  have e_39_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_1
  simp only [← c2604, ← c2605, ← c2606] at e_39_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_0
    linear_combination c2610.trans k2 - hc
  · have hc := e_38_11
    simp only [e_38_10, e_39_1] at hc
    linear_combination c2611.trans k2 - hc

theorem publicBatchWrapper86_f356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 39 11) = band (a (.virt 17511)) (a (.virt 17513)) := by
  have c2612 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2613 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2614 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_2
  simp only [← c2612, ← c2613, ← c2614] at e_39_2
  have hr := e_39_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 39 15) = band (a (.virt 17515)) (a (.virt 17517)) := by
  have c2615 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2616 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2617 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_3
  simp only [← c2615, ← c2616, ← c2617] at e_39_3
  have hr := e_39_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 39 19) = band (a (.wire 39 11)) (a (.wire 39 15)) := by
  have c2618 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2619 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2620 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_4
  simp only [← c2618, ← c2619, ← c2620] at e_39_4
  have hr := e_39_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7805)) (a (.virt 200)) (a (.virt 17519)) (a (.virt 17520)) := by
  have c2621 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2622 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2623 := (publicBatchWrapper86_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2624 := (publicBatchWrapper86_copies82 a h).1
  have c2625 := (publicBatchWrapper86_copies82 a h).2.1
  have c2626 := (publicBatchWrapper86_copies82 a h).2.2.1
  have c2627 := (publicBatchWrapper86_copies82 a h).2.2.2.1
  have c2628 := (publicBatchWrapper86_copies82 a h).2.2.2.2.1
  have c2629 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.1
  have c2630 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.1
  have c2631 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.1
  have c2632 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.1
  have c2633 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.1
  have c2634 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_38_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_12
  simp only [← c2621, k3, ← c2622, k3, ← c2623] at e_38_12
  have e_38_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_13
  simp only [← c2630, ← c2631, k3, ← c2632] at e_38_13
  have e_39_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_5
  simp only [← c2624, ← c2625, ← c2626] at e_39_5
  have e_39_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_6
  simp only [← c2627, ← c2628, ← c2629] at e_39_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_5
    linear_combination c2633.trans k2 - hc
  · have hc := e_38_13
    simp only [e_38_12, e_39_6] at hc
    linear_combination c2634.trans k2 - hc

theorem publicBatchWrapper86_f360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7806)) (a (.virt 200)) (a (.virt 17521)) (a (.virt 17522)) := by
  have c2635 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2636 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2637 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2638 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2639 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2640 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2641 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2642 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2643 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2644 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2645 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2646 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2647 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2648 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_38_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_14
  simp only [← c2635, k3, ← c2636, k3, ← c2637] at e_38_14
  have e_38_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_15
  simp only [← c2644, ← c2645, k3, ← c2646] at e_38_15
  have e_39_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_7
  simp only [← c2638, ← c2639, ← c2640] at e_39_7
  have e_39_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_8
  simp only [← c2641, ← c2642, ← c2643] at e_39_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_7
    linear_combination c2647.trans k2 - hc
  · have hc := e_38_15
    simp only [e_38_14, e_39_8] at hc
    linear_combination c2648.trans k2 - hc

theorem publicBatchWrapper86_f361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7807)) (a (.virt 200)) (a (.virt 17523)) (a (.virt 17524)) := by
  have c2649 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2650 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2651 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2652 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2653 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2654 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2655 := (publicBatchWrapper86_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2656 := (publicBatchWrapper86_copies83 a h).1
  have c2657 := (publicBatchWrapper86_copies83 a h).2.1
  have c2658 := (publicBatchWrapper86_copies83 a h).2.2.1
  have c2659 := (publicBatchWrapper86_copies83 a h).2.2.2.1
  have c2660 := (publicBatchWrapper86_copies83 a h).2.2.2.2.1
  have c2661 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.1
  have c2662 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_38_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_16
  simp only [← c2649, k3, ← c2650, k3, ← c2651] at e_38_16
  have e_38_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_17
  simp only [← c2658, ← c2659, k3, ← c2660] at e_38_17
  have e_39_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_9
  simp only [← c2652, ← c2653, ← c2654] at e_39_9
  have e_39_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_10
  simp only [← c2655, ← c2656, ← c2657] at e_39_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_9
    linear_combination c2661.trans k2 - hc
  · have hc := e_38_17
    simp only [e_38_16, e_39_10] at hc
    linear_combination c2662.trans k2 - hc

theorem publicBatchWrapper86_f362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 7808)) (a (.virt 200)) (a (.virt 17525)) (a (.virt 17526)) := by
  have c2663 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.1
  have c2664 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.1
  have c2665 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.1
  have c2666 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2667 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2668 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2669 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2670 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2671 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2672 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2673 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2674 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2675 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2676 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_38_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_18
  simp only [← c2663, k3, ← c2664, k3, ← c2665] at e_38_18
  have e_38_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row38 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_19
  simp only [← c2672, ← c2673, k3, ← c2674] at e_38_19
  have e_39_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_11
  simp only [← c2666, ← c2667, ← c2668] at e_39_11
  have e_39_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_12
  simp only [← c2669, ← c2670, ← c2671] at e_39_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_11
    linear_combination c2675.trans k2 - hc
  · have hc := e_38_19
    simp only [e_38_18, e_39_12] at hc
    linear_combination c2676.trans k2 - hc

theorem publicBatchWrapper86_f363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 39 55) = band (a (.virt 17519)) (a (.virt 17521)) := by
  have c2677 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2678 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2679 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_13
  simp only [← c2677, ← c2678, ← c2679] at e_39_13
  have hr := e_39_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 39 59) = band (a (.virt 17523)) (a (.virt 17525)) := by
  have c2680 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2681 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2682 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_14
  simp only [← c2680, ← c2681, ← c2682] at e_39_14
  have hr := e_39_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 39 63) = band (a (.wire 39 55)) (a (.wire 39 59)) := by
  have c2683 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2684 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2685 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_15
  simp only [← c2683, ← c2684, ← c2685] at e_39_15
  have hr := e_39_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8005)) (a (.virt 200)) (a (.virt 17527)) (a (.virt 17528)) := by
  have c2686 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2687 := (publicBatchWrapper86_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2688 := (publicBatchWrapper86_copies84 a h).1
  have c2689 := (publicBatchWrapper86_copies84 a h).2.1
  have c2690 := (publicBatchWrapper86_copies84 a h).2.2.1
  have c2691 := (publicBatchWrapper86_copies84 a h).2.2.2.1
  have c2692 := (publicBatchWrapper86_copies84 a h).2.2.2.2.1
  have c2693 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.1
  have c2694 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.1
  have c2695 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.1
  have c2696 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.1
  have c2697 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.1
  have c2698 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2699 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_39_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_16
  simp only [← c2689, ← c2690, ← c2691] at e_39_16
  have e_39_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_17
  simp only [← c2692, ← c2693, ← c2694] at e_39_17
  have e_40_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_0
  simp only [← c2686, k3, ← c2687, k3, ← c2688] at e_40_0
  have e_40_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_1
  simp only [← c2695, ← c2696, k3, ← c2697] at e_40_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_16
    linear_combination c2698.trans k2 - hc
  · have hc := e_40_1
    simp only [e_40_0, e_39_17] at hc
    linear_combination c2699.trans k2 - hc

theorem publicBatchWrapper86_f367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8006)) (a (.virt 200)) (a (.virt 17529)) (a (.virt 17530)) := by
  have c2700 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2701 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2702 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2703 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2704 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2705 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2706 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2707 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2708 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2709 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2710 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2711 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2712 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2713 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_39_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_18
  simp only [← c2703, ← c2704, ← c2705] at e_39_18
  have e_39_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row39 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_19
  simp only [← c2706, ← c2707, ← c2708] at e_39_19
  have e_40_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_2
  simp only [← c2700, k3, ← c2701, k3, ← c2702] at e_40_2
  have e_40_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_3
  simp only [← c2709, ← c2710, k3, ← c2711] at e_40_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_18
    linear_combination c2712.trans k2 - hc
  · have hc := e_40_3
    simp only [e_40_2, e_39_19] at hc
    linear_combination c2713.trans k2 - hc

theorem publicBatchWrapper86_f368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8007)) (a (.virt 200)) (a (.virt 17531)) (a (.virt 17532)) := by
  have c2714 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2715 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2716 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2717 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2718 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2719 := (publicBatchWrapper86_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2720 := (publicBatchWrapper86_copies85 a h).1
  have c2721 := (publicBatchWrapper86_copies85 a h).2.1
  have c2722 := (publicBatchWrapper86_copies85 a h).2.2.1
  have c2723 := (publicBatchWrapper86_copies85 a h).2.2.2.1
  have c2724 := (publicBatchWrapper86_copies85 a h).2.2.2.2.1
  have c2725 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.1
  have c2726 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.1
  have c2727 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_4
  simp only [← c2714, k3, ← c2715, k3, ← c2716] at e_40_4
  have e_40_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_5
  simp only [← c2723, ← c2724, k3, ← c2725] at e_40_5
  have e_41_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_0
  simp only [← c2717, ← c2718, ← c2719] at e_41_0
  have e_41_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_1
  simp only [← c2720, ← c2721, ← c2722] at e_41_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_0
    linear_combination c2726.trans k2 - hc
  · have hc := e_40_5
    simp only [e_40_4, e_41_1] at hc
    linear_combination c2727.trans k2 - hc

theorem publicBatchWrapper86_f369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8008)) (a (.virt 200)) (a (.virt 17533)) (a (.virt 17534)) := by
  have c2728 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.1
  have c2729 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.1
  have c2730 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2731 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2732 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2733 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2734 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2735 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2736 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2737 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2738 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2739 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2740 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2741 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_6
  simp only [← c2728, k3, ← c2729, k3, ← c2730] at e_40_6
  have e_40_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_7
  simp only [← c2737, ← c2738, k3, ← c2739] at e_40_7
  have e_41_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_2
  simp only [← c2731, ← c2732, ← c2733] at e_41_2
  have e_41_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_3
  simp only [← c2734, ← c2735, ← c2736] at e_41_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_2
    linear_combination c2740.trans k2 - hc
  · have hc := e_40_7
    simp only [e_40_6, e_41_3] at hc
    linear_combination c2741.trans k2 - hc

theorem publicBatchWrapper86_f370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 41 19) = band (a (.virt 17527)) (a (.virt 17529)) := by
  have c2742 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2743 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2744 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_4
  simp only [← c2742, ← c2743, ← c2744] at e_41_4
  have hr := e_41_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 41 23) = band (a (.virt 17531)) (a (.virt 17533)) := by
  have c2745 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2746 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2747 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_5
  simp only [← c2745, ← c2746, ← c2747] at e_41_5
  have hr := e_41_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 41 27) = band (a (.wire 41 19)) (a (.wire 41 23)) := by
  have c2748 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2749 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2750 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_6
  simp only [← c2748, ← c2749, ← c2750] at e_41_6
  have hr := e_41_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8205)) (a (.virt 200)) (a (.virt 17535)) (a (.virt 17536)) := by
  have c2751 := (publicBatchWrapper86_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2752 := (publicBatchWrapper86_copies86 a h).1
  have c2753 := (publicBatchWrapper86_copies86 a h).2.1
  have c2754 := (publicBatchWrapper86_copies86 a h).2.2.1
  have c2755 := (publicBatchWrapper86_copies86 a h).2.2.2.1
  have c2756 := (publicBatchWrapper86_copies86 a h).2.2.2.2.1
  have c2757 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.1
  have c2758 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.1
  have c2759 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.1
  have c2760 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.1
  have c2761 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.1
  have c2762 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2763 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2764 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_8
  simp only [← c2751, k3, ← c2752, k3, ← c2753] at e_40_8
  have e_40_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_9
  simp only [← c2760, ← c2761, k3, ← c2762] at e_40_9
  have e_41_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_7
  simp only [← c2754, ← c2755, ← c2756] at e_41_7
  have e_41_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_8
  simp only [← c2757, ← c2758, ← c2759] at e_41_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_7
    linear_combination c2763.trans k2 - hc
  · have hc := e_40_9
    simp only [e_40_8, e_41_8] at hc
    linear_combination c2764.trans k2 - hc

theorem publicBatchWrapper86_f374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8206)) (a (.virt 200)) (a (.virt 17537)) (a (.virt 17538)) := by
  have c2765 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2766 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2767 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2768 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2769 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2770 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2771 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2772 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2773 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2774 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2775 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2776 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2777 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2778 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_10
  simp only [← c2765, k3, ← c2766, k3, ← c2767] at e_40_10
  have e_40_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_11
  simp only [← c2774, ← c2775, k3, ← c2776] at e_40_11
  have e_41_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_9
  simp only [← c2768, ← c2769, ← c2770] at e_41_9
  have e_41_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_10
  simp only [← c2771, ← c2772, ← c2773] at e_41_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_9
    linear_combination c2777.trans k2 - hc
  · have hc := e_40_11
    simp only [e_40_10, e_41_10] at hc
    linear_combination c2778.trans k2 - hc

theorem publicBatchWrapper86_f375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8207)) (a (.virt 200)) (a (.virt 17539)) (a (.virt 17540)) := by
  have c2779 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2780 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2781 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2782 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2783 := (publicBatchWrapper86_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2784 := (publicBatchWrapper86_copies87 a h).1
  have c2785 := (publicBatchWrapper86_copies87 a h).2.1
  have c2786 := (publicBatchWrapper86_copies87 a h).2.2.1
  have c2787 := (publicBatchWrapper86_copies87 a h).2.2.2.1
  have c2788 := (publicBatchWrapper86_copies87 a h).2.2.2.2.1
  have c2789 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.1
  have c2790 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.1
  have c2791 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.1
  have c2792 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_12
  simp only [← c2779, k3, ← c2780, k3, ← c2781] at e_40_12
  have e_40_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_13
  simp only [← c2788, ← c2789, k3, ← c2790] at e_40_13
  have e_41_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_11
  simp only [← c2782, ← c2783, ← c2784] at e_41_11
  have e_41_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_12
  simp only [← c2785, ← c2786, ← c2787] at e_41_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_11
    linear_combination c2791.trans k2 - hc
  · have hc := e_40_13
    simp only [e_40_12, e_41_12] at hc
    linear_combination c2792.trans k2 - hc

theorem publicBatchWrapper86_f376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8208)) (a (.virt 200)) (a (.virt 17541)) (a (.virt 17542)) := by
  have c2793 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.1
  have c2794 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2795 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2796 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2797 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2798 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2799 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2800 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2801 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2802 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2803 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2804 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2805 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2806 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_14
  simp only [← c2793, k3, ← c2794, k3, ← c2795] at e_40_14
  have e_40_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_15
  simp only [← c2802, ← c2803, k3, ← c2804] at e_40_15
  have e_41_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_13
  simp only [← c2796, ← c2797, ← c2798] at e_41_13
  have e_41_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_14
  simp only [← c2799, ← c2800, ← c2801] at e_41_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_13
    linear_combination c2805.trans k2 - hc
  · have hc := e_40_15
    simp only [e_40_14, e_41_14] at hc
    linear_combination c2806.trans k2 - hc

theorem publicBatchWrapper86_f377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 41 63) = band (a (.virt 17535)) (a (.virt 17537)) := by
  have c2807 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2808 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2809 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_15
  simp only [← c2807, ← c2808, ← c2809] at e_41_15
  have hr := e_41_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 41 67) = band (a (.virt 17539)) (a (.virt 17541)) := by
  have c2810 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2811 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2812 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_16
  simp only [← c2810, ← c2811, ← c2812] at e_41_16
  have hr := e_41_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 41 71) = band (a (.wire 41 63)) (a (.wire 41 67)) := by
  have c2813 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2814 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2815 := (publicBatchWrapper86_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_41_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_17
  simp only [← c2813, ← c2814, ← c2815] at e_41_17
  have hr := e_41_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8405)) (a (.virt 200)) (a (.virt 17543)) (a (.virt 17544)) := by
  have c2816 := (publicBatchWrapper86_copies88 a h).1
  have c2817 := (publicBatchWrapper86_copies88 a h).2.1
  have c2818 := (publicBatchWrapper86_copies88 a h).2.2.1
  have c2819 := (publicBatchWrapper86_copies88 a h).2.2.2.1
  have c2820 := (publicBatchWrapper86_copies88 a h).2.2.2.2.1
  have c2821 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.1
  have c2822 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.1
  have c2823 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.1
  have c2824 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.1
  have c2825 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.1
  have c2826 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2827 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2828 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2829 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_16
  simp only [← c2816, k3, ← c2817, k3, ← c2818] at e_40_16
  have e_40_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_17
  simp only [← c2825, ← c2826, k3, ← c2827] at e_40_17
  have e_41_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_18
  simp only [← c2819, ← c2820, ← c2821] at e_41_18
  have e_41_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row41 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_19
  simp only [← c2822, ← c2823, ← c2824] at e_41_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_18
    linear_combination c2828.trans k2 - hc
  · have hc := e_40_17
    simp only [e_40_16, e_41_19] at hc
    linear_combination c2829.trans k2 - hc

theorem publicBatchWrapper86_f381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8406)) (a (.virt 200)) (a (.virt 17545)) (a (.virt 17546)) := by
  have c2830 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2831 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2832 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2833 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2834 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2835 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2836 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2837 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2838 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2839 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2840 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2841 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2842 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2843 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_40_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_18
  simp only [← c2830, k3, ← c2831, k3, ← c2832] at e_40_18
  have e_40_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row40 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_19
  simp only [← c2839, ← c2840, k3, ← c2841] at e_40_19
  have e_42_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c2833, ← c2834, ← c2835] at e_42_0
  have e_42_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_1
  simp only [← c2836, ← c2837, ← c2838] at e_42_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_0
    linear_combination c2842.trans k2 - hc
  · have hc := e_40_19
    simp only [e_40_18, e_42_1] at hc
    linear_combination c2843.trans k2 - hc

theorem publicBatchWrapper86_f382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8407)) (a (.virt 200)) (a (.virt 17547)) (a (.virt 17548)) := by
  have c2844 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2845 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2846 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2847 := (publicBatchWrapper86_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2848 := (publicBatchWrapper86_copies89 a h).1
  have c2849 := (publicBatchWrapper86_copies89 a h).2.1
  have c2850 := (publicBatchWrapper86_copies89 a h).2.2.1
  have c2851 := (publicBatchWrapper86_copies89 a h).2.2.2.1
  have c2852 := (publicBatchWrapper86_copies89 a h).2.2.2.2.1
  have c2853 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.1
  have c2854 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.1
  have c2855 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.1
  have c2856 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.1
  have c2857 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_42_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_2
  simp only [← c2847, ← c2848, ← c2849] at e_42_2
  have e_42_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c2850, ← c2851, ← c2852] at e_42_3
  have e_43_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_0
  simp only [← c2844, k3, ← c2845, k3, ← c2846] at e_43_0
  have e_43_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_1
  simp only [← c2853, ← c2854, k3, ← c2855] at e_43_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_2
    linear_combination c2856.trans k2 - hc
  · have hc := e_43_1
    simp only [e_43_0, e_42_3] at hc
    linear_combination c2857.trans k2 - hc

theorem publicBatchWrapper86_f383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8408)) (a (.virt 200)) (a (.virt 17549)) (a (.virt 17550)) := by
  have c2858 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2859 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2860 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2861 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2862 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2863 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2864 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2865 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2866 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2867 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2868 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2869 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2870 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2871 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_42_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_4
  simp only [← c2861, ← c2862, ← c2863] at e_42_4
  have e_42_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_5
  simp only [← c2864, ← c2865, ← c2866] at e_42_5
  have e_43_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_2
  simp only [← c2858, k3, ← c2859, k3, ← c2860] at e_43_2
  have e_43_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_3
  simp only [← c2867, ← c2868, k3, ← c2869] at e_43_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_4
    linear_combination c2870.trans k2 - hc
  · have hc := e_43_3
    simp only [e_43_2, e_42_5] at hc
    linear_combination c2871.trans k2 - hc

theorem publicBatchWrapper86_f384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 42 27) = band (a (.virt 17543)) (a (.virt 17545)) := by
  have c2872 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2873 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2874 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_42_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_6
  simp only [← c2872, ← c2873, ← c2874] at e_42_6
  have hr := e_42_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 42 31) = band (a (.virt 17547)) (a (.virt 17549)) := by
  have c2875 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2876 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2877 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_42_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2875, ← c2876, ← c2877] at e_42_7
  have hr := e_42_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 42 35) = band (a (.wire 42 27)) (a (.wire 42 31)) := by
  have c2878 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2879 := (publicBatchWrapper86_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2880 := (publicBatchWrapper86_copies90 a h).1
  have e_42_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_8
  simp only [← c2878, ← c2879, ← c2880] at e_42_8
  have hr := e_42_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8605)) (a (.virt 200)) (a (.virt 17551)) (a (.virt 17552)) := by
  have c2881 := (publicBatchWrapper86_copies90 a h).2.1
  have c2882 := (publicBatchWrapper86_copies90 a h).2.2.1
  have c2883 := (publicBatchWrapper86_copies90 a h).2.2.2.1
  have c2884 := (publicBatchWrapper86_copies90 a h).2.2.2.2.1
  have c2885 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.1
  have c2886 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.1
  have c2887 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.1
  have c2888 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.1
  have c2889 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.1
  have c2890 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2891 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2892 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2893 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2894 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_42_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_9
  simp only [← c2884, ← c2885, ← c2886] at e_42_9
  have e_42_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2887, ← c2888, ← c2889] at e_42_10
  have e_43_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_4
  simp only [← c2881, k3, ← c2882, k3, ← c2883] at e_43_4
  have e_43_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_5
  simp only [← c2890, ← c2891, k3, ← c2892] at e_43_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_9
    linear_combination c2893.trans k2 - hc
  · have hc := e_43_5
    simp only [e_43_4, e_42_10] at hc
    linear_combination c2894.trans k2 - hc

theorem publicBatchWrapper86_f388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8606)) (a (.virt 200)) (a (.virt 17553)) (a (.virt 17554)) := by
  have c2895 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2896 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2897 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2898 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2899 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2900 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2901 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2902 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2903 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2904 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2905 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2906 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2907 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2908 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_42_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_11
  simp only [← c2898, ← c2899, ← c2900] at e_42_11
  have e_42_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_12
  simp only [← c2901, ← c2902, ← c2903] at e_42_12
  have e_43_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_6
  simp only [← c2895, k3, ← c2896, k3, ← c2897] at e_43_6
  have e_43_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_7
  simp only [← c2904, ← c2905, k3, ← c2906] at e_43_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_11
    linear_combination c2907.trans k2 - hc
  · have hc := e_43_7
    simp only [e_43_6, e_42_12] at hc
    linear_combination c2908.trans k2 - hc

theorem publicBatchWrapper86_f389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8607)) (a (.virt 200)) (a (.virt 17555)) (a (.virt 17556)) := by
  have c2909 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2910 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2911 := (publicBatchWrapper86_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2912 := (publicBatchWrapper86_copies91 a h).1
  have c2913 := (publicBatchWrapper86_copies91 a h).2.1
  have c2914 := (publicBatchWrapper86_copies91 a h).2.2.1
  have c2915 := (publicBatchWrapper86_copies91 a h).2.2.2.1
  have c2916 := (publicBatchWrapper86_copies91 a h).2.2.2.2.1
  have c2917 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.1
  have c2918 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.1
  have c2919 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.1
  have c2920 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.1
  have c2921 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.1
  have c2922 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_42_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c2912, ← c2913, ← c2914] at e_42_13
  have e_42_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_14
  simp only [← c2915, ← c2916, ← c2917] at e_42_14
  have e_43_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_8
  simp only [← c2909, k3, ← c2910, k3, ← c2911] at e_43_8
  have e_43_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_9
  simp only [← c2918, ← c2919, k3, ← c2920] at e_43_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_13
    linear_combination c2921.trans k2 - hc
  · have hc := e_43_9
    simp only [e_43_8, e_42_14] at hc
    linear_combination c2922.trans k2 - hc

theorem publicBatchWrapper86_f390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8608)) (a (.virt 200)) (a (.virt 17557)) (a (.virt 17558)) := by
  have c2923 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2924 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2925 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2926 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2927 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2928 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2929 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2930 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2931 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2932 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2933 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2934 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2935 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2936 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_42_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_15
  simp only [← c2926, ← c2927, ← c2928] at e_42_15
  have e_42_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_16
  simp only [← c2929, ← c2930, ← c2931] at e_42_16
  have e_43_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_10
  simp only [← c2923, k3, ← c2924, k3, ← c2925] at e_43_10
  have e_43_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_11
  simp only [← c2932, ← c2933, k3, ← c2934] at e_43_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_42_15
    linear_combination c2935.trans k2 - hc
  · have hc := e_43_11
    simp only [e_43_10, e_42_16] at hc
    linear_combination c2936.trans k2 - hc

theorem publicBatchWrapper86_f391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 42 71) = band (a (.virt 17551)) (a (.virt 17553)) := by
  have c2937 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2938 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2939 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_42_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_17
  simp only [← c2937, ← c2938, ← c2939] at e_42_17
  have hr := e_42_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 42 75) = band (a (.virt 17555)) (a (.virt 17557)) := by
  have c2940 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2941 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2942 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_42_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_18
  simp only [← c2940, ← c2941, ← c2942] at e_42_18
  have hr := e_42_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 42 79) = band (a (.wire 42 71)) (a (.wire 42 75)) := by
  have c2943 := (publicBatchWrapper86_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2944 := (publicBatchWrapper86_copies92 a h).1
  have c2945 := (publicBatchWrapper86_copies92 a h).2.1
  have e_42_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row42 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_19
  simp only [← c2943, ← c2944, ← c2945] at e_42_19
  have hr := e_42_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8805)) (a (.virt 200)) (a (.virt 17559)) (a (.virt 17560)) := by
  have c2946 := (publicBatchWrapper86_copies92 a h).2.2.1
  have c2947 := (publicBatchWrapper86_copies92 a h).2.2.2.1
  have c2948 := (publicBatchWrapper86_copies92 a h).2.2.2.2.1
  have c2949 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.1
  have c2950 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.1
  have c2951 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.1
  have c2952 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.1
  have c2953 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.1
  have c2954 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2955 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2956 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2957 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2958 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2959 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_43_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_12
  simp only [← c2946, k3, ← c2947, k3, ← c2948] at e_43_12
  have e_43_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_13
  simp only [← c2955, ← c2956, k3, ← c2957] at e_43_13
  have e_44_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_0
  simp only [← c2949, ← c2950, ← c2951] at e_44_0
  have e_44_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_1
  simp only [← c2952, ← c2953, ← c2954] at e_44_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_0
    linear_combination c2958.trans k2 - hc
  · have hc := e_43_13
    simp only [e_43_12, e_44_1] at hc
    linear_combination c2959.trans k2 - hc

theorem publicBatchWrapper86_f395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8806)) (a (.virt 200)) (a (.virt 17561)) (a (.virt 17562)) := by
  have c2960 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2961 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2962 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2963 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2964 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2965 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2966 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2967 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2968 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2969 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2970 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2971 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2972 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2973 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_43_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_14
  simp only [← c2960, k3, ← c2961, k3, ← c2962] at e_43_14
  have e_43_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_15
  simp only [← c2969, ← c2970, k3, ← c2971] at e_43_15
  have e_44_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_2
  simp only [← c2963, ← c2964, ← c2965] at e_44_2
  have e_44_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_3
  simp only [← c2966, ← c2967, ← c2968] at e_44_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_2
    linear_combination c2972.trans k2 - hc
  · have hc := e_43_15
    simp only [e_43_14, e_44_3] at hc
    linear_combination c2973.trans k2 - hc

theorem publicBatchWrapper86_f396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8807)) (a (.virt 200)) (a (.virt 17563)) (a (.virt 17564)) := by
  have c2974 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2975 := (publicBatchWrapper86_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2976 := (publicBatchWrapper86_copies93 a h).1
  have c2977 := (publicBatchWrapper86_copies93 a h).2.1
  have c2978 := (publicBatchWrapper86_copies93 a h).2.2.1
  have c2979 := (publicBatchWrapper86_copies93 a h).2.2.2.1
  have c2980 := (publicBatchWrapper86_copies93 a h).2.2.2.2.1
  have c2981 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.1
  have c2982 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.1
  have c2983 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.1
  have c2984 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.1
  have c2985 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.1
  have c2986 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2987 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_43_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_16
  simp only [← c2974, k3, ← c2975, k3, ← c2976] at e_43_16
  have e_43_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_17
  simp only [← c2983, ← c2984, k3, ← c2985] at e_43_17
  have e_44_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_4
  simp only [← c2977, ← c2978, ← c2979] at e_44_4
  have e_44_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_5
  simp only [← c2980, ← c2981, ← c2982] at e_44_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_4
    linear_combination c2986.trans k2 - hc
  · have hc := e_43_17
    simp only [e_43_16, e_44_5] at hc
    linear_combination c2987.trans k2 - hc

theorem publicBatchWrapper86_f397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 8808)) (a (.virt 200)) (a (.virt 17565)) (a (.virt 17566)) := by
  have c2988 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2989 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2990 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2991 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2992 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2993 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2994 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2995 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2996 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_43_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_18
  simp only [← c2988, k3, ← c2989, k3, ← c2990] at e_43_18
  have e_43_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row43 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_19
  simp only [← c2997, ← c2998, k3, ← c2999] at e_43_19
  have e_44_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_6
  simp only [← c2991, ← c2992, ← c2993] at e_44_6
  have e_44_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_7
  simp only [← c2994, ← c2995, ← c2996] at e_44_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_6
    linear_combination c3000.trans k2 - hc
  · have hc := e_43_19
    simp only [e_43_18, e_44_7] at hc
    linear_combination c3001.trans k2 - hc

theorem publicBatchWrapper86_f398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 44 35) = band (a (.virt 17559)) (a (.virt 17561)) := by
  have c3002 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3003 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3004 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_8
  simp only [← c3002, ← c3003, ← c3004] at e_44_8
  have hr := e_44_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 44 39) = band (a (.virt 17563)) (a (.virt 17565)) := by
  have c3005 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3006 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3007 := (publicBatchWrapper86_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_44_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_9
  simp only [← c3005, ← c3006, ← c3007] at e_44_9
  have hr := e_44_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 44 43) = band (a (.wire 44 35)) (a (.wire 44 39)) := by
  have c3008 := (publicBatchWrapper86_copies94 a h).1
  have c3009 := (publicBatchWrapper86_copies94 a h).2.1
  have c3010 := (publicBatchWrapper86_copies94 a h).2.2.1
  have e_44_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_10
  simp only [← c3008, ← c3009, ← c3010] at e_44_10
  have hr := e_44_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9005)) (a (.virt 200)) (a (.virt 17567)) (a (.virt 17568)) := by
  have c3011 := (publicBatchWrapper86_copies94 a h).2.2.2.1
  have c3012 := (publicBatchWrapper86_copies94 a h).2.2.2.2.1
  have c3013 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.1
  have c3014 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.1
  have c3015 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.1
  have c3016 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.1
  have c3017 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.1
  have c3018 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3019 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3020 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3021 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3022 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3023 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3024 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_44_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_11
  simp only [← c3014, ← c3015, ← c3016] at e_44_11
  have e_44_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_12
  simp only [← c3017, ← c3018, ← c3019] at e_44_12
  have e_45_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_0
  simp only [← c3011, k3, ← c3012, k3, ← c3013] at e_45_0
  have e_45_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_1
  simp only [← c3020, ← c3021, k3, ← c3022] at e_45_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_11
    linear_combination c3023.trans k2 - hc
  · have hc := e_45_1
    simp only [e_45_0, e_44_12] at hc
    linear_combination c3024.trans k2 - hc

theorem publicBatchWrapper86_f402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9006)) (a (.virt 200)) (a (.virt 17569)) (a (.virt 17570)) := by
  have c3025 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3026 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3027 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3028 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3029 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3030 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3031 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3032 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3033 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3034 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3035 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3036 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3037 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3038 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_44_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_13
  simp only [← c3028, ← c3029, ← c3030] at e_44_13
  have e_44_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_14
  simp only [← c3031, ← c3032, ← c3033] at e_44_14
  have e_45_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_2
  simp only [← c3025, k3, ← c3026, k3, ← c3027] at e_45_2
  have e_45_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_3
  simp only [← c3034, ← c3035, k3, ← c3036] at e_45_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_13
    linear_combination c3037.trans k2 - hc
  · have hc := e_45_3
    simp only [e_45_2, e_44_14] at hc
    linear_combination c3038.trans k2 - hc

theorem publicBatchWrapper86_f403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9007)) (a (.virt 200)) (a (.virt 17571)) (a (.virt 17572)) := by
  have c3039 := (publicBatchWrapper86_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3040 := (publicBatchWrapper86_copies95 a h).1
  have c3041 := (publicBatchWrapper86_copies95 a h).2.1
  have c3042 := (publicBatchWrapper86_copies95 a h).2.2.1
  have c3043 := (publicBatchWrapper86_copies95 a h).2.2.2.1
  have c3044 := (publicBatchWrapper86_copies95 a h).2.2.2.2.1
  have c3045 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.1
  have c3046 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.1
  have c3047 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.1
  have c3048 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.1
  have c3049 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.1
  have c3050 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3051 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3052 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_44_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_15
  simp only [← c3042, ← c3043, ← c3044] at e_44_15
  have e_44_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_16
  simp only [← c3045, ← c3046, ← c3047] at e_44_16
  have e_45_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_4
  simp only [← c3039, k3, ← c3040, k3, ← c3041] at e_45_4
  have e_45_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_5
  simp only [← c3048, ← c3049, k3, ← c3050] at e_45_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_15
    linear_combination c3051.trans k2 - hc
  · have hc := e_45_5
    simp only [e_45_4, e_44_16] at hc
    linear_combination c3052.trans k2 - hc

theorem publicBatchWrapper86_f404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9008)) (a (.virt 200)) (a (.virt 17573)) (a (.virt 17574)) := by
  have c3053 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3054 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3055 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3056 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3057 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3058 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3059 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3060 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3061 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3062 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3063 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3064 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3065 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3066 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_44_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_17
  simp only [← c3056, ← c3057, ← c3058] at e_44_17
  have e_44_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_18
  simp only [← c3059, ← c3060, ← c3061] at e_44_18
  have e_45_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_6
  simp only [← c3053, k3, ← c3054, k3, ← c3055] at e_45_6
  have e_45_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_7
  simp only [← c3062, ← c3063, k3, ← c3064] at e_45_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_44_17
    linear_combination c3065.trans k2 - hc
  · have hc := e_45_7
    simp only [e_45_6, e_44_18] at hc
    linear_combination c3066.trans k2 - hc

theorem publicBatchWrapper86_f405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 44 79) = band (a (.virt 17567)) (a (.virt 17569)) := by
  have c3067 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3068 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3069 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_44_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row44 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_19
  simp only [← c3067, ← c3068, ← c3069] at e_44_19
  have hr := e_44_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 46 3) = band (a (.virt 17571)) (a (.virt 17573)) := by
  have c3070 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3071 := (publicBatchWrapper86_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3072 := (publicBatchWrapper86_copies96 a h).1
  have e_46_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_0
  simp only [← c3070, ← c3071, ← c3072] at e_46_0
  have hr := e_46_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 46 7) = band (a (.wire 44 79)) (a (.wire 46 3)) := by
  have c3073 := (publicBatchWrapper86_copies96 a h).2.1
  have c3074 := (publicBatchWrapper86_copies96 a h).2.2.1
  have c3075 := (publicBatchWrapper86_copies96 a h).2.2.2.1
  have e_46_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_1
  simp only [← c3073, ← c3074, ← c3075] at e_46_1
  have hr := e_46_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9205)) (a (.virt 200)) (a (.virt 17575)) (a (.virt 17576)) := by
  have c3076 := (publicBatchWrapper86_copies96 a h).2.2.2.2.1
  have c3077 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.1
  have c3078 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.1
  have c3079 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.1
  have c3080 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.1
  have c3081 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.1
  have c3082 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3083 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3084 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3085 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3086 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3087 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3088 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3089 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_45_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_8
  simp only [← c3076, k3, ← c3077, k3, ← c3078] at e_45_8
  have e_45_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_9
  simp only [← c3085, ← c3086, k3, ← c3087] at e_45_9
  have e_46_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_2
  simp only [← c3079, ← c3080, ← c3081] at e_46_2
  have e_46_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_3
  simp only [← c3082, ← c3083, ← c3084] at e_46_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_2
    linear_combination c3088.trans k2 - hc
  · have hc := e_45_9
    simp only [e_45_8, e_46_3] at hc
    linear_combination c3089.trans k2 - hc

theorem publicBatchWrapper86_f409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9206)) (a (.virt 200)) (a (.virt 17577)) (a (.virt 17578)) := by
  have c3090 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3091 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3092 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3093 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3094 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3095 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3096 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3097 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3098 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3099 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3100 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3101 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3102 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3103 := (publicBatchWrapper86_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_45_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_10
  simp only [← c3090, k3, ← c3091, k3, ← c3092] at e_45_10
  have e_45_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_11
  simp only [← c3099, ← c3100, k3, ← c3101] at e_45_11
  have e_46_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_4
  simp only [← c3093, ← c3094, ← c3095] at e_46_4
  have e_46_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_5
  simp only [← c3096, ← c3097, ← c3098] at e_46_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_4
    linear_combination c3102.trans k2 - hc
  · have hc := e_45_11
    simp only [e_45_10, e_46_5] at hc
    linear_combination c3103.trans k2 - hc

theorem publicBatchWrapper86_f410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9207)) (a (.virt 200)) (a (.virt 17579)) (a (.virt 17580)) := by
  have c3104 := (publicBatchWrapper86_copies97 a h).1
  have c3105 := (publicBatchWrapper86_copies97 a h).2.1
  have c3106 := (publicBatchWrapper86_copies97 a h).2.2.1
  have c3107 := (publicBatchWrapper86_copies97 a h).2.2.2.1
  have c3108 := (publicBatchWrapper86_copies97 a h).2.2.2.2.1
  have c3109 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.1
  have c3110 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.1
  have c3111 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.1
  have c3112 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.1
  have c3113 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.1
  have c3114 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3115 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3116 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3117 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_45_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_12
  simp only [← c3104, k3, ← c3105, k3, ← c3106] at e_45_12
  have e_45_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_13
  simp only [← c3113, ← c3114, k3, ← c3115] at e_45_13
  have e_46_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c3107, ← c3108, ← c3109] at e_46_6
  have e_46_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_7
  simp only [← c3110, ← c3111, ← c3112] at e_46_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_6
    linear_combination c3116.trans k2 - hc
  · have hc := e_45_13
    simp only [e_45_12, e_46_7] at hc
    linear_combination c3117.trans k2 - hc

theorem publicBatchWrapper86_f411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9208)) (a (.virt 200)) (a (.virt 17581)) (a (.virt 17582)) := by
  have c3118 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3119 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3120 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3121 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3122 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3123 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3124 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3125 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3126 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3127 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3128 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3129 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3130 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3131 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_45_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_14
  simp only [← c3118, k3, ← c3119, k3, ← c3120] at e_45_14
  have e_45_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_15
  simp only [← c3127, ← c3128, k3, ← c3129] at e_45_15
  have e_46_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_8
  simp only [← c3121, ← c3122, ← c3123] at e_46_8
  have e_46_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_9
  simp only [← c3124, ← c3125, ← c3126] at e_46_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_8
    linear_combination c3130.trans k2 - hc
  · have hc := e_45_15
    simp only [e_45_14, e_46_9] at hc
    linear_combination c3131.trans k2 - hc

theorem publicBatchWrapper86_f412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 46 43) = band (a (.virt 17575)) (a (.virt 17577)) := by
  have c3132 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3133 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3134 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_46_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_10
  simp only [← c3132, ← c3133, ← c3134] at e_46_10
  have hr := e_46_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 46 47) = band (a (.virt 17579)) (a (.virt 17581)) := by
  have c3135 := (publicBatchWrapper86_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3136 := (publicBatchWrapper86_copies98 a h).1
  have c3137 := (publicBatchWrapper86_copies98 a h).2.1
  have e_46_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_11
  simp only [← c3135, ← c3136, ← c3137] at e_46_11
  have hr := e_46_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 46 51) = band (a (.wire 46 43)) (a (.wire 46 47)) := by
  have c3138 := (publicBatchWrapper86_copies98 a h).2.2.1
  have c3139 := (publicBatchWrapper86_copies98 a h).2.2.2.1
  have c3140 := (publicBatchWrapper86_copies98 a h).2.2.2.2.1
  have e_46_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_12
  simp only [← c3138, ← c3139, ← c3140] at e_46_12
  have hr := e_46_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9405)) (a (.virt 200)) (a (.virt 17583)) (a (.virt 17584)) := by
  have c3141 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.1
  have c3142 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.1
  have c3143 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.1
  have c3144 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.1
  have c3145 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.1
  have c3146 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3147 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3148 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3149 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3150 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3151 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3152 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3153 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3154 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_45_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_16
  simp only [← c3141, k3, ← c3142, k3, ← c3143] at e_45_16
  have e_45_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_17
  simp only [← c3150, ← c3151, k3, ← c3152] at e_45_17
  have e_46_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_13
  simp only [← c3144, ← c3145, ← c3146] at e_46_13
  have e_46_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_14
  simp only [← c3147, ← c3148, ← c3149] at e_46_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_13
    linear_combination c3153.trans k2 - hc
  · have hc := e_45_17
    simp only [e_45_16, e_46_14] at hc
    linear_combination c3154.trans k2 - hc

theorem publicBatchWrapper86_f416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9406)) (a (.virt 200)) (a (.virt 17585)) (a (.virt 17586)) := by
  have c3155 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3156 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3157 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3158 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3159 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3160 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3161 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3162 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3163 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3164 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3165 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3166 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3167 := (publicBatchWrapper86_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3168 := (publicBatchWrapper86_copies99 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_45_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_18
  simp only [← c3155, k3, ← c3156, k3, ← c3157] at e_45_18
  have e_45_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row45 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_19
  simp only [← c3164, ← c3165, k3, ← c3166] at e_45_19
  have e_46_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_15
  simp only [← c3158, ← c3159, ← c3160] at e_46_15
  have e_46_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_16
  simp only [← c3161, ← c3162, ← c3163] at e_46_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_15
    linear_combination c3167.trans k2 - hc
  · have hc := e_45_19
    simp only [e_45_18, e_46_16] at hc
    linear_combination c3168.trans k2 - hc

theorem publicBatchWrapper86_f417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9407)) (a (.virt 200)) (a (.virt 17587)) (a (.virt 17588)) := by
  have c3169 := (publicBatchWrapper86_copies99 a h).2.1
  have c3170 := (publicBatchWrapper86_copies99 a h).2.2.1
  have c3171 := (publicBatchWrapper86_copies99 a h).2.2.2.1
  have c3172 := (publicBatchWrapper86_copies99 a h).2.2.2.2.1
  have c3173 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.1
  have c3174 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.1
  have c3175 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.1
  have c3176 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.1
  have c3177 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.1
  have c3178 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3179 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3180 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3181 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3182 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_46_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_17
  simp only [← c3172, ← c3173, ← c3174] at e_46_17
  have e_46_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_18
  simp only [← c3175, ← c3176, ← c3177] at e_46_18
  have e_47_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_0
  simp only [← c3169, k3, ← c3170, k3, ← c3171] at e_47_0
  have e_47_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_1
  simp only [← c3178, ← c3179, k3, ← c3180] at e_47_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_17
    linear_combination c3181.trans k2 - hc
  · have hc := e_47_1
    simp only [e_47_0, e_46_18] at hc
    linear_combination c3182.trans k2 - hc

theorem publicBatchWrapper86_f418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9408)) (a (.virt 200)) (a (.virt 17589)) (a (.virt 17590)) := by
  have c3183 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3184 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3185 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3186 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3187 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3188 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3189 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3190 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3191 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3192 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3193 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3194 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3195 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3196 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_46_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row46 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_19
  simp only [← c3186, ← c3187, ← c3188] at e_46_19
  have e_47_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_2
  simp only [← c3183, k3, ← c3184, k3, ← c3185] at e_47_2
  have e_47_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_3
  simp only [← c3192, ← c3193, k3, ← c3194] at e_47_3
  have e_48_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_0
  simp only [← c3189, ← c3190, ← c3191] at e_48_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_19
    linear_combination c3195.trans k2 - hc
  · have hc := e_47_3
    simp only [e_47_2, e_48_0] at hc
    linear_combination c3196.trans k2 - hc

theorem publicBatchWrapper86_f419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 48 7) = band (a (.virt 17583)) (a (.virt 17585)) := by
  have c3197 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3198 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3199 := (publicBatchWrapper86_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_48_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_1
  simp only [← c3197, ← c3198, ← c3199] at e_48_1
  have hr := e_48_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 48 11) = band (a (.virt 17587)) (a (.virt 17589)) := by
  have c3200 := (publicBatchWrapper86_copies100 a h).1
  have c3201 := (publicBatchWrapper86_copies100 a h).2.1
  have c3202 := (publicBatchWrapper86_copies100 a h).2.2.1
  have e_48_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_2
  simp only [← c3200, ← c3201, ← c3202] at e_48_2
  have hr := e_48_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 48 15) = band (a (.wire 48 7)) (a (.wire 48 11)) := by
  have c3203 := (publicBatchWrapper86_copies100 a h).2.2.2.1
  have c3204 := (publicBatchWrapper86_copies100 a h).2.2.2.2.1
  have c3205 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.1
  have e_48_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_3
  simp only [← c3203, ← c3204, ← c3205] at e_48_3
  have hr := e_48_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9605)) (a (.virt 200)) (a (.virt 17591)) (a (.virt 17592)) := by
  have c3206 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.1
  have c3207 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.1
  have c3208 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.1
  have c3209 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.1
  have c3210 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3211 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3212 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3213 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3214 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3215 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3216 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3217 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3218 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3219 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_4
  simp only [← c3206, k3, ← c3207, k3, ← c3208] at e_47_4
  have e_47_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_5
  simp only [← c3215, ← c3216, k3, ← c3217] at e_47_5
  have e_48_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_4
  simp only [← c3209, ← c3210, ← c3211] at e_48_4
  have e_48_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_5
  simp only [← c3212, ← c3213, ← c3214] at e_48_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_4
    linear_combination c3218.trans k2 - hc
  · have hc := e_47_5
    simp only [e_47_4, e_48_5] at hc
    linear_combination c3219.trans k2 - hc

theorem publicBatchWrapper86_f423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9606)) (a (.virt 200)) (a (.virt 17593)) (a (.virt 17594)) := by
  have c3220 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3221 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3222 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3223 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3224 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3225 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3226 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3227 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3228 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3229 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3230 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3231 := (publicBatchWrapper86_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3232 := (publicBatchWrapper86_copies101 a h).1
  have c3233 := (publicBatchWrapper86_copies101 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_6
  simp only [← c3220, k3, ← c3221, k3, ← c3222] at e_47_6
  have e_47_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_7
  simp only [← c3229, ← c3230, k3, ← c3231] at e_47_7
  have e_48_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_6
  simp only [← c3223, ← c3224, ← c3225] at e_48_6
  have e_48_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_7
  simp only [← c3226, ← c3227, ← c3228] at e_48_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_6
    linear_combination c3232.trans k2 - hc
  · have hc := e_47_7
    simp only [e_47_6, e_48_7] at hc
    linear_combination c3233.trans k2 - hc

theorem publicBatchWrapper86_f424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9607)) (a (.virt 200)) (a (.virt 17595)) (a (.virt 17596)) := by
  have c3234 := (publicBatchWrapper86_copies101 a h).2.2.1
  have c3235 := (publicBatchWrapper86_copies101 a h).2.2.2.1
  have c3236 := (publicBatchWrapper86_copies101 a h).2.2.2.2.1
  have c3237 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.1
  have c3238 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.1
  have c3239 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.1
  have c3240 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.1
  have c3241 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.1
  have c3242 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3243 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3244 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3245 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3246 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3247 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_8
  simp only [← c3234, k3, ← c3235, k3, ← c3236] at e_47_8
  have e_47_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_9
  simp only [← c3243, ← c3244, k3, ← c3245] at e_47_9
  have e_48_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_8
  simp only [← c3237, ← c3238, ← c3239] at e_48_8
  have e_48_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_9
  simp only [← c3240, ← c3241, ← c3242] at e_48_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_8
    linear_combination c3246.trans k2 - hc
  · have hc := e_47_9
    simp only [e_47_8, e_48_9] at hc
    linear_combination c3247.trans k2 - hc

theorem publicBatchWrapper86_f425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9608)) (a (.virt 200)) (a (.virt 17597)) (a (.virt 17598)) := by
  have c3248 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3249 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3250 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3251 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3252 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3253 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3254 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3255 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3256 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3257 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3258 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3259 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3260 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3261 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_10
  simp only [← c3248, k3, ← c3249, k3, ← c3250] at e_47_10
  have e_47_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_11
  simp only [← c3257, ← c3258, k3, ← c3259] at e_47_11
  have e_48_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_10
  simp only [← c3251, ← c3252, ← c3253] at e_48_10
  have e_48_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_11
  simp only [← c3254, ← c3255, ← c3256] at e_48_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_10
    linear_combination c3260.trans k2 - hc
  · have hc := e_47_11
    simp only [e_47_10, e_48_11] at hc
    linear_combination c3261.trans k2 - hc

theorem publicBatchWrapper86_f426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 48 51) = band (a (.virt 17591)) (a (.virt 17593)) := by
  have c3262 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3263 := (publicBatchWrapper86_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3264 := (publicBatchWrapper86_copies102 a h).1
  have e_48_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_12
  simp only [← c3262, ← c3263, ← c3264] at e_48_12
  have hr := e_48_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 48 55) = band (a (.virt 17595)) (a (.virt 17597)) := by
  have c3265 := (publicBatchWrapper86_copies102 a h).2.1
  have c3266 := (publicBatchWrapper86_copies102 a h).2.2.1
  have c3267 := (publicBatchWrapper86_copies102 a h).2.2.2.1
  have e_48_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_13
  simp only [← c3265, ← c3266, ← c3267] at e_48_13
  have hr := e_48_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 48 59) = band (a (.wire 48 51)) (a (.wire 48 55)) := by
  have c3268 := (publicBatchWrapper86_copies102 a h).2.2.2.2.1
  have c3269 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.1
  have c3270 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.1
  have e_48_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_14
  simp only [← c3268, ← c3269, ← c3270] at e_48_14
  have hr := e_48_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9805)) (a (.virt 200)) (a (.virt 17599)) (a (.virt 17600)) := by
  have c3271 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.1
  have c3272 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.1
  have c3273 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.1
  have c3274 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3275 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3276 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3277 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3278 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3279 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3280 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3281 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3282 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3283 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3284 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_12
  simp only [← c3271, k3, ← c3272, k3, ← c3273] at e_47_12
  have e_47_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_13
  simp only [← c3280, ← c3281, k3, ← c3282] at e_47_13
  have e_48_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_15
  simp only [← c3274, ← c3275, ← c3276] at e_48_15
  have e_48_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_16
  simp only [← c3277, ← c3278, ← c3279] at e_48_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_15
    linear_combination c3283.trans k2 - hc
  · have hc := e_47_13
    simp only [e_47_12, e_48_16] at hc
    linear_combination c3284.trans k2 - hc

theorem publicBatchWrapper86_f430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9806)) (a (.virt 200)) (a (.virt 17601)) (a (.virt 17602)) := by
  have c3285 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3286 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3287 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3288 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3289 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3290 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3291 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3292 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3293 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3294 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3295 := (publicBatchWrapper86_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3296 := (publicBatchWrapper86_copies103 a h).1
  have c3297 := (publicBatchWrapper86_copies103 a h).2.1
  have c3298 := (publicBatchWrapper86_copies103 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_14
  simp only [← c3285, k3, ← c3286, k3, ← c3287] at e_47_14
  have e_47_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_15
  simp only [← c3294, ← c3295, k3, ← c3296] at e_47_15
  have e_48_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_17
  simp only [← c3288, ← c3289, ← c3290] at e_48_17
  have e_48_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_18
  simp only [← c3291, ← c3292, ← c3293] at e_48_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_17
    linear_combination c3297.trans k2 - hc
  · have hc := e_47_15
    simp only [e_47_14, e_48_18] at hc
    linear_combination c3298.trans k2 - hc

theorem publicBatchWrapper86_f431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9807)) (a (.virt 200)) (a (.virt 17603)) (a (.virt 17604)) := by
  have c3299 := (publicBatchWrapper86_copies103 a h).2.2.2.1
  have c3300 := (publicBatchWrapper86_copies103 a h).2.2.2.2.1
  have c3301 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.1
  have c3302 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.1
  have c3303 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.1
  have c3304 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.1
  have c3305 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.1
  have c3306 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3307 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3308 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3309 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3310 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3311 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3312 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_16
  simp only [← c3299, k3, ← c3300, k3, ← c3301] at e_47_16
  have e_47_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_17
  simp only [← c3308, ← c3309, k3, ← c3310] at e_47_17
  have e_48_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row48 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_19
  simp only [← c3302, ← c3303, ← c3304] at e_48_19
  have e_49_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_0
  simp only [← c3305, ← c3306, ← c3307] at e_49_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_19
    linear_combination c3311.trans k2 - hc
  · have hc := e_47_17
    simp only [e_47_16, e_49_0] at hc
    linear_combination c3312.trans k2 - hc

theorem publicBatchWrapper86_f432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 9808)) (a (.virt 200)) (a (.virt 17605)) (a (.virt 17606)) := by
  have c3313 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3314 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3315 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3316 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3317 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3318 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3319 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3320 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3321 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3322 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3323 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3324 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3325 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3326 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_47_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_18
  simp only [← c3313, k3, ← c3314, k3, ← c3315] at e_47_18
  have e_47_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row47 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_19
  simp only [← c3322, ← c3323, k3, ← c3324] at e_47_19
  have e_49_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_1
  simp only [← c3316, ← c3317, ← c3318] at e_49_1
  have e_49_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_2
  simp only [← c3319, ← c3320, ← c3321] at e_49_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_1
    linear_combination c3325.trans k2 - hc
  · have hc := e_47_19
    simp only [e_47_18, e_49_2] at hc
    linear_combination c3326.trans k2 - hc

theorem publicBatchWrapper86_f433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 49 15) = band (a (.virt 17599)) (a (.virt 17601)) := by
  have c3327 := (publicBatchWrapper86_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3328 := (publicBatchWrapper86_copies104 a h).1
  have c3329 := (publicBatchWrapper86_copies104 a h).2.1
  have e_49_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_3
  simp only [← c3327, ← c3328, ← c3329] at e_49_3
  have hr := e_49_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 49 19) = band (a (.virt 17603)) (a (.virt 17605)) := by
  have c3330 := (publicBatchWrapper86_copies104 a h).2.2.1
  have c3331 := (publicBatchWrapper86_copies104 a h).2.2.2.1
  have c3332 := (publicBatchWrapper86_copies104 a h).2.2.2.2.1
  have e_49_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_4
  simp only [← c3330, ← c3331, ← c3332] at e_49_4
  have hr := e_49_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 49 23) = band (a (.wire 49 15)) (a (.wire 49 19)) := by
  have c3333 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.1
  have c3334 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.1
  have c3335 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.1
  have e_49_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_5
  simp only [← c3333, ← c3334, ← c3335] at e_49_5
  have hr := e_49_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10005)) (a (.virt 200)) (a (.virt 17607)) (a (.virt 17608)) := by
  have c3336 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.1
  have c3337 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.1
  have c3338 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3339 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3340 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3341 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3342 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3343 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3344 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3345 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3346 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3347 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3348 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3349 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_49_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_6
  simp only [← c3339, ← c3340, ← c3341] at e_49_6
  have e_49_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_7
  simp only [← c3342, ← c3343, ← c3344] at e_49_7
  have e_50_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_0
  simp only [← c3336, k3, ← c3337, k3, ← c3338] at e_50_0
  have e_50_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_1
  simp only [← c3345, ← c3346, k3, ← c3347] at e_50_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_6
    linear_combination c3348.trans k2 - hc
  · have hc := e_50_1
    simp only [e_50_0, e_49_7] at hc
    linear_combination c3349.trans k2 - hc

theorem publicBatchWrapper86_f437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10006)) (a (.virt 200)) (a (.virt 17609)) (a (.virt 17610)) := by
  have c3350 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3351 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3352 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3353 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3354 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3355 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3356 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3357 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3358 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3359 := (publicBatchWrapper86_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3360 := (publicBatchWrapper86_copies105 a h).1
  have c3361 := (publicBatchWrapper86_copies105 a h).2.1
  have c3362 := (publicBatchWrapper86_copies105 a h).2.2.1
  have c3363 := (publicBatchWrapper86_copies105 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_49_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_8
  simp only [← c3353, ← c3354, ← c3355] at e_49_8
  have e_49_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_9
  simp only [← c3356, ← c3357, ← c3358] at e_49_9
  have e_50_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_2
  simp only [← c3350, k3, ← c3351, k3, ← c3352] at e_50_2
  have e_50_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_3
  simp only [← c3359, ← c3360, k3, ← c3361] at e_50_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_8
    linear_combination c3362.trans k2 - hc
  · have hc := e_50_3
    simp only [e_50_2, e_49_9] at hc
    linear_combination c3363.trans k2 - hc

theorem publicBatchWrapper86_f438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10007)) (a (.virt 200)) (a (.virt 17611)) (a (.virt 17612)) := by
  have c3364 := (publicBatchWrapper86_copies105 a h).2.2.2.2.1
  have c3365 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.1
  have c3366 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.1
  have c3367 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.1
  have c3368 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.1
  have c3369 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.1
  have c3370 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3371 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3372 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3373 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3374 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3375 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3376 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3377 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_49_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_10
  simp only [← c3367, ← c3368, ← c3369] at e_49_10
  have e_49_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_11
  simp only [← c3370, ← c3371, ← c3372] at e_49_11
  have e_50_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_4
  simp only [← c3364, k3, ← c3365, k3, ← c3366] at e_50_4
  have e_50_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_5
  simp only [← c3373, ← c3374, k3, ← c3375] at e_50_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_10
    linear_combination c3376.trans k2 - hc
  · have hc := e_50_5
    simp only [e_50_4, e_49_11] at hc
    linear_combination c3377.trans k2 - hc

theorem publicBatchWrapper86_f439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10008)) (a (.virt 200)) (a (.virt 17613)) (a (.virt 17614)) := by
  have c3378 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3379 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3380 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3381 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3382 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3383 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3384 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3385 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3386 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3387 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3388 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3389 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3390 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3391 := (publicBatchWrapper86_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_49_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_12
  simp only [← c3381, ← c3382, ← c3383] at e_49_12
  have e_49_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_13
  simp only [← c3384, ← c3385, ← c3386] at e_49_13
  have e_50_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_6
  simp only [← c3378, k3, ← c3379, k3, ← c3380] at e_50_6
  have e_50_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_7
  simp only [← c3387, ← c3388, k3, ← c3389] at e_50_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_12
    linear_combination c3390.trans k2 - hc
  · have hc := e_50_7
    simp only [e_50_6, e_49_13] at hc
    linear_combination c3391.trans k2 - hc

theorem publicBatchWrapper86_f440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 49 59) = band (a (.virt 17607)) (a (.virt 17609)) := by
  have c3392 := (publicBatchWrapper86_copies106 a h).1
  have c3393 := (publicBatchWrapper86_copies106 a h).2.1
  have c3394 := (publicBatchWrapper86_copies106 a h).2.2.1
  have e_49_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_14
  simp only [← c3392, ← c3393, ← c3394] at e_49_14
  have hr := e_49_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 49 63) = band (a (.virt 17611)) (a (.virt 17613)) := by
  have c3395 := (publicBatchWrapper86_copies106 a h).2.2.2.1
  have c3396 := (publicBatchWrapper86_copies106 a h).2.2.2.2.1
  have c3397 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.1
  have e_49_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_15
  simp only [← c3395, ← c3396, ← c3397] at e_49_15
  have hr := e_49_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 49 67) = band (a (.wire 49 59)) (a (.wire 49 63)) := by
  have c3398 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.1
  have c3399 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.1
  have c3400 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.1
  have e_49_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_16
  simp only [← c3398, ← c3399, ← c3400] at e_49_16
  have hr := e_49_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10205)) (a (.virt 200)) (a (.virt 17615)) (a (.virt 17616)) := by
  have c3401 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.1
  have c3402 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3403 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3404 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3405 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3406 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3407 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3408 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3409 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3410 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3411 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3412 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3413 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3414 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_49_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_17
  simp only [← c3404, ← c3405, ← c3406] at e_49_17
  have e_49_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_18
  simp only [← c3407, ← c3408, ← c3409] at e_49_18
  have e_50_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_8
  simp only [← c3401, k3, ← c3402, k3, ← c3403] at e_50_8
  have e_50_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_9
  simp only [← c3410, ← c3411, k3, ← c3412] at e_50_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_17
    linear_combination c3413.trans k2 - hc
  · have hc := e_50_9
    simp only [e_50_8, e_49_18] at hc
    linear_combination c3414.trans k2 - hc

theorem publicBatchWrapper86_f444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10206)) (a (.virt 200)) (a (.virt 17617)) (a (.virt 17618)) := by
  have c3415 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3416 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3417 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3418 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3419 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3420 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3421 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3422 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3423 := (publicBatchWrapper86_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3424 := (publicBatchWrapper86_copies107 a h).1
  have c3425 := (publicBatchWrapper86_copies107 a h).2.1
  have c3426 := (publicBatchWrapper86_copies107 a h).2.2.1
  have c3427 := (publicBatchWrapper86_copies107 a h).2.2.2.1
  have c3428 := (publicBatchWrapper86_copies107 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_49_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row49 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_19
  simp only [← c3418, ← c3419, ← c3420] at e_49_19
  have e_50_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_10
  simp only [← c3415, k3, ← c3416, k3, ← c3417] at e_50_10
  have e_50_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_11
  simp only [← c3424, ← c3425, k3, ← c3426] at e_50_11
  have e_51_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_0
  simp only [← c3421, ← c3422, ← c3423] at e_51_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_49_19
    linear_combination c3427.trans k2 - hc
  · have hc := e_50_11
    simp only [e_50_10, e_51_0] at hc
    linear_combination c3428.trans k2 - hc

theorem publicBatchWrapper86_f445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10207)) (a (.virt 200)) (a (.virt 17619)) (a (.virt 17620)) := by
  have c3429 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.1
  have c3430 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.1
  have c3431 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.1
  have c3432 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.1
  have c3433 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.1
  have c3434 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3435 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3436 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3437 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3438 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3439 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3440 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3441 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3442 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_50_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_12
  simp only [← c3429, k3, ← c3430, k3, ← c3431] at e_50_12
  have e_50_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_13
  simp only [← c3438, ← c3439, k3, ← c3440] at e_50_13
  have e_51_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_1
  simp only [← c3432, ← c3433, ← c3434] at e_51_1
  have e_51_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_2
  simp only [← c3435, ← c3436, ← c3437] at e_51_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_1
    linear_combination c3441.trans k2 - hc
  · have hc := e_50_13
    simp only [e_50_12, e_51_2] at hc
    linear_combination c3442.trans k2 - hc

theorem publicBatchWrapper86_f446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10208)) (a (.virt 200)) (a (.virt 17621)) (a (.virt 17622)) := by
  have c3443 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3444 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3445 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3446 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3447 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3448 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3449 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3450 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3451 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3452 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3453 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3454 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3455 := (publicBatchWrapper86_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3456 := (publicBatchWrapper86_copies108 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_50_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_14
  simp only [← c3443, k3, ← c3444, k3, ← c3445] at e_50_14
  have e_50_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_15
  simp only [← c3452, ← c3453, k3, ← c3454] at e_50_15
  have e_51_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_3
  simp only [← c3446, ← c3447, ← c3448] at e_51_3
  have e_51_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_4
  simp only [← c3449, ← c3450, ← c3451] at e_51_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_3
    linear_combination c3455.trans k2 - hc
  · have hc := e_50_15
    simp only [e_50_14, e_51_4] at hc
    linear_combination c3456.trans k2 - hc

theorem publicBatchWrapper86_f447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 51 23) = band (a (.virt 17615)) (a (.virt 17617)) := by
  have c3457 := (publicBatchWrapper86_copies108 a h).2.1
  have c3458 := (publicBatchWrapper86_copies108 a h).2.2.1
  have c3459 := (publicBatchWrapper86_copies108 a h).2.2.2.1
  have e_51_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_5
  simp only [← c3457, ← c3458, ← c3459] at e_51_5
  have hr := e_51_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 51 27) = band (a (.virt 17619)) (a (.virt 17621)) := by
  have c3460 := (publicBatchWrapper86_copies108 a h).2.2.2.2.1
  have c3461 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.1
  have c3462 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.1
  have e_51_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_6
  simp only [← c3460, ← c3461, ← c3462] at e_51_6
  have hr := e_51_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 51 31) = band (a (.wire 51 23)) (a (.wire 51 27)) := by
  have c3463 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.1
  have c3464 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.1
  have c3465 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.1
  have e_51_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_7
  simp only [← c3463, ← c3464, ← c3465] at e_51_7
  have hr := e_51_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10405)) (a (.virt 200)) (a (.virt 17623)) (a (.virt 17624)) := by
  have c3466 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3467 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3468 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3469 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3470 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3471 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3472 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3473 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3474 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3479 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_50_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_16
  simp only [← c3466, k3, ← c3467, k3, ← c3468] at e_50_16
  have e_50_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_17
  simp only [← c3475, ← c3476, k3, ← c3477] at e_50_17
  have e_51_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_8
  simp only [← c3469, ← c3470, ← c3471] at e_51_8
  have e_51_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_9
  simp only [← c3472, ← c3473, ← c3474] at e_51_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_8
    linear_combination c3478.trans k2 - hc
  · have hc := e_50_17
    simp only [e_50_16, e_51_9] at hc
    linear_combination c3479.trans k2 - hc

theorem publicBatchWrapper86_f451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10406)) (a (.virt 200)) (a (.virt 17625)) (a (.virt 17626)) := by
  have c3480 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3481 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3482 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3483 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3484 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3485 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3486 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3487 := (publicBatchWrapper86_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3488 := (publicBatchWrapper86_copies109 a h).1
  have c3489 := (publicBatchWrapper86_copies109 a h).2.1
  have c3490 := (publicBatchWrapper86_copies109 a h).2.2.1
  have c3491 := (publicBatchWrapper86_copies109 a h).2.2.2.1
  have c3492 := (publicBatchWrapper86_copies109 a h).2.2.2.2.1
  have c3493 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_50_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_18
  simp only [← c3480, k3, ← c3481, k3, ← c3482] at e_50_18
  have e_50_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row50 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_19
  simp only [← c3489, ← c3490, k3, ← c3491] at e_50_19
  have e_51_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_10
  simp only [← c3483, ← c3484, ← c3485] at e_51_10
  have e_51_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_11
  simp only [← c3486, ← c3487, ← c3488] at e_51_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_10
    linear_combination c3492.trans k2 - hc
  · have hc := e_50_19
    simp only [e_50_18, e_51_11] at hc
    linear_combination c3493.trans k2 - hc

theorem publicBatchWrapper86_f452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10407)) (a (.virt 200)) (a (.virt 17627)) (a (.virt 17628)) := by
  have c3494 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.1
  have c3495 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.1
  have c3496 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.1
  have c3497 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.1
  have c3498 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3499 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3500 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3501 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3502 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3503 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3504 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3505 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3506 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3507 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_51_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_12
  simp only [← c3497, ← c3498, ← c3499] at e_51_12
  have e_51_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_13
  simp only [← c3500, ← c3501, ← c3502] at e_51_13
  have e_52_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_0
  simp only [← c3494, k3, ← c3495, k3, ← c3496] at e_52_0
  have e_52_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_1
  simp only [← c3503, ← c3504, k3, ← c3505] at e_52_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_12
    linear_combination c3506.trans k2 - hc
  · have hc := e_52_1
    simp only [e_52_0, e_51_13] at hc
    linear_combination c3507.trans k2 - hc

theorem publicBatchWrapper86_f453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10408)) (a (.virt 200)) (a (.virt 17629)) (a (.virt 17630)) := by
  have c3508 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3509 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3510 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3511 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3512 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3513 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3514 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3515 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3516 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3517 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3518 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3519 := (publicBatchWrapper86_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3520 := (publicBatchWrapper86_copies110 a h).1
  have c3521 := (publicBatchWrapper86_copies110 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_51_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_14
  simp only [← c3511, ← c3512, ← c3513] at e_51_14
  have e_51_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_15
  simp only [← c3514, ← c3515, ← c3516] at e_51_15
  have e_52_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_2
  simp only [← c3508, k3, ← c3509, k3, ← c3510] at e_52_2
  have e_52_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_3
  simp only [← c3517, ← c3518, k3, ← c3519] at e_52_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_14
    linear_combination c3520.trans k2 - hc
  · have hc := e_52_3
    simp only [e_52_2, e_51_15] at hc
    linear_combination c3521.trans k2 - hc

theorem publicBatchWrapper86_f454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 51 67) = band (a (.virt 17623)) (a (.virt 17625)) := by
  have c3522 := (publicBatchWrapper86_copies110 a h).2.2.1
  have c3523 := (publicBatchWrapper86_copies110 a h).2.2.2.1
  have c3524 := (publicBatchWrapper86_copies110 a h).2.2.2.2.1
  have e_51_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_16
  simp only [← c3522, ← c3523, ← c3524] at e_51_16
  have hr := e_51_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 51 71) = band (a (.virt 17627)) (a (.virt 17629)) := by
  have c3525 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.1
  have c3526 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.1
  have c3527 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.1
  have e_51_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_17
  simp only [← c3525, ← c3526, ← c3527] at e_51_17
  have hr := e_51_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 51 75) = band (a (.wire 51 67)) (a (.wire 51 71)) := by
  have c3528 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.1
  have c3529 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.1
  have c3530 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_51_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_18
  simp only [← c3528, ← c3529, ← c3530] at e_51_18
  have hr := e_51_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10605)) (a (.virt 200)) (a (.virt 17631)) (a (.virt 17632)) := by
  have c3531 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3532 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3533 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3534 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3535 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3536 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3537 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3538 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3539 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3540 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3541 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3542 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3543 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3544 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_51_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row51 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_19
  simp only [← c3534, ← c3535, ← c3536] at e_51_19
  have e_52_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_4
  simp only [← c3531, k3, ← c3532, k3, ← c3533] at e_52_4
  have e_52_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_5
  simp only [← c3540, ← c3541, k3, ← c3542] at e_52_5
  have e_53_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_0
  simp only [← c3537, ← c3538, ← c3539] at e_53_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_51_19
    linear_combination c3543.trans k2 - hc
  · have hc := e_52_5
    simp only [e_52_4, e_53_0] at hc
    linear_combination c3544.trans k2 - hc

theorem publicBatchWrapper86_f458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10606)) (a (.virt 200)) (a (.virt 17633)) (a (.virt 17634)) := by
  have c3545 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3546 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3547 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3548 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3549 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3550 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3551 := (publicBatchWrapper86_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3552 := (publicBatchWrapper86_copies111 a h).1
  have c3553 := (publicBatchWrapper86_copies111 a h).2.1
  have c3554 := (publicBatchWrapper86_copies111 a h).2.2.1
  have c3555 := (publicBatchWrapper86_copies111 a h).2.2.2.1
  have c3556 := (publicBatchWrapper86_copies111 a h).2.2.2.2.1
  have c3557 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.1
  have c3558 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_6
  simp only [← c3545, k3, ← c3546, k3, ← c3547] at e_52_6
  have e_52_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_7
  simp only [← c3554, ← c3555, k3, ← c3556] at e_52_7
  have e_53_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_1
  simp only [← c3548, ← c3549, ← c3550] at e_53_1
  have e_53_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_2
  simp only [← c3551, ← c3552, ← c3553] at e_53_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_1
    linear_combination c3557.trans k2 - hc
  · have hc := e_52_7
    simp only [e_52_6, e_53_2] at hc
    linear_combination c3558.trans k2 - hc

theorem publicBatchWrapper86_f459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10607)) (a (.virt 200)) (a (.virt 17635)) (a (.virt 17636)) := by
  have c3559 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.1
  have c3560 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.1
  have c3561 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.1
  have c3562 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3563 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3564 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3565 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3566 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3567 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3568 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3569 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3570 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3571 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3572 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_8
  simp only [← c3559, k3, ← c3560, k3, ← c3561] at e_52_8
  have e_52_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_9
  simp only [← c3568, ← c3569, k3, ← c3570] at e_52_9
  have e_53_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_3
  simp only [← c3562, ← c3563, ← c3564] at e_53_3
  have e_53_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_4
  simp only [← c3565, ← c3566, ← c3567] at e_53_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_3
    linear_combination c3571.trans k2 - hc
  · have hc := e_52_9
    simp only [e_52_8, e_53_4] at hc
    linear_combination c3572.trans k2 - hc

theorem publicBatchWrapper86_f460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10608)) (a (.virt 200)) (a (.virt 17637)) (a (.virt 17638)) := by
  have c3573 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3574 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3575 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3576 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3577 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3578 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3579 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3580 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3581 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3582 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3583 := (publicBatchWrapper86_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3584 := (publicBatchWrapper86_copies112 a h).1
  have c3585 := (publicBatchWrapper86_copies112 a h).2.1
  have c3586 := (publicBatchWrapper86_copies112 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_10
  simp only [← c3573, k3, ← c3574, k3, ← c3575] at e_52_10
  have e_52_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_11
  simp only [← c3582, ← c3583, k3, ← c3584] at e_52_11
  have e_53_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_5
  simp only [← c3576, ← c3577, ← c3578] at e_53_5
  have e_53_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_6
  simp only [← c3579, ← c3580, ← c3581] at e_53_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_5
    linear_combination c3585.trans k2 - hc
  · have hc := e_52_11
    simp only [e_52_10, e_53_6] at hc
    linear_combination c3586.trans k2 - hc

theorem publicBatchWrapper86_f461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 53 31) = band (a (.virt 17631)) (a (.virt 17633)) := by
  have c3587 := (publicBatchWrapper86_copies112 a h).2.2.2.1
  have c3588 := (publicBatchWrapper86_copies112 a h).2.2.2.2.1
  have c3589 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.1
  have e_53_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_7
  simp only [← c3587, ← c3588, ← c3589] at e_53_7
  have hr := e_53_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 53 35) = band (a (.virt 17635)) (a (.virt 17637)) := by
  have c3590 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.1
  have c3591 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.1
  have c3592 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.1
  have e_53_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_8
  simp only [← c3590, ← c3591, ← c3592] at e_53_8
  have hr := e_53_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 53 39) = band (a (.wire 53 31)) (a (.wire 53 35)) := by
  have c3593 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.1
  have c3594 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3595 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_9
  simp only [← c3593, ← c3594, ← c3595] at e_53_9
  have hr := e_53_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10805)) (a (.virt 200)) (a (.virt 17639)) (a (.virt 17640)) := by
  have c3596 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3597 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3598 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3599 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3600 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3601 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3602 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3603 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3604 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3605 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3606 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3607 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3608 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3609 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_12
  simp only [← c3596, k3, ← c3597, k3, ← c3598] at e_52_12
  have e_52_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_13
  simp only [← c3605, ← c3606, k3, ← c3607] at e_52_13
  have e_53_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_10
  simp only [← c3599, ← c3600, ← c3601] at e_53_10
  have e_53_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_11
  simp only [← c3602, ← c3603, ← c3604] at e_53_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_10
    linear_combination c3608.trans k2 - hc
  · have hc := e_52_13
    simp only [e_52_12, e_53_11] at hc
    linear_combination c3609.trans k2 - hc

theorem publicBatchWrapper86_f465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10806)) (a (.virt 200)) (a (.virt 17641)) (a (.virt 17642)) := by
  have c3610 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3611 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3612 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3613 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3614 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3615 := (publicBatchWrapper86_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3616 := (publicBatchWrapper86_copies113 a h).1
  have c3617 := (publicBatchWrapper86_copies113 a h).2.1
  have c3618 := (publicBatchWrapper86_copies113 a h).2.2.1
  have c3619 := (publicBatchWrapper86_copies113 a h).2.2.2.1
  have c3620 := (publicBatchWrapper86_copies113 a h).2.2.2.2.1
  have c3621 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.1
  have c3622 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.1
  have c3623 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_14
  simp only [← c3610, k3, ← c3611, k3, ← c3612] at e_52_14
  have e_52_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_15
  simp only [← c3619, ← c3620, k3, ← c3621] at e_52_15
  have e_53_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_12
  simp only [← c3613, ← c3614, ← c3615] at e_53_12
  have e_53_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_13
  simp only [← c3616, ← c3617, ← c3618] at e_53_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_12
    linear_combination c3622.trans k2 - hc
  · have hc := e_52_15
    simp only [e_52_14, e_53_13] at hc
    linear_combination c3623.trans k2 - hc

theorem publicBatchWrapper86_f466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10807)) (a (.virt 200)) (a (.virt 17643)) (a (.virt 17644)) := by
  have c3624 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.1
  have c3625 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.1
  have c3626 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3627 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3628 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3629 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3630 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3631 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3632 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3633 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3634 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3635 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3636 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3637 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_16
  simp only [← c3624, k3, ← c3625, k3, ← c3626] at e_52_16
  have e_52_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_17
  simp only [← c3633, ← c3634, k3, ← c3635] at e_52_17
  have e_53_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_14
  simp only [← c3627, ← c3628, ← c3629] at e_53_14
  have e_53_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_15
  simp only [← c3630, ← c3631, ← c3632] at e_53_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_14
    linear_combination c3636.trans k2 - hc
  · have hc := e_52_17
    simp only [e_52_16, e_53_15] at hc
    linear_combination c3637.trans k2 - hc

theorem publicBatchWrapper86_f467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 10808)) (a (.virt 200)) (a (.virt 17645)) (a (.virt 17646)) := by
  have c3638 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3639 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3640 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3641 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3642 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3643 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3644 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3645 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3646 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3647 := (publicBatchWrapper86_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3648 := (publicBatchWrapper86_copies114 a h).1
  have c3649 := (publicBatchWrapper86_copies114 a h).2.1
  have c3650 := (publicBatchWrapper86_copies114 a h).2.2.1
  have c3651 := (publicBatchWrapper86_copies114 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_52_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_18
  simp only [← c3638, k3, ← c3639, k3, ← c3640] at e_52_18
  have e_52_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row52 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_19
  simp only [← c3647, ← c3648, k3, ← c3649] at e_52_19
  have e_53_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_16
  simp only [← c3641, ← c3642, ← c3643] at e_53_16
  have e_53_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_17
  simp only [← c3644, ← c3645, ← c3646] at e_53_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_16
    linear_combination c3650.trans k2 - hc
  · have hc := e_52_19
    simp only [e_52_18, e_53_17] at hc
    linear_combination c3651.trans k2 - hc

theorem publicBatchWrapper86_f468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 53 75) = band (a (.virt 17639)) (a (.virt 17641)) := by
  have c3652 := (publicBatchWrapper86_copies114 a h).2.2.2.2.1
  have c3653 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.1
  have c3654 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.1
  have e_53_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_18
  simp only [← c3652, ← c3653, ← c3654] at e_53_18
  have hr := e_53_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 53 79) = band (a (.virt 17643)) (a (.virt 17645)) := by
  have c3655 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.1
  have c3656 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.1
  have c3657 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.1
  have e_53_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row53 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_19
  simp only [← c3655, ← c3656, ← c3657] at e_53_19
  have hr := e_53_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 54 3) = band (a (.wire 53 75)) (a (.wire 53 79)) := by
  have c3658 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3659 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3660 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_0
  simp only [← c3658, ← c3659, ← c3660] at e_54_0
  have hr := e_54_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11005)) (a (.virt 200)) (a (.virt 17647)) (a (.virt 17648)) := by
  have c3661 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3662 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3663 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3664 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3665 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3666 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3667 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3668 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3669 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3670 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3671 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3672 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3673 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3674 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_1
  simp only [← c3664, ← c3665, ← c3666] at e_54_1
  have e_54_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_2
  simp only [← c3667, ← c3668, ← c3669] at e_54_2
  have e_55_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_0
  simp only [← c3661, k3, ← c3662, k3, ← c3663] at e_55_0
  have e_55_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_1
  simp only [← c3670, ← c3671, k3, ← c3672] at e_55_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_1
    linear_combination c3673.trans k2 - hc
  · have hc := e_55_1
    simp only [e_55_0, e_54_2] at hc
    linear_combination c3674.trans k2 - hc

theorem publicBatchWrapper86_f472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11006)) (a (.virt 200)) (a (.virt 17649)) (a (.virt 17650)) := by
  have c3675 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3676 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3677 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3678 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3679 := (publicBatchWrapper86_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3680 := (publicBatchWrapper86_copies115 a h).1
  have c3681 := (publicBatchWrapper86_copies115 a h).2.1
  have c3682 := (publicBatchWrapper86_copies115 a h).2.2.1
  have c3683 := (publicBatchWrapper86_copies115 a h).2.2.2.1
  have c3684 := (publicBatchWrapper86_copies115 a h).2.2.2.2.1
  have c3685 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.1
  have c3686 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.1
  have c3687 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.1
  have c3688 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_3
  simp only [← c3678, ← c3679, ← c3680] at e_54_3
  have e_54_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_4
  simp only [← c3681, ← c3682, ← c3683] at e_54_4
  have e_55_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_2
  simp only [← c3675, k3, ← c3676, k3, ← c3677] at e_55_2
  have e_55_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_3
  simp only [← c3684, ← c3685, k3, ← c3686] at e_55_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_3
    linear_combination c3687.trans k2 - hc
  · have hc := e_55_3
    simp only [e_55_2, e_54_4] at hc
    linear_combination c3688.trans k2 - hc

theorem publicBatchWrapper86_f473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11007)) (a (.virt 200)) (a (.virt 17651)) (a (.virt 17652)) := by
  have c3689 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.1
  have c3690 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3691 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3692 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3693 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3694 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3695 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3696 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3697 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3698 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3699 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3700 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3701 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3702 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_5
  simp only [← c3692, ← c3693, ← c3694] at e_54_5
  have e_54_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_6
  simp only [← c3695, ← c3696, ← c3697] at e_54_6
  have e_55_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_4
  simp only [← c3689, k3, ← c3690, k3, ← c3691] at e_55_4
  have e_55_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_5
  simp only [← c3698, ← c3699, k3, ← c3700] at e_55_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_5
    linear_combination c3701.trans k2 - hc
  · have hc := e_55_5
    simp only [e_55_4, e_54_6] at hc
    linear_combination c3702.trans k2 - hc

theorem publicBatchWrapper86_f474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11008)) (a (.virt 200)) (a (.virt 17653)) (a (.virt 17654)) := by
  have c3703 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3704 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3705 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3706 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3707 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3708 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3709 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3710 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3711 := (publicBatchWrapper86_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3712 := (publicBatchWrapper86_copies116 a h).1
  have c3713 := (publicBatchWrapper86_copies116 a h).2.1
  have c3714 := (publicBatchWrapper86_copies116 a h).2.2.1
  have c3715 := (publicBatchWrapper86_copies116 a h).2.2.2.1
  have c3716 := (publicBatchWrapper86_copies116 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_7
  simp only [← c3706, ← c3707, ← c3708] at e_54_7
  have e_54_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_8
  simp only [← c3709, ← c3710, ← c3711] at e_54_8
  have e_55_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_6
  simp only [← c3703, k3, ← c3704, k3, ← c3705] at e_55_6
  have e_55_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_7
  simp only [← c3712, ← c3713, k3, ← c3714] at e_55_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_7
    linear_combination c3715.trans k2 - hc
  · have hc := e_55_7
    simp only [e_55_6, e_54_8] at hc
    linear_combination c3716.trans k2 - hc

theorem publicBatchWrapper86_f475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 54 39) = band (a (.virt 17647)) (a (.virt 17649)) := by
  have c3717 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.1
  have c3718 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.1
  have c3719 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.1
  have e_54_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_9
  simp only [← c3717, ← c3718, ← c3719] at e_54_9
  have hr := e_54_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 54 43) = band (a (.virt 17651)) (a (.virt 17653)) := by
  have c3720 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.1
  have c3721 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.1
  have c3722 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_54_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_10
  simp only [← c3720, ← c3721, ← c3722] at e_54_10
  have hr := e_54_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 54 47) = band (a (.wire 54 39)) (a (.wire 54 43)) := by
  have c3723 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3724 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3725 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_11
  simp only [← c3723, ← c3724, ← c3725] at e_54_11
  have hr := e_54_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11205)) (a (.virt 200)) (a (.virt 17655)) (a (.virt 17656)) := by
  have c3726 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3727 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3728 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3729 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3730 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3731 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3732 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3733 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3734 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3735 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3736 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3737 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3738 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3739 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_12
  simp only [← c3729, ← c3730, ← c3731] at e_54_12
  have e_54_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_13
  simp only [← c3732, ← c3733, ← c3734] at e_54_13
  have e_55_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_8
  simp only [← c3726, k3, ← c3727, k3, ← c3728] at e_55_8
  have e_55_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_9
  simp only [← c3735, ← c3736, k3, ← c3737] at e_55_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_12
    linear_combination c3738.trans k2 - hc
  · have hc := e_55_9
    simp only [e_55_8, e_54_13] at hc
    linear_combination c3739.trans k2 - hc

theorem publicBatchWrapper86_f479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11206)) (a (.virt 200)) (a (.virt 17657)) (a (.virt 17658)) := by
  have c3740 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3741 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3742 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3743 := (publicBatchWrapper86_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3744 := (publicBatchWrapper86_copies117 a h).1
  have c3745 := (publicBatchWrapper86_copies117 a h).2.1
  have c3746 := (publicBatchWrapper86_copies117 a h).2.2.1
  have c3747 := (publicBatchWrapper86_copies117 a h).2.2.2.1
  have c3748 := (publicBatchWrapper86_copies117 a h).2.2.2.2.1
  have c3749 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.1
  have c3750 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.1
  have c3751 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.1
  have c3752 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.1
  have c3753 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_14
  simp only [← c3743, ← c3744, ← c3745] at e_54_14
  have e_54_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_15
  simp only [← c3746, ← c3747, ← c3748] at e_54_15
  have e_55_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_10
  simp only [← c3740, k3, ← c3741, k3, ← c3742] at e_55_10
  have e_55_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_11
  simp only [← c3749, ← c3750, k3, ← c3751] at e_55_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_14
    linear_combination c3752.trans k2 - hc
  · have hc := e_55_11
    simp only [e_55_10, e_54_15] at hc
    linear_combination c3753.trans k2 - hc

theorem publicBatchWrapper86_f480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11207)) (a (.virt 200)) (a (.virt 17659)) (a (.virt 17660)) := by
  have c3754 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3755 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3756 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3757 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3758 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3759 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3760 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3761 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3762 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3763 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3764 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3765 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3766 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3767 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_16
  simp only [← c3757, ← c3758, ← c3759] at e_54_16
  have e_54_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_17
  simp only [← c3760, ← c3761, ← c3762] at e_54_17
  have e_55_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_12
  simp only [← c3754, k3, ← c3755, k3, ← c3756] at e_55_12
  have e_55_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_13
  simp only [← c3763, ← c3764, k3, ← c3765] at e_55_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_16
    linear_combination c3766.trans k2 - hc
  · have hc := e_55_13
    simp only [e_55_12, e_54_17] at hc
    linear_combination c3767.trans k2 - hc

theorem publicBatchWrapper86_f481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11208)) (a (.virt 200)) (a (.virt 17661)) (a (.virt 17662)) := by
  have c3768 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3769 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3770 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3771 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3772 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3773 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3774 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3775 := (publicBatchWrapper86_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3776 := (publicBatchWrapper86_copies118 a h).1
  have c3777 := (publicBatchWrapper86_copies118 a h).2.1
  have c3778 := (publicBatchWrapper86_copies118 a h).2.2.1
  have c3779 := (publicBatchWrapper86_copies118 a h).2.2.2.1
  have c3780 := (publicBatchWrapper86_copies118 a h).2.2.2.2.1
  have c3781 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_54_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_18
  simp only [← c3771, ← c3772, ← c3773] at e_54_18
  have e_54_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row54 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_19
  simp only [← c3774, ← c3775, ← c3776] at e_54_19
  have e_55_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_14
  simp only [← c3768, k3, ← c3769, k3, ← c3770] at e_55_14
  have e_55_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_15
  simp only [← c3777, ← c3778, k3, ← c3779] at e_55_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_54_18
    linear_combination c3780.trans k2 - hc
  · have hc := e_55_15
    simp only [e_55_14, e_54_19] at hc
    linear_combination c3781.trans k2 - hc

theorem publicBatchWrapper86_f482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 56 3) = band (a (.virt 17655)) (a (.virt 17657)) := by
  have c3782 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.1
  have c3783 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.1
  have c3784 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.1
  have e_56_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_0
  simp only [← c3782, ← c3783, ← c3784] at e_56_0
  have hr := e_56_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 56 7) = band (a (.virt 17659)) (a (.virt 17661)) := by
  have c3785 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.1
  have c3786 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3787 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_1
  simp only [← c3785, ← c3786, ← c3787] at e_56_1
  have hr := e_56_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 56 11) = band (a (.wire 56 3)) (a (.wire 56 7)) := by
  have c3788 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3789 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3790 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_2
  simp only [← c3788, ← c3789, ← c3790] at e_56_2
  have hr := e_56_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11405)) (a (.virt 200)) (a (.virt 17663)) (a (.virt 17664)) := by
  have c3791 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3792 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3793 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3794 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3795 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3796 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3797 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3798 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3799 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3800 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3801 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3802 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3803 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3804 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_55_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_16
  simp only [← c3791, k3, ← c3792, k3, ← c3793] at e_55_16
  have e_55_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_17
  simp only [← c3800, ← c3801, k3, ← c3802] at e_55_17
  have e_56_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_3
  simp only [← c3794, ← c3795, ← c3796] at e_56_3
  have e_56_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_4
  simp only [← c3797, ← c3798, ← c3799] at e_56_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_3
    linear_combination c3803.trans k2 - hc
  · have hc := e_55_17
    simp only [e_55_16, e_56_4] at hc
    linear_combination c3804.trans k2 - hc

theorem publicBatchWrapper86_f486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11406)) (a (.virt 200)) (a (.virt 17665)) (a (.virt 17666)) := by
  have c3805 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3806 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3807 := (publicBatchWrapper86_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3808 := (publicBatchWrapper86_copies119 a h).1
  have c3809 := (publicBatchWrapper86_copies119 a h).2.1
  have c3810 := (publicBatchWrapper86_copies119 a h).2.2.1
  have c3811 := (publicBatchWrapper86_copies119 a h).2.2.2.1
  have c3812 := (publicBatchWrapper86_copies119 a h).2.2.2.2.1
  have c3813 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.1
  have c3814 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.1
  have c3815 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.1
  have c3816 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.1
  have c3817 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.1
  have c3818 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_55_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_18
  simp only [← c3805, k3, ← c3806, k3, ← c3807] at e_55_18
  have e_55_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row55 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_19
  simp only [← c3814, ← c3815, k3, ← c3816] at e_55_19
  have e_56_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_5
  simp only [← c3808, ← c3809, ← c3810] at e_56_5
  have e_56_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_6
  simp only [← c3811, ← c3812, ← c3813] at e_56_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_5
    linear_combination c3817.trans k2 - hc
  · have hc := e_55_19
    simp only [e_55_18, e_56_6] at hc
    linear_combination c3818.trans k2 - hc

theorem publicBatchWrapper86_f487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11407)) (a (.virt 200)) (a (.virt 17667)) (a (.virt 17668)) := by
  have c3819 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3820 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3821 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3822 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3823 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3824 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3825 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3826 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3827 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3828 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3829 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3830 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3831 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3832 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_56_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_7
  simp only [← c3822, ← c3823, ← c3824] at e_56_7
  have e_56_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_8
  simp only [← c3825, ← c3826, ← c3827] at e_56_8
  have e_57_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_0
  simp only [← c3819, k3, ← c3820, k3, ← c3821] at e_57_0
  have e_57_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_1
  simp only [← c3828, ← c3829, k3, ← c3830] at e_57_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_7
    linear_combination c3831.trans k2 - hc
  · have hc := e_57_1
    simp only [e_57_0, e_56_8] at hc
    linear_combination c3832.trans k2 - hc

theorem publicBatchWrapper86_f488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11408)) (a (.virt 200)) (a (.virt 17669)) (a (.virt 17670)) := by
  have c3833 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3834 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3835 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3836 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3837 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3838 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3839 := (publicBatchWrapper86_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3840 := (publicBatchWrapper86_copies120 a h).1
  have c3841 := (publicBatchWrapper86_copies120 a h).2.1
  have c3842 := (publicBatchWrapper86_copies120 a h).2.2.1
  have c3843 := (publicBatchWrapper86_copies120 a h).2.2.2.1
  have c3844 := (publicBatchWrapper86_copies120 a h).2.2.2.2.1
  have c3845 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.1
  have c3846 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_56_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_9
  simp only [← c3836, ← c3837, ← c3838] at e_56_9
  have e_56_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_10
  simp only [← c3839, ← c3840, ← c3841] at e_56_10
  have e_57_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_2
  simp only [← c3833, k3, ← c3834, k3, ← c3835] at e_57_2
  have e_57_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_3
  simp only [← c3842, ← c3843, k3, ← c3844] at e_57_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_9
    linear_combination c3845.trans k2 - hc
  · have hc := e_57_3
    simp only [e_57_2, e_56_10] at hc
    linear_combination c3846.trans k2 - hc

theorem publicBatchWrapper86_f489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 56 47) = band (a (.virt 17663)) (a (.virt 17665)) := by
  have c3847 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.1
  have c3848 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.1
  have c3849 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.1
  have e_56_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_11
  simp only [← c3847, ← c3848, ← c3849] at e_56_11
  have hr := e_56_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 56 51) = band (a (.virt 17667)) (a (.virt 17669)) := by
  have c3850 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3851 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3852 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_12
  simp only [← c3850, ← c3851, ← c3852] at e_56_12
  have hr := e_56_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 56 55) = band (a (.wire 56 47)) (a (.wire 56 51)) := by
  have c3853 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3854 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3855 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_56_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_13
  simp only [← c3853, ← c3854, ← c3855] at e_56_13
  have hr := e_56_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11605)) (a (.virt 200)) (a (.virt 17671)) (a (.virt 17672)) := by
  have c3856 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3857 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3858 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3859 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3860 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3861 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3862 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3863 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3864 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3865 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3866 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3867 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3868 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3869 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_56_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_14
  simp only [← c3859, ← c3860, ← c3861] at e_56_14
  have e_56_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_15
  simp only [← c3862, ← c3863, ← c3864] at e_56_15
  have e_57_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_4
  simp only [← c3856, k3, ← c3857, k3, ← c3858] at e_57_4
  have e_57_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_5
  simp only [← c3865, ← c3866, k3, ← c3867] at e_57_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_14
    linear_combination c3868.trans k2 - hc
  · have hc := e_57_5
    simp only [e_57_4, e_56_15] at hc
    linear_combination c3869.trans k2 - hc

theorem publicBatchWrapper86_f493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11606)) (a (.virt 200)) (a (.virt 17673)) (a (.virt 17674)) := by
  have c3870 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3871 := (publicBatchWrapper86_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3872 := (publicBatchWrapper86_copies121 a h).1
  have c3873 := (publicBatchWrapper86_copies121 a h).2.1
  have c3874 := (publicBatchWrapper86_copies121 a h).2.2.1
  have c3875 := (publicBatchWrapper86_copies121 a h).2.2.2.1
  have c3876 := (publicBatchWrapper86_copies121 a h).2.2.2.2.1
  have c3877 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.1
  have c3878 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.1
  have c3879 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.1
  have c3880 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.1
  have c3881 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.1
  have c3882 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3883 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_56_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_16
  simp only [← c3873, ← c3874, ← c3875] at e_56_16
  have e_56_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_17
  simp only [← c3876, ← c3877, ← c3878] at e_56_17
  have e_57_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_6
  simp only [← c3870, k3, ← c3871, k3, ← c3872] at e_57_6
  have e_57_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_7
  simp only [← c3879, ← c3880, k3, ← c3881] at e_57_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_16
    linear_combination c3882.trans k2 - hc
  · have hc := e_57_7
    simp only [e_57_6, e_56_17] at hc
    linear_combination c3883.trans k2 - hc

theorem publicBatchWrapper86_f494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11607)) (a (.virt 200)) (a (.virt 17675)) (a (.virt 17676)) := by
  have c3884 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3885 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3886 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3887 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3888 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3889 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3890 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3891 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3892 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3893 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3894 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3895 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3896 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3897 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_56_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_18
  simp only [← c3887, ← c3888, ← c3889] at e_56_18
  have e_56_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row56 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_19
  simp only [← c3890, ← c3891, ← c3892] at e_56_19
  have e_57_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_8
  simp only [← c3884, k3, ← c3885, k3, ← c3886] at e_57_8
  have e_57_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_9
  simp only [← c3893, ← c3894, k3, ← c3895] at e_57_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_56_18
    linear_combination c3896.trans k2 - hc
  · have hc := e_57_9
    simp only [e_57_8, e_56_19] at hc
    linear_combination c3897.trans k2 - hc

theorem publicBatchWrapper86_f495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11608)) (a (.virt 200)) (a (.virt 17677)) (a (.virt 17678)) := by
  have c3898 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3899 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3900 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3901 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3902 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3903 := (publicBatchWrapper86_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3904 := (publicBatchWrapper86_copies122 a h).1
  have c3905 := (publicBatchWrapper86_copies122 a h).2.1
  have c3906 := (publicBatchWrapper86_copies122 a h).2.2.1
  have c3907 := (publicBatchWrapper86_copies122 a h).2.2.2.1
  have c3908 := (publicBatchWrapper86_copies122 a h).2.2.2.2.1
  have c3909 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.1
  have c3910 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.1
  have c3911 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_57_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_10
  simp only [← c3898, k3, ← c3899, k3, ← c3900] at e_57_10
  have e_57_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_11
  simp only [← c3907, ← c3908, k3, ← c3909] at e_57_11
  have e_58_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_0
  simp only [← c3901, ← c3902, ← c3903] at e_58_0
  have e_58_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_1
  simp only [← c3904, ← c3905, ← c3906] at e_58_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_0
    linear_combination c3910.trans k2 - hc
  · have hc := e_57_11
    simp only [e_57_10, e_58_1] at hc
    linear_combination c3911.trans k2 - hc

theorem publicBatchWrapper86_f496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 58 11) = band (a (.virt 17671)) (a (.virt 17673)) := by
  have c3912 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.1
  have c3913 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.1
  have c3914 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_58_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_2
  simp only [← c3912, ← c3913, ← c3914] at e_58_2
  have hr := e_58_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 58 15) = band (a (.virt 17675)) (a (.virt 17677)) := by
  have c3915 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3916 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3917 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_3
  simp only [← c3915, ← c3916, ← c3917] at e_58_3
  have hr := e_58_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 58 19) = band (a (.wire 58 11)) (a (.wire 58 15)) := by
  have c3918 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3919 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3920 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_4
  simp only [← c3918, ← c3919, ← c3920] at e_58_4
  have hr := e_58_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11805)) (a (.virt 200)) (a (.virt 17679)) (a (.virt 17680)) := by
  have c3921 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3922 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3923 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3924 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3925 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3926 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3927 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3928 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3929 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3930 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3931 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3932 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3933 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3934 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_57_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_12
  simp only [← c3921, k3, ← c3922, k3, ← c3923] at e_57_12
  have e_57_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_13
  simp only [← c3930, ← c3931, k3, ← c3932] at e_57_13
  have e_58_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_5
  simp only [← c3924, ← c3925, ← c3926] at e_58_5
  have e_58_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_6
  simp only [← c3927, ← c3928, ← c3929] at e_58_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_5
    linear_combination c3933.trans k2 - hc
  · have hc := e_57_13
    simp only [e_57_12, e_58_6] at hc
    linear_combination c3934.trans k2 - hc

theorem publicBatchWrapper86_f500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11806)) (a (.virt 200)) (a (.virt 17681)) (a (.virt 17682)) := by
  have c3935 := (publicBatchWrapper86_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3936 := (publicBatchWrapper86_copies123 a h).1
  have c3937 := (publicBatchWrapper86_copies123 a h).2.1
  have c3938 := (publicBatchWrapper86_copies123 a h).2.2.1
  have c3939 := (publicBatchWrapper86_copies123 a h).2.2.2.1
  have c3940 := (publicBatchWrapper86_copies123 a h).2.2.2.2.1
  have c3941 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.1
  have c3942 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.1
  have c3943 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.1
  have c3944 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.1
  have c3945 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.1
  have c3946 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3947 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3948 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_57_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_14
  simp only [← c3935, k3, ← c3936, k3, ← c3937] at e_57_14
  have e_57_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_15
  simp only [← c3944, ← c3945, k3, ← c3946] at e_57_15
  have e_58_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_7
  simp only [← c3938, ← c3939, ← c3940] at e_58_7
  have e_58_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_8
  simp only [← c3941, ← c3942, ← c3943] at e_58_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_7
    linear_combination c3947.trans k2 - hc
  · have hc := e_57_15
    simp only [e_57_14, e_58_8] at hc
    linear_combination c3948.trans k2 - hc

theorem publicBatchWrapper86_f501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11807)) (a (.virt 200)) (a (.virt 17683)) (a (.virt 17684)) := by
  have c3949 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3950 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3951 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3952 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3953 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3956 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3958 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3959 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3960 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3961 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3962 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_57_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_16
  simp only [← c3949, k3, ← c3950, k3, ← c3951] at e_57_16
  have e_57_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_17
  simp only [← c3958, ← c3959, k3, ← c3960] at e_57_17
  have e_58_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_9
  simp only [← c3952, ← c3953, ← c3954] at e_58_9
  have e_58_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_10
  simp only [← c3955, ← c3956, ← c3957] at e_58_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_9
    linear_combination c3961.trans k2 - hc
  · have hc := e_57_17
    simp only [e_57_16, e_58_10] at hc
    linear_combination c3962.trans k2 - hc

theorem publicBatchWrapper86_f502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 11808)) (a (.virt 200)) (a (.virt 17685)) (a (.virt 17686)) := by
  have c3963 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3964 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3965 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3966 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3967 := (publicBatchWrapper86_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3968 := (publicBatchWrapper86_copies124 a h).1
  have c3969 := (publicBatchWrapper86_copies124 a h).2.1
  have c3970 := (publicBatchWrapper86_copies124 a h).2.2.1
  have c3971 := (publicBatchWrapper86_copies124 a h).2.2.2.1
  have c3972 := (publicBatchWrapper86_copies124 a h).2.2.2.2.1
  have c3973 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.1
  have c3974 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.1
  have c3975 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.1
  have c3976 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_57_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_18
  simp only [← c3963, k3, ← c3964, k3, ← c3965] at e_57_18
  have e_57_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row57 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_19
  simp only [← c3972, ← c3973, k3, ← c3974] at e_57_19
  have e_58_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_11
  simp only [← c3966, ← c3967, ← c3968] at e_58_11
  have e_58_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_12
  simp only [← c3969, ← c3970, ← c3971] at e_58_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_11
    linear_combination c3975.trans k2 - hc
  · have hc := e_57_19
    simp only [e_57_18, e_58_12] at hc
    linear_combination c3976.trans k2 - hc

theorem publicBatchWrapper86_f503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 58 55) = band (a (.virt 17679)) (a (.virt 17681)) := by
  have c3977 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.1
  have c3978 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3979 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_13
  simp only [← c3977, ← c3978, ← c3979] at e_58_13
  have hr := e_58_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 58 59) = band (a (.virt 17683)) (a (.virt 17685)) := by
  have c3980 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3981 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3982 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_14
  simp only [← c3980, ← c3981, ← c3982] at e_58_14
  have hr := e_58_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 58 63) = band (a (.wire 58 55)) (a (.wire 58 59)) := by
  have c3983 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3984 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3985 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_15
  simp only [← c3983, ← c3984, ← c3985] at e_58_15
  have hr := e_58_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12005)) (a (.virt 200)) (a (.virt 17687)) (a (.virt 17688)) := by
  have c3986 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3987 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3988 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3989 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3990 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3991 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3992 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3993 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3994 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3995 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3996 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3997 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3998 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3999 := (publicBatchWrapper86_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_58_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_16
  simp only [← c3989, ← c3990, ← c3991] at e_58_16
  have e_58_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_17
  simp only [← c3992, ← c3993, ← c3994] at e_58_17
  have e_59_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_0
  simp only [← c3986, k3, ← c3987, k3, ← c3988] at e_59_0
  have e_59_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_1
  simp only [← c3995, ← c3996, k3, ← c3997] at e_59_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_16
    linear_combination c3998.trans k2 - hc
  · have hc := e_59_1
    simp only [e_59_0, e_58_17] at hc
    linear_combination c3999.trans k2 - hc

theorem publicBatchWrapper86_f507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12006)) (a (.virt 200)) (a (.virt 17689)) (a (.virt 17690)) := by
  have c4000 := (publicBatchWrapper86_copies125 a h).1
  have c4001 := (publicBatchWrapper86_copies125 a h).2.1
  have c4002 := (publicBatchWrapper86_copies125 a h).2.2.1
  have c4003 := (publicBatchWrapper86_copies125 a h).2.2.2.1
  have c4004 := (publicBatchWrapper86_copies125 a h).2.2.2.2.1
  have c4005 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.1
  have c4006 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.1
  have c4007 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.1
  have c4008 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.1
  have c4009 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.1
  have c4010 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4011 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4012 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4013 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_58_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_18
  simp only [← c4003, ← c4004, ← c4005] at e_58_18
  have e_58_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row58 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_19
  simp only [← c4006, ← c4007, ← c4008] at e_58_19
  have e_59_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_2
  simp only [← c4000, k3, ← c4001, k3, ← c4002] at e_59_2
  have e_59_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_3
  simp only [← c4009, ← c4010, k3, ← c4011] at e_59_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_18
    linear_combination c4012.trans k2 - hc
  · have hc := e_59_3
    simp only [e_59_2, e_58_19] at hc
    linear_combination c4013.trans k2 - hc

theorem publicBatchWrapper86_f508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12007)) (a (.virt 200)) (a (.virt 17691)) (a (.virt 17692)) := by
  have c4014 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4015 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4016 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4017 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4018 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4019 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4020 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4021 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4022 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4023 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4024 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4025 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4026 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4027 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_4
  simp only [← c4014, k3, ← c4015, k3, ← c4016] at e_59_4
  have e_59_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_5
  simp only [← c4023, ← c4024, k3, ← c4025] at e_59_5
  have e_60_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_0
  simp only [← c4017, ← c4018, ← c4019] at e_60_0
  have e_60_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_1
  simp only [← c4020, ← c4021, ← c4022] at e_60_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_0
    linear_combination c4026.trans k2 - hc
  · have hc := e_59_5
    simp only [e_59_4, e_60_1] at hc
    linear_combination c4027.trans k2 - hc

theorem publicBatchWrapper86_f509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12008)) (a (.virt 200)) (a (.virt 17693)) (a (.virt 17694)) := by
  have c4028 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4029 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4030 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4031 := (publicBatchWrapper86_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4032 := (publicBatchWrapper86_copies126 a h).1
  have c4033 := (publicBatchWrapper86_copies126 a h).2.1
  have c4034 := (publicBatchWrapper86_copies126 a h).2.2.1
  have c4035 := (publicBatchWrapper86_copies126 a h).2.2.2.1
  have c4036 := (publicBatchWrapper86_copies126 a h).2.2.2.2.1
  have c4037 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.1
  have c4038 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.1
  have c4039 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.1
  have c4040 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.1
  have c4041 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_6
  simp only [← c4028, k3, ← c4029, k3, ← c4030] at e_59_6
  have e_59_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_7
  simp only [← c4037, ← c4038, k3, ← c4039] at e_59_7
  have e_60_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_2
  simp only [← c4031, ← c4032, ← c4033] at e_60_2
  have e_60_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_3
  simp only [← c4034, ← c4035, ← c4036] at e_60_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_2
    linear_combination c4040.trans k2 - hc
  · have hc := e_59_7
    simp only [e_59_6, e_60_3] at hc
    linear_combination c4041.trans k2 - hc

theorem publicBatchWrapper86_f510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 60 19) = band (a (.virt 17687)) (a (.virt 17689)) := by
  have c4042 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4043 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4044 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_4
  simp only [← c4042, ← c4043, ← c4044] at e_60_4
  have hr := e_60_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 60 23) = band (a (.virt 17691)) (a (.virt 17693)) := by
  have c4045 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4046 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4047 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_5
  simp only [← c4045, ← c4046, ← c4047] at e_60_5
  have hr := e_60_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 60 27) = band (a (.wire 60 19)) (a (.wire 60 23)) := by
  have c4048 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4049 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4050 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_6
  simp only [← c4048, ← c4049, ← c4050] at e_60_6
  have hr := e_60_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12205)) (a (.virt 200)) (a (.virt 17695)) (a (.virt 17696)) := by
  have c4051 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4052 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4053 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4054 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4055 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4056 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4057 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4058 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4059 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4060 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4061 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4062 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4063 := (publicBatchWrapper86_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4064 := (publicBatchWrapper86_copies127 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_8
  simp only [← c4051, k3, ← c4052, k3, ← c4053] at e_59_8
  have e_59_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_9
  simp only [← c4060, ← c4061, k3, ← c4062] at e_59_9
  have e_60_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_7
  simp only [← c4054, ← c4055, ← c4056] at e_60_7
  have e_60_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_8
  simp only [← c4057, ← c4058, ← c4059] at e_60_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_7
    linear_combination c4063.trans k2 - hc
  · have hc := e_59_9
    simp only [e_59_8, e_60_8] at hc
    linear_combination c4064.trans k2 - hc

theorem publicBatchWrapper86_f514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12206)) (a (.virt 200)) (a (.virt 17697)) (a (.virt 17698)) := by
  have c4065 := (publicBatchWrapper86_copies127 a h).2.1
  have c4066 := (publicBatchWrapper86_copies127 a h).2.2.1
  have c4067 := (publicBatchWrapper86_copies127 a h).2.2.2.1
  have c4068 := (publicBatchWrapper86_copies127 a h).2.2.2.2.1
  have c4069 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.1
  have c4070 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.1
  have c4071 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.1
  have c4072 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.1
  have c4073 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.1
  have c4074 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4075 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4076 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4077 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4078 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_10
  simp only [← c4065, k3, ← c4066, k3, ← c4067] at e_59_10
  have e_59_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_11
  simp only [← c4074, ← c4075, k3, ← c4076] at e_59_11
  have e_60_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_9
  simp only [← c4068, ← c4069, ← c4070] at e_60_9
  have e_60_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_10
  simp only [← c4071, ← c4072, ← c4073] at e_60_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_9
    linear_combination c4077.trans k2 - hc
  · have hc := e_59_11
    simp only [e_59_10, e_60_10] at hc
    linear_combination c4078.trans k2 - hc

theorem publicBatchWrapper86_f515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12207)) (a (.virt 200)) (a (.virt 17699)) (a (.virt 17700)) := by
  have c4079 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4080 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4081 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4082 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4083 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4084 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4085 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4086 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4087 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4088 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4089 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4090 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4091 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4092 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_12
  simp only [← c4079, k3, ← c4080, k3, ← c4081] at e_59_12
  have e_59_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_13
  simp only [← c4088, ← c4089, k3, ← c4090] at e_59_13
  have e_60_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_11
  simp only [← c4082, ← c4083, ← c4084] at e_60_11
  have e_60_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_12
  simp only [← c4085, ← c4086, ← c4087] at e_60_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_11
    linear_combination c4091.trans k2 - hc
  · have hc := e_59_13
    simp only [e_59_12, e_60_12] at hc
    linear_combination c4092.trans k2 - hc

theorem publicBatchWrapper86_f516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12208)) (a (.virt 200)) (a (.virt 17701)) (a (.virt 17702)) := by
  have c4093 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4094 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4095 := (publicBatchWrapper86_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4096 := (publicBatchWrapper86_copies128 a h).1
  have c4097 := (publicBatchWrapper86_copies128 a h).2.1
  have c4098 := (publicBatchWrapper86_copies128 a h).2.2.1
  have c4099 := (publicBatchWrapper86_copies128 a h).2.2.2.1
  have c4100 := (publicBatchWrapper86_copies128 a h).2.2.2.2.1
  have c4101 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.1
  have c4102 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.1
  have c4103 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.1
  have c4104 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.1
  have c4105 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.1
  have c4106 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_14
  simp only [← c4093, k3, ← c4094, k3, ← c4095] at e_59_14
  have e_59_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_15
  simp only [← c4102, ← c4103, k3, ← c4104] at e_59_15
  have e_60_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_13
  simp only [← c4096, ← c4097, ← c4098] at e_60_13
  have e_60_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_14
  simp only [← c4099, ← c4100, ← c4101] at e_60_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_13
    linear_combination c4105.trans k2 - hc
  · have hc := e_59_15
    simp only [e_59_14, e_60_14] at hc
    linear_combination c4106.trans k2 - hc

theorem publicBatchWrapper86_f517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 60 63) = band (a (.virt 17695)) (a (.virt 17697)) := by
  have c4107 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4108 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4109 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_15
  simp only [← c4107, ← c4108, ← c4109] at e_60_15
  have hr := e_60_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 60 67) = band (a (.virt 17699)) (a (.virt 17701)) := by
  have c4110 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4111 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4112 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_16
  simp only [← c4110, ← c4111, ← c4112] at e_60_16
  have hr := e_60_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 60 71) = band (a (.wire 60 63)) (a (.wire 60 67)) := by
  have c4113 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4114 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4115 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_17
  simp only [← c4113, ← c4114, ← c4115] at e_60_17
  have hr := e_60_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12405)) (a (.virt 200)) (a (.virt 17703)) (a (.virt 17704)) := by
  have c4116 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4117 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4118 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4119 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4120 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4121 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4122 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4123 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4124 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4125 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4126 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4127 := (publicBatchWrapper86_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4128 := (publicBatchWrapper86_copies129 a h).1
  have c4129 := (publicBatchWrapper86_copies129 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_16
  simp only [← c4116, k3, ← c4117, k3, ← c4118] at e_59_16
  have e_59_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_17
  simp only [← c4125, ← c4126, k3, ← c4127] at e_59_17
  have e_60_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_18
  simp only [← c4119, ← c4120, ← c4121] at e_60_18
  have e_60_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row60 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_19
  simp only [← c4122, ← c4123, ← c4124] at e_60_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_18
    linear_combination c4128.trans k2 - hc
  · have hc := e_59_17
    simp only [e_59_16, e_60_19] at hc
    linear_combination c4129.trans k2 - hc

theorem publicBatchWrapper86_f521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12406)) (a (.virt 200)) (a (.virt 17705)) (a (.virt 17706)) := by
  have c4130 := (publicBatchWrapper86_copies129 a h).2.2.1
  have c4131 := (publicBatchWrapper86_copies129 a h).2.2.2.1
  have c4132 := (publicBatchWrapper86_copies129 a h).2.2.2.2.1
  have c4133 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.1
  have c4134 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.1
  have c4135 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.1
  have c4136 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.1
  have c4137 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.1
  have c4138 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4139 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4140 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4141 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4142 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4143 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_59_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_18
  simp only [← c4130, k3, ← c4131, k3, ← c4132] at e_59_18
  have e_59_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row59 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_19
  simp only [← c4139, ← c4140, k3, ← c4141] at e_59_19
  have e_61_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_0
  simp only [← c4133, ← c4134, ← c4135] at e_61_0
  have e_61_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_1
  simp only [← c4136, ← c4137, ← c4138] at e_61_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_0
    linear_combination c4142.trans k2 - hc
  · have hc := e_59_19
    simp only [e_59_18, e_61_1] at hc
    linear_combination c4143.trans k2 - hc

theorem publicBatchWrapper86_f522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12407)) (a (.virt 200)) (a (.virt 17707)) (a (.virt 17708)) := by
  have c4144 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4145 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4146 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4147 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4148 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4149 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4150 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4151 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4152 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4153 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4154 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4155 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4156 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4157 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_61_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_2
  simp only [← c4147, ← c4148, ← c4149] at e_61_2
  have e_61_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_3
  simp only [← c4150, ← c4151, ← c4152] at e_61_3
  have e_62_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_0
  simp only [← c4144, k3, ← c4145, k3, ← c4146] at e_62_0
  have e_62_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_1
  simp only [← c4153, ← c4154, k3, ← c4155] at e_62_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_2
    linear_combination c4156.trans k2 - hc
  · have hc := e_62_1
    simp only [e_62_0, e_61_3] at hc
    linear_combination c4157.trans k2 - hc

theorem publicBatchWrapper86_f523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12408)) (a (.virt 200)) (a (.virt 17709)) (a (.virt 17710)) := by
  have c4158 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4159 := (publicBatchWrapper86_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4160 := (publicBatchWrapper86_copies130 a h).1
  have c4161 := (publicBatchWrapper86_copies130 a h).2.1
  have c4162 := (publicBatchWrapper86_copies130 a h).2.2.1
  have c4163 := (publicBatchWrapper86_copies130 a h).2.2.2.1
  have c4164 := (publicBatchWrapper86_copies130 a h).2.2.2.2.1
  have c4165 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.1
  have c4166 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.1
  have c4167 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.1
  have c4168 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.1
  have c4169 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.1
  have c4170 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4171 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_61_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_4
  simp only [← c4161, ← c4162, ← c4163] at e_61_4
  have e_61_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_5
  simp only [← c4164, ← c4165, ← c4166] at e_61_5
  have e_62_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_2
  simp only [← c4158, k3, ← c4159, k3, ← c4160] at e_62_2
  have e_62_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_3
  simp only [← c4167, ← c4168, k3, ← c4169] at e_62_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_4
    linear_combination c4170.trans k2 - hc
  · have hc := e_62_3
    simp only [e_62_2, e_61_5] at hc
    linear_combination c4171.trans k2 - hc

theorem publicBatchWrapper86_f524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 61 27) = band (a (.virt 17703)) (a (.virt 17705)) := by
  have c4172 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4173 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4174 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_61_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_6
  simp only [← c4172, ← c4173, ← c4174] at e_61_6
  have hr := e_61_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 61 31) = band (a (.virt 17707)) (a (.virt 17709)) := by
  have c4175 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4176 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4177 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_61_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_7
  simp only [← c4175, ← c4176, ← c4177] at e_61_7
  have hr := e_61_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 61 35) = band (a (.wire 61 27)) (a (.wire 61 31)) := by
  have c4178 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4179 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4180 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_61_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_8
  simp only [← c4178, ← c4179, ← c4180] at e_61_8
  have hr := e_61_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12605)) (a (.virt 200)) (a (.virt 17711)) (a (.virt 17712)) := by
  have c4181 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4182 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4183 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4184 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4185 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4186 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4187 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4188 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4189 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4190 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4191 := (publicBatchWrapper86_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4192 := (publicBatchWrapper86_copies131 a h).1
  have c4193 := (publicBatchWrapper86_copies131 a h).2.1
  have c4194 := (publicBatchWrapper86_copies131 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_61_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_9
  simp only [← c4184, ← c4185, ← c4186] at e_61_9
  have e_61_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_10
  simp only [← c4187, ← c4188, ← c4189] at e_61_10
  have e_62_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_4
  simp only [← c4181, k3, ← c4182, k3, ← c4183] at e_62_4
  have e_62_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_5
  simp only [← c4190, ← c4191, k3, ← c4192] at e_62_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_9
    linear_combination c4193.trans k2 - hc
  · have hc := e_62_5
    simp only [e_62_4, e_61_10] at hc
    linear_combination c4194.trans k2 - hc

theorem publicBatchWrapper86_f528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12606)) (a (.virt 200)) (a (.virt 17713)) (a (.virt 17714)) := by
  have c4195 := (publicBatchWrapper86_copies131 a h).2.2.2.1
  have c4196 := (publicBatchWrapper86_copies131 a h).2.2.2.2.1
  have c4197 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.1
  have c4198 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.1
  have c4199 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.1
  have c4200 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.1
  have c4201 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.1
  have c4202 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4203 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4204 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4205 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4206 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4207 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4208 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_61_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_11
  simp only [← c4198, ← c4199, ← c4200] at e_61_11
  have e_61_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_12
  simp only [← c4201, ← c4202, ← c4203] at e_61_12
  have e_62_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_6
  simp only [← c4195, k3, ← c4196, k3, ← c4197] at e_62_6
  have e_62_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_7
  simp only [← c4204, ← c4205, k3, ← c4206] at e_62_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_11
    linear_combination c4207.trans k2 - hc
  · have hc := e_62_7
    simp only [e_62_6, e_61_12] at hc
    linear_combination c4208.trans k2 - hc

theorem publicBatchWrapper86_f529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12607)) (a (.virt 200)) (a (.virt 17715)) (a (.virt 17716)) := by
  have c4209 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4210 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4211 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4212 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4213 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4214 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4215 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4216 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4217 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4218 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4219 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4220 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4221 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4222 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_61_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_13
  simp only [← c4212, ← c4213, ← c4214] at e_61_13
  have e_61_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_14
  simp only [← c4215, ← c4216, ← c4217] at e_61_14
  have e_62_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_8
  simp only [← c4209, k3, ← c4210, k3, ← c4211] at e_62_8
  have e_62_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_9
  simp only [← c4218, ← c4219, k3, ← c4220] at e_62_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_13
    linear_combination c4221.trans k2 - hc
  · have hc := e_62_9
    simp only [e_62_8, e_61_14] at hc
    linear_combination c4222.trans k2 - hc

theorem publicBatchWrapper86_f530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12608)) (a (.virt 200)) (a (.virt 17717)) (a (.virt 17718)) := by
  have c4223 := (publicBatchWrapper86_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4224 := (publicBatchWrapper86_copies132 a h).1
  have c4225 := (publicBatchWrapper86_copies132 a h).2.1
  have c4226 := (publicBatchWrapper86_copies132 a h).2.2.1
  have c4227 := (publicBatchWrapper86_copies132 a h).2.2.2.1
  have c4228 := (publicBatchWrapper86_copies132 a h).2.2.2.2.1
  have c4229 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.1
  have c4230 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.1
  have c4231 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.1
  have c4232 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.1
  have c4233 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.1
  have c4234 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4235 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4236 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_61_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_15
  simp only [← c4226, ← c4227, ← c4228] at e_61_15
  have e_61_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_16
  simp only [← c4229, ← c4230, ← c4231] at e_61_16
  have e_62_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_10
  simp only [← c4223, k3, ← c4224, k3, ← c4225] at e_62_10
  have e_62_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_11
  simp only [← c4232, ← c4233, k3, ← c4234] at e_62_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_61_15
    linear_combination c4235.trans k2 - hc
  · have hc := e_62_11
    simp only [e_62_10, e_61_16] at hc
    linear_combination c4236.trans k2 - hc

theorem publicBatchWrapper86_f531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 61 71) = band (a (.virt 17711)) (a (.virt 17713)) := by
  have c4237 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4238 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4239 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_61_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_17
  simp only [← c4237, ← c4238, ← c4239] at e_61_17
  have hr := e_61_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 61 75) = band (a (.virt 17715)) (a (.virt 17717)) := by
  have c4240 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4241 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4242 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_61_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_18
  simp only [← c4240, ← c4241, ← c4242] at e_61_18
  have hr := e_61_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 61 79) = band (a (.wire 61 71)) (a (.wire 61 75)) := by
  have c4243 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4244 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4245 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_61_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row61 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_19
  simp only [← c4243, ← c4244, ← c4245] at e_61_19
  have hr := e_61_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12805)) (a (.virt 200)) (a (.virt 17719)) (a (.virt 17720)) := by
  have c4246 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4247 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4248 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4249 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4250 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4251 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4252 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4253 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4254 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4255 := (publicBatchWrapper86_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4256 := (publicBatchWrapper86_copies133 a h).1
  have c4257 := (publicBatchWrapper86_copies133 a h).2.1
  have c4258 := (publicBatchWrapper86_copies133 a h).2.2.1
  have c4259 := (publicBatchWrapper86_copies133 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_62_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_12
  simp only [← c4246, k3, ← c4247, k3, ← c4248] at e_62_12
  have e_62_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_13
  simp only [← c4255, ← c4256, k3, ← c4257] at e_62_13
  have e_63_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_0
  simp only [← c4249, ← c4250, ← c4251] at e_63_0
  have e_63_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_1
  simp only [← c4252, ← c4253, ← c4254] at e_63_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_0
    linear_combination c4258.trans k2 - hc
  · have hc := e_62_13
    simp only [e_62_12, e_63_1] at hc
    linear_combination c4259.trans k2 - hc

theorem publicBatchWrapper86_f535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12806)) (a (.virt 200)) (a (.virt 17721)) (a (.virt 17722)) := by
  have c4260 := (publicBatchWrapper86_copies133 a h).2.2.2.2.1
  have c4261 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.1
  have c4262 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.1
  have c4263 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.1
  have c4264 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.1
  have c4265 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.1
  have c4266 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4267 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4268 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4269 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4270 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4271 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4272 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4273 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_62_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_14
  simp only [← c4260, k3, ← c4261, k3, ← c4262] at e_62_14
  have e_62_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_15
  simp only [← c4269, ← c4270, k3, ← c4271] at e_62_15
  have e_63_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_2
  simp only [← c4263, ← c4264, ← c4265] at e_63_2
  have e_63_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_3
  simp only [← c4266, ← c4267, ← c4268] at e_63_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_2
    linear_combination c4272.trans k2 - hc
  · have hc := e_62_15
    simp only [e_62_14, e_63_3] at hc
    linear_combination c4273.trans k2 - hc

theorem publicBatchWrapper86_f536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12807)) (a (.virt 200)) (a (.virt 17723)) (a (.virt 17724)) := by
  have c4274 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4275 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4276 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4277 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4278 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4279 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4280 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4281 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4282 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4283 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4284 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4285 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4286 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4287 := (publicBatchWrapper86_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_62_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_16
  simp only [← c4274, k3, ← c4275, k3, ← c4276] at e_62_16
  have e_62_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_17
  simp only [← c4283, ← c4284, k3, ← c4285] at e_62_17
  have e_63_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_4
  simp only [← c4277, ← c4278, ← c4279] at e_63_4
  have e_63_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_5
  simp only [← c4280, ← c4281, ← c4282] at e_63_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_4
    linear_combination c4286.trans k2 - hc
  · have hc := e_62_17
    simp only [e_62_16, e_63_5] at hc
    linear_combination c4287.trans k2 - hc

theorem publicBatchWrapper86_f537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 12808)) (a (.virt 200)) (a (.virt 17725)) (a (.virt 17726)) := by
  have c4288 := (publicBatchWrapper86_copies134 a h).1
  have c4289 := (publicBatchWrapper86_copies134 a h).2.1
  have c4290 := (publicBatchWrapper86_copies134 a h).2.2.1
  have c4291 := (publicBatchWrapper86_copies134 a h).2.2.2.1
  have c4292 := (publicBatchWrapper86_copies134 a h).2.2.2.2.1
  have c4293 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.1
  have c4294 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.1
  have c4295 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.1
  have c4296 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.1
  have c4297 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.1
  have c4298 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4299 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4300 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4301 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_62_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_18
  simp only [← c4288, k3, ← c4289, k3, ← c4290] at e_62_18
  have e_62_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row62 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_19
  simp only [← c4297, ← c4298, k3, ← c4299] at e_62_19
  have e_63_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_6
  simp only [← c4291, ← c4292, ← c4293] at e_63_6
  have e_63_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_7
  simp only [← c4294, ← c4295, ← c4296] at e_63_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_6
    linear_combination c4300.trans k2 - hc
  · have hc := e_62_19
    simp only [e_62_18, e_63_7] at hc
    linear_combination c4301.trans k2 - hc

theorem publicBatchWrapper86_f538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 63 35) = band (a (.virt 17719)) (a (.virt 17721)) := by
  have c4302 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4303 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4304 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_8
  simp only [← c4302, ← c4303, ← c4304] at e_63_8
  have hr := e_63_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 63 39) = band (a (.virt 17723)) (a (.virt 17725)) := by
  have c4305 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4306 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4307 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_9
  simp only [← c4305, ← c4306, ← c4307] at e_63_9
  have hr := e_63_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 63 43) = band (a (.wire 63 35)) (a (.wire 63 39)) := by
  have c4308 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4309 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4310 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_10
  simp only [← c4308, ← c4309, ← c4310] at e_63_10
  have hr := e_63_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13005)) (a (.virt 200)) (a (.virt 17727)) (a (.virt 17728)) := by
  have c4311 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4312 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4313 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4314 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4315 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4316 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4317 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4318 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4319 := (publicBatchWrapper86_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4320 := (publicBatchWrapper86_copies135 a h).1
  have c4321 := (publicBatchWrapper86_copies135 a h).2.1
  have c4322 := (publicBatchWrapper86_copies135 a h).2.2.1
  have c4323 := (publicBatchWrapper86_copies135 a h).2.2.2.1
  have c4324 := (publicBatchWrapper86_copies135 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_63_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_11
  simp only [← c4314, ← c4315, ← c4316] at e_63_11
  have e_63_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_12
  simp only [← c4317, ← c4318, ← c4319] at e_63_12
  have e_64_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_0
  simp only [← c4311, k3, ← c4312, k3, ← c4313] at e_64_0
  have e_64_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_1
  simp only [← c4320, ← c4321, k3, ← c4322] at e_64_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_11
    linear_combination c4323.trans k2 - hc
  · have hc := e_64_1
    simp only [e_64_0, e_63_12] at hc
    linear_combination c4324.trans k2 - hc

theorem publicBatchWrapper86_f542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13006)) (a (.virt 200)) (a (.virt 17729)) (a (.virt 17730)) := by
  have c4325 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.1
  have c4326 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.1
  have c4327 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.1
  have c4328 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.1
  have c4329 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.1
  have c4330 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4331 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4332 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4333 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4334 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4335 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4336 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4337 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4338 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_63_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_13
  simp only [← c4328, ← c4329, ← c4330] at e_63_13
  have e_63_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_14
  simp only [← c4331, ← c4332, ← c4333] at e_63_14
  have e_64_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_2
  simp only [← c4325, k3, ← c4326, k3, ← c4327] at e_64_2
  have e_64_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_3
  simp only [← c4334, ← c4335, k3, ← c4336] at e_64_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_13
    linear_combination c4337.trans k2 - hc
  · have hc := e_64_3
    simp only [e_64_2, e_63_14] at hc
    linear_combination c4338.trans k2 - hc

theorem publicBatchWrapper86_f543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13007)) (a (.virt 200)) (a (.virt 17731)) (a (.virt 17732)) := by
  have c4339 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4340 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4341 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4342 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4343 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4344 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4345 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4346 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4347 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4348 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4349 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4350 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4351 := (publicBatchWrapper86_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4352 := (publicBatchWrapper86_copies136 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_63_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_15
  simp only [← c4342, ← c4343, ← c4344] at e_63_15
  have e_63_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_16
  simp only [← c4345, ← c4346, ← c4347] at e_63_16
  have e_64_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_4
  simp only [← c4339, k3, ← c4340, k3, ← c4341] at e_64_4
  have e_64_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_5
  simp only [← c4348, ← c4349, k3, ← c4350] at e_64_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_15
    linear_combination c4351.trans k2 - hc
  · have hc := e_64_5
    simp only [e_64_4, e_63_16] at hc
    linear_combination c4352.trans k2 - hc

theorem publicBatchWrapper86_f544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13008)) (a (.virt 200)) (a (.virt 17733)) (a (.virt 17734)) := by
  have c4353 := (publicBatchWrapper86_copies136 a h).2.1
  have c4354 := (publicBatchWrapper86_copies136 a h).2.2.1
  have c4355 := (publicBatchWrapper86_copies136 a h).2.2.2.1
  have c4356 := (publicBatchWrapper86_copies136 a h).2.2.2.2.1
  have c4357 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.1
  have c4358 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.1
  have c4359 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.1
  have c4360 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.1
  have c4361 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.1
  have c4362 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4363 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4364 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4365 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4366 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_63_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_17
  simp only [← c4356, ← c4357, ← c4358] at e_63_17
  have e_63_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_18
  simp only [← c4359, ← c4360, ← c4361] at e_63_18
  have e_64_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_6
  simp only [← c4353, k3, ← c4354, k3, ← c4355] at e_64_6
  have e_64_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_7
  simp only [← c4362, ← c4363, k3, ← c4364] at e_64_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_63_17
    linear_combination c4365.trans k2 - hc
  · have hc := e_64_7
    simp only [e_64_6, e_63_18] at hc
    linear_combination c4366.trans k2 - hc

theorem publicBatchWrapper86_f545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 63 79) = band (a (.virt 17727)) (a (.virt 17729)) := by
  have c4367 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4368 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4369 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row63 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_19
  simp only [← c4367, ← c4368, ← c4369] at e_63_19
  have hr := e_63_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 65 3) = band (a (.virt 17731)) (a (.virt 17733)) := by
  have c4370 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4371 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4372 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_0
  simp only [← c4370, ← c4371, ← c4372] at e_65_0
  have hr := e_65_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 65 7) = band (a (.wire 63 79)) (a (.wire 65 3)) := by
  have c4373 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4374 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4375 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_1
  simp only [← c4373, ← c4374, ← c4375] at e_65_1
  have hr := e_65_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13205)) (a (.virt 200)) (a (.virt 17735)) (a (.virt 17736)) := by
  have c4376 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4377 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4378 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4379 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4380 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4381 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4382 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4383 := (publicBatchWrapper86_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4384 := (publicBatchWrapper86_copies137 a h).1
  have c4385 := (publicBatchWrapper86_copies137 a h).2.1
  have c4386 := (publicBatchWrapper86_copies137 a h).2.2.1
  have c4387 := (publicBatchWrapper86_copies137 a h).2.2.2.1
  have c4388 := (publicBatchWrapper86_copies137 a h).2.2.2.2.1
  have c4389 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_64_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_8
  simp only [← c4376, k3, ← c4377, k3, ← c4378] at e_64_8
  have e_64_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_9
  simp only [← c4385, ← c4386, k3, ← c4387] at e_64_9
  have e_65_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_2
  simp only [← c4379, ← c4380, ← c4381] at e_65_2
  have e_65_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_3
  simp only [← c4382, ← c4383, ← c4384] at e_65_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_2
    linear_combination c4388.trans k2 - hc
  · have hc := e_64_9
    simp only [e_64_8, e_65_3] at hc
    linear_combination c4389.trans k2 - hc

theorem publicBatchWrapper86_f549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13206)) (a (.virt 200)) (a (.virt 17737)) (a (.virt 17738)) := by
  have c4390 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.1
  have c4391 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.1
  have c4392 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.1
  have c4393 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.1
  have c4394 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4395 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4396 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4397 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4398 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4399 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4400 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4401 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4402 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4403 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_64_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_10
  simp only [← c4390, k3, ← c4391, k3, ← c4392] at e_64_10
  have e_64_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_11
  simp only [← c4399, ← c4400, k3, ← c4401] at e_64_11
  have e_65_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_4
  simp only [← c4393, ← c4394, ← c4395] at e_65_4
  have e_65_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_5
  simp only [← c4396, ← c4397, ← c4398] at e_65_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_4
    linear_combination c4402.trans k2 - hc
  · have hc := e_64_11
    simp only [e_64_10, e_65_5] at hc
    linear_combination c4403.trans k2 - hc

theorem publicBatchWrapper86_f550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13207)) (a (.virt 200)) (a (.virt 17739)) (a (.virt 17740)) := by
  have c4404 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4405 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4406 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4407 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4408 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4409 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4410 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4411 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4412 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4413 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4414 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4415 := (publicBatchWrapper86_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4416 := (publicBatchWrapper86_copies138 a h).1
  have c4417 := (publicBatchWrapper86_copies138 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_64_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_12
  simp only [← c4404, k3, ← c4405, k3, ← c4406] at e_64_12
  have e_64_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_13
  simp only [← c4413, ← c4414, k3, ← c4415] at e_64_13
  have e_65_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_6
  simp only [← c4407, ← c4408, ← c4409] at e_65_6
  have e_65_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_7
  simp only [← c4410, ← c4411, ← c4412] at e_65_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_6
    linear_combination c4416.trans k2 - hc
  · have hc := e_64_13
    simp only [e_64_12, e_65_7] at hc
    linear_combination c4417.trans k2 - hc

theorem publicBatchWrapper86_f551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13208)) (a (.virt 200)) (a (.virt 17741)) (a (.virt 17742)) := by
  have c4418 := (publicBatchWrapper86_copies138 a h).2.2.1
  have c4419 := (publicBatchWrapper86_copies138 a h).2.2.2.1
  have c4420 := (publicBatchWrapper86_copies138 a h).2.2.2.2.1
  have c4421 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.1
  have c4422 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.1
  have c4423 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.1
  have c4424 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.1
  have c4425 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.1
  have c4426 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4427 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4428 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4429 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4430 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_64_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_14
  simp only [← c4418, k3, ← c4419, k3, ← c4420] at e_64_14
  have e_64_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_15
  simp only [← c4427, ← c4428, k3, ← c4429] at e_64_15
  have e_65_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_8
  simp only [← c4421, ← c4422, ← c4423] at e_65_8
  have e_65_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_9
  simp only [← c4424, ← c4425, ← c4426] at e_65_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_8
    linear_combination c4430.trans k2 - hc
  · have hc := e_64_15
    simp only [e_64_14, e_65_9] at hc
    linear_combination c4431.trans k2 - hc

theorem publicBatchWrapper86_f552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 65 43) = band (a (.virt 17735)) (a (.virt 17737)) := by
  have c4432 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_10
  simp only [← c4432, ← c4433, ← c4434] at e_65_10
  have hr := e_65_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 65 47) = band (a (.virt 17739)) (a (.virt 17741)) := by
  have c4435 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4436 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4437 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_11
  simp only [← c4435, ← c4436, ← c4437] at e_65_11
  have hr := e_65_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 65 51) = band (a (.wire 65 43)) (a (.wire 65 47)) := by
  have c4438 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4439 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4440 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_12
  simp only [← c4438, ← c4439, ← c4440] at e_65_12
  have hr := e_65_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13405)) (a (.virt 200)) (a (.virt 17743)) (a (.virt 17744)) := by
  have c4441 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4442 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4443 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4444 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4445 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4446 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4447 := (publicBatchWrapper86_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4448 := (publicBatchWrapper86_copies139 a h).1
  have c4449 := (publicBatchWrapper86_copies139 a h).2.1
  have c4450 := (publicBatchWrapper86_copies139 a h).2.2.1
  have c4451 := (publicBatchWrapper86_copies139 a h).2.2.2.1
  have c4452 := (publicBatchWrapper86_copies139 a h).2.2.2.2.1
  have c4453 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.1
  have c4454 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_64_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_16
  simp only [← c4441, k3, ← c4442, k3, ← c4443] at e_64_16
  have e_64_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_17
  simp only [← c4450, ← c4451, k3, ← c4452] at e_64_17
  have e_65_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_13
  simp only [← c4444, ← c4445, ← c4446] at e_65_13
  have e_65_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_14
  simp only [← c4447, ← c4448, ← c4449] at e_65_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_13
    linear_combination c4453.trans k2 - hc
  · have hc := e_64_17
    simp only [e_64_16, e_65_14] at hc
    linear_combination c4454.trans k2 - hc

theorem publicBatchWrapper86_f556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13406)) (a (.virt 200)) (a (.virt 17745)) (a (.virt 17746)) := by
  have c4455 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.1
  have c4456 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.1
  have c4457 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.1
  have c4458 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4459 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4460 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4461 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4462 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4463 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4464 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4465 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4466 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4467 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4468 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_64_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_18
  simp only [← c4455, k3, ← c4456, k3, ← c4457] at e_64_18
  have e_64_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row64 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_19
  simp only [← c4464, ← c4465, k3, ← c4466] at e_64_19
  have e_65_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_15
  simp only [← c4458, ← c4459, ← c4460] at e_65_15
  have e_65_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_16
  simp only [← c4461, ← c4462, ← c4463] at e_65_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_15
    linear_combination c4467.trans k2 - hc
  · have hc := e_64_19
    simp only [e_64_18, e_65_16] at hc
    linear_combination c4468.trans k2 - hc

theorem publicBatchWrapper86_f557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13407)) (a (.virt 200)) (a (.virt 17747)) (a (.virt 17748)) := by
  have c4469 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4470 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4471 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4472 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4473 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4474 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4475 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4476 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4477 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4478 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4479 := (publicBatchWrapper86_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4480 := (publicBatchWrapper86_copies140 a h).1
  have c4481 := (publicBatchWrapper86_copies140 a h).2.1
  have c4482 := (publicBatchWrapper86_copies140 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_65_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_17
  simp only [← c4472, ← c4473, ← c4474] at e_65_17
  have e_65_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_18
  simp only [← c4475, ← c4476, ← c4477] at e_65_18
  have e_66_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_0
  simp only [← c4469, k3, ← c4470, k3, ← c4471] at e_66_0
  have e_66_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_1
  simp only [← c4478, ← c4479, k3, ← c4480] at e_66_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_17
    linear_combination c4481.trans k2 - hc
  · have hc := e_66_1
    simp only [e_66_0, e_65_18] at hc
    linear_combination c4482.trans k2 - hc

theorem publicBatchWrapper86_f558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13408)) (a (.virt 200)) (a (.virt 17749)) (a (.virt 17750)) := by
  have c4483 := (publicBatchWrapper86_copies140 a h).2.2.2.1
  have c4484 := (publicBatchWrapper86_copies140 a h).2.2.2.2.1
  have c4485 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.1
  have c4486 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.1
  have c4487 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.1
  have c4488 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.1
  have c4489 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.1
  have c4490 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4491 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4492 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4493 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4494 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4495 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4496 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_65_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row65 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_19
  simp only [← c4486, ← c4487, ← c4488] at e_65_19
  have e_66_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_2
  simp only [← c4483, k3, ← c4484, k3, ← c4485] at e_66_2
  have e_66_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_3
  simp only [← c4492, ← c4493, k3, ← c4494] at e_66_3
  have e_67_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_0
  simp only [← c4489, ← c4490, ← c4491] at e_67_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_19
    linear_combination c4495.trans k2 - hc
  · have hc := e_66_3
    simp only [e_66_2, e_67_0] at hc
    linear_combination c4496.trans k2 - hc

theorem publicBatchWrapper86_f559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 67 7) = band (a (.virt 17743)) (a (.virt 17745)) := by
  have c4497 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4498 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4499 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_1
  simp only [← c4497, ← c4498, ← c4499] at e_67_1
  have hr := e_67_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 67 11) = band (a (.virt 17747)) (a (.virt 17749)) := by
  have c4500 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4501 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4502 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_2
  simp only [← c4500, ← c4501, ← c4502] at e_67_2
  have hr := e_67_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 67 15) = band (a (.wire 67 7)) (a (.wire 67 11)) := by
  have c4503 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4504 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4505 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_3
  simp only [← c4503, ← c4504, ← c4505] at e_67_3
  have hr := e_67_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13605)) (a (.virt 200)) (a (.virt 17751)) (a (.virt 17752)) := by
  have c4506 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4507 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4508 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4509 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4510 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4511 := (publicBatchWrapper86_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4512 := (publicBatchWrapper86_copies141 a h).1
  have c4513 := (publicBatchWrapper86_copies141 a h).2.1
  have c4514 := (publicBatchWrapper86_copies141 a h).2.2.1
  have c4515 := (publicBatchWrapper86_copies141 a h).2.2.2.1
  have c4516 := (publicBatchWrapper86_copies141 a h).2.2.2.2.1
  have c4517 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.1
  have c4518 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.1
  have c4519 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_4
  simp only [← c4506, k3, ← c4507, k3, ← c4508] at e_66_4
  have e_66_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_5
  simp only [← c4515, ← c4516, k3, ← c4517] at e_66_5
  have e_67_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_4
  simp only [← c4509, ← c4510, ← c4511] at e_67_4
  have e_67_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_5
  simp only [← c4512, ← c4513, ← c4514] at e_67_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_4
    linear_combination c4518.trans k2 - hc
  · have hc := e_66_5
    simp only [e_66_4, e_67_5] at hc
    linear_combination c4519.trans k2 - hc

theorem publicBatchWrapper86_f563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13606)) (a (.virt 200)) (a (.virt 17753)) (a (.virt 17754)) := by
  have c4520 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.1
  have c4521 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.1
  have c4522 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4523 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4524 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4525 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4526 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4527 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4528 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4529 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4530 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4531 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4532 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4533 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_6
  simp only [← c4520, k3, ← c4521, k3, ← c4522] at e_66_6
  have e_66_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_7
  simp only [← c4529, ← c4530, k3, ← c4531] at e_66_7
  have e_67_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_6
  simp only [← c4523, ← c4524, ← c4525] at e_67_6
  have e_67_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_7
  simp only [← c4526, ← c4527, ← c4528] at e_67_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_6
    linear_combination c4532.trans k2 - hc
  · have hc := e_66_7
    simp only [e_66_6, e_67_7] at hc
    linear_combination c4533.trans k2 - hc

theorem publicBatchWrapper86_f564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13607)) (a (.virt 200)) (a (.virt 17755)) (a (.virt 17756)) := by
  have c4534 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4535 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4536 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4537 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4538 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4539 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4540 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4541 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4542 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4543 := (publicBatchWrapper86_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4544 := (publicBatchWrapper86_copies142 a h).1
  have c4545 := (publicBatchWrapper86_copies142 a h).2.1
  have c4546 := (publicBatchWrapper86_copies142 a h).2.2.1
  have c4547 := (publicBatchWrapper86_copies142 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_8
  simp only [← c4534, k3, ← c4535, k3, ← c4536] at e_66_8
  have e_66_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_9
  simp only [← c4543, ← c4544, k3, ← c4545] at e_66_9
  have e_67_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_8
  simp only [← c4537, ← c4538, ← c4539] at e_67_8
  have e_67_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_9
  simp only [← c4540, ← c4541, ← c4542] at e_67_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_8
    linear_combination c4546.trans k2 - hc
  · have hc := e_66_9
    simp only [e_66_8, e_67_9] at hc
    linear_combination c4547.trans k2 - hc

theorem publicBatchWrapper86_f565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13608)) (a (.virt 200)) (a (.virt 17757)) (a (.virt 17758)) := by
  have c4548 := (publicBatchWrapper86_copies142 a h).2.2.2.2.1
  have c4549 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.1
  have c4550 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.1
  have c4551 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.1
  have c4552 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.1
  have c4553 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.1
  have c4554 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4555 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4556 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4557 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4558 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4559 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4560 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4561 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_10
  simp only [← c4548, k3, ← c4549, k3, ← c4550] at e_66_10
  have e_66_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_11
  simp only [← c4557, ← c4558, k3, ← c4559] at e_66_11
  have e_67_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_10
  simp only [← c4551, ← c4552, ← c4553] at e_67_10
  have e_67_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_11
  simp only [← c4554, ← c4555, ← c4556] at e_67_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_10
    linear_combination c4560.trans k2 - hc
  · have hc := e_66_11
    simp only [e_66_10, e_67_11] at hc
    linear_combination c4561.trans k2 - hc

theorem publicBatchWrapper86_f566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 67 51) = band (a (.virt 17751)) (a (.virt 17753)) := by
  have c4562 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4563 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4564 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_12
  simp only [← c4562, ← c4563, ← c4564] at e_67_12
  have hr := e_67_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 67 55) = band (a (.virt 17755)) (a (.virt 17757)) := by
  have c4565 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4566 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4567 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_13
  simp only [← c4565, ← c4566, ← c4567] at e_67_13
  have hr := e_67_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 67 59) = band (a (.wire 67 51)) (a (.wire 67 55)) := by
  have c4568 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4569 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4570 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_14
  simp only [← c4568, ← c4569, ← c4570] at e_67_14
  have hr := e_67_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13805)) (a (.virt 200)) (a (.virt 17759)) (a (.virt 17760)) := by
  have c4571 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4572 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4573 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4574 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4575 := (publicBatchWrapper86_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4576 := (publicBatchWrapper86_copies143 a h).1
  have c4577 := (publicBatchWrapper86_copies143 a h).2.1
  have c4578 := (publicBatchWrapper86_copies143 a h).2.2.1
  have c4579 := (publicBatchWrapper86_copies143 a h).2.2.2.1
  have c4580 := (publicBatchWrapper86_copies143 a h).2.2.2.2.1
  have c4581 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.1
  have c4582 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.1
  have c4583 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.1
  have c4584 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_12
  simp only [← c4571, k3, ← c4572, k3, ← c4573] at e_66_12
  have e_66_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_13
  simp only [← c4580, ← c4581, k3, ← c4582] at e_66_13
  have e_67_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_15
  simp only [← c4574, ← c4575, ← c4576] at e_67_15
  have e_67_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_16
  simp only [← c4577, ← c4578, ← c4579] at e_67_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_15
    linear_combination c4583.trans k2 - hc
  · have hc := e_66_13
    simp only [e_66_12, e_67_16] at hc
    linear_combination c4584.trans k2 - hc

theorem publicBatchWrapper86_f570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13806)) (a (.virt 200)) (a (.virt 17761)) (a (.virt 17762)) := by
  have c4585 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.1
  have c4586 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4587 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4588 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4589 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4590 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4591 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4592 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4593 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4594 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4595 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4596 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4597 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4598 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_14
  simp only [← c4585, k3, ← c4586, k3, ← c4587] at e_66_14
  have e_66_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_15
  simp only [← c4594, ← c4595, k3, ← c4596] at e_66_15
  have e_67_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_17
  simp only [← c4588, ← c4589, ← c4590] at e_67_17
  have e_67_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_18
  simp only [← c4591, ← c4592, ← c4593] at e_67_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_17
    linear_combination c4597.trans k2 - hc
  · have hc := e_66_15
    simp only [e_66_14, e_67_18] at hc
    linear_combination c4598.trans k2 - hc

theorem publicBatchWrapper86_f571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13807)) (a (.virt 200)) (a (.virt 17763)) (a (.virt 17764)) := by
  have c4599 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4600 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4601 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4602 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4603 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4604 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4605 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4606 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4607 := (publicBatchWrapper86_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4608 := (publicBatchWrapper86_copies144 a h).1
  have c4609 := (publicBatchWrapper86_copies144 a h).2.1
  have c4610 := (publicBatchWrapper86_copies144 a h).2.2.1
  have c4611 := (publicBatchWrapper86_copies144 a h).2.2.2.1
  have c4612 := (publicBatchWrapper86_copies144 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_16
  simp only [← c4599, k3, ← c4600, k3, ← c4601] at e_66_16
  have e_66_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_17
  simp only [← c4608, ← c4609, k3, ← c4610] at e_66_17
  have e_67_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row67 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_19
  simp only [← c4602, ← c4603, ← c4604] at e_67_19
  have e_68_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_0
  simp only [← c4605, ← c4606, ← c4607] at e_68_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_19
    linear_combination c4611.trans k2 - hc
  · have hc := e_66_17
    simp only [e_66_16, e_68_0] at hc
    linear_combination c4612.trans k2 - hc

theorem publicBatchWrapper86_f572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 13808)) (a (.virt 200)) (a (.virt 17765)) (a (.virt 17766)) := by
  have c4613 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.1
  have c4614 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.1
  have c4615 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.1
  have c4616 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.1
  have c4617 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.1
  have c4618 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4619 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4620 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4621 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4622 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4623 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4624 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4625 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4626 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_66_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_18
  simp only [← c4613, k3, ← c4614, k3, ← c4615] at e_66_18
  have e_66_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row66 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_19
  simp only [← c4622, ← c4623, k3, ← c4624] at e_66_19
  have e_68_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_1
  simp only [← c4616, ← c4617, ← c4618] at e_68_1
  have e_68_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_2
  simp only [← c4619, ← c4620, ← c4621] at e_68_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_1
    linear_combination c4625.trans k2 - hc
  · have hc := e_66_19
    simp only [e_66_18, e_68_2] at hc
    linear_combination c4626.trans k2 - hc

theorem publicBatchWrapper86_f573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 68 15) = band (a (.virt 17759)) (a (.virt 17761)) := by
  have c4627 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4628 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4629 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_3
  simp only [← c4627, ← c4628, ← c4629] at e_68_3
  have hr := e_68_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 68 19) = band (a (.virt 17763)) (a (.virt 17765)) := by
  have c4630 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4631 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4632 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_4
  simp only [← c4630, ← c4631, ← c4632] at e_68_4
  have hr := e_68_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 68 23) = band (a (.wire 68 15)) (a (.wire 68 19)) := by
  have c4633 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4634 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4635 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_5
  simp only [← c4633, ← c4634, ← c4635] at e_68_5
  have hr := e_68_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14005)) (a (.virt 200)) (a (.virt 17767)) (a (.virt 17768)) := by
  have c4636 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4637 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4638 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4639 := (publicBatchWrapper86_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4640 := (publicBatchWrapper86_copies145 a h).1
  have c4641 := (publicBatchWrapper86_copies145 a h).2.1
  have c4642 := (publicBatchWrapper86_copies145 a h).2.2.1
  have c4643 := (publicBatchWrapper86_copies145 a h).2.2.2.1
  have c4644 := (publicBatchWrapper86_copies145 a h).2.2.2.2.1
  have c4645 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.1
  have c4646 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.1
  have c4647 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.1
  have c4648 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.1
  have c4649 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_68_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_6
  simp only [← c4639, ← c4640, ← c4641] at e_68_6
  have e_68_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_7
  simp only [← c4642, ← c4643, ← c4644] at e_68_7
  have e_69_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_0
  simp only [← c4636, k3, ← c4637, k3, ← c4638] at e_69_0
  have e_69_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_1
  simp only [← c4645, ← c4646, k3, ← c4647] at e_69_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_6
    linear_combination c4648.trans k2 - hc
  · have hc := e_69_1
    simp only [e_69_0, e_68_7] at hc
    linear_combination c4649.trans k2 - hc

theorem publicBatchWrapper86_f577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14006)) (a (.virt 200)) (a (.virt 17769)) (a (.virt 17770)) := by
  have c4650 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4651 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4652 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4653 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4654 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4655 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4656 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4657 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4658 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4659 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4660 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4661 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4662 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4663 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_68_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_8
  simp only [← c4653, ← c4654, ← c4655] at e_68_8
  have e_68_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_9
  simp only [← c4656, ← c4657, ← c4658] at e_68_9
  have e_69_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_2
  simp only [← c4650, k3, ← c4651, k3, ← c4652] at e_69_2
  have e_69_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_3
  simp only [← c4659, ← c4660, k3, ← c4661] at e_69_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_8
    linear_combination c4662.trans k2 - hc
  · have hc := e_69_3
    simp only [e_69_2, e_68_9] at hc
    linear_combination c4663.trans k2 - hc

theorem publicBatchWrapper86_f578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14007)) (a (.virt 200)) (a (.virt 17771)) (a (.virt 17772)) := by
  have c4664 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4665 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4666 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4667 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4668 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4669 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4670 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4671 := (publicBatchWrapper86_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4672 := (publicBatchWrapper86_copies146 a h).1
  have c4673 := (publicBatchWrapper86_copies146 a h).2.1
  have c4674 := (publicBatchWrapper86_copies146 a h).2.2.1
  have c4675 := (publicBatchWrapper86_copies146 a h).2.2.2.1
  have c4676 := (publicBatchWrapper86_copies146 a h).2.2.2.2.1
  have c4677 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_68_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_10
  simp only [← c4667, ← c4668, ← c4669] at e_68_10
  have e_68_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_11
  simp only [← c4670, ← c4671, ← c4672] at e_68_11
  have e_69_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_4
  simp only [← c4664, k3, ← c4665, k3, ← c4666] at e_69_4
  have e_69_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_5
  simp only [← c4673, ← c4674, k3, ← c4675] at e_69_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_10
    linear_combination c4676.trans k2 - hc
  · have hc := e_69_5
    simp only [e_69_4, e_68_11] at hc
    linear_combination c4677.trans k2 - hc

theorem publicBatchWrapper86_f579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14008)) (a (.virt 200)) (a (.virt 17773)) (a (.virt 17774)) := by
  have c4678 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.1
  have c4679 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.1
  have c4680 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.1
  have c4681 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.1
  have c4682 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4683 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4684 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4685 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4686 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4687 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4688 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4689 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4690 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4691 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_68_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_12
  simp only [← c4681, ← c4682, ← c4683] at e_68_12
  have e_68_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_13
  simp only [← c4684, ← c4685, ← c4686] at e_68_13
  have e_69_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_6
  simp only [← c4678, k3, ← c4679, k3, ← c4680] at e_69_6
  have e_69_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_7
  simp only [← c4687, ← c4688, k3, ← c4689] at e_69_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_12
    linear_combination c4690.trans k2 - hc
  · have hc := e_69_7
    simp only [e_69_6, e_68_13] at hc
    linear_combination c4691.trans k2 - hc

theorem publicBatchWrapper86_f580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 68 59) = band (a (.virt 17767)) (a (.virt 17769)) := by
  have c4692 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4693 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4694 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_14
  simp only [← c4692, ← c4693, ← c4694] at e_68_14
  have hr := e_68_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 68 63) = band (a (.virt 17771)) (a (.virt 17773)) := by
  have c4695 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4696 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4697 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_15
  simp only [← c4695, ← c4696, ← c4697] at e_68_15
  have hr := e_68_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 68 67) = band (a (.wire 68 59)) (a (.wire 68 63)) := by
  have c4698 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4699 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4700 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_68_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_16
  simp only [← c4698, ← c4699, ← c4700] at e_68_16
  have hr := e_68_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14205)) (a (.virt 200)) (a (.virt 17775)) (a (.virt 17776)) := by
  have c4701 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4702 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4703 := (publicBatchWrapper86_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4704 := (publicBatchWrapper86_copies147 a h).1
  have c4705 := (publicBatchWrapper86_copies147 a h).2.1
  have c4706 := (publicBatchWrapper86_copies147 a h).2.2.1
  have c4707 := (publicBatchWrapper86_copies147 a h).2.2.2.1
  have c4708 := (publicBatchWrapper86_copies147 a h).2.2.2.2.1
  have c4709 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.1
  have c4710 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.1
  have c4711 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.1
  have c4712 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.1
  have c4713 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.1
  have c4714 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_68_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_17
  simp only [← c4704, ← c4705, ← c4706] at e_68_17
  have e_68_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_18
  simp only [← c4707, ← c4708, ← c4709] at e_68_18
  have e_69_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_8
  simp only [← c4701, k3, ← c4702, k3, ← c4703] at e_69_8
  have e_69_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_9
  simp only [← c4710, ← c4711, k3, ← c4712] at e_69_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_17
    linear_combination c4713.trans k2 - hc
  · have hc := e_69_9
    simp only [e_69_8, e_68_18] at hc
    linear_combination c4714.trans k2 - hc

theorem publicBatchWrapper86_f584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14206)) (a (.virt 200)) (a (.virt 17777)) (a (.virt 17778)) := by
  have c4715 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4716 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4717 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4718 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4719 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4720 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4721 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4722 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4723 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4724 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4725 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4726 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4727 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4728 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_68_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row68 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_19
  simp only [← c4718, ← c4719, ← c4720] at e_68_19
  have e_69_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_10
  simp only [← c4715, k3, ← c4716, k3, ← c4717] at e_69_10
  have e_69_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_11
  simp only [← c4724, ← c4725, k3, ← c4726] at e_69_11
  have e_70_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_0
  simp only [← c4721, ← c4722, ← c4723] at e_70_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_68_19
    linear_combination c4727.trans k2 - hc
  · have hc := e_69_11
    simp only [e_69_10, e_70_0] at hc
    linear_combination c4728.trans k2 - hc

theorem publicBatchWrapper86_f585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14207)) (a (.virt 200)) (a (.virt 17779)) (a (.virt 17780)) := by
  have c4729 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4730 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4731 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4732 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4733 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4734 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4735 := (publicBatchWrapper86_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4736 := (publicBatchWrapper86_copies148 a h).1
  have c4737 := (publicBatchWrapper86_copies148 a h).2.1
  have c4738 := (publicBatchWrapper86_copies148 a h).2.2.1
  have c4739 := (publicBatchWrapper86_copies148 a h).2.2.2.1
  have c4740 := (publicBatchWrapper86_copies148 a h).2.2.2.2.1
  have c4741 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.1
  have c4742 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_69_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_12
  simp only [← c4729, k3, ← c4730, k3, ← c4731] at e_69_12
  have e_69_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_13
  simp only [← c4738, ← c4739, k3, ← c4740] at e_69_13
  have e_70_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_1
  simp only [← c4732, ← c4733, ← c4734] at e_70_1
  have e_70_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_2
  simp only [← c4735, ← c4736, ← c4737] at e_70_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_1
    linear_combination c4741.trans k2 - hc
  · have hc := e_69_13
    simp only [e_69_12, e_70_2] at hc
    linear_combination c4742.trans k2 - hc

theorem publicBatchWrapper86_f586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14208)) (a (.virt 200)) (a (.virt 17781)) (a (.virt 17782)) := by
  have c4743 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.1
  have c4744 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.1
  have c4745 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.1
  have c4746 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4747 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4748 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4749 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4750 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4751 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4752 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4753 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4754 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4755 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4756 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_69_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_14
  simp only [← c4743, k3, ← c4744, k3, ← c4745] at e_69_14
  have e_69_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_15
  simp only [← c4752, ← c4753, k3, ← c4754] at e_69_15
  have e_70_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_3
  simp only [← c4746, ← c4747, ← c4748] at e_70_3
  have e_70_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_4
  simp only [← c4749, ← c4750, ← c4751] at e_70_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_3
    linear_combination c4755.trans k2 - hc
  · have hc := e_69_15
    simp only [e_69_14, e_70_4] at hc
    linear_combination c4756.trans k2 - hc

theorem publicBatchWrapper86_f587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 70 23) = band (a (.virt 17775)) (a (.virt 17777)) := by
  have c4757 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4758 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4759 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_70_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_5
  simp only [← c4757, ← c4758, ← c4759] at e_70_5
  have hr := e_70_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 70 27) = band (a (.virt 17779)) (a (.virt 17781)) := by
  have c4760 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4761 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4762 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_70_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_6
  simp only [← c4760, ← c4761, ← c4762] at e_70_6
  have hr := e_70_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 70 31) = band (a (.wire 70 23)) (a (.wire 70 27)) := by
  have c4763 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4764 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4765 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_70_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_7
  simp only [← c4763, ← c4764, ← c4765] at e_70_7
  have hr := e_70_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14405)) (a (.virt 200)) (a (.virt 17783)) (a (.virt 17784)) := by
  have c4766 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4767 := (publicBatchWrapper86_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4768 := (publicBatchWrapper86_copies149 a h).1
  have c4769 := (publicBatchWrapper86_copies149 a h).2.1
  have c4770 := (publicBatchWrapper86_copies149 a h).2.2.1
  have c4771 := (publicBatchWrapper86_copies149 a h).2.2.2.1
  have c4772 := (publicBatchWrapper86_copies149 a h).2.2.2.2.1
  have c4773 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.1
  have c4774 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.1
  have c4775 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.1
  have c4776 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.1
  have c4777 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.1
  have c4778 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4779 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_69_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_16
  simp only [← c4766, k3, ← c4767, k3, ← c4768] at e_69_16
  have e_69_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_17
  simp only [← c4775, ← c4776, k3, ← c4777] at e_69_17
  have e_70_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_8
  simp only [← c4769, ← c4770, ← c4771] at e_70_8
  have e_70_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_9
  simp only [← c4772, ← c4773, ← c4774] at e_70_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_8
    linear_combination c4778.trans k2 - hc
  · have hc := e_69_17
    simp only [e_69_16, e_70_9] at hc
    linear_combination c4779.trans k2 - hc

theorem publicBatchWrapper86_f591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14406)) (a (.virt 200)) (a (.virt 17785)) (a (.virt 17786)) := by
  have c4780 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4781 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4782 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4783 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4784 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4785 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4786 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4787 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4788 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4789 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4790 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4791 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4792 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4793 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_69_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_18
  simp only [← c4780, k3, ← c4781, k3, ← c4782] at e_69_18
  have e_69_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row69 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_19
  simp only [← c4789, ← c4790, k3, ← c4791] at e_69_19
  have e_70_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_10
  simp only [← c4783, ← c4784, ← c4785] at e_70_10
  have e_70_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_11
  simp only [← c4786, ← c4787, ← c4788] at e_70_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_10
    linear_combination c4792.trans k2 - hc
  · have hc := e_69_19
    simp only [e_69_18, e_70_11] at hc
    linear_combination c4793.trans k2 - hc

theorem publicBatchWrapper86_f592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14407)) (a (.virt 200)) (a (.virt 17787)) (a (.virt 17788)) := by
  have c4794 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4795 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4796 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4797 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4798 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4799 := (publicBatchWrapper86_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4800 := (publicBatchWrapper86_copies150 a h).1
  have c4801 := (publicBatchWrapper86_copies150 a h).2.1
  have c4802 := (publicBatchWrapper86_copies150 a h).2.2.1
  have c4803 := (publicBatchWrapper86_copies150 a h).2.2.2.1
  have c4804 := (publicBatchWrapper86_copies150 a h).2.2.2.2.1
  have c4805 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.1
  have c4806 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.1
  have c4807 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_70_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_12
  simp only [← c4797, ← c4798, ← c4799] at e_70_12
  have e_70_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_13
  simp only [← c4800, ← c4801, ← c4802] at e_70_13
  have e_71_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_0
  simp only [← c4794, k3, ← c4795, k3, ← c4796] at e_71_0
  have e_71_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_1
  simp only [← c4803, ← c4804, k3, ← c4805] at e_71_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_12
    linear_combination c4806.trans k2 - hc
  · have hc := e_71_1
    simp only [e_71_0, e_70_13] at hc
    linear_combination c4807.trans k2 - hc

theorem publicBatchWrapper86_f593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14408)) (a (.virt 200)) (a (.virt 17789)) (a (.virt 17790)) := by
  have c4808 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.1
  have c4809 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.1
  have c4810 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4811 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4812 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4813 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4814 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4815 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4816 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4817 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4818 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4819 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4820 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4821 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_70_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_14
  simp only [← c4811, ← c4812, ← c4813] at e_70_14
  have e_70_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_15
  simp only [← c4814, ← c4815, ← c4816] at e_70_15
  have e_71_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_2
  simp only [← c4808, k3, ← c4809, k3, ← c4810] at e_71_2
  have e_71_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_3
  simp only [← c4817, ← c4818, k3, ← c4819] at e_71_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_14
    linear_combination c4820.trans k2 - hc
  · have hc := e_71_3
    simp only [e_71_2, e_70_15] at hc
    linear_combination c4821.trans k2 - hc

theorem publicBatchWrapper86_f594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 70 67) = band (a (.virt 17783)) (a (.virt 17785)) := by
  have c4822 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4823 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4824 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_70_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_16
  simp only [← c4822, ← c4823, ← c4824] at e_70_16
  have hr := e_70_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 70 71) = band (a (.virt 17787)) (a (.virt 17789)) := by
  have c4825 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4826 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4827 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_70_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_17
  simp only [← c4825, ← c4826, ← c4827] at e_70_17
  have hr := e_70_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 70 75) = band (a (.wire 70 67)) (a (.wire 70 71)) := by
  have c4828 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4829 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4830 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_70_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_18
  simp only [← c4828, ← c4829, ← c4830] at e_70_18
  have hr := e_70_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14605)) (a (.virt 200)) (a (.virt 17791)) (a (.virt 17792)) := by
  have c4831 := (publicBatchWrapper86_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4832 := (publicBatchWrapper86_copies151 a h).1
  have c4833 := (publicBatchWrapper86_copies151 a h).2.1
  have c4834 := (publicBatchWrapper86_copies151 a h).2.2.1
  have c4835 := (publicBatchWrapper86_copies151 a h).2.2.2.1
  have c4836 := (publicBatchWrapper86_copies151 a h).2.2.2.2.1
  have c4837 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.1
  have c4838 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.1
  have c4839 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.1
  have c4840 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.1
  have c4841 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.1
  have c4842 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4843 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4844 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_70_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row70 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_19
  simp only [← c4834, ← c4835, ← c4836] at e_70_19
  have e_71_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_4
  simp only [← c4831, k3, ← c4832, k3, ← c4833] at e_71_4
  have e_71_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_5
  simp only [← c4840, ← c4841, k3, ← c4842] at e_71_5
  have e_72_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_0
  simp only [← c4837, ← c4838, ← c4839] at e_72_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_70_19
    linear_combination c4843.trans k2 - hc
  · have hc := e_71_5
    simp only [e_71_4, e_72_0] at hc
    linear_combination c4844.trans k2 - hc

theorem publicBatchWrapper86_f598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14606)) (a (.virt 200)) (a (.virt 17793)) (a (.virt 17794)) := by
  have c4845 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4846 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4847 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4848 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4849 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4850 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4851 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4852 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4853 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4854 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4855 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4856 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4857 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4858 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_6
  simp only [← c4845, k3, ← c4846, k3, ← c4847] at e_71_6
  have e_71_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_7
  simp only [← c4854, ← c4855, k3, ← c4856] at e_71_7
  have e_72_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_1
  simp only [← c4848, ← c4849, ← c4850] at e_72_1
  have e_72_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_2
  simp only [← c4851, ← c4852, ← c4853] at e_72_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_1
    linear_combination c4857.trans k2 - hc
  · have hc := e_71_7
    simp only [e_71_6, e_72_2] at hc
    linear_combination c4858.trans k2 - hc

theorem publicBatchWrapper86_f599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14607)) (a (.virt 200)) (a (.virt 17795)) (a (.virt 17796)) := by
  have c4859 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4860 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4861 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4862 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4863 := (publicBatchWrapper86_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4864 := (publicBatchWrapper86_copies152 a h).1
  have c4865 := (publicBatchWrapper86_copies152 a h).2.1
  have c4866 := (publicBatchWrapper86_copies152 a h).2.2.1
  have c4867 := (publicBatchWrapper86_copies152 a h).2.2.2.1
  have c4868 := (publicBatchWrapper86_copies152 a h).2.2.2.2.1
  have c4869 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.1
  have c4870 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.1
  have c4871 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.1
  have c4872 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_8
  simp only [← c4859, k3, ← c4860, k3, ← c4861] at e_71_8
  have e_71_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_9
  simp only [← c4868, ← c4869, k3, ← c4870] at e_71_9
  have e_72_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_3
  simp only [← c4862, ← c4863, ← c4864] at e_72_3
  have e_72_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_4
  simp only [← c4865, ← c4866, ← c4867] at e_72_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_3
    linear_combination c4871.trans k2 - hc
  · have hc := e_71_9
    simp only [e_71_8, e_72_4] at hc
    linear_combination c4872.trans k2 - hc

theorem publicBatchWrapper86_f600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14608)) (a (.virt 200)) (a (.virt 17797)) (a (.virt 17798)) := by
  have c4873 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.1
  have c4874 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4875 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4876 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4877 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4878 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4879 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4880 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4881 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4882 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4883 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4884 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4885 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4886 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_10
  simp only [← c4873, k3, ← c4874, k3, ← c4875] at e_71_10
  have e_71_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_11
  simp only [← c4882, ← c4883, k3, ← c4884] at e_71_11
  have e_72_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_5
  simp only [← c4876, ← c4877, ← c4878] at e_72_5
  have e_72_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_6
  simp only [← c4879, ← c4880, ← c4881] at e_72_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_5
    linear_combination c4885.trans k2 - hc
  · have hc := e_71_11
    simp only [e_71_10, e_72_6] at hc
    linear_combination c4886.trans k2 - hc

theorem publicBatchWrapper86_f601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 72 31) = band (a (.virt 17791)) (a (.virt 17793)) := by
  have c4887 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4888 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4889 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_72_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_7
  simp only [← c4887, ← c4888, ← c4889] at e_72_7
  have hr := e_72_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 72 35) = band (a (.virt 17795)) (a (.virt 17797)) := by
  have c4890 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4891 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4892 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_72_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_8
  simp only [← c4890, ← c4891, ← c4892] at e_72_8
  have hr := e_72_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 72 39) = band (a (.wire 72 31)) (a (.wire 72 35)) := by
  have c4893 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4894 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4895 := (publicBatchWrapper86_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_72_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_9
  simp only [← c4893, ← c4894, ← c4895] at e_72_9
  have hr := e_72_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14805)) (a (.virt 200)) (a (.virt 17799)) (a (.virt 17800)) := by
  have c4896 := (publicBatchWrapper86_copies153 a h).1
  have c4897 := (publicBatchWrapper86_copies153 a h).2.1
  have c4898 := (publicBatchWrapper86_copies153 a h).2.2.1
  have c4899 := (publicBatchWrapper86_copies153 a h).2.2.2.1
  have c4900 := (publicBatchWrapper86_copies153 a h).2.2.2.2.1
  have c4901 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.1
  have c4902 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.1
  have c4903 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.1
  have c4904 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.1
  have c4905 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.1
  have c4906 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4907 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4908 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_12
  simp only [← c4896, k3, ← c4897, k3, ← c4898] at e_71_12
  have e_71_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_13
  simp only [← c4905, ← c4906, k3, ← c4907] at e_71_13
  have e_72_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_10
  simp only [← c4899, ← c4900, ← c4901] at e_72_10
  have e_72_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_11
  simp only [← c4902, ← c4903, ← c4904] at e_72_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_10
    linear_combination c4908.trans k2 - hc
  · have hc := e_71_13
    simp only [e_71_12, e_72_11] at hc
    linear_combination c4909.trans k2 - hc

theorem publicBatchWrapper86_f605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14806)) (a (.virt 200)) (a (.virt 17801)) (a (.virt 17802)) := by
  have c4910 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4913 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4914 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4915 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4916 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4917 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4918 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4919 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4920 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4921 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4922 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4923 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_14
  simp only [← c4910, k3, ← c4911, k3, ← c4912] at e_71_14
  have e_71_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_15
  simp only [← c4919, ← c4920, k3, ← c4921] at e_71_15
  have e_72_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_12
  simp only [← c4913, ← c4914, ← c4915] at e_72_12
  have e_72_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_13
  simp only [← c4916, ← c4917, ← c4918] at e_72_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_12
    linear_combination c4922.trans k2 - hc
  · have hc := e_71_15
    simp only [e_71_14, e_72_13] at hc
    linear_combination c4923.trans k2 - hc

theorem publicBatchWrapper86_f606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14807)) (a (.virt 200)) (a (.virt 17803)) (a (.virt 17804)) := by
  have c4924 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4925 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4926 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4927 := (publicBatchWrapper86_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4928 := (publicBatchWrapper86_copies154 a h).1
  have c4929 := (publicBatchWrapper86_copies154 a h).2.1
  have c4930 := (publicBatchWrapper86_copies154 a h).2.2.1
  have c4931 := (publicBatchWrapper86_copies154 a h).2.2.2.1
  have c4932 := (publicBatchWrapper86_copies154 a h).2.2.2.2.1
  have c4933 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.1
  have c4934 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.1
  have c4935 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.1
  have c4936 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.1
  have c4937 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_16
  simp only [← c4924, k3, ← c4925, k3, ← c4926] at e_71_16
  have e_71_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_17
  simp only [← c4933, ← c4934, k3, ← c4935] at e_71_17
  have e_72_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_14
  simp only [← c4927, ← c4928, ← c4929] at e_72_14
  have e_72_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_15
  simp only [← c4930, ← c4931, ← c4932] at e_72_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_14
    linear_combination c4936.trans k2 - hc
  · have hc := e_71_17
    simp only [e_71_16, e_72_15] at hc
    linear_combination c4937.trans k2 - hc

theorem publicBatchWrapper86_f607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 14808)) (a (.virt 200)) (a (.virt 17805)) (a (.virt 17806)) := by
  have c4938 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4939 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4940 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4941 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4942 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4943 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4944 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4945 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4946 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4947 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4948 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4949 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4950 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4951 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_71_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_18
  simp only [← c4938, k3, ← c4939, k3, ← c4940] at e_71_18
  have e_71_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row71 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_19
  simp only [← c4947, ← c4948, k3, ← c4949] at e_71_19
  have e_72_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_16
  simp only [← c4941, ← c4942, ← c4943] at e_72_16
  have e_72_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_17
  simp only [← c4944, ← c4945, ← c4946] at e_72_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_16
    linear_combination c4950.trans k2 - hc
  · have hc := e_71_19
    simp only [e_71_18, e_72_17] at hc
    linear_combination c4951.trans k2 - hc

theorem publicBatchWrapper86_f608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 72 75) = band (a (.virt 17799)) (a (.virt 17801)) := by
  have c4952 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4953 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4954 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_72_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_18
  simp only [← c4952, ← c4953, ← c4954] at e_72_18
  have hr := e_72_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 72 79) = band (a (.virt 17803)) (a (.virt 17805)) := by
  have c4955 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4956 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4957 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_72_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row72 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_19
  simp only [← c4955, ← c4956, ← c4957] at e_72_19
  have hr := e_72_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 73 3) = band (a (.wire 72 75)) (a (.wire 72 79)) := by
  have c4958 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4959 := (publicBatchWrapper86_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4960 := (publicBatchWrapper86_copies155 a h).1
  have e_73_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_0
  simp only [← c4958, ← c4959, ← c4960] at e_73_0
  have hr := e_73_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15005)) (a (.virt 200)) (a (.virt 17807)) (a (.virt 17808)) := by
  have c4961 := (publicBatchWrapper86_copies155 a h).2.1
  have c4962 := (publicBatchWrapper86_copies155 a h).2.2.1
  have c4963 := (publicBatchWrapper86_copies155 a h).2.2.2.1
  have c4964 := (publicBatchWrapper86_copies155 a h).2.2.2.2.1
  have c4965 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.1
  have c4966 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.1
  have c4967 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.1
  have c4968 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.1
  have c4969 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.1
  have c4970 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4971 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4972 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4973 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4974 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_1
  simp only [← c4964, ← c4965, ← c4966] at e_73_1
  have e_73_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_2
  simp only [← c4967, ← c4968, ← c4969] at e_73_2
  have e_74_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_0
  simp only [← c4961, k3, ← c4962, k3, ← c4963] at e_74_0
  have e_74_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_1
  simp only [← c4970, ← c4971, k3, ← c4972] at e_74_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_1
    linear_combination c4973.trans k2 - hc
  · have hc := e_74_1
    simp only [e_74_0, e_73_2] at hc
    linear_combination c4974.trans k2 - hc

theorem publicBatchWrapper86_f612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15006)) (a (.virt 200)) (a (.virt 17809)) (a (.virt 17810)) := by
  have c4975 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4976 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4977 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4978 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4979 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4980 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4981 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4982 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4983 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4984 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4985 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4986 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4987 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4988 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_3
  simp only [← c4978, ← c4979, ← c4980] at e_73_3
  have e_73_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c4981, ← c4982, ← c4983] at e_73_4
  have e_74_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_2
  simp only [← c4975, k3, ← c4976, k3, ← c4977] at e_74_2
  have e_74_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_3
  simp only [← c4984, ← c4985, k3, ← c4986] at e_74_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_3
    linear_combination c4987.trans k2 - hc
  · have hc := e_74_3
    simp only [e_74_2, e_73_4] at hc
    linear_combination c4988.trans k2 - hc

theorem publicBatchWrapper86_f613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15007)) (a (.virt 200)) (a (.virt 17811)) (a (.virt 17812)) := by
  have c4989 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4990 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4991 := (publicBatchWrapper86_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4992 := (publicBatchWrapper86_copies156 a h).1
  have c4993 := (publicBatchWrapper86_copies156 a h).2.1
  have c4994 := (publicBatchWrapper86_copies156 a h).2.2.1
  have c4995 := (publicBatchWrapper86_copies156 a h).2.2.2.1
  have c4996 := (publicBatchWrapper86_copies156 a h).2.2.2.2.1
  have c4997 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.1
  have c4998 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.1
  have c4999 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.1
  have c5000 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.1
  have c5001 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.1
  have c5002 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_5
  simp only [← c4992, ← c4993, ← c4994] at e_73_5
  have e_73_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_6
  simp only [← c4995, ← c4996, ← c4997] at e_73_6
  have e_74_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_4
  simp only [← c4989, k3, ← c4990, k3, ← c4991] at e_74_4
  have e_74_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_5
  simp only [← c4998, ← c4999, k3, ← c5000] at e_74_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_5
    linear_combination c5001.trans k2 - hc
  · have hc := e_74_5
    simp only [e_74_4, e_73_6] at hc
    linear_combination c5002.trans k2 - hc

theorem publicBatchWrapper86_f614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15008)) (a (.virt 200)) (a (.virt 17813)) (a (.virt 17814)) := by
  have c5003 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5004 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5005 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5006 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5007 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5008 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5009 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5010 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5011 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5012 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5013 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5014 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5015 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5016 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_7
  simp only [← c5006, ← c5007, ← c5008] at e_73_7
  have e_73_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_8
  simp only [← c5009, ← c5010, ← c5011] at e_73_8
  have e_74_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_6
  simp only [← c5003, k3, ← c5004, k3, ← c5005] at e_74_6
  have e_74_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_7
  simp only [← c5012, ← c5013, k3, ← c5014] at e_74_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_7
    linear_combination c5015.trans k2 - hc
  · have hc := e_74_7
    simp only [e_74_6, e_73_8] at hc
    linear_combination c5016.trans k2 - hc

theorem publicBatchWrapper86_f615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 73 39) = band (a (.virt 17807)) (a (.virt 17809)) := by
  have c5017 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5018 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5019 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_73_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_9
  simp only [← c5017, ← c5018, ← c5019] at e_73_9
  have hr := e_73_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 73 43) = band (a (.virt 17811)) (a (.virt 17813)) := by
  have c5020 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5021 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5022 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_73_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_10
  simp only [← c5020, ← c5021, ← c5022] at e_73_10
  have hr := e_73_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 73 47) = band (a (.wire 73 39)) (a (.wire 73 43)) := by
  have c5023 := (publicBatchWrapper86_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5024 := (publicBatchWrapper86_copies157 a h).1
  have c5025 := (publicBatchWrapper86_copies157 a h).2.1
  have e_73_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_11
  simp only [← c5023, ← c5024, ← c5025] at e_73_11
  have hr := e_73_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15205)) (a (.virt 200)) (a (.virt 17815)) (a (.virt 17816)) := by
  have c5026 := (publicBatchWrapper86_copies157 a h).2.2.1
  have c5027 := (publicBatchWrapper86_copies157 a h).2.2.2.1
  have c5028 := (publicBatchWrapper86_copies157 a h).2.2.2.2.1
  have c5029 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.1
  have c5030 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.1
  have c5031 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.1
  have c5032 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.1
  have c5033 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.1
  have c5034 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5035 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5036 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5037 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5038 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5039 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_12
  simp only [← c5029, ← c5030, ← c5031] at e_73_12
  have e_73_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_13
  simp only [← c5032, ← c5033, ← c5034] at e_73_13
  have e_74_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_8
  simp only [← c5026, k3, ← c5027, k3, ← c5028] at e_74_8
  have e_74_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_9
  simp only [← c5035, ← c5036, k3, ← c5037] at e_74_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_12
    linear_combination c5038.trans k2 - hc
  · have hc := e_74_9
    simp only [e_74_8, e_73_13] at hc
    linear_combination c5039.trans k2 - hc

theorem publicBatchWrapper86_f619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15206)) (a (.virt 200)) (a (.virt 17817)) (a (.virt 17818)) := by
  have c5040 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5041 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5042 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5043 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5044 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5045 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5046 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5047 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5048 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5049 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5050 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5051 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5052 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5053 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_14
  simp only [← c5043, ← c5044, ← c5045] at e_73_14
  have e_73_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_15
  simp only [← c5046, ← c5047, ← c5048] at e_73_15
  have e_74_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_10
  simp only [← c5040, k3, ← c5041, k3, ← c5042] at e_74_10
  have e_74_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_11
  simp only [← c5049, ← c5050, k3, ← c5051] at e_74_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_14
    linear_combination c5052.trans k2 - hc
  · have hc := e_74_11
    simp only [e_74_10, e_73_15] at hc
    linear_combination c5053.trans k2 - hc

theorem publicBatchWrapper86_f620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15207)) (a (.virt 200)) (a (.virt 17819)) (a (.virt 17820)) := by
  have c5054 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5055 := (publicBatchWrapper86_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5056 := (publicBatchWrapper86_copies158 a h).1
  have c5057 := (publicBatchWrapper86_copies158 a h).2.1
  have c5058 := (publicBatchWrapper86_copies158 a h).2.2.1
  have c5059 := (publicBatchWrapper86_copies158 a h).2.2.2.1
  have c5060 := (publicBatchWrapper86_copies158 a h).2.2.2.2.1
  have c5061 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.1
  have c5062 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.1
  have c5063 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.1
  have c5064 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.1
  have c5065 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.1
  have c5066 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5067 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_16
  simp only [← c5057, ← c5058, ← c5059] at e_73_16
  have e_73_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_17
  simp only [← c5060, ← c5061, ← c5062] at e_73_17
  have e_74_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_12
  simp only [← c5054, k3, ← c5055, k3, ← c5056] at e_74_12
  have e_74_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_13
  simp only [← c5063, ← c5064, k3, ← c5065] at e_74_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_16
    linear_combination c5066.trans k2 - hc
  · have hc := e_74_13
    simp only [e_74_12, e_73_17] at hc
    linear_combination c5067.trans k2 - hc

theorem publicBatchWrapper86_f621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15208)) (a (.virt 200)) (a (.virt 17821)) (a (.virt 17822)) := by
  have c5068 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5069 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5070 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5071 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5072 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5073 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5074 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5075 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5076 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5077 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5078 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5079 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5080 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5081 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_73_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_18
  simp only [← c5071, ← c5072, ← c5073] at e_73_18
  have e_73_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row73 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_19
  simp only [← c5074, ← c5075, ← c5076] at e_73_19
  have e_74_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_14
  simp only [← c5068, k3, ← c5069, k3, ← c5070] at e_74_14
  have e_74_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_15
  simp only [← c5077, ← c5078, k3, ← c5079] at e_74_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_73_18
    linear_combination c5080.trans k2 - hc
  · have hc := e_74_15
    simp only [e_74_14, e_73_19] at hc
    linear_combination c5081.trans k2 - hc

theorem publicBatchWrapper86_f622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 75 3) = band (a (.virt 17815)) (a (.virt 17817)) := by
  have c5082 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5083 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5084 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_75_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_0
  simp only [← c5082, ← c5083, ← c5084] at e_75_0
  have hr := e_75_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 75 7) = band (a (.virt 17819)) (a (.virt 17821)) := by
  have c5085 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5086 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5087 := (publicBatchWrapper86_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_75_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_1
  simp only [← c5085, ← c5086, ← c5087] at e_75_1
  have hr := e_75_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 75 11) = band (a (.wire 75 3)) (a (.wire 75 7)) := by
  have c5088 := (publicBatchWrapper86_copies159 a h).1
  have c5089 := (publicBatchWrapper86_copies159 a h).2.1
  have c5090 := (publicBatchWrapper86_copies159 a h).2.2.1
  have e_75_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_2
  simp only [← c5088, ← c5089, ← c5090] at e_75_2
  have hr := e_75_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15405)) (a (.virt 200)) (a (.virt 17823)) (a (.virt 17824)) := by
  have c5091 := (publicBatchWrapper86_copies159 a h).2.2.2.1
  have c5092 := (publicBatchWrapper86_copies159 a h).2.2.2.2.1
  have c5093 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.1
  have c5094 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.1
  have c5095 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.1
  have c5096 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.1
  have c5097 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.1
  have c5098 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5099 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5100 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5101 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5102 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5103 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5104 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_74_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_16
  simp only [← c5091, k3, ← c5092, k3, ← c5093] at e_74_16
  have e_74_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_17
  simp only [← c5100, ← c5101, k3, ← c5102] at e_74_17
  have e_75_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_3
  simp only [← c5094, ← c5095, ← c5096] at e_75_3
  have e_75_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_4
  simp only [← c5097, ← c5098, ← c5099] at e_75_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_3
    linear_combination c5103.trans k2 - hc
  · have hc := e_74_17
    simp only [e_74_16, e_75_4] at hc
    linear_combination c5104.trans k2 - hc

theorem publicBatchWrapper86_f626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15406)) (a (.virt 200)) (a (.virt 17825)) (a (.virt 17826)) := by
  have c5105 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5106 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5107 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5108 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5109 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5110 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5111 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5112 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5113 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5114 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5115 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5116 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5117 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5118 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_74_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_18
  simp only [← c5105, k3, ← c5106, k3, ← c5107] at e_74_18
  have e_74_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row74 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_19
  simp only [← c5114, ← c5115, k3, ← c5116] at e_74_19
  have e_75_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_5
  simp only [← c5108, ← c5109, ← c5110] at e_75_5
  have e_75_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_6
  simp only [← c5111, ← c5112, ← c5113] at e_75_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_5
    linear_combination c5117.trans k2 - hc
  · have hc := e_74_19
    simp only [e_74_18, e_75_6] at hc
    linear_combination c5118.trans k2 - hc

theorem publicBatchWrapper86_f627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15407)) (a (.virt 200)) (a (.virt 17827)) (a (.virt 17828)) := by
  have c5119 := (publicBatchWrapper86_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5120 := (publicBatchWrapper86_copies160 a h).1
  have c5121 := (publicBatchWrapper86_copies160 a h).2.1
  have c5122 := (publicBatchWrapper86_copies160 a h).2.2.1
  have c5123 := (publicBatchWrapper86_copies160 a h).2.2.2.1
  have c5124 := (publicBatchWrapper86_copies160 a h).2.2.2.2.1
  have c5125 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.1
  have c5126 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.1
  have c5127 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.1
  have c5128 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.1
  have c5129 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.1
  have c5130 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5131 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5132 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_75_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_7
  simp only [← c5122, ← c5123, ← c5124] at e_75_7
  have e_75_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_8
  simp only [← c5125, ← c5126, ← c5127] at e_75_8
  have e_76_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_0
  simp only [← c5119, k3, ← c5120, k3, ← c5121] at e_76_0
  have e_76_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_1
  simp only [← c5128, ← c5129, k3, ← c5130] at e_76_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_7
    linear_combination c5131.trans k2 - hc
  · have hc := e_76_1
    simp only [e_76_0, e_75_8] at hc
    linear_combination c5132.trans k2 - hc

theorem publicBatchWrapper86_f628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15408)) (a (.virt 200)) (a (.virt 17829)) (a (.virt 17830)) := by
  have c5133 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5134 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5135 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5136 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5137 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5138 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5139 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5140 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5141 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5142 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5143 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5144 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5145 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5146 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_75_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_9
  simp only [← c5136, ← c5137, ← c5138] at e_75_9
  have e_75_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_10
  simp only [← c5139, ← c5140, ← c5141] at e_75_10
  have e_76_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_2
  simp only [← c5133, k3, ← c5134, k3, ← c5135] at e_76_2
  have e_76_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_3
  simp only [← c5142, ← c5143, k3, ← c5144] at e_76_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_9
    linear_combination c5145.trans k2 - hc
  · have hc := e_76_3
    simp only [e_76_2, e_75_10] at hc
    linear_combination c5146.trans k2 - hc

theorem publicBatchWrapper86_f629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 75 47) = band (a (.virt 17823)) (a (.virt 17825)) := by
  have c5147 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5148 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5149 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_75_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_11
  simp only [← c5147, ← c5148, ← c5149] at e_75_11
  have hr := e_75_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 75 51) = band (a (.virt 17827)) (a (.virt 17829)) := by
  have c5150 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5151 := (publicBatchWrapper86_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5152 := (publicBatchWrapper86_copies161 a h).1
  have e_75_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_12
  simp only [← c5150, ← c5151, ← c5152] at e_75_12
  have hr := e_75_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 75 55) = band (a (.wire 75 47)) (a (.wire 75 51)) := by
  have c5153 := (publicBatchWrapper86_copies161 a h).2.1
  have c5154 := (publicBatchWrapper86_copies161 a h).2.2.1
  have c5155 := (publicBatchWrapper86_copies161 a h).2.2.2.1
  have e_75_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_13
  simp only [← c5153, ← c5154, ← c5155] at e_75_13
  have hr := e_75_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15605)) (a (.virt 200)) (a (.virt 17831)) (a (.virt 17832)) := by
  have c5156 := (publicBatchWrapper86_copies161 a h).2.2.2.2.1
  have c5157 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.1
  have c5158 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.1
  have c5159 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.1
  have c5160 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.1
  have c5161 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.1
  have c5162 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5163 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5164 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5165 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5166 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5167 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5168 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5169 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_75_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_14
  simp only [← c5159, ← c5160, ← c5161] at e_75_14
  have e_75_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_15
  simp only [← c5162, ← c5163, ← c5164] at e_75_15
  have e_76_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_4
  simp only [← c5156, k3, ← c5157, k3, ← c5158] at e_76_4
  have e_76_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_5
  simp only [← c5165, ← c5166, k3, ← c5167] at e_76_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_14
    linear_combination c5168.trans k2 - hc
  · have hc := e_76_5
    simp only [e_76_4, e_75_15] at hc
    linear_combination c5169.trans k2 - hc

theorem publicBatchWrapper86_f633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15606)) (a (.virt 200)) (a (.virt 17833)) (a (.virt 17834)) := by
  have c5170 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5171 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5172 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5173 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5174 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5175 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5176 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5177 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5178 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5179 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5180 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5181 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5182 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5183 := (publicBatchWrapper86_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_75_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_16
  simp only [← c5173, ← c5174, ← c5175] at e_75_16
  have e_75_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_17
  simp only [← c5176, ← c5177, ← c5178] at e_75_17
  have e_76_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_6
  simp only [← c5170, k3, ← c5171, k3, ← c5172] at e_76_6
  have e_76_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_7
  simp only [← c5179, ← c5180, k3, ← c5181] at e_76_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_16
    linear_combination c5182.trans k2 - hc
  · have hc := e_76_7
    simp only [e_76_6, e_75_17] at hc
    linear_combination c5183.trans k2 - hc

theorem publicBatchWrapper86_f634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15607)) (a (.virt 200)) (a (.virt 17835)) (a (.virt 17836)) := by
  have c5184 := (publicBatchWrapper86_copies162 a h).1
  have c5185 := (publicBatchWrapper86_copies162 a h).2.1
  have c5186 := (publicBatchWrapper86_copies162 a h).2.2.1
  have c5187 := (publicBatchWrapper86_copies162 a h).2.2.2.1
  have c5188 := (publicBatchWrapper86_copies162 a h).2.2.2.2.1
  have c5189 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.1
  have c5190 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.1
  have c5191 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.1
  have c5192 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.1
  have c5193 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.1
  have c5194 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5195 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5196 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5197 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_75_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_18
  simp only [← c5187, ← c5188, ← c5189] at e_75_18
  have e_75_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row75 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_19
  simp only [← c5190, ← c5191, ← c5192] at e_75_19
  have e_76_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_8
  simp only [← c5184, k3, ← c5185, k3, ← c5186] at e_76_8
  have e_76_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_9
  simp only [← c5193, ← c5194, k3, ← c5195] at e_76_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_75_18
    linear_combination c5196.trans k2 - hc
  · have hc := e_76_9
    simp only [e_76_8, e_75_19] at hc
    linear_combination c5197.trans k2 - hc

theorem publicBatchWrapper86_f635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15608)) (a (.virt 200)) (a (.virt 17837)) (a (.virt 17838)) := by
  have c5198 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5199 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5200 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5201 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5202 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5203 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5204 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5205 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5206 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5207 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5208 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5209 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5210 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5211 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_76_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_10
  simp only [← c5198, k3, ← c5199, k3, ← c5200] at e_76_10
  have e_76_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_11
  simp only [← c5207, ← c5208, k3, ← c5209] at e_76_11
  have e_77_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_0
  simp only [← c5201, ← c5202, ← c5203] at e_77_0
  have e_77_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_1
  simp only [← c5204, ← c5205, ← c5206] at e_77_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_0
    linear_combination c5210.trans k2 - hc
  · have hc := e_76_11
    simp only [e_76_10, e_77_1] at hc
    linear_combination c5211.trans k2 - hc

theorem publicBatchWrapper86_f636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 77 11) = band (a (.virt 17831)) (a (.virt 17833)) := by
  have c5212 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5213 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5214 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_77_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_2
  simp only [← c5212, ← c5213, ← c5214] at e_77_2
  have hr := e_77_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 77 15) = band (a (.virt 17835)) (a (.virt 17837)) := by
  have c5215 := (publicBatchWrapper86_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5216 := (publicBatchWrapper86_copies163 a h).1
  have c5217 := (publicBatchWrapper86_copies163 a h).2.1
  have e_77_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_3
  simp only [← c5215, ← c5216, ← c5217] at e_77_3
  have hr := e_77_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 77 19) = band (a (.wire 77 11)) (a (.wire 77 15)) := by
  have c5218 := (publicBatchWrapper86_copies163 a h).2.2.1
  have c5219 := (publicBatchWrapper86_copies163 a h).2.2.2.1
  have c5220 := (publicBatchWrapper86_copies163 a h).2.2.2.2.1
  have e_77_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_4
  simp only [← c5218, ← c5219, ← c5220] at e_77_4
  have hr := e_77_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15805)) (a (.virt 200)) (a (.virt 17839)) (a (.virt 17840)) := by
  have c5221 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.1
  have c5222 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.1
  have c5223 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.1
  have c5224 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.1
  have c5225 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.1
  have c5226 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5227 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5228 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5229 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5230 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5231 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5232 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5233 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5234 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_76_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_12
  simp only [← c5221, k3, ← c5222, k3, ← c5223] at e_76_12
  have e_76_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_13
  simp only [← c5230, ← c5231, k3, ← c5232] at e_76_13
  have e_77_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_5
  simp only [← c5224, ← c5225, ← c5226] at e_77_5
  have e_77_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_6
  simp only [← c5227, ← c5228, ← c5229] at e_77_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_5
    linear_combination c5233.trans k2 - hc
  · have hc := e_76_13
    simp only [e_76_12, e_77_6] at hc
    linear_combination c5234.trans k2 - hc

theorem publicBatchWrapper86_f640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15806)) (a (.virt 200)) (a (.virt 17841)) (a (.virt 17842)) := by
  have c5235 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5236 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5237 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5238 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5239 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5240 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5241 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5242 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5243 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5244 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5245 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5246 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5247 := (publicBatchWrapper86_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5248 := (publicBatchWrapper86_copies164 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_76_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_14
  simp only [← c5235, k3, ← c5236, k3, ← c5237] at e_76_14
  have e_76_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_15
  simp only [← c5244, ← c5245, k3, ← c5246] at e_76_15
  have e_77_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_7
  simp only [← c5238, ← c5239, ← c5240] at e_77_7
  have e_77_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_8
  simp only [← c5241, ← c5242, ← c5243] at e_77_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_7
    linear_combination c5247.trans k2 - hc
  · have hc := e_76_15
    simp only [e_76_14, e_77_8] at hc
    linear_combination c5248.trans k2 - hc

theorem publicBatchWrapper86_f641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15807)) (a (.virt 200)) (a (.virt 17843)) (a (.virt 17844)) := by
  have c5249 := (publicBatchWrapper86_copies164 a h).2.1
  have c5250 := (publicBatchWrapper86_copies164 a h).2.2.1
  have c5251 := (publicBatchWrapper86_copies164 a h).2.2.2.1
  have c5252 := (publicBatchWrapper86_copies164 a h).2.2.2.2.1
  have c5253 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.1
  have c5254 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.1
  have c5255 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.1
  have c5256 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.1
  have c5257 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.1
  have c5258 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5259 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5260 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5261 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5262 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_76_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_16
  simp only [← c5249, k3, ← c5250, k3, ← c5251] at e_76_16
  have e_76_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_17
  simp only [← c5258, ← c5259, k3, ← c5260] at e_76_17
  have e_77_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_9
  simp only [← c5252, ← c5253, ← c5254] at e_77_9
  have e_77_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_10
  simp only [← c5255, ← c5256, ← c5257] at e_77_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_9
    linear_combination c5261.trans k2 - hc
  · have hc := e_76_17
    simp only [e_76_16, e_77_10] at hc
    linear_combination c5262.trans k2 - hc

theorem publicBatchWrapper86_f642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 15808)) (a (.virt 200)) (a (.virt 17845)) (a (.virt 17846)) := by
  have c5263 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5264 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5265 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5266 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5267 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5268 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5269 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5270 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5271 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5272 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5273 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5274 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5275 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5276 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_76_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_18
  simp only [← c5263, k3, ← c5264, k3, ← c5265] at e_76_18
  have e_76_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row76 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_19
  simp only [← c5272, ← c5273, k3, ← c5274] at e_76_19
  have e_77_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_11
  simp only [← c5266, ← c5267, ← c5268] at e_77_11
  have e_77_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_12
  simp only [← c5269, ← c5270, ← c5271] at e_77_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_11
    linear_combination c5275.trans k2 - hc
  · have hc := e_76_19
    simp only [e_76_18, e_77_12] at hc
    linear_combination c5276.trans k2 - hc

theorem publicBatchWrapper86_f643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 77 55) = band (a (.virt 17839)) (a (.virt 17841)) := by
  have c5277 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5278 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5279 := (publicBatchWrapper86_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_77_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_13
  simp only [← c5277, ← c5278, ← c5279] at e_77_13
  have hr := e_77_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 77 59) = band (a (.virt 17843)) (a (.virt 17845)) := by
  have c5280 := (publicBatchWrapper86_copies165 a h).1
  have c5281 := (publicBatchWrapper86_copies165 a h).2.1
  have c5282 := (publicBatchWrapper86_copies165 a h).2.2.1
  have e_77_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_14
  simp only [← c5280, ← c5281, ← c5282] at e_77_14
  have hr := e_77_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 77 63) = band (a (.wire 77 55)) (a (.wire 77 59)) := by
  have c5283 := (publicBatchWrapper86_copies165 a h).2.2.2.1
  have c5284 := (publicBatchWrapper86_copies165 a h).2.2.2.2.1
  have c5285 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.1
  have e_77_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_15
  simp only [← c5283, ← c5284, ← c5285] at e_77_15
  have hr := e_77_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16005)) (a (.virt 200)) (a (.virt 17847)) (a (.virt 17848)) := by
  have c5286 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.1
  have c5287 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.1
  have c5288 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.1
  have c5289 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.1
  have c5290 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5291 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5292 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5293 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5294 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5295 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5296 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5297 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5298 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5299 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_77_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_16
  simp only [← c5289, ← c5290, ← c5291] at e_77_16
  have e_77_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_17
  simp only [← c5292, ← c5293, ← c5294] at e_77_17
  have e_78_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_0
  simp only [← c5286, k3, ← c5287, k3, ← c5288] at e_78_0
  have e_78_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_1
  simp only [← c5295, ← c5296, k3, ← c5297] at e_78_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_16
    linear_combination c5298.trans k2 - hc
  · have hc := e_78_1
    simp only [e_78_0, e_77_17] at hc
    linear_combination c5299.trans k2 - hc

theorem publicBatchWrapper86_f647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16006)) (a (.virt 200)) (a (.virt 17849)) (a (.virt 17850)) := by
  have c5300 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5301 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5302 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5303 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5304 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5305 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5306 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5307 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5308 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5309 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5310 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5311 := (publicBatchWrapper86_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5312 := (publicBatchWrapper86_copies166 a h).1
  have c5313 := (publicBatchWrapper86_copies166 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_77_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_18
  simp only [← c5303, ← c5304, ← c5305] at e_77_18
  have e_77_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row77 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_19
  simp only [← c5306, ← c5307, ← c5308] at e_77_19
  have e_78_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_2
  simp only [← c5300, k3, ← c5301, k3, ← c5302] at e_78_2
  have e_78_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_3
  simp only [← c5309, ← c5310, k3, ← c5311] at e_78_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_18
    linear_combination c5312.trans k2 - hc
  · have hc := e_78_3
    simp only [e_78_2, e_77_19] at hc
    linear_combination c5313.trans k2 - hc

theorem publicBatchWrapper86_f648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16007)) (a (.virt 200)) (a (.virt 17851)) (a (.virt 17852)) := by
  have c5314 := (publicBatchWrapper86_copies166 a h).2.2.1
  have c5315 := (publicBatchWrapper86_copies166 a h).2.2.2.1
  have c5316 := (publicBatchWrapper86_copies166 a h).2.2.2.2.1
  have c5317 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.1
  have c5318 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.1
  have c5319 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.1
  have c5320 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.1
  have c5321 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.1
  have c5322 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5323 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5324 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5325 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5326 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5327 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_4
  simp only [← c5314, k3, ← c5315, k3, ← c5316] at e_78_4
  have e_78_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_5
  simp only [← c5323, ← c5324, k3, ← c5325] at e_78_5
  have e_79_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_0
  simp only [← c5317, ← c5318, ← c5319] at e_79_0
  have e_79_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_1
  simp only [← c5320, ← c5321, ← c5322] at e_79_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_0
    linear_combination c5326.trans k2 - hc
  · have hc := e_78_5
    simp only [e_78_4, e_79_1] at hc
    linear_combination c5327.trans k2 - hc

theorem publicBatchWrapper86_f649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16008)) (a (.virt 200)) (a (.virt 17853)) (a (.virt 17854)) := by
  have c5328 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5329 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5330 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5331 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5332 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5333 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5334 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5335 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5336 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5337 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5338 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5339 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5340 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5341 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_6
  simp only [← c5328, k3, ← c5329, k3, ← c5330] at e_78_6
  have e_78_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_7
  simp only [← c5337, ← c5338, k3, ← c5339] at e_78_7
  have e_79_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_2
  simp only [← c5331, ← c5332, ← c5333] at e_79_2
  have e_79_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_3
  simp only [← c5334, ← c5335, ← c5336] at e_79_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_2
    linear_combination c5340.trans k2 - hc
  · have hc := e_78_7
    simp only [e_78_6, e_79_3] at hc
    linear_combination c5341.trans k2 - hc

theorem publicBatchWrapper86_f650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 79 19) = band (a (.virt 17847)) (a (.virt 17849)) := by
  have c5342 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5343 := (publicBatchWrapper86_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5344 := (publicBatchWrapper86_copies167 a h).1
  have e_79_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_4
  simp only [← c5342, ← c5343, ← c5344] at e_79_4
  have hr := e_79_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 79 23) = band (a (.virt 17851)) (a (.virt 17853)) := by
  have c5345 := (publicBatchWrapper86_copies167 a h).2.1
  have c5346 := (publicBatchWrapper86_copies167 a h).2.2.1
  have c5347 := (publicBatchWrapper86_copies167 a h).2.2.2.1
  have e_79_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_5
  simp only [← c5345, ← c5346, ← c5347] at e_79_5
  have hr := e_79_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 79 27) = band (a (.wire 79 19)) (a (.wire 79 23)) := by
  have c5348 := (publicBatchWrapper86_copies167 a h).2.2.2.2.1
  have c5349 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.1
  have c5350 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.1
  have e_79_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_6
  simp only [← c5348, ← c5349, ← c5350] at e_79_6
  have hr := e_79_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16205)) (a (.virt 200)) (a (.virt 17855)) (a (.virt 17856)) := by
  have c5351 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.1
  have c5352 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.1
  have c5353 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.1
  have c5354 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5355 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5356 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5357 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5358 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5359 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5360 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5361 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5362 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5363 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5364 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_8
  simp only [← c5351, k3, ← c5352, k3, ← c5353] at e_78_8
  have e_78_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_9
  simp only [← c5360, ← c5361, k3, ← c5362] at e_78_9
  have e_79_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_7
  simp only [← c5354, ← c5355, ← c5356] at e_79_7
  have e_79_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_8
  simp only [← c5357, ← c5358, ← c5359] at e_79_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_7
    linear_combination c5363.trans k2 - hc
  · have hc := e_78_9
    simp only [e_78_8, e_79_8] at hc
    linear_combination c5364.trans k2 - hc

theorem publicBatchWrapper86_f654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16206)) (a (.virt 200)) (a (.virt 17857)) (a (.virt 17858)) := by
  have c5365 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5366 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5367 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5368 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5369 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5370 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5371 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5372 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5373 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5374 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5375 := (publicBatchWrapper86_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5376 := (publicBatchWrapper86_copies168 a h).1
  have c5377 := (publicBatchWrapper86_copies168 a h).2.1
  have c5378 := (publicBatchWrapper86_copies168 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_10
  simp only [← c5365, k3, ← c5366, k3, ← c5367] at e_78_10
  have e_78_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_11
  simp only [← c5374, ← c5375, k3, ← c5376] at e_78_11
  have e_79_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_9
  simp only [← c5368, ← c5369, ← c5370] at e_79_9
  have e_79_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_10
  simp only [← c5371, ← c5372, ← c5373] at e_79_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_9
    linear_combination c5377.trans k2 - hc
  · have hc := e_78_11
    simp only [e_78_10, e_79_10] at hc
    linear_combination c5378.trans k2 - hc

theorem publicBatchWrapper86_f655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16207)) (a (.virt 200)) (a (.virt 17859)) (a (.virt 17860)) := by
  have c5379 := (publicBatchWrapper86_copies168 a h).2.2.2.1
  have c5380 := (publicBatchWrapper86_copies168 a h).2.2.2.2.1
  have c5381 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.1
  have c5382 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.1
  have c5383 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.1
  have c5384 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.1
  have c5385 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.1
  have c5386 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5391 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5392 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_12
  simp only [← c5379, k3, ← c5380, k3, ← c5381] at e_78_12
  have e_78_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_13
  simp only [← c5388, ← c5389, k3, ← c5390] at e_78_13
  have e_79_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_11
  simp only [← c5382, ← c5383, ← c5384] at e_79_11
  have e_79_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_12
  simp only [← c5385, ← c5386, ← c5387] at e_79_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_11
    linear_combination c5391.trans k2 - hc
  · have hc := e_78_13
    simp only [e_78_12, e_79_12] at hc
    linear_combination c5392.trans k2 - hc

theorem publicBatchWrapper86_f656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16208)) (a (.virt 200)) (a (.virt 17861)) (a (.virt 17862)) := by
  have c5393 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5394 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5395 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5396 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5397 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5398 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5399 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5400 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5401 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5402 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5403 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5404 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5405 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5406 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_14
  simp only [← c5393, k3, ← c5394, k3, ← c5395] at e_78_14
  have e_78_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_15
  simp only [← c5402, ← c5403, k3, ← c5404] at e_78_15
  have e_79_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_13
  simp only [← c5396, ← c5397, ← c5398] at e_79_13
  have e_79_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_14
  simp only [← c5399, ← c5400, ← c5401] at e_79_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_13
    linear_combination c5405.trans k2 - hc
  · have hc := e_78_15
    simp only [e_78_14, e_79_14] at hc
    linear_combination c5406.trans k2 - hc

theorem publicBatchWrapper86_f657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 79 63) = band (a (.virt 17855)) (a (.virt 17857)) := by
  have c5407 := (publicBatchWrapper86_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5408 := (publicBatchWrapper86_copies169 a h).1
  have c5409 := (publicBatchWrapper86_copies169 a h).2.1
  have e_79_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_15
  simp only [← c5407, ← c5408, ← c5409] at e_79_15
  have hr := e_79_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 79 67) = band (a (.virt 17859)) (a (.virt 17861)) := by
  have c5410 := (publicBatchWrapper86_copies169 a h).2.2.1
  have c5411 := (publicBatchWrapper86_copies169 a h).2.2.2.1
  have c5412 := (publicBatchWrapper86_copies169 a h).2.2.2.2.1
  have e_79_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_16
  simp only [← c5410, ← c5411, ← c5412] at e_79_16
  have hr := e_79_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 79 71) = band (a (.wire 79 63)) (a (.wire 79 67)) := by
  have c5413 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.1
  have c5414 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.1
  have c5415 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.1
  have e_79_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_17
  simp only [← c5413, ← c5414, ← c5415] at e_79_17
  have hr := e_79_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16405)) (a (.virt 200)) (a (.virt 17863)) (a (.virt 17864)) := by
  have c5416 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.1
  have c5417 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.1
  have c5418 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5419 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5420 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5421 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5422 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5423 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5424 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5425 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5426 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5427 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5428 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5429 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_16
  simp only [← c5416, k3, ← c5417, k3, ← c5418] at e_78_16
  have e_78_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_17
  simp only [← c5425, ← c5426, k3, ← c5427] at e_78_17
  have e_79_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_18
  simp only [← c5419, ← c5420, ← c5421] at e_79_18
  have e_79_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row79 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_19
  simp only [← c5422, ← c5423, ← c5424] at e_79_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_18
    linear_combination c5428.trans k2 - hc
  · have hc := e_78_17
    simp only [e_78_16, e_79_19] at hc
    linear_combination c5429.trans k2 - hc

theorem publicBatchWrapper86_f661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16406)) (a (.virt 200)) (a (.virt 17865)) (a (.virt 17866)) := by
  have c5430 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5431 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5432 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5433 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5434 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5435 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5436 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5437 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5438 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5439 := (publicBatchWrapper86_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5440 := (publicBatchWrapper86_copies170 a h).1
  have c5441 := (publicBatchWrapper86_copies170 a h).2.1
  have c5442 := (publicBatchWrapper86_copies170 a h).2.2.1
  have c5443 := (publicBatchWrapper86_copies170 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_78_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_18
  simp only [← c5430, k3, ← c5431, k3, ← c5432] at e_78_18
  have e_78_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row78 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_19
  simp only [← c5439, ← c5440, k3, ← c5441] at e_78_19
  have e_80_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_0
  simp only [← c5433, ← c5434, ← c5435] at e_80_0
  have e_80_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_1
  simp only [← c5436, ← c5437, ← c5438] at e_80_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_80_0
    linear_combination c5442.trans k2 - hc
  · have hc := e_78_19
    simp only [e_78_18, e_80_1] at hc
    linear_combination c5443.trans k2 - hc

theorem publicBatchWrapper86_f662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16407)) (a (.virt 200)) (a (.virt 17867)) (a (.virt 17868)) := by
  have c5444 := (publicBatchWrapper86_copies170 a h).2.2.2.2.1
  have c5445 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.1
  have c5446 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.1
  have c5447 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.1
  have c5448 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.1
  have c5449 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.1
  have c5450 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5451 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5452 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5453 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5454 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5455 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5456 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5457 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_80_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_2
  simp only [← c5447, ← c5448, ← c5449] at e_80_2
  have e_80_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_3
  simp only [← c5450, ← c5451, ← c5452] at e_80_3
  have e_81_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row81 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_0
  simp only [← c5444, k3, ← c5445, k3, ← c5446] at e_81_0
  have e_81_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row81 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_1
  simp only [← c5453, ← c5454, k3, ← c5455] at e_81_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_80_2
    linear_combination c5456.trans k2 - hc
  · have hc := e_81_1
    simp only [e_81_0, e_80_3] at hc
    linear_combination c5457.trans k2 - hc

theorem publicBatchWrapper86_f663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16408)) (a (.virt 200)) (a (.virt 17869)) (a (.virt 17870)) := by
  have c5458 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5459 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5460 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5461 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5462 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5463 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5464 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5465 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5466 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5467 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5468 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5469 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5470 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5471 := (publicBatchWrapper86_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_80_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_4
  simp only [← c5461, ← c5462, ← c5463] at e_80_4
  have e_80_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_5
  simp only [← c5464, ← c5465, ← c5466] at e_80_5
  have e_81_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row81 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_2
  simp only [← c5458, k3, ← c5459, k3, ← c5460] at e_81_2
  have e_81_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row81 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_3
  simp only [← c5467, ← c5468, k3, ← c5469] at e_81_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_80_4
    linear_combination c5470.trans k2 - hc
  · have hc := e_81_3
    simp only [e_81_2, e_80_5] at hc
    linear_combination c5471.trans k2 - hc

theorem publicBatchWrapper86_f664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 80 27) = band (a (.virt 17863)) (a (.virt 17865)) := by
  have c5472 := (publicBatchWrapper86_copies171 a h).1
  have c5473 := (publicBatchWrapper86_copies171 a h).2.1
  have c5474 := (publicBatchWrapper86_copies171 a h).2.2.1
  have e_80_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_6
  simp only [← c5472, ← c5473, ← c5474] at e_80_6
  have hr := e_80_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 80 31) = band (a (.virt 17867)) (a (.virt 17869)) := by
  have c5475 := (publicBatchWrapper86_copies171 a h).2.2.2.1
  have c5476 := (publicBatchWrapper86_copies171 a h).2.2.2.2.1
  have c5477 := (publicBatchWrapper86_copies171 a h).2.2.2.2.2.1
  have e_80_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_7
  simp only [← c5475, ← c5476, ← c5477] at e_80_7
  have hr := e_80_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 80 35) = band (a (.wire 80 27)) (a (.wire 80 31)) := by
  have c5478 := (publicBatchWrapper86_copies171 a h).2.2.2.2.2.2.1
  have c5479 := (publicBatchWrapper86_copies171 a h).2.2.2.2.2.2.2.1
  have c5480 := (publicBatchWrapper86_copies171 a h).2.2.2.2.2.2.2.2.1
  have e_80_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row80 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_8
  simp only [← c5478, ← c5479, ← c5480] at e_80_8
  have hr := e_80_8
  simp only [band]
  linear_combination hr

end Plonky2Spec.Generated
