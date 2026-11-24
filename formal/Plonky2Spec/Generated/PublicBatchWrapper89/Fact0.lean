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

theorem publicBatchWrapper89_f0 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 0) = a (.virt 17806) := by
  have c0 := (publicBatchWrapper89_copies0 a h).1
  exact c0

theorem publicBatchWrapper89_f1 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 202) = a (.virt 17806) := by
  have c1 := (publicBatchWrapper89_copies0 a h).2.1
  exact c1

theorem publicBatchWrapper89_f2 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 402) = a (.virt 17806) := by
  have c2 := (publicBatchWrapper89_copies0 a h).2.2.1
  exact c2

theorem publicBatchWrapper89_f3 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 602) = a (.virt 17806) := by
  have c3 := (publicBatchWrapper89_copies0 a h).2.2.2.1
  exact c3

theorem publicBatchWrapper89_f4 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 802) = a (.virt 17806) := by
  have c4 := (publicBatchWrapper89_copies0 a h).2.2.2.2.1
  exact c4

theorem publicBatchWrapper89_f5 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 1002) = a (.virt 17806) := by
  have c5 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.1
  exact c5

theorem publicBatchWrapper89_f6 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 1202) = a (.virt 17806) := by
  have c6 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.1
  exact c6

theorem publicBatchWrapper89_f7 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 1402) = a (.virt 17806) := by
  have c7 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.1
  exact c7

theorem publicBatchWrapper89_f8 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 1602) = a (.virt 17806) := by
  have c8 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.1
  exact c8

theorem publicBatchWrapper89_f9 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 1802) = a (.virt 17806) := by
  have c9 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.1
  exact c9

theorem publicBatchWrapper89_f10 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 2002) = a (.virt 17806) := by
  have c10 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c10

theorem publicBatchWrapper89_f11 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 2202) = a (.virt 17806) := by
  have c11 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c11

theorem publicBatchWrapper89_f12 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 2402) = a (.virt 17806) := by
  have c12 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c12

theorem publicBatchWrapper89_f13 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 2602) = a (.virt 17806) := by
  have c13 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c13

theorem publicBatchWrapper89_f14 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 2802) = a (.virt 17806) := by
  have c14 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c14

theorem publicBatchWrapper89_f15 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 3002) = a (.virt 17806) := by
  have c15 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c15

theorem publicBatchWrapper89_f16 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 3202) = a (.virt 17806) := by
  have c16 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c16

theorem publicBatchWrapper89_f17 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 3402) = a (.virt 17806) := by
  have c17 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c17

theorem publicBatchWrapper89_f18 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 3602) = a (.virt 17806) := by
  have c18 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c18

theorem publicBatchWrapper89_f19 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 3802) = a (.virt 17806) := by
  have c19 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c19

theorem publicBatchWrapper89_f20 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 4002) = a (.virt 17806) := by
  have c20 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c20

theorem publicBatchWrapper89_f21 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 4202) = a (.virt 17806) := by
  have c21 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21

theorem publicBatchWrapper89_f22 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 4402) = a (.virt 17806) := by
  have c22 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c22

theorem publicBatchWrapper89_f23 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 4602) = a (.virt 17806) := by
  have c23 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c23

theorem publicBatchWrapper89_f24 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 4802) = a (.virt 17806) := by
  have c24 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c24

theorem publicBatchWrapper89_f25 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 5002) = a (.virt 17806) := by
  have c25 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c25

theorem publicBatchWrapper89_f26 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 5202) = a (.virt 17806) := by
  have c26 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c26

theorem publicBatchWrapper89_f27 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 5402) = a (.virt 17806) := by
  have c27 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c27

theorem publicBatchWrapper89_f28 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 5602) = a (.virt 17806) := by
  have c28 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c28

theorem publicBatchWrapper89_f29 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 5802) = a (.virt 17806) := by
  have c29 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c29

theorem publicBatchWrapper89_f30 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 6002) = a (.virt 17806) := by
  have c30 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c30

theorem publicBatchWrapper89_f31 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 6202) = a (.virt 17806) := by
  have c31 := (publicBatchWrapper89_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c31

theorem publicBatchWrapper89_f32 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 6402) = a (.virt 17806) := by
  have c32 := (publicBatchWrapper89_copies1 a h).1
  exact c32

theorem publicBatchWrapper89_f33 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 6602) = a (.virt 17806) := by
  have c33 := (publicBatchWrapper89_copies1 a h).2.1
  exact c33

theorem publicBatchWrapper89_f34 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 6802) = a (.virt 17806) := by
  have c34 := (publicBatchWrapper89_copies1 a h).2.2.1
  exact c34

theorem publicBatchWrapper89_f35 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 7002) = a (.virt 17806) := by
  have c35 := (publicBatchWrapper89_copies1 a h).2.2.2.1
  exact c35

theorem publicBatchWrapper89_f36 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 7202) = a (.virt 17806) := by
  have c36 := (publicBatchWrapper89_copies1 a h).2.2.2.2.1
  exact c36

theorem publicBatchWrapper89_f37 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 7402) = a (.virt 17806) := by
  have c37 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.1
  exact c37

theorem publicBatchWrapper89_f38 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 7602) = a (.virt 17806) := by
  have c38 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.1
  exact c38

theorem publicBatchWrapper89_f39 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 7802) = a (.virt 17806) := by
  have c39 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.1
  exact c39

theorem publicBatchWrapper89_f40 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 8002) = a (.virt 17806) := by
  have c40 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.1
  exact c40

theorem publicBatchWrapper89_f41 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 8202) = a (.virt 17806) := by
  have c41 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.1
  exact c41

theorem publicBatchWrapper89_f42 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 8402) = a (.virt 17806) := by
  have c42 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c42

theorem publicBatchWrapper89_f43 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 8602) = a (.virt 17806) := by
  have c43 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c43

theorem publicBatchWrapper89_f44 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 8802) = a (.virt 17806) := by
  have c44 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c44

theorem publicBatchWrapper89_f45 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 9002) = a (.virt 17806) := by
  have c45 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c45

theorem publicBatchWrapper89_f46 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 9202) = a (.virt 17806) := by
  have c46 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c46

theorem publicBatchWrapper89_f47 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 9402) = a (.virt 17806) := by
  have c47 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c47

theorem publicBatchWrapper89_f48 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 9602) = a (.virt 17806) := by
  have c48 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c48

theorem publicBatchWrapper89_f49 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 9802) = a (.virt 17806) := by
  have c49 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c49

theorem publicBatchWrapper89_f50 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 10002) = a (.virt 17806) := by
  have c50 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c50

theorem publicBatchWrapper89_f51 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 10202) = a (.virt 17806) := by
  have c51 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c51

theorem publicBatchWrapper89_f52 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 10402) = a (.virt 17806) := by
  have c52 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c52

theorem publicBatchWrapper89_f53 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 10602) = a (.virt 17806) := by
  have c53 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c53

theorem publicBatchWrapper89_f54 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 10802) = a (.virt 17806) := by
  have c54 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c54

theorem publicBatchWrapper89_f55 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 11002) = a (.virt 17806) := by
  have c55 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c55

theorem publicBatchWrapper89_f56 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 11202) = a (.virt 17806) := by
  have c56 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c56

theorem publicBatchWrapper89_f57 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 11402) = a (.virt 17806) := by
  have c57 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c57

theorem publicBatchWrapper89_f58 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 11602) = a (.virt 17806) := by
  have c58 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c58

theorem publicBatchWrapper89_f59 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 11802) = a (.virt 17806) := by
  have c59 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c59

theorem publicBatchWrapper89_f60 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 12002) = a (.virt 17806) := by
  have c60 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c60

theorem publicBatchWrapper89_f61 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 12202) = a (.virt 17806) := by
  have c61 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c61

theorem publicBatchWrapper89_f62 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 12402) = a (.virt 17806) := by
  have c62 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c62

theorem publicBatchWrapper89_f63 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 12602) = a (.virt 17806) := by
  have c63 := (publicBatchWrapper89_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  exact c63

theorem publicBatchWrapper89_f64 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 12802) = a (.virt 17806) := by
  have c64 := (publicBatchWrapper89_copies2 a h).1
  exact c64

theorem publicBatchWrapper89_f65 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 13002) = a (.virt 17806) := by
  have c65 := (publicBatchWrapper89_copies2 a h).2.1
  exact c65

theorem publicBatchWrapper89_f66 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 13202) = a (.virt 17806) := by
  have c66 := (publicBatchWrapper89_copies2 a h).2.2.1
  exact c66

theorem publicBatchWrapper89_f67 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 13402) = a (.virt 17806) := by
  have c67 := (publicBatchWrapper89_copies2 a h).2.2.2.1
  exact c67

theorem publicBatchWrapper89_f68 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 13602) = a (.virt 17806) := by
  have c68 := (publicBatchWrapper89_copies2 a h).2.2.2.2.1
  exact c68

theorem publicBatchWrapper89_f69 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 13802) = a (.virt 17806) := by
  have c69 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.1
  exact c69

theorem publicBatchWrapper89_f70 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 14002) = a (.virt 17806) := by
  have c70 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.1
  exact c70

theorem publicBatchWrapper89_f71 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 14202) = a (.virt 17806) := by
  have c71 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.1
  exact c71

theorem publicBatchWrapper89_f72 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 14402) = a (.virt 17806) := by
  have c72 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.1
  exact c72

theorem publicBatchWrapper89_f73 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 14602) = a (.virt 17806) := by
  have c73 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.1
  exact c73

theorem publicBatchWrapper89_f74 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 14802) = a (.virt 17806) := by
  have c74 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c74

theorem publicBatchWrapper89_f75 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 15002) = a (.virt 17806) := by
  have c75 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c75

theorem publicBatchWrapper89_f76 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 15202) = a (.virt 17806) := by
  have c76 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c76

theorem publicBatchWrapper89_f77 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 15402) = a (.virt 17806) := by
  have c77 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c77

theorem publicBatchWrapper89_f78 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 15602) = a (.virt 17806) := by
  have c78 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c78

theorem publicBatchWrapper89_f79 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 15802) = a (.virt 17806) := by
  have c79 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c79

theorem publicBatchWrapper89_f80 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 16002) = a (.virt 17806) := by
  have c80 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c80

theorem publicBatchWrapper89_f81 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 16202) = a (.virt 17806) := by
  have c81 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c81

theorem publicBatchWrapper89_f82 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 16402) = a (.virt 17806) := by
  have c82 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c82

theorem publicBatchWrapper89_f83 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 16602) = a (.virt 17806) := by
  have c83 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c83

theorem publicBatchWrapper89_f84 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 16802) = a (.virt 17806) := by
  have c84 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c84

theorem publicBatchWrapper89_f85 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 17002) = a (.virt 17806) := by
  have c85 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c85

theorem publicBatchWrapper89_f86 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 17202) = a (.virt 17806) := by
  have c86 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c86

theorem publicBatchWrapper89_f87 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 17402) = a (.virt 17806) := by
  have c87 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c87

theorem publicBatchWrapper89_f88 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.virt 17602) = a (.virt 17806) := by
  have c88 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c88

theorem publicBatchWrapper89_f89 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 200)) (a (.virt 17807)) (a (.virt 17808)) := by
  have c89 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (publicBatchWrapper89_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (publicBatchWrapper89_copies3 a h).1
  have c97 := (publicBatchWrapper89_copies3 a h).2.1
  have c98 := (publicBatchWrapper89_copies3 a h).2.2.1
  have c99 := (publicBatchWrapper89_copies3 a h).2.2.2.1
  have c100 := (publicBatchWrapper89_copies3 a h).2.2.2.2.1
  have c101 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.1
  have c102 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_4 := arithEq_of_rows h (row := 1) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_4
  simp only [← c92, ← c93, ← c94] at e_1_4
  have e_1_5 := arithEq_of_rows h (row := 1) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_5
  simp only [← c95, ← c96, ← c97] at e_1_5
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c89, k3, ← c90, k3, ← c91] at e_2_0
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c98, ← c99, k3, ← c100] at e_2_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_4
    linear_combination c101.trans k2 - hc
  · have hc := e_2_1
    simp only [e_2_0, e_1_5] at hc
    linear_combination c102.trans k2 - hc

theorem publicBatchWrapper89_f90 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 200)) (a (.virt 17809)) (a (.virt 17810)) := by
  have c103 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_6 := arithEq_of_rows h (row := 1) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_6
  simp only [← c106, ← c107, ← c108] at e_1_6
  have e_1_7 := arithEq_of_rows h (row := 1) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_7
  simp only [← c109, ← c110, ← c111] at e_1_7
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c103, k3, ← c104, k3, ← c105] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c112, ← c113, k3, ← c114] at e_2_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_6
    linear_combination c115.trans k2 - hc
  · have hc := e_2_3
    simp only [e_2_2, e_1_7] at hc
    linear_combination c116.trans k2 - hc

theorem publicBatchWrapper89_f91 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5)) (a (.virt 200)) (a (.virt 17811)) (a (.virt 17812)) := by
  have c117 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c127 := (publicBatchWrapper89_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (publicBatchWrapper89_copies4 a h).1
  have c129 := (publicBatchWrapper89_copies4 a h).2.1
  have c130 := (publicBatchWrapper89_copies4 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c120, ← c121, ← c122] at e_1_8
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c123, ← c124, ← c125] at e_1_9
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c117, k3, ← c118, k3, ← c119] at e_2_4
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c126, ← c127, k3, ← c128] at e_2_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_8
    linear_combination c129.trans k2 - hc
  · have hc := e_2_5
    simp only [e_2_4, e_1_9] at hc
    linear_combination c130.trans k2 - hc

theorem publicBatchWrapper89_f92 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6)) (a (.virt 200)) (a (.virt 17813)) (a (.virt 17814)) := by
  have c131 := (publicBatchWrapper89_copies4 a h).2.2.2.1
  have c132 := (publicBatchWrapper89_copies4 a h).2.2.2.2.1
  have c133 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.1
  have c134 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.1
  have c135 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c144 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c134, ← c135, ← c136] at e_1_10
  have e_1_11 := arithEq_of_rows h (row := 1) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_11
  simp only [← c137, ← c138, ← c139] at e_1_11
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c131, k3, ← c132, k3, ← c133] at e_2_6
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c140, ← c141, k3, ← c142] at e_2_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_10
    linear_combination c143.trans k2 - hc
  · have hc := e_2_7
    simp only [e_2_6, e_1_11] at hc
    linear_combination c144.trans k2 - hc

theorem publicBatchWrapper89_f93 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1 51) = band (a (.virt 17807)) (a (.virt 17809)) := by
  have c145 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c145, ← c146, ← c147] at e_1_12
  have hr := e_1_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f94 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1 55) = band (a (.virt 17811)) (a (.virt 17813)) := by
  have c148 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c148, ← c149, ← c150] at e_1_13
  have hr := e_1_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f95 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1 59) = band (a (.wire 1 51)) (a (.wire 1 55)) := by
  have c151 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c151, ← c152, ← c153] at e_1_14
  have hr := e_1_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f96 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 205)) (a (.virt 200)) (a (.virt 17815)) (a (.virt 17816)) := by
  have c154 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (publicBatchWrapper89_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (publicBatchWrapper89_copies5 a h).1
  have c161 := (publicBatchWrapper89_copies5 a h).2.1
  have c162 := (publicBatchWrapper89_copies5 a h).2.2.1
  have c163 := (publicBatchWrapper89_copies5 a h).2.2.2.1
  have c164 := (publicBatchWrapper89_copies5 a h).2.2.2.2.1
  have c165 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.1
  have c166 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.1
  have c167 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_15 := arithEq_of_rows h (row := 1) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_15
  simp only [← c157, ← c158, ← c159] at e_1_15
  have e_1_16 := arithEq_of_rows h (row := 1) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_16
  simp only [← c160, ← c161, ← c162] at e_1_16
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c154, k3, ← c155, k3, ← c156] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c163, ← c164, k3, ← c165] at e_2_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_15
    linear_combination c166.trans k2 - hc
  · have hc := e_2_9
    simp only [e_2_8, e_1_16] at hc
    linear_combination c167.trans k2 - hc

theorem publicBatchWrapper89_f97 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 206)) (a (.virt 200)) (a (.virt 17817)) (a (.virt 17818)) := by
  have c168 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_17 := arithEq_of_rows h (row := 1) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_17
  simp only [← c171, ← c172, ← c173] at e_1_17
  have e_1_18 := arithEq_of_rows h (row := 1) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_18
  simp only [← c174, ← c175, ← c176] at e_1_18
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c168, k3, ← c169, k3, ← c170] at e_2_10
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c177, ← c178, k3, ← c179] at e_2_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_17
    linear_combination c180.trans k2 - hc
  · have hc := e_2_11
    simp only [e_2_10, e_1_18] at hc
    linear_combination c181.trans k2 - hc

theorem publicBatchWrapper89_f98 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 207)) (a (.virt 200)) (a (.virt 17819)) (a (.virt 17820)) := by
  have c182 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (publicBatchWrapper89_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (publicBatchWrapper89_copies6 a h).1
  have c193 := (publicBatchWrapper89_copies6 a h).2.1
  have c194 := (publicBatchWrapper89_copies6 a h).2.2.1
  have c195 := (publicBatchWrapper89_copies6 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1_19 := arithEq_of_rows h (row := 1) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_19
  simp only [← c185, ← c186, ← c187] at e_1_19
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c182, k3, ← c183, k3, ← c184] at e_2_12
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c191, ← c192, k3, ← c193] at e_2_13
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c188, ← c189, ← c190] at e_3_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_1_19
    linear_combination c194.trans k2 - hc
  · have hc := e_2_13
    simp only [e_2_12, e_3_0] at hc
    linear_combination c195.trans k2 - hc

theorem publicBatchWrapper89_f99 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 208)) (a (.virt 200)) (a (.virt 17821)) (a (.virt 17822)) := by
  have c196 := (publicBatchWrapper89_copies6 a h).2.2.2.2.1
  have c197 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.1
  have c198 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.1
  have c199 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c196, k3, ← c197, k3, ← c198] at e_2_14
  have e_2_15 := arithEq_of_rows h (row := 2) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c205, ← c206, k3, ← c207] at e_2_15
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c199, ← c200, ← c201] at e_3_1
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c202, ← c203, ← c204] at e_3_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_1
    linear_combination c208.trans k2 - hc
  · have hc := e_2_15
    simp only [e_2_14, e_3_2] at hc
    linear_combination c209.trans k2 - hc

theorem publicBatchWrapper89_f100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 3 15) = band (a (.virt 17815)) (a (.virt 17817)) := by
  have c210 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c210, ← c211, ← c212] at e_3_3
  have hr := e_3_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 3 19) = band (a (.virt 17819)) (a (.virt 17821)) := by
  have c213 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c213, ← c214, ← c215] at e_3_4
  have hr := e_3_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 3 23) = band (a (.wire 3 15)) (a (.wire 3 19)) := by
  have c216 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c216, ← c217, ← c218] at e_3_5
  have hr := e_3_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 405)) (a (.virt 200)) (a (.virt 17823)) (a (.virt 17824)) := by
  have c219 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (publicBatchWrapper89_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (publicBatchWrapper89_copies7 a h).1
  have c225 := (publicBatchWrapper89_copies7 a h).2.1
  have c226 := (publicBatchWrapper89_copies7 a h).2.2.1
  have c227 := (publicBatchWrapper89_copies7 a h).2.2.2.1
  have c228 := (publicBatchWrapper89_copies7 a h).2.2.2.2.1
  have c229 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.1
  have c230 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.1
  have c231 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2_16 := arithEq_of_rows h (row := 2) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c219, k3, ← c220, k3, ← c221] at e_2_16
  have e_2_17 := arithEq_of_rows h (row := 2) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c228, ← c229, k3, ← c230] at e_2_17
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c222, ← c223, ← c224] at e_3_6
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c225, ← c226, ← c227] at e_3_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_6
    linear_combination c231.trans k2 - hc
  · have hc := e_2_17
    simp only [e_2_16, e_3_7] at hc
    linear_combination c232.trans k2 - hc

theorem publicBatchWrapper89_f104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 406)) (a (.virt 200)) (a (.virt 17825)) (a (.virt 17826)) := by
  have c233 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2_18 := arithEq_of_rows h (row := 2) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c233, k3, ← c234, k3, ← c235] at e_2_18
  have e_2_19 := arithEq_of_rows h (row := 2) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c242, ← c243, k3, ← c244] at e_2_19
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c236, ← c237, ← c238] at e_3_8
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c239, ← c240, ← c241] at e_3_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_8
    linear_combination c245.trans k2 - hc
  · have hc := e_2_19
    simp only [e_2_18, e_3_9] at hc
    linear_combination c246.trans k2 - hc

theorem publicBatchWrapper89_f105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 407)) (a (.virt 200)) (a (.virt 17827)) (a (.virt 17828)) := by
  have c247 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (publicBatchWrapper89_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (publicBatchWrapper89_copies8 a h).1
  have c257 := (publicBatchWrapper89_copies8 a h).2.1
  have c258 := (publicBatchWrapper89_copies8 a h).2.2.1
  have c259 := (publicBatchWrapper89_copies8 a h).2.2.2.1
  have c260 := (publicBatchWrapper89_copies8 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c250, ← c251, ← c252] at e_3_10
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c253, ← c254, ← c255] at e_3_11
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c247, k3, ← c248, k3, ← c249] at e_4_0
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c256, ← c257, k3, ← c258] at e_4_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_10
    linear_combination c259.trans k2 - hc
  · have hc := e_4_1
    simp only [e_4_0, e_3_11] at hc
    linear_combination c260.trans k2 - hc

theorem publicBatchWrapper89_f106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 408)) (a (.virt 200)) (a (.virt 17829)) (a (.virt 17830)) := by
  have c261 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.1
  have c262 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.1
  have c263 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c264, ← c265, ← c266] at e_3_12
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c267, ← c268, ← c269] at e_3_13
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c261, k3, ← c262, k3, ← c263] at e_4_2
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c270, ← c271, k3, ← c272] at e_4_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_12
    linear_combination c273.trans k2 - hc
  · have hc := e_4_3
    simp only [e_4_2, e_3_13] at hc
    linear_combination c274.trans k2 - hc

theorem publicBatchWrapper89_f107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 3 59) = band (a (.virt 17823)) (a (.virt 17825)) := by
  have c275 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c275, ← c276, ← c277] at e_3_14
  have hr := e_3_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 3 63) = band (a (.virt 17827)) (a (.virt 17829)) := by
  have c278 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_15 := arithEq_of_rows h (row := 3) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c278, ← c279, ← c280] at e_3_15
  have hr := e_3_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 3 67) = band (a (.wire 3 59)) (a (.wire 3 63)) := by
  have c281 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_16 := arithEq_of_rows h (row := 3) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c281, ← c282, ← c283] at e_3_16
  have hr := e_3_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 605)) (a (.virt 200)) (a (.virt 17831)) (a (.virt 17832)) := by
  have c284 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (publicBatchWrapper89_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (publicBatchWrapper89_copies9 a h).1
  have c289 := (publicBatchWrapper89_copies9 a h).2.1
  have c290 := (publicBatchWrapper89_copies9 a h).2.2.1
  have c291 := (publicBatchWrapper89_copies9 a h).2.2.2.1
  have c292 := (publicBatchWrapper89_copies9 a h).2.2.2.2.1
  have c293 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.1
  have c294 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.1
  have c295 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_3_17 := arithEq_of_rows h (row := 3) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c287, ← c288, ← c289] at e_3_17
  have e_3_18 := arithEq_of_rows h (row := 3) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c290, ← c291, ← c292] at e_3_18
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c284, k3, ← c285, k3, ← c286] at e_4_4
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c293, ← c294, k3, ← c295] at e_4_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_17
    linear_combination c296.trans k2 - hc
  · have hc := e_4_5
    simp only [e_4_4, e_3_18] at hc
    linear_combination c297.trans k2 - hc

theorem publicBatchWrapper89_f111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 606)) (a (.virt 200)) (a (.virt 17833)) (a (.virt 17834)) := by
  have c298 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_3_19 := arithEq_of_rows h (row := 3) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c301, ← c302, ← c303] at e_3_19
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c298, k3, ← c299, k3, ← c300] at e_4_6
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c307, ← c308, k3, ← c309] at e_4_7
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c304, ← c305, ← c306] at e_5_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_3_19
    linear_combination c310.trans k2 - hc
  · have hc := e_4_7
    simp only [e_4_6, e_5_0] at hc
    linear_combination c311.trans k2 - hc

theorem publicBatchWrapper89_f112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 607)) (a (.virt 200)) (a (.virt 17835)) (a (.virt 17836)) := by
  have c312 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (publicBatchWrapper89_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (publicBatchWrapper89_copies10 a h).1
  have c321 := (publicBatchWrapper89_copies10 a h).2.1
  have c322 := (publicBatchWrapper89_copies10 a h).2.2.1
  have c323 := (publicBatchWrapper89_copies10 a h).2.2.2.1
  have c324 := (publicBatchWrapper89_copies10 a h).2.2.2.2.1
  have c325 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c312, k3, ← c313, k3, ← c314] at e_4_8
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c321, ← c322, k3, ← c323] at e_4_9
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c315, ← c316, ← c317] at e_5_1
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c318, ← c319, ← c320] at e_5_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_1
    linear_combination c324.trans k2 - hc
  · have hc := e_4_9
    simp only [e_4_8, e_5_2] at hc
    linear_combination c325.trans k2 - hc

theorem publicBatchWrapper89_f113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 608)) (a (.virt 200)) (a (.virt 17837)) (a (.virt 17838)) := by
  have c326 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.1
  have c327 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c326, k3, ← c327, k3, ← c328] at e_4_10
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c335, ← c336, k3, ← c337] at e_4_11
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c329, ← c330, ← c331] at e_5_3
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c332, ← c333, ← c334] at e_5_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_3
    linear_combination c338.trans k2 - hc
  · have hc := e_4_11
    simp only [e_4_10, e_5_4] at hc
    linear_combination c339.trans k2 - hc

theorem publicBatchWrapper89_f114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 5 23) = band (a (.virt 17831)) (a (.virt 17833)) := by
  have c340 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c340, ← c341, ← c342] at e_5_5
  have hr := e_5_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 5 27) = band (a (.virt 17835)) (a (.virt 17837)) := by
  have c343 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c343, ← c344, ← c345] at e_5_6
  have hr := e_5_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 5 31) = band (a (.wire 5 23)) (a (.wire 5 27)) := by
  have c346 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_7 := arithEq_of_rows h (row := 5) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c346, ← c347, ← c348] at e_5_7
  have hr := e_5_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 805)) (a (.virt 200)) (a (.virt 17839)) (a (.virt 17840)) := by
  have c349 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (publicBatchWrapper89_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (publicBatchWrapper89_copies11 a h).1
  have c353 := (publicBatchWrapper89_copies11 a h).2.1
  have c354 := (publicBatchWrapper89_copies11 a h).2.2.1
  have c355 := (publicBatchWrapper89_copies11 a h).2.2.2.1
  have c356 := (publicBatchWrapper89_copies11 a h).2.2.2.2.1
  have c357 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.1
  have c358 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.1
  have c359 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have c362 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c349, k3, ← c350, k3, ← c351] at e_4_12
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c358, ← c359, k3, ← c360] at e_4_13
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c352, ← c353, ← c354] at e_5_8
  have e_5_9 := arithEq_of_rows h (row := 5) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c355, ← c356, ← c357] at e_5_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_8
    linear_combination c361.trans k2 - hc
  · have hc := e_4_13
    simp only [e_4_12, e_5_9] at hc
    linear_combination c362.trans k2 - hc

theorem publicBatchWrapper89_f118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 806)) (a (.virt 200)) (a (.virt 17841)) (a (.virt 17842)) := by
  have c363 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c363, k3, ← c364, k3, ← c365] at e_4_14
  have e_4_15 := arithEq_of_rows h (row := 4) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_15
  simp only [← c372, ← c373, k3, ← c374] at e_4_15
  have e_5_10 := arithEq_of_rows h (row := 5) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c366, ← c367, ← c368] at e_5_10
  have e_5_11 := arithEq_of_rows h (row := 5) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c369, ← c370, ← c371] at e_5_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_10
    linear_combination c375.trans k2 - hc
  · have hc := e_4_15
    simp only [e_4_14, e_5_11] at hc
    linear_combination c376.trans k2 - hc

theorem publicBatchWrapper89_f119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 807)) (a (.virt 200)) (a (.virt 17843)) (a (.virt 17844)) := by
  have c377 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (publicBatchWrapper89_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (publicBatchWrapper89_copies12 a h).1
  have c385 := (publicBatchWrapper89_copies12 a h).2.1
  have c386 := (publicBatchWrapper89_copies12 a h).2.2.1
  have c387 := (publicBatchWrapper89_copies12 a h).2.2.2.1
  have c388 := (publicBatchWrapper89_copies12 a h).2.2.2.2.1
  have c389 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.1
  have c390 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_4_16 := arithEq_of_rows h (row := 4) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_16
  simp only [← c377, k3, ← c378, k3, ← c379] at e_4_16
  have e_4_17 := arithEq_of_rows h (row := 4) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_17
  simp only [← c386, ← c387, k3, ← c388] at e_4_17
  have e_5_12 := arithEq_of_rows h (row := 5) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c380, ← c381, ← c382] at e_5_12
  have e_5_13 := arithEq_of_rows h (row := 5) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c383, ← c384, ← c385] at e_5_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_12
    linear_combination c389.trans k2 - hc
  · have hc := e_4_17
    simp only [e_4_16, e_5_13] at hc
    linear_combination c390.trans k2 - hc

theorem publicBatchWrapper89_f120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 808)) (a (.virt 200)) (a (.virt 17845)) (a (.virt 17846)) := by
  have c391 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_4_18 := arithEq_of_rows h (row := 4) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_18
  simp only [← c391, k3, ← c392, k3, ← c393] at e_4_18
  have e_4_19 := arithEq_of_rows h (row := 4) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_19
  simp only [← c400, ← c401, k3, ← c402] at e_4_19
  have e_5_14 := arithEq_of_rows h (row := 5) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c394, ← c395, ← c396] at e_5_14
  have e_5_15 := arithEq_of_rows h (row := 5) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_15
  simp only [← c397, ← c398, ← c399] at e_5_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_14
    linear_combination c403.trans k2 - hc
  · have hc := e_4_19
    simp only [e_4_18, e_5_15] at hc
    linear_combination c404.trans k2 - hc

theorem publicBatchWrapper89_f121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 5 67) = band (a (.virt 17839)) (a (.virt 17841)) := by
  have c405 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_16 := arithEq_of_rows h (row := 5) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_16
  simp only [← c405, ← c406, ← c407] at e_5_16
  have hr := e_5_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 5 71) = band (a (.virt 17843)) (a (.virt 17845)) := by
  have c408 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_17 := arithEq_of_rows h (row := 5) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_17
  simp only [← c408, ← c409, ← c410] at e_5_17
  have hr := e_5_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 5 75) = band (a (.wire 5 67)) (a (.wire 5 71)) := by
  have c411 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_18 := arithEq_of_rows h (row := 5) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_18
  simp only [← c411, ← c412, ← c413] at e_5_18
  have hr := e_5_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1005)) (a (.virt 200)) (a (.virt 17847)) (a (.virt 17848)) := by
  have c414 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (publicBatchWrapper89_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (publicBatchWrapper89_copies13 a h).1
  have c417 := (publicBatchWrapper89_copies13 a h).2.1
  have c418 := (publicBatchWrapper89_copies13 a h).2.2.1
  have c419 := (publicBatchWrapper89_copies13 a h).2.2.2.1
  have c420 := (publicBatchWrapper89_copies13 a h).2.2.2.2.1
  have c421 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.1
  have c422 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.1
  have c423 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_5_19 := arithEq_of_rows h (row := 5) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_19
  simp only [← c417, ← c418, ← c419] at e_5_19
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c414, k3, ← c415, k3, ← c416] at e_6_0
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c423, ← c424, k3, ← c425] at e_6_1
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c420, ← c421, ← c422] at e_7_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_5_19
    linear_combination c426.trans k2 - hc
  · have hc := e_6_1
    simp only [e_6_0, e_7_0] at hc
    linear_combination c427.trans k2 - hc

theorem publicBatchWrapper89_f125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1006)) (a (.virt 200)) (a (.virt 17849)) (a (.virt 17850)) := by
  have c428 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c428, k3, ← c429, k3, ← c430] at e_6_2
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c437, ← c438, k3, ← c439] at e_6_3
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c431, ← c432, ← c433] at e_7_1
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c434, ← c435, ← c436] at e_7_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_1
    linear_combination c440.trans k2 - hc
  · have hc := e_6_3
    simp only [e_6_2, e_7_2] at hc
    linear_combination c441.trans k2 - hc

theorem publicBatchWrapper89_f126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1007)) (a (.virt 200)) (a (.virt 17851)) (a (.virt 17852)) := by
  have c442 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (publicBatchWrapper89_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (publicBatchWrapper89_copies14 a h).1
  have c449 := (publicBatchWrapper89_copies14 a h).2.1
  have c450 := (publicBatchWrapper89_copies14 a h).2.2.1
  have c451 := (publicBatchWrapper89_copies14 a h).2.2.2.1
  have c452 := (publicBatchWrapper89_copies14 a h).2.2.2.2.1
  have c453 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.1
  have c454 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.1
  have c455 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c442, k3, ← c443, k3, ← c444] at e_6_4
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c451, ← c452, k3, ← c453] at e_6_5
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c445, ← c446, ← c447] at e_7_3
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c448, ← c449, ← c450] at e_7_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_3
    linear_combination c454.trans k2 - hc
  · have hc := e_6_5
    simp only [e_6_4, e_7_4] at hc
    linear_combination c455.trans k2 - hc

theorem publicBatchWrapper89_f127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1008)) (a (.virt 200)) (a (.virt 17853)) (a (.virt 17854)) := by
  have c456 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c456, k3, ← c457, k3, ← c458] at e_6_6
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c465, ← c466, k3, ← c467] at e_6_7
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c459, ← c460, ← c461] at e_7_5
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c462, ← c463, ← c464] at e_7_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_5
    linear_combination c468.trans k2 - hc
  · have hc := e_6_7
    simp only [e_6_6, e_7_6] at hc
    linear_combination c469.trans k2 - hc

theorem publicBatchWrapper89_f128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 7 31) = band (a (.virt 17847)) (a (.virt 17849)) := by
  have c470 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c470, ← c471, ← c472] at e_7_7
  have hr := e_7_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 7 35) = band (a (.virt 17851)) (a (.virt 17853)) := by
  have c473 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c473, ← c474, ← c475] at e_7_8
  have hr := e_7_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 7 39) = band (a (.wire 7 31)) (a (.wire 7 35)) := by
  have c476 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c476, ← c477, ← c478] at e_7_9
  have hr := e_7_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1205)) (a (.virt 200)) (a (.virt 17855)) (a (.virt 17856)) := by
  have c479 := (publicBatchWrapper89_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (publicBatchWrapper89_copies15 a h).1
  have c481 := (publicBatchWrapper89_copies15 a h).2.1
  have c482 := (publicBatchWrapper89_copies15 a h).2.2.1
  have c483 := (publicBatchWrapper89_copies15 a h).2.2.2.1
  have c484 := (publicBatchWrapper89_copies15 a h).2.2.2.2.1
  have c485 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.1
  have c486 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.1
  have c487 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c479, k3, ← c480, k3, ← c481] at e_6_8
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c488, ← c489, k3, ← c490] at e_6_9
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c482, ← c483, ← c484] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c485, ← c486, ← c487] at e_7_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_10
    linear_combination c491.trans k2 - hc
  · have hc := e_6_9
    simp only [e_6_8, e_7_11] at hc
    linear_combination c492.trans k2 - hc

theorem publicBatchWrapper89_f132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1206)) (a (.virt 200)) (a (.virt 17857)) (a (.virt 17858)) := by
  have c493 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c493, k3, ← c494, k3, ← c495] at e_6_10
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c502, ← c503, k3, ← c504] at e_6_11
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c496, ← c497, ← c498] at e_7_12
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c499, ← c500, ← c501] at e_7_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_12
    linear_combination c505.trans k2 - hc
  · have hc := e_6_11
    simp only [e_6_10, e_7_13] at hc
    linear_combination c506.trans k2 - hc

theorem publicBatchWrapper89_f133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1207)) (a (.virt 200)) (a (.virt 17859)) (a (.virt 17860)) := by
  have c507 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (publicBatchWrapper89_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (publicBatchWrapper89_copies16 a h).1
  have c513 := (publicBatchWrapper89_copies16 a h).2.1
  have c514 := (publicBatchWrapper89_copies16 a h).2.2.1
  have c515 := (publicBatchWrapper89_copies16 a h).2.2.2.1
  have c516 := (publicBatchWrapper89_copies16 a h).2.2.2.2.1
  have c517 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.1
  have c518 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.1
  have c519 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c507, k3, ← c508, k3, ← c509] at e_6_12
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c516, ← c517, k3, ← c518] at e_6_13
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c510, ← c511, ← c512] at e_7_14
  have e_7_15 := arithEq_of_rows h (row := 7) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_15
  simp only [← c513, ← c514, ← c515] at e_7_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_14
    linear_combination c519.trans k2 - hc
  · have hc := e_6_13
    simp only [e_6_12, e_7_15] at hc
    linear_combination c520.trans k2 - hc

theorem publicBatchWrapper89_f134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1208)) (a (.virt 200)) (a (.virt 17861)) (a (.virt 17862)) := by
  have c521 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c532 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c521, k3, ← c522, k3, ← c523] at e_6_14
  have e_6_15 := arithEq_of_rows h (row := 6) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_15
  simp only [← c530, ← c531, k3, ← c532] at e_6_15
  have e_7_16 := arithEq_of_rows h (row := 7) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_16
  simp only [← c524, ← c525, ← c526] at e_7_16
  have e_7_17 := arithEq_of_rows h (row := 7) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_17
  simp only [← c527, ← c528, ← c529] at e_7_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_7_16
    linear_combination c533.trans k2 - hc
  · have hc := e_6_15
    simp only [e_6_14, e_7_17] at hc
    linear_combination c534.trans k2 - hc

theorem publicBatchWrapper89_f135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 7 75) = band (a (.virt 17855)) (a (.virt 17857)) := by
  have c535 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_18 := arithEq_of_rows h (row := 7) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_18
  simp only [← c535, ← c536, ← c537] at e_7_18
  have hr := e_7_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 7 79) = band (a (.virt 17859)) (a (.virt 17861)) := by
  have c538 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_19 := arithEq_of_rows h (row := 7) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_19
  simp only [← c538, ← c539, ← c540] at e_7_19
  have hr := e_7_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 8 3) = band (a (.wire 7 75)) (a (.wire 7 79)) := by
  have c541 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (publicBatchWrapper89_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c541, ← c542, ← c543] at e_8_0
  have hr := e_8_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1405)) (a (.virt 200)) (a (.virt 17863)) (a (.virt 17864)) := by
  have c544 := (publicBatchWrapper89_copies17 a h).1
  have c545 := (publicBatchWrapper89_copies17 a h).2.1
  have c546 := (publicBatchWrapper89_copies17 a h).2.2.1
  have c547 := (publicBatchWrapper89_copies17 a h).2.2.2.1
  have c548 := (publicBatchWrapper89_copies17 a h).2.2.2.2.1
  have c549 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.1
  have c550 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.1
  have c551 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_16 := arithEq_of_rows h (row := 6) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_16
  simp only [← c544, k3, ← c545, k3, ← c546] at e_6_16
  have e_6_17 := arithEq_of_rows h (row := 6) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_17
  simp only [← c553, ← c554, k3, ← c555] at e_6_17
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c547, ← c548, ← c549] at e_8_1
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c550, ← c551, ← c552] at e_8_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_1
    linear_combination c556.trans k2 - hc
  · have hc := e_6_17
    simp only [e_6_16, e_8_2] at hc
    linear_combination c557.trans k2 - hc

theorem publicBatchWrapper89_f139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1406)) (a (.virt 200)) (a (.virt 17865)) (a (.virt 17866)) := by
  have c558 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_6_18 := arithEq_of_rows h (row := 6) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_18
  simp only [← c558, k3, ← c559, k3, ← c560] at e_6_18
  have e_6_19 := arithEq_of_rows h (row := 6) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_19
  simp only [← c567, ← c568, k3, ← c569] at e_6_19
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c561, ← c562, ← c563] at e_8_3
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c564, ← c565, ← c566] at e_8_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_3
    linear_combination c570.trans k2 - hc
  · have hc := e_6_19
    simp only [e_6_18, e_8_4] at hc
    linear_combination c571.trans k2 - hc

theorem publicBatchWrapper89_f140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1407)) (a (.virt 200)) (a (.virt 17867)) (a (.virt 17868)) := by
  have c572 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (publicBatchWrapper89_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (publicBatchWrapper89_copies18 a h).1
  have c577 := (publicBatchWrapper89_copies18 a h).2.1
  have c578 := (publicBatchWrapper89_copies18 a h).2.2.1
  have c579 := (publicBatchWrapper89_copies18 a h).2.2.2.1
  have c580 := (publicBatchWrapper89_copies18 a h).2.2.2.2.1
  have c581 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.1
  have c582 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.1
  have c583 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c575, ← c576, ← c577] at e_8_5
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c578, ← c579, ← c580] at e_8_6
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c572, k3, ← c573, k3, ← c574] at e_9_0
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c581, ← c582, k3, ← c583] at e_9_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_5
    linear_combination c584.trans k2 - hc
  · have hc := e_9_1
    simp only [e_9_0, e_8_6] at hc
    linear_combination c585.trans k2 - hc

theorem publicBatchWrapper89_f141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1408)) (a (.virt 200)) (a (.virt 17869)) (a (.virt 17870)) := by
  have c586 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c589, ← c590, ← c591] at e_8_7
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c592, ← c593, ← c594] at e_8_8
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c586, k3, ← c587, k3, ← c588] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c595, ← c596, k3, ← c597] at e_9_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_7
    linear_combination c598.trans k2 - hc
  · have hc := e_9_3
    simp only [e_9_2, e_8_8] at hc
    linear_combination c599.trans k2 - hc

theorem publicBatchWrapper89_f142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 8 39) = band (a (.virt 17863)) (a (.virt 17865)) := by
  have c600 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c600, ← c601, ← c602] at e_8_9
  have hr := e_8_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 8 43) = band (a (.virt 17867)) (a (.virt 17869)) := by
  have c603 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c603, ← c604, ← c605] at e_8_10
  have hr := e_8_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 8 47) = band (a (.wire 8 39)) (a (.wire 8 43)) := by
  have c606 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (publicBatchWrapper89_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (publicBatchWrapper89_copies19 a h).1
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c606, ← c607, ← c608] at e_8_11
  have hr := e_8_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1605)) (a (.virt 200)) (a (.virt 17871)) (a (.virt 17872)) := by
  have c609 := (publicBatchWrapper89_copies19 a h).2.1
  have c610 := (publicBatchWrapper89_copies19 a h).2.2.1
  have c611 := (publicBatchWrapper89_copies19 a h).2.2.2.1
  have c612 := (publicBatchWrapper89_copies19 a h).2.2.2.2.1
  have c613 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.1
  have c614 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.1
  have c615 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c612, ← c613, ← c614] at e_8_12
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c615, ← c616, ← c617] at e_8_13
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c609, k3, ← c610, k3, ← c611] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c618, ← c619, k3, ← c620] at e_9_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_12
    linear_combination c621.trans k2 - hc
  · have hc := e_9_5
    simp only [e_9_4, e_8_13] at hc
    linear_combination c622.trans k2 - hc

theorem publicBatchWrapper89_f146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1606)) (a (.virt 200)) (a (.virt 17873)) (a (.virt 17874)) := by
  have c623 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c626, ← c627, ← c628] at e_8_14
  have e_8_15 := arithEq_of_rows h (row := 8) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_15
  simp only [← c629, ← c630, ← c631] at e_8_15
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c623, k3, ← c624, k3, ← c625] at e_9_6
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c632, ← c633, k3, ← c634] at e_9_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_14
    linear_combination c635.trans k2 - hc
  · have hc := e_9_7
    simp only [e_9_6, e_8_15] at hc
    linear_combination c636.trans k2 - hc

theorem publicBatchWrapper89_f147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1607)) (a (.virt 200)) (a (.virt 17875)) (a (.virt 17876)) := by
  have c637 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (publicBatchWrapper89_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (publicBatchWrapper89_copies20 a h).1
  have c641 := (publicBatchWrapper89_copies20 a h).2.1
  have c642 := (publicBatchWrapper89_copies20 a h).2.2.1
  have c643 := (publicBatchWrapper89_copies20 a h).2.2.2.1
  have c644 := (publicBatchWrapper89_copies20 a h).2.2.2.2.1
  have c645 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.1
  have c646 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.1
  have c647 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_8_16 := arithEq_of_rows h (row := 8) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_16
  simp only [← c640, ← c641, ← c642] at e_8_16
  have e_8_17 := arithEq_of_rows h (row := 8) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_17
  simp only [← c643, ← c644, ← c645] at e_8_17
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c637, k3, ← c638, k3, ← c639] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c646, ← c647, k3, ← c648] at e_9_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_16
    linear_combination c649.trans k2 - hc
  · have hc := e_9_9
    simp only [e_9_8, e_8_17] at hc
    linear_combination c650.trans k2 - hc

theorem publicBatchWrapper89_f148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1608)) (a (.virt 200)) (a (.virt 17877)) (a (.virt 17878)) := by
  have c651 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_8_18 := arithEq_of_rows h (row := 8) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_18
  simp only [← c654, ← c655, ← c656] at e_8_18
  have e_8_19 := arithEq_of_rows h (row := 8) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_19
  simp only [← c657, ← c658, ← c659] at e_8_19
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c651, k3, ← c652, k3, ← c653] at e_9_10
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c660, ← c661, k3, ← c662] at e_9_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_8_18
    linear_combination c663.trans k2 - hc
  · have hc := e_9_11
    simp only [e_9_10, e_8_19] at hc
    linear_combination c664.trans k2 - hc

theorem publicBatchWrapper89_f149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 10 3) = band (a (.virt 17871)) (a (.virt 17873)) := by
  have c665 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c665, ← c666, ← c667] at e_10_0
  have hr := e_10_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 10 7) = band (a (.virt 17875)) (a (.virt 17877)) := by
  have c668 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c668, ← c669, ← c670] at e_10_1
  have hr := e_10_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 10 11) = band (a (.wire 10 3)) (a (.wire 10 7)) := by
  have c671 := (publicBatchWrapper89_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (publicBatchWrapper89_copies21 a h).1
  have c673 := (publicBatchWrapper89_copies21 a h).2.1
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c671, ← c672, ← c673] at e_10_2
  have hr := e_10_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1805)) (a (.virt 200)) (a (.virt 17879)) (a (.virt 17880)) := by
  have c674 := (publicBatchWrapper89_copies21 a h).2.2.1
  have c675 := (publicBatchWrapper89_copies21 a h).2.2.2.1
  have c676 := (publicBatchWrapper89_copies21 a h).2.2.2.2.1
  have c677 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.1
  have c678 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.1
  have c679 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c674, k3, ← c675, k3, ← c676] at e_9_12
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c683, ← c684, k3, ← c685] at e_9_13
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c677, ← c678, ← c679] at e_10_3
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c680, ← c681, ← c682] at e_10_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_3
    linear_combination c686.trans k2 - hc
  · have hc := e_9_13
    simp only [e_9_12, e_10_4] at hc
    linear_combination c687.trans k2 - hc

theorem publicBatchWrapper89_f153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1806)) (a (.virt 200)) (a (.virt 17881)) (a (.virt 17882)) := by
  have c688 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c688, k3, ← c689, k3, ← c690] at e_9_14
  have e_9_15 := arithEq_of_rows h (row := 9) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_15
  simp only [← c697, ← c698, k3, ← c699] at e_9_15
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c691, ← c692, ← c693] at e_10_5
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c694, ← c695, ← c696] at e_10_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_5
    linear_combination c700.trans k2 - hc
  · have hc := e_9_15
    simp only [e_9_14, e_10_6] at hc
    linear_combination c701.trans k2 - hc

theorem publicBatchWrapper89_f154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1807)) (a (.virt 200)) (a (.virt 17883)) (a (.virt 17884)) := by
  have c702 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (publicBatchWrapper89_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (publicBatchWrapper89_copies22 a h).1
  have c705 := (publicBatchWrapper89_copies22 a h).2.1
  have c706 := (publicBatchWrapper89_copies22 a h).2.2.1
  have c707 := (publicBatchWrapper89_copies22 a h).2.2.2.1
  have c708 := (publicBatchWrapper89_copies22 a h).2.2.2.2.1
  have c709 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.1
  have c710 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.1
  have c711 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_9_16 := arithEq_of_rows h (row := 9) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_16
  simp only [← c702, k3, ← c703, k3, ← c704] at e_9_16
  have e_9_17 := arithEq_of_rows h (row := 9) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_17
  simp only [← c711, ← c712, k3, ← c713] at e_9_17
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c705, ← c706, ← c707] at e_10_7
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c708, ← c709, ← c710] at e_10_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_7
    linear_combination c714.trans k2 - hc
  · have hc := e_9_17
    simp only [e_9_16, e_10_8] at hc
    linear_combination c715.trans k2 - hc

theorem publicBatchWrapper89_f155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 1808)) (a (.virt 200)) (a (.virt 17885)) (a (.virt 17886)) := by
  have c716 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_9_18 := arithEq_of_rows h (row := 9) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_18
  simp only [← c716, k3, ← c717, k3, ← c718] at e_9_18
  have e_9_19 := arithEq_of_rows h (row := 9) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_19
  simp only [← c725, ← c726, k3, ← c727] at e_9_19
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c719, ← c720, ← c721] at e_10_9
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c722, ← c723, ← c724] at e_10_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_9
    linear_combination c728.trans k2 - hc
  · have hc := e_9_19
    simp only [e_9_18, e_10_10] at hc
    linear_combination c729.trans k2 - hc

theorem publicBatchWrapper89_f156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 10 47) = band (a (.virt 17879)) (a (.virt 17881)) := by
  have c730 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c730, ← c731, ← c732] at e_10_11
  have hr := e_10_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 10 51) = band (a (.virt 17883)) (a (.virt 17885)) := by
  have c733 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (publicBatchWrapper89_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c733, ← c734, ← c735] at e_10_12
  have hr := e_10_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 10 55) = band (a (.wire 10 47)) (a (.wire 10 51)) := by
  have c736 := (publicBatchWrapper89_copies23 a h).1
  have c737 := (publicBatchWrapper89_copies23 a h).2.1
  have c738 := (publicBatchWrapper89_copies23 a h).2.2.1
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c736, ← c737, ← c738] at e_10_13
  have hr := e_10_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2005)) (a (.virt 200)) (a (.virt 17887)) (a (.virt 17888)) := by
  have c739 := (publicBatchWrapper89_copies23 a h).2.2.2.1
  have c740 := (publicBatchWrapper89_copies23 a h).2.2.2.2.1
  have c741 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.1
  have c742 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.1
  have c743 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.1
  have c745 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c751 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c742, ← c743, ← c744] at e_10_14
  have e_10_15 := arithEq_of_rows h (row := 10) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_15
  simp only [← c745, ← c746, ← c747] at e_10_15
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c739, k3, ← c740, k3, ← c741] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c748, ← c749, k3, ← c750] at e_11_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_14
    linear_combination c751.trans k2 - hc
  · have hc := e_11_1
    simp only [e_11_0, e_10_15] at hc
    linear_combination c752.trans k2 - hc

theorem publicBatchWrapper89_f160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2006)) (a (.virt 200)) (a (.virt 17889)) (a (.virt 17890)) := by
  have c753 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c759 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_10_16 := arithEq_of_rows h (row := 10) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_16
  simp only [← c756, ← c757, ← c758] at e_10_16
  have e_10_17 := arithEq_of_rows h (row := 10) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_17
  simp only [← c759, ← c760, ← c761] at e_10_17
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c753, k3, ← c754, k3, ← c755] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c762, ← c763, k3, ← c764] at e_11_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_16
    linear_combination c765.trans k2 - hc
  · have hc := e_11_3
    simp only [e_11_2, e_10_17] at hc
    linear_combination c766.trans k2 - hc

theorem publicBatchWrapper89_f161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2007)) (a (.virt 200)) (a (.virt 17891)) (a (.virt 17892)) := by
  have c767 := (publicBatchWrapper89_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (publicBatchWrapper89_copies24 a h).1
  have c769 := (publicBatchWrapper89_copies24 a h).2.1
  have c770 := (publicBatchWrapper89_copies24 a h).2.2.1
  have c771 := (publicBatchWrapper89_copies24 a h).2.2.2.1
  have c772 := (publicBatchWrapper89_copies24 a h).2.2.2.2.1
  have c773 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.1
  have c774 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.1
  have c775 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have c778 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_10_18 := arithEq_of_rows h (row := 10) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_18
  simp only [← c770, ← c771, ← c772] at e_10_18
  have e_10_19 := arithEq_of_rows h (row := 10) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_19
  simp only [← c773, ← c774, ← c775] at e_10_19
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c767, k3, ← c768, k3, ← c769] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c776, ← c777, k3, ← c778] at e_11_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_10_18
    linear_combination c779.trans k2 - hc
  · have hc := e_11_5
    simp only [e_11_4, e_10_19] at hc
    linear_combination c780.trans k2 - hc

theorem publicBatchWrapper89_f162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2008)) (a (.virt 200)) (a (.virt 17893)) (a (.virt 17894)) := by
  have c781 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c781, k3, ← c782, k3, ← c783] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c790, ← c791, k3, ← c792] at e_11_7
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c784, ← c785, ← c786] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c787, ← c788, ← c789] at e_12_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_0
    linear_combination c793.trans k2 - hc
  · have hc := e_11_7
    simp only [e_11_6, e_12_1] at hc
    linear_combination c794.trans k2 - hc

theorem publicBatchWrapper89_f163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 12 11) = band (a (.virt 17887)) (a (.virt 17889)) := by
  have c795 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c795, ← c796, ← c797] at e_12_2
  have hr := e_12_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 12 15) = band (a (.virt 17891)) (a (.virt 17893)) := by
  have c798 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (publicBatchWrapper89_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (publicBatchWrapper89_copies25 a h).1
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c798, ← c799, ← c800] at e_12_3
  have hr := e_12_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 12 19) = band (a (.wire 12 11)) (a (.wire 12 15)) := by
  have c801 := (publicBatchWrapper89_copies25 a h).2.1
  have c802 := (publicBatchWrapper89_copies25 a h).2.2.1
  have c803 := (publicBatchWrapper89_copies25 a h).2.2.2.1
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c801, ← c802, ← c803] at e_12_4
  have hr := e_12_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2205)) (a (.virt 200)) (a (.virt 17895)) (a (.virt 17896)) := by
  have c804 := (publicBatchWrapper89_copies25 a h).2.2.2.2.1
  have c805 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.1
  have c806 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.1
  have c807 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c814 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c804, k3, ← c805, k3, ← c806] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c813, ← c814, k3, ← c815] at e_11_9
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c807, ← c808, ← c809] at e_12_5
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c810, ← c811, ← c812] at e_12_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_5
    linear_combination c816.trans k2 - hc
  · have hc := e_11_9
    simp only [e_11_8, e_12_6] at hc
    linear_combination c817.trans k2 - hc

theorem publicBatchWrapper89_f167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2206)) (a (.virt 200)) (a (.virt 17897)) (a (.virt 17898)) := by
  have c818 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c829 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (publicBatchWrapper89_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c818, k3, ← c819, k3, ← c820] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c827, ← c828, k3, ← c829] at e_11_11
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c821, ← c822, ← c823] at e_12_7
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c824, ← c825, ← c826] at e_12_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_7
    linear_combination c830.trans k2 - hc
  · have hc := e_11_11
    simp only [e_11_10, e_12_8] at hc
    linear_combination c831.trans k2 - hc

theorem publicBatchWrapper89_f168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2207)) (a (.virt 200)) (a (.virt 17899)) (a (.virt 17900)) := by
  have c832 := (publicBatchWrapper89_copies26 a h).1
  have c833 := (publicBatchWrapper89_copies26 a h).2.1
  have c834 := (publicBatchWrapper89_copies26 a h).2.2.1
  have c835 := (publicBatchWrapper89_copies26 a h).2.2.2.1
  have c836 := (publicBatchWrapper89_copies26 a h).2.2.2.2.1
  have c837 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.1
  have c838 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.1
  have c839 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.1
  have c841 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have c842 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c844 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c845 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c832, k3, ← c833, k3, ← c834] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c841, ← c842, k3, ← c843] at e_11_13
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c835, ← c836, ← c837] at e_12_9
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c838, ← c839, ← c840] at e_12_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_9
    linear_combination c844.trans k2 - hc
  · have hc := e_11_13
    simp only [e_11_12, e_12_10] at hc
    linear_combination c845.trans k2 - hc

theorem publicBatchWrapper89_f169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2208)) (a (.virt 200)) (a (.virt 17901)) (a (.virt 17902)) := by
  have c846 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c847 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c853 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c846, k3, ← c847, k3, ← c848] at e_11_14
  have e_11_15 := arithEq_of_rows h (row := 11) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c855, ← c856, k3, ← c857] at e_11_15
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c849, ← c850, ← c851] at e_12_11
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c852, ← c853, ← c854] at e_12_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_11
    linear_combination c858.trans k2 - hc
  · have hc := e_11_15
    simp only [e_11_14, e_12_12] at hc
    linear_combination c859.trans k2 - hc

theorem publicBatchWrapper89_f170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 12 55) = band (a (.virt 17895)) (a (.virt 17897)) := by
  have c860 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c860, ← c861, ← c862] at e_12_13
  have hr := e_12_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 12 59) = band (a (.virt 17899)) (a (.virt 17901)) := by
  have c863 := (publicBatchWrapper89_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (publicBatchWrapper89_copies27 a h).1
  have c865 := (publicBatchWrapper89_copies27 a h).2.1
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c863, ← c864, ← c865] at e_12_14
  have hr := e_12_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 12 63) = band (a (.wire 12 55)) (a (.wire 12 59)) := by
  have c866 := (publicBatchWrapper89_copies27 a h).2.2.1
  have c867 := (publicBatchWrapper89_copies27 a h).2.2.2.1
  have c868 := (publicBatchWrapper89_copies27 a h).2.2.2.2.1
  have e_12_15 := arithEq_of_rows h (row := 12) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c866, ← c867, ← c868] at e_12_15
  have hr := e_12_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2405)) (a (.virt 200)) (a (.virt 17903)) (a (.virt 17904)) := by
  have c869 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.1
  have c870 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.1
  have c871 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.1
  have c873 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c876 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c880 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_16 := arithEq_of_rows h (row := 11) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c869, k3, ← c870, k3, ← c871] at e_11_16
  have e_11_17 := arithEq_of_rows h (row := 11) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c878, ← c879, k3, ← c880] at e_11_17
  have e_12_16 := arithEq_of_rows h (row := 12) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c872, ← c873, ← c874] at e_12_16
  have e_12_17 := arithEq_of_rows h (row := 12) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c875, ← c876, ← c877] at e_12_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_16
    linear_combination c881.trans k2 - hc
  · have hc := e_11_17
    simp only [e_11_16, e_12_17] at hc
    linear_combination c882.trans k2 - hc

theorem publicBatchWrapper89_f174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2406)) (a (.virt 200)) (a (.virt 17905)) (a (.virt 17906)) := by
  have c883 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (publicBatchWrapper89_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (publicBatchWrapper89_copies28 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_11_18 := arithEq_of_rows h (row := 11) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c883, k3, ← c884, k3, ← c885] at e_11_18
  have e_11_19 := arithEq_of_rows h (row := 11) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c892, ← c893, k3, ← c894] at e_11_19
  have e_12_18 := arithEq_of_rows h (row := 12) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c886, ← c887, ← c888] at e_12_18
  have e_12_19 := arithEq_of_rows h (row := 12) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c889, ← c890, ← c891] at e_12_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_12_18
    linear_combination c895.trans k2 - hc
  · have hc := e_11_19
    simp only [e_11_18, e_12_19] at hc
    linear_combination c896.trans k2 - hc

theorem publicBatchWrapper89_f175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2407)) (a (.virt 200)) (a (.virt 17907)) (a (.virt 17908)) := by
  have c897 := (publicBatchWrapper89_copies28 a h).2.1
  have c898 := (publicBatchWrapper89_copies28 a h).2.2.1
  have c899 := (publicBatchWrapper89_copies28 a h).2.2.2.1
  have c900 := (publicBatchWrapper89_copies28 a h).2.2.2.2.1
  have c901 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.1
  have c902 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.1
  have c903 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.1
  have c904 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have c907 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c897, k3, ← c898, k3, ← c899] at e_13_0
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c906, ← c907, k3, ← c908] at e_13_1
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c900, ← c901, ← c902] at e_14_0
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c903, ← c904, ← c905] at e_14_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_0
    linear_combination c909.trans k2 - hc
  · have hc := e_13_1
    simp only [e_13_0, e_14_1] at hc
    linear_combination c910.trans k2 - hc

theorem publicBatchWrapper89_f176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2408)) (a (.virt 200)) (a (.virt 17909)) (a (.virt 17910)) := by
  have c911 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c911, k3, ← c912, k3, ← c913] at e_13_2
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c920, ← c921, k3, ← c922] at e_13_3
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c914, ← c915, ← c916] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c917, ← c918, ← c919] at e_14_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_2
    linear_combination c923.trans k2 - hc
  · have hc := e_13_3
    simp only [e_13_2, e_14_3] at hc
    linear_combination c924.trans k2 - hc

theorem publicBatchWrapper89_f177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 14 19) = band (a (.virt 17903)) (a (.virt 17905)) := by
  have c925 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c927 := (publicBatchWrapper89_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c925, ← c926, ← c927] at e_14_4
  have hr := e_14_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 14 23) = band (a (.virt 17907)) (a (.virt 17909)) := by
  have c928 := (publicBatchWrapper89_copies29 a h).1
  have c929 := (publicBatchWrapper89_copies29 a h).2.1
  have c930 := (publicBatchWrapper89_copies29 a h).2.2.1
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c928, ← c929, ← c930] at e_14_5
  have hr := e_14_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 14 27) = band (a (.wire 14 19)) (a (.wire 14 23)) := by
  have c931 := (publicBatchWrapper89_copies29 a h).2.2.2.1
  have c932 := (publicBatchWrapper89_copies29 a h).2.2.2.2.1
  have c933 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.1
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c931, ← c932, ← c933] at e_14_6
  have hr := e_14_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2605)) (a (.virt 200)) (a (.virt 17911)) (a (.virt 17912)) := by
  have c934 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.1
  have c935 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.1
  have c937 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c947 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c934, k3, ← c935, k3, ← c936] at e_13_4
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c943, ← c944, k3, ← c945] at e_13_5
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c937, ← c938, ← c939] at e_14_7
  have e_14_8 := arithEq_of_rows h (row := 14) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c940, ← c941, ← c942] at e_14_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_7
    linear_combination c946.trans k2 - hc
  · have hc := e_13_5
    simp only [e_13_4, e_14_8] at hc
    linear_combination c947.trans k2 - hc

theorem publicBatchWrapper89_f181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2606)) (a (.virt 200)) (a (.virt 17913)) (a (.virt 17914)) := by
  have c948 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c953 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c956 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c959 := (publicBatchWrapper89_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (publicBatchWrapper89_copies30 a h).1
  have c961 := (publicBatchWrapper89_copies30 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c948, k3, ← c949, k3, ← c950] at e_13_6
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c957, ← c958, k3, ← c959] at e_13_7
  have e_14_9 := arithEq_of_rows h (row := 14) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c951, ← c952, ← c953] at e_14_9
  have e_14_10 := arithEq_of_rows h (row := 14) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c954, ← c955, ← c956] at e_14_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_9
    linear_combination c960.trans k2 - hc
  · have hc := e_13_7
    simp only [e_13_6, e_14_10] at hc
    linear_combination c961.trans k2 - hc

theorem publicBatchWrapper89_f182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2607)) (a (.virt 200)) (a (.virt 17915)) (a (.virt 17916)) := by
  have c962 := (publicBatchWrapper89_copies30 a h).2.2.1
  have c963 := (publicBatchWrapper89_copies30 a h).2.2.2.1
  have c964 := (publicBatchWrapper89_copies30 a h).2.2.2.2.1
  have c965 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.1
  have c966 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.1
  have c967 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  have c971 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c962, k3, ← c963, k3, ← c964] at e_13_8
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c971, ← c972, k3, ← c973] at e_13_9
  have e_14_11 := arithEq_of_rows h (row := 14) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c965, ← c966, ← c967] at e_14_11
  have e_14_12 := arithEq_of_rows h (row := 14) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c968, ← c969, ← c970] at e_14_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_11
    linear_combination c974.trans k2 - hc
  · have hc := e_13_9
    simp only [e_13_8, e_14_12] at hc
    linear_combination c975.trans k2 - hc

theorem publicBatchWrapper89_f183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2608)) (a (.virt 200)) (a (.virt 17917)) (a (.virt 17918)) := by
  have c976 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c977 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c987 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c989 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c976, k3, ← c977, k3, ← c978] at e_13_10
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c985, ← c986, k3, ← c987] at e_13_11
  have e_14_13 := arithEq_of_rows h (row := 14) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c979, ← c980, ← c981] at e_14_13
  have e_14_14 := arithEq_of_rows h (row := 14) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c982, ← c983, ← c984] at e_14_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_13
    linear_combination c988.trans k2 - hc
  · have hc := e_13_11
    simp only [e_13_10, e_14_14] at hc
    linear_combination c989.trans k2 - hc

theorem publicBatchWrapper89_f184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 14 63) = band (a (.virt 17911)) (a (.virt 17913)) := by
  have c990 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (publicBatchWrapper89_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (publicBatchWrapper89_copies31 a h).1
  have e_14_15 := arithEq_of_rows h (row := 14) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_15
  simp only [← c990, ← c991, ← c992] at e_14_15
  have hr := e_14_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 14 67) = band (a (.virt 17915)) (a (.virt 17917)) := by
  have c993 := (publicBatchWrapper89_copies31 a h).2.1
  have c994 := (publicBatchWrapper89_copies31 a h).2.2.1
  have c995 := (publicBatchWrapper89_copies31 a h).2.2.2.1
  have e_14_16 := arithEq_of_rows h (row := 14) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_16
  simp only [← c993, ← c994, ← c995] at e_14_16
  have hr := e_14_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 14 71) = band (a (.wire 14 63)) (a (.wire 14 67)) := by
  have c996 := (publicBatchWrapper89_copies31 a h).2.2.2.2.1
  have c997 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.1
  have c998 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.1
  have e_14_17 := arithEq_of_rows h (row := 14) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_17
  simp only [← c996, ← c997, ← c998] at e_14_17
  have hr := e_14_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2805)) (a (.virt 200)) (a (.virt 17919)) (a (.virt 17920)) := by
  have c999 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have c1002 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1005 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1008 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1011 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1012 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c999, k3, ← c1000, k3, ← c1001] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c1008, ← c1009, k3, ← c1010] at e_13_13
  have e_14_18 := arithEq_of_rows h (row := 14) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_18
  simp only [← c1002, ← c1003, ← c1004] at e_14_18
  have e_14_19 := arithEq_of_rows h (row := 14) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_19
  simp only [← c1005, ← c1006, ← c1007] at e_14_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_14_18
    linear_combination c1011.trans k2 - hc
  · have hc := e_13_13
    simp only [e_13_12, e_14_19] at hc
    linear_combination c1012.trans k2 - hc

theorem publicBatchWrapper89_f188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2806)) (a (.virt 200)) (a (.virt 17921)) (a (.virt 17922)) := by
  have c1013 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1017 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1019 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1022 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1023 := (publicBatchWrapper89_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1024 := (publicBatchWrapper89_copies32 a h).1
  have c1025 := (publicBatchWrapper89_copies32 a h).2.1
  have c1026 := (publicBatchWrapper89_copies32 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c1013, k3, ← c1014, k3, ← c1015] at e_13_14
  have e_13_15 := arithEq_of_rows h (row := 13) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c1022, ← c1023, k3, ← c1024] at e_13_15
  have e_15_0 := arithEq_of_rows h (row := 15) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c1016, ← c1017, ← c1018] at e_15_0
  have e_15_1 := arithEq_of_rows h (row := 15) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c1019, ← c1020, ← c1021] at e_15_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_0
    linear_combination c1025.trans k2 - hc
  · have hc := e_13_15
    simp only [e_13_14, e_15_1] at hc
    linear_combination c1026.trans k2 - hc

theorem publicBatchWrapper89_f189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2807)) (a (.virt 200)) (a (.virt 17923)) (a (.virt 17924)) := by
  have c1027 := (publicBatchWrapper89_copies32 a h).2.2.2.1
  have c1028 := (publicBatchWrapper89_copies32 a h).2.2.2.2.1
  have c1029 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.1
  have c1030 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_16 := arithEq_of_rows h (row := 13) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c1027, k3, ← c1028, k3, ← c1029] at e_13_16
  have e_13_17 := arithEq_of_rows h (row := 13) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c1036, ← c1037, k3, ← c1038] at e_13_17
  have e_15_2 := arithEq_of_rows h (row := 15) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c1030, ← c1031, ← c1032] at e_15_2
  have e_15_3 := arithEq_of_rows h (row := 15) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c1033, ← c1034, ← c1035] at e_15_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_2
    linear_combination c1039.trans k2 - hc
  · have hc := e_13_17
    simp only [e_13_16, e_15_3] at hc
    linear_combination c1040.trans k2 - hc

theorem publicBatchWrapper89_f190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 2808)) (a (.virt 200)) (a (.virt 17925)) (a (.virt 17926)) := by
  have c1041 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_13_18 := arithEq_of_rows h (row := 13) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c1041, k3, ← c1042, k3, ← c1043] at e_13_18
  have e_13_19 := arithEq_of_rows h (row := 13) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c1050, ← c1051, k3, ← c1052] at e_13_19
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c1044, ← c1045, ← c1046] at e_15_4
  have e_15_5 := arithEq_of_rows h (row := 15) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c1047, ← c1048, ← c1049] at e_15_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_4
    linear_combination c1053.trans k2 - hc
  · have hc := e_13_19
    simp only [e_13_18, e_15_5] at hc
    linear_combination c1054.trans k2 - hc

theorem publicBatchWrapper89_f191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 15 27) = band (a (.virt 17919)) (a (.virt 17921)) := by
  have c1055 := (publicBatchWrapper89_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (publicBatchWrapper89_copies33 a h).1
  have c1057 := (publicBatchWrapper89_copies33 a h).2.1
  have e_15_6 := arithEq_of_rows h (row := 15) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c1055, ← c1056, ← c1057] at e_15_6
  have hr := e_15_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 15 31) = band (a (.virt 17923)) (a (.virt 17925)) := by
  have c1058 := (publicBatchWrapper89_copies33 a h).2.2.1
  have c1059 := (publicBatchWrapper89_copies33 a h).2.2.2.1
  have c1060 := (publicBatchWrapper89_copies33 a h).2.2.2.2.1
  have e_15_7 := arithEq_of_rows h (row := 15) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_7
  simp only [← c1058, ← c1059, ← c1060] at e_15_7
  have hr := e_15_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 15 35) = band (a (.wire 15 27)) (a (.wire 15 31)) := by
  have c1061 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.1
  have c1062 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.1
  have e_15_8 := arithEq_of_rows h (row := 15) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_8
  simp only [← c1061, ← c1062, ← c1063] at e_15_8
  have hr := e_15_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3005)) (a (.virt 200)) (a (.virt 17927)) (a (.virt 17928)) := by
  have c1064 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_15_9 := arithEq_of_rows h (row := 15) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_9
  simp only [← c1067, ← c1068, ← c1069] at e_15_9
  have e_15_10 := arithEq_of_rows h (row := 15) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_10
  simp only [← c1070, ← c1071, ← c1072] at e_15_10
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c1064, k3, ← c1065, k3, ← c1066] at e_16_0
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c1073, ← c1074, k3, ← c1075] at e_16_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_9
    linear_combination c1076.trans k2 - hc
  · have hc := e_16_1
    simp only [e_16_0, e_15_10] at hc
    linear_combination c1077.trans k2 - hc

theorem publicBatchWrapper89_f195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3006)) (a (.virt 200)) (a (.virt 17929)) (a (.virt 17930)) := by
  have c1078 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (publicBatchWrapper89_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (publicBatchWrapper89_copies34 a h).1
  have c1089 := (publicBatchWrapper89_copies34 a h).2.1
  have c1090 := (publicBatchWrapper89_copies34 a h).2.2.1
  have c1091 := (publicBatchWrapper89_copies34 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_15_11 := arithEq_of_rows h (row := 15) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_11
  simp only [← c1081, ← c1082, ← c1083] at e_15_11
  have e_15_12 := arithEq_of_rows h (row := 15) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_12
  simp only [← c1084, ← c1085, ← c1086] at e_15_12
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c1078, k3, ← c1079, k3, ← c1080] at e_16_2
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c1087, ← c1088, k3, ← c1089] at e_16_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_11
    linear_combination c1090.trans k2 - hc
  · have hc := e_16_3
    simp only [e_16_2, e_15_12] at hc
    linear_combination c1091.trans k2 - hc

theorem publicBatchWrapper89_f196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3007)) (a (.virt 200)) (a (.virt 17931)) (a (.virt 17932)) := by
  have c1092 := (publicBatchWrapper89_copies34 a h).2.2.2.2.1
  have c1093 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.1
  have c1094 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.1
  have c1096 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1099 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1102 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1103 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_15_13 := arithEq_of_rows h (row := 15) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_13
  simp only [← c1095, ← c1096, ← c1097] at e_15_13
  have e_15_14 := arithEq_of_rows h (row := 15) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_14
  simp only [← c1098, ← c1099, ← c1100] at e_15_14
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c1092, k3, ← c1093, k3, ← c1094] at e_16_4
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c1101, ← c1102, k3, ← c1103] at e_16_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_13
    linear_combination c1104.trans k2 - hc
  · have hc := e_16_5
    simp only [e_16_4, e_15_14] at hc
    linear_combination c1105.trans k2 - hc

theorem publicBatchWrapper89_f197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3008)) (a (.virt 200)) (a (.virt 17933)) (a (.virt 17934)) := by
  have c1106 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1107 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1108 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (publicBatchWrapper89_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_15_15 := arithEq_of_rows h (row := 15) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_15
  simp only [← c1109, ← c1110, ← c1111] at e_15_15
  have e_15_16 := arithEq_of_rows h (row := 15) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_16
  simp only [← c1112, ← c1113, ← c1114] at e_15_16
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c1106, k3, ← c1107, k3, ← c1108] at e_16_6
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c1115, ← c1116, k3, ← c1117] at e_16_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_15_15
    linear_combination c1118.trans k2 - hc
  · have hc := e_16_7
    simp only [e_16_6, e_15_16] at hc
    linear_combination c1119.trans k2 - hc

theorem publicBatchWrapper89_f198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 15 71) = band (a (.virt 17927)) (a (.virt 17929)) := by
  have c1120 := (publicBatchWrapper89_copies35 a h).1
  have c1121 := (publicBatchWrapper89_copies35 a h).2.1
  have c1122 := (publicBatchWrapper89_copies35 a h).2.2.1
  have e_15_17 := arithEq_of_rows h (row := 15) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_17
  simp only [← c1120, ← c1121, ← c1122] at e_15_17
  have hr := e_15_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 15 75) = band (a (.virt 17931)) (a (.virt 17933)) := by
  have c1123 := (publicBatchWrapper89_copies35 a h).2.2.2.1
  have c1124 := (publicBatchWrapper89_copies35 a h).2.2.2.2.1
  have c1125 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.1
  have e_15_18 := arithEq_of_rows h (row := 15) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_18
  simp only [← c1123, ← c1124, ← c1125] at e_15_18
  have hr := e_15_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 15 79) = band (a (.wire 15 71)) (a (.wire 15 75)) := by
  have c1126 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.1
  have e_15_19 := arithEq_of_rows h (row := 15) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_19
  simp only [← c1126, ← c1127, ← c1128] at e_15_19
  have hr := e_15_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3205)) (a (.virt 200)) (a (.virt 17935)) (a (.virt 17936)) := by
  have c1129 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1140 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c1129, k3, ← c1130, k3, ← c1131] at e_16_8
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c1138, ← c1139, k3, ← c1140] at e_16_9
  have e_17_0 := arithEq_of_rows h (row := 17) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c1132, ← c1133, ← c1134] at e_17_0
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c1135, ← c1136, ← c1137] at e_17_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_0
    linear_combination c1141.trans k2 - hc
  · have hc := e_16_9
    simp only [e_16_8, e_17_1] at hc
    linear_combination c1142.trans k2 - hc

theorem publicBatchWrapper89_f202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3206)) (a (.virt 200)) (a (.virt 17937)) (a (.virt 17938)) := by
  have c1143 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1150 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (publicBatchWrapper89_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (publicBatchWrapper89_copies36 a h).1
  have c1153 := (publicBatchWrapper89_copies36 a h).2.1
  have c1154 := (publicBatchWrapper89_copies36 a h).2.2.1
  have c1155 := (publicBatchWrapper89_copies36 a h).2.2.2.1
  have c1156 := (publicBatchWrapper89_copies36 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c1143, k3, ← c1144, k3, ← c1145] at e_16_10
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1152, ← c1153, k3, ← c1154] at e_16_11
  have e_17_2 := arithEq_of_rows h (row := 17) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c1146, ← c1147, ← c1148] at e_17_2
  have e_17_3 := arithEq_of_rows h (row := 17) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c1149, ← c1150, ← c1151] at e_17_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_2
    linear_combination c1155.trans k2 - hc
  · have hc := e_16_11
    simp only [e_16_10, e_17_3] at hc
    linear_combination c1156.trans k2 - hc

theorem publicBatchWrapper89_f203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3207)) (a (.virt 200)) (a (.virt 17939)) (a (.virt 17940)) := by
  have c1157 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.1
  have c1158 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.1
  have c1159 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have c1162 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1157, k3, ← c1158, k3, ← c1159] at e_16_12
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1166, ← c1167, k3, ← c1168] at e_16_13
  have e_17_4 := arithEq_of_rows h (row := 17) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c1160, ← c1161, ← c1162] at e_17_4
  have e_17_5 := arithEq_of_rows h (row := 17) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c1163, ← c1164, ← c1165] at e_17_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_4
    linear_combination c1169.trans k2 - hc
  · have hc := e_16_13
    simp only [e_16_12, e_17_5] at hc
    linear_combination c1170.trans k2 - hc

theorem publicBatchWrapper89_f204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3208)) (a (.virt 200)) (a (.virt 17941)) (a (.virt 17942)) := by
  have c1171 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1174 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1177 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1179 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1180 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1182 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1183 := (publicBatchWrapper89_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (publicBatchWrapper89_copies37 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1171, k3, ← c1172, k3, ← c1173] at e_16_14
  have e_16_15 := arithEq_of_rows h (row := 16) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_15
  simp only [← c1180, ← c1181, k3, ← c1182] at e_16_15
  have e_17_6 := arithEq_of_rows h (row := 17) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c1174, ← c1175, ← c1176] at e_17_6
  have e_17_7 := arithEq_of_rows h (row := 17) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c1177, ← c1178, ← c1179] at e_17_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_6
    linear_combination c1183.trans k2 - hc
  · have hc := e_16_15
    simp only [e_16_14, e_17_7] at hc
    linear_combination c1184.trans k2 - hc

theorem publicBatchWrapper89_f205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 17 35) = band (a (.virt 17935)) (a (.virt 17937)) := by
  have c1185 := (publicBatchWrapper89_copies37 a h).2.1
  have c1186 := (publicBatchWrapper89_copies37 a h).2.2.1
  have c1187 := (publicBatchWrapper89_copies37 a h).2.2.2.1
  have e_17_8 := arithEq_of_rows h (row := 17) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c1185, ← c1186, ← c1187] at e_17_8
  have hr := e_17_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 17 39) = band (a (.virt 17939)) (a (.virt 17941)) := by
  have c1188 := (publicBatchWrapper89_copies37 a h).2.2.2.2.1
  have c1189 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.1
  have c1190 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.1
  have e_17_9 := arithEq_of_rows h (row := 17) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c1188, ← c1189, ← c1190] at e_17_9
  have hr := e_17_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 17 43) = band (a (.wire 17 35)) (a (.wire 17 39)) := by
  have c1191 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have e_17_10 := arithEq_of_rows h (row := 17) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c1191, ← c1192, ← c1193] at e_17_10
  have hr := e_17_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3405)) (a (.virt 200)) (a (.virt 17943)) (a (.virt 17944)) := by
  have c1194 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1207 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_16_16 := arithEq_of_rows h (row := 16) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_16
  simp only [← c1194, k3, ← c1195, k3, ← c1196] at e_16_16
  have e_16_17 := arithEq_of_rows h (row := 16) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_17
  simp only [← c1203, ← c1204, k3, ← c1205] at e_16_17
  have e_17_11 := arithEq_of_rows h (row := 17) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c1197, ← c1198, ← c1199] at e_17_11
  have e_17_12 := arithEq_of_rows h (row := 17) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c1200, ← c1201, ← c1202] at e_17_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_11
    linear_combination c1206.trans k2 - hc
  · have hc := e_16_17
    simp only [e_16_16, e_17_12] at hc
    linear_combination c1207.trans k2 - hc

theorem publicBatchWrapper89_f209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3406)) (a (.virt 200)) (a (.virt 17945)) (a (.virt 17946)) := by
  have c1208 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (publicBatchWrapper89_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (publicBatchWrapper89_copies38 a h).1
  have c1217 := (publicBatchWrapper89_copies38 a h).2.1
  have c1218 := (publicBatchWrapper89_copies38 a h).2.2.1
  have c1219 := (publicBatchWrapper89_copies38 a h).2.2.2.1
  have c1220 := (publicBatchWrapper89_copies38 a h).2.2.2.2.1
  have c1221 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_16_18 := arithEq_of_rows h (row := 16) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_18
  simp only [← c1208, k3, ← c1209, k3, ← c1210] at e_16_18
  have e_16_19 := arithEq_of_rows h (row := 16) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_19
  simp only [← c1217, ← c1218, k3, ← c1219] at e_16_19
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c1211, ← c1212, ← c1213] at e_17_13
  have e_17_14 := arithEq_of_rows h (row := 17) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c1214, ← c1215, ← c1216] at e_17_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_13
    linear_combination c1220.trans k2 - hc
  · have hc := e_16_19
    simp only [e_16_18, e_17_14] at hc
    linear_combination c1221.trans k2 - hc

theorem publicBatchWrapper89_f210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3407)) (a (.virt 200)) (a (.virt 17947)) (a (.virt 17948)) := by
  have c1222 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.1
  have c1224 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1227 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1233 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_17_15 := arithEq_of_rows h (row := 17) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_15
  simp only [← c1225, ← c1226, ← c1227] at e_17_15
  have e_17_16 := arithEq_of_rows h (row := 17) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_16
  simp only [← c1228, ← c1229, ← c1230] at e_17_16
  have e_18_0 := arithEq_of_rows h (row := 18) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c1222, k3, ← c1223, k3, ← c1224] at e_18_0
  have e_18_1 := arithEq_of_rows h (row := 18) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c1231, ← c1232, k3, ← c1233] at e_18_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_15
    linear_combination c1234.trans k2 - hc
  · have hc := e_18_1
    simp only [e_18_0, e_17_16] at hc
    linear_combination c1235.trans k2 - hc

theorem publicBatchWrapper89_f211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3408)) (a (.virt 200)) (a (.virt 17949)) (a (.virt 17950)) := by
  have c1236 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1239 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1241 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1245 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (publicBatchWrapper89_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (publicBatchWrapper89_copies39 a h).1
  have c1249 := (publicBatchWrapper89_copies39 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_17_17 := arithEq_of_rows h (row := 17) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_17
  simp only [← c1239, ← c1240, ← c1241] at e_17_17
  have e_17_18 := arithEq_of_rows h (row := 17) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_18
  simp only [← c1242, ← c1243, ← c1244] at e_17_18
  have e_18_2 := arithEq_of_rows h (row := 18) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c1236, k3, ← c1237, k3, ← c1238] at e_18_2
  have e_18_3 := arithEq_of_rows h (row := 18) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c1245, ← c1246, k3, ← c1247] at e_18_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_17_17
    linear_combination c1248.trans k2 - hc
  · have hc := e_18_3
    simp only [e_18_2, e_17_18] at hc
    linear_combination c1249.trans k2 - hc

theorem publicBatchWrapper89_f212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 17 79) = band (a (.virt 17943)) (a (.virt 17945)) := by
  have c1250 := (publicBatchWrapper89_copies39 a h).2.2.1
  have c1251 := (publicBatchWrapper89_copies39 a h).2.2.2.1
  have c1252 := (publicBatchWrapper89_copies39 a h).2.2.2.2.1
  have e_17_19 := arithEq_of_rows h (row := 17) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_19
  simp only [← c1250, ← c1251, ← c1252] at e_17_19
  have hr := e_17_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 19 3) = band (a (.virt 17947)) (a (.virt 17949)) := by
  have c1253 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.1
  have c1254 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.1
  have e_19_0 := arithEq_of_rows h (row := 19) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c1253, ← c1254, ← c1255] at e_19_0
  have hr := e_19_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 19 7) = band (a (.wire 17 79)) (a (.wire 19 3)) := by
  have c1256 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have c1258 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_19_1 := arithEq_of_rows h (row := 19) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c1256, ← c1257, ← c1258] at e_19_1
  have hr := e_19_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3605)) (a (.virt 200)) (a (.virt 17951)) (a (.virt 17952)) := by
  have c1259 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1261 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1264 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1267 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1270 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1271 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_4 := arithEq_of_rows h (row := 18) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c1259, k3, ← c1260, k3, ← c1261] at e_18_4
  have e_18_5 := arithEq_of_rows h (row := 18) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c1268, ← c1269, k3, ← c1270] at e_18_5
  have e_19_2 := arithEq_of_rows h (row := 19) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c1262, ← c1263, ← c1264] at e_19_2
  have e_19_3 := arithEq_of_rows h (row := 19) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c1265, ← c1266, ← c1267] at e_19_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_2
    linear_combination c1271.trans k2 - hc
  · have hc := e_18_5
    simp only [e_18_4, e_19_3] at hc
    linear_combination c1272.trans k2 - hc

theorem publicBatchWrapper89_f216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3606)) (a (.virt 200)) (a (.virt 17953)) (a (.virt 17954)) := by
  have c1273 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1274 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (publicBatchWrapper89_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (publicBatchWrapper89_copies40 a h).1
  have c1281 := (publicBatchWrapper89_copies40 a h).2.1
  have c1282 := (publicBatchWrapper89_copies40 a h).2.2.1
  have c1283 := (publicBatchWrapper89_copies40 a h).2.2.2.1
  have c1284 := (publicBatchWrapper89_copies40 a h).2.2.2.2.1
  have c1285 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.1
  have c1286 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_6 := arithEq_of_rows h (row := 18) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c1273, k3, ← c1274, k3, ← c1275] at e_18_6
  have e_18_7 := arithEq_of_rows h (row := 18) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c1282, ← c1283, k3, ← c1284] at e_18_7
  have e_19_4 := arithEq_of_rows h (row := 19) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c1276, ← c1277, ← c1278] at e_19_4
  have e_19_5 := arithEq_of_rows h (row := 19) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c1279, ← c1280, ← c1281] at e_19_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_4
    linear_combination c1285.trans k2 - hc
  · have hc := e_18_7
    simp only [e_18_6, e_19_5] at hc
    linear_combination c1286.trans k2 - hc

theorem publicBatchWrapper89_f217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3607)) (a (.virt 200)) (a (.virt 17955)) (a (.virt 17956)) := by
  have c1287 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1291 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_8 := arithEq_of_rows h (row := 18) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c1287, k3, ← c1288, k3, ← c1289] at e_18_8
  have e_18_9 := arithEq_of_rows h (row := 18) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c1296, ← c1297, k3, ← c1298] at e_18_9
  have e_19_6 := arithEq_of_rows h (row := 19) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c1290, ← c1291, ← c1292] at e_19_6
  have e_19_7 := arithEq_of_rows h (row := 19) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c1293, ← c1294, ← c1295] at e_19_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_6
    linear_combination c1299.trans k2 - hc
  · have hc := e_18_9
    simp only [e_18_8, e_19_7] at hc
    linear_combination c1300.trans k2 - hc

theorem publicBatchWrapper89_f218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3608)) (a (.virt 200)) (a (.virt 17957)) (a (.virt 17958)) := by
  have c1301 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1308 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (publicBatchWrapper89_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1312 := (publicBatchWrapper89_copies41 a h).1
  have c1313 := (publicBatchWrapper89_copies41 a h).2.1
  have c1314 := (publicBatchWrapper89_copies41 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c1301, k3, ← c1302, k3, ← c1303] at e_18_10
  have e_18_11 := arithEq_of_rows h (row := 18) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c1310, ← c1311, k3, ← c1312] at e_18_11
  have e_19_8 := arithEq_of_rows h (row := 19) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c1304, ← c1305, ← c1306] at e_19_8
  have e_19_9 := arithEq_of_rows h (row := 19) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c1307, ← c1308, ← c1309] at e_19_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_8
    linear_combination c1313.trans k2 - hc
  · have hc := e_18_11
    simp only [e_18_10, e_19_9] at hc
    linear_combination c1314.trans k2 - hc

theorem publicBatchWrapper89_f219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 19 43) = band (a (.virt 17951)) (a (.virt 17953)) := by
  have c1315 := (publicBatchWrapper89_copies41 a h).2.2.2.1
  have c1316 := (publicBatchWrapper89_copies41 a h).2.2.2.2.1
  have c1317 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.1
  have e_19_10 := arithEq_of_rows h (row := 19) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c1315, ← c1316, ← c1317] at e_19_10
  have hr := e_19_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 19 47) = band (a (.virt 17955)) (a (.virt 17957)) := by
  have c1318 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.1
  have c1319 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.1
  have e_19_11 := arithEq_of_rows h (row := 19) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c1318, ← c1319, ← c1320] at e_19_11
  have hr := e_19_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 19 51) = band (a (.wire 19 43)) (a (.wire 19 47)) := by
  have c1321 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have c1322 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_19_12 := arithEq_of_rows h (row := 19) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c1321, ← c1322, ← c1323] at e_19_12
  have hr := e_19_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3805)) (a (.virt 200)) (a (.virt 17959)) (a (.virt 17960)) := by
  have c1324 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1325 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1329 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1335 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_12 := arithEq_of_rows h (row := 18) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c1324, k3, ← c1325, k3, ← c1326] at e_18_12
  have e_18_13 := arithEq_of_rows h (row := 18) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c1333, ← c1334, k3, ← c1335] at e_18_13
  have e_19_13 := arithEq_of_rows h (row := 19) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c1327, ← c1328, ← c1329] at e_19_13
  have e_19_14 := arithEq_of_rows h (row := 19) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c1330, ← c1331, ← c1332] at e_19_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_13
    linear_combination c1336.trans k2 - hc
  · have hc := e_18_13
    simp only [e_18_12, e_19_14] at hc
    linear_combination c1337.trans k2 - hc

theorem publicBatchWrapper89_f223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3806)) (a (.virt 200)) (a (.virt 17961)) (a (.virt 17962)) := by
  have c1338 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1342 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (publicBatchWrapper89_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (publicBatchWrapper89_copies42 a h).1
  have c1345 := (publicBatchWrapper89_copies42 a h).2.1
  have c1346 := (publicBatchWrapper89_copies42 a h).2.2.1
  have c1347 := (publicBatchWrapper89_copies42 a h).2.2.2.1
  have c1348 := (publicBatchWrapper89_copies42 a h).2.2.2.2.1
  have c1349 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.1
  have c1350 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.1
  have c1351 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_14 := arithEq_of_rows h (row := 18) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c1338, k3, ← c1339, k3, ← c1340] at e_18_14
  have e_18_15 := arithEq_of_rows h (row := 18) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_15
  simp only [← c1347, ← c1348, k3, ← c1349] at e_18_15
  have e_19_15 := arithEq_of_rows h (row := 19) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_15
  simp only [← c1341, ← c1342, ← c1343] at e_19_15
  have e_19_16 := arithEq_of_rows h (row := 19) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_16
  simp only [← c1344, ← c1345, ← c1346] at e_19_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_15
    linear_combination c1350.trans k2 - hc
  · have hc := e_18_15
    simp only [e_18_14, e_19_16] at hc
    linear_combination c1351.trans k2 - hc

theorem publicBatchWrapper89_f224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3807)) (a (.virt 200)) (a (.virt 17963)) (a (.virt 17964)) := by
  have c1352 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have c1354 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1355 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1358 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1361 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1362 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1363 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_16 := arithEq_of_rows h (row := 18) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_16
  simp only [← c1352, k3, ← c1353, k3, ← c1354] at e_18_16
  have e_18_17 := arithEq_of_rows h (row := 18) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_17
  simp only [← c1361, ← c1362, k3, ← c1363] at e_18_17
  have e_19_17 := arithEq_of_rows h (row := 19) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_17
  simp only [← c1355, ← c1356, ← c1357] at e_19_17
  have e_19_18 := arithEq_of_rows h (row := 19) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_18
  simp only [← c1358, ← c1359, ← c1360] at e_19_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_17
    linear_combination c1364.trans k2 - hc
  · have hc := e_18_17
    simp only [e_18_16, e_19_18] at hc
    linear_combination c1365.trans k2 - hc

theorem publicBatchWrapper89_f225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 3808)) (a (.virt 200)) (a (.virt 17965)) (a (.virt 17966)) := by
  have c1366 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1375 := (publicBatchWrapper89_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (publicBatchWrapper89_copies43 a h).1
  have c1377 := (publicBatchWrapper89_copies43 a h).2.1
  have c1378 := (publicBatchWrapper89_copies43 a h).2.2.1
  have c1379 := (publicBatchWrapper89_copies43 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_18_18 := arithEq_of_rows h (row := 18) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_18
  simp only [← c1366, k3, ← c1367, k3, ← c1368] at e_18_18
  have e_18_19 := arithEq_of_rows h (row := 18) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_19
  simp only [← c1375, ← c1376, k3, ← c1377] at e_18_19
  have e_19_19 := arithEq_of_rows h (row := 19) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_19
  simp only [← c1369, ← c1370, ← c1371] at e_19_19
  have e_20_0 := arithEq_of_rows h (row := 20) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_0
  simp only [← c1372, ← c1373, ← c1374] at e_20_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_19_19
    linear_combination c1378.trans k2 - hc
  · have hc := e_18_19
    simp only [e_18_18, e_20_0] at hc
    linear_combination c1379.trans k2 - hc

theorem publicBatchWrapper89_f226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 20 7) = band (a (.virt 17959)) (a (.virt 17961)) := by
  have c1380 := (publicBatchWrapper89_copies43 a h).2.2.2.2.1
  have c1381 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.1
  have c1382 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.1
  have e_20_1 := arithEq_of_rows h (row := 20) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_1
  simp only [← c1380, ← c1381, ← c1382] at e_20_1
  have hr := e_20_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 20 11) = band (a (.virt 17963)) (a (.virt 17965)) := by
  have c1383 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have e_20_2 := arithEq_of_rows h (row := 20) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_2
  simp only [← c1383, ← c1384, ← c1385] at e_20_2
  have hr := e_20_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 20 15) = band (a (.wire 20 7)) (a (.wire 20 11)) := by
  have c1386 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_20_3 := arithEq_of_rows h (row := 20) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_3
  simp only [← c1386, ← c1387, ← c1388] at e_20_3
  have hr := e_20_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4005)) (a (.virt 200)) (a (.virt 17967)) (a (.virt 17968)) := by
  have c1389 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1392 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_4 := arithEq_of_rows h (row := 20) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_4
  simp only [← c1392, ← c1393, ← c1394] at e_20_4
  have e_20_5 := arithEq_of_rows h (row := 20) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_5
  simp only [← c1395, ← c1396, ← c1397] at e_20_5
  have e_21_0 := arithEq_of_rows h (row := 21) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c1389, k3, ← c1390, k3, ← c1391] at e_21_0
  have e_21_1 := arithEq_of_rows h (row := 21) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c1398, ← c1399, k3, ← c1400] at e_21_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_4
    linear_combination c1401.trans k2 - hc
  · have hc := e_21_1
    simp only [e_21_0, e_20_5] at hc
    linear_combination c1402.trans k2 - hc

theorem publicBatchWrapper89_f230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4006)) (a (.virt 200)) (a (.virt 17969)) (a (.virt 17970)) := by
  have c1403 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (publicBatchWrapper89_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (publicBatchWrapper89_copies44 a h).1
  have c1409 := (publicBatchWrapper89_copies44 a h).2.1
  have c1410 := (publicBatchWrapper89_copies44 a h).2.2.1
  have c1411 := (publicBatchWrapper89_copies44 a h).2.2.2.1
  have c1412 := (publicBatchWrapper89_copies44 a h).2.2.2.2.1
  have c1413 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.1
  have c1414 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_6 := arithEq_of_rows h (row := 20) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_6
  simp only [← c1406, ← c1407, ← c1408] at e_20_6
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c1409, ← c1410, ← c1411] at e_20_7
  have e_21_2 := arithEq_of_rows h (row := 21) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c1403, k3, ← c1404, k3, ← c1405] at e_21_2
  have e_21_3 := arithEq_of_rows h (row := 21) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c1412, ← c1413, k3, ← c1414] at e_21_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_6
    linear_combination c1415.trans k2 - hc
  · have hc := e_21_3
    simp only [e_21_2, e_20_7] at hc
    linear_combination c1416.trans k2 - hc

theorem publicBatchWrapper89_f231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4007)) (a (.virt 200)) (a (.virt 17971)) (a (.virt 17972)) := by
  have c1417 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1423 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1425 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1426 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1428 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1429 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_8 := arithEq_of_rows h (row := 20) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_8
  simp only [← c1420, ← c1421, ← c1422] at e_20_8
  have e_20_9 := arithEq_of_rows h (row := 20) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_9
  simp only [← c1423, ← c1424, ← c1425] at e_20_9
  have e_21_4 := arithEq_of_rows h (row := 21) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c1417, k3, ← c1418, k3, ← c1419] at e_21_4
  have e_21_5 := arithEq_of_rows h (row := 21) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c1426, ← c1427, k3, ← c1428] at e_21_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_8
    linear_combination c1429.trans k2 - hc
  · have hc := e_21_5
    simp only [e_21_4, e_20_9] at hc
    linear_combination c1430.trans k2 - hc

theorem publicBatchWrapper89_f232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4008)) (a (.virt 200)) (a (.virt 17973)) (a (.virt 17974)) := by
  have c1431 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1432 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1434 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1435 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1437 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1438 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1439 := (publicBatchWrapper89_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1440 := (publicBatchWrapper89_copies45 a h).1
  have c1441 := (publicBatchWrapper89_copies45 a h).2.1
  have c1442 := (publicBatchWrapper89_copies45 a h).2.2.1
  have c1443 := (publicBatchWrapper89_copies45 a h).2.2.2.1
  have c1444 := (publicBatchWrapper89_copies45 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_10 := arithEq_of_rows h (row := 20) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_10
  simp only [← c1434, ← c1435, ← c1436] at e_20_10
  have e_20_11 := arithEq_of_rows h (row := 20) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_11
  simp only [← c1437, ← c1438, ← c1439] at e_20_11
  have e_21_6 := arithEq_of_rows h (row := 21) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c1431, k3, ← c1432, k3, ← c1433] at e_21_6
  have e_21_7 := arithEq_of_rows h (row := 21) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c1440, ← c1441, k3, ← c1442] at e_21_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_10
    linear_combination c1443.trans k2 - hc
  · have hc := e_21_7
    simp only [e_21_6, e_20_11] at hc
    linear_combination c1444.trans k2 - hc

theorem publicBatchWrapper89_f233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 20 51) = band (a (.virt 17967)) (a (.virt 17969)) := by
  have c1445 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.1
  have c1446 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.1
  have c1447 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.1
  have e_20_12 := arithEq_of_rows h (row := 20) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_12
  simp only [← c1445, ← c1446, ← c1447] at e_20_12
  have hr := e_20_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 20 55) = band (a (.virt 17971)) (a (.virt 17973)) := by
  have c1448 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_20_13 := arithEq_of_rows h (row := 20) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_13
  simp only [← c1448, ← c1449, ← c1450] at e_20_13
  have hr := e_20_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 20 59) = band (a (.wire 20 51)) (a (.wire 20 55)) := by
  have c1451 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_20_14 := arithEq_of_rows h (row := 20) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_14
  simp only [← c1451, ← c1452, ← c1453] at e_20_14
  have hr := e_20_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4205)) (a (.virt 200)) (a (.virt 17975)) (a (.virt 17976)) := by
  have c1454 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1456 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1459 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1465 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_15 := arithEq_of_rows h (row := 20) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_15
  simp only [← c1457, ← c1458, ← c1459] at e_20_15
  have e_20_16 := arithEq_of_rows h (row := 20) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_16
  simp only [← c1460, ← c1461, ← c1462] at e_20_16
  have e_21_8 := arithEq_of_rows h (row := 21) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1454, k3, ← c1455, k3, ← c1456] at e_21_8
  have e_21_9 := arithEq_of_rows h (row := 21) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c1463, ← c1464, k3, ← c1465] at e_21_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_15
    linear_combination c1466.trans k2 - hc
  · have hc := e_21_9
    simp only [e_21_8, e_20_16] at hc
    linear_combination c1467.trans k2 - hc

theorem publicBatchWrapper89_f237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4206)) (a (.virt 200)) (a (.virt 17977)) (a (.virt 17978)) := by
  have c1468 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1469 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (publicBatchWrapper89_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (publicBatchWrapper89_copies46 a h).1
  have c1473 := (publicBatchWrapper89_copies46 a h).2.1
  have c1474 := (publicBatchWrapper89_copies46 a h).2.2.1
  have c1475 := (publicBatchWrapper89_copies46 a h).2.2.2.1
  have c1476 := (publicBatchWrapper89_copies46 a h).2.2.2.2.1
  have c1477 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.1
  have c1478 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.1
  have c1479 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_17 := arithEq_of_rows h (row := 20) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_17
  simp only [← c1471, ← c1472, ← c1473] at e_20_17
  have e_20_18 := arithEq_of_rows h (row := 20) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_18
  simp only [← c1474, ← c1475, ← c1476] at e_20_18
  have e_21_10 := arithEq_of_rows h (row := 21) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c1468, k3, ← c1469, k3, ← c1470] at e_21_10
  have e_21_11 := arithEq_of_rows h (row := 21) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c1477, ← c1478, k3, ← c1479] at e_21_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_17
    linear_combination c1480.trans k2 - hc
  · have hc := e_21_11
    simp only [e_21_10, e_20_18] at hc
    linear_combination c1481.trans k2 - hc

theorem publicBatchWrapper89_f238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4207)) (a (.virt 200)) (a (.virt 17979)) (a (.virt 17980)) := by
  have c1482 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1486 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1493 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_20_19 := arithEq_of_rows h (row := 20) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_19
  simp only [← c1485, ← c1486, ← c1487] at e_20_19
  have e_21_12 := arithEq_of_rows h (row := 21) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c1482, k3, ← c1483, k3, ← c1484] at e_21_12
  have e_21_13 := arithEq_of_rows h (row := 21) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c1491, ← c1492, k3, ← c1493] at e_21_13
  have e_22_0 := arithEq_of_rows h (row := 22) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_0
  simp only [← c1488, ← c1489, ← c1490] at e_22_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_20_19
    linear_combination c1494.trans k2 - hc
  · have hc := e_21_13
    simp only [e_21_12, e_22_0] at hc
    linear_combination c1495.trans k2 - hc

theorem publicBatchWrapper89_f239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4208)) (a (.virt 200)) (a (.virt 17981)) (a (.virt 17982)) := by
  have c1496 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1499 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1502 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1503 := (publicBatchWrapper89_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (publicBatchWrapper89_copies47 a h).1
  have c1505 := (publicBatchWrapper89_copies47 a h).2.1
  have c1506 := (publicBatchWrapper89_copies47 a h).2.2.1
  have c1507 := (publicBatchWrapper89_copies47 a h).2.2.2.1
  have c1508 := (publicBatchWrapper89_copies47 a h).2.2.2.2.1
  have c1509 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_21_14 := arithEq_of_rows h (row := 21) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c1496, k3, ← c1497, k3, ← c1498] at e_21_14
  have e_21_15 := arithEq_of_rows h (row := 21) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_15
  simp only [← c1505, ← c1506, k3, ← c1507] at e_21_15
  have e_22_1 := arithEq_of_rows h (row := 22) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_1
  simp only [← c1499, ← c1500, ← c1501] at e_22_1
  have e_22_2 := arithEq_of_rows h (row := 22) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_2
  simp only [← c1502, ← c1503, ← c1504] at e_22_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_1
    linear_combination c1508.trans k2 - hc
  · have hc := e_21_15
    simp only [e_21_14, e_22_2] at hc
    linear_combination c1509.trans k2 - hc

theorem publicBatchWrapper89_f240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 22 15) = band (a (.virt 17975)) (a (.virt 17977)) := by
  have c1510 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.1
  have e_22_3 := arithEq_of_rows h (row := 22) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_3
  simp only [← c1510, ← c1511, ← c1512] at e_22_3
  have hr := e_22_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 22 19) = band (a (.virt 17979)) (a (.virt 17981)) := by
  have c1513 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have c1514 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_22_4 := arithEq_of_rows h (row := 22) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_4
  simp only [← c1513, ← c1514, ← c1515] at e_22_4
  have hr := e_22_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 22 23) = band (a (.wire 22 15)) (a (.wire 22 19)) := by
  have c1516 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_22_5 := arithEq_of_rows h (row := 22) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_5
  simp only [← c1516, ← c1517, ← c1518] at e_22_5
  have hr := e_22_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4405)) (a (.virt 200)) (a (.virt 17983)) (a (.virt 17984)) := by
  have c1519 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1520 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1522 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1523 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1526 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1530 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1531 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1532 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_21_16 := arithEq_of_rows h (row := 21) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_16
  simp only [← c1519, k3, ← c1520, k3, ← c1521] at e_21_16
  have e_21_17 := arithEq_of_rows h (row := 21) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_17
  simp only [← c1528, ← c1529, k3, ← c1530] at e_21_17
  have e_22_6 := arithEq_of_rows h (row := 22) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_6
  simp only [← c1522, ← c1523, ← c1524] at e_22_6
  have e_22_7 := arithEq_of_rows h (row := 22) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_7
  simp only [← c1525, ← c1526, ← c1527] at e_22_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_6
    linear_combination c1531.trans k2 - hc
  · have hc := e_21_17
    simp only [e_21_16, e_22_7] at hc
    linear_combination c1532.trans k2 - hc

theorem publicBatchWrapper89_f244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4406)) (a (.virt 200)) (a (.virt 17985)) (a (.virt 17986)) := by
  have c1533 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1535 := (publicBatchWrapper89_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (publicBatchWrapper89_copies48 a h).1
  have c1537 := (publicBatchWrapper89_copies48 a h).2.1
  have c1538 := (publicBatchWrapper89_copies48 a h).2.2.1
  have c1539 := (publicBatchWrapper89_copies48 a h).2.2.2.1
  have c1540 := (publicBatchWrapper89_copies48 a h).2.2.2.2.1
  have c1541 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.1
  have c1542 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.1
  have c1543 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.1
  have c1544 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.1
  have c1545 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have c1546 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_21_18 := arithEq_of_rows h (row := 21) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_18
  simp only [← c1533, k3, ← c1534, k3, ← c1535] at e_21_18
  have e_21_19 := arithEq_of_rows h (row := 21) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_19
  simp only [← c1542, ← c1543, k3, ← c1544] at e_21_19
  have e_22_8 := arithEq_of_rows h (row := 22) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_8
  simp only [← c1536, ← c1537, ← c1538] at e_22_8
  have e_22_9 := arithEq_of_rows h (row := 22) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_9
  simp only [← c1539, ← c1540, ← c1541] at e_22_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_8
    linear_combination c1545.trans k2 - hc
  · have hc := e_21_19
    simp only [e_21_18, e_22_9] at hc
    linear_combination c1546.trans k2 - hc

theorem publicBatchWrapper89_f245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4407)) (a (.virt 200)) (a (.virt 17987)) (a (.virt 17988)) := by
  have c1547 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1548 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1553 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1554 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1556 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1557 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1558 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1559 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_22_10 := arithEq_of_rows h (row := 22) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_10
  simp only [← c1550, ← c1551, ← c1552] at e_22_10
  have e_22_11 := arithEq_of_rows h (row := 22) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c1553, ← c1554, ← c1555] at e_22_11
  have e_23_0 := arithEq_of_rows h (row := 23) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_0
  simp only [← c1547, k3, ← c1548, k3, ← c1549] at e_23_0
  have e_23_1 := arithEq_of_rows h (row := 23) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_1
  simp only [← c1556, ← c1557, k3, ← c1558] at e_23_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_10
    linear_combination c1559.trans k2 - hc
  · have hc := e_23_1
    simp only [e_23_0, e_22_11] at hc
    linear_combination c1560.trans k2 - hc

theorem publicBatchWrapper89_f246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4408)) (a (.virt 200)) (a (.virt 17989)) (a (.virt 17990)) := by
  have c1561 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1562 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (publicBatchWrapper89_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (publicBatchWrapper89_copies49 a h).1
  have c1569 := (publicBatchWrapper89_copies49 a h).2.1
  have c1570 := (publicBatchWrapper89_copies49 a h).2.2.1
  have c1571 := (publicBatchWrapper89_copies49 a h).2.2.2.1
  have c1572 := (publicBatchWrapper89_copies49 a h).2.2.2.2.1
  have c1573 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.1
  have c1574 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_22_12 := arithEq_of_rows h (row := 22) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_12
  simp only [← c1564, ← c1565, ← c1566] at e_22_12
  have e_22_13 := arithEq_of_rows h (row := 22) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_13
  simp only [← c1567, ← c1568, ← c1569] at e_22_13
  have e_23_2 := arithEq_of_rows h (row := 23) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_2
  simp only [← c1561, k3, ← c1562, k3, ← c1563] at e_23_2
  have e_23_3 := arithEq_of_rows h (row := 23) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_3
  simp only [← c1570, ← c1571, k3, ← c1572] at e_23_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_12
    linear_combination c1573.trans k2 - hc
  · have hc := e_23_3
    simp only [e_23_2, e_22_13] at hc
    linear_combination c1574.trans k2 - hc

theorem publicBatchWrapper89_f247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 22 59) = band (a (.virt 17983)) (a (.virt 17985)) := by
  have c1575 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have e_22_14 := arithEq_of_rows h (row := 22) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1575, ← c1576, ← c1577] at e_22_14
  have hr := e_22_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 22 63) = band (a (.virt 17987)) (a (.virt 17989)) := by
  have c1578 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1580 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_22_15 := arithEq_of_rows h (row := 22) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_15
  simp only [← c1578, ← c1579, ← c1580] at e_22_15
  have hr := e_22_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 22 67) = band (a (.wire 22 59)) (a (.wire 22 63)) := by
  have c1581 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1583 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_22_16 := arithEq_of_rows h (row := 22) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_16
  simp only [← c1581, ← c1582, ← c1583] at e_22_16
  have hr := e_22_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4605)) (a (.virt 200)) (a (.virt 17991)) (a (.virt 17992)) := by
  have c1584 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1586 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1590 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1592 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1593 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1595 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1597 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_22_17 := arithEq_of_rows h (row := 22) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_17
  simp only [← c1587, ← c1588, ← c1589] at e_22_17
  have e_22_18 := arithEq_of_rows h (row := 22) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_18
  simp only [← c1590, ← c1591, ← c1592] at e_22_18
  have e_23_4 := arithEq_of_rows h (row := 23) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_4
  simp only [← c1584, k3, ← c1585, k3, ← c1586] at e_23_4
  have e_23_5 := arithEq_of_rows h (row := 23) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_5
  simp only [← c1593, ← c1594, k3, ← c1595] at e_23_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_17
    linear_combination c1596.trans k2 - hc
  · have hc := e_23_5
    simp only [e_23_4, e_22_18] at hc
    linear_combination c1597.trans k2 - hc

theorem publicBatchWrapper89_f251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4606)) (a (.virt 200)) (a (.virt 17993)) (a (.virt 17994)) := by
  have c1598 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1599 := (publicBatchWrapper89_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (publicBatchWrapper89_copies50 a h).1
  have c1601 := (publicBatchWrapper89_copies50 a h).2.1
  have c1602 := (publicBatchWrapper89_copies50 a h).2.2.1
  have c1603 := (publicBatchWrapper89_copies50 a h).2.2.2.1
  have c1604 := (publicBatchWrapper89_copies50 a h).2.2.2.2.1
  have c1605 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.1
  have c1606 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.1
  have c1607 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.1
  have c1609 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have c1610 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_22_19 := arithEq_of_rows h (row := 22) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_19
  simp only [← c1601, ← c1602, ← c1603] at e_22_19
  have e_23_6 := arithEq_of_rows h (row := 23) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_6
  simp only [← c1598, k3, ← c1599, k3, ← c1600] at e_23_6
  have e_23_7 := arithEq_of_rows h (row := 23) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_7
  simp only [← c1607, ← c1608, k3, ← c1609] at e_23_7
  have e_24_0 := arithEq_of_rows h (row := 24) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_0
  simp only [← c1604, ← c1605, ← c1606] at e_24_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_22_19
    linear_combination c1610.trans k2 - hc
  · have hc := e_23_7
    simp only [e_23_6, e_24_0] at hc
    linear_combination c1611.trans k2 - hc

theorem publicBatchWrapper89_f252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4607)) (a (.virt 200)) (a (.virt 17995)) (a (.virt 17996)) := by
  have c1612 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1613 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1614 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1616 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1617 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1619 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1620 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1622 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1623 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_23_8 := arithEq_of_rows h (row := 23) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_8
  simp only [← c1612, k3, ← c1613, k3, ← c1614] at e_23_8
  have e_23_9 := arithEq_of_rows h (row := 23) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_9
  simp only [← c1621, ← c1622, k3, ← c1623] at e_23_9
  have e_24_1 := arithEq_of_rows h (row := 24) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_1
  simp only [← c1615, ← c1616, ← c1617] at e_24_1
  have e_24_2 := arithEq_of_rows h (row := 24) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c1618, ← c1619, ← c1620] at e_24_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_1
    linear_combination c1624.trans k2 - hc
  · have hc := e_23_9
    simp only [e_23_8, e_24_2] at hc
    linear_combination c1625.trans k2 - hc

theorem publicBatchWrapper89_f253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4608)) (a (.virt 200)) (a (.virt 17997)) (a (.virt 17998)) := by
  have c1626 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1628 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1629 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1630 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1631 := (publicBatchWrapper89_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (publicBatchWrapper89_copies51 a h).1
  have c1633 := (publicBatchWrapper89_copies51 a h).2.1
  have c1634 := (publicBatchWrapper89_copies51 a h).2.2.1
  have c1635 := (publicBatchWrapper89_copies51 a h).2.2.2.1
  have c1636 := (publicBatchWrapper89_copies51 a h).2.2.2.2.1
  have c1637 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.1
  have c1638 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_23_10 := arithEq_of_rows h (row := 23) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_10
  simp only [← c1626, k3, ← c1627, k3, ← c1628] at e_23_10
  have e_23_11 := arithEq_of_rows h (row := 23) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_11
  simp only [← c1635, ← c1636, k3, ← c1637] at e_23_11
  have e_24_3 := arithEq_of_rows h (row := 24) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_3
  simp only [← c1629, ← c1630, ← c1631] at e_24_3
  have e_24_4 := arithEq_of_rows h (row := 24) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_4
  simp only [← c1632, ← c1633, ← c1634] at e_24_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_3
    linear_combination c1638.trans k2 - hc
  · have hc := e_23_11
    simp only [e_23_10, e_24_4] at hc
    linear_combination c1639.trans k2 - hc

theorem publicBatchWrapper89_f254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 24 23) = band (a (.virt 17991)) (a (.virt 17993)) := by
  have c1640 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_24_5 := arithEq_of_rows h (row := 24) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c1640, ← c1641, ← c1642] at e_24_5
  have hr := e_24_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 24 27) = band (a (.virt 17995)) (a (.virt 17997)) := by
  have c1643 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_24_6 := arithEq_of_rows h (row := 24) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_6
  simp only [← c1643, ← c1644, ← c1645] at e_24_6
  have hr := e_24_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 24 31) = band (a (.wire 24 23)) (a (.wire 24 27)) := by
  have c1646 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_24_7 := arithEq_of_rows h (row := 24) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_7
  simp only [← c1646, ← c1647, ← c1648] at e_24_7
  have hr := e_24_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4805)) (a (.virt 200)) (a (.virt 17999)) (a (.virt 18000)) := by
  have c1649 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1655 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1657 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1658 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1660 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1661 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_23_12 := arithEq_of_rows h (row := 23) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_12
  simp only [← c1649, k3, ← c1650, k3, ← c1651] at e_23_12
  have e_23_13 := arithEq_of_rows h (row := 23) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_13
  simp only [← c1658, ← c1659, k3, ← c1660] at e_23_13
  have e_24_8 := arithEq_of_rows h (row := 24) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_8
  simp only [← c1652, ← c1653, ← c1654] at e_24_8
  have e_24_9 := arithEq_of_rows h (row := 24) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_9
  simp only [← c1655, ← c1656, ← c1657] at e_24_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_8
    linear_combination c1661.trans k2 - hc
  · have hc := e_23_13
    simp only [e_23_12, e_24_9] at hc
    linear_combination c1662.trans k2 - hc

theorem publicBatchWrapper89_f258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4806)) (a (.virt 200)) (a (.virt 18001)) (a (.virt 18002)) := by
  have c1663 := (publicBatchWrapper89_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (publicBatchWrapper89_copies52 a h).1
  have c1665 := (publicBatchWrapper89_copies52 a h).2.1
  have c1666 := (publicBatchWrapper89_copies52 a h).2.2.1
  have c1667 := (publicBatchWrapper89_copies52 a h).2.2.2.1
  have c1668 := (publicBatchWrapper89_copies52 a h).2.2.2.2.1
  have c1669 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.1
  have c1670 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.1
  have c1671 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.1
  have c1674 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_23_14 := arithEq_of_rows h (row := 23) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_14
  simp only [← c1663, k3, ← c1664, k3, ← c1665] at e_23_14
  have e_23_15 := arithEq_of_rows h (row := 23) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_15
  simp only [← c1672, ← c1673, k3, ← c1674] at e_23_15
  have e_24_10 := arithEq_of_rows h (row := 24) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_10
  simp only [← c1666, ← c1667, ← c1668] at e_24_10
  have e_24_11 := arithEq_of_rows h (row := 24) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_11
  simp only [← c1669, ← c1670, ← c1671] at e_24_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_10
    linear_combination c1675.trans k2 - hc
  · have hc := e_23_15
    simp only [e_23_14, e_24_11] at hc
    linear_combination c1676.trans k2 - hc

theorem publicBatchWrapper89_f259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4807)) (a (.virt 200)) (a (.virt 18003)) (a (.virt 18004)) := by
  have c1677 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1679 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1685 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1688 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1689 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_23_16 := arithEq_of_rows h (row := 23) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_16
  simp only [← c1677, k3, ← c1678, k3, ← c1679] at e_23_16
  have e_23_17 := arithEq_of_rows h (row := 23) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_17
  simp only [← c1686, ← c1687, k3, ← c1688] at e_23_17
  have e_24_12 := arithEq_of_rows h (row := 24) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_12
  simp only [← c1680, ← c1681, ← c1682] at e_24_12
  have e_24_13 := arithEq_of_rows h (row := 24) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_13
  simp only [← c1683, ← c1684, ← c1685] at e_24_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_12
    linear_combination c1689.trans k2 - hc
  · have hc := e_23_17
    simp only [e_23_16, e_24_13] at hc
    linear_combination c1690.trans k2 - hc

theorem publicBatchWrapper89_f260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 4808)) (a (.virt 200)) (a (.virt 18005)) (a (.virt 18006)) := by
  have c1691 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1692 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1694 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (publicBatchWrapper89_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (publicBatchWrapper89_copies53 a h).1
  have c1697 := (publicBatchWrapper89_copies53 a h).2.1
  have c1698 := (publicBatchWrapper89_copies53 a h).2.2.1
  have c1699 := (publicBatchWrapper89_copies53 a h).2.2.2.1
  have c1700 := (publicBatchWrapper89_copies53 a h).2.2.2.2.1
  have c1701 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.1
  have c1702 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.1
  have c1703 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_23_18 := arithEq_of_rows h (row := 23) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_18
  simp only [← c1691, k3, ← c1692, k3, ← c1693] at e_23_18
  have e_23_19 := arithEq_of_rows h (row := 23) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_19
  simp only [← c1700, ← c1701, k3, ← c1702] at e_23_19
  have e_24_14 := arithEq_of_rows h (row := 24) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_14
  simp only [← c1694, ← c1695, ← c1696] at e_24_14
  have e_24_15 := arithEq_of_rows h (row := 24) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_15
  simp only [← c1697, ← c1698, ← c1699] at e_24_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_14
    linear_combination c1703.trans k2 - hc
  · have hc := e_23_19
    simp only [e_23_18, e_24_15] at hc
    linear_combination c1704.trans k2 - hc

theorem publicBatchWrapper89_f261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 24 67) = band (a (.virt 17999)) (a (.virt 18001)) := by
  have c1705 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_24_16 := arithEq_of_rows h (row := 24) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_16
  simp only [← c1705, ← c1706, ← c1707] at e_24_16
  have hr := e_24_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 24 71) = band (a (.virt 18003)) (a (.virt 18005)) := by
  have c1708 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1709 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_24_17 := arithEq_of_rows h (row := 24) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_17
  simp only [← c1708, ← c1709, ← c1710] at e_24_17
  have hr := e_24_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 24 75) = band (a (.wire 24 67)) (a (.wire 24 71)) := by
  have c1711 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_24_18 := arithEq_of_rows h (row := 24) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_18
  simp only [← c1711, ← c1712, ← c1713] at e_24_18
  have hr := e_24_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5005)) (a (.virt 200)) (a (.virt 18007)) (a (.virt 18008)) := by
  have c1714 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1715 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1716 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1717 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1718 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1719 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1720 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1721 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1722 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1723 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1724 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1725 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1726 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1727 := (publicBatchWrapper89_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_24_19 := arithEq_of_rows h (row := 24) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_19
  simp only [← c1717, ← c1718, ← c1719] at e_24_19
  have e_25_0 := arithEq_of_rows h (row := 25) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_0
  simp only [← c1714, k3, ← c1715, k3, ← c1716] at e_25_0
  have e_25_1 := arithEq_of_rows h (row := 25) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_1
  simp only [← c1723, ← c1724, k3, ← c1725] at e_25_1
  have e_26_0 := arithEq_of_rows h (row := 26) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1720, ← c1721, ← c1722] at e_26_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_24_19
    linear_combination c1726.trans k2 - hc
  · have hc := e_25_1
    simp only [e_25_0, e_26_0] at hc
    linear_combination c1727.trans k2 - hc

theorem publicBatchWrapper89_f265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5006)) (a (.virt 200)) (a (.virt 18009)) (a (.virt 18010)) := by
  have c1728 := (publicBatchWrapper89_copies54 a h).1
  have c1729 := (publicBatchWrapper89_copies54 a h).2.1
  have c1730 := (publicBatchWrapper89_copies54 a h).2.2.1
  have c1731 := (publicBatchWrapper89_copies54 a h).2.2.2.1
  have c1732 := (publicBatchWrapper89_copies54 a h).2.2.2.2.1
  have c1733 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.1
  have c1734 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.1
  have c1735 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.1
  have c1736 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.1
  have c1737 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.1
  have c1738 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1739 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1740 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1741 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_2 := arithEq_of_rows h (row := 25) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_2
  simp only [← c1728, k3, ← c1729, k3, ← c1730] at e_25_2
  have e_25_3 := arithEq_of_rows h (row := 25) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_3
  simp only [← c1737, ← c1738, k3, ← c1739] at e_25_3
  have e_26_1 := arithEq_of_rows h (row := 26) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1731, ← c1732, ← c1733] at e_26_1
  have e_26_2 := arithEq_of_rows h (row := 26) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1734, ← c1735, ← c1736] at e_26_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_1
    linear_combination c1740.trans k2 - hc
  · have hc := e_25_3
    simp only [e_25_2, e_26_2] at hc
    linear_combination c1741.trans k2 - hc

theorem publicBatchWrapper89_f266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5007)) (a (.virt 200)) (a (.virt 18011)) (a (.virt 18012)) := by
  have c1742 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1743 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1744 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1745 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1746 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1747 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1748 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1749 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1750 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1751 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1752 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1753 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1754 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1755 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_4 := arithEq_of_rows h (row := 25) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_4
  simp only [← c1742, k3, ← c1743, k3, ← c1744] at e_25_4
  have e_25_5 := arithEq_of_rows h (row := 25) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_5
  simp only [← c1751, ← c1752, k3, ← c1753] at e_25_5
  have e_26_3 := arithEq_of_rows h (row := 26) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1745, ← c1746, ← c1747] at e_26_3
  have e_26_4 := arithEq_of_rows h (row := 26) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1748, ← c1749, ← c1750] at e_26_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_3
    linear_combination c1754.trans k2 - hc
  · have hc := e_25_5
    simp only [e_25_4, e_26_4] at hc
    linear_combination c1755.trans k2 - hc

theorem publicBatchWrapper89_f267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5008)) (a (.virt 200)) (a (.virt 18013)) (a (.virt 18014)) := by
  have c1756 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1757 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1758 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1759 := (publicBatchWrapper89_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1760 := (publicBatchWrapper89_copies55 a h).1
  have c1761 := (publicBatchWrapper89_copies55 a h).2.1
  have c1762 := (publicBatchWrapper89_copies55 a h).2.2.1
  have c1763 := (publicBatchWrapper89_copies55 a h).2.2.2.1
  have c1764 := (publicBatchWrapper89_copies55 a h).2.2.2.2.1
  have c1765 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.1
  have c1766 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.1
  have c1767 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.1
  have c1768 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.1
  have c1769 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_6 := arithEq_of_rows h (row := 25) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_6
  simp only [← c1756, k3, ← c1757, k3, ← c1758] at e_25_6
  have e_25_7 := arithEq_of_rows h (row := 25) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_7
  simp only [← c1765, ← c1766, k3, ← c1767] at e_25_7
  have e_26_5 := arithEq_of_rows h (row := 26) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1759, ← c1760, ← c1761] at e_26_5
  have e_26_6 := arithEq_of_rows h (row := 26) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1762, ← c1763, ← c1764] at e_26_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_5
    linear_combination c1768.trans k2 - hc
  · have hc := e_25_7
    simp only [e_25_6, e_26_6] at hc
    linear_combination c1769.trans k2 - hc

theorem publicBatchWrapper89_f268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 26 31) = band (a (.virt 18007)) (a (.virt 18009)) := by
  have c1770 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1771 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1772 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_7 := arithEq_of_rows h (row := 26) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1770, ← c1771, ← c1772] at e_26_7
  have hr := e_26_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 26 35) = band (a (.virt 18011)) (a (.virt 18013)) := by
  have c1773 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1774 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1775 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_8 := arithEq_of_rows h (row := 26) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1773, ← c1774, ← c1775] at e_26_8
  have hr := e_26_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 26 39) = band (a (.wire 26 31)) (a (.wire 26 35)) := by
  have c1776 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1777 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1778 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_9 := arithEq_of_rows h (row := 26) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1776, ← c1777, ← c1778] at e_26_9
  have hr := e_26_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5205)) (a (.virt 200)) (a (.virt 18015)) (a (.virt 18016)) := by
  have c1779 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1780 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1781 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1782 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1783 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1784 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1785 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1786 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1787 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1788 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1789 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1790 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1791 := (publicBatchWrapper89_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1792 := (publicBatchWrapper89_copies56 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_8 := arithEq_of_rows h (row := 25) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_8
  simp only [← c1779, k3, ← c1780, k3, ← c1781] at e_25_8
  have e_25_9 := arithEq_of_rows h (row := 25) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_9
  simp only [← c1788, ← c1789, k3, ← c1790] at e_25_9
  have e_26_10 := arithEq_of_rows h (row := 26) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1782, ← c1783, ← c1784] at e_26_10
  have e_26_11 := arithEq_of_rows h (row := 26) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1785, ← c1786, ← c1787] at e_26_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_10
    linear_combination c1791.trans k2 - hc
  · have hc := e_25_9
    simp only [e_25_8, e_26_11] at hc
    linear_combination c1792.trans k2 - hc

theorem publicBatchWrapper89_f272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5206)) (a (.virt 200)) (a (.virt 18017)) (a (.virt 18018)) := by
  have c1793 := (publicBatchWrapper89_copies56 a h).2.1
  have c1794 := (publicBatchWrapper89_copies56 a h).2.2.1
  have c1795 := (publicBatchWrapper89_copies56 a h).2.2.2.1
  have c1796 := (publicBatchWrapper89_copies56 a h).2.2.2.2.1
  have c1797 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.1
  have c1798 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.1
  have c1799 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.1
  have c1800 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.1
  have c1801 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.1
  have c1802 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1803 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1804 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1805 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1806 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_10 := arithEq_of_rows h (row := 25) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_10
  simp only [← c1793, k3, ← c1794, k3, ← c1795] at e_25_10
  have e_25_11 := arithEq_of_rows h (row := 25) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_11
  simp only [← c1802, ← c1803, k3, ← c1804] at e_25_11
  have e_26_12 := arithEq_of_rows h (row := 26) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1796, ← c1797, ← c1798] at e_26_12
  have e_26_13 := arithEq_of_rows h (row := 26) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1799, ← c1800, ← c1801] at e_26_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_12
    linear_combination c1805.trans k2 - hc
  · have hc := e_25_11
    simp only [e_25_10, e_26_13] at hc
    linear_combination c1806.trans k2 - hc

theorem publicBatchWrapper89_f273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5207)) (a (.virt 200)) (a (.virt 18019)) (a (.virt 18020)) := by
  have c1807 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1808 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1809 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1810 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1811 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1812 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1813 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1814 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1815 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1816 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1817 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1818 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1819 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1820 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_12 := arithEq_of_rows h (row := 25) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_12
  simp only [← c1807, k3, ← c1808, k3, ← c1809] at e_25_12
  have e_25_13 := arithEq_of_rows h (row := 25) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_13
  simp only [← c1816, ← c1817, k3, ← c1818] at e_25_13
  have e_26_14 := arithEq_of_rows h (row := 26) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1810, ← c1811, ← c1812] at e_26_14
  have e_26_15 := arithEq_of_rows h (row := 26) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_15
  simp only [← c1813, ← c1814, ← c1815] at e_26_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_14
    linear_combination c1819.trans k2 - hc
  · have hc := e_25_13
    simp only [e_25_12, e_26_15] at hc
    linear_combination c1820.trans k2 - hc

theorem publicBatchWrapper89_f274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5208)) (a (.virt 200)) (a (.virt 18021)) (a (.virt 18022)) := by
  have c1821 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1822 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1823 := (publicBatchWrapper89_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1824 := (publicBatchWrapper89_copies57 a h).1
  have c1825 := (publicBatchWrapper89_copies57 a h).2.1
  have c1826 := (publicBatchWrapper89_copies57 a h).2.2.1
  have c1827 := (publicBatchWrapper89_copies57 a h).2.2.2.1
  have c1828 := (publicBatchWrapper89_copies57 a h).2.2.2.2.1
  have c1829 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.1
  have c1830 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.1
  have c1831 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.1
  have c1832 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.1
  have c1833 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.1
  have c1834 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_14 := arithEq_of_rows h (row := 25) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_14
  simp only [← c1821, k3, ← c1822, k3, ← c1823] at e_25_14
  have e_25_15 := arithEq_of_rows h (row := 25) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_15
  simp only [← c1830, ← c1831, k3, ← c1832] at e_25_15
  have e_26_16 := arithEq_of_rows h (row := 26) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_16
  simp only [← c1824, ← c1825, ← c1826] at e_26_16
  have e_26_17 := arithEq_of_rows h (row := 26) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_17
  simp only [← c1827, ← c1828, ← c1829] at e_26_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_26_16
    linear_combination c1833.trans k2 - hc
  · have hc := e_25_15
    simp only [e_25_14, e_26_17] at hc
    linear_combination c1834.trans k2 - hc

theorem publicBatchWrapper89_f275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 26 75) = band (a (.virt 18015)) (a (.virt 18017)) := by
  have c1835 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1836 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1837 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_18 := arithEq_of_rows h (row := 26) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_18
  simp only [← c1835, ← c1836, ← c1837] at e_26_18
  have hr := e_26_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 26 79) = band (a (.virt 18019)) (a (.virt 18021)) := by
  have c1838 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1839 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1840 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_19 := arithEq_of_rows h (row := 26) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_19
  simp only [← c1838, ← c1839, ← c1840] at e_26_19
  have hr := e_26_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 27 3) = band (a (.wire 26 75)) (a (.wire 26 79)) := by
  have c1841 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1842 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1843 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_0 := arithEq_of_rows h (row := 27) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1841, ← c1842, ← c1843] at e_27_0
  have hr := e_27_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5405)) (a (.virt 200)) (a (.virt 18023)) (a (.virt 18024)) := by
  have c1844 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1845 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1846 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1847 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1848 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1849 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1850 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1851 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1852 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1853 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1854 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1855 := (publicBatchWrapper89_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1856 := (publicBatchWrapper89_copies58 a h).1
  have c1857 := (publicBatchWrapper89_copies58 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_16 := arithEq_of_rows h (row := 25) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_16
  simp only [← c1844, k3, ← c1845, k3, ← c1846] at e_25_16
  have e_25_17 := arithEq_of_rows h (row := 25) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_17
  simp only [← c1853, ← c1854, k3, ← c1855] at e_25_17
  have e_27_1 := arithEq_of_rows h (row := 27) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1847, ← c1848, ← c1849] at e_27_1
  have e_27_2 := arithEq_of_rows h (row := 27) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1850, ← c1851, ← c1852] at e_27_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_1
    linear_combination c1856.trans k2 - hc
  · have hc := e_25_17
    simp only [e_25_16, e_27_2] at hc
    linear_combination c1857.trans k2 - hc

theorem publicBatchWrapper89_f279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5406)) (a (.virt 200)) (a (.virt 18025)) (a (.virt 18026)) := by
  have c1858 := (publicBatchWrapper89_copies58 a h).2.2.1
  have c1859 := (publicBatchWrapper89_copies58 a h).2.2.2.1
  have c1860 := (publicBatchWrapper89_copies58 a h).2.2.2.2.1
  have c1861 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.1
  have c1862 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.1
  have c1863 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.1
  have c1864 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.1
  have c1865 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.1
  have c1866 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1867 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1868 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1869 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1870 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1871 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_25_18 := arithEq_of_rows h (row := 25) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_18
  simp only [← c1858, k3, ← c1859, k3, ← c1860] at e_25_18
  have e_25_19 := arithEq_of_rows h (row := 25) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_19
  simp only [← c1867, ← c1868, k3, ← c1869] at e_25_19
  have e_27_3 := arithEq_of_rows h (row := 27) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1861, ← c1862, ← c1863] at e_27_3
  have e_27_4 := arithEq_of_rows h (row := 27) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_4
  simp only [← c1864, ← c1865, ← c1866] at e_27_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_3
    linear_combination c1870.trans k2 - hc
  · have hc := e_25_19
    simp only [e_25_18, e_27_4] at hc
    linear_combination c1871.trans k2 - hc

theorem publicBatchWrapper89_f280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5407)) (a (.virt 200)) (a (.virt 18027)) (a (.virt 18028)) := by
  have c1872 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1873 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1874 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1875 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1876 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1877 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1878 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1879 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1880 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1881 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1882 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1883 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1884 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1885 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_27_5 := arithEq_of_rows h (row := 27) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_5
  simp only [← c1875, ← c1876, ← c1877] at e_27_5
  have e_27_6 := arithEq_of_rows h (row := 27) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_6
  simp only [← c1878, ← c1879, ← c1880] at e_27_6
  have e_28_0 := arithEq_of_rows h (row := 28) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c1872, k3, ← c1873, k3, ← c1874] at e_28_0
  have e_28_1 := arithEq_of_rows h (row := 28) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c1881, ← c1882, k3, ← c1883] at e_28_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_5
    linear_combination c1884.trans k2 - hc
  · have hc := e_28_1
    simp only [e_28_0, e_27_6] at hc
    linear_combination c1885.trans k2 - hc

theorem publicBatchWrapper89_f281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5408)) (a (.virt 200)) (a (.virt 18029)) (a (.virt 18030)) := by
  have c1886 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1887 := (publicBatchWrapper89_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1888 := (publicBatchWrapper89_copies59 a h).1
  have c1889 := (publicBatchWrapper89_copies59 a h).2.1
  have c1890 := (publicBatchWrapper89_copies59 a h).2.2.1
  have c1891 := (publicBatchWrapper89_copies59 a h).2.2.2.1
  have c1892 := (publicBatchWrapper89_copies59 a h).2.2.2.2.1
  have c1893 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.1
  have c1894 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.1
  have c1895 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.1
  have c1896 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.1
  have c1897 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.1
  have c1898 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1899 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_27_7 := arithEq_of_rows h (row := 27) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_7
  simp only [← c1889, ← c1890, ← c1891] at e_27_7
  have e_27_8 := arithEq_of_rows h (row := 27) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_8
  simp only [← c1892, ← c1893, ← c1894] at e_27_8
  have e_28_2 := arithEq_of_rows h (row := 28) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c1886, k3, ← c1887, k3, ← c1888] at e_28_2
  have e_28_3 := arithEq_of_rows h (row := 28) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c1895, ← c1896, k3, ← c1897] at e_28_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_7
    linear_combination c1898.trans k2 - hc
  · have hc := e_28_3
    simp only [e_28_2, e_27_8] at hc
    linear_combination c1899.trans k2 - hc

theorem publicBatchWrapper89_f282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 27 39) = band (a (.virt 18023)) (a (.virt 18025)) := by
  have c1900 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1901 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1902 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_9 := arithEq_of_rows h (row := 27) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_9
  simp only [← c1900, ← c1901, ← c1902] at e_27_9
  have hr := e_27_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 27 43) = band (a (.virt 18027)) (a (.virt 18029)) := by
  have c1903 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1904 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1905 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_10 := arithEq_of_rows h (row := 27) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_10
  simp only [← c1903, ← c1904, ← c1905] at e_27_10
  have hr := e_27_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 27 47) = band (a (.wire 27 39)) (a (.wire 27 43)) := by
  have c1906 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1907 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1908 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_11 := arithEq_of_rows h (row := 27) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_11
  simp only [← c1906, ← c1907, ← c1908] at e_27_11
  have hr := e_27_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5605)) (a (.virt 200)) (a (.virt 18031)) (a (.virt 18032)) := by
  have c1909 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1910 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1911 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1912 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1913 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1914 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1915 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1916 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1917 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1918 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1919 := (publicBatchWrapper89_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1920 := (publicBatchWrapper89_copies60 a h).1
  have c1921 := (publicBatchWrapper89_copies60 a h).2.1
  have c1922 := (publicBatchWrapper89_copies60 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_27_12 := arithEq_of_rows h (row := 27) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_12
  simp only [← c1912, ← c1913, ← c1914] at e_27_12
  have e_27_13 := arithEq_of_rows h (row := 27) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_13
  simp only [← c1915, ← c1916, ← c1917] at e_27_13
  have e_28_4 := arithEq_of_rows h (row := 28) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c1909, k3, ← c1910, k3, ← c1911] at e_28_4
  have e_28_5 := arithEq_of_rows h (row := 28) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1918, ← c1919, k3, ← c1920] at e_28_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_12
    linear_combination c1921.trans k2 - hc
  · have hc := e_28_5
    simp only [e_28_4, e_27_13] at hc
    linear_combination c1922.trans k2 - hc

theorem publicBatchWrapper89_f286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5606)) (a (.virt 200)) (a (.virt 18033)) (a (.virt 18034)) := by
  have c1923 := (publicBatchWrapper89_copies60 a h).2.2.2.1
  have c1924 := (publicBatchWrapper89_copies60 a h).2.2.2.2.1
  have c1925 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.1
  have c1926 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.1
  have c1927 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.1
  have c1928 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.1
  have c1929 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.1
  have c1930 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1931 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1932 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1933 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1934 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1935 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1936 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_27_14 := arithEq_of_rows h (row := 27) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_14
  simp only [← c1926, ← c1927, ← c1928] at e_27_14
  have e_27_15 := arithEq_of_rows h (row := 27) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_15
  simp only [← c1929, ← c1930, ← c1931] at e_27_15
  have e_28_6 := arithEq_of_rows h (row := 28) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1923, k3, ← c1924, k3, ← c1925] at e_28_6
  have e_28_7 := arithEq_of_rows h (row := 28) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1932, ← c1933, k3, ← c1934] at e_28_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_14
    linear_combination c1935.trans k2 - hc
  · have hc := e_28_7
    simp only [e_28_6, e_27_15] at hc
    linear_combination c1936.trans k2 - hc

theorem publicBatchWrapper89_f287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5607)) (a (.virt 200)) (a (.virt 18035)) (a (.virt 18036)) := by
  have c1937 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1938 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1939 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1940 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1941 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1942 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1943 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1944 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1945 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1946 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1947 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1948 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1949 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1950 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_27_16 := arithEq_of_rows h (row := 27) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_16
  simp only [← c1940, ← c1941, ← c1942] at e_27_16
  have e_27_17 := arithEq_of_rows h (row := 27) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_17
  simp only [← c1943, ← c1944, ← c1945] at e_27_17
  have e_28_8 := arithEq_of_rows h (row := 28) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1937, k3, ← c1938, k3, ← c1939] at e_28_8
  have e_28_9 := arithEq_of_rows h (row := 28) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1946, ← c1947, k3, ← c1948] at e_28_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_16
    linear_combination c1949.trans k2 - hc
  · have hc := e_28_9
    simp only [e_28_8, e_27_17] at hc
    linear_combination c1950.trans k2 - hc

theorem publicBatchWrapper89_f288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5608)) (a (.virt 200)) (a (.virt 18037)) (a (.virt 18038)) := by
  have c1951 := (publicBatchWrapper89_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1952 := (publicBatchWrapper89_copies61 a h).1
  have c1953 := (publicBatchWrapper89_copies61 a h).2.1
  have c1954 := (publicBatchWrapper89_copies61 a h).2.2.1
  have c1955 := (publicBatchWrapper89_copies61 a h).2.2.2.1
  have c1956 := (publicBatchWrapper89_copies61 a h).2.2.2.2.1
  have c1957 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.1
  have c1958 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.1
  have c1959 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.1
  have c1960 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.1
  have c1961 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.1
  have c1962 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1963 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1964 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_27_18 := arithEq_of_rows h (row := 27) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_18
  simp only [← c1954, ← c1955, ← c1956] at e_27_18
  have e_27_19 := arithEq_of_rows h (row := 27) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_19
  simp only [← c1957, ← c1958, ← c1959] at e_27_19
  have e_28_10 := arithEq_of_rows h (row := 28) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1951, k3, ← c1952, k3, ← c1953] at e_28_10
  have e_28_11 := arithEq_of_rows h (row := 28) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1960, ← c1961, k3, ← c1962] at e_28_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_27_18
    linear_combination c1963.trans k2 - hc
  · have hc := e_28_11
    simp only [e_28_10, e_27_19] at hc
    linear_combination c1964.trans k2 - hc

theorem publicBatchWrapper89_f289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 29 3) = band (a (.virt 18031)) (a (.virt 18033)) := by
  have c1965 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1966 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1967 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_0 := arithEq_of_rows h (row := 29) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1965, ← c1966, ← c1967] at e_29_0
  have hr := e_29_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 29 7) = band (a (.virt 18035)) (a (.virt 18037)) := by
  have c1968 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1969 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1970 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_1 := arithEq_of_rows h (row := 29) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1968, ← c1969, ← c1970] at e_29_1
  have hr := e_29_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 29 11) = band (a (.wire 29 3)) (a (.wire 29 7)) := by
  have c1971 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1972 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1973 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_2 := arithEq_of_rows h (row := 29) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1971, ← c1972, ← c1973] at e_29_2
  have hr := e_29_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5805)) (a (.virt 200)) (a (.virt 18039)) (a (.virt 18040)) := by
  have c1974 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1975 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1976 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1977 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1978 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1979 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1980 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1981 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1982 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1983 := (publicBatchWrapper89_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1984 := (publicBatchWrapper89_copies62 a h).1
  have c1985 := (publicBatchWrapper89_copies62 a h).2.1
  have c1986 := (publicBatchWrapper89_copies62 a h).2.2.1
  have c1987 := (publicBatchWrapper89_copies62 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_28_12 := arithEq_of_rows h (row := 28) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1974, k3, ← c1975, k3, ← c1976] at e_28_12
  have e_28_13 := arithEq_of_rows h (row := 28) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1983, ← c1984, k3, ← c1985] at e_28_13
  have e_29_3 := arithEq_of_rows h (row := 29) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1977, ← c1978, ← c1979] at e_29_3
  have e_29_4 := arithEq_of_rows h (row := 29) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1980, ← c1981, ← c1982] at e_29_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_3
    linear_combination c1986.trans k2 - hc
  · have hc := e_28_13
    simp only [e_28_12, e_29_4] at hc
    linear_combination c1987.trans k2 - hc

theorem publicBatchWrapper89_f293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5806)) (a (.virt 200)) (a (.virt 18041)) (a (.virt 18042)) := by
  have c1988 := (publicBatchWrapper89_copies62 a h).2.2.2.2.1
  have c1989 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.1
  have c1990 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.1
  have c1991 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.1
  have c1992 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.1
  have c1993 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.1
  have c1994 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1995 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1996 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1997 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1998 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1999 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2000 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2001 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_28_14 := arithEq_of_rows h (row := 28) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1988, k3, ← c1989, k3, ← c1990] at e_28_14
  have e_28_15 := arithEq_of_rows h (row := 28) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_15
  simp only [← c1997, ← c1998, k3, ← c1999] at e_28_15
  have e_29_5 := arithEq_of_rows h (row := 29) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1991, ← c1992, ← c1993] at e_29_5
  have e_29_6 := arithEq_of_rows h (row := 29) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1994, ← c1995, ← c1996] at e_29_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_5
    linear_combination c2000.trans k2 - hc
  · have hc := e_28_15
    simp only [e_28_14, e_29_6] at hc
    linear_combination c2001.trans k2 - hc

theorem publicBatchWrapper89_f294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5807)) (a (.virt 200)) (a (.virt 18043)) (a (.virt 18044)) := by
  have c2002 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2003 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2004 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2005 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2006 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2007 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2008 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2009 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2010 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2011 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2012 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2013 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2014 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2015 := (publicBatchWrapper89_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_28_16 := arithEq_of_rows h (row := 28) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_16
  simp only [← c2002, k3, ← c2003, k3, ← c2004] at e_28_16
  have e_28_17 := arithEq_of_rows h (row := 28) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_17
  simp only [← c2011, ← c2012, k3, ← c2013] at e_28_17
  have e_29_7 := arithEq_of_rows h (row := 29) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c2005, ← c2006, ← c2007] at e_29_7
  have e_29_8 := arithEq_of_rows h (row := 29) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c2008, ← c2009, ← c2010] at e_29_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_7
    linear_combination c2014.trans k2 - hc
  · have hc := e_28_17
    simp only [e_28_16, e_29_8] at hc
    linear_combination c2015.trans k2 - hc

theorem publicBatchWrapper89_f295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 5808)) (a (.virt 200)) (a (.virt 18045)) (a (.virt 18046)) := by
  have c2016 := (publicBatchWrapper89_copies63 a h).1
  have c2017 := (publicBatchWrapper89_copies63 a h).2.1
  have c2018 := (publicBatchWrapper89_copies63 a h).2.2.1
  have c2019 := (publicBatchWrapper89_copies63 a h).2.2.2.1
  have c2020 := (publicBatchWrapper89_copies63 a h).2.2.2.2.1
  have c2021 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.1
  have c2022 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.1
  have c2023 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.1
  have c2024 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.1
  have c2025 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.1
  have c2026 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2027 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2028 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2029 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_28_18 := arithEq_of_rows h (row := 28) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_18
  simp only [← c2016, k3, ← c2017, k3, ← c2018] at e_28_18
  have e_28_19 := arithEq_of_rows h (row := 28) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_19
  simp only [← c2025, ← c2026, k3, ← c2027] at e_28_19
  have e_29_9 := arithEq_of_rows h (row := 29) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c2019, ← c2020, ← c2021] at e_29_9
  have e_29_10 := arithEq_of_rows h (row := 29) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c2022, ← c2023, ← c2024] at e_29_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_9
    linear_combination c2028.trans k2 - hc
  · have hc := e_28_19
    simp only [e_28_18, e_29_10] at hc
    linear_combination c2029.trans k2 - hc

theorem publicBatchWrapper89_f296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 29 47) = band (a (.virt 18039)) (a (.virt 18041)) := by
  have c2030 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2031 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2032 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_11 := arithEq_of_rows h (row := 29) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c2030, ← c2031, ← c2032] at e_29_11
  have hr := e_29_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 29 51) = band (a (.virt 18043)) (a (.virt 18045)) := by
  have c2033 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2034 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2035 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_12 := arithEq_of_rows h (row := 29) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c2033, ← c2034, ← c2035] at e_29_12
  have hr := e_29_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 29 55) = band (a (.wire 29 47)) (a (.wire 29 51)) := by
  have c2036 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2037 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2038 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_13 := arithEq_of_rows h (row := 29) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c2036, ← c2037, ← c2038] at e_29_13
  have hr := e_29_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6005)) (a (.virt 200)) (a (.virt 18047)) (a (.virt 18048)) := by
  have c2039 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2040 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2043 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2046 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2047 := (publicBatchWrapper89_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2048 := (publicBatchWrapper89_copies64 a h).1
  have c2049 := (publicBatchWrapper89_copies64 a h).2.1
  have c2050 := (publicBatchWrapper89_copies64 a h).2.2.1
  have c2051 := (publicBatchWrapper89_copies64 a h).2.2.2.1
  have c2052 := (publicBatchWrapper89_copies64 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_29_14 := arithEq_of_rows h (row := 29) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c2042, ← c2043, ← c2044] at e_29_14
  have e_29_15 := arithEq_of_rows h (row := 29) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_15
  simp only [← c2045, ← c2046, ← c2047] at e_29_15
  have e_30_0 := arithEq_of_rows h (row := 30) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c2039, k3, ← c2040, k3, ← c2041] at e_30_0
  have e_30_1 := arithEq_of_rows h (row := 30) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c2048, ← c2049, k3, ← c2050] at e_30_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_14
    linear_combination c2051.trans k2 - hc
  · have hc := e_30_1
    simp only [e_30_0, e_29_15] at hc
    linear_combination c2052.trans k2 - hc

theorem publicBatchWrapper89_f300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6006)) (a (.virt 200)) (a (.virt 18049)) (a (.virt 18050)) := by
  have c2053 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.1
  have c2054 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.1
  have c2055 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.1
  have c2056 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.1
  have c2057 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.1
  have c2058 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2059 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2060 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2061 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2062 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2063 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2064 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2065 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2066 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_29_16 := arithEq_of_rows h (row := 29) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_16
  simp only [← c2056, ← c2057, ← c2058] at e_29_16
  have e_29_17 := arithEq_of_rows h (row := 29) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_17
  simp only [← c2059, ← c2060, ← c2061] at e_29_17
  have e_30_2 := arithEq_of_rows h (row := 30) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c2053, k3, ← c2054, k3, ← c2055] at e_30_2
  have e_30_3 := arithEq_of_rows h (row := 30) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c2062, ← c2063, k3, ← c2064] at e_30_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_16
    linear_combination c2065.trans k2 - hc
  · have hc := e_30_3
    simp only [e_30_2, e_29_17] at hc
    linear_combination c2066.trans k2 - hc

theorem publicBatchWrapper89_f301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6007)) (a (.virt 200)) (a (.virt 18051)) (a (.virt 18052)) := by
  have c2067 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2068 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2069 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2070 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2071 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2072 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2073 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2074 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2075 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2076 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2077 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2078 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2079 := (publicBatchWrapper89_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2080 := (publicBatchWrapper89_copies65 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_29_18 := arithEq_of_rows h (row := 29) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_18
  simp only [← c2070, ← c2071, ← c2072] at e_29_18
  have e_29_19 := arithEq_of_rows h (row := 29) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_19
  simp only [← c2073, ← c2074, ← c2075] at e_29_19
  have e_30_4 := arithEq_of_rows h (row := 30) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c2067, k3, ← c2068, k3, ← c2069] at e_30_4
  have e_30_5 := arithEq_of_rows h (row := 30) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c2076, ← c2077, k3, ← c2078] at e_30_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_29_18
    linear_combination c2079.trans k2 - hc
  · have hc := e_30_5
    simp only [e_30_4, e_29_19] at hc
    linear_combination c2080.trans k2 - hc

theorem publicBatchWrapper89_f302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6008)) (a (.virt 200)) (a (.virt 18053)) (a (.virt 18054)) := by
  have c2081 := (publicBatchWrapper89_copies65 a h).2.1
  have c2082 := (publicBatchWrapper89_copies65 a h).2.2.1
  have c2083 := (publicBatchWrapper89_copies65 a h).2.2.2.1
  have c2084 := (publicBatchWrapper89_copies65 a h).2.2.2.2.1
  have c2085 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.1
  have c2086 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.1
  have c2087 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.1
  have c2088 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.1
  have c2089 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.1
  have c2090 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2091 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2092 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2093 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2094 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_6 := arithEq_of_rows h (row := 30) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c2081, k3, ← c2082, k3, ← c2083] at e_30_6
  have e_30_7 := arithEq_of_rows h (row := 30) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c2090, ← c2091, k3, ← c2092] at e_30_7
  have e_31_0 := arithEq_of_rows h (row := 31) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c2084, ← c2085, ← c2086] at e_31_0
  have e_31_1 := arithEq_of_rows h (row := 31) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c2087, ← c2088, ← c2089] at e_31_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_0
    linear_combination c2093.trans k2 - hc
  · have hc := e_30_7
    simp only [e_30_6, e_31_1] at hc
    linear_combination c2094.trans k2 - hc

theorem publicBatchWrapper89_f303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 31 11) = band (a (.virt 18047)) (a (.virt 18049)) := by
  have c2095 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2096 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2097 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_2 := arithEq_of_rows h (row := 31) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c2095, ← c2096, ← c2097] at e_31_2
  have hr := e_31_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 31 15) = band (a (.virt 18051)) (a (.virt 18053)) := by
  have c2098 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2099 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2100 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_3 := arithEq_of_rows h (row := 31) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c2098, ← c2099, ← c2100] at e_31_3
  have hr := e_31_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 31 19) = band (a (.wire 31 11)) (a (.wire 31 15)) := by
  have c2101 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2102 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2103 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_4 := arithEq_of_rows h (row := 31) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c2101, ← c2102, ← c2103] at e_31_4
  have hr := e_31_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6205)) (a (.virt 200)) (a (.virt 18055)) (a (.virt 18056)) := by
  have c2104 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2105 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2106 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2107 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2108 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2109 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2110 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2111 := (publicBatchWrapper89_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2112 := (publicBatchWrapper89_copies66 a h).1
  have c2113 := (publicBatchWrapper89_copies66 a h).2.1
  have c2114 := (publicBatchWrapper89_copies66 a h).2.2.1
  have c2115 := (publicBatchWrapper89_copies66 a h).2.2.2.1
  have c2116 := (publicBatchWrapper89_copies66 a h).2.2.2.2.1
  have c2117 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_8 := arithEq_of_rows h (row := 30) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c2104, k3, ← c2105, k3, ← c2106] at e_30_8
  have e_30_9 := arithEq_of_rows h (row := 30) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c2113, ← c2114, k3, ← c2115] at e_30_9
  have e_31_5 := arithEq_of_rows h (row := 31) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c2107, ← c2108, ← c2109] at e_31_5
  have e_31_6 := arithEq_of_rows h (row := 31) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c2110, ← c2111, ← c2112] at e_31_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_5
    linear_combination c2116.trans k2 - hc
  · have hc := e_30_9
    simp only [e_30_8, e_31_6] at hc
    linear_combination c2117.trans k2 - hc

theorem publicBatchWrapper89_f307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6206)) (a (.virt 200)) (a (.virt 18057)) (a (.virt 18058)) := by
  have c2118 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.1
  have c2119 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.1
  have c2120 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.1
  have c2121 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.1
  have c2122 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2123 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2124 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2125 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2126 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2127 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2128 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2129 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2130 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2131 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_10 := arithEq_of_rows h (row := 30) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c2118, k3, ← c2119, k3, ← c2120] at e_30_10
  have e_30_11 := arithEq_of_rows h (row := 30) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c2127, ← c2128, k3, ← c2129] at e_30_11
  have e_31_7 := arithEq_of_rows h (row := 31) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c2121, ← c2122, ← c2123] at e_31_7
  have e_31_8 := arithEq_of_rows h (row := 31) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c2124, ← c2125, ← c2126] at e_31_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_7
    linear_combination c2130.trans k2 - hc
  · have hc := e_30_11
    simp only [e_30_10, e_31_8] at hc
    linear_combination c2131.trans k2 - hc

theorem publicBatchWrapper89_f308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6207)) (a (.virt 200)) (a (.virt 18059)) (a (.virt 18060)) := by
  have c2132 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2133 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2134 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2135 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2136 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2137 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2138 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2139 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2140 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2141 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2142 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2143 := (publicBatchWrapper89_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2144 := (publicBatchWrapper89_copies67 a h).1
  have c2145 := (publicBatchWrapper89_copies67 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_12 := arithEq_of_rows h (row := 30) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c2132, k3, ← c2133, k3, ← c2134] at e_30_12
  have e_30_13 := arithEq_of_rows h (row := 30) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c2141, ← c2142, k3, ← c2143] at e_30_13
  have e_31_9 := arithEq_of_rows h (row := 31) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c2135, ← c2136, ← c2137] at e_31_9
  have e_31_10 := arithEq_of_rows h (row := 31) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c2138, ← c2139, ← c2140] at e_31_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_9
    linear_combination c2144.trans k2 - hc
  · have hc := e_30_13
    simp only [e_30_12, e_31_10] at hc
    linear_combination c2145.trans k2 - hc

theorem publicBatchWrapper89_f309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6208)) (a (.virt 200)) (a (.virt 18061)) (a (.virt 18062)) := by
  have c2146 := (publicBatchWrapper89_copies67 a h).2.2.1
  have c2147 := (publicBatchWrapper89_copies67 a h).2.2.2.1
  have c2148 := (publicBatchWrapper89_copies67 a h).2.2.2.2.1
  have c2149 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.1
  have c2150 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.1
  have c2151 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.1
  have c2152 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.1
  have c2153 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.1
  have c2154 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2155 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2156 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2157 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2158 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2159 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_14 := arithEq_of_rows h (row := 30) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c2146, k3, ← c2147, k3, ← c2148] at e_30_14
  have e_30_15 := arithEq_of_rows h (row := 30) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_15
  simp only [← c2155, ← c2156, k3, ← c2157] at e_30_15
  have e_31_11 := arithEq_of_rows h (row := 31) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c2149, ← c2150, ← c2151] at e_31_11
  have e_31_12 := arithEq_of_rows h (row := 31) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c2152, ← c2153, ← c2154] at e_31_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_11
    linear_combination c2158.trans k2 - hc
  · have hc := e_30_15
    simp only [e_30_14, e_31_12] at hc
    linear_combination c2159.trans k2 - hc

theorem publicBatchWrapper89_f310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 31 55) = band (a (.virt 18055)) (a (.virt 18057)) := by
  have c2160 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2161 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2162 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_13 := arithEq_of_rows h (row := 31) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c2160, ← c2161, ← c2162] at e_31_13
  have hr := e_31_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 31 59) = band (a (.virt 18059)) (a (.virt 18061)) := by
  have c2163 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2164 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2165 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_14 := arithEq_of_rows h (row := 31) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c2163, ← c2164, ← c2165] at e_31_14
  have hr := e_31_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 31 63) = band (a (.wire 31 55)) (a (.wire 31 59)) := by
  have c2166 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2167 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2168 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_31_15 := arithEq_of_rows h (row := 31) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_15
  simp only [← c2166, ← c2167, ← c2168] at e_31_15
  have hr := e_31_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6405)) (a (.virt 200)) (a (.virt 18063)) (a (.virt 18064)) := by
  have c2169 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2170 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2171 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2172 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2173 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2174 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2175 := (publicBatchWrapper89_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2176 := (publicBatchWrapper89_copies68 a h).1
  have c2177 := (publicBatchWrapper89_copies68 a h).2.1
  have c2178 := (publicBatchWrapper89_copies68 a h).2.2.1
  have c2179 := (publicBatchWrapper89_copies68 a h).2.2.2.1
  have c2180 := (publicBatchWrapper89_copies68 a h).2.2.2.2.1
  have c2181 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.1
  have c2182 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_16 := arithEq_of_rows h (row := 30) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_16
  simp only [← c2169, k3, ← c2170, k3, ← c2171] at e_30_16
  have e_30_17 := arithEq_of_rows h (row := 30) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_17
  simp only [← c2178, ← c2179, k3, ← c2180] at e_30_17
  have e_31_16 := arithEq_of_rows h (row := 31) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_16
  simp only [← c2172, ← c2173, ← c2174] at e_31_16
  have e_31_17 := arithEq_of_rows h (row := 31) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_17
  simp only [← c2175, ← c2176, ← c2177] at e_31_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_16
    linear_combination c2181.trans k2 - hc
  · have hc := e_30_17
    simp only [e_30_16, e_31_17] at hc
    linear_combination c2182.trans k2 - hc

theorem publicBatchWrapper89_f314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6406)) (a (.virt 200)) (a (.virt 18065)) (a (.virt 18066)) := by
  have c2183 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.1
  have c2184 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.1
  have c2185 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.1
  have c2186 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2187 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2188 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2189 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2190 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2191 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2192 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2193 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2194 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2195 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2196 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_30_18 := arithEq_of_rows h (row := 30) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_18
  simp only [← c2183, k3, ← c2184, k3, ← c2185] at e_30_18
  have e_30_19 := arithEq_of_rows h (row := 30) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_19
  simp only [← c2192, ← c2193, k3, ← c2194] at e_30_19
  have e_31_18 := arithEq_of_rows h (row := 31) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_18
  simp only [← c2186, ← c2187, ← c2188] at e_31_18
  have e_31_19 := arithEq_of_rows h (row := 31) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_19
  simp only [← c2189, ← c2190, ← c2191] at e_31_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_31_18
    linear_combination c2195.trans k2 - hc
  · have hc := e_30_19
    simp only [e_30_18, e_31_19] at hc
    linear_combination c2196.trans k2 - hc

theorem publicBatchWrapper89_f315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6407)) (a (.virt 200)) (a (.virt 18067)) (a (.virt 18068)) := by
  have c2197 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2198 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2199 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2200 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2201 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2202 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2203 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2204 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2205 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2206 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2207 := (publicBatchWrapper89_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2208 := (publicBatchWrapper89_copies69 a h).1
  have c2209 := (publicBatchWrapper89_copies69 a h).2.1
  have c2210 := (publicBatchWrapper89_copies69 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_0 := arithEq_of_rows h (row := 32) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_0
  simp only [← c2197, k3, ← c2198, k3, ← c2199] at e_32_0
  have e_32_1 := arithEq_of_rows h (row := 32) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_1
  simp only [← c2206, ← c2207, k3, ← c2208] at e_32_1
  have e_33_0 := arithEq_of_rows h (row := 33) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c2200, ← c2201, ← c2202] at e_33_0
  have e_33_1 := arithEq_of_rows h (row := 33) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c2203, ← c2204, ← c2205] at e_33_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_0
    linear_combination c2209.trans k2 - hc
  · have hc := e_32_1
    simp only [e_32_0, e_33_1] at hc
    linear_combination c2210.trans k2 - hc

theorem publicBatchWrapper89_f316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6408)) (a (.virt 200)) (a (.virt 18069)) (a (.virt 18070)) := by
  have c2211 := (publicBatchWrapper89_copies69 a h).2.2.2.1
  have c2212 := (publicBatchWrapper89_copies69 a h).2.2.2.2.1
  have c2213 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.1
  have c2214 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.1
  have c2215 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.1
  have c2216 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.1
  have c2217 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.1
  have c2218 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2219 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2220 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2221 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2222 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2223 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2224 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_2 := arithEq_of_rows h (row := 32) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_2
  simp only [← c2211, k3, ← c2212, k3, ← c2213] at e_32_2
  have e_32_3 := arithEq_of_rows h (row := 32) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_3
  simp only [← c2220, ← c2221, k3, ← c2222] at e_32_3
  have e_33_2 := arithEq_of_rows h (row := 33) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c2214, ← c2215, ← c2216] at e_33_2
  have e_33_3 := arithEq_of_rows h (row := 33) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c2217, ← c2218, ← c2219] at e_33_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_2
    linear_combination c2223.trans k2 - hc
  · have hc := e_32_3
    simp only [e_32_2, e_33_3] at hc
    linear_combination c2224.trans k2 - hc

theorem publicBatchWrapper89_f317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 33 19) = band (a (.virt 18063)) (a (.virt 18065)) := by
  have c2225 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2226 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2227 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_4 := arithEq_of_rows h (row := 33) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c2225, ← c2226, ← c2227] at e_33_4
  have hr := e_33_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 33 23) = band (a (.virt 18067)) (a (.virt 18069)) := by
  have c2228 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2229 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2230 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_5 := arithEq_of_rows h (row := 33) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c2228, ← c2229, ← c2230] at e_33_5
  have hr := e_33_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 33 27) = band (a (.wire 33 19)) (a (.wire 33 23)) := by
  have c2231 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2232 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2233 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_6 := arithEq_of_rows h (row := 33) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c2231, ← c2232, ← c2233] at e_33_6
  have hr := e_33_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6605)) (a (.virt 200)) (a (.virt 18071)) (a (.virt 18072)) := by
  have c2234 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2235 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2236 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2237 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2238 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2239 := (publicBatchWrapper89_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2240 := (publicBatchWrapper89_copies70 a h).1
  have c2241 := (publicBatchWrapper89_copies70 a h).2.1
  have c2242 := (publicBatchWrapper89_copies70 a h).2.2.1
  have c2243 := (publicBatchWrapper89_copies70 a h).2.2.2.1
  have c2244 := (publicBatchWrapper89_copies70 a h).2.2.2.2.1
  have c2245 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.1
  have c2246 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.1
  have c2247 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_4 := arithEq_of_rows h (row := 32) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_4
  simp only [← c2234, k3, ← c2235, k3, ← c2236] at e_32_4
  have e_32_5 := arithEq_of_rows h (row := 32) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_5
  simp only [← c2243, ← c2244, k3, ← c2245] at e_32_5
  have e_33_7 := arithEq_of_rows h (row := 33) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c2237, ← c2238, ← c2239] at e_33_7
  have e_33_8 := arithEq_of_rows h (row := 33) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c2240, ← c2241, ← c2242] at e_33_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_7
    linear_combination c2246.trans k2 - hc
  · have hc := e_32_5
    simp only [e_32_4, e_33_8] at hc
    linear_combination c2247.trans k2 - hc

theorem publicBatchWrapper89_f321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6606)) (a (.virt 200)) (a (.virt 18073)) (a (.virt 18074)) := by
  have c2248 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.1
  have c2249 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.1
  have c2250 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2251 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2252 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2253 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2254 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2255 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2256 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2257 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2258 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2259 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2260 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2261 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_6 := arithEq_of_rows h (row := 32) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_6
  simp only [← c2248, k3, ← c2249, k3, ← c2250] at e_32_6
  have e_32_7 := arithEq_of_rows h (row := 32) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_7
  simp only [← c2257, ← c2258, k3, ← c2259] at e_32_7
  have e_33_9 := arithEq_of_rows h (row := 33) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c2251, ← c2252, ← c2253] at e_33_9
  have e_33_10 := arithEq_of_rows h (row := 33) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c2254, ← c2255, ← c2256] at e_33_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_9
    linear_combination c2260.trans k2 - hc
  · have hc := e_32_7
    simp only [e_32_6, e_33_10] at hc
    linear_combination c2261.trans k2 - hc

theorem publicBatchWrapper89_f322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6607)) (a (.virt 200)) (a (.virt 18075)) (a (.virt 18076)) := by
  have c2262 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2263 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2264 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2265 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2266 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2267 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2268 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2269 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2270 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2271 := (publicBatchWrapper89_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2272 := (publicBatchWrapper89_copies71 a h).1
  have c2273 := (publicBatchWrapper89_copies71 a h).2.1
  have c2274 := (publicBatchWrapper89_copies71 a h).2.2.1
  have c2275 := (publicBatchWrapper89_copies71 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_8 := arithEq_of_rows h (row := 32) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_8
  simp only [← c2262, k3, ← c2263, k3, ← c2264] at e_32_8
  have e_32_9 := arithEq_of_rows h (row := 32) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_9
  simp only [← c2271, ← c2272, k3, ← c2273] at e_32_9
  have e_33_11 := arithEq_of_rows h (row := 33) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c2265, ← c2266, ← c2267] at e_33_11
  have e_33_12 := arithEq_of_rows h (row := 33) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c2268, ← c2269, ← c2270] at e_33_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_11
    linear_combination c2274.trans k2 - hc
  · have hc := e_32_9
    simp only [e_32_8, e_33_12] at hc
    linear_combination c2275.trans k2 - hc

theorem publicBatchWrapper89_f323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6608)) (a (.virt 200)) (a (.virt 18077)) (a (.virt 18078)) := by
  have c2276 := (publicBatchWrapper89_copies71 a h).2.2.2.2.1
  have c2277 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.1
  have c2278 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.1
  have c2279 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.1
  have c2280 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.1
  have c2281 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.1
  have c2282 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2283 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2284 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2285 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2286 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2287 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2288 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2289 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_10 := arithEq_of_rows h (row := 32) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_10
  simp only [← c2276, k3, ← c2277, k3, ← c2278] at e_32_10
  have e_32_11 := arithEq_of_rows h (row := 32) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_11
  simp only [← c2285, ← c2286, k3, ← c2287] at e_32_11
  have e_33_13 := arithEq_of_rows h (row := 33) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c2279, ← c2280, ← c2281] at e_33_13
  have e_33_14 := arithEq_of_rows h (row := 33) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c2282, ← c2283, ← c2284] at e_33_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_13
    linear_combination c2288.trans k2 - hc
  · have hc := e_32_11
    simp only [e_32_10, e_33_14] at hc
    linear_combination c2289.trans k2 - hc

theorem publicBatchWrapper89_f324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 33 63) = band (a (.virt 18071)) (a (.virt 18073)) := by
  have c2290 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2291 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2292 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_15 := arithEq_of_rows h (row := 33) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_15
  simp only [← c2290, ← c2291, ← c2292] at e_33_15
  have hr := e_33_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 33 67) = band (a (.virt 18075)) (a (.virt 18077)) := by
  have c2293 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2294 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2295 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_16 := arithEq_of_rows h (row := 33) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_16
  simp only [← c2293, ← c2294, ← c2295] at e_33_16
  have hr := e_33_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 33 71) = band (a (.wire 33 63)) (a (.wire 33 67)) := by
  have c2296 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2297 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2298 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_17 := arithEq_of_rows h (row := 33) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_17
  simp only [← c2296, ← c2297, ← c2298] at e_33_17
  have hr := e_33_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6805)) (a (.virt 200)) (a (.virt 18079)) (a (.virt 18080)) := by
  have c2299 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2300 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2301 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2302 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2303 := (publicBatchWrapper89_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2304 := (publicBatchWrapper89_copies72 a h).1
  have c2305 := (publicBatchWrapper89_copies72 a h).2.1
  have c2306 := (publicBatchWrapper89_copies72 a h).2.2.1
  have c2307 := (publicBatchWrapper89_copies72 a h).2.2.2.1
  have c2308 := (publicBatchWrapper89_copies72 a h).2.2.2.2.1
  have c2309 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.1
  have c2310 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.1
  have c2311 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.1
  have c2312 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_12 := arithEq_of_rows h (row := 32) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_12
  simp only [← c2299, k3, ← c2300, k3, ← c2301] at e_32_12
  have e_32_13 := arithEq_of_rows h (row := 32) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_13
  simp only [← c2308, ← c2309, k3, ← c2310] at e_32_13
  have e_33_18 := arithEq_of_rows h (row := 33) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_18
  simp only [← c2302, ← c2303, ← c2304] at e_33_18
  have e_33_19 := arithEq_of_rows h (row := 33) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_19
  simp only [← c2305, ← c2306, ← c2307] at e_33_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_33_18
    linear_combination c2311.trans k2 - hc
  · have hc := e_32_13
    simp only [e_32_12, e_33_19] at hc
    linear_combination c2312.trans k2 - hc

theorem publicBatchWrapper89_f328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6806)) (a (.virt 200)) (a (.virt 18081)) (a (.virt 18082)) := by
  have c2313 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.1
  have c2314 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2315 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2316 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2317 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2318 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2319 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2320 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2321 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2322 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2323 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2324 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2325 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2326 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_14 := arithEq_of_rows h (row := 32) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_14
  simp only [← c2313, k3, ← c2314, k3, ← c2315] at e_32_14
  have e_32_15 := arithEq_of_rows h (row := 32) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_15
  simp only [← c2322, ← c2323, k3, ← c2324] at e_32_15
  have e_34_0 := arithEq_of_rows h (row := 34) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c2316, ← c2317, ← c2318] at e_34_0
  have e_34_1 := arithEq_of_rows h (row := 34) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_1
  simp only [← c2319, ← c2320, ← c2321] at e_34_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_0
    linear_combination c2325.trans k2 - hc
  · have hc := e_32_15
    simp only [e_32_14, e_34_1] at hc
    linear_combination c2326.trans k2 - hc

theorem publicBatchWrapper89_f329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6807)) (a (.virt 200)) (a (.virt 18083)) (a (.virt 18084)) := by
  have c2327 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2328 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2329 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2330 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2331 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2332 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2333 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2334 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2335 := (publicBatchWrapper89_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2336 := (publicBatchWrapper89_copies73 a h).1
  have c2337 := (publicBatchWrapper89_copies73 a h).2.1
  have c2338 := (publicBatchWrapper89_copies73 a h).2.2.1
  have c2339 := (publicBatchWrapper89_copies73 a h).2.2.2.1
  have c2340 := (publicBatchWrapper89_copies73 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_16 := arithEq_of_rows h (row := 32) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_16
  simp only [← c2327, k3, ← c2328, k3, ← c2329] at e_32_16
  have e_32_17 := arithEq_of_rows h (row := 32) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_17
  simp only [← c2336, ← c2337, k3, ← c2338] at e_32_17
  have e_34_2 := arithEq_of_rows h (row := 34) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_2
  simp only [← c2330, ← c2331, ← c2332] at e_34_2
  have e_34_3 := arithEq_of_rows h (row := 34) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c2333, ← c2334, ← c2335] at e_34_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_2
    linear_combination c2339.trans k2 - hc
  · have hc := e_32_17
    simp only [e_32_16, e_34_3] at hc
    linear_combination c2340.trans k2 - hc

theorem publicBatchWrapper89_f330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 6808)) (a (.virt 200)) (a (.virt 18085)) (a (.virt 18086)) := by
  have c2341 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.1
  have c2342 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.1
  have c2343 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.1
  have c2344 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.1
  have c2345 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.1
  have c2346 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2347 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2348 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2349 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2350 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2351 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2352 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2353 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2354 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_32_18 := arithEq_of_rows h (row := 32) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_18
  simp only [← c2341, k3, ← c2342, k3, ← c2343] at e_32_18
  have e_32_19 := arithEq_of_rows h (row := 32) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_19
  simp only [← c2350, ← c2351, k3, ← c2352] at e_32_19
  have e_34_4 := arithEq_of_rows h (row := 34) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_4
  simp only [← c2344, ← c2345, ← c2346] at e_34_4
  have e_34_5 := arithEq_of_rows h (row := 34) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_5
  simp only [← c2347, ← c2348, ← c2349] at e_34_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_4
    linear_combination c2353.trans k2 - hc
  · have hc := e_32_19
    simp only [e_32_18, e_34_5] at hc
    linear_combination c2354.trans k2 - hc

theorem publicBatchWrapper89_f331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 34 27) = band (a (.virt 18079)) (a (.virt 18081)) := by
  have c2355 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2356 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2357 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_6 := arithEq_of_rows h (row := 34) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_6
  simp only [← c2355, ← c2356, ← c2357] at e_34_6
  have hr := e_34_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 34 31) = band (a (.virt 18083)) (a (.virt 18085)) := by
  have c2358 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2359 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2360 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_7 := arithEq_of_rows h (row := 34) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_7
  simp only [← c2358, ← c2359, ← c2360] at e_34_7
  have hr := e_34_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 34 35) = band (a (.wire 34 27)) (a (.wire 34 31)) := by
  have c2361 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2362 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2363 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_8 := arithEq_of_rows h (row := 34) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_8
  simp only [← c2361, ← c2362, ← c2363] at e_34_8
  have hr := e_34_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7005)) (a (.virt 200)) (a (.virt 18087)) (a (.virt 18088)) := by
  have c2364 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2365 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2366 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2367 := (publicBatchWrapper89_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2368 := (publicBatchWrapper89_copies74 a h).1
  have c2369 := (publicBatchWrapper89_copies74 a h).2.1
  have c2370 := (publicBatchWrapper89_copies74 a h).2.2.1
  have c2371 := (publicBatchWrapper89_copies74 a h).2.2.2.1
  have c2372 := (publicBatchWrapper89_copies74 a h).2.2.2.2.1
  have c2373 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.1
  have c2374 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.1
  have c2375 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.1
  have c2376 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.1
  have c2377 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_34_9 := arithEq_of_rows h (row := 34) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_9
  simp only [← c2367, ← c2368, ← c2369] at e_34_9
  have e_34_10 := arithEq_of_rows h (row := 34) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_10
  simp only [← c2370, ← c2371, ← c2372] at e_34_10
  have e_35_0 := arithEq_of_rows h (row := 35) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_0
  simp only [← c2364, k3, ← c2365, k3, ← c2366] at e_35_0
  have e_35_1 := arithEq_of_rows h (row := 35) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_1
  simp only [← c2373, ← c2374, k3, ← c2375] at e_35_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_9
    linear_combination c2376.trans k2 - hc
  · have hc := e_35_1
    simp only [e_35_0, e_34_10] at hc
    linear_combination c2377.trans k2 - hc

theorem publicBatchWrapper89_f335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7006)) (a (.virt 200)) (a (.virt 18089)) (a (.virt 18090)) := by
  have c2378 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2379 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2380 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2381 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2382 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2383 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2384 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2385 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2386 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2387 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2388 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2389 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2390 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2391 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_34_11 := arithEq_of_rows h (row := 34) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_11
  simp only [← c2381, ← c2382, ← c2383] at e_34_11
  have e_34_12 := arithEq_of_rows h (row := 34) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_12
  simp only [← c2384, ← c2385, ← c2386] at e_34_12
  have e_35_2 := arithEq_of_rows h (row := 35) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_2
  simp only [← c2378, k3, ← c2379, k3, ← c2380] at e_35_2
  have e_35_3 := arithEq_of_rows h (row := 35) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_3
  simp only [← c2387, ← c2388, k3, ← c2389] at e_35_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_11
    linear_combination c2390.trans k2 - hc
  · have hc := e_35_3
    simp only [e_35_2, e_34_12] at hc
    linear_combination c2391.trans k2 - hc

theorem publicBatchWrapper89_f336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7007)) (a (.virt 200)) (a (.virt 18091)) (a (.virt 18092)) := by
  have c2392 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2393 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2394 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2395 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2396 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2397 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2398 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2399 := (publicBatchWrapper89_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2400 := (publicBatchWrapper89_copies75 a h).1
  have c2401 := (publicBatchWrapper89_copies75 a h).2.1
  have c2402 := (publicBatchWrapper89_copies75 a h).2.2.1
  have c2403 := (publicBatchWrapper89_copies75 a h).2.2.2.1
  have c2404 := (publicBatchWrapper89_copies75 a h).2.2.2.2.1
  have c2405 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_34_13 := arithEq_of_rows h (row := 34) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_13
  simp only [← c2395, ← c2396, ← c2397] at e_34_13
  have e_34_14 := arithEq_of_rows h (row := 34) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_14
  simp only [← c2398, ← c2399, ← c2400] at e_34_14
  have e_35_4 := arithEq_of_rows h (row := 35) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_4
  simp only [← c2392, k3, ← c2393, k3, ← c2394] at e_35_4
  have e_35_5 := arithEq_of_rows h (row := 35) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_5
  simp only [← c2401, ← c2402, k3, ← c2403] at e_35_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_13
    linear_combination c2404.trans k2 - hc
  · have hc := e_35_5
    simp only [e_35_4, e_34_14] at hc
    linear_combination c2405.trans k2 - hc

theorem publicBatchWrapper89_f337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7008)) (a (.virt 200)) (a (.virt 18093)) (a (.virt 18094)) := by
  have c2406 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.1
  have c2407 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.1
  have c2408 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.1
  have c2409 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.1
  have c2410 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2411 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2412 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2413 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2414 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2415 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2416 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2417 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2418 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2419 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_34_15 := arithEq_of_rows h (row := 34) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_15
  simp only [← c2409, ← c2410, ← c2411] at e_34_15
  have e_34_16 := arithEq_of_rows h (row := 34) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_16
  simp only [← c2412, ← c2413, ← c2414] at e_34_16
  have e_35_6 := arithEq_of_rows h (row := 35) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_6
  simp only [← c2406, k3, ← c2407, k3, ← c2408] at e_35_6
  have e_35_7 := arithEq_of_rows h (row := 35) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_7
  simp only [← c2415, ← c2416, k3, ← c2417] at e_35_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_34_15
    linear_combination c2418.trans k2 - hc
  · have hc := e_35_7
    simp only [e_35_6, e_34_16] at hc
    linear_combination c2419.trans k2 - hc

theorem publicBatchWrapper89_f338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 34 71) = band (a (.virt 18087)) (a (.virt 18089)) := by
  have c2420 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2421 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2422 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_17 := arithEq_of_rows h (row := 34) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_17
  simp only [← c2420, ← c2421, ← c2422] at e_34_17
  have hr := e_34_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 34 75) = band (a (.virt 18091)) (a (.virt 18093)) := by
  have c2423 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2424 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2425 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_18 := arithEq_of_rows h (row := 34) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_18
  simp only [← c2423, ← c2424, ← c2425] at e_34_18
  have hr := e_34_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 34 79) = band (a (.wire 34 71)) (a (.wire 34 75)) := by
  have c2426 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2427 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2428 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_19 := arithEq_of_rows h (row := 34) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_19
  simp only [← c2426, ← c2427, ← c2428] at e_34_19
  have hr := e_34_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7205)) (a (.virt 200)) (a (.virt 18095)) (a (.virt 18096)) := by
  have c2429 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2430 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2431 := (publicBatchWrapper89_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2432 := (publicBatchWrapper89_copies76 a h).1
  have c2433 := (publicBatchWrapper89_copies76 a h).2.1
  have c2434 := (publicBatchWrapper89_copies76 a h).2.2.1
  have c2435 := (publicBatchWrapper89_copies76 a h).2.2.2.1
  have c2436 := (publicBatchWrapper89_copies76 a h).2.2.2.2.1
  have c2437 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.1
  have c2438 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.1
  have c2439 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.1
  have c2440 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.1
  have c2441 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.1
  have c2442 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_35_8 := arithEq_of_rows h (row := 35) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_8
  simp only [← c2429, k3, ← c2430, k3, ← c2431] at e_35_8
  have e_35_9 := arithEq_of_rows h (row := 35) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_9
  simp only [← c2438, ← c2439, k3, ← c2440] at e_35_9
  have e_36_0 := arithEq_of_rows h (row := 36) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_0
  simp only [← c2432, ← c2433, ← c2434] at e_36_0
  have e_36_1 := arithEq_of_rows h (row := 36) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_1
  simp only [← c2435, ← c2436, ← c2437] at e_36_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_0
    linear_combination c2441.trans k2 - hc
  · have hc := e_35_9
    simp only [e_35_8, e_36_1] at hc
    linear_combination c2442.trans k2 - hc

theorem publicBatchWrapper89_f342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7206)) (a (.virt 200)) (a (.virt 18097)) (a (.virt 18098)) := by
  have c2443 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2444 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2445 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2446 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2447 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2448 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2449 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2450 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2451 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2452 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2453 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2454 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2455 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2456 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_35_10 := arithEq_of_rows h (row := 35) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_10
  simp only [← c2443, k3, ← c2444, k3, ← c2445] at e_35_10
  have e_35_11 := arithEq_of_rows h (row := 35) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_11
  simp only [← c2452, ← c2453, k3, ← c2454] at e_35_11
  have e_36_2 := arithEq_of_rows h (row := 36) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_2
  simp only [← c2446, ← c2447, ← c2448] at e_36_2
  have e_36_3 := arithEq_of_rows h (row := 36) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_3
  simp only [← c2449, ← c2450, ← c2451] at e_36_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_2
    linear_combination c2455.trans k2 - hc
  · have hc := e_35_11
    simp only [e_35_10, e_36_3] at hc
    linear_combination c2456.trans k2 - hc

theorem publicBatchWrapper89_f343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7207)) (a (.virt 200)) (a (.virt 18099)) (a (.virt 18100)) := by
  have c2457 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2458 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2459 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2460 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2461 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2462 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2463 := (publicBatchWrapper89_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2464 := (publicBatchWrapper89_copies77 a h).1
  have c2465 := (publicBatchWrapper89_copies77 a h).2.1
  have c2466 := (publicBatchWrapper89_copies77 a h).2.2.1
  have c2467 := (publicBatchWrapper89_copies77 a h).2.2.2.1
  have c2468 := (publicBatchWrapper89_copies77 a h).2.2.2.2.1
  have c2469 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.1
  have c2470 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_35_12 := arithEq_of_rows h (row := 35) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_12
  simp only [← c2457, k3, ← c2458, k3, ← c2459] at e_35_12
  have e_35_13 := arithEq_of_rows h (row := 35) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_13
  simp only [← c2466, ← c2467, k3, ← c2468] at e_35_13
  have e_36_4 := arithEq_of_rows h (row := 36) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_4
  simp only [← c2460, ← c2461, ← c2462] at e_36_4
  have e_36_5 := arithEq_of_rows h (row := 36) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_5
  simp only [← c2463, ← c2464, ← c2465] at e_36_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_4
    linear_combination c2469.trans k2 - hc
  · have hc := e_35_13
    simp only [e_35_12, e_36_5] at hc
    linear_combination c2470.trans k2 - hc

theorem publicBatchWrapper89_f344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7208)) (a (.virt 200)) (a (.virt 18101)) (a (.virt 18102)) := by
  have c2471 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.1
  have c2472 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.1
  have c2473 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.1
  have c2474 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2475 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2476 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2477 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2478 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2479 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2480 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2481 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2482 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2483 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2484 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_35_14 := arithEq_of_rows h (row := 35) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_14
  simp only [← c2471, k3, ← c2472, k3, ← c2473] at e_35_14
  have e_35_15 := arithEq_of_rows h (row := 35) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_15
  simp only [← c2480, ← c2481, k3, ← c2482] at e_35_15
  have e_36_6 := arithEq_of_rows h (row := 36) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_6
  simp only [← c2474, ← c2475, ← c2476] at e_36_6
  have e_36_7 := arithEq_of_rows h (row := 36) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_7
  simp only [← c2477, ← c2478, ← c2479] at e_36_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_6
    linear_combination c2483.trans k2 - hc
  · have hc := e_35_15
    simp only [e_35_14, e_36_7] at hc
    linear_combination c2484.trans k2 - hc

theorem publicBatchWrapper89_f345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 36 35) = band (a (.virt 18095)) (a (.virt 18097)) := by
  have c2485 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2486 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2487 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_36_8 := arithEq_of_rows h (row := 36) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_8
  simp only [← c2485, ← c2486, ← c2487] at e_36_8
  have hr := e_36_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 36 39) = band (a (.virt 18099)) (a (.virt 18101)) := by
  have c2488 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2489 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2490 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_36_9 := arithEq_of_rows h (row := 36) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_9
  simp only [← c2488, ← c2489, ← c2490] at e_36_9
  have hr := e_36_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 36 43) = band (a (.wire 36 35)) (a (.wire 36 39)) := by
  have c2491 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2492 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2493 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_36_10 := arithEq_of_rows h (row := 36) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_10
  simp only [← c2491, ← c2492, ← c2493] at e_36_10
  have hr := e_36_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7405)) (a (.virt 200)) (a (.virt 18103)) (a (.virt 18104)) := by
  have c2494 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2495 := (publicBatchWrapper89_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2496 := (publicBatchWrapper89_copies78 a h).1
  have c2497 := (publicBatchWrapper89_copies78 a h).2.1
  have c2498 := (publicBatchWrapper89_copies78 a h).2.2.1
  have c2499 := (publicBatchWrapper89_copies78 a h).2.2.2.1
  have c2500 := (publicBatchWrapper89_copies78 a h).2.2.2.2.1
  have c2501 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.1
  have c2502 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.1
  have c2503 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.1
  have c2504 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.1
  have c2505 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.1
  have c2506 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2507 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_35_16 := arithEq_of_rows h (row := 35) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_16
  simp only [← c2494, k3, ← c2495, k3, ← c2496] at e_35_16
  have e_35_17 := arithEq_of_rows h (row := 35) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_17
  simp only [← c2503, ← c2504, k3, ← c2505] at e_35_17
  have e_36_11 := arithEq_of_rows h (row := 36) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_11
  simp only [← c2497, ← c2498, ← c2499] at e_36_11
  have e_36_12 := arithEq_of_rows h (row := 36) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_12
  simp only [← c2500, ← c2501, ← c2502] at e_36_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_11
    linear_combination c2506.trans k2 - hc
  · have hc := e_35_17
    simp only [e_35_16, e_36_12] at hc
    linear_combination c2507.trans k2 - hc

theorem publicBatchWrapper89_f349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7406)) (a (.virt 200)) (a (.virt 18105)) (a (.virt 18106)) := by
  have c2508 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2509 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2510 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2511 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2512 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2513 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2514 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2515 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2516 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2517 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2518 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2519 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_35_18 := arithEq_of_rows h (row := 35) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_18
  simp only [← c2508, k3, ← c2509, k3, ← c2510] at e_35_18
  have e_35_19 := arithEq_of_rows h (row := 35) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_19
  simp only [← c2517, ← c2518, k3, ← c2519] at e_35_19
  have e_36_13 := arithEq_of_rows h (row := 36) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_13
  simp only [← c2511, ← c2512, ← c2513] at e_36_13
  have e_36_14 := arithEq_of_rows h (row := 36) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_14
  simp only [← c2514, ← c2515, ← c2516] at e_36_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_13
    linear_combination c2520.trans k2 - hc
  · have hc := e_35_19
    simp only [e_35_18, e_36_14] at hc
    linear_combination c2521.trans k2 - hc

theorem publicBatchWrapper89_f350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7407)) (a (.virt 200)) (a (.virt 18107)) (a (.virt 18108)) := by
  have c2522 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2524 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2525 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2526 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2527 := (publicBatchWrapper89_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2528 := (publicBatchWrapper89_copies79 a h).1
  have c2529 := (publicBatchWrapper89_copies79 a h).2.1
  have c2530 := (publicBatchWrapper89_copies79 a h).2.2.1
  have c2531 := (publicBatchWrapper89_copies79 a h).2.2.2.1
  have c2532 := (publicBatchWrapper89_copies79 a h).2.2.2.2.1
  have c2533 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.1
  have c2534 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.1
  have c2535 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_36_15 := arithEq_of_rows h (row := 36) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_15
  simp only [← c2525, ← c2526, ← c2527] at e_36_15
  have e_36_16 := arithEq_of_rows h (row := 36) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_16
  simp only [← c2528, ← c2529, ← c2530] at e_36_16
  have e_37_0 := arithEq_of_rows h (row := 37) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_0
  simp only [← c2522, k3, ← c2523, k3, ← c2524] at e_37_0
  have e_37_1 := arithEq_of_rows h (row := 37) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_1
  simp only [← c2531, ← c2532, k3, ← c2533] at e_37_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_15
    linear_combination c2534.trans k2 - hc
  · have hc := e_37_1
    simp only [e_37_0, e_36_16] at hc
    linear_combination c2535.trans k2 - hc

theorem publicBatchWrapper89_f351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7408)) (a (.virt 200)) (a (.virt 18109)) (a (.virt 18110)) := by
  have c2536 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.1
  have c2537 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.1
  have c2538 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2539 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2540 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2541 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2542 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2543 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2544 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2545 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2546 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2547 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2548 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2549 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_36_17 := arithEq_of_rows h (row := 36) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_17
  simp only [← c2539, ← c2540, ← c2541] at e_36_17
  have e_36_18 := arithEq_of_rows h (row := 36) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_18
  simp only [← c2542, ← c2543, ← c2544] at e_36_18
  have e_37_2 := arithEq_of_rows h (row := 37) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_2
  simp only [← c2536, k3, ← c2537, k3, ← c2538] at e_37_2
  have e_37_3 := arithEq_of_rows h (row := 37) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_3
  simp only [← c2545, ← c2546, k3, ← c2547] at e_37_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_36_17
    linear_combination c2548.trans k2 - hc
  · have hc := e_37_3
    simp only [e_37_2, e_36_18] at hc
    linear_combination c2549.trans k2 - hc

theorem publicBatchWrapper89_f352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 36 79) = band (a (.virt 18103)) (a (.virt 18105)) := by
  have c2550 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2551 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2552 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_36_19 := arithEq_of_rows h (row := 36) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_19
  simp only [← c2550, ← c2551, ← c2552] at e_36_19
  have hr := e_36_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 38 3) = band (a (.virt 18107)) (a (.virt 18109)) := by
  have c2553 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2554 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2555 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_0 := arithEq_of_rows h (row := 38) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_0
  simp only [← c2553, ← c2554, ← c2555] at e_38_0
  have hr := e_38_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 38 7) = band (a (.wire 36 79)) (a (.wire 38 3)) := by
  have c2556 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2557 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2558 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_1 := arithEq_of_rows h (row := 38) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_1
  simp only [← c2556, ← c2557, ← c2558] at e_38_1
  have hr := e_38_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7605)) (a (.virt 200)) (a (.virt 18111)) (a (.virt 18112)) := by
  have c2559 := (publicBatchWrapper89_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2560 := (publicBatchWrapper89_copies80 a h).1
  have c2561 := (publicBatchWrapper89_copies80 a h).2.1
  have c2562 := (publicBatchWrapper89_copies80 a h).2.2.1
  have c2563 := (publicBatchWrapper89_copies80 a h).2.2.2.1
  have c2564 := (publicBatchWrapper89_copies80 a h).2.2.2.2.1
  have c2565 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.1
  have c2566 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.1
  have c2567 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.1
  have c2568 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.1
  have c2569 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.1
  have c2570 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2571 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2572 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_4 := arithEq_of_rows h (row := 37) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_4
  simp only [← c2559, k3, ← c2560, k3, ← c2561] at e_37_4
  have e_37_5 := arithEq_of_rows h (row := 37) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_5
  simp only [← c2568, ← c2569, k3, ← c2570] at e_37_5
  have e_38_2 := arithEq_of_rows h (row := 38) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_2
  simp only [← c2562, ← c2563, ← c2564] at e_38_2
  have e_38_3 := arithEq_of_rows h (row := 38) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_3
  simp only [← c2565, ← c2566, ← c2567] at e_38_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_2
    linear_combination c2571.trans k2 - hc
  · have hc := e_37_5
    simp only [e_37_4, e_38_3] at hc
    linear_combination c2572.trans k2 - hc

theorem publicBatchWrapper89_f356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7606)) (a (.virt 200)) (a (.virt 18113)) (a (.virt 18114)) := by
  have c2573 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2574 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2575 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2576 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2577 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2578 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2579 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2580 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2581 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2582 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2583 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2584 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2585 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2586 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_6 := arithEq_of_rows h (row := 37) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_6
  simp only [← c2573, k3, ← c2574, k3, ← c2575] at e_37_6
  have e_37_7 := arithEq_of_rows h (row := 37) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_7
  simp only [← c2582, ← c2583, k3, ← c2584] at e_37_7
  have e_38_4 := arithEq_of_rows h (row := 38) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_4
  simp only [← c2576, ← c2577, ← c2578] at e_38_4
  have e_38_5 := arithEq_of_rows h (row := 38) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_5
  simp only [← c2579, ← c2580, ← c2581] at e_38_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_4
    linear_combination c2585.trans k2 - hc
  · have hc := e_37_7
    simp only [e_37_6, e_38_5] at hc
    linear_combination c2586.trans k2 - hc

theorem publicBatchWrapper89_f357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7607)) (a (.virt 200)) (a (.virt 18115)) (a (.virt 18116)) := by
  have c2587 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2588 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2589 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2590 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2591 := (publicBatchWrapper89_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2592 := (publicBatchWrapper89_copies81 a h).1
  have c2593 := (publicBatchWrapper89_copies81 a h).2.1
  have c2594 := (publicBatchWrapper89_copies81 a h).2.2.1
  have c2595 := (publicBatchWrapper89_copies81 a h).2.2.2.1
  have c2596 := (publicBatchWrapper89_copies81 a h).2.2.2.2.1
  have c2597 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.1
  have c2598 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.1
  have c2599 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.1
  have c2600 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_8 := arithEq_of_rows h (row := 37) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_8
  simp only [← c2587, k3, ← c2588, k3, ← c2589] at e_37_8
  have e_37_9 := arithEq_of_rows h (row := 37) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_9
  simp only [← c2596, ← c2597, k3, ← c2598] at e_37_9
  have e_38_6 := arithEq_of_rows h (row := 38) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_6
  simp only [← c2590, ← c2591, ← c2592] at e_38_6
  have e_38_7 := arithEq_of_rows h (row := 38) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_7
  simp only [← c2593, ← c2594, ← c2595] at e_38_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_6
    linear_combination c2599.trans k2 - hc
  · have hc := e_37_9
    simp only [e_37_8, e_38_7] at hc
    linear_combination c2600.trans k2 - hc

theorem publicBatchWrapper89_f358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7608)) (a (.virt 200)) (a (.virt 18117)) (a (.virt 18118)) := by
  have c2601 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.1
  have c2602 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2603 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2604 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2605 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2606 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2607 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2608 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2609 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2610 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2611 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2612 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2613 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2614 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_10 := arithEq_of_rows h (row := 37) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_10
  simp only [← c2601, k3, ← c2602, k3, ← c2603] at e_37_10
  have e_37_11 := arithEq_of_rows h (row := 37) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_11
  simp only [← c2610, ← c2611, k3, ← c2612] at e_37_11
  have e_38_8 := arithEq_of_rows h (row := 38) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_8
  simp only [← c2604, ← c2605, ← c2606] at e_38_8
  have e_38_9 := arithEq_of_rows h (row := 38) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_9
  simp only [← c2607, ← c2608, ← c2609] at e_38_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_8
    linear_combination c2613.trans k2 - hc
  · have hc := e_37_11
    simp only [e_37_10, e_38_9] at hc
    linear_combination c2614.trans k2 - hc

theorem publicBatchWrapper89_f359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 38 43) = band (a (.virt 18111)) (a (.virt 18113)) := by
  have c2615 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2616 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2617 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_10 := arithEq_of_rows h (row := 38) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_10
  simp only [← c2615, ← c2616, ← c2617] at e_38_10
  have hr := e_38_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 38 47) = band (a (.virt 18115)) (a (.virt 18117)) := by
  have c2618 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2619 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2620 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_11 := arithEq_of_rows h (row := 38) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_11
  simp only [← c2618, ← c2619, ← c2620] at e_38_11
  have hr := e_38_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 38 51) = band (a (.wire 38 43)) (a (.wire 38 47)) := by
  have c2621 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2622 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2623 := (publicBatchWrapper89_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_38_12 := arithEq_of_rows h (row := 38) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_12
  simp only [← c2621, ← c2622, ← c2623] at e_38_12
  have hr := e_38_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7805)) (a (.virt 200)) (a (.virt 18119)) (a (.virt 18120)) := by
  have c2624 := (publicBatchWrapper89_copies82 a h).1
  have c2625 := (publicBatchWrapper89_copies82 a h).2.1
  have c2626 := (publicBatchWrapper89_copies82 a h).2.2.1
  have c2627 := (publicBatchWrapper89_copies82 a h).2.2.2.1
  have c2628 := (publicBatchWrapper89_copies82 a h).2.2.2.2.1
  have c2629 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.1
  have c2630 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.1
  have c2631 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.1
  have c2632 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.1
  have c2633 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.1
  have c2634 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2635 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2636 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2637 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_12 := arithEq_of_rows h (row := 37) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_12
  simp only [← c2624, k3, ← c2625, k3, ← c2626] at e_37_12
  have e_37_13 := arithEq_of_rows h (row := 37) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_13
  simp only [← c2633, ← c2634, k3, ← c2635] at e_37_13
  have e_38_13 := arithEq_of_rows h (row := 38) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_13
  simp only [← c2627, ← c2628, ← c2629] at e_38_13
  have e_38_14 := arithEq_of_rows h (row := 38) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_14
  simp only [← c2630, ← c2631, ← c2632] at e_38_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_13
    linear_combination c2636.trans k2 - hc
  · have hc := e_37_13
    simp only [e_37_12, e_38_14] at hc
    linear_combination c2637.trans k2 - hc

theorem publicBatchWrapper89_f363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7806)) (a (.virt 200)) (a (.virt 18121)) (a (.virt 18122)) := by
  have c2638 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2639 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2640 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2641 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2642 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2643 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2644 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2645 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2646 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2647 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2648 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2649 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2650 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2651 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_14 := arithEq_of_rows h (row := 37) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_14
  simp only [← c2638, k3, ← c2639, k3, ← c2640] at e_37_14
  have e_37_15 := arithEq_of_rows h (row := 37) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_15
  simp only [← c2647, ← c2648, k3, ← c2649] at e_37_15
  have e_38_15 := arithEq_of_rows h (row := 38) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_15
  simp only [← c2641, ← c2642, ← c2643] at e_38_15
  have e_38_16 := arithEq_of_rows h (row := 38) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_16
  simp only [← c2644, ← c2645, ← c2646] at e_38_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_15
    linear_combination c2650.trans k2 - hc
  · have hc := e_37_15
    simp only [e_37_14, e_38_16] at hc
    linear_combination c2651.trans k2 - hc

theorem publicBatchWrapper89_f364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7807)) (a (.virt 200)) (a (.virt 18123)) (a (.virt 18124)) := by
  have c2652 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2653 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2654 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2655 := (publicBatchWrapper89_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2656 := (publicBatchWrapper89_copies83 a h).1
  have c2657 := (publicBatchWrapper89_copies83 a h).2.1
  have c2658 := (publicBatchWrapper89_copies83 a h).2.2.1
  have c2659 := (publicBatchWrapper89_copies83 a h).2.2.2.1
  have c2660 := (publicBatchWrapper89_copies83 a h).2.2.2.2.1
  have c2661 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.1
  have c2662 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.1
  have c2663 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.1
  have c2664 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.1
  have c2665 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_16 := arithEq_of_rows h (row := 37) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_16
  simp only [← c2652, k3, ← c2653, k3, ← c2654] at e_37_16
  have e_37_17 := arithEq_of_rows h (row := 37) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_17
  simp only [← c2661, ← c2662, k3, ← c2663] at e_37_17
  have e_38_17 := arithEq_of_rows h (row := 38) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_17
  simp only [← c2655, ← c2656, ← c2657] at e_38_17
  have e_38_18 := arithEq_of_rows h (row := 38) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_18
  simp only [← c2658, ← c2659, ← c2660] at e_38_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_17
    linear_combination c2664.trans k2 - hc
  · have hc := e_37_17
    simp only [e_37_16, e_38_18] at hc
    linear_combination c2665.trans k2 - hc

theorem publicBatchWrapper89_f365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 7808)) (a (.virt 200)) (a (.virt 18125)) (a (.virt 18126)) := by
  have c2666 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2667 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2668 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2669 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2670 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2671 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2672 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2673 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2674 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2675 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2676 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2677 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2678 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2679 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_37_18 := arithEq_of_rows h (row := 37) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_18
  simp only [← c2666, k3, ← c2667, k3, ← c2668] at e_37_18
  have e_37_19 := arithEq_of_rows h (row := 37) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_19
  simp only [← c2675, ← c2676, k3, ← c2677] at e_37_19
  have e_38_19 := arithEq_of_rows h (row := 38) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_19
  simp only [← c2669, ← c2670, ← c2671] at e_38_19
  have e_39_0 := arithEq_of_rows h (row := 39) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_0
  simp only [← c2672, ← c2673, ← c2674] at e_39_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_38_19
    linear_combination c2678.trans k2 - hc
  · have hc := e_37_19
    simp only [e_37_18, e_39_0] at hc
    linear_combination c2679.trans k2 - hc

theorem publicBatchWrapper89_f366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 39 7) = band (a (.virt 18119)) (a (.virt 18121)) := by
  have c2680 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2681 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2682 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_1 := arithEq_of_rows h (row := 39) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_1
  simp only [← c2680, ← c2681, ← c2682] at e_39_1
  have hr := e_39_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 39 11) = band (a (.virt 18123)) (a (.virt 18125)) := by
  have c2683 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2684 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2685 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_2 := arithEq_of_rows h (row := 39) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_2
  simp only [← c2683, ← c2684, ← c2685] at e_39_2
  have hr := e_39_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 39 15) = band (a (.wire 39 7)) (a (.wire 39 11)) := by
  have c2686 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2687 := (publicBatchWrapper89_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2688 := (publicBatchWrapper89_copies84 a h).1
  have e_39_3 := arithEq_of_rows h (row := 39) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_3
  simp only [← c2686, ← c2687, ← c2688] at e_39_3
  have hr := e_39_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8005)) (a (.virt 200)) (a (.virt 18127)) (a (.virt 18128)) := by
  have c2689 := (publicBatchWrapper89_copies84 a h).2.1
  have c2690 := (publicBatchWrapper89_copies84 a h).2.2.1
  have c2691 := (publicBatchWrapper89_copies84 a h).2.2.2.1
  have c2692 := (publicBatchWrapper89_copies84 a h).2.2.2.2.1
  have c2693 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.1
  have c2694 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.1
  have c2695 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.1
  have c2696 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.1
  have c2697 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.1
  have c2698 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2699 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2700 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2701 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2702 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_4 := arithEq_of_rows h (row := 39) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_4
  simp only [← c2692, ← c2693, ← c2694] at e_39_4
  have e_39_5 := arithEq_of_rows h (row := 39) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_5
  simp only [← c2695, ← c2696, ← c2697] at e_39_5
  have e_40_0 := arithEq_of_rows h (row := 40) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_0
  simp only [← c2689, k3, ← c2690, k3, ← c2691] at e_40_0
  have e_40_1 := arithEq_of_rows h (row := 40) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_1
  simp only [← c2698, ← c2699, k3, ← c2700] at e_40_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_4
    linear_combination c2701.trans k2 - hc
  · have hc := e_40_1
    simp only [e_40_0, e_39_5] at hc
    linear_combination c2702.trans k2 - hc

theorem publicBatchWrapper89_f370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8006)) (a (.virt 200)) (a (.virt 18129)) (a (.virt 18130)) := by
  have c2703 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2704 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2705 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2706 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2707 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2708 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2709 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2710 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2711 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2712 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2713 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2714 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2715 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2716 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_6 := arithEq_of_rows h (row := 39) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_6
  simp only [← c2706, ← c2707, ← c2708] at e_39_6
  have e_39_7 := arithEq_of_rows h (row := 39) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_7
  simp only [← c2709, ← c2710, ← c2711] at e_39_7
  have e_40_2 := arithEq_of_rows h (row := 40) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_2
  simp only [← c2703, k3, ← c2704, k3, ← c2705] at e_40_2
  have e_40_3 := arithEq_of_rows h (row := 40) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_3
  simp only [← c2712, ← c2713, k3, ← c2714] at e_40_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_6
    linear_combination c2715.trans k2 - hc
  · have hc := e_40_3
    simp only [e_40_2, e_39_7] at hc
    linear_combination c2716.trans k2 - hc

theorem publicBatchWrapper89_f371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8007)) (a (.virt 200)) (a (.virt 18131)) (a (.virt 18132)) := by
  have c2717 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2718 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2719 := (publicBatchWrapper89_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2720 := (publicBatchWrapper89_copies85 a h).1
  have c2721 := (publicBatchWrapper89_copies85 a h).2.1
  have c2722 := (publicBatchWrapper89_copies85 a h).2.2.1
  have c2723 := (publicBatchWrapper89_copies85 a h).2.2.2.1
  have c2724 := (publicBatchWrapper89_copies85 a h).2.2.2.2.1
  have c2725 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.1
  have c2726 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.1
  have c2727 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.1
  have c2728 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.1
  have c2729 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.1
  have c2730 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_8 := arithEq_of_rows h (row := 39) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_8
  simp only [← c2720, ← c2721, ← c2722] at e_39_8
  have e_39_9 := arithEq_of_rows h (row := 39) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_9
  simp only [← c2723, ← c2724, ← c2725] at e_39_9
  have e_40_4 := arithEq_of_rows h (row := 40) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_4
  simp only [← c2717, k3, ← c2718, k3, ← c2719] at e_40_4
  have e_40_5 := arithEq_of_rows h (row := 40) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_5
  simp only [← c2726, ← c2727, k3, ← c2728] at e_40_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_8
    linear_combination c2729.trans k2 - hc
  · have hc := e_40_5
    simp only [e_40_4, e_39_9] at hc
    linear_combination c2730.trans k2 - hc

theorem publicBatchWrapper89_f372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8008)) (a (.virt 200)) (a (.virt 18133)) (a (.virt 18134)) := by
  have c2731 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2732 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2733 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2734 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2735 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2736 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2737 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2738 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2739 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2740 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2741 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2742 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2743 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2744 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_10 := arithEq_of_rows h (row := 39) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_10
  simp only [← c2734, ← c2735, ← c2736] at e_39_10
  have e_39_11 := arithEq_of_rows h (row := 39) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_11
  simp only [← c2737, ← c2738, ← c2739] at e_39_11
  have e_40_6 := arithEq_of_rows h (row := 40) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_6
  simp only [← c2731, k3, ← c2732, k3, ← c2733] at e_40_6
  have e_40_7 := arithEq_of_rows h (row := 40) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_7
  simp only [← c2740, ← c2741, k3, ← c2742] at e_40_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_10
    linear_combination c2743.trans k2 - hc
  · have hc := e_40_7
    simp only [e_40_6, e_39_11] at hc
    linear_combination c2744.trans k2 - hc

theorem publicBatchWrapper89_f373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 39 51) = band (a (.virt 18127)) (a (.virt 18129)) := by
  have c2745 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2746 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2747 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_12 := arithEq_of_rows h (row := 39) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_12
  simp only [← c2745, ← c2746, ← c2747] at e_39_12
  have hr := e_39_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 39 55) = band (a (.virt 18131)) (a (.virt 18133)) := by
  have c2748 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2749 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2750 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_39_13 := arithEq_of_rows h (row := 39) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_13
  simp only [← c2748, ← c2749, ← c2750] at e_39_13
  have hr := e_39_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 39 59) = band (a (.wire 39 51)) (a (.wire 39 55)) := by
  have c2751 := (publicBatchWrapper89_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2752 := (publicBatchWrapper89_copies86 a h).1
  have c2753 := (publicBatchWrapper89_copies86 a h).2.1
  have e_39_14 := arithEq_of_rows h (row := 39) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_14
  simp only [← c2751, ← c2752, ← c2753] at e_39_14
  have hr := e_39_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8205)) (a (.virt 200)) (a (.virt 18135)) (a (.virt 18136)) := by
  have c2754 := (publicBatchWrapper89_copies86 a h).2.2.1
  have c2755 := (publicBatchWrapper89_copies86 a h).2.2.2.1
  have c2756 := (publicBatchWrapper89_copies86 a h).2.2.2.2.1
  have c2757 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.1
  have c2758 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.1
  have c2759 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.1
  have c2760 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.1
  have c2761 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.1
  have c2762 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2763 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2764 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2765 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2766 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2767 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_15 := arithEq_of_rows h (row := 39) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_15
  simp only [← c2757, ← c2758, ← c2759] at e_39_15
  have e_39_16 := arithEq_of_rows h (row := 39) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_16
  simp only [← c2760, ← c2761, ← c2762] at e_39_16
  have e_40_8 := arithEq_of_rows h (row := 40) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_8
  simp only [← c2754, k3, ← c2755, k3, ← c2756] at e_40_8
  have e_40_9 := arithEq_of_rows h (row := 40) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_9
  simp only [← c2763, ← c2764, k3, ← c2765] at e_40_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_15
    linear_combination c2766.trans k2 - hc
  · have hc := e_40_9
    simp only [e_40_8, e_39_16] at hc
    linear_combination c2767.trans k2 - hc

theorem publicBatchWrapper89_f377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8206)) (a (.virt 200)) (a (.virt 18137)) (a (.virt 18138)) := by
  have c2768 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2769 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2770 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2771 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2772 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2773 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2774 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2775 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2776 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2777 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2778 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2779 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2780 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2781 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_17 := arithEq_of_rows h (row := 39) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_17
  simp only [← c2771, ← c2772, ← c2773] at e_39_17
  have e_39_18 := arithEq_of_rows h (row := 39) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_18
  simp only [← c2774, ← c2775, ← c2776] at e_39_18
  have e_40_10 := arithEq_of_rows h (row := 40) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_10
  simp only [← c2768, k3, ← c2769, k3, ← c2770] at e_40_10
  have e_40_11 := arithEq_of_rows h (row := 40) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_11
  simp only [← c2777, ← c2778, k3, ← c2779] at e_40_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_17
    linear_combination c2780.trans k2 - hc
  · have hc := e_40_11
    simp only [e_40_10, e_39_18] at hc
    linear_combination c2781.trans k2 - hc

theorem publicBatchWrapper89_f378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8207)) (a (.virt 200)) (a (.virt 18139)) (a (.virt 18140)) := by
  have c2782 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2783 := (publicBatchWrapper89_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2784 := (publicBatchWrapper89_copies87 a h).1
  have c2785 := (publicBatchWrapper89_copies87 a h).2.1
  have c2786 := (publicBatchWrapper89_copies87 a h).2.2.1
  have c2787 := (publicBatchWrapper89_copies87 a h).2.2.2.1
  have c2788 := (publicBatchWrapper89_copies87 a h).2.2.2.2.1
  have c2789 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.1
  have c2790 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.1
  have c2791 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.1
  have c2792 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.1
  have c2793 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.1
  have c2794 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2795 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_39_19 := arithEq_of_rows h (row := 39) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_19
  simp only [← c2785, ← c2786, ← c2787] at e_39_19
  have e_40_12 := arithEq_of_rows h (row := 40) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_12
  simp only [← c2782, k3, ← c2783, k3, ← c2784] at e_40_12
  have e_40_13 := arithEq_of_rows h (row := 40) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_13
  simp only [← c2791, ← c2792, k3, ← c2793] at e_40_13
  have e_41_0 := arithEq_of_rows h (row := 41) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_0
  simp only [← c2788, ← c2789, ← c2790] at e_41_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_39_19
    linear_combination c2794.trans k2 - hc
  · have hc := e_40_13
    simp only [e_40_12, e_41_0] at hc
    linear_combination c2795.trans k2 - hc

theorem publicBatchWrapper89_f379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8208)) (a (.virt 200)) (a (.virt 18141)) (a (.virt 18142)) := by
  have c2796 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2797 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2798 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2799 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2800 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2801 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2802 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2803 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2804 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2805 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2806 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2807 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2808 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2809 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_40_14 := arithEq_of_rows h (row := 40) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_14
  simp only [← c2796, k3, ← c2797, k3, ← c2798] at e_40_14
  have e_40_15 := arithEq_of_rows h (row := 40) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_15
  simp only [← c2805, ← c2806, k3, ← c2807] at e_40_15
  have e_41_1 := arithEq_of_rows h (row := 41) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_1
  simp only [← c2799, ← c2800, ← c2801] at e_41_1
  have e_41_2 := arithEq_of_rows h (row := 41) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_2
  simp only [← c2802, ← c2803, ← c2804] at e_41_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_1
    linear_combination c2808.trans k2 - hc
  · have hc := e_40_15
    simp only [e_40_14, e_41_2] at hc
    linear_combination c2809.trans k2 - hc

theorem publicBatchWrapper89_f380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 41 15) = band (a (.virt 18135)) (a (.virt 18137)) := by
  have c2810 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2811 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2812 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_3 := arithEq_of_rows h (row := 41) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_3
  simp only [← c2810, ← c2811, ← c2812] at e_41_3
  have hr := e_41_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 41 19) = band (a (.virt 18139)) (a (.virt 18141)) := by
  have c2813 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2814 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2815 := (publicBatchWrapper89_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_41_4 := arithEq_of_rows h (row := 41) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_4
  simp only [← c2813, ← c2814, ← c2815] at e_41_4
  have hr := e_41_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 41 23) = band (a (.wire 41 15)) (a (.wire 41 19)) := by
  have c2816 := (publicBatchWrapper89_copies88 a h).1
  have c2817 := (publicBatchWrapper89_copies88 a h).2.1
  have c2818 := (publicBatchWrapper89_copies88 a h).2.2.1
  have e_41_5 := arithEq_of_rows h (row := 41) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_5
  simp only [← c2816, ← c2817, ← c2818] at e_41_5
  have hr := e_41_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8405)) (a (.virt 200)) (a (.virt 18143)) (a (.virt 18144)) := by
  have c2819 := (publicBatchWrapper89_copies88 a h).2.2.2.1
  have c2820 := (publicBatchWrapper89_copies88 a h).2.2.2.2.1
  have c2821 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.1
  have c2822 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.1
  have c2823 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.1
  have c2824 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.1
  have c2825 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.1
  have c2826 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2827 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2828 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2829 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2830 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2831 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2832 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_40_16 := arithEq_of_rows h (row := 40) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_16
  simp only [← c2819, k3, ← c2820, k3, ← c2821] at e_40_16
  have e_40_17 := arithEq_of_rows h (row := 40) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_17
  simp only [← c2828, ← c2829, k3, ← c2830] at e_40_17
  have e_41_6 := arithEq_of_rows h (row := 41) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_6
  simp only [← c2822, ← c2823, ← c2824] at e_41_6
  have e_41_7 := arithEq_of_rows h (row := 41) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_7
  simp only [← c2825, ← c2826, ← c2827] at e_41_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_6
    linear_combination c2831.trans k2 - hc
  · have hc := e_40_17
    simp only [e_40_16, e_41_7] at hc
    linear_combination c2832.trans k2 - hc

theorem publicBatchWrapper89_f384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8406)) (a (.virt 200)) (a (.virt 18145)) (a (.virt 18146)) := by
  have c2833 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2834 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2835 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2836 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2837 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2838 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2839 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2840 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2841 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2842 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2843 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2844 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2845 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2846 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_40_18 := arithEq_of_rows h (row := 40) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_18
  simp only [← c2833, k3, ← c2834, k3, ← c2835] at e_40_18
  have e_40_19 := arithEq_of_rows h (row := 40) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_19
  simp only [← c2842, ← c2843, k3, ← c2844] at e_40_19
  have e_41_8 := arithEq_of_rows h (row := 41) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_8
  simp only [← c2836, ← c2837, ← c2838] at e_41_8
  have e_41_9 := arithEq_of_rows h (row := 41) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_9
  simp only [← c2839, ← c2840, ← c2841] at e_41_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_8
    linear_combination c2845.trans k2 - hc
  · have hc := e_40_19
    simp only [e_40_18, e_41_9] at hc
    linear_combination c2846.trans k2 - hc

theorem publicBatchWrapper89_f385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8407)) (a (.virt 200)) (a (.virt 18147)) (a (.virt 18148)) := by
  have c2847 := (publicBatchWrapper89_copies88 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2848 := (publicBatchWrapper89_copies89 a h).1
  have c2849 := (publicBatchWrapper89_copies89 a h).2.1
  have c2850 := (publicBatchWrapper89_copies89 a h).2.2.1
  have c2851 := (publicBatchWrapper89_copies89 a h).2.2.2.1
  have c2852 := (publicBatchWrapper89_copies89 a h).2.2.2.2.1
  have c2853 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.1
  have c2854 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.1
  have c2855 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.1
  have c2856 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.1
  have c2857 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.1
  have c2858 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2859 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2860 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_41_10 := arithEq_of_rows h (row := 41) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_10
  simp only [← c2850, ← c2851, ← c2852] at e_41_10
  have e_41_11 := arithEq_of_rows h (row := 41) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_11
  simp only [← c2853, ← c2854, ← c2855] at e_41_11
  have e_42_0 := arithEq_of_rows h (row := 42) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c2847, k3, ← c2848, k3, ← c2849] at e_42_0
  have e_42_1 := arithEq_of_rows h (row := 42) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_1
  simp only [← c2856, ← c2857, k3, ← c2858] at e_42_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_10
    linear_combination c2859.trans k2 - hc
  · have hc := e_42_1
    simp only [e_42_0, e_41_11] at hc
    linear_combination c2860.trans k2 - hc

theorem publicBatchWrapper89_f386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8408)) (a (.virt 200)) (a (.virt 18149)) (a (.virt 18150)) := by
  have c2861 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2862 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2863 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2864 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2865 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2866 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2867 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2868 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2869 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2870 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2871 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2872 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2873 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2874 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_41_12 := arithEq_of_rows h (row := 41) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_12
  simp only [← c2864, ← c2865, ← c2866] at e_41_12
  have e_41_13 := arithEq_of_rows h (row := 41) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_13
  simp only [← c2867, ← c2868, ← c2869] at e_41_13
  have e_42_2 := arithEq_of_rows h (row := 42) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_2
  simp only [← c2861, k3, ← c2862, k3, ← c2863] at e_42_2
  have e_42_3 := arithEq_of_rows h (row := 42) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c2870, ← c2871, k3, ← c2872] at e_42_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_12
    linear_combination c2873.trans k2 - hc
  · have hc := e_42_3
    simp only [e_42_2, e_41_13] at hc
    linear_combination c2874.trans k2 - hc

theorem publicBatchWrapper89_f387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 41 59) = band (a (.virt 18143)) (a (.virt 18145)) := by
  have c2875 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2876 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2877 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_41_14 := arithEq_of_rows h (row := 41) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_14
  simp only [← c2875, ← c2876, ← c2877] at e_41_14
  have hr := e_41_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 41 63) = band (a (.virt 18147)) (a (.virt 18149)) := by
  have c2878 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2879 := (publicBatchWrapper89_copies89 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2880 := (publicBatchWrapper89_copies90 a h).1
  have e_41_15 := arithEq_of_rows h (row := 41) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_15
  simp only [← c2878, ← c2879, ← c2880] at e_41_15
  have hr := e_41_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 41 67) = band (a (.wire 41 59)) (a (.wire 41 63)) := by
  have c2881 := (publicBatchWrapper89_copies90 a h).2.1
  have c2882 := (publicBatchWrapper89_copies90 a h).2.2.1
  have c2883 := (publicBatchWrapper89_copies90 a h).2.2.2.1
  have e_41_16 := arithEq_of_rows h (row := 41) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_16
  simp only [← c2881, ← c2882, ← c2883] at e_41_16
  have hr := e_41_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8605)) (a (.virt 200)) (a (.virt 18151)) (a (.virt 18152)) := by
  have c2884 := (publicBatchWrapper89_copies90 a h).2.2.2.2.1
  have c2885 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.1
  have c2886 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.1
  have c2887 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.1
  have c2888 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.1
  have c2889 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.1
  have c2890 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2891 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2892 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2893 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2894 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2895 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2896 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2897 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_41_17 := arithEq_of_rows h (row := 41) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_17
  simp only [← c2887, ← c2888, ← c2889] at e_41_17
  have e_41_18 := arithEq_of_rows h (row := 41) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_18
  simp only [← c2890, ← c2891, ← c2892] at e_41_18
  have e_42_4 := arithEq_of_rows h (row := 42) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_4
  simp only [← c2884, k3, ← c2885, k3, ← c2886] at e_42_4
  have e_42_5 := arithEq_of_rows h (row := 42) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_5
  simp only [← c2893, ← c2894, k3, ← c2895] at e_42_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_17
    linear_combination c2896.trans k2 - hc
  · have hc := e_42_5
    simp only [e_42_4, e_41_18] at hc
    linear_combination c2897.trans k2 - hc

theorem publicBatchWrapper89_f391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8606)) (a (.virt 200)) (a (.virt 18153)) (a (.virt 18154)) := by
  have c2898 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2899 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2900 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2901 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2902 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2903 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2904 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2905 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2906 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2907 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2908 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2909 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2910 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2911 := (publicBatchWrapper89_copies90 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_41_19 := arithEq_of_rows h (row := 41) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_19
  simp only [← c2901, ← c2902, ← c2903] at e_41_19
  have e_42_6 := arithEq_of_rows h (row := 42) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_6
  simp only [← c2898, k3, ← c2899, k3, ← c2900] at e_42_6
  have e_42_7 := arithEq_of_rows h (row := 42) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2907, ← c2908, k3, ← c2909] at e_42_7
  have e_43_0 := arithEq_of_rows h (row := 43) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_0
  simp only [← c2904, ← c2905, ← c2906] at e_43_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_41_19
    linear_combination c2910.trans k2 - hc
  · have hc := e_42_7
    simp only [e_42_6, e_43_0] at hc
    linear_combination c2911.trans k2 - hc

theorem publicBatchWrapper89_f392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8607)) (a (.virt 200)) (a (.virt 18155)) (a (.virt 18156)) := by
  have c2912 := (publicBatchWrapper89_copies91 a h).1
  have c2913 := (publicBatchWrapper89_copies91 a h).2.1
  have c2914 := (publicBatchWrapper89_copies91 a h).2.2.1
  have c2915 := (publicBatchWrapper89_copies91 a h).2.2.2.1
  have c2916 := (publicBatchWrapper89_copies91 a h).2.2.2.2.1
  have c2917 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.1
  have c2918 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.1
  have c2919 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.1
  have c2920 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.1
  have c2921 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.1
  have c2922 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2923 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2924 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2925 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_42_8 := arithEq_of_rows h (row := 42) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_8
  simp only [← c2912, k3, ← c2913, k3, ← c2914] at e_42_8
  have e_42_9 := arithEq_of_rows h (row := 42) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_9
  simp only [← c2921, ← c2922, k3, ← c2923] at e_42_9
  have e_43_1 := arithEq_of_rows h (row := 43) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_1
  simp only [← c2915, ← c2916, ← c2917] at e_43_1
  have e_43_2 := arithEq_of_rows h (row := 43) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_2
  simp only [← c2918, ← c2919, ← c2920] at e_43_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_1
    linear_combination c2924.trans k2 - hc
  · have hc := e_42_9
    simp only [e_42_8, e_43_2] at hc
    linear_combination c2925.trans k2 - hc

theorem publicBatchWrapper89_f393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8608)) (a (.virt 200)) (a (.virt 18157)) (a (.virt 18158)) := by
  have c2926 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2927 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2928 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2929 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2930 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2931 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2932 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2933 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2934 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2935 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2936 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2937 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2938 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2939 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_42_10 := arithEq_of_rows h (row := 42) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2926, k3, ← c2927, k3, ← c2928] at e_42_10
  have e_42_11 := arithEq_of_rows h (row := 42) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_11
  simp only [← c2935, ← c2936, k3, ← c2937] at e_42_11
  have e_43_3 := arithEq_of_rows h (row := 43) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_3
  simp only [← c2929, ← c2930, ← c2931] at e_43_3
  have e_43_4 := arithEq_of_rows h (row := 43) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_4
  simp only [← c2932, ← c2933, ← c2934] at e_43_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_3
    linear_combination c2938.trans k2 - hc
  · have hc := e_42_11
    simp only [e_42_10, e_43_4] at hc
    linear_combination c2939.trans k2 - hc

theorem publicBatchWrapper89_f394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 43 23) = band (a (.virt 18151)) (a (.virt 18153)) := by
  have c2940 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2941 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2942 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_43_5 := arithEq_of_rows h (row := 43) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_5
  simp only [← c2940, ← c2941, ← c2942] at e_43_5
  have hr := e_43_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 43 27) = band (a (.virt 18155)) (a (.virt 18157)) := by
  have c2943 := (publicBatchWrapper89_copies91 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2944 := (publicBatchWrapper89_copies92 a h).1
  have c2945 := (publicBatchWrapper89_copies92 a h).2.1
  have e_43_6 := arithEq_of_rows h (row := 43) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_6
  simp only [← c2943, ← c2944, ← c2945] at e_43_6
  have hr := e_43_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 43 31) = band (a (.wire 43 23)) (a (.wire 43 27)) := by
  have c2946 := (publicBatchWrapper89_copies92 a h).2.2.1
  have c2947 := (publicBatchWrapper89_copies92 a h).2.2.2.1
  have c2948 := (publicBatchWrapper89_copies92 a h).2.2.2.2.1
  have e_43_7 := arithEq_of_rows h (row := 43) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_7
  simp only [← c2946, ← c2947, ← c2948] at e_43_7
  have hr := e_43_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8805)) (a (.virt 200)) (a (.virt 18159)) (a (.virt 18160)) := by
  have c2949 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.1
  have c2950 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.1
  have c2951 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.1
  have c2952 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.1
  have c2953 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.1
  have c2954 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2955 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2956 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2957 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2958 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2959 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2960 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2961 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2962 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_42_12 := arithEq_of_rows h (row := 42) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_12
  simp only [← c2949, k3, ← c2950, k3, ← c2951] at e_42_12
  have e_42_13 := arithEq_of_rows h (row := 42) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c2958, ← c2959, k3, ← c2960] at e_42_13
  have e_43_8 := arithEq_of_rows h (row := 43) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_8
  simp only [← c2952, ← c2953, ← c2954] at e_43_8
  have e_43_9 := arithEq_of_rows h (row := 43) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_9
  simp only [← c2955, ← c2956, ← c2957] at e_43_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_8
    linear_combination c2961.trans k2 - hc
  · have hc := e_42_13
    simp only [e_42_12, e_43_9] at hc
    linear_combination c2962.trans k2 - hc

theorem publicBatchWrapper89_f398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8806)) (a (.virt 200)) (a (.virt 18161)) (a (.virt 18162)) := by
  have c2963 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2964 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2965 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2966 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2967 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2968 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2969 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2970 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2971 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2972 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2973 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2974 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2975 := (publicBatchWrapper89_copies92 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2976 := (publicBatchWrapper89_copies93 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_42_14 := arithEq_of_rows h (row := 42) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_14
  simp only [← c2963, k3, ← c2964, k3, ← c2965] at e_42_14
  have e_42_15 := arithEq_of_rows h (row := 42) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_15
  simp only [← c2972, ← c2973, k3, ← c2974] at e_42_15
  have e_43_10 := arithEq_of_rows h (row := 43) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_10
  simp only [← c2966, ← c2967, ← c2968] at e_43_10
  have e_43_11 := arithEq_of_rows h (row := 43) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_11
  simp only [← c2969, ← c2970, ← c2971] at e_43_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_10
    linear_combination c2975.trans k2 - hc
  · have hc := e_42_15
    simp only [e_42_14, e_43_11] at hc
    linear_combination c2976.trans k2 - hc

theorem publicBatchWrapper89_f399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8807)) (a (.virt 200)) (a (.virt 18163)) (a (.virt 18164)) := by
  have c2977 := (publicBatchWrapper89_copies93 a h).2.1
  have c2978 := (publicBatchWrapper89_copies93 a h).2.2.1
  have c2979 := (publicBatchWrapper89_copies93 a h).2.2.2.1
  have c2980 := (publicBatchWrapper89_copies93 a h).2.2.2.2.1
  have c2981 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.1
  have c2982 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.1
  have c2983 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.1
  have c2984 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.1
  have c2985 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.1
  have c2986 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2987 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2988 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2989 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2990 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_42_16 := arithEq_of_rows h (row := 42) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_16
  simp only [← c2977, k3, ← c2978, k3, ← c2979] at e_42_16
  have e_42_17 := arithEq_of_rows h (row := 42) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_17
  simp only [← c2986, ← c2987, k3, ← c2988] at e_42_17
  have e_43_12 := arithEq_of_rows h (row := 43) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_12
  simp only [← c2980, ← c2981, ← c2982] at e_43_12
  have e_43_13 := arithEq_of_rows h (row := 43) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_13
  simp only [← c2983, ← c2984, ← c2985] at e_43_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_12
    linear_combination c2989.trans k2 - hc
  · have hc := e_42_17
    simp only [e_42_16, e_43_13] at hc
    linear_combination c2990.trans k2 - hc

theorem publicBatchWrapper89_f400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 8808)) (a (.virt 200)) (a (.virt 18165)) (a (.virt 18166)) := by
  have c2991 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2992 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2993 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2994 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2995 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2996 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2997 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2998 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2999 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3000 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3001 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3002 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3003 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3004 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_42_18 := arithEq_of_rows h (row := 42) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_18
  simp only [← c2991, k3, ← c2992, k3, ← c2993] at e_42_18
  have e_42_19 := arithEq_of_rows h (row := 42) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_19
  simp only [← c3000, ← c3001, k3, ← c3002] at e_42_19
  have e_43_14 := arithEq_of_rows h (row := 43) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_14
  simp only [← c2994, ← c2995, ← c2996] at e_43_14
  have e_43_15 := arithEq_of_rows h (row := 43) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_15
  simp only [← c2997, ← c2998, ← c2999] at e_43_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_14
    linear_combination c3003.trans k2 - hc
  · have hc := e_42_19
    simp only [e_42_18, e_43_15] at hc
    linear_combination c3004.trans k2 - hc

theorem publicBatchWrapper89_f401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 43 67) = band (a (.virt 18159)) (a (.virt 18161)) := by
  have c3005 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3006 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3007 := (publicBatchWrapper89_copies93 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_43_16 := arithEq_of_rows h (row := 43) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_16
  simp only [← c3005, ← c3006, ← c3007] at e_43_16
  have hr := e_43_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 43 71) = band (a (.virt 18163)) (a (.virt 18165)) := by
  have c3008 := (publicBatchWrapper89_copies94 a h).1
  have c3009 := (publicBatchWrapper89_copies94 a h).2.1
  have c3010 := (publicBatchWrapper89_copies94 a h).2.2.1
  have e_43_17 := arithEq_of_rows h (row := 43) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_17
  simp only [← c3008, ← c3009, ← c3010] at e_43_17
  have hr := e_43_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 43 75) = band (a (.wire 43 67)) (a (.wire 43 71)) := by
  have c3011 := (publicBatchWrapper89_copies94 a h).2.2.2.1
  have c3012 := (publicBatchWrapper89_copies94 a h).2.2.2.2.1
  have c3013 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.1
  have e_43_18 := arithEq_of_rows h (row := 43) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_18
  simp only [← c3011, ← c3012, ← c3013] at e_43_18
  have hr := e_43_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9005)) (a (.virt 200)) (a (.virt 18167)) (a (.virt 18168)) := by
  have c3014 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.1
  have c3015 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.1
  have c3016 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.1
  have c3017 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.1
  have c3018 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3019 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3020 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3021 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3022 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3023 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3024 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3025 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3026 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3027 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_43_19 := arithEq_of_rows h (row := 43) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_19
  simp only [← c3017, ← c3018, ← c3019] at e_43_19
  have e_44_0 := arithEq_of_rows h (row := 44) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_0
  simp only [← c3014, k3, ← c3015, k3, ← c3016] at e_44_0
  have e_44_1 := arithEq_of_rows h (row := 44) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_1
  simp only [← c3023, ← c3024, k3, ← c3025] at e_44_1
  have e_45_0 := arithEq_of_rows h (row := 45) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_0
  simp only [← c3020, ← c3021, ← c3022] at e_45_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_43_19
    linear_combination c3026.trans k2 - hc
  · have hc := e_44_1
    simp only [e_44_0, e_45_0] at hc
    linear_combination c3027.trans k2 - hc

theorem publicBatchWrapper89_f405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9006)) (a (.virt 200)) (a (.virt 18169)) (a (.virt 18170)) := by
  have c3028 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3029 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3030 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3031 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3032 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3033 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3034 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3035 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3036 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3037 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3038 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3039 := (publicBatchWrapper89_copies94 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3040 := (publicBatchWrapper89_copies95 a h).1
  have c3041 := (publicBatchWrapper89_copies95 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_2 := arithEq_of_rows h (row := 44) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_2
  simp only [← c3028, k3, ← c3029, k3, ← c3030] at e_44_2
  have e_44_3 := arithEq_of_rows h (row := 44) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_3
  simp only [← c3037, ← c3038, k3, ← c3039] at e_44_3
  have e_45_1 := arithEq_of_rows h (row := 45) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_1
  simp only [← c3031, ← c3032, ← c3033] at e_45_1
  have e_45_2 := arithEq_of_rows h (row := 45) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_2
  simp only [← c3034, ← c3035, ← c3036] at e_45_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_1
    linear_combination c3040.trans k2 - hc
  · have hc := e_44_3
    simp only [e_44_2, e_45_2] at hc
    linear_combination c3041.trans k2 - hc

theorem publicBatchWrapper89_f406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9007)) (a (.virt 200)) (a (.virt 18171)) (a (.virt 18172)) := by
  have c3042 := (publicBatchWrapper89_copies95 a h).2.2.1
  have c3043 := (publicBatchWrapper89_copies95 a h).2.2.2.1
  have c3044 := (publicBatchWrapper89_copies95 a h).2.2.2.2.1
  have c3045 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.1
  have c3046 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.1
  have c3047 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.1
  have c3048 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.1
  have c3049 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.1
  have c3050 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3051 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3052 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3053 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3054 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3055 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_4 := arithEq_of_rows h (row := 44) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_4
  simp only [← c3042, k3, ← c3043, k3, ← c3044] at e_44_4
  have e_44_5 := arithEq_of_rows h (row := 44) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_5
  simp only [← c3051, ← c3052, k3, ← c3053] at e_44_5
  have e_45_3 := arithEq_of_rows h (row := 45) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_3
  simp only [← c3045, ← c3046, ← c3047] at e_45_3
  have e_45_4 := arithEq_of_rows h (row := 45) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_4
  simp only [← c3048, ← c3049, ← c3050] at e_45_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_3
    linear_combination c3054.trans k2 - hc
  · have hc := e_44_5
    simp only [e_44_4, e_45_4] at hc
    linear_combination c3055.trans k2 - hc

theorem publicBatchWrapper89_f407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9008)) (a (.virt 200)) (a (.virt 18173)) (a (.virt 18174)) := by
  have c3056 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3057 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3058 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3059 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3060 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3061 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3062 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3063 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3064 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3065 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3066 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3067 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3068 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3069 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_6 := arithEq_of_rows h (row := 44) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_6
  simp only [← c3056, k3, ← c3057, k3, ← c3058] at e_44_6
  have e_44_7 := arithEq_of_rows h (row := 44) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_7
  simp only [← c3065, ← c3066, k3, ← c3067] at e_44_7
  have e_45_5 := arithEq_of_rows h (row := 45) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_5
  simp only [← c3059, ← c3060, ← c3061] at e_45_5
  have e_45_6 := arithEq_of_rows h (row := 45) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_6
  simp only [← c3062, ← c3063, ← c3064] at e_45_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_5
    linear_combination c3068.trans k2 - hc
  · have hc := e_44_7
    simp only [e_44_6, e_45_6] at hc
    linear_combination c3069.trans k2 - hc

theorem publicBatchWrapper89_f408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 45 31) = band (a (.virt 18167)) (a (.virt 18169)) := by
  have c3070 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3071 := (publicBatchWrapper89_copies95 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3072 := (publicBatchWrapper89_copies96 a h).1
  have e_45_7 := arithEq_of_rows h (row := 45) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_7
  simp only [← c3070, ← c3071, ← c3072] at e_45_7
  have hr := e_45_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 45 35) = band (a (.virt 18171)) (a (.virt 18173)) := by
  have c3073 := (publicBatchWrapper89_copies96 a h).2.1
  have c3074 := (publicBatchWrapper89_copies96 a h).2.2.1
  have c3075 := (publicBatchWrapper89_copies96 a h).2.2.2.1
  have e_45_8 := arithEq_of_rows h (row := 45) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_8
  simp only [← c3073, ← c3074, ← c3075] at e_45_8
  have hr := e_45_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 45 39) = band (a (.wire 45 31)) (a (.wire 45 35)) := by
  have c3076 := (publicBatchWrapper89_copies96 a h).2.2.2.2.1
  have c3077 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.1
  have c3078 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.1
  have e_45_9 := arithEq_of_rows h (row := 45) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_9
  simp only [← c3076, ← c3077, ← c3078] at e_45_9
  have hr := e_45_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9205)) (a (.virt 200)) (a (.virt 18175)) (a (.virt 18176)) := by
  have c3079 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.1
  have c3080 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.1
  have c3081 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.1
  have c3082 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3083 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3084 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3085 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3086 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3087 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3088 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3089 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3090 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3091 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3092 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_8 := arithEq_of_rows h (row := 44) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_8
  simp only [← c3079, k3, ← c3080, k3, ← c3081] at e_44_8
  have e_44_9 := arithEq_of_rows h (row := 44) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_9
  simp only [← c3088, ← c3089, k3, ← c3090] at e_44_9
  have e_45_10 := arithEq_of_rows h (row := 45) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_10
  simp only [← c3082, ← c3083, ← c3084] at e_45_10
  have e_45_11 := arithEq_of_rows h (row := 45) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_11
  simp only [← c3085, ← c3086, ← c3087] at e_45_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_10
    linear_combination c3091.trans k2 - hc
  · have hc := e_44_9
    simp only [e_44_8, e_45_11] at hc
    linear_combination c3092.trans k2 - hc

theorem publicBatchWrapper89_f412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9206)) (a (.virt 200)) (a (.virt 18177)) (a (.virt 18178)) := by
  have c3093 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3094 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3095 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3096 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3097 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3098 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3099 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3100 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3101 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3102 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3103 := (publicBatchWrapper89_copies96 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3104 := (publicBatchWrapper89_copies97 a h).1
  have c3105 := (publicBatchWrapper89_copies97 a h).2.1
  have c3106 := (publicBatchWrapper89_copies97 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_10 := arithEq_of_rows h (row := 44) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_10
  simp only [← c3093, k3, ← c3094, k3, ← c3095] at e_44_10
  have e_44_11 := arithEq_of_rows h (row := 44) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_11
  simp only [← c3102, ← c3103, k3, ← c3104] at e_44_11
  have e_45_12 := arithEq_of_rows h (row := 45) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_12
  simp only [← c3096, ← c3097, ← c3098] at e_45_12
  have e_45_13 := arithEq_of_rows h (row := 45) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_13
  simp only [← c3099, ← c3100, ← c3101] at e_45_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_12
    linear_combination c3105.trans k2 - hc
  · have hc := e_44_11
    simp only [e_44_10, e_45_13] at hc
    linear_combination c3106.trans k2 - hc

theorem publicBatchWrapper89_f413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9207)) (a (.virt 200)) (a (.virt 18179)) (a (.virt 18180)) := by
  have c3107 := (publicBatchWrapper89_copies97 a h).2.2.2.1
  have c3108 := (publicBatchWrapper89_copies97 a h).2.2.2.2.1
  have c3109 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.1
  have c3110 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.1
  have c3111 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.1
  have c3112 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.1
  have c3113 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.1
  have c3114 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3115 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3116 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3117 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3118 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3119 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3120 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_12 := arithEq_of_rows h (row := 44) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_12
  simp only [← c3107, k3, ← c3108, k3, ← c3109] at e_44_12
  have e_44_13 := arithEq_of_rows h (row := 44) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_13
  simp only [← c3116, ← c3117, k3, ← c3118] at e_44_13
  have e_45_14 := arithEq_of_rows h (row := 45) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_14
  simp only [← c3110, ← c3111, ← c3112] at e_45_14
  have e_45_15 := arithEq_of_rows h (row := 45) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_15
  simp only [← c3113, ← c3114, ← c3115] at e_45_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_14
    linear_combination c3119.trans k2 - hc
  · have hc := e_44_13
    simp only [e_44_12, e_45_15] at hc
    linear_combination c3120.trans k2 - hc

theorem publicBatchWrapper89_f414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9208)) (a (.virt 200)) (a (.virt 18181)) (a (.virt 18182)) := by
  have c3121 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3122 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3123 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3124 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3125 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3126 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3127 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3128 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3129 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3130 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3131 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3132 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3133 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3134 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_14 := arithEq_of_rows h (row := 44) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_14
  simp only [← c3121, k3, ← c3122, k3, ← c3123] at e_44_14
  have e_44_15 := arithEq_of_rows h (row := 44) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_15
  simp only [← c3130, ← c3131, k3, ← c3132] at e_44_15
  have e_45_16 := arithEq_of_rows h (row := 45) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_16
  simp only [← c3124, ← c3125, ← c3126] at e_45_16
  have e_45_17 := arithEq_of_rows h (row := 45) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_17
  simp only [← c3127, ← c3128, ← c3129] at e_45_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_45_16
    linear_combination c3133.trans k2 - hc
  · have hc := e_44_15
    simp only [e_44_14, e_45_17] at hc
    linear_combination c3134.trans k2 - hc

theorem publicBatchWrapper89_f415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 45 75) = band (a (.virt 18175)) (a (.virt 18177)) := by
  have c3135 := (publicBatchWrapper89_copies97 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3136 := (publicBatchWrapper89_copies98 a h).1
  have c3137 := (publicBatchWrapper89_copies98 a h).2.1
  have e_45_18 := arithEq_of_rows h (row := 45) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_18
  simp only [← c3135, ← c3136, ← c3137] at e_45_18
  have hr := e_45_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 45 79) = band (a (.virt 18179)) (a (.virt 18181)) := by
  have c3138 := (publicBatchWrapper89_copies98 a h).2.2.1
  have c3139 := (publicBatchWrapper89_copies98 a h).2.2.2.1
  have c3140 := (publicBatchWrapper89_copies98 a h).2.2.2.2.1
  have e_45_19 := arithEq_of_rows h (row := 45) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_19
  simp only [← c3138, ← c3139, ← c3140] at e_45_19
  have hr := e_45_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 46 3) = band (a (.wire 45 75)) (a (.wire 45 79)) := by
  have c3141 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.1
  have c3142 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.1
  have c3143 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.1
  have e_46_0 := arithEq_of_rows h (row := 46) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_0
  simp only [← c3141, ← c3142, ← c3143] at e_46_0
  have hr := e_46_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9405)) (a (.virt 200)) (a (.virt 18183)) (a (.virt 18184)) := by
  have c3144 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.1
  have c3145 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.1
  have c3146 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3147 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3148 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3149 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3150 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3151 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3152 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3153 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3154 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3155 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3156 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3157 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_16 := arithEq_of_rows h (row := 44) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_16
  simp only [← c3144, k3, ← c3145, k3, ← c3146] at e_44_16
  have e_44_17 := arithEq_of_rows h (row := 44) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_17
  simp only [← c3153, ← c3154, k3, ← c3155] at e_44_17
  have e_46_1 := arithEq_of_rows h (row := 46) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_1
  simp only [← c3147, ← c3148, ← c3149] at e_46_1
  have e_46_2 := arithEq_of_rows h (row := 46) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_2
  simp only [← c3150, ← c3151, ← c3152] at e_46_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_1
    linear_combination c3156.trans k2 - hc
  · have hc := e_44_17
    simp only [e_44_16, e_46_2] at hc
    linear_combination c3157.trans k2 - hc

theorem publicBatchWrapper89_f419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9406)) (a (.virt 200)) (a (.virt 18185)) (a (.virt 18186)) := by
  have c3158 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3159 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3160 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3161 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3162 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3163 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3164 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3165 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3166 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3167 := (publicBatchWrapper89_copies98 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3168 := (publicBatchWrapper89_copies99 a h).1
  have c3169 := (publicBatchWrapper89_copies99 a h).2.1
  have c3170 := (publicBatchWrapper89_copies99 a h).2.2.1
  have c3171 := (publicBatchWrapper89_copies99 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_44_18 := arithEq_of_rows h (row := 44) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_18
  simp only [← c3158, k3, ← c3159, k3, ← c3160] at e_44_18
  have e_44_19 := arithEq_of_rows h (row := 44) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_44_19
  simp only [← c3167, ← c3168, k3, ← c3169] at e_44_19
  have e_46_3 := arithEq_of_rows h (row := 46) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_3
  simp only [← c3161, ← c3162, ← c3163] at e_46_3
  have e_46_4 := arithEq_of_rows h (row := 46) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_4
  simp only [← c3164, ← c3165, ← c3166] at e_46_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_3
    linear_combination c3170.trans k2 - hc
  · have hc := e_44_19
    simp only [e_44_18, e_46_4] at hc
    linear_combination c3171.trans k2 - hc

theorem publicBatchWrapper89_f420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9407)) (a (.virt 200)) (a (.virt 18187)) (a (.virt 18188)) := by
  have c3172 := (publicBatchWrapper89_copies99 a h).2.2.2.2.1
  have c3173 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.1
  have c3174 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.1
  have c3175 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.1
  have c3176 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.1
  have c3177 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.1
  have c3178 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3179 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3180 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3181 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3182 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3183 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3184 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3185 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_46_5 := arithEq_of_rows h (row := 46) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_5
  simp only [← c3175, ← c3176, ← c3177] at e_46_5
  have e_46_6 := arithEq_of_rows h (row := 46) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c3178, ← c3179, ← c3180] at e_46_6
  have e_47_0 := arithEq_of_rows h (row := 47) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_0
  simp only [← c3172, k3, ← c3173, k3, ← c3174] at e_47_0
  have e_47_1 := arithEq_of_rows h (row := 47) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_1
  simp only [← c3181, ← c3182, k3, ← c3183] at e_47_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_5
    linear_combination c3184.trans k2 - hc
  · have hc := e_47_1
    simp only [e_47_0, e_46_6] at hc
    linear_combination c3185.trans k2 - hc

theorem publicBatchWrapper89_f421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9408)) (a (.virt 200)) (a (.virt 18189)) (a (.virt 18190)) := by
  have c3186 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3187 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3188 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3189 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3190 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3191 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3192 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3193 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3194 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3195 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3196 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3197 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3198 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3199 := (publicBatchWrapper89_copies99 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_46_7 := arithEq_of_rows h (row := 46) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_7
  simp only [← c3189, ← c3190, ← c3191] at e_46_7
  have e_46_8 := arithEq_of_rows h (row := 46) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_8
  simp only [← c3192, ← c3193, ← c3194] at e_46_8
  have e_47_2 := arithEq_of_rows h (row := 47) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_2
  simp only [← c3186, k3, ← c3187, k3, ← c3188] at e_47_2
  have e_47_3 := arithEq_of_rows h (row := 47) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_3
  simp only [← c3195, ← c3196, k3, ← c3197] at e_47_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_7
    linear_combination c3198.trans k2 - hc
  · have hc := e_47_3
    simp only [e_47_2, e_46_8] at hc
    linear_combination c3199.trans k2 - hc

theorem publicBatchWrapper89_f422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 46 39) = band (a (.virt 18183)) (a (.virt 18185)) := by
  have c3200 := (publicBatchWrapper89_copies100 a h).1
  have c3201 := (publicBatchWrapper89_copies100 a h).2.1
  have c3202 := (publicBatchWrapper89_copies100 a h).2.2.1
  have e_46_9 := arithEq_of_rows h (row := 46) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_9
  simp only [← c3200, ← c3201, ← c3202] at e_46_9
  have hr := e_46_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 46 43) = band (a (.virt 18187)) (a (.virt 18189)) := by
  have c3203 := (publicBatchWrapper89_copies100 a h).2.2.2.1
  have c3204 := (publicBatchWrapper89_copies100 a h).2.2.2.2.1
  have c3205 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.1
  have e_46_10 := arithEq_of_rows h (row := 46) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_10
  simp only [← c3203, ← c3204, ← c3205] at e_46_10
  have hr := e_46_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 46 47) = band (a (.wire 46 39)) (a (.wire 46 43)) := by
  have c3206 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.1
  have c3207 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.1
  have c3208 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.1
  have e_46_11 := arithEq_of_rows h (row := 46) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_11
  simp only [← c3206, ← c3207, ← c3208] at e_46_11
  have hr := e_46_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9605)) (a (.virt 200)) (a (.virt 18191)) (a (.virt 18192)) := by
  have c3209 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.1
  have c3210 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3211 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3212 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3213 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3214 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3215 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3216 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3217 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3218 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3219 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3220 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3221 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3222 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_46_12 := arithEq_of_rows h (row := 46) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_12
  simp only [← c3212, ← c3213, ← c3214] at e_46_12
  have e_46_13 := arithEq_of_rows h (row := 46) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_13
  simp only [← c3215, ← c3216, ← c3217] at e_46_13
  have e_47_4 := arithEq_of_rows h (row := 47) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_4
  simp only [← c3209, k3, ← c3210, k3, ← c3211] at e_47_4
  have e_47_5 := arithEq_of_rows h (row := 47) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_5
  simp only [← c3218, ← c3219, k3, ← c3220] at e_47_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_12
    linear_combination c3221.trans k2 - hc
  · have hc := e_47_5
    simp only [e_47_4, e_46_13] at hc
    linear_combination c3222.trans k2 - hc

theorem publicBatchWrapper89_f426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9606)) (a (.virt 200)) (a (.virt 18193)) (a (.virt 18194)) := by
  have c3223 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3224 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3225 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3226 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3227 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3228 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3229 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3230 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3231 := (publicBatchWrapper89_copies100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3232 := (publicBatchWrapper89_copies101 a h).1
  have c3233 := (publicBatchWrapper89_copies101 a h).2.1
  have c3234 := (publicBatchWrapper89_copies101 a h).2.2.1
  have c3235 := (publicBatchWrapper89_copies101 a h).2.2.2.1
  have c3236 := (publicBatchWrapper89_copies101 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_46_14 := arithEq_of_rows h (row := 46) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_14
  simp only [← c3226, ← c3227, ← c3228] at e_46_14
  have e_46_15 := arithEq_of_rows h (row := 46) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_15
  simp only [← c3229, ← c3230, ← c3231] at e_46_15
  have e_47_6 := arithEq_of_rows h (row := 47) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_6
  simp only [← c3223, k3, ← c3224, k3, ← c3225] at e_47_6
  have e_47_7 := arithEq_of_rows h (row := 47) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_7
  simp only [← c3232, ← c3233, k3, ← c3234] at e_47_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_14
    linear_combination c3235.trans k2 - hc
  · have hc := e_47_7
    simp only [e_47_6, e_46_15] at hc
    linear_combination c3236.trans k2 - hc

theorem publicBatchWrapper89_f427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9607)) (a (.virt 200)) (a (.virt 18195)) (a (.virt 18196)) := by
  have c3237 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.1
  have c3238 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.1
  have c3239 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.1
  have c3240 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.1
  have c3241 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.1
  have c3242 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3243 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3244 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3245 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3246 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3247 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3248 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3249 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3250 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_46_16 := arithEq_of_rows h (row := 46) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_16
  simp only [← c3240, ← c3241, ← c3242] at e_46_16
  have e_46_17 := arithEq_of_rows h (row := 46) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_17
  simp only [← c3243, ← c3244, ← c3245] at e_46_17
  have e_47_8 := arithEq_of_rows h (row := 47) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_8
  simp only [← c3237, k3, ← c3238, k3, ← c3239] at e_47_8
  have e_47_9 := arithEq_of_rows h (row := 47) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_9
  simp only [← c3246, ← c3247, k3, ← c3248] at e_47_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_16
    linear_combination c3249.trans k2 - hc
  · have hc := e_47_9
    simp only [e_47_8, e_46_17] at hc
    linear_combination c3250.trans k2 - hc

theorem publicBatchWrapper89_f428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9608)) (a (.virt 200)) (a (.virt 18197)) (a (.virt 18198)) := by
  have c3251 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3252 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3253 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3254 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3255 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3256 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3257 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3258 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3259 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3260 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3261 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3262 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3263 := (publicBatchWrapper89_copies101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3264 := (publicBatchWrapper89_copies102 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_46_18 := arithEq_of_rows h (row := 46) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_18
  simp only [← c3254, ← c3255, ← c3256] at e_46_18
  have e_46_19 := arithEq_of_rows h (row := 46) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_19
  simp only [← c3257, ← c3258, ← c3259] at e_46_19
  have e_47_10 := arithEq_of_rows h (row := 47) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_10
  simp only [← c3251, k3, ← c3252, k3, ← c3253] at e_47_10
  have e_47_11 := arithEq_of_rows h (row := 47) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_11
  simp only [← c3260, ← c3261, k3, ← c3262] at e_47_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_46_18
    linear_combination c3263.trans k2 - hc
  · have hc := e_47_11
    simp only [e_47_10, e_46_19] at hc
    linear_combination c3264.trans k2 - hc

theorem publicBatchWrapper89_f429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 48 3) = band (a (.virt 18191)) (a (.virt 18193)) := by
  have c3265 := (publicBatchWrapper89_copies102 a h).2.1
  have c3266 := (publicBatchWrapper89_copies102 a h).2.2.1
  have c3267 := (publicBatchWrapper89_copies102 a h).2.2.2.1
  have e_48_0 := arithEq_of_rows h (row := 48) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_0
  simp only [← c3265, ← c3266, ← c3267] at e_48_0
  have hr := e_48_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 48 7) = band (a (.virt 18195)) (a (.virt 18197)) := by
  have c3268 := (publicBatchWrapper89_copies102 a h).2.2.2.2.1
  have c3269 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.1
  have c3270 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.1
  have e_48_1 := arithEq_of_rows h (row := 48) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_1
  simp only [← c3268, ← c3269, ← c3270] at e_48_1
  have hr := e_48_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 48 11) = band (a (.wire 48 3)) (a (.wire 48 7)) := by
  have c3271 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.1
  have c3272 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.1
  have c3273 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.1
  have e_48_2 := arithEq_of_rows h (row := 48) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_2
  simp only [← c3271, ← c3272, ← c3273] at e_48_2
  have hr := e_48_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9805)) (a (.virt 200)) (a (.virt 18199)) (a (.virt 18200)) := by
  have c3274 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3275 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3276 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3277 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3278 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3279 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3280 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3281 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3282 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3283 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3284 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3285 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3286 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3287 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_47_12 := arithEq_of_rows h (row := 47) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_12
  simp only [← c3274, k3, ← c3275, k3, ← c3276] at e_47_12
  have e_47_13 := arithEq_of_rows h (row := 47) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_13
  simp only [← c3283, ← c3284, k3, ← c3285] at e_47_13
  have e_48_3 := arithEq_of_rows h (row := 48) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_3
  simp only [← c3277, ← c3278, ← c3279] at e_48_3
  have e_48_4 := arithEq_of_rows h (row := 48) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_4
  simp only [← c3280, ← c3281, ← c3282] at e_48_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_3
    linear_combination c3286.trans k2 - hc
  · have hc := e_47_13
    simp only [e_47_12, e_48_4] at hc
    linear_combination c3287.trans k2 - hc

theorem publicBatchWrapper89_f433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9806)) (a (.virt 200)) (a (.virt 18201)) (a (.virt 18202)) := by
  have c3288 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3289 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3290 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3291 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3292 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3293 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3294 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3295 := (publicBatchWrapper89_copies102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3296 := (publicBatchWrapper89_copies103 a h).1
  have c3297 := (publicBatchWrapper89_copies103 a h).2.1
  have c3298 := (publicBatchWrapper89_copies103 a h).2.2.1
  have c3299 := (publicBatchWrapper89_copies103 a h).2.2.2.1
  have c3300 := (publicBatchWrapper89_copies103 a h).2.2.2.2.1
  have c3301 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_47_14 := arithEq_of_rows h (row := 47) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_14
  simp only [← c3288, k3, ← c3289, k3, ← c3290] at e_47_14
  have e_47_15 := arithEq_of_rows h (row := 47) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_15
  simp only [← c3297, ← c3298, k3, ← c3299] at e_47_15
  have e_48_5 := arithEq_of_rows h (row := 48) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_5
  simp only [← c3291, ← c3292, ← c3293] at e_48_5
  have e_48_6 := arithEq_of_rows h (row := 48) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_6
  simp only [← c3294, ← c3295, ← c3296] at e_48_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_5
    linear_combination c3300.trans k2 - hc
  · have hc := e_47_15
    simp only [e_47_14, e_48_6] at hc
    linear_combination c3301.trans k2 - hc

theorem publicBatchWrapper89_f434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9807)) (a (.virt 200)) (a (.virt 18203)) (a (.virt 18204)) := by
  have c3302 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.1
  have c3303 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.1
  have c3304 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.1
  have c3305 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.1
  have c3306 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3307 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3308 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3309 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3310 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3311 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3312 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3313 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3314 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3315 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_47_16 := arithEq_of_rows h (row := 47) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_16
  simp only [← c3302, k3, ← c3303, k3, ← c3304] at e_47_16
  have e_47_17 := arithEq_of_rows h (row := 47) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_17
  simp only [← c3311, ← c3312, k3, ← c3313] at e_47_17
  have e_48_7 := arithEq_of_rows h (row := 48) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_7
  simp only [← c3305, ← c3306, ← c3307] at e_48_7
  have e_48_8 := arithEq_of_rows h (row := 48) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_8
  simp only [← c3308, ← c3309, ← c3310] at e_48_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_7
    linear_combination c3314.trans k2 - hc
  · have hc := e_47_17
    simp only [e_47_16, e_48_8] at hc
    linear_combination c3315.trans k2 - hc

theorem publicBatchWrapper89_f435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 9808)) (a (.virt 200)) (a (.virt 18205)) (a (.virt 18206)) := by
  have c3316 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3317 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3318 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3319 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3320 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3321 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3322 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3323 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3324 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3325 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3326 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3327 := (publicBatchWrapper89_copies103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3328 := (publicBatchWrapper89_copies104 a h).1
  have c3329 := (publicBatchWrapper89_copies104 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_47_18 := arithEq_of_rows h (row := 47) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_18
  simp only [← c3316, k3, ← c3317, k3, ← c3318] at e_47_18
  have e_47_19 := arithEq_of_rows h (row := 47) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_19
  simp only [← c3325, ← c3326, k3, ← c3327] at e_47_19
  have e_48_9 := arithEq_of_rows h (row := 48) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_9
  simp only [← c3319, ← c3320, ← c3321] at e_48_9
  have e_48_10 := arithEq_of_rows h (row := 48) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_10
  simp only [← c3322, ← c3323, ← c3324] at e_48_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_9
    linear_combination c3328.trans k2 - hc
  · have hc := e_47_19
    simp only [e_47_18, e_48_10] at hc
    linear_combination c3329.trans k2 - hc

theorem publicBatchWrapper89_f436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 48 47) = band (a (.virt 18199)) (a (.virt 18201)) := by
  have c3330 := (publicBatchWrapper89_copies104 a h).2.2.1
  have c3331 := (publicBatchWrapper89_copies104 a h).2.2.2.1
  have c3332 := (publicBatchWrapper89_copies104 a h).2.2.2.2.1
  have e_48_11 := arithEq_of_rows h (row := 48) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_11
  simp only [← c3330, ← c3331, ← c3332] at e_48_11
  have hr := e_48_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 48 51) = band (a (.virt 18203)) (a (.virt 18205)) := by
  have c3333 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.1
  have c3334 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.1
  have c3335 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.1
  have e_48_12 := arithEq_of_rows h (row := 48) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_12
  simp only [← c3333, ← c3334, ← c3335] at e_48_12
  have hr := e_48_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 48 55) = band (a (.wire 48 47)) (a (.wire 48 51)) := by
  have c3336 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.1
  have c3337 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.1
  have c3338 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_48_13 := arithEq_of_rows h (row := 48) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_13
  simp only [← c3336, ← c3337, ← c3338] at e_48_13
  have hr := e_48_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10005)) (a (.virt 200)) (a (.virt 18207)) (a (.virt 18208)) := by
  have c3339 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3340 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3341 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3342 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3343 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3344 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3345 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3346 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3347 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3348 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3349 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3350 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3351 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3352 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_48_14 := arithEq_of_rows h (row := 48) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_14
  simp only [← c3342, ← c3343, ← c3344] at e_48_14
  have e_48_15 := arithEq_of_rows h (row := 48) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_15
  simp only [← c3345, ← c3346, ← c3347] at e_48_15
  have e_49_0 := arithEq_of_rows h (row := 49) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_0
  simp only [← c3339, k3, ← c3340, k3, ← c3341] at e_49_0
  have e_49_1 := arithEq_of_rows h (row := 49) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_1
  simp only [← c3348, ← c3349, k3, ← c3350] at e_49_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_14
    linear_combination c3351.trans k2 - hc
  · have hc := e_49_1
    simp only [e_49_0, e_48_15] at hc
    linear_combination c3352.trans k2 - hc

theorem publicBatchWrapper89_f440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10006)) (a (.virt 200)) (a (.virt 18209)) (a (.virt 18210)) := by
  have c3353 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3354 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3355 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3356 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3357 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3358 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3359 := (publicBatchWrapper89_copies104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3360 := (publicBatchWrapper89_copies105 a h).1
  have c3361 := (publicBatchWrapper89_copies105 a h).2.1
  have c3362 := (publicBatchWrapper89_copies105 a h).2.2.1
  have c3363 := (publicBatchWrapper89_copies105 a h).2.2.2.1
  have c3364 := (publicBatchWrapper89_copies105 a h).2.2.2.2.1
  have c3365 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.1
  have c3366 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_48_16 := arithEq_of_rows h (row := 48) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_16
  simp only [← c3356, ← c3357, ← c3358] at e_48_16
  have e_48_17 := arithEq_of_rows h (row := 48) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_17
  simp only [← c3359, ← c3360, ← c3361] at e_48_17
  have e_49_2 := arithEq_of_rows h (row := 49) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_2
  simp only [← c3353, k3, ← c3354, k3, ← c3355] at e_49_2
  have e_49_3 := arithEq_of_rows h (row := 49) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_3
  simp only [← c3362, ← c3363, k3, ← c3364] at e_49_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_16
    linear_combination c3365.trans k2 - hc
  · have hc := e_49_3
    simp only [e_49_2, e_48_17] at hc
    linear_combination c3366.trans k2 - hc

theorem publicBatchWrapper89_f441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10007)) (a (.virt 200)) (a (.virt 18211)) (a (.virt 18212)) := by
  have c3367 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.1
  have c3368 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.1
  have c3369 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.1
  have c3370 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3371 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3372 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3373 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3374 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3375 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3376 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3377 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3378 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3379 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3380 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_48_18 := arithEq_of_rows h (row := 48) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_18
  simp only [← c3370, ← c3371, ← c3372] at e_48_18
  have e_48_19 := arithEq_of_rows h (row := 48) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_19
  simp only [← c3373, ← c3374, ← c3375] at e_48_19
  have e_49_4 := arithEq_of_rows h (row := 49) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_4
  simp only [← c3367, k3, ← c3368, k3, ← c3369] at e_49_4
  have e_49_5 := arithEq_of_rows h (row := 49) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_5
  simp only [← c3376, ← c3377, k3, ← c3378] at e_49_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_48_18
    linear_combination c3379.trans k2 - hc
  · have hc := e_49_5
    simp only [e_49_4, e_48_19] at hc
    linear_combination c3380.trans k2 - hc

theorem publicBatchWrapper89_f442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10008)) (a (.virt 200)) (a (.virt 18213)) (a (.virt 18214)) := by
  have c3381 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3382 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3383 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3384 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3385 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3386 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3387 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3388 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3389 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3390 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3391 := (publicBatchWrapper89_copies105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3392 := (publicBatchWrapper89_copies106 a h).1
  have c3393 := (publicBatchWrapper89_copies106 a h).2.1
  have c3394 := (publicBatchWrapper89_copies106 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_6 := arithEq_of_rows h (row := 49) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_6
  simp only [← c3381, k3, ← c3382, k3, ← c3383] at e_49_6
  have e_49_7 := arithEq_of_rows h (row := 49) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_7
  simp only [← c3390, ← c3391, k3, ← c3392] at e_49_7
  have e_50_0 := arithEq_of_rows h (row := 50) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_0
  simp only [← c3384, ← c3385, ← c3386] at e_50_0
  have e_50_1 := arithEq_of_rows h (row := 50) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_1
  simp only [← c3387, ← c3388, ← c3389] at e_50_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_0
    linear_combination c3393.trans k2 - hc
  · have hc := e_49_7
    simp only [e_49_6, e_50_1] at hc
    linear_combination c3394.trans k2 - hc

theorem publicBatchWrapper89_f443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 50 11) = band (a (.virt 18207)) (a (.virt 18209)) := by
  have c3395 := (publicBatchWrapper89_copies106 a h).2.2.2.1
  have c3396 := (publicBatchWrapper89_copies106 a h).2.2.2.2.1
  have c3397 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.1
  have e_50_2 := arithEq_of_rows h (row := 50) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_2
  simp only [← c3395, ← c3396, ← c3397] at e_50_2
  have hr := e_50_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 50 15) = band (a (.virt 18211)) (a (.virt 18213)) := by
  have c3398 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.1
  have c3399 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.1
  have c3400 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.1
  have e_50_3 := arithEq_of_rows h (row := 50) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_3
  simp only [← c3398, ← c3399, ← c3400] at e_50_3
  have hr := e_50_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 50 19) = band (a (.wire 50 11)) (a (.wire 50 15)) := by
  have c3401 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.1
  have c3402 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3403 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_4 := arithEq_of_rows h (row := 50) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_4
  simp only [← c3401, ← c3402, ← c3403] at e_50_4
  have hr := e_50_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10205)) (a (.virt 200)) (a (.virt 18215)) (a (.virt 18216)) := by
  have c3404 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3405 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3406 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3407 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3408 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3409 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3410 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3411 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3412 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3413 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3414 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3415 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3416 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3417 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_8 := arithEq_of_rows h (row := 49) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_8
  simp only [← c3404, k3, ← c3405, k3, ← c3406] at e_49_8
  have e_49_9 := arithEq_of_rows h (row := 49) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_9
  simp only [← c3413, ← c3414, k3, ← c3415] at e_49_9
  have e_50_5 := arithEq_of_rows h (row := 50) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_5
  simp only [← c3407, ← c3408, ← c3409] at e_50_5
  have e_50_6 := arithEq_of_rows h (row := 50) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_6
  simp only [← c3410, ← c3411, ← c3412] at e_50_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_5
    linear_combination c3416.trans k2 - hc
  · have hc := e_49_9
    simp only [e_49_8, e_50_6] at hc
    linear_combination c3417.trans k2 - hc

theorem publicBatchWrapper89_f447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10206)) (a (.virt 200)) (a (.virt 18217)) (a (.virt 18218)) := by
  have c3418 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3419 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3420 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3421 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3422 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3423 := (publicBatchWrapper89_copies106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3424 := (publicBatchWrapper89_copies107 a h).1
  have c3425 := (publicBatchWrapper89_copies107 a h).2.1
  have c3426 := (publicBatchWrapper89_copies107 a h).2.2.1
  have c3427 := (publicBatchWrapper89_copies107 a h).2.2.2.1
  have c3428 := (publicBatchWrapper89_copies107 a h).2.2.2.2.1
  have c3429 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.1
  have c3430 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.1
  have c3431 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_10 := arithEq_of_rows h (row := 49) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_10
  simp only [← c3418, k3, ← c3419, k3, ← c3420] at e_49_10
  have e_49_11 := arithEq_of_rows h (row := 49) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_11
  simp only [← c3427, ← c3428, k3, ← c3429] at e_49_11
  have e_50_7 := arithEq_of_rows h (row := 50) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_7
  simp only [← c3421, ← c3422, ← c3423] at e_50_7
  have e_50_8 := arithEq_of_rows h (row := 50) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_8
  simp only [← c3424, ← c3425, ← c3426] at e_50_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_7
    linear_combination c3430.trans k2 - hc
  · have hc := e_49_11
    simp only [e_49_10, e_50_8] at hc
    linear_combination c3431.trans k2 - hc

theorem publicBatchWrapper89_f448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10207)) (a (.virt 200)) (a (.virt 18219)) (a (.virt 18220)) := by
  have c3432 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.1
  have c3433 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.1
  have c3434 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3435 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3436 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3437 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3438 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3439 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3440 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3441 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3442 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3443 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3444 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3445 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_12 := arithEq_of_rows h (row := 49) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_12
  simp only [← c3432, k3, ← c3433, k3, ← c3434] at e_49_12
  have e_49_13 := arithEq_of_rows h (row := 49) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_13
  simp only [← c3441, ← c3442, k3, ← c3443] at e_49_13
  have e_50_9 := arithEq_of_rows h (row := 50) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_9
  simp only [← c3435, ← c3436, ← c3437] at e_50_9
  have e_50_10 := arithEq_of_rows h (row := 50) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_10
  simp only [← c3438, ← c3439, ← c3440] at e_50_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_9
    linear_combination c3444.trans k2 - hc
  · have hc := e_49_13
    simp only [e_49_12, e_50_10] at hc
    linear_combination c3445.trans k2 - hc

theorem publicBatchWrapper89_f449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10208)) (a (.virt 200)) (a (.virt 18221)) (a (.virt 18222)) := by
  have c3446 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3447 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3448 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3449 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3450 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3451 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3452 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3453 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3454 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3455 := (publicBatchWrapper89_copies107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3456 := (publicBatchWrapper89_copies108 a h).1
  have c3457 := (publicBatchWrapper89_copies108 a h).2.1
  have c3458 := (publicBatchWrapper89_copies108 a h).2.2.1
  have c3459 := (publicBatchWrapper89_copies108 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_14 := arithEq_of_rows h (row := 49) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_14
  simp only [← c3446, k3, ← c3447, k3, ← c3448] at e_49_14
  have e_49_15 := arithEq_of_rows h (row := 49) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_15
  simp only [← c3455, ← c3456, k3, ← c3457] at e_49_15
  have e_50_11 := arithEq_of_rows h (row := 50) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_11
  simp only [← c3449, ← c3450, ← c3451] at e_50_11
  have e_50_12 := arithEq_of_rows h (row := 50) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_12
  simp only [← c3452, ← c3453, ← c3454] at e_50_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_11
    linear_combination c3458.trans k2 - hc
  · have hc := e_49_15
    simp only [e_49_14, e_50_12] at hc
    linear_combination c3459.trans k2 - hc

theorem publicBatchWrapper89_f450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 50 55) = band (a (.virt 18215)) (a (.virt 18217)) := by
  have c3460 := (publicBatchWrapper89_copies108 a h).2.2.2.2.1
  have c3461 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.1
  have c3462 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.1
  have e_50_13 := arithEq_of_rows h (row := 50) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_13
  simp only [← c3460, ← c3461, ← c3462] at e_50_13
  have hr := e_50_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 50 59) = band (a (.virt 18219)) (a (.virt 18221)) := by
  have c3463 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.1
  have c3464 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.1
  have c3465 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.1
  have e_50_14 := arithEq_of_rows h (row := 50) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_14
  simp only [← c3463, ← c3464, ← c3465] at e_50_14
  have hr := e_50_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 50 63) = band (a (.wire 50 55)) (a (.wire 50 59)) := by
  have c3466 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3467 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3468 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_15 := arithEq_of_rows h (row := 50) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_15
  simp only [← c3466, ← c3467, ← c3468] at e_50_15
  have hr := e_50_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10405)) (a (.virt 200)) (a (.virt 18223)) (a (.virt 18224)) := by
  have c3469 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3470 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3471 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3472 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3473 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3474 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3475 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3476 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3477 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3478 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3479 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3480 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3481 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3482 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_16 := arithEq_of_rows h (row := 49) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_16
  simp only [← c3469, k3, ← c3470, k3, ← c3471] at e_49_16
  have e_49_17 := arithEq_of_rows h (row := 49) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_17
  simp only [← c3478, ← c3479, k3, ← c3480] at e_49_17
  have e_50_16 := arithEq_of_rows h (row := 50) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_16
  simp only [← c3472, ← c3473, ← c3474] at e_50_16
  have e_50_17 := arithEq_of_rows h (row := 50) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_17
  simp only [← c3475, ← c3476, ← c3477] at e_50_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_16
    linear_combination c3481.trans k2 - hc
  · have hc := e_49_17
    simp only [e_49_16, e_50_17] at hc
    linear_combination c3482.trans k2 - hc

theorem publicBatchWrapper89_f454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10406)) (a (.virt 200)) (a (.virt 18225)) (a (.virt 18226)) := by
  have c3483 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3484 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3485 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3486 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3487 := (publicBatchWrapper89_copies108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3488 := (publicBatchWrapper89_copies109 a h).1
  have c3489 := (publicBatchWrapper89_copies109 a h).2.1
  have c3490 := (publicBatchWrapper89_copies109 a h).2.2.1
  have c3491 := (publicBatchWrapper89_copies109 a h).2.2.2.1
  have c3492 := (publicBatchWrapper89_copies109 a h).2.2.2.2.1
  have c3493 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.1
  have c3494 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.1
  have c3495 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.1
  have c3496 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_49_18 := arithEq_of_rows h (row := 49) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_18
  simp only [← c3483, k3, ← c3484, k3, ← c3485] at e_49_18
  have e_49_19 := arithEq_of_rows h (row := 49) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_19
  simp only [← c3492, ← c3493, k3, ← c3494] at e_49_19
  have e_50_18 := arithEq_of_rows h (row := 50) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_18
  simp only [← c3486, ← c3487, ← c3488] at e_50_18
  have e_50_19 := arithEq_of_rows h (row := 50) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_19
  simp only [← c3489, ← c3490, ← c3491] at e_50_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_50_18
    linear_combination c3495.trans k2 - hc
  · have hc := e_49_19
    simp only [e_49_18, e_50_19] at hc
    linear_combination c3496.trans k2 - hc

theorem publicBatchWrapper89_f455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10407)) (a (.virt 200)) (a (.virt 18227)) (a (.virt 18228)) := by
  have c3497 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.1
  have c3498 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3499 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3500 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3501 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3502 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3503 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3504 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3505 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3506 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3507 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3508 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3509 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3510 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_0 := arithEq_of_rows h (row := 51) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_0
  simp only [← c3497, k3, ← c3498, k3, ← c3499] at e_51_0
  have e_51_1 := arithEq_of_rows h (row := 51) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_1
  simp only [← c3506, ← c3507, k3, ← c3508] at e_51_1
  have e_52_0 := arithEq_of_rows h (row := 52) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_0
  simp only [← c3500, ← c3501, ← c3502] at e_52_0
  have e_52_1 := arithEq_of_rows h (row := 52) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_1
  simp only [← c3503, ← c3504, ← c3505] at e_52_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_0
    linear_combination c3509.trans k2 - hc
  · have hc := e_51_1
    simp only [e_51_0, e_52_1] at hc
    linear_combination c3510.trans k2 - hc

theorem publicBatchWrapper89_f456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10408)) (a (.virt 200)) (a (.virt 18229)) (a (.virt 18230)) := by
  have c3511 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3512 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3513 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3514 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3515 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3516 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3517 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3518 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3519 := (publicBatchWrapper89_copies109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3520 := (publicBatchWrapper89_copies110 a h).1
  have c3521 := (publicBatchWrapper89_copies110 a h).2.1
  have c3522 := (publicBatchWrapper89_copies110 a h).2.2.1
  have c3523 := (publicBatchWrapper89_copies110 a h).2.2.2.1
  have c3524 := (publicBatchWrapper89_copies110 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_2 := arithEq_of_rows h (row := 51) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_2
  simp only [← c3511, k3, ← c3512, k3, ← c3513] at e_51_2
  have e_51_3 := arithEq_of_rows h (row := 51) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_3
  simp only [← c3520, ← c3521, k3, ← c3522] at e_51_3
  have e_52_2 := arithEq_of_rows h (row := 52) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_2
  simp only [← c3514, ← c3515, ← c3516] at e_52_2
  have e_52_3 := arithEq_of_rows h (row := 52) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_3
  simp only [← c3517, ← c3518, ← c3519] at e_52_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_2
    linear_combination c3523.trans k2 - hc
  · have hc := e_51_3
    simp only [e_51_2, e_52_3] at hc
    linear_combination c3524.trans k2 - hc

theorem publicBatchWrapper89_f457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 52 19) = band (a (.virt 18223)) (a (.virt 18225)) := by
  have c3525 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.1
  have c3526 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.1
  have c3527 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.1
  have e_52_4 := arithEq_of_rows h (row := 52) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_4
  simp only [← c3525, ← c3526, ← c3527] at e_52_4
  have hr := e_52_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 52 23) = band (a (.virt 18227)) (a (.virt 18229)) := by
  have c3528 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.1
  have c3529 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.1
  have c3530 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_52_5 := arithEq_of_rows h (row := 52) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_5
  simp only [← c3528, ← c3529, ← c3530] at e_52_5
  have hr := e_52_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 52 27) = band (a (.wire 52 19)) (a (.wire 52 23)) := by
  have c3531 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3532 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3533 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_52_6 := arithEq_of_rows h (row := 52) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_6
  simp only [← c3531, ← c3532, ← c3533] at e_52_6
  have hr := e_52_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10605)) (a (.virt 200)) (a (.virt 18231)) (a (.virt 18232)) := by
  have c3534 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3535 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3536 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3537 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3538 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3539 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3540 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3541 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3542 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3543 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3544 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3545 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3546 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3547 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_4 := arithEq_of_rows h (row := 51) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_4
  simp only [← c3534, k3, ← c3535, k3, ← c3536] at e_51_4
  have e_51_5 := arithEq_of_rows h (row := 51) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_5
  simp only [← c3543, ← c3544, k3, ← c3545] at e_51_5
  have e_52_7 := arithEq_of_rows h (row := 52) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_7
  simp only [← c3537, ← c3538, ← c3539] at e_52_7
  have e_52_8 := arithEq_of_rows h (row := 52) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_8
  simp only [← c3540, ← c3541, ← c3542] at e_52_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_7
    linear_combination c3546.trans k2 - hc
  · have hc := e_51_5
    simp only [e_51_4, e_52_8] at hc
    linear_combination c3547.trans k2 - hc

theorem publicBatchWrapper89_f461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10606)) (a (.virt 200)) (a (.virt 18233)) (a (.virt 18234)) := by
  have c3548 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3549 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3550 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3551 := (publicBatchWrapper89_copies110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3552 := (publicBatchWrapper89_copies111 a h).1
  have c3553 := (publicBatchWrapper89_copies111 a h).2.1
  have c3554 := (publicBatchWrapper89_copies111 a h).2.2.1
  have c3555 := (publicBatchWrapper89_copies111 a h).2.2.2.1
  have c3556 := (publicBatchWrapper89_copies111 a h).2.2.2.2.1
  have c3557 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.1
  have c3558 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.1
  have c3559 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.1
  have c3560 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.1
  have c3561 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_6 := arithEq_of_rows h (row := 51) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_6
  simp only [← c3548, k3, ← c3549, k3, ← c3550] at e_51_6
  have e_51_7 := arithEq_of_rows h (row := 51) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_7
  simp only [← c3557, ← c3558, k3, ← c3559] at e_51_7
  have e_52_9 := arithEq_of_rows h (row := 52) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_9
  simp only [← c3551, ← c3552, ← c3553] at e_52_9
  have e_52_10 := arithEq_of_rows h (row := 52) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_10
  simp only [← c3554, ← c3555, ← c3556] at e_52_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_9
    linear_combination c3560.trans k2 - hc
  · have hc := e_51_7
    simp only [e_51_6, e_52_10] at hc
    linear_combination c3561.trans k2 - hc

theorem publicBatchWrapper89_f462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10607)) (a (.virt 200)) (a (.virt 18235)) (a (.virt 18236)) := by
  have c3562 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3563 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3564 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3565 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3566 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3567 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3568 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3569 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3570 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3571 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3572 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3573 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3574 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3575 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_8 := arithEq_of_rows h (row := 51) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_8
  simp only [← c3562, k3, ← c3563, k3, ← c3564] at e_51_8
  have e_51_9 := arithEq_of_rows h (row := 51) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_9
  simp only [← c3571, ← c3572, k3, ← c3573] at e_51_9
  have e_52_11 := arithEq_of_rows h (row := 52) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_11
  simp only [← c3565, ← c3566, ← c3567] at e_52_11
  have e_52_12 := arithEq_of_rows h (row := 52) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_12
  simp only [← c3568, ← c3569, ← c3570] at e_52_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_11
    linear_combination c3574.trans k2 - hc
  · have hc := e_51_9
    simp only [e_51_8, e_52_12] at hc
    linear_combination c3575.trans k2 - hc

theorem publicBatchWrapper89_f463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10608)) (a (.virt 200)) (a (.virt 18237)) (a (.virt 18238)) := by
  have c3576 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3577 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3578 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3579 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3580 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3581 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3582 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3583 := (publicBatchWrapper89_copies111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3584 := (publicBatchWrapper89_copies112 a h).1
  have c3585 := (publicBatchWrapper89_copies112 a h).2.1
  have c3586 := (publicBatchWrapper89_copies112 a h).2.2.1
  have c3587 := (publicBatchWrapper89_copies112 a h).2.2.2.1
  have c3588 := (publicBatchWrapper89_copies112 a h).2.2.2.2.1
  have c3589 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_10 := arithEq_of_rows h (row := 51) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_10
  simp only [← c3576, k3, ← c3577, k3, ← c3578] at e_51_10
  have e_51_11 := arithEq_of_rows h (row := 51) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_11
  simp only [← c3585, ← c3586, k3, ← c3587] at e_51_11
  have e_52_13 := arithEq_of_rows h (row := 52) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_13
  simp only [← c3579, ← c3580, ← c3581] at e_52_13
  have e_52_14 := arithEq_of_rows h (row := 52) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_14
  simp only [← c3582, ← c3583, ← c3584] at e_52_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_13
    linear_combination c3588.trans k2 - hc
  · have hc := e_51_11
    simp only [e_51_10, e_52_14] at hc
    linear_combination c3589.trans k2 - hc

theorem publicBatchWrapper89_f464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 52 63) = band (a (.virt 18231)) (a (.virt 18233)) := by
  have c3590 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.1
  have c3591 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.1
  have c3592 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.1
  have e_52_15 := arithEq_of_rows h (row := 52) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_15
  simp only [← c3590, ← c3591, ← c3592] at e_52_15
  have hr := e_52_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 52 67) = band (a (.virt 18235)) (a (.virt 18237)) := by
  have c3593 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.1
  have c3594 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3595 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_52_16 := arithEq_of_rows h (row := 52) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_16
  simp only [← c3593, ← c3594, ← c3595] at e_52_16
  have hr := e_52_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 52 71) = band (a (.wire 52 63)) (a (.wire 52 67)) := by
  have c3596 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3597 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3598 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_52_17 := arithEq_of_rows h (row := 52) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_17
  simp only [← c3596, ← c3597, ← c3598] at e_52_17
  have hr := e_52_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10805)) (a (.virt 200)) (a (.virt 18239)) (a (.virt 18240)) := by
  have c3599 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3600 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3601 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3602 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3603 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3604 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3605 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3606 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3607 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3608 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3609 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3610 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3611 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3612 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_12 := arithEq_of_rows h (row := 51) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_12
  simp only [← c3599, k3, ← c3600, k3, ← c3601] at e_51_12
  have e_51_13 := arithEq_of_rows h (row := 51) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_13
  simp only [← c3608, ← c3609, k3, ← c3610] at e_51_13
  have e_52_18 := arithEq_of_rows h (row := 52) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_18
  simp only [← c3602, ← c3603, ← c3604] at e_52_18
  have e_52_19 := arithEq_of_rows h (row := 52) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_19
  simp only [← c3605, ← c3606, ← c3607] at e_52_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_52_18
    linear_combination c3611.trans k2 - hc
  · have hc := e_51_13
    simp only [e_51_12, e_52_19] at hc
    linear_combination c3612.trans k2 - hc

theorem publicBatchWrapper89_f468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10806)) (a (.virt 200)) (a (.virt 18241)) (a (.virt 18242)) := by
  have c3613 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3614 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3615 := (publicBatchWrapper89_copies112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3616 := (publicBatchWrapper89_copies113 a h).1
  have c3617 := (publicBatchWrapper89_copies113 a h).2.1
  have c3618 := (publicBatchWrapper89_copies113 a h).2.2.1
  have c3619 := (publicBatchWrapper89_copies113 a h).2.2.2.1
  have c3620 := (publicBatchWrapper89_copies113 a h).2.2.2.2.1
  have c3621 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.1
  have c3622 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.1
  have c3623 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.1
  have c3624 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.1
  have c3625 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.1
  have c3626 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_14 := arithEq_of_rows h (row := 51) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_14
  simp only [← c3613, k3, ← c3614, k3, ← c3615] at e_51_14
  have e_51_15 := arithEq_of_rows h (row := 51) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_15
  simp only [← c3622, ← c3623, k3, ← c3624] at e_51_15
  have e_53_0 := arithEq_of_rows h (row := 53) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_0
  simp only [← c3616, ← c3617, ← c3618] at e_53_0
  have e_53_1 := arithEq_of_rows h (row := 53) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_1
  simp only [← c3619, ← c3620, ← c3621] at e_53_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_0
    linear_combination c3625.trans k2 - hc
  · have hc := e_51_15
    simp only [e_51_14, e_53_1] at hc
    linear_combination c3626.trans k2 - hc

theorem publicBatchWrapper89_f469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10807)) (a (.virt 200)) (a (.virt 18243)) (a (.virt 18244)) := by
  have c3627 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3628 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3629 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3630 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3631 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3632 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3633 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3634 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3635 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3636 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3637 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3638 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3639 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3640 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_16 := arithEq_of_rows h (row := 51) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_16
  simp only [← c3627, k3, ← c3628, k3, ← c3629] at e_51_16
  have e_51_17 := arithEq_of_rows h (row := 51) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_17
  simp only [← c3636, ← c3637, k3, ← c3638] at e_51_17
  have e_53_2 := arithEq_of_rows h (row := 53) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_2
  simp only [← c3630, ← c3631, ← c3632] at e_53_2
  have e_53_3 := arithEq_of_rows h (row := 53) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_3
  simp only [← c3633, ← c3634, ← c3635] at e_53_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_2
    linear_combination c3639.trans k2 - hc
  · have hc := e_51_17
    simp only [e_51_16, e_53_3] at hc
    linear_combination c3640.trans k2 - hc

theorem publicBatchWrapper89_f470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 10808)) (a (.virt 200)) (a (.virt 18245)) (a (.virt 18246)) := by
  have c3641 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3642 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3643 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3644 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3645 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3646 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3647 := (publicBatchWrapper89_copies113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3648 := (publicBatchWrapper89_copies114 a h).1
  have c3649 := (publicBatchWrapper89_copies114 a h).2.1
  have c3650 := (publicBatchWrapper89_copies114 a h).2.2.1
  have c3651 := (publicBatchWrapper89_copies114 a h).2.2.2.1
  have c3652 := (publicBatchWrapper89_copies114 a h).2.2.2.2.1
  have c3653 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.1
  have c3654 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_51_18 := arithEq_of_rows h (row := 51) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_18
  simp only [← c3641, k3, ← c3642, k3, ← c3643] at e_51_18
  have e_51_19 := arithEq_of_rows h (row := 51) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_19
  simp only [← c3650, ← c3651, k3, ← c3652] at e_51_19
  have e_53_4 := arithEq_of_rows h (row := 53) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_4
  simp only [← c3644, ← c3645, ← c3646] at e_53_4
  have e_53_5 := arithEq_of_rows h (row := 53) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_5
  simp only [← c3647, ← c3648, ← c3649] at e_53_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_4
    linear_combination c3653.trans k2 - hc
  · have hc := e_51_19
    simp only [e_51_18, e_53_5] at hc
    linear_combination c3654.trans k2 - hc

theorem publicBatchWrapper89_f471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 53 27) = band (a (.virt 18239)) (a (.virt 18241)) := by
  have c3655 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.1
  have c3656 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.1
  have c3657 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.1
  have e_53_6 := arithEq_of_rows h (row := 53) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_6
  simp only [← c3655, ← c3656, ← c3657] at e_53_6
  have hr := e_53_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 53 31) = band (a (.virt 18243)) (a (.virt 18245)) := by
  have c3658 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3659 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3660 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_7 := arithEq_of_rows h (row := 53) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_7
  simp only [← c3658, ← c3659, ← c3660] at e_53_7
  have hr := e_53_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 53 35) = band (a (.wire 53 27)) (a (.wire 53 31)) := by
  have c3661 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3662 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3663 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_8 := arithEq_of_rows h (row := 53) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_8
  simp only [← c3661, ← c3662, ← c3663] at e_53_8
  have hr := e_53_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11005)) (a (.virt 200)) (a (.virt 18247)) (a (.virt 18248)) := by
  have c3664 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3665 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3666 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3667 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3668 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3669 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3670 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3671 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3672 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3673 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3674 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3675 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3676 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3677 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_53_9 := arithEq_of_rows h (row := 53) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_9
  simp only [← c3667, ← c3668, ← c3669] at e_53_9
  have e_53_10 := arithEq_of_rows h (row := 53) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_10
  simp only [← c3670, ← c3671, ← c3672] at e_53_10
  have e_54_0 := arithEq_of_rows h (row := 54) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_0
  simp only [← c3664, k3, ← c3665, k3, ← c3666] at e_54_0
  have e_54_1 := arithEq_of_rows h (row := 54) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_1
  simp only [← c3673, ← c3674, k3, ← c3675] at e_54_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_9
    linear_combination c3676.trans k2 - hc
  · have hc := e_54_1
    simp only [e_54_0, e_53_10] at hc
    linear_combination c3677.trans k2 - hc

theorem publicBatchWrapper89_f475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11006)) (a (.virt 200)) (a (.virt 18249)) (a (.virt 18250)) := by
  have c3678 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3679 := (publicBatchWrapper89_copies114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3680 := (publicBatchWrapper89_copies115 a h).1
  have c3681 := (publicBatchWrapper89_copies115 a h).2.1
  have c3682 := (publicBatchWrapper89_copies115 a h).2.2.1
  have c3683 := (publicBatchWrapper89_copies115 a h).2.2.2.1
  have c3684 := (publicBatchWrapper89_copies115 a h).2.2.2.2.1
  have c3685 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.1
  have c3686 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.1
  have c3687 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.1
  have c3688 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.1
  have c3689 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.1
  have c3690 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3691 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_53_11 := arithEq_of_rows h (row := 53) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_11
  simp only [← c3681, ← c3682, ← c3683] at e_53_11
  have e_53_12 := arithEq_of_rows h (row := 53) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_12
  simp only [← c3684, ← c3685, ← c3686] at e_53_12
  have e_54_2 := arithEq_of_rows h (row := 54) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_2
  simp only [← c3678, k3, ← c3679, k3, ← c3680] at e_54_2
  have e_54_3 := arithEq_of_rows h (row := 54) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_3
  simp only [← c3687, ← c3688, k3, ← c3689] at e_54_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_11
    linear_combination c3690.trans k2 - hc
  · have hc := e_54_3
    simp only [e_54_2, e_53_12] at hc
    linear_combination c3691.trans k2 - hc

theorem publicBatchWrapper89_f476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11007)) (a (.virt 200)) (a (.virt 18251)) (a (.virt 18252)) := by
  have c3692 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3693 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3694 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3695 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3696 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3697 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3698 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3699 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3700 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3701 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3702 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3703 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3704 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3705 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_53_13 := arithEq_of_rows h (row := 53) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_13
  simp only [← c3695, ← c3696, ← c3697] at e_53_13
  have e_53_14 := arithEq_of_rows h (row := 53) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_14
  simp only [← c3698, ← c3699, ← c3700] at e_53_14
  have e_54_4 := arithEq_of_rows h (row := 54) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_4
  simp only [← c3692, k3, ← c3693, k3, ← c3694] at e_54_4
  have e_54_5 := arithEq_of_rows h (row := 54) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_5
  simp only [← c3701, ← c3702, k3, ← c3703] at e_54_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_13
    linear_combination c3704.trans k2 - hc
  · have hc := e_54_5
    simp only [e_54_4, e_53_14] at hc
    linear_combination c3705.trans k2 - hc

theorem publicBatchWrapper89_f477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11008)) (a (.virt 200)) (a (.virt 18253)) (a (.virt 18254)) := by
  have c3706 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3707 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3708 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3709 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3710 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3711 := (publicBatchWrapper89_copies115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3712 := (publicBatchWrapper89_copies116 a h).1
  have c3713 := (publicBatchWrapper89_copies116 a h).2.1
  have c3714 := (publicBatchWrapper89_copies116 a h).2.2.1
  have c3715 := (publicBatchWrapper89_copies116 a h).2.2.2.1
  have c3716 := (publicBatchWrapper89_copies116 a h).2.2.2.2.1
  have c3717 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.1
  have c3718 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.1
  have c3719 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_53_15 := arithEq_of_rows h (row := 53) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_15
  simp only [← c3709, ← c3710, ← c3711] at e_53_15
  have e_53_16 := arithEq_of_rows h (row := 53) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_16
  simp only [← c3712, ← c3713, ← c3714] at e_53_16
  have e_54_6 := arithEq_of_rows h (row := 54) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_6
  simp only [← c3706, k3, ← c3707, k3, ← c3708] at e_54_6
  have e_54_7 := arithEq_of_rows h (row := 54) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_7
  simp only [← c3715, ← c3716, k3, ← c3717] at e_54_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_53_15
    linear_combination c3718.trans k2 - hc
  · have hc := e_54_7
    simp only [e_54_6, e_53_16] at hc
    linear_combination c3719.trans k2 - hc

theorem publicBatchWrapper89_f478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 53 71) = band (a (.virt 18247)) (a (.virt 18249)) := by
  have c3720 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.1
  have c3721 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.1
  have c3722 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_53_17 := arithEq_of_rows h (row := 53) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_17
  simp only [← c3720, ← c3721, ← c3722] at e_53_17
  have hr := e_53_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 53 75) = band (a (.virt 18251)) (a (.virt 18253)) := by
  have c3723 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3724 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3725 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_18 := arithEq_of_rows h (row := 53) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_18
  simp only [← c3723, ← c3724, ← c3725] at e_53_18
  have hr := e_53_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 53 79) = band (a (.wire 53 71)) (a (.wire 53 75)) := by
  have c3726 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3727 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3728 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_53_19 := arithEq_of_rows h (row := 53) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_19
  simp only [← c3726, ← c3727, ← c3728] at e_53_19
  have hr := e_53_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11205)) (a (.virt 200)) (a (.virt 18255)) (a (.virt 18256)) := by
  have c3729 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3730 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3731 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3732 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3733 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3734 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3735 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3736 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3737 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3738 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3739 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3740 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3741 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3742 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_54_8 := arithEq_of_rows h (row := 54) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_8
  simp only [← c3729, k3, ← c3730, k3, ← c3731] at e_54_8
  have e_54_9 := arithEq_of_rows h (row := 54) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_9
  simp only [← c3738, ← c3739, k3, ← c3740] at e_54_9
  have e_55_0 := arithEq_of_rows h (row := 55) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_0
  simp only [← c3732, ← c3733, ← c3734] at e_55_0
  have e_55_1 := arithEq_of_rows h (row := 55) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_1
  simp only [← c3735, ← c3736, ← c3737] at e_55_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_0
    linear_combination c3741.trans k2 - hc
  · have hc := e_54_9
    simp only [e_54_8, e_55_1] at hc
    linear_combination c3742.trans k2 - hc

theorem publicBatchWrapper89_f482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11206)) (a (.virt 200)) (a (.virt 18257)) (a (.virt 18258)) := by
  have c3743 := (publicBatchWrapper89_copies116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3744 := (publicBatchWrapper89_copies117 a h).1
  have c3745 := (publicBatchWrapper89_copies117 a h).2.1
  have c3746 := (publicBatchWrapper89_copies117 a h).2.2.1
  have c3747 := (publicBatchWrapper89_copies117 a h).2.2.2.1
  have c3748 := (publicBatchWrapper89_copies117 a h).2.2.2.2.1
  have c3749 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.1
  have c3750 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.1
  have c3751 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.1
  have c3752 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.1
  have c3753 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.1
  have c3754 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3755 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3756 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_54_10 := arithEq_of_rows h (row := 54) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_10
  simp only [← c3743, k3, ← c3744, k3, ← c3745] at e_54_10
  have e_54_11 := arithEq_of_rows h (row := 54) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_11
  simp only [← c3752, ← c3753, k3, ← c3754] at e_54_11
  have e_55_2 := arithEq_of_rows h (row := 55) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_2
  simp only [← c3746, ← c3747, ← c3748] at e_55_2
  have e_55_3 := arithEq_of_rows h (row := 55) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_3
  simp only [← c3749, ← c3750, ← c3751] at e_55_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_2
    linear_combination c3755.trans k2 - hc
  · have hc := e_54_11
    simp only [e_54_10, e_55_3] at hc
    linear_combination c3756.trans k2 - hc

theorem publicBatchWrapper89_f483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11207)) (a (.virt 200)) (a (.virt 18259)) (a (.virt 18260)) := by
  have c3757 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3758 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3759 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3760 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3761 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3762 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3763 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3764 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3765 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3766 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3767 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3768 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3769 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3770 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_54_12 := arithEq_of_rows h (row := 54) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_12
  simp only [← c3757, k3, ← c3758, k3, ← c3759] at e_54_12
  have e_54_13 := arithEq_of_rows h (row := 54) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_13
  simp only [← c3766, ← c3767, k3, ← c3768] at e_54_13
  have e_55_4 := arithEq_of_rows h (row := 55) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_4
  simp only [← c3760, ← c3761, ← c3762] at e_55_4
  have e_55_5 := arithEq_of_rows h (row := 55) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_5
  simp only [← c3763, ← c3764, ← c3765] at e_55_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_4
    linear_combination c3769.trans k2 - hc
  · have hc := e_54_13
    simp only [e_54_12, e_55_5] at hc
    linear_combination c3770.trans k2 - hc

theorem publicBatchWrapper89_f484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11208)) (a (.virt 200)) (a (.virt 18261)) (a (.virt 18262)) := by
  have c3771 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3772 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3773 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3774 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3775 := (publicBatchWrapper89_copies117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3776 := (publicBatchWrapper89_copies118 a h).1
  have c3777 := (publicBatchWrapper89_copies118 a h).2.1
  have c3778 := (publicBatchWrapper89_copies118 a h).2.2.1
  have c3779 := (publicBatchWrapper89_copies118 a h).2.2.2.1
  have c3780 := (publicBatchWrapper89_copies118 a h).2.2.2.2.1
  have c3781 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.1
  have c3782 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.1
  have c3783 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.1
  have c3784 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_54_14 := arithEq_of_rows h (row := 54) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_14
  simp only [← c3771, k3, ← c3772, k3, ← c3773] at e_54_14
  have e_54_15 := arithEq_of_rows h (row := 54) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_15
  simp only [← c3780, ← c3781, k3, ← c3782] at e_54_15
  have e_55_6 := arithEq_of_rows h (row := 55) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_6
  simp only [← c3774, ← c3775, ← c3776] at e_55_6
  have e_55_7 := arithEq_of_rows h (row := 55) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_7
  simp only [← c3777, ← c3778, ← c3779] at e_55_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_6
    linear_combination c3783.trans k2 - hc
  · have hc := e_54_15
    simp only [e_54_14, e_55_7] at hc
    linear_combination c3784.trans k2 - hc

theorem publicBatchWrapper89_f485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 55 35) = band (a (.virt 18255)) (a (.virt 18257)) := by
  have c3785 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.1
  have c3786 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3787 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_8 := arithEq_of_rows h (row := 55) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_8
  simp only [← c3785, ← c3786, ← c3787] at e_55_8
  have hr := e_55_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 55 39) = band (a (.virt 18259)) (a (.virt 18261)) := by
  have c3788 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3789 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3790 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_9 := arithEq_of_rows h (row := 55) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_9
  simp only [← c3788, ← c3789, ← c3790] at e_55_9
  have hr := e_55_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 55 43) = band (a (.wire 55 35)) (a (.wire 55 39)) := by
  have c3791 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3792 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3793 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_10 := arithEq_of_rows h (row := 55) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_10
  simp only [← c3791, ← c3792, ← c3793] at e_55_10
  have hr := e_55_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11405)) (a (.virt 200)) (a (.virt 18263)) (a (.virt 18264)) := by
  have c3794 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3795 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3796 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3797 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3798 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3799 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3800 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3801 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3802 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3803 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3804 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3805 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3806 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3807 := (publicBatchWrapper89_copies118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_54_16 := arithEq_of_rows h (row := 54) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_16
  simp only [← c3794, k3, ← c3795, k3, ← c3796] at e_54_16
  have e_54_17 := arithEq_of_rows h (row := 54) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_17
  simp only [← c3803, ← c3804, k3, ← c3805] at e_54_17
  have e_55_11 := arithEq_of_rows h (row := 55) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_11
  simp only [← c3797, ← c3798, ← c3799] at e_55_11
  have e_55_12 := arithEq_of_rows h (row := 55) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_12
  simp only [← c3800, ← c3801, ← c3802] at e_55_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_11
    linear_combination c3806.trans k2 - hc
  · have hc := e_54_17
    simp only [e_54_16, e_55_12] at hc
    linear_combination c3807.trans k2 - hc

theorem publicBatchWrapper89_f489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11406)) (a (.virt 200)) (a (.virt 18265)) (a (.virt 18266)) := by
  have c3808 := (publicBatchWrapper89_copies119 a h).1
  have c3809 := (publicBatchWrapper89_copies119 a h).2.1
  have c3810 := (publicBatchWrapper89_copies119 a h).2.2.1
  have c3811 := (publicBatchWrapper89_copies119 a h).2.2.2.1
  have c3812 := (publicBatchWrapper89_copies119 a h).2.2.2.2.1
  have c3813 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.1
  have c3814 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.1
  have c3815 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.1
  have c3816 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.1
  have c3817 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.1
  have c3818 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3819 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3820 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3821 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_54_18 := arithEq_of_rows h (row := 54) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_18
  simp only [← c3808, k3, ← c3809, k3, ← c3810] at e_54_18
  have e_54_19 := arithEq_of_rows h (row := 54) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_19
  simp only [← c3817, ← c3818, k3, ← c3819] at e_54_19
  have e_55_13 := arithEq_of_rows h (row := 55) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_13
  simp only [← c3811, ← c3812, ← c3813] at e_55_13
  have e_55_14 := arithEq_of_rows h (row := 55) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_14
  simp only [← c3814, ← c3815, ← c3816] at e_55_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_13
    linear_combination c3820.trans k2 - hc
  · have hc := e_54_19
    simp only [e_54_18, e_55_14] at hc
    linear_combination c3821.trans k2 - hc

theorem publicBatchWrapper89_f490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11407)) (a (.virt 200)) (a (.virt 18267)) (a (.virt 18268)) := by
  have c3822 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3823 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3824 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3825 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3826 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3827 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3828 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3829 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3830 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3831 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3832 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3833 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3834 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3835 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_55_15 := arithEq_of_rows h (row := 55) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_15
  simp only [← c3825, ← c3826, ← c3827] at e_55_15
  have e_55_16 := arithEq_of_rows h (row := 55) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_16
  simp only [← c3828, ← c3829, ← c3830] at e_55_16
  have e_56_0 := arithEq_of_rows h (row := 56) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_0
  simp only [← c3822, k3, ← c3823, k3, ← c3824] at e_56_0
  have e_56_1 := arithEq_of_rows h (row := 56) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_1
  simp only [← c3831, ← c3832, k3, ← c3833] at e_56_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_15
    linear_combination c3834.trans k2 - hc
  · have hc := e_56_1
    simp only [e_56_0, e_55_16] at hc
    linear_combination c3835.trans k2 - hc

theorem publicBatchWrapper89_f491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11408)) (a (.virt 200)) (a (.virt 18269)) (a (.virt 18270)) := by
  have c3836 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3837 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3838 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3839 := (publicBatchWrapper89_copies119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3840 := (publicBatchWrapper89_copies120 a h).1
  have c3841 := (publicBatchWrapper89_copies120 a h).2.1
  have c3842 := (publicBatchWrapper89_copies120 a h).2.2.1
  have c3843 := (publicBatchWrapper89_copies120 a h).2.2.2.1
  have c3844 := (publicBatchWrapper89_copies120 a h).2.2.2.2.1
  have c3845 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.1
  have c3846 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.1
  have c3847 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.1
  have c3848 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.1
  have c3849 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_55_17 := arithEq_of_rows h (row := 55) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_17
  simp only [← c3839, ← c3840, ← c3841] at e_55_17
  have e_55_18 := arithEq_of_rows h (row := 55) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_18
  simp only [← c3842, ← c3843, ← c3844] at e_55_18
  have e_56_2 := arithEq_of_rows h (row := 56) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_2
  simp only [← c3836, k3, ← c3837, k3, ← c3838] at e_56_2
  have e_56_3 := arithEq_of_rows h (row := 56) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_3
  simp only [← c3845, ← c3846, k3, ← c3847] at e_56_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_55_17
    linear_combination c3848.trans k2 - hc
  · have hc := e_56_3
    simp only [e_56_2, e_55_18] at hc
    linear_combination c3849.trans k2 - hc

theorem publicBatchWrapper89_f492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 55 79) = band (a (.virt 18263)) (a (.virt 18265)) := by
  have c3850 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3851 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3852 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_55_19 := arithEq_of_rows h (row := 55) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_19
  simp only [← c3850, ← c3851, ← c3852] at e_55_19
  have hr := e_55_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 57 3) = band (a (.virt 18267)) (a (.virt 18269)) := by
  have c3853 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3854 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3855 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_57_0 := arithEq_of_rows h (row := 57) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_0
  simp only [← c3853, ← c3854, ← c3855] at e_57_0
  have hr := e_57_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 57 7) = band (a (.wire 55 79)) (a (.wire 57 3)) := by
  have c3856 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3857 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3858 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_57_1 := arithEq_of_rows h (row := 57) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_1
  simp only [← c3856, ← c3857, ← c3858] at e_57_1
  have hr := e_57_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11605)) (a (.virt 200)) (a (.virt 18271)) (a (.virt 18272)) := by
  have c3859 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3860 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3861 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3862 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3863 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3864 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3865 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3866 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3867 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3868 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3869 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3870 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3871 := (publicBatchWrapper89_copies120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3872 := (publicBatchWrapper89_copies121 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_4 := arithEq_of_rows h (row := 56) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_4
  simp only [← c3859, k3, ← c3860, k3, ← c3861] at e_56_4
  have e_56_5 := arithEq_of_rows h (row := 56) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_5
  simp only [← c3868, ← c3869, k3, ← c3870] at e_56_5
  have e_57_2 := arithEq_of_rows h (row := 57) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_2
  simp only [← c3862, ← c3863, ← c3864] at e_57_2
  have e_57_3 := arithEq_of_rows h (row := 57) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_3
  simp only [← c3865, ← c3866, ← c3867] at e_57_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_2
    linear_combination c3871.trans k2 - hc
  · have hc := e_56_5
    simp only [e_56_4, e_57_3] at hc
    linear_combination c3872.trans k2 - hc

theorem publicBatchWrapper89_f496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11606)) (a (.virt 200)) (a (.virt 18273)) (a (.virt 18274)) := by
  have c3873 := (publicBatchWrapper89_copies121 a h).2.1
  have c3874 := (publicBatchWrapper89_copies121 a h).2.2.1
  have c3875 := (publicBatchWrapper89_copies121 a h).2.2.2.1
  have c3876 := (publicBatchWrapper89_copies121 a h).2.2.2.2.1
  have c3877 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.1
  have c3878 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.1
  have c3879 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.1
  have c3880 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.1
  have c3881 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.1
  have c3882 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3883 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3884 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3885 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3886 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_6 := arithEq_of_rows h (row := 56) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_6
  simp only [← c3873, k3, ← c3874, k3, ← c3875] at e_56_6
  have e_56_7 := arithEq_of_rows h (row := 56) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_7
  simp only [← c3882, ← c3883, k3, ← c3884] at e_56_7
  have e_57_4 := arithEq_of_rows h (row := 57) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_4
  simp only [← c3876, ← c3877, ← c3878] at e_57_4
  have e_57_5 := arithEq_of_rows h (row := 57) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_5
  simp only [← c3879, ← c3880, ← c3881] at e_57_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_4
    linear_combination c3885.trans k2 - hc
  · have hc := e_56_7
    simp only [e_56_6, e_57_5] at hc
    linear_combination c3886.trans k2 - hc

theorem publicBatchWrapper89_f497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11607)) (a (.virt 200)) (a (.virt 18275)) (a (.virt 18276)) := by
  have c3887 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3888 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3889 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3890 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3891 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3892 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3893 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3894 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3895 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3896 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3897 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3898 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3899 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3900 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_8 := arithEq_of_rows h (row := 56) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_8
  simp only [← c3887, k3, ← c3888, k3, ← c3889] at e_56_8
  have e_56_9 := arithEq_of_rows h (row := 56) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_9
  simp only [← c3896, ← c3897, k3, ← c3898] at e_56_9
  have e_57_6 := arithEq_of_rows h (row := 57) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_6
  simp only [← c3890, ← c3891, ← c3892] at e_57_6
  have e_57_7 := arithEq_of_rows h (row := 57) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_7
  simp only [← c3893, ← c3894, ← c3895] at e_57_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_6
    linear_combination c3899.trans k2 - hc
  · have hc := e_56_9
    simp only [e_56_8, e_57_7] at hc
    linear_combination c3900.trans k2 - hc

theorem publicBatchWrapper89_f498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11608)) (a (.virt 200)) (a (.virt 18277)) (a (.virt 18278)) := by
  have c3901 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3902 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3903 := (publicBatchWrapper89_copies121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3904 := (publicBatchWrapper89_copies122 a h).1
  have c3905 := (publicBatchWrapper89_copies122 a h).2.1
  have c3906 := (publicBatchWrapper89_copies122 a h).2.2.1
  have c3907 := (publicBatchWrapper89_copies122 a h).2.2.2.1
  have c3908 := (publicBatchWrapper89_copies122 a h).2.2.2.2.1
  have c3909 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.1
  have c3910 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.1
  have c3911 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.1
  have c3912 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.1
  have c3913 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.1
  have c3914 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_10 := arithEq_of_rows h (row := 56) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_10
  simp only [← c3901, k3, ← c3902, k3, ← c3903] at e_56_10
  have e_56_11 := arithEq_of_rows h (row := 56) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_11
  simp only [← c3910, ← c3911, k3, ← c3912] at e_56_11
  have e_57_8 := arithEq_of_rows h (row := 57) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_8
  simp only [← c3904, ← c3905, ← c3906] at e_57_8
  have e_57_9 := arithEq_of_rows h (row := 57) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_9
  simp only [← c3907, ← c3908, ← c3909] at e_57_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_8
    linear_combination c3913.trans k2 - hc
  · have hc := e_56_11
    simp only [e_56_10, e_57_9] at hc
    linear_combination c3914.trans k2 - hc

theorem publicBatchWrapper89_f499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 57 43) = band (a (.virt 18271)) (a (.virt 18273)) := by
  have c3915 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3916 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3917 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_57_10 := arithEq_of_rows h (row := 57) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_10
  simp only [← c3915, ← c3916, ← c3917] at e_57_10
  have hr := e_57_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 57 47) = band (a (.virt 18275)) (a (.virt 18277)) := by
  have c3918 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3919 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3920 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_57_11 := arithEq_of_rows h (row := 57) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_11
  simp only [← c3918, ← c3919, ← c3920] at e_57_11
  have hr := e_57_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 57 51) = band (a (.wire 57 43)) (a (.wire 57 47)) := by
  have c3921 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3922 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3923 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_57_12 := arithEq_of_rows h (row := 57) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_12
  simp only [← c3921, ← c3922, ← c3923] at e_57_12
  have hr := e_57_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11805)) (a (.virt 200)) (a (.virt 18279)) (a (.virt 18280)) := by
  have c3924 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3925 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3926 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3927 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3928 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3929 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3930 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3931 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3932 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3933 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3934 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3935 := (publicBatchWrapper89_copies122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3936 := (publicBatchWrapper89_copies123 a h).1
  have c3937 := (publicBatchWrapper89_copies123 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_12 := arithEq_of_rows h (row := 56) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_12
  simp only [← c3924, k3, ← c3925, k3, ← c3926] at e_56_12
  have e_56_13 := arithEq_of_rows h (row := 56) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_13
  simp only [← c3933, ← c3934, k3, ← c3935] at e_56_13
  have e_57_13 := arithEq_of_rows h (row := 57) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_13
  simp only [← c3927, ← c3928, ← c3929] at e_57_13
  have e_57_14 := arithEq_of_rows h (row := 57) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_14
  simp only [← c3930, ← c3931, ← c3932] at e_57_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_13
    linear_combination c3936.trans k2 - hc
  · have hc := e_56_13
    simp only [e_56_12, e_57_14] at hc
    linear_combination c3937.trans k2 - hc

theorem publicBatchWrapper89_f503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11806)) (a (.virt 200)) (a (.virt 18281)) (a (.virt 18282)) := by
  have c3938 := (publicBatchWrapper89_copies123 a h).2.2.1
  have c3939 := (publicBatchWrapper89_copies123 a h).2.2.2.1
  have c3940 := (publicBatchWrapper89_copies123 a h).2.2.2.2.1
  have c3941 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.1
  have c3942 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.1
  have c3943 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.1
  have c3944 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.1
  have c3945 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.1
  have c3946 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3947 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c3948 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3949 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3950 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3951 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_14 := arithEq_of_rows h (row := 56) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_14
  simp only [← c3938, k3, ← c3939, k3, ← c3940] at e_56_14
  have e_56_15 := arithEq_of_rows h (row := 56) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_15
  simp only [← c3947, ← c3948, k3, ← c3949] at e_56_15
  have e_57_15 := arithEq_of_rows h (row := 57) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_15
  simp only [← c3941, ← c3942, ← c3943] at e_57_15
  have e_57_16 := arithEq_of_rows h (row := 57) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_16
  simp only [← c3944, ← c3945, ← c3946] at e_57_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_15
    linear_combination c3950.trans k2 - hc
  · have hc := e_56_15
    simp only [e_56_14, e_57_16] at hc
    linear_combination c3951.trans k2 - hc

theorem publicBatchWrapper89_f504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11807)) (a (.virt 200)) (a (.virt 18283)) (a (.virt 18284)) := by
  have c3952 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3953 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3954 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3955 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3956 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3957 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3958 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3959 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3960 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3961 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3962 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3963 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3964 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3965 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_16 := arithEq_of_rows h (row := 56) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_16
  simp only [← c3952, k3, ← c3953, k3, ← c3954] at e_56_16
  have e_56_17 := arithEq_of_rows h (row := 56) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_17
  simp only [← c3961, ← c3962, k3, ← c3963] at e_56_17
  have e_57_17 := arithEq_of_rows h (row := 57) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_17
  simp only [← c3955, ← c3956, ← c3957] at e_57_17
  have e_57_18 := arithEq_of_rows h (row := 57) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_18
  simp only [← c3958, ← c3959, ← c3960] at e_57_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_17
    linear_combination c3964.trans k2 - hc
  · have hc := e_56_17
    simp only [e_56_16, e_57_18] at hc
    linear_combination c3965.trans k2 - hc

theorem publicBatchWrapper89_f505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 11808)) (a (.virt 200)) (a (.virt 18285)) (a (.virt 18286)) := by
  have c3966 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3967 := (publicBatchWrapper89_copies123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c3968 := (publicBatchWrapper89_copies124 a h).1
  have c3969 := (publicBatchWrapper89_copies124 a h).2.1
  have c3970 := (publicBatchWrapper89_copies124 a h).2.2.1
  have c3971 := (publicBatchWrapper89_copies124 a h).2.2.2.1
  have c3972 := (publicBatchWrapper89_copies124 a h).2.2.2.2.1
  have c3973 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.1
  have c3974 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.1
  have c3975 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.1
  have c3976 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.1
  have c3977 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.1
  have c3978 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c3979 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_56_18 := arithEq_of_rows h (row := 56) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_18
  simp only [← c3966, k3, ← c3967, k3, ← c3968] at e_56_18
  have e_56_19 := arithEq_of_rows h (row := 56) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_56_19
  simp only [← c3975, ← c3976, k3, ← c3977] at e_56_19
  have e_57_19 := arithEq_of_rows h (row := 57) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_19
  simp only [← c3969, ← c3970, ← c3971] at e_57_19
  have e_58_0 := arithEq_of_rows h (row := 58) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_0
  simp only [← c3972, ← c3973, ← c3974] at e_58_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_57_19
    linear_combination c3978.trans k2 - hc
  · have hc := e_56_19
    simp only [e_56_18, e_58_0] at hc
    linear_combination c3979.trans k2 - hc

theorem publicBatchWrapper89_f506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 58 7) = band (a (.virt 18279)) (a (.virt 18281)) := by
  have c3980 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3981 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3982 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_1 := arithEq_of_rows h (row := 58) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_1
  simp only [← c3980, ← c3981, ← c3982] at e_58_1
  have hr := e_58_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 58 11) = band (a (.virt 18283)) (a (.virt 18285)) := by
  have c3983 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3984 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3985 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_2 := arithEq_of_rows h (row := 58) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_2
  simp only [← c3983, ← c3984, ← c3985] at e_58_2
  have hr := e_58_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 58 15) = band (a (.wire 58 7)) (a (.wire 58 11)) := by
  have c3986 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3987 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3988 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_3 := arithEq_of_rows h (row := 58) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_3
  simp only [← c3986, ← c3987, ← c3988] at e_58_3
  have hr := e_58_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12005)) (a (.virt 200)) (a (.virt 18287)) (a (.virt 18288)) := by
  have c3989 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3990 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3991 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3992 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3993 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3994 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3995 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3996 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3997 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3998 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c3999 := (publicBatchWrapper89_copies124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4000 := (publicBatchWrapper89_copies125 a h).1
  have c4001 := (publicBatchWrapper89_copies125 a h).2.1
  have c4002 := (publicBatchWrapper89_copies125 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_4 := arithEq_of_rows h (row := 58) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_4
  simp only [← c3992, ← c3993, ← c3994] at e_58_4
  have e_58_5 := arithEq_of_rows h (row := 58) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_5
  simp only [← c3995, ← c3996, ← c3997] at e_58_5
  have e_59_0 := arithEq_of_rows h (row := 59) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_0
  simp only [← c3989, k3, ← c3990, k3, ← c3991] at e_59_0
  have e_59_1 := arithEq_of_rows h (row := 59) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_1
  simp only [← c3998, ← c3999, k3, ← c4000] at e_59_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_4
    linear_combination c4001.trans k2 - hc
  · have hc := e_59_1
    simp only [e_59_0, e_58_5] at hc
    linear_combination c4002.trans k2 - hc

theorem publicBatchWrapper89_f510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12006)) (a (.virt 200)) (a (.virt 18289)) (a (.virt 18290)) := by
  have c4003 := (publicBatchWrapper89_copies125 a h).2.2.2.1
  have c4004 := (publicBatchWrapper89_copies125 a h).2.2.2.2.1
  have c4005 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.1
  have c4006 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.1
  have c4007 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.1
  have c4008 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.1
  have c4009 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.1
  have c4010 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4011 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4012 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4013 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4014 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4015 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4016 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_6 := arithEq_of_rows h (row := 58) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_6
  simp only [← c4006, ← c4007, ← c4008] at e_58_6
  have e_58_7 := arithEq_of_rows h (row := 58) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_7
  simp only [← c4009, ← c4010, ← c4011] at e_58_7
  have e_59_2 := arithEq_of_rows h (row := 59) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_2
  simp only [← c4003, k3, ← c4004, k3, ← c4005] at e_59_2
  have e_59_3 := arithEq_of_rows h (row := 59) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_3
  simp only [← c4012, ← c4013, k3, ← c4014] at e_59_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_6
    linear_combination c4015.trans k2 - hc
  · have hc := e_59_3
    simp only [e_59_2, e_58_7] at hc
    linear_combination c4016.trans k2 - hc

theorem publicBatchWrapper89_f511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12007)) (a (.virt 200)) (a (.virt 18291)) (a (.virt 18292)) := by
  have c4017 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4018 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4019 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4020 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4021 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4022 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4023 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4024 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4025 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4026 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4027 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4028 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4029 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4030 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_8 := arithEq_of_rows h (row := 58) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_8
  simp only [← c4020, ← c4021, ← c4022] at e_58_8
  have e_58_9 := arithEq_of_rows h (row := 58) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_9
  simp only [← c4023, ← c4024, ← c4025] at e_58_9
  have e_59_4 := arithEq_of_rows h (row := 59) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_4
  simp only [← c4017, k3, ← c4018, k3, ← c4019] at e_59_4
  have e_59_5 := arithEq_of_rows h (row := 59) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_5
  simp only [← c4026, ← c4027, k3, ← c4028] at e_59_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_8
    linear_combination c4029.trans k2 - hc
  · have hc := e_59_5
    simp only [e_59_4, e_58_9] at hc
    linear_combination c4030.trans k2 - hc

theorem publicBatchWrapper89_f512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12008)) (a (.virt 200)) (a (.virt 18293)) (a (.virt 18294)) := by
  have c4031 := (publicBatchWrapper89_copies125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4032 := (publicBatchWrapper89_copies126 a h).1
  have c4033 := (publicBatchWrapper89_copies126 a h).2.1
  have c4034 := (publicBatchWrapper89_copies126 a h).2.2.1
  have c4035 := (publicBatchWrapper89_copies126 a h).2.2.2.1
  have c4036 := (publicBatchWrapper89_copies126 a h).2.2.2.2.1
  have c4037 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.1
  have c4038 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.1
  have c4039 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.1
  have c4040 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.1
  have c4041 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.1
  have c4042 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4043 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4044 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_10 := arithEq_of_rows h (row := 58) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_10
  simp only [← c4034, ← c4035, ← c4036] at e_58_10
  have e_58_11 := arithEq_of_rows h (row := 58) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_11
  simp only [← c4037, ← c4038, ← c4039] at e_58_11
  have e_59_6 := arithEq_of_rows h (row := 59) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_6
  simp only [← c4031, k3, ← c4032, k3, ← c4033] at e_59_6
  have e_59_7 := arithEq_of_rows h (row := 59) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_7
  simp only [← c4040, ← c4041, k3, ← c4042] at e_59_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_10
    linear_combination c4043.trans k2 - hc
  · have hc := e_59_7
    simp only [e_59_6, e_58_11] at hc
    linear_combination c4044.trans k2 - hc

theorem publicBatchWrapper89_f513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 58 51) = band (a (.virt 18287)) (a (.virt 18289)) := by
  have c4045 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4046 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4047 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_12 := arithEq_of_rows h (row := 58) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_12
  simp only [← c4045, ← c4046, ← c4047] at e_58_12
  have hr := e_58_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 58 55) = band (a (.virt 18291)) (a (.virt 18293)) := by
  have c4048 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4049 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4050 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_13 := arithEq_of_rows h (row := 58) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_13
  simp only [← c4048, ← c4049, ← c4050] at e_58_13
  have hr := e_58_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 58 59) = band (a (.wire 58 51)) (a (.wire 58 55)) := by
  have c4051 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4052 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4053 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_14 := arithEq_of_rows h (row := 58) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_14
  simp only [← c4051, ← c4052, ← c4053] at e_58_14
  have hr := e_58_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12205)) (a (.virt 200)) (a (.virt 18295)) (a (.virt 18296)) := by
  have c4054 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4055 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4056 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4057 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4058 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4059 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4060 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4061 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4062 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4063 := (publicBatchWrapper89_copies126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4064 := (publicBatchWrapper89_copies127 a h).1
  have c4065 := (publicBatchWrapper89_copies127 a h).2.1
  have c4066 := (publicBatchWrapper89_copies127 a h).2.2.1
  have c4067 := (publicBatchWrapper89_copies127 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_15 := arithEq_of_rows h (row := 58) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_15
  simp only [← c4057, ← c4058, ← c4059] at e_58_15
  have e_58_16 := arithEq_of_rows h (row := 58) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_16
  simp only [← c4060, ← c4061, ← c4062] at e_58_16
  have e_59_8 := arithEq_of_rows h (row := 59) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_8
  simp only [← c4054, k3, ← c4055, k3, ← c4056] at e_59_8
  have e_59_9 := arithEq_of_rows h (row := 59) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_9
  simp only [← c4063, ← c4064, k3, ← c4065] at e_59_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_15
    linear_combination c4066.trans k2 - hc
  · have hc := e_59_9
    simp only [e_59_8, e_58_16] at hc
    linear_combination c4067.trans k2 - hc

theorem publicBatchWrapper89_f517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12206)) (a (.virt 200)) (a (.virt 18297)) (a (.virt 18298)) := by
  have c4068 := (publicBatchWrapper89_copies127 a h).2.2.2.2.1
  have c4069 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.1
  have c4070 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.1
  have c4071 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.1
  have c4072 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.1
  have c4073 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.1
  have c4074 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4075 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4076 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4077 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4078 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4079 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4080 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4081 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_17 := arithEq_of_rows h (row := 58) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_17
  simp only [← c4071, ← c4072, ← c4073] at e_58_17
  have e_58_18 := arithEq_of_rows h (row := 58) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_18
  simp only [← c4074, ← c4075, ← c4076] at e_58_18
  have e_59_10 := arithEq_of_rows h (row := 59) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_10
  simp only [← c4068, k3, ← c4069, k3, ← c4070] at e_59_10
  have e_59_11 := arithEq_of_rows h (row := 59) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_11
  simp only [← c4077, ← c4078, k3, ← c4079] at e_59_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_17
    linear_combination c4080.trans k2 - hc
  · have hc := e_59_11
    simp only [e_59_10, e_58_18] at hc
    linear_combination c4081.trans k2 - hc

theorem publicBatchWrapper89_f518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12207)) (a (.virt 200)) (a (.virt 18299)) (a (.virt 18300)) := by
  have c4082 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4083 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4084 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4085 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4086 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4087 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4088 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4089 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4090 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4091 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4092 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4093 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4094 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4095 := (publicBatchWrapper89_copies127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_58_19 := arithEq_of_rows h (row := 58) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_19
  simp only [← c4085, ← c4086, ← c4087] at e_58_19
  have e_59_12 := arithEq_of_rows h (row := 59) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_12
  simp only [← c4082, k3, ← c4083, k3, ← c4084] at e_59_12
  have e_59_13 := arithEq_of_rows h (row := 59) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_13
  simp only [← c4091, ← c4092, k3, ← c4093] at e_59_13
  have e_60_0 := arithEq_of_rows h (row := 60) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_0
  simp only [← c4088, ← c4089, ← c4090] at e_60_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_58_19
    linear_combination c4094.trans k2 - hc
  · have hc := e_59_13
    simp only [e_59_12, e_60_0] at hc
    linear_combination c4095.trans k2 - hc

theorem publicBatchWrapper89_f519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12208)) (a (.virt 200)) (a (.virt 18301)) (a (.virt 18302)) := by
  have c4096 := (publicBatchWrapper89_copies128 a h).1
  have c4097 := (publicBatchWrapper89_copies128 a h).2.1
  have c4098 := (publicBatchWrapper89_copies128 a h).2.2.1
  have c4099 := (publicBatchWrapper89_copies128 a h).2.2.2.1
  have c4100 := (publicBatchWrapper89_copies128 a h).2.2.2.2.1
  have c4101 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.1
  have c4102 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.1
  have c4103 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.1
  have c4104 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.1
  have c4105 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.1
  have c4106 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4107 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4108 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4109 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_59_14 := arithEq_of_rows h (row := 59) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_14
  simp only [← c4096, k3, ← c4097, k3, ← c4098] at e_59_14
  have e_59_15 := arithEq_of_rows h (row := 59) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_15
  simp only [← c4105, ← c4106, k3, ← c4107] at e_59_15
  have e_60_1 := arithEq_of_rows h (row := 60) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_1
  simp only [← c4099, ← c4100, ← c4101] at e_60_1
  have e_60_2 := arithEq_of_rows h (row := 60) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_2
  simp only [← c4102, ← c4103, ← c4104] at e_60_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_1
    linear_combination c4108.trans k2 - hc
  · have hc := e_59_15
    simp only [e_59_14, e_60_2] at hc
    linear_combination c4109.trans k2 - hc

theorem publicBatchWrapper89_f520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 60 15) = band (a (.virt 18295)) (a (.virt 18297)) := by
  have c4110 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4111 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4112 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_3 := arithEq_of_rows h (row := 60) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_3
  simp only [← c4110, ← c4111, ← c4112] at e_60_3
  have hr := e_60_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 60 19) = band (a (.virt 18299)) (a (.virt 18301)) := by
  have c4113 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4114 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4115 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_4 := arithEq_of_rows h (row := 60) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_4
  simp only [← c4113, ← c4114, ← c4115] at e_60_4
  have hr := e_60_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 60 23) = band (a (.wire 60 15)) (a (.wire 60 19)) := by
  have c4116 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4117 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4118 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_5 := arithEq_of_rows h (row := 60) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_5
  simp only [← c4116, ← c4117, ← c4118] at e_60_5
  have hr := e_60_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12405)) (a (.virt 200)) (a (.virt 18303)) (a (.virt 18304)) := by
  have c4119 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4120 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4121 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4122 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4123 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4124 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4125 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4126 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4127 := (publicBatchWrapper89_copies128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4128 := (publicBatchWrapper89_copies129 a h).1
  have c4129 := (publicBatchWrapper89_copies129 a h).2.1
  have c4130 := (publicBatchWrapper89_copies129 a h).2.2.1
  have c4131 := (publicBatchWrapper89_copies129 a h).2.2.2.1
  have c4132 := (publicBatchWrapper89_copies129 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_59_16 := arithEq_of_rows h (row := 59) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_16
  simp only [← c4119, k3, ← c4120, k3, ← c4121] at e_59_16
  have e_59_17 := arithEq_of_rows h (row := 59) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_17
  simp only [← c4128, ← c4129, k3, ← c4130] at e_59_17
  have e_60_6 := arithEq_of_rows h (row := 60) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_6
  simp only [← c4122, ← c4123, ← c4124] at e_60_6
  have e_60_7 := arithEq_of_rows h (row := 60) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_7
  simp only [← c4125, ← c4126, ← c4127] at e_60_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_6
    linear_combination c4131.trans k2 - hc
  · have hc := e_59_17
    simp only [e_59_16, e_60_7] at hc
    linear_combination c4132.trans k2 - hc

theorem publicBatchWrapper89_f524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12406)) (a (.virt 200)) (a (.virt 18305)) (a (.virt 18306)) := by
  have c4133 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.1
  have c4134 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.1
  have c4135 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.1
  have c4136 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.1
  have c4137 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.1
  have c4138 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4139 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4140 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4141 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4142 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4143 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4144 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4145 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4146 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_59_18 := arithEq_of_rows h (row := 59) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_18
  simp only [← c4133, k3, ← c4134, k3, ← c4135] at e_59_18
  have e_59_19 := arithEq_of_rows h (row := 59) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_59_19
  simp only [← c4142, ← c4143, k3, ← c4144] at e_59_19
  have e_60_8 := arithEq_of_rows h (row := 60) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_8
  simp only [← c4136, ← c4137, ← c4138] at e_60_8
  have e_60_9 := arithEq_of_rows h (row := 60) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_9
  simp only [← c4139, ← c4140, ← c4141] at e_60_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_8
    linear_combination c4145.trans k2 - hc
  · have hc := e_59_19
    simp only [e_59_18, e_60_9] at hc
    linear_combination c4146.trans k2 - hc

theorem publicBatchWrapper89_f525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12407)) (a (.virt 200)) (a (.virt 18307)) (a (.virt 18308)) := by
  have c4147 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4148 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4149 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4150 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4151 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4152 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4153 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4154 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4155 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4156 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4157 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4158 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4159 := (publicBatchWrapper89_copies129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4160 := (publicBatchWrapper89_copies130 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_60_10 := arithEq_of_rows h (row := 60) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_10
  simp only [← c4150, ← c4151, ← c4152] at e_60_10
  have e_60_11 := arithEq_of_rows h (row := 60) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_11
  simp only [← c4153, ← c4154, ← c4155] at e_60_11
  have e_61_0 := arithEq_of_rows h (row := 61) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_0
  simp only [← c4147, k3, ← c4148, k3, ← c4149] at e_61_0
  have e_61_1 := arithEq_of_rows h (row := 61) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_1
  simp only [← c4156, ← c4157, k3, ← c4158] at e_61_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_10
    linear_combination c4159.trans k2 - hc
  · have hc := e_61_1
    simp only [e_61_0, e_60_11] at hc
    linear_combination c4160.trans k2 - hc

theorem publicBatchWrapper89_f526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12408)) (a (.virt 200)) (a (.virt 18309)) (a (.virt 18310)) := by
  have c4161 := (publicBatchWrapper89_copies130 a h).2.1
  have c4162 := (publicBatchWrapper89_copies130 a h).2.2.1
  have c4163 := (publicBatchWrapper89_copies130 a h).2.2.2.1
  have c4164 := (publicBatchWrapper89_copies130 a h).2.2.2.2.1
  have c4165 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.1
  have c4166 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.1
  have c4167 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.1
  have c4168 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.1
  have c4169 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.1
  have c4170 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4171 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4172 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4173 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4174 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_60_12 := arithEq_of_rows h (row := 60) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_12
  simp only [← c4164, ← c4165, ← c4166] at e_60_12
  have e_60_13 := arithEq_of_rows h (row := 60) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_13
  simp only [← c4167, ← c4168, ← c4169] at e_60_13
  have e_61_2 := arithEq_of_rows h (row := 61) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_2
  simp only [← c4161, k3, ← c4162, k3, ← c4163] at e_61_2
  have e_61_3 := arithEq_of_rows h (row := 61) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_3
  simp only [← c4170, ← c4171, k3, ← c4172] at e_61_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_12
    linear_combination c4173.trans k2 - hc
  · have hc := e_61_3
    simp only [e_61_2, e_60_13] at hc
    linear_combination c4174.trans k2 - hc

theorem publicBatchWrapper89_f527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 60 59) = band (a (.virt 18303)) (a (.virt 18305)) := by
  have c4175 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4176 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4177 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_14 := arithEq_of_rows h (row := 60) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_14
  simp only [← c4175, ← c4176, ← c4177] at e_60_14
  have hr := e_60_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 60 63) = band (a (.virt 18307)) (a (.virt 18309)) := by
  have c4178 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4179 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4180 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_15 := arithEq_of_rows h (row := 60) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_15
  simp only [← c4178, ← c4179, ← c4180] at e_60_15
  have hr := e_60_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 60 67) = band (a (.wire 60 59)) (a (.wire 60 63)) := by
  have c4181 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4182 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4183 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_60_16 := arithEq_of_rows h (row := 60) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_16
  simp only [← c4181, ← c4182, ← c4183] at e_60_16
  have hr := e_60_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12605)) (a (.virt 200)) (a (.virt 18311)) (a (.virt 18312)) := by
  have c4184 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4185 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4186 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4187 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4188 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4189 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4190 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4191 := (publicBatchWrapper89_copies130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4192 := (publicBatchWrapper89_copies131 a h).1
  have c4193 := (publicBatchWrapper89_copies131 a h).2.1
  have c4194 := (publicBatchWrapper89_copies131 a h).2.2.1
  have c4195 := (publicBatchWrapper89_copies131 a h).2.2.2.1
  have c4196 := (publicBatchWrapper89_copies131 a h).2.2.2.2.1
  have c4197 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_60_17 := arithEq_of_rows h (row := 60) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_17
  simp only [← c4187, ← c4188, ← c4189] at e_60_17
  have e_60_18 := arithEq_of_rows h (row := 60) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_18
  simp only [← c4190, ← c4191, ← c4192] at e_60_18
  have e_61_4 := arithEq_of_rows h (row := 61) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_4
  simp only [← c4184, k3, ← c4185, k3, ← c4186] at e_61_4
  have e_61_5 := arithEq_of_rows h (row := 61) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_5
  simp only [← c4193, ← c4194, k3, ← c4195] at e_61_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_17
    linear_combination c4196.trans k2 - hc
  · have hc := e_61_5
    simp only [e_61_4, e_60_18] at hc
    linear_combination c4197.trans k2 - hc

theorem publicBatchWrapper89_f531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12606)) (a (.virt 200)) (a (.virt 18313)) (a (.virt 18314)) := by
  have c4198 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.1
  have c4199 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.1
  have c4200 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.1
  have c4201 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.1
  have c4202 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4203 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4204 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4205 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4206 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4207 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4208 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4209 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4210 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4211 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_60_19 := arithEq_of_rows h (row := 60) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_60_19
  simp only [← c4201, ← c4202, ← c4203] at e_60_19
  have e_61_6 := arithEq_of_rows h (row := 61) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_6
  simp only [← c4198, k3, ← c4199, k3, ← c4200] at e_61_6
  have e_61_7 := arithEq_of_rows h (row := 61) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_7
  simp only [← c4207, ← c4208, k3, ← c4209] at e_61_7
  have e_62_0 := arithEq_of_rows h (row := 62) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_0
  simp only [← c4204, ← c4205, ← c4206] at e_62_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_60_19
    linear_combination c4210.trans k2 - hc
  · have hc := e_61_7
    simp only [e_61_6, e_62_0] at hc
    linear_combination c4211.trans k2 - hc

theorem publicBatchWrapper89_f532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12607)) (a (.virt 200)) (a (.virt 18315)) (a (.virt 18316)) := by
  have c4212 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4213 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4214 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4215 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4216 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4217 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4218 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4219 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4220 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4221 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4222 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4223 := (publicBatchWrapper89_copies131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4224 := (publicBatchWrapper89_copies132 a h).1
  have c4225 := (publicBatchWrapper89_copies132 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_61_8 := arithEq_of_rows h (row := 61) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_8
  simp only [← c4212, k3, ← c4213, k3, ← c4214] at e_61_8
  have e_61_9 := arithEq_of_rows h (row := 61) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_9
  simp only [← c4221, ← c4222, k3, ← c4223] at e_61_9
  have e_62_1 := arithEq_of_rows h (row := 62) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_1
  simp only [← c4215, ← c4216, ← c4217] at e_62_1
  have e_62_2 := arithEq_of_rows h (row := 62) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_2
  simp only [← c4218, ← c4219, ← c4220] at e_62_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_1
    linear_combination c4224.trans k2 - hc
  · have hc := e_61_9
    simp only [e_61_8, e_62_2] at hc
    linear_combination c4225.trans k2 - hc

theorem publicBatchWrapper89_f533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12608)) (a (.virt 200)) (a (.virt 18317)) (a (.virt 18318)) := by
  have c4226 := (publicBatchWrapper89_copies132 a h).2.2.1
  have c4227 := (publicBatchWrapper89_copies132 a h).2.2.2.1
  have c4228 := (publicBatchWrapper89_copies132 a h).2.2.2.2.1
  have c4229 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.1
  have c4230 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.1
  have c4231 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.1
  have c4232 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.1
  have c4233 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.1
  have c4234 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4235 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4236 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4237 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4238 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4239 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_61_10 := arithEq_of_rows h (row := 61) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_10
  simp only [← c4226, k3, ← c4227, k3, ← c4228] at e_61_10
  have e_61_11 := arithEq_of_rows h (row := 61) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_11
  simp only [← c4235, ← c4236, k3, ← c4237] at e_61_11
  have e_62_3 := arithEq_of_rows h (row := 62) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_3
  simp only [← c4229, ← c4230, ← c4231] at e_62_3
  have e_62_4 := arithEq_of_rows h (row := 62) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_4
  simp only [← c4232, ← c4233, ← c4234] at e_62_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_3
    linear_combination c4238.trans k2 - hc
  · have hc := e_61_11
    simp only [e_61_10, e_62_4] at hc
    linear_combination c4239.trans k2 - hc

theorem publicBatchWrapper89_f534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 62 23) = band (a (.virt 18311)) (a (.virt 18313)) := by
  have c4240 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4241 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4242 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_62_5 := arithEq_of_rows h (row := 62) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_5
  simp only [← c4240, ← c4241, ← c4242] at e_62_5
  have hr := e_62_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 62 27) = band (a (.virt 18315)) (a (.virt 18317)) := by
  have c4243 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4244 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4245 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_62_6 := arithEq_of_rows h (row := 62) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_6
  simp only [← c4243, ← c4244, ← c4245] at e_62_6
  have hr := e_62_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 62 31) = band (a (.wire 62 23)) (a (.wire 62 27)) := by
  have c4246 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4247 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4248 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_62_7 := arithEq_of_rows h (row := 62) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_7
  simp only [← c4246, ← c4247, ← c4248] at e_62_7
  have hr := e_62_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12805)) (a (.virt 200)) (a (.virt 18319)) (a (.virt 18320)) := by
  have c4249 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4250 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4251 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4252 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4253 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4254 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4255 := (publicBatchWrapper89_copies132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4256 := (publicBatchWrapper89_copies133 a h).1
  have c4257 := (publicBatchWrapper89_copies133 a h).2.1
  have c4258 := (publicBatchWrapper89_copies133 a h).2.2.1
  have c4259 := (publicBatchWrapper89_copies133 a h).2.2.2.1
  have c4260 := (publicBatchWrapper89_copies133 a h).2.2.2.2.1
  have c4261 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.1
  have c4262 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_61_12 := arithEq_of_rows h (row := 61) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_12
  simp only [← c4249, k3, ← c4250, k3, ← c4251] at e_61_12
  have e_61_13 := arithEq_of_rows h (row := 61) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_13
  simp only [← c4258, ← c4259, k3, ← c4260] at e_61_13
  have e_62_8 := arithEq_of_rows h (row := 62) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_8
  simp only [← c4252, ← c4253, ← c4254] at e_62_8
  have e_62_9 := arithEq_of_rows h (row := 62) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_9
  simp only [← c4255, ← c4256, ← c4257] at e_62_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_8
    linear_combination c4261.trans k2 - hc
  · have hc := e_61_13
    simp only [e_61_12, e_62_9] at hc
    linear_combination c4262.trans k2 - hc

theorem publicBatchWrapper89_f538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12806)) (a (.virt 200)) (a (.virt 18321)) (a (.virt 18322)) := by
  have c4263 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.1
  have c4264 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.1
  have c4265 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.1
  have c4266 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4267 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4268 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4269 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4270 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4271 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4272 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4273 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4274 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4275 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4276 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_61_14 := arithEq_of_rows h (row := 61) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_14
  simp only [← c4263, k3, ← c4264, k3, ← c4265] at e_61_14
  have e_61_15 := arithEq_of_rows h (row := 61) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_15
  simp only [← c4272, ← c4273, k3, ← c4274] at e_61_15
  have e_62_10 := arithEq_of_rows h (row := 62) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_10
  simp only [← c4266, ← c4267, ← c4268] at e_62_10
  have e_62_11 := arithEq_of_rows h (row := 62) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_11
  simp only [← c4269, ← c4270, ← c4271] at e_62_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_10
    linear_combination c4275.trans k2 - hc
  · have hc := e_61_15
    simp only [e_61_14, e_62_11] at hc
    linear_combination c4276.trans k2 - hc

theorem publicBatchWrapper89_f539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12807)) (a (.virt 200)) (a (.virt 18323)) (a (.virt 18324)) := by
  have c4277 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4278 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4279 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4280 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4281 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4282 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4283 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4284 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4285 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4286 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4287 := (publicBatchWrapper89_copies133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4288 := (publicBatchWrapper89_copies134 a h).1
  have c4289 := (publicBatchWrapper89_copies134 a h).2.1
  have c4290 := (publicBatchWrapper89_copies134 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_61_16 := arithEq_of_rows h (row := 61) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_16
  simp only [← c4277, k3, ← c4278, k3, ← c4279] at e_61_16
  have e_61_17 := arithEq_of_rows h (row := 61) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_17
  simp only [← c4286, ← c4287, k3, ← c4288] at e_61_17
  have e_62_12 := arithEq_of_rows h (row := 62) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_12
  simp only [← c4280, ← c4281, ← c4282] at e_62_12
  have e_62_13 := arithEq_of_rows h (row := 62) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_13
  simp only [← c4283, ← c4284, ← c4285] at e_62_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_12
    linear_combination c4289.trans k2 - hc
  · have hc := e_61_17
    simp only [e_61_16, e_62_13] at hc
    linear_combination c4290.trans k2 - hc

theorem publicBatchWrapper89_f540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 12808)) (a (.virt 200)) (a (.virt 18325)) (a (.virt 18326)) := by
  have c4291 := (publicBatchWrapper89_copies134 a h).2.2.2.1
  have c4292 := (publicBatchWrapper89_copies134 a h).2.2.2.2.1
  have c4293 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.1
  have c4294 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.1
  have c4295 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.1
  have c4296 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.1
  have c4297 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.1
  have c4298 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4299 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4300 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4301 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4302 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4303 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4304 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_61_18 := arithEq_of_rows h (row := 61) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_18
  simp only [← c4291, k3, ← c4292, k3, ← c4293] at e_61_18
  have e_61_19 := arithEq_of_rows h (row := 61) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_61_19
  simp only [← c4300, ← c4301, k3, ← c4302] at e_61_19
  have e_62_14 := arithEq_of_rows h (row := 62) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_14
  simp only [← c4294, ← c4295, ← c4296] at e_62_14
  have e_62_15 := arithEq_of_rows h (row := 62) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_15
  simp only [← c4297, ← c4298, ← c4299] at e_62_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_14
    linear_combination c4303.trans k2 - hc
  · have hc := e_61_19
    simp only [e_61_18, e_62_15] at hc
    linear_combination c4304.trans k2 - hc

theorem publicBatchWrapper89_f541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 62 67) = band (a (.virt 18319)) (a (.virt 18321)) := by
  have c4305 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4306 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4307 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_62_16 := arithEq_of_rows h (row := 62) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_16
  simp only [← c4305, ← c4306, ← c4307] at e_62_16
  have hr := e_62_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 62 71) = band (a (.virt 18323)) (a (.virt 18325)) := by
  have c4308 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4309 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4310 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_62_17 := arithEq_of_rows h (row := 62) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_17
  simp only [← c4308, ← c4309, ← c4310] at e_62_17
  have hr := e_62_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 62 75) = band (a (.wire 62 67)) (a (.wire 62 71)) := by
  have c4311 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4312 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4313 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_62_18 := arithEq_of_rows h (row := 62) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_18
  simp only [← c4311, ← c4312, ← c4313] at e_62_18
  have hr := e_62_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13005)) (a (.virt 200)) (a (.virt 18327)) (a (.virt 18328)) := by
  have c4314 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4315 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4316 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4317 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4318 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4319 := (publicBatchWrapper89_copies134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4320 := (publicBatchWrapper89_copies135 a h).1
  have c4321 := (publicBatchWrapper89_copies135 a h).2.1
  have c4322 := (publicBatchWrapper89_copies135 a h).2.2.1
  have c4323 := (publicBatchWrapper89_copies135 a h).2.2.2.1
  have c4324 := (publicBatchWrapper89_copies135 a h).2.2.2.2.1
  have c4325 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.1
  have c4326 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.1
  have c4327 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_62_19 := arithEq_of_rows h (row := 62) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_62_19
  simp only [← c4317, ← c4318, ← c4319] at e_62_19
  have e_63_0 := arithEq_of_rows h (row := 63) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_0
  simp only [← c4314, k3, ← c4315, k3, ← c4316] at e_63_0
  have e_63_1 := arithEq_of_rows h (row := 63) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_1
  simp only [← c4323, ← c4324, k3, ← c4325] at e_63_1
  have e_64_0 := arithEq_of_rows h (row := 64) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_0
  simp only [← c4320, ← c4321, ← c4322] at e_64_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_62_19
    linear_combination c4326.trans k2 - hc
  · have hc := e_63_1
    simp only [e_63_0, e_64_0] at hc
    linear_combination c4327.trans k2 - hc

theorem publicBatchWrapper89_f545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13006)) (a (.virt 200)) (a (.virt 18329)) (a (.virt 18330)) := by
  have c4328 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.1
  have c4329 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.1
  have c4330 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4331 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4332 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4333 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4334 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4335 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4336 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4337 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4338 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4339 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4340 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4341 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_2 := arithEq_of_rows h (row := 63) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_2
  simp only [← c4328, k3, ← c4329, k3, ← c4330] at e_63_2
  have e_63_3 := arithEq_of_rows h (row := 63) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_3
  simp only [← c4337, ← c4338, k3, ← c4339] at e_63_3
  have e_64_1 := arithEq_of_rows h (row := 64) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_1
  simp only [← c4331, ← c4332, ← c4333] at e_64_1
  have e_64_2 := arithEq_of_rows h (row := 64) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_2
  simp only [← c4334, ← c4335, ← c4336] at e_64_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_1
    linear_combination c4340.trans k2 - hc
  · have hc := e_63_3
    simp only [e_63_2, e_64_2] at hc
    linear_combination c4341.trans k2 - hc

theorem publicBatchWrapper89_f546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13007)) (a (.virt 200)) (a (.virt 18331)) (a (.virt 18332)) := by
  have c4342 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4343 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4344 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4345 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4346 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4347 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4348 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4349 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4350 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4351 := (publicBatchWrapper89_copies135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4352 := (publicBatchWrapper89_copies136 a h).1
  have c4353 := (publicBatchWrapper89_copies136 a h).2.1
  have c4354 := (publicBatchWrapper89_copies136 a h).2.2.1
  have c4355 := (publicBatchWrapper89_copies136 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_4 := arithEq_of_rows h (row := 63) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_4
  simp only [← c4342, k3, ← c4343, k3, ← c4344] at e_63_4
  have e_63_5 := arithEq_of_rows h (row := 63) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_5
  simp only [← c4351, ← c4352, k3, ← c4353] at e_63_5
  have e_64_3 := arithEq_of_rows h (row := 64) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_3
  simp only [← c4345, ← c4346, ← c4347] at e_64_3
  have e_64_4 := arithEq_of_rows h (row := 64) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_4
  simp only [← c4348, ← c4349, ← c4350] at e_64_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_3
    linear_combination c4354.trans k2 - hc
  · have hc := e_63_5
    simp only [e_63_4, e_64_4] at hc
    linear_combination c4355.trans k2 - hc

theorem publicBatchWrapper89_f547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13008)) (a (.virt 200)) (a (.virt 18333)) (a (.virt 18334)) := by
  have c4356 := (publicBatchWrapper89_copies136 a h).2.2.2.2.1
  have c4357 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.1
  have c4358 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.1
  have c4359 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.1
  have c4360 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.1
  have c4361 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.1
  have c4362 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4363 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4364 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4365 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4366 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4367 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4368 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4369 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_6 := arithEq_of_rows h (row := 63) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_6
  simp only [← c4356, k3, ← c4357, k3, ← c4358] at e_63_6
  have e_63_7 := arithEq_of_rows h (row := 63) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_7
  simp only [← c4365, ← c4366, k3, ← c4367] at e_63_7
  have e_64_5 := arithEq_of_rows h (row := 64) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_5
  simp only [← c4359, ← c4360, ← c4361] at e_64_5
  have e_64_6 := arithEq_of_rows h (row := 64) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_6
  simp only [← c4362, ← c4363, ← c4364] at e_64_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_5
    linear_combination c4368.trans k2 - hc
  · have hc := e_63_7
    simp only [e_63_6, e_64_6] at hc
    linear_combination c4369.trans k2 - hc

theorem publicBatchWrapper89_f548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 64 31) = band (a (.virt 18327)) (a (.virt 18329)) := by
  have c4370 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4371 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4372 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_64_7 := arithEq_of_rows h (row := 64) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_7
  simp only [← c4370, ← c4371, ← c4372] at e_64_7
  have hr := e_64_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 64 35) = band (a (.virt 18331)) (a (.virt 18333)) := by
  have c4373 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4374 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4375 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_64_8 := arithEq_of_rows h (row := 64) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_8
  simp only [← c4373, ← c4374, ← c4375] at e_64_8
  have hr := e_64_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 64 39) = band (a (.wire 64 31)) (a (.wire 64 35)) := by
  have c4376 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4377 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4378 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_64_9 := arithEq_of_rows h (row := 64) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_9
  simp only [← c4376, ← c4377, ← c4378] at e_64_9
  have hr := e_64_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13205)) (a (.virt 200)) (a (.virt 18335)) (a (.virt 18336)) := by
  have c4379 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4380 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4381 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4382 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4383 := (publicBatchWrapper89_copies136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4384 := (publicBatchWrapper89_copies137 a h).1
  have c4385 := (publicBatchWrapper89_copies137 a h).2.1
  have c4386 := (publicBatchWrapper89_copies137 a h).2.2.1
  have c4387 := (publicBatchWrapper89_copies137 a h).2.2.2.1
  have c4388 := (publicBatchWrapper89_copies137 a h).2.2.2.2.1
  have c4389 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.1
  have c4390 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.1
  have c4391 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.1
  have c4392 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_8 := arithEq_of_rows h (row := 63) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_8
  simp only [← c4379, k3, ← c4380, k3, ← c4381] at e_63_8
  have e_63_9 := arithEq_of_rows h (row := 63) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_9
  simp only [← c4388, ← c4389, k3, ← c4390] at e_63_9
  have e_64_10 := arithEq_of_rows h (row := 64) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_10
  simp only [← c4382, ← c4383, ← c4384] at e_64_10
  have e_64_11 := arithEq_of_rows h (row := 64) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_11
  simp only [← c4385, ← c4386, ← c4387] at e_64_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_10
    linear_combination c4391.trans k2 - hc
  · have hc := e_63_9
    simp only [e_63_8, e_64_11] at hc
    linear_combination c4392.trans k2 - hc

theorem publicBatchWrapper89_f552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13206)) (a (.virt 200)) (a (.virt 18337)) (a (.virt 18338)) := by
  have c4393 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.1
  have c4394 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4395 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4396 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4397 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4398 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4399 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4400 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4401 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4402 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4403 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4404 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4405 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4406 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_10 := arithEq_of_rows h (row := 63) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_10
  simp only [← c4393, k3, ← c4394, k3, ← c4395] at e_63_10
  have e_63_11 := arithEq_of_rows h (row := 63) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_11
  simp only [← c4402, ← c4403, k3, ← c4404] at e_63_11
  have e_64_12 := arithEq_of_rows h (row := 64) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_12
  simp only [← c4396, ← c4397, ← c4398] at e_64_12
  have e_64_13 := arithEq_of_rows h (row := 64) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_13
  simp only [← c4399, ← c4400, ← c4401] at e_64_13
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_12
    linear_combination c4405.trans k2 - hc
  · have hc := e_63_11
    simp only [e_63_10, e_64_13] at hc
    linear_combination c4406.trans k2 - hc

theorem publicBatchWrapper89_f553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13207)) (a (.virt 200)) (a (.virt 18339)) (a (.virt 18340)) := by
  have c4407 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4408 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4409 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4410 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4411 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4412 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4413 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4414 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4415 := (publicBatchWrapper89_copies137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4416 := (publicBatchWrapper89_copies138 a h).1
  have c4417 := (publicBatchWrapper89_copies138 a h).2.1
  have c4418 := (publicBatchWrapper89_copies138 a h).2.2.1
  have c4419 := (publicBatchWrapper89_copies138 a h).2.2.2.1
  have c4420 := (publicBatchWrapper89_copies138 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_12 := arithEq_of_rows h (row := 63) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_12
  simp only [← c4407, k3, ← c4408, k3, ← c4409] at e_63_12
  have e_63_13 := arithEq_of_rows h (row := 63) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_13
  simp only [← c4416, ← c4417, k3, ← c4418] at e_63_13
  have e_64_14 := arithEq_of_rows h (row := 64) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_14
  simp only [← c4410, ← c4411, ← c4412] at e_64_14
  have e_64_15 := arithEq_of_rows h (row := 64) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_15
  simp only [← c4413, ← c4414, ← c4415] at e_64_15
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_14
    linear_combination c4419.trans k2 - hc
  · have hc := e_63_13
    simp only [e_63_12, e_64_15] at hc
    linear_combination c4420.trans k2 - hc

theorem publicBatchWrapper89_f554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13208)) (a (.virt 200)) (a (.virt 18341)) (a (.virt 18342)) := by
  have c4421 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.1
  have c4422 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.1
  have c4423 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.1
  have c4424 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.1
  have c4425 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.1
  have c4426 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4427 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4428 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4429 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4430 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4431 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4432 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4433 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4434 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_14 := arithEq_of_rows h (row := 63) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_14
  simp only [← c4421, k3, ← c4422, k3, ← c4423] at e_63_14
  have e_63_15 := arithEq_of_rows h (row := 63) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_15
  simp only [← c4430, ← c4431, k3, ← c4432] at e_63_15
  have e_64_16 := arithEq_of_rows h (row := 64) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_16
  simp only [← c4424, ← c4425, ← c4426] at e_64_16
  have e_64_17 := arithEq_of_rows h (row := 64) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_17
  simp only [← c4427, ← c4428, ← c4429] at e_64_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_64_16
    linear_combination c4433.trans k2 - hc
  · have hc := e_63_15
    simp only [e_63_14, e_64_17] at hc
    linear_combination c4434.trans k2 - hc

theorem publicBatchWrapper89_f555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 64 75) = band (a (.virt 18335)) (a (.virt 18337)) := by
  have c4435 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4436 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4437 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_64_18 := arithEq_of_rows h (row := 64) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_18
  simp only [← c4435, ← c4436, ← c4437] at e_64_18
  have hr := e_64_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 64 79) = band (a (.virt 18339)) (a (.virt 18341)) := by
  have c4438 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4439 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4440 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_64_19 := arithEq_of_rows h (row := 64) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_19
  simp only [← c4438, ← c4439, ← c4440] at e_64_19
  have hr := e_64_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 65 3) = band (a (.wire 64 75)) (a (.wire 64 79)) := by
  have c4441 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4442 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4443 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_0 := arithEq_of_rows h (row := 65) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_0
  simp only [← c4441, ← c4442, ← c4443] at e_65_0
  have hr := e_65_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13405)) (a (.virt 200)) (a (.virt 18343)) (a (.virt 18344)) := by
  have c4444 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4445 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4446 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4447 := (publicBatchWrapper89_copies138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4448 := (publicBatchWrapper89_copies139 a h).1
  have c4449 := (publicBatchWrapper89_copies139 a h).2.1
  have c4450 := (publicBatchWrapper89_copies139 a h).2.2.1
  have c4451 := (publicBatchWrapper89_copies139 a h).2.2.2.1
  have c4452 := (publicBatchWrapper89_copies139 a h).2.2.2.2.1
  have c4453 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.1
  have c4454 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.1
  have c4455 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.1
  have c4456 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.1
  have c4457 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_16 := arithEq_of_rows h (row := 63) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_16
  simp only [← c4444, k3, ← c4445, k3, ← c4446] at e_63_16
  have e_63_17 := arithEq_of_rows h (row := 63) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_17
  simp only [← c4453, ← c4454, k3, ← c4455] at e_63_17
  have e_65_1 := arithEq_of_rows h (row := 65) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_1
  simp only [← c4447, ← c4448, ← c4449] at e_65_1
  have e_65_2 := arithEq_of_rows h (row := 65) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_2
  simp only [← c4450, ← c4451, ← c4452] at e_65_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_1
    linear_combination c4456.trans k2 - hc
  · have hc := e_63_17
    simp only [e_63_16, e_65_2] at hc
    linear_combination c4457.trans k2 - hc

theorem publicBatchWrapper89_f559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13406)) (a (.virt 200)) (a (.virt 18345)) (a (.virt 18346)) := by
  have c4458 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4459 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4460 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4461 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4462 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4463 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4464 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4465 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4466 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4467 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4468 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4469 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4470 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4471 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_63_18 := arithEq_of_rows h (row := 63) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_18
  simp only [← c4458, k3, ← c4459, k3, ← c4460] at e_63_18
  have e_63_19 := arithEq_of_rows h (row := 63) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_19
  simp only [← c4467, ← c4468, k3, ← c4469] at e_63_19
  have e_65_3 := arithEq_of_rows h (row := 65) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_3
  simp only [← c4461, ← c4462, ← c4463] at e_65_3
  have e_65_4 := arithEq_of_rows h (row := 65) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_4
  simp only [← c4464, ← c4465, ← c4466] at e_65_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_3
    linear_combination c4470.trans k2 - hc
  · have hc := e_63_19
    simp only [e_63_18, e_65_4] at hc
    linear_combination c4471.trans k2 - hc

theorem publicBatchWrapper89_f560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13407)) (a (.virt 200)) (a (.virt 18347)) (a (.virt 18348)) := by
  have c4472 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4473 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4474 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4475 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4476 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4477 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4478 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4479 := (publicBatchWrapper89_copies139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4480 := (publicBatchWrapper89_copies140 a h).1
  have c4481 := (publicBatchWrapper89_copies140 a h).2.1
  have c4482 := (publicBatchWrapper89_copies140 a h).2.2.1
  have c4483 := (publicBatchWrapper89_copies140 a h).2.2.2.1
  have c4484 := (publicBatchWrapper89_copies140 a h).2.2.2.2.1
  have c4485 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_65_5 := arithEq_of_rows h (row := 65) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_5
  simp only [← c4475, ← c4476, ← c4477] at e_65_5
  have e_65_6 := arithEq_of_rows h (row := 65) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_6
  simp only [← c4478, ← c4479, ← c4480] at e_65_6
  have e_66_0 := arithEq_of_rows h (row := 66) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_0
  simp only [← c4472, k3, ← c4473, k3, ← c4474] at e_66_0
  have e_66_1 := arithEq_of_rows h (row := 66) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_1
  simp only [← c4481, ← c4482, k3, ← c4483] at e_66_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_5
    linear_combination c4484.trans k2 - hc
  · have hc := e_66_1
    simp only [e_66_0, e_65_6] at hc
    linear_combination c4485.trans k2 - hc

theorem publicBatchWrapper89_f561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13408)) (a (.virt 200)) (a (.virt 18349)) (a (.virt 18350)) := by
  have c4486 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.1
  have c4487 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.1
  have c4488 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.1
  have c4489 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.1
  have c4490 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4491 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4492 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4493 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4494 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4495 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4496 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4497 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4498 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4499 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_65_7 := arithEq_of_rows h (row := 65) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_7
  simp only [← c4489, ← c4490, ← c4491] at e_65_7
  have e_65_8 := arithEq_of_rows h (row := 65) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_8
  simp only [← c4492, ← c4493, ← c4494] at e_65_8
  have e_66_2 := arithEq_of_rows h (row := 66) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_2
  simp only [← c4486, k3, ← c4487, k3, ← c4488] at e_66_2
  have e_66_3 := arithEq_of_rows h (row := 66) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_3
  simp only [← c4495, ← c4496, k3, ← c4497] at e_66_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_7
    linear_combination c4498.trans k2 - hc
  · have hc := e_66_3
    simp only [e_66_2, e_65_8] at hc
    linear_combination c4499.trans k2 - hc

theorem publicBatchWrapper89_f562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 65 39) = band (a (.virt 18343)) (a (.virt 18345)) := by
  have c4500 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4501 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4502 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_9 := arithEq_of_rows h (row := 65) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_9
  simp only [← c4500, ← c4501, ← c4502] at e_65_9
  have hr := e_65_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 65 43) = band (a (.virt 18347)) (a (.virt 18349)) := by
  have c4503 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4504 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4505 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_10 := arithEq_of_rows h (row := 65) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_10
  simp only [← c4503, ← c4504, ← c4505] at e_65_10
  have hr := e_65_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 65 47) = band (a (.wire 65 39)) (a (.wire 65 43)) := by
  have c4506 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4507 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4508 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_65_11 := arithEq_of_rows h (row := 65) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_11
  simp only [← c4506, ← c4507, ← c4508] at e_65_11
  have hr := e_65_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13605)) (a (.virt 200)) (a (.virt 18351)) (a (.virt 18352)) := by
  have c4509 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4510 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4511 := (publicBatchWrapper89_copies140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4512 := (publicBatchWrapper89_copies141 a h).1
  have c4513 := (publicBatchWrapper89_copies141 a h).2.1
  have c4514 := (publicBatchWrapper89_copies141 a h).2.2.1
  have c4515 := (publicBatchWrapper89_copies141 a h).2.2.2.1
  have c4516 := (publicBatchWrapper89_copies141 a h).2.2.2.2.1
  have c4517 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.1
  have c4518 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.1
  have c4519 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.1
  have c4520 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.1
  have c4521 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.1
  have c4522 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_65_12 := arithEq_of_rows h (row := 65) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_12
  simp only [← c4512, ← c4513, ← c4514] at e_65_12
  have e_65_13 := arithEq_of_rows h (row := 65) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_13
  simp only [← c4515, ← c4516, ← c4517] at e_65_13
  have e_66_4 := arithEq_of_rows h (row := 66) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_4
  simp only [← c4509, k3, ← c4510, k3, ← c4511] at e_66_4
  have e_66_5 := arithEq_of_rows h (row := 66) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_5
  simp only [← c4518, ← c4519, k3, ← c4520] at e_66_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_12
    linear_combination c4521.trans k2 - hc
  · have hc := e_66_5
    simp only [e_66_4, e_65_13] at hc
    linear_combination c4522.trans k2 - hc

theorem publicBatchWrapper89_f566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13606)) (a (.virt 200)) (a (.virt 18353)) (a (.virt 18354)) := by
  have c4523 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4524 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4525 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4526 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4527 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4528 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4529 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4530 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4531 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4532 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4533 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4534 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4535 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4536 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_65_14 := arithEq_of_rows h (row := 65) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_14
  simp only [← c4526, ← c4527, ← c4528] at e_65_14
  have e_65_15 := arithEq_of_rows h (row := 65) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_15
  simp only [← c4529, ← c4530, ← c4531] at e_65_15
  have e_66_6 := arithEq_of_rows h (row := 66) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_6
  simp only [← c4523, k3, ← c4524, k3, ← c4525] at e_66_6
  have e_66_7 := arithEq_of_rows h (row := 66) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_7
  simp only [← c4532, ← c4533, k3, ← c4534] at e_66_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_14
    linear_combination c4535.trans k2 - hc
  · have hc := e_66_7
    simp only [e_66_6, e_65_15] at hc
    linear_combination c4536.trans k2 - hc

theorem publicBatchWrapper89_f567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13607)) (a (.virt 200)) (a (.virt 18355)) (a (.virt 18356)) := by
  have c4537 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4538 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4539 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4540 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4541 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4542 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4543 := (publicBatchWrapper89_copies141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4544 := (publicBatchWrapper89_copies142 a h).1
  have c4545 := (publicBatchWrapper89_copies142 a h).2.1
  have c4546 := (publicBatchWrapper89_copies142 a h).2.2.1
  have c4547 := (publicBatchWrapper89_copies142 a h).2.2.2.1
  have c4548 := (publicBatchWrapper89_copies142 a h).2.2.2.2.1
  have c4549 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.1
  have c4550 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_65_16 := arithEq_of_rows h (row := 65) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_16
  simp only [← c4540, ← c4541, ← c4542] at e_65_16
  have e_65_17 := arithEq_of_rows h (row := 65) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_17
  simp only [← c4543, ← c4544, ← c4545] at e_65_17
  have e_66_8 := arithEq_of_rows h (row := 66) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_8
  simp only [← c4537, k3, ← c4538, k3, ← c4539] at e_66_8
  have e_66_9 := arithEq_of_rows h (row := 66) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_9
  simp only [← c4546, ← c4547, k3, ← c4548] at e_66_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_16
    linear_combination c4549.trans k2 - hc
  · have hc := e_66_9
    simp only [e_66_8, e_65_17] at hc
    linear_combination c4550.trans k2 - hc

theorem publicBatchWrapper89_f568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13608)) (a (.virt 200)) (a (.virt 18357)) (a (.virt 18358)) := by
  have c4551 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.1
  have c4552 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.1
  have c4553 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.1
  have c4554 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4555 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4556 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4557 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4558 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4559 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4560 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4561 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4562 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4563 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4564 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_65_18 := arithEq_of_rows h (row := 65) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_18
  simp only [← c4554, ← c4555, ← c4556] at e_65_18
  have e_65_19 := arithEq_of_rows h (row := 65) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_65_19
  simp only [← c4557, ← c4558, ← c4559] at e_65_19
  have e_66_10 := arithEq_of_rows h (row := 66) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_10
  simp only [← c4551, k3, ← c4552, k3, ← c4553] at e_66_10
  have e_66_11 := arithEq_of_rows h (row := 66) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_11
  simp only [← c4560, ← c4561, k3, ← c4562] at e_66_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_65_18
    linear_combination c4563.trans k2 - hc
  · have hc := e_66_11
    simp only [e_66_10, e_65_19] at hc
    linear_combination c4564.trans k2 - hc

theorem publicBatchWrapper89_f569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 67 3) = band (a (.virt 18351)) (a (.virt 18353)) := by
  have c4565 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4566 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4567 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_0 := arithEq_of_rows h (row := 67) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_0
  simp only [← c4565, ← c4566, ← c4567] at e_67_0
  have hr := e_67_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 67 7) = band (a (.virt 18355)) (a (.virt 18357)) := by
  have c4568 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4569 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4570 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_1 := arithEq_of_rows h (row := 67) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_1
  simp only [← c4568, ← c4569, ← c4570] at e_67_1
  have hr := e_67_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 67 11) = band (a (.wire 67 3)) (a (.wire 67 7)) := by
  have c4571 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4572 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4573 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_2 := arithEq_of_rows h (row := 67) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_2
  simp only [← c4571, ← c4572, ← c4573] at e_67_2
  have hr := e_67_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13805)) (a (.virt 200)) (a (.virt 18359)) (a (.virt 18360)) := by
  have c4574 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4575 := (publicBatchWrapper89_copies142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4576 := (publicBatchWrapper89_copies143 a h).1
  have c4577 := (publicBatchWrapper89_copies143 a h).2.1
  have c4578 := (publicBatchWrapper89_copies143 a h).2.2.1
  have c4579 := (publicBatchWrapper89_copies143 a h).2.2.2.1
  have c4580 := (publicBatchWrapper89_copies143 a h).2.2.2.2.1
  have c4581 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.1
  have c4582 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.1
  have c4583 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.1
  have c4584 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.1
  have c4585 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.1
  have c4586 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4587 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_66_12 := arithEq_of_rows h (row := 66) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_12
  simp only [← c4574, k3, ← c4575, k3, ← c4576] at e_66_12
  have e_66_13 := arithEq_of_rows h (row := 66) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_13
  simp only [← c4583, ← c4584, k3, ← c4585] at e_66_13
  have e_67_3 := arithEq_of_rows h (row := 67) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_3
  simp only [← c4577, ← c4578, ← c4579] at e_67_3
  have e_67_4 := arithEq_of_rows h (row := 67) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_4
  simp only [← c4580, ← c4581, ← c4582] at e_67_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_3
    linear_combination c4586.trans k2 - hc
  · have hc := e_66_13
    simp only [e_66_12, e_67_4] at hc
    linear_combination c4587.trans k2 - hc

theorem publicBatchWrapper89_f573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13806)) (a (.virt 200)) (a (.virt 18361)) (a (.virt 18362)) := by
  have c4588 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4589 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4590 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4591 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4592 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4593 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4594 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4595 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4596 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4597 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4598 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4599 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4600 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4601 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_66_14 := arithEq_of_rows h (row := 66) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_14
  simp only [← c4588, k3, ← c4589, k3, ← c4590] at e_66_14
  have e_66_15 := arithEq_of_rows h (row := 66) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_15
  simp only [← c4597, ← c4598, k3, ← c4599] at e_66_15
  have e_67_5 := arithEq_of_rows h (row := 67) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_5
  simp only [← c4591, ← c4592, ← c4593] at e_67_5
  have e_67_6 := arithEq_of_rows h (row := 67) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_6
  simp only [← c4594, ← c4595, ← c4596] at e_67_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_5
    linear_combination c4600.trans k2 - hc
  · have hc := e_66_15
    simp only [e_66_14, e_67_6] at hc
    linear_combination c4601.trans k2 - hc

theorem publicBatchWrapper89_f574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13807)) (a (.virt 200)) (a (.virt 18363)) (a (.virt 18364)) := by
  have c4602 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4603 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4604 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4605 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4606 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4607 := (publicBatchWrapper89_copies143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4608 := (publicBatchWrapper89_copies144 a h).1
  have c4609 := (publicBatchWrapper89_copies144 a h).2.1
  have c4610 := (publicBatchWrapper89_copies144 a h).2.2.1
  have c4611 := (publicBatchWrapper89_copies144 a h).2.2.2.1
  have c4612 := (publicBatchWrapper89_copies144 a h).2.2.2.2.1
  have c4613 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.1
  have c4614 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.1
  have c4615 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_66_16 := arithEq_of_rows h (row := 66) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_16
  simp only [← c4602, k3, ← c4603, k3, ← c4604] at e_66_16
  have e_66_17 := arithEq_of_rows h (row := 66) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_17
  simp only [← c4611, ← c4612, k3, ← c4613] at e_66_17
  have e_67_7 := arithEq_of_rows h (row := 67) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_7
  simp only [← c4605, ← c4606, ← c4607] at e_67_7
  have e_67_8 := arithEq_of_rows h (row := 67) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_8
  simp only [← c4608, ← c4609, ← c4610] at e_67_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_7
    linear_combination c4614.trans k2 - hc
  · have hc := e_66_17
    simp only [e_66_16, e_67_8] at hc
    linear_combination c4615.trans k2 - hc

theorem publicBatchWrapper89_f575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 13808)) (a (.virt 200)) (a (.virt 18365)) (a (.virt 18366)) := by
  have c4616 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.1
  have c4617 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.1
  have c4618 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4619 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4620 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4621 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4622 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4623 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4624 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4625 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4626 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4627 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4628 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4629 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_66_18 := arithEq_of_rows h (row := 66) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_18
  simp only [← c4616, k3, ← c4617, k3, ← c4618] at e_66_18
  have e_66_19 := arithEq_of_rows h (row := 66) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_66_19
  simp only [← c4625, ← c4626, k3, ← c4627] at e_66_19
  have e_67_9 := arithEq_of_rows h (row := 67) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_9
  simp only [← c4619, ← c4620, ← c4621] at e_67_9
  have e_67_10 := arithEq_of_rows h (row := 67) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_10
  simp only [← c4622, ← c4623, ← c4624] at e_67_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_9
    linear_combination c4628.trans k2 - hc
  · have hc := e_66_19
    simp only [e_66_18, e_67_10] at hc
    linear_combination c4629.trans k2 - hc

theorem publicBatchWrapper89_f576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 67 47) = band (a (.virt 18359)) (a (.virt 18361)) := by
  have c4630 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4631 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4632 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_11 := arithEq_of_rows h (row := 67) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_11
  simp only [← c4630, ← c4631, ← c4632] at e_67_11
  have hr := e_67_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 67 51) = band (a (.virt 18363)) (a (.virt 18365)) := by
  have c4633 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4634 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4635 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_12 := arithEq_of_rows h (row := 67) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_12
  simp only [← c4633, ← c4634, ← c4635] at e_67_12
  have hr := e_67_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 67 55) = band (a (.wire 67 47)) (a (.wire 67 51)) := by
  have c4636 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4637 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4638 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_67_13 := arithEq_of_rows h (row := 67) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_13
  simp only [← c4636, ← c4637, ← c4638] at e_67_13
  have hr := e_67_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14005)) (a (.virt 200)) (a (.virt 18367)) (a (.virt 18368)) := by
  have c4639 := (publicBatchWrapper89_copies144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4640 := (publicBatchWrapper89_copies145 a h).1
  have c4641 := (publicBatchWrapper89_copies145 a h).2.1
  have c4642 := (publicBatchWrapper89_copies145 a h).2.2.1
  have c4643 := (publicBatchWrapper89_copies145 a h).2.2.2.1
  have c4644 := (publicBatchWrapper89_copies145 a h).2.2.2.2.1
  have c4645 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.1
  have c4646 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.1
  have c4647 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.1
  have c4648 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.1
  have c4649 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.1
  have c4650 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4651 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4652 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_67_14 := arithEq_of_rows h (row := 67) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_14
  simp only [← c4642, ← c4643, ← c4644] at e_67_14
  have e_67_15 := arithEq_of_rows h (row := 67) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_15
  simp only [← c4645, ← c4646, ← c4647] at e_67_15
  have e_68_0 := arithEq_of_rows h (row := 68) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_0
  simp only [← c4639, k3, ← c4640, k3, ← c4641] at e_68_0
  have e_68_1 := arithEq_of_rows h (row := 68) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_1
  simp only [← c4648, ← c4649, k3, ← c4650] at e_68_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_14
    linear_combination c4651.trans k2 - hc
  · have hc := e_68_1
    simp only [e_68_0, e_67_15] at hc
    linear_combination c4652.trans k2 - hc

theorem publicBatchWrapper89_f580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14006)) (a (.virt 200)) (a (.virt 18369)) (a (.virt 18370)) := by
  have c4653 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4654 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4655 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4656 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4657 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4658 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4659 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4660 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4661 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4662 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4663 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4664 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4665 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4666 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_67_16 := arithEq_of_rows h (row := 67) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_16
  simp only [← c4656, ← c4657, ← c4658] at e_67_16
  have e_67_17 := arithEq_of_rows h (row := 67) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_17
  simp only [← c4659, ← c4660, ← c4661] at e_67_17
  have e_68_2 := arithEq_of_rows h (row := 68) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_2
  simp only [← c4653, k3, ← c4654, k3, ← c4655] at e_68_2
  have e_68_3 := arithEq_of_rows h (row := 68) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_3
  simp only [← c4662, ← c4663, k3, ← c4664] at e_68_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_16
    linear_combination c4665.trans k2 - hc
  · have hc := e_68_3
    simp only [e_68_2, e_67_17] at hc
    linear_combination c4666.trans k2 - hc

theorem publicBatchWrapper89_f581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14007)) (a (.virt 200)) (a (.virt 18371)) (a (.virt 18372)) := by
  have c4667 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4668 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4669 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4670 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4671 := (publicBatchWrapper89_copies145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4672 := (publicBatchWrapper89_copies146 a h).1
  have c4673 := (publicBatchWrapper89_copies146 a h).2.1
  have c4674 := (publicBatchWrapper89_copies146 a h).2.2.1
  have c4675 := (publicBatchWrapper89_copies146 a h).2.2.2.1
  have c4676 := (publicBatchWrapper89_copies146 a h).2.2.2.2.1
  have c4677 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.1
  have c4678 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.1
  have c4679 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.1
  have c4680 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_67_18 := arithEq_of_rows h (row := 67) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_18
  simp only [← c4670, ← c4671, ← c4672] at e_67_18
  have e_67_19 := arithEq_of_rows h (row := 67) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_67_19
  simp only [← c4673, ← c4674, ← c4675] at e_67_19
  have e_68_4 := arithEq_of_rows h (row := 68) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_4
  simp only [← c4667, k3, ← c4668, k3, ← c4669] at e_68_4
  have e_68_5 := arithEq_of_rows h (row := 68) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_5
  simp only [← c4676, ← c4677, k3, ← c4678] at e_68_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_67_18
    linear_combination c4679.trans k2 - hc
  · have hc := e_68_5
    simp only [e_68_4, e_67_19] at hc
    linear_combination c4680.trans k2 - hc

theorem publicBatchWrapper89_f582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14008)) (a (.virt 200)) (a (.virt 18373)) (a (.virt 18374)) := by
  have c4681 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.1
  have c4682 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4683 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4684 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4685 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4686 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4687 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4688 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4689 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4690 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4691 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4692 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4693 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4694 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_6 := arithEq_of_rows h (row := 68) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_6
  simp only [← c4681, k3, ← c4682, k3, ← c4683] at e_68_6
  have e_68_7 := arithEq_of_rows h (row := 68) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_7
  simp only [← c4690, ← c4691, k3, ← c4692] at e_68_7
  have e_69_0 := arithEq_of_rows h (row := 69) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_0
  simp only [← c4684, ← c4685, ← c4686] at e_69_0
  have e_69_1 := arithEq_of_rows h (row := 69) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_1
  simp only [← c4687, ← c4688, ← c4689] at e_69_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_0
    linear_combination c4693.trans k2 - hc
  · have hc := e_68_7
    simp only [e_68_6, e_69_1] at hc
    linear_combination c4694.trans k2 - hc

theorem publicBatchWrapper89_f583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 69 11) = band (a (.virt 18367)) (a (.virt 18369)) := by
  have c4695 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4696 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4697 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_69_2 := arithEq_of_rows h (row := 69) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_2
  simp only [← c4695, ← c4696, ← c4697] at e_69_2
  have hr := e_69_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 69 15) = band (a (.virt 18371)) (a (.virt 18373)) := by
  have c4698 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4699 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4700 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_69_3 := arithEq_of_rows h (row := 69) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_3
  simp only [← c4698, ← c4699, ← c4700] at e_69_3
  have hr := e_69_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 69 19) = band (a (.wire 69 11)) (a (.wire 69 15)) := by
  have c4701 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4702 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4703 := (publicBatchWrapper89_copies146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_69_4 := arithEq_of_rows h (row := 69) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_4
  simp only [← c4701, ← c4702, ← c4703] at e_69_4
  have hr := e_69_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14205)) (a (.virt 200)) (a (.virt 18375)) (a (.virt 18376)) := by
  have c4704 := (publicBatchWrapper89_copies147 a h).1
  have c4705 := (publicBatchWrapper89_copies147 a h).2.1
  have c4706 := (publicBatchWrapper89_copies147 a h).2.2.1
  have c4707 := (publicBatchWrapper89_copies147 a h).2.2.2.1
  have c4708 := (publicBatchWrapper89_copies147 a h).2.2.2.2.1
  have c4709 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.1
  have c4710 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.1
  have c4711 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.1
  have c4712 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.1
  have c4713 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.1
  have c4714 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4715 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4716 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4717 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_8 := arithEq_of_rows h (row := 68) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_8
  simp only [← c4704, k3, ← c4705, k3, ← c4706] at e_68_8
  have e_68_9 := arithEq_of_rows h (row := 68) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_9
  simp only [← c4713, ← c4714, k3, ← c4715] at e_68_9
  have e_69_5 := arithEq_of_rows h (row := 69) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_5
  simp only [← c4707, ← c4708, ← c4709] at e_69_5
  have e_69_6 := arithEq_of_rows h (row := 69) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_6
  simp only [← c4710, ← c4711, ← c4712] at e_69_6
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_5
    linear_combination c4716.trans k2 - hc
  · have hc := e_68_9
    simp only [e_68_8, e_69_6] at hc
    linear_combination c4717.trans k2 - hc

theorem publicBatchWrapper89_f587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14206)) (a (.virt 200)) (a (.virt 18377)) (a (.virt 18378)) := by
  have c4718 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4719 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4720 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4721 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4722 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4723 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4724 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4725 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4726 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4727 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4728 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4729 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4730 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4731 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_10 := arithEq_of_rows h (row := 68) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_10
  simp only [← c4718, k3, ← c4719, k3, ← c4720] at e_68_10
  have e_68_11 := arithEq_of_rows h (row := 68) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_11
  simp only [← c4727, ← c4728, k3, ← c4729] at e_68_11
  have e_69_7 := arithEq_of_rows h (row := 69) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_7
  simp only [← c4721, ← c4722, ← c4723] at e_69_7
  have e_69_8 := arithEq_of_rows h (row := 69) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_8
  simp only [← c4724, ← c4725, ← c4726] at e_69_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_7
    linear_combination c4730.trans k2 - hc
  · have hc := e_68_11
    simp only [e_68_10, e_69_8] at hc
    linear_combination c4731.trans k2 - hc

theorem publicBatchWrapper89_f588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14207)) (a (.virt 200)) (a (.virt 18379)) (a (.virt 18380)) := by
  have c4732 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4733 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4734 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4735 := (publicBatchWrapper89_copies147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4736 := (publicBatchWrapper89_copies148 a h).1
  have c4737 := (publicBatchWrapper89_copies148 a h).2.1
  have c4738 := (publicBatchWrapper89_copies148 a h).2.2.1
  have c4739 := (publicBatchWrapper89_copies148 a h).2.2.2.1
  have c4740 := (publicBatchWrapper89_copies148 a h).2.2.2.2.1
  have c4741 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.1
  have c4742 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.1
  have c4743 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.1
  have c4744 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.1
  have c4745 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_12 := arithEq_of_rows h (row := 68) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_12
  simp only [← c4732, k3, ← c4733, k3, ← c4734] at e_68_12
  have e_68_13 := arithEq_of_rows h (row := 68) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_13
  simp only [← c4741, ← c4742, k3, ← c4743] at e_68_13
  have e_69_9 := arithEq_of_rows h (row := 69) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_9
  simp only [← c4735, ← c4736, ← c4737] at e_69_9
  have e_69_10 := arithEq_of_rows h (row := 69) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_10
  simp only [← c4738, ← c4739, ← c4740] at e_69_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_9
    linear_combination c4744.trans k2 - hc
  · have hc := e_68_13
    simp only [e_68_12, e_69_10] at hc
    linear_combination c4745.trans k2 - hc

theorem publicBatchWrapper89_f589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14208)) (a (.virt 200)) (a (.virt 18381)) (a (.virt 18382)) := by
  have c4746 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4747 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4748 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4749 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4750 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4751 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4752 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4753 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4754 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4755 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4756 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4757 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4758 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4759 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_14 := arithEq_of_rows h (row := 68) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_14
  simp only [← c4746, k3, ← c4747, k3, ← c4748] at e_68_14
  have e_68_15 := arithEq_of_rows h (row := 68) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_15
  simp only [← c4755, ← c4756, k3, ← c4757] at e_68_15
  have e_69_11 := arithEq_of_rows h (row := 69) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_11
  simp only [← c4749, ← c4750, ← c4751] at e_69_11
  have e_69_12 := arithEq_of_rows h (row := 69) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_12
  simp only [← c4752, ← c4753, ← c4754] at e_69_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_11
    linear_combination c4758.trans k2 - hc
  · have hc := e_68_15
    simp only [e_68_14, e_69_12] at hc
    linear_combination c4759.trans k2 - hc

theorem publicBatchWrapper89_f590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 69 55) = band (a (.virt 18375)) (a (.virt 18377)) := by
  have c4760 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4761 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4762 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_69_13 := arithEq_of_rows h (row := 69) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_13
  simp only [← c4760, ← c4761, ← c4762] at e_69_13
  have hr := e_69_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 69 59) = band (a (.virt 18379)) (a (.virt 18381)) := by
  have c4763 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4764 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4765 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_69_14 := arithEq_of_rows h (row := 69) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_14
  simp only [← c4763, ← c4764, ← c4765] at e_69_14
  have hr := e_69_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 69 63) = band (a (.wire 69 55)) (a (.wire 69 59)) := by
  have c4766 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4767 := (publicBatchWrapper89_copies148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4768 := (publicBatchWrapper89_copies149 a h).1
  have e_69_15 := arithEq_of_rows h (row := 69) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_15
  simp only [← c4766, ← c4767, ← c4768] at e_69_15
  have hr := e_69_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14405)) (a (.virt 200)) (a (.virt 18383)) (a (.virt 18384)) := by
  have c4769 := (publicBatchWrapper89_copies149 a h).2.1
  have c4770 := (publicBatchWrapper89_copies149 a h).2.2.1
  have c4771 := (publicBatchWrapper89_copies149 a h).2.2.2.1
  have c4772 := (publicBatchWrapper89_copies149 a h).2.2.2.2.1
  have c4773 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.1
  have c4774 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.1
  have c4775 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.1
  have c4776 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.1
  have c4777 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.1
  have c4778 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4779 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4780 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4781 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4782 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_16 := arithEq_of_rows h (row := 68) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_16
  simp only [← c4769, k3, ← c4770, k3, ← c4771] at e_68_16
  have e_68_17 := arithEq_of_rows h (row := 68) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_17
  simp only [← c4778, ← c4779, k3, ← c4780] at e_68_17
  have e_69_16 := arithEq_of_rows h (row := 69) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_16
  simp only [← c4772, ← c4773, ← c4774] at e_69_16
  have e_69_17 := arithEq_of_rows h (row := 69) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_17
  simp only [← c4775, ← c4776, ← c4777] at e_69_17
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_16
    linear_combination c4781.trans k2 - hc
  · have hc := e_68_17
    simp only [e_68_16, e_69_17] at hc
    linear_combination c4782.trans k2 - hc

theorem publicBatchWrapper89_f594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14406)) (a (.virt 200)) (a (.virt 18385)) (a (.virt 18386)) := by
  have c4783 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4784 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4785 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4786 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4787 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4788 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4789 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4790 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4791 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4792 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4793 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4794 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4795 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4796 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_68_18 := arithEq_of_rows h (row := 68) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_18
  simp only [← c4783, k3, ← c4784, k3, ← c4785] at e_68_18
  have e_68_19 := arithEq_of_rows h (row := 68) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_68_19
  simp only [← c4792, ← c4793, k3, ← c4794] at e_68_19
  have e_69_18 := arithEq_of_rows h (row := 69) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_18
  simp only [← c4786, ← c4787, ← c4788] at e_69_18
  have e_69_19 := arithEq_of_rows h (row := 69) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_69_19
  simp only [← c4789, ← c4790, ← c4791] at e_69_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_69_18
    linear_combination c4795.trans k2 - hc
  · have hc := e_68_19
    simp only [e_68_18, e_69_19] at hc
    linear_combination c4796.trans k2 - hc

theorem publicBatchWrapper89_f595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14407)) (a (.virt 200)) (a (.virt 18387)) (a (.virt 18388)) := by
  have c4797 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4798 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4799 := (publicBatchWrapper89_copies149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4800 := (publicBatchWrapper89_copies150 a h).1
  have c4801 := (publicBatchWrapper89_copies150 a h).2.1
  have c4802 := (publicBatchWrapper89_copies150 a h).2.2.1
  have c4803 := (publicBatchWrapper89_copies150 a h).2.2.2.1
  have c4804 := (publicBatchWrapper89_copies150 a h).2.2.2.2.1
  have c4805 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.1
  have c4806 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.1
  have c4807 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.1
  have c4808 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.1
  have c4809 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.1
  have c4810 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_0 := arithEq_of_rows h (row := 70) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_0
  simp only [← c4797, k3, ← c4798, k3, ← c4799] at e_70_0
  have e_70_1 := arithEq_of_rows h (row := 70) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_1
  simp only [← c4806, ← c4807, k3, ← c4808] at e_70_1
  have e_71_0 := arithEq_of_rows h (row := 71) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_0
  simp only [← c4800, ← c4801, ← c4802] at e_71_0
  have e_71_1 := arithEq_of_rows h (row := 71) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_1
  simp only [← c4803, ← c4804, ← c4805] at e_71_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_0
    linear_combination c4809.trans k2 - hc
  · have hc := e_70_1
    simp only [e_70_0, e_71_1] at hc
    linear_combination c4810.trans k2 - hc

theorem publicBatchWrapper89_f596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14408)) (a (.virt 200)) (a (.virt 18389)) (a (.virt 18390)) := by
  have c4811 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4812 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4813 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4814 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4815 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4816 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4817 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4818 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4819 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4820 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4821 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4822 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4823 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4824 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_2 := arithEq_of_rows h (row := 70) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_2
  simp only [← c4811, k3, ← c4812, k3, ← c4813] at e_70_2
  have e_70_3 := arithEq_of_rows h (row := 70) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_3
  simp only [← c4820, ← c4821, k3, ← c4822] at e_70_3
  have e_71_2 := arithEq_of_rows h (row := 71) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_2
  simp only [← c4814, ← c4815, ← c4816] at e_71_2
  have e_71_3 := arithEq_of_rows h (row := 71) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_3
  simp only [← c4817, ← c4818, ← c4819] at e_71_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_2
    linear_combination c4823.trans k2 - hc
  · have hc := e_70_3
    simp only [e_70_2, e_71_3] at hc
    linear_combination c4824.trans k2 - hc

theorem publicBatchWrapper89_f597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 71 19) = band (a (.virt 18383)) (a (.virt 18385)) := by
  have c4825 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4826 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4827 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_4 := arithEq_of_rows h (row := 71) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_4
  simp only [← c4825, ← c4826, ← c4827] at e_71_4
  have hr := e_71_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 71 23) = band (a (.virt 18387)) (a (.virt 18389)) := by
  have c4828 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4829 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4830 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_5 := arithEq_of_rows h (row := 71) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_5
  simp only [← c4828, ← c4829, ← c4830] at e_71_5
  have hr := e_71_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 71 27) = band (a (.wire 71 19)) (a (.wire 71 23)) := by
  have c4831 := (publicBatchWrapper89_copies150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4832 := (publicBatchWrapper89_copies151 a h).1
  have c4833 := (publicBatchWrapper89_copies151 a h).2.1
  have e_71_6 := arithEq_of_rows h (row := 71) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_6
  simp only [← c4831, ← c4832, ← c4833] at e_71_6
  have hr := e_71_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14605)) (a (.virt 200)) (a (.virt 18391)) (a (.virt 18392)) := by
  have c4834 := (publicBatchWrapper89_copies151 a h).2.2.1
  have c4835 := (publicBatchWrapper89_copies151 a h).2.2.2.1
  have c4836 := (publicBatchWrapper89_copies151 a h).2.2.2.2.1
  have c4837 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.1
  have c4838 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.1
  have c4839 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.1
  have c4840 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.1
  have c4841 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.1
  have c4842 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4843 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4844 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4845 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4846 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4847 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_4 := arithEq_of_rows h (row := 70) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_4
  simp only [← c4834, k3, ← c4835, k3, ← c4836] at e_70_4
  have e_70_5 := arithEq_of_rows h (row := 70) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_5
  simp only [← c4843, ← c4844, k3, ← c4845] at e_70_5
  have e_71_7 := arithEq_of_rows h (row := 71) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_7
  simp only [← c4837, ← c4838, ← c4839] at e_71_7
  have e_71_8 := arithEq_of_rows h (row := 71) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_8
  simp only [← c4840, ← c4841, ← c4842] at e_71_8
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_7
    linear_combination c4846.trans k2 - hc
  · have hc := e_70_5
    simp only [e_70_4, e_71_8] at hc
    linear_combination c4847.trans k2 - hc

theorem publicBatchWrapper89_f601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14606)) (a (.virt 200)) (a (.virt 18393)) (a (.virt 18394)) := by
  have c4848 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4849 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4850 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4851 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4852 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4853 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4854 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4855 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4856 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4857 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4858 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4859 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4860 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4861 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_6 := arithEq_of_rows h (row := 70) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_6
  simp only [← c4848, k3, ← c4849, k3, ← c4850] at e_70_6
  have e_70_7 := arithEq_of_rows h (row := 70) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_7
  simp only [← c4857, ← c4858, k3, ← c4859] at e_70_7
  have e_71_9 := arithEq_of_rows h (row := 71) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_9
  simp only [← c4851, ← c4852, ← c4853] at e_71_9
  have e_71_10 := arithEq_of_rows h (row := 71) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_10
  simp only [← c4854, ← c4855, ← c4856] at e_71_10
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_9
    linear_combination c4860.trans k2 - hc
  · have hc := e_70_7
    simp only [e_70_6, e_71_10] at hc
    linear_combination c4861.trans k2 - hc

theorem publicBatchWrapper89_f602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14607)) (a (.virt 200)) (a (.virt 18395)) (a (.virt 18396)) := by
  have c4862 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4863 := (publicBatchWrapper89_copies151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4864 := (publicBatchWrapper89_copies152 a h).1
  have c4865 := (publicBatchWrapper89_copies152 a h).2.1
  have c4866 := (publicBatchWrapper89_copies152 a h).2.2.1
  have c4867 := (publicBatchWrapper89_copies152 a h).2.2.2.1
  have c4868 := (publicBatchWrapper89_copies152 a h).2.2.2.2.1
  have c4869 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.1
  have c4870 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.1
  have c4871 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.1
  have c4872 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.1
  have c4873 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.1
  have c4874 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4875 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_8 := arithEq_of_rows h (row := 70) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_8
  simp only [← c4862, k3, ← c4863, k3, ← c4864] at e_70_8
  have e_70_9 := arithEq_of_rows h (row := 70) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_9
  simp only [← c4871, ← c4872, k3, ← c4873] at e_70_9
  have e_71_11 := arithEq_of_rows h (row := 71) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_11
  simp only [← c4865, ← c4866, ← c4867] at e_71_11
  have e_71_12 := arithEq_of_rows h (row := 71) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_12
  simp only [← c4868, ← c4869, ← c4870] at e_71_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_11
    linear_combination c4874.trans k2 - hc
  · have hc := e_70_9
    simp only [e_70_8, e_71_12] at hc
    linear_combination c4875.trans k2 - hc

theorem publicBatchWrapper89_f603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14608)) (a (.virt 200)) (a (.virt 18397)) (a (.virt 18398)) := by
  have c4876 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4877 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4878 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4879 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4880 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4881 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4882 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4883 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4884 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4885 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4886 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4887 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4888 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4889 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_10 := arithEq_of_rows h (row := 70) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_10
  simp only [← c4876, k3, ← c4877, k3, ← c4878] at e_70_10
  have e_70_11 := arithEq_of_rows h (row := 70) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_11
  simp only [← c4885, ← c4886, k3, ← c4887] at e_70_11
  have e_71_13 := arithEq_of_rows h (row := 71) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_13
  simp only [← c4879, ← c4880, ← c4881] at e_71_13
  have e_71_14 := arithEq_of_rows h (row := 71) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_14
  simp only [← c4882, ← c4883, ← c4884] at e_71_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_13
    linear_combination c4888.trans k2 - hc
  · have hc := e_70_11
    simp only [e_70_10, e_71_14] at hc
    linear_combination c4889.trans k2 - hc

theorem publicBatchWrapper89_f604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 71 63) = band (a (.virt 18391)) (a (.virt 18393)) := by
  have c4890 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4891 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4892 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_71_15 := arithEq_of_rows h (row := 71) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_15
  simp only [← c4890, ← c4891, ← c4892] at e_71_15
  have hr := e_71_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 71 67) = band (a (.virt 18395)) (a (.virt 18397)) := by
  have c4893 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4894 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4895 := (publicBatchWrapper89_copies152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_71_16 := arithEq_of_rows h (row := 71) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_16
  simp only [← c4893, ← c4894, ← c4895] at e_71_16
  have hr := e_71_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 71 71) = band (a (.wire 71 63)) (a (.wire 71 67)) := by
  have c4896 := (publicBatchWrapper89_copies153 a h).1
  have c4897 := (publicBatchWrapper89_copies153 a h).2.1
  have c4898 := (publicBatchWrapper89_copies153 a h).2.2.1
  have e_71_17 := arithEq_of_rows h (row := 71) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_17
  simp only [← c4896, ← c4897, ← c4898] at e_71_17
  have hr := e_71_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14805)) (a (.virt 200)) (a (.virt 18399)) (a (.virt 18400)) := by
  have c4899 := (publicBatchWrapper89_copies153 a h).2.2.2.1
  have c4900 := (publicBatchWrapper89_copies153 a h).2.2.2.2.1
  have c4901 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.1
  have c4902 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.1
  have c4903 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.1
  have c4904 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.1
  have c4905 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.1
  have c4906 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4907 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4908 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4909 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4910 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4911 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4912 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_12 := arithEq_of_rows h (row := 70) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_12
  simp only [← c4899, k3, ← c4900, k3, ← c4901] at e_70_12
  have e_70_13 := arithEq_of_rows h (row := 70) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_13
  simp only [← c4908, ← c4909, k3, ← c4910] at e_70_13
  have e_71_18 := arithEq_of_rows h (row := 71) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_18
  simp only [← c4902, ← c4903, ← c4904] at e_71_18
  have e_71_19 := arithEq_of_rows h (row := 71) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_71_19
  simp only [← c4905, ← c4906, ← c4907] at e_71_19
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_71_18
    linear_combination c4911.trans k2 - hc
  · have hc := e_70_13
    simp only [e_70_12, e_71_19] at hc
    linear_combination c4912.trans k2 - hc

theorem publicBatchWrapper89_f608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14806)) (a (.virt 200)) (a (.virt 18401)) (a (.virt 18402)) := by
  have c4913 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4914 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4915 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4916 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4917 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4918 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4919 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4920 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4921 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4922 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4923 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4924 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4925 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4926 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_14 := arithEq_of_rows h (row := 70) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_14
  simp only [← c4913, k3, ← c4914, k3, ← c4915] at e_70_14
  have e_70_15 := arithEq_of_rows h (row := 70) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_15
  simp only [← c4922, ← c4923, k3, ← c4924] at e_70_15
  have e_72_0 := arithEq_of_rows h (row := 72) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_0
  simp only [← c4916, ← c4917, ← c4918] at e_72_0
  have e_72_1 := arithEq_of_rows h (row := 72) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_1
  simp only [← c4919, ← c4920, ← c4921] at e_72_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_0
    linear_combination c4925.trans k2 - hc
  · have hc := e_70_15
    simp only [e_70_14, e_72_1] at hc
    linear_combination c4926.trans k2 - hc

theorem publicBatchWrapper89_f609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14807)) (a (.virt 200)) (a (.virt 18403)) (a (.virt 18404)) := by
  have c4927 := (publicBatchWrapper89_copies153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4928 := (publicBatchWrapper89_copies154 a h).1
  have c4929 := (publicBatchWrapper89_copies154 a h).2.1
  have c4930 := (publicBatchWrapper89_copies154 a h).2.2.1
  have c4931 := (publicBatchWrapper89_copies154 a h).2.2.2.1
  have c4932 := (publicBatchWrapper89_copies154 a h).2.2.2.2.1
  have c4933 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.1
  have c4934 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.1
  have c4935 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.1
  have c4936 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.1
  have c4937 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.1
  have c4938 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4939 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4940 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_16 := arithEq_of_rows h (row := 70) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_16
  simp only [← c4927, k3, ← c4928, k3, ← c4929] at e_70_16
  have e_70_17 := arithEq_of_rows h (row := 70) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_17
  simp only [← c4936, ← c4937, k3, ← c4938] at e_70_17
  have e_72_2 := arithEq_of_rows h (row := 72) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_2
  simp only [← c4930, ← c4931, ← c4932] at e_72_2
  have e_72_3 := arithEq_of_rows h (row := 72) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_3
  simp only [← c4933, ← c4934, ← c4935] at e_72_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_2
    linear_combination c4939.trans k2 - hc
  · have hc := e_70_17
    simp only [e_70_16, e_72_3] at hc
    linear_combination c4940.trans k2 - hc

theorem publicBatchWrapper89_f610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 14808)) (a (.virt 200)) (a (.virt 18405)) (a (.virt 18406)) := by
  have c4941 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4942 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4943 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4944 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4945 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4946 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4947 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4948 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4949 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4950 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4951 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4952 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4953 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4954 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_70_18 := arithEq_of_rows h (row := 70) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_18
  simp only [← c4941, k3, ← c4942, k3, ← c4943] at e_70_18
  have e_70_19 := arithEq_of_rows h (row := 70) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_70_19
  simp only [← c4950, ← c4951, k3, ← c4952] at e_70_19
  have e_72_4 := arithEq_of_rows h (row := 72) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_4
  simp only [← c4944, ← c4945, ← c4946] at e_72_4
  have e_72_5 := arithEq_of_rows h (row := 72) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_5
  simp only [← c4947, ← c4948, ← c4949] at e_72_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_4
    linear_combination c4953.trans k2 - hc
  · have hc := e_70_19
    simp only [e_70_18, e_72_5] at hc
    linear_combination c4954.trans k2 - hc

theorem publicBatchWrapper89_f611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 72 27) = band (a (.virt 18399)) (a (.virt 18401)) := by
  have c4955 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4956 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4957 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_72_6 := arithEq_of_rows h (row := 72) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_6
  simp only [← c4955, ← c4956, ← c4957] at e_72_6
  have hr := e_72_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 72 31) = band (a (.virt 18403)) (a (.virt 18405)) := by
  have c4958 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4959 := (publicBatchWrapper89_copies154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c4960 := (publicBatchWrapper89_copies155 a h).1
  have e_72_7 := arithEq_of_rows h (row := 72) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_7
  simp only [← c4958, ← c4959, ← c4960] at e_72_7
  have hr := e_72_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 72 35) = band (a (.wire 72 27)) (a (.wire 72 31)) := by
  have c4961 := (publicBatchWrapper89_copies155 a h).2.1
  have c4962 := (publicBatchWrapper89_copies155 a h).2.2.1
  have c4963 := (publicBatchWrapper89_copies155 a h).2.2.2.1
  have e_72_8 := arithEq_of_rows h (row := 72) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_8
  simp only [← c4961, ← c4962, ← c4963] at e_72_8
  have hr := e_72_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15005)) (a (.virt 200)) (a (.virt 18407)) (a (.virt 18408)) := by
  have c4964 := (publicBatchWrapper89_copies155 a h).2.2.2.2.1
  have c4965 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.1
  have c4966 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.1
  have c4967 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.1
  have c4968 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.1
  have c4969 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.1
  have c4970 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c4971 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c4972 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4973 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4974 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4975 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4976 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4977 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_72_9 := arithEq_of_rows h (row := 72) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_9
  simp only [← c4967, ← c4968, ← c4969] at e_72_9
  have e_72_10 := arithEq_of_rows h (row := 72) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_10
  simp only [← c4970, ← c4971, ← c4972] at e_72_10
  have e_73_0 := arithEq_of_rows h (row := 73) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_0
  simp only [← c4964, k3, ← c4965, k3, ← c4966] at e_73_0
  have e_73_1 := arithEq_of_rows h (row := 73) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_1
  simp only [← c4973, ← c4974, k3, ← c4975] at e_73_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_9
    linear_combination c4976.trans k2 - hc
  · have hc := e_73_1
    simp only [e_73_0, e_72_10] at hc
    linear_combination c4977.trans k2 - hc

theorem publicBatchWrapper89_f615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15006)) (a (.virt 200)) (a (.virt 18409)) (a (.virt 18410)) := by
  have c4978 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4979 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4980 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4981 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4982 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4983 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4984 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4985 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4986 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4987 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4988 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4989 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4990 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c4991 := (publicBatchWrapper89_copies155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_72_11 := arithEq_of_rows h (row := 72) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_11
  simp only [← c4981, ← c4982, ← c4983] at e_72_11
  have e_72_12 := arithEq_of_rows h (row := 72) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_12
  simp only [← c4984, ← c4985, ← c4986] at e_72_12
  have e_73_2 := arithEq_of_rows h (row := 73) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_2
  simp only [← c4978, k3, ← c4979, k3, ← c4980] at e_73_2
  have e_73_3 := arithEq_of_rows h (row := 73) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_3
  simp only [← c4987, ← c4988, k3, ← c4989] at e_73_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_11
    linear_combination c4990.trans k2 - hc
  · have hc := e_73_3
    simp only [e_73_2, e_72_12] at hc
    linear_combination c4991.trans k2 - hc

theorem publicBatchWrapper89_f616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15007)) (a (.virt 200)) (a (.virt 18411)) (a (.virt 18412)) := by
  have c4992 := (publicBatchWrapper89_copies156 a h).1
  have c4993 := (publicBatchWrapper89_copies156 a h).2.1
  have c4994 := (publicBatchWrapper89_copies156 a h).2.2.1
  have c4995 := (publicBatchWrapper89_copies156 a h).2.2.2.1
  have c4996 := (publicBatchWrapper89_copies156 a h).2.2.2.2.1
  have c4997 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.1
  have c4998 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.1
  have c4999 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.1
  have c5000 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.1
  have c5001 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.1
  have c5002 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5003 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5004 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5005 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_72_13 := arithEq_of_rows h (row := 72) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_13
  simp only [← c4995, ← c4996, ← c4997] at e_72_13
  have e_72_14 := arithEq_of_rows h (row := 72) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_14
  simp only [← c4998, ← c4999, ← c5000] at e_72_14
  have e_73_4 := arithEq_of_rows h (row := 73) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_4
  simp only [← c4992, k3, ← c4993, k3, ← c4994] at e_73_4
  have e_73_5 := arithEq_of_rows h (row := 73) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_5
  simp only [← c5001, ← c5002, k3, ← c5003] at e_73_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_13
    linear_combination c5004.trans k2 - hc
  · have hc := e_73_5
    simp only [e_73_4, e_72_14] at hc
    linear_combination c5005.trans k2 - hc

theorem publicBatchWrapper89_f617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15008)) (a (.virt 200)) (a (.virt 18413)) (a (.virt 18414)) := by
  have c5006 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5007 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5008 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5009 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5010 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5011 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5012 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5013 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5014 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5015 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5016 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5017 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5018 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5019 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_72_15 := arithEq_of_rows h (row := 72) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_15
  simp only [← c5009, ← c5010, ← c5011] at e_72_15
  have e_72_16 := arithEq_of_rows h (row := 72) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_16
  simp only [← c5012, ← c5013, ← c5014] at e_72_16
  have e_73_6 := arithEq_of_rows h (row := 73) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_6
  simp only [← c5006, k3, ← c5007, k3, ← c5008] at e_73_6
  have e_73_7 := arithEq_of_rows h (row := 73) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_7
  simp only [← c5015, ← c5016, k3, ← c5017] at e_73_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_72_15
    linear_combination c5018.trans k2 - hc
  · have hc := e_73_7
    simp only [e_73_6, e_72_16] at hc
    linear_combination c5019.trans k2 - hc

theorem publicBatchWrapper89_f618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 72 71) = band (a (.virt 18407)) (a (.virt 18409)) := by
  have c5020 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5021 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5022 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_72_17 := arithEq_of_rows h (row := 72) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_17
  simp only [← c5020, ← c5021, ← c5022] at e_72_17
  have hr := e_72_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 72 75) = band (a (.virt 18411)) (a (.virt 18413)) := by
  have c5023 := (publicBatchWrapper89_copies156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5024 := (publicBatchWrapper89_copies157 a h).1
  have c5025 := (publicBatchWrapper89_copies157 a h).2.1
  have e_72_18 := arithEq_of_rows h (row := 72) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_18
  simp only [← c5023, ← c5024, ← c5025] at e_72_18
  have hr := e_72_18
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 72 79) = band (a (.wire 72 71)) (a (.wire 72 75)) := by
  have c5026 := (publicBatchWrapper89_copies157 a h).2.2.1
  have c5027 := (publicBatchWrapper89_copies157 a h).2.2.2.1
  have c5028 := (publicBatchWrapper89_copies157 a h).2.2.2.2.1
  have e_72_19 := arithEq_of_rows h (row := 72) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_72_19
  simp only [← c5026, ← c5027, ← c5028] at e_72_19
  have hr := e_72_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15205)) (a (.virt 200)) (a (.virt 18415)) (a (.virt 18416)) := by
  have c5029 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.1
  have c5030 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.1
  have c5031 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.1
  have c5032 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.1
  have c5033 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.1
  have c5034 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5035 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5036 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5037 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5038 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5039 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5040 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5041 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5042 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_73_8 := arithEq_of_rows h (row := 73) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_8
  simp only [← c5029, k3, ← c5030, k3, ← c5031] at e_73_8
  have e_73_9 := arithEq_of_rows h (row := 73) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_9
  simp only [← c5038, ← c5039, k3, ← c5040] at e_73_9
  have e_74_0 := arithEq_of_rows h (row := 74) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_0
  simp only [← c5032, ← c5033, ← c5034] at e_74_0
  have e_74_1 := arithEq_of_rows h (row := 74) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_1
  simp only [← c5035, ← c5036, ← c5037] at e_74_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_0
    linear_combination c5041.trans k2 - hc
  · have hc := e_73_9
    simp only [e_73_8, e_74_1] at hc
    linear_combination c5042.trans k2 - hc

theorem publicBatchWrapper89_f622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15206)) (a (.virt 200)) (a (.virt 18417)) (a (.virt 18418)) := by
  have c5043 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5044 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5045 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5046 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5047 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5048 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5049 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5050 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5051 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5052 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5053 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5054 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5055 := (publicBatchWrapper89_copies157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5056 := (publicBatchWrapper89_copies158 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_73_10 := arithEq_of_rows h (row := 73) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_10
  simp only [← c5043, k3, ← c5044, k3, ← c5045] at e_73_10
  have e_73_11 := arithEq_of_rows h (row := 73) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_11
  simp only [← c5052, ← c5053, k3, ← c5054] at e_73_11
  have e_74_2 := arithEq_of_rows h (row := 74) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_2
  simp only [← c5046, ← c5047, ← c5048] at e_74_2
  have e_74_3 := arithEq_of_rows h (row := 74) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_3
  simp only [← c5049, ← c5050, ← c5051] at e_74_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_2
    linear_combination c5055.trans k2 - hc
  · have hc := e_73_11
    simp only [e_73_10, e_74_3] at hc
    linear_combination c5056.trans k2 - hc

theorem publicBatchWrapper89_f623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15207)) (a (.virt 200)) (a (.virt 18419)) (a (.virt 18420)) := by
  have c5057 := (publicBatchWrapper89_copies158 a h).2.1
  have c5058 := (publicBatchWrapper89_copies158 a h).2.2.1
  have c5059 := (publicBatchWrapper89_copies158 a h).2.2.2.1
  have c5060 := (publicBatchWrapper89_copies158 a h).2.2.2.2.1
  have c5061 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.1
  have c5062 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.1
  have c5063 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.1
  have c5064 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.1
  have c5065 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.1
  have c5066 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5067 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5068 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5069 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5070 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_73_12 := arithEq_of_rows h (row := 73) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_12
  simp only [← c5057, k3, ← c5058, k3, ← c5059] at e_73_12
  have e_73_13 := arithEq_of_rows h (row := 73) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_13
  simp only [← c5066, ← c5067, k3, ← c5068] at e_73_13
  have e_74_4 := arithEq_of_rows h (row := 74) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_4
  simp only [← c5060, ← c5061, ← c5062] at e_74_4
  have e_74_5 := arithEq_of_rows h (row := 74) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_5
  simp only [← c5063, ← c5064, ← c5065] at e_74_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_4
    linear_combination c5069.trans k2 - hc
  · have hc := e_73_13
    simp only [e_73_12, e_74_5] at hc
    linear_combination c5070.trans k2 - hc

theorem publicBatchWrapper89_f624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15208)) (a (.virt 200)) (a (.virt 18421)) (a (.virt 18422)) := by
  have c5071 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5072 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5073 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5074 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5075 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5076 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5077 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5078 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5079 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5080 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5081 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5082 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5083 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5084 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_73_14 := arithEq_of_rows h (row := 73) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_14
  simp only [← c5071, k3, ← c5072, k3, ← c5073] at e_73_14
  have e_73_15 := arithEq_of_rows h (row := 73) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_15
  simp only [← c5080, ← c5081, k3, ← c5082] at e_73_15
  have e_74_6 := arithEq_of_rows h (row := 74) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_6
  simp only [← c5074, ← c5075, ← c5076] at e_74_6
  have e_74_7 := arithEq_of_rows h (row := 74) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_7
  simp only [← c5077, ← c5078, ← c5079] at e_74_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_6
    linear_combination c5083.trans k2 - hc
  · have hc := e_73_15
    simp only [e_73_14, e_74_7] at hc
    linear_combination c5084.trans k2 - hc

theorem publicBatchWrapper89_f625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 74 35) = band (a (.virt 18415)) (a (.virt 18417)) := by
  have c5085 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5086 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5087 := (publicBatchWrapper89_copies158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_74_8 := arithEq_of_rows h (row := 74) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_8
  simp only [← c5085, ← c5086, ← c5087] at e_74_8
  have hr := e_74_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 74 39) = band (a (.virt 18419)) (a (.virt 18421)) := by
  have c5088 := (publicBatchWrapper89_copies159 a h).1
  have c5089 := (publicBatchWrapper89_copies159 a h).2.1
  have c5090 := (publicBatchWrapper89_copies159 a h).2.2.1
  have e_74_9 := arithEq_of_rows h (row := 74) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_9
  simp only [← c5088, ← c5089, ← c5090] at e_74_9
  have hr := e_74_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 74 43) = band (a (.wire 74 35)) (a (.wire 74 39)) := by
  have c5091 := (publicBatchWrapper89_copies159 a h).2.2.2.1
  have c5092 := (publicBatchWrapper89_copies159 a h).2.2.2.2.1
  have c5093 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.1
  have e_74_10 := arithEq_of_rows h (row := 74) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_10
  simp only [← c5091, ← c5092, ← c5093] at e_74_10
  have hr := e_74_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15405)) (a (.virt 200)) (a (.virt 18423)) (a (.virt 18424)) := by
  have c5094 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.1
  have c5095 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.1
  have c5096 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.1
  have c5097 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.1
  have c5098 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5099 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5100 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5101 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5102 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5103 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5104 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5105 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5106 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5107 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_73_16 := arithEq_of_rows h (row := 73) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_16
  simp only [← c5094, k3, ← c5095, k3, ← c5096] at e_73_16
  have e_73_17 := arithEq_of_rows h (row := 73) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_17
  simp only [← c5103, ← c5104, k3, ← c5105] at e_73_17
  have e_74_11 := arithEq_of_rows h (row := 74) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_11
  simp only [← c5097, ← c5098, ← c5099] at e_74_11
  have e_74_12 := arithEq_of_rows h (row := 74) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_12
  simp only [← c5100, ← c5101, ← c5102] at e_74_12
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_11
    linear_combination c5106.trans k2 - hc
  · have hc := e_73_17
    simp only [e_73_16, e_74_12] at hc
    linear_combination c5107.trans k2 - hc

theorem publicBatchWrapper89_f629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15406)) (a (.virt 200)) (a (.virt 18425)) (a (.virt 18426)) := by
  have c5108 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5109 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5110 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5111 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5112 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5113 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5114 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5115 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5116 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5117 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5118 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5119 := (publicBatchWrapper89_copies159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5120 := (publicBatchWrapper89_copies160 a h).1
  have c5121 := (publicBatchWrapper89_copies160 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_73_18 := arithEq_of_rows h (row := 73) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_18
  simp only [← c5108, k3, ← c5109, k3, ← c5110] at e_73_18
  have e_73_19 := arithEq_of_rows h (row := 73) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_73_19
  simp only [← c5117, ← c5118, k3, ← c5119] at e_73_19
  have e_74_13 := arithEq_of_rows h (row := 74) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_13
  simp only [← c5111, ← c5112, ← c5113] at e_74_13
  have e_74_14 := arithEq_of_rows h (row := 74) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_14
  simp only [← c5114, ← c5115, ← c5116] at e_74_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_13
    linear_combination c5120.trans k2 - hc
  · have hc := e_73_19
    simp only [e_73_18, e_74_14] at hc
    linear_combination c5121.trans k2 - hc

theorem publicBatchWrapper89_f630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15407)) (a (.virt 200)) (a (.virt 18427)) (a (.virt 18428)) := by
  have c5122 := (publicBatchWrapper89_copies160 a h).2.2.1
  have c5123 := (publicBatchWrapper89_copies160 a h).2.2.2.1
  have c5124 := (publicBatchWrapper89_copies160 a h).2.2.2.2.1
  have c5125 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.1
  have c5126 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.1
  have c5127 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.1
  have c5128 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.1
  have c5129 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.1
  have c5130 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5131 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5132 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5133 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5134 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5135 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_74_15 := arithEq_of_rows h (row := 74) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_15
  simp only [← c5125, ← c5126, ← c5127] at e_74_15
  have e_74_16 := arithEq_of_rows h (row := 74) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_16
  simp only [← c5128, ← c5129, ← c5130] at e_74_16
  have e_75_0 := arithEq_of_rows h (row := 75) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_0
  simp only [← c5122, k3, ← c5123, k3, ← c5124] at e_75_0
  have e_75_1 := arithEq_of_rows h (row := 75) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_1
  simp only [← c5131, ← c5132, k3, ← c5133] at e_75_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_15
    linear_combination c5134.trans k2 - hc
  · have hc := e_75_1
    simp only [e_75_0, e_74_16] at hc
    linear_combination c5135.trans k2 - hc

theorem publicBatchWrapper89_f631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15408)) (a (.virt 200)) (a (.virt 18429)) (a (.virt 18430)) := by
  have c5136 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5137 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5138 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5139 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5140 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5141 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5142 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5143 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5144 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5145 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5146 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5147 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5148 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5149 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_74_17 := arithEq_of_rows h (row := 74) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_17
  simp only [← c5139, ← c5140, ← c5141] at e_74_17
  have e_74_18 := arithEq_of_rows h (row := 74) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_18
  simp only [← c5142, ← c5143, ← c5144] at e_74_18
  have e_75_2 := arithEq_of_rows h (row := 75) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_2
  simp only [← c5136, k3, ← c5137, k3, ← c5138] at e_75_2
  have e_75_3 := arithEq_of_rows h (row := 75) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_3
  simp only [← c5145, ← c5146, k3, ← c5147] at e_75_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_74_17
    linear_combination c5148.trans k2 - hc
  · have hc := e_75_3
    simp only [e_75_2, e_74_18] at hc
    linear_combination c5149.trans k2 - hc

theorem publicBatchWrapper89_f632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 74 79) = band (a (.virt 18423)) (a (.virt 18425)) := by
  have c5150 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5151 := (publicBatchWrapper89_copies160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5152 := (publicBatchWrapper89_copies161 a h).1
  have e_74_19 := arithEq_of_rows h (row := 74) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_74_19
  simp only [← c5150, ← c5151, ← c5152] at e_74_19
  have hr := e_74_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 76 3) = band (a (.virt 18427)) (a (.virt 18429)) := by
  have c5153 := (publicBatchWrapper89_copies161 a h).2.1
  have c5154 := (publicBatchWrapper89_copies161 a h).2.2.1
  have c5155 := (publicBatchWrapper89_copies161 a h).2.2.2.1
  have e_76_0 := arithEq_of_rows h (row := 76) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_0
  simp only [← c5153, ← c5154, ← c5155] at e_76_0
  have hr := e_76_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 76 7) = band (a (.wire 74 79)) (a (.wire 76 3)) := by
  have c5156 := (publicBatchWrapper89_copies161 a h).2.2.2.2.1
  have c5157 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.1
  have c5158 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.1
  have e_76_1 := arithEq_of_rows h (row := 76) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_1
  simp only [← c5156, ← c5157, ← c5158] at e_76_1
  have hr := e_76_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15605)) (a (.virt 200)) (a (.virt 18431)) (a (.virt 18432)) := by
  have c5159 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.1
  have c5160 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.1
  have c5161 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.1
  have c5162 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5163 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5164 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5165 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5166 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5167 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5168 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5169 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5170 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5171 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5172 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_4 := arithEq_of_rows h (row := 75) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_4
  simp only [← c5159, k3, ← c5160, k3, ← c5161] at e_75_4
  have e_75_5 := arithEq_of_rows h (row := 75) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_5
  simp only [← c5168, ← c5169, k3, ← c5170] at e_75_5
  have e_76_2 := arithEq_of_rows h (row := 76) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_2
  simp only [← c5162, ← c5163, ← c5164] at e_76_2
  have e_76_3 := arithEq_of_rows h (row := 76) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_3
  simp only [← c5165, ← c5166, ← c5167] at e_76_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_2
    linear_combination c5171.trans k2 - hc
  · have hc := e_75_5
    simp only [e_75_4, e_76_3] at hc
    linear_combination c5172.trans k2 - hc

theorem publicBatchWrapper89_f636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15606)) (a (.virt 200)) (a (.virt 18433)) (a (.virt 18434)) := by
  have c5173 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5174 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5175 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5176 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5177 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5178 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5179 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5180 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5181 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5182 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5183 := (publicBatchWrapper89_copies161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5184 := (publicBatchWrapper89_copies162 a h).1
  have c5185 := (publicBatchWrapper89_copies162 a h).2.1
  have c5186 := (publicBatchWrapper89_copies162 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_6 := arithEq_of_rows h (row := 75) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_6
  simp only [← c5173, k3, ← c5174, k3, ← c5175] at e_75_6
  have e_75_7 := arithEq_of_rows h (row := 75) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_7
  simp only [← c5182, ← c5183, k3, ← c5184] at e_75_7
  have e_76_4 := arithEq_of_rows h (row := 76) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_4
  simp only [← c5176, ← c5177, ← c5178] at e_76_4
  have e_76_5 := arithEq_of_rows h (row := 76) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_5
  simp only [← c5179, ← c5180, ← c5181] at e_76_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_4
    linear_combination c5185.trans k2 - hc
  · have hc := e_75_7
    simp only [e_75_6, e_76_5] at hc
    linear_combination c5186.trans k2 - hc

theorem publicBatchWrapper89_f637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15607)) (a (.virt 200)) (a (.virt 18435)) (a (.virt 18436)) := by
  have c5187 := (publicBatchWrapper89_copies162 a h).2.2.2.1
  have c5188 := (publicBatchWrapper89_copies162 a h).2.2.2.2.1
  have c5189 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.1
  have c5190 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.1
  have c5191 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.1
  have c5192 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.1
  have c5193 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.1
  have c5194 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5195 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5196 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5197 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5198 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5199 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5200 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_8 := arithEq_of_rows h (row := 75) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_8
  simp only [← c5187, k3, ← c5188, k3, ← c5189] at e_75_8
  have e_75_9 := arithEq_of_rows h (row := 75) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_9
  simp only [← c5196, ← c5197, k3, ← c5198] at e_75_9
  have e_76_6 := arithEq_of_rows h (row := 76) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_6
  simp only [← c5190, ← c5191, ← c5192] at e_76_6
  have e_76_7 := arithEq_of_rows h (row := 76) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_7
  simp only [← c5193, ← c5194, ← c5195] at e_76_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_6
    linear_combination c5199.trans k2 - hc
  · have hc := e_75_9
    simp only [e_75_8, e_76_7] at hc
    linear_combination c5200.trans k2 - hc

theorem publicBatchWrapper89_f638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15608)) (a (.virt 200)) (a (.virt 18437)) (a (.virt 18438)) := by
  have c5201 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5202 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5203 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5204 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5205 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5206 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5207 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5208 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5209 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5210 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5211 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5212 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5213 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5214 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_10 := arithEq_of_rows h (row := 75) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_10
  simp only [← c5201, k3, ← c5202, k3, ← c5203] at e_75_10
  have e_75_11 := arithEq_of_rows h (row := 75) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_11
  simp only [← c5210, ← c5211, k3, ← c5212] at e_75_11
  have e_76_8 := arithEq_of_rows h (row := 76) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_8
  simp only [← c5204, ← c5205, ← c5206] at e_76_8
  have e_76_9 := arithEq_of_rows h (row := 76) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_9
  simp only [← c5207, ← c5208, ← c5209] at e_76_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_8
    linear_combination c5213.trans k2 - hc
  · have hc := e_75_11
    simp only [e_75_10, e_76_9] at hc
    linear_combination c5214.trans k2 - hc

theorem publicBatchWrapper89_f639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 76 43) = band (a (.virt 18431)) (a (.virt 18433)) := by
  have c5215 := (publicBatchWrapper89_copies162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5216 := (publicBatchWrapper89_copies163 a h).1
  have c5217 := (publicBatchWrapper89_copies163 a h).2.1
  have e_76_10 := arithEq_of_rows h (row := 76) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_10
  simp only [← c5215, ← c5216, ← c5217] at e_76_10
  have hr := e_76_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 76 47) = band (a (.virt 18435)) (a (.virt 18437)) := by
  have c5218 := (publicBatchWrapper89_copies163 a h).2.2.1
  have c5219 := (publicBatchWrapper89_copies163 a h).2.2.2.1
  have c5220 := (publicBatchWrapper89_copies163 a h).2.2.2.2.1
  have e_76_11 := arithEq_of_rows h (row := 76) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_11
  simp only [← c5218, ← c5219, ← c5220] at e_76_11
  have hr := e_76_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 76 51) = band (a (.wire 76 43)) (a (.wire 76 47)) := by
  have c5221 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.1
  have c5222 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.1
  have c5223 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.1
  have e_76_12 := arithEq_of_rows h (row := 76) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_12
  simp only [← c5221, ← c5222, ← c5223] at e_76_12
  have hr := e_76_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15805)) (a (.virt 200)) (a (.virt 18439)) (a (.virt 18440)) := by
  have c5224 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.1
  have c5225 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.1
  have c5226 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5227 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5228 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5229 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5230 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5231 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5232 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5233 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5234 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5235 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5236 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5237 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_12 := arithEq_of_rows h (row := 75) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_12
  simp only [← c5224, k3, ← c5225, k3, ← c5226] at e_75_12
  have e_75_13 := arithEq_of_rows h (row := 75) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_13
  simp only [← c5233, ← c5234, k3, ← c5235] at e_75_13
  have e_76_13 := arithEq_of_rows h (row := 76) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_13
  simp only [← c5227, ← c5228, ← c5229] at e_76_13
  have e_76_14 := arithEq_of_rows h (row := 76) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_14
  simp only [← c5230, ← c5231, ← c5232] at e_76_14
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_13
    linear_combination c5236.trans k2 - hc
  · have hc := e_75_13
    simp only [e_75_12, e_76_14] at hc
    linear_combination c5237.trans k2 - hc

theorem publicBatchWrapper89_f643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15806)) (a (.virt 200)) (a (.virt 18441)) (a (.virt 18442)) := by
  have c5238 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5239 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5240 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5241 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5242 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5243 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5244 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5245 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5246 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5247 := (publicBatchWrapper89_copies163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5248 := (publicBatchWrapper89_copies164 a h).1
  have c5249 := (publicBatchWrapper89_copies164 a h).2.1
  have c5250 := (publicBatchWrapper89_copies164 a h).2.2.1
  have c5251 := (publicBatchWrapper89_copies164 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_14 := arithEq_of_rows h (row := 75) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_14
  simp only [← c5238, k3, ← c5239, k3, ← c5240] at e_75_14
  have e_75_15 := arithEq_of_rows h (row := 75) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_15
  simp only [← c5247, ← c5248, k3, ← c5249] at e_75_15
  have e_76_15 := arithEq_of_rows h (row := 76) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_15
  simp only [← c5241, ← c5242, ← c5243] at e_76_15
  have e_76_16 := arithEq_of_rows h (row := 76) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_16
  simp only [← c5244, ← c5245, ← c5246] at e_76_16
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_15
    linear_combination c5250.trans k2 - hc
  · have hc := e_75_15
    simp only [e_75_14, e_76_16] at hc
    linear_combination c5251.trans k2 - hc

theorem publicBatchWrapper89_f644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15807)) (a (.virt 200)) (a (.virt 18443)) (a (.virt 18444)) := by
  have c5252 := (publicBatchWrapper89_copies164 a h).2.2.2.2.1
  have c5253 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.1
  have c5254 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.1
  have c5255 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.1
  have c5256 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.1
  have c5257 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.1
  have c5258 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5259 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5260 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5261 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5262 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5263 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5264 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5265 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_16 := arithEq_of_rows h (row := 75) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_16
  simp only [← c5252, k3, ← c5253, k3, ← c5254] at e_75_16
  have e_75_17 := arithEq_of_rows h (row := 75) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_17
  simp only [← c5261, ← c5262, k3, ← c5263] at e_75_17
  have e_76_17 := arithEq_of_rows h (row := 76) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_17
  simp only [← c5255, ← c5256, ← c5257] at e_76_17
  have e_76_18 := arithEq_of_rows h (row := 76) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_18
  simp only [← c5258, ← c5259, ← c5260] at e_76_18
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_17
    linear_combination c5264.trans k2 - hc
  · have hc := e_75_17
    simp only [e_75_16, e_76_18] at hc
    linear_combination c5265.trans k2 - hc

theorem publicBatchWrapper89_f645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 15808)) (a (.virt 200)) (a (.virt 18445)) (a (.virt 18446)) := by
  have c5266 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5267 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5268 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5269 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5270 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5271 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5272 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5273 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5274 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5275 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5276 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5277 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5278 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5279 := (publicBatchWrapper89_copies164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_75_18 := arithEq_of_rows h (row := 75) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_18
  simp only [← c5266, k3, ← c5267, k3, ← c5268] at e_75_18
  have e_75_19 := arithEq_of_rows h (row := 75) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_75_19
  simp only [← c5275, ← c5276, k3, ← c5277] at e_75_19
  have e_76_19 := arithEq_of_rows h (row := 76) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_76_19
  simp only [← c5269, ← c5270, ← c5271] at e_76_19
  have e_77_0 := arithEq_of_rows h (row := 77) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_0
  simp only [← c5272, ← c5273, ← c5274] at e_77_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_76_19
    linear_combination c5278.trans k2 - hc
  · have hc := e_75_19
    simp only [e_75_18, e_77_0] at hc
    linear_combination c5279.trans k2 - hc

theorem publicBatchWrapper89_f646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 77 7) = band (a (.virt 18439)) (a (.virt 18441)) := by
  have c5280 := (publicBatchWrapper89_copies165 a h).1
  have c5281 := (publicBatchWrapper89_copies165 a h).2.1
  have c5282 := (publicBatchWrapper89_copies165 a h).2.2.1
  have e_77_1 := arithEq_of_rows h (row := 77) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_1
  simp only [← c5280, ← c5281, ← c5282] at e_77_1
  have hr := e_77_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 77 11) = band (a (.virt 18443)) (a (.virt 18445)) := by
  have c5283 := (publicBatchWrapper89_copies165 a h).2.2.2.1
  have c5284 := (publicBatchWrapper89_copies165 a h).2.2.2.2.1
  have c5285 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.1
  have e_77_2 := arithEq_of_rows h (row := 77) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_2
  simp only [← c5283, ← c5284, ← c5285] at e_77_2
  have hr := e_77_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 77 15) = band (a (.wire 77 7)) (a (.wire 77 11)) := by
  have c5286 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.1
  have c5287 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.1
  have c5288 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.1
  have e_77_3 := arithEq_of_rows h (row := 77) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_3
  simp only [← c5286, ← c5287, ← c5288] at e_77_3
  have hr := e_77_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16005)) (a (.virt 200)) (a (.virt 18447)) (a (.virt 18448)) := by
  have c5289 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.1
  have c5290 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5291 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5292 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5293 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5294 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5295 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5296 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5297 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5298 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5299 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5300 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5301 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5302 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_4 := arithEq_of_rows h (row := 77) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_4
  simp only [← c5292, ← c5293, ← c5294] at e_77_4
  have e_77_5 := arithEq_of_rows h (row := 77) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_5
  simp only [← c5295, ← c5296, ← c5297] at e_77_5
  have e_78_0 := arithEq_of_rows h (row := 78) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_0
  simp only [← c5289, k3, ← c5290, k3, ← c5291] at e_78_0
  have e_78_1 := arithEq_of_rows h (row := 78) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_1
  simp only [← c5298, ← c5299, k3, ← c5300] at e_78_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_4
    linear_combination c5301.trans k2 - hc
  · have hc := e_78_1
    simp only [e_78_0, e_77_5] at hc
    linear_combination c5302.trans k2 - hc

theorem publicBatchWrapper89_f650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16006)) (a (.virt 200)) (a (.virt 18449)) (a (.virt 18450)) := by
  have c5303 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5304 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5305 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5306 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5307 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5308 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5309 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5310 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5311 := (publicBatchWrapper89_copies165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5312 := (publicBatchWrapper89_copies166 a h).1
  have c5313 := (publicBatchWrapper89_copies166 a h).2.1
  have c5314 := (publicBatchWrapper89_copies166 a h).2.2.1
  have c5315 := (publicBatchWrapper89_copies166 a h).2.2.2.1
  have c5316 := (publicBatchWrapper89_copies166 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_6 := arithEq_of_rows h (row := 77) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_6
  simp only [← c5306, ← c5307, ← c5308] at e_77_6
  have e_77_7 := arithEq_of_rows h (row := 77) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_7
  simp only [← c5309, ← c5310, ← c5311] at e_77_7
  have e_78_2 := arithEq_of_rows h (row := 78) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_2
  simp only [← c5303, k3, ← c5304, k3, ← c5305] at e_78_2
  have e_78_3 := arithEq_of_rows h (row := 78) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_3
  simp only [← c5312, ← c5313, k3, ← c5314] at e_78_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_6
    linear_combination c5315.trans k2 - hc
  · have hc := e_78_3
    simp only [e_78_2, e_77_7] at hc
    linear_combination c5316.trans k2 - hc

theorem publicBatchWrapper89_f651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16007)) (a (.virt 200)) (a (.virt 18451)) (a (.virt 18452)) := by
  have c5317 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.1
  have c5318 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.1
  have c5319 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.1
  have c5320 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.1
  have c5321 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.1
  have c5322 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5323 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5324 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5325 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5326 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5327 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5328 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5329 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5330 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_8 := arithEq_of_rows h (row := 77) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_8
  simp only [← c5320, ← c5321, ← c5322] at e_77_8
  have e_77_9 := arithEq_of_rows h (row := 77) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_9
  simp only [← c5323, ← c5324, ← c5325] at e_77_9
  have e_78_4 := arithEq_of_rows h (row := 78) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_4
  simp only [← c5317, k3, ← c5318, k3, ← c5319] at e_78_4
  have e_78_5 := arithEq_of_rows h (row := 78) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_5
  simp only [← c5326, ← c5327, k3, ← c5328] at e_78_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_8
    linear_combination c5329.trans k2 - hc
  · have hc := e_78_5
    simp only [e_78_4, e_77_9] at hc
    linear_combination c5330.trans k2 - hc

theorem publicBatchWrapper89_f652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16008)) (a (.virt 200)) (a (.virt 18453)) (a (.virt 18454)) := by
  have c5331 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5332 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5333 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5334 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5335 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5336 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5337 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5338 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5339 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5340 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5341 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5342 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5343 := (publicBatchWrapper89_copies166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5344 := (publicBatchWrapper89_copies167 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_10 := arithEq_of_rows h (row := 77) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_10
  simp only [← c5334, ← c5335, ← c5336] at e_77_10
  have e_77_11 := arithEq_of_rows h (row := 77) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_11
  simp only [← c5337, ← c5338, ← c5339] at e_77_11
  have e_78_6 := arithEq_of_rows h (row := 78) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_6
  simp only [← c5331, k3, ← c5332, k3, ← c5333] at e_78_6
  have e_78_7 := arithEq_of_rows h (row := 78) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_7
  simp only [← c5340, ← c5341, k3, ← c5342] at e_78_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_10
    linear_combination c5343.trans k2 - hc
  · have hc := e_78_7
    simp only [e_78_6, e_77_11] at hc
    linear_combination c5344.trans k2 - hc

theorem publicBatchWrapper89_f653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 77 51) = band (a (.virt 18447)) (a (.virt 18449)) := by
  have c5345 := (publicBatchWrapper89_copies167 a h).2.1
  have c5346 := (publicBatchWrapper89_copies167 a h).2.2.1
  have c5347 := (publicBatchWrapper89_copies167 a h).2.2.2.1
  have e_77_12 := arithEq_of_rows h (row := 77) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_12
  simp only [← c5345, ← c5346, ← c5347] at e_77_12
  have hr := e_77_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 77 55) = band (a (.virt 18451)) (a (.virt 18453)) := by
  have c5348 := (publicBatchWrapper89_copies167 a h).2.2.2.2.1
  have c5349 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.1
  have c5350 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.1
  have e_77_13 := arithEq_of_rows h (row := 77) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_13
  simp only [← c5348, ← c5349, ← c5350] at e_77_13
  have hr := e_77_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 77 59) = band (a (.wire 77 51)) (a (.wire 77 55)) := by
  have c5351 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.1
  have c5352 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.1
  have c5353 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.1
  have e_77_14 := arithEq_of_rows h (row := 77) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_14
  simp only [← c5351, ← c5352, ← c5353] at e_77_14
  have hr := e_77_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16205)) (a (.virt 200)) (a (.virt 18455)) (a (.virt 18456)) := by
  have c5354 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5355 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5356 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5357 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5358 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5359 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5360 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5361 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5362 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5363 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5364 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5365 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5366 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5367 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_15 := arithEq_of_rows h (row := 77) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_15
  simp only [← c5357, ← c5358, ← c5359] at e_77_15
  have e_77_16 := arithEq_of_rows h (row := 77) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_16
  simp only [← c5360, ← c5361, ← c5362] at e_77_16
  have e_78_8 := arithEq_of_rows h (row := 78) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_8
  simp only [← c5354, k3, ← c5355, k3, ← c5356] at e_78_8
  have e_78_9 := arithEq_of_rows h (row := 78) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_9
  simp only [← c5363, ← c5364, k3, ← c5365] at e_78_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_15
    linear_combination c5366.trans k2 - hc
  · have hc := e_78_9
    simp only [e_78_8, e_77_16] at hc
    linear_combination c5367.trans k2 - hc

theorem publicBatchWrapper89_f657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16206)) (a (.virt 200)) (a (.virt 18457)) (a (.virt 18458)) := by
  have c5368 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5369 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5370 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5371 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5372 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5373 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5374 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5375 := (publicBatchWrapper89_copies167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5376 := (publicBatchWrapper89_copies168 a h).1
  have c5377 := (publicBatchWrapper89_copies168 a h).2.1
  have c5378 := (publicBatchWrapper89_copies168 a h).2.2.1
  have c5379 := (publicBatchWrapper89_copies168 a h).2.2.2.1
  have c5380 := (publicBatchWrapper89_copies168 a h).2.2.2.2.1
  have c5381 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_17 := arithEq_of_rows h (row := 77) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_17
  simp only [← c5371, ← c5372, ← c5373] at e_77_17
  have e_77_18 := arithEq_of_rows h (row := 77) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_18
  simp only [← c5374, ← c5375, ← c5376] at e_77_18
  have e_78_10 := arithEq_of_rows h (row := 78) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_10
  simp only [← c5368, k3, ← c5369, k3, ← c5370] at e_78_10
  have e_78_11 := arithEq_of_rows h (row := 78) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_11
  simp only [← c5377, ← c5378, k3, ← c5379] at e_78_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_17
    linear_combination c5380.trans k2 - hc
  · have hc := e_78_11
    simp only [e_78_10, e_77_18] at hc
    linear_combination c5381.trans k2 - hc

theorem publicBatchWrapper89_f658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16207)) (a (.virt 200)) (a (.virt 18459)) (a (.virt 18460)) := by
  have c5382 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.1
  have c5383 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.1
  have c5384 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.1
  have c5385 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.1
  have c5386 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5387 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5388 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5389 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5390 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5391 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5392 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5393 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5394 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5395 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_77_19 := arithEq_of_rows h (row := 77) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_77_19
  simp only [← c5385, ← c5386, ← c5387] at e_77_19
  have e_78_12 := arithEq_of_rows h (row := 78) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_12
  simp only [← c5382, k3, ← c5383, k3, ← c5384] at e_78_12
  have e_78_13 := arithEq_of_rows h (row := 78) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_13
  simp only [← c5391, ← c5392, k3, ← c5393] at e_78_13
  have e_79_0 := arithEq_of_rows h (row := 79) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_0
  simp only [← c5388, ← c5389, ← c5390] at e_79_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_77_19
    linear_combination c5394.trans k2 - hc
  · have hc := e_78_13
    simp only [e_78_12, e_79_0] at hc
    linear_combination c5395.trans k2 - hc

theorem publicBatchWrapper89_f659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16208)) (a (.virt 200)) (a (.virt 18461)) (a (.virt 18462)) := by
  have c5396 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5397 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5398 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5399 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5400 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5401 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5402 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5403 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5404 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5405 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5406 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5407 := (publicBatchWrapper89_copies168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5408 := (publicBatchWrapper89_copies169 a h).1
  have c5409 := (publicBatchWrapper89_copies169 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_78_14 := arithEq_of_rows h (row := 78) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_14
  simp only [← c5396, k3, ← c5397, k3, ← c5398] at e_78_14
  have e_78_15 := arithEq_of_rows h (row := 78) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_15
  simp only [← c5405, ← c5406, k3, ← c5407] at e_78_15
  have e_79_1 := arithEq_of_rows h (row := 79) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_1
  simp only [← c5399, ← c5400, ← c5401] at e_79_1
  have e_79_2 := arithEq_of_rows h (row := 79) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_2
  simp only [← c5402, ← c5403, ← c5404] at e_79_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_1
    linear_combination c5408.trans k2 - hc
  · have hc := e_78_15
    simp only [e_78_14, e_79_2] at hc
    linear_combination c5409.trans k2 - hc

theorem publicBatchWrapper89_f660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 79 15) = band (a (.virt 18455)) (a (.virt 18457)) := by
  have c5410 := (publicBatchWrapper89_copies169 a h).2.2.1
  have c5411 := (publicBatchWrapper89_copies169 a h).2.2.2.1
  have c5412 := (publicBatchWrapper89_copies169 a h).2.2.2.2.1
  have e_79_3 := arithEq_of_rows h (row := 79) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_3
  simp only [← c5410, ← c5411, ← c5412] at e_79_3
  have hr := e_79_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 79 19) = band (a (.virt 18459)) (a (.virt 18461)) := by
  have c5413 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.1
  have c5414 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.1
  have c5415 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.1
  have e_79_4 := arithEq_of_rows h (row := 79) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_4
  simp only [← c5413, ← c5414, ← c5415] at e_79_4
  have hr := e_79_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 79 23) = band (a (.wire 79 15)) (a (.wire 79 19)) := by
  have c5416 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.1
  have c5417 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.1
  have c5418 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_79_5 := arithEq_of_rows h (row := 79) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_5
  simp only [← c5416, ← c5417, ← c5418] at e_79_5
  have hr := e_79_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16405)) (a (.virt 200)) (a (.virt 18463)) (a (.virt 18464)) := by
  have c5419 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5420 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5421 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5422 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5423 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5424 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5425 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5426 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5427 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5428 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5429 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5430 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5431 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5432 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_78_16 := arithEq_of_rows h (row := 78) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_16
  simp only [← c5419, k3, ← c5420, k3, ← c5421] at e_78_16
  have e_78_17 := arithEq_of_rows h (row := 78) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_17
  simp only [← c5428, ← c5429, k3, ← c5430] at e_78_17
  have e_79_6 := arithEq_of_rows h (row := 79) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_6
  simp only [← c5422, ← c5423, ← c5424] at e_79_6
  have e_79_7 := arithEq_of_rows h (row := 79) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_7
  simp only [← c5425, ← c5426, ← c5427] at e_79_7
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_6
    linear_combination c5431.trans k2 - hc
  · have hc := e_78_17
    simp only [e_78_16, e_79_7] at hc
    linear_combination c5432.trans k2 - hc

theorem publicBatchWrapper89_f664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16406)) (a (.virt 200)) (a (.virt 18465)) (a (.virt 18466)) := by
  have c5433 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5434 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5435 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5436 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5437 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5438 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5439 := (publicBatchWrapper89_copies169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5440 := (publicBatchWrapper89_copies170 a h).1
  have c5441 := (publicBatchWrapper89_copies170 a h).2.1
  have c5442 := (publicBatchWrapper89_copies170 a h).2.2.1
  have c5443 := (publicBatchWrapper89_copies170 a h).2.2.2.1
  have c5444 := (publicBatchWrapper89_copies170 a h).2.2.2.2.1
  have c5445 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.1
  have c5446 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_78_18 := arithEq_of_rows h (row := 78) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_18
  simp only [← c5433, k3, ← c5434, k3, ← c5435] at e_78_18
  have e_78_19 := arithEq_of_rows h (row := 78) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_78_19
  simp only [← c5442, ← c5443, k3, ← c5444] at e_78_19
  have e_79_8 := arithEq_of_rows h (row := 79) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_8
  simp only [← c5436, ← c5437, ← c5438] at e_79_8
  have e_79_9 := arithEq_of_rows h (row := 79) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_9
  simp only [← c5439, ← c5440, ← c5441] at e_79_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_8
    linear_combination c5445.trans k2 - hc
  · have hc := e_78_19
    simp only [e_78_18, e_79_9] at hc
    linear_combination c5446.trans k2 - hc

theorem publicBatchWrapper89_f665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16407)) (a (.virt 200)) (a (.virt 18467)) (a (.virt 18468)) := by
  have c5447 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.1
  have c5448 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.1
  have c5449 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.1
  have c5450 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5451 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5452 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5453 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5454 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5455 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5456 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5457 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5458 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5459 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5460 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_79_10 := arithEq_of_rows h (row := 79) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_10
  simp only [← c5450, ← c5451, ← c5452] at e_79_10
  have e_79_11 := arithEq_of_rows h (row := 79) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_11
  simp only [← c5453, ← c5454, ← c5455] at e_79_11
  have e_80_0 := arithEq_of_rows h (row := 80) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_0
  simp only [← c5447, k3, ← c5448, k3, ← c5449] at e_80_0
  have e_80_1 := arithEq_of_rows h (row := 80) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_1
  simp only [← c5456, ← c5457, k3, ← c5458] at e_80_1
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_10
    linear_combination c5459.trans k2 - hc
  · have hc := e_80_1
    simp only [e_80_0, e_79_11] at hc
    linear_combination c5460.trans k2 - hc

theorem publicBatchWrapper89_f666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16408)) (a (.virt 200)) (a (.virt 18469)) (a (.virt 18470)) := by
  have c5461 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5462 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5463 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5464 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5465 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5466 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5467 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5468 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5469 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5470 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5471 := (publicBatchWrapper89_copies170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5472 := (publicBatchWrapper89_copies171 a h).1
  have c5473 := (publicBatchWrapper89_copies171 a h).2.1
  have c5474 := (publicBatchWrapper89_copies171 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_79_12 := arithEq_of_rows h (row := 79) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_12
  simp only [← c5464, ← c5465, ← c5466] at e_79_12
  have e_79_13 := arithEq_of_rows h (row := 79) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_13
  simp only [← c5467, ← c5468, ← c5469] at e_79_13
  have e_80_2 := arithEq_of_rows h (row := 80) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_2
  simp only [← c5461, k3, ← c5462, k3, ← c5463] at e_80_2
  have e_80_3 := arithEq_of_rows h (row := 80) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_3
  simp only [← c5470, ← c5471, k3, ← c5472] at e_80_3
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_12
    linear_combination c5473.trans k2 - hc
  · have hc := e_80_3
    simp only [e_80_2, e_79_13] at hc
    linear_combination c5474.trans k2 - hc

theorem publicBatchWrapper89_f667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 79 59) = band (a (.virt 18463)) (a (.virt 18465)) := by
  have c5475 := (publicBatchWrapper89_copies171 a h).2.2.2.1
  have c5476 := (publicBatchWrapper89_copies171 a h).2.2.2.2.1
  have c5477 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.1
  have e_79_14 := arithEq_of_rows h (row := 79) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_14
  simp only [← c5475, ← c5476, ← c5477] at e_79_14
  have hr := e_79_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 79 63) = band (a (.virt 18467)) (a (.virt 18469)) := by
  have c5478 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.1
  have c5479 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.1
  have c5480 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.1
  have e_79_15 := arithEq_of_rows h (row := 79) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_15
  simp only [← c5478, ← c5479, ← c5480] at e_79_15
  have hr := e_79_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 79 67) = band (a (.wire 79 59)) (a (.wire 79 63)) := by
  have c5481 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.1
  have c5482 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5483 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_79_16 := arithEq_of_rows h (row := 79) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_16
  simp only [← c5481, ← c5482, ← c5483] at e_79_16
  have hr := e_79_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16605)) (a (.virt 200)) (a (.virt 18471)) (a (.virt 18472)) := by
  have c5484 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5485 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5486 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5487 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5488 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5489 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5490 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5491 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5492 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5493 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5494 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5495 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5496 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5497 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_79_17 := arithEq_of_rows h (row := 79) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_17
  simp only [← c5487, ← c5488, ← c5489] at e_79_17
  have e_79_18 := arithEq_of_rows h (row := 79) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_18
  simp only [← c5490, ← c5491, ← c5492] at e_79_18
  have e_80_4 := arithEq_of_rows h (row := 80) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_4
  simp only [← c5484, k3, ← c5485, k3, ← c5486] at e_80_4
  have e_80_5 := arithEq_of_rows h (row := 80) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_5
  simp only [← c5493, ← c5494, k3, ← c5495] at e_80_5
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_17
    linear_combination c5496.trans k2 - hc
  · have hc := e_80_5
    simp only [e_80_4, e_79_18] at hc
    linear_combination c5497.trans k2 - hc

theorem publicBatchWrapper89_f671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16606)) (a (.virt 200)) (a (.virt 18473)) (a (.virt 18474)) := by
  have c5498 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5499 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5500 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5501 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5502 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5503 := (publicBatchWrapper89_copies171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5504 := (publicBatchWrapper89_copies172 a h).1
  have c5505 := (publicBatchWrapper89_copies172 a h).2.1
  have c5506 := (publicBatchWrapper89_copies172 a h).2.2.1
  have c5507 := (publicBatchWrapper89_copies172 a h).2.2.2.1
  have c5508 := (publicBatchWrapper89_copies172 a h).2.2.2.2.1
  have c5509 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.1
  have c5510 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.1
  have c5511 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_79_19 := arithEq_of_rows h (row := 79) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_79_19
  simp only [← c5501, ← c5502, ← c5503] at e_79_19
  have e_80_6 := arithEq_of_rows h (row := 80) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_6
  simp only [← c5498, k3, ← c5499, k3, ← c5500] at e_80_6
  have e_80_7 := arithEq_of_rows h (row := 80) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_7
  simp only [← c5507, ← c5508, k3, ← c5509] at e_80_7
  have e_81_0 := arithEq_of_rows h (row := 81) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_0
  simp only [← c5504, ← c5505, ← c5506] at e_81_0
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_79_19
    linear_combination c5510.trans k2 - hc
  · have hc := e_80_7
    simp only [e_80_6, e_81_0] at hc
    linear_combination c5511.trans k2 - hc

theorem publicBatchWrapper89_f672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16607)) (a (.virt 200)) (a (.virt 18475)) (a (.virt 18476)) := by
  have c5512 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.1
  have c5513 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.1
  have c5514 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5515 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5516 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5517 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5518 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5519 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5520 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5521 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5522 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5523 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5524 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5525 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_80_8 := arithEq_of_rows h (row := 80) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_8
  simp only [← c5512, k3, ← c5513, k3, ← c5514] at e_80_8
  have e_80_9 := arithEq_of_rows h (row := 80) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_9
  simp only [← c5521, ← c5522, k3, ← c5523] at e_80_9
  have e_81_1 := arithEq_of_rows h (row := 81) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_1
  simp only [← c5515, ← c5516, ← c5517] at e_81_1
  have e_81_2 := arithEq_of_rows h (row := 81) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_2
  simp only [← c5518, ← c5519, ← c5520] at e_81_2
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_1
    linear_combination c5524.trans k2 - hc
  · have hc := e_80_9
    simp only [e_80_8, e_81_2] at hc
    linear_combination c5525.trans k2 - hc

theorem publicBatchWrapper89_f673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16608)) (a (.virt 200)) (a (.virt 18477)) (a (.virt 18478)) := by
  have c5526 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5527 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5528 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5529 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5530 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5531 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5532 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5533 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5534 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5535 := (publicBatchWrapper89_copies172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5536 := (publicBatchWrapper89_copies173 a h).1
  have c5537 := (publicBatchWrapper89_copies173 a h).2.1
  have c5538 := (publicBatchWrapper89_copies173 a h).2.2.1
  have c5539 := (publicBatchWrapper89_copies173 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_80_10 := arithEq_of_rows h (row := 80) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_10
  simp only [← c5526, k3, ← c5527, k3, ← c5528] at e_80_10
  have e_80_11 := arithEq_of_rows h (row := 80) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_11
  simp only [← c5535, ← c5536, k3, ← c5537] at e_80_11
  have e_81_3 := arithEq_of_rows h (row := 81) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_3
  simp only [← c5529, ← c5530, ← c5531] at e_81_3
  have e_81_4 := arithEq_of_rows h (row := 81) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_4
  simp only [← c5532, ← c5533, ← c5534] at e_81_4
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_3
    linear_combination c5538.trans k2 - hc
  · have hc := e_80_11
    simp only [e_80_10, e_81_4] at hc
    linear_combination c5539.trans k2 - hc

theorem publicBatchWrapper89_f674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 81 23) = band (a (.virt 18471)) (a (.virt 18473)) := by
  have c5540 := (publicBatchWrapper89_copies173 a h).2.2.2.2.1
  have c5541 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.1
  have c5542 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.1
  have e_81_5 := arithEq_of_rows h (row := 81) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_5
  simp only [← c5540, ← c5541, ← c5542] at e_81_5
  have hr := e_81_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 81 27) = band (a (.virt 18475)) (a (.virt 18477)) := by
  have c5543 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.1
  have c5544 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.1
  have c5545 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.1
  have e_81_6 := arithEq_of_rows h (row := 81) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_6
  simp only [← c5543, ← c5544, ← c5545] at e_81_6
  have hr := e_81_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 81 31) = band (a (.wire 81 23)) (a (.wire 81 27)) := by
  have c5546 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.1
  have c5547 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c5548 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_81_7 := arithEq_of_rows h (row := 81) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_7
  simp only [← c5546, ← c5547, ← c5548] at e_81_7
  have hr := e_81_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper89_f677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16805)) (a (.virt 200)) (a (.virt 18479)) (a (.virt 18480)) := by
  have c5549 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5550 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5551 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5552 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5553 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5554 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5555 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5556 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5557 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5558 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5559 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5560 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5561 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5562 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_80_12 := arithEq_of_rows h (row := 80) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_12
  simp only [← c5549, k3, ← c5550, k3, ← c5551] at e_80_12
  have e_80_13 := arithEq_of_rows h (row := 80) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_13
  simp only [← c5558, ← c5559, k3, ← c5560] at e_80_13
  have e_81_8 := arithEq_of_rows h (row := 81) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_8
  simp only [← c5552, ← c5553, ← c5554] at e_81_8
  have e_81_9 := arithEq_of_rows h (row := 81) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_9
  simp only [← c5555, ← c5556, ← c5557] at e_81_9
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_8
    linear_combination c5561.trans k2 - hc
  · have hc := e_80_13
    simp only [e_80_12, e_81_9] at hc
    linear_combination c5562.trans k2 - hc

theorem publicBatchWrapper89_f678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    IsEqual (a (.virt 16806)) (a (.virt 200)) (a (.virt 18481)) (a (.virt 18482)) := by
  have c5563 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5564 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5565 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5566 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c5567 := (publicBatchWrapper89_copies173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c5568 := (publicBatchWrapper89_copies174 a h).1
  have c5569 := (publicBatchWrapper89_copies174 a h).2.1
  have c5570 := (publicBatchWrapper89_copies174 a h).2.2.1
  have c5571 := (publicBatchWrapper89_copies174 a h).2.2.2.1
  have c5572 := (publicBatchWrapper89_copies174 a h).2.2.2.2.1
  have c5573 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.1
  have c5574 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.1
  have c5575 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.1
  have c5576 := (publicBatchWrapper89_copies174 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_80_14 := arithEq_of_rows h (row := 80) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_14
  simp only [← c5563, k3, ← c5564, k3, ← c5565] at e_80_14
  have e_80_15 := arithEq_of_rows h (row := 80) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_80_15
  simp only [← c5572, ← c5573, k3, ← c5574] at e_80_15
  have e_81_10 := arithEq_of_rows h (row := 81) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_10
  simp only [← c5566, ← c5567, ← c5568] at e_81_10
  have e_81_11 := arithEq_of_rows h (row := 81) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_81_11
  simp only [← c5569, ← c5570, ← c5571] at e_81_11
  simp only [k2]
  refine ⟨?_, ?_⟩
  · have hc := e_81_10
    linear_combination c5575.trans k2 - hc
  · have hc := e_80_15
    simp only [e_80_14, e_81_11] at hc
    linear_combination c5576.trans k2 - hc

end Plonky2Spec.Generated
