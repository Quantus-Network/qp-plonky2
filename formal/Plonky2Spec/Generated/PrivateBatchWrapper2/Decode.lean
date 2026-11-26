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
import Plonky2Spec.Generated.PrivateBatchWrapper2.Fact

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `privateBatchWrapper2` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem privateBatchWrapper2_decode (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (IsEqual (a (.virt 5)) (a (.virt 11)) (a (.virt 44)) (a (.virt 45)) ∧
    IsEqual (a (.virt 6)) (a (.virt 11)) (a (.virt 46)) (a (.virt 47)) ∧
    IsEqual (a (.virt 7)) (a (.virt 11)) (a (.virt 48)) (a (.virt 49)) ∧
    IsEqual (a (.virt 8)) (a (.virt 11)) (a (.virt 50)) (a (.virt 51)) ∧
    a (.wire 3 23) = band (a (.virt 44)) (a (.virt 46)) ∧
    a (.wire 3 27) = band (a (.virt 48)) (a (.virt 50)) ∧
    a (.wire 3 31) = band (a (.wire 3 23)) (a (.wire 3 27)) ∧
    IsEqual (a (.virt 19)) (a (.virt 11)) (a (.virt 52)) (a (.virt 53)) ∧
    IsEqual (a (.virt 20)) (a (.virt 11)) (a (.virt 54)) (a (.virt 55)) ∧
    IsEqual (a (.virt 21)) (a (.virt 11)) (a (.virt 56)) (a (.virt 57)) ∧
    IsEqual (a (.virt 22)) (a (.virt 11)) (a (.virt 58)) (a (.virt 59)) ∧
    a (.wire 4 7) = band (a (.virt 52)) (a (.virt 54)) ∧
    a (.wire 4 11) = band (a (.virt 56)) (a (.virt 58)) ∧
    a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11)) ∧
    a (.wire 5 7) = bnot (a (.wire 3 31)) ∧
    a (.virt 12) = bnot (a (.virt 11)) ∧
    a (.wire 5 7) = band (a (.wire 5 7)) (a (.virt 12)) ∧
    a (.wire 5 11) = bselect (a (.wire 5 7)) (a (.virt 5)) (a (.virt 11)) ∧
    a (.wire 5 15) = bselect (a (.wire 5 7)) (a (.virt 6)) (a (.virt 11)) ∧
    a (.wire 5 19) = bselect (a (.wire 5 7)) (a (.virt 7)) (a (.virt 11)) ∧
    a (.wire 5 23) = bselect (a (.wire 5 7)) (a (.virt 8)) (a (.virt 11)) ∧
    a (.wire 5 27) = bselect (a (.wire 5 7)) (a (.virt 9)) (a (.virt 11)) ∧
    a (.wire 5 7) = bor (a (.virt 11)) (a (.wire 5 7)) ∧
    a (.wire 5 31) = bnot (a (.wire 4 15)) ∧
    a (.wire 5 35) = bnot (a (.wire 5 7)) ∧
    a (.wire 4 19) = band (a (.wire 5 31)) (a (.wire 5 35)) ∧
    a (.wire 5 43) = bselect (a (.wire 4 19)) (a (.virt 19)) (a (.wire 5 11)) ∧
    a (.wire 5 51) = bselect (a (.wire 4 19)) (a (.virt 20)) (a (.wire 5 15)) ∧
    a (.wire 5 59) = bselect (a (.wire 4 19)) (a (.virt 21)) (a (.wire 5 19)) ∧
    a (.wire 6 7) = bselect (a (.wire 4 19)) (a (.virt 22)) (a (.wire 5 23)) ∧
    a (.wire 6 15) = bselect (a (.wire 4 19)) (a (.virt 23)) (a (.wire 5 27)) ∧
    a (.wire 0 59) = bor (a (.wire 5 7)) (a (.wire 5 31))) ∧
    (IsEqual (a (.virt 5)) (a (.wire 5 43)) (a (.virt 60)) (a (.virt 61)) ∧
    IsEqual (a (.virt 6)) (a (.wire 5 51)) (a (.virt 62)) (a (.virt 63)) ∧
    IsEqual (a (.virt 7)) (a (.wire 5 59)) (a (.virt 64)) (a (.virt 65)) ∧
    IsEqual (a (.virt 8)) (a (.wire 6 7)) (a (.virt 66)) (a (.virt 67)) ∧
    a (.wire 4 55) = band (a (.virt 60)) (a (.virt 62)) ∧
    a (.wire 4 59) = band (a (.virt 64)) (a (.virt 66)) ∧
    a (.wire 9 3) = band (a (.wire 4 55)) (a (.wire 4 59)) ∧
    a (.wire 10 3) = bor (a (.wire 3 31)) (a (.wire 9 3)) ∧
    a (.wire 10 3) = a (.virt 12) ∧
    a (.virt 0) = a (.virt 0) ∧
    IsEqual (a (.virt 19)) (a (.wire 5 43)) (a (.virt 68)) (a (.virt 69)) ∧
    IsEqual (a (.virt 20)) (a (.wire 5 51)) (a (.virt 70)) (a (.virt 71)) ∧
    IsEqual (a (.virt 21)) (a (.wire 5 59)) (a (.virt 72)) (a (.virt 73)) ∧
    IsEqual (a (.virt 22)) (a (.wire 6 7)) (a (.virt 74)) (a (.virt 75)) ∧
    a (.wire 9 39) = band (a (.virt 68)) (a (.virt 70)) ∧
    a (.wire 9 43) = band (a (.virt 72)) (a (.virt 74)) ∧
    a (.wire 9 47) = band (a (.wire 9 39)) (a (.wire 9 43)) ∧
    a (.wire 10 7) = bor (a (.wire 4 15)) (a (.wire 9 47)) ∧
    a (.wire 10 7) = a (.virt 12) ∧
    a (.virt 14) = a (.virt 0) ∧
    a (.wire 8 59) = bselect (a (.wire 3 31)) (a (.virt 11)) (a (.virt 10)) ∧
    a (.wire 8 59) = a (.virt 11) + a (.wire 8 59) ∧
    a (.wire 11 7) = bselect (a (.wire 4 15)) (a (.virt 11)) (a (.virt 24)) ∧
    a (.wire 10 11) = a (.wire 8 59) + a (.wire 11 7) ∧
    rangeCheck (a (.virt 30)) 32 ∧
    a (.virt 30) = a (.virt 11) + a (.virt 30) ∧
    rangeCheck (a (.virt 35)) 32 ∧
    a (.wire 10 15) = a (.virt 30) + a (.virt 35) ∧
    rangeCheck (a (.virt 25)) 14 ∧
    a (.wire 11 11) = a (.virt 76) - a (.virt 25) ∧
    rangeCheck (a (.wire 11 11)) 14 ∧
    a (.wire 9 51) = a (.wire 10 15) * a (.virt 76)) ∧
    (a (.wire 9 55) = a (.wire 10 11) * a (.wire 11 11) ∧
    a (.wire 11 15) = a (.wire 9 55) - a (.wire 9 51) ∧
    rangeCheck (a (.wire 11 15)) 52 ∧
    a (.wire 5 7) = bnot (a (.wire 3 31)) ∧
    a (.wire 5 31) = bnot (a (.wire 4 15)) ∧
    a (.wire 9 59) = band (a (.wire 5 7)) (a (.wire 5 31)) ∧
    IsEqual (a (.virt 1)) (a (.virt 15)) (a (.virt 77)) (a (.virt 78)) ∧
    IsEqual (a (.virt 2)) (a (.virt 16)) (a (.virt 79)) (a (.virt 80)) ∧
    IsEqual (a (.virt 3)) (a (.virt 17)) (a (.virt 81)) (a (.virt 82)) ∧
    IsEqual (a (.virt 4)) (a (.virt 18)) (a (.virt 83)) (a (.virt 84)) ∧
    a (.wire 17 35) = band (a (.virt 77)) (a (.virt 79)) ∧
    a (.wire 17 39) = band (a (.virt 81)) (a (.virt 83)) ∧
    a (.wire 17 43) = band (a (.wire 17 35)) (a (.wire 17 39)) ∧
    a (.wire 17 47) = band (a (.wire 9 59)) (a (.wire 17 43)) ∧
    a (.wire 17 47) = a (.virt 11) ∧
    a (.wire 5 31) = bnot (a (.wire 4 15)) ∧
    (a (.wire 19 12) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 0 ∧ a (.wire 19 13) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 1 ∧ a (.wire 19 14) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 2 ∧ a (.wire 19 15) = spongeHash perm [a (.virt 36), a (.virt 37), a (.virt 38), a (.virt 39)] 3) ∧
    (a (.wire 20 12) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 0 ∧ a (.wire 20 13) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 1 ∧ a (.wire 20 14) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 2 ∧ a (.wire 20 15) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 3) ∧
    a (.wire 18 11) = bselect (a (.wire 3 31)) (a (.wire 20 12)) (a (.virt 1)) ∧
    a (.wire 18 19) = bselect (a (.wire 3 31)) (a (.wire 20 13)) (a (.virt 2)) ∧
    a (.wire 18 27) = bselect (a (.wire 3 31)) (a (.wire 20 14)) (a (.virt 3)) ∧
    a (.wire 18 35) = bselect (a (.wire 3 31)) (a (.wire 20 15)) (a (.virt 4)) ∧
    (a (.wire 21 12) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 0 ∧ a (.wire 21 13) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 1 ∧ a (.wire 21 14) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 2 ∧ a (.wire 21 15) = spongeHash perm [a (.virt 40), a (.virt 41), a (.virt 42), a (.virt 43)] 3) ∧
    (a (.wire 22 12) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 0 ∧ a (.wire 22 13) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 1 ∧ a (.wire 22 14) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 2 ∧ a (.wire 22 15) = spongeHash perm [a (.wire 21 12), a (.wire 21 13), a (.wire 21 14), a (.wire 21 15)] 3) ∧
    a (.wire 18 43) = bselect (a (.wire 4 15)) (a (.wire 22 12)) (a (.virt 15)) ∧
    a (.wire 18 51) = bselect (a (.wire 4 15)) (a (.wire 22 13)) (a (.virt 16)) ∧
    a (.wire 18 59) = bselect (a (.wire 4 15)) (a (.wire 22 14)) (a (.virt 17)) ∧
    a (.wire 23 7) = bselect (a (.wire 4 15)) (a (.wire 22 15)) (a (.virt 18))) :=
  ⟨⟨privateBatchWrapper2_f0 a h, privateBatchWrapper2_f1 a h, privateBatchWrapper2_f2 a h, privateBatchWrapper2_f3 a h, privateBatchWrapper2_f4 a h, privateBatchWrapper2_f5 a h, privateBatchWrapper2_f6 a h, privateBatchWrapper2_f7 a h, privateBatchWrapper2_f8 a h, privateBatchWrapper2_f9 a h, privateBatchWrapper2_f10 a h, privateBatchWrapper2_f11 a h, privateBatchWrapper2_f12 a h, privateBatchWrapper2_f13 a h, privateBatchWrapper2_f14 a h, privateBatchWrapper2_f15 a h, privateBatchWrapper2_f16 a h, privateBatchWrapper2_f17 a h, privateBatchWrapper2_f18 a h, privateBatchWrapper2_f19 a h, privateBatchWrapper2_f20 a h, privateBatchWrapper2_f21 a h, privateBatchWrapper2_f22 a h, privateBatchWrapper2_f23 a h, privateBatchWrapper2_f24 a h, privateBatchWrapper2_f25 a h, privateBatchWrapper2_f26 a h, privateBatchWrapper2_f27 a h, privateBatchWrapper2_f28 a h, privateBatchWrapper2_f29 a h, privateBatchWrapper2_f30 a h, privateBatchWrapper2_f31 a h⟩, ⟨privateBatchWrapper2_f32 a h, privateBatchWrapper2_f33 a h, privateBatchWrapper2_f34 a h, privateBatchWrapper2_f35 a h, privateBatchWrapper2_f36 a h, privateBatchWrapper2_f37 a h, privateBatchWrapper2_f38 a h, privateBatchWrapper2_f39 a h, privateBatchWrapper2_f40 a h, privateBatchWrapper2_f41 a h, privateBatchWrapper2_f42 a h, privateBatchWrapper2_f43 a h, privateBatchWrapper2_f44 a h, privateBatchWrapper2_f45 a h, privateBatchWrapper2_f46 a h, privateBatchWrapper2_f47 a h, privateBatchWrapper2_f48 a h, privateBatchWrapper2_f49 a h, privateBatchWrapper2_f50 a h, privateBatchWrapper2_f51 a h, privateBatchWrapper2_f52 a h, privateBatchWrapper2_f53 a h, privateBatchWrapper2_f54 a h, privateBatchWrapper2_f55 a h, privateBatchWrapper2_f56 a h, privateBatchWrapper2_f57 a h, privateBatchWrapper2_f58 a h, privateBatchWrapper2_f59 a h, privateBatchWrapper2_f60 a h, privateBatchWrapper2_f61 a h, privateBatchWrapper2_f62 a h, privateBatchWrapper2_f63 a h⟩, ⟨privateBatchWrapper2_f64 a h, privateBatchWrapper2_f65 a h, privateBatchWrapper2_f66 a h, privateBatchWrapper2_f67 a h, privateBatchWrapper2_f68 a h, privateBatchWrapper2_f69 a h, privateBatchWrapper2_f70 a h, privateBatchWrapper2_f71 a h, privateBatchWrapper2_f72 a h, privateBatchWrapper2_f73 a h, privateBatchWrapper2_f74 a h, privateBatchWrapper2_f75 a h, privateBatchWrapper2_f76 a h, privateBatchWrapper2_f77 a h, privateBatchWrapper2_f78 a h, privateBatchWrapper2_f79 a h, privateBatchWrapper2_f80 perm a h hp, privateBatchWrapper2_f81 perm a h hp, privateBatchWrapper2_f82 a h, privateBatchWrapper2_f83 a h, privateBatchWrapper2_f84 a h, privateBatchWrapper2_f85 a h, privateBatchWrapper2_f86 perm a h hp, privateBatchWrapper2_f87 perm a h hp, privateBatchWrapper2_f88 a h, privateBatchWrapper2_f89 a h, privateBatchWrapper2_f90 a h, privateBatchWrapper2_f91 a h⟩⟩

end Plonky2Spec.Generated
