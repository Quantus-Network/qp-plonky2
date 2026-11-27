/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PrivateBatchWrapper4`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge
import Plonky2Spec.Generated.PrivateBatchWrapper4.Fact

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
/-- Every satisfying assignment of `privateBatchWrapper4` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem privateBatchWrapper4_decode (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (IsEqual (a (.virt 5)) (a (.virt 11)) (a (.virt 84)) (a (.virt 85)) ∧
    IsEqual (a (.virt 6)) (a (.virt 11)) (a (.virt 86)) (a (.virt 87)) ∧
    IsEqual (a (.virt 7)) (a (.virt 11)) (a (.virt 88)) (a (.virt 89)) ∧
    IsEqual (a (.virt 8)) (a (.virt 11)) (a (.virt 90)) (a (.virt 91)) ∧
    a (.wire 3 11) = band (a (.virt 84)) (a (.virt 86)) ∧
    a (.wire 3 15) = band (a (.virt 88)) (a (.virt 90)) ∧
    a (.wire 3 19) = band (a (.wire 3 11)) (a (.wire 3 15)) ∧
    IsEqual (a (.virt 19)) (a (.virt 11)) (a (.virt 92)) (a (.virt 93)) ∧
    IsEqual (a (.virt 20)) (a (.virt 11)) (a (.virt 94)) (a (.virt 95)) ∧
    IsEqual (a (.virt 21)) (a (.virt 11)) (a (.virt 96)) (a (.virt 97)) ∧
    IsEqual (a (.virt 22)) (a (.virt 11)) (a (.virt 98)) (a (.virt 99)) ∧
    a (.wire 3 55) = band (a (.virt 92)) (a (.virt 94)) ∧
    a (.wire 3 59) = band (a (.virt 96)) (a (.virt 98)) ∧
    a (.wire 5 3) = band (a (.wire 3 55)) (a (.wire 3 59)) ∧
    IsEqual (a (.virt 30)) (a (.virt 11)) (a (.virt 100)) (a (.virt 101)) ∧
    IsEqual (a (.virt 31)) (a (.virt 11)) (a (.virt 102)) (a (.virt 103)) ∧
    IsEqual (a (.virt 32)) (a (.virt 11)) (a (.virt 104)) (a (.virt 105)) ∧
    IsEqual (a (.virt 33)) (a (.virt 11)) (a (.virt 106)) (a (.virt 107)) ∧
    a (.wire 5 39) = band (a (.virt 100)) (a (.virt 102)) ∧
    a (.wire 5 43) = band (a (.virt 104)) (a (.virt 106)) ∧
    a (.wire 5 47) = band (a (.wire 5 39)) (a (.wire 5 43)) ∧
    IsEqual (a (.virt 41)) (a (.virt 11)) (a (.virt 108)) (a (.virt 109)) ∧
    IsEqual (a (.virt 42)) (a (.virt 11)) (a (.virt 110)) (a (.virt 111)) ∧
    IsEqual (a (.virt 43)) (a (.virt 11)) (a (.virt 112)) (a (.virt 113)) ∧
    IsEqual (a (.virt 44)) (a (.virt 11)) (a (.virt 114)) (a (.virt 115)) ∧
    a (.wire 6 23) = band (a (.virt 108)) (a (.virt 110)) ∧
    a (.wire 6 27) = band (a (.virt 112)) (a (.virt 114)) ∧
    a (.wire 6 31) = band (a (.wire 6 23)) (a (.wire 6 27)) ∧
    a (.wire 7 11) = bnot (a (.wire 3 19)) ∧
    a (.virt 13) = bnot (a (.virt 11)) ∧
    a (.wire 7 11) = band (a (.wire 7 11)) (a (.virt 13)) ∧
    a (.wire 7 15) = bselect (a (.wire 7 11)) (a (.virt 5)) (a (.virt 11))) ∧
    (a (.wire 7 19) = bselect (a (.wire 7 11)) (a (.virt 6)) (a (.virt 11)) ∧
    a (.wire 7 23) = bselect (a (.wire 7 11)) (a (.virt 7)) (a (.virt 11)) ∧
    a (.wire 7 27) = bselect (a (.wire 7 11)) (a (.virt 8)) (a (.virt 11)) ∧
    a (.wire 7 31) = bselect (a (.wire 7 11)) (a (.virt 9)) (a (.virt 11)) ∧
    a (.wire 7 11) = bor (a (.virt 11)) (a (.wire 7 11)) ∧
    a (.wire 7 35) = bnot (a (.wire 5 3)) ∧
    a (.wire 7 39) = bnot (a (.wire 7 11)) ∧
    a (.wire 6 35) = band (a (.wire 7 35)) (a (.wire 7 39)) ∧
    a (.wire 7 47) = bselect (a (.wire 6 35)) (a (.virt 19)) (a (.wire 7 15)) ∧
    a (.wire 7 55) = bselect (a (.wire 6 35)) (a (.virt 20)) (a (.wire 7 19)) ∧
    a (.wire 8 3) = bselect (a (.wire 6 35)) (a (.virt 21)) (a (.wire 7 23)) ∧
    a (.wire 8 11) = bselect (a (.wire 6 35)) (a (.virt 22)) (a (.wire 7 27)) ∧
    a (.wire 8 19) = bselect (a (.wire 6 35)) (a (.virt 23)) (a (.wire 7 31)) ∧
    a (.wire 0 55) = bor (a (.wire 7 11)) (a (.wire 7 35)) ∧
    a (.wire 8 23) = bnot (a (.wire 5 47)) ∧
    a (.wire 8 27) = bnot (a (.wire 0 55)) ∧
    a (.wire 6 39) = band (a (.wire 8 23)) (a (.wire 8 27)) ∧
    a (.wire 8 35) = bselect (a (.wire 6 39)) (a (.virt 30)) (a (.wire 7 47)) ∧
    a (.wire 8 43) = bselect (a (.wire 6 39)) (a (.virt 31)) (a (.wire 7 55)) ∧
    a (.wire 8 51) = bselect (a (.wire 6 39)) (a (.virt 32)) (a (.wire 8 3)) ∧
    a (.wire 8 59) = bselect (a (.wire 6 39)) (a (.virt 33)) (a (.wire 8 11)) ∧
    a (.wire 10 7) = bselect (a (.wire 6 39)) (a (.virt 34)) (a (.wire 8 19)) ∧
    a (.wire 0 59) = bor (a (.wire 0 55)) (a (.wire 8 23)) ∧
    a (.wire 10 11) = bnot (a (.wire 6 31)) ∧
    a (.wire 10 15) = bnot (a (.wire 0 59)) ∧
    a (.wire 6 43) = band (a (.wire 10 11)) (a (.wire 10 15)) ∧
    a (.wire 10 23) = bselect (a (.wire 6 43)) (a (.virt 41)) (a (.wire 8 35)) ∧
    a (.wire 10 31) = bselect (a (.wire 6 43)) (a (.virt 42)) (a (.wire 8 43)) ∧
    a (.wire 10 39) = bselect (a (.wire 6 43)) (a (.virt 43)) (a (.wire 8 51)) ∧
    a (.wire 10 47) = bselect (a (.wire 6 43)) (a (.virt 44)) (a (.wire 8 59)) ∧
    a (.wire 10 55) = bselect (a (.wire 6 43)) (a (.virt 45)) (a (.wire 10 7)) ∧
    a (.wire 11 3) = bor (a (.wire 0 59)) (a (.wire 10 11))) ∧
    (IsEqual (a (.virt 5)) (a (.wire 10 23)) (a (.virt 116)) (a (.virt 117)) ∧
    IsEqual (a (.virt 6)) (a (.wire 10 31)) (a (.virt 118)) (a (.virt 119)) ∧
    IsEqual (a (.virt 7)) (a (.wire 10 39)) (a (.virt 120)) (a (.virt 121)) ∧
    IsEqual (a (.virt 8)) (a (.wire 10 47)) (a (.virt 122)) (a (.virt 123)) ∧
    a (.wire 13 19) = band (a (.virt 116)) (a (.virt 118)) ∧
    a (.wire 13 23) = band (a (.virt 120)) (a (.virt 122)) ∧
    a (.wire 13 27) = band (a (.wire 13 19)) (a (.wire 13 23)) ∧
    a (.wire 11 7) = bor (a (.wire 3 19)) (a (.wire 13 27)) ∧
    a (.wire 11 7) = a (.virt 13) ∧
    a (.virt 0) = a (.virt 0) ∧
    IsEqual (a (.virt 19)) (a (.wire 10 23)) (a (.virt 124)) (a (.virt 125)) ∧
    IsEqual (a (.virt 20)) (a (.wire 10 31)) (a (.virt 126)) (a (.virt 127)) ∧
    IsEqual (a (.virt 21)) (a (.wire 10 39)) (a (.virt 128)) (a (.virt 129)) ∧
    IsEqual (a (.virt 22)) (a (.wire 10 47)) (a (.virt 130)) (a (.virt 131)) ∧
    a (.wire 15 3) = band (a (.virt 124)) (a (.virt 126)) ∧
    a (.wire 15 7) = band (a (.virt 128)) (a (.virt 130)) ∧
    a (.wire 15 11) = band (a (.wire 15 3)) (a (.wire 15 7)) ∧
    a (.wire 11 11) = bor (a (.wire 5 3)) (a (.wire 15 11)) ∧
    a (.wire 11 11) = a (.virt 13) ∧
    a (.virt 14) = a (.virt 0) ∧
    IsEqual (a (.virt 30)) (a (.wire 10 23)) (a (.virt 132)) (a (.virt 133)) ∧
    IsEqual (a (.virt 31)) (a (.wire 10 31)) (a (.virt 134)) (a (.virt 135)) ∧
    IsEqual (a (.virt 32)) (a (.wire 10 39)) (a (.virt 136)) (a (.virt 137)) ∧
    IsEqual (a (.virt 33)) (a (.wire 10 47)) (a (.virt 138)) (a (.virt 139)) ∧
    a (.wire 15 47) = band (a (.virt 132)) (a (.virt 134)) ∧
    a (.wire 15 51) = band (a (.virt 136)) (a (.virt 138)) ∧
    a (.wire 15 55) = band (a (.wire 15 47)) (a (.wire 15 51)) ∧
    a (.wire 11 15) = bor (a (.wire 5 47)) (a (.wire 15 55)) ∧
    a (.wire 11 15) = a (.virt 13) ∧
    a (.virt 25) = a (.virt 0) ∧
    IsEqual (a (.virt 41)) (a (.wire 10 23)) (a (.virt 140)) (a (.virt 141)) ∧
    IsEqual (a (.virt 42)) (a (.wire 10 31)) (a (.virt 142)) (a (.virt 143))) ∧
    (IsEqual (a (.virt 43)) (a (.wire 10 39)) (a (.virt 144)) (a (.virt 145)) ∧
    IsEqual (a (.virt 44)) (a (.wire 10 47)) (a (.virt 146)) (a (.virt 147)) ∧
    a (.wire 17 31) = band (a (.virt 140)) (a (.virt 142)) ∧
    a (.wire 17 35) = band (a (.virt 144)) (a (.virt 146)) ∧
    a (.wire 17 39) = band (a (.wire 17 31)) (a (.wire 17 35)) ∧
    a (.wire 11 19) = bor (a (.wire 6 31)) (a (.wire 17 39)) ∧
    a (.wire 11 19) = a (.virt 13) ∧
    a (.virt 36) = a (.virt 0) ∧
    a (.wire 18 15) = bselect (a (.wire 3 19)) (a (.virt 11)) (a (.virt 10)) ∧
    a (.wire 18 15) = a (.virt 11) + a (.wire 18 15) ∧
    a (.wire 18 23) = bselect (a (.wire 5 3)) (a (.virt 11)) (a (.virt 24)) ∧
    a (.wire 11 23) = a (.wire 18 15) + a (.wire 18 23) ∧
    a (.wire 18 31) = bselect (a (.wire 5 47)) (a (.virt 11)) (a (.virt 35)) ∧
    a (.wire 11 27) = a (.wire 11 23) + a (.wire 18 31) ∧
    a (.wire 18 39) = bselect (a (.wire 6 31)) (a (.virt 11)) (a (.virt 46)) ∧
    a (.wire 11 31) = a (.wire 11 27) + a (.wire 18 39) ∧
    rangeCheck (a (.virt 52)) 32 ∧
    a (.virt 52) = a (.virt 11) + a (.virt 52) ∧
    rangeCheck (a (.virt 57)) 32 ∧
    a (.wire 11 35) = a (.virt 52) + a (.virt 57) ∧
    rangeCheck (a (.virt 62)) 32 ∧
    a (.wire 11 39) = a (.wire 11 35) + a (.virt 62) ∧
    rangeCheck (a (.virt 67)) 32 ∧
    a (.wire 11 43) = a (.wire 11 39) + a (.virt 67) ∧
    rangeCheck (a (.virt 47)) 14 ∧
    a (.wire 18 43) = a (.virt 148) - a (.virt 47) ∧
    rangeCheck (a (.wire 18 43)) 14 ∧
    a (.wire 17 43) = a (.wire 11 43) * a (.virt 148) ∧
    a (.wire 17 47) = a (.wire 11 31) * a (.wire 18 43) ∧
    a (.wire 18 47) = a (.wire 17 47) - a (.wire 17 43) ∧
    rangeCheck (a (.wire 18 47)) 52 ∧
    a (.wire 7 11) = bnot (a (.wire 3 19))) ∧
    (a (.wire 7 35) = bnot (a (.wire 5 3)) ∧
    a (.wire 17 51) = band (a (.wire 7 11)) (a (.wire 7 35)) ∧
    IsEqual (a (.virt 1)) (a (.virt 15)) (a (.virt 149)) (a (.virt 150)) ∧
    IsEqual (a (.virt 2)) (a (.virt 16)) (a (.virt 151)) (a (.virt 152)) ∧
    IsEqual (a (.virt 3)) (a (.virt 17)) (a (.virt 153)) (a (.virt 154)) ∧
    IsEqual (a (.virt 4)) (a (.virt 18)) (a (.virt 155)) (a (.virt 156)) ∧
    a (.wire 27 27) = band (a (.virt 149)) (a (.virt 151)) ∧
    a (.wire 27 31) = band (a (.virt 153)) (a (.virt 155)) ∧
    a (.wire 27 35) = band (a (.wire 27 27)) (a (.wire 27 31)) ∧
    a (.wire 27 39) = band (a (.wire 17 51)) (a (.wire 27 35)) ∧
    a (.wire 27 39) = a (.virt 11) ∧
    a (.wire 8 23) = bnot (a (.wire 5 47)) ∧
    a (.wire 27 43) = band (a (.wire 7 11)) (a (.wire 8 23)) ∧
    IsEqual (a (.virt 1)) (a (.virt 26)) (a (.virt 157)) (a (.virt 158)) ∧
    IsEqual (a (.virt 2)) (a (.virt 27)) (a (.virt 159)) (a (.virt 160)) ∧
    IsEqual (a (.virt 3)) (a (.virt 28)) (a (.virt 161)) (a (.virt 162)) ∧
    IsEqual (a (.virt 4)) (a (.virt 29)) (a (.virt 163)) (a (.virt 164)) ∧
    a (.wire 29 19) = band (a (.virt 157)) (a (.virt 159)) ∧
    a (.wire 29 23) = band (a (.virt 161)) (a (.virt 163)) ∧
    a (.wire 29 27) = band (a (.wire 29 19)) (a (.wire 29 23)) ∧
    a (.wire 29 31) = band (a (.wire 27 43)) (a (.wire 29 27)) ∧
    a (.wire 29 31) = a (.virt 11) ∧
    a (.wire 10 11) = bnot (a (.wire 6 31)) ∧
    a (.wire 29 35) = band (a (.wire 7 11)) (a (.wire 10 11)) ∧
    IsEqual (a (.virt 1)) (a (.virt 37)) (a (.virt 165)) (a (.virt 166)) ∧
    IsEqual (a (.virt 2)) (a (.virt 38)) (a (.virt 167)) (a (.virt 168)) ∧
    IsEqual (a (.virt 3)) (a (.virt 39)) (a (.virt 169)) (a (.virt 170)) ∧
    IsEqual (a (.virt 4)) (a (.virt 40)) (a (.virt 171)) (a (.virt 172)) ∧
    a (.wire 31 11) = band (a (.virt 165)) (a (.virt 167)) ∧
    a (.wire 31 15) = band (a (.virt 169)) (a (.virt 171)) ∧
    a (.wire 31 19) = band (a (.wire 31 11)) (a (.wire 31 15)) ∧
    a (.wire 31 23) = band (a (.wire 29 35)) (a (.wire 31 19))) ∧
    (a (.wire 31 23) = a (.virt 11) ∧
    a (.wire 7 35) = bnot (a (.wire 5 3)) ∧
    a (.wire 8 23) = bnot (a (.wire 5 47)) ∧
    a (.wire 31 27) = band (a (.wire 7 35)) (a (.wire 8 23)) ∧
    IsEqual (a (.virt 15)) (a (.virt 26)) (a (.virt 173)) (a (.virt 174)) ∧
    IsEqual (a (.virt 16)) (a (.virt 27)) (a (.virt 175)) (a (.virt 176)) ∧
    IsEqual (a (.virt 17)) (a (.virt 28)) (a (.virt 177)) (a (.virt 178)) ∧
    IsEqual (a (.virt 18)) (a (.virt 29)) (a (.virt 179)) (a (.virt 180)) ∧
    a (.wire 32 3) = band (a (.virt 173)) (a (.virt 175)) ∧
    a (.wire 32 7) = band (a (.virt 177)) (a (.virt 179)) ∧
    a (.wire 32 11) = band (a (.wire 32 3)) (a (.wire 32 7)) ∧
    a (.wire 32 15) = band (a (.wire 31 27)) (a (.wire 32 11)) ∧
    a (.wire 32 15) = a (.virt 11) ∧
    a (.wire 10 11) = bnot (a (.wire 6 31)) ∧
    a (.wire 32 19) = band (a (.wire 7 35)) (a (.wire 10 11)) ∧
    IsEqual (a (.virt 15)) (a (.virt 37)) (a (.virt 181)) (a (.virt 182)) ∧
    IsEqual (a (.virt 16)) (a (.virt 38)) (a (.virt 183)) (a (.virt 184)) ∧
    IsEqual (a (.virt 17)) (a (.virt 39)) (a (.virt 185)) (a (.virt 186)) ∧
    IsEqual (a (.virt 18)) (a (.virt 40)) (a (.virt 187)) (a (.virt 188)) ∧
    a (.wire 32 55) = band (a (.virt 181)) (a (.virt 183)) ∧
    a (.wire 32 59) = band (a (.virt 185)) (a (.virt 187)) ∧
    a (.wire 34 3) = band (a (.wire 32 55)) (a (.wire 32 59)) ∧
    a (.wire 34 7) = band (a (.wire 32 19)) (a (.wire 34 3)) ∧
    a (.wire 34 7) = a (.virt 11) ∧
    a (.wire 8 23) = bnot (a (.wire 5 47)) ∧
    a (.wire 10 11) = bnot (a (.wire 6 31)) ∧
    a (.wire 34 11) = band (a (.wire 8 23)) (a (.wire 10 11)) ∧
    IsEqual (a (.virt 26)) (a (.virt 37)) (a (.virt 189)) (a (.virt 190)) ∧
    IsEqual (a (.virt 27)) (a (.virt 38)) (a (.virt 191)) (a (.virt 192)) ∧
    IsEqual (a (.virt 28)) (a (.virt 39)) (a (.virt 193)) (a (.virt 194)) ∧
    IsEqual (a (.virt 29)) (a (.virt 40)) (a (.virt 195)) (a (.virt 196)) ∧
    a (.wire 34 47) = band (a (.virt 189)) (a (.virt 191))) ∧
    (a (.wire 34 51) = band (a (.virt 193)) (a (.virt 195)) ∧
    a (.wire 34 55) = band (a (.wire 34 47)) (a (.wire 34 51)) ∧
    a (.wire 34 59) = band (a (.wire 34 11)) (a (.wire 34 55)) ∧
    a (.wire 34 59) = a (.virt 11) ∧
    a (.wire 10 11) = bnot (a (.wire 6 31)) ∧
    (a (.wire 36 12) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 0 ∧ a (.wire 36 13) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 1 ∧ a (.wire 36 14) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 2 ∧ a (.wire 36 15) = spongeHash perm [a (.virt 68), a (.virt 69), a (.virt 70), a (.virt 71)] 3) ∧
    (a (.wire 37 12) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 0 ∧ a (.wire 37 13) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 1 ∧ a (.wire 37 14) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 2 ∧ a (.wire 37 15) = spongeHash perm [a (.wire 36 12), a (.wire 36 13), a (.wire 36 14), a (.wire 36 15)] 3) ∧
    a (.wire 35 43) = bselect (a (.wire 3 19)) (a (.wire 37 12)) (a (.virt 1)) ∧
    a (.wire 35 51) = bselect (a (.wire 3 19)) (a (.wire 37 13)) (a (.virt 2)) ∧
    a (.wire 35 59) = bselect (a (.wire 3 19)) (a (.wire 37 14)) (a (.virt 3)) ∧
    a (.wire 38 7) = bselect (a (.wire 3 19)) (a (.wire 37 15)) (a (.virt 4)) ∧
    (a (.wire 39 12) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 0 ∧ a (.wire 39 13) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 1 ∧ a (.wire 39 14) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 2 ∧ a (.wire 39 15) = spongeHash perm [a (.virt 72), a (.virt 73), a (.virt 74), a (.virt 75)] 3) ∧
    (a (.wire 40 12) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 0 ∧ a (.wire 40 13) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 1 ∧ a (.wire 40 14) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 2 ∧ a (.wire 40 15) = spongeHash perm [a (.wire 39 12), a (.wire 39 13), a (.wire 39 14), a (.wire 39 15)] 3) ∧
    a (.wire 38 15) = bselect (a (.wire 5 3)) (a (.wire 40 12)) (a (.virt 15)) ∧
    a (.wire 38 23) = bselect (a (.wire 5 3)) (a (.wire 40 13)) (a (.virt 16)) ∧
    a (.wire 38 31) = bselect (a (.wire 5 3)) (a (.wire 40 14)) (a (.virt 17)) ∧
    a (.wire 38 39) = bselect (a (.wire 5 3)) (a (.wire 40 15)) (a (.virt 18)) ∧
    (a (.wire 41 12) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 0 ∧ a (.wire 41 13) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 1 ∧ a (.wire 41 14) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 2 ∧ a (.wire 41 15) = spongeHash perm [a (.virt 76), a (.virt 77), a (.virt 78), a (.virt 79)] 3) ∧
    (a (.wire 42 12) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 0 ∧ a (.wire 42 13) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 1 ∧ a (.wire 42 14) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 2 ∧ a (.wire 42 15) = spongeHash perm [a (.wire 41 12), a (.wire 41 13), a (.wire 41 14), a (.wire 41 15)] 3) ∧
    a (.wire 38 47) = bselect (a (.wire 5 47)) (a (.wire 42 12)) (a (.virt 26)) ∧
    a (.wire 38 55) = bselect (a (.wire 5 47)) (a (.wire 42 13)) (a (.virt 27)) ∧
    a (.wire 43 3) = bselect (a (.wire 5 47)) (a (.wire 42 14)) (a (.virt 28)) ∧
    a (.wire 43 11) = bselect (a (.wire 5 47)) (a (.wire 42 15)) (a (.virt 29)) ∧
    (a (.wire 44 12) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 0 ∧ a (.wire 44 13) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 1 ∧ a (.wire 44 14) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 2 ∧ a (.wire 44 15) = spongeHash perm [a (.virt 80), a (.virt 81), a (.virt 82), a (.virt 83)] 3) ∧
    (a (.wire 45 12) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 0 ∧ a (.wire 45 13) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 1 ∧ a (.wire 45 14) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 2 ∧ a (.wire 45 15) = spongeHash perm [a (.wire 44 12), a (.wire 44 13), a (.wire 44 14), a (.wire 44 15)] 3) ∧
    a (.wire 43 19) = bselect (a (.wire 6 31)) (a (.wire 45 12)) (a (.virt 37)) ∧
    a (.wire 43 27) = bselect (a (.wire 6 31)) (a (.wire 45 13)) (a (.virt 38)) ∧
    a (.wire 43 35) = bselect (a (.wire 6 31)) (a (.wire 45 14)) (a (.virt 39)) ∧
    a (.wire 43 43) = bselect (a (.wire 6 31)) (a (.wire 45 15)) (a (.virt 40))) :=
  ⟨⟨privateBatchWrapper4_f0 a h, privateBatchWrapper4_f1 a h, privateBatchWrapper4_f2 a h, privateBatchWrapper4_f3 a h, privateBatchWrapper4_f4 a h, privateBatchWrapper4_f5 a h, privateBatchWrapper4_f6 a h, privateBatchWrapper4_f7 a h, privateBatchWrapper4_f8 a h, privateBatchWrapper4_f9 a h, privateBatchWrapper4_f10 a h, privateBatchWrapper4_f11 a h, privateBatchWrapper4_f12 a h, privateBatchWrapper4_f13 a h, privateBatchWrapper4_f14 a h, privateBatchWrapper4_f15 a h, privateBatchWrapper4_f16 a h, privateBatchWrapper4_f17 a h, privateBatchWrapper4_f18 a h, privateBatchWrapper4_f19 a h, privateBatchWrapper4_f20 a h, privateBatchWrapper4_f21 a h, privateBatchWrapper4_f22 a h, privateBatchWrapper4_f23 a h, privateBatchWrapper4_f24 a h, privateBatchWrapper4_f25 a h, privateBatchWrapper4_f26 a h, privateBatchWrapper4_f27 a h, privateBatchWrapper4_f28 a h, privateBatchWrapper4_f29 a h, privateBatchWrapper4_f30 a h, privateBatchWrapper4_f31 a h⟩, ⟨privateBatchWrapper4_f32 a h, privateBatchWrapper4_f33 a h, privateBatchWrapper4_f34 a h, privateBatchWrapper4_f35 a h, privateBatchWrapper4_f36 a h, privateBatchWrapper4_f37 a h, privateBatchWrapper4_f38 a h, privateBatchWrapper4_f39 a h, privateBatchWrapper4_f40 a h, privateBatchWrapper4_f41 a h, privateBatchWrapper4_f42 a h, privateBatchWrapper4_f43 a h, privateBatchWrapper4_f44 a h, privateBatchWrapper4_f45 a h, privateBatchWrapper4_f46 a h, privateBatchWrapper4_f47 a h, privateBatchWrapper4_f48 a h, privateBatchWrapper4_f49 a h, privateBatchWrapper4_f50 a h, privateBatchWrapper4_f51 a h, privateBatchWrapper4_f52 a h, privateBatchWrapper4_f53 a h, privateBatchWrapper4_f54 a h, privateBatchWrapper4_f55 a h, privateBatchWrapper4_f56 a h, privateBatchWrapper4_f57 a h, privateBatchWrapper4_f58 a h, privateBatchWrapper4_f59 a h, privateBatchWrapper4_f60 a h, privateBatchWrapper4_f61 a h, privateBatchWrapper4_f62 a h, privateBatchWrapper4_f63 a h⟩, ⟨privateBatchWrapper4_f64 a h, privateBatchWrapper4_f65 a h, privateBatchWrapper4_f66 a h, privateBatchWrapper4_f67 a h, privateBatchWrapper4_f68 a h, privateBatchWrapper4_f69 a h, privateBatchWrapper4_f70 a h, privateBatchWrapper4_f71 a h, privateBatchWrapper4_f72 a h, privateBatchWrapper4_f73 a h, privateBatchWrapper4_f74 a h, privateBatchWrapper4_f75 a h, privateBatchWrapper4_f76 a h, privateBatchWrapper4_f77 a h, privateBatchWrapper4_f78 a h, privateBatchWrapper4_f79 a h, privateBatchWrapper4_f80 a h, privateBatchWrapper4_f81 a h, privateBatchWrapper4_f82 a h, privateBatchWrapper4_f83 a h, privateBatchWrapper4_f84 a h, privateBatchWrapper4_f85 a h, privateBatchWrapper4_f86 a h, privateBatchWrapper4_f87 a h, privateBatchWrapper4_f88 a h, privateBatchWrapper4_f89 a h, privateBatchWrapper4_f90 a h, privateBatchWrapper4_f91 a h, privateBatchWrapper4_f92 a h, privateBatchWrapper4_f93 a h, privateBatchWrapper4_f94 a h, privateBatchWrapper4_f95 a h⟩, ⟨privateBatchWrapper4_f96 a h, privateBatchWrapper4_f97 a h, privateBatchWrapper4_f98 a h, privateBatchWrapper4_f99 a h, privateBatchWrapper4_f100 a h, privateBatchWrapper4_f101 a h, privateBatchWrapper4_f102 a h, privateBatchWrapper4_f103 a h, privateBatchWrapper4_f104 a h, privateBatchWrapper4_f105 a h, privateBatchWrapper4_f106 a h, privateBatchWrapper4_f107 a h, privateBatchWrapper4_f108 a h, privateBatchWrapper4_f109 a h, privateBatchWrapper4_f110 a h, privateBatchWrapper4_f111 a h, privateBatchWrapper4_f112 a h, privateBatchWrapper4_f113 a h, privateBatchWrapper4_f114 a h, privateBatchWrapper4_f115 a h, privateBatchWrapper4_f116 a h, privateBatchWrapper4_f117 a h, privateBatchWrapper4_f118 a h, privateBatchWrapper4_f119 a h, privateBatchWrapper4_f120 a h, privateBatchWrapper4_f121 a h, privateBatchWrapper4_f122 a h, privateBatchWrapper4_f123 a h, privateBatchWrapper4_f124 a h, privateBatchWrapper4_f125 a h, privateBatchWrapper4_f126 a h, privateBatchWrapper4_f127 a h⟩, ⟨privateBatchWrapper4_f128 a h, privateBatchWrapper4_f129 a h, privateBatchWrapper4_f130 a h, privateBatchWrapper4_f131 a h, privateBatchWrapper4_f132 a h, privateBatchWrapper4_f133 a h, privateBatchWrapper4_f134 a h, privateBatchWrapper4_f135 a h, privateBatchWrapper4_f136 a h, privateBatchWrapper4_f137 a h, privateBatchWrapper4_f138 a h, privateBatchWrapper4_f139 a h, privateBatchWrapper4_f140 a h, privateBatchWrapper4_f141 a h, privateBatchWrapper4_f142 a h, privateBatchWrapper4_f143 a h, privateBatchWrapper4_f144 a h, privateBatchWrapper4_f145 a h, privateBatchWrapper4_f146 a h, privateBatchWrapper4_f147 a h, privateBatchWrapper4_f148 a h, privateBatchWrapper4_f149 a h, privateBatchWrapper4_f150 a h, privateBatchWrapper4_f151 a h, privateBatchWrapper4_f152 a h, privateBatchWrapper4_f153 a h, privateBatchWrapper4_f154 a h, privateBatchWrapper4_f155 a h, privateBatchWrapper4_f156 a h, privateBatchWrapper4_f157 a h, privateBatchWrapper4_f158 a h, privateBatchWrapper4_f159 a h⟩, ⟨privateBatchWrapper4_f160 a h, privateBatchWrapper4_f161 a h, privateBatchWrapper4_f162 a h, privateBatchWrapper4_f163 a h, privateBatchWrapper4_f164 a h, privateBatchWrapper4_f165 a h, privateBatchWrapper4_f166 a h, privateBatchWrapper4_f167 a h, privateBatchWrapper4_f168 a h, privateBatchWrapper4_f169 a h, privateBatchWrapper4_f170 a h, privateBatchWrapper4_f171 a h, privateBatchWrapper4_f172 a h, privateBatchWrapper4_f173 a h, privateBatchWrapper4_f174 a h, privateBatchWrapper4_f175 a h, privateBatchWrapper4_f176 a h, privateBatchWrapper4_f177 a h, privateBatchWrapper4_f178 a h, privateBatchWrapper4_f179 a h, privateBatchWrapper4_f180 a h, privateBatchWrapper4_f181 a h, privateBatchWrapper4_f182 a h, privateBatchWrapper4_f183 a h, privateBatchWrapper4_f184 a h, privateBatchWrapper4_f185 a h, privateBatchWrapper4_f186 a h, privateBatchWrapper4_f187 a h, privateBatchWrapper4_f188 a h, privateBatchWrapper4_f189 a h, privateBatchWrapper4_f190 a h, privateBatchWrapper4_f191 a h⟩, ⟨privateBatchWrapper4_f192 a h, privateBatchWrapper4_f193 a h, privateBatchWrapper4_f194 a h, privateBatchWrapper4_f195 a h, privateBatchWrapper4_f196 a h, privateBatchWrapper4_f197 perm a h hp, privateBatchWrapper4_f198 perm a h hp, privateBatchWrapper4_f199 a h, privateBatchWrapper4_f200 a h, privateBatchWrapper4_f201 a h, privateBatchWrapper4_f202 a h, privateBatchWrapper4_f203 perm a h hp, privateBatchWrapper4_f204 perm a h hp, privateBatchWrapper4_f205 a h, privateBatchWrapper4_f206 a h, privateBatchWrapper4_f207 a h, privateBatchWrapper4_f208 a h, privateBatchWrapper4_f209 perm a h hp, privateBatchWrapper4_f210 perm a h hp, privateBatchWrapper4_f211 a h, privateBatchWrapper4_f212 a h, privateBatchWrapper4_f213 a h, privateBatchWrapper4_f214 a h, privateBatchWrapper4_f215 perm a h hp, privateBatchWrapper4_f216 perm a h hp, privateBatchWrapper4_f217 a h, privateBatchWrapper4_f218 a h, privateBatchWrapper4_f219 a h, privateBatchWrapper4_f220 a h⟩⟩

end Plonky2Spec.Generated
