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
import Plonky2Spec.Generated.LeafCircuit.Decode

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
/-- Every satisfying assignment of `leafCircuit` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem leafCircuit_decode (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    ((IsBool (a (.virt 13)) ∧
    (a (.wire 1 12) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 0 ∧ a (.wire 1 13) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 1 ∧ a (.wire 1 14) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 2 ∧ a (.wire 1 15) = spongeHash perm [a (.virt 290), a (.virt 291), a (.virt 292), a (.virt 237), a (.virt 238), a (.virt 239), a (.virt 240)] 3) ∧
    (a (.wire 2 12) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 0 ∧ a (.wire 2 13) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 1 ∧ a (.wire 2 14) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 2 ∧ a (.wire 2 15) = spongeHash perm [a (.wire 1 12), a (.wire 1 13), a (.wire 1 14), a (.wire 1 15)] 3) ∧
    a (.wire 2 12) = a (.virt 233) ∧
    a (.wire 2 13) = a (.virt 234) ∧
    a (.wire 2 14) = a (.virt 235) ∧
    a (.wire 2 15) = a (.virt 236) ∧
    rangeCheck (a (.virt 5)) 32 ∧
    rangeCheck (a (.virt 6)) 32 ∧
    rangeCheck (a (.virt 0)) 32 ∧
    rangeCheck (a (.virt 7)) 32 ∧
    (a (.wire 9 12) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 0 ∧ a (.wire 9 13) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 1 ∧ a (.wire 9 14) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 2 ∧ a (.wire 9 15) = spongeHash perm [a (.virt 1), a (.virt 2), a (.virt 3), a (.virt 4), a (.virt 5), a (.virt 6), a (.virt 0), a (.virt 7)] 3) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 10 1), a (.wire 10 2), a (.wire 10 3), a (.wire 10 4), a (.wire 10 5)] ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 10 5)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.virt 292)) ∧
    a (.virt 14) = bor (a (.virt 14)) (a (.virt 14)) ∧
    a (.wire 10 5) = a (.virt 292) * a (.wire 10 5) ∧
    a (.wire 11 3) = a (.virt 294) * a (.wire 10 5) ∧
    a (.wire 8 7) = a (.virt 292) + a (.wire 10 5) ∧
    a (.wire 0 7) = a (.wire 8 7) - a (.wire 11 3) ∧
    a (.wire 0 11) = bnot (a (.wire 0 7)) ∧
    a (.wire 0 11) = band (a (.virt 292)) (a (.wire 0 11)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 10 4) = band (a (.virt 292)) (a (.wire 10 4)) ∧
    a (.wire 11 7) = band (a (.wire 10 4)) (a (.wire 0 11)) ∧
    a (.wire 11 7) = bor (a (.virt 14)) (a (.wire 11 7)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 10 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 10 4) = a (.virt 14) + a (.wire 10 4) ∧
    a (.wire 10 4) = a (.wire 10 4) - a (.virt 14) ∧
    a (.wire 0 15) = bnot (a (.wire 10 4))) ∧
    (a (.wire 11 11) = band (a (.wire 0 11)) (a (.wire 0 15)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 10 3) = band (a (.virt 292)) (a (.wire 10 3)) ∧
    a (.wire 11 15) = band (a (.wire 10 3)) (a (.wire 11 11)) ∧
    a (.wire 8 11) = bor (a (.wire 11 7)) (a (.wire 11 15)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 10 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 10 3) = a (.virt 14) + a (.wire 10 3) ∧
    a (.wire 10 3) = a (.wire 10 3) - a (.virt 14) ∧
    a (.wire 0 19) = bnot (a (.wire 10 3)) ∧
    a (.wire 11 19) = band (a (.wire 11 11)) (a (.wire 0 19)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 10 2) = band (a (.virt 292)) (a (.wire 10 2)) ∧
    a (.wire 11 23) = band (a (.wire 10 2)) (a (.wire 11 19)) ∧
    a (.wire 8 15) = bor (a (.wire 8 11)) (a (.wire 11 23)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 10 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 10 2) = a (.virt 14) + a (.wire 10 2) ∧
    a (.wire 10 2) = a (.wire 10 2) - a (.virt 14) ∧
    a (.wire 0 23) = bnot (a (.wire 10 2)) ∧
    a (.wire 11 27) = band (a (.wire 11 19)) (a (.wire 0 23)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 10 1) = band (a (.virt 292)) (a (.wire 10 1)) ∧
    a (.wire 11 31) = band (a (.wire 10 1)) (a (.wire 11 27)) ∧
    a (.wire 8 19) = bor (a (.wire 8 15)) (a (.wire 11 31)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 10 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 10 1) = a (.virt 14) + a (.wire 10 1) ∧
    a (.wire 10 1) = a (.wire 10 1) - a (.virt 14) ∧
    a (.wire 0 27) = bnot (a (.wire 10 1)) ∧
    a (.wire 11 35) = band (a (.wire 11 27)) (a (.wire 0 27)) ∧
    a (.wire 8 19) = a (.virt 14)) ∧
    (BaseSum 2 (a (.virt 12)) [a (.wire 13 1), a (.wire 13 2), a (.wire 13 3), a (.wire 13 4), a (.wire 13 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 13 5) = band (a (.virt 292)) (a (.wire 13 5)) ∧
    a (.wire 13 5) = band (a (.wire 13 5)) (a (.virt 292)) ∧
    a (.wire 13 5) = bor (a (.virt 14)) (a (.wire 13 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 13 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 13 5) = a (.virt 14) + a (.wire 13 5) ∧
    a (.wire 13 5) = a (.wire 13 5) - a (.virt 14) ∧
    a (.wire 0 31) = bnot (a (.wire 13 5)) ∧
    a (.wire 0 31) = band (a (.virt 292)) (a (.wire 0 31)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 13 4) = band (a (.virt 292)) (a (.wire 13 4)) ∧
    a (.wire 11 39) = band (a (.wire 13 4)) (a (.wire 0 31)) ∧
    a (.wire 8 23) = bor (a (.wire 13 5)) (a (.wire 11 39)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 13 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 13 4) = a (.virt 14) + a (.wire 13 4) ∧
    a (.wire 13 4) = a (.wire 13 4) - a (.virt 14) ∧
    a (.wire 0 35) = bnot (a (.wire 13 4)) ∧
    a (.wire 11 43) = band (a (.wire 0 31)) (a (.wire 0 35)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 13 3) = band (a (.virt 292)) (a (.wire 13 3)) ∧
    a (.wire 11 47) = band (a (.wire 13 3)) (a (.wire 11 43)) ∧
    a (.wire 8 27) = bor (a (.wire 8 23)) (a (.wire 11 47)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 13 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 13 3) = a (.virt 14) + a (.wire 13 3) ∧
    a (.wire 13 3) = a (.wire 13 3) - a (.virt 14) ∧
    a (.wire 0 39) = bnot (a (.wire 13 3)) ∧
    a (.wire 11 51) = band (a (.wire 11 43)) (a (.wire 0 39)) ∧
    a (.virt 292) = bnot (a (.virt 14))) ∧
    (a (.wire 13 2) = band (a (.virt 292)) (a (.wire 13 2)) ∧
    a (.wire 11 55) = band (a (.wire 13 2)) (a (.wire 11 51)) ∧
    a (.wire 8 31) = bor (a (.wire 8 27)) (a (.wire 11 55)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 13 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 13 2) = a (.virt 14) + a (.wire 13 2) ∧
    a (.wire 13 2) = a (.wire 13 2) - a (.virt 14) ∧
    a (.wire 0 43) = bnot (a (.wire 13 2)) ∧
    a (.wire 11 59) = band (a (.wire 11 51)) (a (.wire 0 43)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 13 1) = band (a (.virt 292)) (a (.wire 13 1)) ∧
    a (.wire 14 3) = band (a (.wire 13 1)) (a (.wire 11 59)) ∧
    a (.wire 8 35) = bor (a (.wire 8 31)) (a (.wire 14 3)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 13 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 13 1) = a (.virt 14) + a (.wire 13 1) ∧
    a (.wire 13 1) = a (.wire 13 1) - a (.virt 14) ∧
    a (.wire 0 47) = bnot (a (.wire 13 1)) ∧
    a (.wire 14 7) = band (a (.wire 11 59)) (a (.wire 0 47)) ∧
    rangeCheck (a (.virt 207)) 2 ∧
    IsEqual (a (.virt 207)) (a (.virt 14)) (a (.virt 296)) (a (.virt 297)) ∧
    IsEqual (a (.virt 207)) (a (.virt 292)) (a (.virt 298)) (a (.virt 299)) ∧
    IsEqual (a (.virt 207)) (a (.virt 294)) (a (.virt 300)) (a (.virt 301)) ∧
    IsEqual (a (.virt 207)) (a (.virt 295)) (a (.virt 302)) (a (.virt 303)) ∧
    a (.wire 16 39) = bselect (a (.virt 296)) (a (.wire 9 12)) (a (.virt 15)) ∧
    a (.wire 16 47) = bselect (a (.virt 296)) (a (.wire 9 13)) (a (.virt 16)) ∧
    a (.wire 16 55) = bselect (a (.virt 296)) (a (.wire 9 14)) (a (.virt 17)) ∧
    a (.wire 17 3) = bselect (a (.virt 296)) (a (.wire 9 15)) (a (.virt 18)) ∧
    a (.wire 17 11) = bselect (a (.virt 296)) (a (.virt 15)) (a (.virt 19)) ∧
    a (.wire 17 19) = bselect (a (.virt 298)) (a (.wire 9 12)) (a (.wire 17 11)) ∧
    a (.wire 17 27) = bselect (a (.virt 296)) (a (.virt 16)) (a (.virt 20)) ∧
    a (.wire 17 35) = bselect (a (.virt 298)) (a (.wire 9 13)) (a (.wire 17 27))) ∧
    (a (.wire 17 43) = bselect (a (.virt 296)) (a (.virt 17)) (a (.virt 21)) ∧
    a (.wire 17 51) = bselect (a (.virt 298)) (a (.wire 9 14)) (a (.wire 17 43)) ∧
    a (.wire 17 59) = bselect (a (.virt 296)) (a (.virt 18)) (a (.virt 22)) ∧
    a (.wire 18 7) = bselect (a (.virt 298)) (a (.wire 9 15)) (a (.wire 17 59)) ∧
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) ∧
    a (.wire 18 15) = bselect (a (.wire 8 39)) (a (.virt 19)) (a (.virt 23)) ∧
    a (.wire 18 23) = bselect (a (.virt 300)) (a (.wire 9 12)) (a (.wire 18 15)) ∧
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) ∧
    a (.wire 18 31) = bselect (a (.wire 8 39)) (a (.virt 20)) (a (.virt 24)) ∧
    a (.wire 18 39) = bselect (a (.virt 300)) (a (.wire 9 13)) (a (.wire 18 31)) ∧
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) ∧
    a (.wire 18 47) = bselect (a (.wire 8 39)) (a (.virt 21)) (a (.virt 25)) ∧
    a (.wire 18 55) = bselect (a (.virt 300)) (a (.wire 9 14)) (a (.wire 18 47)) ∧
    a (.wire 8 39) = bor (a (.virt 296)) (a (.virt 298)) ∧
    a (.wire 19 3) = bselect (a (.wire 8 39)) (a (.virt 22)) (a (.virt 26)) ∧
    a (.wire 19 11) = bselect (a (.virt 300)) (a (.wire 9 15)) (a (.wire 19 3)) ∧
    a (.wire 19 19) = bselect (a (.virt 302)) (a (.wire 9 12)) (a (.virt 23)) ∧
    a (.wire 19 27) = bselect (a (.virt 302)) (a (.wire 9 13)) (a (.virt 24)) ∧
    a (.wire 19 35) = bselect (a (.virt 302)) (a (.wire 9 14)) (a (.virt 25)) ∧
    a (.wire 19 43) = bselect (a (.virt 302)) (a (.wire 9 15)) (a (.virt 26)) ∧
    (a (.wire 23 12) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 0 ∧ a (.wire 23 13) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 1 ∧ a (.wire 23 14) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 2 ∧ a (.wire 23 15) = spongeHash perm [a (.wire 16 39), a (.wire 16 47), a (.wire 16 55), a (.wire 17 3), a (.wire 17 19), a (.wire 17 35), a (.wire 17 51), a (.wire 18 7), a (.wire 18 23), a (.wire 18 39), a (.wire 18 55), a (.wire 19 11), a (.wire 19 19), a (.wire 19 27), a (.wire 19 35), a (.wire 19 43)] 3) ∧
    a (.wire 19 51) = bselect (a (.wire 8 35)) (a (.wire 23 12)) (a (.wire 9 12)) ∧
    a (.wire 19 59) = bselect (a (.wire 8 35)) (a (.wire 23 13)) (a (.wire 9 13)) ∧
    a (.wire 24 7) = bselect (a (.wire 8 35)) (a (.wire 23 14)) (a (.wire 9 14)) ∧
    a (.wire 24 15) = bselect (a (.wire 8 35)) (a (.wire 23 15)) (a (.wire 9 15)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 25 1), a (.wire 25 2), a (.wire 25 3), a (.wire 25 4), a (.wire 25 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 25 5) = band (a (.virt 292)) (a (.wire 25 5)) ∧
    a (.wire 25 5) = band (a (.wire 25 5)) (a (.virt 292)) ∧
    a (.wire 25 5) = bor (a (.virt 14)) (a (.wire 25 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 25 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14)) ∧
    (a (.wire 25 5) = a (.virt 14) + a (.wire 25 5) ∧
    a (.wire 25 5) = a (.wire 25 5) - a (.virt 14) ∧
    a (.wire 24 19) = bnot (a (.wire 25 5)) ∧
    a (.wire 24 19) = band (a (.virt 292)) (a (.wire 24 19)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 25 4) = band (a (.virt 292)) (a (.wire 25 4)) ∧
    a (.wire 14 43) = band (a (.wire 25 4)) (a (.wire 24 19)) ∧
    a (.wire 21 19) = bor (a (.wire 25 5)) (a (.wire 14 43)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 25 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 25 4) = a (.virt 14) + a (.wire 25 4) ∧
    a (.wire 25 4) = a (.wire 25 4) - a (.virt 14) ∧
    a (.wire 24 23) = bnot (a (.wire 25 4)) ∧
    a (.wire 14 47) = band (a (.wire 24 19)) (a (.wire 24 23)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 25 3) = band (a (.virt 292)) (a (.wire 25 3)) ∧
    a (.wire 14 51) = band (a (.wire 25 3)) (a (.wire 14 47)) ∧
    a (.wire 21 23) = bor (a (.wire 21 19)) (a (.wire 14 51)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 25 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 25 3) = a (.virt 14) + a (.wire 25 3) ∧
    a (.wire 25 3) = a (.wire 25 3) - a (.virt 14) ∧
    a (.wire 24 27) = bnot (a (.wire 25 3)) ∧
    a (.wire 14 55) = band (a (.wire 14 47)) (a (.wire 24 27)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 25 2) = band (a (.virt 292)) (a (.wire 25 2)) ∧
    a (.wire 14 59) = band (a (.wire 25 2)) (a (.wire 14 55)) ∧
    a (.wire 21 27) = bor (a (.wire 21 23)) (a (.wire 14 59)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 25 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 25 2) = a (.virt 14) + a (.wire 25 2) ∧
    a (.wire 25 2) = a (.wire 25 2) - a (.virt 14)) ∧
    (a (.wire 24 31) = bnot (a (.wire 25 2)) ∧
    a (.wire 26 3) = band (a (.wire 14 55)) (a (.wire 24 31)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 25 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 26 3)) ∧
    a (.wire 21 27) = bor (a (.wire 21 27)) (a (.virt 14)) ∧
    a (.wire 25 1) = a (.virt 292) * a (.wire 25 1) ∧
    a (.wire 26 7) = a (.virt 294) * a (.wire 25 1) ∧
    a (.wire 21 31) = a (.virt 292) + a (.wire 25 1) ∧
    a (.wire 24 35) = a (.wire 21 31) - a (.wire 26 7) ∧
    a (.wire 24 39) = bnot (a (.wire 24 35)) ∧
    a (.wire 26 11) = band (a (.wire 26 3)) (a (.wire 24 39)) ∧
    rangeCheck (a (.virt 208)) 2 ∧
    IsEqual (a (.virt 208)) (a (.virt 14)) (a (.virt 304)) (a (.virt 305)) ∧
    IsEqual (a (.virt 208)) (a (.virt 292)) (a (.virt 306)) (a (.virt 307)) ∧
    IsEqual (a (.virt 208)) (a (.virt 294)) (a (.virt 308)) (a (.virt 309)) ∧
    IsEqual (a (.virt 208)) (a (.virt 295)) (a (.virt 310)) (a (.virt 311)) ∧
    a (.wire 28 31) = bselect (a (.virt 304)) (a (.wire 19 51)) (a (.virt 27)) ∧
    a (.wire 28 39) = bselect (a (.virt 304)) (a (.wire 19 59)) (a (.virt 28)) ∧
    a (.wire 28 47) = bselect (a (.virt 304)) (a (.wire 24 7)) (a (.virt 29)) ∧
    a (.wire 28 55) = bselect (a (.virt 304)) (a (.wire 24 15)) (a (.virt 30)) ∧
    a (.wire 29 3) = bselect (a (.virt 304)) (a (.virt 27)) (a (.virt 31)) ∧
    a (.wire 29 11) = bselect (a (.virt 306)) (a (.wire 19 51)) (a (.wire 29 3)) ∧
    a (.wire 29 19) = bselect (a (.virt 304)) (a (.virt 28)) (a (.virt 32)) ∧
    a (.wire 29 27) = bselect (a (.virt 306)) (a (.wire 19 59)) (a (.wire 29 19)) ∧
    a (.wire 29 35) = bselect (a (.virt 304)) (a (.virt 29)) (a (.virt 33)) ∧
    a (.wire 29 43) = bselect (a (.virt 306)) (a (.wire 24 7)) (a (.wire 29 35)) ∧
    a (.wire 29 51) = bselect (a (.virt 304)) (a (.virt 30)) (a (.virt 34)) ∧
    a (.wire 29 59) = bselect (a (.virt 306)) (a (.wire 24 15)) (a (.wire 29 51)) ∧
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) ∧
    a (.wire 30 7) = bselect (a (.wire 21 35)) (a (.virt 31)) (a (.virt 35)) ∧
    a (.wire 30 15) = bselect (a (.virt 308)) (a (.wire 19 51)) (a (.wire 30 7))) ∧
    (a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) ∧
    a (.wire 30 23) = bselect (a (.wire 21 35)) (a (.virt 32)) (a (.virt 36)) ∧
    a (.wire 30 31) = bselect (a (.virt 308)) (a (.wire 19 59)) (a (.wire 30 23)) ∧
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) ∧
    a (.wire 30 39) = bselect (a (.wire 21 35)) (a (.virt 33)) (a (.virt 37)) ∧
    a (.wire 30 47) = bselect (a (.virt 308)) (a (.wire 24 7)) (a (.wire 30 39)) ∧
    a (.wire 21 35) = bor (a (.virt 304)) (a (.virt 306)) ∧
    a (.wire 30 55) = bselect (a (.wire 21 35)) (a (.virt 34)) (a (.virt 38)) ∧
    a (.wire 31 3) = bselect (a (.virt 308)) (a (.wire 24 15)) (a (.wire 30 55)) ∧
    a (.wire 31 11) = bselect (a (.virt 310)) (a (.wire 19 51)) (a (.virt 35)) ∧
    a (.wire 31 19) = bselect (a (.virt 310)) (a (.wire 19 59)) (a (.virt 36)) ∧
    a (.wire 31 27) = bselect (a (.virt 310)) (a (.wire 24 7)) (a (.virt 37)) ∧
    a (.wire 31 35) = bselect (a (.virt 310)) (a (.wire 24 15)) (a (.virt 38)) ∧
    (a (.wire 35 12) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 0 ∧ a (.wire 35 13) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 1 ∧ a (.wire 35 14) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 2 ∧ a (.wire 35 15) = spongeHash perm [a (.wire 28 31), a (.wire 28 39), a (.wire 28 47), a (.wire 28 55), a (.wire 29 11), a (.wire 29 27), a (.wire 29 43), a (.wire 29 59), a (.wire 30 15), a (.wire 30 31), a (.wire 30 47), a (.wire 31 3), a (.wire 31 11), a (.wire 31 19), a (.wire 31 27), a (.wire 31 35)] 3) ∧
    a (.wire 31 43) = bselect (a (.wire 21 27)) (a (.wire 35 12)) (a (.wire 19 51)) ∧
    a (.wire 31 51) = bselect (a (.wire 21 27)) (a (.wire 35 13)) (a (.wire 19 59)) ∧
    a (.wire 31 59) = bselect (a (.wire 21 27)) (a (.wire 35 14)) (a (.wire 24 7)) ∧
    a (.wire 36 7) = bselect (a (.wire 21 27)) (a (.wire 35 15)) (a (.wire 24 15)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 37 1), a (.wire 37 2), a (.wire 37 3), a (.wire 37 4), a (.wire 37 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 37 5) = band (a (.virt 292)) (a (.wire 37 5)) ∧
    a (.wire 37 5) = band (a (.wire 37 5)) (a (.virt 292)) ∧
    a (.wire 37 5) = bor (a (.virt 14)) (a (.wire 37 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 37 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 37 5) = a (.virt 14) + a (.wire 37 5) ∧
    a (.wire 37 5) = a (.wire 37 5) - a (.virt 14) ∧
    a (.wire 36 11) = bnot (a (.wire 37 5)) ∧
    a (.wire 36 11) = band (a (.virt 292)) (a (.wire 36 11)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 37 4) = band (a (.virt 292)) (a (.wire 37 4)) ∧
    a (.wire 26 47) = band (a (.wire 37 4)) (a (.wire 36 11))) ∧
    (a (.wire 33 15) = bor (a (.wire 37 5)) (a (.wire 26 47)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 37 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 37 4) = a (.virt 14) + a (.wire 37 4) ∧
    a (.wire 37 4) = a (.wire 37 4) - a (.virt 14) ∧
    a (.wire 36 15) = bnot (a (.wire 37 4)) ∧
    a (.wire 26 51) = band (a (.wire 36 11)) (a (.wire 36 15)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 37 3) = band (a (.virt 292)) (a (.wire 37 3)) ∧
    a (.wire 26 55) = band (a (.wire 37 3)) (a (.wire 26 51)) ∧
    a (.wire 33 19) = bor (a (.wire 33 15)) (a (.wire 26 55)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 37 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 37 3) = a (.virt 14) + a (.wire 37 3) ∧
    a (.wire 37 3) = a (.wire 37 3) - a (.virt 14) ∧
    a (.wire 36 19) = bnot (a (.wire 37 3)) ∧
    a (.wire 26 59) = band (a (.wire 26 51)) (a (.wire 36 19)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 37 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 26 59)) ∧
    a (.wire 33 19) = bor (a (.wire 33 19)) (a (.virt 14)) ∧
    a (.wire 37 2) = a (.virt 292) * a (.wire 37 2) ∧
    a (.wire 38 3) = a (.virt 294) * a (.wire 37 2) ∧
    a (.wire 33 23) = a (.virt 292) + a (.wire 37 2) ∧
    a (.wire 36 23) = a (.wire 33 23) - a (.wire 38 3) ∧
    a (.wire 36 27) = bnot (a (.wire 36 23)) ∧
    a (.wire 38 7) = band (a (.wire 26 59)) (a (.wire 36 27)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 37 1) = band (a (.virt 292)) (a (.wire 37 1)) ∧
    a (.wire 38 11) = band (a (.wire 37 1)) (a (.wire 38 7)) ∧
    a (.wire 33 27) = bor (a (.wire 33 19)) (a (.wire 38 11)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 37 1)) ∧
    (a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 37 1) = a (.virt 14) + a (.wire 37 1) ∧
    a (.wire 37 1) = a (.wire 37 1) - a (.virt 14) ∧
    a (.wire 36 31) = bnot (a (.wire 37 1)) ∧
    a (.wire 38 15) = band (a (.wire 38 7)) (a (.wire 36 31)) ∧
    rangeCheck (a (.virt 209)) 2 ∧
    IsEqual (a (.virt 209)) (a (.virt 14)) (a (.virt 312)) (a (.virt 313)) ∧
    IsEqual (a (.virt 209)) (a (.virt 292)) (a (.virt 314)) (a (.virt 315)) ∧
    IsEqual (a (.virt 209)) (a (.virt 294)) (a (.virt 316)) (a (.virt 317)) ∧
    IsEqual (a (.virt 209)) (a (.virt 295)) (a (.virt 318)) (a (.virt 319)) ∧
    a (.wire 40 23) = bselect (a (.virt 312)) (a (.wire 31 43)) (a (.virt 39)) ∧
    a (.wire 40 31) = bselect (a (.virt 312)) (a (.wire 31 51)) (a (.virt 40)) ∧
    a (.wire 40 39) = bselect (a (.virt 312)) (a (.wire 31 59)) (a (.virt 41)) ∧
    a (.wire 40 47) = bselect (a (.virt 312)) (a (.wire 36 7)) (a (.virt 42)) ∧
    a (.wire 40 55) = bselect (a (.virt 312)) (a (.virt 39)) (a (.virt 43)) ∧
    a (.wire 41 3) = bselect (a (.virt 314)) (a (.wire 31 43)) (a (.wire 40 55)) ∧
    a (.wire 41 11) = bselect (a (.virt 312)) (a (.virt 40)) (a (.virt 44)) ∧
    a (.wire 41 19) = bselect (a (.virt 314)) (a (.wire 31 51)) (a (.wire 41 11)) ∧
    a (.wire 41 27) = bselect (a (.virt 312)) (a (.virt 41)) (a (.virt 45)) ∧
    a (.wire 41 35) = bselect (a (.virt 314)) (a (.wire 31 59)) (a (.wire 41 27)) ∧
    a (.wire 41 43) = bselect (a (.virt 312)) (a (.virt 42)) (a (.virt 46)) ∧
    a (.wire 41 51) = bselect (a (.virt 314)) (a (.wire 36 7)) (a (.wire 41 43)) ∧
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) ∧
    a (.wire 41 59) = bselect (a (.wire 33 31)) (a (.virt 43)) (a (.virt 47)) ∧
    a (.wire 43 7) = bselect (a (.virt 316)) (a (.wire 31 43)) (a (.wire 41 59)) ∧
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) ∧
    a (.wire 43 15) = bselect (a (.wire 33 31)) (a (.virt 44)) (a (.virt 48)) ∧
    a (.wire 43 23) = bselect (a (.virt 316)) (a (.wire 31 51)) (a (.wire 43 15)) ∧
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314)) ∧
    a (.wire 43 31) = bselect (a (.wire 33 31)) (a (.virt 45)) (a (.virt 49)) ∧
    a (.wire 43 39) = bselect (a (.virt 316)) (a (.wire 31 59)) (a (.wire 43 31)) ∧
    a (.wire 33 31) = bor (a (.virt 312)) (a (.virt 314))) ∧
    (a (.wire 43 47) = bselect (a (.wire 33 31)) (a (.virt 46)) (a (.virt 50)) ∧
    a (.wire 43 55) = bselect (a (.virt 316)) (a (.wire 36 7)) (a (.wire 43 47)) ∧
    a (.wire 44 3) = bselect (a (.virt 318)) (a (.wire 31 43)) (a (.virt 47)) ∧
    a (.wire 44 11) = bselect (a (.virt 318)) (a (.wire 31 51)) (a (.virt 48)) ∧
    a (.wire 44 19) = bselect (a (.virt 318)) (a (.wire 31 59)) (a (.virt 49)) ∧
    a (.wire 44 27) = bselect (a (.virt 318)) (a (.wire 36 7)) (a (.virt 50)) ∧
    (a (.wire 48 12) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 0 ∧ a (.wire 48 13) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 1 ∧ a (.wire 48 14) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 2 ∧ a (.wire 48 15) = spongeHash perm [a (.wire 40 23), a (.wire 40 31), a (.wire 40 39), a (.wire 40 47), a (.wire 41 3), a (.wire 41 19), a (.wire 41 35), a (.wire 41 51), a (.wire 43 7), a (.wire 43 23), a (.wire 43 39), a (.wire 43 55), a (.wire 44 3), a (.wire 44 11), a (.wire 44 19), a (.wire 44 27)] 3) ∧
    a (.wire 44 35) = bselect (a (.wire 33 27)) (a (.wire 48 12)) (a (.wire 31 43)) ∧
    a (.wire 44 43) = bselect (a (.wire 33 27)) (a (.wire 48 13)) (a (.wire 31 51)) ∧
    a (.wire 44 51) = bselect (a (.wire 33 27)) (a (.wire 48 14)) (a (.wire 31 59)) ∧
    a (.wire 44 59) = bselect (a (.wire 33 27)) (a (.wire 48 15)) (a (.wire 36 7)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 49 1), a (.wire 49 2), a (.wire 49 3), a (.wire 49 4), a (.wire 49 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 49 5) = band (a (.virt 292)) (a (.wire 49 5)) ∧
    a (.wire 49 5) = band (a (.wire 49 5)) (a (.virt 292)) ∧
    a (.wire 49 5) = bor (a (.virt 14)) (a (.wire 49 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 49 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 49 5) = a (.virt 14) + a (.wire 49 5) ∧
    a (.wire 49 5) = a (.wire 49 5) - a (.virt 14) ∧
    a (.wire 50 3) = bnot (a (.wire 49 5)) ∧
    a (.wire 50 3) = band (a (.virt 292)) (a (.wire 50 3)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 49 4) = band (a (.virt 292)) (a (.wire 49 4)) ∧
    a (.wire 38 51) = band (a (.wire 49 4)) (a (.wire 50 3)) ∧
    a (.wire 46 11) = bor (a (.wire 49 5)) (a (.wire 38 51)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 49 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 49 4) = a (.virt 14) + a (.wire 49 4) ∧
    a (.wire 49 4) = a (.wire 49 4) - a (.virt 14) ∧
    a (.wire 50 7) = bnot (a (.wire 49 4)) ∧
    a (.wire 38 55) = band (a (.wire 50 3)) (a (.wire 50 7))) ∧
    (a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 49 3) = band (a (.virt 292)) (a (.wire 49 3)) ∧
    a (.wire 38 59) = band (a (.wire 49 3)) (a (.wire 38 55)) ∧
    a (.wire 46 15) = bor (a (.wire 46 11)) (a (.wire 38 59)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 49 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 49 3) = a (.virt 14) + a (.wire 49 3) ∧
    a (.wire 49 3) = a (.wire 49 3) - a (.virt 14) ∧
    a (.wire 50 11) = bnot (a (.wire 49 3)) ∧
    a (.wire 51 3) = band (a (.wire 38 55)) (a (.wire 50 11)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 49 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 51 3)) ∧
    a (.wire 46 15) = bor (a (.wire 46 15)) (a (.virt 14)) ∧
    a (.wire 49 2) = a (.virt 292) * a (.wire 49 2) ∧
    a (.wire 51 7) = a (.virt 294) * a (.wire 49 2) ∧
    a (.wire 46 19) = a (.virt 292) + a (.wire 49 2) ∧
    a (.wire 50 15) = a (.wire 46 19) - a (.wire 51 7) ∧
    a (.wire 50 19) = bnot (a (.wire 50 15)) ∧
    a (.wire 51 11) = band (a (.wire 51 3)) (a (.wire 50 19)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 49 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 51 11)) ∧
    a (.wire 46 15) = bor (a (.wire 46 15)) (a (.virt 14)) ∧
    a (.wire 49 1) = a (.virt 292) * a (.wire 49 1) ∧
    a (.wire 51 15) = a (.virt 294) * a (.wire 49 1) ∧
    a (.wire 46 23) = a (.virt 292) + a (.wire 49 1) ∧
    a (.wire 50 23) = a (.wire 46 23) - a (.wire 51 15) ∧
    a (.wire 50 27) = bnot (a (.wire 50 23)) ∧
    a (.wire 51 19) = band (a (.wire 51 11)) (a (.wire 50 27)) ∧
    rangeCheck (a (.virt 210)) 2 ∧
    IsEqual (a (.virt 210)) (a (.virt 14)) (a (.virt 320)) (a (.virt 321))) ∧
    (IsEqual (a (.virt 210)) (a (.virt 292)) (a (.virt 322)) (a (.virt 323)) ∧
    IsEqual (a (.virt 210)) (a (.virt 294)) (a (.virt 324)) (a (.virt 325)) ∧
    IsEqual (a (.virt 210)) (a (.virt 295)) (a (.virt 326)) (a (.virt 327)) ∧
    a (.wire 53 19) = bselect (a (.virt 320)) (a (.wire 44 35)) (a (.virt 51)) ∧
    a (.wire 53 27) = bselect (a (.virt 320)) (a (.wire 44 43)) (a (.virt 52)) ∧
    a (.wire 53 35) = bselect (a (.virt 320)) (a (.wire 44 51)) (a (.virt 53)) ∧
    a (.wire 53 43) = bselect (a (.virt 320)) (a (.wire 44 59)) (a (.virt 54)) ∧
    a (.wire 53 51) = bselect (a (.virt 320)) (a (.virt 51)) (a (.virt 55)) ∧
    a (.wire 53 59) = bselect (a (.virt 322)) (a (.wire 44 35)) (a (.wire 53 51)) ∧
    a (.wire 54 7) = bselect (a (.virt 320)) (a (.virt 52)) (a (.virt 56)) ∧
    a (.wire 54 15) = bselect (a (.virt 322)) (a (.wire 44 43)) (a (.wire 54 7)) ∧
    a (.wire 54 23) = bselect (a (.virt 320)) (a (.virt 53)) (a (.virt 57)) ∧
    a (.wire 54 31) = bselect (a (.virt 322)) (a (.wire 44 51)) (a (.wire 54 23)) ∧
    a (.wire 54 39) = bselect (a (.virt 320)) (a (.virt 54)) (a (.virt 58)) ∧
    a (.wire 54 47) = bselect (a (.virt 322)) (a (.wire 44 59)) (a (.wire 54 39)) ∧
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) ∧
    a (.wire 54 55) = bselect (a (.wire 46 27)) (a (.virt 55)) (a (.virt 59)) ∧
    a (.wire 55 3) = bselect (a (.virt 324)) (a (.wire 44 35)) (a (.wire 54 55)) ∧
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) ∧
    a (.wire 55 11) = bselect (a (.wire 46 27)) (a (.virt 56)) (a (.virt 60)) ∧
    a (.wire 55 19) = bselect (a (.virt 324)) (a (.wire 44 43)) (a (.wire 55 11)) ∧
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) ∧
    a (.wire 55 27) = bselect (a (.wire 46 27)) (a (.virt 57)) (a (.virt 61)) ∧
    a (.wire 55 35) = bselect (a (.virt 324)) (a (.wire 44 51)) (a (.wire 55 27)) ∧
    a (.wire 46 27) = bor (a (.virt 320)) (a (.virt 322)) ∧
    a (.wire 55 43) = bselect (a (.wire 46 27)) (a (.virt 58)) (a (.virt 62)) ∧
    a (.wire 55 51) = bselect (a (.virt 324)) (a (.wire 44 59)) (a (.wire 55 43)) ∧
    a (.wire 55 59) = bselect (a (.virt 326)) (a (.wire 44 35)) (a (.virt 59)) ∧
    a (.wire 56 7) = bselect (a (.virt 326)) (a (.wire 44 43)) (a (.virt 60)) ∧
    a (.wire 56 15) = bselect (a (.virt 326)) (a (.wire 44 51)) (a (.virt 61)) ∧
    a (.wire 56 23) = bselect (a (.virt 326)) (a (.wire 44 59)) (a (.virt 62)) ∧
    (a (.wire 60 12) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 0 ∧ a (.wire 60 13) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 1 ∧ a (.wire 60 14) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 2 ∧ a (.wire 60 15) = spongeHash perm [a (.wire 53 19), a (.wire 53 27), a (.wire 53 35), a (.wire 53 43), a (.wire 53 59), a (.wire 54 15), a (.wire 54 31), a (.wire 54 47), a (.wire 55 3), a (.wire 55 19), a (.wire 55 35), a (.wire 55 51), a (.wire 55 59), a (.wire 56 7), a (.wire 56 15), a (.wire 56 23)] 3)) ∧
    (a (.wire 56 31) = bselect (a (.wire 46 15)) (a (.wire 60 12)) (a (.wire 44 35)) ∧
    a (.wire 56 39) = bselect (a (.wire 46 15)) (a (.wire 60 13)) (a (.wire 44 43)) ∧
    a (.wire 56 47) = bselect (a (.wire 46 15)) (a (.wire 60 14)) (a (.wire 44 51)) ∧
    a (.wire 56 55) = bselect (a (.wire 46 15)) (a (.wire 60 15)) (a (.wire 44 59)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 61 1), a (.wire 61 2), a (.wire 61 3), a (.wire 61 4), a (.wire 61 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 61 5) = band (a (.virt 292)) (a (.wire 61 5)) ∧
    a (.wire 61 5) = band (a (.wire 61 5)) (a (.virt 292)) ∧
    a (.wire 61 5) = bor (a (.virt 14)) (a (.wire 61 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 61 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 61 5) = a (.virt 14) + a (.wire 61 5) ∧
    a (.wire 61 5) = a (.wire 61 5) - a (.virt 14) ∧
    a (.wire 56 59) = bnot (a (.wire 61 5)) ∧
    a (.wire 56 59) = band (a (.virt 292)) (a (.wire 56 59)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 61 4) = band (a (.virt 292)) (a (.wire 61 4)) ∧
    a (.wire 51 55) = band (a (.wire 61 4)) (a (.wire 56 59)) ∧
    a (.wire 59 7) = bor (a (.wire 61 5)) (a (.wire 51 55)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 61 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 61 4) = a (.virt 14) + a (.wire 61 4) ∧
    a (.wire 61 4) = a (.wire 61 4) - a (.virt 14) ∧
    a (.wire 62 3) = bnot (a (.wire 61 4)) ∧
    a (.wire 51 59) = band (a (.wire 56 59)) (a (.wire 62 3)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 61 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 51 59)) ∧
    a (.wire 59 7) = bor (a (.wire 59 7)) (a (.virt 14)) ∧
    a (.wire 61 3) = a (.virt 292) * a (.wire 61 3) ∧
    a (.wire 63 3) = a (.virt 294) * a (.wire 61 3) ∧
    a (.wire 59 11) = a (.virt 292) + a (.wire 61 3)) ∧
    (a (.wire 62 7) = a (.wire 59 11) - a (.wire 63 3) ∧
    a (.wire 62 11) = bnot (a (.wire 62 7)) ∧
    a (.wire 63 7) = band (a (.wire 51 59)) (a (.wire 62 11)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 61 2) = band (a (.virt 292)) (a (.wire 61 2)) ∧
    a (.wire 63 11) = band (a (.wire 61 2)) (a (.wire 63 7)) ∧
    a (.wire 59 15) = bor (a (.wire 59 7)) (a (.wire 63 11)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 61 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 61 2) = a (.virt 14) + a (.wire 61 2) ∧
    a (.wire 61 2) = a (.wire 61 2) - a (.virt 14) ∧
    a (.wire 62 15) = bnot (a (.wire 61 2)) ∧
    a (.wire 63 15) = band (a (.wire 63 7)) (a (.wire 62 15)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 61 1) = band (a (.virt 292)) (a (.wire 61 1)) ∧
    a (.wire 63 19) = band (a (.wire 61 1)) (a (.wire 63 15)) ∧
    a (.wire 59 19) = bor (a (.wire 59 15)) (a (.wire 63 19)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 61 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 61 1) = a (.virt 14) + a (.wire 61 1) ∧
    a (.wire 61 1) = a (.wire 61 1) - a (.virt 14) ∧
    a (.wire 62 19) = bnot (a (.wire 61 1)) ∧
    a (.wire 63 23) = band (a (.wire 63 15)) (a (.wire 62 19)) ∧
    rangeCheck (a (.virt 211)) 2 ∧
    IsEqual (a (.virt 211)) (a (.virt 14)) (a (.virt 328)) (a (.virt 329)) ∧
    IsEqual (a (.virt 211)) (a (.virt 292)) (a (.virt 330)) (a (.virt 331)) ∧
    IsEqual (a (.virt 211)) (a (.virt 294)) (a (.virt 332)) (a (.virt 333)) ∧
    IsEqual (a (.virt 211)) (a (.virt 295)) (a (.virt 334)) (a (.virt 335)) ∧
    a (.wire 65 11) = bselect (a (.virt 328)) (a (.wire 56 31)) (a (.virt 63)) ∧
    a (.wire 65 19) = bselect (a (.virt 328)) (a (.wire 56 39)) (a (.virt 64)) ∧
    a (.wire 65 27) = bselect (a (.virt 328)) (a (.wire 56 47)) (a (.virt 65)) ∧
    a (.wire 65 35) = bselect (a (.virt 328)) (a (.wire 56 55)) (a (.virt 66))) ∧
    (a (.wire 65 43) = bselect (a (.virt 328)) (a (.virt 63)) (a (.virt 67)) ∧
    a (.wire 65 51) = bselect (a (.virt 330)) (a (.wire 56 31)) (a (.wire 65 43)) ∧
    a (.wire 65 59) = bselect (a (.virt 328)) (a (.virt 64)) (a (.virt 68)) ∧
    a (.wire 66 7) = bselect (a (.virt 330)) (a (.wire 56 39)) (a (.wire 65 59)) ∧
    a (.wire 66 15) = bselect (a (.virt 328)) (a (.virt 65)) (a (.virt 69)) ∧
    a (.wire 66 23) = bselect (a (.virt 330)) (a (.wire 56 47)) (a (.wire 66 15)) ∧
    a (.wire 66 31) = bselect (a (.virt 328)) (a (.virt 66)) (a (.virt 70)) ∧
    a (.wire 66 39) = bselect (a (.virt 330)) (a (.wire 56 55)) (a (.wire 66 31)) ∧
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) ∧
    a (.wire 66 47) = bselect (a (.wire 59 23)) (a (.virt 67)) (a (.virt 71)) ∧
    a (.wire 66 55) = bselect (a (.virt 332)) (a (.wire 56 31)) (a (.wire 66 47)) ∧
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) ∧
    a (.wire 67 3) = bselect (a (.wire 59 23)) (a (.virt 68)) (a (.virt 72)) ∧
    a (.wire 67 11) = bselect (a (.virt 332)) (a (.wire 56 39)) (a (.wire 67 3)) ∧
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) ∧
    a (.wire 67 19) = bselect (a (.wire 59 23)) (a (.virt 69)) (a (.virt 73)) ∧
    a (.wire 67 27) = bselect (a (.virt 332)) (a (.wire 56 47)) (a (.wire 67 19)) ∧
    a (.wire 59 23) = bor (a (.virt 328)) (a (.virt 330)) ∧
    a (.wire 67 35) = bselect (a (.wire 59 23)) (a (.virt 70)) (a (.virt 74)) ∧
    a (.wire 67 43) = bselect (a (.virt 332)) (a (.wire 56 55)) (a (.wire 67 35)) ∧
    a (.wire 67 51) = bselect (a (.virt 334)) (a (.wire 56 31)) (a (.virt 71)) ∧
    a (.wire 67 59) = bselect (a (.virt 334)) (a (.wire 56 39)) (a (.virt 72)) ∧
    a (.wire 68 7) = bselect (a (.virt 334)) (a (.wire 56 47)) (a (.virt 73)) ∧
    a (.wire 68 15) = bselect (a (.virt 334)) (a (.wire 56 55)) (a (.virt 74)) ∧
    (a (.wire 71 12) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 0 ∧ a (.wire 71 13) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 1 ∧ a (.wire 71 14) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 2 ∧ a (.wire 71 15) = spongeHash perm [a (.wire 65 11), a (.wire 65 19), a (.wire 65 27), a (.wire 65 35), a (.wire 65 51), a (.wire 66 7), a (.wire 66 23), a (.wire 66 39), a (.wire 66 55), a (.wire 67 11), a (.wire 67 27), a (.wire 67 43), a (.wire 67 51), a (.wire 67 59), a (.wire 68 7), a (.wire 68 15)] 3) ∧
    a (.wire 68 23) = bselect (a (.wire 59 19)) (a (.wire 71 12)) (a (.wire 56 31)) ∧
    a (.wire 68 31) = bselect (a (.wire 59 19)) (a (.wire 71 13)) (a (.wire 56 39)) ∧
    a (.wire 68 39) = bselect (a (.wire 59 19)) (a (.wire 71 14)) (a (.wire 56 47)) ∧
    a (.wire 68 47) = bselect (a (.wire 59 19)) (a (.wire 71 15)) (a (.wire 56 55)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 72 1), a (.wire 72 2), a (.wire 72 3), a (.wire 72 4), a (.wire 72 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 72 5) = band (a (.virt 292)) (a (.wire 72 5))) ∧
    (a (.wire 72 5) = band (a (.wire 72 5)) (a (.virt 292)) ∧
    a (.wire 72 5) = bor (a (.virt 14)) (a (.wire 72 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 72 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 72 5) = a (.virt 14) + a (.wire 72 5) ∧
    a (.wire 72 5) = a (.wire 72 5) - a (.virt 14) ∧
    a (.wire 68 51) = bnot (a (.wire 72 5)) ∧
    a (.wire 68 51) = band (a (.virt 292)) (a (.wire 68 51)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 72 4) = band (a (.virt 292)) (a (.wire 72 4)) ∧
    a (.wire 63 59) = band (a (.wire 72 4)) (a (.wire 68 51)) ∧
    a (.wire 73 3) = bor (a (.wire 72 5)) (a (.wire 63 59)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 72 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 72 4) = a (.virt 14) + a (.wire 72 4) ∧
    a (.wire 72 4) = a (.wire 72 4) - a (.virt 14) ∧
    a (.wire 68 55) = bnot (a (.wire 72 4)) ∧
    a (.wire 74 3) = band (a (.wire 68 51)) (a (.wire 68 55)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 72 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 74 3)) ∧
    a (.wire 73 3) = bor (a (.wire 73 3)) (a (.virt 14)) ∧
    a (.wire 72 3) = a (.virt 292) * a (.wire 72 3) ∧
    a (.wire 74 7) = a (.virt 294) * a (.wire 72 3) ∧
    a (.wire 73 7) = a (.virt 292) + a (.wire 72 3) ∧
    a (.wire 68 59) = a (.wire 73 7) - a (.wire 74 7) ∧
    a (.wire 75 3) = bnot (a (.wire 68 59)) ∧
    a (.wire 74 11) = band (a (.wire 74 3)) (a (.wire 75 3)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 72 2) = band (a (.virt 292)) (a (.wire 72 2)) ∧
    a (.wire 74 15) = band (a (.wire 72 2)) (a (.wire 74 11)) ∧
    a (.wire 73 11) = bor (a (.wire 73 3)) (a (.wire 74 15))) ∧
    (a (.virt 14) = a (.virt 14) * a (.wire 72 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 72 2) = a (.virt 14) + a (.wire 72 2) ∧
    a (.wire 72 2) = a (.wire 72 2) - a (.virt 14) ∧
    a (.wire 75 7) = bnot (a (.wire 72 2)) ∧
    a (.wire 74 19) = band (a (.wire 74 11)) (a (.wire 75 7)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 72 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 74 19)) ∧
    a (.wire 73 11) = bor (a (.wire 73 11)) (a (.virt 14)) ∧
    a (.wire 72 1) = a (.virt 292) * a (.wire 72 1) ∧
    a (.wire 74 23) = a (.virt 294) * a (.wire 72 1) ∧
    a (.wire 73 15) = a (.virt 292) + a (.wire 72 1) ∧
    a (.wire 75 11) = a (.wire 73 15) - a (.wire 74 23) ∧
    a (.wire 75 15) = bnot (a (.wire 75 11)) ∧
    a (.wire 74 27) = band (a (.wire 74 19)) (a (.wire 75 15)) ∧
    rangeCheck (a (.virt 212)) 2 ∧
    IsEqual (a (.virt 212)) (a (.virt 14)) (a (.virt 336)) (a (.virt 337)) ∧
    IsEqual (a (.virt 212)) (a (.virt 292)) (a (.virt 338)) (a (.virt 339)) ∧
    IsEqual (a (.virt 212)) (a (.virt 294)) (a (.virt 340)) (a (.virt 341)) ∧
    IsEqual (a (.virt 212)) (a (.virt 295)) (a (.virt 342)) (a (.virt 343)) ∧
    a (.wire 77 7) = bselect (a (.virt 336)) (a (.wire 68 23)) (a (.virt 75)) ∧
    a (.wire 77 15) = bselect (a (.virt 336)) (a (.wire 68 31)) (a (.virt 76)) ∧
    a (.wire 77 23) = bselect (a (.virt 336)) (a (.wire 68 39)) (a (.virt 77)) ∧
    a (.wire 77 31) = bselect (a (.virt 336)) (a (.wire 68 47)) (a (.virt 78)) ∧
    a (.wire 77 39) = bselect (a (.virt 336)) (a (.virt 75)) (a (.virt 79)) ∧
    a (.wire 77 47) = bselect (a (.virt 338)) (a (.wire 68 23)) (a (.wire 77 39)) ∧
    a (.wire 77 55) = bselect (a (.virt 336)) (a (.virt 76)) (a (.virt 80)) ∧
    a (.wire 78 3) = bselect (a (.virt 338)) (a (.wire 68 31)) (a (.wire 77 55)) ∧
    a (.wire 78 11) = bselect (a (.virt 336)) (a (.virt 77)) (a (.virt 81)) ∧
    a (.wire 78 19) = bselect (a (.virt 338)) (a (.wire 68 39)) (a (.wire 78 11)) ∧
    a (.wire 78 27) = bselect (a (.virt 336)) (a (.virt 78)) (a (.virt 82))) ∧
    (a (.wire 78 35) = bselect (a (.virt 338)) (a (.wire 68 47)) (a (.wire 78 27)) ∧
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) ∧
    a (.wire 78 43) = bselect (a (.wire 73 19)) (a (.virt 79)) (a (.virt 83)) ∧
    a (.wire 78 51) = bselect (a (.virt 340)) (a (.wire 68 23)) (a (.wire 78 43)) ∧
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) ∧
    a (.wire 78 59) = bselect (a (.wire 73 19)) (a (.virt 80)) (a (.virt 84)) ∧
    a (.wire 79 7) = bselect (a (.virt 340)) (a (.wire 68 31)) (a (.wire 78 59)) ∧
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) ∧
    a (.wire 79 15) = bselect (a (.wire 73 19)) (a (.virt 81)) (a (.virt 85)) ∧
    a (.wire 79 23) = bselect (a (.virt 340)) (a (.wire 68 39)) (a (.wire 79 15)) ∧
    a (.wire 73 19) = bor (a (.virt 336)) (a (.virt 338)) ∧
    a (.wire 79 31) = bselect (a (.wire 73 19)) (a (.virt 82)) (a (.virt 86)) ∧
    a (.wire 79 39) = bselect (a (.virt 340)) (a (.wire 68 47)) (a (.wire 79 31)) ∧
    a (.wire 79 47) = bselect (a (.virt 342)) (a (.wire 68 23)) (a (.virt 83)) ∧
    a (.wire 79 55) = bselect (a (.virt 342)) (a (.wire 68 31)) (a (.virt 84)) ∧
    a (.wire 80 3) = bselect (a (.virt 342)) (a (.wire 68 39)) (a (.virt 85)) ∧
    a (.wire 80 11) = bselect (a (.virt 342)) (a (.wire 68 47)) (a (.virt 86)) ∧
    (a (.wire 83 12) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 0 ∧ a (.wire 83 13) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 1 ∧ a (.wire 83 14) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 2 ∧ a (.wire 83 15) = spongeHash perm [a (.wire 77 7), a (.wire 77 15), a (.wire 77 23), a (.wire 77 31), a (.wire 77 47), a (.wire 78 3), a (.wire 78 19), a (.wire 78 35), a (.wire 78 51), a (.wire 79 7), a (.wire 79 23), a (.wire 79 39), a (.wire 79 47), a (.wire 79 55), a (.wire 80 3), a (.wire 80 11)] 3) ∧
    a (.wire 80 19) = bselect (a (.wire 73 11)) (a (.wire 83 12)) (a (.wire 68 23)) ∧
    a (.wire 80 27) = bselect (a (.wire 73 11)) (a (.wire 83 13)) (a (.wire 68 31)) ∧
    a (.wire 80 35) = bselect (a (.wire 73 11)) (a (.wire 83 14)) (a (.wire 68 39)) ∧
    a (.wire 80 43) = bselect (a (.wire 73 11)) (a (.wire 83 15)) (a (.wire 68 47)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 84 1), a (.wire 84 2), a (.wire 84 3), a (.wire 84 4), a (.wire 84 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 84 5) = band (a (.virt 292)) (a (.wire 84 5)) ∧
    a (.wire 84 5) = band (a (.wire 84 5)) (a (.virt 292)) ∧
    a (.wire 84 5) = bor (a (.virt 14)) (a (.wire 84 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 84 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 84 5) = a (.virt 14) + a (.wire 84 5) ∧
    a (.wire 84 5) = a (.wire 84 5) - a (.virt 14) ∧
    a (.wire 80 47) = bnot (a (.wire 84 5))) ∧
    (a (.wire 80 47) = band (a (.virt 292)) (a (.wire 80 47)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 84 4) = band (a (.virt 292)) (a (.wire 84 4)) ∧
    a (.wire 85 3) = band (a (.wire 84 4)) (a (.wire 80 47)) ∧
    a (.wire 73 59) = bor (a (.wire 84 5)) (a (.wire 85 3)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 84 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 84 4) = a (.virt 14) + a (.wire 84 4) ∧
    a (.wire 84 4) = a (.wire 84 4) - a (.virt 14) ∧
    a (.wire 80 51) = bnot (a (.wire 84 4)) ∧
    a (.wire 85 7) = band (a (.wire 80 47)) (a (.wire 80 51)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 84 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 85 7)) ∧
    a (.wire 73 59) = bor (a (.wire 73 59)) (a (.virt 14)) ∧
    a (.wire 84 3) = a (.virt 292) * a (.wire 84 3) ∧
    a (.wire 85 11) = a (.virt 294) * a (.wire 84 3) ∧
    a (.wire 86 3) = a (.virt 292) + a (.wire 84 3) ∧
    a (.wire 80 55) = a (.wire 86 3) - a (.wire 85 11) ∧
    a (.wire 80 59) = bnot (a (.wire 80 55)) ∧
    a (.wire 85 15) = band (a (.wire 85 7)) (a (.wire 80 59)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 84 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 85 15)) ∧
    a (.wire 73 59) = bor (a (.wire 73 59)) (a (.virt 14)) ∧
    a (.wire 84 2) = a (.virt 292) * a (.wire 84 2) ∧
    a (.wire 85 19) = a (.virt 294) * a (.wire 84 2) ∧
    a (.wire 86 7) = a (.virt 292) + a (.wire 84 2) ∧
    a (.wire 87 3) = a (.wire 86 7) - a (.wire 85 19) ∧
    a (.wire 87 7) = bnot (a (.wire 87 3)) ∧
    a (.wire 85 23) = band (a (.wire 85 15)) (a (.wire 87 7)) ∧
    a (.virt 292) = bnot (a (.virt 14))) ∧
    (a (.wire 84 1) = band (a (.virt 292)) (a (.wire 84 1)) ∧
    a (.wire 85 27) = band (a (.wire 84 1)) (a (.wire 85 23)) ∧
    a (.wire 86 11) = bor (a (.wire 73 59)) (a (.wire 85 27)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 84 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 84 1) = a (.virt 14) + a (.wire 84 1) ∧
    a (.wire 84 1) = a (.wire 84 1) - a (.virt 14) ∧
    a (.wire 87 11) = bnot (a (.wire 84 1)) ∧
    a (.wire 85 31) = band (a (.wire 85 23)) (a (.wire 87 11)) ∧
    rangeCheck (a (.virt 213)) 2 ∧
    IsEqual (a (.virt 213)) (a (.virt 14)) (a (.virt 344)) (a (.virt 345)) ∧
    IsEqual (a (.virt 213)) (a (.virt 292)) (a (.virt 346)) (a (.virt 347)) ∧
    IsEqual (a (.virt 213)) (a (.virt 294)) (a (.virt 348)) (a (.virt 349)) ∧
    IsEqual (a (.virt 213)) (a (.virt 295)) (a (.virt 350)) (a (.virt 351)) ∧
    a (.wire 90 3) = bselect (a (.virt 344)) (a (.wire 80 19)) (a (.virt 87)) ∧
    a (.wire 90 11) = bselect (a (.virt 344)) (a (.wire 80 27)) (a (.virt 88)) ∧
    a (.wire 90 19) = bselect (a (.virt 344)) (a (.wire 80 35)) (a (.virt 89)) ∧
    a (.wire 90 27) = bselect (a (.virt 344)) (a (.wire 80 43)) (a (.virt 90)) ∧
    a (.wire 90 35) = bselect (a (.virt 344)) (a (.virt 87)) (a (.virt 91)) ∧
    a (.wire 90 43) = bselect (a (.virt 346)) (a (.wire 80 19)) (a (.wire 90 35)) ∧
    a (.wire 90 51) = bselect (a (.virt 344)) (a (.virt 88)) (a (.virt 92)) ∧
    a (.wire 90 59) = bselect (a (.virt 346)) (a (.wire 80 27)) (a (.wire 90 51)) ∧
    a (.wire 91 7) = bselect (a (.virt 344)) (a (.virt 89)) (a (.virt 93)) ∧
    a (.wire 91 15) = bselect (a (.virt 346)) (a (.wire 80 35)) (a (.wire 91 7)) ∧
    a (.wire 91 23) = bselect (a (.virt 344)) (a (.virt 90)) (a (.virt 94)) ∧
    a (.wire 91 31) = bselect (a (.virt 346)) (a (.wire 80 43)) (a (.wire 91 23)) ∧
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) ∧
    a (.wire 91 39) = bselect (a (.wire 86 15)) (a (.virt 91)) (a (.virt 95)) ∧
    a (.wire 91 47) = bselect (a (.virt 348)) (a (.wire 80 19)) (a (.wire 91 39)) ∧
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) ∧
    a (.wire 91 55) = bselect (a (.wire 86 15)) (a (.virt 92)) (a (.virt 96)) ∧
    a (.wire 92 3) = bselect (a (.virt 348)) (a (.wire 80 27)) (a (.wire 91 55))) ∧
    (a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) ∧
    a (.wire 92 11) = bselect (a (.wire 86 15)) (a (.virt 93)) (a (.virt 97)) ∧
    a (.wire 92 19) = bselect (a (.virt 348)) (a (.wire 80 35)) (a (.wire 92 11)) ∧
    a (.wire 86 15) = bor (a (.virt 344)) (a (.virt 346)) ∧
    a (.wire 92 27) = bselect (a (.wire 86 15)) (a (.virt 94)) (a (.virt 98)) ∧
    a (.wire 92 35) = bselect (a (.virt 348)) (a (.wire 80 43)) (a (.wire 92 27)) ∧
    a (.wire 92 43) = bselect (a (.virt 350)) (a (.wire 80 19)) (a (.virt 95)) ∧
    a (.wire 92 51) = bselect (a (.virt 350)) (a (.wire 80 27)) (a (.virt 96)) ∧
    a (.wire 92 59) = bselect (a (.virt 350)) (a (.wire 80 35)) (a (.virt 97)) ∧
    a (.wire 93 7) = bselect (a (.virt 350)) (a (.wire 80 43)) (a (.virt 98)) ∧
    (a (.wire 96 12) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 0 ∧ a (.wire 96 13) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 1 ∧ a (.wire 96 14) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 2 ∧ a (.wire 96 15) = spongeHash perm [a (.wire 90 3), a (.wire 90 11), a (.wire 90 19), a (.wire 90 27), a (.wire 90 43), a (.wire 90 59), a (.wire 91 15), a (.wire 91 31), a (.wire 91 47), a (.wire 92 3), a (.wire 92 19), a (.wire 92 35), a (.wire 92 43), a (.wire 92 51), a (.wire 92 59), a (.wire 93 7)] 3) ∧
    a (.wire 93 15) = bselect (a (.wire 86 11)) (a (.wire 96 12)) (a (.wire 80 19)) ∧
    a (.wire 93 23) = bselect (a (.wire 86 11)) (a (.wire 96 13)) (a (.wire 80 27)) ∧
    a (.wire 93 31) = bselect (a (.wire 86 11)) (a (.wire 96 14)) (a (.wire 80 35)) ∧
    a (.wire 93 39) = bselect (a (.wire 86 11)) (a (.wire 96 15)) (a (.wire 80 43)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 97 1), a (.wire 97 2), a (.wire 97 3), a (.wire 97 4), a (.wire 97 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 97 5) = band (a (.virt 292)) (a (.wire 97 5)) ∧
    a (.wire 97 5) = band (a (.wire 97 5)) (a (.virt 292)) ∧
    a (.wire 97 5) = bor (a (.virt 14)) (a (.wire 97 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 97 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 97 5) = a (.virt 14) + a (.wire 97 5) ∧
    a (.wire 97 5) = a (.wire 97 5) - a (.virt 14) ∧
    a (.wire 93 43) = bnot (a (.wire 97 5)) ∧
    a (.wire 93 43) = band (a (.virt 292)) (a (.wire 93 43)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 97 4) = band (a (.virt 292)) (a (.wire 97 4)) ∧
    a (.wire 89 7) = band (a (.wire 97 4)) (a (.wire 93 43)) ∧
    a (.wire 86 55) = bor (a (.wire 97 5)) (a (.wire 89 7)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 97 4) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14)) ∧
    (a (.wire 97 4) = a (.virt 14) + a (.wire 97 4) ∧
    a (.wire 97 4) = a (.wire 97 4) - a (.virt 14) ∧
    a (.wire 93 47) = bnot (a (.wire 97 4)) ∧
    a (.wire 89 11) = band (a (.wire 93 43)) (a (.wire 93 47)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 97 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 89 11)) ∧
    a (.wire 86 55) = bor (a (.wire 86 55)) (a (.virt 14)) ∧
    a (.wire 97 3) = a (.virt 292) * a (.wire 97 3) ∧
    a (.wire 89 15) = a (.virt 294) * a (.wire 97 3) ∧
    a (.wire 86 59) = a (.virt 292) + a (.wire 97 3) ∧
    a (.wire 93 51) = a (.wire 86 59) - a (.wire 89 15) ∧
    a (.wire 93 55) = bnot (a (.wire 93 51)) ∧
    a (.wire 89 19) = band (a (.wire 89 11)) (a (.wire 93 55)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 97 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 89 19)) ∧
    a (.wire 86 55) = bor (a (.wire 86 55)) (a (.virt 14)) ∧
    a (.wire 97 2) = a (.virt 292) * a (.wire 97 2) ∧
    a (.wire 89 23) = a (.virt 294) * a (.wire 97 2) ∧
    a (.wire 98 3) = a (.virt 292) + a (.wire 97 2) ∧
    a (.wire 93 59) = a (.wire 98 3) - a (.wire 89 23) ∧
    a (.wire 99 3) = bnot (a (.wire 93 59)) ∧
    a (.wire 89 27) = band (a (.wire 89 19)) (a (.wire 99 3)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 97 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 89 27)) ∧
    a (.wire 86 55) = bor (a (.wire 86 55)) (a (.virt 14)) ∧
    a (.wire 97 1) = a (.virt 292) * a (.wire 97 1) ∧
    a (.wire 89 31) = a (.virt 294) * a (.wire 97 1) ∧
    a (.wire 98 7) = a (.virt 292) + a (.wire 97 1) ∧
    a (.wire 99 7) = a (.wire 98 7) - a (.wire 89 31)) ∧
    (a (.wire 99 11) = bnot (a (.wire 99 7)) ∧
    a (.wire 89 35) = band (a (.wire 89 27)) (a (.wire 99 11)) ∧
    rangeCheck (a (.virt 214)) 2 ∧
    IsEqual (a (.virt 214)) (a (.virt 14)) (a (.virt 352)) (a (.virt 353)) ∧
    IsEqual (a (.virt 214)) (a (.virt 292)) (a (.virt 354)) (a (.virt 355)) ∧
    IsEqual (a (.virt 214)) (a (.virt 294)) (a (.virt 356)) (a (.virt 357)) ∧
    IsEqual (a (.virt 214)) (a (.virt 295)) (a (.virt 358)) (a (.virt 359)) ∧
    a (.wire 102 3) = bselect (a (.virt 352)) (a (.wire 93 15)) (a (.virt 99)) ∧
    a (.wire 102 11) = bselect (a (.virt 352)) (a (.wire 93 23)) (a (.virt 100)) ∧
    a (.wire 102 19) = bselect (a (.virt 352)) (a (.wire 93 31)) (a (.virt 101)) ∧
    a (.wire 102 27) = bselect (a (.virt 352)) (a (.wire 93 39)) (a (.virt 102)) ∧
    a (.wire 102 35) = bselect (a (.virt 352)) (a (.virt 99)) (a (.virt 103)) ∧
    a (.wire 102 43) = bselect (a (.virt 354)) (a (.wire 93 15)) (a (.wire 102 35)) ∧
    a (.wire 102 51) = bselect (a (.virt 352)) (a (.virt 100)) (a (.virt 104)) ∧
    a (.wire 102 59) = bselect (a (.virt 354)) (a (.wire 93 23)) (a (.wire 102 51)) ∧
    a (.wire 103 7) = bselect (a (.virt 352)) (a (.virt 101)) (a (.virt 105)) ∧
    a (.wire 103 15) = bselect (a (.virt 354)) (a (.wire 93 31)) (a (.wire 103 7)) ∧
    a (.wire 103 23) = bselect (a (.virt 352)) (a (.virt 102)) (a (.virt 106)) ∧
    a (.wire 103 31) = bselect (a (.virt 354)) (a (.wire 93 39)) (a (.wire 103 23)) ∧
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) ∧
    a (.wire 103 39) = bselect (a (.wire 98 11)) (a (.virt 103)) (a (.virt 107)) ∧
    a (.wire 103 47) = bselect (a (.virt 356)) (a (.wire 93 15)) (a (.wire 103 39)) ∧
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) ∧
    a (.wire 103 55) = bselect (a (.wire 98 11)) (a (.virt 104)) (a (.virt 108)) ∧
    a (.wire 105 3) = bselect (a (.virt 356)) (a (.wire 93 23)) (a (.wire 103 55)) ∧
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) ∧
    a (.wire 105 11) = bselect (a (.wire 98 11)) (a (.virt 105)) (a (.virt 109)) ∧
    a (.wire 105 19) = bselect (a (.virt 356)) (a (.wire 93 31)) (a (.wire 105 11)) ∧
    a (.wire 98 11) = bor (a (.virt 352)) (a (.virt 354)) ∧
    a (.wire 105 27) = bselect (a (.wire 98 11)) (a (.virt 106)) (a (.virt 110)) ∧
    a (.wire 105 35) = bselect (a (.virt 356)) (a (.wire 93 39)) (a (.wire 105 27)) ∧
    a (.wire 105 43) = bselect (a (.virt 358)) (a (.wire 93 15)) (a (.virt 107))) ∧
    (a (.wire 105 51) = bselect (a (.virt 358)) (a (.wire 93 23)) (a (.virt 108)) ∧
    a (.wire 105 59) = bselect (a (.virt 358)) (a (.wire 93 31)) (a (.virt 109)) ∧
    a (.wire 106 7) = bselect (a (.virt 358)) (a (.wire 93 39)) (a (.virt 110)) ∧
    (a (.wire 109 12) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 0 ∧ a (.wire 109 13) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 1 ∧ a (.wire 109 14) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 2 ∧ a (.wire 109 15) = spongeHash perm [a (.wire 102 3), a (.wire 102 11), a (.wire 102 19), a (.wire 102 27), a (.wire 102 43), a (.wire 102 59), a (.wire 103 15), a (.wire 103 31), a (.wire 103 47), a (.wire 105 3), a (.wire 105 19), a (.wire 105 35), a (.wire 105 43), a (.wire 105 51), a (.wire 105 59), a (.wire 106 7)] 3) ∧
    a (.wire 106 15) = bselect (a (.wire 86 55)) (a (.wire 109 12)) (a (.wire 93 15)) ∧
    a (.wire 106 23) = bselect (a (.wire 86 55)) (a (.wire 109 13)) (a (.wire 93 23)) ∧
    a (.wire 106 31) = bselect (a (.wire 86 55)) (a (.wire 109 14)) (a (.wire 93 31)) ∧
    a (.wire 106 39) = bselect (a (.wire 86 55)) (a (.wire 109 15)) (a (.wire 93 39)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 110 1), a (.wire 110 2), a (.wire 110 3), a (.wire 110 4), a (.wire 110 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 110 5) = band (a (.virt 292)) (a (.wire 110 5)) ∧
    a (.wire 110 5) = band (a (.wire 110 5)) (a (.virt 292)) ∧
    a (.wire 110 5) = bor (a (.virt 14)) (a (.wire 110 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 110 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 110 5) = a (.virt 14) + a (.wire 110 5) ∧
    a (.wire 110 5) = a (.wire 110 5) - a (.virt 14) ∧
    a (.wire 106 43) = bnot (a (.wire 110 5)) ∧
    a (.wire 106 43) = band (a (.virt 292)) (a (.wire 106 43)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 110 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 106 43)) ∧
    a (.wire 110 5) = bor (a (.wire 110 5)) (a (.virt 14)) ∧
    a (.wire 110 4) = a (.virt 292) * a (.wire 110 4) ∧
    a (.wire 101 11) = a (.virt 294) * a (.wire 110 4) ∧
    a (.wire 98 51) = a (.virt 292) + a (.wire 110 4) ∧
    a (.wire 106 47) = a (.wire 98 51) - a (.wire 101 11) ∧
    a (.wire 106 51) = bnot (a (.wire 106 47)) ∧
    a (.wire 101 15) = band (a (.wire 106 43)) (a (.wire 106 51)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 110 3) = band (a (.virt 292)) (a (.wire 110 3)) ∧
    a (.wire 101 19) = band (a (.wire 110 3)) (a (.wire 101 15))) ∧
    (a (.wire 98 55) = bor (a (.wire 110 5)) (a (.wire 101 19)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 110 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 110 3) = a (.virt 14) + a (.wire 110 3) ∧
    a (.wire 110 3) = a (.wire 110 3) - a (.virt 14) ∧
    a (.wire 106 55) = bnot (a (.wire 110 3)) ∧
    a (.wire 101 23) = band (a (.wire 101 15)) (a (.wire 106 55)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 110 2) = band (a (.virt 292)) (a (.wire 110 2)) ∧
    a (.wire 101 27) = band (a (.wire 110 2)) (a (.wire 101 23)) ∧
    a (.wire 98 59) = bor (a (.wire 98 55)) (a (.wire 101 27)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 110 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 110 2) = a (.virt 14) + a (.wire 110 2) ∧
    a (.wire 110 2) = a (.wire 110 2) - a (.virt 14) ∧
    a (.wire 106 59) = bnot (a (.wire 110 2)) ∧
    a (.wire 101 31) = band (a (.wire 101 23)) (a (.wire 106 59)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 110 1) = band (a (.virt 292)) (a (.wire 110 1)) ∧
    a (.wire 101 35) = band (a (.wire 110 1)) (a (.wire 101 31)) ∧
    a (.wire 111 3) = bor (a (.wire 98 59)) (a (.wire 101 35)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 110 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 110 1) = a (.virt 14) + a (.wire 110 1) ∧
    a (.wire 110 1) = a (.wire 110 1) - a (.virt 14) ∧
    a (.wire 112 3) = bnot (a (.wire 110 1)) ∧
    a (.wire 101 39) = band (a (.wire 101 31)) (a (.wire 112 3)) ∧
    rangeCheck (a (.virt 215)) 2 ∧
    IsEqual (a (.virt 215)) (a (.virt 14)) (a (.virt 360)) (a (.virt 361)) ∧
    IsEqual (a (.virt 215)) (a (.virt 292)) (a (.virt 362)) (a (.virt 363)) ∧
    IsEqual (a (.virt 215)) (a (.virt 294)) (a (.virt 364)) (a (.virt 365)) ∧
    IsEqual (a (.virt 215)) (a (.virt 295)) (a (.virt 366)) (a (.virt 367))) ∧
    (a (.wire 112 55) = bselect (a (.virt 360)) (a (.wire 106 15)) (a (.virt 111)) ∧
    a (.wire 115 3) = bselect (a (.virt 360)) (a (.wire 106 23)) (a (.virt 112)) ∧
    a (.wire 115 11) = bselect (a (.virt 360)) (a (.wire 106 31)) (a (.virt 113)) ∧
    a (.wire 115 19) = bselect (a (.virt 360)) (a (.wire 106 39)) (a (.virt 114)) ∧
    a (.wire 115 27) = bselect (a (.virt 360)) (a (.virt 111)) (a (.virt 115)) ∧
    a (.wire 115 35) = bselect (a (.virt 362)) (a (.wire 106 15)) (a (.wire 115 27)) ∧
    a (.wire 115 43) = bselect (a (.virt 360)) (a (.virt 112)) (a (.virt 116)) ∧
    a (.wire 115 51) = bselect (a (.virt 362)) (a (.wire 106 23)) (a (.wire 115 43)) ∧
    a (.wire 115 59) = bselect (a (.virt 360)) (a (.virt 113)) (a (.virt 117)) ∧
    a (.wire 116 7) = bselect (a (.virt 362)) (a (.wire 106 31)) (a (.wire 115 59)) ∧
    a (.wire 116 15) = bselect (a (.virt 360)) (a (.virt 114)) (a (.virt 118)) ∧
    a (.wire 116 23) = bselect (a (.virt 362)) (a (.wire 106 39)) (a (.wire 116 15)) ∧
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) ∧
    a (.wire 116 31) = bselect (a (.wire 111 7)) (a (.virt 115)) (a (.virt 119)) ∧
    a (.wire 116 39) = bselect (a (.virt 364)) (a (.wire 106 15)) (a (.wire 116 31)) ∧
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) ∧
    a (.wire 116 47) = bselect (a (.wire 111 7)) (a (.virt 116)) (a (.virt 120)) ∧
    a (.wire 116 55) = bselect (a (.virt 364)) (a (.wire 106 23)) (a (.wire 116 47)) ∧
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) ∧
    a (.wire 117 3) = bselect (a (.wire 111 7)) (a (.virt 117)) (a (.virt 121)) ∧
    a (.wire 117 11) = bselect (a (.virt 364)) (a (.wire 106 31)) (a (.wire 117 3)) ∧
    a (.wire 111 7) = bor (a (.virt 360)) (a (.virt 362)) ∧
    a (.wire 117 19) = bselect (a (.wire 111 7)) (a (.virt 118)) (a (.virt 122)) ∧
    a (.wire 117 27) = bselect (a (.virt 364)) (a (.wire 106 39)) (a (.wire 117 19)) ∧
    a (.wire 117 35) = bselect (a (.virt 366)) (a (.wire 106 15)) (a (.virt 119)) ∧
    a (.wire 117 43) = bselect (a (.virt 366)) (a (.wire 106 23)) (a (.virt 120)) ∧
    a (.wire 117 51) = bselect (a (.virt 366)) (a (.wire 106 31)) (a (.virt 121)) ∧
    a (.wire 117 59) = bselect (a (.virt 366)) (a (.wire 106 39)) (a (.virt 122)) ∧
    (a (.wire 120 12) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 0 ∧ a (.wire 120 13) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 1 ∧ a (.wire 120 14) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 2 ∧ a (.wire 120 15) = spongeHash perm [a (.wire 112 55), a (.wire 115 3), a (.wire 115 11), a (.wire 115 19), a (.wire 115 35), a (.wire 115 51), a (.wire 116 7), a (.wire 116 23), a (.wire 116 39), a (.wire 116 55), a (.wire 117 11), a (.wire 117 27), a (.wire 117 35), a (.wire 117 43), a (.wire 117 51), a (.wire 117 59)] 3) ∧
    a (.wire 121 7) = bselect (a (.wire 111 3)) (a (.wire 120 12)) (a (.wire 106 15)) ∧
    a (.wire 121 15) = bselect (a (.wire 111 3)) (a (.wire 120 13)) (a (.wire 106 23)) ∧
    a (.wire 121 23) = bselect (a (.wire 111 3)) (a (.wire 120 14)) (a (.wire 106 31))) ∧
    (a (.wire 121 31) = bselect (a (.wire 111 3)) (a (.wire 120 15)) (a (.wire 106 39)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 122 1), a (.wire 122 2), a (.wire 122 3), a (.wire 122 4), a (.wire 122 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 122 5) = band (a (.virt 292)) (a (.wire 122 5)) ∧
    a (.wire 122 5) = band (a (.wire 122 5)) (a (.virt 292)) ∧
    a (.wire 122 5) = bor (a (.virt 14)) (a (.wire 122 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 122 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 122 5) = a (.virt 14) + a (.wire 122 5) ∧
    a (.wire 122 5) = a (.wire 122 5) - a (.virt 14) ∧
    a (.wire 121 35) = bnot (a (.wire 122 5)) ∧
    a (.wire 121 35) = band (a (.virt 292)) (a (.wire 121 35)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 122 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 121 35)) ∧
    a (.wire 122 5) = bor (a (.wire 122 5)) (a (.virt 14)) ∧
    a (.wire 122 4) = a (.virt 292) * a (.wire 122 4) ∧
    a (.wire 114 15) = a (.virt 294) * a (.wire 122 4) ∧
    a (.wire 111 47) = a (.virt 292) + a (.wire 122 4) ∧
    a (.wire 121 39) = a (.wire 111 47) - a (.wire 114 15) ∧
    a (.wire 121 43) = bnot (a (.wire 121 39)) ∧
    a (.wire 114 19) = band (a (.wire 121 35)) (a (.wire 121 43)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 122 3) = band (a (.virt 292)) (a (.wire 122 3)) ∧
    a (.wire 114 23) = band (a (.wire 122 3)) (a (.wire 114 19)) ∧
    a (.wire 111 51) = bor (a (.wire 122 5)) (a (.wire 114 23)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 122 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 122 3) = a (.virt 14) + a (.wire 122 3) ∧
    a (.wire 122 3) = a (.wire 122 3) - a (.virt 14) ∧
    a (.wire 121 47) = bnot (a (.wire 122 3)) ∧
    a (.wire 114 27) = band (a (.wire 114 19)) (a (.wire 121 47))) ∧
    (a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 122 2) = band (a (.virt 292)) (a (.wire 122 2)) ∧
    a (.wire 114 31) = band (a (.wire 122 2)) (a (.wire 114 27)) ∧
    a (.wire 111 55) = bor (a (.wire 111 51)) (a (.wire 114 31)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 122 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 122 2) = a (.virt 14) + a (.wire 122 2) ∧
    a (.wire 122 2) = a (.wire 122 2) - a (.virt 14) ∧
    a (.wire 121 51) = bnot (a (.wire 122 2)) ∧
    a (.wire 114 35) = band (a (.wire 114 27)) (a (.wire 121 51)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 122 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 114 35)) ∧
    a (.wire 111 55) = bor (a (.wire 111 55)) (a (.virt 14)) ∧
    a (.wire 122 1) = a (.virt 292) * a (.wire 122 1) ∧
    a (.wire 114 39) = a (.virt 294) * a (.wire 122 1) ∧
    a (.wire 111 59) = a (.virt 292) + a (.wire 122 1) ∧
    a (.wire 121 55) = a (.wire 111 59) - a (.wire 114 39) ∧
    a (.wire 121 59) = bnot (a (.wire 121 55)) ∧
    a (.wire 114 43) = band (a (.wire 114 35)) (a (.wire 121 59)) ∧
    rangeCheck (a (.virt 216)) 2 ∧
    IsEqual (a (.virt 216)) (a (.virt 14)) (a (.virt 368)) (a (.virt 369)) ∧
    IsEqual (a (.virt 216)) (a (.virt 292)) (a (.virt 370)) (a (.virt 371)) ∧
    IsEqual (a (.virt 216)) (a (.virt 294)) (a (.virt 372)) (a (.virt 373)) ∧
    IsEqual (a (.virt 216)) (a (.virt 295)) (a (.virt 374)) (a (.virt 375)) ∧
    a (.wire 124 51) = bselect (a (.virt 368)) (a (.wire 121 7)) (a (.virt 123)) ∧
    a (.wire 124 59) = bselect (a (.virt 368)) (a (.wire 121 15)) (a (.virt 124)) ∧
    a (.wire 126 7) = bselect (a (.virt 368)) (a (.wire 121 23)) (a (.virt 125)) ∧
    a (.wire 126 15) = bselect (a (.virt 368)) (a (.wire 121 31)) (a (.virt 126)) ∧
    a (.wire 126 23) = bselect (a (.virt 368)) (a (.virt 123)) (a (.virt 127)) ∧
    a (.wire 126 31) = bselect (a (.virt 370)) (a (.wire 121 7)) (a (.wire 126 23)) ∧
    a (.wire 126 39) = bselect (a (.virt 368)) (a (.virt 124)) (a (.virt 128))) ∧
    (a (.wire 126 47) = bselect (a (.virt 370)) (a (.wire 121 15)) (a (.wire 126 39)) ∧
    a (.wire 126 55) = bselect (a (.virt 368)) (a (.virt 125)) (a (.virt 129)) ∧
    a (.wire 127 3) = bselect (a (.virt 370)) (a (.wire 121 23)) (a (.wire 126 55)) ∧
    a (.wire 127 11) = bselect (a (.virt 368)) (a (.virt 126)) (a (.virt 130)) ∧
    a (.wire 127 19) = bselect (a (.virt 370)) (a (.wire 121 31)) (a (.wire 127 11)) ∧
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) ∧
    a (.wire 127 27) = bselect (a (.wire 128 3)) (a (.virt 127)) (a (.virt 131)) ∧
    a (.wire 127 35) = bselect (a (.virt 372)) (a (.wire 121 7)) (a (.wire 127 27)) ∧
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) ∧
    a (.wire 127 43) = bselect (a (.wire 128 3)) (a (.virt 128)) (a (.virt 132)) ∧
    a (.wire 127 51) = bselect (a (.virt 372)) (a (.wire 121 15)) (a (.wire 127 43)) ∧
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) ∧
    a (.wire 127 59) = bselect (a (.wire 128 3)) (a (.virt 129)) (a (.virt 133)) ∧
    a (.wire 129 7) = bselect (a (.virt 372)) (a (.wire 121 23)) (a (.wire 127 59)) ∧
    a (.wire 128 3) = bor (a (.virt 368)) (a (.virt 370)) ∧
    a (.wire 129 15) = bselect (a (.wire 128 3)) (a (.virt 130)) (a (.virt 134)) ∧
    a (.wire 129 23) = bselect (a (.virt 372)) (a (.wire 121 31)) (a (.wire 129 15)) ∧
    a (.wire 129 31) = bselect (a (.virt 374)) (a (.wire 121 7)) (a (.virt 131)) ∧
    a (.wire 129 39) = bselect (a (.virt 374)) (a (.wire 121 15)) (a (.virt 132)) ∧
    a (.wire 129 47) = bselect (a (.virt 374)) (a (.wire 121 23)) (a (.virt 133)) ∧
    a (.wire 129 55) = bselect (a (.virt 374)) (a (.wire 121 31)) (a (.virt 134)) ∧
    (a (.wire 132 12) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 0 ∧ a (.wire 132 13) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 1 ∧ a (.wire 132 14) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 2 ∧ a (.wire 132 15) = spongeHash perm [a (.wire 124 51), a (.wire 124 59), a (.wire 126 7), a (.wire 126 15), a (.wire 126 31), a (.wire 126 47), a (.wire 127 3), a (.wire 127 19), a (.wire 127 35), a (.wire 127 51), a (.wire 129 7), a (.wire 129 23), a (.wire 129 31), a (.wire 129 39), a (.wire 129 47), a (.wire 129 55)] 3) ∧
    a (.wire 133 3) = bselect (a (.wire 111 55)) (a (.wire 132 12)) (a (.wire 121 7)) ∧
    a (.wire 133 11) = bselect (a (.wire 111 55)) (a (.wire 132 13)) (a (.wire 121 15)) ∧
    a (.wire 133 19) = bselect (a (.wire 111 55)) (a (.wire 132 14)) (a (.wire 121 23)) ∧
    a (.wire 133 27) = bselect (a (.wire 111 55)) (a (.wire 132 15)) (a (.wire 121 31)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 134 1), a (.wire 134 2), a (.wire 134 3), a (.wire 134 4), a (.wire 134 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 134 5) = band (a (.virt 292)) (a (.wire 134 5)) ∧
    a (.wire 134 5) = band (a (.wire 134 5)) (a (.virt 292)) ∧
    a (.wire 134 5) = bor (a (.virt 14)) (a (.wire 134 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 134 5)) ∧
    (a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 134 5) = a (.virt 14) + a (.wire 134 5) ∧
    a (.wire 134 5) = a (.wire 134 5) - a (.virt 14) ∧
    a (.wire 133 31) = bnot (a (.wire 134 5)) ∧
    a (.wire 133 31) = band (a (.virt 292)) (a (.wire 133 31)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 134 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 133 31)) ∧
    a (.wire 134 5) = bor (a (.wire 134 5)) (a (.virt 14)) ∧
    a (.wire 134 4) = a (.virt 292) * a (.wire 134 4) ∧
    a (.wire 125 19) = a (.virt 294) * a (.wire 134 4) ∧
    a (.wire 128 43) = a (.virt 292) + a (.wire 134 4) ∧
    a (.wire 133 35) = a (.wire 128 43) - a (.wire 125 19) ∧
    a (.wire 133 39) = bnot (a (.wire 133 35)) ∧
    a (.wire 125 23) = band (a (.wire 133 31)) (a (.wire 133 39)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 134 3) = band (a (.virt 292)) (a (.wire 134 3)) ∧
    a (.wire 125 27) = band (a (.wire 134 3)) (a (.wire 125 23)) ∧
    a (.wire 128 47) = bor (a (.wire 134 5)) (a (.wire 125 27)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 134 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 134 3) = a (.virt 14) + a (.wire 134 3) ∧
    a (.wire 134 3) = a (.wire 134 3) - a (.virt 14) ∧
    a (.wire 133 43) = bnot (a (.wire 134 3)) ∧
    a (.wire 125 31) = band (a (.wire 125 23)) (a (.wire 133 43)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 134 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 125 31)) ∧
    a (.wire 128 47) = bor (a (.wire 128 47)) (a (.virt 14)) ∧
    a (.wire 134 2) = a (.virt 292) * a (.wire 134 2) ∧
    a (.wire 125 35) = a (.virt 294) * a (.wire 134 2) ∧
    a (.wire 128 51) = a (.virt 292) + a (.wire 134 2)) ∧
    (a (.wire 133 47) = a (.wire 128 51) - a (.wire 125 35) ∧
    a (.wire 133 51) = bnot (a (.wire 133 47)) ∧
    a (.wire 125 39) = band (a (.wire 125 31)) (a (.wire 133 51)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 134 1) = band (a (.virt 292)) (a (.wire 134 1)) ∧
    a (.wire 125 43) = band (a (.wire 134 1)) (a (.wire 125 39)) ∧
    a (.wire 128 55) = bor (a (.wire 128 47)) (a (.wire 125 43)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 134 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 134 1) = a (.virt 14) + a (.wire 134 1) ∧
    a (.wire 134 1) = a (.wire 134 1) - a (.virt 14) ∧
    a (.wire 133 55) = bnot (a (.wire 134 1)) ∧
    a (.wire 125 47) = band (a (.wire 125 39)) (a (.wire 133 55)) ∧
    rangeCheck (a (.virt 217)) 2 ∧
    IsEqual (a (.virt 217)) (a (.virt 14)) (a (.virt 376)) (a (.virt 377)) ∧
    IsEqual (a (.virt 217)) (a (.virt 292)) (a (.virt 378)) (a (.virt 379)) ∧
    IsEqual (a (.virt 217)) (a (.virt 294)) (a (.virt 380)) (a (.virt 381)) ∧
    IsEqual (a (.virt 217)) (a (.virt 295)) (a (.virt 382)) (a (.virt 383)) ∧
    a (.wire 136 47) = bselect (a (.virt 376)) (a (.wire 133 3)) (a (.virt 135)) ∧
    a (.wire 136 55) = bselect (a (.virt 376)) (a (.wire 133 11)) (a (.virt 136)) ∧
    a (.wire 138 3) = bselect (a (.virt 376)) (a (.wire 133 19)) (a (.virt 137)) ∧
    a (.wire 138 11) = bselect (a (.virt 376)) (a (.wire 133 27)) (a (.virt 138)) ∧
    a (.wire 138 19) = bselect (a (.virt 376)) (a (.virt 135)) (a (.virt 139)) ∧
    a (.wire 138 27) = bselect (a (.virt 378)) (a (.wire 133 3)) (a (.wire 138 19)) ∧
    a (.wire 138 35) = bselect (a (.virt 376)) (a (.virt 136)) (a (.virt 140)) ∧
    a (.wire 138 43) = bselect (a (.virt 378)) (a (.wire 133 11)) (a (.wire 138 35)) ∧
    a (.wire 138 51) = bselect (a (.virt 376)) (a (.virt 137)) (a (.virt 141)) ∧
    a (.wire 138 59) = bselect (a (.virt 378)) (a (.wire 133 19)) (a (.wire 138 51)) ∧
    a (.wire 139 7) = bselect (a (.virt 376)) (a (.virt 138)) (a (.virt 142)) ∧
    a (.wire 139 15) = bselect (a (.virt 378)) (a (.wire 133 27)) (a (.wire 139 7)) ∧
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) ∧
    a (.wire 139 23) = bselect (a (.wire 128 59)) (a (.virt 139)) (a (.virt 143)))) ∧
    ((a (.wire 139 31) = bselect (a (.virt 380)) (a (.wire 133 3)) (a (.wire 139 23)) ∧
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) ∧
    a (.wire 139 39) = bselect (a (.wire 128 59)) (a (.virt 140)) (a (.virt 144)) ∧
    a (.wire 139 47) = bselect (a (.virt 380)) (a (.wire 133 11)) (a (.wire 139 39)) ∧
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) ∧
    a (.wire 139 55) = bselect (a (.wire 128 59)) (a (.virt 141)) (a (.virt 145)) ∧
    a (.wire 140 3) = bselect (a (.virt 380)) (a (.wire 133 19)) (a (.wire 139 55)) ∧
    a (.wire 128 59) = bor (a (.virt 376)) (a (.virt 378)) ∧
    a (.wire 140 11) = bselect (a (.wire 128 59)) (a (.virt 142)) (a (.virt 146)) ∧
    a (.wire 140 19) = bselect (a (.virt 380)) (a (.wire 133 27)) (a (.wire 140 11)) ∧
    a (.wire 140 27) = bselect (a (.virt 382)) (a (.wire 133 3)) (a (.virt 143)) ∧
    a (.wire 140 35) = bselect (a (.virt 382)) (a (.wire 133 11)) (a (.virt 144)) ∧
    a (.wire 140 43) = bselect (a (.virt 382)) (a (.wire 133 19)) (a (.virt 145)) ∧
    a (.wire 140 51) = bselect (a (.virt 382)) (a (.wire 133 27)) (a (.virt 146)) ∧
    (a (.wire 144 12) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 0 ∧ a (.wire 144 13) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 1 ∧ a (.wire 144 14) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 2 ∧ a (.wire 144 15) = spongeHash perm [a (.wire 136 47), a (.wire 136 55), a (.wire 138 3), a (.wire 138 11), a (.wire 138 27), a (.wire 138 43), a (.wire 138 59), a (.wire 139 15), a (.wire 139 31), a (.wire 139 47), a (.wire 140 3), a (.wire 140 19), a (.wire 140 27), a (.wire 140 35), a (.wire 140 43), a (.wire 140 51)] 3) ∧
    a (.wire 140 59) = bselect (a (.wire 128 55)) (a (.wire 144 12)) (a (.wire 133 3)) ∧
    a (.wire 145 7) = bselect (a (.wire 128 55)) (a (.wire 144 13)) (a (.wire 133 11)) ∧
    a (.wire 145 15) = bselect (a (.wire 128 55)) (a (.wire 144 14)) (a (.wire 133 19)) ∧
    a (.wire 145 23) = bselect (a (.wire 128 55)) (a (.wire 144 15)) (a (.wire 133 27)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 146 1), a (.wire 146 2), a (.wire 146 3), a (.wire 146 4), a (.wire 146 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 146 5) = band (a (.virt 292)) (a (.wire 146 5)) ∧
    a (.wire 146 5) = band (a (.wire 146 5)) (a (.virt 292)) ∧
    a (.wire 146 5) = bor (a (.virt 14)) (a (.wire 146 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 146 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 146 5) = a (.virt 14) + a (.wire 146 5) ∧
    a (.wire 146 5) = a (.wire 146 5) - a (.virt 14) ∧
    a (.wire 145 27) = bnot (a (.wire 146 5)) ∧
    a (.wire 145 27) = band (a (.virt 292)) (a (.wire 145 27)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 146 4))) ∧
    (a (.virt 14) = band (a (.virt 14)) (a (.wire 145 27)) ∧
    a (.wire 146 5) = bor (a (.wire 146 5)) (a (.virt 14)) ∧
    a (.wire 146 4) = a (.virt 292) * a (.wire 146 4) ∧
    a (.wire 137 23) = a (.virt 294) * a (.wire 146 4) ∧
    a (.wire 142 39) = a (.virt 292) + a (.wire 146 4) ∧
    a (.wire 145 31) = a (.wire 142 39) - a (.wire 137 23) ∧
    a (.wire 145 35) = bnot (a (.wire 145 31)) ∧
    a (.wire 137 27) = band (a (.wire 145 27)) (a (.wire 145 35)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 146 3) = band (a (.virt 292)) (a (.wire 146 3)) ∧
    a (.wire 137 31) = band (a (.wire 146 3)) (a (.wire 137 27)) ∧
    a (.wire 142 43) = bor (a (.wire 146 5)) (a (.wire 137 31)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 146 3) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 146 3) = a (.virt 14) + a (.wire 146 3) ∧
    a (.wire 146 3) = a (.wire 146 3) - a (.virt 14) ∧
    a (.wire 145 39) = bnot (a (.wire 146 3)) ∧
    a (.wire 137 35) = band (a (.wire 137 27)) (a (.wire 145 39)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 146 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 137 35)) ∧
    a (.wire 142 43) = bor (a (.wire 142 43)) (a (.virt 14)) ∧
    a (.wire 146 2) = a (.virt 292) * a (.wire 146 2) ∧
    a (.wire 137 39) = a (.virt 294) * a (.wire 146 2) ∧
    a (.wire 142 47) = a (.virt 292) + a (.wire 146 2) ∧
    a (.wire 145 43) = a (.wire 142 47) - a (.wire 137 39) ∧
    a (.wire 145 47) = bnot (a (.wire 145 43)) ∧
    a (.wire 137 43) = band (a (.wire 137 35)) (a (.wire 145 47)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 146 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 137 43)) ∧
    a (.wire 142 43) = bor (a (.wire 142 43)) (a (.virt 14))) ∧
    (a (.wire 146 1) = a (.virt 292) * a (.wire 146 1) ∧
    a (.wire 137 47) = a (.virt 294) * a (.wire 146 1) ∧
    a (.wire 142 51) = a (.virt 292) + a (.wire 146 1) ∧
    a (.wire 145 51) = a (.wire 142 51) - a (.wire 137 47) ∧
    a (.wire 145 55) = bnot (a (.wire 145 51)) ∧
    a (.wire 137 51) = band (a (.wire 137 43)) (a (.wire 145 55)) ∧
    rangeCheck (a (.virt 218)) 2 ∧
    IsEqual (a (.virt 218)) (a (.virt 14)) (a (.virt 384)) (a (.virt 385)) ∧
    IsEqual (a (.virt 218)) (a (.virt 292)) (a (.virt 386)) (a (.virt 387)) ∧
    IsEqual (a (.virt 218)) (a (.virt 294)) (a (.virt 388)) (a (.virt 389)) ∧
    IsEqual (a (.virt 218)) (a (.virt 295)) (a (.virt 390)) (a (.virt 391)) ∧
    a (.wire 148 47) = bselect (a (.virt 384)) (a (.wire 140 59)) (a (.virt 147)) ∧
    a (.wire 148 55) = bselect (a (.virt 384)) (a (.wire 145 7)) (a (.virt 148)) ∧
    a (.wire 150 3) = bselect (a (.virt 384)) (a (.wire 145 15)) (a (.virt 149)) ∧
    a (.wire 150 11) = bselect (a (.virt 384)) (a (.wire 145 23)) (a (.virt 150)) ∧
    a (.wire 150 19) = bselect (a (.virt 384)) (a (.virt 147)) (a (.virt 151)) ∧
    a (.wire 150 27) = bselect (a (.virt 386)) (a (.wire 140 59)) (a (.wire 150 19)) ∧
    a (.wire 150 35) = bselect (a (.virt 384)) (a (.virt 148)) (a (.virt 152)) ∧
    a (.wire 150 43) = bselect (a (.virt 386)) (a (.wire 145 7)) (a (.wire 150 35)) ∧
    a (.wire 150 51) = bselect (a (.virt 384)) (a (.virt 149)) (a (.virt 153)) ∧
    a (.wire 150 59) = bselect (a (.virt 386)) (a (.wire 145 15)) (a (.wire 150 51)) ∧
    a (.wire 151 7) = bselect (a (.virt 384)) (a (.virt 150)) (a (.virt 154)) ∧
    a (.wire 151 15) = bselect (a (.virt 386)) (a (.wire 145 23)) (a (.wire 151 7)) ∧
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) ∧
    a (.wire 151 23) = bselect (a (.wire 142 55)) (a (.virt 151)) (a (.virt 155)) ∧
    a (.wire 151 31) = bselect (a (.virt 388)) (a (.wire 140 59)) (a (.wire 151 23)) ∧
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) ∧
    a (.wire 151 39) = bselect (a (.wire 142 55)) (a (.virt 152)) (a (.virt 156)) ∧
    a (.wire 151 47) = bselect (a (.virt 388)) (a (.wire 145 7)) (a (.wire 151 39)) ∧
    a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) ∧
    a (.wire 151 55) = bselect (a (.wire 142 55)) (a (.virt 153)) (a (.virt 157)) ∧
    a (.wire 152 3) = bselect (a (.virt 388)) (a (.wire 145 15)) (a (.wire 151 55))) ∧
    (a (.wire 142 55) = bor (a (.virt 384)) (a (.virt 386)) ∧
    a (.wire 152 11) = bselect (a (.wire 142 55)) (a (.virt 154)) (a (.virt 158)) ∧
    a (.wire 152 19) = bselect (a (.virt 388)) (a (.wire 145 23)) (a (.wire 152 11)) ∧
    a (.wire 152 27) = bselect (a (.virt 390)) (a (.wire 140 59)) (a (.virt 155)) ∧
    a (.wire 152 35) = bselect (a (.virt 390)) (a (.wire 145 7)) (a (.virt 156)) ∧
    a (.wire 152 43) = bselect (a (.virt 390)) (a (.wire 145 15)) (a (.virt 157)) ∧
    a (.wire 152 51) = bselect (a (.virt 390)) (a (.wire 145 23)) (a (.virt 158)) ∧
    (a (.wire 156 12) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 0 ∧ a (.wire 156 13) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 1 ∧ a (.wire 156 14) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 2 ∧ a (.wire 156 15) = spongeHash perm [a (.wire 148 47), a (.wire 148 55), a (.wire 150 3), a (.wire 150 11), a (.wire 150 27), a (.wire 150 43), a (.wire 150 59), a (.wire 151 15), a (.wire 151 31), a (.wire 151 47), a (.wire 152 3), a (.wire 152 19), a (.wire 152 27), a (.wire 152 35), a (.wire 152 43), a (.wire 152 51)] 3) ∧
    a (.wire 152 59) = bselect (a (.wire 142 43)) (a (.wire 156 12)) (a (.wire 140 59)) ∧
    a (.wire 157 7) = bselect (a (.wire 142 43)) (a (.wire 156 13)) (a (.wire 145 7)) ∧
    a (.wire 157 15) = bselect (a (.wire 142 43)) (a (.wire 156 14)) (a (.wire 145 15)) ∧
    a (.wire 157 23) = bselect (a (.wire 142 43)) (a (.wire 156 15)) (a (.wire 145 23)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 158 1), a (.wire 158 2), a (.wire 158 3), a (.wire 158 4), a (.wire 158 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 158 5) = band (a (.virt 292)) (a (.wire 158 5)) ∧
    a (.wire 158 5) = band (a (.wire 158 5)) (a (.virt 292)) ∧
    a (.wire 158 5) = bor (a (.virt 14)) (a (.wire 158 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 158 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 158 5) = a (.virt 14) + a (.wire 158 5) ∧
    a (.wire 158 5) = a (.wire 158 5) - a (.virt 14) ∧
    a (.wire 157 27) = bnot (a (.wire 158 5)) ∧
    a (.wire 157 27) = band (a (.virt 292)) (a (.wire 157 27)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 158 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 157 27)) ∧
    a (.wire 158 5) = bor (a (.wire 158 5)) (a (.virt 14)) ∧
    a (.wire 158 4) = a (.virt 292) * a (.wire 158 4) ∧
    a (.wire 149 27) = a (.virt 294) * a (.wire 158 4) ∧
    a (.wire 154 35) = a (.virt 292) + a (.wire 158 4) ∧
    a (.wire 157 31) = a (.wire 154 35) - a (.wire 149 27) ∧
    a (.wire 157 35) = bnot (a (.wire 157 31))) ∧
    (a (.wire 149 31) = band (a (.wire 157 27)) (a (.wire 157 35)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 158 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 149 31)) ∧
    a (.wire 158 5) = bor (a (.wire 158 5)) (a (.virt 14)) ∧
    a (.wire 158 3) = a (.virt 292) * a (.wire 158 3) ∧
    a (.wire 149 35) = a (.virt 294) * a (.wire 158 3) ∧
    a (.wire 154 39) = a (.virt 292) + a (.wire 158 3) ∧
    a (.wire 157 39) = a (.wire 154 39) - a (.wire 149 35) ∧
    a (.wire 157 43) = bnot (a (.wire 157 39)) ∧
    a (.wire 149 39) = band (a (.wire 149 31)) (a (.wire 157 43)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 158 2) = band (a (.virt 292)) (a (.wire 158 2)) ∧
    a (.wire 149 43) = band (a (.wire 158 2)) (a (.wire 149 39)) ∧
    a (.wire 154 43) = bor (a (.wire 158 5)) (a (.wire 149 43)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 158 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 158 2) = a (.virt 14) + a (.wire 158 2) ∧
    a (.wire 158 2) = a (.wire 158 2) - a (.virt 14) ∧
    a (.wire 157 47) = bnot (a (.wire 158 2)) ∧
    a (.wire 149 47) = band (a (.wire 149 39)) (a (.wire 157 47)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 158 1) = band (a (.virt 292)) (a (.wire 158 1)) ∧
    a (.wire 149 51) = band (a (.wire 158 1)) (a (.wire 149 47)) ∧
    a (.wire 154 47) = bor (a (.wire 154 43)) (a (.wire 149 51)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 158 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 158 1) = a (.virt 14) + a (.wire 158 1) ∧
    a (.wire 158 1) = a (.wire 158 1) - a (.virt 14) ∧
    a (.wire 157 51) = bnot (a (.wire 158 1)) ∧
    a (.wire 149 55) = band (a (.wire 149 47)) (a (.wire 157 51)) ∧
    rangeCheck (a (.virt 219)) 2) ∧
    (IsEqual (a (.virt 219)) (a (.virt 14)) (a (.virt 392)) (a (.virt 393)) ∧
    IsEqual (a (.virt 219)) (a (.virt 292)) (a (.virt 394)) (a (.virt 395)) ∧
    IsEqual (a (.virt 219)) (a (.virt 294)) (a (.virt 396)) (a (.virt 397)) ∧
    IsEqual (a (.virt 219)) (a (.virt 295)) (a (.virt 398)) (a (.virt 399)) ∧
    a (.wire 161 43) = bselect (a (.virt 392)) (a (.wire 152 59)) (a (.virt 159)) ∧
    a (.wire 161 51) = bselect (a (.virt 392)) (a (.wire 157 7)) (a (.virt 160)) ∧
    a (.wire 161 59) = bselect (a (.virt 392)) (a (.wire 157 15)) (a (.virt 161)) ∧
    a (.wire 162 7) = bselect (a (.virt 392)) (a (.wire 157 23)) (a (.virt 162)) ∧
    a (.wire 162 15) = bselect (a (.virt 392)) (a (.virt 159)) (a (.virt 163)) ∧
    a (.wire 162 23) = bselect (a (.virt 394)) (a (.wire 152 59)) (a (.wire 162 15)) ∧
    a (.wire 162 31) = bselect (a (.virt 392)) (a (.virt 160)) (a (.virt 164)) ∧
    a (.wire 162 39) = bselect (a (.virt 394)) (a (.wire 157 7)) (a (.wire 162 31)) ∧
    a (.wire 162 47) = bselect (a (.virt 392)) (a (.virt 161)) (a (.virt 165)) ∧
    a (.wire 162 55) = bselect (a (.virt 394)) (a (.wire 157 15)) (a (.wire 162 47)) ∧
    a (.wire 163 3) = bselect (a (.virt 392)) (a (.virt 162)) (a (.virt 166)) ∧
    a (.wire 163 11) = bselect (a (.virt 394)) (a (.wire 157 23)) (a (.wire 163 3)) ∧
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) ∧
    a (.wire 163 19) = bselect (a (.wire 154 51)) (a (.virt 163)) (a (.virt 167)) ∧
    a (.wire 163 27) = bselect (a (.virt 396)) (a (.wire 152 59)) (a (.wire 163 19)) ∧
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) ∧
    a (.wire 163 35) = bselect (a (.wire 154 51)) (a (.virt 164)) (a (.virt 168)) ∧
    a (.wire 163 43) = bselect (a (.virt 396)) (a (.wire 157 7)) (a (.wire 163 35)) ∧
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) ∧
    a (.wire 163 51) = bselect (a (.wire 154 51)) (a (.virt 165)) (a (.virt 169)) ∧
    a (.wire 163 59) = bselect (a (.virt 396)) (a (.wire 157 15)) (a (.wire 163 51)) ∧
    a (.wire 154 51) = bor (a (.virt 392)) (a (.virt 394)) ∧
    a (.wire 165 7) = bselect (a (.wire 154 51)) (a (.virt 166)) (a (.virt 170)) ∧
    a (.wire 165 15) = bselect (a (.virt 396)) (a (.wire 157 23)) (a (.wire 165 7)) ∧
    a (.wire 165 23) = bselect (a (.virt 398)) (a (.wire 152 59)) (a (.virt 167)) ∧
    a (.wire 165 31) = bselect (a (.virt 398)) (a (.wire 157 7)) (a (.virt 168)) ∧
    a (.wire 165 39) = bselect (a (.virt 398)) (a (.wire 157 15)) (a (.virt 169)) ∧
    a (.wire 165 47) = bselect (a (.virt 398)) (a (.wire 157 23)) (a (.virt 170))) ∧
    ((a (.wire 169 12) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 0 ∧ a (.wire 169 13) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 1 ∧ a (.wire 169 14) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 2 ∧ a (.wire 169 15) = spongeHash perm [a (.wire 161 43), a (.wire 161 51), a (.wire 161 59), a (.wire 162 7), a (.wire 162 23), a (.wire 162 39), a (.wire 162 55), a (.wire 163 11), a (.wire 163 27), a (.wire 163 43), a (.wire 163 59), a (.wire 165 15), a (.wire 165 23), a (.wire 165 31), a (.wire 165 39), a (.wire 165 47)] 3) ∧
    a (.wire 165 55) = bselect (a (.wire 154 47)) (a (.wire 169 12)) (a (.wire 152 59)) ∧
    a (.wire 170 3) = bselect (a (.wire 154 47)) (a (.wire 169 13)) (a (.wire 157 7)) ∧
    a (.wire 170 11) = bselect (a (.wire 154 47)) (a (.wire 169 14)) (a (.wire 157 15)) ∧
    a (.wire 170 19) = bselect (a (.wire 154 47)) (a (.wire 169 15)) (a (.wire 157 23)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 171 1), a (.wire 171 2), a (.wire 171 3), a (.wire 171 4), a (.wire 171 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 171 5) = band (a (.virt 292)) (a (.wire 171 5)) ∧
    a (.wire 171 5) = band (a (.wire 171 5)) (a (.virt 292)) ∧
    a (.wire 171 5) = bor (a (.virt 14)) (a (.wire 171 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 171 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 171 5) = a (.virt 14) + a (.wire 171 5) ∧
    a (.wire 171 5) = a (.wire 171 5) - a (.virt 14) ∧
    a (.wire 170 23) = bnot (a (.wire 171 5)) ∧
    a (.wire 170 23) = band (a (.virt 292)) (a (.wire 170 23)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 171 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 170 23)) ∧
    a (.wire 171 5) = bor (a (.wire 171 5)) (a (.virt 14)) ∧
    a (.wire 171 4) = a (.virt 292) * a (.wire 171 4) ∧
    a (.wire 160 31) = a (.virt 294) * a (.wire 171 4) ∧
    a (.wire 167 31) = a (.virt 292) + a (.wire 171 4) ∧
    a (.wire 170 27) = a (.wire 167 31) - a (.wire 160 31) ∧
    a (.wire 170 31) = bnot (a (.wire 170 27)) ∧
    a (.wire 160 35) = band (a (.wire 170 23)) (a (.wire 170 31)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 171 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 160 35)) ∧
    a (.wire 171 5) = bor (a (.wire 171 5)) (a (.virt 14)) ∧
    a (.wire 171 3) = a (.virt 292) * a (.wire 171 3) ∧
    a (.wire 160 39) = a (.virt 294) * a (.wire 171 3)) ∧
    (a (.wire 167 35) = a (.virt 292) + a (.wire 171 3) ∧
    a (.wire 170 35) = a (.wire 167 35) - a (.wire 160 39) ∧
    a (.wire 170 39) = bnot (a (.wire 170 35)) ∧
    a (.wire 160 43) = band (a (.wire 160 35)) (a (.wire 170 39)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 171 2) = band (a (.virt 292)) (a (.wire 171 2)) ∧
    a (.wire 160 47) = band (a (.wire 171 2)) (a (.wire 160 43)) ∧
    a (.wire 167 39) = bor (a (.wire 171 5)) (a (.wire 160 47)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 171 2) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 171 2) = a (.virt 14) + a (.wire 171 2) ∧
    a (.wire 171 2) = a (.wire 171 2) - a (.virt 14) ∧
    a (.wire 170 43) = bnot (a (.wire 171 2)) ∧
    a (.wire 160 51) = band (a (.wire 160 43)) (a (.wire 170 43)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 171 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 160 51)) ∧
    a (.wire 167 39) = bor (a (.wire 167 39)) (a (.virt 14)) ∧
    a (.wire 171 1) = a (.virt 292) * a (.wire 171 1) ∧
    a (.wire 160 55) = a (.virt 294) * a (.wire 171 1) ∧
    a (.wire 167 43) = a (.virt 292) + a (.wire 171 1) ∧
    a (.wire 170 47) = a (.wire 167 43) - a (.wire 160 55) ∧
    a (.wire 170 51) = bnot (a (.wire 170 47)) ∧
    a (.wire 160 59) = band (a (.wire 160 51)) (a (.wire 170 51)) ∧
    rangeCheck (a (.virt 220)) 2 ∧
    IsEqual (a (.virt 220)) (a (.virt 14)) (a (.virt 400)) (a (.virt 401)) ∧
    IsEqual (a (.virt 220)) (a (.virt 292)) (a (.virt 402)) (a (.virt 403)) ∧
    IsEqual (a (.virt 220)) (a (.virt 294)) (a (.virt 404)) (a (.virt 405)) ∧
    IsEqual (a (.virt 220)) (a (.virt 295)) (a (.virt 406)) (a (.virt 407)) ∧
    a (.wire 174 43) = bselect (a (.virt 400)) (a (.wire 165 55)) (a (.virt 171)) ∧
    a (.wire 174 51) = bselect (a (.virt 400)) (a (.wire 170 3)) (a (.virt 172)) ∧
    a (.wire 174 59) = bselect (a (.virt 400)) (a (.wire 170 11)) (a (.virt 173))) ∧
    (a (.wire 175 7) = bselect (a (.virt 400)) (a (.wire 170 19)) (a (.virt 174)) ∧
    a (.wire 175 15) = bselect (a (.virt 400)) (a (.virt 171)) (a (.virt 175)) ∧
    a (.wire 175 23) = bselect (a (.virt 402)) (a (.wire 165 55)) (a (.wire 175 15)) ∧
    a (.wire 175 31) = bselect (a (.virt 400)) (a (.virt 172)) (a (.virt 176)) ∧
    a (.wire 175 39) = bselect (a (.virt 402)) (a (.wire 170 3)) (a (.wire 175 31)) ∧
    a (.wire 175 47) = bselect (a (.virt 400)) (a (.virt 173)) (a (.virt 177)) ∧
    a (.wire 175 55) = bselect (a (.virt 402)) (a (.wire 170 11)) (a (.wire 175 47)) ∧
    a (.wire 176 3) = bselect (a (.virt 400)) (a (.virt 174)) (a (.virt 178)) ∧
    a (.wire 176 11) = bselect (a (.virt 402)) (a (.wire 170 19)) (a (.wire 176 3)) ∧
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) ∧
    a (.wire 176 19) = bselect (a (.wire 167 47)) (a (.virt 175)) (a (.virt 179)) ∧
    a (.wire 176 27) = bselect (a (.virt 404)) (a (.wire 165 55)) (a (.wire 176 19)) ∧
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) ∧
    a (.wire 176 35) = bselect (a (.wire 167 47)) (a (.virt 176)) (a (.virt 180)) ∧
    a (.wire 176 43) = bselect (a (.virt 404)) (a (.wire 170 3)) (a (.wire 176 35)) ∧
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) ∧
    a (.wire 176 51) = bselect (a (.wire 167 47)) (a (.virt 177)) (a (.virt 181)) ∧
    a (.wire 176 59) = bselect (a (.virt 404)) (a (.wire 170 11)) (a (.wire 176 51)) ∧
    a (.wire 167 47) = bor (a (.virt 400)) (a (.virt 402)) ∧
    a (.wire 177 7) = bselect (a (.wire 167 47)) (a (.virt 178)) (a (.virt 182)) ∧
    a (.wire 177 15) = bselect (a (.virt 404)) (a (.wire 170 19)) (a (.wire 177 7)) ∧
    a (.wire 177 23) = bselect (a (.virt 406)) (a (.wire 165 55)) (a (.virt 179)) ∧
    a (.wire 177 31) = bselect (a (.virt 406)) (a (.wire 170 3)) (a (.virt 180)) ∧
    a (.wire 177 39) = bselect (a (.virt 406)) (a (.wire 170 11)) (a (.virt 181)) ∧
    a (.wire 177 47) = bselect (a (.virt 406)) (a (.wire 170 19)) (a (.virt 182)) ∧
    (a (.wire 181 12) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 0 ∧ a (.wire 181 13) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 1 ∧ a (.wire 181 14) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 2 ∧ a (.wire 181 15) = spongeHash perm [a (.wire 174 43), a (.wire 174 51), a (.wire 174 59), a (.wire 175 7), a (.wire 175 23), a (.wire 175 39), a (.wire 175 55), a (.wire 176 11), a (.wire 176 27), a (.wire 176 43), a (.wire 176 59), a (.wire 177 15), a (.wire 177 23), a (.wire 177 31), a (.wire 177 39), a (.wire 177 47)] 3) ∧
    a (.wire 177 55) = bselect (a (.wire 167 39)) (a (.wire 181 12)) (a (.wire 165 55)) ∧
    a (.wire 182 3) = bselect (a (.wire 167 39)) (a (.wire 181 13)) (a (.wire 170 3)) ∧
    a (.wire 182 11) = bselect (a (.wire 167 39)) (a (.wire 181 14)) (a (.wire 170 11)) ∧
    a (.wire 182 19) = bselect (a (.wire 167 39)) (a (.wire 181 15)) (a (.wire 170 19)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 183 1), a (.wire 183 2), a (.wire 183 3), a (.wire 183 4), a (.wire 183 5)] ∧
    a (.virt 292) = bnot (a (.virt 14))) ∧
    (a (.wire 183 5) = band (a (.virt 292)) (a (.wire 183 5)) ∧
    a (.wire 183 5) = band (a (.wire 183 5)) (a (.virt 292)) ∧
    a (.wire 183 5) = bor (a (.virt 14)) (a (.wire 183 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 183 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 183 5) = a (.virt 14) + a (.wire 183 5) ∧
    a (.wire 183 5) = a (.wire 183 5) - a (.virt 14) ∧
    a (.wire 182 23) = bnot (a (.wire 183 5)) ∧
    a (.wire 182 23) = band (a (.virt 292)) (a (.wire 182 23)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 183 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 182 23)) ∧
    a (.wire 183 5) = bor (a (.wire 183 5)) (a (.virt 14)) ∧
    a (.wire 183 4) = a (.virt 292) * a (.wire 183 4) ∧
    a (.wire 173 35) = a (.virt 294) * a (.wire 183 4) ∧
    a (.wire 179 27) = a (.virt 292) + a (.wire 183 4) ∧
    a (.wire 182 27) = a (.wire 179 27) - a (.wire 173 35) ∧
    a (.wire 182 31) = bnot (a (.wire 182 27)) ∧
    a (.wire 173 39) = band (a (.wire 182 23)) (a (.wire 182 31)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 183 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 173 39)) ∧
    a (.wire 183 5) = bor (a (.wire 183 5)) (a (.virt 14)) ∧
    a (.wire 183 3) = a (.virt 292) * a (.wire 183 3) ∧
    a (.wire 173 43) = a (.virt 294) * a (.wire 183 3) ∧
    a (.wire 179 31) = a (.virt 292) + a (.wire 183 3) ∧
    a (.wire 182 35) = a (.wire 179 31) - a (.wire 173 43) ∧
    a (.wire 182 39) = bnot (a (.wire 182 35)) ∧
    a (.wire 173 47) = band (a (.wire 173 39)) (a (.wire 182 39)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 183 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 173 47))) ∧
    (a (.wire 183 5) = bor (a (.wire 183 5)) (a (.virt 14)) ∧
    a (.wire 183 2) = a (.virt 292) * a (.wire 183 2) ∧
    a (.wire 173 51) = a (.virt 294) * a (.wire 183 2) ∧
    a (.wire 179 35) = a (.virt 292) + a (.wire 183 2) ∧
    a (.wire 182 43) = a (.wire 179 35) - a (.wire 173 51) ∧
    a (.wire 182 47) = bnot (a (.wire 182 43)) ∧
    a (.wire 173 55) = band (a (.wire 173 47)) (a (.wire 182 47)) ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 183 1) = band (a (.virt 292)) (a (.wire 183 1)) ∧
    a (.wire 173 59) = band (a (.wire 183 1)) (a (.wire 173 55)) ∧
    a (.wire 179 39) = bor (a (.wire 183 5)) (a (.wire 173 59)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 183 1) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 183 1) = a (.virt 14) + a (.wire 183 1) ∧
    a (.wire 183 1) = a (.wire 183 1) - a (.virt 14) ∧
    a (.wire 182 51) = bnot (a (.wire 183 1)) ∧
    a (.wire 184 3) = band (a (.wire 173 55)) (a (.wire 182 51)) ∧
    rangeCheck (a (.virt 221)) 2 ∧
    IsEqual (a (.virt 221)) (a (.virt 14)) (a (.virt 408)) (a (.virt 409)) ∧
    IsEqual (a (.virt 221)) (a (.virt 292)) (a (.virt 410)) (a (.virt 411)) ∧
    IsEqual (a (.virt 221)) (a (.virt 294)) (a (.virt 412)) (a (.virt 413)) ∧
    IsEqual (a (.virt 221)) (a (.virt 295)) (a (.virt 414)) (a (.virt 415)) ∧
    a (.wire 186 43) = bselect (a (.virt 408)) (a (.wire 177 55)) (a (.virt 183)) ∧
    a (.wire 186 51) = bselect (a (.virt 408)) (a (.wire 182 3)) (a (.virt 184)) ∧
    a (.wire 186 59) = bselect (a (.virt 408)) (a (.wire 182 11)) (a (.virt 185)) ∧
    a (.wire 187 7) = bselect (a (.virt 408)) (a (.wire 182 19)) (a (.virt 186)) ∧
    a (.wire 187 15) = bselect (a (.virt 408)) (a (.virt 183)) (a (.virt 187)) ∧
    a (.wire 187 23) = bselect (a (.virt 410)) (a (.wire 177 55)) (a (.wire 187 15)) ∧
    a (.wire 187 31) = bselect (a (.virt 408)) (a (.virt 184)) (a (.virt 188)) ∧
    a (.wire 187 39) = bselect (a (.virt 410)) (a (.wire 182 3)) (a (.wire 187 31)) ∧
    a (.wire 187 47) = bselect (a (.virt 408)) (a (.virt 185)) (a (.virt 189)) ∧
    a (.wire 187 55) = bselect (a (.virt 410)) (a (.wire 182 11)) (a (.wire 187 47))) ∧
    (a (.wire 188 3) = bselect (a (.virt 408)) (a (.virt 186)) (a (.virt 190)) ∧
    a (.wire 188 11) = bselect (a (.virt 410)) (a (.wire 182 19)) (a (.wire 188 3)) ∧
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) ∧
    a (.wire 188 19) = bselect (a (.wire 179 43)) (a (.virt 187)) (a (.virt 191)) ∧
    a (.wire 188 27) = bselect (a (.virt 412)) (a (.wire 177 55)) (a (.wire 188 19)) ∧
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) ∧
    a (.wire 188 35) = bselect (a (.wire 179 43)) (a (.virt 188)) (a (.virt 192)) ∧
    a (.wire 188 43) = bselect (a (.virt 412)) (a (.wire 182 3)) (a (.wire 188 35)) ∧
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) ∧
    a (.wire 188 51) = bselect (a (.wire 179 43)) (a (.virt 189)) (a (.virt 193)) ∧
    a (.wire 188 59) = bselect (a (.virt 412)) (a (.wire 182 11)) (a (.wire 188 51)) ∧
    a (.wire 179 43) = bor (a (.virt 408)) (a (.virt 410)) ∧
    a (.wire 189 7) = bselect (a (.wire 179 43)) (a (.virt 190)) (a (.virt 194)) ∧
    a (.wire 189 15) = bselect (a (.virt 412)) (a (.wire 182 19)) (a (.wire 189 7)) ∧
    a (.wire 189 23) = bselect (a (.virt 414)) (a (.wire 177 55)) (a (.virt 191)) ∧
    a (.wire 189 31) = bselect (a (.virt 414)) (a (.wire 182 3)) (a (.virt 192)) ∧
    a (.wire 189 39) = bselect (a (.virt 414)) (a (.wire 182 11)) (a (.virt 193)) ∧
    a (.wire 189 47) = bselect (a (.virt 414)) (a (.wire 182 19)) (a (.virt 194)) ∧
    (a (.wire 193 12) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 0 ∧ a (.wire 193 13) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 1 ∧ a (.wire 193 14) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 2 ∧ a (.wire 193 15) = spongeHash perm [a (.wire 186 43), a (.wire 186 51), a (.wire 186 59), a (.wire 187 7), a (.wire 187 23), a (.wire 187 39), a (.wire 187 55), a (.wire 188 11), a (.wire 188 27), a (.wire 188 43), a (.wire 188 59), a (.wire 189 15), a (.wire 189 23), a (.wire 189 31), a (.wire 189 39), a (.wire 189 47)] 3) ∧
    a (.wire 189 55) = bselect (a (.wire 179 39)) (a (.wire 193 12)) (a (.wire 177 55)) ∧
    a (.wire 194 3) = bselect (a (.wire 179 39)) (a (.wire 193 13)) (a (.wire 182 3)) ∧
    a (.wire 194 11) = bselect (a (.wire 179 39)) (a (.wire 193 14)) (a (.wire 182 11)) ∧
    a (.wire 194 19) = bselect (a (.wire 179 39)) (a (.wire 193 15)) (a (.wire 182 19)) ∧
    BaseSum 2 (a (.virt 12)) [a (.wire 195 1), a (.wire 195 2), a (.wire 195 3), a (.wire 195 4), a (.wire 195 5)] ∧
    a (.virt 292) = bnot (a (.virt 14)) ∧
    a (.wire 195 5) = band (a (.virt 292)) (a (.wire 195 5)) ∧
    a (.wire 195 5) = band (a (.wire 195 5)) (a (.virt 292)) ∧
    a (.wire 195 5) = bor (a (.virt 14)) (a (.wire 195 5)) ∧
    a (.virt 14) = a (.virt 14) * a (.wire 195 5) ∧
    a (.virt 14) = a (.virt 294) * a (.virt 14) ∧
    a (.wire 195 5) = a (.virt 14) + a (.wire 195 5) ∧
    a (.wire 195 5) = a (.wire 195 5) - a (.virt 14)) ∧
    (a (.wire 194 23) = bnot (a (.wire 195 5)) ∧
    a (.wire 194 23) = band (a (.virt 292)) (a (.wire 194 23)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 4)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 194 23)) ∧
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) ∧
    a (.wire 195 4) = a (.virt 292) * a (.wire 195 4) ∧
    a (.wire 184 39) = a (.virt 294) * a (.wire 195 4) ∧
    a (.wire 191 23) = a (.virt 292) + a (.wire 195 4) ∧
    a (.wire 194 27) = a (.wire 191 23) - a (.wire 184 39) ∧
    a (.wire 194 31) = bnot (a (.wire 194 27)) ∧
    a (.wire 184 43) = band (a (.wire 194 23)) (a (.wire 194 31)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 3)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 184 43)) ∧
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) ∧
    a (.wire 195 3) = a (.virt 292) * a (.wire 195 3) ∧
    a (.wire 184 47) = a (.virt 294) * a (.wire 195 3) ∧
    a (.wire 191 27) = a (.virt 292) + a (.wire 195 3) ∧
    a (.wire 194 35) = a (.wire 191 27) - a (.wire 184 47) ∧
    a (.wire 194 39) = bnot (a (.wire 194 35)) ∧
    a (.wire 184 51) = band (a (.wire 184 43)) (a (.wire 194 39)) ∧
    a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 2)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 184 51)) ∧
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) ∧
    a (.wire 195 2) = a (.virt 292) * a (.wire 195 2) ∧
    a (.wire 184 55) = a (.virt 294) * a (.wire 195 2) ∧
    a (.wire 191 31) = a (.virt 292) + a (.wire 195 2) ∧
    a (.wire 194 43) = a (.wire 191 31) - a (.wire 184 55) ∧
    a (.wire 194 47) = bnot (a (.wire 194 43)) ∧
    a (.wire 184 59) = band (a (.wire 184 51)) (a (.wire 194 47))) ∧
    (a (.virt 14) = bnot (a (.virt 292)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 195 1)) ∧
    a (.virt 14) = band (a (.virt 14)) (a (.wire 184 59)) ∧
    a (.wire 195 5) = bor (a (.wire 195 5)) (a (.virt 14)) ∧
    a (.wire 195 1) = a (.virt 292) * a (.wire 195 1) ∧
    a (.wire 196 3) = a (.virt 294) * a (.wire 195 1) ∧
    a (.wire 191 35) = a (.virt 292) + a (.wire 195 1) ∧
    a (.wire 194 51) = a (.wire 191 35) - a (.wire 196 3) ∧
    a (.wire 194 55) = bnot (a (.wire 194 51)) ∧
    a (.wire 196 7) = band (a (.wire 184 59)) (a (.wire 194 55)) ∧
    rangeCheck (a (.virt 222)) 2 ∧
    IsEqual (a (.virt 222)) (a (.virt 14)) (a (.virt 416)) (a (.virt 417)) ∧
    IsEqual (a (.virt 222)) (a (.virt 292)) (a (.virt 418)) (a (.virt 419)) ∧
    IsEqual (a (.virt 222)) (a (.virt 294)) (a (.virt 420)) (a (.virt 421)) ∧
    IsEqual (a (.virt 222)) (a (.virt 295)) (a (.virt 422)) (a (.virt 423)) ∧
    a (.wire 198 47) = bselect (a (.virt 416)) (a (.wire 189 55)) (a (.virt 195)) ∧
    a (.wire 198 55) = bselect (a (.virt 416)) (a (.wire 194 3)) (a (.virt 196)) ∧
    a (.wire 199 3) = bselect (a (.virt 416)) (a (.wire 194 11)) (a (.virt 197)) ∧
    a (.wire 199 11) = bselect (a (.virt 416)) (a (.wire 194 19)) (a (.virt 198)) ∧
    a (.wire 199 19) = bselect (a (.virt 416)) (a (.virt 195)) (a (.virt 199)) ∧
    a (.wire 199 27) = bselect (a (.virt 418)) (a (.wire 189 55)) (a (.wire 199 19)) ∧
    a (.wire 199 35) = bselect (a (.virt 416)) (a (.virt 196)) (a (.virt 200)) ∧
    a (.wire 199 43) = bselect (a (.virt 418)) (a (.wire 194 3)) (a (.wire 199 35)) ∧
    a (.wire 199 51) = bselect (a (.virt 416)) (a (.virt 197)) (a (.virt 201)) ∧
    a (.wire 199 59) = bselect (a (.virt 418)) (a (.wire 194 11)) (a (.wire 199 51)) ∧
    a (.wire 200 7) = bselect (a (.virt 416)) (a (.virt 198)) (a (.virt 202)) ∧
    a (.wire 200 15) = bselect (a (.virt 418)) (a (.wire 194 19)) (a (.wire 200 7)) ∧
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) ∧
    a (.wire 200 23) = bselect (a (.wire 191 39)) (a (.virt 199)) (a (.virt 203)) ∧
    a (.wire 200 31) = bselect (a (.virt 420)) (a (.wire 189 55)) (a (.wire 200 23)) ∧
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) ∧
    a (.wire 200 39) = bselect (a (.wire 191 39)) (a (.virt 200)) (a (.virt 204))) ∧
    (a (.wire 200 47) = bselect (a (.virt 420)) (a (.wire 194 3)) (a (.wire 200 39)) ∧
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) ∧
    a (.wire 200 55) = bselect (a (.wire 191 39)) (a (.virt 201)) (a (.virt 205)) ∧
    a (.wire 201 3) = bselect (a (.virt 420)) (a (.wire 194 11)) (a (.wire 200 55)) ∧
    a (.wire 191 39) = bor (a (.virt 416)) (a (.virt 418)) ∧
    a (.wire 201 11) = bselect (a (.wire 191 39)) (a (.virt 202)) (a (.virt 206)) ∧
    a (.wire 201 19) = bselect (a (.virt 420)) (a (.wire 194 19)) (a (.wire 201 11)) ∧
    a (.wire 201 27) = bselect (a (.virt 422)) (a (.wire 189 55)) (a (.virt 203)) ∧
    a (.wire 201 35) = bselect (a (.virt 422)) (a (.wire 194 3)) (a (.virt 204)) ∧
    a (.wire 201 43) = bselect (a (.virt 422)) (a (.wire 194 11)) (a (.virt 205)) ∧
    a (.wire 201 51) = bselect (a (.virt 422)) (a (.wire 194 19)) (a (.virt 206)) ∧
    (a (.wire 205 12) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 0 ∧ a (.wire 205 13) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 1 ∧ a (.wire 205 14) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 2 ∧ a (.wire 205 15) = spongeHash perm [a (.wire 198 47), a (.wire 198 55), a (.wire 199 3), a (.wire 199 11), a (.wire 199 27), a (.wire 199 43), a (.wire 199 59), a (.wire 200 15), a (.wire 200 31), a (.wire 200 47), a (.wire 201 3), a (.wire 201 19), a (.wire 201 27), a (.wire 201 35), a (.wire 201 43), a (.wire 201 51)] 3) ∧
    a (.wire 201 59) = bselect (a (.wire 195 5)) (a (.wire 205 12)) (a (.wire 189 55)) ∧
    a (.wire 206 7) = bselect (a (.wire 195 5)) (a (.wire 205 13)) (a (.wire 194 3)) ∧
    a (.wire 206 15) = bselect (a (.wire 195 5)) (a (.wire 205 14)) (a (.wire 194 11)) ∧
    a (.wire 206 23) = bselect (a (.wire 195 5)) (a (.wire 205 15)) (a (.wire 194 19)) ∧
    a (.wire 206 27) = a (.wire 201 59) - a (.virt 8) ∧
    a (.wire 196 43) = a (.wire 206 27) * a (.virt 13) ∧
    a (.wire 196 43) = a (.virt 14) ∧
    a (.wire 206 31) = a (.wire 206 7) - a (.virt 9) ∧
    a (.wire 196 47) = a (.wire 206 31) * a (.virt 13) ∧
    a (.wire 196 47) = a (.virt 14) ∧
    a (.wire 206 35) = a (.wire 206 15) - a (.virt 10) ∧
    a (.wire 196 51) = a (.wire 206 35) * a (.virt 13) ∧
    a (.wire 196 51) = a (.virt 14) ∧
    a (.wire 206 39) = a (.wire 206 23) - a (.virt 11) ∧
    a (.wire 196 55) = a (.wire 206 39) * a (.virt 13) ∧
    a (.wire 196 55) = a (.virt 14) ∧
    rangeCheck (a (.virt 249)) 32 ∧
    a (.virt 227) = a (.virt 237) ∧
    a (.virt 228) = a (.virt 238) ∧
    a (.virt 229) = a (.virt 239)) ∧
    (a (.virt 230) = a (.virt 240) ∧
    a (.virt 231) = a (.virt 5) ∧
    a (.virt 232) = a (.virt 6) ∧
    a (.virt 233) = a (.virt 1) ∧
    a (.virt 234) = a (.virt 2) ∧
    a (.virt 235) = a (.virt 3) ∧
    a (.virt 236) = a (.virt 4) ∧
    IsEqual (a (.virt 241)) (a (.virt 14)) (a (.virt 424)) (a (.virt 425)) ∧
    IsEqual (a (.virt 242)) (a (.virt 14)) (a (.virt 426)) (a (.virt 427)) ∧
    IsEqual (a (.virt 243)) (a (.virt 14)) (a (.virt 428)) (a (.virt 429)) ∧
    IsEqual (a (.virt 244)) (a (.virt 14)) (a (.virt 430)) (a (.virt 431)) ∧
    a (.wire 208 31) = band (a (.virt 424)) (a (.virt 426)) ∧
    a (.wire 208 35) = band (a (.virt 428)) (a (.virt 430)) ∧
    a (.wire 208 39) = band (a (.wire 208 31)) (a (.wire 208 35)) ∧
    a (.wire 209 15) = a (.virt 292) - a (.wire 208 39) ∧
    a (.wire 208 43) = a (.wire 208 39) * a (.virt 7) ∧
    a (.wire 208 43) = a (.virt 14) ∧
    a (.virt 13) = a (.wire 209 15) ∧
    (a (.wire 211 12) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 0 ∧ a (.wire 211 13) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 1 ∧ a (.wire 211 14) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 2 ∧ a (.wire 211 15) = spongeHash perm [a (.virt 432), a (.virt 433), a (.virt 292), a (.virt 227), a (.virt 228), a (.virt 229), a (.virt 230), a (.virt 231), a (.virt 232)] 3) ∧
    (a (.wire 212 12) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 0 ∧ a (.wire 212 13) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 1 ∧ a (.wire 212 14) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 2 ∧ a (.wire 212 15) = spongeHash perm [a (.wire 211 12), a (.wire 211 13), a (.wire 211 14), a (.wire 211 15)] 3) ∧
    a (.wire 209 19) = a (.virt 223) - a (.wire 212 12) ∧
    a (.wire 208 47) = a (.wire 209 19) * a (.wire 209 15) ∧
    a (.wire 208 47) = a (.virt 14) ∧
    a (.wire 209 23) = a (.virt 224) - a (.wire 212 13) ∧
    a (.wire 208 51) = a (.wire 209 23) * a (.wire 209 15) ∧
    a (.wire 208 51) = a (.virt 14) ∧
    a (.wire 209 27) = a (.virt 225) - a (.wire 212 14) ∧
    a (.wire 208 55) = a (.wire 209 27) * a (.wire 209 15) ∧
    a (.wire 208 55) = a (.virt 14) ∧
    a (.wire 209 31) = a (.virt 226) - a (.wire 212 15) ∧
    a (.wire 208 59) = a (.wire 209 31) * a (.wire 209 15) ∧
    a (.wire 208 59) = a (.virt 14)) ∧
    ((a (.wire 220 12) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 0 ∧ a (.wire 220 13) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 1 ∧ a (.wire 220 14) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 2 ∧ a (.wire 220 15) = spongeHash perm [a (.virt 245), a (.virt 246), a (.virt 247), a (.virt 248), a (.virt 249), a (.virt 250), a (.virt 251), a (.virt 252), a (.virt 253), a (.virt 254), a (.virt 255), a (.virt 256), a (.virt 257), a (.virt 258), a (.virt 259), a (.virt 260), a (.virt 261), a (.virt 262), a (.virt 263), a (.virt 264), a (.virt 265), a (.virt 266), a (.virt 267), a (.virt 268), a (.virt 269), a (.virt 270), a (.virt 271), a (.virt 272), a (.virt 273), a (.virt 274), a (.virt 275), a (.virt 276), a (.virt 277), a (.virt 278), a (.virt 279), a (.virt 280), a (.virt 281), a (.virt 282), a (.virt 283), a (.virt 284), a (.virt 285), a (.virt 286), a (.virt 287), a (.virt 288), a (.virt 289)] 3) ∧
    a (.wire 209 35) = a (.virt 241) - a (.wire 220 12) ∧
    a (.wire 221 3) = a (.wire 209 35) * a (.wire 209 15) ∧
    a (.wire 221 3) = a (.virt 14) ∧
    a (.wire 209 39) = a (.virt 242) - a (.wire 220 13) ∧
    a (.wire 221 7) = a (.wire 209 39) * a (.wire 209 15) ∧
    a (.wire 221 7) = a (.virt 14) ∧
    a (.wire 209 43) = a (.virt 243) - a (.wire 220 14) ∧
    a (.wire 221 11) = a (.wire 209 43) * a (.wire 209 15) ∧
    a (.wire 221 11) = a (.virt 14) ∧
    a (.wire 209 47) = a (.virt 244) - a (.wire 220 15) ∧
    a (.wire 221 15) = a (.wire 209 47) * a (.wire 209 15) ∧
    a (.wire 221 15) = a (.virt 14) ∧
    a (.wire 209 51) = a (.virt 258) - a (.virt 8) ∧
    a (.wire 221 19) = a (.wire 209 51) * a (.wire 209 15) ∧
    a (.wire 221 19) = a (.virt 14) ∧
    a (.wire 209 55) = a (.virt 259) - a (.virt 9) ∧
    a (.wire 221 23) = a (.wire 209 55) * a (.wire 209 15) ∧
    a (.wire 221 23) = a (.virt 14) ∧
    a (.wire 209 59) = a (.virt 260) - a (.virt 10) ∧
    a (.wire 221 27) = a (.wire 209 59) * a (.wire 209 15) ∧
    a (.wire 221 27) = a (.virt 14) ∧
    a (.wire 222 3) = a (.virt 261) - a (.virt 11) ∧
    a (.wire 221 31) = a (.wire 222 3) * a (.wire 209 15) ∧
    a (.wire 221 31) = a (.virt 14))) :=
  ⟨leafCircuit_decode0 perm a h hp, leafCircuit_decode1 perm a h hp⟩

end Plonky2Spec.Generated
