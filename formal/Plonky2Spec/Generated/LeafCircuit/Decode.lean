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
import Plonky2Spec.Generated.LeafCircuit.Fact

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
theorem leafCircuit_decode0 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (IsBool (a (.virt 13)) ∧
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
    a (.wire 139 23) = bselect (a (.wire 128 59)) (a (.virt 139)) (a (.virt 143))) :=
  ⟨⟨leafCircuit_f0 a h, leafCircuit_f1 perm a h hp, leafCircuit_f2 perm a h hp, leafCircuit_f3 a h, leafCircuit_f4 a h, leafCircuit_f5 a h, leafCircuit_f6 a h, leafCircuit_f7 a h, leafCircuit_f8 a h, leafCircuit_f9 a h, leafCircuit_f10 a h, leafCircuit_f11 perm a h hp, leafCircuit_f12 a h, leafCircuit_f13 a h, leafCircuit_f14 a h, leafCircuit_f15 a h, leafCircuit_f16 a h, leafCircuit_f17 a h, leafCircuit_f18 a h, leafCircuit_f19 a h, leafCircuit_f20 a h, leafCircuit_f21 a h, leafCircuit_f22 a h, leafCircuit_f23 a h, leafCircuit_f24 a h, leafCircuit_f25 a h, leafCircuit_f26 a h, leafCircuit_f27 a h, leafCircuit_f28 a h, leafCircuit_f29 a h, leafCircuit_f30 a h, leafCircuit_f31 a h⟩, ⟨leafCircuit_f32 a h, leafCircuit_f33 a h, leafCircuit_f34 a h, leafCircuit_f35 a h, leafCircuit_f36 a h, leafCircuit_f37 a h, leafCircuit_f38 a h, leafCircuit_f39 a h, leafCircuit_f40 a h, leafCircuit_f41 a h, leafCircuit_f42 a h, leafCircuit_f43 a h, leafCircuit_f44 a h, leafCircuit_f45 a h, leafCircuit_f46 a h, leafCircuit_f47 a h, leafCircuit_f48 a h, leafCircuit_f49 a h, leafCircuit_f50 a h, leafCircuit_f51 a h, leafCircuit_f52 a h, leafCircuit_f53 a h, leafCircuit_f54 a h, leafCircuit_f55 a h, leafCircuit_f56 a h, leafCircuit_f57 a h, leafCircuit_f58 a h, leafCircuit_f59 a h, leafCircuit_f60 a h, leafCircuit_f61 a h, leafCircuit_f62 a h, leafCircuit_f63 a h⟩, ⟨leafCircuit_f64 a h, leafCircuit_f65 a h, leafCircuit_f66 a h, leafCircuit_f67 a h, leafCircuit_f68 a h, leafCircuit_f69 a h, leafCircuit_f70 a h, leafCircuit_f71 a h, leafCircuit_f72 a h, leafCircuit_f73 a h, leafCircuit_f74 a h, leafCircuit_f75 a h, leafCircuit_f76 a h, leafCircuit_f77 a h, leafCircuit_f78 a h, leafCircuit_f79 a h, leafCircuit_f80 a h, leafCircuit_f81 a h, leafCircuit_f82 a h, leafCircuit_f83 a h, leafCircuit_f84 a h, leafCircuit_f85 a h, leafCircuit_f86 a h, leafCircuit_f87 a h, leafCircuit_f88 a h, leafCircuit_f89 a h, leafCircuit_f90 a h, leafCircuit_f91 a h, leafCircuit_f92 a h, leafCircuit_f93 a h, leafCircuit_f94 a h, leafCircuit_f95 a h⟩, ⟨leafCircuit_f96 a h, leafCircuit_f97 a h, leafCircuit_f98 a h, leafCircuit_f99 a h, leafCircuit_f100 a h, leafCircuit_f101 a h, leafCircuit_f102 a h, leafCircuit_f103 a h, leafCircuit_f104 a h, leafCircuit_f105 a h, leafCircuit_f106 a h, leafCircuit_f107 a h, leafCircuit_f108 a h, leafCircuit_f109 a h, leafCircuit_f110 a h, leafCircuit_f111 a h, leafCircuit_f112 a h, leafCircuit_f113 a h, leafCircuit_f114 a h, leafCircuit_f115 a h, leafCircuit_f116 a h, leafCircuit_f117 a h, leafCircuit_f118 a h, leafCircuit_f119 a h, leafCircuit_f120 a h, leafCircuit_f121 a h, leafCircuit_f122 a h, leafCircuit_f123 a h, leafCircuit_f124 a h, leafCircuit_f125 a h, leafCircuit_f126 a h, leafCircuit_f127 a h⟩, ⟨leafCircuit_f128 a h, leafCircuit_f129 a h, leafCircuit_f130 a h, leafCircuit_f131 a h, leafCircuit_f132 a h, leafCircuit_f133 a h, leafCircuit_f134 a h, leafCircuit_f135 a h, leafCircuit_f136 a h, leafCircuit_f137 a h, leafCircuit_f138 a h, leafCircuit_f139 a h, leafCircuit_f140 a h, leafCircuit_f141 a h, leafCircuit_f142 a h, leafCircuit_f143 a h, leafCircuit_f144 a h, leafCircuit_f145 a h, leafCircuit_f146 a h, leafCircuit_f147 a h, leafCircuit_f148 perm a h hp, leafCircuit_f149 a h, leafCircuit_f150 a h, leafCircuit_f151 a h, leafCircuit_f152 a h, leafCircuit_f153 a h, leafCircuit_f154 a h, leafCircuit_f155 a h, leafCircuit_f156 a h, leafCircuit_f157 a h, leafCircuit_f158 a h, leafCircuit_f159 a h⟩, ⟨leafCircuit_f160 a h, leafCircuit_f161 a h, leafCircuit_f162 a h, leafCircuit_f163 a h, leafCircuit_f164 a h, leafCircuit_f165 a h, leafCircuit_f166 a h, leafCircuit_f167 a h, leafCircuit_f168 a h, leafCircuit_f169 a h, leafCircuit_f170 a h, leafCircuit_f171 a h, leafCircuit_f172 a h, leafCircuit_f173 a h, leafCircuit_f174 a h, leafCircuit_f175 a h, leafCircuit_f176 a h, leafCircuit_f177 a h, leafCircuit_f178 a h, leafCircuit_f179 a h, leafCircuit_f180 a h, leafCircuit_f181 a h, leafCircuit_f182 a h, leafCircuit_f183 a h, leafCircuit_f184 a h, leafCircuit_f185 a h, leafCircuit_f186 a h, leafCircuit_f187 a h, leafCircuit_f188 a h, leafCircuit_f189 a h, leafCircuit_f190 a h, leafCircuit_f191 a h⟩, ⟨leafCircuit_f192 a h, leafCircuit_f193 a h, leafCircuit_f194 a h, leafCircuit_f195 a h, leafCircuit_f196 a h, leafCircuit_f197 a h, leafCircuit_f198 a h, leafCircuit_f199 a h, leafCircuit_f200 a h, leafCircuit_f201 a h, leafCircuit_f202 a h, leafCircuit_f203 a h, leafCircuit_f204 a h, leafCircuit_f205 a h, leafCircuit_f206 a h, leafCircuit_f207 a h, leafCircuit_f208 a h, leafCircuit_f209 a h, leafCircuit_f210 a h, leafCircuit_f211 a h, leafCircuit_f212 a h, leafCircuit_f213 a h, leafCircuit_f214 a h, leafCircuit_f215 a h, leafCircuit_f216 a h, leafCircuit_f217 a h, leafCircuit_f218 a h, leafCircuit_f219 a h, leafCircuit_f220 a h, leafCircuit_f221 a h, leafCircuit_f222 a h, leafCircuit_f223 a h⟩, ⟨leafCircuit_f224 a h, leafCircuit_f225 a h, leafCircuit_f226 a h, leafCircuit_f227 a h, leafCircuit_f228 a h, leafCircuit_f229 a h, leafCircuit_f230 a h, leafCircuit_f231 a h, leafCircuit_f232 a h, leafCircuit_f233 a h, leafCircuit_f234 a h, leafCircuit_f235 a h, leafCircuit_f236 a h, leafCircuit_f237 perm a h hp, leafCircuit_f238 a h, leafCircuit_f239 a h, leafCircuit_f240 a h, leafCircuit_f241 a h, leafCircuit_f242 a h, leafCircuit_f243 a h, leafCircuit_f244 a h, leafCircuit_f245 a h, leafCircuit_f246 a h, leafCircuit_f247 a h, leafCircuit_f248 a h, leafCircuit_f249 a h, leafCircuit_f250 a h, leafCircuit_f251 a h, leafCircuit_f252 a h, leafCircuit_f253 a h, leafCircuit_f254 a h, leafCircuit_f255 a h⟩, ⟨leafCircuit_f256 a h, leafCircuit_f257 a h, leafCircuit_f258 a h, leafCircuit_f259 a h, leafCircuit_f260 a h, leafCircuit_f261 a h, leafCircuit_f262 a h, leafCircuit_f263 a h, leafCircuit_f264 a h, leafCircuit_f265 a h, leafCircuit_f266 a h, leafCircuit_f267 a h, leafCircuit_f268 a h, leafCircuit_f269 a h, leafCircuit_f270 a h, leafCircuit_f271 a h, leafCircuit_f272 a h, leafCircuit_f273 a h, leafCircuit_f274 a h, leafCircuit_f275 a h, leafCircuit_f276 a h, leafCircuit_f277 a h, leafCircuit_f278 a h, leafCircuit_f279 a h, leafCircuit_f280 a h, leafCircuit_f281 a h, leafCircuit_f282 a h, leafCircuit_f283 a h, leafCircuit_f284 a h, leafCircuit_f285 a h, leafCircuit_f286 a h, leafCircuit_f287 a h⟩, ⟨leafCircuit_f288 a h, leafCircuit_f289 a h, leafCircuit_f290 a h, leafCircuit_f291 a h, leafCircuit_f292 a h, leafCircuit_f293 a h, leafCircuit_f294 a h, leafCircuit_f295 a h, leafCircuit_f296 a h, leafCircuit_f297 a h, leafCircuit_f298 a h, leafCircuit_f299 a h, leafCircuit_f300 a h, leafCircuit_f301 a h, leafCircuit_f302 a h, leafCircuit_f303 a h, leafCircuit_f304 a h, leafCircuit_f305 a h, leafCircuit_f306 a h, leafCircuit_f307 a h, leafCircuit_f308 a h, leafCircuit_f309 a h, leafCircuit_f310 a h, leafCircuit_f311 a h, leafCircuit_f312 a h, leafCircuit_f313 a h, leafCircuit_f314 a h, leafCircuit_f315 a h, leafCircuit_f316 a h, leafCircuit_f317 a h, leafCircuit_f318 a h, leafCircuit_f319 a h⟩, ⟨leafCircuit_f320 a h, leafCircuit_f321 a h, leafCircuit_f322 a h, leafCircuit_f323 a h, leafCircuit_f324 a h, leafCircuit_f325 a h, leafCircuit_f326 perm a h hp, leafCircuit_f327 a h, leafCircuit_f328 a h, leafCircuit_f329 a h, leafCircuit_f330 a h, leafCircuit_f331 a h, leafCircuit_f332 a h, leafCircuit_f333 a h, leafCircuit_f334 a h, leafCircuit_f335 a h, leafCircuit_f336 a h, leafCircuit_f337 a h, leafCircuit_f338 a h, leafCircuit_f339 a h, leafCircuit_f340 a h, leafCircuit_f341 a h, leafCircuit_f342 a h, leafCircuit_f343 a h, leafCircuit_f344 a h, leafCircuit_f345 a h, leafCircuit_f346 a h, leafCircuit_f347 a h, leafCircuit_f348 a h, leafCircuit_f349 a h, leafCircuit_f350 a h, leafCircuit_f351 a h⟩, ⟨leafCircuit_f352 a h, leafCircuit_f353 a h, leafCircuit_f354 a h, leafCircuit_f355 a h, leafCircuit_f356 a h, leafCircuit_f357 a h, leafCircuit_f358 a h, leafCircuit_f359 a h, leafCircuit_f360 a h, leafCircuit_f361 a h, leafCircuit_f362 a h, leafCircuit_f363 a h, leafCircuit_f364 a h, leafCircuit_f365 a h, leafCircuit_f366 a h, leafCircuit_f367 a h, leafCircuit_f368 a h, leafCircuit_f369 a h, leafCircuit_f370 a h, leafCircuit_f371 a h, leafCircuit_f372 a h, leafCircuit_f373 a h, leafCircuit_f374 a h, leafCircuit_f375 a h, leafCircuit_f376 a h, leafCircuit_f377 a h, leafCircuit_f378 a h, leafCircuit_f379 a h, leafCircuit_f380 a h, leafCircuit_f381 a h, leafCircuit_f382 a h, leafCircuit_f383 a h⟩, ⟨leafCircuit_f384 a h, leafCircuit_f385 a h, leafCircuit_f386 a h, leafCircuit_f387 a h, leafCircuit_f388 a h, leafCircuit_f389 a h, leafCircuit_f390 a h, leafCircuit_f391 a h, leafCircuit_f392 a h, leafCircuit_f393 a h, leafCircuit_f394 a h, leafCircuit_f395 a h, leafCircuit_f396 a h, leafCircuit_f397 a h, leafCircuit_f398 a h, leafCircuit_f399 a h, leafCircuit_f400 a h, leafCircuit_f401 a h, leafCircuit_f402 a h, leafCircuit_f403 a h, leafCircuit_f404 a h, leafCircuit_f405 a h, leafCircuit_f406 a h, leafCircuit_f407 a h, leafCircuit_f408 a h, leafCircuit_f409 a h, leafCircuit_f410 a h, leafCircuit_f411 a h, leafCircuit_f412 a h, leafCircuit_f413 a h, leafCircuit_f414 a h, leafCircuit_f415 perm a h hp⟩, ⟨leafCircuit_f416 a h, leafCircuit_f417 a h, leafCircuit_f418 a h, leafCircuit_f419 a h, leafCircuit_f420 a h, leafCircuit_f421 a h, leafCircuit_f422 a h, leafCircuit_f423 a h, leafCircuit_f424 a h, leafCircuit_f425 a h, leafCircuit_f426 a h, leafCircuit_f427 a h, leafCircuit_f428 a h, leafCircuit_f429 a h, leafCircuit_f430 a h, leafCircuit_f431 a h, leafCircuit_f432 a h, leafCircuit_f433 a h, leafCircuit_f434 a h, leafCircuit_f435 a h, leafCircuit_f436 a h, leafCircuit_f437 a h, leafCircuit_f438 a h, leafCircuit_f439 a h, leafCircuit_f440 a h, leafCircuit_f441 a h, leafCircuit_f442 a h, leafCircuit_f443 a h, leafCircuit_f444 a h, leafCircuit_f445 a h, leafCircuit_f446 a h, leafCircuit_f447 a h⟩, ⟨leafCircuit_f448 a h, leafCircuit_f449 a h, leafCircuit_f450 a h, leafCircuit_f451 a h, leafCircuit_f452 a h, leafCircuit_f453 a h, leafCircuit_f454 a h, leafCircuit_f455 a h, leafCircuit_f456 a h, leafCircuit_f457 a h, leafCircuit_f458 a h, leafCircuit_f459 a h, leafCircuit_f460 a h, leafCircuit_f461 a h, leafCircuit_f462 a h, leafCircuit_f463 a h, leafCircuit_f464 a h, leafCircuit_f465 a h, leafCircuit_f466 a h, leafCircuit_f467 a h, leafCircuit_f468 a h, leafCircuit_f469 a h, leafCircuit_f470 a h, leafCircuit_f471 a h, leafCircuit_f472 a h, leafCircuit_f473 a h, leafCircuit_f474 a h, leafCircuit_f475 a h, leafCircuit_f476 a h, leafCircuit_f477 a h, leafCircuit_f478 a h, leafCircuit_f479 a h⟩, ⟨leafCircuit_f480 a h, leafCircuit_f481 a h, leafCircuit_f482 a h, leafCircuit_f483 a h, leafCircuit_f484 a h, leafCircuit_f485 a h, leafCircuit_f486 a h, leafCircuit_f487 a h, leafCircuit_f488 a h, leafCircuit_f489 a h, leafCircuit_f490 a h, leafCircuit_f491 a h, leafCircuit_f492 a h, leafCircuit_f493 a h, leafCircuit_f494 a h, leafCircuit_f495 a h, leafCircuit_f496 a h, leafCircuit_f497 a h, leafCircuit_f498 a h, leafCircuit_f499 a h, leafCircuit_f500 a h, leafCircuit_f501 a h, leafCircuit_f502 a h, leafCircuit_f503 a h, leafCircuit_f504 perm a h hp, leafCircuit_f505 a h, leafCircuit_f506 a h, leafCircuit_f507 a h, leafCircuit_f508 a h, leafCircuit_f509 a h, leafCircuit_f510 a h, leafCircuit_f511 a h⟩, ⟨leafCircuit_f512 a h, leafCircuit_f513 a h, leafCircuit_f514 a h, leafCircuit_f515 a h, leafCircuit_f516 a h, leafCircuit_f517 a h, leafCircuit_f518 a h, leafCircuit_f519 a h, leafCircuit_f520 a h, leafCircuit_f521 a h, leafCircuit_f522 a h, leafCircuit_f523 a h, leafCircuit_f524 a h, leafCircuit_f525 a h, leafCircuit_f526 a h, leafCircuit_f527 a h, leafCircuit_f528 a h, leafCircuit_f529 a h, leafCircuit_f530 a h, leafCircuit_f531 a h, leafCircuit_f532 a h, leafCircuit_f533 a h, leafCircuit_f534 a h, leafCircuit_f535 a h, leafCircuit_f536 a h, leafCircuit_f537 a h, leafCircuit_f538 a h, leafCircuit_f539 a h, leafCircuit_f540 a h, leafCircuit_f541 a h, leafCircuit_f542 a h, leafCircuit_f543 a h⟩, ⟨leafCircuit_f544 a h, leafCircuit_f545 a h, leafCircuit_f546 a h, leafCircuit_f547 a h, leafCircuit_f548 a h, leafCircuit_f549 a h, leafCircuit_f550 a h, leafCircuit_f551 a h, leafCircuit_f552 a h, leafCircuit_f553 a h, leafCircuit_f554 a h, leafCircuit_f555 a h, leafCircuit_f556 a h, leafCircuit_f557 a h, leafCircuit_f558 a h, leafCircuit_f559 a h, leafCircuit_f560 a h, leafCircuit_f561 a h, leafCircuit_f562 a h, leafCircuit_f563 a h, leafCircuit_f564 a h, leafCircuit_f565 a h, leafCircuit_f566 a h, leafCircuit_f567 a h, leafCircuit_f568 a h, leafCircuit_f569 a h, leafCircuit_f570 a h, leafCircuit_f571 a h, leafCircuit_f572 a h, leafCircuit_f573 a h, leafCircuit_f574 a h, leafCircuit_f575 a h⟩, ⟨leafCircuit_f576 a h, leafCircuit_f577 a h, leafCircuit_f578 a h, leafCircuit_f579 a h, leafCircuit_f580 a h, leafCircuit_f581 a h, leafCircuit_f582 a h, leafCircuit_f583 a h, leafCircuit_f584 a h, leafCircuit_f585 a h, leafCircuit_f586 a h, leafCircuit_f587 a h, leafCircuit_f588 a h, leafCircuit_f589 a h, leafCircuit_f590 a h, leafCircuit_f591 a h, leafCircuit_f592 a h, leafCircuit_f593 perm a h hp, leafCircuit_f594 a h, leafCircuit_f595 a h, leafCircuit_f596 a h, leafCircuit_f597 a h, leafCircuit_f598 a h, leafCircuit_f599 a h, leafCircuit_f600 a h, leafCircuit_f601 a h, leafCircuit_f602 a h, leafCircuit_f603 a h, leafCircuit_f604 a h, leafCircuit_f605 a h, leafCircuit_f606 a h, leafCircuit_f607 a h⟩, ⟨leafCircuit_f608 a h, leafCircuit_f609 a h, leafCircuit_f610 a h, leafCircuit_f611 a h, leafCircuit_f612 a h, leafCircuit_f613 a h, leafCircuit_f614 a h, leafCircuit_f615 a h, leafCircuit_f616 a h, leafCircuit_f617 a h, leafCircuit_f618 a h, leafCircuit_f619 a h, leafCircuit_f620 a h, leafCircuit_f621 a h, leafCircuit_f622 a h, leafCircuit_f623 a h, leafCircuit_f624 a h, leafCircuit_f625 a h, leafCircuit_f626 a h, leafCircuit_f627 a h, leafCircuit_f628 a h, leafCircuit_f629 a h, leafCircuit_f630 a h, leafCircuit_f631 a h, leafCircuit_f632 a h, leafCircuit_f633 a h, leafCircuit_f634 a h, leafCircuit_f635 a h, leafCircuit_f636 a h, leafCircuit_f637 a h, leafCircuit_f638 a h, leafCircuit_f639 a h⟩, ⟨leafCircuit_f640 a h, leafCircuit_f641 a h, leafCircuit_f642 a h, leafCircuit_f643 a h, leafCircuit_f644 a h, leafCircuit_f645 a h, leafCircuit_f646 a h, leafCircuit_f647 a h, leafCircuit_f648 a h, leafCircuit_f649 a h, leafCircuit_f650 a h, leafCircuit_f651 a h, leafCircuit_f652 a h, leafCircuit_f653 a h, leafCircuit_f654 a h, leafCircuit_f655 a h, leafCircuit_f656 a h, leafCircuit_f657 a h, leafCircuit_f658 a h, leafCircuit_f659 a h, leafCircuit_f660 a h, leafCircuit_f661 a h, leafCircuit_f662 a h, leafCircuit_f663 a h, leafCircuit_f664 a h, leafCircuit_f665 a h, leafCircuit_f666 a h, leafCircuit_f667 a h, leafCircuit_f668 a h, leafCircuit_f669 a h, leafCircuit_f670 a h, leafCircuit_f671 a h⟩, ⟨leafCircuit_f672 a h, leafCircuit_f673 a h, leafCircuit_f674 a h, leafCircuit_f675 a h, leafCircuit_f676 a h, leafCircuit_f677 a h, leafCircuit_f678 a h, leafCircuit_f679 a h, leafCircuit_f680 a h, leafCircuit_f681 a h, leafCircuit_f682 perm a h hp, leafCircuit_f683 a h, leafCircuit_f684 a h, leafCircuit_f685 a h, leafCircuit_f686 a h, leafCircuit_f687 a h, leafCircuit_f688 a h, leafCircuit_f689 a h, leafCircuit_f690 a h, leafCircuit_f691 a h, leafCircuit_f692 a h, leafCircuit_f693 a h, leafCircuit_f694 a h, leafCircuit_f695 a h, leafCircuit_f696 a h, leafCircuit_f697 a h, leafCircuit_f698 a h, leafCircuit_f699 a h, leafCircuit_f700 a h, leafCircuit_f701 a h, leafCircuit_f702 a h, leafCircuit_f703 a h⟩, ⟨leafCircuit_f704 a h, leafCircuit_f705 a h, leafCircuit_f706 a h, leafCircuit_f707 a h, leafCircuit_f708 a h, leafCircuit_f709 a h, leafCircuit_f710 a h, leafCircuit_f711 a h, leafCircuit_f712 a h, leafCircuit_f713 a h, leafCircuit_f714 a h, leafCircuit_f715 a h, leafCircuit_f716 a h, leafCircuit_f717 a h, leafCircuit_f718 a h, leafCircuit_f719 a h, leafCircuit_f720 a h, leafCircuit_f721 a h, leafCircuit_f722 a h, leafCircuit_f723 a h, leafCircuit_f724 a h, leafCircuit_f725 a h, leafCircuit_f726 a h, leafCircuit_f727 a h, leafCircuit_f728 a h, leafCircuit_f729 a h, leafCircuit_f730 a h, leafCircuit_f731 a h, leafCircuit_f732 a h, leafCircuit_f733 a h, leafCircuit_f734 a h, leafCircuit_f735 a h⟩, ⟨leafCircuit_f736 a h, leafCircuit_f737 a h, leafCircuit_f738 a h, leafCircuit_f739 a h, leafCircuit_f740 a h, leafCircuit_f741 a h, leafCircuit_f742 a h, leafCircuit_f743 a h, leafCircuit_f744 a h, leafCircuit_f745 a h, leafCircuit_f746 a h, leafCircuit_f747 a h, leafCircuit_f748 a h, leafCircuit_f749 a h, leafCircuit_f750 a h, leafCircuit_f751 a h, leafCircuit_f752 a h, leafCircuit_f753 a h, leafCircuit_f754 a h, leafCircuit_f755 a h, leafCircuit_f756 a h, leafCircuit_f757 a h, leafCircuit_f758 a h, leafCircuit_f759 a h, leafCircuit_f760 a h, leafCircuit_f761 a h, leafCircuit_f762 a h, leafCircuit_f763 a h, leafCircuit_f764 a h, leafCircuit_f765 a h, leafCircuit_f766 a h, leafCircuit_f767 a h⟩, ⟨leafCircuit_f768 a h, leafCircuit_f769 a h, leafCircuit_f770 a h, leafCircuit_f771 perm a h hp, leafCircuit_f772 a h, leafCircuit_f773 a h, leafCircuit_f774 a h, leafCircuit_f775 a h, leafCircuit_f776 a h, leafCircuit_f777 a h, leafCircuit_f778 a h, leafCircuit_f779 a h, leafCircuit_f780 a h, leafCircuit_f781 a h, leafCircuit_f782 a h, leafCircuit_f783 a h, leafCircuit_f784 a h, leafCircuit_f785 a h, leafCircuit_f786 a h, leafCircuit_f787 a h, leafCircuit_f788 a h, leafCircuit_f789 a h, leafCircuit_f790 a h, leafCircuit_f791 a h, leafCircuit_f792 a h, leafCircuit_f793 a h, leafCircuit_f794 a h, leafCircuit_f795 a h, leafCircuit_f796 a h, leafCircuit_f797 a h, leafCircuit_f798 a h, leafCircuit_f799 a h⟩, ⟨leafCircuit_f800 a h, leafCircuit_f801 a h, leafCircuit_f802 a h, leafCircuit_f803 a h, leafCircuit_f804 a h, leafCircuit_f805 a h, leafCircuit_f806 a h, leafCircuit_f807 a h, leafCircuit_f808 a h, leafCircuit_f809 a h, leafCircuit_f810 a h, leafCircuit_f811 a h, leafCircuit_f812 a h, leafCircuit_f813 a h, leafCircuit_f814 a h, leafCircuit_f815 a h, leafCircuit_f816 a h, leafCircuit_f817 a h, leafCircuit_f818 a h, leafCircuit_f819 a h, leafCircuit_f820 a h, leafCircuit_f821 a h, leafCircuit_f822 a h, leafCircuit_f823 a h, leafCircuit_f824 a h, leafCircuit_f825 a h, leafCircuit_f826 a h, leafCircuit_f827 a h, leafCircuit_f828 a h, leafCircuit_f829 a h, leafCircuit_f830 a h, leafCircuit_f831 a h⟩, ⟨leafCircuit_f832 a h, leafCircuit_f833 a h, leafCircuit_f834 a h, leafCircuit_f835 a h, leafCircuit_f836 a h, leafCircuit_f837 a h, leafCircuit_f838 a h, leafCircuit_f839 a h, leafCircuit_f840 a h, leafCircuit_f841 a h, leafCircuit_f842 a h, leafCircuit_f843 a h, leafCircuit_f844 a h, leafCircuit_f845 a h, leafCircuit_f846 a h, leafCircuit_f847 a h, leafCircuit_f848 a h, leafCircuit_f849 a h, leafCircuit_f850 a h, leafCircuit_f851 a h, leafCircuit_f852 a h, leafCircuit_f853 a h, leafCircuit_f854 a h, leafCircuit_f855 a h, leafCircuit_f856 a h, leafCircuit_f857 a h, leafCircuit_f858 a h, leafCircuit_f859 a h, leafCircuit_f860 perm a h hp, leafCircuit_f861 a h, leafCircuit_f862 a h, leafCircuit_f863 a h⟩, ⟨leafCircuit_f864 a h, leafCircuit_f865 a h, leafCircuit_f866 a h, leafCircuit_f867 a h, leafCircuit_f868 a h, leafCircuit_f869 a h, leafCircuit_f870 a h, leafCircuit_f871 a h, leafCircuit_f872 a h, leafCircuit_f873 a h, leafCircuit_f874 a h, leafCircuit_f875 a h, leafCircuit_f876 a h, leafCircuit_f877 a h, leafCircuit_f878 a h, leafCircuit_f879 a h, leafCircuit_f880 a h, leafCircuit_f881 a h, leafCircuit_f882 a h, leafCircuit_f883 a h, leafCircuit_f884 a h, leafCircuit_f885 a h, leafCircuit_f886 a h, leafCircuit_f887 a h, leafCircuit_f888 a h, leafCircuit_f889 a h, leafCircuit_f890 a h, leafCircuit_f891 a h, leafCircuit_f892 a h, leafCircuit_f893 a h, leafCircuit_f894 a h, leafCircuit_f895 a h⟩, ⟨leafCircuit_f896 a h, leafCircuit_f897 a h, leafCircuit_f898 a h, leafCircuit_f899 a h, leafCircuit_f900 a h, leafCircuit_f901 a h, leafCircuit_f902 a h, leafCircuit_f903 a h, leafCircuit_f904 a h, leafCircuit_f905 a h, leafCircuit_f906 a h, leafCircuit_f907 a h, leafCircuit_f908 a h, leafCircuit_f909 a h, leafCircuit_f910 a h, leafCircuit_f911 a h, leafCircuit_f912 a h, leafCircuit_f913 a h, leafCircuit_f914 a h, leafCircuit_f915 a h, leafCircuit_f916 a h, leafCircuit_f917 a h, leafCircuit_f918 a h, leafCircuit_f919 a h, leafCircuit_f920 a h, leafCircuit_f921 a h, leafCircuit_f922 a h, leafCircuit_f923 a h, leafCircuit_f924 a h, leafCircuit_f925 a h, leafCircuit_f926 a h, leafCircuit_f927 a h⟩, ⟨leafCircuit_f928 a h, leafCircuit_f929 a h, leafCircuit_f930 a h, leafCircuit_f931 a h, leafCircuit_f932 a h, leafCircuit_f933 a h, leafCircuit_f934 a h, leafCircuit_f935 a h, leafCircuit_f936 a h, leafCircuit_f937 a h, leafCircuit_f938 a h, leafCircuit_f939 a h, leafCircuit_f940 a h, leafCircuit_f941 a h, leafCircuit_f942 a h, leafCircuit_f943 a h, leafCircuit_f944 a h, leafCircuit_f945 a h, leafCircuit_f946 a h, leafCircuit_f947 a h, leafCircuit_f948 a h, leafCircuit_f949 perm a h hp, leafCircuit_f950 a h, leafCircuit_f951 a h, leafCircuit_f952 a h, leafCircuit_f953 a h, leafCircuit_f954 a h, leafCircuit_f955 a h, leafCircuit_f956 a h, leafCircuit_f957 a h, leafCircuit_f958 a h, leafCircuit_f959 a h⟩, ⟨leafCircuit_f960 a h, leafCircuit_f961 a h, leafCircuit_f962 a h, leafCircuit_f963 a h, leafCircuit_f964 a h, leafCircuit_f965 a h, leafCircuit_f966 a h, leafCircuit_f967 a h, leafCircuit_f968 a h, leafCircuit_f969 a h, leafCircuit_f970 a h, leafCircuit_f971 a h, leafCircuit_f972 a h, leafCircuit_f973 a h, leafCircuit_f974 a h, leafCircuit_f975 a h, leafCircuit_f976 a h, leafCircuit_f977 a h, leafCircuit_f978 a h, leafCircuit_f979 a h, leafCircuit_f980 a h, leafCircuit_f981 a h, leafCircuit_f982 a h, leafCircuit_f983 a h, leafCircuit_f984 a h, leafCircuit_f985 a h, leafCircuit_f986 a h, leafCircuit_f987 a h, leafCircuit_f988 a h, leafCircuit_f989 a h, leafCircuit_f990 a h, leafCircuit_f991 a h⟩, ⟨leafCircuit_f992 a h, leafCircuit_f993 a h, leafCircuit_f994 a h, leafCircuit_f995 a h, leafCircuit_f996 a h, leafCircuit_f997 a h, leafCircuit_f998 a h, leafCircuit_f999 a h, leafCircuit_f1000 a h, leafCircuit_f1001 a h, leafCircuit_f1002 a h, leafCircuit_f1003 a h, leafCircuit_f1004 a h, leafCircuit_f1005 a h, leafCircuit_f1006 a h, leafCircuit_f1007 a h, leafCircuit_f1008 a h, leafCircuit_f1009 a h, leafCircuit_f1010 a h, leafCircuit_f1011 a h, leafCircuit_f1012 a h, leafCircuit_f1013 a h, leafCircuit_f1014 a h, leafCircuit_f1015 a h, leafCircuit_f1016 a h, leafCircuit_f1017 a h, leafCircuit_f1018 a h, leafCircuit_f1019 a h, leafCircuit_f1020 a h, leafCircuit_f1021 a h, leafCircuit_f1022 a h, leafCircuit_f1023 a h⟩⟩

set_option maxHeartbeats 4000000 in
theorem leafCircuit_decode1 (perm : St p → St p) (a : Assignment p) (h : Satisfies (leafCircuit p) a)
    (hp : Poseidon2Rows perm (leafCircuit p) a) :
    (a (.wire 139 31) = bselect (a (.virt 380)) (a (.wire 133 3)) (a (.wire 139 23)) ∧
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
    a (.wire 221 31) = a (.virt 14)) :=
  ⟨⟨leafCircuit_f1024 a h, leafCircuit_f1025 a h, leafCircuit_f1026 a h, leafCircuit_f1027 a h, leafCircuit_f1028 a h, leafCircuit_f1029 a h, leafCircuit_f1030 a h, leafCircuit_f1031 a h, leafCircuit_f1032 a h, leafCircuit_f1033 a h, leafCircuit_f1034 a h, leafCircuit_f1035 a h, leafCircuit_f1036 a h, leafCircuit_f1037 a h, leafCircuit_f1038 perm a h hp, leafCircuit_f1039 a h, leafCircuit_f1040 a h, leafCircuit_f1041 a h, leafCircuit_f1042 a h, leafCircuit_f1043 a h, leafCircuit_f1044 a h, leafCircuit_f1045 a h, leafCircuit_f1046 a h, leafCircuit_f1047 a h, leafCircuit_f1048 a h, leafCircuit_f1049 a h, leafCircuit_f1050 a h, leafCircuit_f1051 a h, leafCircuit_f1052 a h, leafCircuit_f1053 a h, leafCircuit_f1054 a h, leafCircuit_f1055 a h⟩, ⟨leafCircuit_f1056 a h, leafCircuit_f1057 a h, leafCircuit_f1058 a h, leafCircuit_f1059 a h, leafCircuit_f1060 a h, leafCircuit_f1061 a h, leafCircuit_f1062 a h, leafCircuit_f1063 a h, leafCircuit_f1064 a h, leafCircuit_f1065 a h, leafCircuit_f1066 a h, leafCircuit_f1067 a h, leafCircuit_f1068 a h, leafCircuit_f1069 a h, leafCircuit_f1070 a h, leafCircuit_f1071 a h, leafCircuit_f1072 a h, leafCircuit_f1073 a h, leafCircuit_f1074 a h, leafCircuit_f1075 a h, leafCircuit_f1076 a h, leafCircuit_f1077 a h, leafCircuit_f1078 a h, leafCircuit_f1079 a h, leafCircuit_f1080 a h, leafCircuit_f1081 a h, leafCircuit_f1082 a h, leafCircuit_f1083 a h, leafCircuit_f1084 a h, leafCircuit_f1085 a h, leafCircuit_f1086 a h, leafCircuit_f1087 a h⟩, ⟨leafCircuit_f1088 a h, leafCircuit_f1089 a h, leafCircuit_f1090 a h, leafCircuit_f1091 a h, leafCircuit_f1092 a h, leafCircuit_f1093 a h, leafCircuit_f1094 a h, leafCircuit_f1095 a h, leafCircuit_f1096 a h, leafCircuit_f1097 a h, leafCircuit_f1098 a h, leafCircuit_f1099 a h, leafCircuit_f1100 a h, leafCircuit_f1101 a h, leafCircuit_f1102 a h, leafCircuit_f1103 a h, leafCircuit_f1104 a h, leafCircuit_f1105 a h, leafCircuit_f1106 a h, leafCircuit_f1107 a h, leafCircuit_f1108 a h, leafCircuit_f1109 a h, leafCircuit_f1110 a h, leafCircuit_f1111 a h, leafCircuit_f1112 a h, leafCircuit_f1113 a h, leafCircuit_f1114 a h, leafCircuit_f1115 a h, leafCircuit_f1116 a h, leafCircuit_f1117 a h, leafCircuit_f1118 a h, leafCircuit_f1119 a h⟩, ⟨leafCircuit_f1120 a h, leafCircuit_f1121 a h, leafCircuit_f1122 a h, leafCircuit_f1123 a h, leafCircuit_f1124 a h, leafCircuit_f1125 a h, leafCircuit_f1126 a h, leafCircuit_f1127 perm a h hp, leafCircuit_f1128 a h, leafCircuit_f1129 a h, leafCircuit_f1130 a h, leafCircuit_f1131 a h, leafCircuit_f1132 a h, leafCircuit_f1133 a h, leafCircuit_f1134 a h, leafCircuit_f1135 a h, leafCircuit_f1136 a h, leafCircuit_f1137 a h, leafCircuit_f1138 a h, leafCircuit_f1139 a h, leafCircuit_f1140 a h, leafCircuit_f1141 a h, leafCircuit_f1142 a h, leafCircuit_f1143 a h, leafCircuit_f1144 a h, leafCircuit_f1145 a h, leafCircuit_f1146 a h, leafCircuit_f1147 a h, leafCircuit_f1148 a h, leafCircuit_f1149 a h, leafCircuit_f1150 a h, leafCircuit_f1151 a h⟩, ⟨leafCircuit_f1152 a h, leafCircuit_f1153 a h, leafCircuit_f1154 a h, leafCircuit_f1155 a h, leafCircuit_f1156 a h, leafCircuit_f1157 a h, leafCircuit_f1158 a h, leafCircuit_f1159 a h, leafCircuit_f1160 a h, leafCircuit_f1161 a h, leafCircuit_f1162 a h, leafCircuit_f1163 a h, leafCircuit_f1164 a h, leafCircuit_f1165 a h, leafCircuit_f1166 a h, leafCircuit_f1167 a h, leafCircuit_f1168 a h, leafCircuit_f1169 a h, leafCircuit_f1170 a h, leafCircuit_f1171 a h, leafCircuit_f1172 a h, leafCircuit_f1173 a h, leafCircuit_f1174 a h, leafCircuit_f1175 a h, leafCircuit_f1176 a h, leafCircuit_f1177 a h, leafCircuit_f1178 a h, leafCircuit_f1179 a h, leafCircuit_f1180 a h, leafCircuit_f1181 a h, leafCircuit_f1182 a h, leafCircuit_f1183 a h⟩, ⟨leafCircuit_f1184 a h, leafCircuit_f1185 a h, leafCircuit_f1186 a h, leafCircuit_f1187 a h, leafCircuit_f1188 a h, leafCircuit_f1189 a h, leafCircuit_f1190 a h, leafCircuit_f1191 a h, leafCircuit_f1192 a h, leafCircuit_f1193 a h, leafCircuit_f1194 a h, leafCircuit_f1195 a h, leafCircuit_f1196 a h, leafCircuit_f1197 a h, leafCircuit_f1198 a h, leafCircuit_f1199 a h, leafCircuit_f1200 a h, leafCircuit_f1201 a h, leafCircuit_f1202 a h, leafCircuit_f1203 a h, leafCircuit_f1204 a h, leafCircuit_f1205 a h, leafCircuit_f1206 a h, leafCircuit_f1207 a h, leafCircuit_f1208 a h, leafCircuit_f1209 a h, leafCircuit_f1210 a h, leafCircuit_f1211 a h, leafCircuit_f1212 a h, leafCircuit_f1213 a h, leafCircuit_f1214 a h, leafCircuit_f1215 a h⟩, ⟨leafCircuit_f1216 perm a h hp, leafCircuit_f1217 a h, leafCircuit_f1218 a h, leafCircuit_f1219 a h, leafCircuit_f1220 a h, leafCircuit_f1221 a h, leafCircuit_f1222 a h, leafCircuit_f1223 a h, leafCircuit_f1224 a h, leafCircuit_f1225 a h, leafCircuit_f1226 a h, leafCircuit_f1227 a h, leafCircuit_f1228 a h, leafCircuit_f1229 a h, leafCircuit_f1230 a h, leafCircuit_f1231 a h, leafCircuit_f1232 a h, leafCircuit_f1233 a h, leafCircuit_f1234 a h, leafCircuit_f1235 a h, leafCircuit_f1236 a h, leafCircuit_f1237 a h, leafCircuit_f1238 a h, leafCircuit_f1239 a h, leafCircuit_f1240 a h, leafCircuit_f1241 a h, leafCircuit_f1242 a h, leafCircuit_f1243 a h, leafCircuit_f1244 a h, leafCircuit_f1245 a h, leafCircuit_f1246 a h, leafCircuit_f1247 a h⟩, ⟨leafCircuit_f1248 a h, leafCircuit_f1249 a h, leafCircuit_f1250 a h, leafCircuit_f1251 a h, leafCircuit_f1252 a h, leafCircuit_f1253 a h, leafCircuit_f1254 a h, leafCircuit_f1255 a h, leafCircuit_f1256 a h, leafCircuit_f1257 a h, leafCircuit_f1258 a h, leafCircuit_f1259 a h, leafCircuit_f1260 a h, leafCircuit_f1261 a h, leafCircuit_f1262 a h, leafCircuit_f1263 a h, leafCircuit_f1264 a h, leafCircuit_f1265 a h, leafCircuit_f1266 a h, leafCircuit_f1267 a h, leafCircuit_f1268 a h, leafCircuit_f1269 a h, leafCircuit_f1270 a h, leafCircuit_f1271 a h, leafCircuit_f1272 a h, leafCircuit_f1273 a h, leafCircuit_f1274 a h, leafCircuit_f1275 a h, leafCircuit_f1276 a h, leafCircuit_f1277 a h, leafCircuit_f1278 a h, leafCircuit_f1279 a h⟩, ⟨leafCircuit_f1280 a h, leafCircuit_f1281 a h, leafCircuit_f1282 a h, leafCircuit_f1283 a h, leafCircuit_f1284 a h, leafCircuit_f1285 a h, leafCircuit_f1286 a h, leafCircuit_f1287 a h, leafCircuit_f1288 a h, leafCircuit_f1289 a h, leafCircuit_f1290 a h, leafCircuit_f1291 a h, leafCircuit_f1292 a h, leafCircuit_f1293 a h, leafCircuit_f1294 a h, leafCircuit_f1295 a h, leafCircuit_f1296 a h, leafCircuit_f1297 a h, leafCircuit_f1298 a h, leafCircuit_f1299 a h, leafCircuit_f1300 a h, leafCircuit_f1301 a h, leafCircuit_f1302 a h, leafCircuit_f1303 a h, leafCircuit_f1304 a h, leafCircuit_f1305 perm a h hp, leafCircuit_f1306 a h, leafCircuit_f1307 a h, leafCircuit_f1308 a h, leafCircuit_f1309 a h, leafCircuit_f1310 a h, leafCircuit_f1311 a h⟩, ⟨leafCircuit_f1312 a h, leafCircuit_f1313 a h, leafCircuit_f1314 a h, leafCircuit_f1315 a h, leafCircuit_f1316 a h, leafCircuit_f1317 a h, leafCircuit_f1318 a h, leafCircuit_f1319 a h, leafCircuit_f1320 a h, leafCircuit_f1321 a h, leafCircuit_f1322 a h, leafCircuit_f1323 a h, leafCircuit_f1324 a h, leafCircuit_f1325 a h, leafCircuit_f1326 a h, leafCircuit_f1327 a h, leafCircuit_f1328 a h, leafCircuit_f1329 a h, leafCircuit_f1330 a h, leafCircuit_f1331 a h, leafCircuit_f1332 a h, leafCircuit_f1333 a h, leafCircuit_f1334 a h, leafCircuit_f1335 a h, leafCircuit_f1336 a h, leafCircuit_f1337 a h, leafCircuit_f1338 a h, leafCircuit_f1339 a h, leafCircuit_f1340 a h, leafCircuit_f1341 a h, leafCircuit_f1342 a h, leafCircuit_f1343 a h⟩, ⟨leafCircuit_f1344 a h, leafCircuit_f1345 a h, leafCircuit_f1346 a h, leafCircuit_f1347 a h, leafCircuit_f1348 a h, leafCircuit_f1349 a h, leafCircuit_f1350 a h, leafCircuit_f1351 a h, leafCircuit_f1352 a h, leafCircuit_f1353 a h, leafCircuit_f1354 a h, leafCircuit_f1355 a h, leafCircuit_f1356 a h, leafCircuit_f1357 a h, leafCircuit_f1358 a h, leafCircuit_f1359 a h, leafCircuit_f1360 a h, leafCircuit_f1361 a h, leafCircuit_f1362 a h, leafCircuit_f1363 a h, leafCircuit_f1364 a h, leafCircuit_f1365 a h, leafCircuit_f1366 a h, leafCircuit_f1367 a h, leafCircuit_f1368 a h, leafCircuit_f1369 a h, leafCircuit_f1370 a h, leafCircuit_f1371 a h, leafCircuit_f1372 a h, leafCircuit_f1373 a h, leafCircuit_f1374 a h, leafCircuit_f1375 a h⟩, ⟨leafCircuit_f1376 a h, leafCircuit_f1377 a h, leafCircuit_f1378 a h, leafCircuit_f1379 a h, leafCircuit_f1380 a h, leafCircuit_f1381 a h, leafCircuit_f1382 a h, leafCircuit_f1383 a h, leafCircuit_f1384 a h, leafCircuit_f1385 a h, leafCircuit_f1386 a h, leafCircuit_f1387 a h, leafCircuit_f1388 a h, leafCircuit_f1389 a h, leafCircuit_f1390 a h, leafCircuit_f1391 a h, leafCircuit_f1392 a h, leafCircuit_f1393 a h, leafCircuit_f1394 perm a h hp, leafCircuit_f1395 a h, leafCircuit_f1396 a h, leafCircuit_f1397 a h, leafCircuit_f1398 a h, leafCircuit_f1399 a h, leafCircuit_f1400 a h, leafCircuit_f1401 a h, leafCircuit_f1402 a h, leafCircuit_f1403 a h, leafCircuit_f1404 a h, leafCircuit_f1405 a h, leafCircuit_f1406 a h, leafCircuit_f1407 a h⟩, ⟨leafCircuit_f1408 a h, leafCircuit_f1409 a h, leafCircuit_f1410 a h, leafCircuit_f1411 a h, leafCircuit_f1412 a h, leafCircuit_f1413 a h, leafCircuit_f1414 a h, leafCircuit_f1415 a h, leafCircuit_f1416 a h, leafCircuit_f1417 a h, leafCircuit_f1418 a h, leafCircuit_f1419 a h, leafCircuit_f1420 a h, leafCircuit_f1421 a h, leafCircuit_f1422 a h, leafCircuit_f1423 a h, leafCircuit_f1424 a h, leafCircuit_f1425 a h, leafCircuit_f1426 a h, leafCircuit_f1427 a h, leafCircuit_f1428 a h, leafCircuit_f1429 a h, leafCircuit_f1430 a h, leafCircuit_f1431 a h, leafCircuit_f1432 a h, leafCircuit_f1433 a h, leafCircuit_f1434 a h, leafCircuit_f1435 a h, leafCircuit_f1436 a h, leafCircuit_f1437 a h, leafCircuit_f1438 a h, leafCircuit_f1439 a h⟩, ⟨leafCircuit_f1440 a h, leafCircuit_f1441 a h, leafCircuit_f1442 a h, leafCircuit_f1443 a h, leafCircuit_f1444 a h, leafCircuit_f1445 a h, leafCircuit_f1446 a h, leafCircuit_f1447 a h, leafCircuit_f1448 a h, leafCircuit_f1449 a h, leafCircuit_f1450 a h, leafCircuit_f1451 a h, leafCircuit_f1452 a h, leafCircuit_f1453 a h, leafCircuit_f1454 a h, leafCircuit_f1455 a h, leafCircuit_f1456 a h, leafCircuit_f1457 a h, leafCircuit_f1458 a h, leafCircuit_f1459 a h, leafCircuit_f1460 a h, leafCircuit_f1461 a h, leafCircuit_f1462 a h, leafCircuit_f1463 a h, leafCircuit_f1464 a h, leafCircuit_f1465 a h, leafCircuit_f1466 a h, leafCircuit_f1467 a h, leafCircuit_f1468 a h, leafCircuit_f1469 a h, leafCircuit_f1470 a h, leafCircuit_f1471 a h⟩, ⟨leafCircuit_f1472 a h, leafCircuit_f1473 a h, leafCircuit_f1474 a h, leafCircuit_f1475 a h, leafCircuit_f1476 a h, leafCircuit_f1477 a h, leafCircuit_f1478 a h, leafCircuit_f1479 a h, leafCircuit_f1480 a h, leafCircuit_f1481 a h, leafCircuit_f1482 a h, leafCircuit_f1483 perm a h hp, leafCircuit_f1484 a h, leafCircuit_f1485 a h, leafCircuit_f1486 a h, leafCircuit_f1487 a h, leafCircuit_f1488 a h, leafCircuit_f1489 a h, leafCircuit_f1490 a h, leafCircuit_f1491 a h, leafCircuit_f1492 a h, leafCircuit_f1493 a h, leafCircuit_f1494 a h, leafCircuit_f1495 a h, leafCircuit_f1496 a h, leafCircuit_f1497 a h, leafCircuit_f1498 a h, leafCircuit_f1499 a h, leafCircuit_f1500 a h, leafCircuit_f1501 a h, leafCircuit_f1502 a h, leafCircuit_f1503 a h⟩, ⟨leafCircuit_f1504 a h, leafCircuit_f1505 a h, leafCircuit_f1506 a h, leafCircuit_f1507 a h, leafCircuit_f1508 a h, leafCircuit_f1509 a h, leafCircuit_f1510 a h, leafCircuit_f1511 a h, leafCircuit_f1512 a h, leafCircuit_f1513 a h, leafCircuit_f1514 a h, leafCircuit_f1515 a h, leafCircuit_f1516 a h, leafCircuit_f1517 a h, leafCircuit_f1518 a h, leafCircuit_f1519 a h, leafCircuit_f1520 a h, leafCircuit_f1521 a h, leafCircuit_f1522 perm a h hp, leafCircuit_f1523 perm a h hp, leafCircuit_f1524 a h, leafCircuit_f1525 a h, leafCircuit_f1526 a h, leafCircuit_f1527 a h, leafCircuit_f1528 a h, leafCircuit_f1529 a h, leafCircuit_f1530 a h, leafCircuit_f1531 a h, leafCircuit_f1532 a h, leafCircuit_f1533 a h, leafCircuit_f1534 a h, leafCircuit_f1535 a h⟩, ⟨leafCircuit_f1536 perm a h hp, leafCircuit_f1537 a h, leafCircuit_f1538 a h, leafCircuit_f1539 a h, leafCircuit_f1540 a h, leafCircuit_f1541 a h, leafCircuit_f1542 a h, leafCircuit_f1543 a h, leafCircuit_f1544 a h, leafCircuit_f1545 a h, leafCircuit_f1546 a h, leafCircuit_f1547 a h, leafCircuit_f1548 a h, leafCircuit_f1549 a h, leafCircuit_f1550 a h, leafCircuit_f1551 a h, leafCircuit_f1552 a h, leafCircuit_f1553 a h, leafCircuit_f1554 a h, leafCircuit_f1555 a h, leafCircuit_f1556 a h, leafCircuit_f1557 a h, leafCircuit_f1558 a h, leafCircuit_f1559 a h, leafCircuit_f1560 a h⟩⟩

end Plonky2Spec.Generated
