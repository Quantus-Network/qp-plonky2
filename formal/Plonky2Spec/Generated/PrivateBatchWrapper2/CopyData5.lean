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

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

/-- `privateBatchWrapper2.constants`, items `576..608`. -/
def privateBatchWrapper2.constants18 (p : ℕ) : List (Target × ZMod p) := [
    (.virt 9907, 13907817722130613464 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9908, 10952810625341944941 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9909, 6203009912824666656 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9910, 3300053643433736875 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9911, 6744005316595460895 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9912, 15739729861593455007 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9913, 261242591546957104 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9914, 15971894691377196519 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9915, 16201670926897260089 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9916, 8415742645426514154 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9917, 11796957557555108937 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9918, 16294407261451109534 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9919, 5598800393413128961 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9920, 15563455834725815206 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9921, 1764867128134410509 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9922, 7162174829468034425 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9923, 3755871071604584538 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9924, 4342271783305288164 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9925, 12880969007487790240 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9926, 10677306222450711949 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9927, 6504175050764636243 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9928, 5111390403385516632 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9929, 13022173847108028602 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9930, 13993386182188331211 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9931, 17852688572278391749 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9932, 13748656795736045903 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9933, 1633629184600895140 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9934, 4529011729897082393 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9935, 4906629816328734350 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9936, 12354457273844909068 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9937, 11030994060168315240 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9938, 5132927100884704451 /- canonical u64; faithful only at p = goldilocks -/)
  ]

/-- `privateBatchWrapper2.constants`, items `608..638`. -/
def privateBatchWrapper2.constants19 (p : ℕ) : List (Target × ZMod p) := [
    (.virt 9939, 1903334112240650810 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9940, 11300350282455140548 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9941, 12947495166068534688 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9942, 7280050037032595918 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9943, 2325750220496619057 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9944, 9863469300860708700 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9945, 5927058137232203163 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9946, 18296775548535963806 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9947, 16027760200278862284 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9948, 3198226309374699824 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9949, 17159143107915980371 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9950, 4742102661775960192 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9951, 5942453867077433164 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9952, 12359909013026785218 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9953, 12296487080408661428 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9954, 14401603032726340600 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9955, 18340875264408803468 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9956, 8906999085452790476 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9957, 11753455510941625388 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9958, 3159781736502437645 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9959, 3690235712498808163 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9960, 16869124931580068791 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9961, 4404853092538523347 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9962, 576460752303423488 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9963, 65),
    (.virt 9968, 136),
    (.virt 9973, 17),
    (.virt 9982, 7126024226993609386 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9989, 18442240469788262401 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 9990, 33)
  ]

end Plonky2Spec.Generated
