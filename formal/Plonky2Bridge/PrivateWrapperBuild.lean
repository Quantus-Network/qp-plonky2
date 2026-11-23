/-
  The certified builds of the private-batch wrappers (PLAN.md §7, Step 19).

  `Plonky2Bridge/Generated/Wrapper{2,4,8}Built.lean` carry each wrapper's recursion tree
  (`tree`, with the leaf public inputs `leafPis` its `verify_proof` gadgets read) and its
  `buildCert`: the exporter's reading of `CircuitBuilder::build` on the production
  builder — the recursive verifiers laid down over the built leaf, then the wrapper's
  trace replayed on top, to the state digest the trace records of
  `PrivateBatchCircuit::new`'s builder. They prove `build_refines_of_agrees` and
  `build_hashes_of_agrees` by kernel evaluation of the certificate's checks against the
  export. What is left to trust is one declaration per wrapper below: that the deployed
  build agrees with its certificate. `proof_sound` composes them, with the two child
  obligations, into each wrapper's `accepted_sound`.

  WHAT `Agrees` SAYS OF A WRAPPER.
  A wrapper's export is its own part of the production builder: the rows its gadgets
  added, the verifier rows whose free slots its arithmetic ops took, its virtual targets,
  copies, constants and public inputs. `buildCert.rowAt` places each export row in the
  build, and the `σ`-cycles the certificate lists are the whole copy classes those rows'
  wires lie in — including the verifiers' classes of the constants `0` and `1`, which the
  wrapper's wires join. `build_agrees` says the built system a wrapper verifier key
  commits to has those rows, cycles, fixed wires and placements, as `built::verify_agrees`
  checked on the replayed build (`constraint-exporter`), and that the trace is the
  production pre-`build` state (`state_digest`, checked in qp-zk-circuits against
  `PrivateBatchCircuit::new`) and `build` is deterministic.
-/
import Plonky2Bridge.Generated.Wrapper2Built
import Plonky2Bridge.Generated.Wrapper4Built
import Plonky2Bridge.Generated.Wrapper8Built

namespace Plonky2Bridge

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

variable {p : ℕ}

namespace Wrapper2

/-- **TRUSTED — the deployed 2-leaf private-batch build is the certified one.** Checked in
    Rust: `built::verify_agrees` on the `FormalBuiltView` of `CircuitBuilder::build` for the
    replayed `private_batch_wrapper_n2` trace over the built leaf. -/
axiom build_agrees (b : Built p) : BuildOf tree b → b.Agrees (privateBatchWrapper2 p) buildCert

theorem build_refines (b : Built p) : BuildOf tree b → b.Refines (privateBatchWrapper2 p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

theorem build_hashes (b : Built p) :
    BuildOf tree b → b.HashesPublicInputs (privateBatchWrapper2 p) :=
  fun h => build_hashes_of_agrees b (build_agrees b h)

end Wrapper2

namespace Wrapper4

/-- **TRUSTED — the deployed 4-leaf private-batch build is the certified one.** Checked in
    Rust as `Wrapper2.build_agrees`, on the `private_batch_wrapper_n4` trace. -/
axiom build_agrees (b : Built p) : BuildOf tree b → b.Agrees (privateBatchWrapper4 p) buildCert

theorem build_refines (b : Built p) : BuildOf tree b → b.Refines (privateBatchWrapper4 p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

theorem build_hashes (b : Built p) :
    BuildOf tree b → b.HashesPublicInputs (privateBatchWrapper4 p) :=
  fun h => build_hashes_of_agrees b (build_agrees b h)

end Wrapper4

namespace Wrapper8

/-- **TRUSTED — the deployed production (8-leaf, 32-slot) private-batch build is the
    certified one.** Checked in Rust as `Wrapper2.build_agrees`, on the
    `private_batch_wrapper_n8` trace. -/
axiom build_agrees (b : Built p) : BuildOf tree b → b.Agrees (privateBatchWrapper8 p) buildCert

theorem build_refines (b : Built p) : BuildOf tree b → b.Refines (privateBatchWrapper8 p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

theorem build_hashes (b : Built p) :
    BuildOf tree b → b.HashesPublicInputs (privateBatchWrapper8 p) :=
  fun h => build_hashes_of_agrees b (build_agrees b h)

end Wrapper8

end Plonky2Bridge
