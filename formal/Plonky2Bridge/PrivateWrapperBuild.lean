/-
  The certified builds of the private-batch wrappers (PLAN.md §7, Steps 19 and 21).

  `Plonky2Bridge/Generated/Wrapper{2,4,8}Built.lean` carry each wrapper's recursion tree
  (`tree`, with the leaf public inputs `leafPis` its `verify_proof` gadgets read) and its
  `buildCert`: the exporter's reading of `CircuitBuilder::build` on the production
  builder — the recursive verifiers laid down over the built leaf, then the wrapper's
  trace replayed on top, to the state digest the trace records of
  `PrivateBatchCircuit::new`'s builder. They prove `build_refines_of_agrees` and
  `build_hashes_of_agrees` by kernel evaluation of the certificate's checks against the
  export. What is left to trust is one declaration per wrapper, in
  `PrivateWrapperBuild/Wrapper{n}.lean`: that the deployed build agrees with its
  certificate. `proof_sound` composes them, with the child obligation
  `recursive_verifier_transcript` on the complete export, into each wrapper's
  `accepted_sound`.

  This module gathers the `n = 2` and `4` sizes, which every pull request builds. The
  production size, `n = 8`, is over half of a cold Lean build and reaches CI through
  `PrivateWrapperProduction.lean` (the `Plonky2BridgeProduction` target) with the
  `n_inner = 86` public wrapper: pushes to `main`, release pull requests and pull requests
  labelled `formal-full`.

  WHAT `Agrees` SAYS OF A WRAPPER.
  A wrapper's export is the whole production builder before `build`: the wrapper's own
  rows, virtual targets, copies, constants and public inputs first, then those of its
  `verify_proof` gadgets (`built::complete`). `buildCert.rowAt` places each export row in
  the build, and the `σ`-cycles the certificate lists are the whole copy classes those
  rows' wires lie in. `build_agrees` says the built system a wrapper verifier key commits
  to has those rows, cycles, fixed wires and placements (`Refines`, `HashesPublicInputs`),
  and nothing else before the rows `build` appends (`Complete`), as `built::verify_agrees`
  checked on the replayed build (`constraint-exporter`), and that the trace is the
  production pre-`build` state (`state_digest`, checked in qp-zk-circuits against
  `PrivateBatchCircuit::new`) and `build` is deterministic.
-/
import Plonky2Bridge.PrivateWrapperBuild.Wrapper2
import Plonky2Bridge.PrivateWrapperBuild.Wrapper4
