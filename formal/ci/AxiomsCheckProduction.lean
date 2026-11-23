/-
  CI axiom-footprint gate for the production public-wrapper capstone
  (`Plonky2BridgeProduction`, `n_inner = 89`), run with `ci/AxiomsCheck.lean` on the CI
  runs that build that target (pushes to `main`, release pull requests, pull requests
  labelled `formal-full`). The expectations are those of `public_batch_end_to_end_wired`
  and `PublicWrapper2.sound` there: `public_batch_end_to_end_wired_production` reads
  wrapper proofs, so it is gated on the allow-list plus the declarations under
  `proof_sound` and, through the leaf, `LeafCircuit.build_agrees`; `PublicWrapper89.sound`,
  the wiring-to-relation step, on the bare allow-list.
  Import-only.
-/
import Plonky2Bridge.PublicWrapperProduction

#print axioms Plonky2Bridge.public_batch_end_to_end_wired_production
#print axioms Plonky2Bridge.PublicWrapper89.sound
