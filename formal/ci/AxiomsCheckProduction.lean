/-
  CI axiom-footprint gate for the production capstones (`Plonky2BridgeProduction`: the
  `n_inner = 89` public wrapper and the 8-leaf private wrapper), run with
  `ci/AxiomsCheck.lean` on the CI runs that build that target (pushes to `main`, release
  pull requests, pull requests labelled `formal-full`). The expectations are those of the
  `n = 2` and `4` shapes in `ci/AxiomsCheck.lean`: `public_batch_end_to_end_wired_production`
  and `Wrapper8.accepted_sound` read 8-leaf wrapper proofs, so they are gated on the
  allow-list plus the three assumptions, the binding, the child obligation,
  `LeafCircuit.build_agrees` and `Wrapper8.build_agrees`; `PublicWrapper89.sound` and
  `Wrapper8.sound`, the wiring-to-relation steps, on the bare allow-list;
  `private_batch_end_to_end_wired_production` and `Wrapper8.build_*` as their `n = 2`
  counterparts.
  Import-only.
-/
import Plonky2Bridge.PublicWrapperProduction
import Plonky2Bridge.PrivateWrapperProduction

#print axioms Plonky2Bridge.public_batch_end_to_end_wired_production
#print axioms Plonky2Bridge.PublicWrapper89.sound
#print axioms Plonky2Bridge.Wrapper8.build_refines_of_agrees
#print axioms Plonky2Bridge.Wrapper8.build_hashes_of_agrees
#print axioms Plonky2Bridge.Wrapper8.build_refines
#print axioms Plonky2Bridge.Wrapper8.build_hashes
#print axioms Plonky2Bridge.Wrapper8.build_complete
#print axioms Plonky2Bridge.private_batch_end_to_end_wired_production
#print axioms Plonky2Bridge.Wrapper8.sound
#print axioms Plonky2Bridge.Wrapper8.accepted_sound
