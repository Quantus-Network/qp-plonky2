/-
  CI axiom-footprint gate for the two capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus exactly one trusted axiom:
  `WormholeSpec.leaf_proof_sound` for `private_batch_end_to_end` and
  `WormholeSpec.private_batch_proof_sound` for `public_batch_end_to_end`.
  It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge
import Plonky2Bridge.PublicBatch

#print axioms Plonky2Bridge.private_batch_end_to_end
#print axioms Plonky2Bridge.public_batch_end_to_end
