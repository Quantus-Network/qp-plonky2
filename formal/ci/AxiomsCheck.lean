/-
  CI axiom-footprint gate for the two capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus exactly one trusted axiom:
  `WormholeSpec.leaf_proof_sound` for `private_batch_end_to_end` and
  `WormholeSpec.private_batch_proof_sound` for `public_batch_end_to_end`.
  `Wrapper2.sound` (the exporter-generated `n = 2` private-batch wrapper decoded into
  `RPrivateBatch`) is printed as well; it is expected to use no trusted axiom.
  It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge
import Plonky2Bridge.PublicBatch
import Plonky2Bridge.Wrapper2

#print axioms Plonky2Bridge.private_batch_end_to_end
#print axioms Plonky2Bridge.public_batch_end_to_end
#print axioms Plonky2Bridge.Wrapper2.sound
