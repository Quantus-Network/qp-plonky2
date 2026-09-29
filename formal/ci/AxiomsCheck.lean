/-
  CI axiom-footprint gate for the two capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus exactly one trusted axiom:
  `WormholeSpec.leaf_proof_sound` for `private_batch_end_to_end` and
  `WormholeSpec.private_batch_proof_sound` for `public_batch_end_to_end`.
  `private_batch_end_to_end_wired` — the same capstone stated on the exporter-generated
  `n = 2` wrapper wiring — is gated like `private_batch_end_to_end` (allow-list plus
  `WormholeSpec.leaf_proof_sound`), and `public_batch_end_to_end_wired` — on the
  `n_inner = 2` public-batch wrapper wiring — like `public_batch_end_to_end` (allow-list plus
  `WormholeSpec.private_batch_proof_sound`). `Wrapper2.sound` and `PublicWrapper2.sound`, the
  wiring-to-relation steps underneath them, are printed as well and are expected to use no
  trusted axiom.
  It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge
import Plonky2Bridge.PublicBatch
import Plonky2Bridge.Wrapper2
import Plonky2Bridge.PublicWrapper2

#print axioms Plonky2Bridge.private_batch_end_to_end
#print axioms Plonky2Bridge.public_batch_end_to_end
#print axioms Plonky2Bridge.private_batch_end_to_end_wired
#print axioms Plonky2Bridge.Wrapper2.sound
#print axioms Plonky2Bridge.public_batch_end_to_end_wired
#print axioms Plonky2Bridge.PublicWrapper2.sound
