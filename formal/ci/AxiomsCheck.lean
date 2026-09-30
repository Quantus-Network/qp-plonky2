/-
  CI axiom-footprint gate for the two capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus exactly one trusted axiom:
  `WormholeSpec.leaf_proof_sound` for `private_batch_end_to_end` and
  `WormholeSpec.private_batch_proof_sound` for `public_batch_end_to_end`.
  `private_batch_end_to_end_wired` / `private_batch_end_to_end_wired_n4` — the same capstone
  stated on the exporter-generated `n = 2` / `4` wrapper wiring — are gated like
  `private_batch_end_to_end` (allow-list plus `WormholeSpec.leaf_proof_sound`), and
  `public_batch_end_to_end_wired` / `public_batch_end_to_end_wired_n4` — on the
  `n_inner = 2` / `4` public-batch wrapper wiring — like `public_batch_end_to_end` (allow-list
  plus `WormholeSpec.private_batch_proof_sound`). `Wrapper{2,4}.sound` and
  `PublicWrapper{2,4}.sound`, the wiring-to-relation steps underneath them, are printed as
  well and are expected to use no trusted axiom. `leaf_wired` (`LeafCircuit.sound`, the leaf
  circuit's `Rleaf` on the recorded wiring) is the base of the recursion and is gated on the
  bare allow-list.
  It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge
import Plonky2Bridge.PublicBatch
import Plonky2Bridge.PrivateWrapper
import Plonky2Bridge.PublicWrapper
import Plonky2Bridge.LeafWired

#print axioms Plonky2Bridge.private_batch_end_to_end
#print axioms Plonky2Bridge.public_batch_end_to_end
#print axioms Plonky2Bridge.private_batch_end_to_end_wired
#print axioms Plonky2Bridge.Wrapper2.sound
#print axioms Plonky2Bridge.private_batch_end_to_end_wired_n4
#print axioms Plonky2Bridge.Wrapper4.sound
#print axioms Plonky2Bridge.public_batch_end_to_end_wired
#print axioms Plonky2Bridge.PublicWrapper2.sound
#print axioms Plonky2Bridge.public_batch_end_to_end_wired_n4
#print axioms Plonky2Bridge.PublicWrapper4.sound
#print axioms Plonky2Bridge.leaf_wired
#print axioms Plonky2Bridge.LeafCircuit.sound
