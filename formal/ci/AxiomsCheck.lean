/-
  CI axiom-footprint gate for the capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus, where a theorem consumes accepted proofs,
  exactly the one trusted axiom `Plonky2Bridge.proof_sound` (proof-system soundness on the
  exported recursion tree, `Plonky2Bridge/Trusted.lean`).

  * `private_batch_end_to_end` / `public_batch_end_to_end` take the children's relations as
    hypotheses and are gated on the bare allow-list.
  * `private_batch_end_to_end_wired` / `_n4` and `public_batch_end_to_end_wired` / `_n4` — the
    capstones on the exporter-generated `n = 2` / `4` wrapper wiring, with `ProofAccepted`
    for the children — and `LeafCircuit.accepted_sound` / `Wrapper{2,4}.accepted_sound` — an
    accepted proof of the named tree attests its relation — are gated on the allow-list plus
    `Plonky2Bridge.proof_sound`.
  * `Wrapper{2,4}.sound`, `PublicWrapper{2,4}.sound` and `leaf_wired` (`LeafCircuit.sound`),
    the wiring-to-relation steps underneath, use no trusted axiom.
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
#print axioms Plonky2Bridge.LeafCircuit.accepted_sound
#print axioms Plonky2Bridge.Wrapper2.accepted_sound
#print axioms Plonky2Bridge.Wrapper4.accepted_sound
