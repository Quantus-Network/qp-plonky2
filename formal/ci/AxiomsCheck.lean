/-
  CI axiom-footprint gate for the capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus, where a theorem consumes accepted proofs,
  exactly the declarations `proof_sound` rests on: the three assumptions
  (`Plonky2Bridge.poseidon_transcript_oracle`, `Plonky2Bridge.fri_proximity_gap`,
  `Plonky2Bridge.algebraic_sampling`) and the six declarations under `verifier_algebra`:
  the five obligations (`Plonky2Bridge.build_refines_export`,
  `Plonky2Bridge.poseidon2_gate_rows`, `Plonky2Bridge.public_input_wires`,
  `Plonky2Bridge.child_gadget_satisfied`, `Plonky2Bridge.recursive_verifier_transcript`)
  and the binding assumption `Plonky2Bridge.public_input_binding`.
  `proof_sound` and `verifier_algebra` are theorems.

  * `verifier_algebra` is gated on the allow-list plus those six,
    `identical_constraints_satisfies` on the allow-list plus `build_refines_export`, and
    `Built.Refines.satisfies`, the step from the domain identities to `Satisfies`, on the
    bare allow-list.
  * `private_batch_end_to_end` / `public_batch_end_to_end` take the children's relations as
    hypotheses and are gated on the bare allow-list.
  * `private_batch_end_to_end_wired` / `_n4` / `_production` and
    `public_batch_end_to_end_wired` / `_n4` / `_production` — the capstones on the
    exporter-generated wrapper wiring (`n = 2` / `4` / `8` and `n_inner = 2` / `4` / `89`),
    with `ProofAccepted` for the children — and `LeafCircuit.accepted_sound` /
    `Wrapper{2,4,8}.accepted_sound` — an accepted proof of the named tree attests its
    relation — are gated on the allow-list plus those nine declarations.
  * `Wrapper{2,4,8}.sound`, `PublicWrapper{2,4,89}.sound` and `leaf_wired`
    (`LeafCircuit.sound`), the wiring-to-relation steps underneath, use no trusted axiom.
  It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge
import Plonky2Bridge.PublicBatch
import Plonky2Bridge.PrivateWrapper
import Plonky2Bridge.PublicWrapper
import Plonky2Bridge.LeafWired

#print axioms Plonky2Bridge.proof_sound
#print axioms Plonky2Bridge.verifier_algebra
#print axioms Plonky2Bridge.identical_constraints_satisfies
#print axioms Plonky2Bridge.Built.Refines.satisfies
#print axioms Plonky2Bridge.private_batch_end_to_end
#print axioms Plonky2Bridge.public_batch_end_to_end
#print axioms Plonky2Bridge.private_batch_end_to_end_wired
#print axioms Plonky2Bridge.Wrapper2.sound
#print axioms Plonky2Bridge.private_batch_end_to_end_wired_n4
#print axioms Plonky2Bridge.Wrapper4.sound
#print axioms Plonky2Bridge.private_batch_end_to_end_wired_production
#print axioms Plonky2Bridge.Wrapper8.sound
#print axioms Plonky2Bridge.public_batch_end_to_end_wired
#print axioms Plonky2Bridge.PublicWrapper2.sound
#print axioms Plonky2Bridge.public_batch_end_to_end_wired_n4
#print axioms Plonky2Bridge.PublicWrapper4.sound
#print axioms Plonky2Bridge.public_batch_end_to_end_wired_production
#print axioms Plonky2Bridge.PublicWrapper89.sound
#print axioms Plonky2Bridge.leaf_wired
#print axioms Plonky2Bridge.LeafCircuit.sound
#print axioms Plonky2Bridge.LeafCircuit.accepted_sound
#print axioms Plonky2Bridge.Wrapper2.accepted_sound
#print axioms Plonky2Bridge.Wrapper4.accepted_sound
#print axioms Plonky2Bridge.Wrapper8.accepted_sound
