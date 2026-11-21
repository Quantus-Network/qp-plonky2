/-
  CI axiom-footprint gate for the capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus, where a theorem consumes accepted proofs,
  exactly the declarations `proof_sound` rests on: the three assumptions
  (`Plonky2Bridge.poseidon_transcript_oracle`, `Plonky2Bridge.fri_proximity_gap`,
  `Plonky2Bridge.algebraic_sampling`) and the six declarations under `verifier_algebra`:
  the four obligations (`Plonky2Bridge.build_refines_export`,
  `Plonky2Bridge.public_inputs_wired`, `Plonky2Bridge.child_gadget_satisfied`,
  `Plonky2Bridge.recursive_verifier_transcript`), the binding assumption
  `Plonky2Bridge.public_input_binding`, and the trusted `Plonky2Bridge.poseidon_gate_rows`
  (a `PoseidonGate` row computes the proof-system permutation).
  `proof_sound`, `verifier_algebra` and `poseidon2_gate_rows` are theorems. The
  assumptions are implications on their good events (`GoodEvents`, opaque), which the
  capstones take as hypotheses; that changes no footprint.

  * `verifier_algebra` is gated on the allow-list plus those six,
    `identical_constraints_satisfies` on the allow-list plus `build_refines_export`, and
    `Built.Refines.satisfies`, the step from the domain identities to `Satisfies`, and
    `poseidon2_gate_rows`, the Poseidon2 rows from `gate_sound_complete`, on the bare
    allow-list.
  * The leaf's build is certified (Steps 16 and 17). `proof_sound_leaf` is gated on the
    allow-list plus the three assumptions, `public_input_binding` and
    `poseidon_gate_rows`: neither `build_refines_export` nor `public_inputs_wired` nor the
    two child obligations. `Built.Agrees.refines`, `Built.Agrees.hashesPublicInputs` and
    the generated `LeafCircuit.build_refines_of_agrees` / `build_hashes_of_agrees` (the
    certificate checks, kernel evaluations) are gated on the bare allow-list;
    `Built.HashesPublicInputs.hash_eq`, the hash read off the certified wiring, on the
    allow-list plus `poseidon_gate_rows`; and `LeafCircuit.build_refines` /
    `build_hashes` on the allow-list plus the one trusted declaration
    `Plonky2Bridge.LeafCircuit.build_agrees`.
  * `private_batch_end_to_end` / `public_batch_end_to_end` take the children's relations as
    hypotheses and are gated on the bare allow-list.
  * `LeafCircuit.accepted_sound` — an accepted leaf proof attests `Rleaf` — and
    `private_batch_end_to_end_wired` / `_n4` / `_production` — the capstones on the
    exporter-generated private-wrapper wiring (`n = 2` / `4` / `8`), which read only leaf
    proofs — are gated on the footprint of `proof_sound_leaf` plus
    `LeafCircuit.build_agrees`.
  * `Wrapper{2,4,8}.accepted_sound` — an accepted proof of the named tree attests its
    relation — and `public_batch_end_to_end_wired` / `_n4` / `_production` — the capstones
    on the public-wrapper wiring (`n_inner = 2` / `4` / `89`), which read wrapper proofs —
    are gated on the allow-list plus those eight declarations and, through the leaf,
    `LeafCircuit.build_agrees`.
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
#print axioms Plonky2Bridge.poseidon2_gate_rows
#print axioms Plonky2Bridge.proof_sound_leaf
#print axioms Plonky2Bridge.Built.Agrees.refines
#print axioms Plonky2Bridge.LeafCircuit.build_refines_of_agrees
#print axioms Plonky2Bridge.LeafCircuit.build_refines
#print axioms Plonky2Bridge.Built.HashesPublicInputs.hash_eq
#print axioms Plonky2Bridge.Built.Agrees.hashesPublicInputs
#print axioms Plonky2Bridge.LeafCircuit.build_hashes_of_agrees
#print axioms Plonky2Bridge.LeafCircuit.build_hashes
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
