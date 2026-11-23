/-
  CI axiom-footprint gate for the capstone theorems.

  The shell step in `.github/workflows/ci.yml` runs this file and parses `#print axioms`
  output per theorem, asserting the complete allow-list
  `{propext, Classical.choice, Quot.sound}` plus, where a theorem consumes accepted proofs,
  exactly the declarations `proof_sound` rests on: the three assumptions
  (`Plonky2Bridge.poseidon_transcript_oracle`, `Plonky2Bridge.fri_proximity_gap`,
  `Plonky2Bridge.algebraic_sampling`) and the five declarations under `verifier_algebra`:
  the four obligations (`Plonky2Bridge.build_refines_export`,
  `Plonky2Bridge.public_inputs_wired`, `Plonky2Bridge.child_gadget_satisfied`,
  `Plonky2Bridge.recursive_verifier_transcript`) and the binding assumption
  `Plonky2Bridge.public_input_binding`. `proof_sound`, `verifier_algebra`,
  `poseidon2_gate_rows` and `poseidon_gate_rows` are theorems. The assumptions are
  implications on their good events (`GoodEvents`, opaque), which the capstones take as
  hypotheses; that changes no footprint.

  * `verifier_algebra` is gated on the allow-list plus those five,
    `identical_constraints_satisfies` on the allow-list plus `build_refines_export`, and
    `Built.Refines.satisfies`, the step from the domain identities to `Satisfies`,
    `poseidon2_gate_rows`, the Poseidon2 rows from `Poseidon2.gate_sound_complete`, and
    `poseidon_gate_rows`, the `PoseidonGate` rows from `Poseidon.gate_sound_complete`
    (Step 18), on the bare allow-list — as are `Poseidon.gate_sound_complete` itself,
    `Poseidon.goldilocks_perm_test`, the kernel-evaluated test vector that pins the
    model's permutation to the native one, and `publicInputsHash_test`, the one that pins
    the model of the public-input hash to `PoseidonHash::hash_no_pad` (Step 19).
  * The leaf's build is certified (Steps 16 and 17). `proof_sound_leaf` is gated on the
    allow-list plus the three assumptions and `public_input_binding`: neither
    `build_refines_export` nor `public_inputs_wired` nor the two child obligations.
    `Built.Agrees.refines`, `Built.Agrees.hashesPublicInputs` and the generated
    `LeafCircuit.build_refines_of_agrees` / `build_hashes_of_agrees` (the certificate
    checks, kernel evaluations) are gated on the bare allow-list;
    `Built.HashesPublicInputs.hash_eq`, the hash read off the certified wiring, on the
    allow-list; and `LeafCircuit.build_refines` / `build_hashes` on the allow-list plus
    the one trusted declaration `Plonky2Bridge.LeafCircuit.build_agrees`.
  * `private_batch_end_to_end` / `public_batch_end_to_end` take the children's relations as
    hypotheses and are gated on the bare allow-list.
  * `LeafCircuit.accepted_sound` — an accepted leaf proof attests `Rleaf` — and
    `private_batch_end_to_end_wired` / `_n4` / `_production` — the capstones on the
    exporter-generated private-wrapper wiring (`n = 2` / `4` / `8`), which read only leaf
    proofs — are gated on the footprint of `proof_sound_leaf` plus
    `LeafCircuit.build_agrees`.
  * The private wrappers' builds are certified too (Step 19). The generated
    `Wrapper{2,4,8}.build_refines_of_agrees` / `build_hashes_of_agrees` are gated on the
    bare allow-list and `Wrapper{n}.build_refines` / `build_hashes` on the allow-list plus
    the one trusted declaration `Plonky2Bridge.Wrapper{n}.build_agrees`.
  * `Wrapper{2,4,8}.accepted_sound` — an accepted proof of the named tree attests its
    relation — and `public_batch_end_to_end_wired` / `_n4` — the capstones on the
    public-wrapper wiring (`n_inner = 2` / `4`), which read 2-leaf wrapper proofs — are
    gated on the footprint of `proof_sound_leaf` plus the two child obligations
    (`child_gadget_satisfied`, `recursive_verifier_transcript`), `LeafCircuit.build_agrees`
    and the named wrapper's `build_agrees`: neither `build_refines_export` nor
    `public_inputs_wired`, which no capstone takes.
  * `Wrapper{2,4,8}.sound`, `PublicWrapper{2,4}.sound` and `leaf_wired`
    (`LeafCircuit.sound`), the wiring-to-relation steps underneath, use no trusted axiom.
  The production public wrapper (`n_inner = 89`, `Plonky2BridgeProduction`) is gated the
  same way by `ci/AxiomsCheckProduction.lean`, on the runs that build it.
  It is not part of `defaultTargets` or `Plonky2Bridge`; import-only.
-/
import Plonky2Bridge
import Plonky2Bridge.PublicBatch
import Plonky2Bridge.PrivateWrapper
import Plonky2Bridge.PublicWrapper
import Plonky2Bridge.LeafWired
import Plonky2Bridge.Generated.PublicInputsHashVector

#print axioms Plonky2Bridge.proof_sound
#print axioms Plonky2Bridge.verifier_algebra
#print axioms Plonky2Bridge.identical_constraints_satisfies
#print axioms Plonky2Bridge.Built.Refines.satisfies
#print axioms Plonky2Bridge.poseidon2_gate_rows
#print axioms Plonky2Bridge.poseidon_gate_rows
#print axioms Plonky2Spec.Poseidon.gate_sound_complete
#print axioms Plonky2Spec.Poseidon.goldilocks_perm_test
#print axioms Plonky2Bridge.publicInputsHash_test
#print axioms Plonky2Bridge.proof_sound_leaf
#print axioms Plonky2Bridge.Built.Agrees.refines
#print axioms Plonky2Bridge.LeafCircuit.build_refines_of_agrees
#print axioms Plonky2Bridge.LeafCircuit.build_refines
#print axioms Plonky2Bridge.Built.HashesPublicInputs.hash_eq
#print axioms Plonky2Bridge.Built.Agrees.hashesPublicInputs
#print axioms Plonky2Bridge.LeafCircuit.build_hashes_of_agrees
#print axioms Plonky2Bridge.LeafCircuit.build_hashes
#print axioms Plonky2Bridge.Wrapper2.build_refines_of_agrees
#print axioms Plonky2Bridge.Wrapper2.build_hashes_of_agrees
#print axioms Plonky2Bridge.Wrapper2.build_refines
#print axioms Plonky2Bridge.Wrapper2.build_hashes
#print axioms Plonky2Bridge.Wrapper4.build_refines_of_agrees
#print axioms Plonky2Bridge.Wrapper4.build_hashes_of_agrees
#print axioms Plonky2Bridge.Wrapper4.build_refines
#print axioms Plonky2Bridge.Wrapper4.build_hashes
#print axioms Plonky2Bridge.Wrapper8.build_refines_of_agrees
#print axioms Plonky2Bridge.Wrapper8.build_hashes_of_agrees
#print axioms Plonky2Bridge.Wrapper8.build_refines
#print axioms Plonky2Bridge.Wrapper8.build_hashes
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
#print axioms Plonky2Bridge.leaf_wired
#print axioms Plonky2Bridge.LeafCircuit.sound
#print axioms Plonky2Bridge.LeafCircuit.accepted_sound
#print axioms Plonky2Bridge.Wrapper2.accepted_sound
#print axioms Plonky2Bridge.Wrapper4.accepted_sound
#print axioms Plonky2Bridge.Wrapper8.accepted_sound
