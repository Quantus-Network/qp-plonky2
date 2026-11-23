/-
  The certified build of the leaf circuit (PLAN.md §7, Steps 16 and 17).

  `Plonky2Bridge/Generated/LeafBuilt.lean` carries `LeafCircuit.tree`, the leaf as a
  recursion-tree node, and `LeafCircuit.buildCert`, the exporter's
  reading of `CircuitBuilder::build` on the replayed `leaf_circuit.json`, and proves
  `build_refines_of_agrees`: a built system that `Agrees` with the certificate `Refines`
  the leaf export; and `build_hashes_of_agrees`: it hashes the export's public inputs
  into its `PublicInputGate` (`Built.HashesPublicInputs`). The checks behind those proofs
  are kernel evaluations. What is left to trust is the one declaration below: that the
  deployed leaf build agrees with the certificate. It is what the leaf supplies for the
  builder hypotheses of `proof_sound_leaf`, which composes them.

  WHY THIS IS SMALL.
  Nothing is declared of `build` for every circuit (Step 20). `build_agrees`
  is a claim about one build of one circuit, and each part of it is a fact the exporter
  checked on the built system (`built::verify_agrees` in `constraint-exporter`) whose
  digest is the deployed leaf's (`LEAF_CIRCUIT_DIGEST` in `tests/built.rs`, the
  `leaf_build_matches_deployed_digest` test): the export's 223 rows at rows `0..223`, `σ` along the listed cycles, the
  constant wires `build` fixed, the location of each target, and the `PublicInputGate`
  and `PoseidonGate` rows `build` added. The exporter's check is Rust, so it is trusted;
  what the kernel verifies is that these facts imply `Refines` and
  `HashesPublicInputs`.

  WHAT `Agrees` ASSUMES OF THE MODEL.
  `BuildOf tree b` is opaque: it names the built system a verifier key for `tree`
  commits to. `build_agrees` says that system is the one the exporter built from the
  trace, as far as `Agrees` looks. That is the statement "the trace is the deployed
  pre-`build` state, and `build` is deterministic", which the trace test in
  qp-zk-circuits (`formal/traces`, whose `state_digest` is `WormholeCircuit::new`'s
  builder's) and the digest test check.
-/
import Plonky2Bridge.Generated.LeafBuilt

namespace Plonky2Bridge.LeafCircuit

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

variable {p : ℕ}

/-- **TRUSTED — the deployed leaf build is the certified one.** The built system a leaf
    verifier key commits to `Agrees` with `buildCert`: the export's rows at their
    indices, `σ` along the certificate's cycles, the constant wires it lists fixed to the
    export's constants and to `0`, targets placed as `buildCert.loc` says, and the
    `PublicInputGate` and `PoseidonGate` rows at `buildCert.piRow` and
    `buildCert.hashRows`.

    Checked in Rust: `constraint-exporter`'s `built::verify_agrees` on the
    `FormalBuiltView` of `CircuitBuilder::build` for the replayed trace, whose circuit
    digest is the deployed leaf's (`leaf_build_matches_deployed_digest`). -/
axiom build_agrees (b : Built p) : BuildOf tree b → b.Agrees (leafCircuit p) buildCert

/-- **The deployed leaf build refines the leaf export.** `build_agrees` and the certified
    `build_refines_of_agrees`; the refinement hypothesis of `proof_sound_leaf`. -/
theorem build_refines (b : Built p) : BuildOf tree b → b.Refines (leafCircuit p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

/-- **The deployed leaf build hashes the leaf's public inputs.** `build_agrees` and the
    certified `build_hashes_of_agrees`; the hashing hypothesis of `proof_sound_leaf`. -/
theorem build_hashes (b : Built p) : BuildOf tree b → b.HashesPublicInputs (leafCircuit p) :=
  fun h => build_hashes_of_agrees b (build_agrees b h)

end Plonky2Bridge.LeafCircuit
