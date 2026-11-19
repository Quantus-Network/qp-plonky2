/-
  The certified build of the leaf circuit (PLAN.md §7, Step 16).

  `Plonky2Bridge/Generated/LeafBuilt.lean` carries `LeafCircuit.buildCert`, the exporter's
  reading of `CircuitBuilder::build` on the replayed `leaf_circuit.json`, and proves
  `build_refines_of_agrees`: a built system that `Agrees` with the certificate `Refines`
  the leaf export. The checks behind that proof are kernel evaluations. What is left to
  trust is the one declaration below: that the deployed leaf build agrees with the
  certificate. It replaces `build_refines_export` for the leaf, and `proof_sound_leaf`
  composes the two.

  WHY THIS IS SMALLER THAN `build_refines_export`.
  `build_refines_export` asserts a property of `build` for every circuit, including the
  wrappers whose traces are not the deployed builder's state. `build_agrees` is a claim
  about one build of one circuit, and each part of it is a fact the exporter checked on
  the built system (`built::verify_agrees` in `constraint-exporter`) whose digest is the
  deployed leaf's (`built::LEAF_CIRCUIT_DIGEST`, the `leaf_build_matches_deployed_digest`
  test): the export's 223 rows at rows `0..223`, `σ` along the listed cycles, the
  constant wires `build` fixed, and the location of each target. The exporter's
  check is Rust, so it is trusted; what the kernel verifies is that these facts imply
  `Refines`, which is the part `build_refines_export` left to prose.

  WHAT `Agrees` ASSUMES OF THE MODEL.
  `BuildOf tree b` is opaque: it names the built system a verifier key for `tree`
  commits to. `build_agrees` says that system is the one the exporter built from the
  trace, as far as `Agrees` looks. That is the statement "the trace is the deployed
  pre-`build` state, and `build` is deterministic", which the trace test in
  qp-zk-circuits (`formal/traces`, mirrored on `WormholeCircuit::new_internal`) and the
  digest test check.
-/
import Plonky2Bridge.Generated.LeafBuilt

namespace Plonky2Bridge.LeafCircuit

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

variable {p : ℕ}

/-- **TRUSTED — the deployed leaf build is the certified one.** The built system a leaf
    verifier key commits to `Agrees` with `buildCert`: the export's rows at their
    indices, `σ` along the certificate's cycles, the constant wires it lists fixed to the
    export's constants, and targets placed as `buildCert.loc` says.

    Checked in Rust: `constraint-exporter`'s `built::verify_agrees` on the
    `FormalBuiltView` of `CircuitBuilder::build` for the replayed trace, whose circuit
    digest is the deployed leaf's (`leaf_build_matches_deployed_digest`). -/
axiom build_agrees (b : Built p) :
    BuildOf (.node "leaf_circuit" (leafCircuit p) []) b → b.Agrees (leafCircuit p) buildCert

/-- **The deployed leaf build refines the leaf export.** `build_agrees` and the certified
    `build_refines_of_agrees`; the leaf's replacement for `build_refines_export`. -/
theorem build_refines (b : Built p) :
    BuildOf (.node "leaf_circuit" (leafCircuit p) []) b → b.Refines (leafCircuit p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

end Plonky2Bridge.LeafCircuit
