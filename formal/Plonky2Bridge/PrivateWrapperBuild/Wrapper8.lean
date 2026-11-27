/-
  The certified build of the production (8-leaf, 32-slot) private-batch wrapper:
  `build_agrees`, its one trusted declaration, and what follows from it.
  `PrivateWrapperBuild.lean` says what `Agrees` states of a wrapper.
-/
import Plonky2Bridge.Generated.Wrapper8Built

namespace Plonky2Bridge.Wrapper8

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

variable {p : ℕ}

/-- **TRUSTED — the deployed production (8-leaf, 32-slot) private-batch build is the
    certified one.** Checked in Rust as `Wrapper2.build_agrees`, on the
    `private_batch_wrapper_n8` trace. -/
axiom build_agrees (b : Built p) : BuildOf tree b → b.Agrees (privateBatchWrapper8 p) buildCert

theorem build_refines (b : Built p) : BuildOf tree b → b.Refines (privateBatchWrapper8 p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

theorem build_hashes (b : Built p) :
    BuildOf tree b → b.HashesPublicInputs (privateBatchWrapper8 p) :=
  fun h => build_hashes_of_agrees b (build_agrees b h)

theorem build_complete (b : Built p) : BuildOf tree b → b.Complete (privateBatchWrapper8 p) :=
  fun h => (build_agrees b h).complete

end Plonky2Bridge.Wrapper8
