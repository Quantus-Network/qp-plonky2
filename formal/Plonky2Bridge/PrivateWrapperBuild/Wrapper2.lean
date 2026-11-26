/-
  The certified build of the 2-leaf private-batch wrapper: `build_agrees`, its
  one trusted declaration, and what follows from it. `PrivateWrapperBuild.lean` says
  what `Agrees` states of a wrapper.
-/
import Plonky2Bridge.Generated.Wrapper2Built

namespace Plonky2Bridge.Wrapper2

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

variable {p : ℕ}

/-- **TRUSTED — the deployed 2-leaf private-batch build is the certified one.** Checked in
    Rust: `built::verify_agrees` on the `FormalBuiltView` of `CircuitBuilder::build` for the
    replayed `private_batch_wrapper_n2` trace over the built leaf. -/
axiom build_agrees (b : Built p) : BuildOf tree b → b.Agrees (privateBatchWrapper2 p) buildCert

theorem build_refines (b : Built p) : BuildOf tree b → b.Refines (privateBatchWrapper2 p) :=
  fun h => build_refines_of_agrees b (build_agrees b h)

theorem build_hashes (b : Built p) :
    BuildOf tree b → b.HashesPublicInputs (privateBatchWrapper2 p) :=
  fun h => build_hashes_of_agrees b (build_agrees b h)

theorem build_complete (b : Built p) : BuildOf tree b → b.Complete (privateBatchWrapper2 p) :=
  fun h => (build_agrees b h).complete

end Plonky2Bridge.Wrapper2
