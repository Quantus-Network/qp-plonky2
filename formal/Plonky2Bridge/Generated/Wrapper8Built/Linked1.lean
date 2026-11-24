/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Bridge.Generated.Wrapper8Built`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Plonky2Bridge.Trusted
import Plonky2Bridge.Generated.LeafBuilt
import Plonky2Spec.Generated.PrivateBatchWrapper8
import Plonky2Bridge.Generated.Wrapper8Built.Check

namespace Plonky2Bridge.Wrapper8

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ}

/-- **A build the certificate describes hashes the export's public inputs.**
    `Built.Agrees.hashesPublicInputs` on `checks_pi`. -/
theorem build_hashes_of_agrees (b : Built p) (h : b.Agrees (privateBatchWrapper8 p) buildCert) :
    b.HashesPublicInputs (privateBatchWrapper8 p) :=
  h.hashesPublicInputs (by rw [publicInputs_targets]; exact checks_pi)

end Plonky2Bridge.Wrapper8
