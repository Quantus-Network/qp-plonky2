/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Bridge.Generated.Wrapper4Built`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Plonky2Bridge.Trusted
import Plonky2Bridge.Generated.LeafBuilt
import Plonky2Spec.Generated.PrivateBatchWrapper4
import Plonky2Bridge.Generated.Wrapper4Built.Links

namespace Plonky2Bridge.Wrapper4

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ}

/-- Every copy of the export is linked in the certificate. -/
theorem copies_linked : ∀ xy ∈ (privateBatchWrapper4 p).copies, buildCert.Links xy.1 xy.2 := by
  rw [copies_eq]
  exact (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_0 linked_505) (BuildCert.links_append linked_1011 linked_1517)) (BuildCert.links_append (BuildCert.links_append linked_2023 linked_2529) (BuildCert.links_append linked_3035 linked_3541))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_4047 linked_4552) (BuildCert.links_append linked_5058 linked_5564)) (BuildCert.links_append (BuildCert.links_append linked_6070 linked_6576) (BuildCert.links_append linked_7082 linked_7588))))

end Plonky2Bridge.Wrapper4
