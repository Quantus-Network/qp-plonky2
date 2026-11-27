/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Bridge.Generated.Wrapper2Built`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Plonky2Bridge.Trusted
import Plonky2Bridge.Generated.LeafBuilt
import Plonky2Spec.Generated.PrivateBatchWrapper2
import Plonky2Bridge.Generated.Wrapper2Built.Links

namespace Plonky2Bridge.Wrapper2

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ}

/-- Every copy of the export is linked in the certificate. -/
theorem copies_linked : ∀ xy ∈ (privateBatchWrapper2 p).copies, buildCert.Links xy.1 xy.2 := by
  rw [copies_eq]
  exact (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_0 linked_261) (BuildCert.links_append linked_522 linked_783)) (BuildCert.links_append (BuildCert.links_append linked_1044 linked_1305) (BuildCert.links_append linked_1566 linked_1827))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_2089 linked_2350) (BuildCert.links_append linked_2611 linked_2872)) (BuildCert.links_append (BuildCert.links_append linked_3134 linked_3395) (BuildCert.links_append linked_3656 linked_3917))))

end Plonky2Bridge.Wrapper2
