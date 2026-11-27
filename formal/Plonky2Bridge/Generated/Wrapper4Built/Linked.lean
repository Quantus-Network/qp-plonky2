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
  exact (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_0 linked_261) (BuildCert.links_append linked_522 linked_783)) (BuildCert.links_append (BuildCert.links_append linked_1045 linked_1306) (BuildCert.links_append linked_1568 linked_1829))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_2091 linked_2352) (BuildCert.links_append linked_2613 linked_2874)) (BuildCert.links_append (BuildCert.links_append linked_3136 linked_3397) (BuildCert.links_append linked_3659 linked_3920)))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_4182 linked_4443) (BuildCert.links_append linked_4704 linked_4965)) (BuildCert.links_append (BuildCert.links_append linked_5227 linked_5488) (BuildCert.links_append linked_5750 linked_6011))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_6273 linked_6534) (BuildCert.links_append linked_6796 linked_7057)) (BuildCert.links_append (BuildCert.links_append linked_7319 linked_7580) (BuildCert.links_append linked_7842 linked_8103)))))

end Plonky2Bridge.Wrapper4
