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
import Plonky2Bridge.Generated.Wrapper8Built.Links

namespace Plonky2Bridge.Wrapper8

open Plonky2Spec.Wiring
open Plonky2Spec.Generated

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ}

/-- Every copy of the export is linked in the certificate. -/
theorem copies_linked : ∀ xy ∈ (privateBatchWrapper8 p).copies, buildCert.Links xy.1 xy.2 := by
  rw [copies_eq]
  exact (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_0 linked_507) (BuildCert.links_append linked_1015 linked_1523)) (BuildCert.links_append (BuildCert.links_append linked_2031 linked_2538) (BuildCert.links_append linked_3046 linked_3554))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_4062 linked_4569) (BuildCert.links_append linked_5077 linked_5585)) (BuildCert.links_append (BuildCert.links_append linked_6093 linked_6601) (BuildCert.links_append linked_7109 linked_7617)))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_8125 linked_8632) (BuildCert.links_append linked_9140 linked_9648)) (BuildCert.links_append (BuildCert.links_append linked_10156 linked_10663) (BuildCert.links_append linked_11171 linked_11679))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_12187 linked_12694) (BuildCert.links_append linked_13202 linked_13710)) (BuildCert.links_append (BuildCert.links_append linked_14218 linked_14726) (BuildCert.links_append linked_15234 linked_15742)))))

end Plonky2Bridge.Wrapper8
