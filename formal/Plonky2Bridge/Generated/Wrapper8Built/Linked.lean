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
  exact (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_0 linked_262) (BuildCert.links_append linked_524 linked_786)) (BuildCert.links_append (BuildCert.links_append linked_1049 linked_1311) (BuildCert.links_append linked_1574 linked_1836))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_2099 linked_2361) (BuildCert.links_append linked_2623 linked_2885)) (BuildCert.links_append (BuildCert.links_append linked_3148 linked_3410) (BuildCert.links_append linked_3673 linked_3935)))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_4198 linked_4460) (BuildCert.links_append linked_4722 linked_4984)) (BuildCert.links_append (BuildCert.links_append linked_5247 linked_5509) (BuildCert.links_append linked_5772 linked_6034))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_6297 linked_6559) (BuildCert.links_append linked_6821 linked_7083)) (BuildCert.links_append (BuildCert.links_append linked_7346 linked_7608) (BuildCert.links_append linked_7871 linked_8133))))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_8396 linked_8658) (BuildCert.links_append linked_8920 linked_9182)) (BuildCert.links_append (BuildCert.links_append linked_9445 linked_9707) (BuildCert.links_append linked_9970 linked_10232))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_10495 linked_10757) (BuildCert.links_append linked_11019 linked_11281)) (BuildCert.links_append (BuildCert.links_append linked_11544 linked_11806) (BuildCert.links_append linked_12069 linked_12331)))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_12594 linked_12856) (BuildCert.links_append linked_13118 linked_13380)) (BuildCert.links_append (BuildCert.links_append linked_13643 linked_13905) (BuildCert.links_append linked_14168 linked_14430))) (BuildCert.links_append (BuildCert.links_append (BuildCert.links_append linked_14693 linked_14955) (BuildCert.links_append linked_15218 linked_15480)) (BuildCert.links_append (BuildCert.links_append linked_15743 linked_16005) (BuildCert.links_append linked_16268 linked_16530))))))

end Plonky2Bridge.Wrapper8
