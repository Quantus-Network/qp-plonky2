/-
  The public-batch wrapper capstone on the production wiring: `n_inner = 89`, each inner an
  8-leaf 32-slot private batch (`constraint-exporter/traces/public_batch_wrapper_n89.json`).

  Kept out of `Plonky2Bridge` as the `Plonky2BridgeProduction` lake target: the generated
  `Plonky2Spec.Generated.PublicBatchWrapper89` carries some 20,000 gadget-fact theorems and
  `Plonky2Bridge.Generated.PublicWrapper89` 89 forwarding modules, together most of a cold
  Lean build. CI builds this target on pushes to `main`, on release pull requests, and on
  pull requests labelled `formal-full`; the `n_inner = 2` and `4` shapes
  (`Plonky2Bridge/PublicWrapper.lean`) cover every pull request. The axiom gate for it is
  `ci/AxiomsCheckProduction.lean`.
-/
import Plonky2Bridge.Generated.PublicWrapper89

namespace Plonky2Bridge

/-- `public_batch_end_to_end` on the production wiring: 89 inners, each an 8-leaf
    32-slot private batch (`PublicWrapper89.end_to_end_wired`). -/
alias public_batch_end_to_end_wired_production := PublicWrapper89.end_to_end_wired

end Plonky2Bridge
