/-
  The private-batch wrapper capstone on the production wiring: 8 leaves and 32 exit slots
  (`constraint-exporter/traces/private_batch_wrapper_n8.json`).

  Kept out of `Plonky2Bridge` as part of the `Plonky2BridgeProduction` lake target: the
  generated `Plonky2Spec.Generated.PrivateBatchWrapper8` and
  `Plonky2Bridge.Generated.Wrapper8Built` are over half of a cold Lean build. CI builds
  this target on pushes to `main`, on release pull requests, and on pull requests labelled
  `formal-full`; the `n = 2` and `4` shapes (`Plonky2Bridge/PrivateWrapper.lean`) cover
  every pull request. The axiom gate for it is `ci/AxiomsCheckProduction.lean`.
-/
import Plonky2Bridge.Generated.Wrapper8

namespace Plonky2Bridge

/-- `private_batch_end_to_end` on the production wiring: 8 leaves and 32 exit slots
    (`Wrapper8.end_to_end_wired`). -/
alias private_batch_end_to_end_wired_production := Wrapper8.end_to_end_wired

end Plonky2Bridge
