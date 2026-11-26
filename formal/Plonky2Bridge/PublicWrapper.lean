/-
  The public-batch wrapper capstones on the recorded wiring, at the tree sizes CI builds
  on every pull request (`n_inner = 2`, `4`). The production size (`86`) is
  `Plonky2Bridge/PublicWrapperProduction.lean`, the `Plonky2BridgeProduction` target,
  built on `main` and release pull requests: its 19,000 generated gadget facts take the
  bulk of a cold Lean build.

  Each `Generated/PublicWrapper{N}.lean` is emitted by the exporter (`src/public_wrapper.rs`)
  from the `n_inner = N` trace: it reads the inner outputs and the aggregated output off the
  wiring and proves `PublicWrapper{N}.sound : RPublicBatch …` and
  `PublicWrapper{N}.end_to_end_wired` (inners given as accepted private-batch proofs, through
  `proof_sound` and the inner wrapper's `accepted_sound`). This module names the capstones for the
  axiom gate in
  `ci/AxiomsCheck.lean`; the aliases carry the generated statements verbatim.
-/
import Plonky2Bridge.Generated.PublicWrapper2
import Plonky2Bridge.Generated.PublicWrapper4

namespace Plonky2Bridge

/-- `public_batch_end_to_end` on the `n_inner = 2` wiring (`PublicWrapper2.end_to_end_wired`). -/
alias public_batch_end_to_end_wired := PublicWrapper2.end_to_end_wired

/-- `public_batch_end_to_end` on the `n_inner = 4` wiring (`PublicWrapper4.end_to_end_wired`). -/
alias public_batch_end_to_end_wired_n4 := PublicWrapper4.end_to_end_wired

end Plonky2Bridge
