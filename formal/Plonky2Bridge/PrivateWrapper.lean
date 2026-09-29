/-
  The private-batch wrapper capstones on the recorded wiring, at every leaf count with a
  recorded trace (`constraint-exporter/traces/private_batch_wrapper_n*.json`).

  Each `Generated/Wrapper{N}.lean` is emitted by the exporter (`src/private_wrapper.rs`)
  from the `n = N` trace: it reads the children, the preimages and the aggregated output off
  the wiring and proves `Wrapper{N}.sound : RPrivateBatch …` and
  `Wrapper{N}.end_to_end_wired`. This module names the capstones for the axiom gate in
  `ci/AxiomsCheck.lean`; the aliases carry the generated statements verbatim.
-/
import Plonky2Bridge.Generated.Wrapper2
import Plonky2Bridge.Generated.Wrapper4

namespace Plonky2Bridge

/-- `private_batch_end_to_end` on the `n = 2` wiring (`Wrapper2.end_to_end_wired`). -/
alias private_batch_end_to_end_wired := Wrapper2.end_to_end_wired

/-- `private_batch_end_to_end` on the `n = 4` wiring (`Wrapper4.end_to_end_wired`). -/
alias private_batch_end_to_end_wired_n4 := Wrapper4.end_to_end_wired

end Plonky2Bridge
