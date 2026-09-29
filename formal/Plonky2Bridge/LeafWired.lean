/-
  The leaf-circuit capstone on the recorded wiring (PLAN.md Step 9).

  `Generated/Leaf.lean` is emitted by the exporter (`src/leaf.rs`) from the leaf trace
  `constraint-exporter/traces/leaf_circuit.json`: it reads `LeafPublic` / `LeafWitness` off
  the wiring and proves `LeafCircuit.sound : Satisfies → Poseidon2Rows perm → Rleaf …`. This
  module names it for the axiom gate in `ci/AxiomsCheck.lean`; the alias carries the
  generated statement verbatim. Unlike the batch capstones it rests on no trusted
  `WormholeSpec` axiom: the leaf is the base of the recursion, so its relation is
  established directly from the constraints.
-/
import Plonky2Bridge.Generated.Leaf

namespace Plonky2Bridge

/-- `Rleaf` on the recorded leaf wiring (`LeafCircuit.sound`). -/
alias leaf_wired := LeafCircuit.sound

end Plonky2Bridge
