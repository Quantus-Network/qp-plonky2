import Plonky2Spec.Generated.Poseidon2
import Plonky2Spec.Poseidon2
open Plonky2Spec.Poseidon2

-- Probe: can we even state/normalize a single internal-phase state of the
-- *structured* model after unfolding the primitives? (No extracted side yet.)
-- This measures whether the structured model is tractable when expanded.
example {p : ℕ} (w : Fin 130 → ZMod p) : True := by
  trivial
