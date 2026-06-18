/-
  Bridge: the auto-extracted Poseidon2 permutation primitives
  (`Generated/Poseidon2Prims.lean`, produced by running the *real*
  `plonky2/src/gates/poseidon2.rs` helpers over the symbolic field) equal the opaque
  hand model `Plonky2Spec.Poseidon2.{sbox7,mdsLight,internalMix}` (proved by `ring`).

  This machine-checks the *arithmetic* of the Poseidon2 permutation against the live
  Rust: any drift in the S-box, the light-MDS matrix, or the internal diffusion
  breaks `lake build`. The hand model's primitives are `@[irreducible]` (so the
  meaning proof in `Plonky2Spec.Poseidon2` never expands them); here we `unseal`
  them locally to discharge the equalities.

  What remains reviewed (not yet machine-checked here) is only the *round
  composition* — how `Plonky2Spec.Poseidon2.gateConstraints` threads these
  primitives across the 4+22+4 rounds with the checkpoint wires — which mirrors
  `eval_unfiltered` line-by-line and is covered end-to-end by the exporter's
  differential test against the real gate.
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.Ring
import Plonky2Spec.Generated.Poseidon2Prims
import Plonky2Spec.Poseidon2

namespace Plonky2Spec.Poseidon2

variable {p : ℕ}

-- The hand-model `x^7` S-box equals the extracted `sbox7_base`.
unseal sbox7 in
theorem sbox7_extracted (x : ZMod p) : sbox7 x = Generated.sbox7 x := by
  simp only [sbox7, Generated.sbox7]; ring

-- The hand-model light MDS equals the extracted `mds_light_base`, lanewise.
unseal mdsLight applyMat4 in
theorem mdsLight_extracted (s : St p) :
    mdsLight s = Generated.mdsLight (s 0) (s 1) (s 2) (s 3) (s 4) (s 5)
      (s 6) (s 7) (s 8) (s 9) (s 10) (s 11) := by
  funext i
  fin_cases i <;> simp [mdsLight, applyMat4, Generated.mdsLight]

-- The hand-model internal diffusion equals the extracted `internal_mix_base`,
-- lanewise (state `s` ↦ args `w0..w11`, diagonal `diag` ↦ args `w12..w23`).
unseal internalMix in
theorem internalMix_extracted (diag s : St p) :
    internalMix diag s = Generated.internalMix (s 0) (s 1) (s 2) (s 3) (s 4) (s 5)
      (s 6) (s 7) (s 8) (s 9) (s 10) (s 11)
      (diag 0) (diag 1) (diag 2) (diag 3) (diag 4) (diag 5)
      (diag 6) (diag 7) (diag 8) (diag 9) (diag 10) (diag 11) := by
  funext i
  fin_cases i <;> simp [internalMix, Generated.internalMix, Fin.sum_univ_succ] <;> ring

end Plonky2Spec.Poseidon2
