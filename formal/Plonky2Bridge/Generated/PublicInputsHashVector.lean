/-
  AUTO-GENERATED — do not edit by hand.

  A test vector of `PoseidonHash::hash_no_pad`, the proof-system hash of a public-input
  list (`get_public_inputs_hash`), checked against its model `publicInputsHash`
  (`Plonky2Bridge/Trusted.lean`) by kernel evaluation. Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints

  The input is eleven lanes, a leaf's public-input count: a full rate block and a
  partial one. The vector pins what the differential test
  `public_inputs_hash_matches_lean_model` (the exporter's tests) checks at random
  inputs — no padding, overwrite absorption, a permutation per block, the four-lane
  squeeze — on the permutation `goldilocks_perm_test` pins.
-/
import Plonky2Bridge.Trusted

namespace Plonky2Bridge

open Plonky2Spec.Poseidon (goldilocksPrime)

/-- The lanes `1, 2, …, 11`. -/
def publicInputsHashTestInput : List (ZMod goldilocksPrime) := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

/-- `PoseidonHash::hash_no_pad` of `publicInputsHashTestInput`. -/
def publicInputsHashTestOutput : Fin 4 → ZMod goldilocksPrime :=
  ![0xade950b104a7bc8e, 0x7fc4c38dd21f8312, 0x32cd118342193d61, 0x08f4a9a028b8b62c]

set_option maxRecDepth 8192 in
/-- The model of the public-input hash is the native one, on the test vector.
    Kernel-evaluated. -/
theorem publicInputsHash_test :
    publicInputsHash publicInputsHashTestInput = publicInputsHashTestOutput := by
  decide +kernel

end Plonky2Bridge
