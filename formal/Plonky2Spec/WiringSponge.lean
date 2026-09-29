/-
  Steps 8c / 9b — the sponge lift for exported `Poseidon2Gate` rows.

  `hash_n_to_hash_no_pad_p2` (hashing.rs:60-118) absorbs `pad10 inputs` block by block:
  for each eight-element block it adds the block onto the rate lanes of the state and
  permutes it with one `Poseidon2Gate` row. In the builder the additions are `add` calls,
  which fold when an operand is the zero constant: the first row's input wires are wired
  straight to the inputs and to the `one`/`zero` constants, and a later row's input wire is
  either wired to the previous row's output wire (the block element was zero) or to an
  `ArithmeticGate` op summing that output wire and the block element; the capacity lanes
  are always wired to the previous outputs (or to zero). The digest is the last row's first
  four output wires.

  `poseidon2In_first` / `poseidon2In_chain` package the twelve per-lane wire facts of a row
  into `poseidon2In a row = addBlock s blk`, `poseidon2Row_absorb` turns that plus
  `Poseidon2Rows perm` into `poseidon2Out a row = perm (addBlock s blk)`, and the generated
  decode proofs finish by rewriting `spongeHash` through `Sponge.absorbMsg_block8`.
-/
import Plonky2Spec.Wiring
import Plonky2Spec.Sponge

namespace Plonky2Spec.Wiring

open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge

variable {p : ℕ} [Fact p.Prime]

/-- A row absorbing the first block: its rate lanes are the block, its capacity lanes are
    zero. -/
theorem poseidon2In_first {a : Assignment p} {row : ℕ} {b0 b1 b2 b3 b4 b5 b6 b7 : ZMod p}
    (h0 : a (.wire row 0) = b0) (h1 : a (.wire row 1) = b1)
    (h2 : a (.wire row 2) = b2) (h3 : a (.wire row 3) = b3)
    (h4 : a (.wire row 4) = b4) (h5 : a (.wire row 5) = b5)
    (h6 : a (.wire row 6) = b6) (h7 : a (.wire row 7) = b7)
    (h8 : a (.wire row 8) = 0) (h9 : a (.wire row 9) = 0)
    (h10 : a (.wire row 10) = 0) (h11 : a (.wire row 11) = 0) :
    poseidon2In a row = addBlock (fun _ => 0) [b0, b1, b2, b3, b4, b5, b6, b7] := by
  funext j
  fin_cases j <;> simp [poseidon2In, addBlock, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11]

/-- A row absorbing a later block: its rate lanes are the previous row's outputs plus the
    block, its capacity lanes are the previous row's outputs. -/
theorem poseidon2In_chain {a : Assignment p} {row prev : ℕ} {b0 b1 b2 b3 b4 b5 b6 b7 : ZMod p}
    (h0 : a (.wire row 0) = a (.wire prev 12) + b0) (h1 : a (.wire row 1) = a (.wire prev 13) + b1)
    (h2 : a (.wire row 2) = a (.wire prev 14) + b2) (h3 : a (.wire row 3) = a (.wire prev 15) + b3)
    (h4 : a (.wire row 4) = a (.wire prev 16) + b4) (h5 : a (.wire row 5) = a (.wire prev 17) + b5)
    (h6 : a (.wire row 6) = a (.wire prev 18) + b6) (h7 : a (.wire row 7) = a (.wire prev 19) + b7)
    (h8 : a (.wire row 8) = a (.wire prev 20)) (h9 : a (.wire row 9) = a (.wire prev 21))
    (h10 : a (.wire row 10) = a (.wire prev 22)) (h11 : a (.wire row 11) = a (.wire prev 23)) :
    poseidon2In a row = addBlock (poseidon2Out a prev) [b0, b1, b2, b3, b4, b5, b6, b7] := by
  funext j
  fin_cases j <;>
    simp [poseidon2In, poseidon2Out, addBlock, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11]

omit [Fact p.Prime] in
theorem poseidon2Row_absorb (perm : St p → St p) {c : Circuit p} {a : Assignment p}
    (hp : Poseidon2Rows perm c a) {row : ℕ} {r : Row p}
    (hr : c.rows[row]? = some r) (hk : r.kind = .poseidon2) {s : St p} {blk : List (ZMod p)}
    (hin : poseidon2In a row = addBlock s blk) :
    poseidon2Out a row = perm (addBlock s blk) := by
  rw [hp row r hr hk, hin]

end Plonky2Spec.Wiring
