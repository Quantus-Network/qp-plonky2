/-
  Step 8c — the sponge lift for exported `Poseidon2Gate` rows.

  `hash_n_to_hash_no_pad_p2` on four inputs (hashing.rs:60-118) absorbs one block: the
  state fed to the single `Poseidon2Gate` row is `[x0, x1, x2, x3, 1, 0, …, 0]` (inputs,
  the `10*` delimiter, zero fill; `add(zero, ·)` folds so the row's input wires are
  connected straight to the inputs and to the `one`/`zero` constants), and the digest is
  the row's first four output wires. `poseidon2Row_hash4` turns `Poseidon2Rows perm` plus
  the twelve input-wire facts into the `Sponge.spongeHash` form the bridge speaks.
-/
import Plonky2Spec.Wiring
import Plonky2Spec.Sponge

namespace Plonky2Spec.Wiring

open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge

variable {p : ℕ} [Fact p.Prime]

/-- The state a four-input sponge call permutes: `pad10 [x0, x1, x2, x3]` absorbed into
    the zero state. -/
theorem spongeHash_four (perm : St p → St p) (x0 x1 x2 x3 : ZMod p) :
    spongeHash perm [x0, x1, x2, x3]
      = squeeze4 (perm (addBlock (fun _ => 0) [x0, x1, x2, x3, 1, 0, 0, 0])) := by
  rw [spongeHash_short perm _ (by simp [rate])]
  have : pad10 [x0, x1, x2, x3] = [x0, x1, x2, x3, 1, 0, 0, 0] := by
    simp [pad10, rate, List.replicate]
  rw [this]

theorem poseidon2Row_hash4 (perm : St p → St p) {c : Circuit p} {a : Assignment p}
    (hp : Poseidon2Rows perm c a) {row : ℕ} {r : Row p}
    (hr : c.rows[row]? = some r) (hk : r.kind = .poseidon2) {x0 x1 x2 x3 : ZMod p}
    (h0 : a (.wire row 0) = x0) (h1 : a (.wire row 1) = x1)
    (h2 : a (.wire row 2) = x2) (h3 : a (.wire row 3) = x3)
    (h4 : a (.wire row 4) = 1) (h5 : a (.wire row 5) = 0)
    (h6 : a (.wire row 6) = 0) (h7 : a (.wire row 7) = 0)
    (h8 : a (.wire row 8) = 0) (h9 : a (.wire row 9) = 0)
    (h10 : a (.wire row 10) = 0) (h11 : a (.wire row 11) = 0) :
    a (.wire row 12) = spongeHash perm [x0, x1, x2, x3] 0 ∧
    a (.wire row 13) = spongeHash perm [x0, x1, x2, x3] 1 ∧
    a (.wire row 14) = spongeHash perm [x0, x1, x2, x3] 2 ∧
    a (.wire row 15) = spongeHash perm [x0, x1, x2, x3] 3 := by
  have hrow := hp row r hr hk
  have hin : poseidon2In a row = addBlock (fun _ => 0) [x0, x1, x2, x3, 1, 0, 0, 0] := by
    funext j
    fin_cases j <;> simp [poseidon2In, addBlock, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11]
  simp only [spongeHash_four, squeeze4, ← hin, ← hrow]
  exact ⟨rfl, rfl, rfl, rfl⟩

end Plonky2Spec.Wiring
