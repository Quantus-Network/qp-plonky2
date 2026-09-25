/-
  Step 8 (spike) — landing the exported-wiring decode on the object `Plonky2Bridge` consumes.

  `Plonky2Bridge` states the private-batch bridge over `NullSlot` rows and takes the decode
  facts (`IsBool isDummy`, public input `= NullSlot.sel`) as hypotheses. Here those facts
  are *derived* for the 2-slot nullifier-selection path from `Satisfies` on the wiring the
  real `CircuitBuilder` emitted (`Generated/NullifierSelectCircuit.lean`).
-/
import Plonky2Bridge
import Plonky2Spec.Bridges.CircuitBridge

namespace Plonky2Bridge

open Plonky2Spec Plonky2Spec.Wiring Plonky2Spec.Generated

variable {p : ℕ} [Fact p.Prime]

/-- Public-input index of limb `j` of slot `s` (`4s + j`). -/
def limb (s : Fin 2) (j : Fin 4) : Fin 8 := ⟨4 * s.val + j.val, by omega⟩

theorem slotOf_limb (s : Fin 2) (j : Fin 4) : slotOf (limb s j) = s := by
  fin_cases s <;> fin_cases j <;> rfl

/-- The `NullSlot` an assignment induces on slot `s` of `nullifierSelect2`. -/
def wiredSlot (a : Assignment p) (s : Fin 2) : NullSlot p where
  isDummy := a (nullifierSelect2.isDummy s)
  dnull := fun j => a (nullifierSelect2.dnull (limb s j))
  real := fun j => a (nullifierSelect2.real (limb s j))

/-- From the exported wiring alone: each slot's flag is boolean and each public input is
    the slot's `NullSlot.sel` limb — the shape `nullifiers_val_bridge` (`hb`) and
    `PrivateBatchConstraints` consume. -/
theorem nullifierSelect2_wiredSlots (a : Assignment p) (h : Satisfies (nullifierSelect2 p) a) :
    (∀ s, IsBool (wiredSlot a s).isDummy) ∧
    ∀ s j, a (nullifierSelect2.out (limb s j)) = (wiredSlot a s).sel j := by
  obtain ⟨hb, hsel⟩ := nullifierSelect2_decode a h
  refine ⟨hb, fun s j => ?_⟩
  rw [hsel (limb s j), slotOf_limb]
  rfl

end Plonky2Bridge
