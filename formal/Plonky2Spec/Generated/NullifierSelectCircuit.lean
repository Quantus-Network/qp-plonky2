/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` by building the private-batch
  nullifier-selection path for 2 slots with the real `CircuitBuilder` and walking its
  pre-`build` constraint system (`CircuitBuilder::formal_export_view`): placed gate
  rows with their constants, copy constraints, constant targets, and the public-input
  registration order. Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints

  `Bridges/CircuitBridge.lean` proves that every assignment satisfying this system
  decodes each public input to `bselect is_dummy dnull real` — the wiring-level
  counterpart of the hand-stated decode hypotheses in `Plonky2Bridge`.
-/
import Plonky2Spec.Wiring

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

/-- Nullifier selection for 2 slots: `out[4s+j] = select(isDummy[s], dnull[4s+j], real[4s+j])`, each `isDummy[s]` an `assert_bool`ed virtual target, outputs registered as public inputs. -/
def nullifierSelect2 (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, (-1)]⟩  -- row 0
  ]
  copies := [
    (.virt 0, .wire 0 0),
    (.virt 0, .wire 0 1),
    (.virt 0, .wire 0 2),
    (.wire 0 3, .virt 1),
    (.virt 0, .wire 0 4),
    (.virt 6, .wire 0 5),
    (.virt 6, .wire 0 6),
    (.virt 0, .wire 0 8),
    (.virt 2, .wire 0 9),
    (.wire 0 7, .wire 0 10),
    (.virt 0, .wire 0 12),
    (.virt 7, .wire 0 13),
    (.virt 7, .wire 0 14),
    (.virt 0, .wire 0 16),
    (.virt 3, .wire 0 17),
    (.wire 0 15, .wire 0 18),
    (.virt 0, .wire 0 20),
    (.virt 8, .wire 0 21),
    (.virt 8, .wire 0 22),
    (.virt 0, .wire 0 24),
    (.virt 4, .wire 0 25),
    (.wire 0 23, .wire 0 26),
    (.virt 0, .wire 0 28),
    (.virt 9, .wire 0 29),
    (.virt 9, .wire 0 30),
    (.virt 0, .wire 0 32),
    (.virt 5, .wire 0 33),
    (.wire 0 31, .wire 0 34),
    (.virt 10, .wire 0 36),
    (.virt 10, .wire 0 37),
    (.virt 10, .wire 0 38),
    (.wire 0 39, .virt 1),
    (.virt 10, .wire 0 40),
    (.virt 15, .wire 0 41),
    (.virt 15, .wire 0 42),
    (.virt 10, .wire 0 44),
    (.virt 11, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.virt 10, .wire 0 48),
    (.virt 16, .wire 0 49),
    (.virt 16, .wire 0 50),
    (.virt 10, .wire 0 52),
    (.virt 12, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.virt 10, .wire 0 56),
    (.virt 17, .wire 0 57),
    (.virt 17, .wire 0 58),
    (.virt 10, .wire 0 60),
    (.virt 13, .wire 0 61),
    (.wire 0 59, .wire 0 62),
    (.virt 10, .wire 0 64),
    (.virt 18, .wire 0 65),
    (.virt 18, .wire 0 66),
    (.virt 10, .wire 0 68),
    (.virt 14, .wire 0 69),
    (.wire 0 67, .wire 0 70)
  ]
  constants := [
    (.virt 1, 0)
  ]
  publicInputs := [
    .wire 0 11,
    .wire 0 19,
    .wire 0 27,
    .wire 0 35,
    .wire 0 47,
    .wire 0 55,
    .wire 0 63,
    .wire 0 71
  ]

/-- Named targets `isDummy`. -/
def nullifierSelect2.isDummy : Fin 2 → Target :=
  ![.virt 0, .virt 10]

/-- Named targets `dnull`. -/
def nullifierSelect2.dnull : Fin 8 → Target :=
  ![.virt 2, .virt 3, .virt 4, .virt 5, .virt 11, .virt 12, .virt 13, .virt 14]

/-- Named targets `real`. -/
def nullifierSelect2.real : Fin 8 → Target :=
  ![.virt 6, .virt 7, .virt 8, .virt 9, .virt 15, .virt 16, .virt 17, .virt 18]

/-- Named targets `out`. -/
def nullifierSelect2.out : Fin 8 → Target :=
  ![.wire 0 11, .wire 0 19, .wire 0 27, .wire 0 35, .wire 0 47, .wire 0 55, .wire 0 63, .wire 0 71]

end Plonky2Spec.Generated
