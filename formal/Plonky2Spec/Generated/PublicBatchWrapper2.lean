/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from the `n_inner = 2` public-batch wrapper trace
  (`qp-zk-circuits/formal/traces/public_batch_wrapper_n2.json`, recorded by
  `TracingBuilder` while `build_public_batch_constraints` ran on the real builder): the
  pre-`build` constraint system and the gadget calls recorded while building it.
  Each theorem's proof is generated too, one block per recorded gadget call, from
  the ops and copy constraints the builder emitted for it.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

/-- `publicBatchWrapper2.copies`, items `0..32`. -/
def publicBatchWrapper2.copies0 : List (Target × Target) := [
    (.virt 0, .virt 29),
    (.virt 30, .virt 29),
    (.virt 28, .wire 2 0),
    (.virt 28, .wire 2 1),
    (.virt 60, .wire 2 2),
    (.virt 60, .wire 1 48),
    (.virt 3, .wire 1 49),
    (.virt 60, .wire 1 50),
    (.virt 3, .wire 1 52),
    (.virt 61, .wire 1 53),
    (.virt 3, .wire 1 54),
    (.wire 1 55, .wire 2 4),
    (.virt 28, .wire 2 5),
    (.wire 2 3, .wire 2 6),
    (.wire 1 51, .virt 26),
    (.wire 2 7, .virt 26),
    (.virt 28, .wire 2 8),
    (.virt 28, .wire 2 9),
    (.virt 62, .wire 2 10),
    (.virt 62, .wire 1 56),
    (.virt 4, .wire 1 57),
    (.virt 62, .wire 1 58),
    (.virt 4, .wire 1 60),
    (.virt 63, .wire 1 61),
    (.virt 4, .wire 1 62),
    (.wire 1 63, .wire 2 12),
    (.virt 28, .wire 2 13),
    (.wire 2 11, .wire 2 14),
    (.wire 1 59, .virt 26),
    (.wire 2 15, .virt 26),
    (.virt 28, .wire 2 16),
    (.virt 28, .wire 2 17)
  ]

/-- `publicBatchWrapper2.copies`, items `32..64`. -/
def publicBatchWrapper2.copies1 : List (Target × Target) := [
    (.virt 64, .wire 2 18),
    (.virt 64, .wire 1 64),
    (.virt 5, .wire 1 65),
    (.virt 64, .wire 1 66),
    (.virt 5, .wire 1 68),
    (.virt 65, .wire 1 69),
    (.virt 5, .wire 1 70),
    (.wire 1 71, .wire 2 20),
    (.virt 28, .wire 2 21),
    (.wire 2 19, .wire 2 22),
    (.wire 1 67, .virt 26),
    (.wire 2 23, .virt 26),
    (.virt 28, .wire 2 24),
    (.virt 28, .wire 2 25),
    (.virt 66, .wire 2 26),
    (.virt 66, .wire 1 72),
    (.virt 6, .wire 1 73),
    (.virt 66, .wire 1 74),
    (.virt 6, .wire 1 76),
    (.virt 67, .wire 1 77),
    (.virt 6, .wire 1 78),
    (.wire 1 79, .wire 2 28),
    (.virt 28, .wire 2 29),
    (.wire 2 27, .wire 2 30),
    (.wire 1 75, .virt 26),
    (.wire 2 31, .virt 26),
    (.virt 60, .wire 3 0),
    (.virt 62, .wire 3 1),
    (.virt 60, .wire 3 2),
    (.virt 64, .wire 3 4),
    (.virt 66, .wire 3 5),
    (.virt 64, .wire 3 6)
  ]

/-- `publicBatchWrapper2.copies`, items `64..96`. -/
def publicBatchWrapper2.copies2 : List (Target × Target) := [
    (.wire 3 3, .wire 3 8),
    (.wire 3 7, .wire 3 9),
    (.wire 3 3, .wire 3 10),
    (.virt 28, .wire 2 32),
    (.virt 28, .wire 2 33),
    (.virt 68, .wire 2 34),
    (.virt 68, .wire 3 12),
    (.virt 33, .wire 3 13),
    (.virt 68, .wire 3 14),
    (.virt 33, .wire 3 16),
    (.virt 69, .wire 3 17),
    (.virt 33, .wire 3 18),
    (.wire 3 19, .wire 2 36),
    (.virt 28, .wire 2 37),
    (.wire 2 35, .wire 2 38),
    (.wire 3 15, .virt 26),
    (.wire 2 39, .virt 26),
    (.virt 28, .wire 2 40),
    (.virt 28, .wire 2 41),
    (.virt 70, .wire 2 42),
    (.virt 70, .wire 3 20),
    (.virt 34, .wire 3 21),
    (.virt 70, .wire 3 22),
    (.virt 34, .wire 3 24),
    (.virt 71, .wire 3 25),
    (.virt 34, .wire 3 26),
    (.wire 3 27, .wire 2 44),
    (.virt 28, .wire 2 45),
    (.wire 2 43, .wire 2 46),
    (.wire 3 23, .virt 26),
    (.wire 2 47, .virt 26),
    (.virt 28, .wire 2 48)
  ]

/-- `publicBatchWrapper2.copies`, items `96..128`. -/
def publicBatchWrapper2.copies3 : List (Target × Target) := [
    (.virt 28, .wire 2 49),
    (.virt 72, .wire 2 50),
    (.virt 72, .wire 3 28),
    (.virt 35, .wire 3 29),
    (.virt 72, .wire 3 30),
    (.virt 35, .wire 3 32),
    (.virt 73, .wire 3 33),
    (.virt 35, .wire 3 34),
    (.wire 3 35, .wire 2 52),
    (.virt 28, .wire 2 53),
    (.wire 2 51, .wire 2 54),
    (.wire 3 31, .virt 26),
    (.wire 2 55, .virt 26),
    (.virt 28, .wire 2 56),
    (.virt 28, .wire 2 57),
    (.virt 74, .wire 2 58),
    (.virt 74, .wire 3 36),
    (.virt 36, .wire 3 37),
    (.virt 74, .wire 3 38),
    (.virt 36, .wire 3 40),
    (.virt 75, .wire 3 41),
    (.virt 36, .wire 3 42),
    (.wire 3 43, .wire 2 60),
    (.virt 28, .wire 2 61),
    (.wire 2 59, .wire 2 62),
    (.wire 3 39, .virt 26),
    (.wire 2 63, .virt 26),
    (.virt 68, .wire 3 44),
    (.virt 70, .wire 3 45),
    (.virt 68, .wire 3 46),
    (.virt 72, .wire 3 48),
    (.virt 74, .wire 3 49)
  ]

/-- `publicBatchWrapper2.copies`, items `128..160`. -/
def publicBatchWrapper2.copies4 : List (Target × Target) := [
    (.virt 72, .wire 3 50),
    (.wire 3 47, .wire 3 52),
    (.wire 3 51, .wire 3 53),
    (.wire 3 47, .wire 3 54),
    (.virt 28, .wire 2 64),
    (.virt 28, .wire 2 65),
    (.wire 3 11, .wire 2 66),
    (.wire 2 67, .wire 2 68),
    (.virt 3, .wire 2 69),
    (.virt 26, .wire 2 70),
    (.wire 2 67, .wire 2 72),
    (.virt 4, .wire 2 73),
    (.virt 26, .wire 2 74),
    (.wire 2 67, .wire 2 76),
    (.virt 5, .wire 2 77),
    (.virt 26, .wire 2 78),
    (.wire 2 67, .wire 4 0),
    (.virt 6, .wire 4 1),
    (.virt 26, .wire 4 2),
    (.wire 2 67, .wire 4 4),
    (.virt 7, .wire 4 5),
    (.virt 26, .wire 4 6),
    (.wire 2 67, .wire 4 8),
    (.virt 1, .wire 4 9),
    (.virt 26, .wire 4 10),
    (.wire 2 67, .wire 4 12),
    (.virt 2, .wire 4 13),
    (.virt 26, .wire 4 14),
    (.virt 28, .wire 4 16),
    (.virt 28, .wire 4 17),
    (.wire 3 55, .wire 4 18),
    (.virt 28, .wire 4 20)
  ]

/-- `publicBatchWrapper2.copies`, items `160..192`. -/
def publicBatchWrapper2.copies5 : List (Target × Target) := [
    (.virt 28, .wire 4 21),
    (.wire 2 67, .wire 4 22),
    (.wire 4 19, .wire 3 56),
    (.wire 4 23, .wire 3 57),
    (.wire 4 19, .wire 3 58),
    (.wire 3 59, .wire 4 24),
    (.wire 2 71, .wire 4 25),
    (.wire 2 71, .wire 4 26),
    (.wire 3 59, .wire 4 28),
    (.virt 33, .wire 4 29),
    (.wire 4 27, .wire 4 30),
    (.wire 3 59, .wire 4 32),
    (.wire 2 75, .wire 4 33),
    (.wire 2 75, .wire 4 34),
    (.wire 3 59, .wire 4 36),
    (.virt 34, .wire 4 37),
    (.wire 4 35, .wire 4 38),
    (.wire 3 59, .wire 4 40),
    (.wire 2 79, .wire 4 41),
    (.wire 2 79, .wire 4 42),
    (.wire 3 59, .wire 4 44),
    (.virt 35, .wire 4 45),
    (.wire 4 43, .wire 4 46),
    (.wire 3 59, .wire 4 48),
    (.wire 4 3, .wire 4 49),
    (.wire 4 3, .wire 4 50),
    (.wire 3 59, .wire 4 52),
    (.virt 36, .wire 4 53),
    (.wire 4 51, .wire 4 54),
    (.wire 3 59, .wire 4 56),
    (.wire 4 7, .wire 4 57),
    (.wire 4 7, .wire 4 58)
  ]

/-- `publicBatchWrapper2.copies`, items `192..224`. -/
def publicBatchWrapper2.copies6 : List (Target × Target) := [
    (.wire 3 59, .wire 4 60),
    (.virt 37, .wire 4 61),
    (.wire 4 59, .wire 4 62),
    (.wire 3 59, .wire 4 64),
    (.wire 4 11, .wire 4 65),
    (.wire 4 11, .wire 4 66),
    (.wire 3 59, .wire 4 68),
    (.virt 31, .wire 4 69),
    (.wire 4 67, .wire 4 70),
    (.wire 3 59, .wire 4 72),
    (.wire 4 15, .wire 4 73),
    (.wire 4 15, .wire 4 74),
    (.wire 3 59, .wire 4 76),
    (.virt 32, .wire 4 77),
    (.wire 4 75, .wire 4 78),
    (.wire 2 67, .wire 5 0),
    (.wire 4 19, .wire 5 1),
    (.wire 2 67, .wire 5 2),
    (.wire 5 3, .wire 0 64),
    (.virt 28, .wire 0 65),
    (.wire 4 19, .wire 0 66),
    (.virt 28, .wire 6 0),
    (.virt 28, .wire 6 1),
    (.virt 76, .wire 6 2),
    (.virt 1, .wire 6 4),
    (.virt 28, .wire 6 5),
    (.wire 4 71, .wire 6 6),
    (.virt 76, .wire 3 60),
    (.wire 6 7, .wire 3 61),
    (.virt 76, .wire 3 62),
    (.wire 6 7, .wire 3 64),
    (.virt 77, .wire 3 65)
  ]

/-- `publicBatchWrapper2.copies`, items `224..256`. -/
def publicBatchWrapper2.copies7 : List (Target × Target) := [
    (.wire 6 7, .wire 3 66),
    (.wire 3 67, .wire 6 8),
    (.virt 28, .wire 6 9),
    (.wire 6 3, .wire 6 10),
    (.wire 3 63, .virt 26),
    (.wire 6 11, .virt 26),
    (.wire 3 11, .wire 5 4),
    (.virt 76, .wire 5 5),
    (.wire 3 11, .wire 5 6),
    (.wire 5 7, .wire 0 68),
    (.virt 28, .wire 0 69),
    (.virt 76, .wire 0 70),
    (.wire 0 71, .virt 28),
    (.virt 28, .wire 6 12),
    (.virt 28, .wire 6 13),
    (.virt 78, .wire 6 14),
    (.virt 2, .wire 6 16),
    (.virt 28, .wire 6 17),
    (.wire 4 79, .wire 6 18),
    (.virt 78, .wire 3 68),
    (.wire 6 19, .wire 3 69),
    (.virt 78, .wire 3 70),
    (.wire 6 19, .wire 3 72),
    (.virt 79, .wire 3 73),
    (.wire 6 19, .wire 3 74),
    (.wire 3 75, .wire 6 20),
    (.virt 28, .wire 6 21),
    (.wire 6 15, .wire 6 22),
    (.wire 3 71, .virt 26),
    (.wire 6 23, .virt 26),
    (.wire 3 11, .wire 5 8),
    (.virt 78, .wire 5 9)
  ]

/-- `publicBatchWrapper2.copies`, items `256..288`. -/
def publicBatchWrapper2.copies8 : List (Target × Target) := [
    (.wire 3 11, .wire 5 10),
    (.wire 5 11, .wire 0 72),
    (.virt 28, .wire 0 73),
    (.virt 78, .wire 0 74),
    (.wire 0 75, .virt 28),
    (.virt 28, .wire 6 24),
    (.virt 28, .wire 6 25),
    (.virt 80, .wire 6 26),
    (.virt 3, .wire 6 28),
    (.virt 28, .wire 6 29),
    (.wire 4 31, .wire 6 30),
    (.virt 80, .wire 3 76),
    (.wire 6 31, .wire 3 77),
    (.virt 80, .wire 3 78),
    (.wire 6 31, .wire 7 0),
    (.virt 81, .wire 7 1),
    (.wire 6 31, .wire 7 2),
    (.wire 7 3, .wire 6 32),
    (.virt 28, .wire 6 33),
    (.wire 6 27, .wire 6 34),
    (.wire 3 79, .virt 26),
    (.wire 6 35, .virt 26),
    (.virt 28, .wire 6 36),
    (.virt 28, .wire 6 37),
    (.virt 82, .wire 6 38),
    (.virt 4, .wire 6 40),
    (.virt 28, .wire 6 41),
    (.wire 4 39, .wire 6 42),
    (.virt 82, .wire 7 4),
    (.wire 6 43, .wire 7 5),
    (.virt 82, .wire 7 6),
    (.wire 6 43, .wire 7 8)
  ]

/-- `publicBatchWrapper2.copies`, items `288..320`. -/
def publicBatchWrapper2.copies9 : List (Target × Target) := [
    (.virt 83, .wire 7 9),
    (.wire 6 43, .wire 7 10),
    (.wire 7 11, .wire 6 44),
    (.virt 28, .wire 6 45),
    (.wire 6 39, .wire 6 46),
    (.wire 7 7, .virt 26),
    (.wire 6 47, .virt 26),
    (.virt 28, .wire 6 48),
    (.virt 28, .wire 6 49),
    (.virt 84, .wire 6 50),
    (.virt 5, .wire 6 52),
    (.virt 28, .wire 6 53),
    (.wire 4 47, .wire 6 54),
    (.virt 84, .wire 7 12),
    (.wire 6 55, .wire 7 13),
    (.virt 84, .wire 7 14),
    (.wire 6 55, .wire 7 16),
    (.virt 85, .wire 7 17),
    (.wire 6 55, .wire 7 18),
    (.wire 7 19, .wire 6 56),
    (.virt 28, .wire 6 57),
    (.wire 6 51, .wire 6 58),
    (.wire 7 15, .virt 26),
    (.wire 6 59, .virt 26),
    (.virt 28, .wire 6 60),
    (.virt 28, .wire 6 61),
    (.virt 86, .wire 6 62),
    (.virt 6, .wire 6 64),
    (.virt 28, .wire 6 65),
    (.wire 4 55, .wire 6 66),
    (.virt 86, .wire 7 20),
    (.wire 6 67, .wire 7 21)
  ]

/-- `publicBatchWrapper2.copies`, items `320..352`. -/
def publicBatchWrapper2.copies10 : List (Target × Target) := [
    (.virt 86, .wire 7 22),
    (.wire 6 67, .wire 7 24),
    (.virt 87, .wire 7 25),
    (.wire 6 67, .wire 7 26),
    (.wire 7 27, .wire 6 68),
    (.virt 28, .wire 6 69),
    (.wire 6 63, .wire 6 70),
    (.wire 7 23, .virt 26),
    (.wire 6 71, .virt 26),
    (.virt 80, .wire 7 28),
    (.virt 82, .wire 7 29),
    (.virt 80, .wire 7 30),
    (.virt 84, .wire 7 32),
    (.virt 86, .wire 7 33),
    (.virt 84, .wire 7 34),
    (.wire 7 31, .wire 7 36),
    (.wire 7 35, .wire 7 37),
    (.wire 7 31, .wire 7 38),
    (.wire 3 11, .wire 5 12),
    (.wire 7 39, .wire 5 13),
    (.wire 3 11, .wire 5 14),
    (.wire 5 15, .wire 0 76),
    (.virt 28, .wire 0 77),
    (.wire 7 39, .wire 0 78),
    (.wire 0 79, .virt 28),
    (.virt 28, .wire 6 72),
    (.virt 28, .wire 6 73),
    (.virt 88, .wire 6 74),
    (.virt 31, .wire 6 76),
    (.virt 28, .wire 6 77),
    (.wire 4 71, .wire 6 78),
    (.virt 88, .wire 7 40)
  ]

/-- `publicBatchWrapper2.copies`, items `352..384`. -/
def publicBatchWrapper2.copies11 : List (Target × Target) := [
    (.wire 6 79, .wire 7 41),
    (.virt 88, .wire 7 42),
    (.wire 6 79, .wire 7 44),
    (.virt 89, .wire 7 45),
    (.wire 6 79, .wire 7 46),
    (.wire 7 47, .wire 8 0),
    (.virt 28, .wire 8 1),
    (.wire 6 75, .wire 8 2),
    (.wire 7 43, .virt 26),
    (.wire 8 3, .virt 26),
    (.wire 3 55, .wire 5 16),
    (.virt 88, .wire 5 17),
    (.wire 3 55, .wire 5 18),
    (.wire 5 19, .wire 9 0),
    (.virt 28, .wire 9 1),
    (.virt 88, .wire 9 2),
    (.wire 9 3, .virt 28),
    (.virt 28, .wire 8 4),
    (.virt 28, .wire 8 5),
    (.virt 90, .wire 8 6),
    (.virt 32, .wire 8 8),
    (.virt 28, .wire 8 9),
    (.wire 4 79, .wire 8 10),
    (.virt 90, .wire 7 48),
    (.wire 8 11, .wire 7 49),
    (.virt 90, .wire 7 50),
    (.wire 8 11, .wire 7 52),
    (.virt 91, .wire 7 53),
    (.wire 8 11, .wire 7 54),
    (.wire 7 55, .wire 8 12),
    (.virt 28, .wire 8 13),
    (.wire 8 7, .wire 8 14)
  ]

/-- `publicBatchWrapper2.copies`, items `384..416`. -/
def publicBatchWrapper2.copies12 : List (Target × Target) := [
    (.wire 7 51, .virt 26),
    (.wire 8 15, .virt 26),
    (.wire 3 55, .wire 5 20),
    (.virt 90, .wire 5 21),
    (.wire 3 55, .wire 5 22),
    (.wire 5 23, .wire 9 4),
    (.virt 28, .wire 9 5),
    (.virt 90, .wire 9 6),
    (.wire 9 7, .virt 28),
    (.virt 28, .wire 8 16),
    (.virt 28, .wire 8 17),
    (.virt 92, .wire 8 18),
    (.virt 33, .wire 8 20),
    (.virt 28, .wire 8 21),
    (.wire 4 31, .wire 8 22),
    (.virt 92, .wire 7 56),
    (.wire 8 23, .wire 7 57),
    (.virt 92, .wire 7 58),
    (.wire 8 23, .wire 7 60),
    (.virt 93, .wire 7 61),
    (.wire 8 23, .wire 7 62),
    (.wire 7 63, .wire 8 24),
    (.virt 28, .wire 8 25),
    (.wire 8 19, .wire 8 26),
    (.wire 7 59, .virt 26),
    (.wire 8 27, .virt 26),
    (.virt 28, .wire 8 28),
    (.virt 28, .wire 8 29),
    (.virt 94, .wire 8 30),
    (.virt 34, .wire 8 32),
    (.virt 28, .wire 8 33),
    (.wire 4 39, .wire 8 34)
  ]

/-- `publicBatchWrapper2.copies`, items `416..448`. -/
def publicBatchWrapper2.copies13 : List (Target × Target) := [
    (.virt 94, .wire 7 64),
    (.wire 8 35, .wire 7 65),
    (.virt 94, .wire 7 66),
    (.wire 8 35, .wire 7 68),
    (.virt 95, .wire 7 69),
    (.wire 8 35, .wire 7 70),
    (.wire 7 71, .wire 8 36),
    (.virt 28, .wire 8 37),
    (.wire 8 31, .wire 8 38),
    (.wire 7 67, .virt 26),
    (.wire 8 39, .virt 26),
    (.virt 28, .wire 8 40),
    (.virt 28, .wire 8 41),
    (.virt 96, .wire 8 42),
    (.virt 35, .wire 8 44),
    (.virt 28, .wire 8 45),
    (.wire 4 47, .wire 8 46),
    (.virt 96, .wire 7 72),
    (.wire 8 47, .wire 7 73),
    (.virt 96, .wire 7 74),
    (.wire 8 47, .wire 7 76),
    (.virt 97, .wire 7 77),
    (.wire 8 47, .wire 7 78),
    (.wire 7 79, .wire 8 48),
    (.virt 28, .wire 8 49),
    (.wire 8 43, .wire 8 50),
    (.wire 7 75, .virt 26),
    (.wire 8 51, .virt 26),
    (.virt 28, .wire 8 52),
    (.virt 28, .wire 8 53),
    (.virt 98, .wire 8 54),
    (.virt 36, .wire 8 56)
  ]

/-- `publicBatchWrapper2.copies`, items `448..480`. -/
def publicBatchWrapper2.copies14 : List (Target × Target) := [
    (.virt 28, .wire 8 57),
    (.wire 4 55, .wire 8 58),
    (.virt 98, .wire 10 0),
    (.wire 8 59, .wire 10 1),
    (.virt 98, .wire 10 2),
    (.wire 8 59, .wire 10 4),
    (.virt 99, .wire 10 5),
    (.wire 8 59, .wire 10 6),
    (.wire 10 7, .wire 8 60),
    (.virt 28, .wire 8 61),
    (.wire 8 55, .wire 8 62),
    (.wire 10 3, .virt 26),
    (.wire 8 63, .virt 26),
    (.virt 92, .wire 10 8),
    (.virt 94, .wire 10 9),
    (.virt 92, .wire 10 10),
    (.virt 96, .wire 10 12),
    (.virt 98, .wire 10 13),
    (.virt 96, .wire 10 14),
    (.wire 10 11, .wire 10 16),
    (.wire 10 15, .wire 10 17),
    (.wire 10 11, .wire 10 18),
    (.wire 3 55, .wire 5 24),
    (.wire 10 19, .wire 5 25),
    (.wire 3 55, .wire 5 26),
    (.wire 5 27, .wire 9 8),
    (.virt 28, .wire 9 9),
    (.wire 10 19, .wire 9 10),
    (.wire 9 11, .virt 28),
    (.wire 3 11, .wire 8 64),
    (.virt 8, .wire 8 65),
    (.virt 8, .wire 8 66)
  ]

/-- `publicBatchWrapper2.copies`, items `480..512`. -/
def publicBatchWrapper2.copies15 : List (Target × Target) := [
    (.wire 3 11, .wire 8 68),
    (.virt 26, .wire 8 69),
    (.wire 8 67, .wire 8 70),
    (.wire 3 11, .wire 8 72),
    (.virt 9, .wire 8 73),
    (.virt 9, .wire 8 74),
    (.wire 3 11, .wire 8 76),
    (.virt 26, .wire 8 77),
    (.wire 8 75, .wire 8 78),
    (.wire 3 11, .wire 11 0),
    (.virt 10, .wire 11 1),
    (.virt 10, .wire 11 2),
    (.wire 3 11, .wire 11 4),
    (.virt 26, .wire 11 5),
    (.wire 11 3, .wire 11 6),
    (.wire 3 11, .wire 11 8),
    (.virt 11, .wire 11 9),
    (.virt 11, .wire 11 10),
    (.wire 3 11, .wire 11 12),
    (.virt 26, .wire 11 13),
    (.wire 11 11, .wire 11 14),
    (.wire 3 11, .wire 11 16),
    (.virt 12, .wire 11 17),
    (.virt 12, .wire 11 18),
    (.wire 3 11, .wire 11 20),
    (.virt 26, .wire 11 21),
    (.wire 11 19, .wire 11 22),
    (.wire 3 11, .wire 11 24),
    (.virt 13, .wire 11 25),
    (.virt 13, .wire 11 26),
    (.wire 3 11, .wire 11 28),
    (.virt 26, .wire 11 29)
  ]

/-- `publicBatchWrapper2.copies`, items `512..544`. -/
def publicBatchWrapper2.copies16 : List (Target × Target) := [
    (.wire 11 27, .wire 11 30),
    (.wire 3 11, .wire 11 32),
    (.virt 14, .wire 11 33),
    (.virt 14, .wire 11 34),
    (.wire 3 11, .wire 11 36),
    (.virt 26, .wire 11 37),
    (.wire 11 35, .wire 11 38),
    (.wire 3 11, .wire 11 40),
    (.virt 15, .wire 11 41),
    (.virt 15, .wire 11 42),
    (.wire 3 11, .wire 11 44),
    (.virt 26, .wire 11 45),
    (.wire 11 43, .wire 11 46),
    (.wire 3 11, .wire 11 48),
    (.virt 16, .wire 11 49),
    (.virt 16, .wire 11 50),
    (.wire 3 11, .wire 11 52),
    (.virt 26, .wire 11 53),
    (.wire 11 51, .wire 11 54),
    (.wire 3 11, .wire 11 56),
    (.virt 17, .wire 11 57),
    (.virt 17, .wire 11 58),
    (.wire 3 11, .wire 11 60),
    (.virt 26, .wire 11 61),
    (.wire 11 59, .wire 11 62),
    (.wire 3 55, .wire 11 64),
    (.virt 38, .wire 11 65),
    (.virt 38, .wire 11 66),
    (.wire 3 55, .wire 11 68),
    (.virt 26, .wire 11 69),
    (.wire 11 67, .wire 11 70),
    (.wire 3 55, .wire 11 72)
  ]

/-- `publicBatchWrapper2.copies`, items `544..576`. -/
def publicBatchWrapper2.copies17 : List (Target × Target) := [
    (.virt 39, .wire 11 73),
    (.virt 39, .wire 11 74),
    (.wire 3 55, .wire 11 76),
    (.virt 26, .wire 11 77),
    (.wire 11 75, .wire 11 78),
    (.wire 3 55, .wire 12 0),
    (.virt 40, .wire 12 1),
    (.virt 40, .wire 12 2),
    (.wire 3 55, .wire 12 4),
    (.virt 26, .wire 12 5),
    (.wire 12 3, .wire 12 6),
    (.wire 3 55, .wire 12 8),
    (.virt 41, .wire 12 9),
    (.virt 41, .wire 12 10),
    (.wire 3 55, .wire 12 12),
    (.virt 26, .wire 12 13),
    (.wire 12 11, .wire 12 14),
    (.wire 3 55, .wire 12 16),
    (.virt 42, .wire 12 17),
    (.virt 42, .wire 12 18),
    (.wire 3 55, .wire 12 20),
    (.virt 26, .wire 12 21),
    (.wire 12 19, .wire 12 22),
    (.wire 3 55, .wire 12 24),
    (.virt 43, .wire 12 25),
    (.virt 43, .wire 12 26),
    (.wire 3 55, .wire 12 28),
    (.virt 26, .wire 12 29),
    (.wire 12 27, .wire 12 30),
    (.wire 3 55, .wire 12 32),
    (.virt 44, .wire 12 33),
    (.virt 44, .wire 12 34)
  ]

/-- `publicBatchWrapper2.copies`, items `576..608`. -/
def publicBatchWrapper2.copies18 : List (Target × Target) := [
    (.wire 3 55, .wire 12 36),
    (.virt 26, .wire 12 37),
    (.wire 12 35, .wire 12 38),
    (.wire 3 55, .wire 12 40),
    (.virt 45, .wire 12 41),
    (.virt 45, .wire 12 42),
    (.wire 3 55, .wire 12 44),
    (.virt 26, .wire 12 45),
    (.wire 12 43, .wire 12 46),
    (.wire 3 55, .wire 12 48),
    (.virt 46, .wire 12 49),
    (.virt 46, .wire 12 50),
    (.wire 3 55, .wire 12 52),
    (.virt 26, .wire 12 53),
    (.wire 12 51, .wire 12 54),
    (.wire 3 55, .wire 12 56),
    (.virt 47, .wire 12 57),
    (.virt 47, .wire 12 58),
    (.wire 3 55, .wire 12 60),
    (.virt 26, .wire 12 61),
    (.wire 12 59, .wire 12 62),
    (.wire 3 11, .wire 12 64),
    (.virt 18, .wire 12 65),
    (.virt 18, .wire 12 66),
    (.wire 3 11, .wire 12 68),
    (.virt 26, .wire 12 69),
    (.wire 12 67, .wire 12 70),
    (.wire 3 11, .wire 12 72),
    (.virt 19, .wire 12 73),
    (.virt 19, .wire 12 74),
    (.wire 3 11, .wire 12 76),
    (.virt 26, .wire 12 77)
  ]

/-- `publicBatchWrapper2.copies`, items `608..640`. -/
def publicBatchWrapper2.copies19 : List (Target × Target) := [
    (.wire 12 75, .wire 12 78),
    (.wire 3 11, .wire 13 0),
    (.virt 20, .wire 13 1),
    (.virt 20, .wire 13 2),
    (.wire 3 11, .wire 13 4),
    (.virt 26, .wire 13 5),
    (.wire 13 3, .wire 13 6),
    (.wire 3 11, .wire 13 8),
    (.virt 21, .wire 13 9),
    (.virt 21, .wire 13 10),
    (.wire 3 11, .wire 13 12),
    (.virt 26, .wire 13 13),
    (.wire 13 11, .wire 13 14),
    (.wire 3 11, .wire 13 16),
    (.virt 22, .wire 13 17),
    (.virt 22, .wire 13 18),
    (.wire 3 11, .wire 13 20),
    (.virt 26, .wire 13 21),
    (.wire 13 19, .wire 13 22),
    (.wire 3 11, .wire 13 24),
    (.virt 23, .wire 13 25),
    (.virt 23, .wire 13 26),
    (.wire 3 11, .wire 13 28),
    (.virt 26, .wire 13 29),
    (.wire 13 27, .wire 13 30),
    (.wire 3 11, .wire 13 32),
    (.virt 24, .wire 13 33),
    (.virt 24, .wire 13 34),
    (.wire 3 11, .wire 13 36),
    (.virt 26, .wire 13 37),
    (.wire 13 35, .wire 13 38),
    (.wire 3 11, .wire 13 40)
  ]

/-- `publicBatchWrapper2.copies`, items `640..672`. -/
def publicBatchWrapper2.copies20 : List (Target × Target) := [
    (.virt 25, .wire 13 41),
    (.virt 25, .wire 13 42),
    (.wire 3 11, .wire 13 44),
    (.virt 26, .wire 13 45),
    (.wire 13 43, .wire 13 46),
    (.wire 3 55, .wire 13 48),
    (.virt 48, .wire 13 49),
    (.virt 48, .wire 13 50),
    (.wire 3 55, .wire 13 52),
    (.virt 26, .wire 13 53),
    (.wire 13 51, .wire 13 54),
    (.wire 3 55, .wire 13 56),
    (.virt 49, .wire 13 57),
    (.virt 49, .wire 13 58),
    (.wire 3 55, .wire 13 60),
    (.virt 26, .wire 13 61),
    (.wire 13 59, .wire 13 62),
    (.wire 3 55, .wire 13 64),
    (.virt 50, .wire 13 65),
    (.virt 50, .wire 13 66),
    (.wire 3 55, .wire 13 68),
    (.virt 26, .wire 13 69),
    (.wire 13 67, .wire 13 70),
    (.wire 3 55, .wire 13 72),
    (.virt 51, .wire 13 73),
    (.virt 51, .wire 13 74),
    (.wire 3 55, .wire 13 76),
    (.virt 26, .wire 13 77),
    (.wire 13 75, .wire 13 78),
    (.wire 3 55, .wire 14 0),
    (.virt 52, .wire 14 1),
    (.virt 52, .wire 14 2)
  ]

/-- `publicBatchWrapper2.copies`, items `672..693`. -/
def publicBatchWrapper2.copies21 : List (Target × Target) := [
    (.wire 3 55, .wire 14 4),
    (.virt 26, .wire 14 5),
    (.wire 14 3, .wire 14 6),
    (.wire 3 55, .wire 14 8),
    (.virt 53, .wire 14 9),
    (.virt 53, .wire 14 10),
    (.wire 3 55, .wire 14 12),
    (.virt 26, .wire 14 13),
    (.wire 14 11, .wire 14 14),
    (.wire 3 55, .wire 14 16),
    (.virt 54, .wire 14 17),
    (.virt 54, .wire 14 18),
    (.wire 3 55, .wire 14 20),
    (.virt 26, .wire 14 21),
    (.wire 14 19, .wire 14 22),
    (.wire 3 55, .wire 14 24),
    (.virt 55, .wire 14 25),
    (.virt 55, .wire 14 26),
    (.wire 3 55, .wire 14 28),
    (.virt 26, .wire 14 29),
    (.wire 14 27, .wire 14 30)
  ]

/-- The public-batch aggregation wrapper at `n_inner = 2` over the inner private-batch shape named by its recursion gadgets, without the inner verifiers (`wormhole/aggregator/src/public_batch/circuit/circuit_logic.rs`): inner public inputs `inner_pis_0..1`, the `aggregator_address` witness, and the aggregated public inputs. -/
def publicBatchWrapper2 (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 0
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 2
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 3
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 4
    ⟨.arithmetic 20, [(-1), 1]⟩,  -- row 5
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 6
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 7
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 8
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 9
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 10
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 11
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 12
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 13
    ⟨.arithmetic 20, [1, (-1)]⟩  -- row 14
  ]
  copies := ((((publicBatchWrapper2.copies0 ++ publicBatchWrapper2.copies1) ++ (publicBatchWrapper2.copies2 ++ (publicBatchWrapper2.copies3 ++ publicBatchWrapper2.copies4))) ++ ((publicBatchWrapper2.copies5 ++ (publicBatchWrapper2.copies6 ++ publicBatchWrapper2.copies7)) ++ (publicBatchWrapper2.copies8 ++ (publicBatchWrapper2.copies9 ++ publicBatchWrapper2.copies10)))) ++ (((publicBatchWrapper2.copies11 ++ publicBatchWrapper2.copies12) ++ (publicBatchWrapper2.copies13 ++ (publicBatchWrapper2.copies14 ++ publicBatchWrapper2.copies15))) ++ ((publicBatchWrapper2.copies16 ++ (publicBatchWrapper2.copies17 ++ publicBatchWrapper2.copies18)) ++ (publicBatchWrapper2.copies19 ++ (publicBatchWrapper2.copies20 ++ publicBatchWrapper2.copies21)))))
  constants := [
    (.virt 26, 0),
    (.virt 27, 4),
    (.virt 28, 1),
    (.virt 29, 2)
  ]
  publicInputs := [
    .virt 56,
    .virt 57,
    .virt 58,
    .virt 59,
    .wire 4 71,
    .wire 4 79,
    .wire 4 31,
    .wire 4 39,
    .wire 4 47,
    .wire 4 55,
    .wire 4 63,
    .virt 27,
    .wire 8 71,
    .wire 8 79,
    .wire 11 7,
    .wire 11 15,
    .wire 11 23,
    .wire 11 31,
    .wire 11 39,
    .wire 11 47,
    .wire 11 55,
    .wire 11 63,
    .wire 11 71,
    .wire 11 79,
    .wire 12 7,
    .wire 12 15,
    .wire 12 23,
    .wire 12 31,
    .wire 12 39,
    .wire 12 47,
    .wire 12 55,
    .wire 12 63,
    .wire 12 71,
    .wire 12 79,
    .wire 13 7,
    .wire 13 15,
    .wire 13 23,
    .wire 13 31,
    .wire 13 39,
    .wire 13 47,
    .wire 13 55,
    .wire 13 63,
    .wire 13 71,
    .wire 13 79,
    .wire 14 7,
    .wire 14 15,
    .wire 14 23,
    .wire 14 31
  ]

/-- Named targets `inner_pis_0`. -/
def publicBatchWrapper2.inner_pis_0 : Fin 26 → Target :=
  ![.virt 0, .virt 1, .virt 2, .virt 3, .virt 4, .virt 5, .virt 6, .virt 7, .virt 8, .virt 9, .virt 10, .virt 11, .virt 12, .virt 13, .virt 14, .virt 15, .virt 16, .virt 17, .virt 18, .virt 19, .virt 20, .virt 21, .virt 22, .virt 23, .virt 24, .virt 25]

/-- Named targets `inner_pis_1`. -/
def publicBatchWrapper2.inner_pis_1 : Fin 26 → Target :=
  ![.virt 30, .virt 31, .virt 32, .virt 33, .virt 34, .virt 35, .virt 36, .virt 37, .virt 38, .virt 39, .virt 40, .virt 41, .virt 42, .virt 43, .virt 44, .virt 45, .virt 46, .virt 47, .virt 48, .virt 49, .virt 50, .virt 51, .virt 52, .virt 53, .virt 54, .virt 55]

/-- Named targets `aggregator_address`. -/
def publicBatchWrapper2.aggregator_address : Fin 4 → Target :=
  ![.virt 56, .virt 57, .virt 58, .virt 59]

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def publicBatchWrapper2.verifiers : List (String × List Target) :=
  [
    ("private_batch_wrapper_n2", [.virt 0, .virt 1, .virt 2, .virt 3, .virt 4, .virt 5, .virt 6, .virt 7, .virt 8, .virt 9, .virt 10, .virt 11, .virt 12, .virt 13, .virt 14, .virt 15, .virt 16, .virt 17, .virt 18, .virt 19, .virt 20, .virt 21, .virt 22, .virt 23, .virt 24, .virt 25]),
    ("private_batch_wrapper_n2", [.virt 30, .virt 31, .virt 32, .virt 33, .virt 34, .virt 35, .virt 36, .virt 37, .virt 38, .virt 39, .virt 40, .virt 41, .virt 42, .virt 43, .virt 44, .virt 45, .virt 46, .virt 47, .virt 48, .virt 49, .virt 50, .virt 51, .virt 52, .virt 53, .virt 54, .virt 55])
  ]

theorem publicBatchWrapper2_copies0 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 0) = a (.virt 29) ∧ a (.virt 30) = a (.virt 29) ∧ a (.virt 28) = a (.wire 2 0) ∧ a (.virt 28) = a (.wire 2 1) ∧ a (.virt 60) = a (.wire 2 2) ∧ a (.virt 60) = a (.wire 1 48) ∧ a (.virt 3) = a (.wire 1 49) ∧ a (.virt 60) = a (.wire 1 50) ∧ a (.virt 3) = a (.wire 1 52) ∧ a (.virt 61) = a (.wire 1 53) ∧ a (.virt 3) = a (.wire 1 54) ∧ a (.wire 1 55) = a (.wire 2 4) ∧ a (.virt 28) = a (.wire 2 5) ∧ a (.wire 2 3) = a (.wire 2 6) ∧ a (.wire 1 51) = a (.virt 26) ∧ a (.wire 2 7) = a (.virt 26) ∧ a (.virt 28) = a (.wire 2 8) ∧ a (.virt 28) = a (.wire 2 9) ∧ a (.virt 62) = a (.wire 2 10) ∧ a (.virt 62) = a (.wire 1 56) ∧ a (.virt 4) = a (.wire 1 57) ∧ a (.virt 62) = a (.wire 1 58) ∧ a (.virt 4) = a (.wire 1 60) ∧ a (.virt 63) = a (.wire 1 61) ∧ a (.virt 4) = a (.wire 1 62) ∧ a (.wire 1 63) = a (.wire 2 12) ∧ a (.virt 28) = a (.wire 2 13) ∧ a (.wire 2 11) = a (.wire 2 14) ∧ a (.wire 1 59) = a (.virt 26) ∧ a (.wire 2 15) = a (.virt 26) ∧ a (.virt 28) = a (.wire 2 16) ∧ a (.virt 28) = a (.wire 2 17) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies1 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 64) = a (.wire 2 18) ∧ a (.virt 64) = a (.wire 1 64) ∧ a (.virt 5) = a (.wire 1 65) ∧ a (.virt 64) = a (.wire 1 66) ∧ a (.virt 5) = a (.wire 1 68) ∧ a (.virt 65) = a (.wire 1 69) ∧ a (.virt 5) = a (.wire 1 70) ∧ a (.wire 1 71) = a (.wire 2 20) ∧ a (.virt 28) = a (.wire 2 21) ∧ a (.wire 2 19) = a (.wire 2 22) ∧ a (.wire 1 67) = a (.virt 26) ∧ a (.wire 2 23) = a (.virt 26) ∧ a (.virt 28) = a (.wire 2 24) ∧ a (.virt 28) = a (.wire 2 25) ∧ a (.virt 66) = a (.wire 2 26) ∧ a (.virt 66) = a (.wire 1 72) ∧ a (.virt 6) = a (.wire 1 73) ∧ a (.virt 66) = a (.wire 1 74) ∧ a (.virt 6) = a (.wire 1 76) ∧ a (.virt 67) = a (.wire 1 77) ∧ a (.virt 6) = a (.wire 1 78) ∧ a (.wire 1 79) = a (.wire 2 28) ∧ a (.virt 28) = a (.wire 2 29) ∧ a (.wire 2 27) = a (.wire 2 30) ∧ a (.wire 1 75) = a (.virt 26) ∧ a (.wire 2 31) = a (.virt 26) ∧ a (.virt 60) = a (.wire 3 0) ∧ a (.virt 62) = a (.wire 3 1) ∧ a (.virt 60) = a (.wire 3 2) ∧ a (.virt 64) = a (.wire 3 4) ∧ a (.virt 66) = a (.wire 3 5) ∧ a (.virt 64) = a (.wire 3 6) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))
  simp only [publicBatchWrapper2.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies2 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 3) = a (.wire 3 8) ∧ a (.wire 3 7) = a (.wire 3 9) ∧ a (.wire 3 3) = a (.wire 3 10) ∧ a (.virt 28) = a (.wire 2 32) ∧ a (.virt 28) = a (.wire 2 33) ∧ a (.virt 68) = a (.wire 2 34) ∧ a (.virt 68) = a (.wire 3 12) ∧ a (.virt 33) = a (.wire 3 13) ∧ a (.virt 68) = a (.wire 3 14) ∧ a (.virt 33) = a (.wire 3 16) ∧ a (.virt 69) = a (.wire 3 17) ∧ a (.virt 33) = a (.wire 3 18) ∧ a (.wire 3 19) = a (.wire 2 36) ∧ a (.virt 28) = a (.wire 2 37) ∧ a (.wire 2 35) = a (.wire 2 38) ∧ a (.wire 3 15) = a (.virt 26) ∧ a (.wire 2 39) = a (.virt 26) ∧ a (.virt 28) = a (.wire 2 40) ∧ a (.virt 28) = a (.wire 2 41) ∧ a (.virt 70) = a (.wire 2 42) ∧ a (.virt 70) = a (.wire 3 20) ∧ a (.virt 34) = a (.wire 3 21) ∧ a (.virt 70) = a (.wire 3 22) ∧ a (.virt 34) = a (.wire 3 24) ∧ a (.virt 71) = a (.wire 3 25) ∧ a (.virt 34) = a (.wire 3 26) ∧ a (.wire 3 27) = a (.wire 2 44) ∧ a (.virt 28) = a (.wire 2 45) ∧ a (.wire 2 43) = a (.wire 2 46) ∧ a (.wire 3 23) = a (.virt 26) ∧ a (.wire 2 47) = a (.virt 26) ∧ a (.virt 28) = a (.wire 2 48) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies3 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 28) = a (.wire 2 49) ∧ a (.virt 72) = a (.wire 2 50) ∧ a (.virt 72) = a (.wire 3 28) ∧ a (.virt 35) = a (.wire 3 29) ∧ a (.virt 72) = a (.wire 3 30) ∧ a (.virt 35) = a (.wire 3 32) ∧ a (.virt 73) = a (.wire 3 33) ∧ a (.virt 35) = a (.wire 3 34) ∧ a (.wire 3 35) = a (.wire 2 52) ∧ a (.virt 28) = a (.wire 2 53) ∧ a (.wire 2 51) = a (.wire 2 54) ∧ a (.wire 3 31) = a (.virt 26) ∧ a (.wire 2 55) = a (.virt 26) ∧ a (.virt 28) = a (.wire 2 56) ∧ a (.virt 28) = a (.wire 2 57) ∧ a (.virt 74) = a (.wire 2 58) ∧ a (.virt 74) = a (.wire 3 36) ∧ a (.virt 36) = a (.wire 3 37) ∧ a (.virt 74) = a (.wire 3 38) ∧ a (.virt 36) = a (.wire 3 40) ∧ a (.virt 75) = a (.wire 3 41) ∧ a (.virt 36) = a (.wire 3 42) ∧ a (.wire 3 43) = a (.wire 2 60) ∧ a (.virt 28) = a (.wire 2 61) ∧ a (.wire 2 59) = a (.wire 2 62) ∧ a (.wire 3 39) = a (.virt 26) ∧ a (.wire 2 63) = a (.virt 26) ∧ a (.virt 68) = a (.wire 3 44) ∧ a (.virt 70) = a (.wire 3 45) ∧ a (.virt 68) = a (.wire 3 46) ∧ a (.virt 72) = a (.wire 3 48) ∧ a (.virt 74) = a (.wire 3 49) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies3, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies4 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 72) = a (.wire 3 50) ∧ a (.wire 3 47) = a (.wire 3 52) ∧ a (.wire 3 51) = a (.wire 3 53) ∧ a (.wire 3 47) = a (.wire 3 54) ∧ a (.virt 28) = a (.wire 2 64) ∧ a (.virt 28) = a (.wire 2 65) ∧ a (.wire 3 11) = a (.wire 2 66) ∧ a (.wire 2 67) = a (.wire 2 68) ∧ a (.virt 3) = a (.wire 2 69) ∧ a (.virt 26) = a (.wire 2 70) ∧ a (.wire 2 67) = a (.wire 2 72) ∧ a (.virt 4) = a (.wire 2 73) ∧ a (.virt 26) = a (.wire 2 74) ∧ a (.wire 2 67) = a (.wire 2 76) ∧ a (.virt 5) = a (.wire 2 77) ∧ a (.virt 26) = a (.wire 2 78) ∧ a (.wire 2 67) = a (.wire 4 0) ∧ a (.virt 6) = a (.wire 4 1) ∧ a (.virt 26) = a (.wire 4 2) ∧ a (.wire 2 67) = a (.wire 4 4) ∧ a (.virt 7) = a (.wire 4 5) ∧ a (.virt 26) = a (.wire 4 6) ∧ a (.wire 2 67) = a (.wire 4 8) ∧ a (.virt 1) = a (.wire 4 9) ∧ a (.virt 26) = a (.wire 4 10) ∧ a (.wire 2 67) = a (.wire 4 12) ∧ a (.virt 2) = a (.wire 4 13) ∧ a (.virt 26) = a (.wire 4 14) ∧ a (.virt 28) = a (.wire 4 16) ∧ a (.virt 28) = a (.wire 4 17) ∧ a (.wire 3 55) = a (.wire 4 18) ∧ a (.virt 28) = a (.wire 4 20) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies4, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies5 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 28) = a (.wire 4 21) ∧ a (.wire 2 67) = a (.wire 4 22) ∧ a (.wire 4 19) = a (.wire 3 56) ∧ a (.wire 4 23) = a (.wire 3 57) ∧ a (.wire 4 19) = a (.wire 3 58) ∧ a (.wire 3 59) = a (.wire 4 24) ∧ a (.wire 2 71) = a (.wire 4 25) ∧ a (.wire 2 71) = a (.wire 4 26) ∧ a (.wire 3 59) = a (.wire 4 28) ∧ a (.virt 33) = a (.wire 4 29) ∧ a (.wire 4 27) = a (.wire 4 30) ∧ a (.wire 3 59) = a (.wire 4 32) ∧ a (.wire 2 75) = a (.wire 4 33) ∧ a (.wire 2 75) = a (.wire 4 34) ∧ a (.wire 3 59) = a (.wire 4 36) ∧ a (.virt 34) = a (.wire 4 37) ∧ a (.wire 4 35) = a (.wire 4 38) ∧ a (.wire 3 59) = a (.wire 4 40) ∧ a (.wire 2 79) = a (.wire 4 41) ∧ a (.wire 2 79) = a (.wire 4 42) ∧ a (.wire 3 59) = a (.wire 4 44) ∧ a (.virt 35) = a (.wire 4 45) ∧ a (.wire 4 43) = a (.wire 4 46) ∧ a (.wire 3 59) = a (.wire 4 48) ∧ a (.wire 4 3) = a (.wire 4 49) ∧ a (.wire 4 3) = a (.wire 4 50) ∧ a (.wire 3 59) = a (.wire 4 52) ∧ a (.virt 36) = a (.wire 4 53) ∧ a (.wire 4 51) = a (.wire 4 54) ∧ a (.wire 3 59) = a (.wire 4 56) ∧ a (.wire 4 7) = a (.wire 4 57) ∧ a (.wire 4 7) = a (.wire 4 58) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies5, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies6 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 59) = a (.wire 4 60) ∧ a (.virt 37) = a (.wire 4 61) ∧ a (.wire 4 59) = a (.wire 4 62) ∧ a (.wire 3 59) = a (.wire 4 64) ∧ a (.wire 4 11) = a (.wire 4 65) ∧ a (.wire 4 11) = a (.wire 4 66) ∧ a (.wire 3 59) = a (.wire 4 68) ∧ a (.virt 31) = a (.wire 4 69) ∧ a (.wire 4 67) = a (.wire 4 70) ∧ a (.wire 3 59) = a (.wire 4 72) ∧ a (.wire 4 15) = a (.wire 4 73) ∧ a (.wire 4 15) = a (.wire 4 74) ∧ a (.wire 3 59) = a (.wire 4 76) ∧ a (.virt 32) = a (.wire 4 77) ∧ a (.wire 4 75) = a (.wire 4 78) ∧ a (.wire 2 67) = a (.wire 5 0) ∧ a (.wire 4 19) = a (.wire 5 1) ∧ a (.wire 2 67) = a (.wire 5 2) ∧ a (.wire 5 3) = a (.wire 0 64) ∧ a (.virt 28) = a (.wire 0 65) ∧ a (.wire 4 19) = a (.wire 0 66) ∧ a (.virt 28) = a (.wire 6 0) ∧ a (.virt 28) = a (.wire 6 1) ∧ a (.virt 76) = a (.wire 6 2) ∧ a (.virt 1) = a (.wire 6 4) ∧ a (.virt 28) = a (.wire 6 5) ∧ a (.wire 4 71) = a (.wire 6 6) ∧ a (.virt 76) = a (.wire 3 60) ∧ a (.wire 6 7) = a (.wire 3 61) ∧ a (.virt 76) = a (.wire 3 62) ∧ a (.wire 6 7) = a (.wire 3 64) ∧ a (.virt 77) = a (.wire 3 65) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies6, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies7 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 6 7) = a (.wire 3 66) ∧ a (.wire 3 67) = a (.wire 6 8) ∧ a (.virt 28) = a (.wire 6 9) ∧ a (.wire 6 3) = a (.wire 6 10) ∧ a (.wire 3 63) = a (.virt 26) ∧ a (.wire 6 11) = a (.virt 26) ∧ a (.wire 3 11) = a (.wire 5 4) ∧ a (.virt 76) = a (.wire 5 5) ∧ a (.wire 3 11) = a (.wire 5 6) ∧ a (.wire 5 7) = a (.wire 0 68) ∧ a (.virt 28) = a (.wire 0 69) ∧ a (.virt 76) = a (.wire 0 70) ∧ a (.wire 0 71) = a (.virt 28) ∧ a (.virt 28) = a (.wire 6 12) ∧ a (.virt 28) = a (.wire 6 13) ∧ a (.virt 78) = a (.wire 6 14) ∧ a (.virt 2) = a (.wire 6 16) ∧ a (.virt 28) = a (.wire 6 17) ∧ a (.wire 4 79) = a (.wire 6 18) ∧ a (.virt 78) = a (.wire 3 68) ∧ a (.wire 6 19) = a (.wire 3 69) ∧ a (.virt 78) = a (.wire 3 70) ∧ a (.wire 6 19) = a (.wire 3 72) ∧ a (.virt 79) = a (.wire 3 73) ∧ a (.wire 6 19) = a (.wire 3 74) ∧ a (.wire 3 75) = a (.wire 6 20) ∧ a (.virt 28) = a (.wire 6 21) ∧ a (.wire 6 15) = a (.wire 6 22) ∧ a (.wire 3 71) = a (.virt 26) ∧ a (.wire 6 23) = a (.virt 26) ∧ a (.wire 3 11) = a (.wire 5 8) ∧ a (.virt 78) = a (.wire 5 9) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies7, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies8 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 11) = a (.wire 5 10) ∧ a (.wire 5 11) = a (.wire 0 72) ∧ a (.virt 28) = a (.wire 0 73) ∧ a (.virt 78) = a (.wire 0 74) ∧ a (.wire 0 75) = a (.virt 28) ∧ a (.virt 28) = a (.wire 6 24) ∧ a (.virt 28) = a (.wire 6 25) ∧ a (.virt 80) = a (.wire 6 26) ∧ a (.virt 3) = a (.wire 6 28) ∧ a (.virt 28) = a (.wire 6 29) ∧ a (.wire 4 31) = a (.wire 6 30) ∧ a (.virt 80) = a (.wire 3 76) ∧ a (.wire 6 31) = a (.wire 3 77) ∧ a (.virt 80) = a (.wire 3 78) ∧ a (.wire 6 31) = a (.wire 7 0) ∧ a (.virt 81) = a (.wire 7 1) ∧ a (.wire 6 31) = a (.wire 7 2) ∧ a (.wire 7 3) = a (.wire 6 32) ∧ a (.virt 28) = a (.wire 6 33) ∧ a (.wire 6 27) = a (.wire 6 34) ∧ a (.wire 3 79) = a (.virt 26) ∧ a (.wire 6 35) = a (.virt 26) ∧ a (.virt 28) = a (.wire 6 36) ∧ a (.virt 28) = a (.wire 6 37) ∧ a (.virt 82) = a (.wire 6 38) ∧ a (.virt 4) = a (.wire 6 40) ∧ a (.virt 28) = a (.wire 6 41) ∧ a (.wire 4 39) = a (.wire 6 42) ∧ a (.virt 82) = a (.wire 7 4) ∧ a (.wire 6 43) = a (.wire 7 5) ∧ a (.virt 82) = a (.wire 7 6) ∧ a (.wire 6 43) = a (.wire 7 8) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies8, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies9 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 83) = a (.wire 7 9) ∧ a (.wire 6 43) = a (.wire 7 10) ∧ a (.wire 7 11) = a (.wire 6 44) ∧ a (.virt 28) = a (.wire 6 45) ∧ a (.wire 6 39) = a (.wire 6 46) ∧ a (.wire 7 7) = a (.virt 26) ∧ a (.wire 6 47) = a (.virt 26) ∧ a (.virt 28) = a (.wire 6 48) ∧ a (.virt 28) = a (.wire 6 49) ∧ a (.virt 84) = a (.wire 6 50) ∧ a (.virt 5) = a (.wire 6 52) ∧ a (.virt 28) = a (.wire 6 53) ∧ a (.wire 4 47) = a (.wire 6 54) ∧ a (.virt 84) = a (.wire 7 12) ∧ a (.wire 6 55) = a (.wire 7 13) ∧ a (.virt 84) = a (.wire 7 14) ∧ a (.wire 6 55) = a (.wire 7 16) ∧ a (.virt 85) = a (.wire 7 17) ∧ a (.wire 6 55) = a (.wire 7 18) ∧ a (.wire 7 19) = a (.wire 6 56) ∧ a (.virt 28) = a (.wire 6 57) ∧ a (.wire 6 51) = a (.wire 6 58) ∧ a (.wire 7 15) = a (.virt 26) ∧ a (.wire 6 59) = a (.virt 26) ∧ a (.virt 28) = a (.wire 6 60) ∧ a (.virt 28) = a (.wire 6 61) ∧ a (.virt 86) = a (.wire 6 62) ∧ a (.virt 6) = a (.wire 6 64) ∧ a (.virt 28) = a (.wire 6 65) ∧ a (.wire 4 55) = a (.wire 6 66) ∧ a (.virt 86) = a (.wire 7 20) ∧ a (.wire 6 67) = a (.wire 7 21) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies9, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies10 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 86) = a (.wire 7 22) ∧ a (.wire 6 67) = a (.wire 7 24) ∧ a (.virt 87) = a (.wire 7 25) ∧ a (.wire 6 67) = a (.wire 7 26) ∧ a (.wire 7 27) = a (.wire 6 68) ∧ a (.virt 28) = a (.wire 6 69) ∧ a (.wire 6 63) = a (.wire 6 70) ∧ a (.wire 7 23) = a (.virt 26) ∧ a (.wire 6 71) = a (.virt 26) ∧ a (.virt 80) = a (.wire 7 28) ∧ a (.virt 82) = a (.wire 7 29) ∧ a (.virt 80) = a (.wire 7 30) ∧ a (.virt 84) = a (.wire 7 32) ∧ a (.virt 86) = a (.wire 7 33) ∧ a (.virt 84) = a (.wire 7 34) ∧ a (.wire 7 31) = a (.wire 7 36) ∧ a (.wire 7 35) = a (.wire 7 37) ∧ a (.wire 7 31) = a (.wire 7 38) ∧ a (.wire 3 11) = a (.wire 5 12) ∧ a (.wire 7 39) = a (.wire 5 13) ∧ a (.wire 3 11) = a (.wire 5 14) ∧ a (.wire 5 15) = a (.wire 0 76) ∧ a (.virt 28) = a (.wire 0 77) ∧ a (.wire 7 39) = a (.wire 0 78) ∧ a (.wire 0 79) = a (.virt 28) ∧ a (.virt 28) = a (.wire 6 72) ∧ a (.virt 28) = a (.wire 6 73) ∧ a (.virt 88) = a (.wire 6 74) ∧ a (.virt 31) = a (.wire 6 76) ∧ a (.virt 28) = a (.wire 6 77) ∧ a (.wire 4 71) = a (.wire 6 78) ∧ a (.virt 88) = a (.wire 7 40) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies10, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies11 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 6 79) = a (.wire 7 41) ∧ a (.virt 88) = a (.wire 7 42) ∧ a (.wire 6 79) = a (.wire 7 44) ∧ a (.virt 89) = a (.wire 7 45) ∧ a (.wire 6 79) = a (.wire 7 46) ∧ a (.wire 7 47) = a (.wire 8 0) ∧ a (.virt 28) = a (.wire 8 1) ∧ a (.wire 6 75) = a (.wire 8 2) ∧ a (.wire 7 43) = a (.virt 26) ∧ a (.wire 8 3) = a (.virt 26) ∧ a (.wire 3 55) = a (.wire 5 16) ∧ a (.virt 88) = a (.wire 5 17) ∧ a (.wire 3 55) = a (.wire 5 18) ∧ a (.wire 5 19) = a (.wire 9 0) ∧ a (.virt 28) = a (.wire 9 1) ∧ a (.virt 88) = a (.wire 9 2) ∧ a (.wire 9 3) = a (.virt 28) ∧ a (.virt 28) = a (.wire 8 4) ∧ a (.virt 28) = a (.wire 8 5) ∧ a (.virt 90) = a (.wire 8 6) ∧ a (.virt 32) = a (.wire 8 8) ∧ a (.virt 28) = a (.wire 8 9) ∧ a (.wire 4 79) = a (.wire 8 10) ∧ a (.virt 90) = a (.wire 7 48) ∧ a (.wire 8 11) = a (.wire 7 49) ∧ a (.virt 90) = a (.wire 7 50) ∧ a (.wire 8 11) = a (.wire 7 52) ∧ a (.virt 91) = a (.wire 7 53) ∧ a (.wire 8 11) = a (.wire 7 54) ∧ a (.wire 7 55) = a (.wire 8 12) ∧ a (.virt 28) = a (.wire 8 13) ∧ a (.wire 8 7) = a (.wire 8 14) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies11, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies12 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 51) = a (.virt 26) ∧ a (.wire 8 15) = a (.virt 26) ∧ a (.wire 3 55) = a (.wire 5 20) ∧ a (.virt 90) = a (.wire 5 21) ∧ a (.wire 3 55) = a (.wire 5 22) ∧ a (.wire 5 23) = a (.wire 9 4) ∧ a (.virt 28) = a (.wire 9 5) ∧ a (.virt 90) = a (.wire 9 6) ∧ a (.wire 9 7) = a (.virt 28) ∧ a (.virt 28) = a (.wire 8 16) ∧ a (.virt 28) = a (.wire 8 17) ∧ a (.virt 92) = a (.wire 8 18) ∧ a (.virt 33) = a (.wire 8 20) ∧ a (.virt 28) = a (.wire 8 21) ∧ a (.wire 4 31) = a (.wire 8 22) ∧ a (.virt 92) = a (.wire 7 56) ∧ a (.wire 8 23) = a (.wire 7 57) ∧ a (.virt 92) = a (.wire 7 58) ∧ a (.wire 8 23) = a (.wire 7 60) ∧ a (.virt 93) = a (.wire 7 61) ∧ a (.wire 8 23) = a (.wire 7 62) ∧ a (.wire 7 63) = a (.wire 8 24) ∧ a (.virt 28) = a (.wire 8 25) ∧ a (.wire 8 19) = a (.wire 8 26) ∧ a (.wire 7 59) = a (.virt 26) ∧ a (.wire 8 27) = a (.virt 26) ∧ a (.virt 28) = a (.wire 8 28) ∧ a (.virt 28) = a (.wire 8 29) ∧ a (.virt 94) = a (.wire 8 30) ∧ a (.virt 34) = a (.wire 8 32) ∧ a (.virt 28) = a (.wire 8 33) ∧ a (.wire 4 39) = a (.wire 8 34) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies12, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))
  simp only [publicBatchWrapper2.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies13 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 94) = a (.wire 7 64) ∧ a (.wire 8 35) = a (.wire 7 65) ∧ a (.virt 94) = a (.wire 7 66) ∧ a (.wire 8 35) = a (.wire 7 68) ∧ a (.virt 95) = a (.wire 7 69) ∧ a (.wire 8 35) = a (.wire 7 70) ∧ a (.wire 7 71) = a (.wire 8 36) ∧ a (.virt 28) = a (.wire 8 37) ∧ a (.wire 8 31) = a (.wire 8 38) ∧ a (.wire 7 67) = a (.virt 26) ∧ a (.wire 8 39) = a (.virt 26) ∧ a (.virt 28) = a (.wire 8 40) ∧ a (.virt 28) = a (.wire 8 41) ∧ a (.virt 96) = a (.wire 8 42) ∧ a (.virt 35) = a (.wire 8 44) ∧ a (.virt 28) = a (.wire 8 45) ∧ a (.wire 4 47) = a (.wire 8 46) ∧ a (.virt 96) = a (.wire 7 72) ∧ a (.wire 8 47) = a (.wire 7 73) ∧ a (.virt 96) = a (.wire 7 74) ∧ a (.wire 8 47) = a (.wire 7 76) ∧ a (.virt 97) = a (.wire 7 77) ∧ a (.wire 8 47) = a (.wire 7 78) ∧ a (.wire 7 79) = a (.wire 8 48) ∧ a (.virt 28) = a (.wire 8 49) ∧ a (.wire 8 43) = a (.wire 8 50) ∧ a (.wire 7 75) = a (.virt 26) ∧ a (.wire 8 51) = a (.virt 26) ∧ a (.virt 28) = a (.wire 8 52) ∧ a (.virt 28) = a (.wire 8 53) ∧ a (.virt 98) = a (.wire 8 54) ∧ a (.virt 36) = a (.wire 8 56) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies13, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies14 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 28) = a (.wire 8 57) ∧ a (.wire 4 55) = a (.wire 8 58) ∧ a (.virt 98) = a (.wire 10 0) ∧ a (.wire 8 59) = a (.wire 10 1) ∧ a (.virt 98) = a (.wire 10 2) ∧ a (.wire 8 59) = a (.wire 10 4) ∧ a (.virt 99) = a (.wire 10 5) ∧ a (.wire 8 59) = a (.wire 10 6) ∧ a (.wire 10 7) = a (.wire 8 60) ∧ a (.virt 28) = a (.wire 8 61) ∧ a (.wire 8 55) = a (.wire 8 62) ∧ a (.wire 10 3) = a (.virt 26) ∧ a (.wire 8 63) = a (.virt 26) ∧ a (.virt 92) = a (.wire 10 8) ∧ a (.virt 94) = a (.wire 10 9) ∧ a (.virt 92) = a (.wire 10 10) ∧ a (.virt 96) = a (.wire 10 12) ∧ a (.virt 98) = a (.wire 10 13) ∧ a (.virt 96) = a (.wire 10 14) ∧ a (.wire 10 11) = a (.wire 10 16) ∧ a (.wire 10 15) = a (.wire 10 17) ∧ a (.wire 10 11) = a (.wire 10 18) ∧ a (.wire 3 55) = a (.wire 5 24) ∧ a (.wire 10 19) = a (.wire 5 25) ∧ a (.wire 3 55) = a (.wire 5 26) ∧ a (.wire 5 27) = a (.wire 9 8) ∧ a (.virt 28) = a (.wire 9 9) ∧ a (.wire 10 19) = a (.wire 9 10) ∧ a (.wire 9 11) = a (.virt 28) ∧ a (.wire 3 11) = a (.wire 8 64) ∧ a (.virt 8) = a (.wire 8 65) ∧ a (.virt 8) = a (.wire 8 66) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies14, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies15 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 11) = a (.wire 8 68) ∧ a (.virt 26) = a (.wire 8 69) ∧ a (.wire 8 67) = a (.wire 8 70) ∧ a (.wire 3 11) = a (.wire 8 72) ∧ a (.virt 9) = a (.wire 8 73) ∧ a (.virt 9) = a (.wire 8 74) ∧ a (.wire 3 11) = a (.wire 8 76) ∧ a (.virt 26) = a (.wire 8 77) ∧ a (.wire 8 75) = a (.wire 8 78) ∧ a (.wire 3 11) = a (.wire 11 0) ∧ a (.virt 10) = a (.wire 11 1) ∧ a (.virt 10) = a (.wire 11 2) ∧ a (.wire 3 11) = a (.wire 11 4) ∧ a (.virt 26) = a (.wire 11 5) ∧ a (.wire 11 3) = a (.wire 11 6) ∧ a (.wire 3 11) = a (.wire 11 8) ∧ a (.virt 11) = a (.wire 11 9) ∧ a (.virt 11) = a (.wire 11 10) ∧ a (.wire 3 11) = a (.wire 11 12) ∧ a (.virt 26) = a (.wire 11 13) ∧ a (.wire 11 11) = a (.wire 11 14) ∧ a (.wire 3 11) = a (.wire 11 16) ∧ a (.virt 12) = a (.wire 11 17) ∧ a (.virt 12) = a (.wire 11 18) ∧ a (.wire 3 11) = a (.wire 11 20) ∧ a (.virt 26) = a (.wire 11 21) ∧ a (.wire 11 19) = a (.wire 11 22) ∧ a (.wire 3 11) = a (.wire 11 24) ∧ a (.virt 13) = a (.wire 11 25) ∧ a (.virt 13) = a (.wire 11 26) ∧ a (.wire 3 11) = a (.wire 11 28) ∧ a (.virt 26) = a (.wire 11 29) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies15, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies16 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 27) = a (.wire 11 30) ∧ a (.wire 3 11) = a (.wire 11 32) ∧ a (.virt 14) = a (.wire 11 33) ∧ a (.virt 14) = a (.wire 11 34) ∧ a (.wire 3 11) = a (.wire 11 36) ∧ a (.virt 26) = a (.wire 11 37) ∧ a (.wire 11 35) = a (.wire 11 38) ∧ a (.wire 3 11) = a (.wire 11 40) ∧ a (.virt 15) = a (.wire 11 41) ∧ a (.virt 15) = a (.wire 11 42) ∧ a (.wire 3 11) = a (.wire 11 44) ∧ a (.virt 26) = a (.wire 11 45) ∧ a (.wire 11 43) = a (.wire 11 46) ∧ a (.wire 3 11) = a (.wire 11 48) ∧ a (.virt 16) = a (.wire 11 49) ∧ a (.virt 16) = a (.wire 11 50) ∧ a (.wire 3 11) = a (.wire 11 52) ∧ a (.virt 26) = a (.wire 11 53) ∧ a (.wire 11 51) = a (.wire 11 54) ∧ a (.wire 3 11) = a (.wire 11 56) ∧ a (.virt 17) = a (.wire 11 57) ∧ a (.virt 17) = a (.wire 11 58) ∧ a (.wire 3 11) = a (.wire 11 60) ∧ a (.virt 26) = a (.wire 11 61) ∧ a (.wire 11 59) = a (.wire 11 62) ∧ a (.wire 3 55) = a (.wire 11 64) ∧ a (.virt 38) = a (.wire 11 65) ∧ a (.virt 38) = a (.wire 11 66) ∧ a (.wire 3 55) = a (.wire 11 68) ∧ a (.virt 26) = a (.wire 11 69) ∧ a (.wire 11 67) = a (.wire 11 70) ∧ a (.wire 3 55) = a (.wire 11 72) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies16, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies17 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 39) = a (.wire 11 73) ∧ a (.virt 39) = a (.wire 11 74) ∧ a (.wire 3 55) = a (.wire 11 76) ∧ a (.virt 26) = a (.wire 11 77) ∧ a (.wire 11 75) = a (.wire 11 78) ∧ a (.wire 3 55) = a (.wire 12 0) ∧ a (.virt 40) = a (.wire 12 1) ∧ a (.virt 40) = a (.wire 12 2) ∧ a (.wire 3 55) = a (.wire 12 4) ∧ a (.virt 26) = a (.wire 12 5) ∧ a (.wire 12 3) = a (.wire 12 6) ∧ a (.wire 3 55) = a (.wire 12 8) ∧ a (.virt 41) = a (.wire 12 9) ∧ a (.virt 41) = a (.wire 12 10) ∧ a (.wire 3 55) = a (.wire 12 12) ∧ a (.virt 26) = a (.wire 12 13) ∧ a (.wire 12 11) = a (.wire 12 14) ∧ a (.wire 3 55) = a (.wire 12 16) ∧ a (.virt 42) = a (.wire 12 17) ∧ a (.virt 42) = a (.wire 12 18) ∧ a (.wire 3 55) = a (.wire 12 20) ∧ a (.virt 26) = a (.wire 12 21) ∧ a (.wire 12 19) = a (.wire 12 22) ∧ a (.wire 3 55) = a (.wire 12 24) ∧ a (.virt 43) = a (.wire 12 25) ∧ a (.virt 43) = a (.wire 12 26) ∧ a (.wire 3 55) = a (.wire 12 28) ∧ a (.virt 26) = a (.wire 12 29) ∧ a (.wire 12 27) = a (.wire 12 30) ∧ a (.wire 3 55) = a (.wire 12 32) ∧ a (.virt 44) = a (.wire 12 33) ∧ a (.virt 44) = a (.wire 12 34) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies17, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies18 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 55) = a (.wire 12 36) ∧ a (.virt 26) = a (.wire 12 37) ∧ a (.wire 12 35) = a (.wire 12 38) ∧ a (.wire 3 55) = a (.wire 12 40) ∧ a (.virt 45) = a (.wire 12 41) ∧ a (.virt 45) = a (.wire 12 42) ∧ a (.wire 3 55) = a (.wire 12 44) ∧ a (.virt 26) = a (.wire 12 45) ∧ a (.wire 12 43) = a (.wire 12 46) ∧ a (.wire 3 55) = a (.wire 12 48) ∧ a (.virt 46) = a (.wire 12 49) ∧ a (.virt 46) = a (.wire 12 50) ∧ a (.wire 3 55) = a (.wire 12 52) ∧ a (.virt 26) = a (.wire 12 53) ∧ a (.wire 12 51) = a (.wire 12 54) ∧ a (.wire 3 55) = a (.wire 12 56) ∧ a (.virt 47) = a (.wire 12 57) ∧ a (.virt 47) = a (.wire 12 58) ∧ a (.wire 3 55) = a (.wire 12 60) ∧ a (.virt 26) = a (.wire 12 61) ∧ a (.wire 12 59) = a (.wire 12 62) ∧ a (.wire 3 11) = a (.wire 12 64) ∧ a (.virt 18) = a (.wire 12 65) ∧ a (.virt 18) = a (.wire 12 66) ∧ a (.wire 3 11) = a (.wire 12 68) ∧ a (.virt 26) = a (.wire 12 69) ∧ a (.wire 12 67) = a (.wire 12 70) ∧ a (.wire 3 11) = a (.wire 12 72) ∧ a (.virt 19) = a (.wire 12 73) ∧ a (.virt 19) = a (.wire 12 74) ∧ a (.wire 3 11) = a (.wire 12 76) ∧ a (.virt 26) = a (.wire 12 77) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies18, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies19 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 75) = a (.wire 12 78) ∧ a (.wire 3 11) = a (.wire 13 0) ∧ a (.virt 20) = a (.wire 13 1) ∧ a (.virt 20) = a (.wire 13 2) ∧ a (.wire 3 11) = a (.wire 13 4) ∧ a (.virt 26) = a (.wire 13 5) ∧ a (.wire 13 3) = a (.wire 13 6) ∧ a (.wire 3 11) = a (.wire 13 8) ∧ a (.virt 21) = a (.wire 13 9) ∧ a (.virt 21) = a (.wire 13 10) ∧ a (.wire 3 11) = a (.wire 13 12) ∧ a (.virt 26) = a (.wire 13 13) ∧ a (.wire 13 11) = a (.wire 13 14) ∧ a (.wire 3 11) = a (.wire 13 16) ∧ a (.virt 22) = a (.wire 13 17) ∧ a (.virt 22) = a (.wire 13 18) ∧ a (.wire 3 11) = a (.wire 13 20) ∧ a (.virt 26) = a (.wire 13 21) ∧ a (.wire 13 19) = a (.wire 13 22) ∧ a (.wire 3 11) = a (.wire 13 24) ∧ a (.virt 23) = a (.wire 13 25) ∧ a (.virt 23) = a (.wire 13 26) ∧ a (.wire 3 11) = a (.wire 13 28) ∧ a (.virt 26) = a (.wire 13 29) ∧ a (.wire 13 27) = a (.wire 13 30) ∧ a (.wire 3 11) = a (.wire 13 32) ∧ a (.virt 24) = a (.wire 13 33) ∧ a (.virt 24) = a (.wire 13 34) ∧ a (.wire 3 11) = a (.wire 13 36) ∧ a (.virt 26) = a (.wire 13 37) ∧ a (.wire 13 35) = a (.wire 13 38) ∧ a (.wire 3 11) = a (.wire 13 40) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies19, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies20 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 25) = a (.wire 13 41) ∧ a (.virt 25) = a (.wire 13 42) ∧ a (.wire 3 11) = a (.wire 13 44) ∧ a (.virt 26) = a (.wire 13 45) ∧ a (.wire 13 43) = a (.wire 13 46) ∧ a (.wire 3 55) = a (.wire 13 48) ∧ a (.virt 48) = a (.wire 13 49) ∧ a (.virt 48) = a (.wire 13 50) ∧ a (.wire 3 55) = a (.wire 13 52) ∧ a (.virt 26) = a (.wire 13 53) ∧ a (.wire 13 51) = a (.wire 13 54) ∧ a (.wire 3 55) = a (.wire 13 56) ∧ a (.virt 49) = a (.wire 13 57) ∧ a (.virt 49) = a (.wire 13 58) ∧ a (.wire 3 55) = a (.wire 13 60) ∧ a (.virt 26) = a (.wire 13 61) ∧ a (.wire 13 59) = a (.wire 13 62) ∧ a (.wire 3 55) = a (.wire 13 64) ∧ a (.virt 50) = a (.wire 13 65) ∧ a (.virt 50) = a (.wire 13 66) ∧ a (.wire 3 55) = a (.wire 13 68) ∧ a (.virt 26) = a (.wire 13 69) ∧ a (.wire 13 67) = a (.wire 13 70) ∧ a (.wire 3 55) = a (.wire 13 72) ∧ a (.virt 51) = a (.wire 13 73) ∧ a (.virt 51) = a (.wire 13 74) ∧ a (.wire 3 55) = a (.wire 13 76) ∧ a (.virt 26) = a (.wire 13 77) ∧ a (.wire 13 75) = a (.wire 13 78) ∧ a (.wire 3 55) = a (.wire 14 0) ∧ a (.virt 52) = a (.wire 14 1) ∧ a (.virt 52) = a (.wire 14 2) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies20, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies21 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 55) = a (.wire 14 4) ∧ a (.virt 26) = a (.wire 14 5) ∧ a (.wire 14 3) = a (.wire 14 6) ∧ a (.wire 3 55) = a (.wire 14 8) ∧ a (.virt 53) = a (.wire 14 9) ∧ a (.virt 53) = a (.wire 14 10) ∧ a (.wire 3 55) = a (.wire 14 12) ∧ a (.virt 26) = a (.wire 14 13) ∧ a (.wire 14 11) = a (.wire 14 14) ∧ a (.wire 3 55) = a (.wire 14 16) ∧ a (.virt 54) = a (.wire 14 17) ∧ a (.virt 54) = a (.wire 14 18) ∧ a (.wire 3 55) = a (.wire 14 20) ∧ a (.virt 26) = a (.wire 14 21) ∧ a (.wire 14 19) = a (.wire 14 22) ∧ a (.wire 3 55) = a (.wire 14 24) ∧ a (.virt 55) = a (.wire 14 25) ∧ a (.virt 55) = a (.wire 14 26) ∧ a (.wire 3 55) = a (.wire 14 28) ∧ a (.virt 26) = a (.wire 14 29) ∧ a (.wire 14 27) = a (.wire 14 30) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies21, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_consts (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 26) = 0 ∧ a (.virt 27) = 4 ∧ a (.virt 28) = 1 ∧ a (.virt 29) = 2 := by
  have hconst := h.2.2
  simp only [publicBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3⟩ := hconst
  exact ⟨k0, k1, k2, k3⟩

theorem publicBatchWrapper2_f0 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 0) = a (.virt 29) := by
  have c0 := (publicBatchWrapper2_copies0 a h).1
  exact c0

theorem publicBatchWrapper2_f1 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 30) = a (.virt 29) := by
  have c1 := (publicBatchWrapper2_copies0 a h).2.1
  exact c1

theorem publicBatchWrapper2_f2 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 3)) (a (.virt 26)) (a (.virt 60)) (a (.virt 61)) := by
  have c2 := (publicBatchWrapper2_copies0 a h).2.2.1
  have c3 := (publicBatchWrapper2_copies0 a h).2.2.2.1
  have c4 := (publicBatchWrapper2_copies0 a h).2.2.2.2.1
  have c5 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.1
  have c6 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.1
  have c7 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c5, ← c6, ← c7] at e_1_12
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c8, ← c9, ← c10] at e_1_13
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c2, k2, ← c3, k2, ← c4] at e_2_0
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c11, ← c12, k2, ← c13] at e_2_1
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_1_12
    linear_combination c14.trans k0 - hc
  · have hc := e_2_1
    simp only [e_2_0, e_1_13] at hc
    linear_combination c15.trans k0 - hc

theorem publicBatchWrapper2_f3 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 4)) (a (.virt 26)) (a (.virt 62)) (a (.virt 63)) := by
  have c16 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c19, ← c20, ← c21] at e_1_14
  have e_1_15 := arithEq_of_rows h (row := 1) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_15
  simp only [← c22, ← c23, ← c24] at e_1_15
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c16, k2, ← c17, k2, ← c18] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c25, ← c26, k2, ← c27] at e_2_3
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_1_14
    linear_combination c28.trans k0 - hc
  · have hc := e_2_3
    simp only [e_2_2, e_1_15] at hc
    linear_combination c29.trans k0 - hc

theorem publicBatchWrapper2_f4 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 5)) (a (.virt 26)) (a (.virt 64)) (a (.virt 65)) := by
  have c30 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (publicBatchWrapper2_copies1 a h).1
  have c33 := (publicBatchWrapper2_copies1 a h).2.1
  have c34 := (publicBatchWrapper2_copies1 a h).2.2.1
  have c35 := (publicBatchWrapper2_copies1 a h).2.2.2.1
  have c36 := (publicBatchWrapper2_copies1 a h).2.2.2.2.1
  have c37 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.1
  have c38 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.1
  have c39 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.1
  have c42 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_1_16 := arithEq_of_rows h (row := 1) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_16
  simp only [← c33, ← c34, ← c35] at e_1_16
  have e_1_17 := arithEq_of_rows h (row := 1) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_17
  simp only [← c36, ← c37, ← c38] at e_1_17
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c30, k2, ← c31, k2, ← c32] at e_2_4
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c39, ← c40, k2, ← c41] at e_2_5
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_1_16
    linear_combination c42.trans k0 - hc
  · have hc := e_2_5
    simp only [e_2_4, e_1_17] at hc
    linear_combination c43.trans k0 - hc

theorem publicBatchWrapper2_f5 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 6)) (a (.virt 26)) (a (.virt 66)) (a (.virt 67)) := by
  have c44 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_1_18 := arithEq_of_rows h (row := 1) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_18
  simp only [← c47, ← c48, ← c49] at e_1_18
  have e_1_19 := arithEq_of_rows h (row := 1) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_19
  simp only [← c50, ← c51, ← c52] at e_1_19
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c44, k2, ← c45, k2, ← c46] at e_2_6
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c53, ← c54, k2, ← c55] at e_2_7
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_1_18
    linear_combination c56.trans k0 - hc
  · have hc := e_2_7
    simp only [e_2_6, e_1_19] at hc
    linear_combination c57.trans k0 - hc

theorem publicBatchWrapper2_f6 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 3) = band (a (.virt 60)) (a (.virt 62)) := by
  have c58 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c58, ← c59, ← c60] at e_3_0
  have hr := e_3_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f7 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 7) = band (a (.virt 64)) (a (.virt 66)) := by
  have c61 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c61, ← c62, ← c63] at e_3_1
  have hr := e_3_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f8 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 11) = band (a (.wire 3 3)) (a (.wire 3 7)) := by
  have c64 := (publicBatchWrapper2_copies2 a h).1
  have c65 := (publicBatchWrapper2_copies2 a h).2.1
  have c66 := (publicBatchWrapper2_copies2 a h).2.2.1
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c64, ← c65, ← c66] at e_3_2
  have hr := e_3_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f9 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 33)) (a (.virt 26)) (a (.virt 68)) (a (.virt 69)) := by
  have c67 := (publicBatchWrapper2_copies2 a h).2.2.2.1
  have c68 := (publicBatchWrapper2_copies2 a h).2.2.2.2.1
  have c69 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.1
  have c70 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.1
  have c71 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c67, k2, ← c68, k2, ← c69] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c76, ← c77, k2, ← c78] at e_2_9
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c70, ← c71, ← c72] at e_3_3
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c73, ← c74, ← c75] at e_3_4
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_3_3
    linear_combination c79.trans k0 - hc
  · have hc := e_2_9
    simp only [e_2_8, e_3_4] at hc
    linear_combination c80.trans k0 - hc

theorem publicBatchWrapper2_f10 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 34)) (a (.virt 26)) (a (.virt 70)) (a (.virt 71)) := by
  have c81 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c81, k2, ← c82, k2, ← c83] at e_2_10
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c90, ← c91, k2, ← c92] at e_2_11
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c84, ← c85, ← c86] at e_3_5
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c87, ← c88, ← c89] at e_3_6
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_3_5
    linear_combination c93.trans k0 - hc
  · have hc := e_2_11
    simp only [e_2_10, e_3_6] at hc
    linear_combination c94.trans k0 - hc

theorem publicBatchWrapper2_f11 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 35)) (a (.virt 26)) (a (.virt 72)) (a (.virt 73)) := by
  have c95 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (publicBatchWrapper2_copies3 a h).1
  have c97 := (publicBatchWrapper2_copies3 a h).2.1
  have c98 := (publicBatchWrapper2_copies3 a h).2.2.1
  have c99 := (publicBatchWrapper2_copies3 a h).2.2.2.1
  have c100 := (publicBatchWrapper2_copies3 a h).2.2.2.2.1
  have c101 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.1
  have c102 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.1
  have c103 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c95, k2, ← c96, k2, ← c97] at e_2_12
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c104, ← c105, k2, ← c106] at e_2_13
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c98, ← c99, ← c100] at e_3_7
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c101, ← c102, ← c103] at e_3_8
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_3_7
    linear_combination c107.trans k0 - hc
  · have hc := e_2_13
    simp only [e_2_12, e_3_8] at hc
    linear_combination c108.trans k0 - hc

theorem publicBatchWrapper2_f12 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 36)) (a (.virt 26)) (a (.virt 74)) (a (.virt 75)) := by
  have c109 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c109, k2, ← c110, k2, ← c111] at e_2_14
  have e_2_15 := arithEq_of_rows h (row := 2) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c118, ← c119, k2, ← c120] at e_2_15
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c112, ← c113, ← c114] at e_3_9
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c115, ← c116, ← c117] at e_3_10
  simp only [k0]
  refine ⟨?_, ?_⟩
  · have hc := e_3_9
    linear_combination c121.trans k0 - hc
  · have hc := e_2_15
    simp only [e_2_14, e_3_10] at hc
    linear_combination c122.trans k0 - hc

theorem publicBatchWrapper2_f13 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 47) = band (a (.virt 68)) (a (.virt 70)) := by
  have c123 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c123, ← c124, ← c125] at e_3_11
  have hr := e_3_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f14 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 51) = band (a (.virt 72)) (a (.virt 74)) := by
  have c126 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c127 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (publicBatchWrapper2_copies4 a h).1
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c126, ← c127, ← c128] at e_3_12
  have hr := e_3_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f15 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 55) = band (a (.wire 3 47)) (a (.wire 3 51)) := by
  have c129 := (publicBatchWrapper2_copies4 a h).2.1
  have c130 := (publicBatchWrapper2_copies4 a h).2.2.1
  have c131 := (publicBatchWrapper2_copies4 a h).2.2.2.1
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c129, ← c130, ← c131] at e_3_13
  have hr := e_3_13
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f16 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 67) = bnot (a (.wire 3 11)) := by
  have c132 := (publicBatchWrapper2_copies4 a h).2.2.2.2.1
  have c133 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.1
  have c134 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_16 := arithEq_of_rows h (row := 2) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c132, k2, ← c133, k2, ← c134] at e_2_16
  have hr := e_2_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper2_f17 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 28) = bnot (a (.virt 26)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  simp only [bnot, k0, k2]
  ring

theorem publicBatchWrapper2_f18 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 67) = band (a (.wire 2 67)) (a (.virt 28)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  simp only [band, k2]
  ring

theorem publicBatchWrapper2_f19 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 71) = bselect (a (.wire 2 67)) (a (.virt 3)) (a (.virt 26)) := by
  have c135 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_17 := arithEq_of_rows h (row := 2) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c135, ← c136, ← c137, k0] at e_2_17
  have hr := e_2_17
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f20 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 75) = bselect (a (.wire 2 67)) (a (.virt 4)) (a (.virt 26)) := by
  have c138 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_18 := arithEq_of_rows h (row := 2) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c138, ← c139, ← c140, k0] at e_2_18
  have hr := e_2_18
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f21 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 79) = bselect (a (.wire 2 67)) (a (.virt 5)) (a (.virt 26)) := by
  have c141 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_2_19 := arithEq_of_rows h (row := 2) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c141, ← c142, ← c143, k0] at e_2_19
  have hr := e_2_19
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f22 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 3) = bselect (a (.wire 2 67)) (a (.virt 6)) (a (.virt 26)) := by
  have c144 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c145 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c144, ← c145, ← c146, k0] at e_4_0
  have hr := e_4_0
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f23 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 7) = bselect (a (.wire 2 67)) (a (.virt 7)) (a (.virt 26)) := by
  have c147 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c148 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c147, ← c148, ← c149, k0] at e_4_1
  have hr := e_4_1
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f24 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 11) = bselect (a (.wire 2 67)) (a (.virt 1)) (a (.virt 26)) := by
  have c150 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c151 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c150, ← c151, ← c152, k0] at e_4_2
  have hr := e_4_2
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f25 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 15) = bselect (a (.wire 2 67)) (a (.virt 2)) (a (.virt 26)) := by
  have c153 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c153, ← c154, ← c155, k0] at e_4_3
  have hr := e_4_3
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f26 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 67) = bor (a (.virt 26)) (a (.wire 2 67)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  simp only [bor, k0]
  ring

theorem publicBatchWrapper2_f27 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 19) = bnot (a (.wire 3 55)) := by
  have c156 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c156, k2, ← c157, k2, ← c158] at e_4_4
  have hr := e_4_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper2_f28 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 23) = bnot (a (.wire 2 67)) := by
  have c159 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (publicBatchWrapper2_copies5 a h).1
  have c161 := (publicBatchWrapper2_copies5 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c159, k2, ← c160, k2, ← c161] at e_4_5
  have hr := e_4_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper2_f29 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 59) = band (a (.wire 4 19)) (a (.wire 4 23)) := by
  have c162 := (publicBatchWrapper2_copies5 a h).2.2.1
  have c163 := (publicBatchWrapper2_copies5 a h).2.2.2.1
  have c164 := (publicBatchWrapper2_copies5 a h).2.2.2.2.1
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c162, ← c163, ← c164] at e_3_14
  have hr := e_3_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f30 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 31) = bselect (a (.wire 3 59)) (a (.virt 33)) (a (.wire 2 71)) := by
  have c165 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.1
  have c166 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.1
  have c167 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c165, ← c166, ← c167] at e_4_6
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c168, ← c169, ← c170] at e_4_7
  have hr := e_4_7
  simp only [e_4_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f31 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 39) = bselect (a (.wire 3 59)) (a (.virt 34)) (a (.wire 2 75)) := by
  have c171 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c171, ← c172, ← c173] at e_4_8
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c174, ← c175, ← c176] at e_4_9
  have hr := e_4_9
  simp only [e_4_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f32 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 47) = bselect (a (.wire 3 59)) (a (.virt 35)) (a (.wire 2 79)) := by
  have c177 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c177, ← c178, ← c179] at e_4_10
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c180, ← c181, ← c182] at e_4_11
  have hr := e_4_11
  simp only [e_4_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f33 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 55) = bselect (a (.wire 3 59)) (a (.virt 36)) (a (.wire 4 3)) := by
  have c183 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c183, ← c184, ← c185] at e_4_12
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c186, ← c187, ← c188] at e_4_13
  have hr := e_4_13
  simp only [e_4_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f34 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 63) = bselect (a (.wire 3 59)) (a (.virt 37)) (a (.wire 4 7)) := by
  have c189 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (publicBatchWrapper2_copies6 a h).1
  have c193 := (publicBatchWrapper2_copies6 a h).2.1
  have c194 := (publicBatchWrapper2_copies6 a h).2.2.1
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c189, ← c190, ← c191] at e_4_14
  have e_4_15 := arithEq_of_rows h (row := 4) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_15
  simp only [← c192, ← c193, ← c194] at e_4_15
  have hr := e_4_15
  simp only [e_4_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f35 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 71) = bselect (a (.wire 3 59)) (a (.virt 31)) (a (.wire 4 11)) := by
  have c195 := (publicBatchWrapper2_copies6 a h).2.2.2.1
  have c196 := (publicBatchWrapper2_copies6 a h).2.2.2.2.1
  have c197 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.1
  have c198 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.1
  have c199 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.1
  have e_4_16 := arithEq_of_rows h (row := 4) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_16
  simp only [← c195, ← c196, ← c197] at e_4_16
  have e_4_17 := arithEq_of_rows h (row := 4) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_17
  simp only [← c198, ← c199, ← c200] at e_4_17
  have hr := e_4_17
  simp only [e_4_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f36 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 4 79) = bselect (a (.wire 3 59)) (a (.virt 32)) (a (.wire 4 15)) := by
  have c201 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_18 := arithEq_of_rows h (row := 4) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_18
  simp only [← c201, ← c202, ← c203] at e_4_18
  have e_4_19 := arithEq_of_rows h (row := 4) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_19
  simp only [← c204, ← c205, ← c206] at e_4_19
  have hr := e_4_19
  simp only [e_4_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f37 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 67) = bor (a (.wire 2 67)) (a (.wire 4 19)) := by
  have c207 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_0_16 := arithEq_of_rows h (row := 0) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_16
  simp only [← c210, ← c211, k2, ← c212] at e_0_16
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c207, ← c208, ← c209] at e_5_0
  have hr := e_0_16
  simp only [e_5_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f38 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 1)) (a (.wire 4 71)) (a (.virt 76)) (a (.virt 77)) := by
  have c213 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (publicBatchWrapper2_copies7 a h).1
  have c225 := (publicBatchWrapper2_copies7 a h).2.1
  have c226 := (publicBatchWrapper2_copies7 a h).2.2.1
  have c227 := (publicBatchWrapper2_copies7 a h).2.2.2.1
  have c228 := (publicBatchWrapper2_copies7 a h).2.2.2.2.1
  have c229 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_3_15 := arithEq_of_rows h (row := 3) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c219, ← c220, ← c221] at e_3_15
  have e_3_16 := arithEq_of_rows h (row := 3) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c222, ← c223, ← c224] at e_3_16
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c213, k2, ← c214, k2, ← c215] at e_6_0
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c216, ← c217, k2, ← c218] at e_6_1
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c225, ← c226, k2, ← c227] at e_6_2
  refine ⟨?_, ?_⟩
  · have hc := e_3_15
    simp only [e_6_1] at hc
    linear_combination c228.trans k0 - hc
  · have hc := e_6_2
    simp only [e_6_0, e_3_16, e_6_1] at hc
    linear_combination c229.trans k0 - hc

theorem publicBatchWrapper2_f39 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 71) = bor (a (.wire 3 11)) (a (.virt 76)) := by
  have c230 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.1
  have c231 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_0_17 := arithEq_of_rows h (row := 0) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c233, ← c234, k2, ← c235] at e_0_17
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c230, ← c231, ← c232] at e_5_1
  have hr := e_0_17
  simp only [e_5_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f40 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 71) = a (.virt 28) := by
  have c236 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c236

theorem publicBatchWrapper2_f41 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 2)) (a (.wire 4 79)) (a (.virt 78)) (a (.virt 79)) := by
  have c237 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_3_17 := arithEq_of_rows h (row := 3) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c243, ← c244, ← c245] at e_3_17
  have e_3_18 := arithEq_of_rows h (row := 3) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c246, ← c247, ← c248] at e_3_18
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c237, k2, ← c238, k2, ← c239] at e_6_3
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c240, ← c241, k2, ← c242] at e_6_4
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c249, ← c250, k2, ← c251] at e_6_5
  refine ⟨?_, ?_⟩
  · have hc := e_3_17
    simp only [e_6_4] at hc
    linear_combination c252.trans k0 - hc
  · have hc := e_6_5
    simp only [e_6_3, e_3_18, e_6_4] at hc
    linear_combination c253.trans k0 - hc

theorem publicBatchWrapper2_f42 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 75) = bor (a (.wire 3 11)) (a (.virt 78)) := by
  have c254 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (publicBatchWrapper2_copies8 a h).1
  have c257 := (publicBatchWrapper2_copies8 a h).2.1
  have c258 := (publicBatchWrapper2_copies8 a h).2.2.1
  have c259 := (publicBatchWrapper2_copies8 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_0_18 := arithEq_of_rows h (row := 0) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c257, ← c258, k2, ← c259] at e_0_18
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c254, ← c255, ← c256] at e_5_2
  have hr := e_0_18
  simp only [e_5_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f43 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 75) = a (.virt 28) := by
  have c260 := (publicBatchWrapper2_copies8 a h).2.2.2.2.1
  exact c260

theorem publicBatchWrapper2_f44 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 3)) (a (.wire 4 31)) (a (.virt 80)) (a (.virt 81)) := by
  have c261 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.1
  have c262 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.1
  have c263 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_3_19 := arithEq_of_rows h (row := 3) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c267, ← c268, ← c269] at e_3_19
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c261, k2, ← c262, k2, ← c263] at e_6_6
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c264, ← c265, k2, ← c266] at e_6_7
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c273, ← c274, k2, ← c275] at e_6_8
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c270, ← c271, ← c272] at e_7_0
  refine ⟨?_, ?_⟩
  · have hc := e_3_19
    simp only [e_6_7] at hc
    linear_combination c276.trans k0 - hc
  · have hc := e_6_8
    simp only [e_6_6, e_7_0, e_6_7] at hc
    linear_combination c277.trans k0 - hc

theorem publicBatchWrapper2_f45 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 4)) (a (.wire 4 39)) (a (.virt 82)) (a (.virt 83)) := by
  have c278 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (publicBatchWrapper2_copies9 a h).1
  have c289 := (publicBatchWrapper2_copies9 a h).2.1
  have c290 := (publicBatchWrapper2_copies9 a h).2.2.1
  have c291 := (publicBatchWrapper2_copies9 a h).2.2.2.1
  have c292 := (publicBatchWrapper2_copies9 a h).2.2.2.2.1
  have c293 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.1
  have c294 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c278, k2, ← c279, k2, ← c280] at e_6_9
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c281, ← c282, k2, ← c283] at e_6_10
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c290, ← c291, k2, ← c292] at e_6_11
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c284, ← c285, ← c286] at e_7_1
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c287, ← c288, ← c289] at e_7_2
  refine ⟨?_, ?_⟩
  · have hc := e_7_1
    simp only [e_6_10] at hc
    linear_combination c293.trans k0 - hc
  · have hc := e_6_11
    simp only [e_6_9, e_7_2, e_6_10] at hc
    linear_combination c294.trans k0 - hc

theorem publicBatchWrapper2_f46 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 5)) (a (.wire 4 47)) (a (.virt 84)) (a (.virt 85)) := by
  have c295 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c295, k2, ← c296, k2, ← c297] at e_6_12
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c298, ← c299, k2, ← c300] at e_6_13
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c307, ← c308, k2, ← c309] at e_6_14
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c301, ← c302, ← c303] at e_7_3
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c304, ← c305, ← c306] at e_7_4
  refine ⟨?_, ?_⟩
  · have hc := e_7_3
    simp only [e_6_13] at hc
    linear_combination c310.trans k0 - hc
  · have hc := e_6_14
    simp only [e_6_12, e_7_4, e_6_13] at hc
    linear_combination c311.trans k0 - hc

theorem publicBatchWrapper2_f47 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 6)) (a (.wire 4 55)) (a (.virt 86)) (a (.virt 87)) := by
  have c312 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (publicBatchWrapper2_copies10 a h).1
  have c321 := (publicBatchWrapper2_copies10 a h).2.1
  have c322 := (publicBatchWrapper2_copies10 a h).2.2.1
  have c323 := (publicBatchWrapper2_copies10 a h).2.2.2.1
  have c324 := (publicBatchWrapper2_copies10 a h).2.2.2.2.1
  have c325 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.1
  have c326 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.1
  have c327 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_6_15 := arithEq_of_rows h (row := 6) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_15
  simp only [← c312, k2, ← c313, k2, ← c314] at e_6_15
  have e_6_16 := arithEq_of_rows h (row := 6) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_16
  simp only [← c315, ← c316, k2, ← c317] at e_6_16
  have e_6_17 := arithEq_of_rows h (row := 6) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_17
  simp only [← c324, ← c325, k2, ← c326] at e_6_17
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c318, ← c319, ← c320] at e_7_5
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c321, ← c322, ← c323] at e_7_6
  refine ⟨?_, ?_⟩
  · have hc := e_7_5
    simp only [e_6_16] at hc
    linear_combination c327.trans k0 - hc
  · have hc := e_6_17
    simp only [e_6_15, e_7_6, e_6_16] at hc
    linear_combination c328.trans k0 - hc

theorem publicBatchWrapper2_f48 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 31) = band (a (.virt 80)) (a (.virt 82)) := by
  have c329 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c329, ← c330, ← c331] at e_7_7
  have hr := e_7_7
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f49 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 35) = band (a (.virt 84)) (a (.virt 86)) := by
  have c332 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c332, ← c333, ← c334] at e_7_8
  have hr := e_7_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f50 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 39) = band (a (.wire 7 31)) (a (.wire 7 35)) := by
  have c335 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c335, ← c336, ← c337] at e_7_9
  have hr := e_7_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f51 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 79) = bor (a (.wire 3 11)) (a (.wire 7 39)) := by
  have c338 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_0_19 := arithEq_of_rows h (row := 0) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c341, ← c342, k2, ← c343] at e_0_19
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c338, ← c339, ← c340] at e_5_3
  have hr := e_0_19
  simp only [e_5_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f52 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 79) = a (.virt 28) := by
  have c344 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c344

theorem publicBatchWrapper2_f53 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 31)) (a (.wire 4 71)) (a (.virt 88)) (a (.virt 89)) := by
  have c345 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (publicBatchWrapper2_copies11 a h).1
  have c353 := (publicBatchWrapper2_copies11 a h).2.1
  have c354 := (publicBatchWrapper2_copies11 a h).2.2.1
  have c355 := (publicBatchWrapper2_copies11 a h).2.2.2.1
  have c356 := (publicBatchWrapper2_copies11 a h).2.2.2.2.1
  have c357 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.1
  have c358 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.1
  have c359 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_6_18 := arithEq_of_rows h (row := 6) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_18
  simp only [← c345, k2, ← c346, k2, ← c347] at e_6_18
  have e_6_19 := arithEq_of_rows h (row := 6) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_19
  simp only [← c348, ← c349, k2, ← c350] at e_6_19
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c351, ← c352, ← c353] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c354, ← c355, ← c356] at e_7_11
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c357, ← c358, k2, ← c359] at e_8_0
  refine ⟨?_, ?_⟩
  · have hc := e_7_10
    simp only [e_6_19] at hc
    linear_combination c360.trans k0 - hc
  · have hc := e_8_0
    simp only [e_6_18, e_7_11, e_6_19] at hc
    linear_combination c361.trans k0 - hc

theorem publicBatchWrapper2_f54 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 3) = bor (a (.wire 3 55)) (a (.virt 88)) := by
  have c362 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c362, ← c363, ← c364] at e_5_4
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c365, ← c366, k2, ← c367] at e_9_0
  have hr := e_9_0
  simp only [e_5_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f55 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 3) = a (.virt 28) := by
  have c368 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c368

theorem publicBatchWrapper2_f56 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 32)) (a (.wire 4 79)) (a (.virt 90)) (a (.virt 91)) := by
  have c369 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (publicBatchWrapper2_copies12 a h).1
  have c385 := (publicBatchWrapper2_copies12 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c375, ← c376, ← c377] at e_7_12
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c378, ← c379, ← c380] at e_7_13
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c369, k2, ← c370, k2, ← c371] at e_8_1
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c372, ← c373, k2, ← c374] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c381, ← c382, k2, ← c383] at e_8_3
  refine ⟨?_, ?_⟩
  · have hc := e_7_12
    simp only [e_8_2] at hc
    linear_combination c384.trans k0 - hc
  · have hc := e_8_3
    simp only [e_8_1, e_7_13, e_8_2] at hc
    linear_combination c385.trans k0 - hc

theorem publicBatchWrapper2_f57 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 7) = bor (a (.wire 3 55)) (a (.virt 90)) := by
  have c386 := (publicBatchWrapper2_copies12 a h).2.2.1
  have c387 := (publicBatchWrapper2_copies12 a h).2.2.2.1
  have c388 := (publicBatchWrapper2_copies12 a h).2.2.2.2.1
  have c389 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.1
  have c390 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.1
  have c391 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c386, ← c387, ← c388] at e_5_5
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c389, ← c390, k2, ← c391] at e_9_1
  have hr := e_9_1
  simp only [e_5_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f58 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 7) = a (.virt 28) := by
  have c392 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.1
  exact c392

theorem publicBatchWrapper2_f59 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 33)) (a (.wire 4 31)) (a (.virt 92)) (a (.virt 93)) := by
  have c393 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c405 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c399, ← c400, ← c401] at e_7_14
  have e_7_15 := arithEq_of_rows h (row := 7) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_15
  simp only [← c402, ← c403, ← c404] at e_7_15
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c393, k2, ← c394, k2, ← c395] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c396, ← c397, k2, ← c398] at e_8_5
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c405, ← c406, k2, ← c407] at e_8_6
  refine ⟨?_, ?_⟩
  · have hc := e_7_14
    simp only [e_8_5] at hc
    linear_combination c408.trans k0 - hc
  · have hc := e_8_6
    simp only [e_8_4, e_7_15, e_8_5] at hc
    linear_combination c409.trans k0 - hc

theorem publicBatchWrapper2_f60 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 34)) (a (.wire 4 39)) (a (.virt 94)) (a (.virt 95)) := by
  have c410 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (publicBatchWrapper2_copies13 a h).1
  have c417 := (publicBatchWrapper2_copies13 a h).2.1
  have c418 := (publicBatchWrapper2_copies13 a h).2.2.1
  have c419 := (publicBatchWrapper2_copies13 a h).2.2.2.1
  have c420 := (publicBatchWrapper2_copies13 a h).2.2.2.2.1
  have c421 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.1
  have c422 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.1
  have c423 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_7_16 := arithEq_of_rows h (row := 7) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_16
  simp only [← c416, ← c417, ← c418] at e_7_16
  have e_7_17 := arithEq_of_rows h (row := 7) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_17
  simp only [← c419, ← c420, ← c421] at e_7_17
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c410, k2, ← c411, k2, ← c412] at e_8_7
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c413, ← c414, k2, ← c415] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c422, ← c423, k2, ← c424] at e_8_9
  refine ⟨?_, ?_⟩
  · have hc := e_7_16
    simp only [e_8_8] at hc
    linear_combination c425.trans k0 - hc
  · have hc := e_8_9
    simp only [e_8_7, e_7_17, e_8_8] at hc
    linear_combination c426.trans k0 - hc

theorem publicBatchWrapper2_f61 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 35)) (a (.wire 4 47)) (a (.virt 96)) (a (.virt 97)) := by
  have c427 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_7_18 := arithEq_of_rows h (row := 7) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_18
  simp only [← c433, ← c434, ← c435] at e_7_18
  have e_7_19 := arithEq_of_rows h (row := 7) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_19
  simp only [← c436, ← c437, ← c438] at e_7_19
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c427, k2, ← c428, k2, ← c429] at e_8_10
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c430, ← c431, k2, ← c432] at e_8_11
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c439, ← c440, k2, ← c441] at e_8_12
  refine ⟨?_, ?_⟩
  · have hc := e_7_18
    simp only [e_8_11] at hc
    linear_combination c442.trans k0 - hc
  · have hc := e_8_12
    simp only [e_8_10, e_7_19, e_8_11] at hc
    linear_combination c443.trans k0 - hc

theorem publicBatchWrapper2_f62 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 36)) (a (.wire 4 55)) (a (.virt 98)) (a (.virt 99)) := by
  have c444 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (publicBatchWrapper2_copies14 a h).1
  have c449 := (publicBatchWrapper2_copies14 a h).2.1
  have c450 := (publicBatchWrapper2_copies14 a h).2.2.1
  have c451 := (publicBatchWrapper2_copies14 a h).2.2.2.1
  have c452 := (publicBatchWrapper2_copies14 a h).2.2.2.2.1
  have c453 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.1
  have c454 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.1
  have c455 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c444, k2, ← c445, k2, ← c446] at e_8_13
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c447, ← c448, k2, ← c449] at e_8_14
  have e_8_15 := arithEq_of_rows h (row := 8) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_15
  simp only [← c456, ← c457, k2, ← c458] at e_8_15
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c450, ← c451, ← c452] at e_10_0
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c453, ← c454, ← c455] at e_10_1
  refine ⟨?_, ?_⟩
  · have hc := e_10_0
    simp only [e_8_14] at hc
    linear_combination c459.trans k0 - hc
  · have hc := e_8_15
    simp only [e_8_13, e_10_1, e_8_14] at hc
    linear_combination c460.trans k0 - hc

theorem publicBatchWrapper2_f63 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 11) = band (a (.virt 92)) (a (.virt 94)) := by
  have c461 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c461, ← c462, ← c463] at e_10_2
  have hr := e_10_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f64 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 15) = band (a (.virt 96)) (a (.virt 98)) := by
  have c464 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c464, ← c465, ← c466] at e_10_3
  have hr := e_10_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f65 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 19) = band (a (.wire 10 11)) (a (.wire 10 15)) := by
  have c467 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c467, ← c468, ← c469] at e_10_4
  have hr := e_10_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f66 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 11) = bor (a (.wire 3 55)) (a (.wire 10 19)) := by
  have c470 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c470, ← c471, ← c472] at e_5_6
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c473, ← c474, k2, ← c475] at e_9_2
  have hr := e_9_2
  simp only [e_5_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f67 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 11) = a (.virt 28) := by
  have c476 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c476

theorem publicBatchWrapper2_f68 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 8 71) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 8)) := by
  have c477 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (publicBatchWrapper2_copies15 a h).1
  have c481 := (publicBatchWrapper2_copies15 a h).2.1
  have c482 := (publicBatchWrapper2_copies15 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_8_16 := arithEq_of_rows h (row := 8) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_16
  simp only [← c477, ← c478, ← c479] at e_8_16
  have e_8_17 := arithEq_of_rows h (row := 8) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_17
  simp only [← c480, ← c481, k0, ← c482] at e_8_17
  have hr := e_8_17
  simp only [e_8_16] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f69 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 8 79) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 9)) := by
  have c483 := (publicBatchWrapper2_copies15 a h).2.2.2.1
  have c484 := (publicBatchWrapper2_copies15 a h).2.2.2.2.1
  have c485 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.1
  have c486 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.1
  have c487 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_8_18 := arithEq_of_rows h (row := 8) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_18
  simp only [← c483, ← c484, ← c485] at e_8_18
  have e_8_19 := arithEq_of_rows h (row := 8) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_19
  simp only [← c486, ← c487, k0, ← c488] at e_8_19
  have hr := e_8_19
  simp only [e_8_18] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f70 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 7) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 10)) := by
  have c489 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c489, ← c490, ← c491] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c492, ← c493, k0, ← c494] at e_11_1
  have hr := e_11_1
  simp only [e_11_0] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f71 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 15) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 11)) := by
  have c495 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c495, ← c496, ← c497] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c498, ← c499, k0, ← c500] at e_11_3
  have hr := e_11_3
  simp only [e_11_2] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f72 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 23) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 12)) := by
  have c501 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c501, ← c502, ← c503] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c504, ← c505, k0, ← c506] at e_11_5
  have hr := e_11_5
  simp only [e_11_4] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f73 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 31) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 13)) := by
  have c507 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (publicBatchWrapper2_copies16 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c507, ← c508, ← c509] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c510, ← c511, k0, ← c512] at e_11_7
  have hr := e_11_7
  simp only [e_11_6] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f74 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 39) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 14)) := by
  have c513 := (publicBatchWrapper2_copies16 a h).2.1
  have c514 := (publicBatchWrapper2_copies16 a h).2.2.1
  have c515 := (publicBatchWrapper2_copies16 a h).2.2.2.1
  have c516 := (publicBatchWrapper2_copies16 a h).2.2.2.2.1
  have c517 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.1
  have c518 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c513, ← c514, ← c515] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c516, ← c517, k0, ← c518] at e_11_9
  have hr := e_11_9
  simp only [e_11_8] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f75 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 47) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 15)) := by
  have c519 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c519, ← c520, ← c521] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c522, ← c523, k0, ← c524] at e_11_11
  have hr := e_11_11
  simp only [e_11_10] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f76 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 55) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 16)) := by
  have c525 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c525, ← c526, ← c527] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c528, ← c529, k0, ← c530] at e_11_13
  have hr := e_11_13
  simp only [e_11_12] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f77 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 63) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 17)) := by
  have c531 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c532 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c535 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c531, ← c532, ← c533] at e_11_14
  have e_11_15 := arithEq_of_rows h (row := 11) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c534, ← c535, k0, ← c536] at e_11_15
  have hr := e_11_15
  simp only [e_11_14] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f78 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 71) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 38)) := by
  have c537 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c541 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_16 := arithEq_of_rows h (row := 11) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c537, ← c538, ← c539] at e_11_16
  have e_11_17 := arithEq_of_rows h (row := 11) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c540, ← c541, k0, ← c542] at e_11_17
  have hr := e_11_17
  simp only [e_11_16] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f79 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 79) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 39)) := by
  have c543 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (publicBatchWrapper2_copies17 a h).1
  have c545 := (publicBatchWrapper2_copies17 a h).2.1
  have c546 := (publicBatchWrapper2_copies17 a h).2.2.1
  have c547 := (publicBatchWrapper2_copies17 a h).2.2.2.1
  have c548 := (publicBatchWrapper2_copies17 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_11_18 := arithEq_of_rows h (row := 11) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c543, ← c544, ← c545] at e_11_18
  have e_11_19 := arithEq_of_rows h (row := 11) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c546, ← c547, k0, ← c548] at e_11_19
  have hr := e_11_19
  simp only [e_11_18] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f80 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 7) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 40)) := by
  have c549 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.1
  have c550 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.1
  have c551 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c549, ← c550, ← c551] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c552, ← c553, k0, ← c554] at e_12_1
  have hr := e_12_1
  simp only [e_12_0] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f81 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 15) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 41)) := by
  have c555 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c555, ← c556, ← c557] at e_12_2
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c558, ← c559, k0, ← c560] at e_12_3
  have hr := e_12_3
  simp only [e_12_2] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f82 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 23) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 42)) := by
  have c561 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c561, ← c562, ← c563] at e_12_4
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c564, ← c565, k0, ← c566] at e_12_5
  have hr := e_12_5
  simp only [e_12_4] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f83 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 31) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 43)) := by
  have c567 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c567, ← c568, ← c569] at e_12_6
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c570, ← c571, k0, ← c572] at e_12_7
  have hr := e_12_7
  simp only [e_12_6] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f84 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 39) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 44)) := by
  have c573 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (publicBatchWrapper2_copies18 a h).1
  have c577 := (publicBatchWrapper2_copies18 a h).2.1
  have c578 := (publicBatchWrapper2_copies18 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c573, ← c574, ← c575] at e_12_8
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c576, ← c577, k0, ← c578] at e_12_9
  have hr := e_12_9
  simp only [e_12_8] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f85 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 47) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 45)) := by
  have c579 := (publicBatchWrapper2_copies18 a h).2.2.2.1
  have c580 := (publicBatchWrapper2_copies18 a h).2.2.2.2.1
  have c581 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.1
  have c582 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.1
  have c583 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c579, ← c580, ← c581] at e_12_10
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c582, ← c583, k0, ← c584] at e_12_11
  have hr := e_12_11
  simp only [e_12_10] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f86 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 55) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 46)) := by
  have c585 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c585, ← c586, ← c587] at e_12_12
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c588, ← c589, k0, ← c590] at e_12_13
  have hr := e_12_13
  simp only [e_12_12] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f87 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 63) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 47)) := by
  have c591 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c591, ← c592, ← c593] at e_12_14
  have e_12_15 := arithEq_of_rows h (row := 12) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c594, ← c595, k0, ← c596] at e_12_15
  have hr := e_12_15
  simp only [e_12_14] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f88 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 71) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 18)) := by
  have c597 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c600 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_16 := arithEq_of_rows h (row := 12) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c597, ← c598, ← c599] at e_12_16
  have e_12_17 := arithEq_of_rows h (row := 12) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c600, ← c601, k0, ← c602] at e_12_17
  have hr := e_12_17
  simp only [e_12_16] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f89 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 79) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 19)) := by
  have c603 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (publicBatchWrapper2_copies19 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_12_18 := arithEq_of_rows h (row := 12) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c603, ← c604, ← c605] at e_12_18
  have e_12_19 := arithEq_of_rows h (row := 12) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c606, ← c607, k0, ← c608] at e_12_19
  have hr := e_12_19
  simp only [e_12_18] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f90 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 7) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 20)) := by
  have c609 := (publicBatchWrapper2_copies19 a h).2.1
  have c610 := (publicBatchWrapper2_copies19 a h).2.2.1
  have c611 := (publicBatchWrapper2_copies19 a h).2.2.2.1
  have c612 := (publicBatchWrapper2_copies19 a h).2.2.2.2.1
  have c613 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.1
  have c614 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c609, ← c610, ← c611] at e_13_0
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c612, ← c613, k0, ← c614] at e_13_1
  have hr := e_13_1
  simp only [e_13_0] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f91 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 15) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 21)) := by
  have c615 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c615, ← c616, ← c617] at e_13_2
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c618, ← c619, k0, ← c620] at e_13_3
  have hr := e_13_3
  simp only [e_13_2] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f92 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 23) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 22)) := by
  have c621 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c621, ← c622, ← c623] at e_13_4
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c624, ← c625, k0, ← c626] at e_13_5
  have hr := e_13_5
  simp only [e_13_4] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f93 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 31) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 23)) := by
  have c627 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c627, ← c628, ← c629] at e_13_6
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c630, ← c631, k0, ← c632] at e_13_7
  have hr := e_13_7
  simp only [e_13_6] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f94 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 39) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 24)) := by
  have c633 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c633, ← c634, ← c635] at e_13_8
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c636, ← c637, k0, ← c638] at e_13_9
  have hr := e_13_9
  simp only [e_13_8] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f95 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 47) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 25)) := by
  have c639 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (publicBatchWrapper2_copies20 a h).1
  have c641 := (publicBatchWrapper2_copies20 a h).2.1
  have c642 := (publicBatchWrapper2_copies20 a h).2.2.1
  have c643 := (publicBatchWrapper2_copies20 a h).2.2.2.1
  have c644 := (publicBatchWrapper2_copies20 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c639, ← c640, ← c641] at e_13_10
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c642, ← c643, k0, ← c644] at e_13_11
  have hr := e_13_11
  simp only [e_13_10] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f96 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 55) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 48)) := by
  have c645 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.1
  have c646 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.1
  have c647 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c645, ← c646, ← c647] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c648, ← c649, k0, ← c650] at e_13_13
  have hr := e_13_13
  simp only [e_13_12] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f97 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 63) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 49)) := by
  have c651 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c651, ← c652, ← c653] at e_13_14
  have e_13_15 := arithEq_of_rows h (row := 13) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c654, ← c655, k0, ← c656] at e_13_15
  have hr := e_13_15
  simp only [e_13_14] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f98 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 71) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 50)) := by
  have c657 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_16 := arithEq_of_rows h (row := 13) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c657, ← c658, ← c659] at e_13_16
  have e_13_17 := arithEq_of_rows h (row := 13) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c660, ← c661, k0, ← c662] at e_13_17
  have hr := e_13_17
  simp only [e_13_16] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f99 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 79) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 51)) := by
  have c663 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_13_18 := arithEq_of_rows h (row := 13) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c663, ← c664, ← c665] at e_13_18
  have e_13_19 := arithEq_of_rows h (row := 13) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c666, ← c667, k0, ← c668] at e_13_19
  have hr := e_13_19
  simp only [e_13_18] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f100 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 7) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 52)) := by
  have c669 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (publicBatchWrapper2_copies21 a h).1
  have c673 := (publicBatchWrapper2_copies21 a h).2.1
  have c674 := (publicBatchWrapper2_copies21 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c669, ← c670, ← c671] at e_14_0
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c672, ← c673, k0, ← c674] at e_14_1
  have hr := e_14_1
  simp only [e_14_0] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f101 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 15) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 53)) := by
  have c675 := (publicBatchWrapper2_copies21 a h).2.2.2.1
  have c676 := (publicBatchWrapper2_copies21 a h).2.2.2.2.1
  have c677 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.1
  have c678 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.1
  have c679 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c675, ← c676, ← c677] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c678, ← c679, k0, ← c680] at e_14_3
  have hr := e_14_3
  simp only [e_14_2] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f102 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 23) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 54)) := by
  have c681 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c681, ← c682, ← c683] at e_14_4
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c684, ← c685, k0, ← c686] at e_14_5
  have hr := e_14_5
  simp only [e_14_4] at hr
  simp only [bselect, k0]
  linear_combination hr

theorem publicBatchWrapper2_f103 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 31) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 55)) := by
  have c687 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper2_consts a h
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c687, ← c688, ← c689] at e_14_6
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c690, ← c691, k0, ← c692] at e_14_7
  have hr := e_14_7
  simp only [e_14_6] at hr
  simp only [bselect, k0]
  linear_combination hr

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `publicBatchWrapper2` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem publicBatchWrapper2_decode (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    (a (.virt 0) = a (.virt 29) ∧
    a (.virt 30) = a (.virt 29) ∧
    IsEqual (a (.virt 3)) (a (.virt 26)) (a (.virt 60)) (a (.virt 61)) ∧
    IsEqual (a (.virt 4)) (a (.virt 26)) (a (.virt 62)) (a (.virt 63)) ∧
    IsEqual (a (.virt 5)) (a (.virt 26)) (a (.virt 64)) (a (.virt 65)) ∧
    IsEqual (a (.virt 6)) (a (.virt 26)) (a (.virt 66)) (a (.virt 67)) ∧
    a (.wire 3 3) = band (a (.virt 60)) (a (.virt 62)) ∧
    a (.wire 3 7) = band (a (.virt 64)) (a (.virt 66)) ∧
    a (.wire 3 11) = band (a (.wire 3 3)) (a (.wire 3 7)) ∧
    IsEqual (a (.virt 33)) (a (.virt 26)) (a (.virt 68)) (a (.virt 69)) ∧
    IsEqual (a (.virt 34)) (a (.virt 26)) (a (.virt 70)) (a (.virt 71)) ∧
    IsEqual (a (.virt 35)) (a (.virt 26)) (a (.virt 72)) (a (.virt 73)) ∧
    IsEqual (a (.virt 36)) (a (.virt 26)) (a (.virt 74)) (a (.virt 75)) ∧
    a (.wire 3 47) = band (a (.virt 68)) (a (.virt 70)) ∧
    a (.wire 3 51) = band (a (.virt 72)) (a (.virt 74)) ∧
    a (.wire 3 55) = band (a (.wire 3 47)) (a (.wire 3 51)) ∧
    a (.wire 2 67) = bnot (a (.wire 3 11)) ∧
    a (.virt 28) = bnot (a (.virt 26)) ∧
    a (.wire 2 67) = band (a (.wire 2 67)) (a (.virt 28)) ∧
    a (.wire 2 71) = bselect (a (.wire 2 67)) (a (.virt 3)) (a (.virt 26)) ∧
    a (.wire 2 75) = bselect (a (.wire 2 67)) (a (.virt 4)) (a (.virt 26)) ∧
    a (.wire 2 79) = bselect (a (.wire 2 67)) (a (.virt 5)) (a (.virt 26)) ∧
    a (.wire 4 3) = bselect (a (.wire 2 67)) (a (.virt 6)) (a (.virt 26)) ∧
    a (.wire 4 7) = bselect (a (.wire 2 67)) (a (.virt 7)) (a (.virt 26)) ∧
    a (.wire 4 11) = bselect (a (.wire 2 67)) (a (.virt 1)) (a (.virt 26)) ∧
    a (.wire 4 15) = bselect (a (.wire 2 67)) (a (.virt 2)) (a (.virt 26)) ∧
    a (.wire 2 67) = bor (a (.virt 26)) (a (.wire 2 67)) ∧
    a (.wire 4 19) = bnot (a (.wire 3 55)) ∧
    a (.wire 4 23) = bnot (a (.wire 2 67)) ∧
    a (.wire 3 59) = band (a (.wire 4 19)) (a (.wire 4 23)) ∧
    a (.wire 4 31) = bselect (a (.wire 3 59)) (a (.virt 33)) (a (.wire 2 71)) ∧
    a (.wire 4 39) = bselect (a (.wire 3 59)) (a (.virt 34)) (a (.wire 2 75))) ∧
    (a (.wire 4 47) = bselect (a (.wire 3 59)) (a (.virt 35)) (a (.wire 2 79)) ∧
    a (.wire 4 55) = bselect (a (.wire 3 59)) (a (.virt 36)) (a (.wire 4 3)) ∧
    a (.wire 4 63) = bselect (a (.wire 3 59)) (a (.virt 37)) (a (.wire 4 7)) ∧
    a (.wire 4 71) = bselect (a (.wire 3 59)) (a (.virt 31)) (a (.wire 4 11)) ∧
    a (.wire 4 79) = bselect (a (.wire 3 59)) (a (.virt 32)) (a (.wire 4 15)) ∧
    a (.wire 0 67) = bor (a (.wire 2 67)) (a (.wire 4 19)) ∧
    IsEqual (a (.virt 1)) (a (.wire 4 71)) (a (.virt 76)) (a (.virt 77)) ∧
    a (.wire 0 71) = bor (a (.wire 3 11)) (a (.virt 76)) ∧
    a (.wire 0 71) = a (.virt 28) ∧
    IsEqual (a (.virt 2)) (a (.wire 4 79)) (a (.virt 78)) (a (.virt 79)) ∧
    a (.wire 0 75) = bor (a (.wire 3 11)) (a (.virt 78)) ∧
    a (.wire 0 75) = a (.virt 28) ∧
    IsEqual (a (.virt 3)) (a (.wire 4 31)) (a (.virt 80)) (a (.virt 81)) ∧
    IsEqual (a (.virt 4)) (a (.wire 4 39)) (a (.virt 82)) (a (.virt 83)) ∧
    IsEqual (a (.virt 5)) (a (.wire 4 47)) (a (.virt 84)) (a (.virt 85)) ∧
    IsEqual (a (.virt 6)) (a (.wire 4 55)) (a (.virt 86)) (a (.virt 87)) ∧
    a (.wire 7 31) = band (a (.virt 80)) (a (.virt 82)) ∧
    a (.wire 7 35) = band (a (.virt 84)) (a (.virt 86)) ∧
    a (.wire 7 39) = band (a (.wire 7 31)) (a (.wire 7 35)) ∧
    a (.wire 0 79) = bor (a (.wire 3 11)) (a (.wire 7 39)) ∧
    a (.wire 0 79) = a (.virt 28) ∧
    IsEqual (a (.virt 31)) (a (.wire 4 71)) (a (.virt 88)) (a (.virt 89)) ∧
    a (.wire 9 3) = bor (a (.wire 3 55)) (a (.virt 88)) ∧
    a (.wire 9 3) = a (.virt 28) ∧
    IsEqual (a (.virt 32)) (a (.wire 4 79)) (a (.virt 90)) (a (.virt 91)) ∧
    a (.wire 9 7) = bor (a (.wire 3 55)) (a (.virt 90)) ∧
    a (.wire 9 7) = a (.virt 28) ∧
    IsEqual (a (.virt 33)) (a (.wire 4 31)) (a (.virt 92)) (a (.virt 93)) ∧
    IsEqual (a (.virt 34)) (a (.wire 4 39)) (a (.virt 94)) (a (.virt 95)) ∧
    IsEqual (a (.virt 35)) (a (.wire 4 47)) (a (.virt 96)) (a (.virt 97)) ∧
    IsEqual (a (.virt 36)) (a (.wire 4 55)) (a (.virt 98)) (a (.virt 99)) ∧
    a (.wire 10 11) = band (a (.virt 92)) (a (.virt 94))) ∧
    (a (.wire 10 15) = band (a (.virt 96)) (a (.virt 98)) ∧
    a (.wire 10 19) = band (a (.wire 10 11)) (a (.wire 10 15)) ∧
    a (.wire 9 11) = bor (a (.wire 3 55)) (a (.wire 10 19)) ∧
    a (.wire 9 11) = a (.virt 28) ∧
    a (.wire 8 71) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 8)) ∧
    a (.wire 8 79) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 9)) ∧
    a (.wire 11 7) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 10)) ∧
    a (.wire 11 15) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 11)) ∧
    a (.wire 11 23) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 12)) ∧
    a (.wire 11 31) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 13)) ∧
    a (.wire 11 39) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 14)) ∧
    a (.wire 11 47) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 15)) ∧
    a (.wire 11 55) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 16)) ∧
    a (.wire 11 63) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 17)) ∧
    a (.wire 11 71) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 38)) ∧
    a (.wire 11 79) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 39)) ∧
    a (.wire 12 7) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 40)) ∧
    a (.wire 12 15) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 41)) ∧
    a (.wire 12 23) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 42)) ∧
    a (.wire 12 31) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 43)) ∧
    a (.wire 12 39) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 44)) ∧
    a (.wire 12 47) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 45)) ∧
    a (.wire 12 55) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 46)) ∧
    a (.wire 12 63) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 47)) ∧
    a (.wire 12 71) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 18)) ∧
    a (.wire 12 79) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 19)) ∧
    a (.wire 13 7) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 20)) ∧
    a (.wire 13 15) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 21)) ∧
    a (.wire 13 23) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 22)) ∧
    a (.wire 13 31) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 23)) ∧
    a (.wire 13 39) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 24)) ∧
    a (.wire 13 47) = bselect (a (.wire 3 11)) (a (.virt 26)) (a (.virt 25))) ∧
    (a (.wire 13 55) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 48)) ∧
    a (.wire 13 63) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 49)) ∧
    a (.wire 13 71) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 50)) ∧
    a (.wire 13 79) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 51)) ∧
    a (.wire 14 7) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 52)) ∧
    a (.wire 14 15) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 53)) ∧
    a (.wire 14 23) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 54)) ∧
    a (.wire 14 31) = bselect (a (.wire 3 55)) (a (.virt 26)) (a (.virt 55))) :=
  ⟨⟨publicBatchWrapper2_f0 a h, publicBatchWrapper2_f1 a h, publicBatchWrapper2_f2 a h, publicBatchWrapper2_f3 a h, publicBatchWrapper2_f4 a h, publicBatchWrapper2_f5 a h, publicBatchWrapper2_f6 a h, publicBatchWrapper2_f7 a h, publicBatchWrapper2_f8 a h, publicBatchWrapper2_f9 a h, publicBatchWrapper2_f10 a h, publicBatchWrapper2_f11 a h, publicBatchWrapper2_f12 a h, publicBatchWrapper2_f13 a h, publicBatchWrapper2_f14 a h, publicBatchWrapper2_f15 a h, publicBatchWrapper2_f16 a h, publicBatchWrapper2_f17 a h, publicBatchWrapper2_f18 a h, publicBatchWrapper2_f19 a h, publicBatchWrapper2_f20 a h, publicBatchWrapper2_f21 a h, publicBatchWrapper2_f22 a h, publicBatchWrapper2_f23 a h, publicBatchWrapper2_f24 a h, publicBatchWrapper2_f25 a h, publicBatchWrapper2_f26 a h, publicBatchWrapper2_f27 a h, publicBatchWrapper2_f28 a h, publicBatchWrapper2_f29 a h, publicBatchWrapper2_f30 a h, publicBatchWrapper2_f31 a h⟩, ⟨publicBatchWrapper2_f32 a h, publicBatchWrapper2_f33 a h, publicBatchWrapper2_f34 a h, publicBatchWrapper2_f35 a h, publicBatchWrapper2_f36 a h, publicBatchWrapper2_f37 a h, publicBatchWrapper2_f38 a h, publicBatchWrapper2_f39 a h, publicBatchWrapper2_f40 a h, publicBatchWrapper2_f41 a h, publicBatchWrapper2_f42 a h, publicBatchWrapper2_f43 a h, publicBatchWrapper2_f44 a h, publicBatchWrapper2_f45 a h, publicBatchWrapper2_f46 a h, publicBatchWrapper2_f47 a h, publicBatchWrapper2_f48 a h, publicBatchWrapper2_f49 a h, publicBatchWrapper2_f50 a h, publicBatchWrapper2_f51 a h, publicBatchWrapper2_f52 a h, publicBatchWrapper2_f53 a h, publicBatchWrapper2_f54 a h, publicBatchWrapper2_f55 a h, publicBatchWrapper2_f56 a h, publicBatchWrapper2_f57 a h, publicBatchWrapper2_f58 a h, publicBatchWrapper2_f59 a h, publicBatchWrapper2_f60 a h, publicBatchWrapper2_f61 a h, publicBatchWrapper2_f62 a h, publicBatchWrapper2_f63 a h⟩, ⟨publicBatchWrapper2_f64 a h, publicBatchWrapper2_f65 a h, publicBatchWrapper2_f66 a h, publicBatchWrapper2_f67 a h, publicBatchWrapper2_f68 a h, publicBatchWrapper2_f69 a h, publicBatchWrapper2_f70 a h, publicBatchWrapper2_f71 a h, publicBatchWrapper2_f72 a h, publicBatchWrapper2_f73 a h, publicBatchWrapper2_f74 a h, publicBatchWrapper2_f75 a h, publicBatchWrapper2_f76 a h, publicBatchWrapper2_f77 a h, publicBatchWrapper2_f78 a h, publicBatchWrapper2_f79 a h, publicBatchWrapper2_f80 a h, publicBatchWrapper2_f81 a h, publicBatchWrapper2_f82 a h, publicBatchWrapper2_f83 a h, publicBatchWrapper2_f84 a h, publicBatchWrapper2_f85 a h, publicBatchWrapper2_f86 a h, publicBatchWrapper2_f87 a h, publicBatchWrapper2_f88 a h, publicBatchWrapper2_f89 a h, publicBatchWrapper2_f90 a h, publicBatchWrapper2_f91 a h, publicBatchWrapper2_f92 a h, publicBatchWrapper2_f93 a h, publicBatchWrapper2_f94 a h, publicBatchWrapper2_f95 a h⟩, ⟨publicBatchWrapper2_f96 a h, publicBatchWrapper2_f97 a h, publicBatchWrapper2_f98 a h, publicBatchWrapper2_f99 a h, publicBatchWrapper2_f100 a h, publicBatchWrapper2_f101 a h, publicBatchWrapper2_f102 a h, publicBatchWrapper2_f103 a h⟩⟩

end Plonky2Spec.Generated
