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

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

variable {p : ℕ} [Fact p.Prime]

/-- `publicBatchWrapper2.copies`, items `0..32`. -/
def publicBatchWrapper2.copies0 : List (Target × Target) := [
    (.virt 19078, .wire 0 0),
    (.virt 19078, .wire 0 1),
    (.virt 19080, .wire 0 2),
    (.virt 19080, .wire 1 0),
    (.virt 9488, .wire 1 1),
    (.virt 19080, .wire 1 2),
    (.virt 9488, .wire 1 4),
    (.virt 19081, .wire 1 5),
    (.virt 9488, .wire 1 6),
    (.wire 1 7, .wire 0 4),
    (.virt 19078, .wire 0 5),
    (.wire 0 3, .wire 0 6),
    (.wire 1 3, .virt 19079),
    (.wire 0 7, .virt 19079),
    (.virt 19078, .wire 0 8),
    (.virt 19078, .wire 0 9),
    (.virt 19082, .wire 0 10),
    (.virt 19082, .wire 1 8),
    (.virt 9489, .wire 1 9),
    (.virt 19082, .wire 1 10),
    (.virt 9489, .wire 1 12),
    (.virt 19083, .wire 1 13),
    (.virt 9489, .wire 1 14),
    (.wire 1 15, .wire 0 12),
    (.virt 19078, .wire 0 13),
    (.wire 0 11, .wire 0 14),
    (.wire 1 11, .virt 19079),
    (.wire 0 15, .virt 19079),
    (.virt 19078, .wire 0 16),
    (.virt 19078, .wire 0 17),
    (.virt 19084, .wire 0 18),
    (.virt 19084, .wire 1 16)
  ]

/-- `publicBatchWrapper2.copies`, items `32..64`. -/
def publicBatchWrapper2.copies1 : List (Target × Target) := [
    (.virt 9490, .wire 1 17),
    (.virt 19084, .wire 1 18),
    (.virt 9490, .wire 1 20),
    (.virt 19085, .wire 1 21),
    (.virt 9490, .wire 1 22),
    (.wire 1 23, .wire 0 20),
    (.virt 19078, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.wire 1 19, .virt 19079),
    (.wire 0 23, .virt 19079),
    (.virt 19078, .wire 0 24),
    (.virt 19078, .wire 0 25),
    (.virt 19086, .wire 0 26),
    (.virt 19086, .wire 1 24),
    (.virt 9491, .wire 1 25),
    (.virt 19086, .wire 1 26),
    (.virt 9491, .wire 1 28),
    (.virt 19087, .wire 1 29),
    (.virt 9491, .wire 1 30),
    (.wire 1 31, .wire 0 28),
    (.virt 19078, .wire 0 29),
    (.wire 0 27, .wire 0 30),
    (.wire 1 27, .virt 19079),
    (.wire 0 31, .virt 19079),
    (.virt 19080, .wire 1 32),
    (.virt 19082, .wire 1 33),
    (.virt 19080, .wire 1 34),
    (.virt 19084, .wire 1 36),
    (.virt 19086, .wire 1 37),
    (.virt 19084, .wire 1 38),
    (.wire 1 35, .wire 1 40),
    (.wire 1 39, .wire 1 41)
  ]

/-- `publicBatchWrapper2.copies`, items `64..96`. -/
def publicBatchWrapper2.copies2 : List (Target × Target) := [
    (.wire 1 35, .wire 1 42),
    (.virt 19078, .wire 0 32),
    (.virt 19078, .wire 0 33),
    (.virt 19088, .wire 0 34),
    (.virt 19088, .wire 1 44),
    (.virt 19025, .wire 1 45),
    (.virt 19088, .wire 1 46),
    (.virt 19025, .wire 1 48),
    (.virt 19089, .wire 1 49),
    (.virt 19025, .wire 1 50),
    (.wire 1 51, .wire 0 36),
    (.virt 19078, .wire 0 37),
    (.wire 0 35, .wire 0 38),
    (.wire 1 47, .virt 19079),
    (.wire 0 39, .virt 19079),
    (.virt 19078, .wire 0 40),
    (.virt 19078, .wire 0 41),
    (.virt 19090, .wire 0 42),
    (.virt 19090, .wire 1 52),
    (.virt 19026, .wire 1 53),
    (.virt 19090, .wire 1 54),
    (.virt 19026, .wire 1 56),
    (.virt 19091, .wire 1 57),
    (.virt 19026, .wire 1 58),
    (.wire 1 59, .wire 0 44),
    (.virt 19078, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.wire 1 55, .virt 19079),
    (.wire 0 47, .virt 19079),
    (.virt 19078, .wire 0 48),
    (.virt 19078, .wire 0 49),
    (.virt 19092, .wire 0 50)
  ]

/-- `publicBatchWrapper2.copies`, items `96..128`. -/
def publicBatchWrapper2.copies3 : List (Target × Target) := [
    (.virt 19092, .wire 1 60),
    (.virt 19027, .wire 1 61),
    (.virt 19092, .wire 1 62),
    (.virt 19027, .wire 1 64),
    (.virt 19093, .wire 1 65),
    (.virt 19027, .wire 1 66),
    (.wire 1 67, .wire 0 52),
    (.virt 19078, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.wire 1 63, .virt 19079),
    (.wire 0 55, .virt 19079),
    (.virt 19078, .wire 0 56),
    (.virt 19078, .wire 0 57),
    (.virt 19094, .wire 0 58),
    (.virt 19094, .wire 1 68),
    (.virt 19028, .wire 1 69),
    (.virt 19094, .wire 1 70),
    (.virt 19028, .wire 1 72),
    (.virt 19095, .wire 1 73),
    (.virt 19028, .wire 1 74),
    (.wire 1 75, .wire 0 60),
    (.virt 19078, .wire 0 61),
    (.wire 0 59, .wire 0 62),
    (.wire 1 71, .virt 19079),
    (.wire 0 63, .virt 19079),
    (.virt 19088, .wire 1 76),
    (.virt 19090, .wire 1 77),
    (.virt 19088, .wire 1 78),
    (.virt 19092, .wire 2 0),
    (.virt 19094, .wire 2 1),
    (.virt 19092, .wire 2 2),
    (.wire 1 79, .wire 2 4)
  ]

/-- `publicBatchWrapper2.copies`, items `128..160`. -/
def publicBatchWrapper2.copies4 : List (Target × Target) := [
    (.wire 2 3, .wire 2 5),
    (.wire 1 79, .wire 2 6),
    (.virt 19078, .wire 0 64),
    (.virt 19078, .wire 0 65),
    (.wire 1 43, .wire 0 66),
    (.wire 0 67, .wire 0 68),
    (.virt 9488, .wire 0 69),
    (.virt 19079, .wire 0 70),
    (.wire 0 67, .wire 0 72),
    (.virt 9489, .wire 0 73),
    (.virt 19079, .wire 0 74),
    (.wire 0 67, .wire 0 76),
    (.virt 9490, .wire 0 77),
    (.virt 19079, .wire 0 78),
    (.wire 0 67, .wire 3 0),
    (.virt 9491, .wire 3 1),
    (.virt 19079, .wire 3 2),
    (.wire 0 67, .wire 3 4),
    (.virt 9492, .wire 3 5),
    (.virt 19079, .wire 3 6),
    (.wire 0 67, .wire 3 8),
    (.virt 9486, .wire 3 9),
    (.virt 19079, .wire 3 10),
    (.wire 0 67, .wire 3 12),
    (.virt 9487, .wire 3 13),
    (.virt 19079, .wire 3 14),
    (.virt 19078, .wire 3 16),
    (.virt 19078, .wire 3 17),
    (.wire 2 7, .wire 3 18),
    (.virt 19078, .wire 3 20),
    (.virt 19078, .wire 3 21),
    (.wire 0 67, .wire 3 22)
  ]

/-- `publicBatchWrapper2.copies`, items `160..192`. -/
def publicBatchWrapper2.copies5 : List (Target × Target) := [
    (.wire 3 19, .wire 2 8),
    (.wire 3 23, .wire 2 9),
    (.wire 3 19, .wire 2 10),
    (.wire 2 11, .wire 3 24),
    (.wire 0 71, .wire 3 25),
    (.wire 0 71, .wire 3 26),
    (.wire 2 11, .wire 3 28),
    (.virt 19025, .wire 3 29),
    (.wire 3 27, .wire 3 30),
    (.wire 2 11, .wire 3 32),
    (.wire 0 75, .wire 3 33),
    (.wire 0 75, .wire 3 34),
    (.wire 2 11, .wire 3 36),
    (.virt 19026, .wire 3 37),
    (.wire 3 35, .wire 3 38),
    (.wire 2 11, .wire 3 40),
    (.wire 0 79, .wire 3 41),
    (.wire 0 79, .wire 3 42),
    (.wire 2 11, .wire 3 44),
    (.virt 19027, .wire 3 45),
    (.wire 3 43, .wire 3 46),
    (.wire 2 11, .wire 3 48),
    (.wire 3 3, .wire 3 49),
    (.wire 3 3, .wire 3 50),
    (.wire 2 11, .wire 3 52),
    (.virt 19028, .wire 3 53),
    (.wire 3 51, .wire 3 54),
    (.wire 2 11, .wire 3 56),
    (.wire 3 7, .wire 3 57),
    (.wire 3 7, .wire 3 58),
    (.wire 2 11, .wire 3 60),
    (.virt 19029, .wire 3 61)
  ]

/-- `publicBatchWrapper2.copies`, items `192..224`. -/
def publicBatchWrapper2.copies6 : List (Target × Target) := [
    (.wire 3 59, .wire 3 62),
    (.wire 2 11, .wire 3 64),
    (.wire 3 11, .wire 3 65),
    (.wire 3 11, .wire 3 66),
    (.wire 2 11, .wire 3 68),
    (.virt 19023, .wire 3 69),
    (.wire 3 67, .wire 3 70),
    (.wire 2 11, .wire 3 72),
    (.wire 3 15, .wire 3 73),
    (.wire 3 15, .wire 3 74),
    (.wire 2 11, .wire 3 76),
    (.virt 19024, .wire 3 77),
    (.wire 3 75, .wire 3 78),
    (.wire 0 67, .wire 4 0),
    (.wire 3 19, .wire 4 1),
    (.wire 0 67, .wire 4 2),
    (.wire 4 3, .wire 5 0),
    (.virt 19078, .wire 5 1),
    (.wire 3 19, .wire 5 2),
    (.virt 19078, .wire 6 0),
    (.virt 19078, .wire 6 1),
    (.virt 19096, .wire 6 2),
    (.virt 9486, .wire 6 4),
    (.virt 19078, .wire 6 5),
    (.wire 3 71, .wire 6 6),
    (.virt 19096, .wire 2 12),
    (.wire 6 7, .wire 2 13),
    (.virt 19096, .wire 2 14),
    (.wire 6 7, .wire 2 16),
    (.virt 19097, .wire 2 17),
    (.wire 6 7, .wire 2 18),
    (.wire 2 19, .wire 6 8)
  ]

/-- `publicBatchWrapper2.copies`, items `224..256`. -/
def publicBatchWrapper2.copies7 : List (Target × Target) := [
    (.virt 19078, .wire 6 9),
    (.wire 6 3, .wire 6 10),
    (.wire 2 15, .virt 19079),
    (.wire 6 11, .virt 19079),
    (.wire 1 43, .wire 4 4),
    (.virt 19096, .wire 4 5),
    (.wire 1 43, .wire 4 6),
    (.wire 4 7, .wire 5 4),
    (.virt 19078, .wire 5 5),
    (.virt 19096, .wire 5 6),
    (.wire 5 7, .virt 19078),
    (.virt 19078, .wire 6 12),
    (.virt 19078, .wire 6 13),
    (.virt 19098, .wire 6 14),
    (.virt 9487, .wire 6 16),
    (.virt 19078, .wire 6 17),
    (.wire 3 79, .wire 6 18),
    (.virt 19098, .wire 2 20),
    (.wire 6 19, .wire 2 21),
    (.virt 19098, .wire 2 22),
    (.wire 6 19, .wire 2 24),
    (.virt 19099, .wire 2 25),
    (.wire 6 19, .wire 2 26),
    (.wire 2 27, .wire 6 20),
    (.virt 19078, .wire 6 21),
    (.wire 6 15, .wire 6 22),
    (.wire 2 23, .virt 19079),
    (.wire 6 23, .virt 19079),
    (.wire 1 43, .wire 4 8),
    (.virt 19098, .wire 4 9),
    (.wire 1 43, .wire 4 10),
    (.wire 4 11, .wire 5 8)
  ]

/-- `publicBatchWrapper2.copies`, items `256..288`. -/
def publicBatchWrapper2.copies8 : List (Target × Target) := [
    (.virt 19078, .wire 5 9),
    (.virt 19098, .wire 5 10),
    (.wire 5 11, .virt 19078),
    (.virt 19078, .wire 6 24),
    (.virt 19078, .wire 6 25),
    (.virt 19100, .wire 6 26),
    (.virt 9488, .wire 6 28),
    (.virt 19078, .wire 6 29),
    (.wire 3 31, .wire 6 30),
    (.virt 19100, .wire 2 28),
    (.wire 6 31, .wire 2 29),
    (.virt 19100, .wire 2 30),
    (.wire 6 31, .wire 2 32),
    (.virt 19101, .wire 2 33),
    (.wire 6 31, .wire 2 34),
    (.wire 2 35, .wire 6 32),
    (.virt 19078, .wire 6 33),
    (.wire 6 27, .wire 6 34),
    (.wire 2 31, .virt 19079),
    (.wire 6 35, .virt 19079),
    (.virt 19078, .wire 6 36),
    (.virt 19078, .wire 6 37),
    (.virt 19102, .wire 6 38),
    (.virt 9489, .wire 6 40),
    (.virt 19078, .wire 6 41),
    (.wire 3 39, .wire 6 42),
    (.virt 19102, .wire 2 36),
    (.wire 6 43, .wire 2 37),
    (.virt 19102, .wire 2 38),
    (.wire 6 43, .wire 2 40),
    (.virt 19103, .wire 2 41),
    (.wire 6 43, .wire 2 42)
  ]

/-- `publicBatchWrapper2.copies`, items `288..320`. -/
def publicBatchWrapper2.copies9 : List (Target × Target) := [
    (.wire 2 43, .wire 6 44),
    (.virt 19078, .wire 6 45),
    (.wire 6 39, .wire 6 46),
    (.wire 2 39, .virt 19079),
    (.wire 6 47, .virt 19079),
    (.virt 19078, .wire 6 48),
    (.virt 19078, .wire 6 49),
    (.virt 19104, .wire 6 50),
    (.virt 9490, .wire 6 52),
    (.virt 19078, .wire 6 53),
    (.wire 3 47, .wire 6 54),
    (.virt 19104, .wire 2 44),
    (.wire 6 55, .wire 2 45),
    (.virt 19104, .wire 2 46),
    (.wire 6 55, .wire 2 48),
    (.virt 19105, .wire 2 49),
    (.wire 6 55, .wire 2 50),
    (.wire 2 51, .wire 6 56),
    (.virt 19078, .wire 6 57),
    (.wire 6 51, .wire 6 58),
    (.wire 2 47, .virt 19079),
    (.wire 6 59, .virt 19079),
    (.virt 19078, .wire 6 60),
    (.virt 19078, .wire 6 61),
    (.virt 19106, .wire 6 62),
    (.virt 9491, .wire 6 64),
    (.virt 19078, .wire 6 65),
    (.wire 3 55, .wire 6 66),
    (.virt 19106, .wire 2 52),
    (.wire 6 67, .wire 2 53),
    (.virt 19106, .wire 2 54),
    (.wire 6 67, .wire 2 56)
  ]

/-- `publicBatchWrapper2.copies`, items `320..352`. -/
def publicBatchWrapper2.copies10 : List (Target × Target) := [
    (.virt 19107, .wire 2 57),
    (.wire 6 67, .wire 2 58),
    (.wire 2 59, .wire 6 68),
    (.virt 19078, .wire 6 69),
    (.wire 6 63, .wire 6 70),
    (.wire 2 55, .virt 19079),
    (.wire 6 71, .virt 19079),
    (.virt 19100, .wire 2 60),
    (.virt 19102, .wire 2 61),
    (.virt 19100, .wire 2 62),
    (.virt 19104, .wire 2 64),
    (.virt 19106, .wire 2 65),
    (.virt 19104, .wire 2 66),
    (.wire 2 63, .wire 2 68),
    (.wire 2 67, .wire 2 69),
    (.wire 2 63, .wire 2 70),
    (.wire 1 43, .wire 4 12),
    (.wire 2 71, .wire 4 13),
    (.wire 1 43, .wire 4 14),
    (.wire 4 15, .wire 5 12),
    (.virt 19078, .wire 5 13),
    (.wire 2 71, .wire 5 14),
    (.wire 5 15, .virt 19078),
    (.virt 19078, .wire 6 72),
    (.virt 19078, .wire 6 73),
    (.virt 19108, .wire 6 74),
    (.virt 19023, .wire 6 76),
    (.virt 19078, .wire 6 77),
    (.wire 3 71, .wire 6 78),
    (.virt 19108, .wire 2 72),
    (.wire 6 79, .wire 2 73),
    (.virt 19108, .wire 2 74)
  ]

/-- `publicBatchWrapper2.copies`, items `352..384`. -/
def publicBatchWrapper2.copies11 : List (Target × Target) := [
    (.wire 6 79, .wire 2 76),
    (.virt 19109, .wire 2 77),
    (.wire 6 79, .wire 2 78),
    (.wire 2 79, .wire 7 0),
    (.virt 19078, .wire 7 1),
    (.wire 6 75, .wire 7 2),
    (.wire 2 75, .virt 19079),
    (.wire 7 3, .virt 19079),
    (.wire 2 7, .wire 4 16),
    (.virt 19108, .wire 4 17),
    (.wire 2 7, .wire 4 18),
    (.wire 4 19, .wire 5 16),
    (.virt 19078, .wire 5 17),
    (.virt 19108, .wire 5 18),
    (.wire 5 19, .virt 19078),
    (.virt 19078, .wire 7 4),
    (.virt 19078, .wire 7 5),
    (.virt 19110, .wire 7 6),
    (.virt 19024, .wire 7 8),
    (.virt 19078, .wire 7 9),
    (.wire 3 79, .wire 7 10),
    (.virt 19110, .wire 8 0),
    (.wire 7 11, .wire 8 1),
    (.virt 19110, .wire 8 2),
    (.wire 7 11, .wire 8 4),
    (.virt 19111, .wire 8 5),
    (.wire 7 11, .wire 8 6),
    (.wire 8 7, .wire 7 12),
    (.virt 19078, .wire 7 13),
    (.wire 7 7, .wire 7 14),
    (.wire 8 3, .virt 19079),
    (.wire 7 15, .virt 19079)
  ]

/-- `publicBatchWrapper2.copies`, items `384..416`. -/
def publicBatchWrapper2.copies12 : List (Target × Target) := [
    (.wire 2 7, .wire 4 20),
    (.virt 19110, .wire 4 21),
    (.wire 2 7, .wire 4 22),
    (.wire 4 23, .wire 5 20),
    (.virt 19078, .wire 5 21),
    (.virt 19110, .wire 5 22),
    (.wire 5 23, .virt 19078),
    (.virt 19078, .wire 7 16),
    (.virt 19078, .wire 7 17),
    (.virt 19112, .wire 7 18),
    (.virt 19025, .wire 7 20),
    (.virt 19078, .wire 7 21),
    (.wire 3 31, .wire 7 22),
    (.virt 19112, .wire 8 8),
    (.wire 7 23, .wire 8 9),
    (.virt 19112, .wire 8 10),
    (.wire 7 23, .wire 8 12),
    (.virt 19113, .wire 8 13),
    (.wire 7 23, .wire 8 14),
    (.wire 8 15, .wire 7 24),
    (.virt 19078, .wire 7 25),
    (.wire 7 19, .wire 7 26),
    (.wire 8 11, .virt 19079),
    (.wire 7 27, .virt 19079),
    (.virt 19078, .wire 7 28),
    (.virt 19078, .wire 7 29),
    (.virt 19114, .wire 7 30),
    (.virt 19026, .wire 7 32),
    (.virt 19078, .wire 7 33),
    (.wire 3 39, .wire 7 34),
    (.virt 19114, .wire 8 16),
    (.wire 7 35, .wire 8 17)
  ]

/-- `publicBatchWrapper2.copies`, items `416..448`. -/
def publicBatchWrapper2.copies13 : List (Target × Target) := [
    (.virt 19114, .wire 8 18),
    (.wire 7 35, .wire 8 20),
    (.virt 19115, .wire 8 21),
    (.wire 7 35, .wire 8 22),
    (.wire 8 23, .wire 7 36),
    (.virt 19078, .wire 7 37),
    (.wire 7 31, .wire 7 38),
    (.wire 8 19, .virt 19079),
    (.wire 7 39, .virt 19079),
    (.virt 19078, .wire 7 40),
    (.virt 19078, .wire 7 41),
    (.virt 19116, .wire 7 42),
    (.virt 19027, .wire 7 44),
    (.virt 19078, .wire 7 45),
    (.wire 3 47, .wire 7 46),
    (.virt 19116, .wire 8 24),
    (.wire 7 47, .wire 8 25),
    (.virt 19116, .wire 8 26),
    (.wire 7 47, .wire 8 28),
    (.virt 19117, .wire 8 29),
    (.wire 7 47, .wire 8 30),
    (.wire 8 31, .wire 7 48),
    (.virt 19078, .wire 7 49),
    (.wire 7 43, .wire 7 50),
    (.wire 8 27, .virt 19079),
    (.wire 7 51, .virt 19079),
    (.virt 19078, .wire 7 52),
    (.virt 19078, .wire 7 53),
    (.virt 19118, .wire 7 54),
    (.virt 19028, .wire 7 56),
    (.virt 19078, .wire 7 57),
    (.wire 3 55, .wire 7 58)
  ]

/-- `publicBatchWrapper2.copies`, items `448..480`. -/
def publicBatchWrapper2.copies14 : List (Target × Target) := [
    (.virt 19118, .wire 8 32),
    (.wire 7 59, .wire 8 33),
    (.virt 19118, .wire 8 34),
    (.wire 7 59, .wire 8 36),
    (.virt 19119, .wire 8 37),
    (.wire 7 59, .wire 8 38),
    (.wire 8 39, .wire 7 60),
    (.virt 19078, .wire 7 61),
    (.wire 7 55, .wire 7 62),
    (.wire 8 35, .virt 19079),
    (.wire 7 63, .virt 19079),
    (.virt 19112, .wire 8 40),
    (.virt 19114, .wire 8 41),
    (.virt 19112, .wire 8 42),
    (.virt 19116, .wire 8 44),
    (.virt 19118, .wire 8 45),
    (.virt 19116, .wire 8 46),
    (.wire 8 43, .wire 8 48),
    (.wire 8 47, .wire 8 49),
    (.wire 8 43, .wire 8 50),
    (.wire 2 7, .wire 4 24),
    (.wire 8 51, .wire 4 25),
    (.wire 2 7, .wire 4 26),
    (.wire 4 27, .wire 5 24),
    (.virt 19078, .wire 5 25),
    (.wire 8 51, .wire 5 26),
    (.wire 5 27, .virt 19078),
    (.wire 1 43, .wire 7 64),
    (.virt 9493, .wire 7 65),
    (.virt 9493, .wire 7 66),
    (.wire 1 43, .wire 7 68),
    (.virt 19079, .wire 7 69)
  ]

/-- `publicBatchWrapper2.copies`, items `480..512`. -/
def publicBatchWrapper2.copies15 : List (Target × Target) := [
    (.wire 7 67, .wire 7 70),
    (.wire 1 43, .wire 7 72),
    (.virt 9494, .wire 7 73),
    (.virt 9494, .wire 7 74),
    (.wire 1 43, .wire 7 76),
    (.virt 19079, .wire 7 77),
    (.wire 7 75, .wire 7 78),
    (.wire 1 43, .wire 9 0),
    (.virt 9495, .wire 9 1),
    (.virt 9495, .wire 9 2),
    (.wire 1 43, .wire 9 4),
    (.virt 19079, .wire 9 5),
    (.wire 9 3, .wire 9 6),
    (.wire 1 43, .wire 9 8),
    (.virt 9496, .wire 9 9),
    (.virt 9496, .wire 9 10),
    (.wire 1 43, .wire 9 12),
    (.virt 19079, .wire 9 13),
    (.wire 9 11, .wire 9 14),
    (.wire 1 43, .wire 9 16),
    (.virt 9497, .wire 9 17),
    (.virt 9497, .wire 9 18),
    (.wire 1 43, .wire 9 20),
    (.virt 19079, .wire 9 21),
    (.wire 9 19, .wire 9 22),
    (.wire 1 43, .wire 9 24),
    (.virt 9498, .wire 9 25),
    (.virt 9498, .wire 9 26),
    (.wire 1 43, .wire 9 28),
    (.virt 19079, .wire 9 29),
    (.wire 9 27, .wire 9 30),
    (.wire 1 43, .wire 9 32)
  ]

/-- `publicBatchWrapper2.copies`, items `512..544`. -/
def publicBatchWrapper2.copies16 : List (Target × Target) := [
    (.virt 9499, .wire 9 33),
    (.virt 9499, .wire 9 34),
    (.wire 1 43, .wire 9 36),
    (.virt 19079, .wire 9 37),
    (.wire 9 35, .wire 9 38),
    (.wire 1 43, .wire 9 40),
    (.virt 9500, .wire 9 41),
    (.virt 9500, .wire 9 42),
    (.wire 1 43, .wire 9 44),
    (.virt 19079, .wire 9 45),
    (.wire 9 43, .wire 9 46),
    (.wire 1 43, .wire 9 48),
    (.virt 9501, .wire 9 49),
    (.virt 9501, .wire 9 50),
    (.wire 1 43, .wire 9 52),
    (.virt 19079, .wire 9 53),
    (.wire 9 51, .wire 9 54),
    (.wire 1 43, .wire 9 56),
    (.virt 9502, .wire 9 57),
    (.virt 9502, .wire 9 58),
    (.wire 1 43, .wire 9 60),
    (.virt 19079, .wire 9 61),
    (.wire 9 59, .wire 9 62),
    (.wire 1 43, .wire 9 64),
    (.virt 9503, .wire 9 65),
    (.virt 9503, .wire 9 66),
    (.wire 1 43, .wire 9 68),
    (.virt 19079, .wire 9 69),
    (.wire 9 67, .wire 9 70),
    (.wire 1 43, .wire 9 72),
    (.virt 9504, .wire 9 73),
    (.virt 9504, .wire 9 74)
  ]

/-- `publicBatchWrapper2.copies`, items `544..576`. -/
def publicBatchWrapper2.copies17 : List (Target × Target) := [
    (.wire 1 43, .wire 9 76),
    (.virt 19079, .wire 9 77),
    (.wire 9 75, .wire 9 78),
    (.wire 1 43, .wire 10 0),
    (.virt 9505, .wire 10 1),
    (.virt 9505, .wire 10 2),
    (.wire 1 43, .wire 10 4),
    (.virt 19079, .wire 10 5),
    (.wire 10 3, .wire 10 6),
    (.wire 1 43, .wire 10 8),
    (.virt 9506, .wire 10 9),
    (.virt 9506, .wire 10 10),
    (.wire 1 43, .wire 10 12),
    (.virt 19079, .wire 10 13),
    (.wire 10 11, .wire 10 14),
    (.wire 1 43, .wire 10 16),
    (.virt 9507, .wire 10 17),
    (.virt 9507, .wire 10 18),
    (.wire 1 43, .wire 10 20),
    (.virt 19079, .wire 10 21),
    (.wire 10 19, .wire 10 22),
    (.wire 1 43, .wire 10 24),
    (.virt 9508, .wire 10 25),
    (.virt 9508, .wire 10 26),
    (.wire 1 43, .wire 10 28),
    (.virt 19079, .wire 10 29),
    (.wire 10 27, .wire 10 30),
    (.wire 1 43, .wire 10 32),
    (.virt 9509, .wire 10 33),
    (.virt 9509, .wire 10 34),
    (.wire 1 43, .wire 10 36),
    (.virt 19079, .wire 10 37)
  ]

/-- `publicBatchWrapper2.copies`, items `576..608`. -/
def publicBatchWrapper2.copies18 : List (Target × Target) := [
    (.wire 10 35, .wire 10 38),
    (.wire 1 43, .wire 10 40),
    (.virt 9510, .wire 10 41),
    (.virt 9510, .wire 10 42),
    (.wire 1 43, .wire 10 44),
    (.virt 19079, .wire 10 45),
    (.wire 10 43, .wire 10 46),
    (.wire 1 43, .wire 10 48),
    (.virt 9511, .wire 10 49),
    (.virt 9511, .wire 10 50),
    (.wire 1 43, .wire 10 52),
    (.virt 19079, .wire 10 53),
    (.wire 10 51, .wire 10 54),
    (.wire 1 43, .wire 10 56),
    (.virt 9512, .wire 10 57),
    (.virt 9512, .wire 10 58),
    (.wire 1 43, .wire 10 60),
    (.virt 19079, .wire 10 61),
    (.wire 10 59, .wire 10 62),
    (.wire 2 7, .wire 10 64),
    (.virt 19030, .wire 10 65),
    (.virt 19030, .wire 10 66),
    (.wire 2 7, .wire 10 68),
    (.virt 19079, .wire 10 69),
    (.wire 10 67, .wire 10 70),
    (.wire 2 7, .wire 10 72),
    (.virt 19031, .wire 10 73),
    (.virt 19031, .wire 10 74),
    (.wire 2 7, .wire 10 76),
    (.virt 19079, .wire 10 77),
    (.wire 10 75, .wire 10 78),
    (.wire 2 7, .wire 11 0)
  ]

/-- `publicBatchWrapper2.copies`, items `608..640`. -/
def publicBatchWrapper2.copies19 : List (Target × Target) := [
    (.virt 19032, .wire 11 1),
    (.virt 19032, .wire 11 2),
    (.wire 2 7, .wire 11 4),
    (.virt 19079, .wire 11 5),
    (.wire 11 3, .wire 11 6),
    (.wire 2 7, .wire 11 8),
    (.virt 19033, .wire 11 9),
    (.virt 19033, .wire 11 10),
    (.wire 2 7, .wire 11 12),
    (.virt 19079, .wire 11 13),
    (.wire 11 11, .wire 11 14),
    (.wire 2 7, .wire 11 16),
    (.virt 19034, .wire 11 17),
    (.virt 19034, .wire 11 18),
    (.wire 2 7, .wire 11 20),
    (.virt 19079, .wire 11 21),
    (.wire 11 19, .wire 11 22),
    (.wire 2 7, .wire 11 24),
    (.virt 19035, .wire 11 25),
    (.virt 19035, .wire 11 26),
    (.wire 2 7, .wire 11 28),
    (.virt 19079, .wire 11 29),
    (.wire 11 27, .wire 11 30),
    (.wire 2 7, .wire 11 32),
    (.virt 19036, .wire 11 33),
    (.virt 19036, .wire 11 34),
    (.wire 2 7, .wire 11 36),
    (.virt 19079, .wire 11 37),
    (.wire 11 35, .wire 11 38),
    (.wire 2 7, .wire 11 40),
    (.virt 19037, .wire 11 41),
    (.virt 19037, .wire 11 42)
  ]

/-- `publicBatchWrapper2.copies`, items `640..672`. -/
def publicBatchWrapper2.copies20 : List (Target × Target) := [
    (.wire 2 7, .wire 11 44),
    (.virt 19079, .wire 11 45),
    (.wire 11 43, .wire 11 46),
    (.wire 2 7, .wire 11 48),
    (.virt 19038, .wire 11 49),
    (.virt 19038, .wire 11 50),
    (.wire 2 7, .wire 11 52),
    (.virt 19079, .wire 11 53),
    (.wire 11 51, .wire 11 54),
    (.wire 2 7, .wire 11 56),
    (.virt 19039, .wire 11 57),
    (.virt 19039, .wire 11 58),
    (.wire 2 7, .wire 11 60),
    (.virt 19079, .wire 11 61),
    (.wire 11 59, .wire 11 62),
    (.wire 2 7, .wire 11 64),
    (.virt 19040, .wire 11 65),
    (.virt 19040, .wire 11 66),
    (.wire 2 7, .wire 11 68),
    (.virt 19079, .wire 11 69),
    (.wire 11 67, .wire 11 70),
    (.wire 2 7, .wire 11 72),
    (.virt 19041, .wire 11 73),
    (.virt 19041, .wire 11 74),
    (.wire 2 7, .wire 11 76),
    (.virt 19079, .wire 11 77),
    (.wire 11 75, .wire 11 78),
    (.wire 2 7, .wire 12 0),
    (.virt 19042, .wire 12 1),
    (.virt 19042, .wire 12 2),
    (.wire 2 7, .wire 12 4),
    (.virt 19079, .wire 12 5)
  ]

/-- `publicBatchWrapper2.copies`, items `672..704`. -/
def publicBatchWrapper2.copies21 : List (Target × Target) := [
    (.wire 12 3, .wire 12 6),
    (.wire 2 7, .wire 12 8),
    (.virt 19043, .wire 12 9),
    (.virt 19043, .wire 12 10),
    (.wire 2 7, .wire 12 12),
    (.virt 19079, .wire 12 13),
    (.wire 12 11, .wire 12 14),
    (.wire 2 7, .wire 12 16),
    (.virt 19044, .wire 12 17),
    (.virt 19044, .wire 12 18),
    (.wire 2 7, .wire 12 20),
    (.virt 19079, .wire 12 21),
    (.wire 12 19, .wire 12 22),
    (.wire 2 7, .wire 12 24),
    (.virt 19045, .wire 12 25),
    (.virt 19045, .wire 12 26),
    (.wire 2 7, .wire 12 28),
    (.virt 19079, .wire 12 29),
    (.wire 12 27, .wire 12 30),
    (.wire 2 7, .wire 12 32),
    (.virt 19046, .wire 12 33),
    (.virt 19046, .wire 12 34),
    (.wire 2 7, .wire 12 36),
    (.virt 19079, .wire 12 37),
    (.wire 12 35, .wire 12 38),
    (.wire 2 7, .wire 12 40),
    (.virt 19047, .wire 12 41),
    (.virt 19047, .wire 12 42),
    (.wire 2 7, .wire 12 44),
    (.virt 19079, .wire 12 45),
    (.wire 12 43, .wire 12 46),
    (.wire 2 7, .wire 12 48)
  ]

/-- `publicBatchWrapper2.copies`, items `704..736`. -/
def publicBatchWrapper2.copies22 : List (Target × Target) := [
    (.virt 19048, .wire 12 49),
    (.virt 19048, .wire 12 50),
    (.wire 2 7, .wire 12 52),
    (.virt 19079, .wire 12 53),
    (.wire 12 51, .wire 12 54),
    (.wire 2 7, .wire 12 56),
    (.virt 19049, .wire 12 57),
    (.virt 19049, .wire 12 58),
    (.wire 2 7, .wire 12 60),
    (.virt 19079, .wire 12 61),
    (.wire 12 59, .wire 12 62),
    (.wire 1 43, .wire 12 64),
    (.virt 9513, .wire 12 65),
    (.virt 9513, .wire 12 66),
    (.wire 1 43, .wire 12 68),
    (.virt 19079, .wire 12 69),
    (.wire 12 67, .wire 12 70),
    (.wire 1 43, .wire 12 72),
    (.virt 9514, .wire 12 73),
    (.virt 9514, .wire 12 74),
    (.wire 1 43, .wire 12 76),
    (.virt 19079, .wire 12 77),
    (.wire 12 75, .wire 12 78),
    (.wire 1 43, .wire 13 0),
    (.virt 9515, .wire 13 1),
    (.virt 9515, .wire 13 2),
    (.wire 1 43, .wire 13 4),
    (.virt 19079, .wire 13 5),
    (.wire 13 3, .wire 13 6),
    (.wire 1 43, .wire 13 8),
    (.virt 9516, .wire 13 9),
    (.virt 9516, .wire 13 10)
  ]

/-- `publicBatchWrapper2.copies`, items `736..768`. -/
def publicBatchWrapper2.copies23 : List (Target × Target) := [
    (.wire 1 43, .wire 13 12),
    (.virt 19079, .wire 13 13),
    (.wire 13 11, .wire 13 14),
    (.wire 1 43, .wire 13 16),
    (.virt 9517, .wire 13 17),
    (.virt 9517, .wire 13 18),
    (.wire 1 43, .wire 13 20),
    (.virt 19079, .wire 13 21),
    (.wire 13 19, .wire 13 22),
    (.wire 1 43, .wire 13 24),
    (.virt 9518, .wire 13 25),
    (.virt 9518, .wire 13 26),
    (.wire 1 43, .wire 13 28),
    (.virt 19079, .wire 13 29),
    (.wire 13 27, .wire 13 30),
    (.wire 1 43, .wire 13 32),
    (.virt 9519, .wire 13 33),
    (.virt 9519, .wire 13 34),
    (.wire 1 43, .wire 13 36),
    (.virt 19079, .wire 13 37),
    (.wire 13 35, .wire 13 38),
    (.wire 1 43, .wire 13 40),
    (.virt 9520, .wire 13 41),
    (.virt 9520, .wire 13 42),
    (.wire 1 43, .wire 13 44),
    (.virt 19079, .wire 13 45),
    (.wire 13 43, .wire 13 46),
    (.wire 2 7, .wire 13 48),
    (.virt 19050, .wire 13 49),
    (.virt 19050, .wire 13 50),
    (.wire 2 7, .wire 13 52),
    (.virt 19079, .wire 13 53)
  ]

/-- `publicBatchWrapper2.copies`, items `768..800`. -/
def publicBatchWrapper2.copies24 : List (Target × Target) := [
    (.wire 13 51, .wire 13 54),
    (.wire 2 7, .wire 13 56),
    (.virt 19051, .wire 13 57),
    (.virt 19051, .wire 13 58),
    (.wire 2 7, .wire 13 60),
    (.virt 19079, .wire 13 61),
    (.wire 13 59, .wire 13 62),
    (.wire 2 7, .wire 13 64),
    (.virt 19052, .wire 13 65),
    (.virt 19052, .wire 13 66),
    (.wire 2 7, .wire 13 68),
    (.virt 19079, .wire 13 69),
    (.wire 13 67, .wire 13 70),
    (.wire 2 7, .wire 13 72),
    (.virt 19053, .wire 13 73),
    (.virt 19053, .wire 13 74),
    (.wire 2 7, .wire 13 76),
    (.virt 19079, .wire 13 77),
    (.wire 13 75, .wire 13 78),
    (.wire 2 7, .wire 14 0),
    (.virt 19054, .wire 14 1),
    (.virt 19054, .wire 14 2),
    (.wire 2 7, .wire 14 4),
    (.virt 19079, .wire 14 5),
    (.wire 14 3, .wire 14 6),
    (.wire 2 7, .wire 14 8),
    (.virt 19055, .wire 14 9),
    (.virt 19055, .wire 14 10),
    (.wire 2 7, .wire 14 12),
    (.virt 19079, .wire 14 13),
    (.wire 14 11, .wire 14 14),
    (.wire 2 7, .wire 14 16)
  ]

/-- `publicBatchWrapper2.copies`, items `800..811`. -/
def publicBatchWrapper2.copies25 : List (Target × Target) := [
    (.virt 19056, .wire 14 17),
    (.virt 19056, .wire 14 18),
    (.wire 2 7, .wire 14 20),
    (.virt 19079, .wire 14 21),
    (.wire 14 19, .wire 14 22),
    (.wire 2 7, .wire 14 24),
    (.virt 19057, .wire 14 25),
    (.virt 19057, .wire 14 26),
    (.wire 2 7, .wire 14 28),
    (.virt 19079, .wire 14 29),
    (.wire 14 27, .wire 14 30)
  ]

/-- The public-batch aggregation wrapper at `n_inner = 2` over `2`-leaf private batches, without the inner verifiers (`wormhole/aggregator/src/public_batch/circuit/circuit_logic.rs`): inner public inputs `inner_pis_0/1`, the `aggregator_address` witness, and the aggregated public inputs. -/
def publicBatchWrapper2 (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 2
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 3
    ⟨.arithmetic 20, [(-1), 1]⟩,  -- row 4
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 5
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 6
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 7
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 8
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 9
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 10
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 11
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 12
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 13
    ⟨.arithmetic 20, [1, (-1)]⟩  -- row 14
  ]
  copies := ((((publicBatchWrapper2.copies0 ++ (publicBatchWrapper2.copies1 ++ publicBatchWrapper2.copies2)) ++ (publicBatchWrapper2.copies3 ++ (publicBatchWrapper2.copies4 ++ publicBatchWrapper2.copies5))) ++ ((publicBatchWrapper2.copies6 ++ (publicBatchWrapper2.copies7 ++ publicBatchWrapper2.copies8)) ++ ((publicBatchWrapper2.copies9 ++ publicBatchWrapper2.copies10) ++ (publicBatchWrapper2.copies11 ++ publicBatchWrapper2.copies12)))) ++ (((publicBatchWrapper2.copies13 ++ (publicBatchWrapper2.copies14 ++ publicBatchWrapper2.copies15)) ++ (publicBatchWrapper2.copies16 ++ (publicBatchWrapper2.copies17 ++ publicBatchWrapper2.copies18))) ++ ((publicBatchWrapper2.copies19 ++ (publicBatchWrapper2.copies20 ++ publicBatchWrapper2.copies21)) ++ ((publicBatchWrapper2.copies22 ++ publicBatchWrapper2.copies23) ++ (publicBatchWrapper2.copies24 ++ publicBatchWrapper2.copies25)))))
  constants := [
    (.virt 19078, 1),
    (.virt 19079, 0),
    (.virt 19120, 8)
  ]
  publicInputs := [
    .virt 19074,
    .virt 19075,
    .virt 19076,
    .virt 19077,
    .wire 3 71,
    .wire 3 79,
    .wire 3 31,
    .wire 3 39,
    .wire 3 47,
    .wire 3 55,
    .wire 3 63,
    .virt 19120,
    .wire 7 71,
    .wire 7 79,
    .wire 9 7,
    .wire 9 15,
    .wire 9 23,
    .wire 9 31,
    .wire 9 39,
    .wire 9 47,
    .wire 9 55,
    .wire 9 63,
    .wire 9 71,
    .wire 9 79,
    .wire 10 7,
    .wire 10 15,
    .wire 10 23,
    .wire 10 31,
    .wire 10 39,
    .wire 10 47,
    .wire 10 55,
    .wire 10 63,
    .wire 10 71,
    .wire 10 79,
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
def publicBatchWrapper2.inner_pis_0 : Fin 52 → Target :=
  ![.virt 9485, .virt 9486, .virt 9487, .virt 9488, .virt 9489, .virt 9490, .virt 9491, .virt 9492, .virt 9493, .virt 9494, .virt 9495, .virt 9496, .virt 9497, .virt 9498, .virt 9499, .virt 9500, .virt 9501, .virt 9502, .virt 9503, .virt 9504, .virt 9505, .virt 9506, .virt 9507, .virt 9508, .virt 9509, .virt 9510, .virt 9511, .virt 9512, .virt 9513, .virt 9514, .virt 9515, .virt 9516, .virt 9517, .virt 9518, .virt 9519, .virt 9520, .virt 9521, .virt 9522, .virt 9523, .virt 9524, .virt 9525, .virt 9526, .virt 9527, .virt 9528, .virt 9529, .virt 9530, .virt 9531, .virt 9532, .virt 9533, .virt 9534, .virt 9535, .virt 9536]

/-- Named targets `inner_pis_1`. -/
def publicBatchWrapper2.inner_pis_1 : Fin 52 → Target :=
  ![.virt 19022, .virt 19023, .virt 19024, .virt 19025, .virt 19026, .virt 19027, .virt 19028, .virt 19029, .virt 19030, .virt 19031, .virt 19032, .virt 19033, .virt 19034, .virt 19035, .virt 19036, .virt 19037, .virt 19038, .virt 19039, .virt 19040, .virt 19041, .virt 19042, .virt 19043, .virt 19044, .virt 19045, .virt 19046, .virt 19047, .virt 19048, .virt 19049, .virt 19050, .virt 19051, .virt 19052, .virt 19053, .virt 19054, .virt 19055, .virt 19056, .virt 19057, .virt 19058, .virt 19059, .virt 19060, .virt 19061, .virt 19062, .virt 19063, .virt 19064, .virt 19065, .virt 19066, .virt 19067, .virt 19068, .virt 19069, .virt 19070, .virt 19071, .virt 19072, .virt 19073]

/-- Named targets `aggregator_address`. -/
def publicBatchWrapper2.aggregator_address : Fin 4 → Target :=
  ![.virt 19074, .virt 19075, .virt 19076, .virt 19077]

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `publicBatchWrapper2` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem publicBatchWrapper2_decode (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    (IsEqual (a (.virt 9488)) (a (.virt 19079)) (a (.virt 19080)) (a (.virt 19081)) ∧
    IsEqual (a (.virt 9489)) (a (.virt 19079)) (a (.virt 19082)) (a (.virt 19083)) ∧
    IsEqual (a (.virt 9490)) (a (.virt 19079)) (a (.virt 19084)) (a (.virt 19085)) ∧
    IsEqual (a (.virt 9491)) (a (.virt 19079)) (a (.virt 19086)) (a (.virt 19087)) ∧
    a (.wire 1 35) = band (a (.virt 19080)) (a (.virt 19082)) ∧
    a (.wire 1 39) = band (a (.virt 19084)) (a (.virt 19086)) ∧
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) ∧
    IsEqual (a (.virt 19025)) (a (.virt 19079)) (a (.virt 19088)) (a (.virt 19089)) ∧
    IsEqual (a (.virt 19026)) (a (.virt 19079)) (a (.virt 19090)) (a (.virt 19091)) ∧
    IsEqual (a (.virt 19027)) (a (.virt 19079)) (a (.virt 19092)) (a (.virt 19093)) ∧
    IsEqual (a (.virt 19028)) (a (.virt 19079)) (a (.virt 19094)) (a (.virt 19095)) ∧
    a (.wire 1 79) = band (a (.virt 19088)) (a (.virt 19090)) ∧
    a (.wire 2 3) = band (a (.virt 19092)) (a (.virt 19094)) ∧
    a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) ∧
    a (.wire 0 67) = bnot (a (.wire 1 43)) ∧
    a (.virt 19078) = bnot (a (.virt 19079)) ∧
    a (.wire 0 67) = band (a (.wire 0 67)) (a (.virt 19078)) ∧
    a (.wire 0 71) = bselect (a (.wire 0 67)) (a (.virt 9488)) (a (.virt 19079)) ∧
    a (.wire 0 75) = bselect (a (.wire 0 67)) (a (.virt 9489)) (a (.virt 19079)) ∧
    a (.wire 0 79) = bselect (a (.wire 0 67)) (a (.virt 9490)) (a (.virt 19079)) ∧
    a (.wire 3 3) = bselect (a (.wire 0 67)) (a (.virt 9491)) (a (.virt 19079)) ∧
    a (.wire 3 7) = bselect (a (.wire 0 67)) (a (.virt 9492)) (a (.virt 19079)) ∧
    a (.wire 3 11) = bselect (a (.wire 0 67)) (a (.virt 9486)) (a (.virt 19079)) ∧
    a (.wire 3 15) = bselect (a (.wire 0 67)) (a (.virt 9487)) (a (.virt 19079)) ∧
    a (.wire 0 67) = bor (a (.virt 19079)) (a (.wire 0 67)) ∧
    a (.wire 3 19) = bnot (a (.wire 2 7)) ∧
    a (.wire 3 23) = bnot (a (.wire 0 67)) ∧
    a (.wire 2 11) = band (a (.wire 3 19)) (a (.wire 3 23)) ∧
    a (.wire 3 31) = bselect (a (.wire 2 11)) (a (.virt 19025)) (a (.wire 0 71)) ∧
    a (.wire 3 39) = bselect (a (.wire 2 11)) (a (.virt 19026)) (a (.wire 0 75)) ∧
    a (.wire 3 47) = bselect (a (.wire 2 11)) (a (.virt 19027)) (a (.wire 0 79)) ∧
    a (.wire 3 55) = bselect (a (.wire 2 11)) (a (.virt 19028)) (a (.wire 3 3))) ∧
    (a (.wire 3 63) = bselect (a (.wire 2 11)) (a (.virt 19029)) (a (.wire 3 7)) ∧
    a (.wire 3 71) = bselect (a (.wire 2 11)) (a (.virt 19023)) (a (.wire 3 11)) ∧
    a (.wire 3 79) = bselect (a (.wire 2 11)) (a (.virt 19024)) (a (.wire 3 15)) ∧
    a (.wire 5 3) = bor (a (.wire 0 67)) (a (.wire 3 19)) ∧
    IsEqual (a (.virt 9486)) (a (.wire 3 71)) (a (.virt 19096)) (a (.virt 19097)) ∧
    a (.wire 5 7) = bor (a (.wire 1 43)) (a (.virt 19096)) ∧
    a (.wire 5 7) = a (.virt 19078) ∧
    IsEqual (a (.virt 9487)) (a (.wire 3 79)) (a (.virt 19098)) (a (.virt 19099)) ∧
    a (.wire 5 11) = bor (a (.wire 1 43)) (a (.virt 19098)) ∧
    a (.wire 5 11) = a (.virt 19078) ∧
    IsEqual (a (.virt 9488)) (a (.wire 3 31)) (a (.virt 19100)) (a (.virt 19101)) ∧
    IsEqual (a (.virt 9489)) (a (.wire 3 39)) (a (.virt 19102)) (a (.virt 19103)) ∧
    IsEqual (a (.virt 9490)) (a (.wire 3 47)) (a (.virt 19104)) (a (.virt 19105)) ∧
    IsEqual (a (.virt 9491)) (a (.wire 3 55)) (a (.virt 19106)) (a (.virt 19107)) ∧
    a (.wire 2 63) = band (a (.virt 19100)) (a (.virt 19102)) ∧
    a (.wire 2 67) = band (a (.virt 19104)) (a (.virt 19106)) ∧
    a (.wire 2 71) = band (a (.wire 2 63)) (a (.wire 2 67)) ∧
    a (.wire 5 15) = bor (a (.wire 1 43)) (a (.wire 2 71)) ∧
    a (.wire 5 15) = a (.virt 19078) ∧
    IsEqual (a (.virt 19023)) (a (.wire 3 71)) (a (.virt 19108)) (a (.virt 19109)) ∧
    a (.wire 5 19) = bor (a (.wire 2 7)) (a (.virt 19108)) ∧
    a (.wire 5 19) = a (.virt 19078) ∧
    IsEqual (a (.virt 19024)) (a (.wire 3 79)) (a (.virt 19110)) (a (.virt 19111)) ∧
    a (.wire 5 23) = bor (a (.wire 2 7)) (a (.virt 19110)) ∧
    a (.wire 5 23) = a (.virt 19078) ∧
    IsEqual (a (.virt 19025)) (a (.wire 3 31)) (a (.virt 19112)) (a (.virt 19113)) ∧
    IsEqual (a (.virt 19026)) (a (.wire 3 39)) (a (.virt 19114)) (a (.virt 19115)) ∧
    IsEqual (a (.virt 19027)) (a (.wire 3 47)) (a (.virt 19116)) (a (.virt 19117)) ∧
    IsEqual (a (.virt 19028)) (a (.wire 3 55)) (a (.virt 19118)) (a (.virt 19119)) ∧
    a (.wire 8 43) = band (a (.virt 19112)) (a (.virt 19114)) ∧
    a (.wire 8 47) = band (a (.virt 19116)) (a (.virt 19118)) ∧
    a (.wire 8 51) = band (a (.wire 8 43)) (a (.wire 8 47))) ∧
    (a (.wire 5 27) = bor (a (.wire 2 7)) (a (.wire 8 51)) ∧
    a (.wire 5 27) = a (.virt 19078) ∧
    a (.wire 7 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9493)) ∧
    a (.wire 7 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9494)) ∧
    a (.wire 9 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9495)) ∧
    a (.wire 9 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9496)) ∧
    a (.wire 9 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9497)) ∧
    a (.wire 9 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9498)) ∧
    a (.wire 9 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9499)) ∧
    a (.wire 9 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9500)) ∧
    a (.wire 9 55) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9501)) ∧
    a (.wire 9 63) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9502)) ∧
    a (.wire 9 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9503)) ∧
    a (.wire 9 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9504)) ∧
    a (.wire 10 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9505)) ∧
    a (.wire 10 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9506)) ∧
    a (.wire 10 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9507)) ∧
    a (.wire 10 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9508)) ∧
    a (.wire 10 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9509)) ∧
    a (.wire 10 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9510)) ∧
    a (.wire 10 55) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9511)) ∧
    a (.wire 10 63) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9512)) ∧
    a (.wire 10 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19030)) ∧
    a (.wire 10 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19031)) ∧
    a (.wire 11 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19032)) ∧
    a (.wire 11 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19033)) ∧
    a (.wire 11 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19034)) ∧
    a (.wire 11 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19035)) ∧
    a (.wire 11 39) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19036)) ∧
    a (.wire 11 47) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19037)) ∧
    a (.wire 11 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19038)) ∧
    a (.wire 11 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19039))) ∧
    (a (.wire 11 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19040)) ∧
    a (.wire 11 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19041)) ∧
    a (.wire 12 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19042)) ∧
    a (.wire 12 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19043)) ∧
    a (.wire 12 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19044)) ∧
    a (.wire 12 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19045)) ∧
    a (.wire 12 39) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19046)) ∧
    a (.wire 12 47) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19047)) ∧
    a (.wire 12 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19048)) ∧
    a (.wire 12 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19049)) ∧
    a (.wire 12 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9513)) ∧
    a (.wire 12 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9514)) ∧
    a (.wire 13 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9515)) ∧
    a (.wire 13 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9516)) ∧
    a (.wire 13 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9517)) ∧
    a (.wire 13 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9518)) ∧
    a (.wire 13 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9519)) ∧
    a (.wire 13 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9520)) ∧
    a (.wire 13 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19050)) ∧
    a (.wire 13 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19051)) ∧
    a (.wire 13 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19052)) ∧
    a (.wire 13 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19053)) ∧
    a (.wire 14 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19054)) ∧
    a (.wire 14 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19055)) ∧
    a (.wire 14 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19056)) ∧
    a (.wire 14 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19057))) := by
  have hcopy := h.2.1
  simp only [publicBatchWrapper2, List.forall_mem_append] at hcopy
  have hcopy0 := hcopy.1.1.1.1
  simp only [publicBatchWrapper2.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy0
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31⟩ := hcopy0
  have hcopy1 := hcopy.1.1.1.2.1
  simp only [publicBatchWrapper2.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy1
  obtain ⟨c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63⟩ := hcopy1
  have hcopy2 := hcopy.1.1.1.2.2
  simp only [publicBatchWrapper2.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy2
  obtain ⟨c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95⟩ := hcopy2
  have hcopy3 := hcopy.1.1.2.1
  simp only [publicBatchWrapper2.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy3
  obtain ⟨c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127⟩ := hcopy3
  have hcopy4 := hcopy.1.1.2.2.1
  simp only [publicBatchWrapper2.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy4
  obtain ⟨c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149, c150, c151, c152, c153, c154, c155, c156, c157, c158, c159⟩ := hcopy4
  have hcopy5 := hcopy.1.1.2.2.2
  simp only [publicBatchWrapper2.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy5
  obtain ⟨c160, c161, c162, c163, c164, c165, c166, c167, c168, c169, c170, c171, c172, c173, c174, c175, c176, c177, c178, c179, c180, c181, c182, c183, c184, c185, c186, c187, c188, c189, c190, c191⟩ := hcopy5
  have hcopy6 := hcopy.1.2.1.1
  simp only [publicBatchWrapper2.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy6
  obtain ⟨c192, c193, c194, c195, c196, c197, c198, c199, c200, c201, c202, c203, c204, c205, c206, c207, c208, c209, c210, c211, c212, c213, c214, c215, c216, c217, c218, c219, c220, c221, c222, c223⟩ := hcopy6
  have hcopy7 := hcopy.1.2.1.2.1
  simp only [publicBatchWrapper2.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy7
  obtain ⟨c224, c225, c226, c227, c228, c229, c230, c231, c232, c233, c234, c235, c236, c237, c238, c239, c240, c241, c242, c243, c244, c245, c246, c247, c248, c249, c250, c251, c252, c253, c254, c255⟩ := hcopy7
  have hcopy8 := hcopy.1.2.1.2.2
  simp only [publicBatchWrapper2.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy8
  obtain ⟨c256, c257, c258, c259, c260, c261, c262, c263, c264, c265, c266, c267, c268, c269, c270, c271, c272, c273, c274, c275, c276, c277, c278, c279, c280, c281, c282, c283, c284, c285, c286, c287⟩ := hcopy8
  have hcopy9 := hcopy.1.2.2.1.1
  simp only [publicBatchWrapper2.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy9
  obtain ⟨c288, c289, c290, c291, c292, c293, c294, c295, c296, c297, c298, c299, c300, c301, c302, c303, c304, c305, c306, c307, c308, c309, c310, c311, c312, c313, c314, c315, c316, c317, c318, c319⟩ := hcopy9
  have hcopy10 := hcopy.1.2.2.1.2
  simp only [publicBatchWrapper2.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy10
  obtain ⟨c320, c321, c322, c323, c324, c325, c326, c327, c328, c329, c330, c331, c332, c333, c334, c335, c336, c337, c338, c339, c340, c341, c342, c343, c344, c345, c346, c347, c348, c349, c350, c351⟩ := hcopy10
  have hcopy11 := hcopy.1.2.2.2.1
  simp only [publicBatchWrapper2.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy11
  obtain ⟨c352, c353, c354, c355, c356, c357, c358, c359, c360, c361, c362, c363, c364, c365, c366, c367, c368, c369, c370, c371, c372, c373, c374, c375, c376, c377, c378, c379, c380, c381, c382, c383⟩ := hcopy11
  have hcopy12 := hcopy.1.2.2.2.2
  simp only [publicBatchWrapper2.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy12
  obtain ⟨c384, c385, c386, c387, c388, c389, c390, c391, c392, c393, c394, c395, c396, c397, c398, c399, c400, c401, c402, c403, c404, c405, c406, c407, c408, c409, c410, c411, c412, c413, c414, c415⟩ := hcopy12
  have hcopy13 := hcopy.2.1.1.1
  simp only [publicBatchWrapper2.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy13
  obtain ⟨c416, c417, c418, c419, c420, c421, c422, c423, c424, c425, c426, c427, c428, c429, c430, c431, c432, c433, c434, c435, c436, c437, c438, c439, c440, c441, c442, c443, c444, c445, c446, c447⟩ := hcopy13
  have hcopy14 := hcopy.2.1.1.2.1
  simp only [publicBatchWrapper2.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy14
  obtain ⟨c448, c449, c450, c451, c452, c453, c454, c455, c456, c457, c458, c459, c460, c461, c462, c463, c464, c465, c466, c467, c468, c469, c470, c471, c472, c473, c474, c475, c476, c477, c478, c479⟩ := hcopy14
  have hcopy15 := hcopy.2.1.1.2.2
  simp only [publicBatchWrapper2.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy15
  obtain ⟨c480, c481, c482, c483, c484, c485, c486, c487, c488, c489, c490, c491, c492, c493, c494, c495, c496, c497, c498, c499, c500, c501, c502, c503, c504, c505, c506, c507, c508, c509, c510, c511⟩ := hcopy15
  have hcopy16 := hcopy.2.1.2.1
  simp only [publicBatchWrapper2.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy16
  obtain ⟨c512, c513, c514, c515, c516, c517, c518, c519, c520, c521, c522, c523, c524, c525, c526, c527, c528, c529, c530, c531, c532, c533, c534, c535, c536, c537, c538, c539, c540, c541, c542, c543⟩ := hcopy16
  have hcopy17 := hcopy.2.1.2.2.1
  simp only [publicBatchWrapper2.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy17
  obtain ⟨c544, c545, c546, c547, c548, c549, c550, c551, c552, c553, c554, c555, c556, c557, c558, c559, c560, c561, c562, c563, c564, c565, c566, c567, c568, c569, c570, c571, c572, c573, c574, c575⟩ := hcopy17
  have hcopy18 := hcopy.2.1.2.2.2
  simp only [publicBatchWrapper2.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy18
  obtain ⟨c576, c577, c578, c579, c580, c581, c582, c583, c584, c585, c586, c587, c588, c589, c590, c591, c592, c593, c594, c595, c596, c597, c598, c599, c600, c601, c602, c603, c604, c605, c606, c607⟩ := hcopy18
  have hcopy19 := hcopy.2.2.1.1
  simp only [publicBatchWrapper2.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy19
  obtain ⟨c608, c609, c610, c611, c612, c613, c614, c615, c616, c617, c618, c619, c620, c621, c622, c623, c624, c625, c626, c627, c628, c629, c630, c631, c632, c633, c634, c635, c636, c637, c638, c639⟩ := hcopy19
  have hcopy20 := hcopy.2.2.1.2.1
  simp only [publicBatchWrapper2.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy20
  obtain ⟨c640, c641, c642, c643, c644, c645, c646, c647, c648, c649, c650, c651, c652, c653, c654, c655, c656, c657, c658, c659, c660, c661, c662, c663, c664, c665, c666, c667, c668, c669, c670, c671⟩ := hcopy20
  have hcopy21 := hcopy.2.2.1.2.2
  simp only [publicBatchWrapper2.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy21
  obtain ⟨c672, c673, c674, c675, c676, c677, c678, c679, c680, c681, c682, c683, c684, c685, c686, c687, c688, c689, c690, c691, c692, c693, c694, c695, c696, c697, c698, c699, c700, c701, c702, c703⟩ := hcopy21
  have hcopy22 := hcopy.2.2.2.1.1
  simp only [publicBatchWrapper2.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy22
  obtain ⟨c704, c705, c706, c707, c708, c709, c710, c711, c712, c713, c714, c715, c716, c717, c718, c719, c720, c721, c722, c723, c724, c725, c726, c727, c728, c729, c730, c731, c732, c733, c734, c735⟩ := hcopy22
  have hcopy23 := hcopy.2.2.2.1.2
  simp only [publicBatchWrapper2.copies23, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy23
  obtain ⟨c736, c737, c738, c739, c740, c741, c742, c743, c744, c745, c746, c747, c748, c749, c750, c751, c752, c753, c754, c755, c756, c757, c758, c759, c760, c761, c762, c763, c764, c765, c766, c767⟩ := hcopy23
  have hcopy24 := hcopy.2.2.2.2.1
  simp only [publicBatchWrapper2.copies24, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy24
  obtain ⟨c768, c769, c770, c771, c772, c773, c774, c775, c776, c777, c778, c779, c780, c781, c782, c783, c784, c785, c786, c787, c788, c789, c790, c791, c792, c793, c794, c795, c796, c797, c798, c799⟩ := hcopy24
  have hcopy25 := hcopy.2.2.2.2.2
  simp only [publicBatchWrapper2.copies25, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy25
  obtain ⟨c800, c801, c802, c803, c804, c805, c806, c807, c808, c809, c810⟩ := hcopy25
  have hconst := h.2.2
  simp only [publicBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2⟩ := hconst
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, k0, ← c1, k0, ← c2] at e_0_0
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_0
  simp only [← c3, ← c4, ← c5] at e_1_0
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_1
  simp only [← c6, ← c7, ← c8] at e_1_1
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_1
  simp only [← c9, ← c10, k0, ← c11] at e_0_1
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_2
  simp only [← c14, k0, ← c15, k0, ← c16] at e_0_2
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_2
  simp only [← c17, ← c18, ← c19] at e_1_2
  have e_1_3 := arithEq_of_rows h (row := 1) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_3
  simp only [← c20, ← c21, ← c22] at e_1_3
  have e_0_3 := arithEq_of_rows h (row := 0) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_3
  simp only [← c23, ← c24, k0, ← c25] at e_0_3
  have e_0_4 := arithEq_of_rows h (row := 0) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_4
  simp only [← c28, k0, ← c29, k0, ← c30] at e_0_4
  have e_1_4 := arithEq_of_rows h (row := 1) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_4
  simp only [← c31, ← c32, ← c33] at e_1_4
  have e_1_5 := arithEq_of_rows h (row := 1) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_5
  simp only [← c34, ← c35, ← c36] at e_1_5
  have e_0_5 := arithEq_of_rows h (row := 0) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_5
  simp only [← c37, ← c38, k0, ← c39] at e_0_5
  have e_0_6 := arithEq_of_rows h (row := 0) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_6
  simp only [← c42, k0, ← c43, k0, ← c44] at e_0_6
  have e_1_6 := arithEq_of_rows h (row := 1) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_6
  simp only [← c45, ← c46, ← c47] at e_1_6
  have e_1_7 := arithEq_of_rows h (row := 1) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_7
  simp only [← c48, ← c49, ← c50] at e_1_7
  have e_0_7 := arithEq_of_rows h (row := 0) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_7
  simp only [← c51, ← c52, k0, ← c53] at e_0_7
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c56, ← c57, ← c58] at e_1_8
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c59, ← c60, ← c61] at e_1_9
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c62, ← c63, ← c64] at e_1_10
  have e_0_8 := arithEq_of_rows h (row := 0) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_8
  simp only [← c65, k0, ← c66, k0, ← c67] at e_0_8
  have e_1_11 := arithEq_of_rows h (row := 1) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_11
  simp only [← c68, ← c69, ← c70] at e_1_11
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c71, ← c72, ← c73] at e_1_12
  have e_0_9 := arithEq_of_rows h (row := 0) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_9
  simp only [← c74, ← c75, k0, ← c76] at e_0_9
  have e_0_10 := arithEq_of_rows h (row := 0) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_10
  simp only [← c79, k0, ← c80, k0, ← c81] at e_0_10
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c82, ← c83, ← c84] at e_1_13
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c85, ← c86, ← c87] at e_1_14
  have e_0_11 := arithEq_of_rows h (row := 0) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_11
  simp only [← c88, ← c89, k0, ← c90] at e_0_11
  have e_0_12 := arithEq_of_rows h (row := 0) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c93, k0, ← c94, k0, ← c95] at e_0_12
  have e_1_15 := arithEq_of_rows h (row := 1) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_15
  simp only [← c96, ← c97, ← c98] at e_1_15
  have e_1_16 := arithEq_of_rows h (row := 1) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_16
  simp only [← c99, ← c100, ← c101] at e_1_16
  have e_0_13 := arithEq_of_rows h (row := 0) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c102, ← c103, k0, ← c104] at e_0_13
  have e_0_14 := arithEq_of_rows h (row := 0) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c107, k0, ← c108, k0, ← c109] at e_0_14
  have e_1_17 := arithEq_of_rows h (row := 1) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_17
  simp only [← c110, ← c111, ← c112] at e_1_17
  have e_1_18 := arithEq_of_rows h (row := 1) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_18
  simp only [← c113, ← c114, ← c115] at e_1_18
  have e_0_15 := arithEq_of_rows h (row := 0) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_15
  simp only [← c116, ← c117, k0, ← c118] at e_0_15
  have e_1_19 := arithEq_of_rows h (row := 1) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_1_19
  simp only [← c121, ← c122, ← c123] at e_1_19
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c124, ← c125, ← c126] at e_2_0
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c127, ← c128, ← c129] at e_2_1
  have e_0_16 := arithEq_of_rows h (row := 0) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_16
  simp only [← c130, k0, ← c131, k0, ← c132] at e_0_16
  have e_0_17 := arithEq_of_rows h (row := 0) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c133, ← c134, ← c135, k1] at e_0_17
  have e_0_18 := arithEq_of_rows h (row := 0) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c136, ← c137, ← c138, k1] at e_0_18
  have e_0_19 := arithEq_of_rows h (row := 0) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c139, ← c140, ← c141, k1] at e_0_19
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c142, ← c143, ← c144, k1] at e_3_0
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c145, ← c146, ← c147, k1] at e_3_1
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c148, ← c149, ← c150, k1] at e_3_2
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c151, ← c152, ← c153, k1] at e_3_3
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c154, k0, ← c155, k0, ← c156] at e_3_4
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c157, k0, ← c158, k0, ← c159] at e_3_5
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c160, ← c161, ← c162] at e_2_2
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c163, ← c164, ← c165] at e_3_6
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c166, ← c167, ← c168] at e_3_7
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c169, ← c170, ← c171] at e_3_8
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c172, ← c173, ← c174] at e_3_9
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c175, ← c176, ← c177] at e_3_10
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c178, ← c179, ← c180] at e_3_11
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c181, ← c182, ← c183] at e_3_12
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c184, ← c185, ← c186] at e_3_13
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c187, ← c188, ← c189] at e_3_14
  have e_3_15 := arithEq_of_rows h (row := 3) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c190, ← c191, ← c192] at e_3_15
  have e_3_16 := arithEq_of_rows h (row := 3) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c193, ← c194, ← c195] at e_3_16
  have e_3_17 := arithEq_of_rows h (row := 3) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c196, ← c197, ← c198] at e_3_17
  have e_3_18 := arithEq_of_rows h (row := 3) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c199, ← c200, ← c201] at e_3_18
  have e_3_19 := arithEq_of_rows h (row := 3) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c202, ← c203, ← c204] at e_3_19
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c205, ← c206, ← c207] at e_4_0
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c208, ← c209, k0, ← c210] at e_5_0
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c211, k0, ← c212, k0, ← c213] at e_6_0
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c214, ← c215, k0, ← c216] at e_6_1
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c217, ← c218, ← c219] at e_2_3
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c220, ← c221, ← c222] at e_2_4
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c223, ← c224, k0, ← c225] at e_6_2
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c228, ← c229, ← c230] at e_4_1
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c231, ← c232, k0, ← c233] at e_5_1
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c235, k0, ← c236, k0, ← c237] at e_6_3
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c238, ← c239, k0, ← c240] at e_6_4
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c241, ← c242, ← c243] at e_2_5
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c244, ← c245, ← c246] at e_2_6
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c247, ← c248, k0, ← c249] at e_6_5
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c252, ← c253, ← c254] at e_4_2
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c255, ← c256, k0, ← c257] at e_5_2
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c259, k0, ← c260, k0, ← c261] at e_6_6
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c262, ← c263, k0, ← c264] at e_6_7
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c265, ← c266, ← c267] at e_2_7
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c268, ← c269, ← c270] at e_2_8
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c271, ← c272, k0, ← c273] at e_6_8
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c276, k0, ← c277, k0, ← c278] at e_6_9
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c279, ← c280, k0, ← c281] at e_6_10
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c282, ← c283, ← c284] at e_2_9
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c285, ← c286, ← c287] at e_2_10
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c288, ← c289, k0, ← c290] at e_6_11
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c293, k0, ← c294, k0, ← c295] at e_6_12
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c296, ← c297, k0, ← c298] at e_6_13
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c299, ← c300, ← c301] at e_2_11
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c302, ← c303, ← c304] at e_2_12
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c305, ← c306, k0, ← c307] at e_6_14
  have e_6_15 := arithEq_of_rows h (row := 6) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_15
  simp only [← c310, k0, ← c311, k0, ← c312] at e_6_15
  have e_6_16 := arithEq_of_rows h (row := 6) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_16
  simp only [← c313, ← c314, k0, ← c315] at e_6_16
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c316, ← c317, ← c318] at e_2_13
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c319, ← c320, ← c321] at e_2_14
  have e_6_17 := arithEq_of_rows h (row := 6) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_17
  simp only [← c322, ← c323, k0, ← c324] at e_6_17
  have e_2_15 := arithEq_of_rows h (row := 2) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c327, ← c328, ← c329] at e_2_15
  have e_2_16 := arithEq_of_rows h (row := 2) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c330, ← c331, ← c332] at e_2_16
  have e_2_17 := arithEq_of_rows h (row := 2) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c333, ← c334, ← c335] at e_2_17
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c336, ← c337, ← c338] at e_4_3
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c339, ← c340, k0, ← c341] at e_5_3
  have e_6_18 := arithEq_of_rows h (row := 6) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_18
  simp only [← c343, k0, ← c344, k0, ← c345] at e_6_18
  have e_6_19 := arithEq_of_rows h (row := 6) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_19
  simp only [← c346, ← c347, k0, ← c348] at e_6_19
  have e_2_18 := arithEq_of_rows h (row := 2) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c349, ← c350, ← c351] at e_2_18
  have e_2_19 := arithEq_of_rows h (row := 2) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c352, ← c353, ← c354] at e_2_19
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c355, ← c356, k0, ← c357] at e_7_0
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c360, ← c361, ← c362] at e_4_4
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c363, ← c364, k0, ← c365] at e_5_4
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c367, k0, ← c368, k0, ← c369] at e_7_1
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c370, ← c371, k0, ← c372] at e_7_2
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c373, ← c374, ← c375] at e_8_0
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c376, ← c377, ← c378] at e_8_1
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c379, ← c380, k0, ← c381] at e_7_3
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c384, ← c385, ← c386] at e_4_5
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c387, ← c388, k0, ← c389] at e_5_5
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c391, k0, ← c392, k0, ← c393] at e_7_4
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c394, ← c395, k0, ← c396] at e_7_5
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c397, ← c398, ← c399] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c400, ← c401, ← c402] at e_8_3
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c403, ← c404, k0, ← c405] at e_7_6
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c408, k0, ← c409, k0, ← c410] at e_7_7
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c411, ← c412, k0, ← c413] at e_7_8
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c414, ← c415, ← c416] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c417, ← c418, ← c419] at e_8_5
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c420, ← c421, k0, ← c422] at e_7_9
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c425, k0, ← c426, k0, ← c427] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c428, ← c429, k0, ← c430] at e_7_11
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c431, ← c432, ← c433] at e_8_6
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c434, ← c435, ← c436] at e_8_7
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c437, ← c438, k0, ← c439] at e_7_12
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c442, k0, ← c443, k0, ← c444] at e_7_13
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c445, ← c446, k0, ← c447] at e_7_14
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c448, ← c449, ← c450] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c451, ← c452, ← c453] at e_8_9
  have e_7_15 := arithEq_of_rows h (row := 7) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_15
  simp only [← c454, ← c455, k0, ← c456] at e_7_15
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c459, ← c460, ← c461] at e_8_10
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c462, ← c463, ← c464] at e_8_11
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c465, ← c466, ← c467] at e_8_12
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c468, ← c469, ← c470] at e_4_6
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c471, ← c472, k0, ← c473] at e_5_6
  have e_7_16 := arithEq_of_rows h (row := 7) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_16
  simp only [← c475, ← c476, ← c477] at e_7_16
  have e_7_17 := arithEq_of_rows h (row := 7) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_17
  simp only [← c478, ← c479, k1, ← c480] at e_7_17
  have e_7_18 := arithEq_of_rows h (row := 7) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_18
  simp only [← c481, ← c482, ← c483] at e_7_18
  have e_7_19 := arithEq_of_rows h (row := 7) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_19
  simp only [← c484, ← c485, k1, ← c486] at e_7_19
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c487, ← c488, ← c489] at e_9_0
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c490, ← c491, k1, ← c492] at e_9_1
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c493, ← c494, ← c495] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c496, ← c497, k1, ← c498] at e_9_3
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c499, ← c500, ← c501] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c502, ← c503, k1, ← c504] at e_9_5
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c505, ← c506, ← c507] at e_9_6
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c508, ← c509, k1, ← c510] at e_9_7
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c511, ← c512, ← c513] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c514, ← c515, k1, ← c516] at e_9_9
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c517, ← c518, ← c519] at e_9_10
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c520, ← c521, k1, ← c522] at e_9_11
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c523, ← c524, ← c525] at e_9_12
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c526, ← c527, k1, ← c528] at e_9_13
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c529, ← c530, ← c531] at e_9_14
  have e_9_15 := arithEq_of_rows h (row := 9) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_15
  simp only [← c532, ← c533, k1, ← c534] at e_9_15
  have e_9_16 := arithEq_of_rows h (row := 9) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_16
  simp only [← c535, ← c536, ← c537] at e_9_16
  have e_9_17 := arithEq_of_rows h (row := 9) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_17
  simp only [← c538, ← c539, k1, ← c540] at e_9_17
  have e_9_18 := arithEq_of_rows h (row := 9) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_18
  simp only [← c541, ← c542, ← c543] at e_9_18
  have e_9_19 := arithEq_of_rows h (row := 9) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_19
  simp only [← c544, ← c545, k1, ← c546] at e_9_19
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c547, ← c548, ← c549] at e_10_0
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c550, ← c551, k1, ← c552] at e_10_1
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c553, ← c554, ← c555] at e_10_2
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c556, ← c557, k1, ← c558] at e_10_3
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c559, ← c560, ← c561] at e_10_4
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c562, ← c563, k1, ← c564] at e_10_5
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c565, ← c566, ← c567] at e_10_6
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c568, ← c569, k1, ← c570] at e_10_7
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c571, ← c572, ← c573] at e_10_8
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c574, ← c575, k1, ← c576] at e_10_9
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c577, ← c578, ← c579] at e_10_10
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c580, ← c581, k1, ← c582] at e_10_11
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c583, ← c584, ← c585] at e_10_12
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c586, ← c587, k1, ← c588] at e_10_13
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c589, ← c590, ← c591] at e_10_14
  have e_10_15 := arithEq_of_rows h (row := 10) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_15
  simp only [← c592, ← c593, k1, ← c594] at e_10_15
  have e_10_16 := arithEq_of_rows h (row := 10) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_16
  simp only [← c595, ← c596, ← c597] at e_10_16
  have e_10_17 := arithEq_of_rows h (row := 10) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_17
  simp only [← c598, ← c599, k1, ← c600] at e_10_17
  have e_10_18 := arithEq_of_rows h (row := 10) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_18
  simp only [← c601, ← c602, ← c603] at e_10_18
  have e_10_19 := arithEq_of_rows h (row := 10) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_19
  simp only [← c604, ← c605, k1, ← c606] at e_10_19
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c607, ← c608, ← c609] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c610, ← c611, k1, ← c612] at e_11_1
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c613, ← c614, ← c615] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c616, ← c617, k1, ← c618] at e_11_3
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c619, ← c620, ← c621] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c622, ← c623, k1, ← c624] at e_11_5
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c625, ← c626, ← c627] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c628, ← c629, k1, ← c630] at e_11_7
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c631, ← c632, ← c633] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c634, ← c635, k1, ← c636] at e_11_9
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c637, ← c638, ← c639] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c640, ← c641, k1, ← c642] at e_11_11
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c643, ← c644, ← c645] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c646, ← c647, k1, ← c648] at e_11_13
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c649, ← c650, ← c651] at e_11_14
  have e_11_15 := arithEq_of_rows h (row := 11) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c652, ← c653, k1, ← c654] at e_11_15
  have e_11_16 := arithEq_of_rows h (row := 11) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c655, ← c656, ← c657] at e_11_16
  have e_11_17 := arithEq_of_rows h (row := 11) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c658, ← c659, k1, ← c660] at e_11_17
  have e_11_18 := arithEq_of_rows h (row := 11) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c661, ← c662, ← c663] at e_11_18
  have e_11_19 := arithEq_of_rows h (row := 11) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c664, ← c665, k1, ← c666] at e_11_19
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c667, ← c668, ← c669] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c670, ← c671, k1, ← c672] at e_12_1
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c673, ← c674, ← c675] at e_12_2
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c676, ← c677, k1, ← c678] at e_12_3
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c679, ← c680, ← c681] at e_12_4
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c682, ← c683, k1, ← c684] at e_12_5
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c685, ← c686, ← c687] at e_12_6
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c688, ← c689, k1, ← c690] at e_12_7
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c691, ← c692, ← c693] at e_12_8
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c694, ← c695, k1, ← c696] at e_12_9
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c697, ← c698, ← c699] at e_12_10
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c700, ← c701, k1, ← c702] at e_12_11
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c703, ← c704, ← c705] at e_12_12
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c706, ← c707, k1, ← c708] at e_12_13
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c709, ← c710, ← c711] at e_12_14
  have e_12_15 := arithEq_of_rows h (row := 12) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c712, ← c713, k1, ← c714] at e_12_15
  have e_12_16 := arithEq_of_rows h (row := 12) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c715, ← c716, ← c717] at e_12_16
  have e_12_17 := arithEq_of_rows h (row := 12) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c718, ← c719, k1, ← c720] at e_12_17
  have e_12_18 := arithEq_of_rows h (row := 12) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c721, ← c722, ← c723] at e_12_18
  have e_12_19 := arithEq_of_rows h (row := 12) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c724, ← c725, k1, ← c726] at e_12_19
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c727, ← c728, ← c729] at e_13_0
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c730, ← c731, k1, ← c732] at e_13_1
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c733, ← c734, ← c735] at e_13_2
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c736, ← c737, k1, ← c738] at e_13_3
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c739, ← c740, ← c741] at e_13_4
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c742, ← c743, k1, ← c744] at e_13_5
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c745, ← c746, ← c747] at e_13_6
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c748, ← c749, k1, ← c750] at e_13_7
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c751, ← c752, ← c753] at e_13_8
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c754, ← c755, k1, ← c756] at e_13_9
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c757, ← c758, ← c759] at e_13_10
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c760, ← c761, k1, ← c762] at e_13_11
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c763, ← c764, ← c765] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c766, ← c767, k1, ← c768] at e_13_13
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c769, ← c770, ← c771] at e_13_14
  have e_13_15 := arithEq_of_rows h (row := 13) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c772, ← c773, k1, ← c774] at e_13_15
  have e_13_16 := arithEq_of_rows h (row := 13) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c775, ← c776, ← c777] at e_13_16
  have e_13_17 := arithEq_of_rows h (row := 13) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c778, ← c779, k1, ← c780] at e_13_17
  have e_13_18 := arithEq_of_rows h (row := 13) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c781, ← c782, ← c783] at e_13_18
  have e_13_19 := arithEq_of_rows h (row := 13) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c784, ← c785, k1, ← c786] at e_13_19
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c787, ← c788, ← c789] at e_14_0
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c790, ← c791, k1, ← c792] at e_14_1
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c793, ← c794, ← c795] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c796, ← c797, k1, ← c798] at e_14_3
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c799, ← c800, ← c801] at e_14_4
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c802, ← c803, k1, ← c804] at e_14_5
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c805, ← c806, ← c807] at e_14_6
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c808, ← c809, k1, ← c810] at e_14_7
  have f0 : IsEqual (a (.virt 9488)) (a (.virt 19079)) (a (.virt 19080)) (a (.virt 19081)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_0
      linear_combination c12.trans k1 - hc
    · have hc := e_0_1
      simp only [e_0_0, e_1_1] at hc
      linear_combination c13.trans k1 - hc
  have f1 : IsEqual (a (.virt 9489)) (a (.virt 19079)) (a (.virt 19082)) (a (.virt 19083)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_2
      linear_combination c26.trans k1 - hc
    · have hc := e_0_3
      simp only [e_0_2, e_1_3] at hc
      linear_combination c27.trans k1 - hc
  have f2 : IsEqual (a (.virt 9490)) (a (.virt 19079)) (a (.virt 19084)) (a (.virt 19085)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_4
      linear_combination c40.trans k1 - hc
    · have hc := e_0_5
      simp only [e_0_4, e_1_5] at hc
      linear_combination c41.trans k1 - hc
  have f3 : IsEqual (a (.virt 9491)) (a (.virt 19079)) (a (.virt 19086)) (a (.virt 19087)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_6
      linear_combination c54.trans k1 - hc
    · have hc := e_0_7
      simp only [e_0_6, e_1_7] at hc
      linear_combination c55.trans k1 - hc
  have f4 : a (.wire 1 35) = band (a (.virt 19080)) (a (.virt 19082)) := by
    have hr := e_1_8
    simp only [band]
    linear_combination hr
  have f5 : a (.wire 1 39) = band (a (.virt 19084)) (a (.virt 19086)) := by
    have hr := e_1_9
    simp only [band]
    linear_combination hr
  have f6 : a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) := by
    have hr := e_1_10
    simp only [band]
    linear_combination hr
  have f7 : IsEqual (a (.virt 19025)) (a (.virt 19079)) (a (.virt 19088)) (a (.virt 19089)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_11
      linear_combination c77.trans k1 - hc
    · have hc := e_0_9
      simp only [e_0_8, e_1_12] at hc
      linear_combination c78.trans k1 - hc
  have f8 : IsEqual (a (.virt 19026)) (a (.virt 19079)) (a (.virt 19090)) (a (.virt 19091)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_13
      linear_combination c91.trans k1 - hc
    · have hc := e_0_11
      simp only [e_0_10, e_1_14] at hc
      linear_combination c92.trans k1 - hc
  have f9 : IsEqual (a (.virt 19027)) (a (.virt 19079)) (a (.virt 19092)) (a (.virt 19093)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_15
      linear_combination c105.trans k1 - hc
    · have hc := e_0_13
      simp only [e_0_12, e_1_16] at hc
      linear_combination c106.trans k1 - hc
  have f10 : IsEqual (a (.virt 19028)) (a (.virt 19079)) (a (.virt 19094)) (a (.virt 19095)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_17
      linear_combination c119.trans k1 - hc
    · have hc := e_0_15
      simp only [e_0_14, e_1_18] at hc
      linear_combination c120.trans k1 - hc
  have f11 : a (.wire 1 79) = band (a (.virt 19088)) (a (.virt 19090)) := by
    have hr := e_1_19
    simp only [band]
    linear_combination hr
  have f12 : a (.wire 2 3) = band (a (.virt 19092)) (a (.virt 19094)) := by
    have hr := e_2_0
    simp only [band]
    linear_combination hr
  have f13 : a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) := by
    have hr := e_2_1
    simp only [band]
    linear_combination hr
  have f14 : a (.wire 0 67) = bnot (a (.wire 1 43)) := by
    have hr := e_0_16
    simp only [bnot]
    linear_combination hr
  have f15 : a (.virt 19078) = bnot (a (.virt 19079)) := by
    simp only [bnot, k0, k1]
    ring
  have f16 : a (.wire 0 67) = band (a (.wire 0 67)) (a (.virt 19078)) := by
    simp only [band, k0]
    ring
  have f17 : a (.wire 0 71) = bselect (a (.wire 0 67)) (a (.virt 9488)) (a (.virt 19079)) := by
    have hr := e_0_17
    simp only [bselect, k1]
    linear_combination hr
  have f18 : a (.wire 0 75) = bselect (a (.wire 0 67)) (a (.virt 9489)) (a (.virt 19079)) := by
    have hr := e_0_18
    simp only [bselect, k1]
    linear_combination hr
  have f19 : a (.wire 0 79) = bselect (a (.wire 0 67)) (a (.virt 9490)) (a (.virt 19079)) := by
    have hr := e_0_19
    simp only [bselect, k1]
    linear_combination hr
  have f20 : a (.wire 3 3) = bselect (a (.wire 0 67)) (a (.virt 9491)) (a (.virt 19079)) := by
    have hr := e_3_0
    simp only [bselect, k1]
    linear_combination hr
  have f21 : a (.wire 3 7) = bselect (a (.wire 0 67)) (a (.virt 9492)) (a (.virt 19079)) := by
    have hr := e_3_1
    simp only [bselect, k1]
    linear_combination hr
  have f22 : a (.wire 3 11) = bselect (a (.wire 0 67)) (a (.virt 9486)) (a (.virt 19079)) := by
    have hr := e_3_2
    simp only [bselect, k1]
    linear_combination hr
  have f23 : a (.wire 3 15) = bselect (a (.wire 0 67)) (a (.virt 9487)) (a (.virt 19079)) := by
    have hr := e_3_3
    simp only [bselect, k1]
    linear_combination hr
  have f24 : a (.wire 0 67) = bor (a (.virt 19079)) (a (.wire 0 67)) := by
    simp only [bor, k1]
    ring
  have f25 : a (.wire 3 19) = bnot (a (.wire 2 7)) := by
    have hr := e_3_4
    simp only [bnot]
    linear_combination hr
  have f26 : a (.wire 3 23) = bnot (a (.wire 0 67)) := by
    have hr := e_3_5
    simp only [bnot]
    linear_combination hr
  have f27 : a (.wire 2 11) = band (a (.wire 3 19)) (a (.wire 3 23)) := by
    have hr := e_2_2
    simp only [band]
    linear_combination hr
  have f28 : a (.wire 3 31) = bselect (a (.wire 2 11)) (a (.virt 19025)) (a (.wire 0 71)) := by
    have hr := e_3_7
    simp only [e_3_6] at hr
    simp only [bselect]
    linear_combination hr
  have f29 : a (.wire 3 39) = bselect (a (.wire 2 11)) (a (.virt 19026)) (a (.wire 0 75)) := by
    have hr := e_3_9
    simp only [e_3_8] at hr
    simp only [bselect]
    linear_combination hr
  have f30 : a (.wire 3 47) = bselect (a (.wire 2 11)) (a (.virt 19027)) (a (.wire 0 79)) := by
    have hr := e_3_11
    simp only [e_3_10] at hr
    simp only [bselect]
    linear_combination hr
  have f31 : a (.wire 3 55) = bselect (a (.wire 2 11)) (a (.virt 19028)) (a (.wire 3 3)) := by
    have hr := e_3_13
    simp only [e_3_12] at hr
    simp only [bselect]
    linear_combination hr
  have f32 : a (.wire 3 63) = bselect (a (.wire 2 11)) (a (.virt 19029)) (a (.wire 3 7)) := by
    have hr := e_3_15
    simp only [e_3_14] at hr
    simp only [bselect]
    linear_combination hr
  have f33 : a (.wire 3 71) = bselect (a (.wire 2 11)) (a (.virt 19023)) (a (.wire 3 11)) := by
    have hr := e_3_17
    simp only [e_3_16] at hr
    simp only [bselect]
    linear_combination hr
  have f34 : a (.wire 3 79) = bselect (a (.wire 2 11)) (a (.virt 19024)) (a (.wire 3 15)) := by
    have hr := e_3_19
    simp only [e_3_18] at hr
    simp only [bselect]
    linear_combination hr
  have f35 : a (.wire 5 3) = bor (a (.wire 0 67)) (a (.wire 3 19)) := by
    have hr := e_5_0
    simp only [e_4_0] at hr
    simp only [bor]
    linear_combination hr
  have f36 : IsEqual (a (.virt 9486)) (a (.wire 3 71)) (a (.virt 19096)) (a (.virt 19097)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_3
      simp only [e_6_1] at hc
      linear_combination c226.trans k1 - hc
    · have hc := e_6_2
      simp only [e_6_0, e_2_4, e_6_1] at hc
      linear_combination c227.trans k1 - hc
  have f37 : a (.wire 5 7) = bor (a (.wire 1 43)) (a (.virt 19096)) := by
    have hr := e_5_1
    simp only [e_4_1] at hr
    simp only [bor]
    linear_combination hr
  have f38 : a (.wire 5 7) = a (.virt 19078) := by
    exact c234
  have f39 : IsEqual (a (.virt 9487)) (a (.wire 3 79)) (a (.virt 19098)) (a (.virt 19099)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_5
      simp only [e_6_4] at hc
      linear_combination c250.trans k1 - hc
    · have hc := e_6_5
      simp only [e_6_3, e_2_6, e_6_4] at hc
      linear_combination c251.trans k1 - hc
  have f40 : a (.wire 5 11) = bor (a (.wire 1 43)) (a (.virt 19098)) := by
    have hr := e_5_2
    simp only [e_4_2] at hr
    simp only [bor]
    linear_combination hr
  have f41 : a (.wire 5 11) = a (.virt 19078) := by
    exact c258
  have f42 : IsEqual (a (.virt 9488)) (a (.wire 3 31)) (a (.virt 19100)) (a (.virt 19101)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_7
      simp only [e_6_7] at hc
      linear_combination c274.trans k1 - hc
    · have hc := e_6_8
      simp only [e_6_6, e_2_8, e_6_7] at hc
      linear_combination c275.trans k1 - hc
  have f43 : IsEqual (a (.virt 9489)) (a (.wire 3 39)) (a (.virt 19102)) (a (.virt 19103)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_9
      simp only [e_6_10] at hc
      linear_combination c291.trans k1 - hc
    · have hc := e_6_11
      simp only [e_6_9, e_2_10, e_6_10] at hc
      linear_combination c292.trans k1 - hc
  have f44 : IsEqual (a (.virt 9490)) (a (.wire 3 47)) (a (.virt 19104)) (a (.virt 19105)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_11
      simp only [e_6_13] at hc
      linear_combination c308.trans k1 - hc
    · have hc := e_6_14
      simp only [e_6_12, e_2_12, e_6_13] at hc
      linear_combination c309.trans k1 - hc
  have f45 : IsEqual (a (.virt 9491)) (a (.wire 3 55)) (a (.virt 19106)) (a (.virt 19107)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_13
      simp only [e_6_16] at hc
      linear_combination c325.trans k1 - hc
    · have hc := e_6_17
      simp only [e_6_15, e_2_14, e_6_16] at hc
      linear_combination c326.trans k1 - hc
  have f46 : a (.wire 2 63) = band (a (.virt 19100)) (a (.virt 19102)) := by
    have hr := e_2_15
    simp only [band]
    linear_combination hr
  have f47 : a (.wire 2 67) = band (a (.virt 19104)) (a (.virt 19106)) := by
    have hr := e_2_16
    simp only [band]
    linear_combination hr
  have f48 : a (.wire 2 71) = band (a (.wire 2 63)) (a (.wire 2 67)) := by
    have hr := e_2_17
    simp only [band]
    linear_combination hr
  have f49 : a (.wire 5 15) = bor (a (.wire 1 43)) (a (.wire 2 71)) := by
    have hr := e_5_3
    simp only [e_4_3] at hr
    simp only [bor]
    linear_combination hr
  have f50 : a (.wire 5 15) = a (.virt 19078) := by
    exact c342
  have f51 : IsEqual (a (.virt 19023)) (a (.wire 3 71)) (a (.virt 19108)) (a (.virt 19109)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_2_18
      simp only [e_6_19] at hc
      linear_combination c358.trans k1 - hc
    · have hc := e_7_0
      simp only [e_6_18, e_2_19, e_6_19] at hc
      linear_combination c359.trans k1 - hc
  have f52 : a (.wire 5 19) = bor (a (.wire 2 7)) (a (.virt 19108)) := by
    have hr := e_5_4
    simp only [e_4_4] at hr
    simp only [bor]
    linear_combination hr
  have f53 : a (.wire 5 19) = a (.virt 19078) := by
    exact c366
  have f54 : IsEqual (a (.virt 19024)) (a (.wire 3 79)) (a (.virt 19110)) (a (.virt 19111)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_8_0
      simp only [e_7_2] at hc
      linear_combination c382.trans k1 - hc
    · have hc := e_7_3
      simp only [e_7_1, e_8_1, e_7_2] at hc
      linear_combination c383.trans k1 - hc
  have f55 : a (.wire 5 23) = bor (a (.wire 2 7)) (a (.virt 19110)) := by
    have hr := e_5_5
    simp only [e_4_5] at hr
    simp only [bor]
    linear_combination hr
  have f56 : a (.wire 5 23) = a (.virt 19078) := by
    exact c390
  have f57 : IsEqual (a (.virt 19025)) (a (.wire 3 31)) (a (.virt 19112)) (a (.virt 19113)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_8_2
      simp only [e_7_5] at hc
      linear_combination c406.trans k1 - hc
    · have hc := e_7_6
      simp only [e_7_4, e_8_3, e_7_5] at hc
      linear_combination c407.trans k1 - hc
  have f58 : IsEqual (a (.virt 19026)) (a (.wire 3 39)) (a (.virt 19114)) (a (.virt 19115)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_8_4
      simp only [e_7_8] at hc
      linear_combination c423.trans k1 - hc
    · have hc := e_7_9
      simp only [e_7_7, e_8_5, e_7_8] at hc
      linear_combination c424.trans k1 - hc
  have f59 : IsEqual (a (.virt 19027)) (a (.wire 3 47)) (a (.virt 19116)) (a (.virt 19117)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_8_6
      simp only [e_7_11] at hc
      linear_combination c440.trans k1 - hc
    · have hc := e_7_12
      simp only [e_7_10, e_8_7, e_7_11] at hc
      linear_combination c441.trans k1 - hc
  have f60 : IsEqual (a (.virt 19028)) (a (.wire 3 55)) (a (.virt 19118)) (a (.virt 19119)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_8_8
      simp only [e_7_14] at hc
      linear_combination c457.trans k1 - hc
    · have hc := e_7_15
      simp only [e_7_13, e_8_9, e_7_14] at hc
      linear_combination c458.trans k1 - hc
  have f61 : a (.wire 8 43) = band (a (.virt 19112)) (a (.virt 19114)) := by
    have hr := e_8_10
    simp only [band]
    linear_combination hr
  have f62 : a (.wire 8 47) = band (a (.virt 19116)) (a (.virt 19118)) := by
    have hr := e_8_11
    simp only [band]
    linear_combination hr
  have f63 : a (.wire 8 51) = band (a (.wire 8 43)) (a (.wire 8 47)) := by
    have hr := e_8_12
    simp only [band]
    linear_combination hr
  have f64 : a (.wire 5 27) = bor (a (.wire 2 7)) (a (.wire 8 51)) := by
    have hr := e_5_6
    simp only [e_4_6] at hr
    simp only [bor]
    linear_combination hr
  have f65 : a (.wire 5 27) = a (.virt 19078) := by
    exact c474
  have f66 : a (.wire 7 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9493)) := by
    have hr := e_7_17
    simp only [e_7_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f67 : a (.wire 7 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9494)) := by
    have hr := e_7_19
    simp only [e_7_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f68 : a (.wire 9 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9495)) := by
    have hr := e_9_1
    simp only [e_9_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f69 : a (.wire 9 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9496)) := by
    have hr := e_9_3
    simp only [e_9_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f70 : a (.wire 9 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9497)) := by
    have hr := e_9_5
    simp only [e_9_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f71 : a (.wire 9 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9498)) := by
    have hr := e_9_7
    simp only [e_9_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f72 : a (.wire 9 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9499)) := by
    have hr := e_9_9
    simp only [e_9_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f73 : a (.wire 9 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9500)) := by
    have hr := e_9_11
    simp only [e_9_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f74 : a (.wire 9 55) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9501)) := by
    have hr := e_9_13
    simp only [e_9_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f75 : a (.wire 9 63) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9502)) := by
    have hr := e_9_15
    simp only [e_9_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f76 : a (.wire 9 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9503)) := by
    have hr := e_9_17
    simp only [e_9_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f77 : a (.wire 9 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9504)) := by
    have hr := e_9_19
    simp only [e_9_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f78 : a (.wire 10 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9505)) := by
    have hr := e_10_1
    simp only [e_10_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f79 : a (.wire 10 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9506)) := by
    have hr := e_10_3
    simp only [e_10_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f80 : a (.wire 10 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9507)) := by
    have hr := e_10_5
    simp only [e_10_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f81 : a (.wire 10 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9508)) := by
    have hr := e_10_7
    simp only [e_10_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f82 : a (.wire 10 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9509)) := by
    have hr := e_10_9
    simp only [e_10_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f83 : a (.wire 10 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9510)) := by
    have hr := e_10_11
    simp only [e_10_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f84 : a (.wire 10 55) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9511)) := by
    have hr := e_10_13
    simp only [e_10_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f85 : a (.wire 10 63) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9512)) := by
    have hr := e_10_15
    simp only [e_10_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f86 : a (.wire 10 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19030)) := by
    have hr := e_10_17
    simp only [e_10_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f87 : a (.wire 10 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19031)) := by
    have hr := e_10_19
    simp only [e_10_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f88 : a (.wire 11 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19032)) := by
    have hr := e_11_1
    simp only [e_11_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f89 : a (.wire 11 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19033)) := by
    have hr := e_11_3
    simp only [e_11_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f90 : a (.wire 11 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19034)) := by
    have hr := e_11_5
    simp only [e_11_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f91 : a (.wire 11 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19035)) := by
    have hr := e_11_7
    simp only [e_11_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f92 : a (.wire 11 39) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19036)) := by
    have hr := e_11_9
    simp only [e_11_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f93 : a (.wire 11 47) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19037)) := by
    have hr := e_11_11
    simp only [e_11_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f94 : a (.wire 11 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19038)) := by
    have hr := e_11_13
    simp only [e_11_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f95 : a (.wire 11 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19039)) := by
    have hr := e_11_15
    simp only [e_11_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f96 : a (.wire 11 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19040)) := by
    have hr := e_11_17
    simp only [e_11_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f97 : a (.wire 11 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19041)) := by
    have hr := e_11_19
    simp only [e_11_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f98 : a (.wire 12 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19042)) := by
    have hr := e_12_1
    simp only [e_12_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f99 : a (.wire 12 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19043)) := by
    have hr := e_12_3
    simp only [e_12_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f100 : a (.wire 12 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19044)) := by
    have hr := e_12_5
    simp only [e_12_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f101 : a (.wire 12 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19045)) := by
    have hr := e_12_7
    simp only [e_12_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f102 : a (.wire 12 39) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19046)) := by
    have hr := e_12_9
    simp only [e_12_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f103 : a (.wire 12 47) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19047)) := by
    have hr := e_12_11
    simp only [e_12_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f104 : a (.wire 12 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19048)) := by
    have hr := e_12_13
    simp only [e_12_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f105 : a (.wire 12 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19049)) := by
    have hr := e_12_15
    simp only [e_12_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f106 : a (.wire 12 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9513)) := by
    have hr := e_12_17
    simp only [e_12_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f107 : a (.wire 12 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9514)) := by
    have hr := e_12_19
    simp only [e_12_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f108 : a (.wire 13 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9515)) := by
    have hr := e_13_1
    simp only [e_13_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f109 : a (.wire 13 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9516)) := by
    have hr := e_13_3
    simp only [e_13_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f110 : a (.wire 13 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9517)) := by
    have hr := e_13_5
    simp only [e_13_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f111 : a (.wire 13 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9518)) := by
    have hr := e_13_7
    simp only [e_13_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f112 : a (.wire 13 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9519)) := by
    have hr := e_13_9
    simp only [e_13_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f113 : a (.wire 13 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9520)) := by
    have hr := e_13_11
    simp only [e_13_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f114 : a (.wire 13 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19050)) := by
    have hr := e_13_13
    simp only [e_13_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f115 : a (.wire 13 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19051)) := by
    have hr := e_13_15
    simp only [e_13_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f116 : a (.wire 13 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19052)) := by
    have hr := e_13_17
    simp only [e_13_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f117 : a (.wire 13 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19053)) := by
    have hr := e_13_19
    simp only [e_13_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f118 : a (.wire 14 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19054)) := by
    have hr := e_14_1
    simp only [e_14_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f119 : a (.wire 14 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19055)) := by
    have hr := e_14_3
    simp only [e_14_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f120 : a (.wire 14 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19056)) := by
    have hr := e_14_5
    simp only [e_14_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f121 : a (.wire 14 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19057)) := by
    have hr := e_14_7
    simp only [e_14_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  exact ⟨⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31⟩, ⟨f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60, f61, f62, f63⟩, ⟨f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75, f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90, f91, f92, f93, f94, f95⟩, ⟨f96, f97, f98, f99, f100, f101, f102, f103, f104, f105, f106, f107, f108, f109, f110, f111, f112, f113, f114, f115, f116, f117, f118, f119, f120, f121⟩⟩

end Plonky2Spec.Generated
