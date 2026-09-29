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

/-- The public-batch aggregation wrapper at `n_inner = 2` over `2`-leaf private batches, without the inner verifiers (`wormhole/aggregator/src/public_batch/circuit/circuit_logic.rs`): inner public inputs `inner_pis_0..1`, the `aggregator_address` witness, and the aggregated public inputs. -/
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

theorem publicBatchWrapper2_copies0 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19078) = a (.wire 0 0) ∧ a (.virt 19078) = a (.wire 0 1) ∧ a (.virt 19080) = a (.wire 0 2) ∧ a (.virt 19080) = a (.wire 1 0) ∧ a (.virt 9488) = a (.wire 1 1) ∧ a (.virt 19080) = a (.wire 1 2) ∧ a (.virt 9488) = a (.wire 1 4) ∧ a (.virt 19081) = a (.wire 1 5) ∧ a (.virt 9488) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 4) ∧ a (.virt 19078) = a (.wire 0 5) ∧ a (.wire 0 3) = a (.wire 0 6) ∧ a (.wire 1 3) = a (.virt 19079) ∧ a (.wire 0 7) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 0 8) ∧ a (.virt 19078) = a (.wire 0 9) ∧ a (.virt 19082) = a (.wire 0 10) ∧ a (.virt 19082) = a (.wire 1 8) ∧ a (.virt 9489) = a (.wire 1 9) ∧ a (.virt 19082) = a (.wire 1 10) ∧ a (.virt 9489) = a (.wire 1 12) ∧ a (.virt 19083) = a (.wire 1 13) ∧ a (.virt 9489) = a (.wire 1 14) ∧ a (.wire 1 15) = a (.wire 0 12) ∧ a (.virt 19078) = a (.wire 0 13) ∧ a (.wire 0 11) = a (.wire 0 14) ∧ a (.wire 1 11) = a (.virt 19079) ∧ a (.wire 0 15) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 0 16) ∧ a (.virt 19078) = a (.wire 0 17) ∧ a (.virt 19084) = a (.wire 0 18) ∧ a (.virt 19084) = a (.wire 1 16) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies1 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 9490) = a (.wire 1 17) ∧ a (.virt 19084) = a (.wire 1 18) ∧ a (.virt 9490) = a (.wire 1 20) ∧ a (.virt 19085) = a (.wire 1 21) ∧ a (.virt 9490) = a (.wire 1 22) ∧ a (.wire 1 23) = a (.wire 0 20) ∧ a (.virt 19078) = a (.wire 0 21) ∧ a (.wire 0 19) = a (.wire 0 22) ∧ a (.wire 1 19) = a (.virt 19079) ∧ a (.wire 0 23) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 0 24) ∧ a (.virt 19078) = a (.wire 0 25) ∧ a (.virt 19086) = a (.wire 0 26) ∧ a (.virt 19086) = a (.wire 1 24) ∧ a (.virt 9491) = a (.wire 1 25) ∧ a (.virt 19086) = a (.wire 1 26) ∧ a (.virt 9491) = a (.wire 1 28) ∧ a (.virt 19087) = a (.wire 1 29) ∧ a (.virt 9491) = a (.wire 1 30) ∧ a (.wire 1 31) = a (.wire 0 28) ∧ a (.virt 19078) = a (.wire 0 29) ∧ a (.wire 0 27) = a (.wire 0 30) ∧ a (.wire 1 27) = a (.virt 19079) ∧ a (.wire 0 31) = a (.virt 19079) ∧ a (.virt 19080) = a (.wire 1 32) ∧ a (.virt 19082) = a (.wire 1 33) ∧ a (.virt 19080) = a (.wire 1 34) ∧ a (.virt 19084) = a (.wire 1 36) ∧ a (.virt 19086) = a (.wire 1 37) ∧ a (.virt 19084) = a (.wire 1 38) ∧ a (.wire 1 35) = a (.wire 1 40) ∧ a (.wire 1 39) = a (.wire 1 41) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies2 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 35) = a (.wire 1 42) ∧ a (.virt 19078) = a (.wire 0 32) ∧ a (.virt 19078) = a (.wire 0 33) ∧ a (.virt 19088) = a (.wire 0 34) ∧ a (.virt 19088) = a (.wire 1 44) ∧ a (.virt 19025) = a (.wire 1 45) ∧ a (.virt 19088) = a (.wire 1 46) ∧ a (.virt 19025) = a (.wire 1 48) ∧ a (.virt 19089) = a (.wire 1 49) ∧ a (.virt 19025) = a (.wire 1 50) ∧ a (.wire 1 51) = a (.wire 0 36) ∧ a (.virt 19078) = a (.wire 0 37) ∧ a (.wire 0 35) = a (.wire 0 38) ∧ a (.wire 1 47) = a (.virt 19079) ∧ a (.wire 0 39) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 0 40) ∧ a (.virt 19078) = a (.wire 0 41) ∧ a (.virt 19090) = a (.wire 0 42) ∧ a (.virt 19090) = a (.wire 1 52) ∧ a (.virt 19026) = a (.wire 1 53) ∧ a (.virt 19090) = a (.wire 1 54) ∧ a (.virt 19026) = a (.wire 1 56) ∧ a (.virt 19091) = a (.wire 1 57) ∧ a (.virt 19026) = a (.wire 1 58) ∧ a (.wire 1 59) = a (.wire 0 44) ∧ a (.virt 19078) = a (.wire 0 45) ∧ a (.wire 0 43) = a (.wire 0 46) ∧ a (.wire 1 55) = a (.virt 19079) ∧ a (.wire 0 47) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 0 48) ∧ a (.virt 19078) = a (.wire 0 49) ∧ a (.virt 19092) = a (.wire 0 50) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies3 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19092) = a (.wire 1 60) ∧ a (.virt 19027) = a (.wire 1 61) ∧ a (.virt 19092) = a (.wire 1 62) ∧ a (.virt 19027) = a (.wire 1 64) ∧ a (.virt 19093) = a (.wire 1 65) ∧ a (.virt 19027) = a (.wire 1 66) ∧ a (.wire 1 67) = a (.wire 0 52) ∧ a (.virt 19078) = a (.wire 0 53) ∧ a (.wire 0 51) = a (.wire 0 54) ∧ a (.wire 1 63) = a (.virt 19079) ∧ a (.wire 0 55) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 0 56) ∧ a (.virt 19078) = a (.wire 0 57) ∧ a (.virt 19094) = a (.wire 0 58) ∧ a (.virt 19094) = a (.wire 1 68) ∧ a (.virt 19028) = a (.wire 1 69) ∧ a (.virt 19094) = a (.wire 1 70) ∧ a (.virt 19028) = a (.wire 1 72) ∧ a (.virt 19095) = a (.wire 1 73) ∧ a (.virt 19028) = a (.wire 1 74) ∧ a (.wire 1 75) = a (.wire 0 60) ∧ a (.virt 19078) = a (.wire 0 61) ∧ a (.wire 0 59) = a (.wire 0 62) ∧ a (.wire 1 71) = a (.virt 19079) ∧ a (.wire 0 63) = a (.virt 19079) ∧ a (.virt 19088) = a (.wire 1 76) ∧ a (.virt 19090) = a (.wire 1 77) ∧ a (.virt 19088) = a (.wire 1 78) ∧ a (.virt 19092) = a (.wire 2 0) ∧ a (.virt 19094) = a (.wire 2 1) ∧ a (.virt 19092) = a (.wire 2 2) ∧ a (.wire 1 79) = a (.wire 2 4) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies3, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies4 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 3) = a (.wire 2 5) ∧ a (.wire 1 79) = a (.wire 2 6) ∧ a (.virt 19078) = a (.wire 0 64) ∧ a (.virt 19078) = a (.wire 0 65) ∧ a (.wire 1 43) = a (.wire 0 66) ∧ a (.wire 0 67) = a (.wire 0 68) ∧ a (.virt 9488) = a (.wire 0 69) ∧ a (.virt 19079) = a (.wire 0 70) ∧ a (.wire 0 67) = a (.wire 0 72) ∧ a (.virt 9489) = a (.wire 0 73) ∧ a (.virt 19079) = a (.wire 0 74) ∧ a (.wire 0 67) = a (.wire 0 76) ∧ a (.virt 9490) = a (.wire 0 77) ∧ a (.virt 19079) = a (.wire 0 78) ∧ a (.wire 0 67) = a (.wire 3 0) ∧ a (.virt 9491) = a (.wire 3 1) ∧ a (.virt 19079) = a (.wire 3 2) ∧ a (.wire 0 67) = a (.wire 3 4) ∧ a (.virt 9492) = a (.wire 3 5) ∧ a (.virt 19079) = a (.wire 3 6) ∧ a (.wire 0 67) = a (.wire 3 8) ∧ a (.virt 9486) = a (.wire 3 9) ∧ a (.virt 19079) = a (.wire 3 10) ∧ a (.wire 0 67) = a (.wire 3 12) ∧ a (.virt 9487) = a (.wire 3 13) ∧ a (.virt 19079) = a (.wire 3 14) ∧ a (.virt 19078) = a (.wire 3 16) ∧ a (.virt 19078) = a (.wire 3 17) ∧ a (.wire 2 7) = a (.wire 3 18) ∧ a (.virt 19078) = a (.wire 3 20) ∧ a (.virt 19078) = a (.wire 3 21) ∧ a (.wire 0 67) = a (.wire 3 22) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies4, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies5 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 19) = a (.wire 2 8) ∧ a (.wire 3 23) = a (.wire 2 9) ∧ a (.wire 3 19) = a (.wire 2 10) ∧ a (.wire 2 11) = a (.wire 3 24) ∧ a (.wire 0 71) = a (.wire 3 25) ∧ a (.wire 0 71) = a (.wire 3 26) ∧ a (.wire 2 11) = a (.wire 3 28) ∧ a (.virt 19025) = a (.wire 3 29) ∧ a (.wire 3 27) = a (.wire 3 30) ∧ a (.wire 2 11) = a (.wire 3 32) ∧ a (.wire 0 75) = a (.wire 3 33) ∧ a (.wire 0 75) = a (.wire 3 34) ∧ a (.wire 2 11) = a (.wire 3 36) ∧ a (.virt 19026) = a (.wire 3 37) ∧ a (.wire 3 35) = a (.wire 3 38) ∧ a (.wire 2 11) = a (.wire 3 40) ∧ a (.wire 0 79) = a (.wire 3 41) ∧ a (.wire 0 79) = a (.wire 3 42) ∧ a (.wire 2 11) = a (.wire 3 44) ∧ a (.virt 19027) = a (.wire 3 45) ∧ a (.wire 3 43) = a (.wire 3 46) ∧ a (.wire 2 11) = a (.wire 3 48) ∧ a (.wire 3 3) = a (.wire 3 49) ∧ a (.wire 3 3) = a (.wire 3 50) ∧ a (.wire 2 11) = a (.wire 3 52) ∧ a (.virt 19028) = a (.wire 3 53) ∧ a (.wire 3 51) = a (.wire 3 54) ∧ a (.wire 2 11) = a (.wire 3 56) ∧ a (.wire 3 7) = a (.wire 3 57) ∧ a (.wire 3 7) = a (.wire 3 58) ∧ a (.wire 2 11) = a (.wire 3 60) ∧ a (.virt 19029) = a (.wire 3 61) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies5, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies6 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 59) = a (.wire 3 62) ∧ a (.wire 2 11) = a (.wire 3 64) ∧ a (.wire 3 11) = a (.wire 3 65) ∧ a (.wire 3 11) = a (.wire 3 66) ∧ a (.wire 2 11) = a (.wire 3 68) ∧ a (.virt 19023) = a (.wire 3 69) ∧ a (.wire 3 67) = a (.wire 3 70) ∧ a (.wire 2 11) = a (.wire 3 72) ∧ a (.wire 3 15) = a (.wire 3 73) ∧ a (.wire 3 15) = a (.wire 3 74) ∧ a (.wire 2 11) = a (.wire 3 76) ∧ a (.virt 19024) = a (.wire 3 77) ∧ a (.wire 3 75) = a (.wire 3 78) ∧ a (.wire 0 67) = a (.wire 4 0) ∧ a (.wire 3 19) = a (.wire 4 1) ∧ a (.wire 0 67) = a (.wire 4 2) ∧ a (.wire 4 3) = a (.wire 5 0) ∧ a (.virt 19078) = a (.wire 5 1) ∧ a (.wire 3 19) = a (.wire 5 2) ∧ a (.virt 19078) = a (.wire 6 0) ∧ a (.virt 19078) = a (.wire 6 1) ∧ a (.virt 19096) = a (.wire 6 2) ∧ a (.virt 9486) = a (.wire 6 4) ∧ a (.virt 19078) = a (.wire 6 5) ∧ a (.wire 3 71) = a (.wire 6 6) ∧ a (.virt 19096) = a (.wire 2 12) ∧ a (.wire 6 7) = a (.wire 2 13) ∧ a (.virt 19096) = a (.wire 2 14) ∧ a (.wire 6 7) = a (.wire 2 16) ∧ a (.virt 19097) = a (.wire 2 17) ∧ a (.wire 6 7) = a (.wire 2 18) ∧ a (.wire 2 19) = a (.wire 6 8) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies6, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies7 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19078) = a (.wire 6 9) ∧ a (.wire 6 3) = a (.wire 6 10) ∧ a (.wire 2 15) = a (.virt 19079) ∧ a (.wire 6 11) = a (.virt 19079) ∧ a (.wire 1 43) = a (.wire 4 4) ∧ a (.virt 19096) = a (.wire 4 5) ∧ a (.wire 1 43) = a (.wire 4 6) ∧ a (.wire 4 7) = a (.wire 5 4) ∧ a (.virt 19078) = a (.wire 5 5) ∧ a (.virt 19096) = a (.wire 5 6) ∧ a (.wire 5 7) = a (.virt 19078) ∧ a (.virt 19078) = a (.wire 6 12) ∧ a (.virt 19078) = a (.wire 6 13) ∧ a (.virt 19098) = a (.wire 6 14) ∧ a (.virt 9487) = a (.wire 6 16) ∧ a (.virt 19078) = a (.wire 6 17) ∧ a (.wire 3 79) = a (.wire 6 18) ∧ a (.virt 19098) = a (.wire 2 20) ∧ a (.wire 6 19) = a (.wire 2 21) ∧ a (.virt 19098) = a (.wire 2 22) ∧ a (.wire 6 19) = a (.wire 2 24) ∧ a (.virt 19099) = a (.wire 2 25) ∧ a (.wire 6 19) = a (.wire 2 26) ∧ a (.wire 2 27) = a (.wire 6 20) ∧ a (.virt 19078) = a (.wire 6 21) ∧ a (.wire 6 15) = a (.wire 6 22) ∧ a (.wire 2 23) = a (.virt 19079) ∧ a (.wire 6 23) = a (.virt 19079) ∧ a (.wire 1 43) = a (.wire 4 8) ∧ a (.virt 19098) = a (.wire 4 9) ∧ a (.wire 1 43) = a (.wire 4 10) ∧ a (.wire 4 11) = a (.wire 5 8) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies7, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies8 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19078) = a (.wire 5 9) ∧ a (.virt 19098) = a (.wire 5 10) ∧ a (.wire 5 11) = a (.virt 19078) ∧ a (.virt 19078) = a (.wire 6 24) ∧ a (.virt 19078) = a (.wire 6 25) ∧ a (.virt 19100) = a (.wire 6 26) ∧ a (.virt 9488) = a (.wire 6 28) ∧ a (.virt 19078) = a (.wire 6 29) ∧ a (.wire 3 31) = a (.wire 6 30) ∧ a (.virt 19100) = a (.wire 2 28) ∧ a (.wire 6 31) = a (.wire 2 29) ∧ a (.virt 19100) = a (.wire 2 30) ∧ a (.wire 6 31) = a (.wire 2 32) ∧ a (.virt 19101) = a (.wire 2 33) ∧ a (.wire 6 31) = a (.wire 2 34) ∧ a (.wire 2 35) = a (.wire 6 32) ∧ a (.virt 19078) = a (.wire 6 33) ∧ a (.wire 6 27) = a (.wire 6 34) ∧ a (.wire 2 31) = a (.virt 19079) ∧ a (.wire 6 35) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 6 36) ∧ a (.virt 19078) = a (.wire 6 37) ∧ a (.virt 19102) = a (.wire 6 38) ∧ a (.virt 9489) = a (.wire 6 40) ∧ a (.virt 19078) = a (.wire 6 41) ∧ a (.wire 3 39) = a (.wire 6 42) ∧ a (.virt 19102) = a (.wire 2 36) ∧ a (.wire 6 43) = a (.wire 2 37) ∧ a (.virt 19102) = a (.wire 2 38) ∧ a (.wire 6 43) = a (.wire 2 40) ∧ a (.virt 19103) = a (.wire 2 41) ∧ a (.wire 6 43) = a (.wire 2 42) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies8, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies9 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 43) = a (.wire 6 44) ∧ a (.virt 19078) = a (.wire 6 45) ∧ a (.wire 6 39) = a (.wire 6 46) ∧ a (.wire 2 39) = a (.virt 19079) ∧ a (.wire 6 47) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 6 48) ∧ a (.virt 19078) = a (.wire 6 49) ∧ a (.virt 19104) = a (.wire 6 50) ∧ a (.virt 9490) = a (.wire 6 52) ∧ a (.virt 19078) = a (.wire 6 53) ∧ a (.wire 3 47) = a (.wire 6 54) ∧ a (.virt 19104) = a (.wire 2 44) ∧ a (.wire 6 55) = a (.wire 2 45) ∧ a (.virt 19104) = a (.wire 2 46) ∧ a (.wire 6 55) = a (.wire 2 48) ∧ a (.virt 19105) = a (.wire 2 49) ∧ a (.wire 6 55) = a (.wire 2 50) ∧ a (.wire 2 51) = a (.wire 6 56) ∧ a (.virt 19078) = a (.wire 6 57) ∧ a (.wire 6 51) = a (.wire 6 58) ∧ a (.wire 2 47) = a (.virt 19079) ∧ a (.wire 6 59) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 6 60) ∧ a (.virt 19078) = a (.wire 6 61) ∧ a (.virt 19106) = a (.wire 6 62) ∧ a (.virt 9491) = a (.wire 6 64) ∧ a (.virt 19078) = a (.wire 6 65) ∧ a (.wire 3 55) = a (.wire 6 66) ∧ a (.virt 19106) = a (.wire 2 52) ∧ a (.wire 6 67) = a (.wire 2 53) ∧ a (.virt 19106) = a (.wire 2 54) ∧ a (.wire 6 67) = a (.wire 2 56) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies9, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies10 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19107) = a (.wire 2 57) ∧ a (.wire 6 67) = a (.wire 2 58) ∧ a (.wire 2 59) = a (.wire 6 68) ∧ a (.virt 19078) = a (.wire 6 69) ∧ a (.wire 6 63) = a (.wire 6 70) ∧ a (.wire 2 55) = a (.virt 19079) ∧ a (.wire 6 71) = a (.virt 19079) ∧ a (.virt 19100) = a (.wire 2 60) ∧ a (.virt 19102) = a (.wire 2 61) ∧ a (.virt 19100) = a (.wire 2 62) ∧ a (.virt 19104) = a (.wire 2 64) ∧ a (.virt 19106) = a (.wire 2 65) ∧ a (.virt 19104) = a (.wire 2 66) ∧ a (.wire 2 63) = a (.wire 2 68) ∧ a (.wire 2 67) = a (.wire 2 69) ∧ a (.wire 2 63) = a (.wire 2 70) ∧ a (.wire 1 43) = a (.wire 4 12) ∧ a (.wire 2 71) = a (.wire 4 13) ∧ a (.wire 1 43) = a (.wire 4 14) ∧ a (.wire 4 15) = a (.wire 5 12) ∧ a (.virt 19078) = a (.wire 5 13) ∧ a (.wire 2 71) = a (.wire 5 14) ∧ a (.wire 5 15) = a (.virt 19078) ∧ a (.virt 19078) = a (.wire 6 72) ∧ a (.virt 19078) = a (.wire 6 73) ∧ a (.virt 19108) = a (.wire 6 74) ∧ a (.virt 19023) = a (.wire 6 76) ∧ a (.virt 19078) = a (.wire 6 77) ∧ a (.wire 3 71) = a (.wire 6 78) ∧ a (.virt 19108) = a (.wire 2 72) ∧ a (.wire 6 79) = a (.wire 2 73) ∧ a (.virt 19108) = a (.wire 2 74) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies10, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies11 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 6 79) = a (.wire 2 76) ∧ a (.virt 19109) = a (.wire 2 77) ∧ a (.wire 6 79) = a (.wire 2 78) ∧ a (.wire 2 79) = a (.wire 7 0) ∧ a (.virt 19078) = a (.wire 7 1) ∧ a (.wire 6 75) = a (.wire 7 2) ∧ a (.wire 2 75) = a (.virt 19079) ∧ a (.wire 7 3) = a (.virt 19079) ∧ a (.wire 2 7) = a (.wire 4 16) ∧ a (.virt 19108) = a (.wire 4 17) ∧ a (.wire 2 7) = a (.wire 4 18) ∧ a (.wire 4 19) = a (.wire 5 16) ∧ a (.virt 19078) = a (.wire 5 17) ∧ a (.virt 19108) = a (.wire 5 18) ∧ a (.wire 5 19) = a (.virt 19078) ∧ a (.virt 19078) = a (.wire 7 4) ∧ a (.virt 19078) = a (.wire 7 5) ∧ a (.virt 19110) = a (.wire 7 6) ∧ a (.virt 19024) = a (.wire 7 8) ∧ a (.virt 19078) = a (.wire 7 9) ∧ a (.wire 3 79) = a (.wire 7 10) ∧ a (.virt 19110) = a (.wire 8 0) ∧ a (.wire 7 11) = a (.wire 8 1) ∧ a (.virt 19110) = a (.wire 8 2) ∧ a (.wire 7 11) = a (.wire 8 4) ∧ a (.virt 19111) = a (.wire 8 5) ∧ a (.wire 7 11) = a (.wire 8 6) ∧ a (.wire 8 7) = a (.wire 7 12) ∧ a (.virt 19078) = a (.wire 7 13) ∧ a (.wire 7 7) = a (.wire 7 14) ∧ a (.wire 8 3) = a (.virt 19079) ∧ a (.wire 7 15) = a (.virt 19079) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies11, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies12 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 7) = a (.wire 4 20) ∧ a (.virt 19110) = a (.wire 4 21) ∧ a (.wire 2 7) = a (.wire 4 22) ∧ a (.wire 4 23) = a (.wire 5 20) ∧ a (.virt 19078) = a (.wire 5 21) ∧ a (.virt 19110) = a (.wire 5 22) ∧ a (.wire 5 23) = a (.virt 19078) ∧ a (.virt 19078) = a (.wire 7 16) ∧ a (.virt 19078) = a (.wire 7 17) ∧ a (.virt 19112) = a (.wire 7 18) ∧ a (.virt 19025) = a (.wire 7 20) ∧ a (.virt 19078) = a (.wire 7 21) ∧ a (.wire 3 31) = a (.wire 7 22) ∧ a (.virt 19112) = a (.wire 8 8) ∧ a (.wire 7 23) = a (.wire 8 9) ∧ a (.virt 19112) = a (.wire 8 10) ∧ a (.wire 7 23) = a (.wire 8 12) ∧ a (.virt 19113) = a (.wire 8 13) ∧ a (.wire 7 23) = a (.wire 8 14) ∧ a (.wire 8 15) = a (.wire 7 24) ∧ a (.virt 19078) = a (.wire 7 25) ∧ a (.wire 7 19) = a (.wire 7 26) ∧ a (.wire 8 11) = a (.virt 19079) ∧ a (.wire 7 27) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 7 28) ∧ a (.virt 19078) = a (.wire 7 29) ∧ a (.virt 19114) = a (.wire 7 30) ∧ a (.virt 19026) = a (.wire 7 32) ∧ a (.virt 19078) = a (.wire 7 33) ∧ a (.wire 3 39) = a (.wire 7 34) ∧ a (.virt 19114) = a (.wire 8 16) ∧ a (.wire 7 35) = a (.wire 8 17) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies12, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies13 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19114) = a (.wire 8 18) ∧ a (.wire 7 35) = a (.wire 8 20) ∧ a (.virt 19115) = a (.wire 8 21) ∧ a (.wire 7 35) = a (.wire 8 22) ∧ a (.wire 8 23) = a (.wire 7 36) ∧ a (.virt 19078) = a (.wire 7 37) ∧ a (.wire 7 31) = a (.wire 7 38) ∧ a (.wire 8 19) = a (.virt 19079) ∧ a (.wire 7 39) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 7 40) ∧ a (.virt 19078) = a (.wire 7 41) ∧ a (.virt 19116) = a (.wire 7 42) ∧ a (.virt 19027) = a (.wire 7 44) ∧ a (.virt 19078) = a (.wire 7 45) ∧ a (.wire 3 47) = a (.wire 7 46) ∧ a (.virt 19116) = a (.wire 8 24) ∧ a (.wire 7 47) = a (.wire 8 25) ∧ a (.virt 19116) = a (.wire 8 26) ∧ a (.wire 7 47) = a (.wire 8 28) ∧ a (.virt 19117) = a (.wire 8 29) ∧ a (.wire 7 47) = a (.wire 8 30) ∧ a (.wire 8 31) = a (.wire 7 48) ∧ a (.virt 19078) = a (.wire 7 49) ∧ a (.wire 7 43) = a (.wire 7 50) ∧ a (.wire 8 27) = a (.virt 19079) ∧ a (.wire 7 51) = a (.virt 19079) ∧ a (.virt 19078) = a (.wire 7 52) ∧ a (.virt 19078) = a (.wire 7 53) ∧ a (.virt 19118) = a (.wire 7 54) ∧ a (.virt 19028) = a (.wire 7 56) ∧ a (.virt 19078) = a (.wire 7 57) ∧ a (.wire 3 55) = a (.wire 7 58) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies13, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies14 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19118) = a (.wire 8 32) ∧ a (.wire 7 59) = a (.wire 8 33) ∧ a (.virt 19118) = a (.wire 8 34) ∧ a (.wire 7 59) = a (.wire 8 36) ∧ a (.virt 19119) = a (.wire 8 37) ∧ a (.wire 7 59) = a (.wire 8 38) ∧ a (.wire 8 39) = a (.wire 7 60) ∧ a (.virt 19078) = a (.wire 7 61) ∧ a (.wire 7 55) = a (.wire 7 62) ∧ a (.wire 8 35) = a (.virt 19079) ∧ a (.wire 7 63) = a (.virt 19079) ∧ a (.virt 19112) = a (.wire 8 40) ∧ a (.virt 19114) = a (.wire 8 41) ∧ a (.virt 19112) = a (.wire 8 42) ∧ a (.virt 19116) = a (.wire 8 44) ∧ a (.virt 19118) = a (.wire 8 45) ∧ a (.virt 19116) = a (.wire 8 46) ∧ a (.wire 8 43) = a (.wire 8 48) ∧ a (.wire 8 47) = a (.wire 8 49) ∧ a (.wire 8 43) = a (.wire 8 50) ∧ a (.wire 2 7) = a (.wire 4 24) ∧ a (.wire 8 51) = a (.wire 4 25) ∧ a (.wire 2 7) = a (.wire 4 26) ∧ a (.wire 4 27) = a (.wire 5 24) ∧ a (.virt 19078) = a (.wire 5 25) ∧ a (.wire 8 51) = a (.wire 5 26) ∧ a (.wire 5 27) = a (.virt 19078) ∧ a (.wire 1 43) = a (.wire 7 64) ∧ a (.virt 9493) = a (.wire 7 65) ∧ a (.virt 9493) = a (.wire 7 66) ∧ a (.wire 1 43) = a (.wire 7 68) ∧ a (.virt 19079) = a (.wire 7 69) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies14, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies15 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 67) = a (.wire 7 70) ∧ a (.wire 1 43) = a (.wire 7 72) ∧ a (.virt 9494) = a (.wire 7 73) ∧ a (.virt 9494) = a (.wire 7 74) ∧ a (.wire 1 43) = a (.wire 7 76) ∧ a (.virt 19079) = a (.wire 7 77) ∧ a (.wire 7 75) = a (.wire 7 78) ∧ a (.wire 1 43) = a (.wire 9 0) ∧ a (.virt 9495) = a (.wire 9 1) ∧ a (.virt 9495) = a (.wire 9 2) ∧ a (.wire 1 43) = a (.wire 9 4) ∧ a (.virt 19079) = a (.wire 9 5) ∧ a (.wire 9 3) = a (.wire 9 6) ∧ a (.wire 1 43) = a (.wire 9 8) ∧ a (.virt 9496) = a (.wire 9 9) ∧ a (.virt 9496) = a (.wire 9 10) ∧ a (.wire 1 43) = a (.wire 9 12) ∧ a (.virt 19079) = a (.wire 9 13) ∧ a (.wire 9 11) = a (.wire 9 14) ∧ a (.wire 1 43) = a (.wire 9 16) ∧ a (.virt 9497) = a (.wire 9 17) ∧ a (.virt 9497) = a (.wire 9 18) ∧ a (.wire 1 43) = a (.wire 9 20) ∧ a (.virt 19079) = a (.wire 9 21) ∧ a (.wire 9 19) = a (.wire 9 22) ∧ a (.wire 1 43) = a (.wire 9 24) ∧ a (.virt 9498) = a (.wire 9 25) ∧ a (.virt 9498) = a (.wire 9 26) ∧ a (.wire 1 43) = a (.wire 9 28) ∧ a (.virt 19079) = a (.wire 9 29) ∧ a (.wire 9 27) = a (.wire 9 30) ∧ a (.wire 1 43) = a (.wire 9 32) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies15, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies16 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 9499) = a (.wire 9 33) ∧ a (.virt 9499) = a (.wire 9 34) ∧ a (.wire 1 43) = a (.wire 9 36) ∧ a (.virt 19079) = a (.wire 9 37) ∧ a (.wire 9 35) = a (.wire 9 38) ∧ a (.wire 1 43) = a (.wire 9 40) ∧ a (.virt 9500) = a (.wire 9 41) ∧ a (.virt 9500) = a (.wire 9 42) ∧ a (.wire 1 43) = a (.wire 9 44) ∧ a (.virt 19079) = a (.wire 9 45) ∧ a (.wire 9 43) = a (.wire 9 46) ∧ a (.wire 1 43) = a (.wire 9 48) ∧ a (.virt 9501) = a (.wire 9 49) ∧ a (.virt 9501) = a (.wire 9 50) ∧ a (.wire 1 43) = a (.wire 9 52) ∧ a (.virt 19079) = a (.wire 9 53) ∧ a (.wire 9 51) = a (.wire 9 54) ∧ a (.wire 1 43) = a (.wire 9 56) ∧ a (.virt 9502) = a (.wire 9 57) ∧ a (.virt 9502) = a (.wire 9 58) ∧ a (.wire 1 43) = a (.wire 9 60) ∧ a (.virt 19079) = a (.wire 9 61) ∧ a (.wire 9 59) = a (.wire 9 62) ∧ a (.wire 1 43) = a (.wire 9 64) ∧ a (.virt 9503) = a (.wire 9 65) ∧ a (.virt 9503) = a (.wire 9 66) ∧ a (.wire 1 43) = a (.wire 9 68) ∧ a (.virt 19079) = a (.wire 9 69) ∧ a (.wire 9 67) = a (.wire 9 70) ∧ a (.wire 1 43) = a (.wire 9 72) ∧ a (.virt 9504) = a (.wire 9 73) ∧ a (.virt 9504) = a (.wire 9 74) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies16, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies17 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 43) = a (.wire 9 76) ∧ a (.virt 19079) = a (.wire 9 77) ∧ a (.wire 9 75) = a (.wire 9 78) ∧ a (.wire 1 43) = a (.wire 10 0) ∧ a (.virt 9505) = a (.wire 10 1) ∧ a (.virt 9505) = a (.wire 10 2) ∧ a (.wire 1 43) = a (.wire 10 4) ∧ a (.virt 19079) = a (.wire 10 5) ∧ a (.wire 10 3) = a (.wire 10 6) ∧ a (.wire 1 43) = a (.wire 10 8) ∧ a (.virt 9506) = a (.wire 10 9) ∧ a (.virt 9506) = a (.wire 10 10) ∧ a (.wire 1 43) = a (.wire 10 12) ∧ a (.virt 19079) = a (.wire 10 13) ∧ a (.wire 10 11) = a (.wire 10 14) ∧ a (.wire 1 43) = a (.wire 10 16) ∧ a (.virt 9507) = a (.wire 10 17) ∧ a (.virt 9507) = a (.wire 10 18) ∧ a (.wire 1 43) = a (.wire 10 20) ∧ a (.virt 19079) = a (.wire 10 21) ∧ a (.wire 10 19) = a (.wire 10 22) ∧ a (.wire 1 43) = a (.wire 10 24) ∧ a (.virt 9508) = a (.wire 10 25) ∧ a (.virt 9508) = a (.wire 10 26) ∧ a (.wire 1 43) = a (.wire 10 28) ∧ a (.virt 19079) = a (.wire 10 29) ∧ a (.wire 10 27) = a (.wire 10 30) ∧ a (.wire 1 43) = a (.wire 10 32) ∧ a (.virt 9509) = a (.wire 10 33) ∧ a (.virt 9509) = a (.wire 10 34) ∧ a (.wire 1 43) = a (.wire 10 36) ∧ a (.virt 19079) = a (.wire 10 37) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies17, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies18 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 35) = a (.wire 10 38) ∧ a (.wire 1 43) = a (.wire 10 40) ∧ a (.virt 9510) = a (.wire 10 41) ∧ a (.virt 9510) = a (.wire 10 42) ∧ a (.wire 1 43) = a (.wire 10 44) ∧ a (.virt 19079) = a (.wire 10 45) ∧ a (.wire 10 43) = a (.wire 10 46) ∧ a (.wire 1 43) = a (.wire 10 48) ∧ a (.virt 9511) = a (.wire 10 49) ∧ a (.virt 9511) = a (.wire 10 50) ∧ a (.wire 1 43) = a (.wire 10 52) ∧ a (.virt 19079) = a (.wire 10 53) ∧ a (.wire 10 51) = a (.wire 10 54) ∧ a (.wire 1 43) = a (.wire 10 56) ∧ a (.virt 9512) = a (.wire 10 57) ∧ a (.virt 9512) = a (.wire 10 58) ∧ a (.wire 1 43) = a (.wire 10 60) ∧ a (.virt 19079) = a (.wire 10 61) ∧ a (.wire 10 59) = a (.wire 10 62) ∧ a (.wire 2 7) = a (.wire 10 64) ∧ a (.virt 19030) = a (.wire 10 65) ∧ a (.virt 19030) = a (.wire 10 66) ∧ a (.wire 2 7) = a (.wire 10 68) ∧ a (.virt 19079) = a (.wire 10 69) ∧ a (.wire 10 67) = a (.wire 10 70) ∧ a (.wire 2 7) = a (.wire 10 72) ∧ a (.virt 19031) = a (.wire 10 73) ∧ a (.virt 19031) = a (.wire 10 74) ∧ a (.wire 2 7) = a (.wire 10 76) ∧ a (.virt 19079) = a (.wire 10 77) ∧ a (.wire 10 75) = a (.wire 10 78) ∧ a (.wire 2 7) = a (.wire 11 0) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies18, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies19 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19032) = a (.wire 11 1) ∧ a (.virt 19032) = a (.wire 11 2) ∧ a (.wire 2 7) = a (.wire 11 4) ∧ a (.virt 19079) = a (.wire 11 5) ∧ a (.wire 11 3) = a (.wire 11 6) ∧ a (.wire 2 7) = a (.wire 11 8) ∧ a (.virt 19033) = a (.wire 11 9) ∧ a (.virt 19033) = a (.wire 11 10) ∧ a (.wire 2 7) = a (.wire 11 12) ∧ a (.virt 19079) = a (.wire 11 13) ∧ a (.wire 11 11) = a (.wire 11 14) ∧ a (.wire 2 7) = a (.wire 11 16) ∧ a (.virt 19034) = a (.wire 11 17) ∧ a (.virt 19034) = a (.wire 11 18) ∧ a (.wire 2 7) = a (.wire 11 20) ∧ a (.virt 19079) = a (.wire 11 21) ∧ a (.wire 11 19) = a (.wire 11 22) ∧ a (.wire 2 7) = a (.wire 11 24) ∧ a (.virt 19035) = a (.wire 11 25) ∧ a (.virt 19035) = a (.wire 11 26) ∧ a (.wire 2 7) = a (.wire 11 28) ∧ a (.virt 19079) = a (.wire 11 29) ∧ a (.wire 11 27) = a (.wire 11 30) ∧ a (.wire 2 7) = a (.wire 11 32) ∧ a (.virt 19036) = a (.wire 11 33) ∧ a (.virt 19036) = a (.wire 11 34) ∧ a (.wire 2 7) = a (.wire 11 36) ∧ a (.virt 19079) = a (.wire 11 37) ∧ a (.wire 11 35) = a (.wire 11 38) ∧ a (.wire 2 7) = a (.wire 11 40) ∧ a (.virt 19037) = a (.wire 11 41) ∧ a (.virt 19037) = a (.wire 11 42) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies19, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [publicBatchWrapper2.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies20 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 7) = a (.wire 11 44) ∧ a (.virt 19079) = a (.wire 11 45) ∧ a (.wire 11 43) = a (.wire 11 46) ∧ a (.wire 2 7) = a (.wire 11 48) ∧ a (.virt 19038) = a (.wire 11 49) ∧ a (.virt 19038) = a (.wire 11 50) ∧ a (.wire 2 7) = a (.wire 11 52) ∧ a (.virt 19079) = a (.wire 11 53) ∧ a (.wire 11 51) = a (.wire 11 54) ∧ a (.wire 2 7) = a (.wire 11 56) ∧ a (.virt 19039) = a (.wire 11 57) ∧ a (.virt 19039) = a (.wire 11 58) ∧ a (.wire 2 7) = a (.wire 11 60) ∧ a (.virt 19079) = a (.wire 11 61) ∧ a (.wire 11 59) = a (.wire 11 62) ∧ a (.wire 2 7) = a (.wire 11 64) ∧ a (.virt 19040) = a (.wire 11 65) ∧ a (.virt 19040) = a (.wire 11 66) ∧ a (.wire 2 7) = a (.wire 11 68) ∧ a (.virt 19079) = a (.wire 11 69) ∧ a (.wire 11 67) = a (.wire 11 70) ∧ a (.wire 2 7) = a (.wire 11 72) ∧ a (.virt 19041) = a (.wire 11 73) ∧ a (.virt 19041) = a (.wire 11 74) ∧ a (.wire 2 7) = a (.wire 11 76) ∧ a (.virt 19079) = a (.wire 11 77) ∧ a (.wire 11 75) = a (.wire 11 78) ∧ a (.wire 2 7) = a (.wire 12 0) ∧ a (.virt 19042) = a (.wire 12 1) ∧ a (.virt 19042) = a (.wire 12 2) ∧ a (.wire 2 7) = a (.wire 12 4) ∧ a (.virt 19079) = a (.wire 12 5) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies20, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies21 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 3) = a (.wire 12 6) ∧ a (.wire 2 7) = a (.wire 12 8) ∧ a (.virt 19043) = a (.wire 12 9) ∧ a (.virt 19043) = a (.wire 12 10) ∧ a (.wire 2 7) = a (.wire 12 12) ∧ a (.virt 19079) = a (.wire 12 13) ∧ a (.wire 12 11) = a (.wire 12 14) ∧ a (.wire 2 7) = a (.wire 12 16) ∧ a (.virt 19044) = a (.wire 12 17) ∧ a (.virt 19044) = a (.wire 12 18) ∧ a (.wire 2 7) = a (.wire 12 20) ∧ a (.virt 19079) = a (.wire 12 21) ∧ a (.wire 12 19) = a (.wire 12 22) ∧ a (.wire 2 7) = a (.wire 12 24) ∧ a (.virt 19045) = a (.wire 12 25) ∧ a (.virt 19045) = a (.wire 12 26) ∧ a (.wire 2 7) = a (.wire 12 28) ∧ a (.virt 19079) = a (.wire 12 29) ∧ a (.wire 12 27) = a (.wire 12 30) ∧ a (.wire 2 7) = a (.wire 12 32) ∧ a (.virt 19046) = a (.wire 12 33) ∧ a (.virt 19046) = a (.wire 12 34) ∧ a (.wire 2 7) = a (.wire 12 36) ∧ a (.virt 19079) = a (.wire 12 37) ∧ a (.wire 12 35) = a (.wire 12 38) ∧ a (.wire 2 7) = a (.wire 12 40) ∧ a (.virt 19047) = a (.wire 12 41) ∧ a (.virt 19047) = a (.wire 12 42) ∧ a (.wire 2 7) = a (.wire 12 44) ∧ a (.virt 19079) = a (.wire 12 45) ∧ a (.wire 12 43) = a (.wire 12 46) ∧ a (.wire 2 7) = a (.wire 12 48) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies21, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies22 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19048) = a (.wire 12 49) ∧ a (.virt 19048) = a (.wire 12 50) ∧ a (.wire 2 7) = a (.wire 12 52) ∧ a (.virt 19079) = a (.wire 12 53) ∧ a (.wire 12 51) = a (.wire 12 54) ∧ a (.wire 2 7) = a (.wire 12 56) ∧ a (.virt 19049) = a (.wire 12 57) ∧ a (.virt 19049) = a (.wire 12 58) ∧ a (.wire 2 7) = a (.wire 12 60) ∧ a (.virt 19079) = a (.wire 12 61) ∧ a (.wire 12 59) = a (.wire 12 62) ∧ a (.wire 1 43) = a (.wire 12 64) ∧ a (.virt 9513) = a (.wire 12 65) ∧ a (.virt 9513) = a (.wire 12 66) ∧ a (.wire 1 43) = a (.wire 12 68) ∧ a (.virt 19079) = a (.wire 12 69) ∧ a (.wire 12 67) = a (.wire 12 70) ∧ a (.wire 1 43) = a (.wire 12 72) ∧ a (.virt 9514) = a (.wire 12 73) ∧ a (.virt 9514) = a (.wire 12 74) ∧ a (.wire 1 43) = a (.wire 12 76) ∧ a (.virt 19079) = a (.wire 12 77) ∧ a (.wire 12 75) = a (.wire 12 78) ∧ a (.wire 1 43) = a (.wire 13 0) ∧ a (.virt 9515) = a (.wire 13 1) ∧ a (.virt 9515) = a (.wire 13 2) ∧ a (.wire 1 43) = a (.wire 13 4) ∧ a (.virt 19079) = a (.wire 13 5) ∧ a (.wire 13 3) = a (.wire 13 6) ∧ a (.wire 1 43) = a (.wire 13 8) ∧ a (.virt 9516) = a (.wire 13 9) ∧ a (.virt 9516) = a (.wire 13 10) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies22, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies23 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 43) = a (.wire 13 12) ∧ a (.virt 19079) = a (.wire 13 13) ∧ a (.wire 13 11) = a (.wire 13 14) ∧ a (.wire 1 43) = a (.wire 13 16) ∧ a (.virt 9517) = a (.wire 13 17) ∧ a (.virt 9517) = a (.wire 13 18) ∧ a (.wire 1 43) = a (.wire 13 20) ∧ a (.virt 19079) = a (.wire 13 21) ∧ a (.wire 13 19) = a (.wire 13 22) ∧ a (.wire 1 43) = a (.wire 13 24) ∧ a (.virt 9518) = a (.wire 13 25) ∧ a (.virt 9518) = a (.wire 13 26) ∧ a (.wire 1 43) = a (.wire 13 28) ∧ a (.virt 19079) = a (.wire 13 29) ∧ a (.wire 13 27) = a (.wire 13 30) ∧ a (.wire 1 43) = a (.wire 13 32) ∧ a (.virt 9519) = a (.wire 13 33) ∧ a (.virt 9519) = a (.wire 13 34) ∧ a (.wire 1 43) = a (.wire 13 36) ∧ a (.virt 19079) = a (.wire 13 37) ∧ a (.wire 13 35) = a (.wire 13 38) ∧ a (.wire 1 43) = a (.wire 13 40) ∧ a (.virt 9520) = a (.wire 13 41) ∧ a (.virt 9520) = a (.wire 13 42) ∧ a (.wire 1 43) = a (.wire 13 44) ∧ a (.virt 19079) = a (.wire 13 45) ∧ a (.wire 13 43) = a (.wire 13 46) ∧ a (.wire 2 7) = a (.wire 13 48) ∧ a (.virt 19050) = a (.wire 13 49) ∧ a (.virt 19050) = a (.wire 13 50) ∧ a (.wire 2 7) = a (.wire 13 52) ∧ a (.virt 19079) = a (.wire 13 53) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies23, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies23, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies24 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 51) = a (.wire 13 54) ∧ a (.wire 2 7) = a (.wire 13 56) ∧ a (.virt 19051) = a (.wire 13 57) ∧ a (.virt 19051) = a (.wire 13 58) ∧ a (.wire 2 7) = a (.wire 13 60) ∧ a (.virt 19079) = a (.wire 13 61) ∧ a (.wire 13 59) = a (.wire 13 62) ∧ a (.wire 2 7) = a (.wire 13 64) ∧ a (.virt 19052) = a (.wire 13 65) ∧ a (.virt 19052) = a (.wire 13 66) ∧ a (.wire 2 7) = a (.wire 13 68) ∧ a (.virt 19079) = a (.wire 13 69) ∧ a (.wire 13 67) = a (.wire 13 70) ∧ a (.wire 2 7) = a (.wire 13 72) ∧ a (.virt 19053) = a (.wire 13 73) ∧ a (.virt 19053) = a (.wire 13 74) ∧ a (.wire 2 7) = a (.wire 13 76) ∧ a (.virt 19079) = a (.wire 13 77) ∧ a (.wire 13 75) = a (.wire 13 78) ∧ a (.wire 2 7) = a (.wire 14 0) ∧ a (.virt 19054) = a (.wire 14 1) ∧ a (.virt 19054) = a (.wire 14 2) ∧ a (.wire 2 7) = a (.wire 14 4) ∧ a (.virt 19079) = a (.wire 14 5) ∧ a (.wire 14 3) = a (.wire 14 6) ∧ a (.wire 2 7) = a (.wire 14 8) ∧ a (.virt 19055) = a (.wire 14 9) ∧ a (.virt 19055) = a (.wire 14 10) ∧ a (.wire 2 7) = a (.wire 14 12) ∧ a (.virt 19079) = a (.wire 14 13) ∧ a (.wire 14 11) = a (.wire 14 14) ∧ a (.wire 2 7) = a (.wire 14 16) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies24, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper2.copies24, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_copies25 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19056) = a (.wire 14 17) ∧ a (.virt 19056) = a (.wire 14 18) ∧ a (.wire 2 7) = a (.wire 14 20) ∧ a (.virt 19079) = a (.wire 14 21) ∧ a (.wire 14 19) = a (.wire 14 22) ∧ a (.wire 2 7) = a (.wire 14 24) ∧ a (.virt 19057) = a (.wire 14 25) ∧ a (.virt 19057) = a (.wire 14 26) ∧ a (.wire 2 7) = a (.wire 14 28) ∧ a (.virt 19079) = a (.wire 14 29) ∧ a (.wire 14 27) = a (.wire 14 30) := by
  have hc : ∀ q ∈ publicBatchWrapper2.copies25, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper2.copies25, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper2_consts (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19078) = 1 ∧ a (.virt 19079) = 0 ∧ a (.virt 19120) = 8 := by
  have hconst := h.2.2
  simp only [publicBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2⟩ := hconst
  exact ⟨k0, k1, k2⟩

theorem publicBatchWrapper2_f0 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9488)) (a (.virt 19079)) (a (.virt 19080)) (a (.virt 19081)) := by
  have c0 := (publicBatchWrapper2_copies0 a h).1
  have c1 := (publicBatchWrapper2_copies0 a h).2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c0, k0, ← c1, k0, ← c2] at e_0_0
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_1
  simp only [← c9, ← c10, k0, ← c11] at e_0_1
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_0
  simp only [← c3, ← c4, ← c5] at e_1_0
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_1
  simp only [← c6, ← c7, ← c8] at e_1_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_0
    linear_combination c12.trans k1 - hc
  · have hc := e_0_1
    simp only [e_0_0, e_1_1] at hc
    linear_combination c13.trans k1 - hc

theorem publicBatchWrapper2_f1 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9489)) (a (.virt 19079)) (a (.virt 19082)) (a (.virt 19083)) := by
  have c14 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_2
  simp only [← c14, k0, ← c15, k0, ← c16] at e_0_2
  have e_0_3 := arithEq_of_rows h (row := 0) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_3
  simp only [← c23, ← c24, k0, ← c25] at e_0_3
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_2
  simp only [← c17, ← c18, ← c19] at e_1_2
  have e_1_3 := arithEq_of_rows h (row := 1) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_3
  simp only [← c20, ← c21, ← c22] at e_1_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_2
    linear_combination c26.trans k1 - hc
  · have hc := e_0_3
    simp only [e_0_2, e_1_3] at hc
    linear_combination c27.trans k1 - hc

theorem publicBatchWrapper2_f2 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9490)) (a (.virt 19079)) (a (.virt 19084)) (a (.virt 19085)) := by
  have c28 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (publicBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_4 := arithEq_of_rows h (row := 0) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_4
  simp only [← c28, k0, ← c29, k0, ← c30] at e_0_4
  have e_0_5 := arithEq_of_rows h (row := 0) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_5
  simp only [← c37, ← c38, k0, ← c39] at e_0_5
  have e_1_4 := arithEq_of_rows h (row := 1) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_4
  simp only [← c31, ← c32, ← c33] at e_1_4
  have e_1_5 := arithEq_of_rows h (row := 1) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_5
  simp only [← c34, ← c35, ← c36] at e_1_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_4
    linear_combination c40.trans k1 - hc
  · have hc := e_0_5
    simp only [e_0_4, e_1_5] at hc
    linear_combination c41.trans k1 - hc

theorem publicBatchWrapper2_f3 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9491)) (a (.virt 19079)) (a (.virt 19086)) (a (.virt 19087)) := by
  have c42 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_6 := arithEq_of_rows h (row := 0) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_6
  simp only [← c42, k0, ← c43, k0, ← c44] at e_0_6
  have e_0_7 := arithEq_of_rows h (row := 0) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_7
  simp only [← c51, ← c52, k0, ← c53] at e_0_7
  have e_1_6 := arithEq_of_rows h (row := 1) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_6
  simp only [← c45, ← c46, ← c47] at e_1_6
  have e_1_7 := arithEq_of_rows h (row := 1) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_7
  simp only [← c48, ← c49, ← c50] at e_1_7
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_6
    linear_combination c54.trans k1 - hc
  · have hc := e_0_7
    simp only [e_0_6, e_1_7] at hc
    linear_combination c55.trans k1 - hc

theorem publicBatchWrapper2_f4 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 35) = band (a (.virt 19080)) (a (.virt 19082)) := by
  have c56 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c56, ← c57, ← c58] at e_1_8
  have hr := e_1_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f5 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 39) = band (a (.virt 19084)) (a (.virt 19086)) := by
  have c59 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c59, ← c60, ← c61] at e_1_9
  have hr := e_1_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f6 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) := by
  have c62 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (publicBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (publicBatchWrapper2_copies2 a h).1
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c62, ← c63, ← c64] at e_1_10
  have hr := e_1_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f7 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19025)) (a (.virt 19079)) (a (.virt 19088)) (a (.virt 19089)) := by
  have c65 := (publicBatchWrapper2_copies2 a h).2.1
  have c66 := (publicBatchWrapper2_copies2 a h).2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_8 := arithEq_of_rows h (row := 0) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_8
  simp only [← c65, k0, ← c66, k0, ← c67] at e_0_8
  have e_0_9 := arithEq_of_rows h (row := 0) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_9
  simp only [← c74, ← c75, k0, ← c76] at e_0_9
  have e_1_11 := arithEq_of_rows h (row := 1) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_11
  simp only [← c68, ← c69, ← c70] at e_1_11
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c71, ← c72, ← c73] at e_1_12
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_11
    linear_combination c77.trans k1 - hc
  · have hc := e_0_9
    simp only [e_0_8, e_1_12] at hc
    linear_combination c78.trans k1 - hc

theorem publicBatchWrapper2_f8 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19026)) (a (.virt 19079)) (a (.virt 19090)) (a (.virt 19091)) := by
  have c79 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_10 := arithEq_of_rows h (row := 0) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_10
  simp only [← c79, k0, ← c80, k0, ← c81] at e_0_10
  have e_0_11 := arithEq_of_rows h (row := 0) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_11
  simp only [← c88, ← c89, k0, ← c90] at e_0_11
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c82, ← c83, ← c84] at e_1_13
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c85, ← c86, ← c87] at e_1_14
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_13
    linear_combination c91.trans k1 - hc
  · have hc := e_0_11
    simp only [e_0_10, e_1_14] at hc
    linear_combination c92.trans k1 - hc

theorem publicBatchWrapper2_f9 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19027)) (a (.virt 19079)) (a (.virt 19092)) (a (.virt 19093)) := by
  have c93 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (publicBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_12 := arithEq_of_rows h (row := 0) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c93, k0, ← c94, k0, ← c95] at e_0_12
  have e_0_13 := arithEq_of_rows h (row := 0) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c102, ← c103, k0, ← c104] at e_0_13
  have e_1_15 := arithEq_of_rows h (row := 1) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_15
  simp only [← c96, ← c97, ← c98] at e_1_15
  have e_1_16 := arithEq_of_rows h (row := 1) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_16
  simp only [← c99, ← c100, ← c101] at e_1_16
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_15
    linear_combination c105.trans k1 - hc
  · have hc := e_0_13
    simp only [e_0_12, e_1_16] at hc
    linear_combination c106.trans k1 - hc

theorem publicBatchWrapper2_f10 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19028)) (a (.virt 19079)) (a (.virt 19094)) (a (.virt 19095)) := by
  have c107 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_14 := arithEq_of_rows h (row := 0) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c107, k0, ← c108, k0, ← c109] at e_0_14
  have e_0_15 := arithEq_of_rows h (row := 0) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_15
  simp only [← c116, ← c117, k0, ← c118] at e_0_15
  have e_1_17 := arithEq_of_rows h (row := 1) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_17
  simp only [← c110, ← c111, ← c112] at e_1_17
  have e_1_18 := arithEq_of_rows h (row := 1) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_18
  simp only [← c113, ← c114, ← c115] at e_1_18
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_17
    linear_combination c119.trans k1 - hc
  · have hc := e_0_15
    simp only [e_0_14, e_1_18] at hc
    linear_combination c120.trans k1 - hc

theorem publicBatchWrapper2_f11 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 1 79) = band (a (.virt 19088)) (a (.virt 19090)) := by
  have c121 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_19 := arithEq_of_rows h (row := 1) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_19
  simp only [← c121, ← c122, ← c123] at e_1_19
  have hr := e_1_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f12 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 3) = band (a (.virt 19092)) (a (.virt 19094)) := by
  have c124 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c124, ← c125, ← c126] at e_2_0
  have hr := e_2_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f13 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) := by
  have c127 := (publicBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (publicBatchWrapper2_copies4 a h).1
  have c129 := (publicBatchWrapper2_copies4 a h).2.1
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c127, ← c128, ← c129] at e_2_1
  have hr := e_2_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f14 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 67) = bnot (a (.wire 1 43)) := by
  have c130 := (publicBatchWrapper2_copies4 a h).2.2.1
  have c131 := (publicBatchWrapper2_copies4 a h).2.2.2.1
  have c132 := (publicBatchWrapper2_copies4 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_16 := arithEq_of_rows h (row := 0) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_16
  simp only [← c130, k0, ← c131, k0, ← c132] at e_0_16
  have hr := e_0_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper2_f15 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.virt 19078) = bnot (a (.virt 19079)) := by
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  simp only [bnot, k0, k1]
  ring

theorem publicBatchWrapper2_f16 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 67) = band (a (.wire 0 67)) (a (.virt 19078)) := by
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  simp only [band, k0]
  ring

theorem publicBatchWrapper2_f17 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 71) = bselect (a (.wire 0 67)) (a (.virt 9488)) (a (.virt 19079)) := by
  have c133 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.1
  have c134 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.1
  have c135 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_17 := arithEq_of_rows h (row := 0) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c133, ← c134, ← c135, k1] at e_0_17
  have hr := e_0_17
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f18 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 75) = bselect (a (.wire 0 67)) (a (.virt 9489)) (a (.virt 19079)) := by
  have c136 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_18 := arithEq_of_rows h (row := 0) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c136, ← c137, ← c138, k1] at e_0_18
  have hr := e_0_18
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f19 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 79) = bselect (a (.wire 0 67)) (a (.virt 9490)) (a (.virt 19079)) := by
  have c139 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_0_19 := arithEq_of_rows h (row := 0) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c139, ← c140, ← c141, k1] at e_0_19
  have hr := e_0_19
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f20 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 3) = bselect (a (.wire 0 67)) (a (.virt 9491)) (a (.virt 19079)) := by
  have c142 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c144 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c142, ← c143, ← c144, k1] at e_3_0
  have hr := e_3_0
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f21 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 7) = bselect (a (.wire 0 67)) (a (.virt 9492)) (a (.virt 19079)) := by
  have c145 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c145, ← c146, ← c147, k1] at e_3_1
  have hr := e_3_1
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f22 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 11) = bselect (a (.wire 0 67)) (a (.virt 9486)) (a (.virt 19079)) := by
  have c148 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c148, ← c149, ← c150, k1] at e_3_2
  have hr := e_3_2
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f23 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 15) = bselect (a (.wire 0 67)) (a (.virt 9487)) (a (.virt 19079)) := by
  have c151 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c151, ← c152, ← c153, k1] at e_3_3
  have hr := e_3_3
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f24 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 0 67) = bor (a (.virt 19079)) (a (.wire 0 67)) := by
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  simp only [bor, k1]
  ring

theorem publicBatchWrapper2_f25 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 19) = bnot (a (.wire 2 7)) := by
  have c154 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c154, k0, ← c155, k0, ← c156] at e_3_4
  have hr := e_3_4
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper2_f26 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 23) = bnot (a (.wire 0 67)) := by
  have c157 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (publicBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c157, k0, ← c158, k0, ← c159] at e_3_5
  have hr := e_3_5
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper2_f27 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 11) = band (a (.wire 3 19)) (a (.wire 3 23)) := by
  have c160 := (publicBatchWrapper2_copies5 a h).1
  have c161 := (publicBatchWrapper2_copies5 a h).2.1
  have c162 := (publicBatchWrapper2_copies5 a h).2.2.1
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c160, ← c161, ← c162] at e_2_2
  have hr := e_2_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f28 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 31) = bselect (a (.wire 2 11)) (a (.virt 19025)) (a (.wire 0 71)) := by
  have c163 := (publicBatchWrapper2_copies5 a h).2.2.2.1
  have c164 := (publicBatchWrapper2_copies5 a h).2.2.2.2.1
  have c165 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.1
  have c166 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.1
  have c167 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.1
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c163, ← c164, ← c165] at e_3_6
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c166, ← c167, ← c168] at e_3_7
  have hr := e_3_7
  simp only [e_3_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f29 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 39) = bselect (a (.wire 2 11)) (a (.virt 19026)) (a (.wire 0 75)) := by
  have c169 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c169, ← c170, ← c171] at e_3_8
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c172, ← c173, ← c174] at e_3_9
  have hr := e_3_9
  simp only [e_3_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f30 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 47) = bselect (a (.wire 2 11)) (a (.virt 19027)) (a (.wire 0 79)) := by
  have c175 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c175, ← c176, ← c177] at e_3_10
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c178, ← c179, ← c180] at e_3_11
  have hr := e_3_11
  simp only [e_3_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f31 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 55) = bselect (a (.wire 2 11)) (a (.virt 19028)) (a (.wire 3 3)) := by
  have c181 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c181, ← c182, ← c183] at e_3_12
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c184, ← c185, ← c186] at e_3_13
  have hr := e_3_13
  simp only [e_3_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f32 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 63) = bselect (a (.wire 2 11)) (a (.virt 19029)) (a (.wire 3 7)) := by
  have c187 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (publicBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (publicBatchWrapper2_copies6 a h).1
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c187, ← c188, ← c189] at e_3_14
  have e_3_15 := arithEq_of_rows h (row := 3) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c190, ← c191, ← c192] at e_3_15
  have hr := e_3_15
  simp only [e_3_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f33 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 71) = bselect (a (.wire 2 11)) (a (.virt 19023)) (a (.wire 3 11)) := by
  have c193 := (publicBatchWrapper2_copies6 a h).2.1
  have c194 := (publicBatchWrapper2_copies6 a h).2.2.1
  have c195 := (publicBatchWrapper2_copies6 a h).2.2.2.1
  have c196 := (publicBatchWrapper2_copies6 a h).2.2.2.2.1
  have c197 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.1
  have c198 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.1
  have e_3_16 := arithEq_of_rows h (row := 3) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c193, ← c194, ← c195] at e_3_16
  have e_3_17 := arithEq_of_rows h (row := 3) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c196, ← c197, ← c198] at e_3_17
  have hr := e_3_17
  simp only [e_3_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f34 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 3 79) = bselect (a (.wire 2 11)) (a (.virt 19024)) (a (.wire 3 15)) := by
  have c199 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_18 := arithEq_of_rows h (row := 3) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c199, ← c200, ← c201] at e_3_18
  have e_3_19 := arithEq_of_rows h (row := 3) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c202, ← c203, ← c204] at e_3_19
  have hr := e_3_19
  simp only [e_3_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper2_f35 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 3) = bor (a (.wire 0 67)) (a (.wire 3 19)) := by
  have c205 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c205, ← c206, ← c207] at e_4_0
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c208, ← c209, k0, ← c210] at e_5_0
  have hr := e_5_0
  simp only [e_4_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f36 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9486)) (a (.wire 3 71)) (a (.virt 19096)) (a (.virt 19097)) := by
  have c211 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (publicBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c217, ← c218, ← c219] at e_2_3
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c220, ← c221, ← c222] at e_2_4
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c211, k0, ← c212, k0, ← c213] at e_6_0
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c214, ← c215, k0, ← c216] at e_6_1
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c223, ← c224, k0, ← c225] at e_6_2
  refine ⟨?_, ?_⟩
  · have hc := e_2_3
    simp only [e_6_1] at hc
    linear_combination c226.trans k1 - hc
  · have hc := e_6_2
    simp only [e_6_0, e_2_4, e_6_1] at hc
    linear_combination c227.trans k1 - hc

theorem publicBatchWrapper2_f37 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 7) = bor (a (.wire 1 43)) (a (.virt 19096)) := by
  have c228 := (publicBatchWrapper2_copies7 a h).2.2.2.2.1
  have c229 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.1
  have c230 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.1
  have c231 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c228, ← c229, ← c230] at e_4_1
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c231, ← c232, k0, ← c233] at e_5_1
  have hr := e_5_1
  simp only [e_4_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f38 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 7) = a (.virt 19078) := by
  have c234 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c234

theorem publicBatchWrapper2_f39 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9487)) (a (.wire 3 79)) (a (.virt 19098)) (a (.virt 19099)) := by
  have c235 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c241, ← c242, ← c243] at e_2_5
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c244, ← c245, ← c246] at e_2_6
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c235, k0, ← c236, k0, ← c237] at e_6_3
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c238, ← c239, k0, ← c240] at e_6_4
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c247, ← c248, k0, ← c249] at e_6_5
  refine ⟨?_, ?_⟩
  · have hc := e_2_5
    simp only [e_6_4] at hc
    linear_combination c250.trans k1 - hc
  · have hc := e_6_5
    simp only [e_6_3, e_2_6, e_6_4] at hc
    linear_combination c251.trans k1 - hc

theorem publicBatchWrapper2_f40 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 11) = bor (a (.wire 1 43)) (a (.virt 19098)) := by
  have c252 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (publicBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (publicBatchWrapper2_copies8 a h).1
  have c257 := (publicBatchWrapper2_copies8 a h).2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c252, ← c253, ← c254] at e_4_2
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c255, ← c256, k0, ← c257] at e_5_2
  have hr := e_5_2
  simp only [e_4_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f41 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 11) = a (.virt 19078) := by
  have c258 := (publicBatchWrapper2_copies8 a h).2.2.1
  exact c258

theorem publicBatchWrapper2_f42 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9488)) (a (.wire 3 31)) (a (.virt 19100)) (a (.virt 19101)) := by
  have c259 := (publicBatchWrapper2_copies8 a h).2.2.2.1
  have c260 := (publicBatchWrapper2_copies8 a h).2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c265, ← c266, ← c267] at e_2_7
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c268, ← c269, ← c270] at e_2_8
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c259, k0, ← c260, k0, ← c261] at e_6_6
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c262, ← c263, k0, ← c264] at e_6_7
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c271, ← c272, k0, ← c273] at e_6_8
  refine ⟨?_, ?_⟩
  · have hc := e_2_7
    simp only [e_6_7] at hc
    linear_combination c274.trans k1 - hc
  · have hc := e_6_8
    simp only [e_6_6, e_2_8, e_6_7] at hc
    linear_combination c275.trans k1 - hc

theorem publicBatchWrapper2_f43 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9489)) (a (.wire 3 39)) (a (.virt 19102)) (a (.virt 19103)) := by
  have c276 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (publicBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c282, ← c283, ← c284] at e_2_9
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c285, ← c286, ← c287] at e_2_10
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c276, k0, ← c277, k0, ← c278] at e_6_9
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c279, ← c280, k0, ← c281] at e_6_10
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c288, ← c289, k0, ← c290] at e_6_11
  refine ⟨?_, ?_⟩
  · have hc := e_2_9
    simp only [e_6_10] at hc
    linear_combination c291.trans k1 - hc
  · have hc := e_6_11
    simp only [e_6_9, e_2_10, e_6_10] at hc
    linear_combination c292.trans k1 - hc

theorem publicBatchWrapper2_f44 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9490)) (a (.wire 3 47)) (a (.virt 19104)) (a (.virt 19105)) := by
  have c293 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.1
  have c294 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c299, ← c300, ← c301] at e_2_11
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c302, ← c303, ← c304] at e_2_12
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c293, k0, ← c294, k0, ← c295] at e_6_12
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c296, ← c297, k0, ← c298] at e_6_13
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c305, ← c306, k0, ← c307] at e_6_14
  refine ⟨?_, ?_⟩
  · have hc := e_2_11
    simp only [e_6_13] at hc
    linear_combination c308.trans k1 - hc
  · have hc := e_6_14
    simp only [e_6_12, e_2_12, e_6_13] at hc
    linear_combination c309.trans k1 - hc

theorem publicBatchWrapper2_f45 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 9491)) (a (.wire 3 55)) (a (.virt 19106)) (a (.virt 19107)) := by
  have c310 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (publicBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c316, ← c317, ← c318] at e_2_13
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c319, ← c320, ← c321] at e_2_14
  have e_6_15 := arithEq_of_rows h (row := 6) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_15
  simp only [← c310, k0, ← c311, k0, ← c312] at e_6_15
  have e_6_16 := arithEq_of_rows h (row := 6) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_16
  simp only [← c313, ← c314, k0, ← c315] at e_6_16
  have e_6_17 := arithEq_of_rows h (row := 6) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_17
  simp only [← c322, ← c323, k0, ← c324] at e_6_17
  refine ⟨?_, ?_⟩
  · have hc := e_2_13
    simp only [e_6_16] at hc
    linear_combination c325.trans k1 - hc
  · have hc := e_6_17
    simp only [e_6_15, e_2_14, e_6_16] at hc
    linear_combination c326.trans k1 - hc

theorem publicBatchWrapper2_f46 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 63) = band (a (.virt 19100)) (a (.virt 19102)) := by
  have c327 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have e_2_15 := arithEq_of_rows h (row := 2) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c327, ← c328, ← c329] at e_2_15
  have hr := e_2_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f47 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 67) = band (a (.virt 19104)) (a (.virt 19106)) := by
  have c330 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_16 := arithEq_of_rows h (row := 2) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c330, ← c331, ← c332] at e_2_16
  have hr := e_2_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f48 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 2 71) = band (a (.wire 2 63)) (a (.wire 2 67)) := by
  have c333 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_17 := arithEq_of_rows h (row := 2) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c333, ← c334, ← c335] at e_2_17
  have hr := e_2_17
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f49 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 15) = bor (a (.wire 1 43)) (a (.wire 2 71)) := by
  have c336 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c336, ← c337, ← c338] at e_4_3
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c339, ← c340, k0, ← c341] at e_5_3
  have hr := e_5_3
  simp only [e_4_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f50 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 15) = a (.virt 19078) := by
  have c342 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c342

theorem publicBatchWrapper2_f51 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19023)) (a (.wire 3 71)) (a (.virt 19108)) (a (.virt 19109)) := by
  have c343 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (publicBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_2_18 := arithEq_of_rows h (row := 2) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c349, ← c350, ← c351] at e_2_18
  have e_2_19 := arithEq_of_rows h (row := 2) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c352, ← c353, ← c354] at e_2_19
  have e_6_18 := arithEq_of_rows h (row := 6) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_18
  simp only [← c343, k0, ← c344, k0, ← c345] at e_6_18
  have e_6_19 := arithEq_of_rows h (row := 6) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_19
  simp only [← c346, ← c347, k0, ← c348] at e_6_19
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c355, ← c356, k0, ← c357] at e_7_0
  refine ⟨?_, ?_⟩
  · have hc := e_2_18
    simp only [e_6_19] at hc
    linear_combination c358.trans k1 - hc
  · have hc := e_7_0
    simp only [e_6_18, e_2_19, e_6_19] at hc
    linear_combination c359.trans k1 - hc

theorem publicBatchWrapper2_f52 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 19) = bor (a (.wire 2 7)) (a (.virt 19108)) := by
  have c360 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have c362 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c360, ← c361, ← c362] at e_4_4
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c363, ← c364, k0, ← c365] at e_5_4
  have hr := e_5_4
  simp only [e_4_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f53 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 19) = a (.virt 19078) := by
  have c366 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c366

theorem publicBatchWrapper2_f54 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19024)) (a (.wire 3 79)) (a (.virt 19110)) (a (.virt 19111)) := by
  have c367 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (publicBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c367, k0, ← c368, k0, ← c369] at e_7_1
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c370, ← c371, k0, ← c372] at e_7_2
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c379, ← c380, k0, ← c381] at e_7_3
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c373, ← c374, ← c375] at e_8_0
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c376, ← c377, ← c378] at e_8_1
  refine ⟨?_, ?_⟩
  · have hc := e_8_0
    simp only [e_7_2] at hc
    linear_combination c382.trans k1 - hc
  · have hc := e_7_3
    simp only [e_7_1, e_8_1, e_7_2] at hc
    linear_combination c383.trans k1 - hc

theorem publicBatchWrapper2_f55 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 23) = bor (a (.wire 2 7)) (a (.virt 19110)) := by
  have c384 := (publicBatchWrapper2_copies12 a h).1
  have c385 := (publicBatchWrapper2_copies12 a h).2.1
  have c386 := (publicBatchWrapper2_copies12 a h).2.2.1
  have c387 := (publicBatchWrapper2_copies12 a h).2.2.2.1
  have c388 := (publicBatchWrapper2_copies12 a h).2.2.2.2.1
  have c389 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c384, ← c385, ← c386] at e_4_5
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c387, ← c388, k0, ← c389] at e_5_5
  have hr := e_5_5
  simp only [e_4_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f56 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 23) = a (.virt 19078) := by
  have c390 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.1
  exact c390

theorem publicBatchWrapper2_f57 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19025)) (a (.wire 3 31)) (a (.virt 19112)) (a (.virt 19113)) := by
  have c391 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c391, k0, ← c392, k0, ← c393] at e_7_4
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c394, ← c395, k0, ← c396] at e_7_5
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c403, ← c404, k0, ← c405] at e_7_6
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c397, ← c398, ← c399] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c400, ← c401, ← c402] at e_8_3
  refine ⟨?_, ?_⟩
  · have hc := e_8_2
    simp only [e_7_5] at hc
    linear_combination c406.trans k1 - hc
  · have hc := e_7_6
    simp only [e_7_4, e_8_3, e_7_5] at hc
    linear_combination c407.trans k1 - hc

theorem publicBatchWrapper2_f58 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19026)) (a (.wire 3 39)) (a (.virt 19114)) (a (.virt 19115)) := by
  have c408 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (publicBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c408, k0, ← c409, k0, ← c410] at e_7_7
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c411, ← c412, k0, ← c413] at e_7_8
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c420, ← c421, k0, ← c422] at e_7_9
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c414, ← c415, ← c416] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c417, ← c418, ← c419] at e_8_5
  refine ⟨?_, ?_⟩
  · have hc := e_8_4
    simp only [e_7_8] at hc
    linear_combination c423.trans k1 - hc
  · have hc := e_7_9
    simp only [e_7_7, e_8_5, e_7_8] at hc
    linear_combination c424.trans k1 - hc

theorem publicBatchWrapper2_f59 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19027)) (a (.wire 3 47)) (a (.virt 19116)) (a (.virt 19117)) := by
  have c425 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c425, k0, ← c426, k0, ← c427] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c428, ← c429, k0, ← c430] at e_7_11
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c437, ← c438, k0, ← c439] at e_7_12
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c431, ← c432, ← c433] at e_8_6
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c434, ← c435, ← c436] at e_8_7
  refine ⟨?_, ?_⟩
  · have hc := e_8_6
    simp only [e_7_11] at hc
    linear_combination c440.trans k1 - hc
  · have hc := e_7_12
    simp only [e_7_10, e_8_7, e_7_11] at hc
    linear_combination c441.trans k1 - hc

theorem publicBatchWrapper2_f60 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    IsEqual (a (.virt 19028)) (a (.wire 3 55)) (a (.virt 19118)) (a (.virt 19119)) := by
  have c442 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (publicBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c442, k0, ← c443, k0, ← c444] at e_7_13
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c445, ← c446, k0, ← c447] at e_7_14
  have e_7_15 := arithEq_of_rows h (row := 7) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_15
  simp only [← c454, ← c455, k0, ← c456] at e_7_15
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c448, ← c449, ← c450] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c451, ← c452, ← c453] at e_8_9
  refine ⟨?_, ?_⟩
  · have hc := e_8_8
    simp only [e_7_14] at hc
    linear_combination c457.trans k1 - hc
  · have hc := e_7_15
    simp only [e_7_13, e_8_9, e_7_14] at hc
    linear_combination c458.trans k1 - hc

theorem publicBatchWrapper2_f61 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 8 43) = band (a (.virt 19112)) (a (.virt 19114)) := by
  have c459 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c459, ← c460, ← c461] at e_8_10
  have hr := e_8_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f62 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 8 47) = band (a (.virt 19116)) (a (.virt 19118)) := by
  have c462 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c462, ← c463, ← c464] at e_8_11
  have hr := e_8_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f63 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 8 51) = band (a (.wire 8 43)) (a (.wire 8 47)) := by
  have c465 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c465, ← c466, ← c467] at e_8_12
  have hr := e_8_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper2_f64 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 27) = bor (a (.wire 2 7)) (a (.wire 8 51)) := by
  have c468 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c468, ← c469, ← c470] at e_4_6
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c471, ← c472, k0, ← c473] at e_5_6
  have hr := e_5_6
  simp only [e_4_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper2_f65 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 5 27) = a (.virt 19078) := by
  have c474 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c474

theorem publicBatchWrapper2_f66 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9493)) := by
  have c475 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (publicBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (publicBatchWrapper2_copies15 a h).1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_16 := arithEq_of_rows h (row := 7) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_16
  simp only [← c475, ← c476, ← c477] at e_7_16
  have e_7_17 := arithEq_of_rows h (row := 7) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_17
  simp only [← c478, ← c479, k1, ← c480] at e_7_17
  have hr := e_7_17
  simp only [e_7_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f67 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 7 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9494)) := by
  have c481 := (publicBatchWrapper2_copies15 a h).2.1
  have c482 := (publicBatchWrapper2_copies15 a h).2.2.1
  have c483 := (publicBatchWrapper2_copies15 a h).2.2.2.1
  have c484 := (publicBatchWrapper2_copies15 a h).2.2.2.2.1
  have c485 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.1
  have c486 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_7_18 := arithEq_of_rows h (row := 7) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_18
  simp only [← c481, ← c482, ← c483] at e_7_18
  have e_7_19 := arithEq_of_rows h (row := 7) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_19
  simp only [← c484, ← c485, k1, ← c486] at e_7_19
  have hr := e_7_19
  simp only [e_7_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f68 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9495)) := by
  have c487 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c487, ← c488, ← c489] at e_9_0
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c490, ← c491, k1, ← c492] at e_9_1
  have hr := e_9_1
  simp only [e_9_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f69 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9496)) := by
  have c493 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c493, ← c494, ← c495] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c496, ← c497, k1, ← c498] at e_9_3
  have hr := e_9_3
  simp only [e_9_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f70 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9497)) := by
  have c499 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c499, ← c500, ← c501] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c502, ← c503, k1, ← c504] at e_9_5
  have hr := e_9_5
  simp only [e_9_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f71 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9498)) := by
  have c505 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c505, ← c506, ← c507] at e_9_6
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c508, ← c509, k1, ← c510] at e_9_7
  have hr := e_9_7
  simp only [e_9_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f72 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9499)) := by
  have c511 := (publicBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (publicBatchWrapper2_copies16 a h).1
  have c513 := (publicBatchWrapper2_copies16 a h).2.1
  have c514 := (publicBatchWrapper2_copies16 a h).2.2.1
  have c515 := (publicBatchWrapper2_copies16 a h).2.2.2.1
  have c516 := (publicBatchWrapper2_copies16 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c511, ← c512, ← c513] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c514, ← c515, k1, ← c516] at e_9_9
  have hr := e_9_9
  simp only [e_9_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f73 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9500)) := by
  have c517 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.1
  have c518 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.1
  have c519 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c517, ← c518, ← c519] at e_9_10
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c520, ← c521, k1, ← c522] at e_9_11
  have hr := e_9_11
  simp only [e_9_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f74 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 55) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9501)) := by
  have c523 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c523, ← c524, ← c525] at e_9_12
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c526, ← c527, k1, ← c528] at e_9_13
  have hr := e_9_13
  simp only [e_9_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f75 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 63) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9502)) := by
  have c529 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c532 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c529, ← c530, ← c531] at e_9_14
  have e_9_15 := arithEq_of_rows h (row := 9) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_15
  simp only [← c532, ← c533, k1, ← c534] at e_9_15
  have hr := e_9_15
  simp only [e_9_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f76 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9503)) := by
  have c535 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_16 := arithEq_of_rows h (row := 9) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_16
  simp only [← c535, ← c536, ← c537] at e_9_16
  have e_9_17 := arithEq_of_rows h (row := 9) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_17
  simp only [← c538, ← c539, k1, ← c540] at e_9_17
  have hr := e_9_17
  simp only [e_9_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f77 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 9 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9504)) := by
  have c541 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (publicBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (publicBatchWrapper2_copies17 a h).1
  have c545 := (publicBatchWrapper2_copies17 a h).2.1
  have c546 := (publicBatchWrapper2_copies17 a h).2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_9_18 := arithEq_of_rows h (row := 9) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_18
  simp only [← c541, ← c542, ← c543] at e_9_18
  have e_9_19 := arithEq_of_rows h (row := 9) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_19
  simp only [← c544, ← c545, k1, ← c546] at e_9_19
  have hr := e_9_19
  simp only [e_9_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f78 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9505)) := by
  have c547 := (publicBatchWrapper2_copies17 a h).2.2.2.1
  have c548 := (publicBatchWrapper2_copies17 a h).2.2.2.2.1
  have c549 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.1
  have c550 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.1
  have c551 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c547, ← c548, ← c549] at e_10_0
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c550, ← c551, k1, ← c552] at e_10_1
  have hr := e_10_1
  simp only [e_10_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f79 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9506)) := by
  have c553 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c553, ← c554, ← c555] at e_10_2
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c556, ← c557, k1, ← c558] at e_10_3
  have hr := e_10_3
  simp only [e_10_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f80 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9507)) := by
  have c559 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c559, ← c560, ← c561] at e_10_4
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c562, ← c563, k1, ← c564] at e_10_5
  have hr := e_10_5
  simp only [e_10_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f81 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9508)) := by
  have c565 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c565, ← c566, ← c567] at e_10_6
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c568, ← c569, k1, ← c570] at e_10_7
  have hr := e_10_7
  simp only [e_10_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f82 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9509)) := by
  have c571 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (publicBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (publicBatchWrapper2_copies18 a h).1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c571, ← c572, ← c573] at e_10_8
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c574, ← c575, k1, ← c576] at e_10_9
  have hr := e_10_9
  simp only [e_10_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f83 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9510)) := by
  have c577 := (publicBatchWrapper2_copies18 a h).2.1
  have c578 := (publicBatchWrapper2_copies18 a h).2.2.1
  have c579 := (publicBatchWrapper2_copies18 a h).2.2.2.1
  have c580 := (publicBatchWrapper2_copies18 a h).2.2.2.2.1
  have c581 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.1
  have c582 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c577, ← c578, ← c579] at e_10_10
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c580, ← c581, k1, ← c582] at e_10_11
  have hr := e_10_11
  simp only [e_10_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f84 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 55) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9511)) := by
  have c583 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c583, ← c584, ← c585] at e_10_12
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c586, ← c587, k1, ← c588] at e_10_13
  have hr := e_10_13
  simp only [e_10_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f85 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 63) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9512)) := by
  have c589 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c589, ← c590, ← c591] at e_10_14
  have e_10_15 := arithEq_of_rows h (row := 10) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_15
  simp only [← c592, ← c593, k1, ← c594] at e_10_15
  have hr := e_10_15
  simp only [e_10_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f86 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19030)) := by
  have c595 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c600 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_16 := arithEq_of_rows h (row := 10) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_16
  simp only [← c595, ← c596, ← c597] at e_10_16
  have e_10_17 := arithEq_of_rows h (row := 10) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_17
  simp only [← c598, ← c599, k1, ← c600] at e_10_17
  have hr := e_10_17
  simp only [e_10_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f87 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 10 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19031)) := by
  have c601 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_10_18 := arithEq_of_rows h (row := 10) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_18
  simp only [← c601, ← c602, ← c603] at e_10_18
  have e_10_19 := arithEq_of_rows h (row := 10) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_19
  simp only [← c604, ← c605, k1, ← c606] at e_10_19
  have hr := e_10_19
  simp only [e_10_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f88 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19032)) := by
  have c607 := (publicBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (publicBatchWrapper2_copies19 a h).1
  have c609 := (publicBatchWrapper2_copies19 a h).2.1
  have c610 := (publicBatchWrapper2_copies19 a h).2.2.1
  have c611 := (publicBatchWrapper2_copies19 a h).2.2.2.1
  have c612 := (publicBatchWrapper2_copies19 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c607, ← c608, ← c609] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c610, ← c611, k1, ← c612] at e_11_1
  have hr := e_11_1
  simp only [e_11_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f89 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19033)) := by
  have c613 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.1
  have c614 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.1
  have c615 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c613, ← c614, ← c615] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c616, ← c617, k1, ← c618] at e_11_3
  have hr := e_11_3
  simp only [e_11_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f90 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19034)) := by
  have c619 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c619, ← c620, ← c621] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c622, ← c623, k1, ← c624] at e_11_5
  have hr := e_11_5
  simp only [e_11_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f91 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19035)) := by
  have c625 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c625, ← c626, ← c627] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c628, ← c629, k1, ← c630] at e_11_7
  have hr := e_11_7
  simp only [e_11_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f92 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 39) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19036)) := by
  have c631 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c631, ← c632, ← c633] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c634, ← c635, k1, ← c636] at e_11_9
  have hr := e_11_9
  simp only [e_11_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f93 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 47) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19037)) := by
  have c637 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (publicBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (publicBatchWrapper2_copies20 a h).1
  have c641 := (publicBatchWrapper2_copies20 a h).2.1
  have c642 := (publicBatchWrapper2_copies20 a h).2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c637, ← c638, ← c639] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c640, ← c641, k1, ← c642] at e_11_11
  have hr := e_11_11
  simp only [e_11_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f94 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19038)) := by
  have c643 := (publicBatchWrapper2_copies20 a h).2.2.2.1
  have c644 := (publicBatchWrapper2_copies20 a h).2.2.2.2.1
  have c645 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.1
  have c646 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.1
  have c647 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c643, ← c644, ← c645] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c646, ← c647, k1, ← c648] at e_11_13
  have hr := e_11_13
  simp only [e_11_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f95 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19039)) := by
  have c649 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c649, ← c650, ← c651] at e_11_14
  have e_11_15 := arithEq_of_rows h (row := 11) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c652, ← c653, k1, ← c654] at e_11_15
  have hr := e_11_15
  simp only [e_11_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f96 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19040)) := by
  have c655 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_16 := arithEq_of_rows h (row := 11) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c655, ← c656, ← c657] at e_11_16
  have e_11_17 := arithEq_of_rows h (row := 11) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c658, ← c659, k1, ← c660] at e_11_17
  have hr := e_11_17
  simp only [e_11_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f97 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 11 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19041)) := by
  have c661 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_11_18 := arithEq_of_rows h (row := 11) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c661, ← c662, ← c663] at e_11_18
  have e_11_19 := arithEq_of_rows h (row := 11) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c664, ← c665, k1, ← c666] at e_11_19
  have hr := e_11_19
  simp only [e_11_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f98 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19042)) := by
  have c667 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (publicBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (publicBatchWrapper2_copies21 a h).1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c667, ← c668, ← c669] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c670, ← c671, k1, ← c672] at e_12_1
  have hr := e_12_1
  simp only [e_12_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f99 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19043)) := by
  have c673 := (publicBatchWrapper2_copies21 a h).2.1
  have c674 := (publicBatchWrapper2_copies21 a h).2.2.1
  have c675 := (publicBatchWrapper2_copies21 a h).2.2.2.1
  have c676 := (publicBatchWrapper2_copies21 a h).2.2.2.2.1
  have c677 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.1
  have c678 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c673, ← c674, ← c675] at e_12_2
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c676, ← c677, k1, ← c678] at e_12_3
  have hr := e_12_3
  simp only [e_12_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f100 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19044)) := by
  have c679 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c679, ← c680, ← c681] at e_12_4
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c682, ← c683, k1, ← c684] at e_12_5
  have hr := e_12_5
  simp only [e_12_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f101 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19045)) := by
  have c685 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c685, ← c686, ← c687] at e_12_6
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c688, ← c689, k1, ← c690] at e_12_7
  have hr := e_12_7
  simp only [e_12_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f102 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 39) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19046)) := by
  have c691 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c691, ← c692, ← c693] at e_12_8
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c694, ← c695, k1, ← c696] at e_12_9
  have hr := e_12_9
  simp only [e_12_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f103 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 47) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19047)) := by
  have c697 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c697, ← c698, ← c699] at e_12_10
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c700, ← c701, k1, ← c702] at e_12_11
  have hr := e_12_11
  simp only [e_12_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f104 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19048)) := by
  have c703 := (publicBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (publicBatchWrapper2_copies22 a h).1
  have c705 := (publicBatchWrapper2_copies22 a h).2.1
  have c706 := (publicBatchWrapper2_copies22 a h).2.2.1
  have c707 := (publicBatchWrapper2_copies22 a h).2.2.2.1
  have c708 := (publicBatchWrapper2_copies22 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c703, ← c704, ← c705] at e_12_12
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c706, ← c707, k1, ← c708] at e_12_13
  have hr := e_12_13
  simp only [e_12_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f105 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19049)) := by
  have c709 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.1
  have c710 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.1
  have c711 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c709, ← c710, ← c711] at e_12_14
  have e_12_15 := arithEq_of_rows h (row := 12) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c712, ← c713, k1, ← c714] at e_12_15
  have hr := e_12_15
  simp only [e_12_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f106 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 71) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9513)) := by
  have c715 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_16 := arithEq_of_rows h (row := 12) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c715, ← c716, ← c717] at e_12_16
  have e_12_17 := arithEq_of_rows h (row := 12) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c718, ← c719, k1, ← c720] at e_12_17
  have hr := e_12_17
  simp only [e_12_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f107 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 12 79) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9514)) := by
  have c721 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_12_18 := arithEq_of_rows h (row := 12) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c721, ← c722, ← c723] at e_12_18
  have e_12_19 := arithEq_of_rows h (row := 12) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c724, ← c725, k1, ← c726] at e_12_19
  have hr := e_12_19
  simp only [e_12_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f108 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 7) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9515)) := by
  have c727 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c727, ← c728, ← c729] at e_13_0
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c730, ← c731, k1, ← c732] at e_13_1
  have hr := e_13_1
  simp only [e_13_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f109 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 15) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9516)) := by
  have c733 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (publicBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c736 := (publicBatchWrapper2_copies23 a h).1
  have c737 := (publicBatchWrapper2_copies23 a h).2.1
  have c738 := (publicBatchWrapper2_copies23 a h).2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c733, ← c734, ← c735] at e_13_2
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c736, ← c737, k1, ← c738] at e_13_3
  have hr := e_13_3
  simp only [e_13_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f110 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 23) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9517)) := by
  have c739 := (publicBatchWrapper2_copies23 a h).2.2.2.1
  have c740 := (publicBatchWrapper2_copies23 a h).2.2.2.2.1
  have c741 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.1
  have c742 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.1
  have c743 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c739, ← c740, ← c741] at e_13_4
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c742, ← c743, k1, ← c744] at e_13_5
  have hr := e_13_5
  simp only [e_13_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f111 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 31) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9518)) := by
  have c745 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c745, ← c746, ← c747] at e_13_6
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c748, ← c749, k1, ← c750] at e_13_7
  have hr := e_13_7
  simp only [e_13_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f112 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 39) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9519)) := by
  have c751 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c753 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c751, ← c752, ← c753] at e_13_8
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c754, ← c755, k1, ← c756] at e_13_9
  have hr := e_13_9
  simp only [e_13_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f113 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 47) = bselect (a (.wire 1 43)) (a (.virt 19079)) (a (.virt 9520)) := by
  have c757 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c759 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c757, ← c758, ← c759] at e_13_10
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c760, ← c761, k1, ← c762] at e_13_11
  have hr := e_13_11
  simp only [e_13_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f114 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 55) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19050)) := by
  have c763 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (publicBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (publicBatchWrapper2_copies24 a h).1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c763, ← c764, ← c765] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c766, ← c767, k1, ← c768] at e_13_13
  have hr := e_13_13
  simp only [e_13_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f115 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 63) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19051)) := by
  have c769 := (publicBatchWrapper2_copies24 a h).2.1
  have c770 := (publicBatchWrapper2_copies24 a h).2.2.1
  have c771 := (publicBatchWrapper2_copies24 a h).2.2.2.1
  have c772 := (publicBatchWrapper2_copies24 a h).2.2.2.2.1
  have c773 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.1
  have c774 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c769, ← c770, ← c771] at e_13_14
  have e_13_15 := arithEq_of_rows h (row := 13) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c772, ← c773, k1, ← c774] at e_13_15
  have hr := e_13_15
  simp only [e_13_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f116 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 71) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19052)) := by
  have c775 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have c778 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_16 := arithEq_of_rows h (row := 13) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c775, ← c776, ← c777] at e_13_16
  have e_13_17 := arithEq_of_rows h (row := 13) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c778, ← c779, k1, ← c780] at e_13_17
  have hr := e_13_17
  simp only [e_13_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f117 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 13 79) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19053)) := by
  have c781 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_13_18 := arithEq_of_rows h (row := 13) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c781, ← c782, ← c783] at e_13_18
  have e_13_19 := arithEq_of_rows h (row := 13) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c784, ← c785, k1, ← c786] at e_13_19
  have hr := e_13_19
  simp only [e_13_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f118 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 7) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19054)) := by
  have c787 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c787, ← c788, ← c789] at e_14_0
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c790, ← c791, k1, ← c792] at e_14_1
  have hr := e_14_1
  simp only [e_14_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f119 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 15) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19055)) := by
  have c793 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c793, ← c794, ← c795] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c796, ← c797, k1, ← c798] at e_14_3
  have hr := e_14_3
  simp only [e_14_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f120 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 23) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19056)) := by
  have c799 := (publicBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (publicBatchWrapper2_copies25 a h).1
  have c801 := (publicBatchWrapper2_copies25 a h).2.1
  have c802 := (publicBatchWrapper2_copies25 a h).2.2.1
  have c803 := (publicBatchWrapper2_copies25 a h).2.2.2.1
  have c804 := (publicBatchWrapper2_copies25 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c799, ← c800, ← c801] at e_14_4
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c802, ← c803, k1, ← c804] at e_14_5
  have hr := e_14_5
  simp only [e_14_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper2_f121 (a : Assignment p) (h : Satisfies (publicBatchWrapper2 p) a) :
    a (.wire 14 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19057)) := by
  have c805 := (publicBatchWrapper2_copies25 a h).2.2.2.2.2.1
  have c806 := (publicBatchWrapper2_copies25 a h).2.2.2.2.2.2.1
  have c807 := (publicBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (publicBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (publicBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (publicBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2⟩ := publicBatchWrapper2_consts a h
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c805, ← c806, ← c807] at e_14_6
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c808, ← c809, k1, ← c810] at e_14_7
  have hr := e_14_7
  simp only [e_14_6] at hr
  simp only [bselect, k1]
  linear_combination hr

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
    a (.wire 14 31) = bselect (a (.wire 2 7)) (a (.virt 19079)) (a (.virt 19057))) :=
  ⟨⟨publicBatchWrapper2_f0 a h, publicBatchWrapper2_f1 a h, publicBatchWrapper2_f2 a h, publicBatchWrapper2_f3 a h, publicBatchWrapper2_f4 a h, publicBatchWrapper2_f5 a h, publicBatchWrapper2_f6 a h, publicBatchWrapper2_f7 a h, publicBatchWrapper2_f8 a h, publicBatchWrapper2_f9 a h, publicBatchWrapper2_f10 a h, publicBatchWrapper2_f11 a h, publicBatchWrapper2_f12 a h, publicBatchWrapper2_f13 a h, publicBatchWrapper2_f14 a h, publicBatchWrapper2_f15 a h, publicBatchWrapper2_f16 a h, publicBatchWrapper2_f17 a h, publicBatchWrapper2_f18 a h, publicBatchWrapper2_f19 a h, publicBatchWrapper2_f20 a h, publicBatchWrapper2_f21 a h, publicBatchWrapper2_f22 a h, publicBatchWrapper2_f23 a h, publicBatchWrapper2_f24 a h, publicBatchWrapper2_f25 a h, publicBatchWrapper2_f26 a h, publicBatchWrapper2_f27 a h, publicBatchWrapper2_f28 a h, publicBatchWrapper2_f29 a h, publicBatchWrapper2_f30 a h, publicBatchWrapper2_f31 a h⟩, ⟨publicBatchWrapper2_f32 a h, publicBatchWrapper2_f33 a h, publicBatchWrapper2_f34 a h, publicBatchWrapper2_f35 a h, publicBatchWrapper2_f36 a h, publicBatchWrapper2_f37 a h, publicBatchWrapper2_f38 a h, publicBatchWrapper2_f39 a h, publicBatchWrapper2_f40 a h, publicBatchWrapper2_f41 a h, publicBatchWrapper2_f42 a h, publicBatchWrapper2_f43 a h, publicBatchWrapper2_f44 a h, publicBatchWrapper2_f45 a h, publicBatchWrapper2_f46 a h, publicBatchWrapper2_f47 a h, publicBatchWrapper2_f48 a h, publicBatchWrapper2_f49 a h, publicBatchWrapper2_f50 a h, publicBatchWrapper2_f51 a h, publicBatchWrapper2_f52 a h, publicBatchWrapper2_f53 a h, publicBatchWrapper2_f54 a h, publicBatchWrapper2_f55 a h, publicBatchWrapper2_f56 a h, publicBatchWrapper2_f57 a h, publicBatchWrapper2_f58 a h, publicBatchWrapper2_f59 a h, publicBatchWrapper2_f60 a h, publicBatchWrapper2_f61 a h, publicBatchWrapper2_f62 a h, publicBatchWrapper2_f63 a h⟩, ⟨publicBatchWrapper2_f64 a h, publicBatchWrapper2_f65 a h, publicBatchWrapper2_f66 a h, publicBatchWrapper2_f67 a h, publicBatchWrapper2_f68 a h, publicBatchWrapper2_f69 a h, publicBatchWrapper2_f70 a h, publicBatchWrapper2_f71 a h, publicBatchWrapper2_f72 a h, publicBatchWrapper2_f73 a h, publicBatchWrapper2_f74 a h, publicBatchWrapper2_f75 a h, publicBatchWrapper2_f76 a h, publicBatchWrapper2_f77 a h, publicBatchWrapper2_f78 a h, publicBatchWrapper2_f79 a h, publicBatchWrapper2_f80 a h, publicBatchWrapper2_f81 a h, publicBatchWrapper2_f82 a h, publicBatchWrapper2_f83 a h, publicBatchWrapper2_f84 a h, publicBatchWrapper2_f85 a h, publicBatchWrapper2_f86 a h, publicBatchWrapper2_f87 a h, publicBatchWrapper2_f88 a h, publicBatchWrapper2_f89 a h, publicBatchWrapper2_f90 a h, publicBatchWrapper2_f91 a h, publicBatchWrapper2_f92 a h, publicBatchWrapper2_f93 a h, publicBatchWrapper2_f94 a h, publicBatchWrapper2_f95 a h⟩, ⟨publicBatchWrapper2_f96 a h, publicBatchWrapper2_f97 a h, publicBatchWrapper2_f98 a h, publicBatchWrapper2_f99 a h, publicBatchWrapper2_f100 a h, publicBatchWrapper2_f101 a h, publicBatchWrapper2_f102 a h, publicBatchWrapper2_f103 a h, publicBatchWrapper2_f104 a h, publicBatchWrapper2_f105 a h, publicBatchWrapper2_f106 a h, publicBatchWrapper2_f107 a h, publicBatchWrapper2_f108 a h, publicBatchWrapper2_f109 a h, publicBatchWrapper2_f110 a h, publicBatchWrapper2_f111 a h, publicBatchWrapper2_f112 a h, publicBatchWrapper2_f113 a h, publicBatchWrapper2_f114 a h, publicBatchWrapper2_f115 a h, publicBatchWrapper2_f116 a h, publicBatchWrapper2_f117 a h, publicBatchWrapper2_f118 a h, publicBatchWrapper2_f119 a h, publicBatchWrapper2_f120 a h, publicBatchWrapper2_f121 a h⟩⟩

end Plonky2Spec.Generated
