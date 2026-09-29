/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from the `n_inner = 4` public-batch wrapper trace
  (`qp-zk-circuits/formal/traces/public_batch_wrapper_n4.json`, recorded by
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

/-- `publicBatchWrapper4.copies`, items `0..32`. -/
def publicBatchWrapper4.copies0 : List (Target × Target) := [
    (.virt 38152, .wire 0 0),
    (.virt 38152, .wire 0 1),
    (.virt 38154, .wire 0 2),
    (.virt 38154, .wire 1 0),
    (.virt 9488, .wire 1 1),
    (.virt 38154, .wire 1 2),
    (.virt 9488, .wire 1 4),
    (.virt 38155, .wire 1 5),
    (.virt 9488, .wire 1 6),
    (.wire 1 7, .wire 0 4),
    (.virt 38152, .wire 0 5),
    (.wire 0 3, .wire 0 6),
    (.wire 1 3, .virt 38153),
    (.wire 0 7, .virt 38153),
    (.virt 38152, .wire 0 8),
    (.virt 38152, .wire 0 9),
    (.virt 38156, .wire 0 10),
    (.virt 38156, .wire 1 8),
    (.virt 9489, .wire 1 9),
    (.virt 38156, .wire 1 10),
    (.virt 9489, .wire 1 12),
    (.virt 38157, .wire 1 13),
    (.virt 9489, .wire 1 14),
    (.wire 1 15, .wire 0 12),
    (.virt 38152, .wire 0 13),
    (.wire 0 11, .wire 0 14),
    (.wire 1 11, .virt 38153),
    (.wire 0 15, .virt 38153),
    (.virt 38152, .wire 0 16),
    (.virt 38152, .wire 0 17),
    (.virt 38158, .wire 0 18),
    (.virt 38158, .wire 1 16)
  ]

/-- `publicBatchWrapper4.copies`, items `32..64`. -/
def publicBatchWrapper4.copies1 : List (Target × Target) := [
    (.virt 9490, .wire 1 17),
    (.virt 38158, .wire 1 18),
    (.virt 9490, .wire 1 20),
    (.virt 38159, .wire 1 21),
    (.virt 9490, .wire 1 22),
    (.wire 1 23, .wire 0 20),
    (.virt 38152, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.wire 1 19, .virt 38153),
    (.wire 0 23, .virt 38153),
    (.virt 38152, .wire 0 24),
    (.virt 38152, .wire 0 25),
    (.virt 38160, .wire 0 26),
    (.virt 38160, .wire 1 24),
    (.virt 9491, .wire 1 25),
    (.virt 38160, .wire 1 26),
    (.virt 9491, .wire 1 28),
    (.virt 38161, .wire 1 29),
    (.virt 9491, .wire 1 30),
    (.wire 1 31, .wire 0 28),
    (.virt 38152, .wire 0 29),
    (.wire 0 27, .wire 0 30),
    (.wire 1 27, .virt 38153),
    (.wire 0 31, .virt 38153),
    (.virt 38154, .wire 1 32),
    (.virt 38156, .wire 1 33),
    (.virt 38154, .wire 1 34),
    (.virt 38158, .wire 1 36),
    (.virt 38160, .wire 1 37),
    (.virt 38158, .wire 1 38),
    (.wire 1 35, .wire 1 40),
    (.wire 1 39, .wire 1 41)
  ]

/-- `publicBatchWrapper4.copies`, items `64..96`. -/
def publicBatchWrapper4.copies2 : List (Target × Target) := [
    (.wire 1 35, .wire 1 42),
    (.virt 38152, .wire 0 32),
    (.virt 38152, .wire 0 33),
    (.virt 38162, .wire 0 34),
    (.virt 38162, .wire 1 44),
    (.virt 19025, .wire 1 45),
    (.virt 38162, .wire 1 46),
    (.virt 19025, .wire 1 48),
    (.virt 38163, .wire 1 49),
    (.virt 19025, .wire 1 50),
    (.wire 1 51, .wire 0 36),
    (.virt 38152, .wire 0 37),
    (.wire 0 35, .wire 0 38),
    (.wire 1 47, .virt 38153),
    (.wire 0 39, .virt 38153),
    (.virt 38152, .wire 0 40),
    (.virt 38152, .wire 0 41),
    (.virt 38164, .wire 0 42),
    (.virt 38164, .wire 1 52),
    (.virt 19026, .wire 1 53),
    (.virt 38164, .wire 1 54),
    (.virt 19026, .wire 1 56),
    (.virt 38165, .wire 1 57),
    (.virt 19026, .wire 1 58),
    (.wire 1 59, .wire 0 44),
    (.virt 38152, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.wire 1 55, .virt 38153),
    (.wire 0 47, .virt 38153),
    (.virt 38152, .wire 0 48),
    (.virt 38152, .wire 0 49),
    (.virt 38166, .wire 0 50)
  ]

/-- `publicBatchWrapper4.copies`, items `96..128`. -/
def publicBatchWrapper4.copies3 : List (Target × Target) := [
    (.virt 38166, .wire 1 60),
    (.virt 19027, .wire 1 61),
    (.virt 38166, .wire 1 62),
    (.virt 19027, .wire 1 64),
    (.virt 38167, .wire 1 65),
    (.virt 19027, .wire 1 66),
    (.wire 1 67, .wire 0 52),
    (.virt 38152, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.wire 1 63, .virt 38153),
    (.wire 0 55, .virt 38153),
    (.virt 38152, .wire 0 56),
    (.virt 38152, .wire 0 57),
    (.virt 38168, .wire 0 58),
    (.virt 38168, .wire 1 68),
    (.virt 19028, .wire 1 69),
    (.virt 38168, .wire 1 70),
    (.virt 19028, .wire 1 72),
    (.virt 38169, .wire 1 73),
    (.virt 19028, .wire 1 74),
    (.wire 1 75, .wire 0 60),
    (.virt 38152, .wire 0 61),
    (.wire 0 59, .wire 0 62),
    (.wire 1 71, .virt 38153),
    (.wire 0 63, .virt 38153),
    (.virt 38162, .wire 1 76),
    (.virt 38164, .wire 1 77),
    (.virt 38162, .wire 1 78),
    (.virt 38166, .wire 2 0),
    (.virt 38168, .wire 2 1),
    (.virt 38166, .wire 2 2),
    (.wire 1 79, .wire 2 4)
  ]

/-- `publicBatchWrapper4.copies`, items `128..160`. -/
def publicBatchWrapper4.copies4 : List (Target × Target) := [
    (.wire 2 3, .wire 2 5),
    (.wire 1 79, .wire 2 6),
    (.virt 38152, .wire 0 64),
    (.virt 38152, .wire 0 65),
    (.virt 38170, .wire 0 66),
    (.virt 38170, .wire 2 8),
    (.virt 28562, .wire 2 9),
    (.virt 38170, .wire 2 10),
    (.virt 28562, .wire 2 12),
    (.virt 38171, .wire 2 13),
    (.virt 28562, .wire 2 14),
    (.wire 2 15, .wire 0 68),
    (.virt 38152, .wire 0 69),
    (.wire 0 67, .wire 0 70),
    (.wire 2 11, .virt 38153),
    (.wire 0 71, .virt 38153),
    (.virt 38152, .wire 0 72),
    (.virt 38152, .wire 0 73),
    (.virt 38172, .wire 0 74),
    (.virt 38172, .wire 2 16),
    (.virt 28563, .wire 2 17),
    (.virt 38172, .wire 2 18),
    (.virt 28563, .wire 2 20),
    (.virt 38173, .wire 2 21),
    (.virt 28563, .wire 2 22),
    (.wire 2 23, .wire 0 76),
    (.virt 38152, .wire 0 77),
    (.wire 0 75, .wire 0 78),
    (.wire 2 19, .virt 38153),
    (.wire 0 79, .virt 38153),
    (.virt 38152, .wire 3 0),
    (.virt 38152, .wire 3 1)
  ]

/-- `publicBatchWrapper4.copies`, items `160..192`. -/
def publicBatchWrapper4.copies5 : List (Target × Target) := [
    (.virt 38174, .wire 3 2),
    (.virt 38174, .wire 2 24),
    (.virt 28564, .wire 2 25),
    (.virt 38174, .wire 2 26),
    (.virt 28564, .wire 2 28),
    (.virt 38175, .wire 2 29),
    (.virt 28564, .wire 2 30),
    (.wire 2 31, .wire 3 4),
    (.virt 38152, .wire 3 5),
    (.wire 3 3, .wire 3 6),
    (.wire 2 27, .virt 38153),
    (.wire 3 7, .virt 38153),
    (.virt 38152, .wire 3 8),
    (.virt 38152, .wire 3 9),
    (.virt 38176, .wire 3 10),
    (.virt 38176, .wire 2 32),
    (.virt 28565, .wire 2 33),
    (.virt 38176, .wire 2 34),
    (.virt 28565, .wire 2 36),
    (.virt 38177, .wire 2 37),
    (.virt 28565, .wire 2 38),
    (.wire 2 39, .wire 3 12),
    (.virt 38152, .wire 3 13),
    (.wire 3 11, .wire 3 14),
    (.wire 2 35, .virt 38153),
    (.wire 3 15, .virt 38153),
    (.virt 38170, .wire 2 40),
    (.virt 38172, .wire 2 41),
    (.virt 38170, .wire 2 42),
    (.virt 38174, .wire 2 44),
    (.virt 38176, .wire 2 45),
    (.virt 38174, .wire 2 46)
  ]

/-- `publicBatchWrapper4.copies`, items `192..224`. -/
def publicBatchWrapper4.copies6 : List (Target × Target) := [
    (.wire 2 43, .wire 2 48),
    (.wire 2 47, .wire 2 49),
    (.wire 2 43, .wire 2 50),
    (.virt 38152, .wire 3 16),
    (.virt 38152, .wire 3 17),
    (.virt 38178, .wire 3 18),
    (.virt 38178, .wire 2 52),
    (.virt 38099, .wire 2 53),
    (.virt 38178, .wire 2 54),
    (.virt 38099, .wire 2 56),
    (.virt 38179, .wire 2 57),
    (.virt 38099, .wire 2 58),
    (.wire 2 59, .wire 3 20),
    (.virt 38152, .wire 3 21),
    (.wire 3 19, .wire 3 22),
    (.wire 2 55, .virt 38153),
    (.wire 3 23, .virt 38153),
    (.virt 38152, .wire 3 24),
    (.virt 38152, .wire 3 25),
    (.virt 38180, .wire 3 26),
    (.virt 38180, .wire 2 60),
    (.virt 38100, .wire 2 61),
    (.virt 38180, .wire 2 62),
    (.virt 38100, .wire 2 64),
    (.virt 38181, .wire 2 65),
    (.virt 38100, .wire 2 66),
    (.wire 2 67, .wire 3 28),
    (.virt 38152, .wire 3 29),
    (.wire 3 27, .wire 3 30),
    (.wire 2 63, .virt 38153),
    (.wire 3 31, .virt 38153),
    (.virt 38152, .wire 3 32)
  ]

/-- `publicBatchWrapper4.copies`, items `224..256`. -/
def publicBatchWrapper4.copies7 : List (Target × Target) := [
    (.virt 38152, .wire 3 33),
    (.virt 38182, .wire 3 34),
    (.virt 38182, .wire 2 68),
    (.virt 38101, .wire 2 69),
    (.virt 38182, .wire 2 70),
    (.virt 38101, .wire 2 72),
    (.virt 38183, .wire 2 73),
    (.virt 38101, .wire 2 74),
    (.wire 2 75, .wire 3 36),
    (.virt 38152, .wire 3 37),
    (.wire 3 35, .wire 3 38),
    (.wire 2 71, .virt 38153),
    (.wire 3 39, .virt 38153),
    (.virt 38152, .wire 3 40),
    (.virt 38152, .wire 3 41),
    (.virt 38184, .wire 3 42),
    (.virt 38184, .wire 2 76),
    (.virt 38102, .wire 2 77),
    (.virt 38184, .wire 2 78),
    (.virt 38102, .wire 4 0),
    (.virt 38185, .wire 4 1),
    (.virt 38102, .wire 4 2),
    (.wire 4 3, .wire 3 44),
    (.virt 38152, .wire 3 45),
    (.wire 3 43, .wire 3 46),
    (.wire 2 79, .virt 38153),
    (.wire 3 47, .virt 38153),
    (.virt 38178, .wire 4 4),
    (.virt 38180, .wire 4 5),
    (.virt 38178, .wire 4 6),
    (.virt 38182, .wire 4 8),
    (.virt 38184, .wire 4 9)
  ]

/-- `publicBatchWrapper4.copies`, items `256..288`. -/
def publicBatchWrapper4.copies8 : List (Target × Target) := [
    (.virt 38182, .wire 4 10),
    (.wire 4 7, .wire 4 12),
    (.wire 4 11, .wire 4 13),
    (.wire 4 7, .wire 4 14),
    (.virt 38152, .wire 3 48),
    (.virt 38152, .wire 3 49),
    (.wire 1 43, .wire 3 50),
    (.wire 3 51, .wire 3 52),
    (.virt 9488, .wire 3 53),
    (.virt 38153, .wire 3 54),
    (.wire 3 51, .wire 3 56),
    (.virt 9489, .wire 3 57),
    (.virt 38153, .wire 3 58),
    (.wire 3 51, .wire 3 60),
    (.virt 9490, .wire 3 61),
    (.virt 38153, .wire 3 62),
    (.wire 3 51, .wire 3 64),
    (.virt 9491, .wire 3 65),
    (.virt 38153, .wire 3 66),
    (.wire 3 51, .wire 3 68),
    (.virt 9492, .wire 3 69),
    (.virt 38153, .wire 3 70),
    (.wire 3 51, .wire 3 72),
    (.virt 9486, .wire 3 73),
    (.virt 38153, .wire 3 74),
    (.wire 3 51, .wire 3 76),
    (.virt 9487, .wire 3 77),
    (.virt 38153, .wire 3 78),
    (.virt 38152, .wire 5 0),
    (.virt 38152, .wire 5 1),
    (.wire 2 7, .wire 5 2),
    (.virt 38152, .wire 5 4)
  ]

/-- `publicBatchWrapper4.copies`, items `288..320`. -/
def publicBatchWrapper4.copies9 : List (Target × Target) := [
    (.virt 38152, .wire 5 5),
    (.wire 3 51, .wire 5 6),
    (.wire 5 3, .wire 4 16),
    (.wire 5 7, .wire 4 17),
    (.wire 5 3, .wire 4 18),
    (.wire 4 19, .wire 5 8),
    (.wire 3 55, .wire 5 9),
    (.wire 3 55, .wire 5 10),
    (.wire 4 19, .wire 5 12),
    (.virt 19025, .wire 5 13),
    (.wire 5 11, .wire 5 14),
    (.wire 4 19, .wire 5 16),
    (.wire 3 59, .wire 5 17),
    (.wire 3 59, .wire 5 18),
    (.wire 4 19, .wire 5 20),
    (.virt 19026, .wire 5 21),
    (.wire 5 19, .wire 5 22),
    (.wire 4 19, .wire 5 24),
    (.wire 3 63, .wire 5 25),
    (.wire 3 63, .wire 5 26),
    (.wire 4 19, .wire 5 28),
    (.virt 19027, .wire 5 29),
    (.wire 5 27, .wire 5 30),
    (.wire 4 19, .wire 5 32),
    (.wire 3 67, .wire 5 33),
    (.wire 3 67, .wire 5 34),
    (.wire 4 19, .wire 5 36),
    (.virt 19028, .wire 5 37),
    (.wire 5 35, .wire 5 38),
    (.wire 4 19, .wire 5 40),
    (.wire 3 71, .wire 5 41),
    (.wire 3 71, .wire 5 42)
  ]

/-- `publicBatchWrapper4.copies`, items `320..352`. -/
def publicBatchWrapper4.copies10 : List (Target × Target) := [
    (.wire 4 19, .wire 5 44),
    (.virt 19029, .wire 5 45),
    (.wire 5 43, .wire 5 46),
    (.wire 4 19, .wire 5 48),
    (.wire 3 75, .wire 5 49),
    (.wire 3 75, .wire 5 50),
    (.wire 4 19, .wire 5 52),
    (.virt 19023, .wire 5 53),
    (.wire 5 51, .wire 5 54),
    (.wire 4 19, .wire 5 56),
    (.wire 3 79, .wire 5 57),
    (.wire 3 79, .wire 5 58),
    (.wire 4 19, .wire 5 60),
    (.virt 19024, .wire 5 61),
    (.wire 5 59, .wire 5 62),
    (.wire 3 51, .wire 6 0),
    (.wire 5 3, .wire 6 1),
    (.wire 3 51, .wire 6 2),
    (.wire 6 3, .wire 7 0),
    (.virt 38152, .wire 7 1),
    (.wire 5 3, .wire 7 2),
    (.virt 38152, .wire 5 64),
    (.virt 38152, .wire 5 65),
    (.wire 2 51, .wire 5 66),
    (.virt 38152, .wire 5 68),
    (.virt 38152, .wire 5 69),
    (.wire 7 3, .wire 5 70),
    (.wire 5 67, .wire 4 20),
    (.wire 5 71, .wire 4 21),
    (.wire 5 67, .wire 4 22),
    (.wire 4 23, .wire 5 72),
    (.wire 5 15, .wire 5 73)
  ]

/-- `publicBatchWrapper4.copies`, items `352..384`. -/
def publicBatchWrapper4.copies11 : List (Target × Target) := [
    (.wire 5 15, .wire 5 74),
    (.wire 4 23, .wire 5 76),
    (.virt 28562, .wire 5 77),
    (.wire 5 75, .wire 5 78),
    (.wire 4 23, .wire 8 0),
    (.wire 5 23, .wire 8 1),
    (.wire 5 23, .wire 8 2),
    (.wire 4 23, .wire 8 4),
    (.virt 28563, .wire 8 5),
    (.wire 8 3, .wire 8 6),
    (.wire 4 23, .wire 8 8),
    (.wire 5 31, .wire 8 9),
    (.wire 5 31, .wire 8 10),
    (.wire 4 23, .wire 8 12),
    (.virt 28564, .wire 8 13),
    (.wire 8 11, .wire 8 14),
    (.wire 4 23, .wire 8 16),
    (.wire 5 39, .wire 8 17),
    (.wire 5 39, .wire 8 18),
    (.wire 4 23, .wire 8 20),
    (.virt 28565, .wire 8 21),
    (.wire 8 19, .wire 8 22),
    (.wire 4 23, .wire 8 24),
    (.wire 5 47, .wire 8 25),
    (.wire 5 47, .wire 8 26),
    (.wire 4 23, .wire 8 28),
    (.virt 28566, .wire 8 29),
    (.wire 8 27, .wire 8 30),
    (.wire 4 23, .wire 8 32),
    (.wire 5 55, .wire 8 33),
    (.wire 5 55, .wire 8 34),
    (.wire 4 23, .wire 8 36)
  ]

/-- `publicBatchWrapper4.copies`, items `384..416`. -/
def publicBatchWrapper4.copies12 : List (Target × Target) := [
    (.virt 28560, .wire 8 37),
    (.wire 8 35, .wire 8 38),
    (.wire 4 23, .wire 8 40),
    (.wire 5 63, .wire 8 41),
    (.wire 5 63, .wire 8 42),
    (.wire 4 23, .wire 8 44),
    (.virt 28561, .wire 8 45),
    (.wire 8 43, .wire 8 46),
    (.wire 7 3, .wire 6 4),
    (.wire 5 67, .wire 6 5),
    (.wire 7 3, .wire 6 6),
    (.wire 6 7, .wire 7 4),
    (.virt 38152, .wire 7 5),
    (.wire 5 67, .wire 7 6),
    (.virt 38152, .wire 8 48),
    (.virt 38152, .wire 8 49),
    (.wire 4 15, .wire 8 50),
    (.virt 38152, .wire 8 52),
    (.virt 38152, .wire 8 53),
    (.wire 7 7, .wire 8 54),
    (.wire 8 51, .wire 4 24),
    (.wire 8 55, .wire 4 25),
    (.wire 8 51, .wire 4 26),
    (.wire 4 27, .wire 8 56),
    (.wire 5 79, .wire 8 57),
    (.wire 5 79, .wire 8 58),
    (.wire 4 27, .wire 8 60),
    (.virt 38099, .wire 8 61),
    (.wire 8 59, .wire 8 62),
    (.wire 4 27, .wire 8 64),
    (.wire 8 7, .wire 8 65),
    (.wire 8 7, .wire 8 66)
  ]

/-- `publicBatchWrapper4.copies`, items `416..448`. -/
def publicBatchWrapper4.copies13 : List (Target × Target) := [
    (.wire 4 27, .wire 8 68),
    (.virt 38100, .wire 8 69),
    (.wire 8 67, .wire 8 70),
    (.wire 4 27, .wire 8 72),
    (.wire 8 15, .wire 8 73),
    (.wire 8 15, .wire 8 74),
    (.wire 4 27, .wire 8 76),
    (.virt 38101, .wire 8 77),
    (.wire 8 75, .wire 8 78),
    (.wire 4 27, .wire 9 0),
    (.wire 8 23, .wire 9 1),
    (.wire 8 23, .wire 9 2),
    (.wire 4 27, .wire 9 4),
    (.virt 38102, .wire 9 5),
    (.wire 9 3, .wire 9 6),
    (.wire 4 27, .wire 9 8),
    (.wire 8 31, .wire 9 9),
    (.wire 8 31, .wire 9 10),
    (.wire 4 27, .wire 9 12),
    (.virt 38103, .wire 9 13),
    (.wire 9 11, .wire 9 14),
    (.wire 4 27, .wire 9 16),
    (.wire 8 39, .wire 9 17),
    (.wire 8 39, .wire 9 18),
    (.wire 4 27, .wire 9 20),
    (.virt 38097, .wire 9 21),
    (.wire 9 19, .wire 9 22),
    (.wire 4 27, .wire 9 24),
    (.wire 8 47, .wire 9 25),
    (.wire 8 47, .wire 9 26),
    (.wire 4 27, .wire 9 28),
    (.virt 38098, .wire 9 29)
  ]

/-- `publicBatchWrapper4.copies`, items `448..480`. -/
def publicBatchWrapper4.copies14 : List (Target × Target) := [
    (.wire 9 27, .wire 9 30),
    (.wire 7 7, .wire 6 8),
    (.wire 8 51, .wire 6 9),
    (.wire 7 7, .wire 6 10),
    (.wire 6 11, .wire 7 8),
    (.virt 38152, .wire 7 9),
    (.wire 8 51, .wire 7 10),
    (.virt 38152, .wire 9 32),
    (.virt 38152, .wire 9 33),
    (.virt 38186, .wire 9 34),
    (.virt 9486, .wire 9 36),
    (.virt 38152, .wire 9 37),
    (.wire 9 23, .wire 9 38),
    (.virt 38186, .wire 4 28),
    (.wire 9 39, .wire 4 29),
    (.virt 38186, .wire 4 30),
    (.wire 9 39, .wire 4 32),
    (.virt 38187, .wire 4 33),
    (.wire 9 39, .wire 4 34),
    (.wire 4 35, .wire 9 40),
    (.virt 38152, .wire 9 41),
    (.wire 9 35, .wire 9 42),
    (.wire 4 31, .virt 38153),
    (.wire 9 43, .virt 38153),
    (.wire 1 43, .wire 6 12),
    (.virt 38186, .wire 6 13),
    (.wire 1 43, .wire 6 14),
    (.wire 6 15, .wire 7 12),
    (.virt 38152, .wire 7 13),
    (.virt 38186, .wire 7 14),
    (.wire 7 15, .virt 38152),
    (.virt 38152, .wire 9 44)
  ]

/-- `publicBatchWrapper4.copies`, items `480..512`. -/
def publicBatchWrapper4.copies15 : List (Target × Target) := [
    (.virt 38152, .wire 9 45),
    (.virt 38188, .wire 9 46),
    (.virt 9487, .wire 9 48),
    (.virt 38152, .wire 9 49),
    (.wire 9 31, .wire 9 50),
    (.virt 38188, .wire 4 36),
    (.wire 9 51, .wire 4 37),
    (.virt 38188, .wire 4 38),
    (.wire 9 51, .wire 4 40),
    (.virt 38189, .wire 4 41),
    (.wire 9 51, .wire 4 42),
    (.wire 4 43, .wire 9 52),
    (.virt 38152, .wire 9 53),
    (.wire 9 47, .wire 9 54),
    (.wire 4 39, .virt 38153),
    (.wire 9 55, .virt 38153),
    (.wire 1 43, .wire 6 16),
    (.virt 38188, .wire 6 17),
    (.wire 1 43, .wire 6 18),
    (.wire 6 19, .wire 7 16),
    (.virt 38152, .wire 7 17),
    (.virt 38188, .wire 7 18),
    (.wire 7 19, .virt 38152),
    (.virt 38152, .wire 9 56),
    (.virt 38152, .wire 9 57),
    (.virt 38190, .wire 9 58),
    (.virt 9488, .wire 9 60),
    (.virt 38152, .wire 9 61),
    (.wire 8 63, .wire 9 62),
    (.virt 38190, .wire 4 44),
    (.wire 9 63, .wire 4 45),
    (.virt 38190, .wire 4 46)
  ]

/-- `publicBatchWrapper4.copies`, items `512..544`. -/
def publicBatchWrapper4.copies16 : List (Target × Target) := [
    (.wire 9 63, .wire 4 48),
    (.virt 38191, .wire 4 49),
    (.wire 9 63, .wire 4 50),
    (.wire 4 51, .wire 9 64),
    (.virt 38152, .wire 9 65),
    (.wire 9 59, .wire 9 66),
    (.wire 4 47, .virt 38153),
    (.wire 9 67, .virt 38153),
    (.virt 38152, .wire 9 68),
    (.virt 38152, .wire 9 69),
    (.virt 38192, .wire 9 70),
    (.virt 9489, .wire 9 72),
    (.virt 38152, .wire 9 73),
    (.wire 8 71, .wire 9 74),
    (.virt 38192, .wire 4 52),
    (.wire 9 75, .wire 4 53),
    (.virt 38192, .wire 4 54),
    (.wire 9 75, .wire 4 56),
    (.virt 38193, .wire 4 57),
    (.wire 9 75, .wire 4 58),
    (.wire 4 59, .wire 9 76),
    (.virt 38152, .wire 9 77),
    (.wire 9 71, .wire 9 78),
    (.wire 4 55, .virt 38153),
    (.wire 9 79, .virt 38153),
    (.virt 38152, .wire 10 0),
    (.virt 38152, .wire 10 1),
    (.virt 38194, .wire 10 2),
    (.virt 9490, .wire 10 4),
    (.virt 38152, .wire 10 5),
    (.wire 8 79, .wire 10 6),
    (.virt 38194, .wire 4 60)
  ]

/-- `publicBatchWrapper4.copies`, items `544..576`. -/
def publicBatchWrapper4.copies17 : List (Target × Target) := [
    (.wire 10 7, .wire 4 61),
    (.virt 38194, .wire 4 62),
    (.wire 10 7, .wire 4 64),
    (.virt 38195, .wire 4 65),
    (.wire 10 7, .wire 4 66),
    (.wire 4 67, .wire 10 8),
    (.virt 38152, .wire 10 9),
    (.wire 10 3, .wire 10 10),
    (.wire 4 63, .virt 38153),
    (.wire 10 11, .virt 38153),
    (.virt 38152, .wire 10 12),
    (.virt 38152, .wire 10 13),
    (.virt 38196, .wire 10 14),
    (.virt 9491, .wire 10 16),
    (.virt 38152, .wire 10 17),
    (.wire 9 7, .wire 10 18),
    (.virt 38196, .wire 4 68),
    (.wire 10 19, .wire 4 69),
    (.virt 38196, .wire 4 70),
    (.wire 10 19, .wire 4 72),
    (.virt 38197, .wire 4 73),
    (.wire 10 19, .wire 4 74),
    (.wire 4 75, .wire 10 20),
    (.virt 38152, .wire 10 21),
    (.wire 10 15, .wire 10 22),
    (.wire 4 71, .virt 38153),
    (.wire 10 23, .virt 38153),
    (.virt 38190, .wire 4 76),
    (.virt 38192, .wire 4 77),
    (.virt 38190, .wire 4 78),
    (.virt 38194, .wire 11 0),
    (.virt 38196, .wire 11 1)
  ]

/-- `publicBatchWrapper4.copies`, items `576..608`. -/
def publicBatchWrapper4.copies18 : List (Target × Target) := [
    (.virt 38194, .wire 11 2),
    (.wire 4 79, .wire 11 4),
    (.wire 11 3, .wire 11 5),
    (.wire 4 79, .wire 11 6),
    (.wire 1 43, .wire 6 20),
    (.wire 11 7, .wire 6 21),
    (.wire 1 43, .wire 6 22),
    (.wire 6 23, .wire 7 20),
    (.virt 38152, .wire 7 21),
    (.wire 11 7, .wire 7 22),
    (.wire 7 23, .virt 38152),
    (.virt 38152, .wire 10 24),
    (.virt 38152, .wire 10 25),
    (.virt 38198, .wire 10 26),
    (.virt 19023, .wire 10 28),
    (.virt 38152, .wire 10 29),
    (.wire 9 23, .wire 10 30),
    (.virt 38198, .wire 11 8),
    (.wire 10 31, .wire 11 9),
    (.virt 38198, .wire 11 10),
    (.wire 10 31, .wire 11 12),
    (.virt 38199, .wire 11 13),
    (.wire 10 31, .wire 11 14),
    (.wire 11 15, .wire 10 32),
    (.virt 38152, .wire 10 33),
    (.wire 10 27, .wire 10 34),
    (.wire 11 11, .virt 38153),
    (.wire 10 35, .virt 38153),
    (.wire 2 7, .wire 6 24),
    (.virt 38198, .wire 6 25),
    (.wire 2 7, .wire 6 26),
    (.wire 6 27, .wire 7 24)
  ]

/-- `publicBatchWrapper4.copies`, items `608..640`. -/
def publicBatchWrapper4.copies19 : List (Target × Target) := [
    (.virt 38152, .wire 7 25),
    (.virt 38198, .wire 7 26),
    (.wire 7 27, .virt 38152),
    (.virt 38152, .wire 10 36),
    (.virt 38152, .wire 10 37),
    (.virt 38200, .wire 10 38),
    (.virt 19024, .wire 10 40),
    (.virt 38152, .wire 10 41),
    (.wire 9 31, .wire 10 42),
    (.virt 38200, .wire 11 16),
    (.wire 10 43, .wire 11 17),
    (.virt 38200, .wire 11 18),
    (.wire 10 43, .wire 11 20),
    (.virt 38201, .wire 11 21),
    (.wire 10 43, .wire 11 22),
    (.wire 11 23, .wire 10 44),
    (.virt 38152, .wire 10 45),
    (.wire 10 39, .wire 10 46),
    (.wire 11 19, .virt 38153),
    (.wire 10 47, .virt 38153),
    (.wire 2 7, .wire 6 28),
    (.virt 38200, .wire 6 29),
    (.wire 2 7, .wire 6 30),
    (.wire 6 31, .wire 7 28),
    (.virt 38152, .wire 7 29),
    (.virt 38200, .wire 7 30),
    (.wire 7 31, .virt 38152),
    (.virt 38152, .wire 10 48),
    (.virt 38152, .wire 10 49),
    (.virt 38202, .wire 10 50),
    (.virt 19025, .wire 10 52),
    (.virt 38152, .wire 10 53)
  ]

/-- `publicBatchWrapper4.copies`, items `640..672`. -/
def publicBatchWrapper4.copies20 : List (Target × Target) := [
    (.wire 8 63, .wire 10 54),
    (.virt 38202, .wire 11 24),
    (.wire 10 55, .wire 11 25),
    (.virt 38202, .wire 11 26),
    (.wire 10 55, .wire 11 28),
    (.virt 38203, .wire 11 29),
    (.wire 10 55, .wire 11 30),
    (.wire 11 31, .wire 10 56),
    (.virt 38152, .wire 10 57),
    (.wire 10 51, .wire 10 58),
    (.wire 11 27, .virt 38153),
    (.wire 10 59, .virt 38153),
    (.virt 38152, .wire 10 60),
    (.virt 38152, .wire 10 61),
    (.virt 38204, .wire 10 62),
    (.virt 19026, .wire 10 64),
    (.virt 38152, .wire 10 65),
    (.wire 8 71, .wire 10 66),
    (.virt 38204, .wire 11 32),
    (.wire 10 67, .wire 11 33),
    (.virt 38204, .wire 11 34),
    (.wire 10 67, .wire 11 36),
    (.virt 38205, .wire 11 37),
    (.wire 10 67, .wire 11 38),
    (.wire 11 39, .wire 10 68),
    (.virt 38152, .wire 10 69),
    (.wire 10 63, .wire 10 70),
    (.wire 11 35, .virt 38153),
    (.wire 10 71, .virt 38153),
    (.virt 38152, .wire 10 72),
    (.virt 38152, .wire 10 73),
    (.virt 38206, .wire 10 74)
  ]

/-- `publicBatchWrapper4.copies`, items `672..704`. -/
def publicBatchWrapper4.copies21 : List (Target × Target) := [
    (.virt 19027, .wire 10 76),
    (.virt 38152, .wire 10 77),
    (.wire 8 79, .wire 10 78),
    (.virt 38206, .wire 11 40),
    (.wire 10 79, .wire 11 41),
    (.virt 38206, .wire 11 42),
    (.wire 10 79, .wire 11 44),
    (.virt 38207, .wire 11 45),
    (.wire 10 79, .wire 11 46),
    (.wire 11 47, .wire 12 0),
    (.virt 38152, .wire 12 1),
    (.wire 10 75, .wire 12 2),
    (.wire 11 43, .virt 38153),
    (.wire 12 3, .virt 38153),
    (.virt 38152, .wire 12 4),
    (.virt 38152, .wire 12 5),
    (.virt 38208, .wire 12 6),
    (.virt 19028, .wire 12 8),
    (.virt 38152, .wire 12 9),
    (.wire 9 7, .wire 12 10),
    (.virt 38208, .wire 11 48),
    (.wire 12 11, .wire 11 49),
    (.virt 38208, .wire 11 50),
    (.wire 12 11, .wire 11 52),
    (.virt 38209, .wire 11 53),
    (.wire 12 11, .wire 11 54),
    (.wire 11 55, .wire 12 12),
    (.virt 38152, .wire 12 13),
    (.wire 12 7, .wire 12 14),
    (.wire 11 51, .virt 38153),
    (.wire 12 15, .virt 38153),
    (.virt 38202, .wire 11 56)
  ]

/-- `publicBatchWrapper4.copies`, items `704..736`. -/
def publicBatchWrapper4.copies22 : List (Target × Target) := [
    (.virt 38204, .wire 11 57),
    (.virt 38202, .wire 11 58),
    (.virt 38206, .wire 11 60),
    (.virt 38208, .wire 11 61),
    (.virt 38206, .wire 11 62),
    (.wire 11 59, .wire 11 64),
    (.wire 11 63, .wire 11 65),
    (.wire 11 59, .wire 11 66),
    (.wire 2 7, .wire 6 32),
    (.wire 11 67, .wire 6 33),
    (.wire 2 7, .wire 6 34),
    (.wire 6 35, .wire 7 32),
    (.virt 38152, .wire 7 33),
    (.wire 11 67, .wire 7 34),
    (.wire 7 35, .virt 38152),
    (.virt 38152, .wire 12 16),
    (.virt 38152, .wire 12 17),
    (.virt 38210, .wire 12 18),
    (.virt 28560, .wire 12 20),
    (.virt 38152, .wire 12 21),
    (.wire 9 23, .wire 12 22),
    (.virt 38210, .wire 11 68),
    (.wire 12 23, .wire 11 69),
    (.virt 38210, .wire 11 70),
    (.wire 12 23, .wire 11 72),
    (.virt 38211, .wire 11 73),
    (.wire 12 23, .wire 11 74),
    (.wire 11 75, .wire 12 24),
    (.virt 38152, .wire 12 25),
    (.wire 12 19, .wire 12 26),
    (.wire 11 71, .virt 38153),
    (.wire 12 27, .virt 38153)
  ]

/-- `publicBatchWrapper4.copies`, items `736..768`. -/
def publicBatchWrapper4.copies23 : List (Target × Target) := [
    (.wire 2 51, .wire 6 36),
    (.virt 38210, .wire 6 37),
    (.wire 2 51, .wire 6 38),
    (.wire 6 39, .wire 7 36),
    (.virt 38152, .wire 7 37),
    (.virt 38210, .wire 7 38),
    (.wire 7 39, .virt 38152),
    (.virt 38152, .wire 12 28),
    (.virt 38152, .wire 12 29),
    (.virt 38212, .wire 12 30),
    (.virt 28561, .wire 12 32),
    (.virt 38152, .wire 12 33),
    (.wire 9 31, .wire 12 34),
    (.virt 38212, .wire 11 76),
    (.wire 12 35, .wire 11 77),
    (.virt 38212, .wire 11 78),
    (.wire 12 35, .wire 13 0),
    (.virt 38213, .wire 13 1),
    (.wire 12 35, .wire 13 2),
    (.wire 13 3, .wire 12 36),
    (.virt 38152, .wire 12 37),
    (.wire 12 31, .wire 12 38),
    (.wire 11 79, .virt 38153),
    (.wire 12 39, .virt 38153),
    (.wire 2 51, .wire 6 40),
    (.virt 38212, .wire 6 41),
    (.wire 2 51, .wire 6 42),
    (.wire 6 43, .wire 7 40),
    (.virt 38152, .wire 7 41),
    (.virt 38212, .wire 7 42),
    (.wire 7 43, .virt 38152),
    (.virt 38152, .wire 12 40)
  ]

/-- `publicBatchWrapper4.copies`, items `768..800`. -/
def publicBatchWrapper4.copies24 : List (Target × Target) := [
    (.virt 38152, .wire 12 41),
    (.virt 38214, .wire 12 42),
    (.virt 28562, .wire 12 44),
    (.virt 38152, .wire 12 45),
    (.wire 8 63, .wire 12 46),
    (.virt 38214, .wire 13 4),
    (.wire 12 47, .wire 13 5),
    (.virt 38214, .wire 13 6),
    (.wire 12 47, .wire 13 8),
    (.virt 38215, .wire 13 9),
    (.wire 12 47, .wire 13 10),
    (.wire 13 11, .wire 12 48),
    (.virt 38152, .wire 12 49),
    (.wire 12 43, .wire 12 50),
    (.wire 13 7, .virt 38153),
    (.wire 12 51, .virt 38153),
    (.virt 38152, .wire 12 52),
    (.virt 38152, .wire 12 53),
    (.virt 38216, .wire 12 54),
    (.virt 28563, .wire 12 56),
    (.virt 38152, .wire 12 57),
    (.wire 8 71, .wire 12 58),
    (.virt 38216, .wire 13 12),
    (.wire 12 59, .wire 13 13),
    (.virt 38216, .wire 13 14),
    (.wire 12 59, .wire 13 16),
    (.virt 38217, .wire 13 17),
    (.wire 12 59, .wire 13 18),
    (.wire 13 19, .wire 12 60),
    (.virt 38152, .wire 12 61),
    (.wire 12 55, .wire 12 62),
    (.wire 13 15, .virt 38153)
  ]

/-- `publicBatchWrapper4.copies`, items `800..832`. -/
def publicBatchWrapper4.copies25 : List (Target × Target) := [
    (.wire 12 63, .virt 38153),
    (.virt 38152, .wire 12 64),
    (.virt 38152, .wire 12 65),
    (.virt 38218, .wire 12 66),
    (.virt 28564, .wire 12 68),
    (.virt 38152, .wire 12 69),
    (.wire 8 79, .wire 12 70),
    (.virt 38218, .wire 13 20),
    (.wire 12 71, .wire 13 21),
    (.virt 38218, .wire 13 22),
    (.wire 12 71, .wire 13 24),
    (.virt 38219, .wire 13 25),
    (.wire 12 71, .wire 13 26),
    (.wire 13 27, .wire 12 72),
    (.virt 38152, .wire 12 73),
    (.wire 12 67, .wire 12 74),
    (.wire 13 23, .virt 38153),
    (.wire 12 75, .virt 38153),
    (.virt 38152, .wire 12 76),
    (.virt 38152, .wire 12 77),
    (.virt 38220, .wire 12 78),
    (.virt 28565, .wire 14 0),
    (.virt 38152, .wire 14 1),
    (.wire 9 7, .wire 14 2),
    (.virt 38220, .wire 13 28),
    (.wire 14 3, .wire 13 29),
    (.virt 38220, .wire 13 30),
    (.wire 14 3, .wire 13 32),
    (.virt 38221, .wire 13 33),
    (.wire 14 3, .wire 13 34),
    (.wire 13 35, .wire 14 4),
    (.virt 38152, .wire 14 5)
  ]

/-- `publicBatchWrapper4.copies`, items `832..864`. -/
def publicBatchWrapper4.copies26 : List (Target × Target) := [
    (.wire 12 79, .wire 14 6),
    (.wire 13 31, .virt 38153),
    (.wire 14 7, .virt 38153),
    (.virt 38214, .wire 13 36),
    (.virt 38216, .wire 13 37),
    (.virt 38214, .wire 13 38),
    (.virt 38218, .wire 13 40),
    (.virt 38220, .wire 13 41),
    (.virt 38218, .wire 13 42),
    (.wire 13 39, .wire 13 44),
    (.wire 13 43, .wire 13 45),
    (.wire 13 39, .wire 13 46),
    (.wire 2 51, .wire 6 44),
    (.wire 13 47, .wire 6 45),
    (.wire 2 51, .wire 6 46),
    (.wire 6 47, .wire 7 44),
    (.virt 38152, .wire 7 45),
    (.wire 13 47, .wire 7 46),
    (.wire 7 47, .virt 38152),
    (.virt 38152, .wire 14 8),
    (.virt 38152, .wire 14 9),
    (.virt 38222, .wire 14 10),
    (.virt 38097, .wire 14 12),
    (.virt 38152, .wire 14 13),
    (.wire 9 23, .wire 14 14),
    (.virt 38222, .wire 13 48),
    (.wire 14 15, .wire 13 49),
    (.virt 38222, .wire 13 50),
    (.wire 14 15, .wire 13 52),
    (.virt 38223, .wire 13 53),
    (.wire 14 15, .wire 13 54),
    (.wire 13 55, .wire 14 16)
  ]

/-- `publicBatchWrapper4.copies`, items `864..896`. -/
def publicBatchWrapper4.copies27 : List (Target × Target) := [
    (.virt 38152, .wire 14 17),
    (.wire 14 11, .wire 14 18),
    (.wire 13 51, .virt 38153),
    (.wire 14 19, .virt 38153),
    (.wire 4 15, .wire 6 48),
    (.virt 38222, .wire 6 49),
    (.wire 4 15, .wire 6 50),
    (.wire 6 51, .wire 7 48),
    (.virt 38152, .wire 7 49),
    (.virt 38222, .wire 7 50),
    (.wire 7 51, .virt 38152),
    (.virt 38152, .wire 14 20),
    (.virt 38152, .wire 14 21),
    (.virt 38224, .wire 14 22),
    (.virt 38098, .wire 14 24),
    (.virt 38152, .wire 14 25),
    (.wire 9 31, .wire 14 26),
    (.virt 38224, .wire 13 56),
    (.wire 14 27, .wire 13 57),
    (.virt 38224, .wire 13 58),
    (.wire 14 27, .wire 13 60),
    (.virt 38225, .wire 13 61),
    (.wire 14 27, .wire 13 62),
    (.wire 13 63, .wire 14 28),
    (.virt 38152, .wire 14 29),
    (.wire 14 23, .wire 14 30),
    (.wire 13 59, .virt 38153),
    (.wire 14 31, .virt 38153),
    (.wire 4 15, .wire 6 52),
    (.virt 38224, .wire 6 53),
    (.wire 4 15, .wire 6 54),
    (.wire 6 55, .wire 7 52)
  ]

/-- `publicBatchWrapper4.copies`, items `896..928`. -/
def publicBatchWrapper4.copies28 : List (Target × Target) := [
    (.virt 38152, .wire 7 53),
    (.virt 38224, .wire 7 54),
    (.wire 7 55, .virt 38152),
    (.virt 38152, .wire 14 32),
    (.virt 38152, .wire 14 33),
    (.virt 38226, .wire 14 34),
    (.virt 38099, .wire 14 36),
    (.virt 38152, .wire 14 37),
    (.wire 8 63, .wire 14 38),
    (.virt 38226, .wire 13 64),
    (.wire 14 39, .wire 13 65),
    (.virt 38226, .wire 13 66),
    (.wire 14 39, .wire 13 68),
    (.virt 38227, .wire 13 69),
    (.wire 14 39, .wire 13 70),
    (.wire 13 71, .wire 14 40),
    (.virt 38152, .wire 14 41),
    (.wire 14 35, .wire 14 42),
    (.wire 13 67, .virt 38153),
    (.wire 14 43, .virt 38153),
    (.virt 38152, .wire 14 44),
    (.virt 38152, .wire 14 45),
    (.virt 38228, .wire 14 46),
    (.virt 38100, .wire 14 48),
    (.virt 38152, .wire 14 49),
    (.wire 8 71, .wire 14 50),
    (.virt 38228, .wire 13 72),
    (.wire 14 51, .wire 13 73),
    (.virt 38228, .wire 13 74),
    (.wire 14 51, .wire 13 76),
    (.virt 38229, .wire 13 77),
    (.wire 14 51, .wire 13 78)
  ]

/-- `publicBatchWrapper4.copies`, items `928..960`. -/
def publicBatchWrapper4.copies29 : List (Target × Target) := [
    (.wire 13 79, .wire 14 52),
    (.virt 38152, .wire 14 53),
    (.wire 14 47, .wire 14 54),
    (.wire 13 75, .virt 38153),
    (.wire 14 55, .virt 38153),
    (.virt 38152, .wire 14 56),
    (.virt 38152, .wire 14 57),
    (.virt 38230, .wire 14 58),
    (.virt 38101, .wire 14 60),
    (.virt 38152, .wire 14 61),
    (.wire 8 79, .wire 14 62),
    (.virt 38230, .wire 15 0),
    (.wire 14 63, .wire 15 1),
    (.virt 38230, .wire 15 2),
    (.wire 14 63, .wire 15 4),
    (.virt 38231, .wire 15 5),
    (.wire 14 63, .wire 15 6),
    (.wire 15 7, .wire 14 64),
    (.virt 38152, .wire 14 65),
    (.wire 14 59, .wire 14 66),
    (.wire 15 3, .virt 38153),
    (.wire 14 67, .virt 38153),
    (.virt 38152, .wire 14 68),
    (.virt 38152, .wire 14 69),
    (.virt 38232, .wire 14 70),
    (.virt 38102, .wire 14 72),
    (.virt 38152, .wire 14 73),
    (.wire 9 7, .wire 14 74),
    (.virt 38232, .wire 15 8),
    (.wire 14 75, .wire 15 9),
    (.virt 38232, .wire 15 10),
    (.wire 14 75, .wire 15 12)
  ]

/-- `publicBatchWrapper4.copies`, items `960..992`. -/
def publicBatchWrapper4.copies30 : List (Target × Target) := [
    (.virt 38233, .wire 15 13),
    (.wire 14 75, .wire 15 14),
    (.wire 15 15, .wire 14 76),
    (.virt 38152, .wire 14 77),
    (.wire 14 71, .wire 14 78),
    (.wire 15 11, .virt 38153),
    (.wire 14 79, .virt 38153),
    (.virt 38226, .wire 15 16),
    (.virt 38228, .wire 15 17),
    (.virt 38226, .wire 15 18),
    (.virt 38230, .wire 15 20),
    (.virt 38232, .wire 15 21),
    (.virt 38230, .wire 15 22),
    (.wire 15 19, .wire 15 24),
    (.wire 15 23, .wire 15 25),
    (.wire 15 19, .wire 15 26),
    (.wire 4 15, .wire 6 56),
    (.wire 15 27, .wire 6 57),
    (.wire 4 15, .wire 6 58),
    (.wire 6 59, .wire 7 56),
    (.virt 38152, .wire 7 57),
    (.wire 15 27, .wire 7 58),
    (.wire 7 59, .virt 38152),
    (.wire 1 43, .wire 16 0),
    (.virt 9493, .wire 16 1),
    (.virt 9493, .wire 16 2),
    (.wire 1 43, .wire 16 4),
    (.virt 38153, .wire 16 5),
    (.wire 16 3, .wire 16 6),
    (.wire 1 43, .wire 16 8),
    (.virt 9494, .wire 16 9),
    (.virt 9494, .wire 16 10)
  ]

/-- `publicBatchWrapper4.copies`, items `992..1024`. -/
def publicBatchWrapper4.copies31 : List (Target × Target) := [
    (.wire 1 43, .wire 16 12),
    (.virt 38153, .wire 16 13),
    (.wire 16 11, .wire 16 14),
    (.wire 1 43, .wire 16 16),
    (.virt 9495, .wire 16 17),
    (.virt 9495, .wire 16 18),
    (.wire 1 43, .wire 16 20),
    (.virt 38153, .wire 16 21),
    (.wire 16 19, .wire 16 22),
    (.wire 1 43, .wire 16 24),
    (.virt 9496, .wire 16 25),
    (.virt 9496, .wire 16 26),
    (.wire 1 43, .wire 16 28),
    (.virt 38153, .wire 16 29),
    (.wire 16 27, .wire 16 30),
    (.wire 1 43, .wire 16 32),
    (.virt 9497, .wire 16 33),
    (.virt 9497, .wire 16 34),
    (.wire 1 43, .wire 16 36),
    (.virt 38153, .wire 16 37),
    (.wire 16 35, .wire 16 38),
    (.wire 1 43, .wire 16 40),
    (.virt 9498, .wire 16 41),
    (.virt 9498, .wire 16 42),
    (.wire 1 43, .wire 16 44),
    (.virt 38153, .wire 16 45),
    (.wire 16 43, .wire 16 46),
    (.wire 1 43, .wire 16 48),
    (.virt 9499, .wire 16 49),
    (.virt 9499, .wire 16 50),
    (.wire 1 43, .wire 16 52),
    (.virt 38153, .wire 16 53)
  ]

/-- `publicBatchWrapper4.copies`, items `1024..1056`. -/
def publicBatchWrapper4.copies32 : List (Target × Target) := [
    (.wire 16 51, .wire 16 54),
    (.wire 1 43, .wire 16 56),
    (.virt 9500, .wire 16 57),
    (.virt 9500, .wire 16 58),
    (.wire 1 43, .wire 16 60),
    (.virt 38153, .wire 16 61),
    (.wire 16 59, .wire 16 62),
    (.wire 1 43, .wire 16 64),
    (.virt 9501, .wire 16 65),
    (.virt 9501, .wire 16 66),
    (.wire 1 43, .wire 16 68),
    (.virt 38153, .wire 16 69),
    (.wire 16 67, .wire 16 70),
    (.wire 1 43, .wire 16 72),
    (.virt 9502, .wire 16 73),
    (.virt 9502, .wire 16 74),
    (.wire 1 43, .wire 16 76),
    (.virt 38153, .wire 16 77),
    (.wire 16 75, .wire 16 78),
    (.wire 1 43, .wire 17 0),
    (.virt 9503, .wire 17 1),
    (.virt 9503, .wire 17 2),
    (.wire 1 43, .wire 17 4),
    (.virt 38153, .wire 17 5),
    (.wire 17 3, .wire 17 6),
    (.wire 1 43, .wire 17 8),
    (.virt 9504, .wire 17 9),
    (.virt 9504, .wire 17 10),
    (.wire 1 43, .wire 17 12),
    (.virt 38153, .wire 17 13),
    (.wire 17 11, .wire 17 14),
    (.wire 1 43, .wire 17 16)
  ]

/-- `publicBatchWrapper4.copies`, items `1056..1088`. -/
def publicBatchWrapper4.copies33 : List (Target × Target) := [
    (.virt 9505, .wire 17 17),
    (.virt 9505, .wire 17 18),
    (.wire 1 43, .wire 17 20),
    (.virt 38153, .wire 17 21),
    (.wire 17 19, .wire 17 22),
    (.wire 1 43, .wire 17 24),
    (.virt 9506, .wire 17 25),
    (.virt 9506, .wire 17 26),
    (.wire 1 43, .wire 17 28),
    (.virt 38153, .wire 17 29),
    (.wire 17 27, .wire 17 30),
    (.wire 1 43, .wire 17 32),
    (.virt 9507, .wire 17 33),
    (.virt 9507, .wire 17 34),
    (.wire 1 43, .wire 17 36),
    (.virt 38153, .wire 17 37),
    (.wire 17 35, .wire 17 38),
    (.wire 1 43, .wire 17 40),
    (.virt 9508, .wire 17 41),
    (.virt 9508, .wire 17 42),
    (.wire 1 43, .wire 17 44),
    (.virt 38153, .wire 17 45),
    (.wire 17 43, .wire 17 46),
    (.wire 1 43, .wire 17 48),
    (.virt 9509, .wire 17 49),
    (.virt 9509, .wire 17 50),
    (.wire 1 43, .wire 17 52),
    (.virt 38153, .wire 17 53),
    (.wire 17 51, .wire 17 54),
    (.wire 1 43, .wire 17 56),
    (.virt 9510, .wire 17 57),
    (.virt 9510, .wire 17 58)
  ]

/-- `publicBatchWrapper4.copies`, items `1088..1120`. -/
def publicBatchWrapper4.copies34 : List (Target × Target) := [
    (.wire 1 43, .wire 17 60),
    (.virt 38153, .wire 17 61),
    (.wire 17 59, .wire 17 62),
    (.wire 1 43, .wire 17 64),
    (.virt 9511, .wire 17 65),
    (.virt 9511, .wire 17 66),
    (.wire 1 43, .wire 17 68),
    (.virt 38153, .wire 17 69),
    (.wire 17 67, .wire 17 70),
    (.wire 1 43, .wire 17 72),
    (.virt 9512, .wire 17 73),
    (.virt 9512, .wire 17 74),
    (.wire 1 43, .wire 17 76),
    (.virt 38153, .wire 17 77),
    (.wire 17 75, .wire 17 78),
    (.wire 2 7, .wire 18 0),
    (.virt 19030, .wire 18 1),
    (.virt 19030, .wire 18 2),
    (.wire 2 7, .wire 18 4),
    (.virt 38153, .wire 18 5),
    (.wire 18 3, .wire 18 6),
    (.wire 2 7, .wire 18 8),
    (.virt 19031, .wire 18 9),
    (.virt 19031, .wire 18 10),
    (.wire 2 7, .wire 18 12),
    (.virt 38153, .wire 18 13),
    (.wire 18 11, .wire 18 14),
    (.wire 2 7, .wire 18 16),
    (.virt 19032, .wire 18 17),
    (.virt 19032, .wire 18 18),
    (.wire 2 7, .wire 18 20),
    (.virt 38153, .wire 18 21)
  ]

/-- `publicBatchWrapper4.copies`, items `1120..1152`. -/
def publicBatchWrapper4.copies35 : List (Target × Target) := [
    (.wire 18 19, .wire 18 22),
    (.wire 2 7, .wire 18 24),
    (.virt 19033, .wire 18 25),
    (.virt 19033, .wire 18 26),
    (.wire 2 7, .wire 18 28),
    (.virt 38153, .wire 18 29),
    (.wire 18 27, .wire 18 30),
    (.wire 2 7, .wire 18 32),
    (.virt 19034, .wire 18 33),
    (.virt 19034, .wire 18 34),
    (.wire 2 7, .wire 18 36),
    (.virt 38153, .wire 18 37),
    (.wire 18 35, .wire 18 38),
    (.wire 2 7, .wire 18 40),
    (.virt 19035, .wire 18 41),
    (.virt 19035, .wire 18 42),
    (.wire 2 7, .wire 18 44),
    (.virt 38153, .wire 18 45),
    (.wire 18 43, .wire 18 46),
    (.wire 2 7, .wire 18 48),
    (.virt 19036, .wire 18 49),
    (.virt 19036, .wire 18 50),
    (.wire 2 7, .wire 18 52),
    (.virt 38153, .wire 18 53),
    (.wire 18 51, .wire 18 54),
    (.wire 2 7, .wire 18 56),
    (.virt 19037, .wire 18 57),
    (.virt 19037, .wire 18 58),
    (.wire 2 7, .wire 18 60),
    (.virt 38153, .wire 18 61),
    (.wire 18 59, .wire 18 62),
    (.wire 2 7, .wire 18 64)
  ]

/-- `publicBatchWrapper4.copies`, items `1152..1184`. -/
def publicBatchWrapper4.copies36 : List (Target × Target) := [
    (.virt 19038, .wire 18 65),
    (.virt 19038, .wire 18 66),
    (.wire 2 7, .wire 18 68),
    (.virt 38153, .wire 18 69),
    (.wire 18 67, .wire 18 70),
    (.wire 2 7, .wire 18 72),
    (.virt 19039, .wire 18 73),
    (.virt 19039, .wire 18 74),
    (.wire 2 7, .wire 18 76),
    (.virt 38153, .wire 18 77),
    (.wire 18 75, .wire 18 78),
    (.wire 2 7, .wire 19 0),
    (.virt 19040, .wire 19 1),
    (.virt 19040, .wire 19 2),
    (.wire 2 7, .wire 19 4),
    (.virt 38153, .wire 19 5),
    (.wire 19 3, .wire 19 6),
    (.wire 2 7, .wire 19 8),
    (.virt 19041, .wire 19 9),
    (.virt 19041, .wire 19 10),
    (.wire 2 7, .wire 19 12),
    (.virt 38153, .wire 19 13),
    (.wire 19 11, .wire 19 14),
    (.wire 2 7, .wire 19 16),
    (.virt 19042, .wire 19 17),
    (.virt 19042, .wire 19 18),
    (.wire 2 7, .wire 19 20),
    (.virt 38153, .wire 19 21),
    (.wire 19 19, .wire 19 22),
    (.wire 2 7, .wire 19 24),
    (.virt 19043, .wire 19 25),
    (.virt 19043, .wire 19 26)
  ]

/-- `publicBatchWrapper4.copies`, items `1184..1216`. -/
def publicBatchWrapper4.copies37 : List (Target × Target) := [
    (.wire 2 7, .wire 19 28),
    (.virt 38153, .wire 19 29),
    (.wire 19 27, .wire 19 30),
    (.wire 2 7, .wire 19 32),
    (.virt 19044, .wire 19 33),
    (.virt 19044, .wire 19 34),
    (.wire 2 7, .wire 19 36),
    (.virt 38153, .wire 19 37),
    (.wire 19 35, .wire 19 38),
    (.wire 2 7, .wire 19 40),
    (.virt 19045, .wire 19 41),
    (.virt 19045, .wire 19 42),
    (.wire 2 7, .wire 19 44),
    (.virt 38153, .wire 19 45),
    (.wire 19 43, .wire 19 46),
    (.wire 2 7, .wire 19 48),
    (.virt 19046, .wire 19 49),
    (.virt 19046, .wire 19 50),
    (.wire 2 7, .wire 19 52),
    (.virt 38153, .wire 19 53),
    (.wire 19 51, .wire 19 54),
    (.wire 2 7, .wire 19 56),
    (.virt 19047, .wire 19 57),
    (.virt 19047, .wire 19 58),
    (.wire 2 7, .wire 19 60),
    (.virt 38153, .wire 19 61),
    (.wire 19 59, .wire 19 62),
    (.wire 2 7, .wire 19 64),
    (.virt 19048, .wire 19 65),
    (.virt 19048, .wire 19 66),
    (.wire 2 7, .wire 19 68),
    (.virt 38153, .wire 19 69)
  ]

/-- `publicBatchWrapper4.copies`, items `1216..1248`. -/
def publicBatchWrapper4.copies38 : List (Target × Target) := [
    (.wire 19 67, .wire 19 70),
    (.wire 2 7, .wire 19 72),
    (.virt 19049, .wire 19 73),
    (.virt 19049, .wire 19 74),
    (.wire 2 7, .wire 19 76),
    (.virt 38153, .wire 19 77),
    (.wire 19 75, .wire 19 78),
    (.wire 2 51, .wire 20 0),
    (.virt 28567, .wire 20 1),
    (.virt 28567, .wire 20 2),
    (.wire 2 51, .wire 20 4),
    (.virt 38153, .wire 20 5),
    (.wire 20 3, .wire 20 6),
    (.wire 2 51, .wire 20 8),
    (.virt 28568, .wire 20 9),
    (.virt 28568, .wire 20 10),
    (.wire 2 51, .wire 20 12),
    (.virt 38153, .wire 20 13),
    (.wire 20 11, .wire 20 14),
    (.wire 2 51, .wire 20 16),
    (.virt 28569, .wire 20 17),
    (.virt 28569, .wire 20 18),
    (.wire 2 51, .wire 20 20),
    (.virt 38153, .wire 20 21),
    (.wire 20 19, .wire 20 22),
    (.wire 2 51, .wire 20 24),
    (.virt 28570, .wire 20 25),
    (.virt 28570, .wire 20 26),
    (.wire 2 51, .wire 20 28),
    (.virt 38153, .wire 20 29),
    (.wire 20 27, .wire 20 30),
    (.wire 2 51, .wire 20 32)
  ]

/-- `publicBatchWrapper4.copies`, items `1248..1280`. -/
def publicBatchWrapper4.copies39 : List (Target × Target) := [
    (.virt 28571, .wire 20 33),
    (.virt 28571, .wire 20 34),
    (.wire 2 51, .wire 20 36),
    (.virt 38153, .wire 20 37),
    (.wire 20 35, .wire 20 38),
    (.wire 2 51, .wire 20 40),
    (.virt 28572, .wire 20 41),
    (.virt 28572, .wire 20 42),
    (.wire 2 51, .wire 20 44),
    (.virt 38153, .wire 20 45),
    (.wire 20 43, .wire 20 46),
    (.wire 2 51, .wire 20 48),
    (.virt 28573, .wire 20 49),
    (.virt 28573, .wire 20 50),
    (.wire 2 51, .wire 20 52),
    (.virt 38153, .wire 20 53),
    (.wire 20 51, .wire 20 54),
    (.wire 2 51, .wire 20 56),
    (.virt 28574, .wire 20 57),
    (.virt 28574, .wire 20 58),
    (.wire 2 51, .wire 20 60),
    (.virt 38153, .wire 20 61),
    (.wire 20 59, .wire 20 62),
    (.wire 2 51, .wire 20 64),
    (.virt 28575, .wire 20 65),
    (.virt 28575, .wire 20 66),
    (.wire 2 51, .wire 20 68),
    (.virt 38153, .wire 20 69),
    (.wire 20 67, .wire 20 70),
    (.wire 2 51, .wire 20 72),
    (.virt 28576, .wire 20 73),
    (.virt 28576, .wire 20 74)
  ]

/-- `publicBatchWrapper4.copies`, items `1280..1312`. -/
def publicBatchWrapper4.copies40 : List (Target × Target) := [
    (.wire 2 51, .wire 20 76),
    (.virt 38153, .wire 20 77),
    (.wire 20 75, .wire 20 78),
    (.wire 2 51, .wire 21 0),
    (.virt 28577, .wire 21 1),
    (.virt 28577, .wire 21 2),
    (.wire 2 51, .wire 21 4),
    (.virt 38153, .wire 21 5),
    (.wire 21 3, .wire 21 6),
    (.wire 2 51, .wire 21 8),
    (.virt 28578, .wire 21 9),
    (.virt 28578, .wire 21 10),
    (.wire 2 51, .wire 21 12),
    (.virt 38153, .wire 21 13),
    (.wire 21 11, .wire 21 14),
    (.wire 2 51, .wire 21 16),
    (.virt 28579, .wire 21 17),
    (.virt 28579, .wire 21 18),
    (.wire 2 51, .wire 21 20),
    (.virt 38153, .wire 21 21),
    (.wire 21 19, .wire 21 22),
    (.wire 2 51, .wire 21 24),
    (.virt 28580, .wire 21 25),
    (.virt 28580, .wire 21 26),
    (.wire 2 51, .wire 21 28),
    (.virt 38153, .wire 21 29),
    (.wire 21 27, .wire 21 30),
    (.wire 2 51, .wire 21 32),
    (.virt 28581, .wire 21 33),
    (.virt 28581, .wire 21 34),
    (.wire 2 51, .wire 21 36),
    (.virt 38153, .wire 21 37)
  ]

/-- `publicBatchWrapper4.copies`, items `1312..1344`. -/
def publicBatchWrapper4.copies41 : List (Target × Target) := [
    (.wire 21 35, .wire 21 38),
    (.wire 2 51, .wire 21 40),
    (.virt 28582, .wire 21 41),
    (.virt 28582, .wire 21 42),
    (.wire 2 51, .wire 21 44),
    (.virt 38153, .wire 21 45),
    (.wire 21 43, .wire 21 46),
    (.wire 2 51, .wire 21 48),
    (.virt 28583, .wire 21 49),
    (.virt 28583, .wire 21 50),
    (.wire 2 51, .wire 21 52),
    (.virt 38153, .wire 21 53),
    (.wire 21 51, .wire 21 54),
    (.wire 2 51, .wire 21 56),
    (.virt 28584, .wire 21 57),
    (.virt 28584, .wire 21 58),
    (.wire 2 51, .wire 21 60),
    (.virt 38153, .wire 21 61),
    (.wire 21 59, .wire 21 62),
    (.wire 2 51, .wire 21 64),
    (.virt 28585, .wire 21 65),
    (.virt 28585, .wire 21 66),
    (.wire 2 51, .wire 21 68),
    (.virt 38153, .wire 21 69),
    (.wire 21 67, .wire 21 70),
    (.wire 2 51, .wire 21 72),
    (.virt 28586, .wire 21 73),
    (.virt 28586, .wire 21 74),
    (.wire 2 51, .wire 21 76),
    (.virt 38153, .wire 21 77),
    (.wire 21 75, .wire 21 78),
    (.wire 4 15, .wire 22 0)
  ]

/-- `publicBatchWrapper4.copies`, items `1344..1376`. -/
def publicBatchWrapper4.copies42 : List (Target × Target) := [
    (.virt 38104, .wire 22 1),
    (.virt 38104, .wire 22 2),
    (.wire 4 15, .wire 22 4),
    (.virt 38153, .wire 22 5),
    (.wire 22 3, .wire 22 6),
    (.wire 4 15, .wire 22 8),
    (.virt 38105, .wire 22 9),
    (.virt 38105, .wire 22 10),
    (.wire 4 15, .wire 22 12),
    (.virt 38153, .wire 22 13),
    (.wire 22 11, .wire 22 14),
    (.wire 4 15, .wire 22 16),
    (.virt 38106, .wire 22 17),
    (.virt 38106, .wire 22 18),
    (.wire 4 15, .wire 22 20),
    (.virt 38153, .wire 22 21),
    (.wire 22 19, .wire 22 22),
    (.wire 4 15, .wire 22 24),
    (.virt 38107, .wire 22 25),
    (.virt 38107, .wire 22 26),
    (.wire 4 15, .wire 22 28),
    (.virt 38153, .wire 22 29),
    (.wire 22 27, .wire 22 30),
    (.wire 4 15, .wire 22 32),
    (.virt 38108, .wire 22 33),
    (.virt 38108, .wire 22 34),
    (.wire 4 15, .wire 22 36),
    (.virt 38153, .wire 22 37),
    (.wire 22 35, .wire 22 38),
    (.wire 4 15, .wire 22 40),
    (.virt 38109, .wire 22 41),
    (.virt 38109, .wire 22 42)
  ]

/-- `publicBatchWrapper4.copies`, items `1376..1408`. -/
def publicBatchWrapper4.copies43 : List (Target × Target) := [
    (.wire 4 15, .wire 22 44),
    (.virt 38153, .wire 22 45),
    (.wire 22 43, .wire 22 46),
    (.wire 4 15, .wire 22 48),
    (.virt 38110, .wire 22 49),
    (.virt 38110, .wire 22 50),
    (.wire 4 15, .wire 22 52),
    (.virt 38153, .wire 22 53),
    (.wire 22 51, .wire 22 54),
    (.wire 4 15, .wire 22 56),
    (.virt 38111, .wire 22 57),
    (.virt 38111, .wire 22 58),
    (.wire 4 15, .wire 22 60),
    (.virt 38153, .wire 22 61),
    (.wire 22 59, .wire 22 62),
    (.wire 4 15, .wire 22 64),
    (.virt 38112, .wire 22 65),
    (.virt 38112, .wire 22 66),
    (.wire 4 15, .wire 22 68),
    (.virt 38153, .wire 22 69),
    (.wire 22 67, .wire 22 70),
    (.wire 4 15, .wire 22 72),
    (.virt 38113, .wire 22 73),
    (.virt 38113, .wire 22 74),
    (.wire 4 15, .wire 22 76),
    (.virt 38153, .wire 22 77),
    (.wire 22 75, .wire 22 78),
    (.wire 4 15, .wire 23 0),
    (.virt 38114, .wire 23 1),
    (.virt 38114, .wire 23 2),
    (.wire 4 15, .wire 23 4),
    (.virt 38153, .wire 23 5)
  ]

/-- `publicBatchWrapper4.copies`, items `1408..1440`. -/
def publicBatchWrapper4.copies44 : List (Target × Target) := [
    (.wire 23 3, .wire 23 6),
    (.wire 4 15, .wire 23 8),
    (.virt 38115, .wire 23 9),
    (.virt 38115, .wire 23 10),
    (.wire 4 15, .wire 23 12),
    (.virt 38153, .wire 23 13),
    (.wire 23 11, .wire 23 14),
    (.wire 4 15, .wire 23 16),
    (.virt 38116, .wire 23 17),
    (.virt 38116, .wire 23 18),
    (.wire 4 15, .wire 23 20),
    (.virt 38153, .wire 23 21),
    (.wire 23 19, .wire 23 22),
    (.wire 4 15, .wire 23 24),
    (.virt 38117, .wire 23 25),
    (.virt 38117, .wire 23 26),
    (.wire 4 15, .wire 23 28),
    (.virt 38153, .wire 23 29),
    (.wire 23 27, .wire 23 30),
    (.wire 4 15, .wire 23 32),
    (.virt 38118, .wire 23 33),
    (.virt 38118, .wire 23 34),
    (.wire 4 15, .wire 23 36),
    (.virt 38153, .wire 23 37),
    (.wire 23 35, .wire 23 38),
    (.wire 4 15, .wire 23 40),
    (.virt 38119, .wire 23 41),
    (.virt 38119, .wire 23 42),
    (.wire 4 15, .wire 23 44),
    (.virt 38153, .wire 23 45),
    (.wire 23 43, .wire 23 46),
    (.wire 4 15, .wire 23 48)
  ]

/-- `publicBatchWrapper4.copies`, items `1440..1472`. -/
def publicBatchWrapper4.copies45 : List (Target × Target) := [
    (.virt 38120, .wire 23 49),
    (.virt 38120, .wire 23 50),
    (.wire 4 15, .wire 23 52),
    (.virt 38153, .wire 23 53),
    (.wire 23 51, .wire 23 54),
    (.wire 4 15, .wire 23 56),
    (.virt 38121, .wire 23 57),
    (.virt 38121, .wire 23 58),
    (.wire 4 15, .wire 23 60),
    (.virt 38153, .wire 23 61),
    (.wire 23 59, .wire 23 62),
    (.wire 4 15, .wire 23 64),
    (.virt 38122, .wire 23 65),
    (.virt 38122, .wire 23 66),
    (.wire 4 15, .wire 23 68),
    (.virt 38153, .wire 23 69),
    (.wire 23 67, .wire 23 70),
    (.wire 4 15, .wire 23 72),
    (.virt 38123, .wire 23 73),
    (.virt 38123, .wire 23 74),
    (.wire 4 15, .wire 23 76),
    (.virt 38153, .wire 23 77),
    (.wire 23 75, .wire 23 78),
    (.wire 1 43, .wire 24 0),
    (.virt 9513, .wire 24 1),
    (.virt 9513, .wire 24 2),
    (.wire 1 43, .wire 24 4),
    (.virt 38153, .wire 24 5),
    (.wire 24 3, .wire 24 6),
    (.wire 1 43, .wire 24 8),
    (.virt 9514, .wire 24 9),
    (.virt 9514, .wire 24 10)
  ]

/-- `publicBatchWrapper4.copies`, items `1472..1504`. -/
def publicBatchWrapper4.copies46 : List (Target × Target) := [
    (.wire 1 43, .wire 24 12),
    (.virt 38153, .wire 24 13),
    (.wire 24 11, .wire 24 14),
    (.wire 1 43, .wire 24 16),
    (.virt 9515, .wire 24 17),
    (.virt 9515, .wire 24 18),
    (.wire 1 43, .wire 24 20),
    (.virt 38153, .wire 24 21),
    (.wire 24 19, .wire 24 22),
    (.wire 1 43, .wire 24 24),
    (.virt 9516, .wire 24 25),
    (.virt 9516, .wire 24 26),
    (.wire 1 43, .wire 24 28),
    (.virt 38153, .wire 24 29),
    (.wire 24 27, .wire 24 30),
    (.wire 1 43, .wire 24 32),
    (.virt 9517, .wire 24 33),
    (.virt 9517, .wire 24 34),
    (.wire 1 43, .wire 24 36),
    (.virt 38153, .wire 24 37),
    (.wire 24 35, .wire 24 38),
    (.wire 1 43, .wire 24 40),
    (.virt 9518, .wire 24 41),
    (.virt 9518, .wire 24 42),
    (.wire 1 43, .wire 24 44),
    (.virt 38153, .wire 24 45),
    (.wire 24 43, .wire 24 46),
    (.wire 1 43, .wire 24 48),
    (.virt 9519, .wire 24 49),
    (.virt 9519, .wire 24 50),
    (.wire 1 43, .wire 24 52),
    (.virt 38153, .wire 24 53)
  ]

/-- `publicBatchWrapper4.copies`, items `1504..1536`. -/
def publicBatchWrapper4.copies47 : List (Target × Target) := [
    (.wire 24 51, .wire 24 54),
    (.wire 1 43, .wire 24 56),
    (.virt 9520, .wire 24 57),
    (.virt 9520, .wire 24 58),
    (.wire 1 43, .wire 24 60),
    (.virt 38153, .wire 24 61),
    (.wire 24 59, .wire 24 62),
    (.wire 2 7, .wire 24 64),
    (.virt 19050, .wire 24 65),
    (.virt 19050, .wire 24 66),
    (.wire 2 7, .wire 24 68),
    (.virt 38153, .wire 24 69),
    (.wire 24 67, .wire 24 70),
    (.wire 2 7, .wire 24 72),
    (.virt 19051, .wire 24 73),
    (.virt 19051, .wire 24 74),
    (.wire 2 7, .wire 24 76),
    (.virt 38153, .wire 24 77),
    (.wire 24 75, .wire 24 78),
    (.wire 2 7, .wire 25 0),
    (.virt 19052, .wire 25 1),
    (.virt 19052, .wire 25 2),
    (.wire 2 7, .wire 25 4),
    (.virt 38153, .wire 25 5),
    (.wire 25 3, .wire 25 6),
    (.wire 2 7, .wire 25 8),
    (.virt 19053, .wire 25 9),
    (.virt 19053, .wire 25 10),
    (.wire 2 7, .wire 25 12),
    (.virt 38153, .wire 25 13),
    (.wire 25 11, .wire 25 14),
    (.wire 2 7, .wire 25 16)
  ]

/-- `publicBatchWrapper4.copies`, items `1536..1568`. -/
def publicBatchWrapper4.copies48 : List (Target × Target) := [
    (.virt 19054, .wire 25 17),
    (.virt 19054, .wire 25 18),
    (.wire 2 7, .wire 25 20),
    (.virt 38153, .wire 25 21),
    (.wire 25 19, .wire 25 22),
    (.wire 2 7, .wire 25 24),
    (.virt 19055, .wire 25 25),
    (.virt 19055, .wire 25 26),
    (.wire 2 7, .wire 25 28),
    (.virt 38153, .wire 25 29),
    (.wire 25 27, .wire 25 30),
    (.wire 2 7, .wire 25 32),
    (.virt 19056, .wire 25 33),
    (.virt 19056, .wire 25 34),
    (.wire 2 7, .wire 25 36),
    (.virt 38153, .wire 25 37),
    (.wire 25 35, .wire 25 38),
    (.wire 2 7, .wire 25 40),
    (.virt 19057, .wire 25 41),
    (.virt 19057, .wire 25 42),
    (.wire 2 7, .wire 25 44),
    (.virt 38153, .wire 25 45),
    (.wire 25 43, .wire 25 46),
    (.wire 2 51, .wire 25 48),
    (.virt 28587, .wire 25 49),
    (.virt 28587, .wire 25 50),
    (.wire 2 51, .wire 25 52),
    (.virt 38153, .wire 25 53),
    (.wire 25 51, .wire 25 54),
    (.wire 2 51, .wire 25 56),
    (.virt 28588, .wire 25 57),
    (.virt 28588, .wire 25 58)
  ]

/-- `publicBatchWrapper4.copies`, items `1568..1600`. -/
def publicBatchWrapper4.copies49 : List (Target × Target) := [
    (.wire 2 51, .wire 25 60),
    (.virt 38153, .wire 25 61),
    (.wire 25 59, .wire 25 62),
    (.wire 2 51, .wire 25 64),
    (.virt 28589, .wire 25 65),
    (.virt 28589, .wire 25 66),
    (.wire 2 51, .wire 25 68),
    (.virt 38153, .wire 25 69),
    (.wire 25 67, .wire 25 70),
    (.wire 2 51, .wire 25 72),
    (.virt 28590, .wire 25 73),
    (.virt 28590, .wire 25 74),
    (.wire 2 51, .wire 25 76),
    (.virt 38153, .wire 25 77),
    (.wire 25 75, .wire 25 78),
    (.wire 2 51, .wire 26 0),
    (.virt 28591, .wire 26 1),
    (.virt 28591, .wire 26 2),
    (.wire 2 51, .wire 26 4),
    (.virt 38153, .wire 26 5),
    (.wire 26 3, .wire 26 6),
    (.wire 2 51, .wire 26 8),
    (.virt 28592, .wire 26 9),
    (.virt 28592, .wire 26 10),
    (.wire 2 51, .wire 26 12),
    (.virt 38153, .wire 26 13),
    (.wire 26 11, .wire 26 14),
    (.wire 2 51, .wire 26 16),
    (.virt 28593, .wire 26 17),
    (.virt 28593, .wire 26 18),
    (.wire 2 51, .wire 26 20),
    (.virt 38153, .wire 26 21)
  ]

/-- `publicBatchWrapper4.copies`, items `1600..1632`. -/
def publicBatchWrapper4.copies50 : List (Target × Target) := [
    (.wire 26 19, .wire 26 22),
    (.wire 2 51, .wire 26 24),
    (.virt 28594, .wire 26 25),
    (.virt 28594, .wire 26 26),
    (.wire 2 51, .wire 26 28),
    (.virt 38153, .wire 26 29),
    (.wire 26 27, .wire 26 30),
    (.wire 4 15, .wire 26 32),
    (.virt 38124, .wire 26 33),
    (.virt 38124, .wire 26 34),
    (.wire 4 15, .wire 26 36),
    (.virt 38153, .wire 26 37),
    (.wire 26 35, .wire 26 38),
    (.wire 4 15, .wire 26 40),
    (.virt 38125, .wire 26 41),
    (.virt 38125, .wire 26 42),
    (.wire 4 15, .wire 26 44),
    (.virt 38153, .wire 26 45),
    (.wire 26 43, .wire 26 46),
    (.wire 4 15, .wire 26 48),
    (.virt 38126, .wire 26 49),
    (.virt 38126, .wire 26 50),
    (.wire 4 15, .wire 26 52),
    (.virt 38153, .wire 26 53),
    (.wire 26 51, .wire 26 54),
    (.wire 4 15, .wire 26 56),
    (.virt 38127, .wire 26 57),
    (.virt 38127, .wire 26 58),
    (.wire 4 15, .wire 26 60),
    (.virt 38153, .wire 26 61),
    (.wire 26 59, .wire 26 62),
    (.wire 4 15, .wire 26 64)
  ]

/-- `publicBatchWrapper4.copies`, items `1632..1655`. -/
def publicBatchWrapper4.copies51 : List (Target × Target) := [
    (.virt 38128, .wire 26 65),
    (.virt 38128, .wire 26 66),
    (.wire 4 15, .wire 26 68),
    (.virt 38153, .wire 26 69),
    (.wire 26 67, .wire 26 70),
    (.wire 4 15, .wire 26 72),
    (.virt 38129, .wire 26 73),
    (.virt 38129, .wire 26 74),
    (.wire 4 15, .wire 26 76),
    (.virt 38153, .wire 26 77),
    (.wire 26 75, .wire 26 78),
    (.wire 4 15, .wire 27 0),
    (.virt 38130, .wire 27 1),
    (.virt 38130, .wire 27 2),
    (.wire 4 15, .wire 27 4),
    (.virt 38153, .wire 27 5),
    (.wire 27 3, .wire 27 6),
    (.wire 4 15, .wire 27 8),
    (.virt 38131, .wire 27 9),
    (.virt 38131, .wire 27 10),
    (.wire 4 15, .wire 27 12),
    (.virt 38153, .wire 27 13),
    (.wire 27 11, .wire 27 14)
  ]

/-- The public-batch aggregation wrapper at `n_inner = 4` over `2`-leaf private batches, without the inner verifiers (`wormhole/aggregator/src/public_batch/circuit/circuit_logic.rs`): inner public inputs `inner_pis_0..3`, the `aggregator_address` witness, and the aggregated public inputs. -/
def publicBatchWrapper4 (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 2
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 3
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 4
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 5
    ⟨.arithmetic 20, [(-1), 1]⟩,  -- row 6
    ⟨.arithmetic 20, [1, 1]⟩,  -- row 7
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 8
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 9
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 10
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 11
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 12
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 13
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 14
    ⟨.arithmetic 20, [1, 0]⟩,  -- row 15
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 16
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 17
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 18
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 19
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 20
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 21
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 22
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 23
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 24
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 25
    ⟨.arithmetic 20, [1, (-1)]⟩,  -- row 26
    ⟨.arithmetic 20, [1, (-1)]⟩  -- row 27
  ]
  copies := (((((publicBatchWrapper4.copies0 ++ (publicBatchWrapper4.copies1 ++ publicBatchWrapper4.copies2)) ++ (publicBatchWrapper4.copies3 ++ (publicBatchWrapper4.copies4 ++ publicBatchWrapper4.copies5))) ++ ((publicBatchWrapper4.copies6 ++ (publicBatchWrapper4.copies7 ++ publicBatchWrapper4.copies8)) ++ ((publicBatchWrapper4.copies9 ++ publicBatchWrapper4.copies10) ++ (publicBatchWrapper4.copies11 ++ publicBatchWrapper4.copies12)))) ++ (((publicBatchWrapper4.copies13 ++ (publicBatchWrapper4.copies14 ++ publicBatchWrapper4.copies15)) ++ (publicBatchWrapper4.copies16 ++ (publicBatchWrapper4.copies17 ++ publicBatchWrapper4.copies18))) ++ ((publicBatchWrapper4.copies19 ++ (publicBatchWrapper4.copies20 ++ publicBatchWrapper4.copies21)) ++ ((publicBatchWrapper4.copies22 ++ publicBatchWrapper4.copies23) ++ (publicBatchWrapper4.copies24 ++ publicBatchWrapper4.copies25))))) ++ ((((publicBatchWrapper4.copies26 ++ (publicBatchWrapper4.copies27 ++ publicBatchWrapper4.copies28)) ++ (publicBatchWrapper4.copies29 ++ (publicBatchWrapper4.copies30 ++ publicBatchWrapper4.copies31))) ++ ((publicBatchWrapper4.copies32 ++ (publicBatchWrapper4.copies33 ++ publicBatchWrapper4.copies34)) ++ ((publicBatchWrapper4.copies35 ++ publicBatchWrapper4.copies36) ++ (publicBatchWrapper4.copies37 ++ publicBatchWrapper4.copies38)))) ++ (((publicBatchWrapper4.copies39 ++ (publicBatchWrapper4.copies40 ++ publicBatchWrapper4.copies41)) ++ (publicBatchWrapper4.copies42 ++ (publicBatchWrapper4.copies43 ++ publicBatchWrapper4.copies44))) ++ ((publicBatchWrapper4.copies45 ++ (publicBatchWrapper4.copies46 ++ publicBatchWrapper4.copies47)) ++ ((publicBatchWrapper4.copies48 ++ publicBatchWrapper4.copies49) ++ (publicBatchWrapper4.copies50 ++ publicBatchWrapper4.copies51))))))
  constants := [
    (.virt 38152, 1),
    (.virt 38153, 0),
    (.virt 38234, 16)
  ]
  publicInputs := [
    .virt 38148,
    .virt 38149,
    .virt 38150,
    .virt 38151,
    .wire 9 23,
    .wire 9 31,
    .wire 8 63,
    .wire 8 71,
    .wire 8 79,
    .wire 9 7,
    .wire 9 15,
    .virt 38234,
    .wire 16 7,
    .wire 16 15,
    .wire 16 23,
    .wire 16 31,
    .wire 16 39,
    .wire 16 47,
    .wire 16 55,
    .wire 16 63,
    .wire 16 71,
    .wire 16 79,
    .wire 17 7,
    .wire 17 15,
    .wire 17 23,
    .wire 17 31,
    .wire 17 39,
    .wire 17 47,
    .wire 17 55,
    .wire 17 63,
    .wire 17 71,
    .wire 17 79,
    .wire 18 7,
    .wire 18 15,
    .wire 18 23,
    .wire 18 31,
    .wire 18 39,
    .wire 18 47,
    .wire 18 55,
    .wire 18 63,
    .wire 18 71,
    .wire 18 79,
    .wire 19 7,
    .wire 19 15,
    .wire 19 23,
    .wire 19 31,
    .wire 19 39,
    .wire 19 47,
    .wire 19 55,
    .wire 19 63,
    .wire 19 71,
    .wire 19 79,
    .wire 20 7,
    .wire 20 15,
    .wire 20 23,
    .wire 20 31,
    .wire 20 39,
    .wire 20 47,
    .wire 20 55,
    .wire 20 63,
    .wire 20 71,
    .wire 20 79,
    .wire 21 7,
    .wire 21 15,
    .wire 21 23,
    .wire 21 31,
    .wire 21 39,
    .wire 21 47,
    .wire 21 55,
    .wire 21 63,
    .wire 21 71,
    .wire 21 79,
    .wire 22 7,
    .wire 22 15,
    .wire 22 23,
    .wire 22 31,
    .wire 22 39,
    .wire 22 47,
    .wire 22 55,
    .wire 22 63,
    .wire 22 71,
    .wire 22 79,
    .wire 23 7,
    .wire 23 15,
    .wire 23 23,
    .wire 23 31,
    .wire 23 39,
    .wire 23 47,
    .wire 23 55,
    .wire 23 63,
    .wire 23 71,
    .wire 23 79,
    .wire 24 7,
    .wire 24 15,
    .wire 24 23,
    .wire 24 31,
    .wire 24 39,
    .wire 24 47,
    .wire 24 55,
    .wire 24 63,
    .wire 24 71,
    .wire 24 79,
    .wire 25 7,
    .wire 25 15,
    .wire 25 23,
    .wire 25 31,
    .wire 25 39,
    .wire 25 47,
    .wire 25 55,
    .wire 25 63,
    .wire 25 71,
    .wire 25 79,
    .wire 26 7,
    .wire 26 15,
    .wire 26 23,
    .wire 26 31,
    .wire 26 39,
    .wire 26 47,
    .wire 26 55,
    .wire 26 63,
    .wire 26 71,
    .wire 26 79,
    .wire 27 7,
    .wire 27 15
  ]

/-- Named targets `inner_pis_0`. -/
def publicBatchWrapper4.inner_pis_0 : Fin 52 → Target :=
  ![.virt 9485, .virt 9486, .virt 9487, .virt 9488, .virt 9489, .virt 9490, .virt 9491, .virt 9492, .virt 9493, .virt 9494, .virt 9495, .virt 9496, .virt 9497, .virt 9498, .virt 9499, .virt 9500, .virt 9501, .virt 9502, .virt 9503, .virt 9504, .virt 9505, .virt 9506, .virt 9507, .virt 9508, .virt 9509, .virt 9510, .virt 9511, .virt 9512, .virt 9513, .virt 9514, .virt 9515, .virt 9516, .virt 9517, .virt 9518, .virt 9519, .virt 9520, .virt 9521, .virt 9522, .virt 9523, .virt 9524, .virt 9525, .virt 9526, .virt 9527, .virt 9528, .virt 9529, .virt 9530, .virt 9531, .virt 9532, .virt 9533, .virt 9534, .virt 9535, .virt 9536]

/-- Named targets `inner_pis_1`. -/
def publicBatchWrapper4.inner_pis_1 : Fin 52 → Target :=
  ![.virt 19022, .virt 19023, .virt 19024, .virt 19025, .virt 19026, .virt 19027, .virt 19028, .virt 19029, .virt 19030, .virt 19031, .virt 19032, .virt 19033, .virt 19034, .virt 19035, .virt 19036, .virt 19037, .virt 19038, .virt 19039, .virt 19040, .virt 19041, .virt 19042, .virt 19043, .virt 19044, .virt 19045, .virt 19046, .virt 19047, .virt 19048, .virt 19049, .virt 19050, .virt 19051, .virt 19052, .virt 19053, .virt 19054, .virt 19055, .virt 19056, .virt 19057, .virt 19058, .virt 19059, .virt 19060, .virt 19061, .virt 19062, .virt 19063, .virt 19064, .virt 19065, .virt 19066, .virt 19067, .virt 19068, .virt 19069, .virt 19070, .virt 19071, .virt 19072, .virt 19073]

/-- Named targets `inner_pis_2`. -/
def publicBatchWrapper4.inner_pis_2 : Fin 52 → Target :=
  ![.virt 28559, .virt 28560, .virt 28561, .virt 28562, .virt 28563, .virt 28564, .virt 28565, .virt 28566, .virt 28567, .virt 28568, .virt 28569, .virt 28570, .virt 28571, .virt 28572, .virt 28573, .virt 28574, .virt 28575, .virt 28576, .virt 28577, .virt 28578, .virt 28579, .virt 28580, .virt 28581, .virt 28582, .virt 28583, .virt 28584, .virt 28585, .virt 28586, .virt 28587, .virt 28588, .virt 28589, .virt 28590, .virt 28591, .virt 28592, .virt 28593, .virt 28594, .virt 28595, .virt 28596, .virt 28597, .virt 28598, .virt 28599, .virt 28600, .virt 28601, .virt 28602, .virt 28603, .virt 28604, .virt 28605, .virt 28606, .virt 28607, .virt 28608, .virt 28609, .virt 28610]

/-- Named targets `inner_pis_3`. -/
def publicBatchWrapper4.inner_pis_3 : Fin 52 → Target :=
  ![.virt 38096, .virt 38097, .virt 38098, .virt 38099, .virt 38100, .virt 38101, .virt 38102, .virt 38103, .virt 38104, .virt 38105, .virt 38106, .virt 38107, .virt 38108, .virt 38109, .virt 38110, .virt 38111, .virt 38112, .virt 38113, .virt 38114, .virt 38115, .virt 38116, .virt 38117, .virt 38118, .virt 38119, .virt 38120, .virt 38121, .virt 38122, .virt 38123, .virt 38124, .virt 38125, .virt 38126, .virt 38127, .virt 38128, .virt 38129, .virt 38130, .virt 38131, .virt 38132, .virt 38133, .virt 38134, .virt 38135, .virt 38136, .virt 38137, .virt 38138, .virt 38139, .virt 38140, .virt 38141, .virt 38142, .virt 38143, .virt 38144, .virt 38145, .virt 38146, .virt 38147]

/-- Named targets `aggregator_address`. -/
def publicBatchWrapper4.aggregator_address : Fin 4 → Target :=
  ![.virt 38148, .virt 38149, .virt 38150, .virt 38151]

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `publicBatchWrapper4` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem publicBatchWrapper4_decode (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    (IsEqual (a (.virt 9488)) (a (.virt 38153)) (a (.virt 38154)) (a (.virt 38155)) ∧
    IsEqual (a (.virt 9489)) (a (.virt 38153)) (a (.virt 38156)) (a (.virt 38157)) ∧
    IsEqual (a (.virt 9490)) (a (.virt 38153)) (a (.virt 38158)) (a (.virt 38159)) ∧
    IsEqual (a (.virt 9491)) (a (.virt 38153)) (a (.virt 38160)) (a (.virt 38161)) ∧
    a (.wire 1 35) = band (a (.virt 38154)) (a (.virt 38156)) ∧
    a (.wire 1 39) = band (a (.virt 38158)) (a (.virt 38160)) ∧
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) ∧
    IsEqual (a (.virt 19025)) (a (.virt 38153)) (a (.virt 38162)) (a (.virt 38163)) ∧
    IsEqual (a (.virt 19026)) (a (.virt 38153)) (a (.virt 38164)) (a (.virt 38165)) ∧
    IsEqual (a (.virt 19027)) (a (.virt 38153)) (a (.virt 38166)) (a (.virt 38167)) ∧
    IsEqual (a (.virt 19028)) (a (.virt 38153)) (a (.virt 38168)) (a (.virt 38169)) ∧
    a (.wire 1 79) = band (a (.virt 38162)) (a (.virt 38164)) ∧
    a (.wire 2 3) = band (a (.virt 38166)) (a (.virt 38168)) ∧
    a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) ∧
    IsEqual (a (.virt 28562)) (a (.virt 38153)) (a (.virt 38170)) (a (.virt 38171)) ∧
    IsEqual (a (.virt 28563)) (a (.virt 38153)) (a (.virt 38172)) (a (.virt 38173)) ∧
    IsEqual (a (.virt 28564)) (a (.virt 38153)) (a (.virt 38174)) (a (.virt 38175)) ∧
    IsEqual (a (.virt 28565)) (a (.virt 38153)) (a (.virt 38176)) (a (.virt 38177)) ∧
    a (.wire 2 43) = band (a (.virt 38170)) (a (.virt 38172)) ∧
    a (.wire 2 47) = band (a (.virt 38174)) (a (.virt 38176)) ∧
    a (.wire 2 51) = band (a (.wire 2 43)) (a (.wire 2 47)) ∧
    IsEqual (a (.virt 38099)) (a (.virt 38153)) (a (.virt 38178)) (a (.virt 38179)) ∧
    IsEqual (a (.virt 38100)) (a (.virt 38153)) (a (.virt 38180)) (a (.virt 38181)) ∧
    IsEqual (a (.virt 38101)) (a (.virt 38153)) (a (.virt 38182)) (a (.virt 38183)) ∧
    IsEqual (a (.virt 38102)) (a (.virt 38153)) (a (.virt 38184)) (a (.virt 38185)) ∧
    a (.wire 4 7) = band (a (.virt 38178)) (a (.virt 38180)) ∧
    a (.wire 4 11) = band (a (.virt 38182)) (a (.virt 38184)) ∧
    a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11)) ∧
    a (.wire 3 51) = bnot (a (.wire 1 43)) ∧
    a (.virt 38152) = bnot (a (.virt 38153)) ∧
    a (.wire 3 51) = band (a (.wire 3 51)) (a (.virt 38152)) ∧
    a (.wire 3 55) = bselect (a (.wire 3 51)) (a (.virt 9488)) (a (.virt 38153))) ∧
    (a (.wire 3 59) = bselect (a (.wire 3 51)) (a (.virt 9489)) (a (.virt 38153)) ∧
    a (.wire 3 63) = bselect (a (.wire 3 51)) (a (.virt 9490)) (a (.virt 38153)) ∧
    a (.wire 3 67) = bselect (a (.wire 3 51)) (a (.virt 9491)) (a (.virt 38153)) ∧
    a (.wire 3 71) = bselect (a (.wire 3 51)) (a (.virt 9492)) (a (.virt 38153)) ∧
    a (.wire 3 75) = bselect (a (.wire 3 51)) (a (.virt 9486)) (a (.virt 38153)) ∧
    a (.wire 3 79) = bselect (a (.wire 3 51)) (a (.virt 9487)) (a (.virt 38153)) ∧
    a (.wire 3 51) = bor (a (.virt 38153)) (a (.wire 3 51)) ∧
    a (.wire 5 3) = bnot (a (.wire 2 7)) ∧
    a (.wire 5 7) = bnot (a (.wire 3 51)) ∧
    a (.wire 4 19) = band (a (.wire 5 3)) (a (.wire 5 7)) ∧
    a (.wire 5 15) = bselect (a (.wire 4 19)) (a (.virt 19025)) (a (.wire 3 55)) ∧
    a (.wire 5 23) = bselect (a (.wire 4 19)) (a (.virt 19026)) (a (.wire 3 59)) ∧
    a (.wire 5 31) = bselect (a (.wire 4 19)) (a (.virt 19027)) (a (.wire 3 63)) ∧
    a (.wire 5 39) = bselect (a (.wire 4 19)) (a (.virt 19028)) (a (.wire 3 67)) ∧
    a (.wire 5 47) = bselect (a (.wire 4 19)) (a (.virt 19029)) (a (.wire 3 71)) ∧
    a (.wire 5 55) = bselect (a (.wire 4 19)) (a (.virt 19023)) (a (.wire 3 75)) ∧
    a (.wire 5 63) = bselect (a (.wire 4 19)) (a (.virt 19024)) (a (.wire 3 79)) ∧
    a (.wire 7 3) = bor (a (.wire 3 51)) (a (.wire 5 3)) ∧
    a (.wire 5 67) = bnot (a (.wire 2 51)) ∧
    a (.wire 5 71) = bnot (a (.wire 7 3)) ∧
    a (.wire 4 23) = band (a (.wire 5 67)) (a (.wire 5 71)) ∧
    a (.wire 5 79) = bselect (a (.wire 4 23)) (a (.virt 28562)) (a (.wire 5 15)) ∧
    a (.wire 8 7) = bselect (a (.wire 4 23)) (a (.virt 28563)) (a (.wire 5 23)) ∧
    a (.wire 8 15) = bselect (a (.wire 4 23)) (a (.virt 28564)) (a (.wire 5 31)) ∧
    a (.wire 8 23) = bselect (a (.wire 4 23)) (a (.virt 28565)) (a (.wire 5 39)) ∧
    a (.wire 8 31) = bselect (a (.wire 4 23)) (a (.virt 28566)) (a (.wire 5 47)) ∧
    a (.wire 8 39) = bselect (a (.wire 4 23)) (a (.virt 28560)) (a (.wire 5 55)) ∧
    a (.wire 8 47) = bselect (a (.wire 4 23)) (a (.virt 28561)) (a (.wire 5 63)) ∧
    a (.wire 7 7) = bor (a (.wire 7 3)) (a (.wire 5 67)) ∧
    a (.wire 8 51) = bnot (a (.wire 4 15)) ∧
    a (.wire 8 55) = bnot (a (.wire 7 7)) ∧
    a (.wire 4 27) = band (a (.wire 8 51)) (a (.wire 8 55))) ∧
    (a (.wire 8 63) = bselect (a (.wire 4 27)) (a (.virt 38099)) (a (.wire 5 79)) ∧
    a (.wire 8 71) = bselect (a (.wire 4 27)) (a (.virt 38100)) (a (.wire 8 7)) ∧
    a (.wire 8 79) = bselect (a (.wire 4 27)) (a (.virt 38101)) (a (.wire 8 15)) ∧
    a (.wire 9 7) = bselect (a (.wire 4 27)) (a (.virt 38102)) (a (.wire 8 23)) ∧
    a (.wire 9 15) = bselect (a (.wire 4 27)) (a (.virt 38103)) (a (.wire 8 31)) ∧
    a (.wire 9 23) = bselect (a (.wire 4 27)) (a (.virt 38097)) (a (.wire 8 39)) ∧
    a (.wire 9 31) = bselect (a (.wire 4 27)) (a (.virt 38098)) (a (.wire 8 47)) ∧
    a (.wire 7 11) = bor (a (.wire 7 7)) (a (.wire 8 51)) ∧
    IsEqual (a (.virt 9486)) (a (.wire 9 23)) (a (.virt 38186)) (a (.virt 38187)) ∧
    a (.wire 7 15) = bor (a (.wire 1 43)) (a (.virt 38186)) ∧
    a (.wire 7 15) = a (.virt 38152) ∧
    IsEqual (a (.virt 9487)) (a (.wire 9 31)) (a (.virt 38188)) (a (.virt 38189)) ∧
    a (.wire 7 19) = bor (a (.wire 1 43)) (a (.virt 38188)) ∧
    a (.wire 7 19) = a (.virt 38152) ∧
    IsEqual (a (.virt 9488)) (a (.wire 8 63)) (a (.virt 38190)) (a (.virt 38191)) ∧
    IsEqual (a (.virt 9489)) (a (.wire 8 71)) (a (.virt 38192)) (a (.virt 38193)) ∧
    IsEqual (a (.virt 9490)) (a (.wire 8 79)) (a (.virt 38194)) (a (.virt 38195)) ∧
    IsEqual (a (.virt 9491)) (a (.wire 9 7)) (a (.virt 38196)) (a (.virt 38197)) ∧
    a (.wire 4 79) = band (a (.virt 38190)) (a (.virt 38192)) ∧
    a (.wire 11 3) = band (a (.virt 38194)) (a (.virt 38196)) ∧
    a (.wire 11 7) = band (a (.wire 4 79)) (a (.wire 11 3)) ∧
    a (.wire 7 23) = bor (a (.wire 1 43)) (a (.wire 11 7)) ∧
    a (.wire 7 23) = a (.virt 38152) ∧
    IsEqual (a (.virt 19023)) (a (.wire 9 23)) (a (.virt 38198)) (a (.virt 38199)) ∧
    a (.wire 7 27) = bor (a (.wire 2 7)) (a (.virt 38198)) ∧
    a (.wire 7 27) = a (.virt 38152) ∧
    IsEqual (a (.virt 19024)) (a (.wire 9 31)) (a (.virt 38200)) (a (.virt 38201)) ∧
    a (.wire 7 31) = bor (a (.wire 2 7)) (a (.virt 38200)) ∧
    a (.wire 7 31) = a (.virt 38152) ∧
    IsEqual (a (.virt 19025)) (a (.wire 8 63)) (a (.virt 38202)) (a (.virt 38203)) ∧
    IsEqual (a (.virt 19026)) (a (.wire 8 71)) (a (.virt 38204)) (a (.virt 38205)) ∧
    IsEqual (a (.virt 19027)) (a (.wire 8 79)) (a (.virt 38206)) (a (.virt 38207))) ∧
    (IsEqual (a (.virt 19028)) (a (.wire 9 7)) (a (.virt 38208)) (a (.virt 38209)) ∧
    a (.wire 11 59) = band (a (.virt 38202)) (a (.virt 38204)) ∧
    a (.wire 11 63) = band (a (.virt 38206)) (a (.virt 38208)) ∧
    a (.wire 11 67) = band (a (.wire 11 59)) (a (.wire 11 63)) ∧
    a (.wire 7 35) = bor (a (.wire 2 7)) (a (.wire 11 67)) ∧
    a (.wire 7 35) = a (.virt 38152) ∧
    IsEqual (a (.virt 28560)) (a (.wire 9 23)) (a (.virt 38210)) (a (.virt 38211)) ∧
    a (.wire 7 39) = bor (a (.wire 2 51)) (a (.virt 38210)) ∧
    a (.wire 7 39) = a (.virt 38152) ∧
    IsEqual (a (.virt 28561)) (a (.wire 9 31)) (a (.virt 38212)) (a (.virt 38213)) ∧
    a (.wire 7 43) = bor (a (.wire 2 51)) (a (.virt 38212)) ∧
    a (.wire 7 43) = a (.virt 38152) ∧
    IsEqual (a (.virt 28562)) (a (.wire 8 63)) (a (.virt 38214)) (a (.virt 38215)) ∧
    IsEqual (a (.virt 28563)) (a (.wire 8 71)) (a (.virt 38216)) (a (.virt 38217)) ∧
    IsEqual (a (.virt 28564)) (a (.wire 8 79)) (a (.virt 38218)) (a (.virt 38219)) ∧
    IsEqual (a (.virt 28565)) (a (.wire 9 7)) (a (.virt 38220)) (a (.virt 38221)) ∧
    a (.wire 13 39) = band (a (.virt 38214)) (a (.virt 38216)) ∧
    a (.wire 13 43) = band (a (.virt 38218)) (a (.virt 38220)) ∧
    a (.wire 13 47) = band (a (.wire 13 39)) (a (.wire 13 43)) ∧
    a (.wire 7 47) = bor (a (.wire 2 51)) (a (.wire 13 47)) ∧
    a (.wire 7 47) = a (.virt 38152) ∧
    IsEqual (a (.virt 38097)) (a (.wire 9 23)) (a (.virt 38222)) (a (.virt 38223)) ∧
    a (.wire 7 51) = bor (a (.wire 4 15)) (a (.virt 38222)) ∧
    a (.wire 7 51) = a (.virt 38152) ∧
    IsEqual (a (.virt 38098)) (a (.wire 9 31)) (a (.virt 38224)) (a (.virt 38225)) ∧
    a (.wire 7 55) = bor (a (.wire 4 15)) (a (.virt 38224)) ∧
    a (.wire 7 55) = a (.virt 38152) ∧
    IsEqual (a (.virt 38099)) (a (.wire 8 63)) (a (.virt 38226)) (a (.virt 38227)) ∧
    IsEqual (a (.virt 38100)) (a (.wire 8 71)) (a (.virt 38228)) (a (.virt 38229)) ∧
    IsEqual (a (.virt 38101)) (a (.wire 8 79)) (a (.virt 38230)) (a (.virt 38231)) ∧
    IsEqual (a (.virt 38102)) (a (.wire 9 7)) (a (.virt 38232)) (a (.virt 38233)) ∧
    a (.wire 15 19) = band (a (.virt 38226)) (a (.virt 38228))) ∧
    (a (.wire 15 23) = band (a (.virt 38230)) (a (.virt 38232)) ∧
    a (.wire 15 27) = band (a (.wire 15 19)) (a (.wire 15 23)) ∧
    a (.wire 7 59) = bor (a (.wire 4 15)) (a (.wire 15 27)) ∧
    a (.wire 7 59) = a (.virt 38152) ∧
    a (.wire 16 7) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9493)) ∧
    a (.wire 16 15) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9494)) ∧
    a (.wire 16 23) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9495)) ∧
    a (.wire 16 31) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9496)) ∧
    a (.wire 16 39) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9497)) ∧
    a (.wire 16 47) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9498)) ∧
    a (.wire 16 55) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9499)) ∧
    a (.wire 16 63) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9500)) ∧
    a (.wire 16 71) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9501)) ∧
    a (.wire 16 79) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9502)) ∧
    a (.wire 17 7) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9503)) ∧
    a (.wire 17 15) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9504)) ∧
    a (.wire 17 23) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9505)) ∧
    a (.wire 17 31) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9506)) ∧
    a (.wire 17 39) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9507)) ∧
    a (.wire 17 47) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9508)) ∧
    a (.wire 17 55) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9509)) ∧
    a (.wire 17 63) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9510)) ∧
    a (.wire 17 71) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9511)) ∧
    a (.wire 17 79) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9512)) ∧
    a (.wire 18 7) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19030)) ∧
    a (.wire 18 15) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19031)) ∧
    a (.wire 18 23) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19032)) ∧
    a (.wire 18 31) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19033)) ∧
    a (.wire 18 39) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19034)) ∧
    a (.wire 18 47) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19035)) ∧
    a (.wire 18 55) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19036)) ∧
    a (.wire 18 63) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19037))) ∧
    (a (.wire 18 71) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19038)) ∧
    a (.wire 18 79) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19039)) ∧
    a (.wire 19 7) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19040)) ∧
    a (.wire 19 15) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19041)) ∧
    a (.wire 19 23) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19042)) ∧
    a (.wire 19 31) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19043)) ∧
    a (.wire 19 39) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19044)) ∧
    a (.wire 19 47) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19045)) ∧
    a (.wire 19 55) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19046)) ∧
    a (.wire 19 63) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19047)) ∧
    a (.wire 19 71) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19048)) ∧
    a (.wire 19 79) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19049)) ∧
    a (.wire 20 7) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28567)) ∧
    a (.wire 20 15) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28568)) ∧
    a (.wire 20 23) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28569)) ∧
    a (.wire 20 31) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28570)) ∧
    a (.wire 20 39) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28571)) ∧
    a (.wire 20 47) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28572)) ∧
    a (.wire 20 55) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28573)) ∧
    a (.wire 20 63) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28574)) ∧
    a (.wire 20 71) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28575)) ∧
    a (.wire 20 79) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28576)) ∧
    a (.wire 21 7) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28577)) ∧
    a (.wire 21 15) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28578)) ∧
    a (.wire 21 23) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28579)) ∧
    a (.wire 21 31) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28580)) ∧
    a (.wire 21 39) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28581)) ∧
    a (.wire 21 47) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28582)) ∧
    a (.wire 21 55) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28583)) ∧
    a (.wire 21 63) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28584)) ∧
    a (.wire 21 71) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28585)) ∧
    a (.wire 21 79) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28586))) ∧
    (a (.wire 22 7) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38104)) ∧
    a (.wire 22 15) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38105)) ∧
    a (.wire 22 23) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38106)) ∧
    a (.wire 22 31) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38107)) ∧
    a (.wire 22 39) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38108)) ∧
    a (.wire 22 47) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38109)) ∧
    a (.wire 22 55) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38110)) ∧
    a (.wire 22 63) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38111)) ∧
    a (.wire 22 71) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38112)) ∧
    a (.wire 22 79) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38113)) ∧
    a (.wire 23 7) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38114)) ∧
    a (.wire 23 15) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38115)) ∧
    a (.wire 23 23) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38116)) ∧
    a (.wire 23 31) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38117)) ∧
    a (.wire 23 39) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38118)) ∧
    a (.wire 23 47) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38119)) ∧
    a (.wire 23 55) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38120)) ∧
    a (.wire 23 63) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38121)) ∧
    a (.wire 23 71) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38122)) ∧
    a (.wire 23 79) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38123)) ∧
    a (.wire 24 7) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9513)) ∧
    a (.wire 24 15) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9514)) ∧
    a (.wire 24 23) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9515)) ∧
    a (.wire 24 31) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9516)) ∧
    a (.wire 24 39) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9517)) ∧
    a (.wire 24 47) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9518)) ∧
    a (.wire 24 55) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9519)) ∧
    a (.wire 24 63) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9520)) ∧
    a (.wire 24 71) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19050)) ∧
    a (.wire 24 79) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19051)) ∧
    a (.wire 25 7) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19052)) ∧
    a (.wire 25 15) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19053))) ∧
    (a (.wire 25 23) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19054)) ∧
    a (.wire 25 31) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19055)) ∧
    a (.wire 25 39) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19056)) ∧
    a (.wire 25 47) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19057)) ∧
    a (.wire 25 55) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28587)) ∧
    a (.wire 25 63) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28588)) ∧
    a (.wire 25 71) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28589)) ∧
    a (.wire 25 79) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28590)) ∧
    a (.wire 26 7) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28591)) ∧
    a (.wire 26 15) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28592)) ∧
    a (.wire 26 23) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28593)) ∧
    a (.wire 26 31) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28594)) ∧
    a (.wire 26 39) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38124)) ∧
    a (.wire 26 47) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38125)) ∧
    a (.wire 26 55) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38126)) ∧
    a (.wire 26 63) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38127)) ∧
    a (.wire 26 71) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38128)) ∧
    a (.wire 26 79) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38129)) ∧
    a (.wire 27 7) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38130)) ∧
    a (.wire 27 15) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38131))) := by
  have hcopy := h.2.1
  simp only [publicBatchWrapper4, List.forall_mem_append] at hcopy
  have hcopy0 := hcopy.1.1.1.1.1
  simp only [publicBatchWrapper4.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy0
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31⟩ := hcopy0
  have hcopy1 := hcopy.1.1.1.1.2.1
  simp only [publicBatchWrapper4.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy1
  obtain ⟨c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63⟩ := hcopy1
  have hcopy2 := hcopy.1.1.1.1.2.2
  simp only [publicBatchWrapper4.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy2
  obtain ⟨c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95⟩ := hcopy2
  have hcopy3 := hcopy.1.1.1.2.1
  simp only [publicBatchWrapper4.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy3
  obtain ⟨c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127⟩ := hcopy3
  have hcopy4 := hcopy.1.1.1.2.2.1
  simp only [publicBatchWrapper4.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy4
  obtain ⟨c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149, c150, c151, c152, c153, c154, c155, c156, c157, c158, c159⟩ := hcopy4
  have hcopy5 := hcopy.1.1.1.2.2.2
  simp only [publicBatchWrapper4.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy5
  obtain ⟨c160, c161, c162, c163, c164, c165, c166, c167, c168, c169, c170, c171, c172, c173, c174, c175, c176, c177, c178, c179, c180, c181, c182, c183, c184, c185, c186, c187, c188, c189, c190, c191⟩ := hcopy5
  have hcopy6 := hcopy.1.1.2.1.1
  simp only [publicBatchWrapper4.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy6
  obtain ⟨c192, c193, c194, c195, c196, c197, c198, c199, c200, c201, c202, c203, c204, c205, c206, c207, c208, c209, c210, c211, c212, c213, c214, c215, c216, c217, c218, c219, c220, c221, c222, c223⟩ := hcopy6
  have hcopy7 := hcopy.1.1.2.1.2.1
  simp only [publicBatchWrapper4.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy7
  obtain ⟨c224, c225, c226, c227, c228, c229, c230, c231, c232, c233, c234, c235, c236, c237, c238, c239, c240, c241, c242, c243, c244, c245, c246, c247, c248, c249, c250, c251, c252, c253, c254, c255⟩ := hcopy7
  have hcopy8 := hcopy.1.1.2.1.2.2
  simp only [publicBatchWrapper4.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy8
  obtain ⟨c256, c257, c258, c259, c260, c261, c262, c263, c264, c265, c266, c267, c268, c269, c270, c271, c272, c273, c274, c275, c276, c277, c278, c279, c280, c281, c282, c283, c284, c285, c286, c287⟩ := hcopy8
  have hcopy9 := hcopy.1.1.2.2.1.1
  simp only [publicBatchWrapper4.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy9
  obtain ⟨c288, c289, c290, c291, c292, c293, c294, c295, c296, c297, c298, c299, c300, c301, c302, c303, c304, c305, c306, c307, c308, c309, c310, c311, c312, c313, c314, c315, c316, c317, c318, c319⟩ := hcopy9
  have hcopy10 := hcopy.1.1.2.2.1.2
  simp only [publicBatchWrapper4.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy10
  obtain ⟨c320, c321, c322, c323, c324, c325, c326, c327, c328, c329, c330, c331, c332, c333, c334, c335, c336, c337, c338, c339, c340, c341, c342, c343, c344, c345, c346, c347, c348, c349, c350, c351⟩ := hcopy10
  have hcopy11 := hcopy.1.1.2.2.2.1
  simp only [publicBatchWrapper4.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy11
  obtain ⟨c352, c353, c354, c355, c356, c357, c358, c359, c360, c361, c362, c363, c364, c365, c366, c367, c368, c369, c370, c371, c372, c373, c374, c375, c376, c377, c378, c379, c380, c381, c382, c383⟩ := hcopy11
  have hcopy12 := hcopy.1.1.2.2.2.2
  simp only [publicBatchWrapper4.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy12
  obtain ⟨c384, c385, c386, c387, c388, c389, c390, c391, c392, c393, c394, c395, c396, c397, c398, c399, c400, c401, c402, c403, c404, c405, c406, c407, c408, c409, c410, c411, c412, c413, c414, c415⟩ := hcopy12
  have hcopy13 := hcopy.1.2.1.1.1
  simp only [publicBatchWrapper4.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy13
  obtain ⟨c416, c417, c418, c419, c420, c421, c422, c423, c424, c425, c426, c427, c428, c429, c430, c431, c432, c433, c434, c435, c436, c437, c438, c439, c440, c441, c442, c443, c444, c445, c446, c447⟩ := hcopy13
  have hcopy14 := hcopy.1.2.1.1.2.1
  simp only [publicBatchWrapper4.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy14
  obtain ⟨c448, c449, c450, c451, c452, c453, c454, c455, c456, c457, c458, c459, c460, c461, c462, c463, c464, c465, c466, c467, c468, c469, c470, c471, c472, c473, c474, c475, c476, c477, c478, c479⟩ := hcopy14
  have hcopy15 := hcopy.1.2.1.1.2.2
  simp only [publicBatchWrapper4.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy15
  obtain ⟨c480, c481, c482, c483, c484, c485, c486, c487, c488, c489, c490, c491, c492, c493, c494, c495, c496, c497, c498, c499, c500, c501, c502, c503, c504, c505, c506, c507, c508, c509, c510, c511⟩ := hcopy15
  have hcopy16 := hcopy.1.2.1.2.1
  simp only [publicBatchWrapper4.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy16
  obtain ⟨c512, c513, c514, c515, c516, c517, c518, c519, c520, c521, c522, c523, c524, c525, c526, c527, c528, c529, c530, c531, c532, c533, c534, c535, c536, c537, c538, c539, c540, c541, c542, c543⟩ := hcopy16
  have hcopy17 := hcopy.1.2.1.2.2.1
  simp only [publicBatchWrapper4.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy17
  obtain ⟨c544, c545, c546, c547, c548, c549, c550, c551, c552, c553, c554, c555, c556, c557, c558, c559, c560, c561, c562, c563, c564, c565, c566, c567, c568, c569, c570, c571, c572, c573, c574, c575⟩ := hcopy17
  have hcopy18 := hcopy.1.2.1.2.2.2
  simp only [publicBatchWrapper4.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy18
  obtain ⟨c576, c577, c578, c579, c580, c581, c582, c583, c584, c585, c586, c587, c588, c589, c590, c591, c592, c593, c594, c595, c596, c597, c598, c599, c600, c601, c602, c603, c604, c605, c606, c607⟩ := hcopy18
  have hcopy19 := hcopy.1.2.2.1.1
  simp only [publicBatchWrapper4.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy19
  obtain ⟨c608, c609, c610, c611, c612, c613, c614, c615, c616, c617, c618, c619, c620, c621, c622, c623, c624, c625, c626, c627, c628, c629, c630, c631, c632, c633, c634, c635, c636, c637, c638, c639⟩ := hcopy19
  have hcopy20 := hcopy.1.2.2.1.2.1
  simp only [publicBatchWrapper4.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy20
  obtain ⟨c640, c641, c642, c643, c644, c645, c646, c647, c648, c649, c650, c651, c652, c653, c654, c655, c656, c657, c658, c659, c660, c661, c662, c663, c664, c665, c666, c667, c668, c669, c670, c671⟩ := hcopy20
  have hcopy21 := hcopy.1.2.2.1.2.2
  simp only [publicBatchWrapper4.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy21
  obtain ⟨c672, c673, c674, c675, c676, c677, c678, c679, c680, c681, c682, c683, c684, c685, c686, c687, c688, c689, c690, c691, c692, c693, c694, c695, c696, c697, c698, c699, c700, c701, c702, c703⟩ := hcopy21
  have hcopy22 := hcopy.1.2.2.2.1.1
  simp only [publicBatchWrapper4.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy22
  obtain ⟨c704, c705, c706, c707, c708, c709, c710, c711, c712, c713, c714, c715, c716, c717, c718, c719, c720, c721, c722, c723, c724, c725, c726, c727, c728, c729, c730, c731, c732, c733, c734, c735⟩ := hcopy22
  have hcopy23 := hcopy.1.2.2.2.1.2
  simp only [publicBatchWrapper4.copies23, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy23
  obtain ⟨c736, c737, c738, c739, c740, c741, c742, c743, c744, c745, c746, c747, c748, c749, c750, c751, c752, c753, c754, c755, c756, c757, c758, c759, c760, c761, c762, c763, c764, c765, c766, c767⟩ := hcopy23
  have hcopy24 := hcopy.1.2.2.2.2.1
  simp only [publicBatchWrapper4.copies24, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy24
  obtain ⟨c768, c769, c770, c771, c772, c773, c774, c775, c776, c777, c778, c779, c780, c781, c782, c783, c784, c785, c786, c787, c788, c789, c790, c791, c792, c793, c794, c795, c796, c797, c798, c799⟩ := hcopy24
  have hcopy25 := hcopy.1.2.2.2.2.2
  simp only [publicBatchWrapper4.copies25, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy25
  obtain ⟨c800, c801, c802, c803, c804, c805, c806, c807, c808, c809, c810, c811, c812, c813, c814, c815, c816, c817, c818, c819, c820, c821, c822, c823, c824, c825, c826, c827, c828, c829, c830, c831⟩ := hcopy25
  have hcopy26 := hcopy.2.1.1.1.1
  simp only [publicBatchWrapper4.copies26, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy26
  obtain ⟨c832, c833, c834, c835, c836, c837, c838, c839, c840, c841, c842, c843, c844, c845, c846, c847, c848, c849, c850, c851, c852, c853, c854, c855, c856, c857, c858, c859, c860, c861, c862, c863⟩ := hcopy26
  have hcopy27 := hcopy.2.1.1.1.2.1
  simp only [publicBatchWrapper4.copies27, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy27
  obtain ⟨c864, c865, c866, c867, c868, c869, c870, c871, c872, c873, c874, c875, c876, c877, c878, c879, c880, c881, c882, c883, c884, c885, c886, c887, c888, c889, c890, c891, c892, c893, c894, c895⟩ := hcopy27
  have hcopy28 := hcopy.2.1.1.1.2.2
  simp only [publicBatchWrapper4.copies28, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy28
  obtain ⟨c896, c897, c898, c899, c900, c901, c902, c903, c904, c905, c906, c907, c908, c909, c910, c911, c912, c913, c914, c915, c916, c917, c918, c919, c920, c921, c922, c923, c924, c925, c926, c927⟩ := hcopy28
  have hcopy29 := hcopy.2.1.1.2.1
  simp only [publicBatchWrapper4.copies29, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy29
  obtain ⟨c928, c929, c930, c931, c932, c933, c934, c935, c936, c937, c938, c939, c940, c941, c942, c943, c944, c945, c946, c947, c948, c949, c950, c951, c952, c953, c954, c955, c956, c957, c958, c959⟩ := hcopy29
  have hcopy30 := hcopy.2.1.1.2.2.1
  simp only [publicBatchWrapper4.copies30, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy30
  obtain ⟨c960, c961, c962, c963, c964, c965, c966, c967, c968, c969, c970, c971, c972, c973, c974, c975, c976, c977, c978, c979, c980, c981, c982, c983, c984, c985, c986, c987, c988, c989, c990, c991⟩ := hcopy30
  have hcopy31 := hcopy.2.1.1.2.2.2
  simp only [publicBatchWrapper4.copies31, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy31
  obtain ⟨c992, c993, c994, c995, c996, c997, c998, c999, c1000, c1001, c1002, c1003, c1004, c1005, c1006, c1007, c1008, c1009, c1010, c1011, c1012, c1013, c1014, c1015, c1016, c1017, c1018, c1019, c1020, c1021, c1022, c1023⟩ := hcopy31
  have hcopy32 := hcopy.2.1.2.1.1
  simp only [publicBatchWrapper4.copies32, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy32
  obtain ⟨c1024, c1025, c1026, c1027, c1028, c1029, c1030, c1031, c1032, c1033, c1034, c1035, c1036, c1037, c1038, c1039, c1040, c1041, c1042, c1043, c1044, c1045, c1046, c1047, c1048, c1049, c1050, c1051, c1052, c1053, c1054, c1055⟩ := hcopy32
  have hcopy33 := hcopy.2.1.2.1.2.1
  simp only [publicBatchWrapper4.copies33, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy33
  obtain ⟨c1056, c1057, c1058, c1059, c1060, c1061, c1062, c1063, c1064, c1065, c1066, c1067, c1068, c1069, c1070, c1071, c1072, c1073, c1074, c1075, c1076, c1077, c1078, c1079, c1080, c1081, c1082, c1083, c1084, c1085, c1086, c1087⟩ := hcopy33
  have hcopy34 := hcopy.2.1.2.1.2.2
  simp only [publicBatchWrapper4.copies34, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy34
  obtain ⟨c1088, c1089, c1090, c1091, c1092, c1093, c1094, c1095, c1096, c1097, c1098, c1099, c1100, c1101, c1102, c1103, c1104, c1105, c1106, c1107, c1108, c1109, c1110, c1111, c1112, c1113, c1114, c1115, c1116, c1117, c1118, c1119⟩ := hcopy34
  have hcopy35 := hcopy.2.1.2.2.1.1
  simp only [publicBatchWrapper4.copies35, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy35
  obtain ⟨c1120, c1121, c1122, c1123, c1124, c1125, c1126, c1127, c1128, c1129, c1130, c1131, c1132, c1133, c1134, c1135, c1136, c1137, c1138, c1139, c1140, c1141, c1142, c1143, c1144, c1145, c1146, c1147, c1148, c1149, c1150, c1151⟩ := hcopy35
  have hcopy36 := hcopy.2.1.2.2.1.2
  simp only [publicBatchWrapper4.copies36, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy36
  obtain ⟨c1152, c1153, c1154, c1155, c1156, c1157, c1158, c1159, c1160, c1161, c1162, c1163, c1164, c1165, c1166, c1167, c1168, c1169, c1170, c1171, c1172, c1173, c1174, c1175, c1176, c1177, c1178, c1179, c1180, c1181, c1182, c1183⟩ := hcopy36
  have hcopy37 := hcopy.2.1.2.2.2.1
  simp only [publicBatchWrapper4.copies37, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy37
  obtain ⟨c1184, c1185, c1186, c1187, c1188, c1189, c1190, c1191, c1192, c1193, c1194, c1195, c1196, c1197, c1198, c1199, c1200, c1201, c1202, c1203, c1204, c1205, c1206, c1207, c1208, c1209, c1210, c1211, c1212, c1213, c1214, c1215⟩ := hcopy37
  have hcopy38 := hcopy.2.1.2.2.2.2
  simp only [publicBatchWrapper4.copies38, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy38
  obtain ⟨c1216, c1217, c1218, c1219, c1220, c1221, c1222, c1223, c1224, c1225, c1226, c1227, c1228, c1229, c1230, c1231, c1232, c1233, c1234, c1235, c1236, c1237, c1238, c1239, c1240, c1241, c1242, c1243, c1244, c1245, c1246, c1247⟩ := hcopy38
  have hcopy39 := hcopy.2.2.1.1.1
  simp only [publicBatchWrapper4.copies39, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy39
  obtain ⟨c1248, c1249, c1250, c1251, c1252, c1253, c1254, c1255, c1256, c1257, c1258, c1259, c1260, c1261, c1262, c1263, c1264, c1265, c1266, c1267, c1268, c1269, c1270, c1271, c1272, c1273, c1274, c1275, c1276, c1277, c1278, c1279⟩ := hcopy39
  have hcopy40 := hcopy.2.2.1.1.2.1
  simp only [publicBatchWrapper4.copies40, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy40
  obtain ⟨c1280, c1281, c1282, c1283, c1284, c1285, c1286, c1287, c1288, c1289, c1290, c1291, c1292, c1293, c1294, c1295, c1296, c1297, c1298, c1299, c1300, c1301, c1302, c1303, c1304, c1305, c1306, c1307, c1308, c1309, c1310, c1311⟩ := hcopy40
  have hcopy41 := hcopy.2.2.1.1.2.2
  simp only [publicBatchWrapper4.copies41, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy41
  obtain ⟨c1312, c1313, c1314, c1315, c1316, c1317, c1318, c1319, c1320, c1321, c1322, c1323, c1324, c1325, c1326, c1327, c1328, c1329, c1330, c1331, c1332, c1333, c1334, c1335, c1336, c1337, c1338, c1339, c1340, c1341, c1342, c1343⟩ := hcopy41
  have hcopy42 := hcopy.2.2.1.2.1
  simp only [publicBatchWrapper4.copies42, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy42
  obtain ⟨c1344, c1345, c1346, c1347, c1348, c1349, c1350, c1351, c1352, c1353, c1354, c1355, c1356, c1357, c1358, c1359, c1360, c1361, c1362, c1363, c1364, c1365, c1366, c1367, c1368, c1369, c1370, c1371, c1372, c1373, c1374, c1375⟩ := hcopy42
  have hcopy43 := hcopy.2.2.1.2.2.1
  simp only [publicBatchWrapper4.copies43, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy43
  obtain ⟨c1376, c1377, c1378, c1379, c1380, c1381, c1382, c1383, c1384, c1385, c1386, c1387, c1388, c1389, c1390, c1391, c1392, c1393, c1394, c1395, c1396, c1397, c1398, c1399, c1400, c1401, c1402, c1403, c1404, c1405, c1406, c1407⟩ := hcopy43
  have hcopy44 := hcopy.2.2.1.2.2.2
  simp only [publicBatchWrapper4.copies44, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy44
  obtain ⟨c1408, c1409, c1410, c1411, c1412, c1413, c1414, c1415, c1416, c1417, c1418, c1419, c1420, c1421, c1422, c1423, c1424, c1425, c1426, c1427, c1428, c1429, c1430, c1431, c1432, c1433, c1434, c1435, c1436, c1437, c1438, c1439⟩ := hcopy44
  have hcopy45 := hcopy.2.2.2.1.1
  simp only [publicBatchWrapper4.copies45, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy45
  obtain ⟨c1440, c1441, c1442, c1443, c1444, c1445, c1446, c1447, c1448, c1449, c1450, c1451, c1452, c1453, c1454, c1455, c1456, c1457, c1458, c1459, c1460, c1461, c1462, c1463, c1464, c1465, c1466, c1467, c1468, c1469, c1470, c1471⟩ := hcopy45
  have hcopy46 := hcopy.2.2.2.1.2.1
  simp only [publicBatchWrapper4.copies46, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy46
  obtain ⟨c1472, c1473, c1474, c1475, c1476, c1477, c1478, c1479, c1480, c1481, c1482, c1483, c1484, c1485, c1486, c1487, c1488, c1489, c1490, c1491, c1492, c1493, c1494, c1495, c1496, c1497, c1498, c1499, c1500, c1501, c1502, c1503⟩ := hcopy46
  have hcopy47 := hcopy.2.2.2.1.2.2
  simp only [publicBatchWrapper4.copies47, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy47
  obtain ⟨c1504, c1505, c1506, c1507, c1508, c1509, c1510, c1511, c1512, c1513, c1514, c1515, c1516, c1517, c1518, c1519, c1520, c1521, c1522, c1523, c1524, c1525, c1526, c1527, c1528, c1529, c1530, c1531, c1532, c1533, c1534, c1535⟩ := hcopy47
  have hcopy48 := hcopy.2.2.2.2.1.1
  simp only [publicBatchWrapper4.copies48, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy48
  obtain ⟨c1536, c1537, c1538, c1539, c1540, c1541, c1542, c1543, c1544, c1545, c1546, c1547, c1548, c1549, c1550, c1551, c1552, c1553, c1554, c1555, c1556, c1557, c1558, c1559, c1560, c1561, c1562, c1563, c1564, c1565, c1566, c1567⟩ := hcopy48
  have hcopy49 := hcopy.2.2.2.2.1.2
  simp only [publicBatchWrapper4.copies49, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy49
  obtain ⟨c1568, c1569, c1570, c1571, c1572, c1573, c1574, c1575, c1576, c1577, c1578, c1579, c1580, c1581, c1582, c1583, c1584, c1585, c1586, c1587, c1588, c1589, c1590, c1591, c1592, c1593, c1594, c1595, c1596, c1597, c1598, c1599⟩ := hcopy49
  have hcopy50 := hcopy.2.2.2.2.2.1
  simp only [publicBatchWrapper4.copies50, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy50
  obtain ⟨c1600, c1601, c1602, c1603, c1604, c1605, c1606, c1607, c1608, c1609, c1610, c1611, c1612, c1613, c1614, c1615, c1616, c1617, c1618, c1619, c1620, c1621, c1622, c1623, c1624, c1625, c1626, c1627, c1628, c1629, c1630, c1631⟩ := hcopy50
  have hcopy51 := hcopy.2.2.2.2.2.2
  simp only [publicBatchWrapper4.copies51, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hcopy51
  obtain ⟨c1632, c1633, c1634, c1635, c1636, c1637, c1638, c1639, c1640, c1641, c1642, c1643, c1644, c1645, c1646, c1647, c1648, c1649, c1650, c1651, c1652, c1653, c1654⟩ := hcopy51
  have hconst := h.2.2
  simp only [publicBatchWrapper4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
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
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c133, ← c134, ← c135] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c136, ← c137, ← c138] at e_2_3
  have e_0_17 := arithEq_of_rows h (row := 0) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c139, ← c140, k0, ← c141] at e_0_17
  have e_0_18 := arithEq_of_rows h (row := 0) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c144, k0, ← c145, k0, ← c146] at e_0_18
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c147, ← c148, ← c149] at e_2_4
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c150, ← c151, ← c152] at e_2_5
  have e_0_19 := arithEq_of_rows h (row := 0) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c153, ← c154, k0, ← c155] at e_0_19
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c158, k0, ← c159, k0, ← c160] at e_3_0
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c161, ← c162, ← c163] at e_2_6
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c164, ← c165, ← c166] at e_2_7
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c167, ← c168, k0, ← c169] at e_3_1
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c172, k0, ← c173, k0, ← c174] at e_3_2
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c175, ← c176, ← c177] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c178, ← c179, ← c180] at e_2_9
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c181, ← c182, k0, ← c183] at e_3_3
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c186, ← c187, ← c188] at e_2_10
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c189, ← c190, ← c191] at e_2_11
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c192, ← c193, ← c194] at e_2_12
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c195, k0, ← c196, k0, ← c197] at e_3_4
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c198, ← c199, ← c200] at e_2_13
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c201, ← c202, ← c203] at e_2_14
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c204, ← c205, k0, ← c206] at e_3_5
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c209, k0, ← c210, k0, ← c211] at e_3_6
  have e_2_15 := arithEq_of_rows h (row := 2) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c212, ← c213, ← c214] at e_2_15
  have e_2_16 := arithEq_of_rows h (row := 2) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c215, ← c216, ← c217] at e_2_16
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c218, ← c219, k0, ← c220] at e_3_7
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c223, k0, ← c224, k0, ← c225] at e_3_8
  have e_2_17 := arithEq_of_rows h (row := 2) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c226, ← c227, ← c228] at e_2_17
  have e_2_18 := arithEq_of_rows h (row := 2) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c229, ← c230, ← c231] at e_2_18
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c232, ← c233, k0, ← c234] at e_3_9
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c237, k0, ← c238, k0, ← c239] at e_3_10
  have e_2_19 := arithEq_of_rows h (row := 2) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c240, ← c241, ← c242] at e_2_19
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c243, ← c244, ← c245] at e_4_0
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c246, ← c247, k0, ← c248] at e_3_11
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c251, ← c252, ← c253] at e_4_1
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c254, ← c255, ← c256] at e_4_2
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c257, ← c258, ← c259] at e_4_3
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c260, k0, ← c261, k0, ← c262] at e_3_12
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c263, ← c264, ← c265, k1] at e_3_13
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c266, ← c267, ← c268, k1] at e_3_14
  have e_3_15 := arithEq_of_rows h (row := 3) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c269, ← c270, ← c271, k1] at e_3_15
  have e_3_16 := arithEq_of_rows h (row := 3) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c272, ← c273, ← c274, k1] at e_3_16
  have e_3_17 := arithEq_of_rows h (row := 3) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c275, ← c276, ← c277, k1] at e_3_17
  have e_3_18 := arithEq_of_rows h (row := 3) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c278, ← c279, ← c280, k1] at e_3_18
  have e_3_19 := arithEq_of_rows h (row := 3) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c281, ← c282, ← c283, k1] at e_3_19
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c284, k0, ← c285, k0, ← c286] at e_5_0
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c287, k0, ← c288, k0, ← c289] at e_5_1
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c290, ← c291, ← c292] at e_4_4
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c293, ← c294, ← c295] at e_5_2
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c296, ← c297, ← c298] at e_5_3
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c299, ← c300, ← c301] at e_5_4
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c302, ← c303, ← c304] at e_5_5
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c305, ← c306, ← c307] at e_5_6
  have e_5_7 := arithEq_of_rows h (row := 5) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c308, ← c309, ← c310] at e_5_7
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c311, ← c312, ← c313] at e_5_8
  have e_5_9 := arithEq_of_rows h (row := 5) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c314, ← c315, ← c316] at e_5_9
  have e_5_10 := arithEq_of_rows h (row := 5) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c317, ← c318, ← c319] at e_5_10
  have e_5_11 := arithEq_of_rows h (row := 5) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c320, ← c321, ← c322] at e_5_11
  have e_5_12 := arithEq_of_rows h (row := 5) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c323, ← c324, ← c325] at e_5_12
  have e_5_13 := arithEq_of_rows h (row := 5) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c326, ← c327, ← c328] at e_5_13
  have e_5_14 := arithEq_of_rows h (row := 5) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c329, ← c330, ← c331] at e_5_14
  have e_5_15 := arithEq_of_rows h (row := 5) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_15
  simp only [← c332, ← c333, ← c334] at e_5_15
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c335, ← c336, ← c337] at e_6_0
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c338, ← c339, k0, ← c340] at e_7_0
  have e_5_16 := arithEq_of_rows h (row := 5) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_16
  simp only [← c341, k0, ← c342, k0, ← c343] at e_5_16
  have e_5_17 := arithEq_of_rows h (row := 5) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_17
  simp only [← c344, k0, ← c345, k0, ← c346] at e_5_17
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c347, ← c348, ← c349] at e_4_5
  have e_5_18 := arithEq_of_rows h (row := 5) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_18
  simp only [← c350, ← c351, ← c352] at e_5_18
  have e_5_19 := arithEq_of_rows h (row := 5) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_5_19
  simp only [← c353, ← c354, ← c355] at e_5_19
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c356, ← c357, ← c358] at e_8_0
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c359, ← c360, ← c361] at e_8_1
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c362, ← c363, ← c364] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c365, ← c366, ← c367] at e_8_3
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c368, ← c369, ← c370] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c371, ← c372, ← c373] at e_8_5
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c374, ← c375, ← c376] at e_8_6
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c377, ← c378, ← c379] at e_8_7
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c380, ← c381, ← c382] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c383, ← c384, ← c385] at e_8_9
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c386, ← c387, ← c388] at e_8_10
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c389, ← c390, ← c391] at e_8_11
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c392, ← c393, ← c394] at e_6_1
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c395, ← c396, k0, ← c397] at e_7_1
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c398, k0, ← c399, k0, ← c400] at e_8_12
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c401, k0, ← c402, k0, ← c403] at e_8_13
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c404, ← c405, ← c406] at e_4_6
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c407, ← c408, ← c409] at e_8_14
  have e_8_15 := arithEq_of_rows h (row := 8) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_15
  simp only [← c410, ← c411, ← c412] at e_8_15
  have e_8_16 := arithEq_of_rows h (row := 8) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_16
  simp only [← c413, ← c414, ← c415] at e_8_16
  have e_8_17 := arithEq_of_rows h (row := 8) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_17
  simp only [← c416, ← c417, ← c418] at e_8_17
  have e_8_18 := arithEq_of_rows h (row := 8) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_18
  simp only [← c419, ← c420, ← c421] at e_8_18
  have e_8_19 := arithEq_of_rows h (row := 8) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_8_19
  simp only [← c422, ← c423, ← c424] at e_8_19
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c425, ← c426, ← c427] at e_9_0
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c428, ← c429, ← c430] at e_9_1
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c431, ← c432, ← c433] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c434, ← c435, ← c436] at e_9_3
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c437, ← c438, ← c439] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c440, ← c441, ← c442] at e_9_5
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c443, ← c444, ← c445] at e_9_6
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c446, ← c447, ← c448] at e_9_7
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c449, ← c450, ← c451] at e_6_2
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c452, ← c453, k0, ← c454] at e_7_2
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c455, k0, ← c456, k0, ← c457] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c458, ← c459, k0, ← c460] at e_9_9
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c461, ← c462, ← c463] at e_4_7
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c464, ← c465, ← c466] at e_4_8
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c467, ← c468, k0, ← c469] at e_9_10
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c472, ← c473, ← c474] at e_6_3
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c475, ← c476, k0, ← c477] at e_7_3
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c479, k0, ← c480, k0, ← c481] at e_9_11
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c482, ← c483, k0, ← c484] at e_9_12
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c485, ← c486, ← c487] at e_4_9
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c488, ← c489, ← c490] at e_4_10
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c491, ← c492, k0, ← c493] at e_9_13
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c496, ← c497, ← c498] at e_6_4
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c499, ← c500, k0, ← c501] at e_7_4
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c503, k0, ← c504, k0, ← c505] at e_9_14
  have e_9_15 := arithEq_of_rows h (row := 9) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_15
  simp only [← c506, ← c507, k0, ← c508] at e_9_15
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c509, ← c510, ← c511] at e_4_11
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c512, ← c513, ← c514] at e_4_12
  have e_9_16 := arithEq_of_rows h (row := 9) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_16
  simp only [← c515, ← c516, k0, ← c517] at e_9_16
  have e_9_17 := arithEq_of_rows h (row := 9) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_17
  simp only [← c520, k0, ← c521, k0, ← c522] at e_9_17
  have e_9_18 := arithEq_of_rows h (row := 9) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_18
  simp only [← c523, ← c524, k0, ← c525] at e_9_18
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c526, ← c527, ← c528] at e_4_13
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c529, ← c530, ← c531] at e_4_14
  have e_9_19 := arithEq_of_rows h (row := 9) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_9_19
  simp only [← c532, ← c533, k0, ← c534] at e_9_19
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c537, k0, ← c538, k0, ← c539] at e_10_0
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c540, ← c541, k0, ← c542] at e_10_1
  have e_4_15 := arithEq_of_rows h (row := 4) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_15
  simp only [← c543, ← c544, ← c545] at e_4_15
  have e_4_16 := arithEq_of_rows h (row := 4) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_16
  simp only [← c546, ← c547, ← c548] at e_4_16
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c549, ← c550, k0, ← c551] at e_10_2
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c554, k0, ← c555, k0, ← c556] at e_10_3
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c557, ← c558, k0, ← c559] at e_10_4
  have e_4_17 := arithEq_of_rows h (row := 4) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_17
  simp only [← c560, ← c561, ← c562] at e_4_17
  have e_4_18 := arithEq_of_rows h (row := 4) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_18
  simp only [← c563, ← c564, ← c565] at e_4_18
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c566, ← c567, k0, ← c568] at e_10_5
  have e_4_19 := arithEq_of_rows h (row := 4) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_4_19
  simp only [← c571, ← c572, ← c573] at e_4_19
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c574, ← c575, ← c576] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c577, ← c578, ← c579] at e_11_1
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c580, ← c581, ← c582] at e_6_5
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c583, ← c584, k0, ← c585] at e_7_5
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c587, k0, ← c588, k0, ← c589] at e_10_6
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c590, ← c591, k0, ← c592] at e_10_7
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c593, ← c594, ← c595] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c596, ← c597, ← c598] at e_11_3
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c599, ← c600, k0, ← c601] at e_10_8
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c604, ← c605, ← c606] at e_6_6
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c607, ← c608, k0, ← c609] at e_7_6
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c611, k0, ← c612, k0, ← c613] at e_10_9
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c614, ← c615, k0, ← c616] at e_10_10
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c617, ← c618, ← c619] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c620, ← c621, ← c622] at e_11_5
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c623, ← c624, k0, ← c625] at e_10_11
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c628, ← c629, ← c630] at e_6_7
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c631, ← c632, k0, ← c633] at e_7_7
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c635, k0, ← c636, k0, ← c637] at e_10_12
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c638, ← c639, k0, ← c640] at e_10_13
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c641, ← c642, ← c643] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c644, ← c645, ← c646] at e_11_7
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c647, ← c648, k0, ← c649] at e_10_14
  have e_10_15 := arithEq_of_rows h (row := 10) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_15
  simp only [← c652, k0, ← c653, k0, ← c654] at e_10_15
  have e_10_16 := arithEq_of_rows h (row := 10) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_16
  simp only [← c655, ← c656, k0, ← c657] at e_10_16
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c658, ← c659, ← c660] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c661, ← c662, ← c663] at e_11_9
  have e_10_17 := arithEq_of_rows h (row := 10) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_17
  simp only [← c664, ← c665, k0, ← c666] at e_10_17
  have e_10_18 := arithEq_of_rows h (row := 10) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_18
  simp only [← c669, k0, ← c670, k0, ← c671] at e_10_18
  have e_10_19 := arithEq_of_rows h (row := 10) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_10_19
  simp only [← c672, ← c673, k0, ← c674] at e_10_19
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c675, ← c676, ← c677] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c678, ← c679, ← c680] at e_11_11
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c681, ← c682, k0, ← c683] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c686, k0, ← c687, k0, ← c688] at e_12_1
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c689, ← c690, k0, ← c691] at e_12_2
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c692, ← c693, ← c694] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c695, ← c696, ← c697] at e_11_13
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c698, ← c699, k0, ← c700] at e_12_3
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c703, ← c704, ← c705] at e_11_14
  have e_11_15 := arithEq_of_rows h (row := 11) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c706, ← c707, ← c708] at e_11_15
  have e_11_16 := arithEq_of_rows h (row := 11) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c709, ← c710, ← c711] at e_11_16
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c712, ← c713, ← c714] at e_6_8
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c715, ← c716, k0, ← c717] at e_7_8
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c719, k0, ← c720, k0, ← c721] at e_12_4
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c722, ← c723, k0, ← c724] at e_12_5
  have e_11_17 := arithEq_of_rows h (row := 11) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c725, ← c726, ← c727] at e_11_17
  have e_11_18 := arithEq_of_rows h (row := 11) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c728, ← c729, ← c730] at e_11_18
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c731, ← c732, k0, ← c733] at e_12_6
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c736, ← c737, ← c738] at e_6_9
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c739, ← c740, k0, ← c741] at e_7_9
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c743, k0, ← c744, k0, ← c745] at e_12_7
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c746, ← c747, k0, ← c748] at e_12_8
  have e_11_19 := arithEq_of_rows h (row := 11) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c749, ← c750, ← c751] at e_11_19
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c752, ← c753, ← c754] at e_13_0
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c755, ← c756, k0, ← c757] at e_12_9
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c760, ← c761, ← c762] at e_6_10
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c763, ← c764, k0, ← c765] at e_7_10
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c767, k0, ← c768, k0, ← c769] at e_12_10
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c770, ← c771, k0, ← c772] at e_12_11
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c773, ← c774, ← c775] at e_13_1
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c776, ← c777, ← c778] at e_13_2
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c779, ← c780, k0, ← c781] at e_12_12
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c784, k0, ← c785, k0, ← c786] at e_12_13
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c787, ← c788, k0, ← c789] at e_12_14
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c790, ← c791, ← c792] at e_13_3
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c793, ← c794, ← c795] at e_13_4
  have e_12_15 := arithEq_of_rows h (row := 12) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c796, ← c797, k0, ← c798] at e_12_15
  have e_12_16 := arithEq_of_rows h (row := 12) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c801, k0, ← c802, k0, ← c803] at e_12_16
  have e_12_17 := arithEq_of_rows h (row := 12) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c804, ← c805, k0, ← c806] at e_12_17
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c807, ← c808, ← c809] at e_13_5
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c810, ← c811, ← c812] at e_13_6
  have e_12_18 := arithEq_of_rows h (row := 12) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c813, ← c814, k0, ← c815] at e_12_18
  have e_12_19 := arithEq_of_rows h (row := 12) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c818, k0, ← c819, k0, ← c820] at e_12_19
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c821, ← c822, k0, ← c823] at e_14_0
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c824, ← c825, ← c826] at e_13_7
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c827, ← c828, ← c829] at e_13_8
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c830, ← c831, k0, ← c832] at e_14_1
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c835, ← c836, ← c837] at e_13_9
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c838, ← c839, ← c840] at e_13_10
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c841, ← c842, ← c843] at e_13_11
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c844, ← c845, ← c846] at e_6_11
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c847, ← c848, k0, ← c849] at e_7_11
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c851, k0, ← c852, k0, ← c853] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c854, ← c855, k0, ← c856] at e_14_3
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c857, ← c858, ← c859] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c860, ← c861, ← c862] at e_13_13
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c863, ← c864, k0, ← c865] at e_14_4
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c868, ← c869, ← c870] at e_6_12
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c871, ← c872, k0, ← c873] at e_7_12
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c875, k0, ← c876, k0, ← c877] at e_14_5
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c878, ← c879, k0, ← c880] at e_14_6
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c881, ← c882, ← c883] at e_13_14
  have e_13_15 := arithEq_of_rows h (row := 13) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c884, ← c885, ← c886] at e_13_15
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c887, ← c888, k0, ← c889] at e_14_7
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c892, ← c893, ← c894] at e_6_13
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c895, ← c896, k0, ← c897] at e_7_13
  have e_14_8 := arithEq_of_rows h (row := 14) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c899, k0, ← c900, k0, ← c901] at e_14_8
  have e_14_9 := arithEq_of_rows h (row := 14) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c902, ← c903, k0, ← c904] at e_14_9
  have e_13_16 := arithEq_of_rows h (row := 13) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c905, ← c906, ← c907] at e_13_16
  have e_13_17 := arithEq_of_rows h (row := 13) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c908, ← c909, ← c910] at e_13_17
  have e_14_10 := arithEq_of_rows h (row := 14) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c911, ← c912, k0, ← c913] at e_14_10
  have e_14_11 := arithEq_of_rows h (row := 14) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c916, k0, ← c917, k0, ← c918] at e_14_11
  have e_14_12 := arithEq_of_rows h (row := 14) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c919, ← c920, k0, ← c921] at e_14_12
  have e_13_18 := arithEq_of_rows h (row := 13) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c922, ← c923, ← c924] at e_13_18
  have e_13_19 := arithEq_of_rows h (row := 13) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c925, ← c926, ← c927] at e_13_19
  have e_14_13 := arithEq_of_rows h (row := 14) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c928, ← c929, k0, ← c930] at e_14_13
  have e_14_14 := arithEq_of_rows h (row := 14) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c933, k0, ← c934, k0, ← c935] at e_14_14
  have e_14_15 := arithEq_of_rows h (row := 14) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_15
  simp only [← c936, ← c937, k0, ← c938] at e_14_15
  have e_15_0 := arithEq_of_rows h (row := 15) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c939, ← c940, ← c941] at e_15_0
  have e_15_1 := arithEq_of_rows h (row := 15) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c942, ← c943, ← c944] at e_15_1
  have e_14_16 := arithEq_of_rows h (row := 14) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_16
  simp only [← c945, ← c946, k0, ← c947] at e_14_16
  have e_14_17 := arithEq_of_rows h (row := 14) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_17
  simp only [← c950, k0, ← c951, k0, ← c952] at e_14_17
  have e_14_18 := arithEq_of_rows h (row := 14) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_18
  simp only [← c953, ← c954, k0, ← c955] at e_14_18
  have e_15_2 := arithEq_of_rows h (row := 15) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c956, ← c957, ← c958] at e_15_2
  have e_15_3 := arithEq_of_rows h (row := 15) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c959, ← c960, ← c961] at e_15_3
  have e_14_19 := arithEq_of_rows h (row := 14) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_14_19
  simp only [← c962, ← c963, k0, ← c964] at e_14_19
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c967, ← c968, ← c969] at e_15_4
  have e_15_5 := arithEq_of_rows h (row := 15) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c970, ← c971, ← c972] at e_15_5
  have e_15_6 := arithEq_of_rows h (row := 15) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c973, ← c974, ← c975] at e_15_6
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c976, ← c977, ← c978] at e_6_14
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c979, ← c980, k0, ← c981] at e_7_14
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c983, ← c984, ← c985] at e_16_0
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c986, ← c987, k1, ← c988] at e_16_1
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c989, ← c990, ← c991] at e_16_2
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c992, ← c993, k1, ← c994] at e_16_3
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c995, ← c996, ← c997] at e_16_4
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c998, ← c999, k1, ← c1000] at e_16_5
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c1001, ← c1002, ← c1003] at e_16_6
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c1004, ← c1005, k1, ← c1006] at e_16_7
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c1007, ← c1008, ← c1009] at e_16_8
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c1010, ← c1011, k1, ← c1012] at e_16_9
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c1013, ← c1014, ← c1015] at e_16_10
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1016, ← c1017, k1, ← c1018] at e_16_11
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1019, ← c1020, ← c1021] at e_16_12
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1022, ← c1023, k1, ← c1024] at e_16_13
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1025, ← c1026, ← c1027] at e_16_14
  have e_16_15 := arithEq_of_rows h (row := 16) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_15
  simp only [← c1028, ← c1029, k1, ← c1030] at e_16_15
  have e_16_16 := arithEq_of_rows h (row := 16) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_16
  simp only [← c1031, ← c1032, ← c1033] at e_16_16
  have e_16_17 := arithEq_of_rows h (row := 16) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_17
  simp only [← c1034, ← c1035, k1, ← c1036] at e_16_17
  have e_16_18 := arithEq_of_rows h (row := 16) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_18
  simp only [← c1037, ← c1038, ← c1039] at e_16_18
  have e_16_19 := arithEq_of_rows h (row := 16) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_16_19
  simp only [← c1040, ← c1041, k1, ← c1042] at e_16_19
  have e_17_0 := arithEq_of_rows h (row := 17) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c1043, ← c1044, ← c1045] at e_17_0
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c1046, ← c1047, k1, ← c1048] at e_17_1
  have e_17_2 := arithEq_of_rows h (row := 17) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c1049, ← c1050, ← c1051] at e_17_2
  have e_17_3 := arithEq_of_rows h (row := 17) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c1052, ← c1053, k1, ← c1054] at e_17_3
  have e_17_4 := arithEq_of_rows h (row := 17) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c1055, ← c1056, ← c1057] at e_17_4
  have e_17_5 := arithEq_of_rows h (row := 17) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c1058, ← c1059, k1, ← c1060] at e_17_5
  have e_17_6 := arithEq_of_rows h (row := 17) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c1061, ← c1062, ← c1063] at e_17_6
  have e_17_7 := arithEq_of_rows h (row := 17) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c1064, ← c1065, k1, ← c1066] at e_17_7
  have e_17_8 := arithEq_of_rows h (row := 17) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c1067, ← c1068, ← c1069] at e_17_8
  have e_17_9 := arithEq_of_rows h (row := 17) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c1070, ← c1071, k1, ← c1072] at e_17_9
  have e_17_10 := arithEq_of_rows h (row := 17) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c1073, ← c1074, ← c1075] at e_17_10
  have e_17_11 := arithEq_of_rows h (row := 17) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c1076, ← c1077, k1, ← c1078] at e_17_11
  have e_17_12 := arithEq_of_rows h (row := 17) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c1079, ← c1080, ← c1081] at e_17_12
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c1082, ← c1083, k1, ← c1084] at e_17_13
  have e_17_14 := arithEq_of_rows h (row := 17) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c1085, ← c1086, ← c1087] at e_17_14
  have e_17_15 := arithEq_of_rows h (row := 17) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_15
  simp only [← c1088, ← c1089, k1, ← c1090] at e_17_15
  have e_17_16 := arithEq_of_rows h (row := 17) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_16
  simp only [← c1091, ← c1092, ← c1093] at e_17_16
  have e_17_17 := arithEq_of_rows h (row := 17) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_17
  simp only [← c1094, ← c1095, k1, ← c1096] at e_17_17
  have e_17_18 := arithEq_of_rows h (row := 17) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_18
  simp only [← c1097, ← c1098, ← c1099] at e_17_18
  have e_17_19 := arithEq_of_rows h (row := 17) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_17_19
  simp only [← c1100, ← c1101, k1, ← c1102] at e_17_19
  have e_18_0 := arithEq_of_rows h (row := 18) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c1103, ← c1104, ← c1105] at e_18_0
  have e_18_1 := arithEq_of_rows h (row := 18) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c1106, ← c1107, k1, ← c1108] at e_18_1
  have e_18_2 := arithEq_of_rows h (row := 18) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c1109, ← c1110, ← c1111] at e_18_2
  have e_18_3 := arithEq_of_rows h (row := 18) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c1112, ← c1113, k1, ← c1114] at e_18_3
  have e_18_4 := arithEq_of_rows h (row := 18) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c1115, ← c1116, ← c1117] at e_18_4
  have e_18_5 := arithEq_of_rows h (row := 18) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c1118, ← c1119, k1, ← c1120] at e_18_5
  have e_18_6 := arithEq_of_rows h (row := 18) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c1121, ← c1122, ← c1123] at e_18_6
  have e_18_7 := arithEq_of_rows h (row := 18) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c1124, ← c1125, k1, ← c1126] at e_18_7
  have e_18_8 := arithEq_of_rows h (row := 18) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c1127, ← c1128, ← c1129] at e_18_8
  have e_18_9 := arithEq_of_rows h (row := 18) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c1130, ← c1131, k1, ← c1132] at e_18_9
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c1133, ← c1134, ← c1135] at e_18_10
  have e_18_11 := arithEq_of_rows h (row := 18) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c1136, ← c1137, k1, ← c1138] at e_18_11
  have e_18_12 := arithEq_of_rows h (row := 18) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c1139, ← c1140, ← c1141] at e_18_12
  have e_18_13 := arithEq_of_rows h (row := 18) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c1142, ← c1143, k1, ← c1144] at e_18_13
  have e_18_14 := arithEq_of_rows h (row := 18) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c1145, ← c1146, ← c1147] at e_18_14
  have e_18_15 := arithEq_of_rows h (row := 18) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_15
  simp only [← c1148, ← c1149, k1, ← c1150] at e_18_15
  have e_18_16 := arithEq_of_rows h (row := 18) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_16
  simp only [← c1151, ← c1152, ← c1153] at e_18_16
  have e_18_17 := arithEq_of_rows h (row := 18) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_17
  simp only [← c1154, ← c1155, k1, ← c1156] at e_18_17
  have e_18_18 := arithEq_of_rows h (row := 18) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_18
  simp only [← c1157, ← c1158, ← c1159] at e_18_18
  have e_18_19 := arithEq_of_rows h (row := 18) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_18_19
  simp only [← c1160, ← c1161, k1, ← c1162] at e_18_19
  have e_19_0 := arithEq_of_rows h (row := 19) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c1163, ← c1164, ← c1165] at e_19_0
  have e_19_1 := arithEq_of_rows h (row := 19) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c1166, ← c1167, k1, ← c1168] at e_19_1
  have e_19_2 := arithEq_of_rows h (row := 19) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c1169, ← c1170, ← c1171] at e_19_2
  have e_19_3 := arithEq_of_rows h (row := 19) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c1172, ← c1173, k1, ← c1174] at e_19_3
  have e_19_4 := arithEq_of_rows h (row := 19) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c1175, ← c1176, ← c1177] at e_19_4
  have e_19_5 := arithEq_of_rows h (row := 19) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c1178, ← c1179, k1, ← c1180] at e_19_5
  have e_19_6 := arithEq_of_rows h (row := 19) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c1181, ← c1182, ← c1183] at e_19_6
  have e_19_7 := arithEq_of_rows h (row := 19) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c1184, ← c1185, k1, ← c1186] at e_19_7
  have e_19_8 := arithEq_of_rows h (row := 19) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c1187, ← c1188, ← c1189] at e_19_8
  have e_19_9 := arithEq_of_rows h (row := 19) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c1190, ← c1191, k1, ← c1192] at e_19_9
  have e_19_10 := arithEq_of_rows h (row := 19) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c1193, ← c1194, ← c1195] at e_19_10
  have e_19_11 := arithEq_of_rows h (row := 19) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c1196, ← c1197, k1, ← c1198] at e_19_11
  have e_19_12 := arithEq_of_rows h (row := 19) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c1199, ← c1200, ← c1201] at e_19_12
  have e_19_13 := arithEq_of_rows h (row := 19) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c1202, ← c1203, k1, ← c1204] at e_19_13
  have e_19_14 := arithEq_of_rows h (row := 19) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c1205, ← c1206, ← c1207] at e_19_14
  have e_19_15 := arithEq_of_rows h (row := 19) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_15
  simp only [← c1208, ← c1209, k1, ← c1210] at e_19_15
  have e_19_16 := arithEq_of_rows h (row := 19) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_16
  simp only [← c1211, ← c1212, ← c1213] at e_19_16
  have e_19_17 := arithEq_of_rows h (row := 19) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_17
  simp only [← c1214, ← c1215, k1, ← c1216] at e_19_17
  have e_19_18 := arithEq_of_rows h (row := 19) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_18
  simp only [← c1217, ← c1218, ← c1219] at e_19_18
  have e_19_19 := arithEq_of_rows h (row := 19) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_19_19
  simp only [← c1220, ← c1221, k1, ← c1222] at e_19_19
  have e_20_0 := arithEq_of_rows h (row := 20) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_0
  simp only [← c1223, ← c1224, ← c1225] at e_20_0
  have e_20_1 := arithEq_of_rows h (row := 20) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_1
  simp only [← c1226, ← c1227, k1, ← c1228] at e_20_1
  have e_20_2 := arithEq_of_rows h (row := 20) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_2
  simp only [← c1229, ← c1230, ← c1231] at e_20_2
  have e_20_3 := arithEq_of_rows h (row := 20) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_3
  simp only [← c1232, ← c1233, k1, ← c1234] at e_20_3
  have e_20_4 := arithEq_of_rows h (row := 20) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_4
  simp only [← c1235, ← c1236, ← c1237] at e_20_4
  have e_20_5 := arithEq_of_rows h (row := 20) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_5
  simp only [← c1238, ← c1239, k1, ← c1240] at e_20_5
  have e_20_6 := arithEq_of_rows h (row := 20) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_6
  simp only [← c1241, ← c1242, ← c1243] at e_20_6
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c1244, ← c1245, k1, ← c1246] at e_20_7
  have e_20_8 := arithEq_of_rows h (row := 20) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_8
  simp only [← c1247, ← c1248, ← c1249] at e_20_8
  have e_20_9 := arithEq_of_rows h (row := 20) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_9
  simp only [← c1250, ← c1251, k1, ← c1252] at e_20_9
  have e_20_10 := arithEq_of_rows h (row := 20) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_10
  simp only [← c1253, ← c1254, ← c1255] at e_20_10
  have e_20_11 := arithEq_of_rows h (row := 20) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_11
  simp only [← c1256, ← c1257, k1, ← c1258] at e_20_11
  have e_20_12 := arithEq_of_rows h (row := 20) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_12
  simp only [← c1259, ← c1260, ← c1261] at e_20_12
  have e_20_13 := arithEq_of_rows h (row := 20) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_13
  simp only [← c1262, ← c1263, k1, ← c1264] at e_20_13
  have e_20_14 := arithEq_of_rows h (row := 20) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_14
  simp only [← c1265, ← c1266, ← c1267] at e_20_14
  have e_20_15 := arithEq_of_rows h (row := 20) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_15
  simp only [← c1268, ← c1269, k1, ← c1270] at e_20_15
  have e_20_16 := arithEq_of_rows h (row := 20) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_16
  simp only [← c1271, ← c1272, ← c1273] at e_20_16
  have e_20_17 := arithEq_of_rows h (row := 20) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_17
  simp only [← c1274, ← c1275, k1, ← c1276] at e_20_17
  have e_20_18 := arithEq_of_rows h (row := 20) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_18
  simp only [← c1277, ← c1278, ← c1279] at e_20_18
  have e_20_19 := arithEq_of_rows h (row := 20) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_20_19
  simp only [← c1280, ← c1281, k1, ← c1282] at e_20_19
  have e_21_0 := arithEq_of_rows h (row := 21) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c1283, ← c1284, ← c1285] at e_21_0
  have e_21_1 := arithEq_of_rows h (row := 21) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c1286, ← c1287, k1, ← c1288] at e_21_1
  have e_21_2 := arithEq_of_rows h (row := 21) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c1289, ← c1290, ← c1291] at e_21_2
  have e_21_3 := arithEq_of_rows h (row := 21) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c1292, ← c1293, k1, ← c1294] at e_21_3
  have e_21_4 := arithEq_of_rows h (row := 21) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c1295, ← c1296, ← c1297] at e_21_4
  have e_21_5 := arithEq_of_rows h (row := 21) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c1298, ← c1299, k1, ← c1300] at e_21_5
  have e_21_6 := arithEq_of_rows h (row := 21) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c1301, ← c1302, ← c1303] at e_21_6
  have e_21_7 := arithEq_of_rows h (row := 21) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c1304, ← c1305, k1, ← c1306] at e_21_7
  have e_21_8 := arithEq_of_rows h (row := 21) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1307, ← c1308, ← c1309] at e_21_8
  have e_21_9 := arithEq_of_rows h (row := 21) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c1310, ← c1311, k1, ← c1312] at e_21_9
  have e_21_10 := arithEq_of_rows h (row := 21) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c1313, ← c1314, ← c1315] at e_21_10
  have e_21_11 := arithEq_of_rows h (row := 21) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c1316, ← c1317, k1, ← c1318] at e_21_11
  have e_21_12 := arithEq_of_rows h (row := 21) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c1319, ← c1320, ← c1321] at e_21_12
  have e_21_13 := arithEq_of_rows h (row := 21) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c1322, ← c1323, k1, ← c1324] at e_21_13
  have e_21_14 := arithEq_of_rows h (row := 21) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c1325, ← c1326, ← c1327] at e_21_14
  have e_21_15 := arithEq_of_rows h (row := 21) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_15
  simp only [← c1328, ← c1329, k1, ← c1330] at e_21_15
  have e_21_16 := arithEq_of_rows h (row := 21) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_16
  simp only [← c1331, ← c1332, ← c1333] at e_21_16
  have e_21_17 := arithEq_of_rows h (row := 21) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_17
  simp only [← c1334, ← c1335, k1, ← c1336] at e_21_17
  have e_21_18 := arithEq_of_rows h (row := 21) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_18
  simp only [← c1337, ← c1338, ← c1339] at e_21_18
  have e_21_19 := arithEq_of_rows h (row := 21) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_21_19
  simp only [← c1340, ← c1341, k1, ← c1342] at e_21_19
  have e_22_0 := arithEq_of_rows h (row := 22) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_0
  simp only [← c1343, ← c1344, ← c1345] at e_22_0
  have e_22_1 := arithEq_of_rows h (row := 22) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_1
  simp only [← c1346, ← c1347, k1, ← c1348] at e_22_1
  have e_22_2 := arithEq_of_rows h (row := 22) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_2
  simp only [← c1349, ← c1350, ← c1351] at e_22_2
  have e_22_3 := arithEq_of_rows h (row := 22) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_3
  simp only [← c1352, ← c1353, k1, ← c1354] at e_22_3
  have e_22_4 := arithEq_of_rows h (row := 22) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_4
  simp only [← c1355, ← c1356, ← c1357] at e_22_4
  have e_22_5 := arithEq_of_rows h (row := 22) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_5
  simp only [← c1358, ← c1359, k1, ← c1360] at e_22_5
  have e_22_6 := arithEq_of_rows h (row := 22) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_6
  simp only [← c1361, ← c1362, ← c1363] at e_22_6
  have e_22_7 := arithEq_of_rows h (row := 22) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_7
  simp only [← c1364, ← c1365, k1, ← c1366] at e_22_7
  have e_22_8 := arithEq_of_rows h (row := 22) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_8
  simp only [← c1367, ← c1368, ← c1369] at e_22_8
  have e_22_9 := arithEq_of_rows h (row := 22) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_9
  simp only [← c1370, ← c1371, k1, ← c1372] at e_22_9
  have e_22_10 := arithEq_of_rows h (row := 22) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_10
  simp only [← c1373, ← c1374, ← c1375] at e_22_10
  have e_22_11 := arithEq_of_rows h (row := 22) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c1376, ← c1377, k1, ← c1378] at e_22_11
  have e_22_12 := arithEq_of_rows h (row := 22) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_12
  simp only [← c1379, ← c1380, ← c1381] at e_22_12
  have e_22_13 := arithEq_of_rows h (row := 22) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_13
  simp only [← c1382, ← c1383, k1, ← c1384] at e_22_13
  have e_22_14 := arithEq_of_rows h (row := 22) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1385, ← c1386, ← c1387] at e_22_14
  have e_22_15 := arithEq_of_rows h (row := 22) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_15
  simp only [← c1388, ← c1389, k1, ← c1390] at e_22_15
  have e_22_16 := arithEq_of_rows h (row := 22) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_16
  simp only [← c1391, ← c1392, ← c1393] at e_22_16
  have e_22_17 := arithEq_of_rows h (row := 22) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_17
  simp only [← c1394, ← c1395, k1, ← c1396] at e_22_17
  have e_22_18 := arithEq_of_rows h (row := 22) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_18
  simp only [← c1397, ← c1398, ← c1399] at e_22_18
  have e_22_19 := arithEq_of_rows h (row := 22) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_22_19
  simp only [← c1400, ← c1401, k1, ← c1402] at e_22_19
  have e_23_0 := arithEq_of_rows h (row := 23) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_0
  simp only [← c1403, ← c1404, ← c1405] at e_23_0
  have e_23_1 := arithEq_of_rows h (row := 23) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_1
  simp only [← c1406, ← c1407, k1, ← c1408] at e_23_1
  have e_23_2 := arithEq_of_rows h (row := 23) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_2
  simp only [← c1409, ← c1410, ← c1411] at e_23_2
  have e_23_3 := arithEq_of_rows h (row := 23) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_3
  simp only [← c1412, ← c1413, k1, ← c1414] at e_23_3
  have e_23_4 := arithEq_of_rows h (row := 23) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_4
  simp only [← c1415, ← c1416, ← c1417] at e_23_4
  have e_23_5 := arithEq_of_rows h (row := 23) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_5
  simp only [← c1418, ← c1419, k1, ← c1420] at e_23_5
  have e_23_6 := arithEq_of_rows h (row := 23) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_6
  simp only [← c1421, ← c1422, ← c1423] at e_23_6
  have e_23_7 := arithEq_of_rows h (row := 23) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_7
  simp only [← c1424, ← c1425, k1, ← c1426] at e_23_7
  have e_23_8 := arithEq_of_rows h (row := 23) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_8
  simp only [← c1427, ← c1428, ← c1429] at e_23_8
  have e_23_9 := arithEq_of_rows h (row := 23) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_9
  simp only [← c1430, ← c1431, k1, ← c1432] at e_23_9
  have e_23_10 := arithEq_of_rows h (row := 23) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_10
  simp only [← c1433, ← c1434, ← c1435] at e_23_10
  have e_23_11 := arithEq_of_rows h (row := 23) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_11
  simp only [← c1436, ← c1437, k1, ← c1438] at e_23_11
  have e_23_12 := arithEq_of_rows h (row := 23) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_12
  simp only [← c1439, ← c1440, ← c1441] at e_23_12
  have e_23_13 := arithEq_of_rows h (row := 23) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_13
  simp only [← c1442, ← c1443, k1, ← c1444] at e_23_13
  have e_23_14 := arithEq_of_rows h (row := 23) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_14
  simp only [← c1445, ← c1446, ← c1447] at e_23_14
  have e_23_15 := arithEq_of_rows h (row := 23) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_15
  simp only [← c1448, ← c1449, k1, ← c1450] at e_23_15
  have e_23_16 := arithEq_of_rows h (row := 23) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_16
  simp only [← c1451, ← c1452, ← c1453] at e_23_16
  have e_23_17 := arithEq_of_rows h (row := 23) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_17
  simp only [← c1454, ← c1455, k1, ← c1456] at e_23_17
  have e_23_18 := arithEq_of_rows h (row := 23) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_18
  simp only [← c1457, ← c1458, ← c1459] at e_23_18
  have e_23_19 := arithEq_of_rows h (row := 23) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_23_19
  simp only [← c1460, ← c1461, k1, ← c1462] at e_23_19
  have e_24_0 := arithEq_of_rows h (row := 24) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_0
  simp only [← c1463, ← c1464, ← c1465] at e_24_0
  have e_24_1 := arithEq_of_rows h (row := 24) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_1
  simp only [← c1466, ← c1467, k1, ← c1468] at e_24_1
  have e_24_2 := arithEq_of_rows h (row := 24) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c1469, ← c1470, ← c1471] at e_24_2
  have e_24_3 := arithEq_of_rows h (row := 24) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_3
  simp only [← c1472, ← c1473, k1, ← c1474] at e_24_3
  have e_24_4 := arithEq_of_rows h (row := 24) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_4
  simp only [← c1475, ← c1476, ← c1477] at e_24_4
  have e_24_5 := arithEq_of_rows h (row := 24) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c1478, ← c1479, k1, ← c1480] at e_24_5
  have e_24_6 := arithEq_of_rows h (row := 24) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_6
  simp only [← c1481, ← c1482, ← c1483] at e_24_6
  have e_24_7 := arithEq_of_rows h (row := 24) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_7
  simp only [← c1484, ← c1485, k1, ← c1486] at e_24_7
  have e_24_8 := arithEq_of_rows h (row := 24) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_8
  simp only [← c1487, ← c1488, ← c1489] at e_24_8
  have e_24_9 := arithEq_of_rows h (row := 24) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_9
  simp only [← c1490, ← c1491, k1, ← c1492] at e_24_9
  have e_24_10 := arithEq_of_rows h (row := 24) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_10
  simp only [← c1493, ← c1494, ← c1495] at e_24_10
  have e_24_11 := arithEq_of_rows h (row := 24) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_11
  simp only [← c1496, ← c1497, k1, ← c1498] at e_24_11
  have e_24_12 := arithEq_of_rows h (row := 24) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_12
  simp only [← c1499, ← c1500, ← c1501] at e_24_12
  have e_24_13 := arithEq_of_rows h (row := 24) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_13
  simp only [← c1502, ← c1503, k1, ← c1504] at e_24_13
  have e_24_14 := arithEq_of_rows h (row := 24) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_14
  simp only [← c1505, ← c1506, ← c1507] at e_24_14
  have e_24_15 := arithEq_of_rows h (row := 24) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_15
  simp only [← c1508, ← c1509, k1, ← c1510] at e_24_15
  have e_24_16 := arithEq_of_rows h (row := 24) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_16
  simp only [← c1511, ← c1512, ← c1513] at e_24_16
  have e_24_17 := arithEq_of_rows h (row := 24) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_17
  simp only [← c1514, ← c1515, k1, ← c1516] at e_24_17
  have e_24_18 := arithEq_of_rows h (row := 24) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_18
  simp only [← c1517, ← c1518, ← c1519] at e_24_18
  have e_24_19 := arithEq_of_rows h (row := 24) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_24_19
  simp only [← c1520, ← c1521, k1, ← c1522] at e_24_19
  have e_25_0 := arithEq_of_rows h (row := 25) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_0
  simp only [← c1523, ← c1524, ← c1525] at e_25_0
  have e_25_1 := arithEq_of_rows h (row := 25) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_1
  simp only [← c1526, ← c1527, k1, ← c1528] at e_25_1
  have e_25_2 := arithEq_of_rows h (row := 25) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_2
  simp only [← c1529, ← c1530, ← c1531] at e_25_2
  have e_25_3 := arithEq_of_rows h (row := 25) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_3
  simp only [← c1532, ← c1533, k1, ← c1534] at e_25_3
  have e_25_4 := arithEq_of_rows h (row := 25) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_4
  simp only [← c1535, ← c1536, ← c1537] at e_25_4
  have e_25_5 := arithEq_of_rows h (row := 25) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_5
  simp only [← c1538, ← c1539, k1, ← c1540] at e_25_5
  have e_25_6 := arithEq_of_rows h (row := 25) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_6
  simp only [← c1541, ← c1542, ← c1543] at e_25_6
  have e_25_7 := arithEq_of_rows h (row := 25) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_7
  simp only [← c1544, ← c1545, k1, ← c1546] at e_25_7
  have e_25_8 := arithEq_of_rows h (row := 25) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_8
  simp only [← c1547, ← c1548, ← c1549] at e_25_8
  have e_25_9 := arithEq_of_rows h (row := 25) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_9
  simp only [← c1550, ← c1551, k1, ← c1552] at e_25_9
  have e_25_10 := arithEq_of_rows h (row := 25) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_10
  simp only [← c1553, ← c1554, ← c1555] at e_25_10
  have e_25_11 := arithEq_of_rows h (row := 25) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_11
  simp only [← c1556, ← c1557, k1, ← c1558] at e_25_11
  have e_25_12 := arithEq_of_rows h (row := 25) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_12
  simp only [← c1559, ← c1560, ← c1561] at e_25_12
  have e_25_13 := arithEq_of_rows h (row := 25) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_13
  simp only [← c1562, ← c1563, k1, ← c1564] at e_25_13
  have e_25_14 := arithEq_of_rows h (row := 25) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_14
  simp only [← c1565, ← c1566, ← c1567] at e_25_14
  have e_25_15 := arithEq_of_rows h (row := 25) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_15
  simp only [← c1568, ← c1569, k1, ← c1570] at e_25_15
  have e_25_16 := arithEq_of_rows h (row := 25) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_16
  simp only [← c1571, ← c1572, ← c1573] at e_25_16
  have e_25_17 := arithEq_of_rows h (row := 25) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_17
  simp only [← c1574, ← c1575, k1, ← c1576] at e_25_17
  have e_25_18 := arithEq_of_rows h (row := 25) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_18
  simp only [← c1577, ← c1578, ← c1579] at e_25_18
  have e_25_19 := arithEq_of_rows h (row := 25) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_25_19
  simp only [← c1580, ← c1581, k1, ← c1582] at e_25_19
  have e_26_0 := arithEq_of_rows h (row := 26) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1583, ← c1584, ← c1585] at e_26_0
  have e_26_1 := arithEq_of_rows h (row := 26) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1586, ← c1587, k1, ← c1588] at e_26_1
  have e_26_2 := arithEq_of_rows h (row := 26) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1589, ← c1590, ← c1591] at e_26_2
  have e_26_3 := arithEq_of_rows h (row := 26) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1592, ← c1593, k1, ← c1594] at e_26_3
  have e_26_4 := arithEq_of_rows h (row := 26) (i := 4) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1595, ← c1596, ← c1597] at e_26_4
  have e_26_5 := arithEq_of_rows h (row := 26) (i := 5) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1598, ← c1599, k1, ← c1600] at e_26_5
  have e_26_6 := arithEq_of_rows h (row := 26) (i := 6) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1601, ← c1602, ← c1603] at e_26_6
  have e_26_7 := arithEq_of_rows h (row := 26) (i := 7) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1604, ← c1605, k1, ← c1606] at e_26_7
  have e_26_8 := arithEq_of_rows h (row := 26) (i := 8) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1607, ← c1608, ← c1609] at e_26_8
  have e_26_9 := arithEq_of_rows h (row := 26) (i := 9) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1610, ← c1611, k1, ← c1612] at e_26_9
  have e_26_10 := arithEq_of_rows h (row := 26) (i := 10) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1613, ← c1614, ← c1615] at e_26_10
  have e_26_11 := arithEq_of_rows h (row := 26) (i := 11) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1616, ← c1617, k1, ← c1618] at e_26_11
  have e_26_12 := arithEq_of_rows h (row := 26) (i := 12) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1619, ← c1620, ← c1621] at e_26_12
  have e_26_13 := arithEq_of_rows h (row := 26) (i := 13) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1622, ← c1623, k1, ← c1624] at e_26_13
  have e_26_14 := arithEq_of_rows h (row := 26) (i := 14) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1625, ← c1626, ← c1627] at e_26_14
  have e_26_15 := arithEq_of_rows h (row := 26) (i := 15) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_15
  simp only [← c1628, ← c1629, k1, ← c1630] at e_26_15
  have e_26_16 := arithEq_of_rows h (row := 26) (i := 16) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_16
  simp only [← c1631, ← c1632, ← c1633] at e_26_16
  have e_26_17 := arithEq_of_rows h (row := 26) (i := 17) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_17
  simp only [← c1634, ← c1635, k1, ← c1636] at e_26_17
  have e_26_18 := arithEq_of_rows h (row := 26) (i := 18) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_18
  simp only [← c1637, ← c1638, ← c1639] at e_26_18
  have e_26_19 := arithEq_of_rows h (row := 26) (i := 19) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_26_19
  simp only [← c1640, ← c1641, k1, ← c1642] at e_26_19
  have e_27_0 := arithEq_of_rows h (row := 27) (i := 0) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1643, ← c1644, ← c1645] at e_27_0
  have e_27_1 := arithEq_of_rows h (row := 27) (i := 1) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1646, ← c1647, k1, ← c1648] at e_27_1
  have e_27_2 := arithEq_of_rows h (row := 27) (i := 2) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1649, ← c1650, ← c1651] at e_27_2
  have e_27_3 := arithEq_of_rows h (row := 27) (i := 3) rfl (by norm_num)
  norm_num only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1652, ← c1653, k1, ← c1654] at e_27_3
  have f0 : IsEqual (a (.virt 9488)) (a (.virt 38153)) (a (.virt 38154)) (a (.virt 38155)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_0
      linear_combination c12.trans k1 - hc
    · have hc := e_0_1
      simp only [e_0_0, e_1_1] at hc
      linear_combination c13.trans k1 - hc
  have f1 : IsEqual (a (.virt 9489)) (a (.virt 38153)) (a (.virt 38156)) (a (.virt 38157)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_2
      linear_combination c26.trans k1 - hc
    · have hc := e_0_3
      simp only [e_0_2, e_1_3] at hc
      linear_combination c27.trans k1 - hc
  have f2 : IsEqual (a (.virt 9490)) (a (.virt 38153)) (a (.virt 38158)) (a (.virt 38159)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_4
      linear_combination c40.trans k1 - hc
    · have hc := e_0_5
      simp only [e_0_4, e_1_5] at hc
      linear_combination c41.trans k1 - hc
  have f3 : IsEqual (a (.virt 9491)) (a (.virt 38153)) (a (.virt 38160)) (a (.virt 38161)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_6
      linear_combination c54.trans k1 - hc
    · have hc := e_0_7
      simp only [e_0_6, e_1_7] at hc
      linear_combination c55.trans k1 - hc
  have f4 : a (.wire 1 35) = band (a (.virt 38154)) (a (.virt 38156)) := by
    have hr := e_1_8
    simp only [band]
    linear_combination hr
  have f5 : a (.wire 1 39) = band (a (.virt 38158)) (a (.virt 38160)) := by
    have hr := e_1_9
    simp only [band]
    linear_combination hr
  have f6 : a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) := by
    have hr := e_1_10
    simp only [band]
    linear_combination hr
  have f7 : IsEqual (a (.virt 19025)) (a (.virt 38153)) (a (.virt 38162)) (a (.virt 38163)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_11
      linear_combination c77.trans k1 - hc
    · have hc := e_0_9
      simp only [e_0_8, e_1_12] at hc
      linear_combination c78.trans k1 - hc
  have f8 : IsEqual (a (.virt 19026)) (a (.virt 38153)) (a (.virt 38164)) (a (.virt 38165)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_13
      linear_combination c91.trans k1 - hc
    · have hc := e_0_11
      simp only [e_0_10, e_1_14] at hc
      linear_combination c92.trans k1 - hc
  have f9 : IsEqual (a (.virt 19027)) (a (.virt 38153)) (a (.virt 38166)) (a (.virt 38167)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_15
      linear_combination c105.trans k1 - hc
    · have hc := e_0_13
      simp only [e_0_12, e_1_16] at hc
      linear_combination c106.trans k1 - hc
  have f10 : IsEqual (a (.virt 19028)) (a (.virt 38153)) (a (.virt 38168)) (a (.virt 38169)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_1_17
      linear_combination c119.trans k1 - hc
    · have hc := e_0_15
      simp only [e_0_14, e_1_18] at hc
      linear_combination c120.trans k1 - hc
  have f11 : a (.wire 1 79) = band (a (.virt 38162)) (a (.virt 38164)) := by
    have hr := e_1_19
    simp only [band]
    linear_combination hr
  have f12 : a (.wire 2 3) = band (a (.virt 38166)) (a (.virt 38168)) := by
    have hr := e_2_0
    simp only [band]
    linear_combination hr
  have f13 : a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) := by
    have hr := e_2_1
    simp only [band]
    linear_combination hr
  have f14 : IsEqual (a (.virt 28562)) (a (.virt 38153)) (a (.virt 38170)) (a (.virt 38171)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_2
      linear_combination c142.trans k1 - hc
    · have hc := e_0_17
      simp only [e_0_16, e_2_3] at hc
      linear_combination c143.trans k1 - hc
  have f15 : IsEqual (a (.virt 28563)) (a (.virt 38153)) (a (.virt 38172)) (a (.virt 38173)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_4
      linear_combination c156.trans k1 - hc
    · have hc := e_0_19
      simp only [e_0_18, e_2_5] at hc
      linear_combination c157.trans k1 - hc
  have f16 : IsEqual (a (.virt 28564)) (a (.virt 38153)) (a (.virt 38174)) (a (.virt 38175)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_6
      linear_combination c170.trans k1 - hc
    · have hc := e_3_1
      simp only [e_3_0, e_2_7] at hc
      linear_combination c171.trans k1 - hc
  have f17 : IsEqual (a (.virt 28565)) (a (.virt 38153)) (a (.virt 38176)) (a (.virt 38177)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_8
      linear_combination c184.trans k1 - hc
    · have hc := e_3_3
      simp only [e_3_2, e_2_9] at hc
      linear_combination c185.trans k1 - hc
  have f18 : a (.wire 2 43) = band (a (.virt 38170)) (a (.virt 38172)) := by
    have hr := e_2_10
    simp only [band]
    linear_combination hr
  have f19 : a (.wire 2 47) = band (a (.virt 38174)) (a (.virt 38176)) := by
    have hr := e_2_11
    simp only [band]
    linear_combination hr
  have f20 : a (.wire 2 51) = band (a (.wire 2 43)) (a (.wire 2 47)) := by
    have hr := e_2_12
    simp only [band]
    linear_combination hr
  have f21 : IsEqual (a (.virt 38099)) (a (.virt 38153)) (a (.virt 38178)) (a (.virt 38179)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_13
      linear_combination c207.trans k1 - hc
    · have hc := e_3_5
      simp only [e_3_4, e_2_14] at hc
      linear_combination c208.trans k1 - hc
  have f22 : IsEqual (a (.virt 38100)) (a (.virt 38153)) (a (.virt 38180)) (a (.virt 38181)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_15
      linear_combination c221.trans k1 - hc
    · have hc := e_3_7
      simp only [e_3_6, e_2_16] at hc
      linear_combination c222.trans k1 - hc
  have f23 : IsEqual (a (.virt 38101)) (a (.virt 38153)) (a (.virt 38182)) (a (.virt 38183)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_17
      linear_combination c235.trans k1 - hc
    · have hc := e_3_9
      simp only [e_3_8, e_2_18] at hc
      linear_combination c236.trans k1 - hc
  have f24 : IsEqual (a (.virt 38102)) (a (.virt 38153)) (a (.virt 38184)) (a (.virt 38185)) := by
    simp only [k1]
    refine ⟨?_, ?_⟩
    · have hc := e_2_19
      linear_combination c249.trans k1 - hc
    · have hc := e_3_11
      simp only [e_3_10, e_4_0] at hc
      linear_combination c250.trans k1 - hc
  have f25 : a (.wire 4 7) = band (a (.virt 38178)) (a (.virt 38180)) := by
    have hr := e_4_1
    simp only [band]
    linear_combination hr
  have f26 : a (.wire 4 11) = band (a (.virt 38182)) (a (.virt 38184)) := by
    have hr := e_4_2
    simp only [band]
    linear_combination hr
  have f27 : a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11)) := by
    have hr := e_4_3
    simp only [band]
    linear_combination hr
  have f28 : a (.wire 3 51) = bnot (a (.wire 1 43)) := by
    have hr := e_3_12
    simp only [bnot]
    linear_combination hr
  have f29 : a (.virt 38152) = bnot (a (.virt 38153)) := by
    simp only [bnot, k0, k1]
    ring
  have f30 : a (.wire 3 51) = band (a (.wire 3 51)) (a (.virt 38152)) := by
    simp only [band, k0]
    ring
  have f31 : a (.wire 3 55) = bselect (a (.wire 3 51)) (a (.virt 9488)) (a (.virt 38153)) := by
    have hr := e_3_13
    simp only [bselect, k1]
    linear_combination hr
  have f32 : a (.wire 3 59) = bselect (a (.wire 3 51)) (a (.virt 9489)) (a (.virt 38153)) := by
    have hr := e_3_14
    simp only [bselect, k1]
    linear_combination hr
  have f33 : a (.wire 3 63) = bselect (a (.wire 3 51)) (a (.virt 9490)) (a (.virt 38153)) := by
    have hr := e_3_15
    simp only [bselect, k1]
    linear_combination hr
  have f34 : a (.wire 3 67) = bselect (a (.wire 3 51)) (a (.virt 9491)) (a (.virt 38153)) := by
    have hr := e_3_16
    simp only [bselect, k1]
    linear_combination hr
  have f35 : a (.wire 3 71) = bselect (a (.wire 3 51)) (a (.virt 9492)) (a (.virt 38153)) := by
    have hr := e_3_17
    simp only [bselect, k1]
    linear_combination hr
  have f36 : a (.wire 3 75) = bselect (a (.wire 3 51)) (a (.virt 9486)) (a (.virt 38153)) := by
    have hr := e_3_18
    simp only [bselect, k1]
    linear_combination hr
  have f37 : a (.wire 3 79) = bselect (a (.wire 3 51)) (a (.virt 9487)) (a (.virt 38153)) := by
    have hr := e_3_19
    simp only [bselect, k1]
    linear_combination hr
  have f38 : a (.wire 3 51) = bor (a (.virt 38153)) (a (.wire 3 51)) := by
    simp only [bor, k1]
    ring
  have f39 : a (.wire 5 3) = bnot (a (.wire 2 7)) := by
    have hr := e_5_0
    simp only [bnot]
    linear_combination hr
  have f40 : a (.wire 5 7) = bnot (a (.wire 3 51)) := by
    have hr := e_5_1
    simp only [bnot]
    linear_combination hr
  have f41 : a (.wire 4 19) = band (a (.wire 5 3)) (a (.wire 5 7)) := by
    have hr := e_4_4
    simp only [band]
    linear_combination hr
  have f42 : a (.wire 5 15) = bselect (a (.wire 4 19)) (a (.virt 19025)) (a (.wire 3 55)) := by
    have hr := e_5_3
    simp only [e_5_2] at hr
    simp only [bselect]
    linear_combination hr
  have f43 : a (.wire 5 23) = bselect (a (.wire 4 19)) (a (.virt 19026)) (a (.wire 3 59)) := by
    have hr := e_5_5
    simp only [e_5_4] at hr
    simp only [bselect]
    linear_combination hr
  have f44 : a (.wire 5 31) = bselect (a (.wire 4 19)) (a (.virt 19027)) (a (.wire 3 63)) := by
    have hr := e_5_7
    simp only [e_5_6] at hr
    simp only [bselect]
    linear_combination hr
  have f45 : a (.wire 5 39) = bselect (a (.wire 4 19)) (a (.virt 19028)) (a (.wire 3 67)) := by
    have hr := e_5_9
    simp only [e_5_8] at hr
    simp only [bselect]
    linear_combination hr
  have f46 : a (.wire 5 47) = bselect (a (.wire 4 19)) (a (.virt 19029)) (a (.wire 3 71)) := by
    have hr := e_5_11
    simp only [e_5_10] at hr
    simp only [bselect]
    linear_combination hr
  have f47 : a (.wire 5 55) = bselect (a (.wire 4 19)) (a (.virt 19023)) (a (.wire 3 75)) := by
    have hr := e_5_13
    simp only [e_5_12] at hr
    simp only [bselect]
    linear_combination hr
  have f48 : a (.wire 5 63) = bselect (a (.wire 4 19)) (a (.virt 19024)) (a (.wire 3 79)) := by
    have hr := e_5_15
    simp only [e_5_14] at hr
    simp only [bselect]
    linear_combination hr
  have f49 : a (.wire 7 3) = bor (a (.wire 3 51)) (a (.wire 5 3)) := by
    have hr := e_7_0
    simp only [e_6_0] at hr
    simp only [bor]
    linear_combination hr
  have f50 : a (.wire 5 67) = bnot (a (.wire 2 51)) := by
    have hr := e_5_16
    simp only [bnot]
    linear_combination hr
  have f51 : a (.wire 5 71) = bnot (a (.wire 7 3)) := by
    have hr := e_5_17
    simp only [bnot]
    linear_combination hr
  have f52 : a (.wire 4 23) = band (a (.wire 5 67)) (a (.wire 5 71)) := by
    have hr := e_4_5
    simp only [band]
    linear_combination hr
  have f53 : a (.wire 5 79) = bselect (a (.wire 4 23)) (a (.virt 28562)) (a (.wire 5 15)) := by
    have hr := e_5_19
    simp only [e_5_18] at hr
    simp only [bselect]
    linear_combination hr
  have f54 : a (.wire 8 7) = bselect (a (.wire 4 23)) (a (.virt 28563)) (a (.wire 5 23)) := by
    have hr := e_8_1
    simp only [e_8_0] at hr
    simp only [bselect]
    linear_combination hr
  have f55 : a (.wire 8 15) = bselect (a (.wire 4 23)) (a (.virt 28564)) (a (.wire 5 31)) := by
    have hr := e_8_3
    simp only [e_8_2] at hr
    simp only [bselect]
    linear_combination hr
  have f56 : a (.wire 8 23) = bselect (a (.wire 4 23)) (a (.virt 28565)) (a (.wire 5 39)) := by
    have hr := e_8_5
    simp only [e_8_4] at hr
    simp only [bselect]
    linear_combination hr
  have f57 : a (.wire 8 31) = bselect (a (.wire 4 23)) (a (.virt 28566)) (a (.wire 5 47)) := by
    have hr := e_8_7
    simp only [e_8_6] at hr
    simp only [bselect]
    linear_combination hr
  have f58 : a (.wire 8 39) = bselect (a (.wire 4 23)) (a (.virt 28560)) (a (.wire 5 55)) := by
    have hr := e_8_9
    simp only [e_8_8] at hr
    simp only [bselect]
    linear_combination hr
  have f59 : a (.wire 8 47) = bselect (a (.wire 4 23)) (a (.virt 28561)) (a (.wire 5 63)) := by
    have hr := e_8_11
    simp only [e_8_10] at hr
    simp only [bselect]
    linear_combination hr
  have f60 : a (.wire 7 7) = bor (a (.wire 7 3)) (a (.wire 5 67)) := by
    have hr := e_7_1
    simp only [e_6_1] at hr
    simp only [bor]
    linear_combination hr
  have f61 : a (.wire 8 51) = bnot (a (.wire 4 15)) := by
    have hr := e_8_12
    simp only [bnot]
    linear_combination hr
  have f62 : a (.wire 8 55) = bnot (a (.wire 7 7)) := by
    have hr := e_8_13
    simp only [bnot]
    linear_combination hr
  have f63 : a (.wire 4 27) = band (a (.wire 8 51)) (a (.wire 8 55)) := by
    have hr := e_4_6
    simp only [band]
    linear_combination hr
  have f64 : a (.wire 8 63) = bselect (a (.wire 4 27)) (a (.virt 38099)) (a (.wire 5 79)) := by
    have hr := e_8_15
    simp only [e_8_14] at hr
    simp only [bselect]
    linear_combination hr
  have f65 : a (.wire 8 71) = bselect (a (.wire 4 27)) (a (.virt 38100)) (a (.wire 8 7)) := by
    have hr := e_8_17
    simp only [e_8_16] at hr
    simp only [bselect]
    linear_combination hr
  have f66 : a (.wire 8 79) = bselect (a (.wire 4 27)) (a (.virt 38101)) (a (.wire 8 15)) := by
    have hr := e_8_19
    simp only [e_8_18] at hr
    simp only [bselect]
    linear_combination hr
  have f67 : a (.wire 9 7) = bselect (a (.wire 4 27)) (a (.virt 38102)) (a (.wire 8 23)) := by
    have hr := e_9_1
    simp only [e_9_0] at hr
    simp only [bselect]
    linear_combination hr
  have f68 : a (.wire 9 15) = bselect (a (.wire 4 27)) (a (.virt 38103)) (a (.wire 8 31)) := by
    have hr := e_9_3
    simp only [e_9_2] at hr
    simp only [bselect]
    linear_combination hr
  have f69 : a (.wire 9 23) = bselect (a (.wire 4 27)) (a (.virt 38097)) (a (.wire 8 39)) := by
    have hr := e_9_5
    simp only [e_9_4] at hr
    simp only [bselect]
    linear_combination hr
  have f70 : a (.wire 9 31) = bselect (a (.wire 4 27)) (a (.virt 38098)) (a (.wire 8 47)) := by
    have hr := e_9_7
    simp only [e_9_6] at hr
    simp only [bselect]
    linear_combination hr
  have f71 : a (.wire 7 11) = bor (a (.wire 7 7)) (a (.wire 8 51)) := by
    have hr := e_7_2
    simp only [e_6_2] at hr
    simp only [bor]
    linear_combination hr
  have f72 : IsEqual (a (.virt 9486)) (a (.wire 9 23)) (a (.virt 38186)) (a (.virt 38187)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_4_7
      simp only [e_9_9] at hc
      linear_combination c470.trans k1 - hc
    · have hc := e_9_10
      simp only [e_9_8, e_4_8, e_9_9] at hc
      linear_combination c471.trans k1 - hc
  have f73 : a (.wire 7 15) = bor (a (.wire 1 43)) (a (.virt 38186)) := by
    have hr := e_7_3
    simp only [e_6_3] at hr
    simp only [bor]
    linear_combination hr
  have f74 : a (.wire 7 15) = a (.virt 38152) := by
    exact c478
  have f75 : IsEqual (a (.virt 9487)) (a (.wire 9 31)) (a (.virt 38188)) (a (.virt 38189)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_4_9
      simp only [e_9_12] at hc
      linear_combination c494.trans k1 - hc
    · have hc := e_9_13
      simp only [e_9_11, e_4_10, e_9_12] at hc
      linear_combination c495.trans k1 - hc
  have f76 : a (.wire 7 19) = bor (a (.wire 1 43)) (a (.virt 38188)) := by
    have hr := e_7_4
    simp only [e_6_4] at hr
    simp only [bor]
    linear_combination hr
  have f77 : a (.wire 7 19) = a (.virt 38152) := by
    exact c502
  have f78 : IsEqual (a (.virt 9488)) (a (.wire 8 63)) (a (.virt 38190)) (a (.virt 38191)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_4_11
      simp only [e_9_15] at hc
      linear_combination c518.trans k1 - hc
    · have hc := e_9_16
      simp only [e_9_14, e_4_12, e_9_15] at hc
      linear_combination c519.trans k1 - hc
  have f79 : IsEqual (a (.virt 9489)) (a (.wire 8 71)) (a (.virt 38192)) (a (.virt 38193)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_4_13
      simp only [e_9_18] at hc
      linear_combination c535.trans k1 - hc
    · have hc := e_9_19
      simp only [e_9_17, e_4_14, e_9_18] at hc
      linear_combination c536.trans k1 - hc
  have f80 : IsEqual (a (.virt 9490)) (a (.wire 8 79)) (a (.virt 38194)) (a (.virt 38195)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_4_15
      simp only [e_10_1] at hc
      linear_combination c552.trans k1 - hc
    · have hc := e_10_2
      simp only [e_10_0, e_4_16, e_10_1] at hc
      linear_combination c553.trans k1 - hc
  have f81 : IsEqual (a (.virt 9491)) (a (.wire 9 7)) (a (.virt 38196)) (a (.virt 38197)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_4_17
      simp only [e_10_4] at hc
      linear_combination c569.trans k1 - hc
    · have hc := e_10_5
      simp only [e_10_3, e_4_18, e_10_4] at hc
      linear_combination c570.trans k1 - hc
  have f82 : a (.wire 4 79) = band (a (.virt 38190)) (a (.virt 38192)) := by
    have hr := e_4_19
    simp only [band]
    linear_combination hr
  have f83 : a (.wire 11 3) = band (a (.virt 38194)) (a (.virt 38196)) := by
    have hr := e_11_0
    simp only [band]
    linear_combination hr
  have f84 : a (.wire 11 7) = band (a (.wire 4 79)) (a (.wire 11 3)) := by
    have hr := e_11_1
    simp only [band]
    linear_combination hr
  have f85 : a (.wire 7 23) = bor (a (.wire 1 43)) (a (.wire 11 7)) := by
    have hr := e_7_5
    simp only [e_6_5] at hr
    simp only [bor]
    linear_combination hr
  have f86 : a (.wire 7 23) = a (.virt 38152) := by
    exact c586
  have f87 : IsEqual (a (.virt 19023)) (a (.wire 9 23)) (a (.virt 38198)) (a (.virt 38199)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_2
      simp only [e_10_7] at hc
      linear_combination c602.trans k1 - hc
    · have hc := e_10_8
      simp only [e_10_6, e_11_3, e_10_7] at hc
      linear_combination c603.trans k1 - hc
  have f88 : a (.wire 7 27) = bor (a (.wire 2 7)) (a (.virt 38198)) := by
    have hr := e_7_6
    simp only [e_6_6] at hr
    simp only [bor]
    linear_combination hr
  have f89 : a (.wire 7 27) = a (.virt 38152) := by
    exact c610
  have f90 : IsEqual (a (.virt 19024)) (a (.wire 9 31)) (a (.virt 38200)) (a (.virt 38201)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_4
      simp only [e_10_10] at hc
      linear_combination c626.trans k1 - hc
    · have hc := e_10_11
      simp only [e_10_9, e_11_5, e_10_10] at hc
      linear_combination c627.trans k1 - hc
  have f91 : a (.wire 7 31) = bor (a (.wire 2 7)) (a (.virt 38200)) := by
    have hr := e_7_7
    simp only [e_6_7] at hr
    simp only [bor]
    linear_combination hr
  have f92 : a (.wire 7 31) = a (.virt 38152) := by
    exact c634
  have f93 : IsEqual (a (.virt 19025)) (a (.wire 8 63)) (a (.virt 38202)) (a (.virt 38203)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_6
      simp only [e_10_13] at hc
      linear_combination c650.trans k1 - hc
    · have hc := e_10_14
      simp only [e_10_12, e_11_7, e_10_13] at hc
      linear_combination c651.trans k1 - hc
  have f94 : IsEqual (a (.virt 19026)) (a (.wire 8 71)) (a (.virt 38204)) (a (.virt 38205)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_8
      simp only [e_10_16] at hc
      linear_combination c667.trans k1 - hc
    · have hc := e_10_17
      simp only [e_10_15, e_11_9, e_10_16] at hc
      linear_combination c668.trans k1 - hc
  have f95 : IsEqual (a (.virt 19027)) (a (.wire 8 79)) (a (.virt 38206)) (a (.virt 38207)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_10
      simp only [e_10_19] at hc
      linear_combination c684.trans k1 - hc
    · have hc := e_12_0
      simp only [e_10_18, e_11_11, e_10_19] at hc
      linear_combination c685.trans k1 - hc
  have f96 : IsEqual (a (.virt 19028)) (a (.wire 9 7)) (a (.virt 38208)) (a (.virt 38209)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_12
      simp only [e_12_2] at hc
      linear_combination c701.trans k1 - hc
    · have hc := e_12_3
      simp only [e_12_1, e_11_13, e_12_2] at hc
      linear_combination c702.trans k1 - hc
  have f97 : a (.wire 11 59) = band (a (.virt 38202)) (a (.virt 38204)) := by
    have hr := e_11_14
    simp only [band]
    linear_combination hr
  have f98 : a (.wire 11 63) = band (a (.virt 38206)) (a (.virt 38208)) := by
    have hr := e_11_15
    simp only [band]
    linear_combination hr
  have f99 : a (.wire 11 67) = band (a (.wire 11 59)) (a (.wire 11 63)) := by
    have hr := e_11_16
    simp only [band]
    linear_combination hr
  have f100 : a (.wire 7 35) = bor (a (.wire 2 7)) (a (.wire 11 67)) := by
    have hr := e_7_8
    simp only [e_6_8] at hr
    simp only [bor]
    linear_combination hr
  have f101 : a (.wire 7 35) = a (.virt 38152) := by
    exact c718
  have f102 : IsEqual (a (.virt 28560)) (a (.wire 9 23)) (a (.virt 38210)) (a (.virt 38211)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_17
      simp only [e_12_5] at hc
      linear_combination c734.trans k1 - hc
    · have hc := e_12_6
      simp only [e_12_4, e_11_18, e_12_5] at hc
      linear_combination c735.trans k1 - hc
  have f103 : a (.wire 7 39) = bor (a (.wire 2 51)) (a (.virt 38210)) := by
    have hr := e_7_9
    simp only [e_6_9] at hr
    simp only [bor]
    linear_combination hr
  have f104 : a (.wire 7 39) = a (.virt 38152) := by
    exact c742
  have f105 : IsEqual (a (.virt 28561)) (a (.wire 9 31)) (a (.virt 38212)) (a (.virt 38213)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_11_19
      simp only [e_12_8] at hc
      linear_combination c758.trans k1 - hc
    · have hc := e_12_9
      simp only [e_12_7, e_13_0, e_12_8] at hc
      linear_combination c759.trans k1 - hc
  have f106 : a (.wire 7 43) = bor (a (.wire 2 51)) (a (.virt 38212)) := by
    have hr := e_7_10
    simp only [e_6_10] at hr
    simp only [bor]
    linear_combination hr
  have f107 : a (.wire 7 43) = a (.virt 38152) := by
    exact c766
  have f108 : IsEqual (a (.virt 28562)) (a (.wire 8 63)) (a (.virt 38214)) (a (.virt 38215)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_1
      simp only [e_12_11] at hc
      linear_combination c782.trans k1 - hc
    · have hc := e_12_12
      simp only [e_12_10, e_13_2, e_12_11] at hc
      linear_combination c783.trans k1 - hc
  have f109 : IsEqual (a (.virt 28563)) (a (.wire 8 71)) (a (.virt 38216)) (a (.virt 38217)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_3
      simp only [e_12_14] at hc
      linear_combination c799.trans k1 - hc
    · have hc := e_12_15
      simp only [e_12_13, e_13_4, e_12_14] at hc
      linear_combination c800.trans k1 - hc
  have f110 : IsEqual (a (.virt 28564)) (a (.wire 8 79)) (a (.virt 38218)) (a (.virt 38219)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_5
      simp only [e_12_17] at hc
      linear_combination c816.trans k1 - hc
    · have hc := e_12_18
      simp only [e_12_16, e_13_6, e_12_17] at hc
      linear_combination c817.trans k1 - hc
  have f111 : IsEqual (a (.virt 28565)) (a (.wire 9 7)) (a (.virt 38220)) (a (.virt 38221)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_7
      simp only [e_14_0] at hc
      linear_combination c833.trans k1 - hc
    · have hc := e_14_1
      simp only [e_12_19, e_13_8, e_14_0] at hc
      linear_combination c834.trans k1 - hc
  have f112 : a (.wire 13 39) = band (a (.virt 38214)) (a (.virt 38216)) := by
    have hr := e_13_9
    simp only [band]
    linear_combination hr
  have f113 : a (.wire 13 43) = band (a (.virt 38218)) (a (.virt 38220)) := by
    have hr := e_13_10
    simp only [band]
    linear_combination hr
  have f114 : a (.wire 13 47) = band (a (.wire 13 39)) (a (.wire 13 43)) := by
    have hr := e_13_11
    simp only [band]
    linear_combination hr
  have f115 : a (.wire 7 47) = bor (a (.wire 2 51)) (a (.wire 13 47)) := by
    have hr := e_7_11
    simp only [e_6_11] at hr
    simp only [bor]
    linear_combination hr
  have f116 : a (.wire 7 47) = a (.virt 38152) := by
    exact c850
  have f117 : IsEqual (a (.virt 38097)) (a (.wire 9 23)) (a (.virt 38222)) (a (.virt 38223)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_12
      simp only [e_14_3] at hc
      linear_combination c866.trans k1 - hc
    · have hc := e_14_4
      simp only [e_14_2, e_13_13, e_14_3] at hc
      linear_combination c867.trans k1 - hc
  have f118 : a (.wire 7 51) = bor (a (.wire 4 15)) (a (.virt 38222)) := by
    have hr := e_7_12
    simp only [e_6_12] at hr
    simp only [bor]
    linear_combination hr
  have f119 : a (.wire 7 51) = a (.virt 38152) := by
    exact c874
  have f120 : IsEqual (a (.virt 38098)) (a (.wire 9 31)) (a (.virt 38224)) (a (.virt 38225)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_14
      simp only [e_14_6] at hc
      linear_combination c890.trans k1 - hc
    · have hc := e_14_7
      simp only [e_14_5, e_13_15, e_14_6] at hc
      linear_combination c891.trans k1 - hc
  have f121 : a (.wire 7 55) = bor (a (.wire 4 15)) (a (.virt 38224)) := by
    have hr := e_7_13
    simp only [e_6_13] at hr
    simp only [bor]
    linear_combination hr
  have f122 : a (.wire 7 55) = a (.virt 38152) := by
    exact c898
  have f123 : IsEqual (a (.virt 38099)) (a (.wire 8 63)) (a (.virt 38226)) (a (.virt 38227)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_16
      simp only [e_14_9] at hc
      linear_combination c914.trans k1 - hc
    · have hc := e_14_10
      simp only [e_14_8, e_13_17, e_14_9] at hc
      linear_combination c915.trans k1 - hc
  have f124 : IsEqual (a (.virt 38100)) (a (.wire 8 71)) (a (.virt 38228)) (a (.virt 38229)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_13_18
      simp only [e_14_12] at hc
      linear_combination c931.trans k1 - hc
    · have hc := e_14_13
      simp only [e_14_11, e_13_19, e_14_12] at hc
      linear_combination c932.trans k1 - hc
  have f125 : IsEqual (a (.virt 38101)) (a (.wire 8 79)) (a (.virt 38230)) (a (.virt 38231)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_15_0
      simp only [e_14_15] at hc
      linear_combination c948.trans k1 - hc
    · have hc := e_14_16
      simp only [e_14_14, e_15_1, e_14_15] at hc
      linear_combination c949.trans k1 - hc
  have f126 : IsEqual (a (.virt 38102)) (a (.wire 9 7)) (a (.virt 38232)) (a (.virt 38233)) := by
    refine ⟨?_, ?_⟩
    · have hc := e_15_2
      simp only [e_14_18] at hc
      linear_combination c965.trans k1 - hc
    · have hc := e_14_19
      simp only [e_14_17, e_15_3, e_14_18] at hc
      linear_combination c966.trans k1 - hc
  have f127 : a (.wire 15 19) = band (a (.virt 38226)) (a (.virt 38228)) := by
    have hr := e_15_4
    simp only [band]
    linear_combination hr
  have f128 : a (.wire 15 23) = band (a (.virt 38230)) (a (.virt 38232)) := by
    have hr := e_15_5
    simp only [band]
    linear_combination hr
  have f129 : a (.wire 15 27) = band (a (.wire 15 19)) (a (.wire 15 23)) := by
    have hr := e_15_6
    simp only [band]
    linear_combination hr
  have f130 : a (.wire 7 59) = bor (a (.wire 4 15)) (a (.wire 15 27)) := by
    have hr := e_7_14
    simp only [e_6_14] at hr
    simp only [bor]
    linear_combination hr
  have f131 : a (.wire 7 59) = a (.virt 38152) := by
    exact c982
  have f132 : a (.wire 16 7) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9493)) := by
    have hr := e_16_1
    simp only [e_16_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f133 : a (.wire 16 15) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9494)) := by
    have hr := e_16_3
    simp only [e_16_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f134 : a (.wire 16 23) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9495)) := by
    have hr := e_16_5
    simp only [e_16_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f135 : a (.wire 16 31) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9496)) := by
    have hr := e_16_7
    simp only [e_16_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f136 : a (.wire 16 39) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9497)) := by
    have hr := e_16_9
    simp only [e_16_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f137 : a (.wire 16 47) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9498)) := by
    have hr := e_16_11
    simp only [e_16_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f138 : a (.wire 16 55) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9499)) := by
    have hr := e_16_13
    simp only [e_16_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f139 : a (.wire 16 63) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9500)) := by
    have hr := e_16_15
    simp only [e_16_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f140 : a (.wire 16 71) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9501)) := by
    have hr := e_16_17
    simp only [e_16_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f141 : a (.wire 16 79) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9502)) := by
    have hr := e_16_19
    simp only [e_16_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f142 : a (.wire 17 7) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9503)) := by
    have hr := e_17_1
    simp only [e_17_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f143 : a (.wire 17 15) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9504)) := by
    have hr := e_17_3
    simp only [e_17_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f144 : a (.wire 17 23) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9505)) := by
    have hr := e_17_5
    simp only [e_17_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f145 : a (.wire 17 31) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9506)) := by
    have hr := e_17_7
    simp only [e_17_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f146 : a (.wire 17 39) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9507)) := by
    have hr := e_17_9
    simp only [e_17_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f147 : a (.wire 17 47) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9508)) := by
    have hr := e_17_11
    simp only [e_17_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f148 : a (.wire 17 55) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9509)) := by
    have hr := e_17_13
    simp only [e_17_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f149 : a (.wire 17 63) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9510)) := by
    have hr := e_17_15
    simp only [e_17_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f150 : a (.wire 17 71) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9511)) := by
    have hr := e_17_17
    simp only [e_17_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f151 : a (.wire 17 79) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9512)) := by
    have hr := e_17_19
    simp only [e_17_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f152 : a (.wire 18 7) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19030)) := by
    have hr := e_18_1
    simp only [e_18_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f153 : a (.wire 18 15) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19031)) := by
    have hr := e_18_3
    simp only [e_18_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f154 : a (.wire 18 23) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19032)) := by
    have hr := e_18_5
    simp only [e_18_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f155 : a (.wire 18 31) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19033)) := by
    have hr := e_18_7
    simp only [e_18_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f156 : a (.wire 18 39) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19034)) := by
    have hr := e_18_9
    simp only [e_18_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f157 : a (.wire 18 47) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19035)) := by
    have hr := e_18_11
    simp only [e_18_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f158 : a (.wire 18 55) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19036)) := by
    have hr := e_18_13
    simp only [e_18_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f159 : a (.wire 18 63) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19037)) := by
    have hr := e_18_15
    simp only [e_18_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f160 : a (.wire 18 71) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19038)) := by
    have hr := e_18_17
    simp only [e_18_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f161 : a (.wire 18 79) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19039)) := by
    have hr := e_18_19
    simp only [e_18_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f162 : a (.wire 19 7) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19040)) := by
    have hr := e_19_1
    simp only [e_19_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f163 : a (.wire 19 15) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19041)) := by
    have hr := e_19_3
    simp only [e_19_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f164 : a (.wire 19 23) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19042)) := by
    have hr := e_19_5
    simp only [e_19_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f165 : a (.wire 19 31) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19043)) := by
    have hr := e_19_7
    simp only [e_19_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f166 : a (.wire 19 39) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19044)) := by
    have hr := e_19_9
    simp only [e_19_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f167 : a (.wire 19 47) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19045)) := by
    have hr := e_19_11
    simp only [e_19_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f168 : a (.wire 19 55) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19046)) := by
    have hr := e_19_13
    simp only [e_19_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f169 : a (.wire 19 63) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19047)) := by
    have hr := e_19_15
    simp only [e_19_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f170 : a (.wire 19 71) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19048)) := by
    have hr := e_19_17
    simp only [e_19_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f171 : a (.wire 19 79) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19049)) := by
    have hr := e_19_19
    simp only [e_19_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f172 : a (.wire 20 7) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28567)) := by
    have hr := e_20_1
    simp only [e_20_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f173 : a (.wire 20 15) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28568)) := by
    have hr := e_20_3
    simp only [e_20_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f174 : a (.wire 20 23) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28569)) := by
    have hr := e_20_5
    simp only [e_20_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f175 : a (.wire 20 31) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28570)) := by
    have hr := e_20_7
    simp only [e_20_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f176 : a (.wire 20 39) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28571)) := by
    have hr := e_20_9
    simp only [e_20_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f177 : a (.wire 20 47) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28572)) := by
    have hr := e_20_11
    simp only [e_20_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f178 : a (.wire 20 55) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28573)) := by
    have hr := e_20_13
    simp only [e_20_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f179 : a (.wire 20 63) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28574)) := by
    have hr := e_20_15
    simp only [e_20_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f180 : a (.wire 20 71) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28575)) := by
    have hr := e_20_17
    simp only [e_20_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f181 : a (.wire 20 79) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28576)) := by
    have hr := e_20_19
    simp only [e_20_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f182 : a (.wire 21 7) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28577)) := by
    have hr := e_21_1
    simp only [e_21_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f183 : a (.wire 21 15) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28578)) := by
    have hr := e_21_3
    simp only [e_21_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f184 : a (.wire 21 23) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28579)) := by
    have hr := e_21_5
    simp only [e_21_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f185 : a (.wire 21 31) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28580)) := by
    have hr := e_21_7
    simp only [e_21_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f186 : a (.wire 21 39) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28581)) := by
    have hr := e_21_9
    simp only [e_21_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f187 : a (.wire 21 47) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28582)) := by
    have hr := e_21_11
    simp only [e_21_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f188 : a (.wire 21 55) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28583)) := by
    have hr := e_21_13
    simp only [e_21_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f189 : a (.wire 21 63) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28584)) := by
    have hr := e_21_15
    simp only [e_21_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f190 : a (.wire 21 71) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28585)) := by
    have hr := e_21_17
    simp only [e_21_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f191 : a (.wire 21 79) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28586)) := by
    have hr := e_21_19
    simp only [e_21_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f192 : a (.wire 22 7) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38104)) := by
    have hr := e_22_1
    simp only [e_22_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f193 : a (.wire 22 15) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38105)) := by
    have hr := e_22_3
    simp only [e_22_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f194 : a (.wire 22 23) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38106)) := by
    have hr := e_22_5
    simp only [e_22_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f195 : a (.wire 22 31) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38107)) := by
    have hr := e_22_7
    simp only [e_22_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f196 : a (.wire 22 39) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38108)) := by
    have hr := e_22_9
    simp only [e_22_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f197 : a (.wire 22 47) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38109)) := by
    have hr := e_22_11
    simp only [e_22_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f198 : a (.wire 22 55) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38110)) := by
    have hr := e_22_13
    simp only [e_22_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f199 : a (.wire 22 63) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38111)) := by
    have hr := e_22_15
    simp only [e_22_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f200 : a (.wire 22 71) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38112)) := by
    have hr := e_22_17
    simp only [e_22_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f201 : a (.wire 22 79) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38113)) := by
    have hr := e_22_19
    simp only [e_22_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f202 : a (.wire 23 7) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38114)) := by
    have hr := e_23_1
    simp only [e_23_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f203 : a (.wire 23 15) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38115)) := by
    have hr := e_23_3
    simp only [e_23_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f204 : a (.wire 23 23) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38116)) := by
    have hr := e_23_5
    simp only [e_23_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f205 : a (.wire 23 31) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38117)) := by
    have hr := e_23_7
    simp only [e_23_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f206 : a (.wire 23 39) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38118)) := by
    have hr := e_23_9
    simp only [e_23_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f207 : a (.wire 23 47) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38119)) := by
    have hr := e_23_11
    simp only [e_23_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f208 : a (.wire 23 55) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38120)) := by
    have hr := e_23_13
    simp only [e_23_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f209 : a (.wire 23 63) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38121)) := by
    have hr := e_23_15
    simp only [e_23_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f210 : a (.wire 23 71) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38122)) := by
    have hr := e_23_17
    simp only [e_23_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f211 : a (.wire 23 79) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38123)) := by
    have hr := e_23_19
    simp only [e_23_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f212 : a (.wire 24 7) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9513)) := by
    have hr := e_24_1
    simp only [e_24_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f213 : a (.wire 24 15) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9514)) := by
    have hr := e_24_3
    simp only [e_24_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f214 : a (.wire 24 23) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9515)) := by
    have hr := e_24_5
    simp only [e_24_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f215 : a (.wire 24 31) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9516)) := by
    have hr := e_24_7
    simp only [e_24_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f216 : a (.wire 24 39) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9517)) := by
    have hr := e_24_9
    simp only [e_24_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f217 : a (.wire 24 47) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9518)) := by
    have hr := e_24_11
    simp only [e_24_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f218 : a (.wire 24 55) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9519)) := by
    have hr := e_24_13
    simp only [e_24_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f219 : a (.wire 24 63) = bselect (a (.wire 1 43)) (a (.virt 38153)) (a (.virt 9520)) := by
    have hr := e_24_15
    simp only [e_24_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f220 : a (.wire 24 71) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19050)) := by
    have hr := e_24_17
    simp only [e_24_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f221 : a (.wire 24 79) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19051)) := by
    have hr := e_24_19
    simp only [e_24_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f222 : a (.wire 25 7) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19052)) := by
    have hr := e_25_1
    simp only [e_25_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f223 : a (.wire 25 15) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19053)) := by
    have hr := e_25_3
    simp only [e_25_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f224 : a (.wire 25 23) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19054)) := by
    have hr := e_25_5
    simp only [e_25_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f225 : a (.wire 25 31) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19055)) := by
    have hr := e_25_7
    simp only [e_25_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f226 : a (.wire 25 39) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19056)) := by
    have hr := e_25_9
    simp only [e_25_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f227 : a (.wire 25 47) = bselect (a (.wire 2 7)) (a (.virt 38153)) (a (.virt 19057)) := by
    have hr := e_25_11
    simp only [e_25_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f228 : a (.wire 25 55) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28587)) := by
    have hr := e_25_13
    simp only [e_25_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f229 : a (.wire 25 63) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28588)) := by
    have hr := e_25_15
    simp only [e_25_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f230 : a (.wire 25 71) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28589)) := by
    have hr := e_25_17
    simp only [e_25_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f231 : a (.wire 25 79) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28590)) := by
    have hr := e_25_19
    simp only [e_25_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f232 : a (.wire 26 7) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28591)) := by
    have hr := e_26_1
    simp only [e_26_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f233 : a (.wire 26 15) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28592)) := by
    have hr := e_26_3
    simp only [e_26_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f234 : a (.wire 26 23) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28593)) := by
    have hr := e_26_5
    simp only [e_26_4] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f235 : a (.wire 26 31) = bselect (a (.wire 2 51)) (a (.virt 38153)) (a (.virt 28594)) := by
    have hr := e_26_7
    simp only [e_26_6] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f236 : a (.wire 26 39) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38124)) := by
    have hr := e_26_9
    simp only [e_26_8] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f237 : a (.wire 26 47) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38125)) := by
    have hr := e_26_11
    simp only [e_26_10] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f238 : a (.wire 26 55) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38126)) := by
    have hr := e_26_13
    simp only [e_26_12] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f239 : a (.wire 26 63) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38127)) := by
    have hr := e_26_15
    simp only [e_26_14] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f240 : a (.wire 26 71) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38128)) := by
    have hr := e_26_17
    simp only [e_26_16] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f241 : a (.wire 26 79) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38129)) := by
    have hr := e_26_19
    simp only [e_26_18] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f242 : a (.wire 27 7) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38130)) := by
    have hr := e_27_1
    simp only [e_27_0] at hr
    simp only [bselect, k1]
    linear_combination hr
  have f243 : a (.wire 27 15) = bselect (a (.wire 4 15)) (a (.virt 38153)) (a (.virt 38131)) := by
    have hr := e_27_3
    simp only [e_27_2] at hr
    simp only [bselect, k1]
    linear_combination hr
  exact ⟨⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31⟩, ⟨f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60, f61, f62, f63⟩, ⟨f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75, f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90, f91, f92, f93, f94, f95⟩, ⟨f96, f97, f98, f99, f100, f101, f102, f103, f104, f105, f106, f107, f108, f109, f110, f111, f112, f113, f114, f115, f116, f117, f118, f119, f120, f121, f122, f123, f124, f125, f126, f127⟩, ⟨f128, f129, f130, f131, f132, f133, f134, f135, f136, f137, f138, f139, f140, f141, f142, f143, f144, f145, f146, f147, f148, f149, f150, f151, f152, f153, f154, f155, f156, f157, f158, f159⟩, ⟨f160, f161, f162, f163, f164, f165, f166, f167, f168, f169, f170, f171, f172, f173, f174, f175, f176, f177, f178, f179, f180, f181, f182, f183, f184, f185, f186, f187, f188, f189, f190, f191⟩, ⟨f192, f193, f194, f195, f196, f197, f198, f199, f200, f201, f202, f203, f204, f205, f206, f207, f208, f209, f210, f211, f212, f213, f214, f215, f216, f217, f218, f219, f220, f221, f222, f223⟩, ⟨f224, f225, f226, f227, f228, f229, f230, f231, f232, f233, f234, f235, f236, f237, f238, f239, f240, f241, f242, f243⟩⟩

end Plonky2Spec.Generated
