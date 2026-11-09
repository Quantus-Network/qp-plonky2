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

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

/-- `publicBatchWrapper4.copies`, items `0..32`. -/
def publicBatchWrapper4.copies0 : List (Target × Target) := [
    (.virt 5918, .virt 23782),
    (.virt 11862, .virt 23782),
    (.virt 17806, .virt 23782),
    (.virt 23750, .virt 23782),
    (.virt 23780, .wire 0 0),
    (.virt 23780, .wire 0 1),
    (.virt 23783, .wire 0 2),
    (.virt 23783, .wire 1 0),
    (.virt 5921, .wire 1 1),
    (.virt 23783, .wire 1 2),
    (.virt 5921, .wire 1 4),
    (.virt 23784, .wire 1 5),
    (.virt 5921, .wire 1 6),
    (.wire 1 7, .wire 0 4),
    (.virt 23780, .wire 0 5),
    (.wire 0 3, .wire 0 6),
    (.wire 1 3, .virt 23781),
    (.wire 0 7, .virt 23781),
    (.virt 23780, .wire 0 8),
    (.virt 23780, .wire 0 9),
    (.virt 23785, .wire 0 10),
    (.virt 23785, .wire 1 8),
    (.virt 5922, .wire 1 9),
    (.virt 23785, .wire 1 10),
    (.virt 5922, .wire 1 12),
    (.virt 23786, .wire 1 13),
    (.virt 5922, .wire 1 14),
    (.wire 1 15, .wire 0 12),
    (.virt 23780, .wire 0 13),
    (.wire 0 11, .wire 0 14),
    (.wire 1 11, .virt 23781),
    (.wire 0 15, .virt 23781)
  ]

/-- `publicBatchWrapper4.copies`, items `32..64`. -/
def publicBatchWrapper4.copies1 : List (Target × Target) := [
    (.virt 23780, .wire 0 16),
    (.virt 23780, .wire 0 17),
    (.virt 23787, .wire 0 18),
    (.virt 23787, .wire 1 16),
    (.virt 5923, .wire 1 17),
    (.virt 23787, .wire 1 18),
    (.virt 5923, .wire 1 20),
    (.virt 23788, .wire 1 21),
    (.virt 5923, .wire 1 22),
    (.wire 1 23, .wire 0 20),
    (.virt 23780, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.wire 1 19, .virt 23781),
    (.wire 0 23, .virt 23781),
    (.virt 23780, .wire 0 24),
    (.virt 23780, .wire 0 25),
    (.virt 23789, .wire 0 26),
    (.virt 23789, .wire 1 24),
    (.virt 5924, .wire 1 25),
    (.virt 23789, .wire 1 26),
    (.virt 5924, .wire 1 28),
    (.virt 23790, .wire 1 29),
    (.virt 5924, .wire 1 30),
    (.wire 1 31, .wire 0 28),
    (.virt 23780, .wire 0 29),
    (.wire 0 27, .wire 0 30),
    (.wire 1 27, .virt 23781),
    (.wire 0 31, .virt 23781),
    (.virt 23783, .wire 1 32),
    (.virt 23785, .wire 1 33),
    (.virt 23783, .wire 1 34),
    (.virt 23787, .wire 1 36)
  ]

/-- `publicBatchWrapper4.copies`, items `64..96`. -/
def publicBatchWrapper4.copies2 : List (Target × Target) := [
    (.virt 23789, .wire 1 37),
    (.virt 23787, .wire 1 38),
    (.wire 1 35, .wire 1 40),
    (.wire 1 39, .wire 1 41),
    (.wire 1 35, .wire 1 42),
    (.virt 23780, .wire 0 32),
    (.virt 23780, .wire 0 33),
    (.virt 23791, .wire 0 34),
    (.virt 23791, .wire 1 44),
    (.virt 11865, .wire 1 45),
    (.virt 23791, .wire 1 46),
    (.virt 11865, .wire 1 48),
    (.virt 23792, .wire 1 49),
    (.virt 11865, .wire 1 50),
    (.wire 1 51, .wire 0 36),
    (.virt 23780, .wire 0 37),
    (.wire 0 35, .wire 0 38),
    (.wire 1 47, .virt 23781),
    (.wire 0 39, .virt 23781),
    (.virt 23780, .wire 0 40),
    (.virt 23780, .wire 0 41),
    (.virt 23793, .wire 0 42),
    (.virt 23793, .wire 1 52),
    (.virt 11866, .wire 1 53),
    (.virt 23793, .wire 1 54),
    (.virt 11866, .wire 1 56),
    (.virt 23794, .wire 1 57),
    (.virt 11866, .wire 1 58),
    (.wire 1 59, .wire 0 44),
    (.virt 23780, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.wire 1 55, .virt 23781)
  ]

/-- `publicBatchWrapper4.copies`, items `96..128`. -/
def publicBatchWrapper4.copies3 : List (Target × Target) := [
    (.wire 0 47, .virt 23781),
    (.virt 23780, .wire 0 48),
    (.virt 23780, .wire 0 49),
    (.virt 23795, .wire 0 50),
    (.virt 23795, .wire 1 60),
    (.virt 11867, .wire 1 61),
    (.virt 23795, .wire 1 62),
    (.virt 11867, .wire 1 64),
    (.virt 23796, .wire 1 65),
    (.virt 11867, .wire 1 66),
    (.wire 1 67, .wire 0 52),
    (.virt 23780, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.wire 1 63, .virt 23781),
    (.wire 0 55, .virt 23781),
    (.virt 23780, .wire 0 56),
    (.virt 23780, .wire 0 57),
    (.virt 23797, .wire 0 58),
    (.virt 23797, .wire 1 68),
    (.virt 11868, .wire 1 69),
    (.virt 23797, .wire 1 70),
    (.virt 11868, .wire 1 72),
    (.virt 23798, .wire 1 73),
    (.virt 11868, .wire 1 74),
    (.wire 1 75, .wire 0 60),
    (.virt 23780, .wire 0 61),
    (.wire 0 59, .wire 0 62),
    (.wire 1 71, .virt 23781),
    (.wire 0 63, .virt 23781),
    (.virt 23791, .wire 1 76),
    (.virt 23793, .wire 1 77),
    (.virt 23791, .wire 1 78)
  ]

/-- `publicBatchWrapper4.copies`, items `128..160`. -/
def publicBatchWrapper4.copies4 : List (Target × Target) := [
    (.virt 23795, .wire 2 0),
    (.virt 23797, .wire 2 1),
    (.virt 23795, .wire 2 2),
    (.wire 1 79, .wire 2 4),
    (.wire 2 3, .wire 2 5),
    (.wire 1 79, .wire 2 6),
    (.virt 23780, .wire 0 64),
    (.virt 23780, .wire 0 65),
    (.virt 23799, .wire 0 66),
    (.virt 23799, .wire 2 8),
    (.virt 17809, .wire 2 9),
    (.virt 23799, .wire 2 10),
    (.virt 17809, .wire 2 12),
    (.virt 23800, .wire 2 13),
    (.virt 17809, .wire 2 14),
    (.wire 2 15, .wire 0 68),
    (.virt 23780, .wire 0 69),
    (.wire 0 67, .wire 0 70),
    (.wire 2 11, .virt 23781),
    (.wire 0 71, .virt 23781),
    (.virt 23780, .wire 0 72),
    (.virt 23780, .wire 0 73),
    (.virt 23801, .wire 0 74),
    (.virt 23801, .wire 2 16),
    (.virt 17810, .wire 2 17),
    (.virt 23801, .wire 2 18),
    (.virt 17810, .wire 2 20),
    (.virt 23802, .wire 2 21),
    (.virt 17810, .wire 2 22),
    (.wire 2 23, .wire 0 76),
    (.virt 23780, .wire 0 77),
    (.wire 0 75, .wire 0 78)
  ]

/-- `publicBatchWrapper4.copies`, items `160..192`. -/
def publicBatchWrapper4.copies5 : List (Target × Target) := [
    (.wire 2 19, .virt 23781),
    (.wire 0 79, .virt 23781),
    (.virt 23780, .wire 3 0),
    (.virt 23780, .wire 3 1),
    (.virt 23803, .wire 3 2),
    (.virt 23803, .wire 2 24),
    (.virt 17811, .wire 2 25),
    (.virt 23803, .wire 2 26),
    (.virt 17811, .wire 2 28),
    (.virt 23804, .wire 2 29),
    (.virt 17811, .wire 2 30),
    (.wire 2 31, .wire 3 4),
    (.virt 23780, .wire 3 5),
    (.wire 3 3, .wire 3 6),
    (.wire 2 27, .virt 23781),
    (.wire 3 7, .virt 23781),
    (.virt 23780, .wire 3 8),
    (.virt 23780, .wire 3 9),
    (.virt 23805, .wire 3 10),
    (.virt 23805, .wire 2 32),
    (.virt 17812, .wire 2 33),
    (.virt 23805, .wire 2 34),
    (.virt 17812, .wire 2 36),
    (.virt 23806, .wire 2 37),
    (.virt 17812, .wire 2 38),
    (.wire 2 39, .wire 3 12),
    (.virt 23780, .wire 3 13),
    (.wire 3 11, .wire 3 14),
    (.wire 2 35, .virt 23781),
    (.wire 3 15, .virt 23781),
    (.virt 23799, .wire 2 40),
    (.virt 23801, .wire 2 41)
  ]

/-- `publicBatchWrapper4.copies`, items `192..224`. -/
def publicBatchWrapper4.copies6 : List (Target × Target) := [
    (.virt 23799, .wire 2 42),
    (.virt 23803, .wire 2 44),
    (.virt 23805, .wire 2 45),
    (.virt 23803, .wire 2 46),
    (.wire 2 43, .wire 2 48),
    (.wire 2 47, .wire 2 49),
    (.wire 2 43, .wire 2 50),
    (.virt 23780, .wire 3 16),
    (.virt 23780, .wire 3 17),
    (.virt 23807, .wire 3 18),
    (.virt 23807, .wire 2 52),
    (.virt 23753, .wire 2 53),
    (.virt 23807, .wire 2 54),
    (.virt 23753, .wire 2 56),
    (.virt 23808, .wire 2 57),
    (.virt 23753, .wire 2 58),
    (.wire 2 59, .wire 3 20),
    (.virt 23780, .wire 3 21),
    (.wire 3 19, .wire 3 22),
    (.wire 2 55, .virt 23781),
    (.wire 3 23, .virt 23781),
    (.virt 23780, .wire 3 24),
    (.virt 23780, .wire 3 25),
    (.virt 23809, .wire 3 26),
    (.virt 23809, .wire 2 60),
    (.virt 23754, .wire 2 61),
    (.virt 23809, .wire 2 62),
    (.virt 23754, .wire 2 64),
    (.virt 23810, .wire 2 65),
    (.virt 23754, .wire 2 66),
    (.wire 2 67, .wire 3 28),
    (.virt 23780, .wire 3 29)
  ]

/-- `publicBatchWrapper4.copies`, items `224..256`. -/
def publicBatchWrapper4.copies7 : List (Target × Target) := [
    (.wire 3 27, .wire 3 30),
    (.wire 2 63, .virt 23781),
    (.wire 3 31, .virt 23781),
    (.virt 23780, .wire 3 32),
    (.virt 23780, .wire 3 33),
    (.virt 23811, .wire 3 34),
    (.virt 23811, .wire 2 68),
    (.virt 23755, .wire 2 69),
    (.virt 23811, .wire 2 70),
    (.virt 23755, .wire 2 72),
    (.virt 23812, .wire 2 73),
    (.virt 23755, .wire 2 74),
    (.wire 2 75, .wire 3 36),
    (.virt 23780, .wire 3 37),
    (.wire 3 35, .wire 3 38),
    (.wire 2 71, .virt 23781),
    (.wire 3 39, .virt 23781),
    (.virt 23780, .wire 3 40),
    (.virt 23780, .wire 3 41),
    (.virt 23813, .wire 3 42),
    (.virt 23813, .wire 2 76),
    (.virt 23756, .wire 2 77),
    (.virt 23813, .wire 2 78),
    (.virt 23756, .wire 4 0),
    (.virt 23814, .wire 4 1),
    (.virt 23756, .wire 4 2),
    (.wire 4 3, .wire 3 44),
    (.virt 23780, .wire 3 45),
    (.wire 3 43, .wire 3 46),
    (.wire 2 79, .virt 23781),
    (.wire 3 47, .virt 23781),
    (.virt 23807, .wire 4 4)
  ]

/-- `publicBatchWrapper4.copies`, items `256..288`. -/
def publicBatchWrapper4.copies8 : List (Target × Target) := [
    (.virt 23809, .wire 4 5),
    (.virt 23807, .wire 4 6),
    (.virt 23811, .wire 4 8),
    (.virt 23813, .wire 4 9),
    (.virt 23811, .wire 4 10),
    (.wire 4 7, .wire 4 12),
    (.wire 4 11, .wire 4 13),
    (.wire 4 7, .wire 4 14),
    (.virt 23780, .wire 3 48),
    (.virt 23780, .wire 3 49),
    (.wire 1 43, .wire 3 50),
    (.wire 3 51, .wire 3 52),
    (.virt 5921, .wire 3 53),
    (.virt 23781, .wire 3 54),
    (.wire 3 51, .wire 3 56),
    (.virt 5922, .wire 3 57),
    (.virt 23781, .wire 3 58),
    (.wire 3 51, .wire 3 60),
    (.virt 5923, .wire 3 61),
    (.virt 23781, .wire 3 62),
    (.wire 3 51, .wire 3 64),
    (.virt 5924, .wire 3 65),
    (.virt 23781, .wire 3 66),
    (.wire 3 51, .wire 3 68),
    (.virt 5925, .wire 3 69),
    (.virt 23781, .wire 3 70),
    (.wire 3 51, .wire 3 72),
    (.virt 5919, .wire 3 73),
    (.virt 23781, .wire 3 74),
    (.wire 3 51, .wire 3 76),
    (.virt 5920, .wire 3 77),
    (.virt 23781, .wire 3 78)
  ]

/-- `publicBatchWrapper4.copies`, items `288..320`. -/
def publicBatchWrapper4.copies9 : List (Target × Target) := [
    (.virt 23780, .wire 5 0),
    (.virt 23780, .wire 5 1),
    (.wire 2 7, .wire 5 2),
    (.virt 23780, .wire 5 4),
    (.virt 23780, .wire 5 5),
    (.wire 3 51, .wire 5 6),
    (.wire 5 3, .wire 4 16),
    (.wire 5 7, .wire 4 17),
    (.wire 5 3, .wire 4 18),
    (.wire 4 19, .wire 5 8),
    (.wire 3 55, .wire 5 9),
    (.wire 3 55, .wire 5 10),
    (.wire 4 19, .wire 5 12),
    (.virt 11865, .wire 5 13),
    (.wire 5 11, .wire 5 14),
    (.wire 4 19, .wire 5 16),
    (.wire 3 59, .wire 5 17),
    (.wire 3 59, .wire 5 18),
    (.wire 4 19, .wire 5 20),
    (.virt 11866, .wire 5 21),
    (.wire 5 19, .wire 5 22),
    (.wire 4 19, .wire 5 24),
    (.wire 3 63, .wire 5 25),
    (.wire 3 63, .wire 5 26),
    (.wire 4 19, .wire 5 28),
    (.virt 11867, .wire 5 29),
    (.wire 5 27, .wire 5 30),
    (.wire 4 19, .wire 5 32),
    (.wire 3 67, .wire 5 33),
    (.wire 3 67, .wire 5 34),
    (.wire 4 19, .wire 5 36),
    (.virt 11868, .wire 5 37)
  ]

/-- `publicBatchWrapper4.copies`, items `320..352`. -/
def publicBatchWrapper4.copies10 : List (Target × Target) := [
    (.wire 5 35, .wire 5 38),
    (.wire 4 19, .wire 5 40),
    (.wire 3 71, .wire 5 41),
    (.wire 3 71, .wire 5 42),
    (.wire 4 19, .wire 5 44),
    (.virt 11869, .wire 5 45),
    (.wire 5 43, .wire 5 46),
    (.wire 4 19, .wire 5 48),
    (.wire 3 75, .wire 5 49),
    (.wire 3 75, .wire 5 50),
    (.wire 4 19, .wire 5 52),
    (.virt 11863, .wire 5 53),
    (.wire 5 51, .wire 5 54),
    (.wire 4 19, .wire 5 56),
    (.wire 3 79, .wire 5 57),
    (.wire 3 79, .wire 5 58),
    (.wire 4 19, .wire 5 60),
    (.virt 11864, .wire 5 61),
    (.wire 5 59, .wire 5 62),
    (.wire 3 51, .wire 6 0),
    (.wire 5 3, .wire 6 1),
    (.wire 3 51, .wire 6 2),
    (.wire 6 3, .wire 7 0),
    (.virt 23780, .wire 7 1),
    (.wire 5 3, .wire 7 2),
    (.virt 23780, .wire 5 64),
    (.virt 23780, .wire 5 65),
    (.wire 2 51, .wire 5 66),
    (.virt 23780, .wire 5 68),
    (.virt 23780, .wire 5 69),
    (.wire 7 3, .wire 5 70),
    (.wire 5 67, .wire 4 20)
  ]

/-- `publicBatchWrapper4.copies`, items `352..384`. -/
def publicBatchWrapper4.copies11 : List (Target × Target) := [
    (.wire 5 71, .wire 4 21),
    (.wire 5 67, .wire 4 22),
    (.wire 4 23, .wire 5 72),
    (.wire 5 15, .wire 5 73),
    (.wire 5 15, .wire 5 74),
    (.wire 4 23, .wire 5 76),
    (.virt 17809, .wire 5 77),
    (.wire 5 75, .wire 5 78),
    (.wire 4 23, .wire 8 0),
    (.wire 5 23, .wire 8 1),
    (.wire 5 23, .wire 8 2),
    (.wire 4 23, .wire 8 4),
    (.virt 17810, .wire 8 5),
    (.wire 8 3, .wire 8 6),
    (.wire 4 23, .wire 8 8),
    (.wire 5 31, .wire 8 9),
    (.wire 5 31, .wire 8 10),
    (.wire 4 23, .wire 8 12),
    (.virt 17811, .wire 8 13),
    (.wire 8 11, .wire 8 14),
    (.wire 4 23, .wire 8 16),
    (.wire 5 39, .wire 8 17),
    (.wire 5 39, .wire 8 18),
    (.wire 4 23, .wire 8 20),
    (.virt 17812, .wire 8 21),
    (.wire 8 19, .wire 8 22),
    (.wire 4 23, .wire 8 24),
    (.wire 5 47, .wire 8 25),
    (.wire 5 47, .wire 8 26),
    (.wire 4 23, .wire 8 28),
    (.virt 17813, .wire 8 29),
    (.wire 8 27, .wire 8 30)
  ]

/-- `publicBatchWrapper4.copies`, items `384..416`. -/
def publicBatchWrapper4.copies12 : List (Target × Target) := [
    (.wire 4 23, .wire 8 32),
    (.wire 5 55, .wire 8 33),
    (.wire 5 55, .wire 8 34),
    (.wire 4 23, .wire 8 36),
    (.virt 17807, .wire 8 37),
    (.wire 8 35, .wire 8 38),
    (.wire 4 23, .wire 8 40),
    (.wire 5 63, .wire 8 41),
    (.wire 5 63, .wire 8 42),
    (.wire 4 23, .wire 8 44),
    (.virt 17808, .wire 8 45),
    (.wire 8 43, .wire 8 46),
    (.wire 7 3, .wire 6 4),
    (.wire 5 67, .wire 6 5),
    (.wire 7 3, .wire 6 6),
    (.wire 6 7, .wire 7 4),
    (.virt 23780, .wire 7 5),
    (.wire 5 67, .wire 7 6),
    (.virt 23780, .wire 8 48),
    (.virt 23780, .wire 8 49),
    (.wire 4 15, .wire 8 50),
    (.virt 23780, .wire 8 52),
    (.virt 23780, .wire 8 53),
    (.wire 7 7, .wire 8 54),
    (.wire 8 51, .wire 4 24),
    (.wire 8 55, .wire 4 25),
    (.wire 8 51, .wire 4 26),
    (.wire 4 27, .wire 8 56),
    (.wire 5 79, .wire 8 57),
    (.wire 5 79, .wire 8 58),
    (.wire 4 27, .wire 8 60),
    (.virt 23753, .wire 8 61)
  ]

/-- `publicBatchWrapper4.copies`, items `416..448`. -/
def publicBatchWrapper4.copies13 : List (Target × Target) := [
    (.wire 8 59, .wire 8 62),
    (.wire 4 27, .wire 8 64),
    (.wire 8 7, .wire 8 65),
    (.wire 8 7, .wire 8 66),
    (.wire 4 27, .wire 8 68),
    (.virt 23754, .wire 8 69),
    (.wire 8 67, .wire 8 70),
    (.wire 4 27, .wire 8 72),
    (.wire 8 15, .wire 8 73),
    (.wire 8 15, .wire 8 74),
    (.wire 4 27, .wire 8 76),
    (.virt 23755, .wire 8 77),
    (.wire 8 75, .wire 8 78),
    (.wire 4 27, .wire 9 0),
    (.wire 8 23, .wire 9 1),
    (.wire 8 23, .wire 9 2),
    (.wire 4 27, .wire 9 4),
    (.virt 23756, .wire 9 5),
    (.wire 9 3, .wire 9 6),
    (.wire 4 27, .wire 9 8),
    (.wire 8 31, .wire 9 9),
    (.wire 8 31, .wire 9 10),
    (.wire 4 27, .wire 9 12),
    (.virt 23757, .wire 9 13),
    (.wire 9 11, .wire 9 14),
    (.wire 4 27, .wire 9 16),
    (.wire 8 39, .wire 9 17),
    (.wire 8 39, .wire 9 18),
    (.wire 4 27, .wire 9 20),
    (.virt 23751, .wire 9 21),
    (.wire 9 19, .wire 9 22),
    (.wire 4 27, .wire 9 24)
  ]

/-- `publicBatchWrapper4.copies`, items `448..480`. -/
def publicBatchWrapper4.copies14 : List (Target × Target) := [
    (.wire 8 47, .wire 9 25),
    (.wire 8 47, .wire 9 26),
    (.wire 4 27, .wire 9 28),
    (.virt 23752, .wire 9 29),
    (.wire 9 27, .wire 9 30),
    (.wire 7 7, .wire 6 8),
    (.wire 8 51, .wire 6 9),
    (.wire 7 7, .wire 6 10),
    (.wire 6 11, .wire 7 8),
    (.virt 23780, .wire 7 9),
    (.wire 8 51, .wire 7 10),
    (.virt 23780, .wire 9 32),
    (.virt 23780, .wire 9 33),
    (.virt 23815, .wire 9 34),
    (.virt 5919, .wire 9 36),
    (.virt 23780, .wire 9 37),
    (.wire 9 23, .wire 9 38),
    (.virt 23815, .wire 4 28),
    (.wire 9 39, .wire 4 29),
    (.virt 23815, .wire 4 30),
    (.wire 9 39, .wire 4 32),
    (.virt 23816, .wire 4 33),
    (.wire 9 39, .wire 4 34),
    (.wire 4 35, .wire 9 40),
    (.virt 23780, .wire 9 41),
    (.wire 9 35, .wire 9 42),
    (.wire 4 31, .virt 23781),
    (.wire 9 43, .virt 23781),
    (.wire 1 43, .wire 6 12),
    (.virt 23815, .wire 6 13),
    (.wire 1 43, .wire 6 14),
    (.wire 6 15, .wire 7 12)
  ]

/-- `publicBatchWrapper4.copies`, items `480..512`. -/
def publicBatchWrapper4.copies15 : List (Target × Target) := [
    (.virt 23780, .wire 7 13),
    (.virt 23815, .wire 7 14),
    (.wire 7 15, .virt 23780),
    (.virt 23780, .wire 9 44),
    (.virt 23780, .wire 9 45),
    (.virt 23817, .wire 9 46),
    (.virt 5920, .wire 9 48),
    (.virt 23780, .wire 9 49),
    (.wire 9 31, .wire 9 50),
    (.virt 23817, .wire 4 36),
    (.wire 9 51, .wire 4 37),
    (.virt 23817, .wire 4 38),
    (.wire 9 51, .wire 4 40),
    (.virt 23818, .wire 4 41),
    (.wire 9 51, .wire 4 42),
    (.wire 4 43, .wire 9 52),
    (.virt 23780, .wire 9 53),
    (.wire 9 47, .wire 9 54),
    (.wire 4 39, .virt 23781),
    (.wire 9 55, .virt 23781),
    (.wire 1 43, .wire 6 16),
    (.virt 23817, .wire 6 17),
    (.wire 1 43, .wire 6 18),
    (.wire 6 19, .wire 7 16),
    (.virt 23780, .wire 7 17),
    (.virt 23817, .wire 7 18),
    (.wire 7 19, .virt 23780),
    (.virt 23780, .wire 9 56),
    (.virt 23780, .wire 9 57),
    (.virt 23819, .wire 9 58),
    (.virt 5921, .wire 9 60),
    (.virt 23780, .wire 9 61)
  ]

/-- `publicBatchWrapper4.copies`, items `512..544`. -/
def publicBatchWrapper4.copies16 : List (Target × Target) := [
    (.wire 8 63, .wire 9 62),
    (.virt 23819, .wire 4 44),
    (.wire 9 63, .wire 4 45),
    (.virt 23819, .wire 4 46),
    (.wire 9 63, .wire 4 48),
    (.virt 23820, .wire 4 49),
    (.wire 9 63, .wire 4 50),
    (.wire 4 51, .wire 9 64),
    (.virt 23780, .wire 9 65),
    (.wire 9 59, .wire 9 66),
    (.wire 4 47, .virt 23781),
    (.wire 9 67, .virt 23781),
    (.virt 23780, .wire 9 68),
    (.virt 23780, .wire 9 69),
    (.virt 23821, .wire 9 70),
    (.virt 5922, .wire 9 72),
    (.virt 23780, .wire 9 73),
    (.wire 8 71, .wire 9 74),
    (.virt 23821, .wire 4 52),
    (.wire 9 75, .wire 4 53),
    (.virt 23821, .wire 4 54),
    (.wire 9 75, .wire 4 56),
    (.virt 23822, .wire 4 57),
    (.wire 9 75, .wire 4 58),
    (.wire 4 59, .wire 9 76),
    (.virt 23780, .wire 9 77),
    (.wire 9 71, .wire 9 78),
    (.wire 4 55, .virt 23781),
    (.wire 9 79, .virt 23781),
    (.virt 23780, .wire 10 0),
    (.virt 23780, .wire 10 1),
    (.virt 23823, .wire 10 2)
  ]

/-- `publicBatchWrapper4.copies`, items `544..576`. -/
def publicBatchWrapper4.copies17 : List (Target × Target) := [
    (.virt 5923, .wire 10 4),
    (.virt 23780, .wire 10 5),
    (.wire 8 79, .wire 10 6),
    (.virt 23823, .wire 4 60),
    (.wire 10 7, .wire 4 61),
    (.virt 23823, .wire 4 62),
    (.wire 10 7, .wire 4 64),
    (.virt 23824, .wire 4 65),
    (.wire 10 7, .wire 4 66),
    (.wire 4 67, .wire 10 8),
    (.virt 23780, .wire 10 9),
    (.wire 10 3, .wire 10 10),
    (.wire 4 63, .virt 23781),
    (.wire 10 11, .virt 23781),
    (.virt 23780, .wire 10 12),
    (.virt 23780, .wire 10 13),
    (.virt 23825, .wire 10 14),
    (.virt 5924, .wire 10 16),
    (.virt 23780, .wire 10 17),
    (.wire 9 7, .wire 10 18),
    (.virt 23825, .wire 4 68),
    (.wire 10 19, .wire 4 69),
    (.virt 23825, .wire 4 70),
    (.wire 10 19, .wire 4 72),
    (.virt 23826, .wire 4 73),
    (.wire 10 19, .wire 4 74),
    (.wire 4 75, .wire 10 20),
    (.virt 23780, .wire 10 21),
    (.wire 10 15, .wire 10 22),
    (.wire 4 71, .virt 23781),
    (.wire 10 23, .virt 23781),
    (.virt 23819, .wire 4 76)
  ]

/-- `publicBatchWrapper4.copies`, items `576..608`. -/
def publicBatchWrapper4.copies18 : List (Target × Target) := [
    (.virt 23821, .wire 4 77),
    (.virt 23819, .wire 4 78),
    (.virt 23823, .wire 11 0),
    (.virt 23825, .wire 11 1),
    (.virt 23823, .wire 11 2),
    (.wire 4 79, .wire 11 4),
    (.wire 11 3, .wire 11 5),
    (.wire 4 79, .wire 11 6),
    (.wire 1 43, .wire 6 20),
    (.wire 11 7, .wire 6 21),
    (.wire 1 43, .wire 6 22),
    (.wire 6 23, .wire 7 20),
    (.virt 23780, .wire 7 21),
    (.wire 11 7, .wire 7 22),
    (.wire 7 23, .virt 23780),
    (.virt 23780, .wire 10 24),
    (.virt 23780, .wire 10 25),
    (.virt 23827, .wire 10 26),
    (.virt 11863, .wire 10 28),
    (.virt 23780, .wire 10 29),
    (.wire 9 23, .wire 10 30),
    (.virt 23827, .wire 11 8),
    (.wire 10 31, .wire 11 9),
    (.virt 23827, .wire 11 10),
    (.wire 10 31, .wire 11 12),
    (.virt 23828, .wire 11 13),
    (.wire 10 31, .wire 11 14),
    (.wire 11 15, .wire 10 32),
    (.virt 23780, .wire 10 33),
    (.wire 10 27, .wire 10 34),
    (.wire 11 11, .virt 23781),
    (.wire 10 35, .virt 23781)
  ]

/-- `publicBatchWrapper4.copies`, items `608..640`. -/
def publicBatchWrapper4.copies19 : List (Target × Target) := [
    (.wire 2 7, .wire 6 24),
    (.virt 23827, .wire 6 25),
    (.wire 2 7, .wire 6 26),
    (.wire 6 27, .wire 7 24),
    (.virt 23780, .wire 7 25),
    (.virt 23827, .wire 7 26),
    (.wire 7 27, .virt 23780),
    (.virt 23780, .wire 10 36),
    (.virt 23780, .wire 10 37),
    (.virt 23829, .wire 10 38),
    (.virt 11864, .wire 10 40),
    (.virt 23780, .wire 10 41),
    (.wire 9 31, .wire 10 42),
    (.virt 23829, .wire 11 16),
    (.wire 10 43, .wire 11 17),
    (.virt 23829, .wire 11 18),
    (.wire 10 43, .wire 11 20),
    (.virt 23830, .wire 11 21),
    (.wire 10 43, .wire 11 22),
    (.wire 11 23, .wire 10 44),
    (.virt 23780, .wire 10 45),
    (.wire 10 39, .wire 10 46),
    (.wire 11 19, .virt 23781),
    (.wire 10 47, .virt 23781),
    (.wire 2 7, .wire 6 28),
    (.virt 23829, .wire 6 29),
    (.wire 2 7, .wire 6 30),
    (.wire 6 31, .wire 7 28),
    (.virt 23780, .wire 7 29),
    (.virt 23829, .wire 7 30),
    (.wire 7 31, .virt 23780),
    (.virt 23780, .wire 10 48)
  ]

/-- `publicBatchWrapper4.copies`, items `640..672`. -/
def publicBatchWrapper4.copies20 : List (Target × Target) := [
    (.virt 23780, .wire 10 49),
    (.virt 23831, .wire 10 50),
    (.virt 11865, .wire 10 52),
    (.virt 23780, .wire 10 53),
    (.wire 8 63, .wire 10 54),
    (.virt 23831, .wire 11 24),
    (.wire 10 55, .wire 11 25),
    (.virt 23831, .wire 11 26),
    (.wire 10 55, .wire 11 28),
    (.virt 23832, .wire 11 29),
    (.wire 10 55, .wire 11 30),
    (.wire 11 31, .wire 10 56),
    (.virt 23780, .wire 10 57),
    (.wire 10 51, .wire 10 58),
    (.wire 11 27, .virt 23781),
    (.wire 10 59, .virt 23781),
    (.virt 23780, .wire 10 60),
    (.virt 23780, .wire 10 61),
    (.virt 23833, .wire 10 62),
    (.virt 11866, .wire 10 64),
    (.virt 23780, .wire 10 65),
    (.wire 8 71, .wire 10 66),
    (.virt 23833, .wire 11 32),
    (.wire 10 67, .wire 11 33),
    (.virt 23833, .wire 11 34),
    (.wire 10 67, .wire 11 36),
    (.virt 23834, .wire 11 37),
    (.wire 10 67, .wire 11 38),
    (.wire 11 39, .wire 10 68),
    (.virt 23780, .wire 10 69),
    (.wire 10 63, .wire 10 70),
    (.wire 11 35, .virt 23781)
  ]

/-- `publicBatchWrapper4.copies`, items `672..704`. -/
def publicBatchWrapper4.copies21 : List (Target × Target) := [
    (.wire 10 71, .virt 23781),
    (.virt 23780, .wire 10 72),
    (.virt 23780, .wire 10 73),
    (.virt 23835, .wire 10 74),
    (.virt 11867, .wire 10 76),
    (.virt 23780, .wire 10 77),
    (.wire 8 79, .wire 10 78),
    (.virt 23835, .wire 11 40),
    (.wire 10 79, .wire 11 41),
    (.virt 23835, .wire 11 42),
    (.wire 10 79, .wire 11 44),
    (.virt 23836, .wire 11 45),
    (.wire 10 79, .wire 11 46),
    (.wire 11 47, .wire 12 0),
    (.virt 23780, .wire 12 1),
    (.wire 10 75, .wire 12 2),
    (.wire 11 43, .virt 23781),
    (.wire 12 3, .virt 23781),
    (.virt 23780, .wire 12 4),
    (.virt 23780, .wire 12 5),
    (.virt 23837, .wire 12 6),
    (.virt 11868, .wire 12 8),
    (.virt 23780, .wire 12 9),
    (.wire 9 7, .wire 12 10),
    (.virt 23837, .wire 11 48),
    (.wire 12 11, .wire 11 49),
    (.virt 23837, .wire 11 50),
    (.wire 12 11, .wire 11 52),
    (.virt 23838, .wire 11 53),
    (.wire 12 11, .wire 11 54),
    (.wire 11 55, .wire 12 12),
    (.virt 23780, .wire 12 13)
  ]

/-- `publicBatchWrapper4.copies`, items `704..736`. -/
def publicBatchWrapper4.copies22 : List (Target × Target) := [
    (.wire 12 7, .wire 12 14),
    (.wire 11 51, .virt 23781),
    (.wire 12 15, .virt 23781),
    (.virt 23831, .wire 11 56),
    (.virt 23833, .wire 11 57),
    (.virt 23831, .wire 11 58),
    (.virt 23835, .wire 11 60),
    (.virt 23837, .wire 11 61),
    (.virt 23835, .wire 11 62),
    (.wire 11 59, .wire 11 64),
    (.wire 11 63, .wire 11 65),
    (.wire 11 59, .wire 11 66),
    (.wire 2 7, .wire 6 32),
    (.wire 11 67, .wire 6 33),
    (.wire 2 7, .wire 6 34),
    (.wire 6 35, .wire 7 32),
    (.virt 23780, .wire 7 33),
    (.wire 11 67, .wire 7 34),
    (.wire 7 35, .virt 23780),
    (.virt 23780, .wire 12 16),
    (.virt 23780, .wire 12 17),
    (.virt 23839, .wire 12 18),
    (.virt 17807, .wire 12 20),
    (.virt 23780, .wire 12 21),
    (.wire 9 23, .wire 12 22),
    (.virt 23839, .wire 11 68),
    (.wire 12 23, .wire 11 69),
    (.virt 23839, .wire 11 70),
    (.wire 12 23, .wire 11 72),
    (.virt 23840, .wire 11 73),
    (.wire 12 23, .wire 11 74),
    (.wire 11 75, .wire 12 24)
  ]

/-- `publicBatchWrapper4.copies`, items `736..768`. -/
def publicBatchWrapper4.copies23 : List (Target × Target) := [
    (.virt 23780, .wire 12 25),
    (.wire 12 19, .wire 12 26),
    (.wire 11 71, .virt 23781),
    (.wire 12 27, .virt 23781),
    (.wire 2 51, .wire 6 36),
    (.virt 23839, .wire 6 37),
    (.wire 2 51, .wire 6 38),
    (.wire 6 39, .wire 7 36),
    (.virt 23780, .wire 7 37),
    (.virt 23839, .wire 7 38),
    (.wire 7 39, .virt 23780),
    (.virt 23780, .wire 12 28),
    (.virt 23780, .wire 12 29),
    (.virt 23841, .wire 12 30),
    (.virt 17808, .wire 12 32),
    (.virt 23780, .wire 12 33),
    (.wire 9 31, .wire 12 34),
    (.virt 23841, .wire 11 76),
    (.wire 12 35, .wire 11 77),
    (.virt 23841, .wire 11 78),
    (.wire 12 35, .wire 13 0),
    (.virt 23842, .wire 13 1),
    (.wire 12 35, .wire 13 2),
    (.wire 13 3, .wire 12 36),
    (.virt 23780, .wire 12 37),
    (.wire 12 31, .wire 12 38),
    (.wire 11 79, .virt 23781),
    (.wire 12 39, .virt 23781),
    (.wire 2 51, .wire 6 40),
    (.virt 23841, .wire 6 41),
    (.wire 2 51, .wire 6 42),
    (.wire 6 43, .wire 7 40)
  ]

/-- `publicBatchWrapper4.copies`, items `768..800`. -/
def publicBatchWrapper4.copies24 : List (Target × Target) := [
    (.virt 23780, .wire 7 41),
    (.virt 23841, .wire 7 42),
    (.wire 7 43, .virt 23780),
    (.virt 23780, .wire 12 40),
    (.virt 23780, .wire 12 41),
    (.virt 23843, .wire 12 42),
    (.virt 17809, .wire 12 44),
    (.virt 23780, .wire 12 45),
    (.wire 8 63, .wire 12 46),
    (.virt 23843, .wire 13 4),
    (.wire 12 47, .wire 13 5),
    (.virt 23843, .wire 13 6),
    (.wire 12 47, .wire 13 8),
    (.virt 23844, .wire 13 9),
    (.wire 12 47, .wire 13 10),
    (.wire 13 11, .wire 12 48),
    (.virt 23780, .wire 12 49),
    (.wire 12 43, .wire 12 50),
    (.wire 13 7, .virt 23781),
    (.wire 12 51, .virt 23781),
    (.virt 23780, .wire 12 52),
    (.virt 23780, .wire 12 53),
    (.virt 23845, .wire 12 54),
    (.virt 17810, .wire 12 56),
    (.virt 23780, .wire 12 57),
    (.wire 8 71, .wire 12 58),
    (.virt 23845, .wire 13 12),
    (.wire 12 59, .wire 13 13),
    (.virt 23845, .wire 13 14),
    (.wire 12 59, .wire 13 16),
    (.virt 23846, .wire 13 17),
    (.wire 12 59, .wire 13 18)
  ]

/-- `publicBatchWrapper4.copies`, items `800..832`. -/
def publicBatchWrapper4.copies25 : List (Target × Target) := [
    (.wire 13 19, .wire 12 60),
    (.virt 23780, .wire 12 61),
    (.wire 12 55, .wire 12 62),
    (.wire 13 15, .virt 23781),
    (.wire 12 63, .virt 23781),
    (.virt 23780, .wire 12 64),
    (.virt 23780, .wire 12 65),
    (.virt 23847, .wire 12 66),
    (.virt 17811, .wire 12 68),
    (.virt 23780, .wire 12 69),
    (.wire 8 79, .wire 12 70),
    (.virt 23847, .wire 13 20),
    (.wire 12 71, .wire 13 21),
    (.virt 23847, .wire 13 22),
    (.wire 12 71, .wire 13 24),
    (.virt 23848, .wire 13 25),
    (.wire 12 71, .wire 13 26),
    (.wire 13 27, .wire 12 72),
    (.virt 23780, .wire 12 73),
    (.wire 12 67, .wire 12 74),
    (.wire 13 23, .virt 23781),
    (.wire 12 75, .virt 23781),
    (.virt 23780, .wire 12 76),
    (.virt 23780, .wire 12 77),
    (.virt 23849, .wire 12 78),
    (.virt 17812, .wire 14 0),
    (.virt 23780, .wire 14 1),
    (.wire 9 7, .wire 14 2),
    (.virt 23849, .wire 13 28),
    (.wire 14 3, .wire 13 29),
    (.virt 23849, .wire 13 30),
    (.wire 14 3, .wire 13 32)
  ]

/-- `publicBatchWrapper4.copies`, items `832..864`. -/
def publicBatchWrapper4.copies26 : List (Target × Target) := [
    (.virt 23850, .wire 13 33),
    (.wire 14 3, .wire 13 34),
    (.wire 13 35, .wire 14 4),
    (.virt 23780, .wire 14 5),
    (.wire 12 79, .wire 14 6),
    (.wire 13 31, .virt 23781),
    (.wire 14 7, .virt 23781),
    (.virt 23843, .wire 13 36),
    (.virt 23845, .wire 13 37),
    (.virt 23843, .wire 13 38),
    (.virt 23847, .wire 13 40),
    (.virt 23849, .wire 13 41),
    (.virt 23847, .wire 13 42),
    (.wire 13 39, .wire 13 44),
    (.wire 13 43, .wire 13 45),
    (.wire 13 39, .wire 13 46),
    (.wire 2 51, .wire 6 44),
    (.wire 13 47, .wire 6 45),
    (.wire 2 51, .wire 6 46),
    (.wire 6 47, .wire 7 44),
    (.virt 23780, .wire 7 45),
    (.wire 13 47, .wire 7 46),
    (.wire 7 47, .virt 23780),
    (.virt 23780, .wire 14 8),
    (.virt 23780, .wire 14 9),
    (.virt 23851, .wire 14 10),
    (.virt 23751, .wire 14 12),
    (.virt 23780, .wire 14 13),
    (.wire 9 23, .wire 14 14),
    (.virt 23851, .wire 13 48),
    (.wire 14 15, .wire 13 49),
    (.virt 23851, .wire 13 50)
  ]

/-- `publicBatchWrapper4.copies`, items `864..896`. -/
def publicBatchWrapper4.copies27 : List (Target × Target) := [
    (.wire 14 15, .wire 13 52),
    (.virt 23852, .wire 13 53),
    (.wire 14 15, .wire 13 54),
    (.wire 13 55, .wire 14 16),
    (.virt 23780, .wire 14 17),
    (.wire 14 11, .wire 14 18),
    (.wire 13 51, .virt 23781),
    (.wire 14 19, .virt 23781),
    (.wire 4 15, .wire 6 48),
    (.virt 23851, .wire 6 49),
    (.wire 4 15, .wire 6 50),
    (.wire 6 51, .wire 7 48),
    (.virt 23780, .wire 7 49),
    (.virt 23851, .wire 7 50),
    (.wire 7 51, .virt 23780),
    (.virt 23780, .wire 14 20),
    (.virt 23780, .wire 14 21),
    (.virt 23853, .wire 14 22),
    (.virt 23752, .wire 14 24),
    (.virt 23780, .wire 14 25),
    (.wire 9 31, .wire 14 26),
    (.virt 23853, .wire 13 56),
    (.wire 14 27, .wire 13 57),
    (.virt 23853, .wire 13 58),
    (.wire 14 27, .wire 13 60),
    (.virt 23854, .wire 13 61),
    (.wire 14 27, .wire 13 62),
    (.wire 13 63, .wire 14 28),
    (.virt 23780, .wire 14 29),
    (.wire 14 23, .wire 14 30),
    (.wire 13 59, .virt 23781),
    (.wire 14 31, .virt 23781)
  ]

/-- `publicBatchWrapper4.copies`, items `896..928`. -/
def publicBatchWrapper4.copies28 : List (Target × Target) := [
    (.wire 4 15, .wire 6 52),
    (.virt 23853, .wire 6 53),
    (.wire 4 15, .wire 6 54),
    (.wire 6 55, .wire 7 52),
    (.virt 23780, .wire 7 53),
    (.virt 23853, .wire 7 54),
    (.wire 7 55, .virt 23780),
    (.virt 23780, .wire 14 32),
    (.virt 23780, .wire 14 33),
    (.virt 23855, .wire 14 34),
    (.virt 23753, .wire 14 36),
    (.virt 23780, .wire 14 37),
    (.wire 8 63, .wire 14 38),
    (.virt 23855, .wire 13 64),
    (.wire 14 39, .wire 13 65),
    (.virt 23855, .wire 13 66),
    (.wire 14 39, .wire 13 68),
    (.virt 23856, .wire 13 69),
    (.wire 14 39, .wire 13 70),
    (.wire 13 71, .wire 14 40),
    (.virt 23780, .wire 14 41),
    (.wire 14 35, .wire 14 42),
    (.wire 13 67, .virt 23781),
    (.wire 14 43, .virt 23781),
    (.virt 23780, .wire 14 44),
    (.virt 23780, .wire 14 45),
    (.virt 23857, .wire 14 46),
    (.virt 23754, .wire 14 48),
    (.virt 23780, .wire 14 49),
    (.wire 8 71, .wire 14 50),
    (.virt 23857, .wire 13 72),
    (.wire 14 51, .wire 13 73)
  ]

/-- `publicBatchWrapper4.copies`, items `928..960`. -/
def publicBatchWrapper4.copies29 : List (Target × Target) := [
    (.virt 23857, .wire 13 74),
    (.wire 14 51, .wire 13 76),
    (.virt 23858, .wire 13 77),
    (.wire 14 51, .wire 13 78),
    (.wire 13 79, .wire 14 52),
    (.virt 23780, .wire 14 53),
    (.wire 14 47, .wire 14 54),
    (.wire 13 75, .virt 23781),
    (.wire 14 55, .virt 23781),
    (.virt 23780, .wire 14 56),
    (.virt 23780, .wire 14 57),
    (.virt 23859, .wire 14 58),
    (.virt 23755, .wire 14 60),
    (.virt 23780, .wire 14 61),
    (.wire 8 79, .wire 14 62),
    (.virt 23859, .wire 15 0),
    (.wire 14 63, .wire 15 1),
    (.virt 23859, .wire 15 2),
    (.wire 14 63, .wire 15 4),
    (.virt 23860, .wire 15 5),
    (.wire 14 63, .wire 15 6),
    (.wire 15 7, .wire 14 64),
    (.virt 23780, .wire 14 65),
    (.wire 14 59, .wire 14 66),
    (.wire 15 3, .virt 23781),
    (.wire 14 67, .virt 23781),
    (.virt 23780, .wire 14 68),
    (.virt 23780, .wire 14 69),
    (.virt 23861, .wire 14 70),
    (.virt 23756, .wire 14 72),
    (.virt 23780, .wire 14 73),
    (.wire 9 7, .wire 14 74)
  ]

/-- `publicBatchWrapper4.copies`, items `960..992`. -/
def publicBatchWrapper4.copies30 : List (Target × Target) := [
    (.virt 23861, .wire 15 8),
    (.wire 14 75, .wire 15 9),
    (.virt 23861, .wire 15 10),
    (.wire 14 75, .wire 15 12),
    (.virt 23862, .wire 15 13),
    (.wire 14 75, .wire 15 14),
    (.wire 15 15, .wire 14 76),
    (.virt 23780, .wire 14 77),
    (.wire 14 71, .wire 14 78),
    (.wire 15 11, .virt 23781),
    (.wire 14 79, .virt 23781),
    (.virt 23855, .wire 15 16),
    (.virt 23857, .wire 15 17),
    (.virt 23855, .wire 15 18),
    (.virt 23859, .wire 15 20),
    (.virt 23861, .wire 15 21),
    (.virt 23859, .wire 15 22),
    (.wire 15 19, .wire 15 24),
    (.wire 15 23, .wire 15 25),
    (.wire 15 19, .wire 15 26),
    (.wire 4 15, .wire 6 56),
    (.wire 15 27, .wire 6 57),
    (.wire 4 15, .wire 6 58),
    (.wire 6 59, .wire 7 56),
    (.virt 23780, .wire 7 57),
    (.wire 15 27, .wire 7 58),
    (.wire 7 59, .virt 23780),
    (.wire 1 43, .wire 16 0),
    (.virt 5926, .wire 16 1),
    (.virt 5926, .wire 16 2),
    (.wire 1 43, .wire 16 4),
    (.virt 23781, .wire 16 5)
  ]

/-- `publicBatchWrapper4.copies`, items `992..1024`. -/
def publicBatchWrapper4.copies31 : List (Target × Target) := [
    (.wire 16 3, .wire 16 6),
    (.wire 1 43, .wire 16 8),
    (.virt 5927, .wire 16 9),
    (.virt 5927, .wire 16 10),
    (.wire 1 43, .wire 16 12),
    (.virt 23781, .wire 16 13),
    (.wire 16 11, .wire 16 14),
    (.wire 1 43, .wire 16 16),
    (.virt 5928, .wire 16 17),
    (.virt 5928, .wire 16 18),
    (.wire 1 43, .wire 16 20),
    (.virt 23781, .wire 16 21),
    (.wire 16 19, .wire 16 22),
    (.wire 1 43, .wire 16 24),
    (.virt 5929, .wire 16 25),
    (.virt 5929, .wire 16 26),
    (.wire 1 43, .wire 16 28),
    (.virt 23781, .wire 16 29),
    (.wire 16 27, .wire 16 30),
    (.wire 1 43, .wire 16 32),
    (.virt 5930, .wire 16 33),
    (.virt 5930, .wire 16 34),
    (.wire 1 43, .wire 16 36),
    (.virt 23781, .wire 16 37),
    (.wire 16 35, .wire 16 38),
    (.wire 1 43, .wire 16 40),
    (.virt 5931, .wire 16 41),
    (.virt 5931, .wire 16 42),
    (.wire 1 43, .wire 16 44),
    (.virt 23781, .wire 16 45),
    (.wire 16 43, .wire 16 46),
    (.wire 1 43, .wire 16 48)
  ]

/-- `publicBatchWrapper4.copies`, items `1024..1056`. -/
def publicBatchWrapper4.copies32 : List (Target × Target) := [
    (.virt 5932, .wire 16 49),
    (.virt 5932, .wire 16 50),
    (.wire 1 43, .wire 16 52),
    (.virt 23781, .wire 16 53),
    (.wire 16 51, .wire 16 54),
    (.wire 1 43, .wire 16 56),
    (.virt 5933, .wire 16 57),
    (.virt 5933, .wire 16 58),
    (.wire 1 43, .wire 16 60),
    (.virt 23781, .wire 16 61),
    (.wire 16 59, .wire 16 62),
    (.wire 1 43, .wire 16 64),
    (.virt 5934, .wire 16 65),
    (.virt 5934, .wire 16 66),
    (.wire 1 43, .wire 16 68),
    (.virt 23781, .wire 16 69),
    (.wire 16 67, .wire 16 70),
    (.wire 1 43, .wire 16 72),
    (.virt 5935, .wire 16 73),
    (.virt 5935, .wire 16 74),
    (.wire 1 43, .wire 16 76),
    (.virt 23781, .wire 16 77),
    (.wire 16 75, .wire 16 78),
    (.wire 2 7, .wire 17 0),
    (.virt 11870, .wire 17 1),
    (.virt 11870, .wire 17 2),
    (.wire 2 7, .wire 17 4),
    (.virt 23781, .wire 17 5),
    (.wire 17 3, .wire 17 6),
    (.wire 2 7, .wire 17 8),
    (.virt 11871, .wire 17 9),
    (.virt 11871, .wire 17 10)
  ]

/-- `publicBatchWrapper4.copies`, items `1056..1088`. -/
def publicBatchWrapper4.copies33 : List (Target × Target) := [
    (.wire 2 7, .wire 17 12),
    (.virt 23781, .wire 17 13),
    (.wire 17 11, .wire 17 14),
    (.wire 2 7, .wire 17 16),
    (.virt 11872, .wire 17 17),
    (.virt 11872, .wire 17 18),
    (.wire 2 7, .wire 17 20),
    (.virt 23781, .wire 17 21),
    (.wire 17 19, .wire 17 22),
    (.wire 2 7, .wire 17 24),
    (.virt 11873, .wire 17 25),
    (.virt 11873, .wire 17 26),
    (.wire 2 7, .wire 17 28),
    (.virt 23781, .wire 17 29),
    (.wire 17 27, .wire 17 30),
    (.wire 2 7, .wire 17 32),
    (.virt 11874, .wire 17 33),
    (.virt 11874, .wire 17 34),
    (.wire 2 7, .wire 17 36),
    (.virt 23781, .wire 17 37),
    (.wire 17 35, .wire 17 38),
    (.wire 2 7, .wire 17 40),
    (.virt 11875, .wire 17 41),
    (.virt 11875, .wire 17 42),
    (.wire 2 7, .wire 17 44),
    (.virt 23781, .wire 17 45),
    (.wire 17 43, .wire 17 46),
    (.wire 2 7, .wire 17 48),
    (.virt 11876, .wire 17 49),
    (.virt 11876, .wire 17 50),
    (.wire 2 7, .wire 17 52),
    (.virt 23781, .wire 17 53)
  ]

/-- `publicBatchWrapper4.copies`, items `1088..1120`. -/
def publicBatchWrapper4.copies34 : List (Target × Target) := [
    (.wire 17 51, .wire 17 54),
    (.wire 2 7, .wire 17 56),
    (.virt 11877, .wire 17 57),
    (.virt 11877, .wire 17 58),
    (.wire 2 7, .wire 17 60),
    (.virt 23781, .wire 17 61),
    (.wire 17 59, .wire 17 62),
    (.wire 2 7, .wire 17 64),
    (.virt 11878, .wire 17 65),
    (.virt 11878, .wire 17 66),
    (.wire 2 7, .wire 17 68),
    (.virt 23781, .wire 17 69),
    (.wire 17 67, .wire 17 70),
    (.wire 2 7, .wire 17 72),
    (.virt 11879, .wire 17 73),
    (.virt 11879, .wire 17 74),
    (.wire 2 7, .wire 17 76),
    (.virt 23781, .wire 17 77),
    (.wire 17 75, .wire 17 78),
    (.wire 2 51, .wire 18 0),
    (.virt 17814, .wire 18 1),
    (.virt 17814, .wire 18 2),
    (.wire 2 51, .wire 18 4),
    (.virt 23781, .wire 18 5),
    (.wire 18 3, .wire 18 6),
    (.wire 2 51, .wire 18 8),
    (.virt 17815, .wire 18 9),
    (.virt 17815, .wire 18 10),
    (.wire 2 51, .wire 18 12),
    (.virt 23781, .wire 18 13),
    (.wire 18 11, .wire 18 14),
    (.wire 2 51, .wire 18 16)
  ]

/-- `publicBatchWrapper4.copies`, items `1120..1152`. -/
def publicBatchWrapper4.copies35 : List (Target × Target) := [
    (.virt 17816, .wire 18 17),
    (.virt 17816, .wire 18 18),
    (.wire 2 51, .wire 18 20),
    (.virt 23781, .wire 18 21),
    (.wire 18 19, .wire 18 22),
    (.wire 2 51, .wire 18 24),
    (.virt 17817, .wire 18 25),
    (.virt 17817, .wire 18 26),
    (.wire 2 51, .wire 18 28),
    (.virt 23781, .wire 18 29),
    (.wire 18 27, .wire 18 30),
    (.wire 2 51, .wire 18 32),
    (.virt 17818, .wire 18 33),
    (.virt 17818, .wire 18 34),
    (.wire 2 51, .wire 18 36),
    (.virt 23781, .wire 18 37),
    (.wire 18 35, .wire 18 38),
    (.wire 2 51, .wire 18 40),
    (.virt 17819, .wire 18 41),
    (.virt 17819, .wire 18 42),
    (.wire 2 51, .wire 18 44),
    (.virt 23781, .wire 18 45),
    (.wire 18 43, .wire 18 46),
    (.wire 2 51, .wire 18 48),
    (.virt 17820, .wire 18 49),
    (.virt 17820, .wire 18 50),
    (.wire 2 51, .wire 18 52),
    (.virt 23781, .wire 18 53),
    (.wire 18 51, .wire 18 54),
    (.wire 2 51, .wire 18 56),
    (.virt 17821, .wire 18 57),
    (.virt 17821, .wire 18 58)
  ]

/-- `publicBatchWrapper4.copies`, items `1152..1184`. -/
def publicBatchWrapper4.copies36 : List (Target × Target) := [
    (.wire 2 51, .wire 18 60),
    (.virt 23781, .wire 18 61),
    (.wire 18 59, .wire 18 62),
    (.wire 2 51, .wire 18 64),
    (.virt 17822, .wire 18 65),
    (.virt 17822, .wire 18 66),
    (.wire 2 51, .wire 18 68),
    (.virt 23781, .wire 18 69),
    (.wire 18 67, .wire 18 70),
    (.wire 2 51, .wire 18 72),
    (.virt 17823, .wire 18 73),
    (.virt 17823, .wire 18 74),
    (.wire 2 51, .wire 18 76),
    (.virt 23781, .wire 18 77),
    (.wire 18 75, .wire 18 78),
    (.wire 4 15, .wire 19 0),
    (.virt 23758, .wire 19 1),
    (.virt 23758, .wire 19 2),
    (.wire 4 15, .wire 19 4),
    (.virt 23781, .wire 19 5),
    (.wire 19 3, .wire 19 6),
    (.wire 4 15, .wire 19 8),
    (.virt 23759, .wire 19 9),
    (.virt 23759, .wire 19 10),
    (.wire 4 15, .wire 19 12),
    (.virt 23781, .wire 19 13),
    (.wire 19 11, .wire 19 14),
    (.wire 4 15, .wire 19 16),
    (.virt 23760, .wire 19 17),
    (.virt 23760, .wire 19 18),
    (.wire 4 15, .wire 19 20),
    (.virt 23781, .wire 19 21)
  ]

/-- `publicBatchWrapper4.copies`, items `1184..1216`. -/
def publicBatchWrapper4.copies37 : List (Target × Target) := [
    (.wire 19 19, .wire 19 22),
    (.wire 4 15, .wire 19 24),
    (.virt 23761, .wire 19 25),
    (.virt 23761, .wire 19 26),
    (.wire 4 15, .wire 19 28),
    (.virt 23781, .wire 19 29),
    (.wire 19 27, .wire 19 30),
    (.wire 4 15, .wire 19 32),
    (.virt 23762, .wire 19 33),
    (.virt 23762, .wire 19 34),
    (.wire 4 15, .wire 19 36),
    (.virt 23781, .wire 19 37),
    (.wire 19 35, .wire 19 38),
    (.wire 4 15, .wire 19 40),
    (.virt 23763, .wire 19 41),
    (.virt 23763, .wire 19 42),
    (.wire 4 15, .wire 19 44),
    (.virt 23781, .wire 19 45),
    (.wire 19 43, .wire 19 46),
    (.wire 4 15, .wire 19 48),
    (.virt 23764, .wire 19 49),
    (.virt 23764, .wire 19 50),
    (.wire 4 15, .wire 19 52),
    (.virt 23781, .wire 19 53),
    (.wire 19 51, .wire 19 54),
    (.wire 4 15, .wire 19 56),
    (.virt 23765, .wire 19 57),
    (.virt 23765, .wire 19 58),
    (.wire 4 15, .wire 19 60),
    (.virt 23781, .wire 19 61),
    (.wire 19 59, .wire 19 62),
    (.wire 4 15, .wire 19 64)
  ]

/-- `publicBatchWrapper4.copies`, items `1216..1248`. -/
def publicBatchWrapper4.copies38 : List (Target × Target) := [
    (.virt 23766, .wire 19 65),
    (.virt 23766, .wire 19 66),
    (.wire 4 15, .wire 19 68),
    (.virt 23781, .wire 19 69),
    (.wire 19 67, .wire 19 70),
    (.wire 4 15, .wire 19 72),
    (.virt 23767, .wire 19 73),
    (.virt 23767, .wire 19 74),
    (.wire 4 15, .wire 19 76),
    (.virt 23781, .wire 19 77),
    (.wire 19 75, .wire 19 78),
    (.wire 1 43, .wire 20 0),
    (.virt 5936, .wire 20 1),
    (.virt 5936, .wire 20 2),
    (.wire 1 43, .wire 20 4),
    (.virt 23781, .wire 20 5),
    (.wire 20 3, .wire 20 6),
    (.wire 1 43, .wire 20 8),
    (.virt 5937, .wire 20 9),
    (.virt 5937, .wire 20 10),
    (.wire 1 43, .wire 20 12),
    (.virt 23781, .wire 20 13),
    (.wire 20 11, .wire 20 14),
    (.wire 1 43, .wire 20 16),
    (.virt 5938, .wire 20 17),
    (.virt 5938, .wire 20 18),
    (.wire 1 43, .wire 20 20),
    (.virt 23781, .wire 20 21),
    (.wire 20 19, .wire 20 22),
    (.wire 1 43, .wire 20 24),
    (.virt 5939, .wire 20 25),
    (.virt 5939, .wire 20 26)
  ]

/-- `publicBatchWrapper4.copies`, items `1248..1280`. -/
def publicBatchWrapper4.copies39 : List (Target × Target) := [
    (.wire 1 43, .wire 20 28),
    (.virt 23781, .wire 20 29),
    (.wire 20 27, .wire 20 30),
    (.wire 1 43, .wire 20 32),
    (.virt 5940, .wire 20 33),
    (.virt 5940, .wire 20 34),
    (.wire 1 43, .wire 20 36),
    (.virt 23781, .wire 20 37),
    (.wire 20 35, .wire 20 38),
    (.wire 1 43, .wire 20 40),
    (.virt 5941, .wire 20 41),
    (.virt 5941, .wire 20 42),
    (.wire 1 43, .wire 20 44),
    (.virt 23781, .wire 20 45),
    (.wire 20 43, .wire 20 46),
    (.wire 1 43, .wire 20 48),
    (.virt 5942, .wire 20 49),
    (.virt 5942, .wire 20 50),
    (.wire 1 43, .wire 20 52),
    (.virt 23781, .wire 20 53),
    (.wire 20 51, .wire 20 54),
    (.wire 1 43, .wire 20 56),
    (.virt 5943, .wire 20 57),
    (.virt 5943, .wire 20 58),
    (.wire 1 43, .wire 20 60),
    (.virt 23781, .wire 20 61),
    (.wire 20 59, .wire 20 62),
    (.wire 2 7, .wire 20 64),
    (.virt 11880, .wire 20 65),
    (.virt 11880, .wire 20 66),
    (.wire 2 7, .wire 20 68),
    (.virt 23781, .wire 20 69)
  ]

/-- `publicBatchWrapper4.copies`, items `1280..1312`. -/
def publicBatchWrapper4.copies40 : List (Target × Target) := [
    (.wire 20 67, .wire 20 70),
    (.wire 2 7, .wire 20 72),
    (.virt 11881, .wire 20 73),
    (.virt 11881, .wire 20 74),
    (.wire 2 7, .wire 20 76),
    (.virt 23781, .wire 20 77),
    (.wire 20 75, .wire 20 78),
    (.wire 2 7, .wire 21 0),
    (.virt 11882, .wire 21 1),
    (.virt 11882, .wire 21 2),
    (.wire 2 7, .wire 21 4),
    (.virt 23781, .wire 21 5),
    (.wire 21 3, .wire 21 6),
    (.wire 2 7, .wire 21 8),
    (.virt 11883, .wire 21 9),
    (.virt 11883, .wire 21 10),
    (.wire 2 7, .wire 21 12),
    (.virt 23781, .wire 21 13),
    (.wire 21 11, .wire 21 14),
    (.wire 2 7, .wire 21 16),
    (.virt 11884, .wire 21 17),
    (.virt 11884, .wire 21 18),
    (.wire 2 7, .wire 21 20),
    (.virt 23781, .wire 21 21),
    (.wire 21 19, .wire 21 22),
    (.wire 2 7, .wire 21 24),
    (.virt 11885, .wire 21 25),
    (.virt 11885, .wire 21 26),
    (.wire 2 7, .wire 21 28),
    (.virt 23781, .wire 21 29),
    (.wire 21 27, .wire 21 30),
    (.wire 2 7, .wire 21 32)
  ]

/-- `publicBatchWrapper4.copies`, items `1312..1344`. -/
def publicBatchWrapper4.copies41 : List (Target × Target) := [
    (.virt 11886, .wire 21 33),
    (.virt 11886, .wire 21 34),
    (.wire 2 7, .wire 21 36),
    (.virt 23781, .wire 21 37),
    (.wire 21 35, .wire 21 38),
    (.wire 2 7, .wire 21 40),
    (.virt 11887, .wire 21 41),
    (.virt 11887, .wire 21 42),
    (.wire 2 7, .wire 21 44),
    (.virt 23781, .wire 21 45),
    (.wire 21 43, .wire 21 46),
    (.wire 2 51, .wire 21 48),
    (.virt 17824, .wire 21 49),
    (.virt 17824, .wire 21 50),
    (.wire 2 51, .wire 21 52),
    (.virt 23781, .wire 21 53),
    (.wire 21 51, .wire 21 54),
    (.wire 2 51, .wire 21 56),
    (.virt 17825, .wire 21 57),
    (.virt 17825, .wire 21 58),
    (.wire 2 51, .wire 21 60),
    (.virt 23781, .wire 21 61),
    (.wire 21 59, .wire 21 62),
    (.wire 2 51, .wire 21 64),
    (.virt 17826, .wire 21 65),
    (.virt 17826, .wire 21 66),
    (.wire 2 51, .wire 21 68),
    (.virt 23781, .wire 21 69),
    (.wire 21 67, .wire 21 70),
    (.wire 2 51, .wire 21 72),
    (.virt 17827, .wire 21 73),
    (.virt 17827, .wire 21 74)
  ]

/-- `publicBatchWrapper4.copies`, items `1344..1376`. -/
def publicBatchWrapper4.copies42 : List (Target × Target) := [
    (.wire 2 51, .wire 21 76),
    (.virt 23781, .wire 21 77),
    (.wire 21 75, .wire 21 78),
    (.wire 2 51, .wire 22 0),
    (.virt 17828, .wire 22 1),
    (.virt 17828, .wire 22 2),
    (.wire 2 51, .wire 22 4),
    (.virt 23781, .wire 22 5),
    (.wire 22 3, .wire 22 6),
    (.wire 2 51, .wire 22 8),
    (.virt 17829, .wire 22 9),
    (.virt 17829, .wire 22 10),
    (.wire 2 51, .wire 22 12),
    (.virt 23781, .wire 22 13),
    (.wire 22 11, .wire 22 14),
    (.wire 2 51, .wire 22 16),
    (.virt 17830, .wire 22 17),
    (.virt 17830, .wire 22 18),
    (.wire 2 51, .wire 22 20),
    (.virt 23781, .wire 22 21),
    (.wire 22 19, .wire 22 22),
    (.wire 2 51, .wire 22 24),
    (.virt 17831, .wire 22 25),
    (.virt 17831, .wire 22 26),
    (.wire 2 51, .wire 22 28),
    (.virt 23781, .wire 22 29),
    (.wire 22 27, .wire 22 30),
    (.wire 4 15, .wire 22 32),
    (.virt 23768, .wire 22 33),
    (.virt 23768, .wire 22 34),
    (.wire 4 15, .wire 22 36),
    (.virt 23781, .wire 22 37)
  ]

/-- `publicBatchWrapper4.copies`, items `1376..1408`. -/
def publicBatchWrapper4.copies43 : List (Target × Target) := [
    (.wire 22 35, .wire 22 38),
    (.wire 4 15, .wire 22 40),
    (.virt 23769, .wire 22 41),
    (.virt 23769, .wire 22 42),
    (.wire 4 15, .wire 22 44),
    (.virt 23781, .wire 22 45),
    (.wire 22 43, .wire 22 46),
    (.wire 4 15, .wire 22 48),
    (.virt 23770, .wire 22 49),
    (.virt 23770, .wire 22 50),
    (.wire 4 15, .wire 22 52),
    (.virt 23781, .wire 22 53),
    (.wire 22 51, .wire 22 54),
    (.wire 4 15, .wire 22 56),
    (.virt 23771, .wire 22 57),
    (.virt 23771, .wire 22 58),
    (.wire 4 15, .wire 22 60),
    (.virt 23781, .wire 22 61),
    (.wire 22 59, .wire 22 62),
    (.wire 4 15, .wire 22 64),
    (.virt 23772, .wire 22 65),
    (.virt 23772, .wire 22 66),
    (.wire 4 15, .wire 22 68),
    (.virt 23781, .wire 22 69),
    (.wire 22 67, .wire 22 70),
    (.wire 4 15, .wire 22 72),
    (.virt 23773, .wire 22 73),
    (.virt 23773, .wire 22 74),
    (.wire 4 15, .wire 22 76),
    (.virt 23781, .wire 22 77),
    (.wire 22 75, .wire 22 78),
    (.wire 4 15, .wire 23 0)
  ]

/-- `publicBatchWrapper4.copies`, items `1408..1419`. -/
def publicBatchWrapper4.copies44 : List (Target × Target) := [
    (.virt 23774, .wire 23 1),
    (.virt 23774, .wire 23 2),
    (.wire 4 15, .wire 23 4),
    (.virt 23781, .wire 23 5),
    (.wire 23 3, .wire 23 6),
    (.wire 4 15, .wire 23 8),
    (.virt 23775, .wire 23 9),
    (.virt 23775, .wire 23 10),
    (.wire 4 15, .wire 23 12),
    (.virt 23781, .wire 23 13),
    (.wire 23 11, .wire 23 14)
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
    ⟨.arithmetic 20, [1, (-1)]⟩  -- row 23
  ]
  copies := (((((publicBatchWrapper4.copies0 ++ publicBatchWrapper4.copies1) ++ (publicBatchWrapper4.copies2 ++ (publicBatchWrapper4.copies3 ++ publicBatchWrapper4.copies4))) ++ ((publicBatchWrapper4.copies5 ++ (publicBatchWrapper4.copies6 ++ publicBatchWrapper4.copies7)) ++ (publicBatchWrapper4.copies8 ++ (publicBatchWrapper4.copies9 ++ publicBatchWrapper4.copies10)))) ++ (((publicBatchWrapper4.copies11 ++ publicBatchWrapper4.copies12) ++ (publicBatchWrapper4.copies13 ++ (publicBatchWrapper4.copies14 ++ publicBatchWrapper4.copies15))) ++ ((publicBatchWrapper4.copies16 ++ (publicBatchWrapper4.copies17 ++ publicBatchWrapper4.copies18)) ++ (publicBatchWrapper4.copies19 ++ (publicBatchWrapper4.copies20 ++ publicBatchWrapper4.copies21))))) ++ ((((publicBatchWrapper4.copies22 ++ publicBatchWrapper4.copies23) ++ (publicBatchWrapper4.copies24 ++ (publicBatchWrapper4.copies25 ++ publicBatchWrapper4.copies26))) ++ ((publicBatchWrapper4.copies27 ++ (publicBatchWrapper4.copies28 ++ publicBatchWrapper4.copies29)) ++ (publicBatchWrapper4.copies30 ++ (publicBatchWrapper4.copies31 ++ publicBatchWrapper4.copies32)))) ++ (((publicBatchWrapper4.copies33 ++ (publicBatchWrapper4.copies34 ++ publicBatchWrapper4.copies35)) ++ (publicBatchWrapper4.copies36 ++ (publicBatchWrapper4.copies37 ++ publicBatchWrapper4.copies38))) ++ ((publicBatchWrapper4.copies39 ++ (publicBatchWrapper4.copies40 ++ publicBatchWrapper4.copies41)) ++ (publicBatchWrapper4.copies42 ++ (publicBatchWrapper4.copies43 ++ publicBatchWrapper4.copies44))))))
  constants := [
    (.virt 23780, 1),
    (.virt 23781, 0),
    (.virt 23782, 2),
    (.virt 23863, 8)
  ]
  publicInputs := [
    .virt 23776,
    .virt 23777,
    .virt 23778,
    .virt 23779,
    .wire 9 23,
    .wire 9 31,
    .wire 8 63,
    .wire 8 71,
    .wire 8 79,
    .wire 9 7,
    .wire 9 15,
    .virt 23863,
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
    .wire 23 15
  ]

/-- Named targets `inner_pis_0`. -/
def publicBatchWrapper4.inner_pis_0 : Fin 26 → Target :=
  ![.virt 5918, .virt 5919, .virt 5920, .virt 5921, .virt 5922, .virt 5923, .virt 5924, .virt 5925, .virt 5926, .virt 5927, .virt 5928, .virt 5929, .virt 5930, .virt 5931, .virt 5932, .virt 5933, .virt 5934, .virt 5935, .virt 5936, .virt 5937, .virt 5938, .virt 5939, .virt 5940, .virt 5941, .virt 5942, .virt 5943]

/-- Named targets `inner_pis_1`. -/
def publicBatchWrapper4.inner_pis_1 : Fin 26 → Target :=
  ![.virt 11862, .virt 11863, .virt 11864, .virt 11865, .virt 11866, .virt 11867, .virt 11868, .virt 11869, .virt 11870, .virt 11871, .virt 11872, .virt 11873, .virt 11874, .virt 11875, .virt 11876, .virt 11877, .virt 11878, .virt 11879, .virt 11880, .virt 11881, .virt 11882, .virt 11883, .virt 11884, .virt 11885, .virt 11886, .virt 11887]

/-- Named targets `inner_pis_2`. -/
def publicBatchWrapper4.inner_pis_2 : Fin 26 → Target :=
  ![.virt 17806, .virt 17807, .virt 17808, .virt 17809, .virt 17810, .virt 17811, .virt 17812, .virt 17813, .virt 17814, .virt 17815, .virt 17816, .virt 17817, .virt 17818, .virt 17819, .virt 17820, .virt 17821, .virt 17822, .virt 17823, .virt 17824, .virt 17825, .virt 17826, .virt 17827, .virt 17828, .virt 17829, .virt 17830, .virt 17831]

/-- Named targets `inner_pis_3`. -/
def publicBatchWrapper4.inner_pis_3 : Fin 26 → Target :=
  ![.virt 23750, .virt 23751, .virt 23752, .virt 23753, .virt 23754, .virt 23755, .virt 23756, .virt 23757, .virt 23758, .virt 23759, .virt 23760, .virt 23761, .virt 23762, .virt 23763, .virt 23764, .virt 23765, .virt 23766, .virt 23767, .virt 23768, .virt 23769, .virt 23770, .virt 23771, .virt 23772, .virt 23773, .virt 23774, .virt 23775]

/-- Named targets `aggregator_address`. -/
def publicBatchWrapper4.aggregator_address : Fin 4 → Target :=
  ![.virt 23776, .virt 23777, .virt 23778, .virt 23779]

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def publicBatchWrapper4.verifiers : List (String × List Target) :=
  [
    ("private_batch_wrapper_n2", [.virt 5918, .virt 5919, .virt 5920, .virt 5921, .virt 5922, .virt 5923, .virt 5924, .virt 5925, .virt 5926, .virt 5927, .virt 5928, .virt 5929, .virt 5930, .virt 5931, .virt 5932, .virt 5933, .virt 5934, .virt 5935, .virt 5936, .virt 5937, .virt 5938, .virt 5939, .virt 5940, .virt 5941, .virt 5942, .virt 5943]),
    ("private_batch_wrapper_n2", [.virt 11862, .virt 11863, .virt 11864, .virt 11865, .virt 11866, .virt 11867, .virt 11868, .virt 11869, .virt 11870, .virt 11871, .virt 11872, .virt 11873, .virt 11874, .virt 11875, .virt 11876, .virt 11877, .virt 11878, .virt 11879, .virt 11880, .virt 11881, .virt 11882, .virt 11883, .virt 11884, .virt 11885, .virt 11886, .virt 11887]),
    ("private_batch_wrapper_n2", [.virt 17806, .virt 17807, .virt 17808, .virt 17809, .virt 17810, .virt 17811, .virt 17812, .virt 17813, .virt 17814, .virt 17815, .virt 17816, .virt 17817, .virt 17818, .virt 17819, .virt 17820, .virt 17821, .virt 17822, .virt 17823, .virt 17824, .virt 17825, .virt 17826, .virt 17827, .virt 17828, .virt 17829, .virt 17830, .virt 17831]),
    ("private_batch_wrapper_n2", [.virt 23750, .virt 23751, .virt 23752, .virt 23753, .virt 23754, .virt 23755, .virt 23756, .virt 23757, .virt 23758, .virt 23759, .virt 23760, .virt 23761, .virt 23762, .virt 23763, .virt 23764, .virt 23765, .virt 23766, .virt 23767, .virt 23768, .virt 23769, .virt 23770, .virt 23771, .virt 23772, .virt 23773, .virt 23774, .virt 23775])
  ]

theorem publicBatchWrapper4_copies0 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 5918) = a (.virt 23782) ∧ a (.virt 11862) = a (.virt 23782) ∧ a (.virt 17806) = a (.virt 23782) ∧ a (.virt 23750) = a (.virt 23782) ∧ a (.virt 23780) = a (.wire 0 0) ∧ a (.virt 23780) = a (.wire 0 1) ∧ a (.virt 23783) = a (.wire 0 2) ∧ a (.virt 23783) = a (.wire 1 0) ∧ a (.virt 5921) = a (.wire 1 1) ∧ a (.virt 23783) = a (.wire 1 2) ∧ a (.virt 5921) = a (.wire 1 4) ∧ a (.virt 23784) = a (.wire 1 5) ∧ a (.virt 5921) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 4) ∧ a (.virt 23780) = a (.wire 0 5) ∧ a (.wire 0 3) = a (.wire 0 6) ∧ a (.wire 1 3) = a (.virt 23781) ∧ a (.wire 0 7) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 0 8) ∧ a (.virt 23780) = a (.wire 0 9) ∧ a (.virt 23785) = a (.wire 0 10) ∧ a (.virt 23785) = a (.wire 1 8) ∧ a (.virt 5922) = a (.wire 1 9) ∧ a (.virt 23785) = a (.wire 1 10) ∧ a (.virt 5922) = a (.wire 1 12) ∧ a (.virt 23786) = a (.wire 1 13) ∧ a (.virt 5922) = a (.wire 1 14) ∧ a (.wire 1 15) = a (.wire 0 12) ∧ a (.virt 23780) = a (.wire 0 13) ∧ a (.wire 0 11) = a (.wire 0 14) ∧ a (.wire 1 11) = a (.virt 23781) ∧ a (.wire 0 15) = a (.virt 23781) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies1 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = a (.wire 0 16) ∧ a (.virt 23780) = a (.wire 0 17) ∧ a (.virt 23787) = a (.wire 0 18) ∧ a (.virt 23787) = a (.wire 1 16) ∧ a (.virt 5923) = a (.wire 1 17) ∧ a (.virt 23787) = a (.wire 1 18) ∧ a (.virt 5923) = a (.wire 1 20) ∧ a (.virt 23788) = a (.wire 1 21) ∧ a (.virt 5923) = a (.wire 1 22) ∧ a (.wire 1 23) = a (.wire 0 20) ∧ a (.virt 23780) = a (.wire 0 21) ∧ a (.wire 0 19) = a (.wire 0 22) ∧ a (.wire 1 19) = a (.virt 23781) ∧ a (.wire 0 23) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 0 24) ∧ a (.virt 23780) = a (.wire 0 25) ∧ a (.virt 23789) = a (.wire 0 26) ∧ a (.virt 23789) = a (.wire 1 24) ∧ a (.virt 5924) = a (.wire 1 25) ∧ a (.virt 23789) = a (.wire 1 26) ∧ a (.virt 5924) = a (.wire 1 28) ∧ a (.virt 23790) = a (.wire 1 29) ∧ a (.virt 5924) = a (.wire 1 30) ∧ a (.wire 1 31) = a (.wire 0 28) ∧ a (.virt 23780) = a (.wire 0 29) ∧ a (.wire 0 27) = a (.wire 0 30) ∧ a (.wire 1 27) = a (.virt 23781) ∧ a (.wire 0 31) = a (.virt 23781) ∧ a (.virt 23783) = a (.wire 1 32) ∧ a (.virt 23785) = a (.wire 1 33) ∧ a (.virt 23783) = a (.wire 1 34) ∧ a (.virt 23787) = a (.wire 1 36) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper4.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies2 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23789) = a (.wire 1 37) ∧ a (.virt 23787) = a (.wire 1 38) ∧ a (.wire 1 35) = a (.wire 1 40) ∧ a (.wire 1 39) = a (.wire 1 41) ∧ a (.wire 1 35) = a (.wire 1 42) ∧ a (.virt 23780) = a (.wire 0 32) ∧ a (.virt 23780) = a (.wire 0 33) ∧ a (.virt 23791) = a (.wire 0 34) ∧ a (.virt 23791) = a (.wire 1 44) ∧ a (.virt 11865) = a (.wire 1 45) ∧ a (.virt 23791) = a (.wire 1 46) ∧ a (.virt 11865) = a (.wire 1 48) ∧ a (.virt 23792) = a (.wire 1 49) ∧ a (.virt 11865) = a (.wire 1 50) ∧ a (.wire 1 51) = a (.wire 0 36) ∧ a (.virt 23780) = a (.wire 0 37) ∧ a (.wire 0 35) = a (.wire 0 38) ∧ a (.wire 1 47) = a (.virt 23781) ∧ a (.wire 0 39) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 0 40) ∧ a (.virt 23780) = a (.wire 0 41) ∧ a (.virt 23793) = a (.wire 0 42) ∧ a (.virt 23793) = a (.wire 1 52) ∧ a (.virt 11866) = a (.wire 1 53) ∧ a (.virt 23793) = a (.wire 1 54) ∧ a (.virt 11866) = a (.wire 1 56) ∧ a (.virt 23794) = a (.wire 1 57) ∧ a (.virt 11866) = a (.wire 1 58) ∧ a (.wire 1 59) = a (.wire 0 44) ∧ a (.virt 23780) = a (.wire 0 45) ∧ a (.wire 0 43) = a (.wire 0 46) ∧ a (.wire 1 55) = a (.virt 23781) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies3 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 0 47) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 0 48) ∧ a (.virt 23780) = a (.wire 0 49) ∧ a (.virt 23795) = a (.wire 0 50) ∧ a (.virt 23795) = a (.wire 1 60) ∧ a (.virt 11867) = a (.wire 1 61) ∧ a (.virt 23795) = a (.wire 1 62) ∧ a (.virt 11867) = a (.wire 1 64) ∧ a (.virt 23796) = a (.wire 1 65) ∧ a (.virt 11867) = a (.wire 1 66) ∧ a (.wire 1 67) = a (.wire 0 52) ∧ a (.virt 23780) = a (.wire 0 53) ∧ a (.wire 0 51) = a (.wire 0 54) ∧ a (.wire 1 63) = a (.virt 23781) ∧ a (.wire 0 55) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 0 56) ∧ a (.virt 23780) = a (.wire 0 57) ∧ a (.virt 23797) = a (.wire 0 58) ∧ a (.virt 23797) = a (.wire 1 68) ∧ a (.virt 11868) = a (.wire 1 69) ∧ a (.virt 23797) = a (.wire 1 70) ∧ a (.virt 11868) = a (.wire 1 72) ∧ a (.virt 23798) = a (.wire 1 73) ∧ a (.virt 11868) = a (.wire 1 74) ∧ a (.wire 1 75) = a (.wire 0 60) ∧ a (.virt 23780) = a (.wire 0 61) ∧ a (.wire 0 59) = a (.wire 0 62) ∧ a (.wire 1 71) = a (.virt 23781) ∧ a (.wire 0 63) = a (.virt 23781) ∧ a (.virt 23791) = a (.wire 1 76) ∧ a (.virt 23793) = a (.wire 1 77) ∧ a (.virt 23791) = a (.wire 1 78) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies3, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies4 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23795) = a (.wire 2 0) ∧ a (.virt 23797) = a (.wire 2 1) ∧ a (.virt 23795) = a (.wire 2 2) ∧ a (.wire 1 79) = a (.wire 2 4) ∧ a (.wire 2 3) = a (.wire 2 5) ∧ a (.wire 1 79) = a (.wire 2 6) ∧ a (.virt 23780) = a (.wire 0 64) ∧ a (.virt 23780) = a (.wire 0 65) ∧ a (.virt 23799) = a (.wire 0 66) ∧ a (.virt 23799) = a (.wire 2 8) ∧ a (.virt 17809) = a (.wire 2 9) ∧ a (.virt 23799) = a (.wire 2 10) ∧ a (.virt 17809) = a (.wire 2 12) ∧ a (.virt 23800) = a (.wire 2 13) ∧ a (.virt 17809) = a (.wire 2 14) ∧ a (.wire 2 15) = a (.wire 0 68) ∧ a (.virt 23780) = a (.wire 0 69) ∧ a (.wire 0 67) = a (.wire 0 70) ∧ a (.wire 2 11) = a (.virt 23781) ∧ a (.wire 0 71) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 0 72) ∧ a (.virt 23780) = a (.wire 0 73) ∧ a (.virt 23801) = a (.wire 0 74) ∧ a (.virt 23801) = a (.wire 2 16) ∧ a (.virt 17810) = a (.wire 2 17) ∧ a (.virt 23801) = a (.wire 2 18) ∧ a (.virt 17810) = a (.wire 2 20) ∧ a (.virt 23802) = a (.wire 2 21) ∧ a (.virt 17810) = a (.wire 2 22) ∧ a (.wire 2 23) = a (.wire 0 76) ∧ a (.virt 23780) = a (.wire 0 77) ∧ a (.wire 0 75) = a (.wire 0 78) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies4, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies5 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 19) = a (.virt 23781) ∧ a (.wire 0 79) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 3 0) ∧ a (.virt 23780) = a (.wire 3 1) ∧ a (.virt 23803) = a (.wire 3 2) ∧ a (.virt 23803) = a (.wire 2 24) ∧ a (.virt 17811) = a (.wire 2 25) ∧ a (.virt 23803) = a (.wire 2 26) ∧ a (.virt 17811) = a (.wire 2 28) ∧ a (.virt 23804) = a (.wire 2 29) ∧ a (.virt 17811) = a (.wire 2 30) ∧ a (.wire 2 31) = a (.wire 3 4) ∧ a (.virt 23780) = a (.wire 3 5) ∧ a (.wire 3 3) = a (.wire 3 6) ∧ a (.wire 2 27) = a (.virt 23781) ∧ a (.wire 3 7) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 3 8) ∧ a (.virt 23780) = a (.wire 3 9) ∧ a (.virt 23805) = a (.wire 3 10) ∧ a (.virt 23805) = a (.wire 2 32) ∧ a (.virt 17812) = a (.wire 2 33) ∧ a (.virt 23805) = a (.wire 2 34) ∧ a (.virt 17812) = a (.wire 2 36) ∧ a (.virt 23806) = a (.wire 2 37) ∧ a (.virt 17812) = a (.wire 2 38) ∧ a (.wire 2 39) = a (.wire 3 12) ∧ a (.virt 23780) = a (.wire 3 13) ∧ a (.wire 3 11) = a (.wire 3 14) ∧ a (.wire 2 35) = a (.virt 23781) ∧ a (.wire 3 15) = a (.virt 23781) ∧ a (.virt 23799) = a (.wire 2 40) ∧ a (.virt 23801) = a (.wire 2 41) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies5, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies6 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23799) = a (.wire 2 42) ∧ a (.virt 23803) = a (.wire 2 44) ∧ a (.virt 23805) = a (.wire 2 45) ∧ a (.virt 23803) = a (.wire 2 46) ∧ a (.wire 2 43) = a (.wire 2 48) ∧ a (.wire 2 47) = a (.wire 2 49) ∧ a (.wire 2 43) = a (.wire 2 50) ∧ a (.virt 23780) = a (.wire 3 16) ∧ a (.virt 23780) = a (.wire 3 17) ∧ a (.virt 23807) = a (.wire 3 18) ∧ a (.virt 23807) = a (.wire 2 52) ∧ a (.virt 23753) = a (.wire 2 53) ∧ a (.virt 23807) = a (.wire 2 54) ∧ a (.virt 23753) = a (.wire 2 56) ∧ a (.virt 23808) = a (.wire 2 57) ∧ a (.virt 23753) = a (.wire 2 58) ∧ a (.wire 2 59) = a (.wire 3 20) ∧ a (.virt 23780) = a (.wire 3 21) ∧ a (.wire 3 19) = a (.wire 3 22) ∧ a (.wire 2 55) = a (.virt 23781) ∧ a (.wire 3 23) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 3 24) ∧ a (.virt 23780) = a (.wire 3 25) ∧ a (.virt 23809) = a (.wire 3 26) ∧ a (.virt 23809) = a (.wire 2 60) ∧ a (.virt 23754) = a (.wire 2 61) ∧ a (.virt 23809) = a (.wire 2 62) ∧ a (.virt 23754) = a (.wire 2 64) ∧ a (.virt 23810) = a (.wire 2 65) ∧ a (.virt 23754) = a (.wire 2 66) ∧ a (.wire 2 67) = a (.wire 3 28) ∧ a (.virt 23780) = a (.wire 3 29) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies6, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies7 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 27) = a (.wire 3 30) ∧ a (.wire 2 63) = a (.virt 23781) ∧ a (.wire 3 31) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 3 32) ∧ a (.virt 23780) = a (.wire 3 33) ∧ a (.virt 23811) = a (.wire 3 34) ∧ a (.virt 23811) = a (.wire 2 68) ∧ a (.virt 23755) = a (.wire 2 69) ∧ a (.virt 23811) = a (.wire 2 70) ∧ a (.virt 23755) = a (.wire 2 72) ∧ a (.virt 23812) = a (.wire 2 73) ∧ a (.virt 23755) = a (.wire 2 74) ∧ a (.wire 2 75) = a (.wire 3 36) ∧ a (.virt 23780) = a (.wire 3 37) ∧ a (.wire 3 35) = a (.wire 3 38) ∧ a (.wire 2 71) = a (.virt 23781) ∧ a (.wire 3 39) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 3 40) ∧ a (.virt 23780) = a (.wire 3 41) ∧ a (.virt 23813) = a (.wire 3 42) ∧ a (.virt 23813) = a (.wire 2 76) ∧ a (.virt 23756) = a (.wire 2 77) ∧ a (.virt 23813) = a (.wire 2 78) ∧ a (.virt 23756) = a (.wire 4 0) ∧ a (.virt 23814) = a (.wire 4 1) ∧ a (.virt 23756) = a (.wire 4 2) ∧ a (.wire 4 3) = a (.wire 3 44) ∧ a (.virt 23780) = a (.wire 3 45) ∧ a (.wire 3 43) = a (.wire 3 46) ∧ a (.wire 2 79) = a (.virt 23781) ∧ a (.wire 3 47) = a (.virt 23781) ∧ a (.virt 23807) = a (.wire 4 4) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies7, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies8 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23809) = a (.wire 4 5) ∧ a (.virt 23807) = a (.wire 4 6) ∧ a (.virt 23811) = a (.wire 4 8) ∧ a (.virt 23813) = a (.wire 4 9) ∧ a (.virt 23811) = a (.wire 4 10) ∧ a (.wire 4 7) = a (.wire 4 12) ∧ a (.wire 4 11) = a (.wire 4 13) ∧ a (.wire 4 7) = a (.wire 4 14) ∧ a (.virt 23780) = a (.wire 3 48) ∧ a (.virt 23780) = a (.wire 3 49) ∧ a (.wire 1 43) = a (.wire 3 50) ∧ a (.wire 3 51) = a (.wire 3 52) ∧ a (.virt 5921) = a (.wire 3 53) ∧ a (.virt 23781) = a (.wire 3 54) ∧ a (.wire 3 51) = a (.wire 3 56) ∧ a (.virt 5922) = a (.wire 3 57) ∧ a (.virt 23781) = a (.wire 3 58) ∧ a (.wire 3 51) = a (.wire 3 60) ∧ a (.virt 5923) = a (.wire 3 61) ∧ a (.virt 23781) = a (.wire 3 62) ∧ a (.wire 3 51) = a (.wire 3 64) ∧ a (.virt 5924) = a (.wire 3 65) ∧ a (.virt 23781) = a (.wire 3 66) ∧ a (.wire 3 51) = a (.wire 3 68) ∧ a (.virt 5925) = a (.wire 3 69) ∧ a (.virt 23781) = a (.wire 3 70) ∧ a (.wire 3 51) = a (.wire 3 72) ∧ a (.virt 5919) = a (.wire 3 73) ∧ a (.virt 23781) = a (.wire 3 74) ∧ a (.wire 3 51) = a (.wire 3 76) ∧ a (.virt 5920) = a (.wire 3 77) ∧ a (.virt 23781) = a (.wire 3 78) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies8, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies9 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = a (.wire 5 0) ∧ a (.virt 23780) = a (.wire 5 1) ∧ a (.wire 2 7) = a (.wire 5 2) ∧ a (.virt 23780) = a (.wire 5 4) ∧ a (.virt 23780) = a (.wire 5 5) ∧ a (.wire 3 51) = a (.wire 5 6) ∧ a (.wire 5 3) = a (.wire 4 16) ∧ a (.wire 5 7) = a (.wire 4 17) ∧ a (.wire 5 3) = a (.wire 4 18) ∧ a (.wire 4 19) = a (.wire 5 8) ∧ a (.wire 3 55) = a (.wire 5 9) ∧ a (.wire 3 55) = a (.wire 5 10) ∧ a (.wire 4 19) = a (.wire 5 12) ∧ a (.virt 11865) = a (.wire 5 13) ∧ a (.wire 5 11) = a (.wire 5 14) ∧ a (.wire 4 19) = a (.wire 5 16) ∧ a (.wire 3 59) = a (.wire 5 17) ∧ a (.wire 3 59) = a (.wire 5 18) ∧ a (.wire 4 19) = a (.wire 5 20) ∧ a (.virt 11866) = a (.wire 5 21) ∧ a (.wire 5 19) = a (.wire 5 22) ∧ a (.wire 4 19) = a (.wire 5 24) ∧ a (.wire 3 63) = a (.wire 5 25) ∧ a (.wire 3 63) = a (.wire 5 26) ∧ a (.wire 4 19) = a (.wire 5 28) ∧ a (.virt 11867) = a (.wire 5 29) ∧ a (.wire 5 27) = a (.wire 5 30) ∧ a (.wire 4 19) = a (.wire 5 32) ∧ a (.wire 3 67) = a (.wire 5 33) ∧ a (.wire 3 67) = a (.wire 5 34) ∧ a (.wire 4 19) = a (.wire 5 36) ∧ a (.virt 11868) = a (.wire 5 37) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies9, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies10 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 35) = a (.wire 5 38) ∧ a (.wire 4 19) = a (.wire 5 40) ∧ a (.wire 3 71) = a (.wire 5 41) ∧ a (.wire 3 71) = a (.wire 5 42) ∧ a (.wire 4 19) = a (.wire 5 44) ∧ a (.virt 11869) = a (.wire 5 45) ∧ a (.wire 5 43) = a (.wire 5 46) ∧ a (.wire 4 19) = a (.wire 5 48) ∧ a (.wire 3 75) = a (.wire 5 49) ∧ a (.wire 3 75) = a (.wire 5 50) ∧ a (.wire 4 19) = a (.wire 5 52) ∧ a (.virt 11863) = a (.wire 5 53) ∧ a (.wire 5 51) = a (.wire 5 54) ∧ a (.wire 4 19) = a (.wire 5 56) ∧ a (.wire 3 79) = a (.wire 5 57) ∧ a (.wire 3 79) = a (.wire 5 58) ∧ a (.wire 4 19) = a (.wire 5 60) ∧ a (.virt 11864) = a (.wire 5 61) ∧ a (.wire 5 59) = a (.wire 5 62) ∧ a (.wire 3 51) = a (.wire 6 0) ∧ a (.wire 5 3) = a (.wire 6 1) ∧ a (.wire 3 51) = a (.wire 6 2) ∧ a (.wire 6 3) = a (.wire 7 0) ∧ a (.virt 23780) = a (.wire 7 1) ∧ a (.wire 5 3) = a (.wire 7 2) ∧ a (.virt 23780) = a (.wire 5 64) ∧ a (.virt 23780) = a (.wire 5 65) ∧ a (.wire 2 51) = a (.wire 5 66) ∧ a (.virt 23780) = a (.wire 5 68) ∧ a (.virt 23780) = a (.wire 5 69) ∧ a (.wire 7 3) = a (.wire 5 70) ∧ a (.wire 5 67) = a (.wire 4 20) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies10, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies11 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 71) = a (.wire 4 21) ∧ a (.wire 5 67) = a (.wire 4 22) ∧ a (.wire 4 23) = a (.wire 5 72) ∧ a (.wire 5 15) = a (.wire 5 73) ∧ a (.wire 5 15) = a (.wire 5 74) ∧ a (.wire 4 23) = a (.wire 5 76) ∧ a (.virt 17809) = a (.wire 5 77) ∧ a (.wire 5 75) = a (.wire 5 78) ∧ a (.wire 4 23) = a (.wire 8 0) ∧ a (.wire 5 23) = a (.wire 8 1) ∧ a (.wire 5 23) = a (.wire 8 2) ∧ a (.wire 4 23) = a (.wire 8 4) ∧ a (.virt 17810) = a (.wire 8 5) ∧ a (.wire 8 3) = a (.wire 8 6) ∧ a (.wire 4 23) = a (.wire 8 8) ∧ a (.wire 5 31) = a (.wire 8 9) ∧ a (.wire 5 31) = a (.wire 8 10) ∧ a (.wire 4 23) = a (.wire 8 12) ∧ a (.virt 17811) = a (.wire 8 13) ∧ a (.wire 8 11) = a (.wire 8 14) ∧ a (.wire 4 23) = a (.wire 8 16) ∧ a (.wire 5 39) = a (.wire 8 17) ∧ a (.wire 5 39) = a (.wire 8 18) ∧ a (.wire 4 23) = a (.wire 8 20) ∧ a (.virt 17812) = a (.wire 8 21) ∧ a (.wire 8 19) = a (.wire 8 22) ∧ a (.wire 4 23) = a (.wire 8 24) ∧ a (.wire 5 47) = a (.wire 8 25) ∧ a (.wire 5 47) = a (.wire 8 26) ∧ a (.wire 4 23) = a (.wire 8 28) ∧ a (.virt 17813) = a (.wire 8 29) ∧ a (.wire 8 27) = a (.wire 8 30) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies11, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies12 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 23) = a (.wire 8 32) ∧ a (.wire 5 55) = a (.wire 8 33) ∧ a (.wire 5 55) = a (.wire 8 34) ∧ a (.wire 4 23) = a (.wire 8 36) ∧ a (.virt 17807) = a (.wire 8 37) ∧ a (.wire 8 35) = a (.wire 8 38) ∧ a (.wire 4 23) = a (.wire 8 40) ∧ a (.wire 5 63) = a (.wire 8 41) ∧ a (.wire 5 63) = a (.wire 8 42) ∧ a (.wire 4 23) = a (.wire 8 44) ∧ a (.virt 17808) = a (.wire 8 45) ∧ a (.wire 8 43) = a (.wire 8 46) ∧ a (.wire 7 3) = a (.wire 6 4) ∧ a (.wire 5 67) = a (.wire 6 5) ∧ a (.wire 7 3) = a (.wire 6 6) ∧ a (.wire 6 7) = a (.wire 7 4) ∧ a (.virt 23780) = a (.wire 7 5) ∧ a (.wire 5 67) = a (.wire 7 6) ∧ a (.virt 23780) = a (.wire 8 48) ∧ a (.virt 23780) = a (.wire 8 49) ∧ a (.wire 4 15) = a (.wire 8 50) ∧ a (.virt 23780) = a (.wire 8 52) ∧ a (.virt 23780) = a (.wire 8 53) ∧ a (.wire 7 7) = a (.wire 8 54) ∧ a (.wire 8 51) = a (.wire 4 24) ∧ a (.wire 8 55) = a (.wire 4 25) ∧ a (.wire 8 51) = a (.wire 4 26) ∧ a (.wire 4 27) = a (.wire 8 56) ∧ a (.wire 5 79) = a (.wire 8 57) ∧ a (.wire 5 79) = a (.wire 8 58) ∧ a (.wire 4 27) = a (.wire 8 60) ∧ a (.virt 23753) = a (.wire 8 61) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies12, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper4.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies13 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 59) = a (.wire 8 62) ∧ a (.wire 4 27) = a (.wire 8 64) ∧ a (.wire 8 7) = a (.wire 8 65) ∧ a (.wire 8 7) = a (.wire 8 66) ∧ a (.wire 4 27) = a (.wire 8 68) ∧ a (.virt 23754) = a (.wire 8 69) ∧ a (.wire 8 67) = a (.wire 8 70) ∧ a (.wire 4 27) = a (.wire 8 72) ∧ a (.wire 8 15) = a (.wire 8 73) ∧ a (.wire 8 15) = a (.wire 8 74) ∧ a (.wire 4 27) = a (.wire 8 76) ∧ a (.virt 23755) = a (.wire 8 77) ∧ a (.wire 8 75) = a (.wire 8 78) ∧ a (.wire 4 27) = a (.wire 9 0) ∧ a (.wire 8 23) = a (.wire 9 1) ∧ a (.wire 8 23) = a (.wire 9 2) ∧ a (.wire 4 27) = a (.wire 9 4) ∧ a (.virt 23756) = a (.wire 9 5) ∧ a (.wire 9 3) = a (.wire 9 6) ∧ a (.wire 4 27) = a (.wire 9 8) ∧ a (.wire 8 31) = a (.wire 9 9) ∧ a (.wire 8 31) = a (.wire 9 10) ∧ a (.wire 4 27) = a (.wire 9 12) ∧ a (.virt 23757) = a (.wire 9 13) ∧ a (.wire 9 11) = a (.wire 9 14) ∧ a (.wire 4 27) = a (.wire 9 16) ∧ a (.wire 8 39) = a (.wire 9 17) ∧ a (.wire 8 39) = a (.wire 9 18) ∧ a (.wire 4 27) = a (.wire 9 20) ∧ a (.virt 23751) = a (.wire 9 21) ∧ a (.wire 9 19) = a (.wire 9 22) ∧ a (.wire 4 27) = a (.wire 9 24) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies13, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies14 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 47) = a (.wire 9 25) ∧ a (.wire 8 47) = a (.wire 9 26) ∧ a (.wire 4 27) = a (.wire 9 28) ∧ a (.virt 23752) = a (.wire 9 29) ∧ a (.wire 9 27) = a (.wire 9 30) ∧ a (.wire 7 7) = a (.wire 6 8) ∧ a (.wire 8 51) = a (.wire 6 9) ∧ a (.wire 7 7) = a (.wire 6 10) ∧ a (.wire 6 11) = a (.wire 7 8) ∧ a (.virt 23780) = a (.wire 7 9) ∧ a (.wire 8 51) = a (.wire 7 10) ∧ a (.virt 23780) = a (.wire 9 32) ∧ a (.virt 23780) = a (.wire 9 33) ∧ a (.virt 23815) = a (.wire 9 34) ∧ a (.virt 5919) = a (.wire 9 36) ∧ a (.virt 23780) = a (.wire 9 37) ∧ a (.wire 9 23) = a (.wire 9 38) ∧ a (.virt 23815) = a (.wire 4 28) ∧ a (.wire 9 39) = a (.wire 4 29) ∧ a (.virt 23815) = a (.wire 4 30) ∧ a (.wire 9 39) = a (.wire 4 32) ∧ a (.virt 23816) = a (.wire 4 33) ∧ a (.wire 9 39) = a (.wire 4 34) ∧ a (.wire 4 35) = a (.wire 9 40) ∧ a (.virt 23780) = a (.wire 9 41) ∧ a (.wire 9 35) = a (.wire 9 42) ∧ a (.wire 4 31) = a (.virt 23781) ∧ a (.wire 9 43) = a (.virt 23781) ∧ a (.wire 1 43) = a (.wire 6 12) ∧ a (.virt 23815) = a (.wire 6 13) ∧ a (.wire 1 43) = a (.wire 6 14) ∧ a (.wire 6 15) = a (.wire 7 12) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies14, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies15 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = a (.wire 7 13) ∧ a (.virt 23815) = a (.wire 7 14) ∧ a (.wire 7 15) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 9 44) ∧ a (.virt 23780) = a (.wire 9 45) ∧ a (.virt 23817) = a (.wire 9 46) ∧ a (.virt 5920) = a (.wire 9 48) ∧ a (.virt 23780) = a (.wire 9 49) ∧ a (.wire 9 31) = a (.wire 9 50) ∧ a (.virt 23817) = a (.wire 4 36) ∧ a (.wire 9 51) = a (.wire 4 37) ∧ a (.virt 23817) = a (.wire 4 38) ∧ a (.wire 9 51) = a (.wire 4 40) ∧ a (.virt 23818) = a (.wire 4 41) ∧ a (.wire 9 51) = a (.wire 4 42) ∧ a (.wire 4 43) = a (.wire 9 52) ∧ a (.virt 23780) = a (.wire 9 53) ∧ a (.wire 9 47) = a (.wire 9 54) ∧ a (.wire 4 39) = a (.virt 23781) ∧ a (.wire 9 55) = a (.virt 23781) ∧ a (.wire 1 43) = a (.wire 6 16) ∧ a (.virt 23817) = a (.wire 6 17) ∧ a (.wire 1 43) = a (.wire 6 18) ∧ a (.wire 6 19) = a (.wire 7 16) ∧ a (.virt 23780) = a (.wire 7 17) ∧ a (.virt 23817) = a (.wire 7 18) ∧ a (.wire 7 19) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 9 56) ∧ a (.virt 23780) = a (.wire 9 57) ∧ a (.virt 23819) = a (.wire 9 58) ∧ a (.virt 5921) = a (.wire 9 60) ∧ a (.virt 23780) = a (.wire 9 61) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies15, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies16 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 63) = a (.wire 9 62) ∧ a (.virt 23819) = a (.wire 4 44) ∧ a (.wire 9 63) = a (.wire 4 45) ∧ a (.virt 23819) = a (.wire 4 46) ∧ a (.wire 9 63) = a (.wire 4 48) ∧ a (.virt 23820) = a (.wire 4 49) ∧ a (.wire 9 63) = a (.wire 4 50) ∧ a (.wire 4 51) = a (.wire 9 64) ∧ a (.virt 23780) = a (.wire 9 65) ∧ a (.wire 9 59) = a (.wire 9 66) ∧ a (.wire 4 47) = a (.virt 23781) ∧ a (.wire 9 67) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 9 68) ∧ a (.virt 23780) = a (.wire 9 69) ∧ a (.virt 23821) = a (.wire 9 70) ∧ a (.virt 5922) = a (.wire 9 72) ∧ a (.virt 23780) = a (.wire 9 73) ∧ a (.wire 8 71) = a (.wire 9 74) ∧ a (.virt 23821) = a (.wire 4 52) ∧ a (.wire 9 75) = a (.wire 4 53) ∧ a (.virt 23821) = a (.wire 4 54) ∧ a (.wire 9 75) = a (.wire 4 56) ∧ a (.virt 23822) = a (.wire 4 57) ∧ a (.wire 9 75) = a (.wire 4 58) ∧ a (.wire 4 59) = a (.wire 9 76) ∧ a (.virt 23780) = a (.wire 9 77) ∧ a (.wire 9 71) = a (.wire 9 78) ∧ a (.wire 4 55) = a (.virt 23781) ∧ a (.wire 9 79) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 10 0) ∧ a (.virt 23780) = a (.wire 10 1) ∧ a (.virt 23823) = a (.wire 10 2) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies16, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies17 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 5923) = a (.wire 10 4) ∧ a (.virt 23780) = a (.wire 10 5) ∧ a (.wire 8 79) = a (.wire 10 6) ∧ a (.virt 23823) = a (.wire 4 60) ∧ a (.wire 10 7) = a (.wire 4 61) ∧ a (.virt 23823) = a (.wire 4 62) ∧ a (.wire 10 7) = a (.wire 4 64) ∧ a (.virt 23824) = a (.wire 4 65) ∧ a (.wire 10 7) = a (.wire 4 66) ∧ a (.wire 4 67) = a (.wire 10 8) ∧ a (.virt 23780) = a (.wire 10 9) ∧ a (.wire 10 3) = a (.wire 10 10) ∧ a (.wire 4 63) = a (.virt 23781) ∧ a (.wire 10 11) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 10 12) ∧ a (.virt 23780) = a (.wire 10 13) ∧ a (.virt 23825) = a (.wire 10 14) ∧ a (.virt 5924) = a (.wire 10 16) ∧ a (.virt 23780) = a (.wire 10 17) ∧ a (.wire 9 7) = a (.wire 10 18) ∧ a (.virt 23825) = a (.wire 4 68) ∧ a (.wire 10 19) = a (.wire 4 69) ∧ a (.virt 23825) = a (.wire 4 70) ∧ a (.wire 10 19) = a (.wire 4 72) ∧ a (.virt 23826) = a (.wire 4 73) ∧ a (.wire 10 19) = a (.wire 4 74) ∧ a (.wire 4 75) = a (.wire 10 20) ∧ a (.virt 23780) = a (.wire 10 21) ∧ a (.wire 10 15) = a (.wire 10 22) ∧ a (.wire 4 71) = a (.virt 23781) ∧ a (.wire 10 23) = a (.virt 23781) ∧ a (.virt 23819) = a (.wire 4 76) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies17, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies18 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23821) = a (.wire 4 77) ∧ a (.virt 23819) = a (.wire 4 78) ∧ a (.virt 23823) = a (.wire 11 0) ∧ a (.virt 23825) = a (.wire 11 1) ∧ a (.virt 23823) = a (.wire 11 2) ∧ a (.wire 4 79) = a (.wire 11 4) ∧ a (.wire 11 3) = a (.wire 11 5) ∧ a (.wire 4 79) = a (.wire 11 6) ∧ a (.wire 1 43) = a (.wire 6 20) ∧ a (.wire 11 7) = a (.wire 6 21) ∧ a (.wire 1 43) = a (.wire 6 22) ∧ a (.wire 6 23) = a (.wire 7 20) ∧ a (.virt 23780) = a (.wire 7 21) ∧ a (.wire 11 7) = a (.wire 7 22) ∧ a (.wire 7 23) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 10 24) ∧ a (.virt 23780) = a (.wire 10 25) ∧ a (.virt 23827) = a (.wire 10 26) ∧ a (.virt 11863) = a (.wire 10 28) ∧ a (.virt 23780) = a (.wire 10 29) ∧ a (.wire 9 23) = a (.wire 10 30) ∧ a (.virt 23827) = a (.wire 11 8) ∧ a (.wire 10 31) = a (.wire 11 9) ∧ a (.virt 23827) = a (.wire 11 10) ∧ a (.wire 10 31) = a (.wire 11 12) ∧ a (.virt 23828) = a (.wire 11 13) ∧ a (.wire 10 31) = a (.wire 11 14) ∧ a (.wire 11 15) = a (.wire 10 32) ∧ a (.virt 23780) = a (.wire 10 33) ∧ a (.wire 10 27) = a (.wire 10 34) ∧ a (.wire 11 11) = a (.virt 23781) ∧ a (.wire 10 35) = a (.virt 23781) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies18, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies19 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 7) = a (.wire 6 24) ∧ a (.virt 23827) = a (.wire 6 25) ∧ a (.wire 2 7) = a (.wire 6 26) ∧ a (.wire 6 27) = a (.wire 7 24) ∧ a (.virt 23780) = a (.wire 7 25) ∧ a (.virt 23827) = a (.wire 7 26) ∧ a (.wire 7 27) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 10 36) ∧ a (.virt 23780) = a (.wire 10 37) ∧ a (.virt 23829) = a (.wire 10 38) ∧ a (.virt 11864) = a (.wire 10 40) ∧ a (.virt 23780) = a (.wire 10 41) ∧ a (.wire 9 31) = a (.wire 10 42) ∧ a (.virt 23829) = a (.wire 11 16) ∧ a (.wire 10 43) = a (.wire 11 17) ∧ a (.virt 23829) = a (.wire 11 18) ∧ a (.wire 10 43) = a (.wire 11 20) ∧ a (.virt 23830) = a (.wire 11 21) ∧ a (.wire 10 43) = a (.wire 11 22) ∧ a (.wire 11 23) = a (.wire 10 44) ∧ a (.virt 23780) = a (.wire 10 45) ∧ a (.wire 10 39) = a (.wire 10 46) ∧ a (.wire 11 19) = a (.virt 23781) ∧ a (.wire 10 47) = a (.virt 23781) ∧ a (.wire 2 7) = a (.wire 6 28) ∧ a (.virt 23829) = a (.wire 6 29) ∧ a (.wire 2 7) = a (.wire 6 30) ∧ a (.wire 6 31) = a (.wire 7 28) ∧ a (.virt 23780) = a (.wire 7 29) ∧ a (.virt 23829) = a (.wire 7 30) ∧ a (.wire 7 31) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 10 48) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies19, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies20 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = a (.wire 10 49) ∧ a (.virt 23831) = a (.wire 10 50) ∧ a (.virt 11865) = a (.wire 10 52) ∧ a (.virt 23780) = a (.wire 10 53) ∧ a (.wire 8 63) = a (.wire 10 54) ∧ a (.virt 23831) = a (.wire 11 24) ∧ a (.wire 10 55) = a (.wire 11 25) ∧ a (.virt 23831) = a (.wire 11 26) ∧ a (.wire 10 55) = a (.wire 11 28) ∧ a (.virt 23832) = a (.wire 11 29) ∧ a (.wire 10 55) = a (.wire 11 30) ∧ a (.wire 11 31) = a (.wire 10 56) ∧ a (.virt 23780) = a (.wire 10 57) ∧ a (.wire 10 51) = a (.wire 10 58) ∧ a (.wire 11 27) = a (.virt 23781) ∧ a (.wire 10 59) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 10 60) ∧ a (.virt 23780) = a (.wire 10 61) ∧ a (.virt 23833) = a (.wire 10 62) ∧ a (.virt 11866) = a (.wire 10 64) ∧ a (.virt 23780) = a (.wire 10 65) ∧ a (.wire 8 71) = a (.wire 10 66) ∧ a (.virt 23833) = a (.wire 11 32) ∧ a (.wire 10 67) = a (.wire 11 33) ∧ a (.virt 23833) = a (.wire 11 34) ∧ a (.wire 10 67) = a (.wire 11 36) ∧ a (.virt 23834) = a (.wire 11 37) ∧ a (.wire 10 67) = a (.wire 11 38) ∧ a (.wire 11 39) = a (.wire 10 68) ∧ a (.virt 23780) = a (.wire 10 69) ∧ a (.wire 10 63) = a (.wire 10 70) ∧ a (.wire 11 35) = a (.virt 23781) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies20, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies21 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 10 71) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 10 72) ∧ a (.virt 23780) = a (.wire 10 73) ∧ a (.virt 23835) = a (.wire 10 74) ∧ a (.virt 11867) = a (.wire 10 76) ∧ a (.virt 23780) = a (.wire 10 77) ∧ a (.wire 8 79) = a (.wire 10 78) ∧ a (.virt 23835) = a (.wire 11 40) ∧ a (.wire 10 79) = a (.wire 11 41) ∧ a (.virt 23835) = a (.wire 11 42) ∧ a (.wire 10 79) = a (.wire 11 44) ∧ a (.virt 23836) = a (.wire 11 45) ∧ a (.wire 10 79) = a (.wire 11 46) ∧ a (.wire 11 47) = a (.wire 12 0) ∧ a (.virt 23780) = a (.wire 12 1) ∧ a (.wire 10 75) = a (.wire 12 2) ∧ a (.wire 11 43) = a (.virt 23781) ∧ a (.wire 12 3) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 12 4) ∧ a (.virt 23780) = a (.wire 12 5) ∧ a (.virt 23837) = a (.wire 12 6) ∧ a (.virt 11868) = a (.wire 12 8) ∧ a (.virt 23780) = a (.wire 12 9) ∧ a (.wire 9 7) = a (.wire 12 10) ∧ a (.virt 23837) = a (.wire 11 48) ∧ a (.wire 12 11) = a (.wire 11 49) ∧ a (.virt 23837) = a (.wire 11 50) ∧ a (.wire 12 11) = a (.wire 11 52) ∧ a (.virt 23838) = a (.wire 11 53) ∧ a (.wire 12 11) = a (.wire 11 54) ∧ a (.wire 11 55) = a (.wire 12 12) ∧ a (.virt 23780) = a (.wire 12 13) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies21, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies22 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 12 7) = a (.wire 12 14) ∧ a (.wire 11 51) = a (.virt 23781) ∧ a (.wire 12 15) = a (.virt 23781) ∧ a (.virt 23831) = a (.wire 11 56) ∧ a (.virt 23833) = a (.wire 11 57) ∧ a (.virt 23831) = a (.wire 11 58) ∧ a (.virt 23835) = a (.wire 11 60) ∧ a (.virt 23837) = a (.wire 11 61) ∧ a (.virt 23835) = a (.wire 11 62) ∧ a (.wire 11 59) = a (.wire 11 64) ∧ a (.wire 11 63) = a (.wire 11 65) ∧ a (.wire 11 59) = a (.wire 11 66) ∧ a (.wire 2 7) = a (.wire 6 32) ∧ a (.wire 11 67) = a (.wire 6 33) ∧ a (.wire 2 7) = a (.wire 6 34) ∧ a (.wire 6 35) = a (.wire 7 32) ∧ a (.virt 23780) = a (.wire 7 33) ∧ a (.wire 11 67) = a (.wire 7 34) ∧ a (.wire 7 35) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 12 16) ∧ a (.virt 23780) = a (.wire 12 17) ∧ a (.virt 23839) = a (.wire 12 18) ∧ a (.virt 17807) = a (.wire 12 20) ∧ a (.virt 23780) = a (.wire 12 21) ∧ a (.wire 9 23) = a (.wire 12 22) ∧ a (.virt 23839) = a (.wire 11 68) ∧ a (.wire 12 23) = a (.wire 11 69) ∧ a (.virt 23839) = a (.wire 11 70) ∧ a (.wire 12 23) = a (.wire 11 72) ∧ a (.virt 23840) = a (.wire 11 73) ∧ a (.wire 12 23) = a (.wire 11 74) ∧ a (.wire 11 75) = a (.wire 12 24) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies22, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies23 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = a (.wire 12 25) ∧ a (.wire 12 19) = a (.wire 12 26) ∧ a (.wire 11 71) = a (.virt 23781) ∧ a (.wire 12 27) = a (.virt 23781) ∧ a (.wire 2 51) = a (.wire 6 36) ∧ a (.virt 23839) = a (.wire 6 37) ∧ a (.wire 2 51) = a (.wire 6 38) ∧ a (.wire 6 39) = a (.wire 7 36) ∧ a (.virt 23780) = a (.wire 7 37) ∧ a (.virt 23839) = a (.wire 7 38) ∧ a (.wire 7 39) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 12 28) ∧ a (.virt 23780) = a (.wire 12 29) ∧ a (.virt 23841) = a (.wire 12 30) ∧ a (.virt 17808) = a (.wire 12 32) ∧ a (.virt 23780) = a (.wire 12 33) ∧ a (.wire 9 31) = a (.wire 12 34) ∧ a (.virt 23841) = a (.wire 11 76) ∧ a (.wire 12 35) = a (.wire 11 77) ∧ a (.virt 23841) = a (.wire 11 78) ∧ a (.wire 12 35) = a (.wire 13 0) ∧ a (.virt 23842) = a (.wire 13 1) ∧ a (.wire 12 35) = a (.wire 13 2) ∧ a (.wire 13 3) = a (.wire 12 36) ∧ a (.virt 23780) = a (.wire 12 37) ∧ a (.wire 12 31) = a (.wire 12 38) ∧ a (.wire 11 79) = a (.virt 23781) ∧ a (.wire 12 39) = a (.virt 23781) ∧ a (.wire 2 51) = a (.wire 6 40) ∧ a (.virt 23841) = a (.wire 6 41) ∧ a (.wire 2 51) = a (.wire 6 42) ∧ a (.wire 6 43) = a (.wire 7 40) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies23, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq)))))
  simp only [publicBatchWrapper4.copies23, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies24 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = a (.wire 7 41) ∧ a (.virt 23841) = a (.wire 7 42) ∧ a (.wire 7 43) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 12 40) ∧ a (.virt 23780) = a (.wire 12 41) ∧ a (.virt 23843) = a (.wire 12 42) ∧ a (.virt 17809) = a (.wire 12 44) ∧ a (.virt 23780) = a (.wire 12 45) ∧ a (.wire 8 63) = a (.wire 12 46) ∧ a (.virt 23843) = a (.wire 13 4) ∧ a (.wire 12 47) = a (.wire 13 5) ∧ a (.virt 23843) = a (.wire 13 6) ∧ a (.wire 12 47) = a (.wire 13 8) ∧ a (.virt 23844) = a (.wire 13 9) ∧ a (.wire 12 47) = a (.wire 13 10) ∧ a (.wire 13 11) = a (.wire 12 48) ∧ a (.virt 23780) = a (.wire 12 49) ∧ a (.wire 12 43) = a (.wire 12 50) ∧ a (.wire 13 7) = a (.virt 23781) ∧ a (.wire 12 51) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 12 52) ∧ a (.virt 23780) = a (.wire 12 53) ∧ a (.virt 23845) = a (.wire 12 54) ∧ a (.virt 17810) = a (.wire 12 56) ∧ a (.virt 23780) = a (.wire 12 57) ∧ a (.wire 8 71) = a (.wire 12 58) ∧ a (.virt 23845) = a (.wire 13 12) ∧ a (.wire 12 59) = a (.wire 13 13) ∧ a (.virt 23845) = a (.wire 13 14) ∧ a (.wire 12 59) = a (.wire 13 16) ∧ a (.virt 23846) = a (.wire 13 17) ∧ a (.wire 12 59) = a (.wire 13 18) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies24, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies24, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies25 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 13 19) = a (.wire 12 60) ∧ a (.virt 23780) = a (.wire 12 61) ∧ a (.wire 12 55) = a (.wire 12 62) ∧ a (.wire 13 15) = a (.virt 23781) ∧ a (.wire 12 63) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 12 64) ∧ a (.virt 23780) = a (.wire 12 65) ∧ a (.virt 23847) = a (.wire 12 66) ∧ a (.virt 17811) = a (.wire 12 68) ∧ a (.virt 23780) = a (.wire 12 69) ∧ a (.wire 8 79) = a (.wire 12 70) ∧ a (.virt 23847) = a (.wire 13 20) ∧ a (.wire 12 71) = a (.wire 13 21) ∧ a (.virt 23847) = a (.wire 13 22) ∧ a (.wire 12 71) = a (.wire 13 24) ∧ a (.virt 23848) = a (.wire 13 25) ∧ a (.wire 12 71) = a (.wire 13 26) ∧ a (.wire 13 27) = a (.wire 12 72) ∧ a (.virt 23780) = a (.wire 12 73) ∧ a (.wire 12 67) = a (.wire 12 74) ∧ a (.wire 13 23) = a (.virt 23781) ∧ a (.wire 12 75) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 12 76) ∧ a (.virt 23780) = a (.wire 12 77) ∧ a (.virt 23849) = a (.wire 12 78) ∧ a (.virt 17812) = a (.wire 14 0) ∧ a (.virt 23780) = a (.wire 14 1) ∧ a (.wire 9 7) = a (.wire 14 2) ∧ a (.virt 23849) = a (.wire 13 28) ∧ a (.wire 14 3) = a (.wire 13 29) ∧ a (.virt 23849) = a (.wire 13 30) ∧ a (.wire 14 3) = a (.wire 13 32) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies25, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies25, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies26 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23850) = a (.wire 13 33) ∧ a (.wire 14 3) = a (.wire 13 34) ∧ a (.wire 13 35) = a (.wire 14 4) ∧ a (.virt 23780) = a (.wire 14 5) ∧ a (.wire 12 79) = a (.wire 14 6) ∧ a (.wire 13 31) = a (.virt 23781) ∧ a (.wire 14 7) = a (.virt 23781) ∧ a (.virt 23843) = a (.wire 13 36) ∧ a (.virt 23845) = a (.wire 13 37) ∧ a (.virt 23843) = a (.wire 13 38) ∧ a (.virt 23847) = a (.wire 13 40) ∧ a (.virt 23849) = a (.wire 13 41) ∧ a (.virt 23847) = a (.wire 13 42) ∧ a (.wire 13 39) = a (.wire 13 44) ∧ a (.wire 13 43) = a (.wire 13 45) ∧ a (.wire 13 39) = a (.wire 13 46) ∧ a (.wire 2 51) = a (.wire 6 44) ∧ a (.wire 13 47) = a (.wire 6 45) ∧ a (.wire 2 51) = a (.wire 6 46) ∧ a (.wire 6 47) = a (.wire 7 44) ∧ a (.virt 23780) = a (.wire 7 45) ∧ a (.wire 13 47) = a (.wire 7 46) ∧ a (.wire 7 47) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 14 8) ∧ a (.virt 23780) = a (.wire 14 9) ∧ a (.virt 23851) = a (.wire 14 10) ∧ a (.virt 23751) = a (.wire 14 12) ∧ a (.virt 23780) = a (.wire 14 13) ∧ a (.wire 9 23) = a (.wire 14 14) ∧ a (.virt 23851) = a (.wire 13 48) ∧ a (.wire 14 15) = a (.wire 13 49) ∧ a (.virt 23851) = a (.wire 13 50) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies26, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies26, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies27 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 14 15) = a (.wire 13 52) ∧ a (.virt 23852) = a (.wire 13 53) ∧ a (.wire 14 15) = a (.wire 13 54) ∧ a (.wire 13 55) = a (.wire 14 16) ∧ a (.virt 23780) = a (.wire 14 17) ∧ a (.wire 14 11) = a (.wire 14 18) ∧ a (.wire 13 51) = a (.virt 23781) ∧ a (.wire 14 19) = a (.virt 23781) ∧ a (.wire 4 15) = a (.wire 6 48) ∧ a (.virt 23851) = a (.wire 6 49) ∧ a (.wire 4 15) = a (.wire 6 50) ∧ a (.wire 6 51) = a (.wire 7 48) ∧ a (.virt 23780) = a (.wire 7 49) ∧ a (.virt 23851) = a (.wire 7 50) ∧ a (.wire 7 51) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 14 20) ∧ a (.virt 23780) = a (.wire 14 21) ∧ a (.virt 23853) = a (.wire 14 22) ∧ a (.virt 23752) = a (.wire 14 24) ∧ a (.virt 23780) = a (.wire 14 25) ∧ a (.wire 9 31) = a (.wire 14 26) ∧ a (.virt 23853) = a (.wire 13 56) ∧ a (.wire 14 27) = a (.wire 13 57) ∧ a (.virt 23853) = a (.wire 13 58) ∧ a (.wire 14 27) = a (.wire 13 60) ∧ a (.virt 23854) = a (.wire 13 61) ∧ a (.wire 14 27) = a (.wire 13 62) ∧ a (.wire 13 63) = a (.wire 14 28) ∧ a (.virt 23780) = a (.wire 14 29) ∧ a (.wire 14 23) = a (.wire 14 30) ∧ a (.wire 13 59) = a (.virt 23781) ∧ a (.wire 14 31) = a (.virt 23781) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies27, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies27, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies28 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 15) = a (.wire 6 52) ∧ a (.virt 23853) = a (.wire 6 53) ∧ a (.wire 4 15) = a (.wire 6 54) ∧ a (.wire 6 55) = a (.wire 7 52) ∧ a (.virt 23780) = a (.wire 7 53) ∧ a (.virt 23853) = a (.wire 7 54) ∧ a (.wire 7 55) = a (.virt 23780) ∧ a (.virt 23780) = a (.wire 14 32) ∧ a (.virt 23780) = a (.wire 14 33) ∧ a (.virt 23855) = a (.wire 14 34) ∧ a (.virt 23753) = a (.wire 14 36) ∧ a (.virt 23780) = a (.wire 14 37) ∧ a (.wire 8 63) = a (.wire 14 38) ∧ a (.virt 23855) = a (.wire 13 64) ∧ a (.wire 14 39) = a (.wire 13 65) ∧ a (.virt 23855) = a (.wire 13 66) ∧ a (.wire 14 39) = a (.wire 13 68) ∧ a (.virt 23856) = a (.wire 13 69) ∧ a (.wire 14 39) = a (.wire 13 70) ∧ a (.wire 13 71) = a (.wire 14 40) ∧ a (.virt 23780) = a (.wire 14 41) ∧ a (.wire 14 35) = a (.wire 14 42) ∧ a (.wire 13 67) = a (.virt 23781) ∧ a (.wire 14 43) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 14 44) ∧ a (.virt 23780) = a (.wire 14 45) ∧ a (.virt 23857) = a (.wire 14 46) ∧ a (.virt 23754) = a (.wire 14 48) ∧ a (.virt 23780) = a (.wire 14 49) ∧ a (.wire 8 71) = a (.wire 14 50) ∧ a (.virt 23857) = a (.wire 13 72) ∧ a (.wire 14 51) = a (.wire 13 73) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies28, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies28, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies29 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23857) = a (.wire 13 74) ∧ a (.wire 14 51) = a (.wire 13 76) ∧ a (.virt 23858) = a (.wire 13 77) ∧ a (.wire 14 51) = a (.wire 13 78) ∧ a (.wire 13 79) = a (.wire 14 52) ∧ a (.virt 23780) = a (.wire 14 53) ∧ a (.wire 14 47) = a (.wire 14 54) ∧ a (.wire 13 75) = a (.virt 23781) ∧ a (.wire 14 55) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 14 56) ∧ a (.virt 23780) = a (.wire 14 57) ∧ a (.virt 23859) = a (.wire 14 58) ∧ a (.virt 23755) = a (.wire 14 60) ∧ a (.virt 23780) = a (.wire 14 61) ∧ a (.wire 8 79) = a (.wire 14 62) ∧ a (.virt 23859) = a (.wire 15 0) ∧ a (.wire 14 63) = a (.wire 15 1) ∧ a (.virt 23859) = a (.wire 15 2) ∧ a (.wire 14 63) = a (.wire 15 4) ∧ a (.virt 23860) = a (.wire 15 5) ∧ a (.wire 14 63) = a (.wire 15 6) ∧ a (.wire 15 7) = a (.wire 14 64) ∧ a (.virt 23780) = a (.wire 14 65) ∧ a (.wire 14 59) = a (.wire 14 66) ∧ a (.wire 15 3) = a (.virt 23781) ∧ a (.wire 14 67) = a (.virt 23781) ∧ a (.virt 23780) = a (.wire 14 68) ∧ a (.virt 23780) = a (.wire 14 69) ∧ a (.virt 23861) = a (.wire 14 70) ∧ a (.virt 23756) = a (.wire 14 72) ∧ a (.virt 23780) = a (.wire 14 73) ∧ a (.wire 9 7) = a (.wire 14 74) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies29, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies29, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies30 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23861) = a (.wire 15 8) ∧ a (.wire 14 75) = a (.wire 15 9) ∧ a (.virt 23861) = a (.wire 15 10) ∧ a (.wire 14 75) = a (.wire 15 12) ∧ a (.virt 23862) = a (.wire 15 13) ∧ a (.wire 14 75) = a (.wire 15 14) ∧ a (.wire 15 15) = a (.wire 14 76) ∧ a (.virt 23780) = a (.wire 14 77) ∧ a (.wire 14 71) = a (.wire 14 78) ∧ a (.wire 15 11) = a (.virt 23781) ∧ a (.wire 14 79) = a (.virt 23781) ∧ a (.virt 23855) = a (.wire 15 16) ∧ a (.virt 23857) = a (.wire 15 17) ∧ a (.virt 23855) = a (.wire 15 18) ∧ a (.virt 23859) = a (.wire 15 20) ∧ a (.virt 23861) = a (.wire 15 21) ∧ a (.virt 23859) = a (.wire 15 22) ∧ a (.wire 15 19) = a (.wire 15 24) ∧ a (.wire 15 23) = a (.wire 15 25) ∧ a (.wire 15 19) = a (.wire 15 26) ∧ a (.wire 4 15) = a (.wire 6 56) ∧ a (.wire 15 27) = a (.wire 6 57) ∧ a (.wire 4 15) = a (.wire 6 58) ∧ a (.wire 6 59) = a (.wire 7 56) ∧ a (.virt 23780) = a (.wire 7 57) ∧ a (.wire 15 27) = a (.wire 7 58) ∧ a (.wire 7 59) = a (.virt 23780) ∧ a (.wire 1 43) = a (.wire 16 0) ∧ a (.virt 5926) = a (.wire 16 1) ∧ a (.virt 5926) = a (.wire 16 2) ∧ a (.wire 1 43) = a (.wire 16 4) ∧ a (.virt 23781) = a (.wire 16 5) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies30, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies30, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies31 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 3) = a (.wire 16 6) ∧ a (.wire 1 43) = a (.wire 16 8) ∧ a (.virt 5927) = a (.wire 16 9) ∧ a (.virt 5927) = a (.wire 16 10) ∧ a (.wire 1 43) = a (.wire 16 12) ∧ a (.virt 23781) = a (.wire 16 13) ∧ a (.wire 16 11) = a (.wire 16 14) ∧ a (.wire 1 43) = a (.wire 16 16) ∧ a (.virt 5928) = a (.wire 16 17) ∧ a (.virt 5928) = a (.wire 16 18) ∧ a (.wire 1 43) = a (.wire 16 20) ∧ a (.virt 23781) = a (.wire 16 21) ∧ a (.wire 16 19) = a (.wire 16 22) ∧ a (.wire 1 43) = a (.wire 16 24) ∧ a (.virt 5929) = a (.wire 16 25) ∧ a (.virt 5929) = a (.wire 16 26) ∧ a (.wire 1 43) = a (.wire 16 28) ∧ a (.virt 23781) = a (.wire 16 29) ∧ a (.wire 16 27) = a (.wire 16 30) ∧ a (.wire 1 43) = a (.wire 16 32) ∧ a (.virt 5930) = a (.wire 16 33) ∧ a (.virt 5930) = a (.wire 16 34) ∧ a (.wire 1 43) = a (.wire 16 36) ∧ a (.virt 23781) = a (.wire 16 37) ∧ a (.wire 16 35) = a (.wire 16 38) ∧ a (.wire 1 43) = a (.wire 16 40) ∧ a (.virt 5931) = a (.wire 16 41) ∧ a (.virt 5931) = a (.wire 16 42) ∧ a (.wire 1 43) = a (.wire 16 44) ∧ a (.virt 23781) = a (.wire 16 45) ∧ a (.wire 16 43) = a (.wire 16 46) ∧ a (.wire 1 43) = a (.wire 16 48) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies31, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies31, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies32 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 5932) = a (.wire 16 49) ∧ a (.virt 5932) = a (.wire 16 50) ∧ a (.wire 1 43) = a (.wire 16 52) ∧ a (.virt 23781) = a (.wire 16 53) ∧ a (.wire 16 51) = a (.wire 16 54) ∧ a (.wire 1 43) = a (.wire 16 56) ∧ a (.virt 5933) = a (.wire 16 57) ∧ a (.virt 5933) = a (.wire 16 58) ∧ a (.wire 1 43) = a (.wire 16 60) ∧ a (.virt 23781) = a (.wire 16 61) ∧ a (.wire 16 59) = a (.wire 16 62) ∧ a (.wire 1 43) = a (.wire 16 64) ∧ a (.virt 5934) = a (.wire 16 65) ∧ a (.virt 5934) = a (.wire 16 66) ∧ a (.wire 1 43) = a (.wire 16 68) ∧ a (.virt 23781) = a (.wire 16 69) ∧ a (.wire 16 67) = a (.wire 16 70) ∧ a (.wire 1 43) = a (.wire 16 72) ∧ a (.virt 5935) = a (.wire 16 73) ∧ a (.virt 5935) = a (.wire 16 74) ∧ a (.wire 1 43) = a (.wire 16 76) ∧ a (.virt 23781) = a (.wire 16 77) ∧ a (.wire 16 75) = a (.wire 16 78) ∧ a (.wire 2 7) = a (.wire 17 0) ∧ a (.virt 11870) = a (.wire 17 1) ∧ a (.virt 11870) = a (.wire 17 2) ∧ a (.wire 2 7) = a (.wire 17 4) ∧ a (.virt 23781) = a (.wire 17 5) ∧ a (.wire 17 3) = a (.wire 17 6) ∧ a (.wire 2 7) = a (.wire 17 8) ∧ a (.virt 11871) = a (.wire 17 9) ∧ a (.virt 11871) = a (.wire 17 10) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies32, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies32, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies33 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 7) = a (.wire 17 12) ∧ a (.virt 23781) = a (.wire 17 13) ∧ a (.wire 17 11) = a (.wire 17 14) ∧ a (.wire 2 7) = a (.wire 17 16) ∧ a (.virt 11872) = a (.wire 17 17) ∧ a (.virt 11872) = a (.wire 17 18) ∧ a (.wire 2 7) = a (.wire 17 20) ∧ a (.virt 23781) = a (.wire 17 21) ∧ a (.wire 17 19) = a (.wire 17 22) ∧ a (.wire 2 7) = a (.wire 17 24) ∧ a (.virt 11873) = a (.wire 17 25) ∧ a (.virt 11873) = a (.wire 17 26) ∧ a (.wire 2 7) = a (.wire 17 28) ∧ a (.virt 23781) = a (.wire 17 29) ∧ a (.wire 17 27) = a (.wire 17 30) ∧ a (.wire 2 7) = a (.wire 17 32) ∧ a (.virt 11874) = a (.wire 17 33) ∧ a (.virt 11874) = a (.wire 17 34) ∧ a (.wire 2 7) = a (.wire 17 36) ∧ a (.virt 23781) = a (.wire 17 37) ∧ a (.wire 17 35) = a (.wire 17 38) ∧ a (.wire 2 7) = a (.wire 17 40) ∧ a (.virt 11875) = a (.wire 17 41) ∧ a (.virt 11875) = a (.wire 17 42) ∧ a (.wire 2 7) = a (.wire 17 44) ∧ a (.virt 23781) = a (.wire 17 45) ∧ a (.wire 17 43) = a (.wire 17 46) ∧ a (.wire 2 7) = a (.wire 17 48) ∧ a (.virt 11876) = a (.wire 17 49) ∧ a (.virt 11876) = a (.wire 17 50) ∧ a (.wire 2 7) = a (.wire 17 52) ∧ a (.virt 23781) = a (.wire 17 53) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies33, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies33, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies34 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 51) = a (.wire 17 54) ∧ a (.wire 2 7) = a (.wire 17 56) ∧ a (.virt 11877) = a (.wire 17 57) ∧ a (.virt 11877) = a (.wire 17 58) ∧ a (.wire 2 7) = a (.wire 17 60) ∧ a (.virt 23781) = a (.wire 17 61) ∧ a (.wire 17 59) = a (.wire 17 62) ∧ a (.wire 2 7) = a (.wire 17 64) ∧ a (.virt 11878) = a (.wire 17 65) ∧ a (.virt 11878) = a (.wire 17 66) ∧ a (.wire 2 7) = a (.wire 17 68) ∧ a (.virt 23781) = a (.wire 17 69) ∧ a (.wire 17 67) = a (.wire 17 70) ∧ a (.wire 2 7) = a (.wire 17 72) ∧ a (.virt 11879) = a (.wire 17 73) ∧ a (.virt 11879) = a (.wire 17 74) ∧ a (.wire 2 7) = a (.wire 17 76) ∧ a (.virt 23781) = a (.wire 17 77) ∧ a (.wire 17 75) = a (.wire 17 78) ∧ a (.wire 2 51) = a (.wire 18 0) ∧ a (.virt 17814) = a (.wire 18 1) ∧ a (.virt 17814) = a (.wire 18 2) ∧ a (.wire 2 51) = a (.wire 18 4) ∧ a (.virt 23781) = a (.wire 18 5) ∧ a (.wire 18 3) = a (.wire 18 6) ∧ a (.wire 2 51) = a (.wire 18 8) ∧ a (.virt 17815) = a (.wire 18 9) ∧ a (.virt 17815) = a (.wire 18 10) ∧ a (.wire 2 51) = a (.wire 18 12) ∧ a (.virt 23781) = a (.wire 18 13) ∧ a (.wire 18 11) = a (.wire 18 14) ∧ a (.wire 2 51) = a (.wire 18 16) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies34, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies34, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies35 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 17816) = a (.wire 18 17) ∧ a (.virt 17816) = a (.wire 18 18) ∧ a (.wire 2 51) = a (.wire 18 20) ∧ a (.virt 23781) = a (.wire 18 21) ∧ a (.wire 18 19) = a (.wire 18 22) ∧ a (.wire 2 51) = a (.wire 18 24) ∧ a (.virt 17817) = a (.wire 18 25) ∧ a (.virt 17817) = a (.wire 18 26) ∧ a (.wire 2 51) = a (.wire 18 28) ∧ a (.virt 23781) = a (.wire 18 29) ∧ a (.wire 18 27) = a (.wire 18 30) ∧ a (.wire 2 51) = a (.wire 18 32) ∧ a (.virt 17818) = a (.wire 18 33) ∧ a (.virt 17818) = a (.wire 18 34) ∧ a (.wire 2 51) = a (.wire 18 36) ∧ a (.virt 23781) = a (.wire 18 37) ∧ a (.wire 18 35) = a (.wire 18 38) ∧ a (.wire 2 51) = a (.wire 18 40) ∧ a (.virt 17819) = a (.wire 18 41) ∧ a (.virt 17819) = a (.wire 18 42) ∧ a (.wire 2 51) = a (.wire 18 44) ∧ a (.virt 23781) = a (.wire 18 45) ∧ a (.wire 18 43) = a (.wire 18 46) ∧ a (.wire 2 51) = a (.wire 18 48) ∧ a (.virt 17820) = a (.wire 18 49) ∧ a (.virt 17820) = a (.wire 18 50) ∧ a (.wire 2 51) = a (.wire 18 52) ∧ a (.virt 23781) = a (.wire 18 53) ∧ a (.wire 18 51) = a (.wire 18 54) ∧ a (.wire 2 51) = a (.wire 18 56) ∧ a (.virt 17821) = a (.wire 18 57) ∧ a (.virt 17821) = a (.wire 18 58) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies35, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies35, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies36 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 51) = a (.wire 18 60) ∧ a (.virt 23781) = a (.wire 18 61) ∧ a (.wire 18 59) = a (.wire 18 62) ∧ a (.wire 2 51) = a (.wire 18 64) ∧ a (.virt 17822) = a (.wire 18 65) ∧ a (.virt 17822) = a (.wire 18 66) ∧ a (.wire 2 51) = a (.wire 18 68) ∧ a (.virt 23781) = a (.wire 18 69) ∧ a (.wire 18 67) = a (.wire 18 70) ∧ a (.wire 2 51) = a (.wire 18 72) ∧ a (.virt 17823) = a (.wire 18 73) ∧ a (.virt 17823) = a (.wire 18 74) ∧ a (.wire 2 51) = a (.wire 18 76) ∧ a (.virt 23781) = a (.wire 18 77) ∧ a (.wire 18 75) = a (.wire 18 78) ∧ a (.wire 4 15) = a (.wire 19 0) ∧ a (.virt 23758) = a (.wire 19 1) ∧ a (.virt 23758) = a (.wire 19 2) ∧ a (.wire 4 15) = a (.wire 19 4) ∧ a (.virt 23781) = a (.wire 19 5) ∧ a (.wire 19 3) = a (.wire 19 6) ∧ a (.wire 4 15) = a (.wire 19 8) ∧ a (.virt 23759) = a (.wire 19 9) ∧ a (.virt 23759) = a (.wire 19 10) ∧ a (.wire 4 15) = a (.wire 19 12) ∧ a (.virt 23781) = a (.wire 19 13) ∧ a (.wire 19 11) = a (.wire 19 14) ∧ a (.wire 4 15) = a (.wire 19 16) ∧ a (.virt 23760) = a (.wire 19 17) ∧ a (.virt 23760) = a (.wire 19 18) ∧ a (.wire 4 15) = a (.wire 19 20) ∧ a (.virt 23781) = a (.wire 19 21) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies36, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies36, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies37 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 19) = a (.wire 19 22) ∧ a (.wire 4 15) = a (.wire 19 24) ∧ a (.virt 23761) = a (.wire 19 25) ∧ a (.virt 23761) = a (.wire 19 26) ∧ a (.wire 4 15) = a (.wire 19 28) ∧ a (.virt 23781) = a (.wire 19 29) ∧ a (.wire 19 27) = a (.wire 19 30) ∧ a (.wire 4 15) = a (.wire 19 32) ∧ a (.virt 23762) = a (.wire 19 33) ∧ a (.virt 23762) = a (.wire 19 34) ∧ a (.wire 4 15) = a (.wire 19 36) ∧ a (.virt 23781) = a (.wire 19 37) ∧ a (.wire 19 35) = a (.wire 19 38) ∧ a (.wire 4 15) = a (.wire 19 40) ∧ a (.virt 23763) = a (.wire 19 41) ∧ a (.virt 23763) = a (.wire 19 42) ∧ a (.wire 4 15) = a (.wire 19 44) ∧ a (.virt 23781) = a (.wire 19 45) ∧ a (.wire 19 43) = a (.wire 19 46) ∧ a (.wire 4 15) = a (.wire 19 48) ∧ a (.virt 23764) = a (.wire 19 49) ∧ a (.virt 23764) = a (.wire 19 50) ∧ a (.wire 4 15) = a (.wire 19 52) ∧ a (.virt 23781) = a (.wire 19 53) ∧ a (.wire 19 51) = a (.wire 19 54) ∧ a (.wire 4 15) = a (.wire 19 56) ∧ a (.virt 23765) = a (.wire 19 57) ∧ a (.virt 23765) = a (.wire 19 58) ∧ a (.wire 4 15) = a (.wire 19 60) ∧ a (.virt 23781) = a (.wire 19 61) ∧ a (.wire 19 59) = a (.wire 19 62) ∧ a (.wire 4 15) = a (.wire 19 64) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies37, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies37, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies38 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23766) = a (.wire 19 65) ∧ a (.virt 23766) = a (.wire 19 66) ∧ a (.wire 4 15) = a (.wire 19 68) ∧ a (.virt 23781) = a (.wire 19 69) ∧ a (.wire 19 67) = a (.wire 19 70) ∧ a (.wire 4 15) = a (.wire 19 72) ∧ a (.virt 23767) = a (.wire 19 73) ∧ a (.virt 23767) = a (.wire 19 74) ∧ a (.wire 4 15) = a (.wire 19 76) ∧ a (.virt 23781) = a (.wire 19 77) ∧ a (.wire 19 75) = a (.wire 19 78) ∧ a (.wire 1 43) = a (.wire 20 0) ∧ a (.virt 5936) = a (.wire 20 1) ∧ a (.virt 5936) = a (.wire 20 2) ∧ a (.wire 1 43) = a (.wire 20 4) ∧ a (.virt 23781) = a (.wire 20 5) ∧ a (.wire 20 3) = a (.wire 20 6) ∧ a (.wire 1 43) = a (.wire 20 8) ∧ a (.virt 5937) = a (.wire 20 9) ∧ a (.virt 5937) = a (.wire 20 10) ∧ a (.wire 1 43) = a (.wire 20 12) ∧ a (.virt 23781) = a (.wire 20 13) ∧ a (.wire 20 11) = a (.wire 20 14) ∧ a (.wire 1 43) = a (.wire 20 16) ∧ a (.virt 5938) = a (.wire 20 17) ∧ a (.virt 5938) = a (.wire 20 18) ∧ a (.wire 1 43) = a (.wire 20 20) ∧ a (.virt 23781) = a (.wire 20 21) ∧ a (.wire 20 19) = a (.wire 20 22) ∧ a (.wire 1 43) = a (.wire 20 24) ∧ a (.virt 5939) = a (.wire 20 25) ∧ a (.virt 5939) = a (.wire 20 26) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies38, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies38, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies39 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 1 43) = a (.wire 20 28) ∧ a (.virt 23781) = a (.wire 20 29) ∧ a (.wire 20 27) = a (.wire 20 30) ∧ a (.wire 1 43) = a (.wire 20 32) ∧ a (.virt 5940) = a (.wire 20 33) ∧ a (.virt 5940) = a (.wire 20 34) ∧ a (.wire 1 43) = a (.wire 20 36) ∧ a (.virt 23781) = a (.wire 20 37) ∧ a (.wire 20 35) = a (.wire 20 38) ∧ a (.wire 1 43) = a (.wire 20 40) ∧ a (.virt 5941) = a (.wire 20 41) ∧ a (.virt 5941) = a (.wire 20 42) ∧ a (.wire 1 43) = a (.wire 20 44) ∧ a (.virt 23781) = a (.wire 20 45) ∧ a (.wire 20 43) = a (.wire 20 46) ∧ a (.wire 1 43) = a (.wire 20 48) ∧ a (.virt 5942) = a (.wire 20 49) ∧ a (.virt 5942) = a (.wire 20 50) ∧ a (.wire 1 43) = a (.wire 20 52) ∧ a (.virt 23781) = a (.wire 20 53) ∧ a (.wire 20 51) = a (.wire 20 54) ∧ a (.wire 1 43) = a (.wire 20 56) ∧ a (.virt 5943) = a (.wire 20 57) ∧ a (.virt 5943) = a (.wire 20 58) ∧ a (.wire 1 43) = a (.wire 20 60) ∧ a (.virt 23781) = a (.wire 20 61) ∧ a (.wire 20 59) = a (.wire 20 62) ∧ a (.wire 2 7) = a (.wire 20 64) ∧ a (.virt 11880) = a (.wire 20 65) ∧ a (.virt 11880) = a (.wire 20 66) ∧ a (.wire 2 7) = a (.wire 20 68) ∧ a (.virt 23781) = a (.wire 20 69) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies39, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies39, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies40 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 67) = a (.wire 20 70) ∧ a (.wire 2 7) = a (.wire 20 72) ∧ a (.virt 11881) = a (.wire 20 73) ∧ a (.virt 11881) = a (.wire 20 74) ∧ a (.wire 2 7) = a (.wire 20 76) ∧ a (.virt 23781) = a (.wire 20 77) ∧ a (.wire 20 75) = a (.wire 20 78) ∧ a (.wire 2 7) = a (.wire 21 0) ∧ a (.virt 11882) = a (.wire 21 1) ∧ a (.virt 11882) = a (.wire 21 2) ∧ a (.wire 2 7) = a (.wire 21 4) ∧ a (.virt 23781) = a (.wire 21 5) ∧ a (.wire 21 3) = a (.wire 21 6) ∧ a (.wire 2 7) = a (.wire 21 8) ∧ a (.virt 11883) = a (.wire 21 9) ∧ a (.virt 11883) = a (.wire 21 10) ∧ a (.wire 2 7) = a (.wire 21 12) ∧ a (.virt 23781) = a (.wire 21 13) ∧ a (.wire 21 11) = a (.wire 21 14) ∧ a (.wire 2 7) = a (.wire 21 16) ∧ a (.virt 11884) = a (.wire 21 17) ∧ a (.virt 11884) = a (.wire 21 18) ∧ a (.wire 2 7) = a (.wire 21 20) ∧ a (.virt 23781) = a (.wire 21 21) ∧ a (.wire 21 19) = a (.wire 21 22) ∧ a (.wire 2 7) = a (.wire 21 24) ∧ a (.virt 11885) = a (.wire 21 25) ∧ a (.virt 11885) = a (.wire 21 26) ∧ a (.wire 2 7) = a (.wire 21 28) ∧ a (.virt 23781) = a (.wire 21 29) ∧ a (.wire 21 27) = a (.wire 21 30) ∧ a (.wire 2 7) = a (.wire 21 32) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies40, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies40, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies41 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 11886) = a (.wire 21 33) ∧ a (.virt 11886) = a (.wire 21 34) ∧ a (.wire 2 7) = a (.wire 21 36) ∧ a (.virt 23781) = a (.wire 21 37) ∧ a (.wire 21 35) = a (.wire 21 38) ∧ a (.wire 2 7) = a (.wire 21 40) ∧ a (.virt 11887) = a (.wire 21 41) ∧ a (.virt 11887) = a (.wire 21 42) ∧ a (.wire 2 7) = a (.wire 21 44) ∧ a (.virt 23781) = a (.wire 21 45) ∧ a (.wire 21 43) = a (.wire 21 46) ∧ a (.wire 2 51) = a (.wire 21 48) ∧ a (.virt 17824) = a (.wire 21 49) ∧ a (.virt 17824) = a (.wire 21 50) ∧ a (.wire 2 51) = a (.wire 21 52) ∧ a (.virt 23781) = a (.wire 21 53) ∧ a (.wire 21 51) = a (.wire 21 54) ∧ a (.wire 2 51) = a (.wire 21 56) ∧ a (.virt 17825) = a (.wire 21 57) ∧ a (.virt 17825) = a (.wire 21 58) ∧ a (.wire 2 51) = a (.wire 21 60) ∧ a (.virt 23781) = a (.wire 21 61) ∧ a (.wire 21 59) = a (.wire 21 62) ∧ a (.wire 2 51) = a (.wire 21 64) ∧ a (.virt 17826) = a (.wire 21 65) ∧ a (.virt 17826) = a (.wire 21 66) ∧ a (.wire 2 51) = a (.wire 21 68) ∧ a (.virt 23781) = a (.wire 21 69) ∧ a (.wire 21 67) = a (.wire 21 70) ∧ a (.wire 2 51) = a (.wire 21 72) ∧ a (.virt 17827) = a (.wire 21 73) ∧ a (.virt 17827) = a (.wire 21 74) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies41, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies41, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies42 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 51) = a (.wire 21 76) ∧ a (.virt 23781) = a (.wire 21 77) ∧ a (.wire 21 75) = a (.wire 21 78) ∧ a (.wire 2 51) = a (.wire 22 0) ∧ a (.virt 17828) = a (.wire 22 1) ∧ a (.virt 17828) = a (.wire 22 2) ∧ a (.wire 2 51) = a (.wire 22 4) ∧ a (.virt 23781) = a (.wire 22 5) ∧ a (.wire 22 3) = a (.wire 22 6) ∧ a (.wire 2 51) = a (.wire 22 8) ∧ a (.virt 17829) = a (.wire 22 9) ∧ a (.virt 17829) = a (.wire 22 10) ∧ a (.wire 2 51) = a (.wire 22 12) ∧ a (.virt 23781) = a (.wire 22 13) ∧ a (.wire 22 11) = a (.wire 22 14) ∧ a (.wire 2 51) = a (.wire 22 16) ∧ a (.virt 17830) = a (.wire 22 17) ∧ a (.virt 17830) = a (.wire 22 18) ∧ a (.wire 2 51) = a (.wire 22 20) ∧ a (.virt 23781) = a (.wire 22 21) ∧ a (.wire 22 19) = a (.wire 22 22) ∧ a (.wire 2 51) = a (.wire 22 24) ∧ a (.virt 17831) = a (.wire 22 25) ∧ a (.virt 17831) = a (.wire 22 26) ∧ a (.wire 2 51) = a (.wire 22 28) ∧ a (.virt 23781) = a (.wire 22 29) ∧ a (.wire 22 27) = a (.wire 22 30) ∧ a (.wire 4 15) = a (.wire 22 32) ∧ a (.virt 23768) = a (.wire 22 33) ∧ a (.virt 23768) = a (.wire 22 34) ∧ a (.wire 4 15) = a (.wire 22 36) ∧ a (.virt 23781) = a (.wire 22 37) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies42, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [publicBatchWrapper4.copies42, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies43 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 35) = a (.wire 22 38) ∧ a (.wire 4 15) = a (.wire 22 40) ∧ a (.virt 23769) = a (.wire 22 41) ∧ a (.virt 23769) = a (.wire 22 42) ∧ a (.wire 4 15) = a (.wire 22 44) ∧ a (.virt 23781) = a (.wire 22 45) ∧ a (.wire 22 43) = a (.wire 22 46) ∧ a (.wire 4 15) = a (.wire 22 48) ∧ a (.virt 23770) = a (.wire 22 49) ∧ a (.virt 23770) = a (.wire 22 50) ∧ a (.wire 4 15) = a (.wire 22 52) ∧ a (.virt 23781) = a (.wire 22 53) ∧ a (.wire 22 51) = a (.wire 22 54) ∧ a (.wire 4 15) = a (.wire 22 56) ∧ a (.virt 23771) = a (.wire 22 57) ∧ a (.virt 23771) = a (.wire 22 58) ∧ a (.wire 4 15) = a (.wire 22 60) ∧ a (.virt 23781) = a (.wire 22 61) ∧ a (.wire 22 59) = a (.wire 22 62) ∧ a (.wire 4 15) = a (.wire 22 64) ∧ a (.virt 23772) = a (.wire 22 65) ∧ a (.virt 23772) = a (.wire 22 66) ∧ a (.wire 4 15) = a (.wire 22 68) ∧ a (.virt 23781) = a (.wire 22 69) ∧ a (.wire 22 67) = a (.wire 22 70) ∧ a (.wire 4 15) = a (.wire 22 72) ∧ a (.virt 23773) = a (.wire 22 73) ∧ a (.virt 23773) = a (.wire 22 74) ∧ a (.wire 4 15) = a (.wire 22 76) ∧ a (.virt 23781) = a (.wire 22 77) ∧ a (.wire 22 75) = a (.wire 22 78) ∧ a (.wire 4 15) = a (.wire 23 0) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies43, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [publicBatchWrapper4.copies43, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_copies44 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23774) = a (.wire 23 1) ∧ a (.virt 23774) = a (.wire 23 2) ∧ a (.wire 4 15) = a (.wire 23 4) ∧ a (.virt 23781) = a (.wire 23 5) ∧ a (.wire 23 3) = a (.wire 23 6) ∧ a (.wire 4 15) = a (.wire 23 8) ∧ a (.virt 23775) = a (.wire 23 9) ∧ a (.virt 23775) = a (.wire 23 10) ∧ a (.wire 4 15) = a (.wire 23 12) ∧ a (.virt 23781) = a (.wire 23 13) ∧ a (.wire 23 11) = a (.wire 23 14) := by
  have hc : ∀ q ∈ publicBatchWrapper4.copies44, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [publicBatchWrapper4.copies44, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem publicBatchWrapper4_consts (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = 1 ∧ a (.virt 23781) = 0 ∧ a (.virt 23782) = 2 ∧ a (.virt 23863) = 8 := by
  have hconst := h.2.2
  simp only [publicBatchWrapper4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3⟩ := hconst
  exact ⟨k0, k1, k2, k3⟩

theorem publicBatchWrapper4_f0 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 5918) = a (.virt 23782) := by
  have c0 := (publicBatchWrapper4_copies0 a h).1
  exact c0

theorem publicBatchWrapper4_f1 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 11862) = a (.virt 23782) := by
  have c1 := (publicBatchWrapper4_copies0 a h).2.1
  exact c1

theorem publicBatchWrapper4_f2 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 17806) = a (.virt 23782) := by
  have c2 := (publicBatchWrapper4_copies0 a h).2.2.1
  exact c2

theorem publicBatchWrapper4_f3 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23750) = a (.virt 23782) := by
  have c3 := (publicBatchWrapper4_copies0 a h).2.2.2.1
  exact c3

theorem publicBatchWrapper4_f4 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5921)) (a (.virt 23781)) (a (.virt 23783)) (a (.virt 23784)) := by
  have c4 := (publicBatchWrapper4_copies0 a h).2.2.2.2.1
  have c5 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.1
  have c6 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.1
  have c7 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c14 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_0 := arithEq_of_rows h (row := 0) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_0
  simp only [← c4, k0, ← c5, k0, ← c6] at e_0_0
  have e_0_1 := arithEq_of_rows h (row := 0) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_1
  simp only [← c13, ← c14, k0, ← c15] at e_0_1
  have e_1_0 := arithEq_of_rows h (row := 1) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_0
  simp only [← c7, ← c8, ← c9] at e_1_0
  have e_1_1 := arithEq_of_rows h (row := 1) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_1
  simp only [← c10, ← c11, ← c12] at e_1_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_0
    linear_combination c16.trans k1 - hc
  · have hc := e_0_1
    simp only [e_0_0, e_1_1] at hc
    linear_combination c17.trans k1 - hc

theorem publicBatchWrapper4_f5 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5922)) (a (.virt 23781)) (a (.virt 23785)) (a (.virt 23786)) := by
  have c18 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (publicBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_2 := arithEq_of_rows h (row := 0) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_2
  simp only [← c18, k0, ← c19, k0, ← c20] at e_0_2
  have e_0_3 := arithEq_of_rows h (row := 0) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_3
  simp only [← c27, ← c28, k0, ← c29] at e_0_3
  have e_1_2 := arithEq_of_rows h (row := 1) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_2
  simp only [← c21, ← c22, ← c23] at e_1_2
  have e_1_3 := arithEq_of_rows h (row := 1) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_3
  simp only [← c24, ← c25, ← c26] at e_1_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_2
    linear_combination c30.trans k1 - hc
  · have hc := e_0_3
    simp only [e_0_2, e_1_3] at hc
    linear_combination c31.trans k1 - hc

theorem publicBatchWrapper4_f6 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5923)) (a (.virt 23781)) (a (.virt 23787)) (a (.virt 23788)) := by
  have c32 := (publicBatchWrapper4_copies1 a h).1
  have c33 := (publicBatchWrapper4_copies1 a h).2.1
  have c34 := (publicBatchWrapper4_copies1 a h).2.2.1
  have c35 := (publicBatchWrapper4_copies1 a h).2.2.2.1
  have c36 := (publicBatchWrapper4_copies1 a h).2.2.2.2.1
  have c37 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.1
  have c38 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.1
  have c39 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.1
  have c42 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_4 := arithEq_of_rows h (row := 0) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_4
  simp only [← c32, k0, ← c33, k0, ← c34] at e_0_4
  have e_0_5 := arithEq_of_rows h (row := 0) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_5
  simp only [← c41, ← c42, k0, ← c43] at e_0_5
  have e_1_4 := arithEq_of_rows h (row := 1) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_4
  simp only [← c35, ← c36, ← c37] at e_1_4
  have e_1_5 := arithEq_of_rows h (row := 1) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_5
  simp only [← c38, ← c39, ← c40] at e_1_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_4
    linear_combination c44.trans k1 - hc
  · have hc := e_0_5
    simp only [e_0_4, e_1_5] at hc
    linear_combination c45.trans k1 - hc

theorem publicBatchWrapper4_f7 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5924)) (a (.virt 23781)) (a (.virt 23789)) (a (.virt 23790)) := by
  have c46 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_6 := arithEq_of_rows h (row := 0) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_6
  simp only [← c46, k0, ← c47, k0, ← c48] at e_0_6
  have e_0_7 := arithEq_of_rows h (row := 0) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_7
  simp only [← c55, ← c56, k0, ← c57] at e_0_7
  have e_1_6 := arithEq_of_rows h (row := 1) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_6
  simp only [← c49, ← c50, ← c51] at e_1_6
  have e_1_7 := arithEq_of_rows h (row := 1) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_7
  simp only [← c52, ← c53, ← c54] at e_1_7
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_6
    linear_combination c58.trans k1 - hc
  · have hc := e_0_7
    simp only [e_0_6, e_1_7] at hc
    linear_combination c59.trans k1 - hc

theorem publicBatchWrapper4_f8 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 1 35) = band (a (.virt 23783)) (a (.virt 23785)) := by
  have c60 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c62 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c60, ← c61, ← c62] at e_1_8
  have hr := e_1_8
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f9 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 1 39) = band (a (.virt 23787)) (a (.virt 23789)) := by
  have c63 := (publicBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (publicBatchWrapper4_copies2 a h).1
  have c65 := (publicBatchWrapper4_copies2 a h).2.1
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c63, ← c64, ← c65] at e_1_9
  have hr := e_1_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f10 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) := by
  have c66 := (publicBatchWrapper4_copies2 a h).2.2.1
  have c67 := (publicBatchWrapper4_copies2 a h).2.2.2.1
  have c68 := (publicBatchWrapper4_copies2 a h).2.2.2.2.1
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c66, ← c67, ← c68] at e_1_10
  have hr := e_1_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f11 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11865)) (a (.virt 23781)) (a (.virt 23791)) (a (.virt 23792)) := by
  have c69 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.1
  have c70 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.1
  have c71 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_8 := arithEq_of_rows h (row := 0) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_8
  simp only [← c69, k0, ← c70, k0, ← c71] at e_0_8
  have e_0_9 := arithEq_of_rows h (row := 0) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_9
  simp only [← c78, ← c79, k0, ← c80] at e_0_9
  have e_1_11 := arithEq_of_rows h (row := 1) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_11
  simp only [← c72, ← c73, ← c74] at e_1_11
  have e_1_12 := arithEq_of_rows h (row := 1) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_12
  simp only [← c75, ← c76, ← c77] at e_1_12
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_11
    linear_combination c81.trans k1 - hc
  · have hc := e_0_9
    simp only [e_0_8, e_1_12] at hc
    linear_combination c82.trans k1 - hc

theorem publicBatchWrapper4_f12 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11866)) (a (.virt 23781)) (a (.virt 23793)) (a (.virt 23794)) := by
  have c83 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (publicBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (publicBatchWrapper4_copies3 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_10 := arithEq_of_rows h (row := 0) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_10
  simp only [← c83, k0, ← c84, k0, ← c85] at e_0_10
  have e_0_11 := arithEq_of_rows h (row := 0) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_11
  simp only [← c92, ← c93, k0, ← c94] at e_0_11
  have e_1_13 := arithEq_of_rows h (row := 1) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_13
  simp only [← c86, ← c87, ← c88] at e_1_13
  have e_1_14 := arithEq_of_rows h (row := 1) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_14
  simp only [← c89, ← c90, ← c91] at e_1_14
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_13
    linear_combination c95.trans k1 - hc
  · have hc := e_0_11
    simp only [e_0_10, e_1_14] at hc
    linear_combination c96.trans k1 - hc

theorem publicBatchWrapper4_f13 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11867)) (a (.virt 23781)) (a (.virt 23795)) (a (.virt 23796)) := by
  have c97 := (publicBatchWrapper4_copies3 a h).2.1
  have c98 := (publicBatchWrapper4_copies3 a h).2.2.1
  have c99 := (publicBatchWrapper4_copies3 a h).2.2.2.1
  have c100 := (publicBatchWrapper4_copies3 a h).2.2.2.2.1
  have c101 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.1
  have c102 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.1
  have c103 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_12 := arithEq_of_rows h (row := 0) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c97, k0, ← c98, k0, ← c99] at e_0_12
  have e_0_13 := arithEq_of_rows h (row := 0) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c106, ← c107, k0, ← c108] at e_0_13
  have e_1_15 := arithEq_of_rows h (row := 1) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_15
  simp only [← c100, ← c101, ← c102] at e_1_15
  have e_1_16 := arithEq_of_rows h (row := 1) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_16
  simp only [← c103, ← c104, ← c105] at e_1_16
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_15
    linear_combination c109.trans k1 - hc
  · have hc := e_0_13
    simp only [e_0_12, e_1_16] at hc
    linear_combination c110.trans k1 - hc

theorem publicBatchWrapper4_f14 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11868)) (a (.virt 23781)) (a (.virt 23797)) (a (.virt 23798)) := by
  have c111 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_14 := arithEq_of_rows h (row := 0) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c111, k0, ← c112, k0, ← c113] at e_0_14
  have e_0_15 := arithEq_of_rows h (row := 0) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_15
  simp only [← c120, ← c121, k0, ← c122] at e_0_15
  have e_1_17 := arithEq_of_rows h (row := 1) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_17
  simp only [← c114, ← c115, ← c116] at e_1_17
  have e_1_18 := arithEq_of_rows h (row := 1) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_18
  simp only [← c117, ← c118, ← c119] at e_1_18
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_1_17
    linear_combination c123.trans k1 - hc
  · have hc := e_0_15
    simp only [e_0_14, e_1_18] at hc
    linear_combination c124.trans k1 - hc

theorem publicBatchWrapper4_f15 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 1 79) = band (a (.virt 23791)) (a (.virt 23793)) := by
  have c125 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c127 := (publicBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_1_19 := arithEq_of_rows h (row := 1) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_19
  simp only [← c125, ← c126, ← c127] at e_1_19
  have hr := e_1_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f16 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 3) = band (a (.virt 23795)) (a (.virt 23797)) := by
  have c128 := (publicBatchWrapper4_copies4 a h).1
  have c129 := (publicBatchWrapper4_copies4 a h).2.1
  have c130 := (publicBatchWrapper4_copies4 a h).2.2.1
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c128, ← c129, ← c130] at e_2_0
  have hr := e_2_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f17 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) := by
  have c131 := (publicBatchWrapper4_copies4 a h).2.2.2.1
  have c132 := (publicBatchWrapper4_copies4 a h).2.2.2.2.1
  have c133 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.1
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c131, ← c132, ← c133] at e_2_1
  have hr := e_2_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f18 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17809)) (a (.virt 23781)) (a (.virt 23799)) (a (.virt 23800)) := by
  have c134 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.1
  have c135 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c144 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c145 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_16 := arithEq_of_rows h (row := 0) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_16
  simp only [← c134, k0, ← c135, k0, ← c136] at e_0_16
  have e_0_17 := arithEq_of_rows h (row := 0) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_17
  simp only [← c143, ← c144, k0, ← c145] at e_0_17
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c137, ← c138, ← c139] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c140, ← c141, ← c142] at e_2_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_2
    linear_combination c146.trans k1 - hc
  · have hc := e_0_17
    simp only [e_0_16, e_2_3] at hc
    linear_combination c147.trans k1 - hc

theorem publicBatchWrapper4_f19 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17810)) (a (.virt 23781)) (a (.virt 23801)) (a (.virt 23802)) := by
  have c148 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c151 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (publicBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (publicBatchWrapper4_copies5 a h).1
  have c161 := (publicBatchWrapper4_copies5 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_0_18 := arithEq_of_rows h (row := 0) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_18
  simp only [← c148, k0, ← c149, k0, ← c150] at e_0_18
  have e_0_19 := arithEq_of_rows h (row := 0) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_19
  simp only [← c157, ← c158, k0, ← c159] at e_0_19
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c151, ← c152, ← c153] at e_2_4
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c154, ← c155, ← c156] at e_2_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_4
    linear_combination c160.trans k1 - hc
  · have hc := e_0_19
    simp only [e_0_18, e_2_5] at hc
    linear_combination c161.trans k1 - hc

theorem publicBatchWrapper4_f20 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17811)) (a (.virt 23781)) (a (.virt 23803)) (a (.virt 23804)) := by
  have c162 := (publicBatchWrapper4_copies5 a h).2.2.1
  have c163 := (publicBatchWrapper4_copies5 a h).2.2.2.1
  have c164 := (publicBatchWrapper4_copies5 a h).2.2.2.2.1
  have c165 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.1
  have c166 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.1
  have c167 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c165, ← c166, ← c167] at e_2_6
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c168, ← c169, ← c170] at e_2_7
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c162, k0, ← c163, k0, ← c164] at e_3_0
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c171, ← c172, k0, ← c173] at e_3_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_6
    linear_combination c174.trans k1 - hc
  · have hc := e_3_1
    simp only [e_3_0, e_2_7] at hc
    linear_combination c175.trans k1 - hc

theorem publicBatchWrapper4_f21 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17812)) (a (.virt 23781)) (a (.virt 23805)) (a (.virt 23806)) := by
  have c176 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c179, ← c180, ← c181] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c182, ← c183, ← c184] at e_2_9
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c176, k0, ← c177, k0, ← c178] at e_3_2
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c185, ← c186, k0, ← c187] at e_3_3
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_8
    linear_combination c188.trans k1 - hc
  · have hc := e_3_3
    simp only [e_3_2, e_2_9] at hc
    linear_combination c189.trans k1 - hc

theorem publicBatchWrapper4_f22 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 43) = band (a (.virt 23799)) (a (.virt 23801)) := by
  have c190 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (publicBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (publicBatchWrapper4_copies6 a h).1
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c190, ← c191, ← c192] at e_2_10
  have hr := e_2_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f23 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 47) = band (a (.virt 23803)) (a (.virt 23805)) := by
  have c193 := (publicBatchWrapper4_copies6 a h).2.1
  have c194 := (publicBatchWrapper4_copies6 a h).2.2.1
  have c195 := (publicBatchWrapper4_copies6 a h).2.2.2.1
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c193, ← c194, ← c195] at e_2_11
  have hr := e_2_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f24 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 2 51) = band (a (.wire 2 43)) (a (.wire 2 47)) := by
  have c196 := (publicBatchWrapper4_copies6 a h).2.2.2.2.1
  have c197 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.1
  have c198 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.1
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c196, ← c197, ← c198] at e_2_12
  have hr := e_2_12
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f25 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23753)) (a (.virt 23781)) (a (.virt 23807)) (a (.virt 23808)) := by
  have c199 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c202, ← c203, ← c204] at e_2_13
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c205, ← c206, ← c207] at e_2_14
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c199, k0, ← c200, k0, ← c201] at e_3_4
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c208, ← c209, k0, ← c210] at e_3_5
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_13
    linear_combination c211.trans k1 - hc
  · have hc := e_3_5
    simp only [e_3_4, e_2_14] at hc
    linear_combination c212.trans k1 - hc

theorem publicBatchWrapper4_f26 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23754)) (a (.virt 23781)) (a (.virt 23809)) (a (.virt 23810)) := by
  have c213 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (publicBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (publicBatchWrapper4_copies7 a h).1
  have c225 := (publicBatchWrapper4_copies7 a h).2.1
  have c226 := (publicBatchWrapper4_copies7 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_2_15 := arithEq_of_rows h (row := 2) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_15
  simp only [← c216, ← c217, ← c218] at e_2_15
  have e_2_16 := arithEq_of_rows h (row := 2) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_16
  simp only [← c219, ← c220, ← c221] at e_2_16
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c213, k0, ← c214, k0, ← c215] at e_3_6
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c222, ← c223, k0, ← c224] at e_3_7
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_15
    linear_combination c225.trans k1 - hc
  · have hc := e_3_7
    simp only [e_3_6, e_2_16] at hc
    linear_combination c226.trans k1 - hc

theorem publicBatchWrapper4_f27 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23755)) (a (.virt 23781)) (a (.virt 23811)) (a (.virt 23812)) := by
  have c227 := (publicBatchWrapper4_copies7 a h).2.2.2.1
  have c228 := (publicBatchWrapper4_copies7 a h).2.2.2.2.1
  have c229 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.1
  have c230 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.1
  have c231 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_2_17 := arithEq_of_rows h (row := 2) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_17
  simp only [← c230, ← c231, ← c232] at e_2_17
  have e_2_18 := arithEq_of_rows h (row := 2) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_18
  simp only [← c233, ← c234, ← c235] at e_2_18
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c227, k0, ← c228, k0, ← c229] at e_3_8
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c236, ← c237, k0, ← c238] at e_3_9
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_17
    linear_combination c239.trans k1 - hc
  · have hc := e_3_9
    simp only [e_3_8, e_2_18] at hc
    linear_combination c240.trans k1 - hc

theorem publicBatchWrapper4_f28 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23756)) (a (.virt 23781)) (a (.virt 23813)) (a (.virt 23814)) := by
  have c241 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_2_19 := arithEq_of_rows h (row := 2) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_19
  simp only [← c244, ← c245, ← c246] at e_2_19
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c241, k0, ← c242, k0, ← c243] at e_3_10
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c250, ← c251, k0, ← c252] at e_3_11
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c247, ← c248, ← c249] at e_4_0
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_19
    linear_combination c253.trans k1 - hc
  · have hc := e_3_11
    simp only [e_3_10, e_4_0] at hc
    linear_combination c254.trans k1 - hc

theorem publicBatchWrapper4_f29 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 7) = band (a (.virt 23807)) (a (.virt 23809)) := by
  have c255 := (publicBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (publicBatchWrapper4_copies8 a h).1
  have c257 := (publicBatchWrapper4_copies8 a h).2.1
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c255, ← c256, ← c257] at e_4_1
  have hr := e_4_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f30 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 11) = band (a (.virt 23811)) (a (.virt 23813)) := by
  have c258 := (publicBatchWrapper4_copies8 a h).2.2.1
  have c259 := (publicBatchWrapper4_copies8 a h).2.2.2.1
  have c260 := (publicBatchWrapper4_copies8 a h).2.2.2.2.1
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c258, ← c259, ← c260] at e_4_2
  have hr := e_4_2
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f31 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11)) := by
  have c261 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.1
  have c262 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.1
  have c263 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.1
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c261, ← c262, ← c263] at e_4_3
  have hr := e_4_3
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f32 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 51) = bnot (a (.wire 1 43)) := by
  have c264 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c264, k0, ← c265, k0, ← c266] at e_3_12
  have hr := e_3_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f33 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.virt 23780) = bnot (a (.virt 23781)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  simp only [bnot, k0, k1]
  ring

theorem publicBatchWrapper4_f34 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 51) = band (a (.wire 3 51)) (a (.virt 23780)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  simp only [band, k0]
  ring

theorem publicBatchWrapper4_f35 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 55) = bselect (a (.wire 3 51)) (a (.virt 5921)) (a (.virt 23781)) := by
  have c267 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c267, ← c268, ← c269, k1] at e_3_13
  have hr := e_3_13
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f36 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 59) = bselect (a (.wire 3 51)) (a (.virt 5922)) (a (.virt 23781)) := by
  have c270 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c270, ← c271, ← c272, k1] at e_3_14
  have hr := e_3_14
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f37 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 63) = bselect (a (.wire 3 51)) (a (.virt 5923)) (a (.virt 23781)) := by
  have c273 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_15 := arithEq_of_rows h (row := 3) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_15
  simp only [← c273, ← c274, ← c275, k1] at e_3_15
  have hr := e_3_15
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f38 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 67) = bselect (a (.wire 3 51)) (a (.virt 5924)) (a (.virt 23781)) := by
  have c276 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c278 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_16 := arithEq_of_rows h (row := 3) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_16
  simp only [← c276, ← c277, ← c278, k1] at e_3_16
  have hr := e_3_16
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f39 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 71) = bselect (a (.wire 3 51)) (a (.virt 5925)) (a (.virt 23781)) := by
  have c279 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_17 := arithEq_of_rows h (row := 3) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_17
  simp only [← c279, ← c280, ← c281, k1] at e_3_17
  have hr := e_3_17
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f40 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 75) = bselect (a (.wire 3 51)) (a (.virt 5919)) (a (.virt 23781)) := by
  have c282 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_18 := arithEq_of_rows h (row := 3) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_18
  simp only [← c282, ← c283, ← c284, k1] at e_3_18
  have hr := e_3_18
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f41 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 79) = bselect (a (.wire 3 51)) (a (.virt 5920)) (a (.virt 23781)) := by
  have c285 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (publicBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_3_19 := arithEq_of_rows h (row := 3) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_19
  simp only [← c285, ← c286, ← c287, k1] at e_3_19
  have hr := e_3_19
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f42 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 3 51) = bor (a (.virt 23781)) (a (.wire 3 51)) := by
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  simp only [bor, k1]
  ring

theorem publicBatchWrapper4_f43 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 3) = bnot (a (.wire 2 7)) := by
  have c288 := (publicBatchWrapper4_copies9 a h).1
  have c289 := (publicBatchWrapper4_copies9 a h).2.1
  have c290 := (publicBatchWrapper4_copies9 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c288, k0, ← c289, k0, ← c290] at e_5_0
  have hr := e_5_0
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f44 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 7) = bnot (a (.wire 3 51)) := by
  have c291 := (publicBatchWrapper4_copies9 a h).2.2.2.1
  have c292 := (publicBatchWrapper4_copies9 a h).2.2.2.2.1
  have c293 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c291, k0, ← c292, k0, ← c293] at e_5_1
  have hr := e_5_1
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f45 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 19) = band (a (.wire 5 3)) (a (.wire 5 7)) := by
  have c294 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.1
  have c295 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.1
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c294, ← c295, ← c296] at e_4_4
  have hr := e_4_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f46 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 15) = bselect (a (.wire 4 19)) (a (.virt 11865)) (a (.wire 3 55)) := by
  have c297 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c297, ← c298, ← c299] at e_5_2
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c300, ← c301, ← c302] at e_5_3
  have hr := e_5_3
  simp only [e_5_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f47 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 23) = bselect (a (.wire 4 19)) (a (.virt 11866)) (a (.wire 3 59)) := by
  have c303 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c303, ← c304, ← c305] at e_5_4
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c306, ← c307, ← c308] at e_5_5
  have hr := e_5_5
  simp only [e_5_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f48 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 31) = bselect (a (.wire 4 19)) (a (.virt 11867)) (a (.wire 3 63)) := by
  have c309 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c309, ← c310, ← c311] at e_5_6
  have e_5_7 := arithEq_of_rows h (row := 5) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c312, ← c313, ← c314] at e_5_7
  have hr := e_5_7
  simp only [e_5_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f49 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 39) = bselect (a (.wire 4 19)) (a (.virt 11868)) (a (.wire 3 67)) := by
  have c315 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (publicBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (publicBatchWrapper4_copies10 a h).1
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c315, ← c316, ← c317] at e_5_8
  have e_5_9 := arithEq_of_rows h (row := 5) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c318, ← c319, ← c320] at e_5_9
  have hr := e_5_9
  simp only [e_5_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f50 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 47) = bselect (a (.wire 4 19)) (a (.virt 11869)) (a (.wire 3 71)) := by
  have c321 := (publicBatchWrapper4_copies10 a h).2.1
  have c322 := (publicBatchWrapper4_copies10 a h).2.2.1
  have c323 := (publicBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (publicBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.1
  have c326 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.1
  have e_5_10 := arithEq_of_rows h (row := 5) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c321, ← c322, ← c323] at e_5_10
  have e_5_11 := arithEq_of_rows h (row := 5) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c324, ← c325, ← c326] at e_5_11
  have hr := e_5_11
  simp only [e_5_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f51 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 55) = bselect (a (.wire 4 19)) (a (.virt 11863)) (a (.wire 3 75)) := by
  have c327 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_12 := arithEq_of_rows h (row := 5) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c327, ← c328, ← c329] at e_5_12
  have e_5_13 := arithEq_of_rows h (row := 5) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c330, ← c331, ← c332] at e_5_13
  have hr := e_5_13
  simp only [e_5_12] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f52 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 63) = bselect (a (.wire 4 19)) (a (.virt 11864)) (a (.wire 3 79)) := by
  have c333 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_14 := arithEq_of_rows h (row := 5) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c333, ← c334, ← c335] at e_5_14
  have e_5_15 := arithEq_of_rows h (row := 5) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_15
  simp only [← c336, ← c337, ← c338] at e_5_15
  have hr := e_5_15
  simp only [e_5_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f53 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 3) = bor (a (.wire 3 51)) (a (.wire 5 3)) := by
  have c339 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c339, ← c340, ← c341] at e_6_0
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c342, ← c343, k0, ← c344] at e_7_0
  have hr := e_7_0
  simp only [e_6_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f54 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 67) = bnot (a (.wire 2 51)) := by
  have c345 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_5_16 := arithEq_of_rows h (row := 5) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_16
  simp only [← c345, k0, ← c346, k0, ← c347] at e_5_16
  have hr := e_5_16
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f55 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 71) = bnot (a (.wire 7 3)) := by
  have c348 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_5_17 := arithEq_of_rows h (row := 5) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_17
  simp only [← c348, k0, ← c349, k0, ← c350] at e_5_17
  have hr := e_5_17
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f56 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 23) = band (a (.wire 5 67)) (a (.wire 5 71)) := by
  have c351 := (publicBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (publicBatchWrapper4_copies11 a h).1
  have c353 := (publicBatchWrapper4_copies11 a h).2.1
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c351, ← c352, ← c353] at e_4_5
  have hr := e_4_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f57 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 5 79) = bselect (a (.wire 4 23)) (a (.virt 17809)) (a (.wire 5 15)) := by
  have c354 := (publicBatchWrapper4_copies11 a h).2.2.1
  have c355 := (publicBatchWrapper4_copies11 a h).2.2.2.1
  have c356 := (publicBatchWrapper4_copies11 a h).2.2.2.2.1
  have c357 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.1
  have c358 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.1
  have c359 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.1
  have e_5_18 := arithEq_of_rows h (row := 5) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_18
  simp only [← c354, ← c355, ← c356] at e_5_18
  have e_5_19 := arithEq_of_rows h (row := 5) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_19
  simp only [← c357, ← c358, ← c359] at e_5_19
  have hr := e_5_19
  simp only [e_5_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f58 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 7) = bselect (a (.wire 4 23)) (a (.virt 17810)) (a (.wire 5 23)) := by
  have c360 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have c362 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c360, ← c361, ← c362] at e_8_0
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c363, ← c364, ← c365] at e_8_1
  have hr := e_8_1
  simp only [e_8_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f59 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 15) = bselect (a (.wire 4 23)) (a (.virt 17811)) (a (.wire 5 31)) := by
  have c366 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c366, ← c367, ← c368] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c369, ← c370, ← c371] at e_8_3
  have hr := e_8_3
  simp only [e_8_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f60 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 23) = bselect (a (.wire 4 23)) (a (.virt 17812)) (a (.wire 5 39)) := by
  have c372 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c372, ← c373, ← c374] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c375, ← c376, ← c377] at e_8_5
  have hr := e_8_5
  simp only [e_8_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f61 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 31) = bselect (a (.wire 4 23)) (a (.virt 17813)) (a (.wire 5 47)) := by
  have c378 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (publicBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c378, ← c379, ← c380] at e_8_6
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c381, ← c382, ← c383] at e_8_7
  have hr := e_8_7
  simp only [e_8_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f62 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 39) = bselect (a (.wire 4 23)) (a (.virt 17807)) (a (.wire 5 55)) := by
  have c384 := (publicBatchWrapper4_copies12 a h).1
  have c385 := (publicBatchWrapper4_copies12 a h).2.1
  have c386 := (publicBatchWrapper4_copies12 a h).2.2.1
  have c387 := (publicBatchWrapper4_copies12 a h).2.2.2.1
  have c388 := (publicBatchWrapper4_copies12 a h).2.2.2.2.1
  have c389 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.1
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c384, ← c385, ← c386] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c387, ← c388, ← c389] at e_8_9
  have hr := e_8_9
  simp only [e_8_8] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f63 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 47) = bselect (a (.wire 4 23)) (a (.virt 17808)) (a (.wire 5 63)) := by
  have c390 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.1
  have c391 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c390, ← c391, ← c392] at e_8_10
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c393, ← c394, ← c395] at e_8_11
  have hr := e_8_11
  simp only [e_8_10] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f64 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 7) = bor (a (.wire 7 3)) (a (.wire 5 67)) := by
  have c396 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c396, ← c397, ← c398] at e_6_1
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c399, ← c400, k0, ← c401] at e_7_1
  have hr := e_7_1
  simp only [e_6_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f65 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 51) = bnot (a (.wire 4 15)) := by
  have c402 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c402, k0, ← c403, k0, ← c404] at e_8_12
  have hr := e_8_12
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f66 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 55) = bnot (a (.wire 7 7)) := by
  have c405 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c405, k0, ← c406, k0, ← c407] at e_8_13
  have hr := e_8_13
  simp only [bnot]
  linear_combination hr

theorem publicBatchWrapper4_f67 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 27) = band (a (.wire 8 51)) (a (.wire 8 55)) := by
  have c408 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c408, ← c409, ← c410] at e_4_6
  have hr := e_4_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f68 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 63) = bselect (a (.wire 4 27)) (a (.virt 23753)) (a (.wire 5 79)) := by
  have c411 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (publicBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (publicBatchWrapper4_copies13 a h).1
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c411, ← c412, ← c413] at e_8_14
  have e_8_15 := arithEq_of_rows h (row := 8) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_15
  simp only [← c414, ← c415, ← c416] at e_8_15
  have hr := e_8_15
  simp only [e_8_14] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f69 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 71) = bselect (a (.wire 4 27)) (a (.virt 23754)) (a (.wire 8 7)) := by
  have c417 := (publicBatchWrapper4_copies13 a h).2.1
  have c418 := (publicBatchWrapper4_copies13 a h).2.2.1
  have c419 := (publicBatchWrapper4_copies13 a h).2.2.2.1
  have c420 := (publicBatchWrapper4_copies13 a h).2.2.2.2.1
  have c421 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.1
  have c422 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.1
  have e_8_16 := arithEq_of_rows h (row := 8) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_16
  simp only [← c417, ← c418, ← c419] at e_8_16
  have e_8_17 := arithEq_of_rows h (row := 8) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_17
  simp only [← c420, ← c421, ← c422] at e_8_17
  have hr := e_8_17
  simp only [e_8_16] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f70 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 8 79) = bselect (a (.wire 4 27)) (a (.virt 23755)) (a (.wire 8 15)) := by
  have c423 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_18 := arithEq_of_rows h (row := 8) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_18
  simp only [← c423, ← c424, ← c425] at e_8_18
  have e_8_19 := arithEq_of_rows h (row := 8) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_19
  simp only [← c426, ← c427, ← c428] at e_8_19
  have hr := e_8_19
  simp only [e_8_18] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f71 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 9 7) = bselect (a (.wire 4 27)) (a (.virt 23756)) (a (.wire 8 23)) := by
  have c429 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c429, ← c430, ← c431] at e_9_0
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c432, ← c433, ← c434] at e_9_1
  have hr := e_9_1
  simp only [e_9_0] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f72 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 9 15) = bselect (a (.wire 4 27)) (a (.virt 23757)) (a (.wire 8 31)) := by
  have c435 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c435, ← c436, ← c437] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c438, ← c439, ← c440] at e_9_3
  have hr := e_9_3
  simp only [e_9_2] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f73 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 9 23) = bselect (a (.wire 4 27)) (a (.virt 23751)) (a (.wire 8 39)) := by
  have c441 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c441, ← c442, ← c443] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c444, ← c445, ← c446] at e_9_5
  have hr := e_9_5
  simp only [e_9_4] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f74 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 9 31) = bselect (a (.wire 4 27)) (a (.virt 23752)) (a (.wire 8 47)) := by
  have c447 := (publicBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (publicBatchWrapper4_copies14 a h).1
  have c449 := (publicBatchWrapper4_copies14 a h).2.1
  have c450 := (publicBatchWrapper4_copies14 a h).2.2.1
  have c451 := (publicBatchWrapper4_copies14 a h).2.2.2.1
  have c452 := (publicBatchWrapper4_copies14 a h).2.2.2.2.1
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c447, ← c448, ← c449] at e_9_6
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c450, ← c451, ← c452] at e_9_7
  have hr := e_9_7
  simp only [e_9_6] at hr
  simp only [bselect]
  linear_combination hr

theorem publicBatchWrapper4_f75 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 11) = bor (a (.wire 7 7)) (a (.wire 8 51)) := by
  have c453 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.1
  have c454 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.1
  have c455 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c453, ← c454, ← c455] at e_6_2
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c456, ← c457, k0, ← c458] at e_7_2
  have hr := e_7_2
  simp only [e_6_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f76 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5919)) (a (.wire 9 23)) (a (.virt 23815)) (a (.virt 23816)) := by
  have c459 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c465, ← c466, ← c467] at e_4_7
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c468, ← c469, ← c470] at e_4_8
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c459, k0, ← c460, k0, ← c461] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c462, ← c463, k0, ← c464] at e_9_9
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c471, ← c472, k0, ← c473] at e_9_10
  refine ⟨?_, ?_⟩
  · have hc := e_4_7
    simp only [e_9_9] at hc
    linear_combination c474.trans k1 - hc
  · have hc := e_9_10
    simp only [e_9_8, e_4_8, e_9_9] at hc
    linear_combination c475.trans k1 - hc

theorem publicBatchWrapper4_f77 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 15) = bor (a (.wire 1 43)) (a (.virt 23815)) := by
  have c476 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (publicBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (publicBatchWrapper4_copies15 a h).1
  have c481 := (publicBatchWrapper4_copies15 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c476, ← c477, ← c478] at e_6_3
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c479, ← c480, k0, ← c481] at e_7_3
  have hr := e_7_3
  simp only [e_6_3] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f78 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 15) = a (.virt 23780) := by
  have c482 := (publicBatchWrapper4_copies15 a h).2.2.1
  exact c482

theorem publicBatchWrapper4_f79 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5920)) (a (.wire 9 31)) (a (.virt 23817)) (a (.virt 23818)) := by
  have c483 := (publicBatchWrapper4_copies15 a h).2.2.2.1
  have c484 := (publicBatchWrapper4_copies15 a h).2.2.2.2.1
  have c485 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.1
  have c486 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.1
  have c487 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c489, ← c490, ← c491] at e_4_9
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c492, ← c493, ← c494] at e_4_10
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c483, k0, ← c484, k0, ← c485] at e_9_11
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c486, ← c487, k0, ← c488] at e_9_12
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c495, ← c496, k0, ← c497] at e_9_13
  refine ⟨?_, ?_⟩
  · have hc := e_4_9
    simp only [e_9_12] at hc
    linear_combination c498.trans k1 - hc
  · have hc := e_9_13
    simp only [e_9_11, e_4_10, e_9_12] at hc
    linear_combination c499.trans k1 - hc

theorem publicBatchWrapper4_f80 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 19) = bor (a (.wire 1 43)) (a (.virt 23817)) := by
  have c500 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c500, ← c501, ← c502] at e_6_4
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c503, ← c504, k0, ← c505] at e_7_4
  have hr := e_7_4
  simp only [e_6_4] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f81 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 19) = a (.virt 23780) := by
  have c506 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c506

theorem publicBatchWrapper4_f82 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5921)) (a (.wire 8 63)) (a (.virt 23819)) (a (.virt 23820)) := by
  have c507 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (publicBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (publicBatchWrapper4_copies16 a h).1
  have c513 := (publicBatchWrapper4_copies16 a h).2.1
  have c514 := (publicBatchWrapper4_copies16 a h).2.2.1
  have c515 := (publicBatchWrapper4_copies16 a h).2.2.2.1
  have c516 := (publicBatchWrapper4_copies16 a h).2.2.2.2.1
  have c517 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.1
  have c518 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.1
  have c519 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c513, ← c514, ← c515] at e_4_11
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c516, ← c517, ← c518] at e_4_12
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c507, k0, ← c508, k0, ← c509] at e_9_14
  have e_9_15 := arithEq_of_rows h (row := 9) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_15
  simp only [← c510, ← c511, k0, ← c512] at e_9_15
  have e_9_16 := arithEq_of_rows h (row := 9) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_16
  simp only [← c519, ← c520, k0, ← c521] at e_9_16
  refine ⟨?_, ?_⟩
  · have hc := e_4_11
    simp only [e_9_15] at hc
    linear_combination c522.trans k1 - hc
  · have hc := e_9_16
    simp only [e_9_14, e_4_12, e_9_15] at hc
    linear_combination c523.trans k1 - hc

theorem publicBatchWrapper4_f83 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5922)) (a (.wire 8 71)) (a (.virt 23821)) (a (.virt 23822)) := by
  have c524 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c532 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c535 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c530, ← c531, ← c532] at e_4_13
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c533, ← c534, ← c535] at e_4_14
  have e_9_17 := arithEq_of_rows h (row := 9) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_17
  simp only [← c524, k0, ← c525, k0, ← c526] at e_9_17
  have e_9_18 := arithEq_of_rows h (row := 9) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_18
  simp only [← c527, ← c528, k0, ← c529] at e_9_18
  have e_9_19 := arithEq_of_rows h (row := 9) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_19
  simp only [← c536, ← c537, k0, ← c538] at e_9_19
  refine ⟨?_, ?_⟩
  · have hc := e_4_13
    simp only [e_9_18] at hc
    linear_combination c539.trans k1 - hc
  · have hc := e_9_19
    simp only [e_9_17, e_4_14, e_9_18] at hc
    linear_combination c540.trans k1 - hc

theorem publicBatchWrapper4_f84 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5923)) (a (.wire 8 79)) (a (.virt 23823)) (a (.virt 23824)) := by
  have c541 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (publicBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (publicBatchWrapper4_copies17 a h).1
  have c545 := (publicBatchWrapper4_copies17 a h).2.1
  have c546 := (publicBatchWrapper4_copies17 a h).2.2.1
  have c547 := (publicBatchWrapper4_copies17 a h).2.2.2.1
  have c548 := (publicBatchWrapper4_copies17 a h).2.2.2.2.1
  have c549 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.1
  have c550 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.1
  have c551 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_4_15 := arithEq_of_rows h (row := 4) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_15
  simp only [← c547, ← c548, ← c549] at e_4_15
  have e_4_16 := arithEq_of_rows h (row := 4) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_16
  simp only [← c550, ← c551, ← c552] at e_4_16
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c541, k0, ← c542, k0, ← c543] at e_10_0
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c544, ← c545, k0, ← c546] at e_10_1
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c553, ← c554, k0, ← c555] at e_10_2
  refine ⟨?_, ?_⟩
  · have hc := e_4_15
    simp only [e_10_1] at hc
    linear_combination c556.trans k1 - hc
  · have hc := e_10_2
    simp only [e_10_0, e_4_16, e_10_1] at hc
    linear_combination c557.trans k1 - hc

theorem publicBatchWrapper4_f85 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 5924)) (a (.wire 9 7)) (a (.virt 23825)) (a (.virt 23826)) := by
  have c558 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_4_17 := arithEq_of_rows h (row := 4) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_17
  simp only [← c564, ← c565, ← c566] at e_4_17
  have e_4_18 := arithEq_of_rows h (row := 4) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_18
  simp only [← c567, ← c568, ← c569] at e_4_18
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c558, k0, ← c559, k0, ← c560] at e_10_3
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c561, ← c562, k0, ← c563] at e_10_4
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c570, ← c571, k0, ← c572] at e_10_5
  refine ⟨?_, ?_⟩
  · have hc := e_4_17
    simp only [e_10_4] at hc
    linear_combination c573.trans k1 - hc
  · have hc := e_10_5
    simp only [e_10_3, e_4_18, e_10_4] at hc
    linear_combination c574.trans k1 - hc

theorem publicBatchWrapper4_f86 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 4 79) = band (a (.virt 23819)) (a (.virt 23821)) := by
  have c575 := (publicBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (publicBatchWrapper4_copies18 a h).1
  have c577 := (publicBatchWrapper4_copies18 a h).2.1
  have e_4_19 := arithEq_of_rows h (row := 4) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_19
  simp only [← c575, ← c576, ← c577] at e_4_19
  have hr := e_4_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f87 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 11 3) = band (a (.virt 23823)) (a (.virt 23825)) := by
  have c578 := (publicBatchWrapper4_copies18 a h).2.2.1
  have c579 := (publicBatchWrapper4_copies18 a h).2.2.2.1
  have c580 := (publicBatchWrapper4_copies18 a h).2.2.2.2.1
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c578, ← c579, ← c580] at e_11_0
  have hr := e_11_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f88 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 11 7) = band (a (.wire 4 79)) (a (.wire 11 3)) := by
  have c581 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.1
  have c582 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.1
  have c583 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.1
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c581, ← c582, ← c583] at e_11_1
  have hr := e_11_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f89 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 23) = bor (a (.wire 1 43)) (a (.wire 11 7)) := by
  have c584 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c584, ← c585, ← c586] at e_6_5
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c587, ← c588, k0, ← c589] at e_7_5
  have hr := e_7_5
  simp only [e_6_5] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f90 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 23) = a (.virt 23780) := by
  have c590 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c590

theorem publicBatchWrapper4_f91 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11863)) (a (.wire 9 23)) (a (.virt 23827)) (a (.virt 23828)) := by
  have c591 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c600 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (publicBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c591, k0, ← c592, k0, ← c593] at e_10_6
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c594, ← c595, k0, ← c596] at e_10_7
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c603, ← c604, k0, ← c605] at e_10_8
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c597, ← c598, ← c599] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c600, ← c601, ← c602] at e_11_3
  refine ⟨?_, ?_⟩
  · have hc := e_11_2
    simp only [e_10_7] at hc
    linear_combination c606.trans k1 - hc
  · have hc := e_10_8
    simp only [e_10_6, e_11_3, e_10_7] at hc
    linear_combination c607.trans k1 - hc

theorem publicBatchWrapper4_f92 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 27) = bor (a (.wire 2 7)) (a (.virt 23827)) := by
  have c608 := (publicBatchWrapper4_copies19 a h).1
  have c609 := (publicBatchWrapper4_copies19 a h).2.1
  have c610 := (publicBatchWrapper4_copies19 a h).2.2.1
  have c611 := (publicBatchWrapper4_copies19 a h).2.2.2.1
  have c612 := (publicBatchWrapper4_copies19 a h).2.2.2.2.1
  have c613 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c608, ← c609, ← c610] at e_6_6
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c611, ← c612, k0, ← c613] at e_7_6
  have hr := e_7_6
  simp only [e_6_6] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f93 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 27) = a (.virt 23780) := by
  have c614 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.1
  exact c614

theorem publicBatchWrapper4_f94 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11864)) (a (.wire 9 31)) (a (.virt 23829)) (a (.virt 23830)) := by
  have c615 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c615, k0, ← c616, k0, ← c617] at e_10_9
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c618, ← c619, k0, ← c620] at e_10_10
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c627, ← c628, k0, ← c629] at e_10_11
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c621, ← c622, ← c623] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c624, ← c625, ← c626] at e_11_5
  refine ⟨?_, ?_⟩
  · have hc := e_11_4
    simp only [e_10_10] at hc
    linear_combination c630.trans k1 - hc
  · have hc := e_10_11
    simp only [e_10_9, e_11_5, e_10_10] at hc
    linear_combination c631.trans k1 - hc

theorem publicBatchWrapper4_f95 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 31) = bor (a (.wire 2 7)) (a (.virt 23829)) := by
  have c632 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c632, ← c633, ← c634] at e_6_7
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c635, ← c636, k0, ← c637] at e_7_7
  have hr := e_7_7
  simp only [e_6_7] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f96 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 31) = a (.virt 23780) := by
  have c638 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c638

theorem publicBatchWrapper4_f97 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11865)) (a (.wire 8 63)) (a (.virt 23831)) (a (.virt 23832)) := by
  have c639 := (publicBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (publicBatchWrapper4_copies20 a h).1
  have c641 := (publicBatchWrapper4_copies20 a h).2.1
  have c642 := (publicBatchWrapper4_copies20 a h).2.2.1
  have c643 := (publicBatchWrapper4_copies20 a h).2.2.2.1
  have c644 := (publicBatchWrapper4_copies20 a h).2.2.2.2.1
  have c645 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.1
  have c646 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.1
  have c647 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c639, k0, ← c640, k0, ← c641] at e_10_12
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c642, ← c643, k0, ← c644] at e_10_13
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c651, ← c652, k0, ← c653] at e_10_14
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c645, ← c646, ← c647] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c648, ← c649, ← c650] at e_11_7
  refine ⟨?_, ?_⟩
  · have hc := e_11_6
    simp only [e_10_13] at hc
    linear_combination c654.trans k1 - hc
  · have hc := e_10_14
    simp only [e_10_12, e_11_7, e_10_13] at hc
    linear_combination c655.trans k1 - hc

theorem publicBatchWrapper4_f98 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11866)) (a (.wire 8 71)) (a (.virt 23833)) (a (.virt 23834)) := by
  have c656 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (publicBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (publicBatchWrapper4_copies21 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_10_15 := arithEq_of_rows h (row := 10) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_15
  simp only [← c656, k0, ← c657, k0, ← c658] at e_10_15
  have e_10_16 := arithEq_of_rows h (row := 10) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_16
  simp only [← c659, ← c660, k0, ← c661] at e_10_16
  have e_10_17 := arithEq_of_rows h (row := 10) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_17
  simp only [← c668, ← c669, k0, ← c670] at e_10_17
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c662, ← c663, ← c664] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c665, ← c666, ← c667] at e_11_9
  refine ⟨?_, ?_⟩
  · have hc := e_11_8
    simp only [e_10_16] at hc
    linear_combination c671.trans k1 - hc
  · have hc := e_10_17
    simp only [e_10_15, e_11_9, e_10_16] at hc
    linear_combination c672.trans k1 - hc

theorem publicBatchWrapper4_f99 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11867)) (a (.wire 8 79)) (a (.virt 23835)) (a (.virt 23836)) := by
  have c673 := (publicBatchWrapper4_copies21 a h).2.1
  have c674 := (publicBatchWrapper4_copies21 a h).2.2.1
  have c675 := (publicBatchWrapper4_copies21 a h).2.2.2.1
  have c676 := (publicBatchWrapper4_copies21 a h).2.2.2.2.1
  have c677 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.1
  have c678 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.1
  have c679 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_10_18 := arithEq_of_rows h (row := 10) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_18
  simp only [← c673, k0, ← c674, k0, ← c675] at e_10_18
  have e_10_19 := arithEq_of_rows h (row := 10) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_19
  simp only [← c676, ← c677, k0, ← c678] at e_10_19
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c679, ← c680, ← c681] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c682, ← c683, ← c684] at e_11_11
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c685, ← c686, k0, ← c687] at e_12_0
  refine ⟨?_, ?_⟩
  · have hc := e_11_10
    simp only [e_10_19] at hc
    linear_combination c688.trans k1 - hc
  · have hc := e_12_0
    simp only [e_10_18, e_11_11, e_10_19] at hc
    linear_combination c689.trans k1 - hc

theorem publicBatchWrapper4_f100 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 11868)) (a (.wire 9 7)) (a (.virt 23837)) (a (.virt 23838)) := by
  have c690 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (publicBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (publicBatchWrapper4_copies22 a h).1
  have c705 := (publicBatchWrapper4_copies22 a h).2.1
  have c706 := (publicBatchWrapper4_copies22 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c696, ← c697, ← c698] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c699, ← c700, ← c701] at e_11_13
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c690, k0, ← c691, k0, ← c692] at e_12_1
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c693, ← c694, k0, ← c695] at e_12_2
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c702, ← c703, k0, ← c704] at e_12_3
  refine ⟨?_, ?_⟩
  · have hc := e_11_12
    simp only [e_12_2] at hc
    linear_combination c705.trans k1 - hc
  · have hc := e_12_3
    simp only [e_12_1, e_11_13, e_12_2] at hc
    linear_combination c706.trans k1 - hc

theorem publicBatchWrapper4_f101 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 11 59) = band (a (.virt 23831)) (a (.virt 23833)) := by
  have c707 := (publicBatchWrapper4_copies22 a h).2.2.2.1
  have c708 := (publicBatchWrapper4_copies22 a h).2.2.2.2.1
  have c709 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.1
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c707, ← c708, ← c709] at e_11_14
  have hr := e_11_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f102 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 11 63) = band (a (.virt 23835)) (a (.virt 23837)) := by
  have c710 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.1
  have c711 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.1
  have e_11_15 := arithEq_of_rows h (row := 11) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_15
  simp only [← c710, ← c711, ← c712] at e_11_15
  have hr := e_11_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f103 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 11 67) = band (a (.wire 11 59)) (a (.wire 11 63)) := by
  have c713 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_11_16 := arithEq_of_rows h (row := 11) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_16
  simp only [← c713, ← c714, ← c715] at e_11_16
  have hr := e_11_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f104 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 35) = bor (a (.wire 2 7)) (a (.wire 11 67)) := by
  have c716 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c716, ← c717, ← c718] at e_6_8
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c719, ← c720, k0, ← c721] at e_7_8
  have hr := e_7_8
  simp only [e_6_8] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f105 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 35) = a (.virt 23780) := by
  have c722 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c722

theorem publicBatchWrapper4_f106 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17807)) (a (.wire 9 23)) (a (.virt 23839)) (a (.virt 23840)) := by
  have c723 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c733 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (publicBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c736 := (publicBatchWrapper4_copies23 a h).1
  have c737 := (publicBatchWrapper4_copies23 a h).2.1
  have c738 := (publicBatchWrapper4_copies23 a h).2.2.1
  have c739 := (publicBatchWrapper4_copies23 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_11_17 := arithEq_of_rows h (row := 11) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_17
  simp only [← c729, ← c730, ← c731] at e_11_17
  have e_11_18 := arithEq_of_rows h (row := 11) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_18
  simp only [← c732, ← c733, ← c734] at e_11_18
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c723, k0, ← c724, k0, ← c725] at e_12_4
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c726, ← c727, k0, ← c728] at e_12_5
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c735, ← c736, k0, ← c737] at e_12_6
  refine ⟨?_, ?_⟩
  · have hc := e_11_17
    simp only [e_12_5] at hc
    linear_combination c738.trans k1 - hc
  · have hc := e_12_6
    simp only [e_12_4, e_11_18, e_12_5] at hc
    linear_combination c739.trans k1 - hc

theorem publicBatchWrapper4_f107 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 39) = bor (a (.wire 2 51)) (a (.virt 23839)) := by
  have c740 := (publicBatchWrapper4_copies23 a h).2.2.2.2.1
  have c741 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.1
  have c742 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.1
  have c743 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.1
  have c745 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c740, ← c741, ← c742] at e_6_9
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c743, ← c744, k0, ← c745] at e_7_9
  have hr := e_7_9
  simp only [e_6_9] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f108 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 39) = a (.virt 23780) := by
  have c746 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c746

theorem publicBatchWrapper4_f109 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17808)) (a (.wire 9 31)) (a (.virt 23841)) (a (.virt 23842)) := by
  have c747 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c751 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c753 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c759 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_11_19 := arithEq_of_rows h (row := 11) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_19
  simp only [← c753, ← c754, ← c755] at e_11_19
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c747, k0, ← c748, k0, ← c749] at e_12_7
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c750, ← c751, k0, ← c752] at e_12_8
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c759, ← c760, k0, ← c761] at e_12_9
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c756, ← c757, ← c758] at e_13_0
  refine ⟨?_, ?_⟩
  · have hc := e_11_19
    simp only [e_12_8] at hc
    linear_combination c762.trans k1 - hc
  · have hc := e_12_9
    simp only [e_12_7, e_13_0, e_12_8] at hc
    linear_combination c763.trans k1 - hc

theorem publicBatchWrapper4_f110 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 43) = bor (a (.wire 2 51)) (a (.virt 23841)) := by
  have c764 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (publicBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (publicBatchWrapper4_copies24 a h).1
  have c769 := (publicBatchWrapper4_copies24 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c764, ← c765, ← c766] at e_6_10
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c767, ← c768, k0, ← c769] at e_7_10
  have hr := e_7_10
  simp only [e_6_10] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f111 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 43) = a (.virt 23780) := by
  have c770 := (publicBatchWrapper4_copies24 a h).2.2.1
  exact c770

theorem publicBatchWrapper4_f112 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17809)) (a (.wire 8 63)) (a (.virt 23843)) (a (.virt 23844)) := by
  have c771 := (publicBatchWrapper4_copies24 a h).2.2.2.1
  have c772 := (publicBatchWrapper4_copies24 a h).2.2.2.2.1
  have c773 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.1
  have c774 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.1
  have c775 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have c778 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c781 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c771, k0, ← c772, k0, ← c773] at e_12_10
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c774, ← c775, k0, ← c776] at e_12_11
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c783, ← c784, k0, ← c785] at e_12_12
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c777, ← c778, ← c779] at e_13_1
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c780, ← c781, ← c782] at e_13_2
  refine ⟨?_, ?_⟩
  · have hc := e_13_1
    simp only [e_12_11] at hc
    linear_combination c786.trans k1 - hc
  · have hc := e_12_12
    simp only [e_12_10, e_13_2, e_12_11] at hc
    linear_combination c787.trans k1 - hc

theorem publicBatchWrapper4_f113 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17810)) (a (.wire 8 71)) (a (.virt 23845)) (a (.virt 23846)) := by
  have c788 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (publicBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (publicBatchWrapper4_copies25 a h).1
  have c801 := (publicBatchWrapper4_copies25 a h).2.1
  have c802 := (publicBatchWrapper4_copies25 a h).2.2.1
  have c803 := (publicBatchWrapper4_copies25 a h).2.2.2.1
  have c804 := (publicBatchWrapper4_copies25 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c788, k0, ← c789, k0, ← c790] at e_12_13
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c791, ← c792, k0, ← c793] at e_12_14
  have e_12_15 := arithEq_of_rows h (row := 12) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_15
  simp only [← c800, ← c801, k0, ← c802] at e_12_15
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c794, ← c795, ← c796] at e_13_3
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c797, ← c798, ← c799] at e_13_4
  refine ⟨?_, ?_⟩
  · have hc := e_13_3
    simp only [e_12_14] at hc
    linear_combination c803.trans k1 - hc
  · have hc := e_12_15
    simp only [e_12_13, e_13_4, e_12_14] at hc
    linear_combination c804.trans k1 - hc

theorem publicBatchWrapper4_f114 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17811)) (a (.wire 8 79)) (a (.virt 23847)) (a (.virt 23848)) := by
  have c805 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.1
  have c806 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.1
  have c807 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c814 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_12_16 := arithEq_of_rows h (row := 12) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_16
  simp only [← c805, k0, ← c806, k0, ← c807] at e_12_16
  have e_12_17 := arithEq_of_rows h (row := 12) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_17
  simp only [← c808, ← c809, k0, ← c810] at e_12_17
  have e_12_18 := arithEq_of_rows h (row := 12) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_18
  simp only [← c817, ← c818, k0, ← c819] at e_12_18
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c811, ← c812, ← c813] at e_13_5
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c814, ← c815, ← c816] at e_13_6
  refine ⟨?_, ?_⟩
  · have hc := e_13_5
    simp only [e_12_17] at hc
    linear_combination c820.trans k1 - hc
  · have hc := e_12_18
    simp only [e_12_16, e_13_6, e_12_17] at hc
    linear_combination c821.trans k1 - hc

theorem publicBatchWrapper4_f115 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 17812)) (a (.wire 9 7)) (a (.virt 23849)) (a (.virt 23850)) := by
  have c822 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c829 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (publicBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c832 := (publicBatchWrapper4_copies26 a h).1
  have c833 := (publicBatchWrapper4_copies26 a h).2.1
  have c834 := (publicBatchWrapper4_copies26 a h).2.2.1
  have c835 := (publicBatchWrapper4_copies26 a h).2.2.2.1
  have c836 := (publicBatchWrapper4_copies26 a h).2.2.2.2.1
  have c837 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.1
  have c838 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_12_19 := arithEq_of_rows h (row := 12) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_19
  simp only [← c822, k0, ← c823, k0, ← c824] at e_12_19
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c828, ← c829, ← c830] at e_13_7
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c831, ← c832, ← c833] at e_13_8
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c825, ← c826, k0, ← c827] at e_14_0
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c834, ← c835, k0, ← c836] at e_14_1
  refine ⟨?_, ?_⟩
  · have hc := e_13_7
    simp only [e_14_0] at hc
    linear_combination c837.trans k1 - hc
  · have hc := e_14_1
    simp only [e_12_19, e_13_8, e_14_0] at hc
    linear_combination c838.trans k1 - hc

theorem publicBatchWrapper4_f116 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 13 39) = band (a (.virt 23843)) (a (.virt 23845)) := by
  have c839 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.1
  have c841 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c839, ← c840, ← c841] at e_13_9
  have hr := e_13_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f117 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 13 43) = band (a (.virt 23847)) (a (.virt 23849)) := by
  have c842 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c844 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c842, ← c843, ← c844] at e_13_10
  have hr := e_13_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f118 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 13 47) = band (a (.wire 13 39)) (a (.wire 13 43)) := by
  have c845 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c847 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c845, ← c846, ← c847] at e_13_11
  have hr := e_13_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f119 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 47) = bor (a (.wire 2 51)) (a (.wire 13 47)) := by
  have c848 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c853 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c848, ← c849, ← c850] at e_6_11
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c851, ← c852, k0, ← c853] at e_7_11
  have hr := e_7_11
  simp only [e_6_11] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f120 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 47) = a (.virt 23780) := by
  have c854 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c854

theorem publicBatchWrapper4_f121 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23751)) (a (.wire 9 23)) (a (.virt 23851)) (a (.virt 23852)) := by
  have c855 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c860 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c863 := (publicBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (publicBatchWrapper4_copies27 a h).1
  have c865 := (publicBatchWrapper4_copies27 a h).2.1
  have c866 := (publicBatchWrapper4_copies27 a h).2.2.1
  have c867 := (publicBatchWrapper4_copies27 a h).2.2.2.1
  have c868 := (publicBatchWrapper4_copies27 a h).2.2.2.2.1
  have c869 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.1
  have c870 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.1
  have c871 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c861, ← c862, ← c863] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c864, ← c865, ← c866] at e_13_13
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c855, k0, ← c856, k0, ← c857] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c858, ← c859, k0, ← c860] at e_14_3
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c867, ← c868, k0, ← c869] at e_14_4
  refine ⟨?_, ?_⟩
  · have hc := e_13_12
    simp only [e_14_3] at hc
    linear_combination c870.trans k1 - hc
  · have hc := e_14_4
    simp only [e_14_2, e_13_13, e_14_3] at hc
    linear_combination c871.trans k1 - hc

theorem publicBatchWrapper4_f122 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 51) = bor (a (.wire 4 15)) (a (.virt 23851)) := by
  have c872 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.1
  have c873 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c876 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c872, ← c873, ← c874] at e_6_12
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c875, ← c876, k0, ← c877] at e_7_12
  have hr := e_7_12
  simp only [e_6_12] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f123 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 51) = a (.virt 23780) := by
  have c878 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c878

theorem publicBatchWrapper4_f124 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23752)) (a (.wire 9 31)) (a (.virt 23853)) (a (.virt 23854)) := by
  have c879 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c880 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c883 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (publicBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c885, ← c886, ← c887] at e_13_14
  have e_13_15 := arithEq_of_rows h (row := 13) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_15
  simp only [← c888, ← c889, ← c890] at e_13_15
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c879, k0, ← c880, k0, ← c881] at e_14_5
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c882, ← c883, k0, ← c884] at e_14_6
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c891, ← c892, k0, ← c893] at e_14_7
  refine ⟨?_, ?_⟩
  · have hc := e_13_14
    simp only [e_14_6] at hc
    linear_combination c894.trans k1 - hc
  · have hc := e_14_7
    simp only [e_14_5, e_13_15, e_14_6] at hc
    linear_combination c895.trans k1 - hc

theorem publicBatchWrapper4_f125 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 55) = bor (a (.wire 4 15)) (a (.virt 23853)) := by
  have c896 := (publicBatchWrapper4_copies28 a h).1
  have c897 := (publicBatchWrapper4_copies28 a h).2.1
  have c898 := (publicBatchWrapper4_copies28 a h).2.2.1
  have c899 := (publicBatchWrapper4_copies28 a h).2.2.2.1
  have c900 := (publicBatchWrapper4_copies28 a h).2.2.2.2.1
  have c901 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c896, ← c897, ← c898] at e_6_13
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c899, ← c900, k0, ← c901] at e_7_13
  have hr := e_7_13
  simp only [e_6_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f126 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 55) = a (.virt 23780) := by
  have c902 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.1
  exact c902

theorem publicBatchWrapper4_f127 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23753)) (a (.wire 8 63)) (a (.virt 23855)) (a (.virt 23856)) := by
  have c903 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.1
  have c904 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have c907 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_13_16 := arithEq_of_rows h (row := 13) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_16
  simp only [← c909, ← c910, ← c911] at e_13_16
  have e_13_17 := arithEq_of_rows h (row := 13) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_17
  simp only [← c912, ← c913, ← c914] at e_13_17
  have e_14_8 := arithEq_of_rows h (row := 14) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c903, k0, ← c904, k0, ← c905] at e_14_8
  have e_14_9 := arithEq_of_rows h (row := 14) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c906, ← c907, k0, ← c908] at e_14_9
  have e_14_10 := arithEq_of_rows h (row := 14) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c915, ← c916, k0, ← c917] at e_14_10
  refine ⟨?_, ?_⟩
  · have hc := e_13_16
    simp only [e_14_9] at hc
    linear_combination c918.trans k1 - hc
  · have hc := e_14_10
    simp only [e_14_8, e_13_17, e_14_9] at hc
    linear_combination c919.trans k1 - hc

theorem publicBatchWrapper4_f128 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23754)) (a (.wire 8 71)) (a (.virt 23857)) (a (.virt 23858)) := by
  have c920 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c925 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c927 := (publicBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c928 := (publicBatchWrapper4_copies29 a h).1
  have c929 := (publicBatchWrapper4_copies29 a h).2.1
  have c930 := (publicBatchWrapper4_copies29 a h).2.2.1
  have c931 := (publicBatchWrapper4_copies29 a h).2.2.2.1
  have c932 := (publicBatchWrapper4_copies29 a h).2.2.2.2.1
  have c933 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.1
  have c934 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.1
  have c935 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_13_18 := arithEq_of_rows h (row := 13) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_18
  simp only [← c926, ← c927, ← c928] at e_13_18
  have e_13_19 := arithEq_of_rows h (row := 13) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_19
  simp only [← c929, ← c930, ← c931] at e_13_19
  have e_14_11 := arithEq_of_rows h (row := 14) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c920, k0, ← c921, k0, ← c922] at e_14_11
  have e_14_12 := arithEq_of_rows h (row := 14) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c923, ← c924, k0, ← c925] at e_14_12
  have e_14_13 := arithEq_of_rows h (row := 14) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c932, ← c933, k0, ← c934] at e_14_13
  refine ⟨?_, ?_⟩
  · have hc := e_13_18
    simp only [e_14_12] at hc
    linear_combination c935.trans k1 - hc
  · have hc := e_14_13
    simp only [e_14_11, e_13_19, e_14_12] at hc
    linear_combination c936.trans k1 - hc

theorem publicBatchWrapper4_f129 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23755)) (a (.wire 8 79)) (a (.virt 23859)) (a (.virt 23860)) := by
  have c937 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c947 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c953 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_14_14 := arithEq_of_rows h (row := 14) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c937, k0, ← c938, k0, ← c939] at e_14_14
  have e_14_15 := arithEq_of_rows h (row := 14) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_15
  simp only [← c940, ← c941, k0, ← c942] at e_14_15
  have e_14_16 := arithEq_of_rows h (row := 14) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_16
  simp only [← c949, ← c950, k0, ← c951] at e_14_16
  have e_15_0 := arithEq_of_rows h (row := 15) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c943, ← c944, ← c945] at e_15_0
  have e_15_1 := arithEq_of_rows h (row := 15) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c946, ← c947, ← c948] at e_15_1
  refine ⟨?_, ?_⟩
  · have hc := e_15_0
    simp only [e_14_15] at hc
    linear_combination c952.trans k1 - hc
  · have hc := e_14_16
    simp only [e_14_14, e_15_1, e_14_15] at hc
    linear_combination c953.trans k1 - hc

theorem publicBatchWrapper4_f130 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    IsEqual (a (.virt 23756)) (a (.wire 9 7)) (a (.virt 23861)) (a (.virt 23862)) := by
  have c954 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c956 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c959 := (publicBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (publicBatchWrapper4_copies30 a h).1
  have c961 := (publicBatchWrapper4_copies30 a h).2.1
  have c962 := (publicBatchWrapper4_copies30 a h).2.2.1
  have c963 := (publicBatchWrapper4_copies30 a h).2.2.2.1
  have c964 := (publicBatchWrapper4_copies30 a h).2.2.2.2.1
  have c965 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.1
  have c966 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.1
  have c967 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_14_17 := arithEq_of_rows h (row := 14) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_17
  simp only [← c954, k0, ← c955, k0, ← c956] at e_14_17
  have e_14_18 := arithEq_of_rows h (row := 14) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_18
  simp only [← c957, ← c958, k0, ← c959] at e_14_18
  have e_14_19 := arithEq_of_rows h (row := 14) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_19
  simp only [← c966, ← c967, k0, ← c968] at e_14_19
  have e_15_2 := arithEq_of_rows h (row := 15) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c960, ← c961, ← c962] at e_15_2
  have e_15_3 := arithEq_of_rows h (row := 15) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c963, ← c964, ← c965] at e_15_3
  refine ⟨?_, ?_⟩
  · have hc := e_15_2
    simp only [e_14_18] at hc
    linear_combination c969.trans k1 - hc
  · have hc := e_14_19
    simp only [e_14_17, e_15_3, e_14_18] at hc
    linear_combination c970.trans k1 - hc

theorem publicBatchWrapper4_f131 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 15 19) = band (a (.virt 23855)) (a (.virt 23857)) := by
  have c971 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c971, ← c972, ← c973] at e_15_4
  have hr := e_15_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f132 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 15 23) = band (a (.virt 23859)) (a (.virt 23861)) := by
  have c974 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_5 := arithEq_of_rows h (row := 15) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c974, ← c975, ← c976] at e_15_5
  have hr := e_15_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f133 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 15 27) = band (a (.wire 15 19)) (a (.wire 15 23)) := by
  have c977 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_6 := arithEq_of_rows h (row := 15) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c977, ← c978, ← c979] at e_15_6
  have hr := e_15_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper4_f134 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 59) = bor (a (.wire 4 15)) (a (.wire 15 27)) := by
  have c980 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c980, ← c981, ← c982] at e_6_14
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c983, ← c984, k0, ← c985] at e_7_14
  have hr := e_7_14
  simp only [e_6_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper4_f135 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 7 59) = a (.virt 23780) := by
  have c986 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c986

theorem publicBatchWrapper4_f136 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 7) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5926)) := by
  have c987 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c989 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c990 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (publicBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (publicBatchWrapper4_copies31 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c987, ← c988, ← c989] at e_16_0
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c990, ← c991, k1, ← c992] at e_16_1
  have hr := e_16_1
  simp only [e_16_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f137 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 15) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5927)) := by
  have c993 := (publicBatchWrapper4_copies31 a h).2.1
  have c994 := (publicBatchWrapper4_copies31 a h).2.2.1
  have c995 := (publicBatchWrapper4_copies31 a h).2.2.2.1
  have c996 := (publicBatchWrapper4_copies31 a h).2.2.2.2.1
  have c997 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.1
  have c998 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c993, ← c994, ← c995] at e_16_2
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c996, ← c997, k1, ← c998] at e_16_3
  have hr := e_16_3
  simp only [e_16_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f138 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 23) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5928)) := by
  have c999 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have c1002 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c999, ← c1000, ← c1001] at e_16_4
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c1002, ← c1003, k1, ← c1004] at e_16_5
  have hr := e_16_5
  simp only [e_16_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f139 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 31) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5929)) := by
  have c1005 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1008 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c1005, ← c1006, ← c1007] at e_16_6
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c1008, ← c1009, k1, ← c1010] at e_16_7
  have hr := e_16_7
  simp only [e_16_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f140 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 39) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5930)) := by
  have c1011 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1012 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c1011, ← c1012, ← c1013] at e_16_8
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c1014, ← c1015, k1, ← c1016] at e_16_9
  have hr := e_16_9
  simp only [e_16_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f141 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 47) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5931)) := by
  have c1017 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1019 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1022 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c1017, ← c1018, ← c1019] at e_16_10
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1020, ← c1021, k1, ← c1022] at e_16_11
  have hr := e_16_11
  simp only [e_16_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f142 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 55) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5932)) := by
  have c1023 := (publicBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1024 := (publicBatchWrapper4_copies32 a h).1
  have c1025 := (publicBatchWrapper4_copies32 a h).2.1
  have c1026 := (publicBatchWrapper4_copies32 a h).2.2.1
  have c1027 := (publicBatchWrapper4_copies32 a h).2.2.2.1
  have c1028 := (publicBatchWrapper4_copies32 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1023, ← c1024, ← c1025] at e_16_12
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1026, ← c1027, k1, ← c1028] at e_16_13
  have hr := e_16_13
  simp only [e_16_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f143 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 63) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5933)) := by
  have c1029 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.1
  have c1030 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1029, ← c1030, ← c1031] at e_16_14
  have e_16_15 := arithEq_of_rows h (row := 16) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_15
  simp only [← c1032, ← c1033, k1, ← c1034] at e_16_15
  have hr := e_16_15
  simp only [e_16_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f144 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 71) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5934)) := by
  have c1035 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_16 := arithEq_of_rows h (row := 16) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_16
  simp only [← c1035, ← c1036, ← c1037] at e_16_16
  have e_16_17 := arithEq_of_rows h (row := 16) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_17
  simp only [← c1038, ← c1039, k1, ← c1040] at e_16_17
  have hr := e_16_17
  simp only [e_16_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f145 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 16 79) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5935)) := by
  have c1041 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_16_18 := arithEq_of_rows h (row := 16) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_18
  simp only [← c1041, ← c1042, ← c1043] at e_16_18
  have e_16_19 := arithEq_of_rows h (row := 16) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_19
  simp only [← c1044, ← c1045, k1, ← c1046] at e_16_19
  have hr := e_16_19
  simp only [e_16_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f146 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 7) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11870)) := by
  have c1047 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_0 := arithEq_of_rows h (row := 17) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c1047, ← c1048, ← c1049] at e_17_0
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c1050, ← c1051, k1, ← c1052] at e_17_1
  have hr := e_17_1
  simp only [e_17_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f147 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 15) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11871)) := by
  have c1053 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (publicBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (publicBatchWrapper4_copies33 a h).1
  have c1057 := (publicBatchWrapper4_copies33 a h).2.1
  have c1058 := (publicBatchWrapper4_copies33 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_2 := arithEq_of_rows h (row := 17) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c1053, ← c1054, ← c1055] at e_17_2
  have e_17_3 := arithEq_of_rows h (row := 17) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c1056, ← c1057, k1, ← c1058] at e_17_3
  have hr := e_17_3
  simp only [e_17_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f148 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 23) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11872)) := by
  have c1059 := (publicBatchWrapper4_copies33 a h).2.2.2.1
  have c1060 := (publicBatchWrapper4_copies33 a h).2.2.2.2.1
  have c1061 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.1
  have c1062 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_4 := arithEq_of_rows h (row := 17) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c1059, ← c1060, ← c1061] at e_17_4
  have e_17_5 := arithEq_of_rows h (row := 17) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c1062, ← c1063, k1, ← c1064] at e_17_5
  have hr := e_17_5
  simp only [e_17_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f149 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 31) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11873)) := by
  have c1065 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_6 := arithEq_of_rows h (row := 17) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c1065, ← c1066, ← c1067] at e_17_6
  have e_17_7 := arithEq_of_rows h (row := 17) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c1068, ← c1069, k1, ← c1070] at e_17_7
  have hr := e_17_7
  simp only [e_17_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f150 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 39) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11874)) := by
  have c1071 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_8 := arithEq_of_rows h (row := 17) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c1071, ← c1072, ← c1073] at e_17_8
  have e_17_9 := arithEq_of_rows h (row := 17) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c1074, ← c1075, k1, ← c1076] at e_17_9
  have hr := e_17_9
  simp only [e_17_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f151 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 47) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11875)) := by
  have c1077 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1078 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_10 := arithEq_of_rows h (row := 17) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c1077, ← c1078, ← c1079] at e_17_10
  have e_17_11 := arithEq_of_rows h (row := 17) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c1080, ← c1081, k1, ← c1082] at e_17_11
  have hr := e_17_11
  simp only [e_17_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f152 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 55) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11876)) := by
  have c1083 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (publicBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (publicBatchWrapper4_copies34 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_12 := arithEq_of_rows h (row := 17) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c1083, ← c1084, ← c1085] at e_17_12
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c1086, ← c1087, k1, ← c1088] at e_17_13
  have hr := e_17_13
  simp only [e_17_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f153 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 63) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11877)) := by
  have c1089 := (publicBatchWrapper4_copies34 a h).2.1
  have c1090 := (publicBatchWrapper4_copies34 a h).2.2.1
  have c1091 := (publicBatchWrapper4_copies34 a h).2.2.2.1
  have c1092 := (publicBatchWrapper4_copies34 a h).2.2.2.2.1
  have c1093 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.1
  have c1094 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_14 := arithEq_of_rows h (row := 17) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c1089, ← c1090, ← c1091] at e_17_14
  have e_17_15 := arithEq_of_rows h (row := 17) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_15
  simp only [← c1092, ← c1093, k1, ← c1094] at e_17_15
  have hr := e_17_15
  simp only [e_17_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f154 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 71) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11878)) := by
  have c1095 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.1
  have c1096 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1099 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_16 := arithEq_of_rows h (row := 17) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_16
  simp only [← c1095, ← c1096, ← c1097] at e_17_16
  have e_17_17 := arithEq_of_rows h (row := 17) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_17
  simp only [← c1098, ← c1099, k1, ← c1100] at e_17_17
  have hr := e_17_17
  simp only [e_17_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f155 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 17 79) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11879)) := by
  have c1101 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1102 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1103 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1106 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_17_18 := arithEq_of_rows h (row := 17) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_18
  simp only [← c1101, ← c1102, ← c1103] at e_17_18
  have e_17_19 := arithEq_of_rows h (row := 17) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_19
  simp only [← c1104, ← c1105, k1, ← c1106] at e_17_19
  have hr := e_17_19
  simp only [e_17_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f156 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 7) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17814)) := by
  have c1107 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1108 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_0 := arithEq_of_rows h (row := 18) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c1107, ← c1108, ← c1109] at e_18_0
  have e_18_1 := arithEq_of_rows h (row := 18) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c1110, ← c1111, k1, ← c1112] at e_18_1
  have hr := e_18_1
  simp only [e_18_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f157 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 15) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17815)) := by
  have c1113 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_2 := arithEq_of_rows h (row := 18) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c1113, ← c1114, ← c1115] at e_18_2
  have e_18_3 := arithEq_of_rows h (row := 18) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c1116, ← c1117, k1, ← c1118] at e_18_3
  have hr := e_18_3
  simp only [e_18_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f158 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 23) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17816)) := by
  have c1119 := (publicBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1120 := (publicBatchWrapper4_copies35 a h).1
  have c1121 := (publicBatchWrapper4_copies35 a h).2.1
  have c1122 := (publicBatchWrapper4_copies35 a h).2.2.1
  have c1123 := (publicBatchWrapper4_copies35 a h).2.2.2.1
  have c1124 := (publicBatchWrapper4_copies35 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_4 := arithEq_of_rows h (row := 18) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c1119, ← c1120, ← c1121] at e_18_4
  have e_18_5 := arithEq_of_rows h (row := 18) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c1122, ← c1123, k1, ← c1124] at e_18_5
  have hr := e_18_5
  simp only [e_18_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f159 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 31) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17817)) := by
  have c1125 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.1
  have c1126 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_6 := arithEq_of_rows h (row := 18) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c1125, ← c1126, ← c1127] at e_18_6
  have e_18_7 := arithEq_of_rows h (row := 18) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c1128, ← c1129, k1, ← c1130] at e_18_7
  have hr := e_18_7
  simp only [e_18_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f160 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 39) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17818)) := by
  have c1131 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_8 := arithEq_of_rows h (row := 18) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c1131, ← c1132, ← c1133] at e_18_8
  have e_18_9 := arithEq_of_rows h (row := 18) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c1134, ← c1135, k1, ← c1136] at e_18_9
  have hr := e_18_9
  simp only [e_18_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f161 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 47) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17819)) := by
  have c1137 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1140 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c1137, ← c1138, ← c1139] at e_18_10
  have e_18_11 := arithEq_of_rows h (row := 18) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c1140, ← c1141, k1, ← c1142] at e_18_11
  have hr := e_18_11
  simp only [e_18_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f162 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 55) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17820)) := by
  have c1143 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_12 := arithEq_of_rows h (row := 18) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c1143, ← c1144, ← c1145] at e_18_12
  have e_18_13 := arithEq_of_rows h (row := 18) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c1146, ← c1147, k1, ← c1148] at e_18_13
  have hr := e_18_13
  simp only [e_18_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f163 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 63) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17821)) := by
  have c1149 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1150 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (publicBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (publicBatchWrapper4_copies36 a h).1
  have c1153 := (publicBatchWrapper4_copies36 a h).2.1
  have c1154 := (publicBatchWrapper4_copies36 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_14 := arithEq_of_rows h (row := 18) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c1149, ← c1150, ← c1151] at e_18_14
  have e_18_15 := arithEq_of_rows h (row := 18) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_15
  simp only [← c1152, ← c1153, k1, ← c1154] at e_18_15
  have hr := e_18_15
  simp only [e_18_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f164 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 71) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17822)) := by
  have c1155 := (publicBatchWrapper4_copies36 a h).2.2.2.1
  have c1156 := (publicBatchWrapper4_copies36 a h).2.2.2.2.1
  have c1157 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.1
  have c1158 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.1
  have c1159 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_16 := arithEq_of_rows h (row := 18) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_16
  simp only [← c1155, ← c1156, ← c1157] at e_18_16
  have e_18_17 := arithEq_of_rows h (row := 18) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_17
  simp only [← c1158, ← c1159, k1, ← c1160] at e_18_17
  have hr := e_18_17
  simp only [e_18_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f165 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 18 79) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17823)) := by
  have c1161 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have c1162 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_18_18 := arithEq_of_rows h (row := 18) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_18
  simp only [← c1161, ← c1162, ← c1163] at e_18_18
  have e_18_19 := arithEq_of_rows h (row := 18) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_19
  simp only [← c1164, ← c1165, k1, ← c1166] at e_18_19
  have hr := e_18_19
  simp only [e_18_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f166 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 7) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23758)) := by
  have c1167 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_0 := arithEq_of_rows h (row := 19) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c1167, ← c1168, ← c1169] at e_19_0
  have e_19_1 := arithEq_of_rows h (row := 19) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c1170, ← c1171, k1, ← c1172] at e_19_1
  have hr := e_19_1
  simp only [e_19_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f167 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 15) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23759)) := by
  have c1173 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1174 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1177 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_2 := arithEq_of_rows h (row := 19) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c1173, ← c1174, ← c1175] at e_19_2
  have e_19_3 := arithEq_of_rows h (row := 19) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c1176, ← c1177, k1, ← c1178] at e_19_3
  have hr := e_19_3
  simp only [e_19_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f168 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 23) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23760)) := by
  have c1179 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1180 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1182 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1183 := (publicBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (publicBatchWrapper4_copies37 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_4 := arithEq_of_rows h (row := 19) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c1179, ← c1180, ← c1181] at e_19_4
  have e_19_5 := arithEq_of_rows h (row := 19) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c1182, ← c1183, k1, ← c1184] at e_19_5
  have hr := e_19_5
  simp only [e_19_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f169 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 31) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23761)) := by
  have c1185 := (publicBatchWrapper4_copies37 a h).2.1
  have c1186 := (publicBatchWrapper4_copies37 a h).2.2.1
  have c1187 := (publicBatchWrapper4_copies37 a h).2.2.2.1
  have c1188 := (publicBatchWrapper4_copies37 a h).2.2.2.2.1
  have c1189 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.1
  have c1190 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_6 := arithEq_of_rows h (row := 19) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c1185, ← c1186, ← c1187] at e_19_6
  have e_19_7 := arithEq_of_rows h (row := 19) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c1188, ← c1189, k1, ← c1190] at e_19_7
  have hr := e_19_7
  simp only [e_19_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f170 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 39) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23762)) := by
  have c1191 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_8 := arithEq_of_rows h (row := 19) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c1191, ← c1192, ← c1193] at e_19_8
  have e_19_9 := arithEq_of_rows h (row := 19) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c1194, ← c1195, k1, ← c1196] at e_19_9
  have hr := e_19_9
  simp only [e_19_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f171 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 47) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23763)) := by
  have c1197 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_10 := arithEq_of_rows h (row := 19) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c1197, ← c1198, ← c1199] at e_19_10
  have e_19_11 := arithEq_of_rows h (row := 19) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c1200, ← c1201, k1, ← c1202] at e_19_11
  have hr := e_19_11
  simp only [e_19_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f172 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 55) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23764)) := by
  have c1203 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1207 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_12 := arithEq_of_rows h (row := 19) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c1203, ← c1204, ← c1205] at e_19_12
  have e_19_13 := arithEq_of_rows h (row := 19) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c1206, ← c1207, k1, ← c1208] at e_19_13
  have hr := e_19_13
  simp only [e_19_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f173 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 63) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23765)) := by
  have c1209 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_14 := arithEq_of_rows h (row := 19) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c1209, ← c1210, ← c1211] at e_19_14
  have e_19_15 := arithEq_of_rows h (row := 19) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_15
  simp only [← c1212, ← c1213, k1, ← c1214] at e_19_15
  have hr := e_19_15
  simp only [e_19_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f174 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 71) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23766)) := by
  have c1215 := (publicBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (publicBatchWrapper4_copies38 a h).1
  have c1217 := (publicBatchWrapper4_copies38 a h).2.1
  have c1218 := (publicBatchWrapper4_copies38 a h).2.2.1
  have c1219 := (publicBatchWrapper4_copies38 a h).2.2.2.1
  have c1220 := (publicBatchWrapper4_copies38 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_16 := arithEq_of_rows h (row := 19) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_16
  simp only [← c1215, ← c1216, ← c1217] at e_19_16
  have e_19_17 := arithEq_of_rows h (row := 19) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_17
  simp only [← c1218, ← c1219, k1, ← c1220] at e_19_17
  have hr := e_19_17
  simp only [e_19_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f175 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 19 79) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23767)) := by
  have c1221 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.1
  have c1222 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.1
  have c1224 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_19_18 := arithEq_of_rows h (row := 19) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_18
  simp only [← c1221, ← c1222, ← c1223] at e_19_18
  have e_19_19 := arithEq_of_rows h (row := 19) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_19
  simp only [← c1224, ← c1225, k1, ← c1226] at e_19_19
  have hr := e_19_19
  simp only [e_19_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f176 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 7) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5936)) := by
  have c1227 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_0 := arithEq_of_rows h (row := 20) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_0
  simp only [← c1227, ← c1228, ← c1229] at e_20_0
  have e_20_1 := arithEq_of_rows h (row := 20) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_1
  simp only [← c1230, ← c1231, k1, ← c1232] at e_20_1
  have hr := e_20_1
  simp only [e_20_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f177 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 15) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5937)) := by
  have c1233 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1236 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_2 := arithEq_of_rows h (row := 20) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_2
  simp only [← c1233, ← c1234, ← c1235] at e_20_2
  have e_20_3 := arithEq_of_rows h (row := 20) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_3
  simp only [← c1236, ← c1237, k1, ← c1238] at e_20_3
  have hr := e_20_3
  simp only [e_20_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f178 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 23) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5938)) := by
  have c1239 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1241 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_4 := arithEq_of_rows h (row := 20) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_4
  simp only [← c1239, ← c1240, ← c1241] at e_20_4
  have e_20_5 := arithEq_of_rows h (row := 20) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_5
  simp only [← c1242, ← c1243, k1, ← c1244] at e_20_5
  have hr := e_20_5
  simp only [e_20_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f179 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 31) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5939)) := by
  have c1245 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (publicBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (publicBatchWrapper4_copies39 a h).1
  have c1249 := (publicBatchWrapper4_copies39 a h).2.1
  have c1250 := (publicBatchWrapper4_copies39 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_6 := arithEq_of_rows h (row := 20) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_6
  simp only [← c1245, ← c1246, ← c1247] at e_20_6
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c1248, ← c1249, k1, ← c1250] at e_20_7
  have hr := e_20_7
  simp only [e_20_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f180 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 39) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5940)) := by
  have c1251 := (publicBatchWrapper4_copies39 a h).2.2.2.1
  have c1252 := (publicBatchWrapper4_copies39 a h).2.2.2.2.1
  have c1253 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.1
  have c1254 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.1
  have c1256 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_8 := arithEq_of_rows h (row := 20) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_8
  simp only [← c1251, ← c1252, ← c1253] at e_20_8
  have e_20_9 := arithEq_of_rows h (row := 20) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_9
  simp only [← c1254, ← c1255, k1, ← c1256] at e_20_9
  have hr := e_20_9
  simp only [e_20_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f181 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 47) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5941)) := by
  have c1257 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have c1258 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1261 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_10 := arithEq_of_rows h (row := 20) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_10
  simp only [← c1257, ← c1258, ← c1259] at e_20_10
  have e_20_11 := arithEq_of_rows h (row := 20) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_11
  simp only [← c1260, ← c1261, k1, ← c1262] at e_20_11
  have hr := e_20_11
  simp only [e_20_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f182 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 55) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5942)) := by
  have c1263 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1264 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1267 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_12 := arithEq_of_rows h (row := 20) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_12
  simp only [← c1263, ← c1264, ← c1265] at e_20_12
  have e_20_13 := arithEq_of_rows h (row := 20) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_13
  simp only [← c1266, ← c1267, k1, ← c1268] at e_20_13
  have hr := e_20_13
  simp only [e_20_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f183 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 63) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5943)) := by
  have c1269 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1270 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1271 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1274 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_14 := arithEq_of_rows h (row := 20) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_14
  simp only [← c1269, ← c1270, ← c1271] at e_20_14
  have e_20_15 := arithEq_of_rows h (row := 20) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_15
  simp only [← c1272, ← c1273, k1, ← c1274] at e_20_15
  have hr := e_20_15
  simp only [e_20_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f184 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 71) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11880)) := by
  have c1275 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (publicBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (publicBatchWrapper4_copies40 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_16 := arithEq_of_rows h (row := 20) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_16
  simp only [← c1275, ← c1276, ← c1277] at e_20_16
  have e_20_17 := arithEq_of_rows h (row := 20) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_17
  simp only [← c1278, ← c1279, k1, ← c1280] at e_20_17
  have hr := e_20_17
  simp only [e_20_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f185 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 20 79) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11881)) := by
  have c1281 := (publicBatchWrapper4_copies40 a h).2.1
  have c1282 := (publicBatchWrapper4_copies40 a h).2.2.1
  have c1283 := (publicBatchWrapper4_copies40 a h).2.2.2.1
  have c1284 := (publicBatchWrapper4_copies40 a h).2.2.2.2.1
  have c1285 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.1
  have c1286 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_20_18 := arithEq_of_rows h (row := 20) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_18
  simp only [← c1281, ← c1282, ← c1283] at e_20_18
  have e_20_19 := arithEq_of_rows h (row := 20) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_19
  simp only [← c1284, ← c1285, k1, ← c1286] at e_20_19
  have hr := e_20_19
  simp only [e_20_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f186 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 7) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11882)) := by
  have c1287 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1291 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_0 := arithEq_of_rows h (row := 21) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c1287, ← c1288, ← c1289] at e_21_0
  have e_21_1 := arithEq_of_rows h (row := 21) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c1290, ← c1291, k1, ← c1292] at e_21_1
  have hr := e_21_1
  simp only [e_21_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f187 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 15) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11883)) := by
  have c1293 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_2 := arithEq_of_rows h (row := 21) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c1293, ← c1294, ← c1295] at e_21_2
  have e_21_3 := arithEq_of_rows h (row := 21) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c1296, ← c1297, k1, ← c1298] at e_21_3
  have hr := e_21_3
  simp only [e_21_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f188 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 23) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11884)) := by
  have c1299 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_4 := arithEq_of_rows h (row := 21) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c1299, ← c1300, ← c1301] at e_21_4
  have e_21_5 := arithEq_of_rows h (row := 21) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c1302, ← c1303, k1, ← c1304] at e_21_5
  have hr := e_21_5
  simp only [e_21_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f189 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 31) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11885)) := by
  have c1305 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1308 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_6 := arithEq_of_rows h (row := 21) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c1305, ← c1306, ← c1307] at e_21_6
  have e_21_7 := arithEq_of_rows h (row := 21) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c1308, ← c1309, k1, ← c1310] at e_21_7
  have hr := e_21_7
  simp only [e_21_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f190 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 39) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11886)) := by
  have c1311 := (publicBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1312 := (publicBatchWrapper4_copies41 a h).1
  have c1313 := (publicBatchWrapper4_copies41 a h).2.1
  have c1314 := (publicBatchWrapper4_copies41 a h).2.2.1
  have c1315 := (publicBatchWrapper4_copies41 a h).2.2.2.1
  have c1316 := (publicBatchWrapper4_copies41 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_8 := arithEq_of_rows h (row := 21) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1311, ← c1312, ← c1313] at e_21_8
  have e_21_9 := arithEq_of_rows h (row := 21) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c1314, ← c1315, k1, ← c1316] at e_21_9
  have hr := e_21_9
  simp only [e_21_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f191 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 47) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11887)) := by
  have c1317 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.1
  have c1318 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.1
  have c1319 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.1
  have c1321 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have c1322 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_10 := arithEq_of_rows h (row := 21) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c1317, ← c1318, ← c1319] at e_21_10
  have e_21_11 := arithEq_of_rows h (row := 21) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c1320, ← c1321, k1, ← c1322] at e_21_11
  have hr := e_21_11
  simp only [e_21_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f192 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 55) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17824)) := by
  have c1323 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1325 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_12 := arithEq_of_rows h (row := 21) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c1323, ← c1324, ← c1325] at e_21_12
  have e_21_13 := arithEq_of_rows h (row := 21) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c1326, ← c1327, k1, ← c1328] at e_21_13
  have hr := e_21_13
  simp only [e_21_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f193 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 63) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17825)) := by
  have c1329 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_14 := arithEq_of_rows h (row := 21) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c1329, ← c1330, ← c1331] at e_21_14
  have e_21_15 := arithEq_of_rows h (row := 21) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_15
  simp only [← c1332, ← c1333, k1, ← c1334] at e_21_15
  have hr := e_21_15
  simp only [e_21_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f194 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 71) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17826)) := by
  have c1335 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_16 := arithEq_of_rows h (row := 21) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_16
  simp only [← c1335, ← c1336, ← c1337] at e_21_16
  have e_21_17 := arithEq_of_rows h (row := 21) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_17
  simp only [← c1338, ← c1339, k1, ← c1340] at e_21_17
  have hr := e_21_17
  simp only [e_21_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f195 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 21 79) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17827)) := by
  have c1341 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1342 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (publicBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (publicBatchWrapper4_copies42 a h).1
  have c1345 := (publicBatchWrapper4_copies42 a h).2.1
  have c1346 := (publicBatchWrapper4_copies42 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_21_18 := arithEq_of_rows h (row := 21) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_18
  simp only [← c1341, ← c1342, ← c1343] at e_21_18
  have e_21_19 := arithEq_of_rows h (row := 21) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_19
  simp only [← c1344, ← c1345, k1, ← c1346] at e_21_19
  have hr := e_21_19
  simp only [e_21_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f196 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 7) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17828)) := by
  have c1347 := (publicBatchWrapper4_copies42 a h).2.2.2.1
  have c1348 := (publicBatchWrapper4_copies42 a h).2.2.2.2.1
  have c1349 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.1
  have c1350 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.1
  have c1351 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.1
  have c1352 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_0 := arithEq_of_rows h (row := 22) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_0
  simp only [← c1347, ← c1348, ← c1349] at e_22_0
  have e_22_1 := arithEq_of_rows h (row := 22) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_1
  simp only [← c1350, ← c1351, k1, ← c1352] at e_22_1
  have hr := e_22_1
  simp only [e_22_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f197 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 15) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17829)) := by
  have c1353 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have c1354 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1355 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1358 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_2 := arithEq_of_rows h (row := 22) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_2
  simp only [← c1353, ← c1354, ← c1355] at e_22_2
  have e_22_3 := arithEq_of_rows h (row := 22) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_3
  simp only [← c1356, ← c1357, k1, ← c1358] at e_22_3
  have hr := e_22_3
  simp only [e_22_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f198 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 23) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17830)) := by
  have c1359 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1361 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1362 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1363 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_4 := arithEq_of_rows h (row := 22) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_4
  simp only [← c1359, ← c1360, ← c1361] at e_22_4
  have e_22_5 := arithEq_of_rows h (row := 22) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_5
  simp only [← c1362, ← c1363, k1, ← c1364] at e_22_5
  have hr := e_22_5
  simp only [e_22_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f199 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 31) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17831)) := by
  have c1365 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_6 := arithEq_of_rows h (row := 22) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_6
  simp only [← c1365, ← c1366, ← c1367] at e_22_6
  have e_22_7 := arithEq_of_rows h (row := 22) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_7
  simp only [← c1368, ← c1369, k1, ← c1370] at e_22_7
  have hr := e_22_7
  simp only [e_22_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f200 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 39) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23768)) := by
  have c1371 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1375 := (publicBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (publicBatchWrapper4_copies43 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_8 := arithEq_of_rows h (row := 22) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_8
  simp only [← c1371, ← c1372, ← c1373] at e_22_8
  have e_22_9 := arithEq_of_rows h (row := 22) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_9
  simp only [← c1374, ← c1375, k1, ← c1376] at e_22_9
  have hr := e_22_9
  simp only [e_22_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f201 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 47) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23769)) := by
  have c1377 := (publicBatchWrapper4_copies43 a h).2.1
  have c1378 := (publicBatchWrapper4_copies43 a h).2.2.1
  have c1379 := (publicBatchWrapper4_copies43 a h).2.2.2.1
  have c1380 := (publicBatchWrapper4_copies43 a h).2.2.2.2.1
  have c1381 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.1
  have c1382 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_10 := arithEq_of_rows h (row := 22) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_10
  simp only [← c1377, ← c1378, ← c1379] at e_22_10
  have e_22_11 := arithEq_of_rows h (row := 22) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c1380, ← c1381, k1, ← c1382] at e_22_11
  have hr := e_22_11
  simp only [e_22_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f202 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 55) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23770)) := by
  have c1383 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have c1386 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_12 := arithEq_of_rows h (row := 22) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_12
  simp only [← c1383, ← c1384, ← c1385] at e_22_12
  have e_22_13 := arithEq_of_rows h (row := 22) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_13
  simp only [← c1386, ← c1387, k1, ← c1388] at e_22_13
  have hr := e_22_13
  simp only [e_22_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f203 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 63) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23771)) := by
  have c1389 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1392 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_14 := arithEq_of_rows h (row := 22) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1389, ← c1390, ← c1391] at e_22_14
  have e_22_15 := arithEq_of_rows h (row := 22) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_15
  simp only [← c1392, ← c1393, k1, ← c1394] at e_22_15
  have hr := e_22_15
  simp only [e_22_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f204 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 71) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23772)) := by
  have c1395 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_16 := arithEq_of_rows h (row := 22) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_16
  simp only [← c1395, ← c1396, ← c1397] at e_22_16
  have e_22_17 := arithEq_of_rows h (row := 22) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_17
  simp only [← c1398, ← c1399, k1, ← c1400] at e_22_17
  have hr := e_22_17
  simp only [e_22_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f205 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 22 79) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23773)) := by
  have c1401 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_22_18 := arithEq_of_rows h (row := 22) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_18
  simp only [← c1401, ← c1402, ← c1403] at e_22_18
  have e_22_19 := arithEq_of_rows h (row := 22) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_19
  simp only [← c1404, ← c1405, k1, ← c1406] at e_22_19
  have hr := e_22_19
  simp only [e_22_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f206 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 23 7) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23774)) := by
  have c1407 := (publicBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (publicBatchWrapper4_copies44 a h).1
  have c1409 := (publicBatchWrapper4_copies44 a h).2.1
  have c1410 := (publicBatchWrapper4_copies44 a h).2.2.1
  have c1411 := (publicBatchWrapper4_copies44 a h).2.2.2.1
  have c1412 := (publicBatchWrapper4_copies44 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_23_0 := arithEq_of_rows h (row := 23) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_0
  simp only [← c1407, ← c1408, ← c1409] at e_23_0
  have e_23_1 := arithEq_of_rows h (row := 23) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_1
  simp only [← c1410, ← c1411, k1, ← c1412] at e_23_1
  have hr := e_23_1
  simp only [e_23_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper4_f207 (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    a (.wire 23 15) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23775)) := by
  have c1413 := (publicBatchWrapper4_copies44 a h).2.2.2.2.2.1
  have c1414 := (publicBatchWrapper4_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (publicBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (publicBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.1
  have c1417 := (publicBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (publicBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper4_consts a h
  have e_23_2 := arithEq_of_rows h (row := 23) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_2
  simp only [← c1413, ← c1414, ← c1415] at e_23_2
  have e_23_3 := arithEq_of_rows h (row := 23) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_23_3
  simp only [← c1416, ← c1417, k1, ← c1418] at e_23_3
  have hr := e_23_3
  simp only [e_23_2] at hr
  simp only [bselect, k1]
  linear_combination hr

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `publicBatchWrapper4` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem publicBatchWrapper4_decode (a : Assignment p) (h : Satisfies (publicBatchWrapper4 p) a) :
    (a (.virt 5918) = a (.virt 23782) ∧
    a (.virt 11862) = a (.virt 23782) ∧
    a (.virt 17806) = a (.virt 23782) ∧
    a (.virt 23750) = a (.virt 23782) ∧
    IsEqual (a (.virt 5921)) (a (.virt 23781)) (a (.virt 23783)) (a (.virt 23784)) ∧
    IsEqual (a (.virt 5922)) (a (.virt 23781)) (a (.virt 23785)) (a (.virt 23786)) ∧
    IsEqual (a (.virt 5923)) (a (.virt 23781)) (a (.virt 23787)) (a (.virt 23788)) ∧
    IsEqual (a (.virt 5924)) (a (.virt 23781)) (a (.virt 23789)) (a (.virt 23790)) ∧
    a (.wire 1 35) = band (a (.virt 23783)) (a (.virt 23785)) ∧
    a (.wire 1 39) = band (a (.virt 23787)) (a (.virt 23789)) ∧
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) ∧
    IsEqual (a (.virt 11865)) (a (.virt 23781)) (a (.virt 23791)) (a (.virt 23792)) ∧
    IsEqual (a (.virt 11866)) (a (.virt 23781)) (a (.virt 23793)) (a (.virt 23794)) ∧
    IsEqual (a (.virt 11867)) (a (.virt 23781)) (a (.virt 23795)) (a (.virt 23796)) ∧
    IsEqual (a (.virt 11868)) (a (.virt 23781)) (a (.virt 23797)) (a (.virt 23798)) ∧
    a (.wire 1 79) = band (a (.virt 23791)) (a (.virt 23793)) ∧
    a (.wire 2 3) = band (a (.virt 23795)) (a (.virt 23797)) ∧
    a (.wire 2 7) = band (a (.wire 1 79)) (a (.wire 2 3)) ∧
    IsEqual (a (.virt 17809)) (a (.virt 23781)) (a (.virt 23799)) (a (.virt 23800)) ∧
    IsEqual (a (.virt 17810)) (a (.virt 23781)) (a (.virt 23801)) (a (.virt 23802)) ∧
    IsEqual (a (.virt 17811)) (a (.virt 23781)) (a (.virt 23803)) (a (.virt 23804)) ∧
    IsEqual (a (.virt 17812)) (a (.virt 23781)) (a (.virt 23805)) (a (.virt 23806)) ∧
    a (.wire 2 43) = band (a (.virt 23799)) (a (.virt 23801)) ∧
    a (.wire 2 47) = band (a (.virt 23803)) (a (.virt 23805)) ∧
    a (.wire 2 51) = band (a (.wire 2 43)) (a (.wire 2 47)) ∧
    IsEqual (a (.virt 23753)) (a (.virt 23781)) (a (.virt 23807)) (a (.virt 23808)) ∧
    IsEqual (a (.virt 23754)) (a (.virt 23781)) (a (.virt 23809)) (a (.virt 23810)) ∧
    IsEqual (a (.virt 23755)) (a (.virt 23781)) (a (.virt 23811)) (a (.virt 23812)) ∧
    IsEqual (a (.virt 23756)) (a (.virt 23781)) (a (.virt 23813)) (a (.virt 23814)) ∧
    a (.wire 4 7) = band (a (.virt 23807)) (a (.virt 23809)) ∧
    a (.wire 4 11) = band (a (.virt 23811)) (a (.virt 23813)) ∧
    a (.wire 4 15) = band (a (.wire 4 7)) (a (.wire 4 11))) ∧
    (a (.wire 3 51) = bnot (a (.wire 1 43)) ∧
    a (.virt 23780) = bnot (a (.virt 23781)) ∧
    a (.wire 3 51) = band (a (.wire 3 51)) (a (.virt 23780)) ∧
    a (.wire 3 55) = bselect (a (.wire 3 51)) (a (.virt 5921)) (a (.virt 23781)) ∧
    a (.wire 3 59) = bselect (a (.wire 3 51)) (a (.virt 5922)) (a (.virt 23781)) ∧
    a (.wire 3 63) = bselect (a (.wire 3 51)) (a (.virt 5923)) (a (.virt 23781)) ∧
    a (.wire 3 67) = bselect (a (.wire 3 51)) (a (.virt 5924)) (a (.virt 23781)) ∧
    a (.wire 3 71) = bselect (a (.wire 3 51)) (a (.virt 5925)) (a (.virt 23781)) ∧
    a (.wire 3 75) = bselect (a (.wire 3 51)) (a (.virt 5919)) (a (.virt 23781)) ∧
    a (.wire 3 79) = bselect (a (.wire 3 51)) (a (.virt 5920)) (a (.virt 23781)) ∧
    a (.wire 3 51) = bor (a (.virt 23781)) (a (.wire 3 51)) ∧
    a (.wire 5 3) = bnot (a (.wire 2 7)) ∧
    a (.wire 5 7) = bnot (a (.wire 3 51)) ∧
    a (.wire 4 19) = band (a (.wire 5 3)) (a (.wire 5 7)) ∧
    a (.wire 5 15) = bselect (a (.wire 4 19)) (a (.virt 11865)) (a (.wire 3 55)) ∧
    a (.wire 5 23) = bselect (a (.wire 4 19)) (a (.virt 11866)) (a (.wire 3 59)) ∧
    a (.wire 5 31) = bselect (a (.wire 4 19)) (a (.virt 11867)) (a (.wire 3 63)) ∧
    a (.wire 5 39) = bselect (a (.wire 4 19)) (a (.virt 11868)) (a (.wire 3 67)) ∧
    a (.wire 5 47) = bselect (a (.wire 4 19)) (a (.virt 11869)) (a (.wire 3 71)) ∧
    a (.wire 5 55) = bselect (a (.wire 4 19)) (a (.virt 11863)) (a (.wire 3 75)) ∧
    a (.wire 5 63) = bselect (a (.wire 4 19)) (a (.virt 11864)) (a (.wire 3 79)) ∧
    a (.wire 7 3) = bor (a (.wire 3 51)) (a (.wire 5 3)) ∧
    a (.wire 5 67) = bnot (a (.wire 2 51)) ∧
    a (.wire 5 71) = bnot (a (.wire 7 3)) ∧
    a (.wire 4 23) = band (a (.wire 5 67)) (a (.wire 5 71)) ∧
    a (.wire 5 79) = bselect (a (.wire 4 23)) (a (.virt 17809)) (a (.wire 5 15)) ∧
    a (.wire 8 7) = bselect (a (.wire 4 23)) (a (.virt 17810)) (a (.wire 5 23)) ∧
    a (.wire 8 15) = bselect (a (.wire 4 23)) (a (.virt 17811)) (a (.wire 5 31)) ∧
    a (.wire 8 23) = bselect (a (.wire 4 23)) (a (.virt 17812)) (a (.wire 5 39)) ∧
    a (.wire 8 31) = bselect (a (.wire 4 23)) (a (.virt 17813)) (a (.wire 5 47)) ∧
    a (.wire 8 39) = bselect (a (.wire 4 23)) (a (.virt 17807)) (a (.wire 5 55)) ∧
    a (.wire 8 47) = bselect (a (.wire 4 23)) (a (.virt 17808)) (a (.wire 5 63))) ∧
    (a (.wire 7 7) = bor (a (.wire 7 3)) (a (.wire 5 67)) ∧
    a (.wire 8 51) = bnot (a (.wire 4 15)) ∧
    a (.wire 8 55) = bnot (a (.wire 7 7)) ∧
    a (.wire 4 27) = band (a (.wire 8 51)) (a (.wire 8 55)) ∧
    a (.wire 8 63) = bselect (a (.wire 4 27)) (a (.virt 23753)) (a (.wire 5 79)) ∧
    a (.wire 8 71) = bselect (a (.wire 4 27)) (a (.virt 23754)) (a (.wire 8 7)) ∧
    a (.wire 8 79) = bselect (a (.wire 4 27)) (a (.virt 23755)) (a (.wire 8 15)) ∧
    a (.wire 9 7) = bselect (a (.wire 4 27)) (a (.virt 23756)) (a (.wire 8 23)) ∧
    a (.wire 9 15) = bselect (a (.wire 4 27)) (a (.virt 23757)) (a (.wire 8 31)) ∧
    a (.wire 9 23) = bselect (a (.wire 4 27)) (a (.virt 23751)) (a (.wire 8 39)) ∧
    a (.wire 9 31) = bselect (a (.wire 4 27)) (a (.virt 23752)) (a (.wire 8 47)) ∧
    a (.wire 7 11) = bor (a (.wire 7 7)) (a (.wire 8 51)) ∧
    IsEqual (a (.virt 5919)) (a (.wire 9 23)) (a (.virt 23815)) (a (.virt 23816)) ∧
    a (.wire 7 15) = bor (a (.wire 1 43)) (a (.virt 23815)) ∧
    a (.wire 7 15) = a (.virt 23780) ∧
    IsEqual (a (.virt 5920)) (a (.wire 9 31)) (a (.virt 23817)) (a (.virt 23818)) ∧
    a (.wire 7 19) = bor (a (.wire 1 43)) (a (.virt 23817)) ∧
    a (.wire 7 19) = a (.virt 23780) ∧
    IsEqual (a (.virt 5921)) (a (.wire 8 63)) (a (.virt 23819)) (a (.virt 23820)) ∧
    IsEqual (a (.virt 5922)) (a (.wire 8 71)) (a (.virt 23821)) (a (.virt 23822)) ∧
    IsEqual (a (.virt 5923)) (a (.wire 8 79)) (a (.virt 23823)) (a (.virt 23824)) ∧
    IsEqual (a (.virt 5924)) (a (.wire 9 7)) (a (.virt 23825)) (a (.virt 23826)) ∧
    a (.wire 4 79) = band (a (.virt 23819)) (a (.virt 23821)) ∧
    a (.wire 11 3) = band (a (.virt 23823)) (a (.virt 23825)) ∧
    a (.wire 11 7) = band (a (.wire 4 79)) (a (.wire 11 3)) ∧
    a (.wire 7 23) = bor (a (.wire 1 43)) (a (.wire 11 7)) ∧
    a (.wire 7 23) = a (.virt 23780) ∧
    IsEqual (a (.virt 11863)) (a (.wire 9 23)) (a (.virt 23827)) (a (.virt 23828)) ∧
    a (.wire 7 27) = bor (a (.wire 2 7)) (a (.virt 23827)) ∧
    a (.wire 7 27) = a (.virt 23780) ∧
    IsEqual (a (.virt 11864)) (a (.wire 9 31)) (a (.virt 23829)) (a (.virt 23830)) ∧
    a (.wire 7 31) = bor (a (.wire 2 7)) (a (.virt 23829))) ∧
    (a (.wire 7 31) = a (.virt 23780) ∧
    IsEqual (a (.virt 11865)) (a (.wire 8 63)) (a (.virt 23831)) (a (.virt 23832)) ∧
    IsEqual (a (.virt 11866)) (a (.wire 8 71)) (a (.virt 23833)) (a (.virt 23834)) ∧
    IsEqual (a (.virt 11867)) (a (.wire 8 79)) (a (.virt 23835)) (a (.virt 23836)) ∧
    IsEqual (a (.virt 11868)) (a (.wire 9 7)) (a (.virt 23837)) (a (.virt 23838)) ∧
    a (.wire 11 59) = band (a (.virt 23831)) (a (.virt 23833)) ∧
    a (.wire 11 63) = band (a (.virt 23835)) (a (.virt 23837)) ∧
    a (.wire 11 67) = band (a (.wire 11 59)) (a (.wire 11 63)) ∧
    a (.wire 7 35) = bor (a (.wire 2 7)) (a (.wire 11 67)) ∧
    a (.wire 7 35) = a (.virt 23780) ∧
    IsEqual (a (.virt 17807)) (a (.wire 9 23)) (a (.virt 23839)) (a (.virt 23840)) ∧
    a (.wire 7 39) = bor (a (.wire 2 51)) (a (.virt 23839)) ∧
    a (.wire 7 39) = a (.virt 23780) ∧
    IsEqual (a (.virt 17808)) (a (.wire 9 31)) (a (.virt 23841)) (a (.virt 23842)) ∧
    a (.wire 7 43) = bor (a (.wire 2 51)) (a (.virt 23841)) ∧
    a (.wire 7 43) = a (.virt 23780) ∧
    IsEqual (a (.virt 17809)) (a (.wire 8 63)) (a (.virt 23843)) (a (.virt 23844)) ∧
    IsEqual (a (.virt 17810)) (a (.wire 8 71)) (a (.virt 23845)) (a (.virt 23846)) ∧
    IsEqual (a (.virt 17811)) (a (.wire 8 79)) (a (.virt 23847)) (a (.virt 23848)) ∧
    IsEqual (a (.virt 17812)) (a (.wire 9 7)) (a (.virt 23849)) (a (.virt 23850)) ∧
    a (.wire 13 39) = band (a (.virt 23843)) (a (.virt 23845)) ∧
    a (.wire 13 43) = band (a (.virt 23847)) (a (.virt 23849)) ∧
    a (.wire 13 47) = band (a (.wire 13 39)) (a (.wire 13 43)) ∧
    a (.wire 7 47) = bor (a (.wire 2 51)) (a (.wire 13 47)) ∧
    a (.wire 7 47) = a (.virt 23780) ∧
    IsEqual (a (.virt 23751)) (a (.wire 9 23)) (a (.virt 23851)) (a (.virt 23852)) ∧
    a (.wire 7 51) = bor (a (.wire 4 15)) (a (.virt 23851)) ∧
    a (.wire 7 51) = a (.virt 23780) ∧
    IsEqual (a (.virt 23752)) (a (.wire 9 31)) (a (.virt 23853)) (a (.virt 23854)) ∧
    a (.wire 7 55) = bor (a (.wire 4 15)) (a (.virt 23853)) ∧
    a (.wire 7 55) = a (.virt 23780) ∧
    IsEqual (a (.virt 23753)) (a (.wire 8 63)) (a (.virt 23855)) (a (.virt 23856))) ∧
    (IsEqual (a (.virt 23754)) (a (.wire 8 71)) (a (.virt 23857)) (a (.virt 23858)) ∧
    IsEqual (a (.virt 23755)) (a (.wire 8 79)) (a (.virt 23859)) (a (.virt 23860)) ∧
    IsEqual (a (.virt 23756)) (a (.wire 9 7)) (a (.virt 23861)) (a (.virt 23862)) ∧
    a (.wire 15 19) = band (a (.virt 23855)) (a (.virt 23857)) ∧
    a (.wire 15 23) = band (a (.virt 23859)) (a (.virt 23861)) ∧
    a (.wire 15 27) = band (a (.wire 15 19)) (a (.wire 15 23)) ∧
    a (.wire 7 59) = bor (a (.wire 4 15)) (a (.wire 15 27)) ∧
    a (.wire 7 59) = a (.virt 23780) ∧
    a (.wire 16 7) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5926)) ∧
    a (.wire 16 15) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5927)) ∧
    a (.wire 16 23) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5928)) ∧
    a (.wire 16 31) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5929)) ∧
    a (.wire 16 39) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5930)) ∧
    a (.wire 16 47) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5931)) ∧
    a (.wire 16 55) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5932)) ∧
    a (.wire 16 63) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5933)) ∧
    a (.wire 16 71) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5934)) ∧
    a (.wire 16 79) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5935)) ∧
    a (.wire 17 7) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11870)) ∧
    a (.wire 17 15) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11871)) ∧
    a (.wire 17 23) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11872)) ∧
    a (.wire 17 31) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11873)) ∧
    a (.wire 17 39) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11874)) ∧
    a (.wire 17 47) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11875)) ∧
    a (.wire 17 55) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11876)) ∧
    a (.wire 17 63) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11877)) ∧
    a (.wire 17 71) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11878)) ∧
    a (.wire 17 79) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11879)) ∧
    a (.wire 18 7) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17814)) ∧
    a (.wire 18 15) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17815)) ∧
    a (.wire 18 23) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17816)) ∧
    a (.wire 18 31) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17817))) ∧
    (a (.wire 18 39) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17818)) ∧
    a (.wire 18 47) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17819)) ∧
    a (.wire 18 55) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17820)) ∧
    a (.wire 18 63) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17821)) ∧
    a (.wire 18 71) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17822)) ∧
    a (.wire 18 79) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17823)) ∧
    a (.wire 19 7) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23758)) ∧
    a (.wire 19 15) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23759)) ∧
    a (.wire 19 23) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23760)) ∧
    a (.wire 19 31) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23761)) ∧
    a (.wire 19 39) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23762)) ∧
    a (.wire 19 47) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23763)) ∧
    a (.wire 19 55) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23764)) ∧
    a (.wire 19 63) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23765)) ∧
    a (.wire 19 71) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23766)) ∧
    a (.wire 19 79) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23767)) ∧
    a (.wire 20 7) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5936)) ∧
    a (.wire 20 15) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5937)) ∧
    a (.wire 20 23) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5938)) ∧
    a (.wire 20 31) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5939)) ∧
    a (.wire 20 39) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5940)) ∧
    a (.wire 20 47) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5941)) ∧
    a (.wire 20 55) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5942)) ∧
    a (.wire 20 63) = bselect (a (.wire 1 43)) (a (.virt 23781)) (a (.virt 5943)) ∧
    a (.wire 20 71) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11880)) ∧
    a (.wire 20 79) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11881)) ∧
    a (.wire 21 7) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11882)) ∧
    a (.wire 21 15) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11883)) ∧
    a (.wire 21 23) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11884)) ∧
    a (.wire 21 31) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11885)) ∧
    a (.wire 21 39) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11886)) ∧
    a (.wire 21 47) = bselect (a (.wire 2 7)) (a (.virt 23781)) (a (.virt 11887))) ∧
    (a (.wire 21 55) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17824)) ∧
    a (.wire 21 63) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17825)) ∧
    a (.wire 21 71) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17826)) ∧
    a (.wire 21 79) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17827)) ∧
    a (.wire 22 7) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17828)) ∧
    a (.wire 22 15) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17829)) ∧
    a (.wire 22 23) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17830)) ∧
    a (.wire 22 31) = bselect (a (.wire 2 51)) (a (.virt 23781)) (a (.virt 17831)) ∧
    a (.wire 22 39) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23768)) ∧
    a (.wire 22 47) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23769)) ∧
    a (.wire 22 55) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23770)) ∧
    a (.wire 22 63) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23771)) ∧
    a (.wire 22 71) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23772)) ∧
    a (.wire 22 79) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23773)) ∧
    a (.wire 23 7) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23774)) ∧
    a (.wire 23 15) = bselect (a (.wire 4 15)) (a (.virt 23781)) (a (.virt 23775))) :=
  ⟨⟨publicBatchWrapper4_f0 a h, publicBatchWrapper4_f1 a h, publicBatchWrapper4_f2 a h, publicBatchWrapper4_f3 a h, publicBatchWrapper4_f4 a h, publicBatchWrapper4_f5 a h, publicBatchWrapper4_f6 a h, publicBatchWrapper4_f7 a h, publicBatchWrapper4_f8 a h, publicBatchWrapper4_f9 a h, publicBatchWrapper4_f10 a h, publicBatchWrapper4_f11 a h, publicBatchWrapper4_f12 a h, publicBatchWrapper4_f13 a h, publicBatchWrapper4_f14 a h, publicBatchWrapper4_f15 a h, publicBatchWrapper4_f16 a h, publicBatchWrapper4_f17 a h, publicBatchWrapper4_f18 a h, publicBatchWrapper4_f19 a h, publicBatchWrapper4_f20 a h, publicBatchWrapper4_f21 a h, publicBatchWrapper4_f22 a h, publicBatchWrapper4_f23 a h, publicBatchWrapper4_f24 a h, publicBatchWrapper4_f25 a h, publicBatchWrapper4_f26 a h, publicBatchWrapper4_f27 a h, publicBatchWrapper4_f28 a h, publicBatchWrapper4_f29 a h, publicBatchWrapper4_f30 a h, publicBatchWrapper4_f31 a h⟩, ⟨publicBatchWrapper4_f32 a h, publicBatchWrapper4_f33 a h, publicBatchWrapper4_f34 a h, publicBatchWrapper4_f35 a h, publicBatchWrapper4_f36 a h, publicBatchWrapper4_f37 a h, publicBatchWrapper4_f38 a h, publicBatchWrapper4_f39 a h, publicBatchWrapper4_f40 a h, publicBatchWrapper4_f41 a h, publicBatchWrapper4_f42 a h, publicBatchWrapper4_f43 a h, publicBatchWrapper4_f44 a h, publicBatchWrapper4_f45 a h, publicBatchWrapper4_f46 a h, publicBatchWrapper4_f47 a h, publicBatchWrapper4_f48 a h, publicBatchWrapper4_f49 a h, publicBatchWrapper4_f50 a h, publicBatchWrapper4_f51 a h, publicBatchWrapper4_f52 a h, publicBatchWrapper4_f53 a h, publicBatchWrapper4_f54 a h, publicBatchWrapper4_f55 a h, publicBatchWrapper4_f56 a h, publicBatchWrapper4_f57 a h, publicBatchWrapper4_f58 a h, publicBatchWrapper4_f59 a h, publicBatchWrapper4_f60 a h, publicBatchWrapper4_f61 a h, publicBatchWrapper4_f62 a h, publicBatchWrapper4_f63 a h⟩, ⟨publicBatchWrapper4_f64 a h, publicBatchWrapper4_f65 a h, publicBatchWrapper4_f66 a h, publicBatchWrapper4_f67 a h, publicBatchWrapper4_f68 a h, publicBatchWrapper4_f69 a h, publicBatchWrapper4_f70 a h, publicBatchWrapper4_f71 a h, publicBatchWrapper4_f72 a h, publicBatchWrapper4_f73 a h, publicBatchWrapper4_f74 a h, publicBatchWrapper4_f75 a h, publicBatchWrapper4_f76 a h, publicBatchWrapper4_f77 a h, publicBatchWrapper4_f78 a h, publicBatchWrapper4_f79 a h, publicBatchWrapper4_f80 a h, publicBatchWrapper4_f81 a h, publicBatchWrapper4_f82 a h, publicBatchWrapper4_f83 a h, publicBatchWrapper4_f84 a h, publicBatchWrapper4_f85 a h, publicBatchWrapper4_f86 a h, publicBatchWrapper4_f87 a h, publicBatchWrapper4_f88 a h, publicBatchWrapper4_f89 a h, publicBatchWrapper4_f90 a h, publicBatchWrapper4_f91 a h, publicBatchWrapper4_f92 a h, publicBatchWrapper4_f93 a h, publicBatchWrapper4_f94 a h, publicBatchWrapper4_f95 a h⟩, ⟨publicBatchWrapper4_f96 a h, publicBatchWrapper4_f97 a h, publicBatchWrapper4_f98 a h, publicBatchWrapper4_f99 a h, publicBatchWrapper4_f100 a h, publicBatchWrapper4_f101 a h, publicBatchWrapper4_f102 a h, publicBatchWrapper4_f103 a h, publicBatchWrapper4_f104 a h, publicBatchWrapper4_f105 a h, publicBatchWrapper4_f106 a h, publicBatchWrapper4_f107 a h, publicBatchWrapper4_f108 a h, publicBatchWrapper4_f109 a h, publicBatchWrapper4_f110 a h, publicBatchWrapper4_f111 a h, publicBatchWrapper4_f112 a h, publicBatchWrapper4_f113 a h, publicBatchWrapper4_f114 a h, publicBatchWrapper4_f115 a h, publicBatchWrapper4_f116 a h, publicBatchWrapper4_f117 a h, publicBatchWrapper4_f118 a h, publicBatchWrapper4_f119 a h, publicBatchWrapper4_f120 a h, publicBatchWrapper4_f121 a h, publicBatchWrapper4_f122 a h, publicBatchWrapper4_f123 a h, publicBatchWrapper4_f124 a h, publicBatchWrapper4_f125 a h, publicBatchWrapper4_f126 a h, publicBatchWrapper4_f127 a h⟩, ⟨publicBatchWrapper4_f128 a h, publicBatchWrapper4_f129 a h, publicBatchWrapper4_f130 a h, publicBatchWrapper4_f131 a h, publicBatchWrapper4_f132 a h, publicBatchWrapper4_f133 a h, publicBatchWrapper4_f134 a h, publicBatchWrapper4_f135 a h, publicBatchWrapper4_f136 a h, publicBatchWrapper4_f137 a h, publicBatchWrapper4_f138 a h, publicBatchWrapper4_f139 a h, publicBatchWrapper4_f140 a h, publicBatchWrapper4_f141 a h, publicBatchWrapper4_f142 a h, publicBatchWrapper4_f143 a h, publicBatchWrapper4_f144 a h, publicBatchWrapper4_f145 a h, publicBatchWrapper4_f146 a h, publicBatchWrapper4_f147 a h, publicBatchWrapper4_f148 a h, publicBatchWrapper4_f149 a h, publicBatchWrapper4_f150 a h, publicBatchWrapper4_f151 a h, publicBatchWrapper4_f152 a h, publicBatchWrapper4_f153 a h, publicBatchWrapper4_f154 a h, publicBatchWrapper4_f155 a h, publicBatchWrapper4_f156 a h, publicBatchWrapper4_f157 a h, publicBatchWrapper4_f158 a h, publicBatchWrapper4_f159 a h⟩, ⟨publicBatchWrapper4_f160 a h, publicBatchWrapper4_f161 a h, publicBatchWrapper4_f162 a h, publicBatchWrapper4_f163 a h, publicBatchWrapper4_f164 a h, publicBatchWrapper4_f165 a h, publicBatchWrapper4_f166 a h, publicBatchWrapper4_f167 a h, publicBatchWrapper4_f168 a h, publicBatchWrapper4_f169 a h, publicBatchWrapper4_f170 a h, publicBatchWrapper4_f171 a h, publicBatchWrapper4_f172 a h, publicBatchWrapper4_f173 a h, publicBatchWrapper4_f174 a h, publicBatchWrapper4_f175 a h, publicBatchWrapper4_f176 a h, publicBatchWrapper4_f177 a h, publicBatchWrapper4_f178 a h, publicBatchWrapper4_f179 a h, publicBatchWrapper4_f180 a h, publicBatchWrapper4_f181 a h, publicBatchWrapper4_f182 a h, publicBatchWrapper4_f183 a h, publicBatchWrapper4_f184 a h, publicBatchWrapper4_f185 a h, publicBatchWrapper4_f186 a h, publicBatchWrapper4_f187 a h, publicBatchWrapper4_f188 a h, publicBatchWrapper4_f189 a h, publicBatchWrapper4_f190 a h, publicBatchWrapper4_f191 a h⟩, ⟨publicBatchWrapper4_f192 a h, publicBatchWrapper4_f193 a h, publicBatchWrapper4_f194 a h, publicBatchWrapper4_f195 a h, publicBatchWrapper4_f196 a h, publicBatchWrapper4_f197 a h, publicBatchWrapper4_f198 a h, publicBatchWrapper4_f199 a h, publicBatchWrapper4_f200 a h, publicBatchWrapper4_f201 a h, publicBatchWrapper4_f202 a h, publicBatchWrapper4_f203 a h, publicBatchWrapper4_f204 a h, publicBatchWrapper4_f205 a h, publicBatchWrapper4_f206 a h, publicBatchWrapper4_f207 a h⟩⟩

end Plonky2Spec.Generated
