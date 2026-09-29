/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from the `n = 2` private-batch wrapper trace
  (`qp-zk-circuits/formal/traces/private_batch_wrapper_n2.json`, recorded by
  `TracingBuilder` while `build_private_batch_constraints` ran on the real builder): the
  pre-`build` constraint system and the gadget calls recorded while building it.
  Each theorem's proof is generated too, one block per recorded gadget call, from
  the ops and copy constraints the builder emitted for it.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.WiringSponge

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash)

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

/-- `privateBatchWrapper2.copies`, items `0..32`. -/
def privateBatchWrapper2.copies0 : List (Target × Target) := [
    (.virt 18978, .wire 0 0),
    (.virt 18978, .wire 0 1),
    (.virt 18981, .wire 0 2),
    (.virt 18981, .wire 1 0),
    (.virt 9479, .wire 1 1),
    (.virt 18981, .wire 1 2),
    (.virt 9479, .wire 1 4),
    (.virt 18982, .wire 1 5),
    (.virt 9479, .wire 1 6),
    (.wire 1 7, .wire 0 4),
    (.virt 18978, .wire 0 5),
    (.wire 0 3, .wire 0 6),
    (.wire 1 3, .virt 18979),
    (.wire 0 7, .virt 18979),
    (.virt 18978, .wire 0 8),
    (.virt 18978, .wire 0 9),
    (.virt 18983, .wire 0 10),
    (.virt 18983, .wire 1 8),
    (.virt 9480, .wire 1 9),
    (.virt 18983, .wire 1 10),
    (.virt 9480, .wire 1 12),
    (.virt 18984, .wire 1 13),
    (.virt 9480, .wire 1 14),
    (.wire 1 15, .wire 0 12),
    (.virt 18978, .wire 0 13),
    (.wire 0 11, .wire 0 14),
    (.wire 1 11, .virt 18979),
    (.wire 0 15, .virt 18979),
    (.virt 18978, .wire 0 16),
    (.virt 18978, .wire 0 17),
    (.virt 18985, .wire 0 18),
    (.virt 18985, .wire 1 16)
  ]

/-- `privateBatchWrapper2.copies`, items `32..64`. -/
def privateBatchWrapper2.copies1 : List (Target × Target) := [
    (.virt 9481, .wire 1 17),
    (.virt 18985, .wire 1 18),
    (.virt 9481, .wire 1 20),
    (.virt 18986, .wire 1 21),
    (.virt 9481, .wire 1 22),
    (.wire 1 23, .wire 0 20),
    (.virt 18978, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.wire 1 19, .virt 18979),
    (.wire 0 23, .virt 18979),
    (.virt 18978, .wire 0 24),
    (.virt 18978, .wire 0 25),
    (.virt 18987, .wire 0 26),
    (.virt 18987, .wire 1 24),
    (.virt 9482, .wire 1 25),
    (.virt 18987, .wire 1 26),
    (.virt 9482, .wire 1 28),
    (.virt 18988, .wire 1 29),
    (.virt 9482, .wire 1 30),
    (.wire 1 31, .wire 0 28),
    (.virt 18978, .wire 0 29),
    (.wire 0 27, .wire 0 30),
    (.wire 1 27, .virt 18979),
    (.wire 0 31, .virt 18979),
    (.virt 18981, .wire 1 32),
    (.virt 18983, .wire 1 33),
    (.virt 18981, .wire 1 34),
    (.virt 18985, .wire 1 36),
    (.virt 18987, .wire 1 37),
    (.virt 18985, .wire 1 38),
    (.wire 1 35, .wire 1 40),
    (.wire 1 39, .wire 1 41)
  ]

/-- `privateBatchWrapper2.copies`, items `64..96`. -/
def privateBatchWrapper2.copies2 : List (Target × Target) := [
    (.wire 1 35, .wire 1 42),
    (.virt 18978, .wire 0 32),
    (.virt 18978, .wire 0 33),
    (.virt 18989, .wire 0 34),
    (.virt 18989, .wire 1 44),
    (.virt 18964, .wire 1 45),
    (.virt 18989, .wire 1 46),
    (.virt 18964, .wire 1 48),
    (.virt 18990, .wire 1 49),
    (.virt 18964, .wire 1 50),
    (.wire 1 51, .wire 0 36),
    (.virt 18978, .wire 0 37),
    (.wire 0 35, .wire 0 38),
    (.wire 1 47, .virt 18979),
    (.wire 0 39, .virt 18979),
    (.virt 18978, .wire 0 40),
    (.virt 18978, .wire 0 41),
    (.virt 18991, .wire 0 42),
    (.virt 18991, .wire 1 52),
    (.virt 18965, .wire 1 53),
    (.virt 18991, .wire 1 54),
    (.virt 18965, .wire 1 56),
    (.virt 18992, .wire 1 57),
    (.virt 18965, .wire 1 58),
    (.wire 1 59, .wire 0 44),
    (.virt 18978, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.wire 1 55, .virt 18979),
    (.wire 0 47, .virt 18979),
    (.virt 18978, .wire 0 48),
    (.virt 18978, .wire 0 49),
    (.virt 18993, .wire 0 50)
  ]

/-- `privateBatchWrapper2.copies`, items `96..128`. -/
def privateBatchWrapper2.copies3 : List (Target × Target) := [
    (.virt 18993, .wire 2 0),
    (.virt 18966, .wire 2 1),
    (.virt 18993, .wire 2 2),
    (.virt 18966, .wire 2 4),
    (.virt 18994, .wire 2 5),
    (.virt 18966, .wire 2 6),
    (.wire 2 7, .wire 0 52),
    (.virt 18978, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.wire 2 3, .virt 18979),
    (.wire 0 55, .virt 18979),
    (.virt 18978, .wire 0 56),
    (.virt 18978, .wire 0 57),
    (.virt 18995, .wire 0 58),
    (.virt 18995, .wire 2 8),
    (.virt 18967, .wire 2 9),
    (.virt 18995, .wire 2 10),
    (.virt 18967, .wire 2 12),
    (.virt 18996, .wire 2 13),
    (.virt 18967, .wire 2 14),
    (.wire 2 15, .wire 3 0),
    (.virt 18978, .wire 3 1),
    (.wire 0 59, .wire 3 2),
    (.wire 2 11, .virt 18979),
    (.wire 3 3, .virt 18979),
    (.virt 18989, .wire 2 16),
    (.virt 18991, .wire 2 17),
    (.virt 18989, .wire 2 18),
    (.virt 18993, .wire 2 20),
    (.virt 18995, .wire 2 21),
    (.virt 18993, .wire 2 22),
    (.wire 2 19, .wire 2 24)
  ]

/-- `privateBatchWrapper2.copies`, items `128..160`. -/
def privateBatchWrapper2.copies4 : List (Target × Target) := [
    (.wire 2 23, .wire 2 25),
    (.wire 2 19, .wire 2 26),
    (.virt 18978, .wire 3 4),
    (.virt 18978, .wire 3 5),
    (.wire 1 43, .wire 3 6),
    (.wire 3 7, .wire 3 8),
    (.virt 9479, .wire 3 9),
    (.virt 18979, .wire 3 10),
    (.wire 3 7, .wire 3 12),
    (.virt 9480, .wire 3 13),
    (.virt 18979, .wire 3 14),
    (.wire 3 7, .wire 3 16),
    (.virt 9481, .wire 3 17),
    (.virt 18979, .wire 3 18),
    (.wire 3 7, .wire 3 20),
    (.virt 9482, .wire 3 21),
    (.virt 18979, .wire 3 22),
    (.wire 3 7, .wire 3 24),
    (.virt 9483, .wire 3 25),
    (.virt 18979, .wire 3 26),
    (.wire 3 7, .wire 3 28),
    (.virt 9466, .wire 3 29),
    (.virt 18979, .wire 3 30),
    (.virt 18978, .wire 3 32),
    (.virt 18978, .wire 3 33),
    (.wire 2 27, .wire 3 34),
    (.virt 18978, .wire 3 36),
    (.virt 18978, .wire 3 37),
    (.wire 3 7, .wire 3 38),
    (.wire 3 35, .wire 2 28),
    (.wire 3 39, .wire 2 29),
    (.wire 3 35, .wire 2 30)
  ]

/-- `privateBatchWrapper2.copies`, items `160..192`. -/
def privateBatchWrapper2.copies5 : List (Target × Target) := [
    (.wire 2 31, .wire 3 40),
    (.wire 3 11, .wire 3 41),
    (.wire 3 11, .wire 3 42),
    (.wire 2 31, .wire 3 44),
    (.virt 18964, .wire 3 45),
    (.wire 3 43, .wire 3 46),
    (.wire 2 31, .wire 3 48),
    (.wire 3 15, .wire 3 49),
    (.wire 3 15, .wire 3 50),
    (.wire 2 31, .wire 3 52),
    (.virt 18965, .wire 3 53),
    (.wire 3 51, .wire 3 54),
    (.wire 2 31, .wire 3 56),
    (.wire 3 19, .wire 3 57),
    (.wire 3 19, .wire 3 58),
    (.wire 2 31, .wire 4 0),
    (.virt 18966, .wire 4 1),
    (.wire 3 59, .wire 4 2),
    (.wire 2 31, .wire 4 4),
    (.wire 3 23, .wire 4 5),
    (.wire 3 23, .wire 4 6),
    (.wire 2 31, .wire 4 8),
    (.virt 18967, .wire 4 9),
    (.wire 4 7, .wire 4 10),
    (.wire 2 31, .wire 4 12),
    (.wire 3 27, .wire 4 13),
    (.wire 3 27, .wire 4 14),
    (.wire 2 31, .wire 4 16),
    (.virt 18968, .wire 4 17),
    (.wire 4 15, .wire 4 18),
    (.wire 2 31, .wire 4 20),
    (.wire 3 31, .wire 4 21)
  ]

/-- `privateBatchWrapper2.copies`, items `192..224`. -/
def privateBatchWrapper2.copies6 : List (Target × Target) := [
    (.wire 3 31, .wire 4 22),
    (.wire 2 31, .wire 4 24),
    (.virt 18951, .wire 4 25),
    (.wire 4 23, .wire 4 26),
    (.wire 3 7, .wire 5 0),
    (.wire 3 35, .wire 5 1),
    (.wire 3 7, .wire 5 2),
    (.wire 5 3, .wire 6 0),
    (.virt 18978, .wire 6 1),
    (.wire 3 35, .wire 6 2),
    (.virt 18978, .wire 4 28),
    (.virt 18978, .wire 4 29),
    (.virt 18997, .wire 4 30),
    (.virt 9479, .wire 4 32),
    (.virt 18978, .wire 4 33),
    (.wire 3 47, .wire 4 34),
    (.virt 18997, .wire 2 32),
    (.wire 4 35, .wire 2 33),
    (.virt 18997, .wire 2 34),
    (.wire 4 35, .wire 2 36),
    (.virt 18998, .wire 2 37),
    (.wire 4 35, .wire 2 38),
    (.wire 2 39, .wire 4 36),
    (.virt 18978, .wire 4 37),
    (.wire 4 31, .wire 4 38),
    (.wire 2 35, .virt 18979),
    (.wire 4 39, .virt 18979),
    (.virt 18978, .wire 4 40),
    (.virt 18978, .wire 4 41),
    (.virt 18999, .wire 4 42),
    (.virt 9480, .wire 4 44),
    (.virt 18978, .wire 4 45)
  ]

/-- `privateBatchWrapper2.copies`, items `224..256`. -/
def privateBatchWrapper2.copies7 : List (Target × Target) := [
    (.wire 3 55, .wire 4 46),
    (.virt 18999, .wire 2 40),
    (.wire 4 47, .wire 2 41),
    (.virt 18999, .wire 2 42),
    (.wire 4 47, .wire 2 44),
    (.virt 19000, .wire 2 45),
    (.wire 4 47, .wire 2 46),
    (.wire 2 47, .wire 4 48),
    (.virt 18978, .wire 4 49),
    (.wire 4 43, .wire 4 50),
    (.wire 2 43, .virt 18979),
    (.wire 4 51, .virt 18979),
    (.virt 18978, .wire 4 52),
    (.virt 18978, .wire 4 53),
    (.virt 19001, .wire 4 54),
    (.virt 9481, .wire 4 56),
    (.virt 18978, .wire 4 57),
    (.wire 4 3, .wire 4 58),
    (.virt 19001, .wire 2 48),
    (.wire 4 59, .wire 2 49),
    (.virt 19001, .wire 2 50),
    (.wire 4 59, .wire 2 52),
    (.virt 19002, .wire 2 53),
    (.wire 4 59, .wire 2 54),
    (.wire 2 55, .wire 7 0),
    (.virt 18978, .wire 7 1),
    (.wire 4 55, .wire 7 2),
    (.wire 2 51, .virt 18979),
    (.wire 7 3, .virt 18979),
    (.virt 18978, .wire 7 4),
    (.virt 18978, .wire 7 5),
    (.virt 19003, .wire 7 6)
  ]

/-- `privateBatchWrapper2.copies`, items `256..288`. -/
def privateBatchWrapper2.copies8 : List (Target × Target) := [
    (.virt 9482, .wire 7 8),
    (.virt 18978, .wire 7 9),
    (.wire 4 11, .wire 7 10),
    (.virt 19003, .wire 2 56),
    (.wire 7 11, .wire 2 57),
    (.virt 19003, .wire 2 58),
    (.wire 7 11, .wire 8 0),
    (.virt 19004, .wire 8 1),
    (.wire 7 11, .wire 8 2),
    (.wire 8 3, .wire 7 12),
    (.virt 18978, .wire 7 13),
    (.wire 7 7, .wire 7 14),
    (.wire 2 59, .virt 18979),
    (.wire 7 15, .virt 18979),
    (.virt 18997, .wire 8 4),
    (.virt 18999, .wire 8 5),
    (.virt 18997, .wire 8 6),
    (.virt 19001, .wire 8 8),
    (.virt 19003, .wire 8 9),
    (.virt 19001, .wire 8 10),
    (.wire 8 7, .wire 8 12),
    (.wire 8 11, .wire 8 13),
    (.wire 8 7, .wire 8 14),
    (.wire 1 43, .wire 5 4),
    (.wire 8 15, .wire 5 5),
    (.wire 1 43, .wire 5 6),
    (.wire 5 7, .wire 6 4),
    (.virt 18978, .wire 6 5),
    (.wire 8 15, .wire 6 6),
    (.wire 6 7, .virt 18978),
    (.virt 9463, .virt 9463),
    (.virt 18978, .wire 7 16)
  ]

/-- `privateBatchWrapper2.copies`, items `288..320`. -/
def privateBatchWrapper2.copies9 : List (Target × Target) := [
    (.virt 18978, .wire 7 17),
    (.virt 19005, .wire 7 18),
    (.virt 9466, .wire 7 20),
    (.virt 18978, .wire 7 21),
    (.wire 4 27, .wire 7 22),
    (.virt 19005, .wire 8 16),
    (.wire 7 23, .wire 8 17),
    (.virt 19005, .wire 8 18),
    (.wire 7 23, .wire 8 20),
    (.virt 19006, .wire 8 21),
    (.wire 7 23, .wire 8 22),
    (.wire 8 23, .wire 7 24),
    (.virt 18978, .wire 7 25),
    (.wire 7 19, .wire 7 26),
    (.wire 8 19, .virt 18979),
    (.wire 7 27, .virt 18979),
    (.wire 1 43, .wire 5 8),
    (.virt 19005, .wire 5 9),
    (.wire 1 43, .wire 5 10),
    (.wire 5 11, .wire 6 8),
    (.virt 18978, .wire 6 9),
    (.virt 19005, .wire 6 10),
    (.wire 6 11, .virt 18978),
    (.virt 18978, .wire 7 28),
    (.virt 18978, .wire 7 29),
    (.virt 19007, .wire 7 30),
    (.virt 18964, .wire 7 32),
    (.virt 18978, .wire 7 33),
    (.wire 3 47, .wire 7 34),
    (.virt 19007, .wire 8 24),
    (.wire 7 35, .wire 8 25),
    (.virt 19007, .wire 8 26)
  ]

/-- `privateBatchWrapper2.copies`, items `320..352`. -/
def privateBatchWrapper2.copies10 : List (Target × Target) := [
    (.wire 7 35, .wire 8 28),
    (.virt 19008, .wire 8 29),
    (.wire 7 35, .wire 8 30),
    (.wire 8 31, .wire 7 36),
    (.virt 18978, .wire 7 37),
    (.wire 7 31, .wire 7 38),
    (.wire 8 27, .virt 18979),
    (.wire 7 39, .virt 18979),
    (.virt 18978, .wire 7 40),
    (.virt 18978, .wire 7 41),
    (.virt 19009, .wire 7 42),
    (.virt 18965, .wire 7 44),
    (.virt 18978, .wire 7 45),
    (.wire 3 55, .wire 7 46),
    (.virt 19009, .wire 8 32),
    (.wire 7 47, .wire 8 33),
    (.virt 19009, .wire 8 34),
    (.wire 7 47, .wire 8 36),
    (.virt 19010, .wire 8 37),
    (.wire 7 47, .wire 8 38),
    (.wire 8 39, .wire 7 48),
    (.virt 18978, .wire 7 49),
    (.wire 7 43, .wire 7 50),
    (.wire 8 35, .virt 18979),
    (.wire 7 51, .virt 18979),
    (.virt 18978, .wire 7 52),
    (.virt 18978, .wire 7 53),
    (.virt 19011, .wire 7 54),
    (.virt 18966, .wire 7 56),
    (.virt 18978, .wire 7 57),
    (.wire 4 3, .wire 7 58),
    (.virt 19011, .wire 8 40)
  ]

/-- `privateBatchWrapper2.copies`, items `352..384`. -/
def privateBatchWrapper2.copies11 : List (Target × Target) := [
    (.wire 7 59, .wire 8 41),
    (.virt 19011, .wire 8 42),
    (.wire 7 59, .wire 8 44),
    (.virt 19012, .wire 8 45),
    (.wire 7 59, .wire 8 46),
    (.wire 8 47, .wire 9 0),
    (.virt 18978, .wire 9 1),
    (.wire 7 55, .wire 9 2),
    (.wire 8 43, .virt 18979),
    (.wire 9 3, .virt 18979),
    (.virt 18978, .wire 9 4),
    (.virt 18978, .wire 9 5),
    (.virt 19013, .wire 9 6),
    (.virt 18967, .wire 9 8),
    (.virt 18978, .wire 9 9),
    (.wire 4 11, .wire 9 10),
    (.virt 19013, .wire 8 48),
    (.wire 9 11, .wire 8 49),
    (.virt 19013, .wire 8 50),
    (.wire 9 11, .wire 8 52),
    (.virt 19014, .wire 8 53),
    (.wire 9 11, .wire 8 54),
    (.wire 8 55, .wire 9 12),
    (.virt 18978, .wire 9 13),
    (.wire 9 7, .wire 9 14),
    (.wire 8 51, .virt 18979),
    (.wire 9 15, .virt 18979),
    (.virt 19007, .wire 8 56),
    (.virt 19009, .wire 8 57),
    (.virt 19007, .wire 8 58),
    (.virt 19011, .wire 10 0),
    (.virt 19013, .wire 10 1)
  ]

/-- `privateBatchWrapper2.copies`, items `384..416`. -/
def privateBatchWrapper2.copies12 : List (Target × Target) := [
    (.virt 19011, .wire 10 2),
    (.wire 8 59, .wire 10 4),
    (.wire 10 3, .wire 10 5),
    (.wire 8 59, .wire 10 6),
    (.wire 2 27, .wire 5 12),
    (.wire 10 7, .wire 5 13),
    (.wire 2 27, .wire 5 14),
    (.wire 5 15, .wire 6 12),
    (.virt 18978, .wire 6 13),
    (.wire 10 7, .wire 6 14),
    (.wire 6 15, .virt 18978),
    (.virt 18948, .virt 9463),
    (.virt 18978, .wire 9 16),
    (.virt 18978, .wire 9 17),
    (.virt 19015, .wire 9 18),
    (.virt 18951, .wire 9 20),
    (.virt 18978, .wire 9 21),
    (.wire 4 27, .wire 9 22),
    (.virt 19015, .wire 10 8),
    (.wire 9 23, .wire 10 9),
    (.virt 19015, .wire 10 10),
    (.wire 9 23, .wire 10 12),
    (.virt 19016, .wire 10 13),
    (.wire 9 23, .wire 10 14),
    (.wire 10 15, .wire 9 24),
    (.virt 18978, .wire 9 25),
    (.wire 9 19, .wire 9 26),
    (.wire 10 11, .virt 18979),
    (.wire 9 27, .virt 18979),
    (.wire 2 27, .wire 5 16),
    (.virt 19015, .wire 5 17),
    (.wire 2 27, .wire 5 18)
  ]

/-- `privateBatchWrapper2.copies`, items `416..448`. -/
def privateBatchWrapper2.copies13 : List (Target × Target) := [
    (.wire 5 19, .wire 6 16),
    (.virt 18978, .wire 6 17),
    (.virt 19015, .wire 6 18),
    (.wire 6 19, .virt 18978),
    (.wire 1 43, .wire 9 28),
    (.virt 9471, .wire 9 29),
    (.virt 9471, .wire 9 30),
    (.wire 1 43, .wire 9 32),
    (.virt 18979, .wire 9 33),
    (.wire 9 31, .wire 9 34),
    (.wire 1 43, .wire 9 36),
    (.virt 9472, .wire 9 37),
    (.virt 9472, .wire 9 38),
    (.wire 1 43, .wire 9 40),
    (.virt 18979, .wire 9 41),
    (.wire 9 39, .wire 9 42),
    (.wire 1 43, .wire 9 44),
    (.virt 9473, .wire 9 45),
    (.virt 9473, .wire 9 46),
    (.wire 1 43, .wire 9 48),
    (.virt 18979, .wire 9 49),
    (.wire 9 47, .wire 9 50),
    (.wire 1 43, .wire 9 52),
    (.virt 9474, .wire 9 53),
    (.virt 9474, .wire 9 54),
    (.wire 1 43, .wire 9 56),
    (.virt 18979, .wire 9 57),
    (.wire 9 55, .wire 9 58),
    (.wire 1 43, .wire 11 0),
    (.virt 9464, .wire 11 1),
    (.virt 9464, .wire 11 2),
    (.wire 1 43, .wire 11 4)
  ]

/-- `privateBatchWrapper2.copies`, items `448..480`. -/
def privateBatchWrapper2.copies14 : List (Target × Target) := [
    (.virt 18979, .wire 11 5),
    (.wire 11 3, .wire 11 6),
    (.wire 1 43, .wire 11 8),
    (.virt 9475, .wire 11 9),
    (.virt 9475, .wire 11 10),
    (.wire 1 43, .wire 11 12),
    (.virt 18979, .wire 11 13),
    (.wire 11 11, .wire 11 14),
    (.wire 1 43, .wire 11 16),
    (.virt 9476, .wire 11 17),
    (.virt 9476, .wire 11 18),
    (.wire 1 43, .wire 11 20),
    (.virt 18979, .wire 11 21),
    (.wire 11 19, .wire 11 22),
    (.wire 1 43, .wire 11 24),
    (.virt 9477, .wire 11 25),
    (.virt 9477, .wire 11 26),
    (.wire 1 43, .wire 11 28),
    (.virt 18979, .wire 11 29),
    (.wire 11 27, .wire 11 30),
    (.wire 1 43, .wire 11 32),
    (.virt 9478, .wire 11 33),
    (.virt 9478, .wire 11 34),
    (.wire 1 43, .wire 11 36),
    (.virt 18979, .wire 11 37),
    (.wire 11 35, .wire 11 38),
    (.wire 1 43, .wire 11 40),
    (.virt 9465, .wire 11 41),
    (.virt 9465, .wire 11 42),
    (.wire 1 43, .wire 11 44),
    (.virt 18979, .wire 11 45),
    (.wire 11 43, .wire 11 46)
  ]

/-- `privateBatchWrapper2.copies`, items `480..512`. -/
def privateBatchWrapper2.copies15 : List (Target × Target) := [
    (.wire 2 27, .wire 11 48),
    (.virt 18956, .wire 11 49),
    (.virt 18956, .wire 11 50),
    (.wire 2 27, .wire 11 52),
    (.virt 18979, .wire 11 53),
    (.wire 11 51, .wire 11 54),
    (.wire 2 27, .wire 11 56),
    (.virt 18957, .wire 11 57),
    (.virt 18957, .wire 11 58),
    (.wire 2 27, .wire 12 0),
    (.virt 18979, .wire 12 1),
    (.wire 11 59, .wire 12 2),
    (.wire 2 27, .wire 12 4),
    (.virt 18958, .wire 12 5),
    (.virt 18958, .wire 12 6),
    (.wire 2 27, .wire 12 8),
    (.virt 18979, .wire 12 9),
    (.wire 12 7, .wire 12 10),
    (.wire 2 27, .wire 12 12),
    (.virt 18959, .wire 12 13),
    (.virt 18959, .wire 12 14),
    (.wire 2 27, .wire 12 16),
    (.virt 18979, .wire 12 17),
    (.wire 12 15, .wire 12 18),
    (.wire 2 27, .wire 12 20),
    (.virt 18949, .wire 12 21),
    (.virt 18949, .wire 12 22),
    (.wire 2 27, .wire 12 24),
    (.virt 18979, .wire 12 25),
    (.wire 12 23, .wire 12 26),
    (.wire 2 27, .wire 12 28),
    (.virt 18960, .wire 12 29)
  ]

/-- `privateBatchWrapper2.copies`, items `512..544`. -/
def privateBatchWrapper2.copies16 : List (Target × Target) := [
    (.virt 18960, .wire 12 30),
    (.wire 2 27, .wire 12 32),
    (.virt 18979, .wire 12 33),
    (.wire 12 31, .wire 12 34),
    (.wire 2 27, .wire 12 36),
    (.virt 18961, .wire 12 37),
    (.virt 18961, .wire 12 38),
    (.wire 2 27, .wire 12 40),
    (.virt 18979, .wire 12 41),
    (.wire 12 39, .wire 12 42),
    (.wire 2 27, .wire 12 44),
    (.virt 18962, .wire 12 45),
    (.virt 18962, .wire 12 46),
    (.wire 2 27, .wire 12 48),
    (.virt 18979, .wire 12 49),
    (.wire 12 47, .wire 12 50),
    (.wire 2 27, .wire 12 52),
    (.virt 18963, .wire 12 53),
    (.virt 18963, .wire 12 54),
    (.wire 2 27, .wire 12 56),
    (.virt 18979, .wire 12 57),
    (.wire 12 55, .wire 12 58),
    (.wire 2 27, .wire 13 0),
    (.virt 18950, .wire 13 1),
    (.virt 18950, .wire 13 2),
    (.wire 2 27, .wire 13 4),
    (.virt 18979, .wire 13 5),
    (.wire 13 3, .wire 13 6),
    (.wire 1 43, .wire 13 8),
    (.virt 9484, .wire 13 9),
    (.virt 9484, .wire 13 10),
    (.wire 1 43, .wire 13 12)
  ]

/-- `privateBatchWrapper2.copies`, items `544..576`. -/
def privateBatchWrapper2.copies17 : List (Target × Target) := [
    (.virt 18979, .wire 13 13),
    (.wire 13 11, .wire 13 14),
    (.wire 2 27, .wire 13 16),
    (.virt 18969, .wire 13 17),
    (.virt 18969, .wire 13 18),
    (.wire 2 27, .wire 13 20),
    (.virt 18979, .wire 13 21),
    (.wire 13 19, .wire 13 22),
    (.wire 13 15, .wire 6 20),
    (.virt 18978, .wire 6 21),
    (.wire 13 23, .wire 6 22),
    (.wire 11 7, .wire 6 24),
    (.virt 18978, .wire 6 25),
    (.wire 11 47, .wire 6 26),
    (.wire 6 27, .wire 6 28),
    (.virt 18978, .wire 6 29),
    (.wire 12 27, .wire 6 30),
    (.wire 6 31, .wire 6 32),
    (.virt 18978, .wire 6 33),
    (.wire 13 7, .wire 6 34),
    (.virt 19017, .wire 13 24),
    (.virt 18978, .wire 13 25),
    (.wire 4 27, .wire 13 26),
    (.wire 14 15, .virt 18979),
    (.wire 14 16, .virt 18979),
    (.wire 14 17, .virt 18979),
    (.wire 14 18, .virt 18979),
    (.wire 14 19, .virt 18979),
    (.wire 14 20, .virt 18979),
    (.wire 14 21, .virt 18979),
    (.wire 14 22, .virt 18979),
    (.wire 14 23, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `576..608`. -/
def privateBatchWrapper2.copies18 : List (Target × Target) := [
    (.wire 14 24, .virt 18979),
    (.wire 14 25, .virt 18979),
    (.wire 14 26, .virt 18979),
    (.wire 14 27, .virt 18979),
    (.wire 14 28, .virt 18979),
    (.wire 14 29, .virt 18979),
    (.wire 14 30, .virt 18979),
    (.wire 14 31, .virt 18979),
    (.wire 14 32, .virt 18979),
    (.wire 14 33, .virt 18979),
    (.wire 14 34, .virt 18979),
    (.wire 14 35, .virt 18979),
    (.wire 14 36, .virt 18979),
    (.wire 14 37, .virt 18979),
    (.wire 14 38, .virt 18979),
    (.wire 14 39, .virt 18979),
    (.wire 14 40, .virt 18979),
    (.wire 14 41, .virt 18979),
    (.wire 14 42, .virt 18979),
    (.wire 14 43, .virt 18979),
    (.wire 14 44, .virt 18979),
    (.wire 14 45, .virt 18979),
    (.wire 14 46, .virt 18979),
    (.wire 14 47, .virt 18979),
    (.wire 14 48, .virt 18979),
    (.wire 14 49, .virt 18979),
    (.wire 14 50, .virt 18979),
    (.wire 14 51, .virt 18979),
    (.wire 14 52, .virt 18979),
    (.wire 14 53, .virt 18979),
    (.wire 14 54, .virt 18979),
    (.wire 14 55, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `608..640`. -/
def privateBatchWrapper2.copies19 : List (Target × Target) := [
    (.wire 14 56, .virt 18979),
    (.wire 14 57, .virt 18979),
    (.wire 14 58, .virt 18979),
    (.wire 14 59, .virt 18979),
    (.wire 14 0, .wire 13 27),
    (.wire 6 35, .wire 10 16),
    (.virt 19017, .wire 10 17),
    (.wire 6 35, .wire 10 18),
    (.wire 6 23, .wire 10 20),
    (.wire 13 27, .wire 10 21),
    (.wire 6 23, .wire 10 22),
    (.wire 10 23, .wire 13 28),
    (.virt 18978, .wire 13 29),
    (.wire 10 19, .wire 13 30),
    (.wire 15 53, .virt 18979),
    (.wire 15 54, .virt 18979),
    (.wire 15 55, .virt 18979),
    (.wire 15 56, .virt 18979),
    (.wire 15 57, .virt 18979),
    (.wire 15 58, .virt 18979),
    (.wire 15 59, .virt 18979),
    (.wire 15 0, .wire 13 31),
    (.virt 18978, .wire 13 32),
    (.virt 18978, .wire 13 33),
    (.virt 19019, .wire 13 34),
    (.wire 9 35, .wire 13 36),
    (.virt 18978, .wire 13 37),
    (.wire 9 35, .wire 13 38),
    (.virt 19019, .wire 10 24),
    (.wire 13 39, .wire 10 25),
    (.virt 19019, .wire 10 26),
    (.wire 13 39, .wire 10 28)
  ]

/-- `privateBatchWrapper2.copies`, items `640..672`. -/
def privateBatchWrapper2.copies20 : List (Target × Target) := [
    (.virt 19020, .wire 10 29),
    (.wire 13 39, .wire 10 30),
    (.wire 10 31, .wire 13 40),
    (.virt 18978, .wire 13 41),
    (.wire 13 35, .wire 13 42),
    (.wire 10 27, .virt 18979),
    (.wire 13 43, .virt 18979),
    (.virt 18978, .wire 13 44),
    (.virt 18978, .wire 13 45),
    (.virt 19021, .wire 13 46),
    (.wire 9 43, .wire 13 48),
    (.virt 18978, .wire 13 49),
    (.wire 9 43, .wire 13 50),
    (.virt 19021, .wire 10 32),
    (.wire 13 51, .wire 10 33),
    (.virt 19021, .wire 10 34),
    (.wire 13 51, .wire 10 36),
    (.virt 19022, .wire 10 37),
    (.wire 13 51, .wire 10 38),
    (.wire 10 39, .wire 13 52),
    (.virt 18978, .wire 13 53),
    (.wire 13 47, .wire 13 54),
    (.wire 10 35, .virt 18979),
    (.wire 13 55, .virt 18979),
    (.virt 18978, .wire 13 56),
    (.virt 18978, .wire 13 57),
    (.virt 19023, .wire 13 58),
    (.wire 9 51, .wire 16 0),
    (.virt 18978, .wire 16 1),
    (.wire 9 51, .wire 16 2),
    (.virt 19023, .wire 10 40),
    (.wire 16 3, .wire 10 41)
  ]

/-- `privateBatchWrapper2.copies`, items `672..704`. -/
def privateBatchWrapper2.copies21 : List (Target × Target) := [
    (.virt 19023, .wire 10 42),
    (.wire 16 3, .wire 10 44),
    (.virt 19024, .wire 10 45),
    (.wire 16 3, .wire 10 46),
    (.wire 10 47, .wire 16 4),
    (.virt 18978, .wire 16 5),
    (.wire 13 59, .wire 16 6),
    (.wire 10 43, .virt 18979),
    (.wire 16 7, .virt 18979),
    (.virt 18978, .wire 16 8),
    (.virt 18978, .wire 16 9),
    (.virt 19025, .wire 16 10),
    (.wire 9 59, .wire 16 12),
    (.virt 18978, .wire 16 13),
    (.wire 9 59, .wire 16 14),
    (.virt 19025, .wire 10 48),
    (.wire 16 15, .wire 10 49),
    (.virt 19025, .wire 10 50),
    (.wire 16 15, .wire 10 52),
    (.virt 19026, .wire 10 53),
    (.wire 16 15, .wire 10 54),
    (.wire 10 55, .wire 16 16),
    (.virt 18978, .wire 16 17),
    (.wire 16 11, .wire 16 18),
    (.wire 10 51, .virt 18979),
    (.wire 16 19, .virt 18979),
    (.virt 19019, .wire 10 56),
    (.virt 19021, .wire 10 57),
    (.virt 19019, .wire 10 58),
    (.virt 19023, .wire 17 0),
    (.virt 19025, .wire 17 1),
    (.virt 19023, .wire 17 2)
  ]

/-- `privateBatchWrapper2.copies`, items `704..736`. -/
def privateBatchWrapper2.copies22 : List (Target × Target) := [
    (.wire 10 59, .wire 17 4),
    (.wire 17 3, .wire 17 5),
    (.wire 10 59, .wire 17 6),
    (.wire 17 7, .wire 16 20),
    (.wire 11 7, .wire 16 21),
    (.virt 18979, .wire 16 22),
    (.virt 18978, .wire 16 24),
    (.virt 18978, .wire 16 25),
    (.virt 19027, .wire 16 26),
    (.wire 11 15, .wire 16 28),
    (.virt 18978, .wire 16 29),
    (.wire 9 35, .wire 16 30),
    (.virt 19027, .wire 17 8),
    (.wire 16 31, .wire 17 9),
    (.virt 19027, .wire 17 10),
    (.wire 16 31, .wire 17 12),
    (.virt 19028, .wire 17 13),
    (.wire 16 31, .wire 17 14),
    (.wire 17 15, .wire 16 32),
    (.virt 18978, .wire 16 33),
    (.wire 16 27, .wire 16 34),
    (.wire 17 11, .virt 18979),
    (.wire 16 35, .virt 18979),
    (.virt 18978, .wire 16 36),
    (.virt 18978, .wire 16 37),
    (.virt 19029, .wire 16 38),
    (.wire 11 23, .wire 16 40),
    (.virt 18978, .wire 16 41),
    (.wire 9 43, .wire 16 42),
    (.virt 19029, .wire 17 16),
    (.wire 16 43, .wire 17 17),
    (.virt 19029, .wire 17 18)
  ]

/-- `privateBatchWrapper2.copies`, items `736..768`. -/
def privateBatchWrapper2.copies23 : List (Target × Target) := [
    (.wire 16 43, .wire 17 20),
    (.virt 19030, .wire 17 21),
    (.wire 16 43, .wire 17 22),
    (.wire 17 23, .wire 16 44),
    (.virt 18978, .wire 16 45),
    (.wire 16 39, .wire 16 46),
    (.wire 17 19, .virt 18979),
    (.wire 16 47, .virt 18979),
    (.virt 18978, .wire 16 48),
    (.virt 18978, .wire 16 49),
    (.virt 19031, .wire 16 50),
    (.wire 11 31, .wire 16 52),
    (.virt 18978, .wire 16 53),
    (.wire 9 51, .wire 16 54),
    (.virt 19031, .wire 17 24),
    (.wire 16 55, .wire 17 25),
    (.virt 19031, .wire 17 26),
    (.wire 16 55, .wire 17 28),
    (.virt 19032, .wire 17 29),
    (.wire 16 55, .wire 17 30),
    (.wire 17 31, .wire 16 56),
    (.virt 18978, .wire 16 57),
    (.wire 16 51, .wire 16 58),
    (.wire 17 27, .virt 18979),
    (.wire 16 59, .virt 18979),
    (.virt 18978, .wire 18 0),
    (.virt 18978, .wire 18 1),
    (.virt 19033, .wire 18 2),
    (.wire 11 39, .wire 18 4),
    (.virt 18978, .wire 18 5),
    (.wire 9 59, .wire 18 6),
    (.virt 19033, .wire 17 32)
  ]

/-- `privateBatchWrapper2.copies`, items `768..800`. -/
def privateBatchWrapper2.copies24 : List (Target × Target) := [
    (.wire 18 7, .wire 17 33),
    (.virt 19033, .wire 17 34),
    (.wire 18 7, .wire 17 36),
    (.virt 19034, .wire 17 37),
    (.wire 18 7, .wire 17 38),
    (.wire 17 39, .wire 18 8),
    (.virt 18978, .wire 18 9),
    (.wire 18 3, .wire 18 10),
    (.wire 17 35, .virt 18979),
    (.wire 18 11, .virt 18979),
    (.virt 19027, .wire 17 40),
    (.virt 19029, .wire 17 41),
    (.virt 19027, .wire 17 42),
    (.virt 19031, .wire 17 44),
    (.virt 19033, .wire 17 45),
    (.virt 19031, .wire 17 46),
    (.wire 17 43, .wire 17 48),
    (.wire 17 47, .wire 17 49),
    (.wire 17 43, .wire 17 50),
    (.wire 17 51, .wire 18 12),
    (.wire 11 47, .wire 18 13),
    (.virt 18979, .wire 18 14),
    (.wire 16 23, .wire 6 36),
    (.virt 18978, .wire 6 37),
    (.wire 18 15, .wire 6 38),
    (.virt 18978, .wire 18 16),
    (.virt 18978, .wire 18 17),
    (.virt 19035, .wire 18 18),
    (.wire 11 55, .wire 18 20),
    (.virt 18978, .wire 18 21),
    (.wire 9 35, .wire 18 22),
    (.virt 19035, .wire 17 52)
  ]

/-- `privateBatchWrapper2.copies`, items `800..832`. -/
def privateBatchWrapper2.copies25 : List (Target × Target) := [
    (.wire 18 23, .wire 17 53),
    (.virt 19035, .wire 17 54),
    (.wire 18 23, .wire 17 56),
    (.virt 19036, .wire 17 57),
    (.wire 18 23, .wire 17 58),
    (.wire 17 59, .wire 18 24),
    (.virt 18978, .wire 18 25),
    (.wire 18 19, .wire 18 26),
    (.wire 17 55, .virt 18979),
    (.wire 18 27, .virt 18979),
    (.virt 18978, .wire 18 28),
    (.virt 18978, .wire 18 29),
    (.virt 19037, .wire 18 30),
    (.wire 12 3, .wire 18 32),
    (.virt 18978, .wire 18 33),
    (.wire 9 43, .wire 18 34),
    (.virt 19037, .wire 19 0),
    (.wire 18 35, .wire 19 1),
    (.virt 19037, .wire 19 2),
    (.wire 18 35, .wire 19 4),
    (.virt 19038, .wire 19 5),
    (.wire 18 35, .wire 19 6),
    (.wire 19 7, .wire 18 36),
    (.virt 18978, .wire 18 37),
    (.wire 18 31, .wire 18 38),
    (.wire 19 3, .virt 18979),
    (.wire 18 39, .virt 18979),
    (.virt 18978, .wire 18 40),
    (.virt 18978, .wire 18 41),
    (.virt 19039, .wire 18 42),
    (.wire 12 11, .wire 18 44),
    (.virt 18978, .wire 18 45)
  ]

/-- `privateBatchWrapper2.copies`, items `832..864`. -/
def privateBatchWrapper2.copies26 : List (Target × Target) := [
    (.wire 9 51, .wire 18 46),
    (.virt 19039, .wire 19 8),
    (.wire 18 47, .wire 19 9),
    (.virt 19039, .wire 19 10),
    (.wire 18 47, .wire 19 12),
    (.virt 19040, .wire 19 13),
    (.wire 18 47, .wire 19 14),
    (.wire 19 15, .wire 18 48),
    (.virt 18978, .wire 18 49),
    (.wire 18 43, .wire 18 50),
    (.wire 19 11, .virt 18979),
    (.wire 18 51, .virt 18979),
    (.virt 18978, .wire 18 52),
    (.virt 18978, .wire 18 53),
    (.virt 19041, .wire 18 54),
    (.wire 12 19, .wire 18 56),
    (.virt 18978, .wire 18 57),
    (.wire 9 59, .wire 18 58),
    (.virt 19041, .wire 19 16),
    (.wire 18 59, .wire 19 17),
    (.virt 19041, .wire 19 18),
    (.wire 18 59, .wire 19 20),
    (.virt 19042, .wire 19 21),
    (.wire 18 59, .wire 19 22),
    (.wire 19 23, .wire 20 0),
    (.virt 18978, .wire 20 1),
    (.wire 18 55, .wire 20 2),
    (.wire 19 19, .virt 18979),
    (.wire 20 3, .virt 18979),
    (.virt 19035, .wire 19 24),
    (.virt 19037, .wire 19 25),
    (.virt 19035, .wire 19 26)
  ]

/-- `privateBatchWrapper2.copies`, items `864..896`. -/
def privateBatchWrapper2.copies27 : List (Target × Target) := [
    (.virt 19039, .wire 19 28),
    (.virt 19041, .wire 19 29),
    (.virt 19039, .wire 19 30),
    (.wire 19 27, .wire 19 32),
    (.wire 19 31, .wire 19 33),
    (.wire 19 27, .wire 19 34),
    (.wire 19 35, .wire 20 4),
    (.wire 12 27, .wire 20 5),
    (.virt 18979, .wire 20 6),
    (.wire 6 39, .wire 6 40),
    (.virt 18978, .wire 6 41),
    (.wire 20 7, .wire 6 42),
    (.virt 18978, .wire 20 8),
    (.virt 18978, .wire 20 9),
    (.virt 19043, .wire 20 10),
    (.wire 12 35, .wire 20 12),
    (.virt 18978, .wire 20 13),
    (.wire 9 35, .wire 20 14),
    (.virt 19043, .wire 19 36),
    (.wire 20 15, .wire 19 37),
    (.virt 19043, .wire 19 38),
    (.wire 20 15, .wire 19 40),
    (.virt 19044, .wire 19 41),
    (.wire 20 15, .wire 19 42),
    (.wire 19 43, .wire 20 16),
    (.virt 18978, .wire 20 17),
    (.wire 20 11, .wire 20 18),
    (.wire 19 39, .virt 18979),
    (.wire 20 19, .virt 18979),
    (.virt 18978, .wire 20 20),
    (.virt 18978, .wire 20 21),
    (.virt 19045, .wire 20 22)
  ]

/-- `privateBatchWrapper2.copies`, items `896..928`. -/
def privateBatchWrapper2.copies28 : List (Target × Target) := [
    (.wire 12 43, .wire 20 24),
    (.virt 18978, .wire 20 25),
    (.wire 9 43, .wire 20 26),
    (.virt 19045, .wire 19 44),
    (.wire 20 27, .wire 19 45),
    (.virt 19045, .wire 19 46),
    (.wire 20 27, .wire 19 48),
    (.virt 19046, .wire 19 49),
    (.wire 20 27, .wire 19 50),
    (.wire 19 51, .wire 20 28),
    (.virt 18978, .wire 20 29),
    (.wire 20 23, .wire 20 30),
    (.wire 19 47, .virt 18979),
    (.wire 20 31, .virt 18979),
    (.virt 18978, .wire 20 32),
    (.virt 18978, .wire 20 33),
    (.virt 19047, .wire 20 34),
    (.wire 12 51, .wire 20 36),
    (.virt 18978, .wire 20 37),
    (.wire 9 51, .wire 20 38),
    (.virt 19047, .wire 19 52),
    (.wire 20 39, .wire 19 53),
    (.virt 19047, .wire 19 54),
    (.wire 20 39, .wire 19 56),
    (.virt 19048, .wire 19 57),
    (.wire 20 39, .wire 19 58),
    (.wire 19 59, .wire 20 40),
    (.virt 18978, .wire 20 41),
    (.wire 20 35, .wire 20 42),
    (.wire 19 55, .virt 18979),
    (.wire 20 43, .virt 18979),
    (.virt 18978, .wire 20 44)
  ]

/-- `privateBatchWrapper2.copies`, items `928..960`. -/
def privateBatchWrapper2.copies29 : List (Target × Target) := [
    (.virt 18978, .wire 20 45),
    (.virt 19049, .wire 20 46),
    (.wire 12 59, .wire 20 48),
    (.virt 18978, .wire 20 49),
    (.wire 9 59, .wire 20 50),
    (.virt 19049, .wire 21 0),
    (.wire 20 51, .wire 21 1),
    (.virt 19049, .wire 21 2),
    (.wire 20 51, .wire 21 4),
    (.virt 19050, .wire 21 5),
    (.wire 20 51, .wire 21 6),
    (.wire 21 7, .wire 20 52),
    (.virt 18978, .wire 20 53),
    (.wire 20 47, .wire 20 54),
    (.wire 21 3, .virt 18979),
    (.wire 20 55, .virt 18979),
    (.virt 19043, .wire 21 8),
    (.virt 19045, .wire 21 9),
    (.virt 19043, .wire 21 10),
    (.virt 19047, .wire 21 12),
    (.virt 19049, .wire 21 13),
    (.virt 19047, .wire 21 14),
    (.wire 21 11, .wire 21 16),
    (.wire 21 15, .wire 21 17),
    (.wire 21 11, .wire 21 18),
    (.wire 21 19, .wire 20 56),
    (.wire 13 7, .wire 20 57),
    (.virt 18979, .wire 20 58),
    (.wire 6 43, .wire 6 44),
    (.virt 18978, .wire 6 45),
    (.wire 20 59, .wire 6 46),
    (.virt 18979, .wire 22 0)
  ]

/-- `privateBatchWrapper2.copies`, items `960..992`. -/
def privateBatchWrapper2.copies30 : List (Target × Target) := [
    (.wire 6 47, .wire 22 1),
    (.wire 6 47, .wire 22 2),
    (.virt 18979, .wire 22 4),
    (.virt 18979, .wire 22 5),
    (.wire 22 3, .wire 22 6),
    (.virt 18979, .wire 22 8),
    (.wire 9 35, .wire 22 9),
    (.wire 9 35, .wire 22 10),
    (.virt 18979, .wire 22 12),
    (.virt 18979, .wire 22 13),
    (.wire 22 11, .wire 22 14),
    (.virt 18979, .wire 22 16),
    (.wire 9 43, .wire 22 17),
    (.wire 9 43, .wire 22 18),
    (.virt 18979, .wire 22 20),
    (.virt 18979, .wire 22 21),
    (.wire 22 19, .wire 22 22),
    (.virt 18979, .wire 22 24),
    (.wire 9 51, .wire 22 25),
    (.wire 9 51, .wire 22 26),
    (.virt 18979, .wire 22 28),
    (.virt 18979, .wire 22 29),
    (.wire 22 27, .wire 22 30),
    (.virt 18979, .wire 22 32),
    (.wire 9 59, .wire 22 33),
    (.wire 9 59, .wire 22 34),
    (.virt 18979, .wire 22 36),
    (.virt 18979, .wire 22 37),
    (.wire 22 35, .wire 22 38),
    (.wire 23 33, .virt 18979),
    (.wire 23 34, .virt 18979),
    (.wire 23 35, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `992..1024`. -/
def privateBatchWrapper2.copies31 : List (Target × Target) := [
    (.wire 23 36, .virt 18979),
    (.wire 23 37, .virt 18979),
    (.wire 23 38, .virt 18979),
    (.wire 23 39, .virt 18979),
    (.wire 23 40, .virt 18979),
    (.wire 23 41, .virt 18979),
    (.wire 23 42, .virt 18979),
    (.wire 23 43, .virt 18979),
    (.wire 23 44, .virt 18979),
    (.wire 23 45, .virt 18979),
    (.wire 23 46, .virt 18979),
    (.wire 23 47, .virt 18979),
    (.wire 23 48, .virt 18979),
    (.wire 23 49, .virt 18979),
    (.wire 23 50, .virt 18979),
    (.wire 23 51, .virt 18979),
    (.wire 23 52, .virt 18979),
    (.wire 23 53, .virt 18979),
    (.wire 23 54, .virt 18979),
    (.wire 23 55, .virt 18979),
    (.wire 23 56, .virt 18979),
    (.wire 23 57, .virt 18979),
    (.wire 23 58, .virt 18979),
    (.wire 23 59, .virt 18979),
    (.wire 23 0, .wire 22 7),
    (.virt 18978, .wire 22 40),
    (.virt 18978, .wire 22 41),
    (.virt 19051, .wire 22 42),
    (.wire 9 35, .wire 22 44),
    (.virt 18978, .wire 22 45),
    (.wire 11 15, .wire 22 46),
    (.virt 19051, .wire 21 20)
  ]

/-- `privateBatchWrapper2.copies`, items `1024..1056`. -/
def privateBatchWrapper2.copies32 : List (Target × Target) := [
    (.wire 22 47, .wire 21 21),
    (.virt 19051, .wire 21 22),
    (.wire 22 47, .wire 21 24),
    (.virt 19052, .wire 21 25),
    (.wire 22 47, .wire 21 26),
    (.wire 21 27, .wire 22 48),
    (.virt 18978, .wire 22 49),
    (.wire 22 43, .wire 22 50),
    (.wire 21 23, .virt 18979),
    (.wire 22 51, .virt 18979),
    (.virt 18978, .wire 22 52),
    (.virt 18978, .wire 22 53),
    (.virt 19053, .wire 22 54),
    (.wire 9 43, .wire 22 56),
    (.virt 18978, .wire 22 57),
    (.wire 11 23, .wire 22 58),
    (.virt 19053, .wire 21 28),
    (.wire 22 59, .wire 21 29),
    (.virt 19053, .wire 21 30),
    (.wire 22 59, .wire 21 32),
    (.virt 19054, .wire 21 33),
    (.wire 22 59, .wire 21 34),
    (.wire 21 35, .wire 24 0),
    (.virt 18978, .wire 24 1),
    (.wire 22 55, .wire 24 2),
    (.wire 21 31, .virt 18979),
    (.wire 24 3, .virt 18979),
    (.virt 18978, .wire 24 4),
    (.virt 18978, .wire 24 5),
    (.virt 19055, .wire 24 6),
    (.wire 9 51, .wire 24 8),
    (.virt 18978, .wire 24 9)
  ]

/-- `privateBatchWrapper2.copies`, items `1056..1088`. -/
def privateBatchWrapper2.copies33 : List (Target × Target) := [
    (.wire 11 31, .wire 24 10),
    (.virt 19055, .wire 21 36),
    (.wire 24 11, .wire 21 37),
    (.virt 19055, .wire 21 38),
    (.wire 24 11, .wire 21 40),
    (.virt 19056, .wire 21 41),
    (.wire 24 11, .wire 21 42),
    (.wire 21 43, .wire 24 12),
    (.virt 18978, .wire 24 13),
    (.wire 24 7, .wire 24 14),
    (.wire 21 39, .virt 18979),
    (.wire 24 15, .virt 18979),
    (.virt 18978, .wire 24 16),
    (.virt 18978, .wire 24 17),
    (.virt 19057, .wire 24 18),
    (.wire 9 59, .wire 24 20),
    (.virt 18978, .wire 24 21),
    (.wire 11 39, .wire 24 22),
    (.virt 19057, .wire 21 44),
    (.wire 24 23, .wire 21 45),
    (.virt 19057, .wire 21 46),
    (.wire 24 23, .wire 21 48),
    (.virt 19058, .wire 21 49),
    (.wire 24 23, .wire 21 50),
    (.wire 21 51, .wire 24 24),
    (.virt 18978, .wire 24 25),
    (.wire 24 19, .wire 24 26),
    (.wire 21 47, .virt 18979),
    (.wire 24 27, .virt 18979),
    (.virt 19051, .wire 21 52),
    (.virt 19053, .wire 21 53),
    (.virt 19051, .wire 21 54)
  ]

/-- `privateBatchWrapper2.copies`, items `1088..1120`. -/
def privateBatchWrapper2.copies34 : List (Target × Target) := [
    (.virt 19055, .wire 21 56),
    (.virt 19057, .wire 21 57),
    (.virt 19055, .wire 21 58),
    (.wire 21 55, .wire 25 0),
    (.wire 21 59, .wire 25 1),
    (.wire 21 55, .wire 25 2),
    (.virt 18978, .wire 24 28),
    (.virt 18978, .wire 24 29),
    (.virt 19059, .wire 24 30),
    (.virt 19059, .wire 25 4),
    (.wire 22 47, .wire 25 5),
    (.virt 19059, .wire 25 6),
    (.wire 22 47, .wire 25 8),
    (.virt 19060, .wire 25 9),
    (.wire 22 47, .wire 25 10),
    (.wire 25 11, .wire 24 32),
    (.virt 18978, .wire 24 33),
    (.wire 24 31, .wire 24 34),
    (.wire 25 7, .virt 18979),
    (.wire 24 35, .virt 18979),
    (.virt 18978, .wire 24 36),
    (.virt 18978, .wire 24 37),
    (.virt 19061, .wire 24 38),
    (.virt 19061, .wire 25 12),
    (.wire 22 59, .wire 25 13),
    (.virt 19061, .wire 25 14),
    (.wire 22 59, .wire 25 16),
    (.virt 19062, .wire 25 17),
    (.wire 22 59, .wire 25 18),
    (.wire 25 19, .wire 24 40),
    (.virt 18978, .wire 24 41),
    (.wire 24 39, .wire 24 42)
  ]

/-- `privateBatchWrapper2.copies`, items `1120..1152`. -/
def privateBatchWrapper2.copies35 : List (Target × Target) := [
    (.wire 25 15, .virt 18979),
    (.wire 24 43, .virt 18979),
    (.virt 18978, .wire 24 44),
    (.virt 18978, .wire 24 45),
    (.virt 19063, .wire 24 46),
    (.virt 19063, .wire 25 20),
    (.wire 24 11, .wire 25 21),
    (.virt 19063, .wire 25 22),
    (.wire 24 11, .wire 25 24),
    (.virt 19064, .wire 25 25),
    (.wire 24 11, .wire 25 26),
    (.wire 25 27, .wire 24 48),
    (.virt 18978, .wire 24 49),
    (.wire 24 47, .wire 24 50),
    (.wire 25 23, .virt 18979),
    (.wire 24 51, .virt 18979),
    (.virt 18978, .wire 24 52),
    (.virt 18978, .wire 24 53),
    (.virt 19065, .wire 24 54),
    (.virt 19065, .wire 25 28),
    (.wire 24 23, .wire 25 29),
    (.virt 19065, .wire 25 30),
    (.wire 24 23, .wire 25 32),
    (.virt 19066, .wire 25 33),
    (.wire 24 23, .wire 25 34),
    (.wire 25 35, .wire 24 56),
    (.virt 18978, .wire 24 57),
    (.wire 24 55, .wire 24 58),
    (.wire 25 31, .virt 18979),
    (.wire 24 59, .virt 18979),
    (.virt 19059, .wire 25 36),
    (.virt 19061, .wire 25 37)
  ]

/-- `privateBatchWrapper2.copies`, items `1152..1184`. -/
def privateBatchWrapper2.copies36 : List (Target × Target) := [
    (.virt 19059, .wire 25 38),
    (.virt 19063, .wire 25 40),
    (.virt 19065, .wire 25 41),
    (.virt 19063, .wire 25 42),
    (.wire 25 39, .wire 25 44),
    (.wire 25 43, .wire 25 45),
    (.wire 25 39, .wire 25 46),
    (.wire 25 47, .wire 26 0),
    (.wire 11 7, .wire 26 1),
    (.virt 18979, .wire 26 2),
    (.virt 18978, .wire 26 4),
    (.virt 18978, .wire 26 5),
    (.virt 19067, .wire 26 6),
    (.wire 11 15, .wire 26 8),
    (.virt 18978, .wire 26 9),
    (.wire 11 15, .wire 26 10),
    (.virt 19067, .wire 25 48),
    (.wire 26 11, .wire 25 49),
    (.virt 19067, .wire 25 50),
    (.wire 26 11, .wire 25 52),
    (.virt 19068, .wire 25 53),
    (.wire 26 11, .wire 25 54),
    (.wire 25 55, .wire 26 12),
    (.virt 18978, .wire 26 13),
    (.wire 26 7, .wire 26 14),
    (.wire 25 51, .virt 18979),
    (.wire 26 15, .virt 18979),
    (.virt 18978, .wire 26 16),
    (.virt 18978, .wire 26 17),
    (.virt 19069, .wire 26 18),
    (.wire 11 23, .wire 26 20),
    (.virt 18978, .wire 26 21)
  ]

/-- `privateBatchWrapper2.copies`, items `1184..1216`. -/
def privateBatchWrapper2.copies37 : List (Target × Target) := [
    (.wire 11 23, .wire 26 22),
    (.virt 19069, .wire 25 56),
    (.wire 26 23, .wire 25 57),
    (.virt 19069, .wire 25 58),
    (.wire 26 23, .wire 27 0),
    (.virt 19070, .wire 27 1),
    (.wire 26 23, .wire 27 2),
    (.wire 27 3, .wire 26 24),
    (.virt 18978, .wire 26 25),
    (.wire 26 19, .wire 26 26),
    (.wire 25 59, .virt 18979),
    (.wire 26 27, .virt 18979),
    (.virt 18978, .wire 26 28),
    (.virt 18978, .wire 26 29),
    (.virt 19071, .wire 26 30),
    (.wire 11 31, .wire 26 32),
    (.virt 18978, .wire 26 33),
    (.wire 11 31, .wire 26 34),
    (.virt 19071, .wire 27 4),
    (.wire 26 35, .wire 27 5),
    (.virt 19071, .wire 27 6),
    (.wire 26 35, .wire 27 8),
    (.virt 19072, .wire 27 9),
    (.wire 26 35, .wire 27 10),
    (.wire 27 11, .wire 26 36),
    (.virt 18978, .wire 26 37),
    (.wire 26 31, .wire 26 38),
    (.wire 27 7, .virt 18979),
    (.wire 26 39, .virt 18979),
    (.virt 18978, .wire 26 40),
    (.virt 18978, .wire 26 41),
    (.virt 19073, .wire 26 42)
  ]

/-- `privateBatchWrapper2.copies`, items `1216..1248`. -/
def privateBatchWrapper2.copies38 : List (Target × Target) := [
    (.wire 11 39, .wire 26 44),
    (.virt 18978, .wire 26 45),
    (.wire 11 39, .wire 26 46),
    (.virt 19073, .wire 27 12),
    (.wire 26 47, .wire 27 13),
    (.virt 19073, .wire 27 14),
    (.wire 26 47, .wire 27 16),
    (.virt 19074, .wire 27 17),
    (.wire 26 47, .wire 27 18),
    (.wire 27 19, .wire 26 48),
    (.virt 18978, .wire 26 49),
    (.wire 26 43, .wire 26 50),
    (.wire 27 15, .virt 18979),
    (.wire 26 51, .virt 18979),
    (.virt 19067, .wire 27 20),
    (.virt 19069, .wire 27 21),
    (.virt 19067, .wire 27 22),
    (.virt 19071, .wire 27 24),
    (.virt 19073, .wire 27 25),
    (.virt 19071, .wire 27 26),
    (.wire 27 23, .wire 27 28),
    (.wire 27 27, .wire 27 29),
    (.wire 27 23, .wire 27 30),
    (.wire 27 31, .wire 26 52),
    (.wire 11 47, .wire 26 53),
    (.virt 18979, .wire 26 54),
    (.wire 26 3, .wire 6 48),
    (.virt 18978, .wire 6 49),
    (.wire 26 55, .wire 6 50),
    (.virt 18978, .wire 26 56),
    (.virt 18978, .wire 26 57),
    (.virt 19075, .wire 26 58)
  ]

/-- `privateBatchWrapper2.copies`, items `1248..1280`. -/
def privateBatchWrapper2.copies39 : List (Target × Target) := [
    (.wire 11 55, .wire 28 0),
    (.virt 18978, .wire 28 1),
    (.wire 11 15, .wire 28 2),
    (.virt 19075, .wire 27 32),
    (.wire 28 3, .wire 27 33),
    (.virt 19075, .wire 27 34),
    (.wire 28 3, .wire 27 36),
    (.virt 19076, .wire 27 37),
    (.wire 28 3, .wire 27 38),
    (.wire 27 39, .wire 28 4),
    (.virt 18978, .wire 28 5),
    (.wire 26 59, .wire 28 6),
    (.wire 27 35, .virt 18979),
    (.wire 28 7, .virt 18979),
    (.virt 18978, .wire 28 8),
    (.virt 18978, .wire 28 9),
    (.virt 19077, .wire 28 10),
    (.wire 12 3, .wire 28 12),
    (.virt 18978, .wire 28 13),
    (.wire 11 23, .wire 28 14),
    (.virt 19077, .wire 27 40),
    (.wire 28 15, .wire 27 41),
    (.virt 19077, .wire 27 42),
    (.wire 28 15, .wire 27 44),
    (.virt 19078, .wire 27 45),
    (.wire 28 15, .wire 27 46),
    (.wire 27 47, .wire 28 16),
    (.virt 18978, .wire 28 17),
    (.wire 28 11, .wire 28 18),
    (.wire 27 43, .virt 18979),
    (.wire 28 19, .virt 18979),
    (.virt 18978, .wire 28 20)
  ]

/-- `privateBatchWrapper2.copies`, items `1280..1312`. -/
def privateBatchWrapper2.copies40 : List (Target × Target) := [
    (.virt 18978, .wire 28 21),
    (.virt 19079, .wire 28 22),
    (.wire 12 11, .wire 28 24),
    (.virt 18978, .wire 28 25),
    (.wire 11 31, .wire 28 26),
    (.virt 19079, .wire 27 48),
    (.wire 28 27, .wire 27 49),
    (.virt 19079, .wire 27 50),
    (.wire 28 27, .wire 27 52),
    (.virt 19080, .wire 27 53),
    (.wire 28 27, .wire 27 54),
    (.wire 27 55, .wire 28 28),
    (.virt 18978, .wire 28 29),
    (.wire 28 23, .wire 28 30),
    (.wire 27 51, .virt 18979),
    (.wire 28 31, .virt 18979),
    (.virt 18978, .wire 28 32),
    (.virt 18978, .wire 28 33),
    (.virt 19081, .wire 28 34),
    (.wire 12 19, .wire 28 36),
    (.virt 18978, .wire 28 37),
    (.wire 11 39, .wire 28 38),
    (.virt 19081, .wire 27 56),
    (.wire 28 39, .wire 27 57),
    (.virt 19081, .wire 27 58),
    (.wire 28 39, .wire 29 0),
    (.virt 19082, .wire 29 1),
    (.wire 28 39, .wire 29 2),
    (.wire 29 3, .wire 28 40),
    (.virt 18978, .wire 28 41),
    (.wire 28 35, .wire 28 42),
    (.wire 27 59, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `1312..1344`. -/
def privateBatchWrapper2.copies41 : List (Target × Target) := [
    (.wire 28 43, .virt 18979),
    (.virt 19075, .wire 29 4),
    (.virt 19077, .wire 29 5),
    (.virt 19075, .wire 29 6),
    (.virt 19079, .wire 29 8),
    (.virt 19081, .wire 29 9),
    (.virt 19079, .wire 29 10),
    (.wire 29 7, .wire 29 12),
    (.wire 29 11, .wire 29 13),
    (.wire 29 7, .wire 29 14),
    (.wire 29 15, .wire 28 44),
    (.wire 12 27, .wire 28 45),
    (.virt 18979, .wire 28 46),
    (.wire 6 51, .wire 6 52),
    (.virt 18978, .wire 6 53),
    (.wire 28 47, .wire 6 54),
    (.virt 18978, .wire 28 48),
    (.virt 18978, .wire 28 49),
    (.virt 19083, .wire 28 50),
    (.wire 12 35, .wire 28 52),
    (.virt 18978, .wire 28 53),
    (.wire 11 15, .wire 28 54),
    (.virt 19083, .wire 29 16),
    (.wire 28 55, .wire 29 17),
    (.virt 19083, .wire 29 18),
    (.wire 28 55, .wire 29 20),
    (.virt 19084, .wire 29 21),
    (.wire 28 55, .wire 29 22),
    (.wire 29 23, .wire 28 56),
    (.virt 18978, .wire 28 57),
    (.wire 28 51, .wire 28 58),
    (.wire 29 19, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `1344..1376`. -/
def privateBatchWrapper2.copies42 : List (Target × Target) := [
    (.wire 28 59, .virt 18979),
    (.virt 18978, .wire 30 0),
    (.virt 18978, .wire 30 1),
    (.virt 19085, .wire 30 2),
    (.wire 12 43, .wire 30 4),
    (.virt 18978, .wire 30 5),
    (.wire 11 23, .wire 30 6),
    (.virt 19085, .wire 29 24),
    (.wire 30 7, .wire 29 25),
    (.virt 19085, .wire 29 26),
    (.wire 30 7, .wire 29 28),
    (.virt 19086, .wire 29 29),
    (.wire 30 7, .wire 29 30),
    (.wire 29 31, .wire 30 8),
    (.virt 18978, .wire 30 9),
    (.wire 30 3, .wire 30 10),
    (.wire 29 27, .virt 18979),
    (.wire 30 11, .virt 18979),
    (.virt 18978, .wire 30 12),
    (.virt 18978, .wire 30 13),
    (.virt 19087, .wire 30 14),
    (.wire 12 51, .wire 30 16),
    (.virt 18978, .wire 30 17),
    (.wire 11 31, .wire 30 18),
    (.virt 19087, .wire 29 32),
    (.wire 30 19, .wire 29 33),
    (.virt 19087, .wire 29 34),
    (.wire 30 19, .wire 29 36),
    (.virt 19088, .wire 29 37),
    (.wire 30 19, .wire 29 38),
    (.wire 29 39, .wire 30 20),
    (.virt 18978, .wire 30 21)
  ]

/-- `privateBatchWrapper2.copies`, items `1376..1408`. -/
def privateBatchWrapper2.copies43 : List (Target × Target) := [
    (.wire 30 15, .wire 30 22),
    (.wire 29 35, .virt 18979),
    (.wire 30 23, .virt 18979),
    (.virt 18978, .wire 30 24),
    (.virt 18978, .wire 30 25),
    (.virt 19089, .wire 30 26),
    (.wire 12 59, .wire 30 28),
    (.virt 18978, .wire 30 29),
    (.wire 11 39, .wire 30 30),
    (.virt 19089, .wire 29 40),
    (.wire 30 31, .wire 29 41),
    (.virt 19089, .wire 29 42),
    (.wire 30 31, .wire 29 44),
    (.virt 19090, .wire 29 45),
    (.wire 30 31, .wire 29 46),
    (.wire 29 47, .wire 30 32),
    (.virt 18978, .wire 30 33),
    (.wire 30 27, .wire 30 34),
    (.wire 29 43, .virt 18979),
    (.wire 30 35, .virt 18979),
    (.virt 19083, .wire 29 48),
    (.virt 19085, .wire 29 49),
    (.virt 19083, .wire 29 50),
    (.virt 19087, .wire 29 52),
    (.virt 19089, .wire 29 53),
    (.virt 19087, .wire 29 54),
    (.wire 29 51, .wire 29 56),
    (.wire 29 55, .wire 29 57),
    (.wire 29 51, .wire 29 58),
    (.wire 29 59, .wire 30 36),
    (.wire 13 7, .wire 30 37),
    (.virt 18979, .wire 30 38)
  ]

/-- `privateBatchWrapper2.copies`, items `1408..1440`. -/
def privateBatchWrapper2.copies44 : List (Target × Target) := [
    (.wire 6 55, .wire 6 56),
    (.virt 18978, .wire 6 57),
    (.wire 30 39, .wire 6 58),
    (.wire 25 3, .wire 30 40),
    (.wire 6 59, .wire 30 41),
    (.wire 6 59, .wire 30 42),
    (.wire 25 3, .wire 30 44),
    (.virt 18979, .wire 30 45),
    (.wire 30 43, .wire 30 46),
    (.wire 25 3, .wire 30 48),
    (.wire 11 15, .wire 30 49),
    (.wire 11 15, .wire 30 50),
    (.wire 25 3, .wire 30 52),
    (.virt 18979, .wire 30 53),
    (.wire 30 51, .wire 30 54),
    (.wire 25 3, .wire 30 56),
    (.wire 11 23, .wire 30 57),
    (.wire 11 23, .wire 30 58),
    (.wire 25 3, .wire 31 0),
    (.virt 18979, .wire 31 1),
    (.wire 30 59, .wire 31 2),
    (.wire 25 3, .wire 31 4),
    (.wire 11 31, .wire 31 5),
    (.wire 11 31, .wire 31 6),
    (.wire 25 3, .wire 31 8),
    (.virt 18979, .wire 31 9),
    (.wire 31 7, .wire 31 10),
    (.wire 25 3, .wire 31 12),
    (.wire 11 39, .wire 31 13),
    (.wire 11 39, .wire 31 14),
    (.wire 25 3, .wire 31 16),
    (.virt 18979, .wire 31 17)
  ]

/-- `privateBatchWrapper2.copies`, items `1440..1472`. -/
def privateBatchWrapper2.copies45 : List (Target × Target) := [
    (.wire 31 15, .wire 31 18),
    (.wire 32 33, .virt 18979),
    (.wire 32 34, .virt 18979),
    (.wire 32 35, .virt 18979),
    (.wire 32 36, .virt 18979),
    (.wire 32 37, .virt 18979),
    (.wire 32 38, .virt 18979),
    (.wire 32 39, .virt 18979),
    (.wire 32 40, .virt 18979),
    (.wire 32 41, .virt 18979),
    (.wire 32 42, .virt 18979),
    (.wire 32 43, .virt 18979),
    (.wire 32 44, .virt 18979),
    (.wire 32 45, .virt 18979),
    (.wire 32 46, .virt 18979),
    (.wire 32 47, .virt 18979),
    (.wire 32 48, .virt 18979),
    (.wire 32 49, .virt 18979),
    (.wire 32 50, .virt 18979),
    (.wire 32 51, .virt 18979),
    (.wire 32 52, .virt 18979),
    (.wire 32 53, .virt 18979),
    (.wire 32 54, .virt 18979),
    (.wire 32 55, .virt 18979),
    (.wire 32 56, .virt 18979),
    (.wire 32 57, .virt 18979),
    (.wire 32 58, .virt 18979),
    (.wire 32 59, .virt 18979),
    (.wire 32 0, .wire 30 47),
    (.virt 18978, .wire 31 20),
    (.virt 18978, .wire 31 21),
    (.virt 19091, .wire 31 22)
  ]

/-- `privateBatchWrapper2.copies`, items `1472..1504`. -/
def privateBatchWrapper2.copies46 : List (Target × Target) := [
    (.wire 9 35, .wire 31 24),
    (.virt 18978, .wire 31 25),
    (.wire 11 55, .wire 31 26),
    (.virt 19091, .wire 33 0),
    (.wire 31 27, .wire 33 1),
    (.virt 19091, .wire 33 2),
    (.wire 31 27, .wire 33 4),
    (.virt 19092, .wire 33 5),
    (.wire 31 27, .wire 33 6),
    (.wire 33 7, .wire 31 28),
    (.virt 18978, .wire 31 29),
    (.wire 31 23, .wire 31 30),
    (.wire 33 3, .virt 18979),
    (.wire 31 31, .virt 18979),
    (.virt 18978, .wire 31 32),
    (.virt 18978, .wire 31 33),
    (.virt 19093, .wire 31 34),
    (.wire 9 43, .wire 31 36),
    (.virt 18978, .wire 31 37),
    (.wire 12 3, .wire 31 38),
    (.virt 19093, .wire 33 8),
    (.wire 31 39, .wire 33 9),
    (.virt 19093, .wire 33 10),
    (.wire 31 39, .wire 33 12),
    (.virt 19094, .wire 33 13),
    (.wire 31 39, .wire 33 14),
    (.wire 33 15, .wire 31 40),
    (.virt 18978, .wire 31 41),
    (.wire 31 35, .wire 31 42),
    (.wire 33 11, .virt 18979),
    (.wire 31 43, .virt 18979),
    (.virt 18978, .wire 31 44)
  ]

/-- `privateBatchWrapper2.copies`, items `1504..1536`. -/
def privateBatchWrapper2.copies47 : List (Target × Target) := [
    (.virt 18978, .wire 31 45),
    (.virt 19095, .wire 31 46),
    (.wire 9 51, .wire 31 48),
    (.virt 18978, .wire 31 49),
    (.wire 12 11, .wire 31 50),
    (.virt 19095, .wire 33 16),
    (.wire 31 51, .wire 33 17),
    (.virt 19095, .wire 33 18),
    (.wire 31 51, .wire 33 20),
    (.virt 19096, .wire 33 21),
    (.wire 31 51, .wire 33 22),
    (.wire 33 23, .wire 31 52),
    (.virt 18978, .wire 31 53),
    (.wire 31 47, .wire 31 54),
    (.wire 33 19, .virt 18979),
    (.wire 31 55, .virt 18979),
    (.virt 18978, .wire 31 56),
    (.virt 18978, .wire 31 57),
    (.virt 19097, .wire 31 58),
    (.wire 9 59, .wire 34 0),
    (.virt 18978, .wire 34 1),
    (.wire 12 19, .wire 34 2),
    (.virt 19097, .wire 33 24),
    (.wire 34 3, .wire 33 25),
    (.virt 19097, .wire 33 26),
    (.wire 34 3, .wire 33 28),
    (.virt 19098, .wire 33 29),
    (.wire 34 3, .wire 33 30),
    (.wire 33 31, .wire 34 4),
    (.virt 18978, .wire 34 5),
    (.wire 31 59, .wire 34 6),
    (.wire 33 27, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `1536..1568`. -/
def privateBatchWrapper2.copies48 : List (Target × Target) := [
    (.wire 34 7, .virt 18979),
    (.virt 19091, .wire 33 32),
    (.virt 19093, .wire 33 33),
    (.virt 19091, .wire 33 34),
    (.virt 19095, .wire 33 36),
    (.virt 19097, .wire 33 37),
    (.virt 19095, .wire 33 38),
    (.wire 33 35, .wire 33 40),
    (.wire 33 39, .wire 33 41),
    (.wire 33 35, .wire 33 42),
    (.virt 18978, .wire 34 8),
    (.virt 18978, .wire 34 9),
    (.virt 19099, .wire 34 10),
    (.wire 11 15, .wire 34 12),
    (.virt 18978, .wire 34 13),
    (.wire 11 55, .wire 34 14),
    (.virt 19099, .wire 33 44),
    (.wire 34 15, .wire 33 45),
    (.virt 19099, .wire 33 46),
    (.wire 34 15, .wire 33 48),
    (.virt 19100, .wire 33 49),
    (.wire 34 15, .wire 33 50),
    (.wire 33 51, .wire 34 16),
    (.virt 18978, .wire 34 17),
    (.wire 34 11, .wire 34 18),
    (.wire 33 47, .virt 18979),
    (.wire 34 19, .virt 18979),
    (.virt 18978, .wire 34 20),
    (.virt 18978, .wire 34 21),
    (.virt 19101, .wire 34 22),
    (.wire 11 23, .wire 34 24),
    (.virt 18978, .wire 34 25)
  ]

/-- `privateBatchWrapper2.copies`, items `1568..1600`. -/
def privateBatchWrapper2.copies49 : List (Target × Target) := [
    (.wire 12 3, .wire 34 26),
    (.virt 19101, .wire 33 52),
    (.wire 34 27, .wire 33 53),
    (.virt 19101, .wire 33 54),
    (.wire 34 27, .wire 33 56),
    (.virt 19102, .wire 33 57),
    (.wire 34 27, .wire 33 58),
    (.wire 33 59, .wire 34 28),
    (.virt 18978, .wire 34 29),
    (.wire 34 23, .wire 34 30),
    (.wire 33 55, .virt 18979),
    (.wire 34 31, .virt 18979),
    (.virt 18978, .wire 34 32),
    (.virt 18978, .wire 34 33),
    (.virt 19103, .wire 34 34),
    (.wire 11 31, .wire 34 36),
    (.virt 18978, .wire 34 37),
    (.wire 12 11, .wire 34 38),
    (.virt 19103, .wire 35 0),
    (.wire 34 39, .wire 35 1),
    (.virt 19103, .wire 35 2),
    (.wire 34 39, .wire 35 4),
    (.virt 19104, .wire 35 5),
    (.wire 34 39, .wire 35 6),
    (.wire 35 7, .wire 34 40),
    (.virt 18978, .wire 34 41),
    (.wire 34 35, .wire 34 42),
    (.wire 35 3, .virt 18979),
    (.wire 34 43, .virt 18979),
    (.virt 18978, .wire 34 44),
    (.virt 18978, .wire 34 45),
    (.virt 19105, .wire 34 46)
  ]

/-- `privateBatchWrapper2.copies`, items `1600..1632`. -/
def privateBatchWrapper2.copies50 : List (Target × Target) := [
    (.wire 11 39, .wire 34 48),
    (.virt 18978, .wire 34 49),
    (.wire 12 19, .wire 34 50),
    (.virt 19105, .wire 35 8),
    (.wire 34 51, .wire 35 9),
    (.virt 19105, .wire 35 10),
    (.wire 34 51, .wire 35 12),
    (.virt 19106, .wire 35 13),
    (.wire 34 51, .wire 35 14),
    (.wire 35 15, .wire 34 52),
    (.virt 18978, .wire 34 53),
    (.wire 34 47, .wire 34 54),
    (.wire 35 11, .virt 18979),
    (.wire 34 55, .virt 18979),
    (.virt 19099, .wire 35 16),
    (.virt 19101, .wire 35 17),
    (.virt 19099, .wire 35 18),
    (.virt 19103, .wire 35 20),
    (.virt 19105, .wire 35 21),
    (.virt 19103, .wire 35 22),
    (.wire 35 19, .wire 35 24),
    (.wire 35 23, .wire 35 25),
    (.wire 35 19, .wire 35 26),
    (.wire 33 43, .wire 5 20),
    (.wire 35 27, .wire 5 21),
    (.wire 33 43, .wire 5 22),
    (.wire 5 23, .wire 36 0),
    (.virt 18978, .wire 36 1),
    (.wire 35 27, .wire 36 2),
    (.virt 18978, .wire 34 56),
    (.virt 18978, .wire 34 57),
    (.virt 19107, .wire 34 58)
  ]

/-- `privateBatchWrapper2.copies`, items `1632..1664`. -/
def privateBatchWrapper2.copies51 : List (Target × Target) := [
    (.virt 19107, .wire 35 28),
    (.wire 31 27, .wire 35 29),
    (.virt 19107, .wire 35 30),
    (.wire 31 27, .wire 35 32),
    (.virt 19108, .wire 35 33),
    (.wire 31 27, .wire 35 34),
    (.wire 35 35, .wire 37 0),
    (.virt 18978, .wire 37 1),
    (.wire 34 59, .wire 37 2),
    (.wire 35 31, .virt 18979),
    (.wire 37 3, .virt 18979),
    (.virt 18978, .wire 37 4),
    (.virt 18978, .wire 37 5),
    (.virt 19109, .wire 37 6),
    (.virt 19109, .wire 35 36),
    (.wire 31 39, .wire 35 37),
    (.virt 19109, .wire 35 38),
    (.wire 31 39, .wire 35 40),
    (.virt 19110, .wire 35 41),
    (.wire 31 39, .wire 35 42),
    (.wire 35 43, .wire 37 8),
    (.virt 18978, .wire 37 9),
    (.wire 37 7, .wire 37 10),
    (.wire 35 39, .virt 18979),
    (.wire 37 11, .virt 18979),
    (.virt 18978, .wire 37 12),
    (.virt 18978, .wire 37 13),
    (.virt 19111, .wire 37 14),
    (.virt 19111, .wire 35 44),
    (.wire 31 51, .wire 35 45),
    (.virt 19111, .wire 35 46),
    (.wire 31 51, .wire 35 48)
  ]

/-- `privateBatchWrapper2.copies`, items `1664..1696`. -/
def privateBatchWrapper2.copies52 : List (Target × Target) := [
    (.virt 19112, .wire 35 49),
    (.wire 31 51, .wire 35 50),
    (.wire 35 51, .wire 37 16),
    (.virt 18978, .wire 37 17),
    (.wire 37 15, .wire 37 18),
    (.wire 35 47, .virt 18979),
    (.wire 37 19, .virt 18979),
    (.virt 18978, .wire 37 20),
    (.virt 18978, .wire 37 21),
    (.virt 19113, .wire 37 22),
    (.virt 19113, .wire 35 52),
    (.wire 34 3, .wire 35 53),
    (.virt 19113, .wire 35 54),
    (.wire 34 3, .wire 35 56),
    (.virt 19114, .wire 35 57),
    (.wire 34 3, .wire 35 58),
    (.wire 35 59, .wire 37 24),
    (.virt 18978, .wire 37 25),
    (.wire 37 23, .wire 37 26),
    (.wire 35 55, .virt 18979),
    (.wire 37 27, .virt 18979),
    (.virt 19107, .wire 38 0),
    (.virt 19109, .wire 38 1),
    (.virt 19107, .wire 38 2),
    (.virt 19111, .wire 38 4),
    (.virt 19113, .wire 38 5),
    (.virt 19111, .wire 38 6),
    (.wire 38 3, .wire 38 8),
    (.wire 38 7, .wire 38 9),
    (.wire 38 3, .wire 38 10),
    (.wire 38 11, .wire 37 28),
    (.wire 11 7, .wire 37 29)
  ]

/-- `privateBatchWrapper2.copies`, items `1696..1728`. -/
def privateBatchWrapper2.copies53 : List (Target × Target) := [
    (.virt 18979, .wire 37 30),
    (.virt 18978, .wire 37 32),
    (.virt 18978, .wire 37 33),
    (.virt 19115, .wire 37 34),
    (.virt 19115, .wire 38 12),
    (.wire 34 15, .wire 38 13),
    (.virt 19115, .wire 38 14),
    (.wire 34 15, .wire 38 16),
    (.virt 19116, .wire 38 17),
    (.wire 34 15, .wire 38 18),
    (.wire 38 19, .wire 37 36),
    (.virt 18978, .wire 37 37),
    (.wire 37 35, .wire 37 38),
    (.wire 38 15, .virt 18979),
    (.wire 37 39, .virt 18979),
    (.virt 18978, .wire 37 40),
    (.virt 18978, .wire 37 41),
    (.virt 19117, .wire 37 42),
    (.virt 19117, .wire 38 20),
    (.wire 34 27, .wire 38 21),
    (.virt 19117, .wire 38 22),
    (.wire 34 27, .wire 38 24),
    (.virt 19118, .wire 38 25),
    (.wire 34 27, .wire 38 26),
    (.wire 38 27, .wire 37 44),
    (.virt 18978, .wire 37 45),
    (.wire 37 43, .wire 37 46),
    (.wire 38 23, .virt 18979),
    (.wire 37 47, .virt 18979),
    (.virt 18978, .wire 37 48),
    (.virt 18978, .wire 37 49),
    (.virt 19119, .wire 37 50)
  ]

/-- `privateBatchWrapper2.copies`, items `1728..1760`. -/
def privateBatchWrapper2.copies54 : List (Target × Target) := [
    (.virt 19119, .wire 38 28),
    (.wire 34 39, .wire 38 29),
    (.virt 19119, .wire 38 30),
    (.wire 34 39, .wire 38 32),
    (.virt 19120, .wire 38 33),
    (.wire 34 39, .wire 38 34),
    (.wire 38 35, .wire 37 52),
    (.virt 18978, .wire 37 53),
    (.wire 37 51, .wire 37 54),
    (.wire 38 31, .virt 18979),
    (.wire 37 55, .virt 18979),
    (.virt 18978, .wire 37 56),
    (.virt 18978, .wire 37 57),
    (.virt 19121, .wire 37 58),
    (.virt 19121, .wire 38 36),
    (.wire 34 51, .wire 38 37),
    (.virt 19121, .wire 38 38),
    (.wire 34 51, .wire 38 40),
    (.virt 19122, .wire 38 41),
    (.wire 34 51, .wire 38 42),
    (.wire 38 43, .wire 39 0),
    (.virt 18978, .wire 39 1),
    (.wire 37 59, .wire 39 2),
    (.wire 38 39, .virt 18979),
    (.wire 39 3, .virt 18979),
    (.virt 19115, .wire 38 44),
    (.virt 19117, .wire 38 45),
    (.virt 19115, .wire 38 46),
    (.virt 19119, .wire 38 48),
    (.virt 19121, .wire 38 49),
    (.virt 19119, .wire 38 50),
    (.wire 38 47, .wire 38 52)
  ]

/-- `privateBatchWrapper2.copies`, items `1760..1792`. -/
def privateBatchWrapper2.copies55 : List (Target × Target) := [
    (.wire 38 51, .wire 38 53),
    (.wire 38 47, .wire 38 54),
    (.wire 38 55, .wire 39 4),
    (.wire 11 47, .wire 39 5),
    (.virt 18979, .wire 39 6),
    (.wire 37 31, .wire 36 4),
    (.virt 18978, .wire 36 5),
    (.wire 39 7, .wire 36 6),
    (.virt 18978, .wire 39 8),
    (.virt 18978, .wire 39 9),
    (.virt 19123, .wire 39 10),
    (.wire 11 55, .wire 39 12),
    (.virt 18978, .wire 39 13),
    (.wire 11 55, .wire 39 14),
    (.virt 19123, .wire 38 56),
    (.wire 39 15, .wire 38 57),
    (.virt 19123, .wire 38 58),
    (.wire 39 15, .wire 40 0),
    (.virt 19124, .wire 40 1),
    (.wire 39 15, .wire 40 2),
    (.wire 40 3, .wire 39 16),
    (.virt 18978, .wire 39 17),
    (.wire 39 11, .wire 39 18),
    (.wire 38 59, .virt 18979),
    (.wire 39 19, .virt 18979),
    (.virt 18978, .wire 39 20),
    (.virt 18978, .wire 39 21),
    (.virt 19125, .wire 39 22),
    (.wire 12 3, .wire 39 24),
    (.virt 18978, .wire 39 25),
    (.wire 12 3, .wire 39 26),
    (.virt 19125, .wire 40 4)
  ]

/-- `privateBatchWrapper2.copies`, items `1792..1824`. -/
def privateBatchWrapper2.copies56 : List (Target × Target) := [
    (.wire 39 27, .wire 40 5),
    (.virt 19125, .wire 40 6),
    (.wire 39 27, .wire 40 8),
    (.virt 19126, .wire 40 9),
    (.wire 39 27, .wire 40 10),
    (.wire 40 11, .wire 39 28),
    (.virt 18978, .wire 39 29),
    (.wire 39 23, .wire 39 30),
    (.wire 40 7, .virt 18979),
    (.wire 39 31, .virt 18979),
    (.virt 18978, .wire 39 32),
    (.virt 18978, .wire 39 33),
    (.virt 19127, .wire 39 34),
    (.wire 12 11, .wire 39 36),
    (.virt 18978, .wire 39 37),
    (.wire 12 11, .wire 39 38),
    (.virt 19127, .wire 40 12),
    (.wire 39 39, .wire 40 13),
    (.virt 19127, .wire 40 14),
    (.wire 39 39, .wire 40 16),
    (.virt 19128, .wire 40 17),
    (.wire 39 39, .wire 40 18),
    (.wire 40 19, .wire 39 40),
    (.virt 18978, .wire 39 41),
    (.wire 39 35, .wire 39 42),
    (.wire 40 15, .virt 18979),
    (.wire 39 43, .virt 18979),
    (.virt 18978, .wire 39 44),
    (.virt 18978, .wire 39 45),
    (.virt 19129, .wire 39 46),
    (.wire 12 19, .wire 39 48),
    (.virt 18978, .wire 39 49)
  ]

/-- `privateBatchWrapper2.copies`, items `1824..1856`. -/
def privateBatchWrapper2.copies57 : List (Target × Target) := [
    (.wire 12 19, .wire 39 50),
    (.virt 19129, .wire 40 20),
    (.wire 39 51, .wire 40 21),
    (.virt 19129, .wire 40 22),
    (.wire 39 51, .wire 40 24),
    (.virt 19130, .wire 40 25),
    (.wire 39 51, .wire 40 26),
    (.wire 40 27, .wire 39 52),
    (.virt 18978, .wire 39 53),
    (.wire 39 47, .wire 39 54),
    (.wire 40 23, .virt 18979),
    (.wire 39 55, .virt 18979),
    (.virt 19123, .wire 40 28),
    (.virt 19125, .wire 40 29),
    (.virt 19123, .wire 40 30),
    (.virt 19127, .wire 40 32),
    (.virt 19129, .wire 40 33),
    (.virt 19127, .wire 40 34),
    (.wire 40 31, .wire 40 36),
    (.wire 40 35, .wire 40 37),
    (.wire 40 31, .wire 40 38),
    (.wire 40 39, .wire 39 56),
    (.wire 12 27, .wire 39 57),
    (.virt 18979, .wire 39 58),
    (.wire 36 7, .wire 36 8),
    (.virt 18978, .wire 36 9),
    (.wire 39 59, .wire 36 10),
    (.virt 18978, .wire 41 0),
    (.virt 18978, .wire 41 1),
    (.virt 19131, .wire 41 2),
    (.wire 12 35, .wire 41 4),
    (.virt 18978, .wire 41 5)
  ]

/-- `privateBatchWrapper2.copies`, items `1856..1888`. -/
def privateBatchWrapper2.copies58 : List (Target × Target) := [
    (.wire 11 55, .wire 41 6),
    (.virt 19131, .wire 40 40),
    (.wire 41 7, .wire 40 41),
    (.virt 19131, .wire 40 42),
    (.wire 41 7, .wire 40 44),
    (.virt 19132, .wire 40 45),
    (.wire 41 7, .wire 40 46),
    (.wire 40 47, .wire 41 8),
    (.virt 18978, .wire 41 9),
    (.wire 41 3, .wire 41 10),
    (.wire 40 43, .virt 18979),
    (.wire 41 11, .virt 18979),
    (.virt 18978, .wire 41 12),
    (.virt 18978, .wire 41 13),
    (.virt 19133, .wire 41 14),
    (.wire 12 43, .wire 41 16),
    (.virt 18978, .wire 41 17),
    (.wire 12 3, .wire 41 18),
    (.virt 19133, .wire 40 48),
    (.wire 41 19, .wire 40 49),
    (.virt 19133, .wire 40 50),
    (.wire 41 19, .wire 40 52),
    (.virt 19134, .wire 40 53),
    (.wire 41 19, .wire 40 54),
    (.wire 40 55, .wire 41 20),
    (.virt 18978, .wire 41 21),
    (.wire 41 15, .wire 41 22),
    (.wire 40 51, .virt 18979),
    (.wire 41 23, .virt 18979),
    (.virt 18978, .wire 41 24),
    (.virt 18978, .wire 41 25),
    (.virt 19135, .wire 41 26)
  ]

/-- `privateBatchWrapper2.copies`, items `1888..1920`. -/
def privateBatchWrapper2.copies59 : List (Target × Target) := [
    (.wire 12 51, .wire 41 28),
    (.virt 18978, .wire 41 29),
    (.wire 12 11, .wire 41 30),
    (.virt 19135, .wire 40 56),
    (.wire 41 31, .wire 40 57),
    (.virt 19135, .wire 40 58),
    (.wire 41 31, .wire 42 0),
    (.virt 19136, .wire 42 1),
    (.wire 41 31, .wire 42 2),
    (.wire 42 3, .wire 41 32),
    (.virt 18978, .wire 41 33),
    (.wire 41 27, .wire 41 34),
    (.wire 40 59, .virt 18979),
    (.wire 41 35, .virt 18979),
    (.virt 18978, .wire 41 36),
    (.virt 18978, .wire 41 37),
    (.virt 19137, .wire 41 38),
    (.wire 12 59, .wire 41 40),
    (.virt 18978, .wire 41 41),
    (.wire 12 19, .wire 41 42),
    (.virt 19137, .wire 42 4),
    (.wire 41 43, .wire 42 5),
    (.virt 19137, .wire 42 6),
    (.wire 41 43, .wire 42 8),
    (.virt 19138, .wire 42 9),
    (.wire 41 43, .wire 42 10),
    (.wire 42 11, .wire 41 44),
    (.virt 18978, .wire 41 45),
    (.wire 41 39, .wire 41 46),
    (.wire 42 7, .virt 18979),
    (.wire 41 47, .virt 18979),
    (.virt 19131, .wire 42 12)
  ]

/-- `privateBatchWrapper2.copies`, items `1920..1952`. -/
def privateBatchWrapper2.copies60 : List (Target × Target) := [
    (.virt 19133, .wire 42 13),
    (.virt 19131, .wire 42 14),
    (.virt 19135, .wire 42 16),
    (.virt 19137, .wire 42 17),
    (.virt 19135, .wire 42 18),
    (.wire 42 15, .wire 42 20),
    (.wire 42 19, .wire 42 21),
    (.wire 42 15, .wire 42 22),
    (.wire 42 23, .wire 41 48),
    (.wire 13 7, .wire 41 49),
    (.virt 18979, .wire 41 50),
    (.wire 36 11, .wire 36 12),
    (.virt 18978, .wire 36 13),
    (.wire 41 51, .wire 36 14),
    (.wire 36 3, .wire 41 52),
    (.wire 36 15, .wire 41 53),
    (.wire 36 15, .wire 41 54),
    (.wire 36 3, .wire 41 56),
    (.virt 18979, .wire 41 57),
    (.wire 41 55, .wire 41 58),
    (.wire 36 3, .wire 43 0),
    (.wire 11 55, .wire 43 1),
    (.wire 11 55, .wire 43 2),
    (.wire 36 3, .wire 43 4),
    (.virt 18979, .wire 43 5),
    (.wire 43 3, .wire 43 6),
    (.wire 36 3, .wire 43 8),
    (.wire 12 3, .wire 43 9),
    (.wire 12 3, .wire 43 10),
    (.wire 36 3, .wire 43 12),
    (.virt 18979, .wire 43 13),
    (.wire 43 11, .wire 43 14)
  ]

/-- `privateBatchWrapper2.copies`, items `1952..1984`. -/
def privateBatchWrapper2.copies61 : List (Target × Target) := [
    (.wire 36 3, .wire 43 16),
    (.wire 12 11, .wire 43 17),
    (.wire 12 11, .wire 43 18),
    (.wire 36 3, .wire 43 20),
    (.virt 18979, .wire 43 21),
    (.wire 43 19, .wire 43 22),
    (.wire 36 3, .wire 43 24),
    (.wire 12 19, .wire 43 25),
    (.wire 12 19, .wire 43 26),
    (.wire 36 3, .wire 43 28),
    (.virt 18979, .wire 43 29),
    (.wire 43 27, .wire 43 30),
    (.wire 44 33, .virt 18979),
    (.wire 44 34, .virt 18979),
    (.wire 44 35, .virt 18979),
    (.wire 44 36, .virt 18979),
    (.wire 44 37, .virt 18979),
    (.wire 44 38, .virt 18979),
    (.wire 44 39, .virt 18979),
    (.wire 44 40, .virt 18979),
    (.wire 44 41, .virt 18979),
    (.wire 44 42, .virt 18979),
    (.wire 44 43, .virt 18979),
    (.wire 44 44, .virt 18979),
    (.wire 44 45, .virt 18979),
    (.wire 44 46, .virt 18979),
    (.wire 44 47, .virt 18979),
    (.wire 44 48, .virt 18979),
    (.wire 44 49, .virt 18979),
    (.wire 44 50, .virt 18979),
    (.wire 44 51, .virt 18979),
    (.wire 44 52, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `1984..2016`. -/
def privateBatchWrapper2.copies62 : List (Target × Target) := [
    (.wire 44 53, .virt 18979),
    (.wire 44 54, .virt 18979),
    (.wire 44 55, .virt 18979),
    (.wire 44 56, .virt 18979),
    (.wire 44 57, .virt 18979),
    (.wire 44 58, .virt 18979),
    (.wire 44 59, .virt 18979),
    (.wire 44 0, .wire 41 59),
    (.virt 18978, .wire 43 32),
    (.virt 18978, .wire 43 33),
    (.virt 19139, .wire 43 34),
    (.wire 9 35, .wire 43 36),
    (.virt 18978, .wire 43 37),
    (.wire 12 35, .wire 43 38),
    (.virt 19139, .wire 42 24),
    (.wire 43 39, .wire 42 25),
    (.virt 19139, .wire 42 26),
    (.wire 43 39, .wire 42 28),
    (.virt 19140, .wire 42 29),
    (.wire 43 39, .wire 42 30),
    (.wire 42 31, .wire 43 40),
    (.virt 18978, .wire 43 41),
    (.wire 43 35, .wire 43 42),
    (.wire 42 27, .virt 18979),
    (.wire 43 43, .virt 18979),
    (.virt 18978, .wire 43 44),
    (.virt 18978, .wire 43 45),
    (.virt 19141, .wire 43 46),
    (.wire 9 43, .wire 43 48),
    (.virt 18978, .wire 43 49),
    (.wire 12 43, .wire 43 50),
    (.virt 19141, .wire 42 32)
  ]

/-- `privateBatchWrapper2.copies`, items `2016..2048`. -/
def privateBatchWrapper2.copies63 : List (Target × Target) := [
    (.wire 43 51, .wire 42 33),
    (.virt 19141, .wire 42 34),
    (.wire 43 51, .wire 42 36),
    (.virt 19142, .wire 42 37),
    (.wire 43 51, .wire 42 38),
    (.wire 42 39, .wire 43 52),
    (.virt 18978, .wire 43 53),
    (.wire 43 47, .wire 43 54),
    (.wire 42 35, .virt 18979),
    (.wire 43 55, .virt 18979),
    (.virt 18978, .wire 43 56),
    (.virt 18978, .wire 43 57),
    (.virt 19143, .wire 43 58),
    (.wire 9 51, .wire 45 0),
    (.virt 18978, .wire 45 1),
    (.wire 12 51, .wire 45 2),
    (.virt 19143, .wire 42 40),
    (.wire 45 3, .wire 42 41),
    (.virt 19143, .wire 42 42),
    (.wire 45 3, .wire 42 44),
    (.virt 19144, .wire 42 45),
    (.wire 45 3, .wire 42 46),
    (.wire 42 47, .wire 45 4),
    (.virt 18978, .wire 45 5),
    (.wire 43 59, .wire 45 6),
    (.wire 42 43, .virt 18979),
    (.wire 45 7, .virt 18979),
    (.virt 18978, .wire 45 8),
    (.virt 18978, .wire 45 9),
    (.virt 19145, .wire 45 10),
    (.wire 9 59, .wire 45 12),
    (.virt 18978, .wire 45 13)
  ]

/-- `privateBatchWrapper2.copies`, items `2048..2080`. -/
def privateBatchWrapper2.copies64 : List (Target × Target) := [
    (.wire 12 59, .wire 45 14),
    (.virt 19145, .wire 42 48),
    (.wire 45 15, .wire 42 49),
    (.virt 19145, .wire 42 50),
    (.wire 45 15, .wire 42 52),
    (.virt 19146, .wire 42 53),
    (.wire 45 15, .wire 42 54),
    (.wire 42 55, .wire 45 16),
    (.virt 18978, .wire 45 17),
    (.wire 45 11, .wire 45 18),
    (.wire 42 51, .virt 18979),
    (.wire 45 19, .virt 18979),
    (.virt 19139, .wire 42 56),
    (.virt 19141, .wire 42 57),
    (.virt 19139, .wire 42 58),
    (.virt 19143, .wire 46 0),
    (.virt 19145, .wire 46 1),
    (.virt 19143, .wire 46 2),
    (.wire 42 59, .wire 46 4),
    (.wire 46 3, .wire 46 5),
    (.wire 42 59, .wire 46 6),
    (.virt 18978, .wire 45 20),
    (.virt 18978, .wire 45 21),
    (.virt 19147, .wire 45 22),
    (.wire 11 15, .wire 45 24),
    (.virt 18978, .wire 45 25),
    (.wire 12 35, .wire 45 26),
    (.virt 19147, .wire 46 8),
    (.wire 45 27, .wire 46 9),
    (.virt 19147, .wire 46 10),
    (.wire 45 27, .wire 46 12),
    (.virt 19148, .wire 46 13)
  ]

/-- `privateBatchWrapper2.copies`, items `2080..2112`. -/
def privateBatchWrapper2.copies65 : List (Target × Target) := [
    (.wire 45 27, .wire 46 14),
    (.wire 46 15, .wire 45 28),
    (.virt 18978, .wire 45 29),
    (.wire 45 23, .wire 45 30),
    (.wire 46 11, .virt 18979),
    (.wire 45 31, .virt 18979),
    (.virt 18978, .wire 45 32),
    (.virt 18978, .wire 45 33),
    (.virt 19149, .wire 45 34),
    (.wire 11 23, .wire 45 36),
    (.virt 18978, .wire 45 37),
    (.wire 12 43, .wire 45 38),
    (.virt 19149, .wire 46 16),
    (.wire 45 39, .wire 46 17),
    (.virt 19149, .wire 46 18),
    (.wire 45 39, .wire 46 20),
    (.virt 19150, .wire 46 21),
    (.wire 45 39, .wire 46 22),
    (.wire 46 23, .wire 45 40),
    (.virt 18978, .wire 45 41),
    (.wire 45 35, .wire 45 42),
    (.wire 46 19, .virt 18979),
    (.wire 45 43, .virt 18979),
    (.virt 18978, .wire 45 44),
    (.virt 18978, .wire 45 45),
    (.virt 19151, .wire 45 46),
    (.wire 11 31, .wire 45 48),
    (.virt 18978, .wire 45 49),
    (.wire 12 51, .wire 45 50),
    (.virt 19151, .wire 46 24),
    (.wire 45 51, .wire 46 25),
    (.virt 19151, .wire 46 26)
  ]

/-- `privateBatchWrapper2.copies`, items `2112..2144`. -/
def privateBatchWrapper2.copies66 : List (Target × Target) := [
    (.wire 45 51, .wire 46 28),
    (.virt 19152, .wire 46 29),
    (.wire 45 51, .wire 46 30),
    (.wire 46 31, .wire 45 52),
    (.virt 18978, .wire 45 53),
    (.wire 45 47, .wire 45 54),
    (.wire 46 27, .virt 18979),
    (.wire 45 55, .virt 18979),
    (.virt 18978, .wire 45 56),
    (.virt 18978, .wire 45 57),
    (.virt 19153, .wire 45 58),
    (.wire 11 39, .wire 47 0),
    (.virt 18978, .wire 47 1),
    (.wire 12 59, .wire 47 2),
    (.virt 19153, .wire 46 32),
    (.wire 47 3, .wire 46 33),
    (.virt 19153, .wire 46 34),
    (.wire 47 3, .wire 46 36),
    (.virt 19154, .wire 46 37),
    (.wire 47 3, .wire 46 38),
    (.wire 46 39, .wire 47 4),
    (.virt 18978, .wire 47 5),
    (.wire 45 59, .wire 47 6),
    (.wire 46 35, .virt 18979),
    (.wire 47 7, .virt 18979),
    (.virt 19147, .wire 46 40),
    (.virt 19149, .wire 46 41),
    (.virt 19147, .wire 46 42),
    (.virt 19151, .wire 46 44),
    (.virt 19153, .wire 46 45),
    (.virt 19151, .wire 46 46),
    (.wire 46 43, .wire 46 48)
  ]

/-- `privateBatchWrapper2.copies`, items `2144..2176`. -/
def privateBatchWrapper2.copies67 : List (Target × Target) := [
    (.wire 46 47, .wire 46 49),
    (.wire 46 43, .wire 46 50),
    (.wire 46 7, .wire 5 24),
    (.wire 46 51, .wire 5 25),
    (.wire 46 7, .wire 5 26),
    (.wire 5 27, .wire 36 16),
    (.virt 18978, .wire 36 17),
    (.wire 46 51, .wire 36 18),
    (.virt 18978, .wire 47 8),
    (.virt 18978, .wire 47 9),
    (.virt 19155, .wire 47 10),
    (.wire 11 55, .wire 47 12),
    (.virt 18978, .wire 47 13),
    (.wire 12 35, .wire 47 14),
    (.virt 19155, .wire 46 52),
    (.wire 47 15, .wire 46 53),
    (.virt 19155, .wire 46 54),
    (.wire 47 15, .wire 46 56),
    (.virt 19156, .wire 46 57),
    (.wire 47 15, .wire 46 58),
    (.wire 46 59, .wire 47 16),
    (.virt 18978, .wire 47 17),
    (.wire 47 11, .wire 47 18),
    (.wire 46 55, .virt 18979),
    (.wire 47 19, .virt 18979),
    (.virt 18978, .wire 47 20),
    (.virt 18978, .wire 47 21),
    (.virt 19157, .wire 47 22),
    (.wire 12 3, .wire 47 24),
    (.virt 18978, .wire 47 25),
    (.wire 12 43, .wire 47 26),
    (.virt 19157, .wire 48 0)
  ]

/-- `privateBatchWrapper2.copies`, items `2176..2208`. -/
def privateBatchWrapper2.copies68 : List (Target × Target) := [
    (.wire 47 27, .wire 48 1),
    (.virt 19157, .wire 48 2),
    (.wire 47 27, .wire 48 4),
    (.virt 19158, .wire 48 5),
    (.wire 47 27, .wire 48 6),
    (.wire 48 7, .wire 47 28),
    (.virt 18978, .wire 47 29),
    (.wire 47 23, .wire 47 30),
    (.wire 48 3, .virt 18979),
    (.wire 47 31, .virt 18979),
    (.virt 18978, .wire 47 32),
    (.virt 18978, .wire 47 33),
    (.virt 19159, .wire 47 34),
    (.wire 12 11, .wire 47 36),
    (.virt 18978, .wire 47 37),
    (.wire 12 51, .wire 47 38),
    (.virt 19159, .wire 48 8),
    (.wire 47 39, .wire 48 9),
    (.virt 19159, .wire 48 10),
    (.wire 47 39, .wire 48 12),
    (.virt 19160, .wire 48 13),
    (.wire 47 39, .wire 48 14),
    (.wire 48 15, .wire 47 40),
    (.virt 18978, .wire 47 41),
    (.wire 47 35, .wire 47 42),
    (.wire 48 11, .virt 18979),
    (.wire 47 43, .virt 18979),
    (.virt 18978, .wire 47 44),
    (.virt 18978, .wire 47 45),
    (.virt 19161, .wire 47 46),
    (.wire 12 19, .wire 47 48),
    (.virt 18978, .wire 47 49)
  ]

/-- `privateBatchWrapper2.copies`, items `2208..2240`. -/
def privateBatchWrapper2.copies69 : List (Target × Target) := [
    (.wire 12 59, .wire 47 50),
    (.virt 19161, .wire 48 16),
    (.wire 47 51, .wire 48 17),
    (.virt 19161, .wire 48 18),
    (.wire 47 51, .wire 48 20),
    (.virt 19162, .wire 48 21),
    (.wire 47 51, .wire 48 22),
    (.wire 48 23, .wire 47 52),
    (.virt 18978, .wire 47 53),
    (.wire 47 47, .wire 47 54),
    (.wire 48 19, .virt 18979),
    (.wire 47 55, .virt 18979),
    (.virt 19155, .wire 48 24),
    (.virt 19157, .wire 48 25),
    (.virt 19155, .wire 48 26),
    (.virt 19159, .wire 48 28),
    (.virt 19161, .wire 48 29),
    (.virt 19159, .wire 48 30),
    (.wire 48 27, .wire 48 32),
    (.wire 48 31, .wire 48 33),
    (.wire 48 27, .wire 48 34),
    (.wire 36 19, .wire 5 28),
    (.wire 48 35, .wire 5 29),
    (.wire 36 19, .wire 5 30),
    (.wire 5 31, .wire 36 20),
    (.virt 18978, .wire 36 21),
    (.wire 48 35, .wire 36 22),
    (.virt 18978, .wire 47 56),
    (.virt 18978, .wire 47 57),
    (.virt 19163, .wire 47 58),
    (.virt 19163, .wire 48 36),
    (.wire 43 39, .wire 48 37)
  ]

/-- `privateBatchWrapper2.copies`, items `2240..2272`. -/
def privateBatchWrapper2.copies70 : List (Target × Target) := [
    (.virt 19163, .wire 48 38),
    (.wire 43 39, .wire 48 40),
    (.virt 19164, .wire 48 41),
    (.wire 43 39, .wire 48 42),
    (.wire 48 43, .wire 49 0),
    (.virt 18978, .wire 49 1),
    (.wire 47 59, .wire 49 2),
    (.wire 48 39, .virt 18979),
    (.wire 49 3, .virt 18979),
    (.virt 18978, .wire 49 4),
    (.virt 18978, .wire 49 5),
    (.virt 19165, .wire 49 6),
    (.virt 19165, .wire 48 44),
    (.wire 43 51, .wire 48 45),
    (.virt 19165, .wire 48 46),
    (.wire 43 51, .wire 48 48),
    (.virt 19166, .wire 48 49),
    (.wire 43 51, .wire 48 50),
    (.wire 48 51, .wire 49 8),
    (.virt 18978, .wire 49 9),
    (.wire 49 7, .wire 49 10),
    (.wire 48 47, .virt 18979),
    (.wire 49 11, .virt 18979),
    (.virt 18978, .wire 49 12),
    (.virt 18978, .wire 49 13),
    (.virt 19167, .wire 49 14),
    (.virt 19167, .wire 48 52),
    (.wire 45 3, .wire 48 53),
    (.virt 19167, .wire 48 54),
    (.wire 45 3, .wire 48 56),
    (.virt 19168, .wire 48 57),
    (.wire 45 3, .wire 48 58)
  ]

/-- `privateBatchWrapper2.copies`, items `2272..2304`. -/
def privateBatchWrapper2.copies71 : List (Target × Target) := [
    (.wire 48 59, .wire 49 16),
    (.virt 18978, .wire 49 17),
    (.wire 49 15, .wire 49 18),
    (.wire 48 55, .virt 18979),
    (.wire 49 19, .virt 18979),
    (.virt 18978, .wire 49 20),
    (.virt 18978, .wire 49 21),
    (.virt 19169, .wire 49 22),
    (.virt 19169, .wire 50 0),
    (.wire 45 15, .wire 50 1),
    (.virt 19169, .wire 50 2),
    (.wire 45 15, .wire 50 4),
    (.virt 19170, .wire 50 5),
    (.wire 45 15, .wire 50 6),
    (.wire 50 7, .wire 49 24),
    (.virt 18978, .wire 49 25),
    (.wire 49 23, .wire 49 26),
    (.wire 50 3, .virt 18979),
    (.wire 49 27, .virt 18979),
    (.virt 19163, .wire 50 8),
    (.virt 19165, .wire 50 9),
    (.virt 19163, .wire 50 10),
    (.virt 19167, .wire 50 12),
    (.virt 19169, .wire 50 13),
    (.virt 19167, .wire 50 14),
    (.wire 50 11, .wire 50 16),
    (.wire 50 15, .wire 50 17),
    (.wire 50 11, .wire 50 18),
    (.wire 50 19, .wire 49 28),
    (.wire 11 7, .wire 49 29),
    (.virt 18979, .wire 49 30),
    (.virt 18978, .wire 49 32)
  ]

/-- `privateBatchWrapper2.copies`, items `2304..2336`. -/
def privateBatchWrapper2.copies72 : List (Target × Target) := [
    (.virt 18978, .wire 49 33),
    (.virt 19171, .wire 49 34),
    (.virt 19171, .wire 50 20),
    (.wire 45 27, .wire 50 21),
    (.virt 19171, .wire 50 22),
    (.wire 45 27, .wire 50 24),
    (.virt 19172, .wire 50 25),
    (.wire 45 27, .wire 50 26),
    (.wire 50 27, .wire 49 36),
    (.virt 18978, .wire 49 37),
    (.wire 49 35, .wire 49 38),
    (.wire 50 23, .virt 18979),
    (.wire 49 39, .virt 18979),
    (.virt 18978, .wire 49 40),
    (.virt 18978, .wire 49 41),
    (.virt 19173, .wire 49 42),
    (.virt 19173, .wire 50 28),
    (.wire 45 39, .wire 50 29),
    (.virt 19173, .wire 50 30),
    (.wire 45 39, .wire 50 32),
    (.virt 19174, .wire 50 33),
    (.wire 45 39, .wire 50 34),
    (.wire 50 35, .wire 49 44),
    (.virt 18978, .wire 49 45),
    (.wire 49 43, .wire 49 46),
    (.wire 50 31, .virt 18979),
    (.wire 49 47, .virt 18979),
    (.virt 18978, .wire 49 48),
    (.virt 18978, .wire 49 49),
    (.virt 19175, .wire 49 50),
    (.virt 19175, .wire 50 36),
    (.wire 45 51, .wire 50 37)
  ]

/-- `privateBatchWrapper2.copies`, items `2336..2368`. -/
def privateBatchWrapper2.copies73 : List (Target × Target) := [
    (.virt 19175, .wire 50 38),
    (.wire 45 51, .wire 50 40),
    (.virt 19176, .wire 50 41),
    (.wire 45 51, .wire 50 42),
    (.wire 50 43, .wire 49 52),
    (.virt 18978, .wire 49 53),
    (.wire 49 51, .wire 49 54),
    (.wire 50 39, .virt 18979),
    (.wire 49 55, .virt 18979),
    (.virt 18978, .wire 49 56),
    (.virt 18978, .wire 49 57),
    (.virt 19177, .wire 49 58),
    (.virt 19177, .wire 50 44),
    (.wire 47 3, .wire 50 45),
    (.virt 19177, .wire 50 46),
    (.wire 47 3, .wire 50 48),
    (.virt 19178, .wire 50 49),
    (.wire 47 3, .wire 50 50),
    (.wire 50 51, .wire 51 0),
    (.virt 18978, .wire 51 1),
    (.wire 49 59, .wire 51 2),
    (.wire 50 47, .virt 18979),
    (.wire 51 3, .virt 18979),
    (.virt 19171, .wire 50 52),
    (.virt 19173, .wire 50 53),
    (.virt 19171, .wire 50 54),
    (.virt 19175, .wire 50 56),
    (.virt 19177, .wire 50 57),
    (.virt 19175, .wire 50 58),
    (.wire 50 55, .wire 52 0),
    (.wire 50 59, .wire 52 1),
    (.wire 50 55, .wire 52 2)
  ]

/-- `privateBatchWrapper2.copies`, items `2368..2400`. -/
def privateBatchWrapper2.copies74 : List (Target × Target) := [
    (.wire 52 3, .wire 51 4),
    (.wire 11 47, .wire 51 5),
    (.virt 18979, .wire 51 6),
    (.wire 49 31, .wire 36 24),
    (.virt 18978, .wire 36 25),
    (.wire 51 7, .wire 36 26),
    (.virt 18978, .wire 51 8),
    (.virt 18978, .wire 51 9),
    (.virt 19179, .wire 51 10),
    (.virt 19179, .wire 52 4),
    (.wire 47 15, .wire 52 5),
    (.virt 19179, .wire 52 6),
    (.wire 47 15, .wire 52 8),
    (.virt 19180, .wire 52 9),
    (.wire 47 15, .wire 52 10),
    (.wire 52 11, .wire 51 12),
    (.virt 18978, .wire 51 13),
    (.wire 51 11, .wire 51 14),
    (.wire 52 7, .virt 18979),
    (.wire 51 15, .virt 18979),
    (.virt 18978, .wire 51 16),
    (.virt 18978, .wire 51 17),
    (.virt 19181, .wire 51 18),
    (.virt 19181, .wire 52 12),
    (.wire 47 27, .wire 52 13),
    (.virt 19181, .wire 52 14),
    (.wire 47 27, .wire 52 16),
    (.virt 19182, .wire 52 17),
    (.wire 47 27, .wire 52 18),
    (.wire 52 19, .wire 51 20),
    (.virt 18978, .wire 51 21),
    (.wire 51 19, .wire 51 22)
  ]

/-- `privateBatchWrapper2.copies`, items `2400..2432`. -/
def privateBatchWrapper2.copies75 : List (Target × Target) := [
    (.wire 52 15, .virt 18979),
    (.wire 51 23, .virt 18979),
    (.virt 18978, .wire 51 24),
    (.virt 18978, .wire 51 25),
    (.virt 19183, .wire 51 26),
    (.virt 19183, .wire 52 20),
    (.wire 47 39, .wire 52 21),
    (.virt 19183, .wire 52 22),
    (.wire 47 39, .wire 52 24),
    (.virt 19184, .wire 52 25),
    (.wire 47 39, .wire 52 26),
    (.wire 52 27, .wire 51 28),
    (.virt 18978, .wire 51 29),
    (.wire 51 27, .wire 51 30),
    (.wire 52 23, .virt 18979),
    (.wire 51 31, .virt 18979),
    (.virt 18978, .wire 51 32),
    (.virt 18978, .wire 51 33),
    (.virt 19185, .wire 51 34),
    (.virt 19185, .wire 52 28),
    (.wire 47 51, .wire 52 29),
    (.virt 19185, .wire 52 30),
    (.wire 47 51, .wire 52 32),
    (.virt 19186, .wire 52 33),
    (.wire 47 51, .wire 52 34),
    (.wire 52 35, .wire 51 36),
    (.virt 18978, .wire 51 37),
    (.wire 51 35, .wire 51 38),
    (.wire 52 31, .virt 18979),
    (.wire 51 39, .virt 18979),
    (.virt 19179, .wire 52 36),
    (.virt 19181, .wire 52 37)
  ]

/-- `privateBatchWrapper2.copies`, items `2432..2464`. -/
def privateBatchWrapper2.copies76 : List (Target × Target) := [
    (.virt 19179, .wire 52 38),
    (.virt 19183, .wire 52 40),
    (.virt 19185, .wire 52 41),
    (.virt 19183, .wire 52 42),
    (.wire 52 39, .wire 52 44),
    (.wire 52 43, .wire 52 45),
    (.wire 52 39, .wire 52 46),
    (.wire 52 47, .wire 51 40),
    (.wire 12 27, .wire 51 41),
    (.virt 18979, .wire 51 42),
    (.wire 36 27, .wire 36 28),
    (.virt 18978, .wire 36 29),
    (.wire 51 43, .wire 36 30),
    (.virt 18978, .wire 51 44),
    (.virt 18978, .wire 51 45),
    (.virt 19187, .wire 51 46),
    (.wire 12 35, .wire 51 48),
    (.virt 18978, .wire 51 49),
    (.wire 12 35, .wire 51 50),
    (.virt 19187, .wire 52 48),
    (.wire 51 51, .wire 52 49),
    (.virt 19187, .wire 52 50),
    (.wire 51 51, .wire 52 52),
    (.virt 19188, .wire 52 53),
    (.wire 51 51, .wire 52 54),
    (.wire 52 55, .wire 51 52),
    (.virt 18978, .wire 51 53),
    (.wire 51 47, .wire 51 54),
    (.wire 52 51, .virt 18979),
    (.wire 51 55, .virt 18979),
    (.virt 18978, .wire 51 56),
    (.virt 18978, .wire 51 57)
  ]

/-- `privateBatchWrapper2.copies`, items `2464..2496`. -/
def privateBatchWrapper2.copies77 : List (Target × Target) := [
    (.virt 19189, .wire 51 58),
    (.wire 12 43, .wire 53 0),
    (.virt 18978, .wire 53 1),
    (.wire 12 43, .wire 53 2),
    (.virt 19189, .wire 52 56),
    (.wire 53 3, .wire 52 57),
    (.virt 19189, .wire 52 58),
    (.wire 53 3, .wire 54 0),
    (.virt 19190, .wire 54 1),
    (.wire 53 3, .wire 54 2),
    (.wire 54 3, .wire 53 4),
    (.virt 18978, .wire 53 5),
    (.wire 51 59, .wire 53 6),
    (.wire 52 59, .virt 18979),
    (.wire 53 7, .virt 18979),
    (.virt 18978, .wire 53 8),
    (.virt 18978, .wire 53 9),
    (.virt 19191, .wire 53 10),
    (.wire 12 51, .wire 53 12),
    (.virt 18978, .wire 53 13),
    (.wire 12 51, .wire 53 14),
    (.virt 19191, .wire 54 4),
    (.wire 53 15, .wire 54 5),
    (.virt 19191, .wire 54 6),
    (.wire 53 15, .wire 54 8),
    (.virt 19192, .wire 54 9),
    (.wire 53 15, .wire 54 10),
    (.wire 54 11, .wire 53 16),
    (.virt 18978, .wire 53 17),
    (.wire 53 11, .wire 53 18),
    (.wire 54 7, .virt 18979),
    (.wire 53 19, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `2496..2528`. -/
def privateBatchWrapper2.copies78 : List (Target × Target) := [
    (.virt 18978, .wire 53 20),
    (.virt 18978, .wire 53 21),
    (.virt 19193, .wire 53 22),
    (.wire 12 59, .wire 53 24),
    (.virt 18978, .wire 53 25),
    (.wire 12 59, .wire 53 26),
    (.virt 19193, .wire 54 12),
    (.wire 53 27, .wire 54 13),
    (.virt 19193, .wire 54 14),
    (.wire 53 27, .wire 54 16),
    (.virt 19194, .wire 54 17),
    (.wire 53 27, .wire 54 18),
    (.wire 54 19, .wire 53 28),
    (.virt 18978, .wire 53 29),
    (.wire 53 23, .wire 53 30),
    (.wire 54 15, .virt 18979),
    (.wire 53 31, .virt 18979),
    (.virt 19187, .wire 54 20),
    (.virt 19189, .wire 54 21),
    (.virt 19187, .wire 54 22),
    (.virt 19191, .wire 54 24),
    (.virt 19193, .wire 54 25),
    (.virt 19191, .wire 54 26),
    (.wire 54 23, .wire 54 28),
    (.wire 54 27, .wire 54 29),
    (.wire 54 23, .wire 54 30),
    (.wire 54 31, .wire 53 32),
    (.wire 13 7, .wire 53 33),
    (.virt 18979, .wire 53 34),
    (.wire 36 31, .wire 36 32),
    (.virt 18978, .wire 36 33),
    (.wire 53 35, .wire 36 34)
  ]

/-- `privateBatchWrapper2.copies`, items `2528..2560`. -/
def privateBatchWrapper2.copies79 : List (Target × Target) := [
    (.wire 36 23, .wire 53 36),
    (.wire 36 35, .wire 53 37),
    (.wire 36 35, .wire 53 38),
    (.wire 36 23, .wire 53 40),
    (.virt 18979, .wire 53 41),
    (.wire 53 39, .wire 53 42),
    (.wire 36 23, .wire 53 44),
    (.wire 12 35, .wire 53 45),
    (.wire 12 35, .wire 53 46),
    (.wire 36 23, .wire 53 48),
    (.virt 18979, .wire 53 49),
    (.wire 53 47, .wire 53 50),
    (.wire 36 23, .wire 53 52),
    (.wire 12 43, .wire 53 53),
    (.wire 12 43, .wire 53 54),
    (.wire 36 23, .wire 53 56),
    (.virt 18979, .wire 53 57),
    (.wire 53 55, .wire 53 58),
    (.wire 36 23, .wire 55 0),
    (.wire 12 51, .wire 55 1),
    (.wire 12 51, .wire 55 2),
    (.wire 36 23, .wire 55 4),
    (.virt 18979, .wire 55 5),
    (.wire 55 3, .wire 55 6),
    (.wire 36 23, .wire 55 8),
    (.wire 12 59, .wire 55 9),
    (.wire 12 59, .wire 55 10),
    (.wire 36 23, .wire 55 12),
    (.virt 18979, .wire 55 13),
    (.wire 55 11, .wire 55 14),
    (.wire 56 33, .virt 18979),
    (.wire 56 34, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `2560..2592`. -/
def privateBatchWrapper2.copies80 : List (Target × Target) := [
    (.wire 56 35, .virt 18979),
    (.wire 56 36, .virt 18979),
    (.wire 56 37, .virt 18979),
    (.wire 56 38, .virt 18979),
    (.wire 56 39, .virt 18979),
    (.wire 56 40, .virt 18979),
    (.wire 56 41, .virt 18979),
    (.wire 56 42, .virt 18979),
    (.wire 56 43, .virt 18979),
    (.wire 56 44, .virt 18979),
    (.wire 56 45, .virt 18979),
    (.wire 56 46, .virt 18979),
    (.wire 56 47, .virt 18979),
    (.wire 56 48, .virt 18979),
    (.wire 56 49, .virt 18979),
    (.wire 56 50, .virt 18979),
    (.wire 56 51, .virt 18979),
    (.wire 56 52, .virt 18979),
    (.wire 56 53, .virt 18979),
    (.wire 56 54, .virt 18979),
    (.wire 56 55, .virt 18979),
    (.wire 56 56, .virt 18979),
    (.wire 56 57, .virt 18979),
    (.wire 56 58, .virt 18979),
    (.wire 56 59, .virt 18979),
    (.wire 56 0, .wire 53 43),
    (.wire 3 7, .wire 54 32),
    (.wire 3 35, .wire 54 33),
    (.wire 3 7, .wire 54 34),
    (.virt 18978, .wire 55 16),
    (.virt 18978, .wire 55 17),
    (.virt 19195, .wire 55 18)
  ]

/-- `privateBatchWrapper2.copies`, items `2592..2624`. -/
def privateBatchWrapper2.copies81 : List (Target × Target) := [
    (.virt 9467, .wire 55 20),
    (.virt 18978, .wire 55 21),
    (.virt 18952, .wire 55 22),
    (.virt 19195, .wire 54 36),
    (.wire 55 23, .wire 54 37),
    (.virt 19195, .wire 54 38),
    (.wire 55 23, .wire 54 40),
    (.virt 19196, .wire 54 41),
    (.wire 55 23, .wire 54 42),
    (.wire 54 43, .wire 55 24),
    (.virt 18978, .wire 55 25),
    (.wire 55 19, .wire 55 26),
    (.wire 54 39, .virt 18979),
    (.wire 55 27, .virt 18979),
    (.virt 18978, .wire 55 28),
    (.virt 18978, .wire 55 29),
    (.virt 19197, .wire 55 30),
    (.virt 9468, .wire 55 32),
    (.virt 18978, .wire 55 33),
    (.virt 18953, .wire 55 34),
    (.virt 19197, .wire 54 44),
    (.wire 55 35, .wire 54 45),
    (.virt 19197, .wire 54 46),
    (.wire 55 35, .wire 54 48),
    (.virt 19198, .wire 54 49),
    (.wire 55 35, .wire 54 50),
    (.wire 54 51, .wire 55 36),
    (.virt 18978, .wire 55 37),
    (.wire 55 31, .wire 55 38),
    (.wire 54 47, .virt 18979),
    (.wire 55 39, .virt 18979),
    (.virt 18978, .wire 55 40)
  ]

/-- `privateBatchWrapper2.copies`, items `2624..2656`. -/
def privateBatchWrapper2.copies82 : List (Target × Target) := [
    (.virt 18978, .wire 55 41),
    (.virt 19199, .wire 55 42),
    (.virt 9469, .wire 55 44),
    (.virt 18978, .wire 55 45),
    (.virt 18954, .wire 55 46),
    (.virt 19199, .wire 54 52),
    (.wire 55 47, .wire 54 53),
    (.virt 19199, .wire 54 54),
    (.wire 55 47, .wire 54 56),
    (.virt 19200, .wire 54 57),
    (.wire 55 47, .wire 54 58),
    (.wire 54 59, .wire 55 48),
    (.virt 18978, .wire 55 49),
    (.wire 55 43, .wire 55 50),
    (.wire 54 55, .virt 18979),
    (.wire 55 51, .virt 18979),
    (.virt 18978, .wire 55 52),
    (.virt 18978, .wire 55 53),
    (.virt 19201, .wire 55 54),
    (.virt 9470, .wire 55 56),
    (.virt 18978, .wire 55 57),
    (.virt 18955, .wire 55 58),
    (.virt 19201, .wire 57 0),
    (.wire 55 59, .wire 57 1),
    (.virt 19201, .wire 57 2),
    (.wire 55 59, .wire 57 4),
    (.virt 19202, .wire 57 5),
    (.wire 55 59, .wire 57 6),
    (.wire 57 7, .wire 58 0),
    (.virt 18978, .wire 58 1),
    (.wire 55 55, .wire 58 2),
    (.wire 57 3, .virt 18979)
  ]

/-- `privateBatchWrapper2.copies`, items `2656..2688`. -/
def privateBatchWrapper2.copies83 : List (Target × Target) := [
    (.wire 58 3, .virt 18979),
    (.virt 19195, .wire 57 8),
    (.virt 19197, .wire 57 9),
    (.virt 19195, .wire 57 10),
    (.virt 19199, .wire 57 12),
    (.virt 19201, .wire 57 13),
    (.virt 19199, .wire 57 14),
    (.wire 57 11, .wire 57 16),
    (.wire 57 15, .wire 57 17),
    (.wire 57 11, .wire 57 18),
    (.wire 54 35, .wire 57 20),
    (.wire 57 19, .wire 57 21),
    (.wire 54 35, .wire 57 22),
    (.wire 57 23, .virt 18979),
    (.virt 18970, .wire 59 0),
    (.virt 18971, .wire 59 1),
    (.virt 18972, .wire 59 2),
    (.virt 18973, .wire 59 3),
    (.virt 18978, .wire 59 4),
    (.virt 18979, .wire 59 5),
    (.virt 18979, .wire 59 6),
    (.virt 18979, .wire 59 7),
    (.virt 18979, .wire 59 8),
    (.virt 18979, .wire 59 9),
    (.virt 18979, .wire 59 10),
    (.virt 18979, .wire 59 11),
    (.wire 59 12, .wire 60 0),
    (.wire 59 13, .wire 60 1),
    (.wire 59 14, .wire 60 2),
    (.wire 59 15, .wire 60 3),
    (.virt 18978, .wire 60 4),
    (.virt 18979, .wire 60 5)
  ]

/-- `privateBatchWrapper2.copies`, items `2688..2720`. -/
def privateBatchWrapper2.copies84 : List (Target × Target) := [
    (.virt 18979, .wire 60 6),
    (.virt 18979, .wire 60 7),
    (.virt 18979, .wire 60 8),
    (.virt 18979, .wire 60 9),
    (.virt 18979, .wire 60 10),
    (.virt 18979, .wire 60 11),
    (.wire 1 43, .wire 58 4),
    (.virt 9467, .wire 58 5),
    (.virt 9467, .wire 58 6),
    (.wire 1 43, .wire 58 8),
    (.wire 60 12, .wire 58 9),
    (.wire 58 7, .wire 58 10),
    (.wire 1 43, .wire 58 12),
    (.virt 9468, .wire 58 13),
    (.virt 9468, .wire 58 14),
    (.wire 1 43, .wire 58 16),
    (.wire 60 13, .wire 58 17),
    (.wire 58 15, .wire 58 18),
    (.wire 1 43, .wire 58 20),
    (.virt 9469, .wire 58 21),
    (.virt 9469, .wire 58 22),
    (.wire 1 43, .wire 58 24),
    (.wire 60 14, .wire 58 25),
    (.wire 58 23, .wire 58 26),
    (.wire 1 43, .wire 58 28),
    (.virt 9470, .wire 58 29),
    (.virt 9470, .wire 58 30),
    (.wire 1 43, .wire 58 32),
    (.wire 60 15, .wire 58 33),
    (.wire 58 31, .wire 58 34),
    (.virt 18974, .wire 61 0),
    (.virt 18975, .wire 61 1)
  ]

/-- `privateBatchWrapper2.copies`, items `2720..2752`. -/
def privateBatchWrapper2.copies85 : List (Target × Target) := [
    (.virt 18976, .wire 61 2),
    (.virt 18977, .wire 61 3),
    (.virt 18978, .wire 61 4),
    (.virt 18979, .wire 61 5),
    (.virt 18979, .wire 61 6),
    (.virt 18979, .wire 61 7),
    (.virt 18979, .wire 61 8),
    (.virt 18979, .wire 61 9),
    (.virt 18979, .wire 61 10),
    (.virt 18979, .wire 61 11),
    (.wire 61 12, .wire 62 0),
    (.wire 61 13, .wire 62 1),
    (.wire 61 14, .wire 62 2),
    (.wire 61 15, .wire 62 3),
    (.virt 18978, .wire 62 4),
    (.virt 18979, .wire 62 5),
    (.virt 18979, .wire 62 6),
    (.virt 18979, .wire 62 7),
    (.virt 18979, .wire 62 8),
    (.virt 18979, .wire 62 9),
    (.virt 18979, .wire 62 10),
    (.virt 18979, .wire 62 11),
    (.wire 2 27, .wire 58 36),
    (.virt 18952, .wire 58 37),
    (.virt 18952, .wire 58 38),
    (.wire 2 27, .wire 58 40),
    (.wire 62 12, .wire 58 41),
    (.wire 58 39, .wire 58 42),
    (.wire 2 27, .wire 58 44),
    (.virt 18953, .wire 58 45),
    (.virt 18953, .wire 58 46),
    (.wire 2 27, .wire 58 48)
  ]

/-- `privateBatchWrapper2.copies`, items `2752..2784`. -/
def privateBatchWrapper2.copies86 : List (Target × Target) := [
    (.wire 62 13, .wire 58 49),
    (.wire 58 47, .wire 58 50),
    (.wire 2 27, .wire 58 52),
    (.virt 18954, .wire 58 53),
    (.virt 18954, .wire 58 54),
    (.wire 2 27, .wire 58 56),
    (.wire 62 14, .wire 58 57),
    (.wire 58 55, .wire 58 58),
    (.wire 2 27, .wire 63 0),
    (.virt 18955, .wire 63 1),
    (.virt 18955, .wire 63 2),
    (.wire 2 27, .wire 63 4),
    (.wire 62 15, .wire 63 5),
    (.wire 63 3, .wire 63 6),
    (.virt 19203, .wire 63 8),
    (.virt 19203, .wire 63 9),
    (.virt 19203, .wire 63 10),
    (.wire 63 11, .virt 18979),
    (.virt 19203, .wire 63 12),
    (.wire 58 11, .wire 63 13),
    (.wire 58 11, .wire 63 14),
    (.virt 19203, .wire 63 16),
    (.wire 58 43, .wire 63 17),
    (.wire 63 15, .wire 63 18),
    (.virt 19203, .wire 63 20),
    (.wire 58 43, .wire 63 21),
    (.wire 58 43, .wire 63 22),
    (.virt 19203, .wire 63 24),
    (.wire 58 11, .wire 63 25),
    (.wire 63 23, .wire 63 26),
    (.virt 19203, .wire 63 28),
    (.wire 58 19, .wire 63 29)
  ]

/-- `privateBatchWrapper2.copies`, items `2784..2816`. -/
def privateBatchWrapper2.copies87 : List (Target × Target) := [
    (.wire 58 19, .wire 63 30),
    (.virt 19203, .wire 63 32),
    (.wire 58 51, .wire 63 33),
    (.wire 63 31, .wire 63 34),
    (.virt 19203, .wire 63 36),
    (.wire 58 51, .wire 63 37),
    (.wire 58 51, .wire 63 38),
    (.virt 19203, .wire 63 40),
    (.wire 58 19, .wire 63 41),
    (.wire 63 39, .wire 63 42),
    (.virt 19203, .wire 63 44),
    (.wire 58 27, .wire 63 45),
    (.wire 58 27, .wire 63 46),
    (.virt 19203, .wire 63 48),
    (.wire 58 59, .wire 63 49),
    (.wire 63 47, .wire 63 50),
    (.virt 19203, .wire 63 52),
    (.wire 58 59, .wire 63 53),
    (.wire 58 59, .wire 63 54),
    (.virt 19203, .wire 63 56),
    (.wire 58 27, .wire 63 57),
    (.wire 63 55, .wire 63 58),
    (.virt 19203, .wire 64 0),
    (.wire 58 35, .wire 64 1),
    (.wire 58 35, .wire 64 2),
    (.virt 19203, .wire 64 4),
    (.wire 63 7, .wire 64 5),
    (.wire 64 3, .wire 64 6),
    (.virt 19203, .wire 64 8),
    (.wire 63 7, .wire 64 9),
    (.wire 63 7, .wire 64 10),
    (.virt 19203, .wire 64 12)
  ]

/-- `privateBatchWrapper2.copies`, items `2816..2818`. -/
def privateBatchWrapper2.copies88 : List (Target × Target) := [
    (.wire 58 35, .wire 64 13),
    (.wire 64 11, .wire 64 14)
  ]

/-- The private-batch aggregation wrapper at `n = 2` without the leaf verifiers (`wormhole/aggregator/src/private_batch/circuit/circuit_logic.rs`): leaf public inputs `leaf_pis_0..1`, dummy-nullifier preimages `dummy_pre_image_0..1`, the permutation switches `switches`, and the aggregated public inputs. -/
def privateBatchWrapper2 (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 2
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 3
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 4
    ⟨.arithmetic 15, [(-1), 1]⟩,  -- row 5
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 6
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 7
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 8
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 9
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 10
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 11
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 12
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 13
    ⟨.baseSum2 59, []⟩,  -- row 14
    ⟨.baseSum2 59, []⟩,  -- row 15
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 16
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 17
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 18
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 19
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 20
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 21
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 22
    ⟨.baseSum2 59, []⟩,  -- row 23
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 24
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 25
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 26
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 27
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 28
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 29
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 30
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 31
    ⟨.baseSum2 59, []⟩,  -- row 32
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 33
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 34
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 35
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 36
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 37
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 38
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 39
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 40
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 41
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 42
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 43
    ⟨.baseSum2 59, []⟩,  -- row 44
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 45
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 46
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 47
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 48
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 49
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 50
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 51
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 52
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 53
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 54
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 55
    ⟨.baseSum2 59, []⟩,  -- row 56
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 57
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 58
    ⟨.poseidon2, []⟩,  -- row 59
    ⟨.poseidon2, []⟩,  -- row 60
    ⟨.poseidon2, []⟩,  -- row 61
    ⟨.poseidon2, []⟩,  -- row 62
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 63
    ⟨.arithmetic 15, [1, (-1)]⟩  -- row 64
  ]
  copies := ((((((privateBatchWrapper2.copies0 ++ privateBatchWrapper2.copies1) ++ (privateBatchWrapper2.copies2 ++ (privateBatchWrapper2.copies3 ++ privateBatchWrapper2.copies4))) ++ ((privateBatchWrapper2.copies5 ++ (privateBatchWrapper2.copies6 ++ privateBatchWrapper2.copies7)) ++ (privateBatchWrapper2.copies8 ++ (privateBatchWrapper2.copies9 ++ privateBatchWrapper2.copies10)))) ++ (((privateBatchWrapper2.copies11 ++ privateBatchWrapper2.copies12) ++ (privateBatchWrapper2.copies13 ++ (privateBatchWrapper2.copies14 ++ privateBatchWrapper2.copies15))) ++ ((privateBatchWrapper2.copies16 ++ (privateBatchWrapper2.copies17 ++ privateBatchWrapper2.copies18)) ++ (privateBatchWrapper2.copies19 ++ (privateBatchWrapper2.copies20 ++ privateBatchWrapper2.copies21))))) ++ ((((privateBatchWrapper2.copies22 ++ privateBatchWrapper2.copies23) ++ (privateBatchWrapper2.copies24 ++ (privateBatchWrapper2.copies25 ++ privateBatchWrapper2.copies26))) ++ ((privateBatchWrapper2.copies27 ++ (privateBatchWrapper2.copies28 ++ privateBatchWrapper2.copies29)) ++ (privateBatchWrapper2.copies30 ++ (privateBatchWrapper2.copies31 ++ privateBatchWrapper2.copies32)))) ++ (((privateBatchWrapper2.copies33 ++ privateBatchWrapper2.copies34) ++ (privateBatchWrapper2.copies35 ++ (privateBatchWrapper2.copies36 ++ privateBatchWrapper2.copies37))) ++ ((privateBatchWrapper2.copies38 ++ (privateBatchWrapper2.copies39 ++ privateBatchWrapper2.copies40)) ++ (privateBatchWrapper2.copies41 ++ (privateBatchWrapper2.copies42 ++ privateBatchWrapper2.copies43)))))) ++ (((((privateBatchWrapper2.copies44 ++ privateBatchWrapper2.copies45) ++ (privateBatchWrapper2.copies46 ++ (privateBatchWrapper2.copies47 ++ privateBatchWrapper2.copies48))) ++ ((privateBatchWrapper2.copies49 ++ (privateBatchWrapper2.copies50 ++ privateBatchWrapper2.copies51)) ++ (privateBatchWrapper2.copies52 ++ (privateBatchWrapper2.copies53 ++ privateBatchWrapper2.copies54)))) ++ (((privateBatchWrapper2.copies55 ++ privateBatchWrapper2.copies56) ++ (privateBatchWrapper2.copies57 ++ (privateBatchWrapper2.copies58 ++ privateBatchWrapper2.copies59))) ++ ((privateBatchWrapper2.copies60 ++ (privateBatchWrapper2.copies61 ++ privateBatchWrapper2.copies62)) ++ (privateBatchWrapper2.copies63 ++ (privateBatchWrapper2.copies64 ++ privateBatchWrapper2.copies65))))) ++ ((((privateBatchWrapper2.copies66 ++ privateBatchWrapper2.copies67) ++ (privateBatchWrapper2.copies68 ++ (privateBatchWrapper2.copies69 ++ privateBatchWrapper2.copies70))) ++ ((privateBatchWrapper2.copies71 ++ (privateBatchWrapper2.copies72 ++ privateBatchWrapper2.copies73)) ++ (privateBatchWrapper2.copies74 ++ (privateBatchWrapper2.copies75 ++ privateBatchWrapper2.copies76)))) ++ (((privateBatchWrapper2.copies77 ++ (privateBatchWrapper2.copies78 ++ privateBatchWrapper2.copies79)) ++ (privateBatchWrapper2.copies80 ++ (privateBatchWrapper2.copies81 ++ privateBatchWrapper2.copies82))) ++ ((privateBatchWrapper2.copies83 ++ (privateBatchWrapper2.copies84 ++ privateBatchWrapper2.copies85)) ++ (privateBatchWrapper2.copies86 ++ (privateBatchWrapper2.copies87 ++ privateBatchWrapper2.copies88)))))))
  constants := [
    (.virt 18978, 1),
    (.virt 18979, 0),
    (.virt 18980, 4),
    (.virt 19017, 10000),
    (.virt 19018, 576460752303423488 /- canonical u64; faithful only at p = goldilocks -/)
  ]
  publicInputs := [
    .virt 18980,
    .virt 9463,
    .wire 4 27,
    .wire 3 47,
    .wire 3 55,
    .wire 4 3,
    .wire 4 11,
    .wire 4 19,
    .wire 22 7,
    .wire 22 15,
    .wire 22 23,
    .wire 22 31,
    .wire 22 39,
    .wire 30 47,
    .wire 30 55,
    .wire 31 3,
    .wire 31 11,
    .wire 31 19,
    .wire 41 59,
    .wire 43 7,
    .wire 43 15,
    .wire 43 23,
    .wire 43 31,
    .wire 53 43,
    .wire 53 51,
    .wire 53 59,
    .wire 55 7,
    .wire 55 15,
    .wire 63 19,
    .wire 63 35,
    .wire 63 51,
    .wire 64 7,
    .wire 63 27,
    .wire 63 43,
    .wire 63 59,
    .wire 64 15,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979,
    .virt 18979
  ]

/-- Named targets `leaf_pis_0`. -/
def privateBatchWrapper2.leaf_pis_0 : Fin 22 → Target :=
  ![.virt 9463, .virt 9464, .virt 9465, .virt 9466, .virt 9467, .virt 9468, .virt 9469, .virt 9470, .virt 9471, .virt 9472, .virt 9473, .virt 9474, .virt 9475, .virt 9476, .virt 9477, .virt 9478, .virt 9479, .virt 9480, .virt 9481, .virt 9482, .virt 9483, .virt 9484]

/-- Named targets `leaf_pis_1`. -/
def privateBatchWrapper2.leaf_pis_1 : Fin 22 → Target :=
  ![.virt 18948, .virt 18949, .virt 18950, .virt 18951, .virt 18952, .virt 18953, .virt 18954, .virt 18955, .virt 18956, .virt 18957, .virt 18958, .virt 18959, .virt 18960, .virt 18961, .virt 18962, .virt 18963, .virt 18964, .virt 18965, .virt 18966, .virt 18967, .virt 18968, .virt 18969]

/-- Named targets `dummy_pre_image_0`. -/
def privateBatchWrapper2.dummy_pre_image_0 : Fin 4 → Target :=
  ![.virt 18970, .virt 18971, .virt 18972, .virt 18973]

/-- Named targets `dummy_pre_image_1`. -/
def privateBatchWrapper2.dummy_pre_image_1 : Fin 4 → Target :=
  ![.virt 18974, .virt 18975, .virt 18976, .virt 18977]

/-- Named target `switches`. -/
def privateBatchWrapper2.switches : Target := .virt 19203

theorem privateBatchWrapper2_copies0 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 0 0) ∧ a (.virt 18978) = a (.wire 0 1) ∧ a (.virt 18981) = a (.wire 0 2) ∧ a (.virt 18981) = a (.wire 1 0) ∧ a (.virt 9479) = a (.wire 1 1) ∧ a (.virt 18981) = a (.wire 1 2) ∧ a (.virt 9479) = a (.wire 1 4) ∧ a (.virt 18982) = a (.wire 1 5) ∧ a (.virt 9479) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 4) ∧ a (.virt 18978) = a (.wire 0 5) ∧ a (.wire 0 3) = a (.wire 0 6) ∧ a (.wire 1 3) = a (.virt 18979) ∧ a (.wire 0 7) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 0 8) ∧ a (.virt 18978) = a (.wire 0 9) ∧ a (.virt 18983) = a (.wire 0 10) ∧ a (.virt 18983) = a (.wire 1 8) ∧ a (.virt 9480) = a (.wire 1 9) ∧ a (.virt 18983) = a (.wire 1 10) ∧ a (.virt 9480) = a (.wire 1 12) ∧ a (.virt 18984) = a (.wire 1 13) ∧ a (.virt 9480) = a (.wire 1 14) ∧ a (.wire 1 15) = a (.wire 0 12) ∧ a (.virt 18978) = a (.wire 0 13) ∧ a (.wire 0 11) = a (.wire 0 14) ∧ a (.wire 1 11) = a (.virt 18979) ∧ a (.wire 0 15) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 0 16) ∧ a (.virt 18978) = a (.wire 0 17) ∧ a (.virt 18985) = a (.wire 0 18) ∧ a (.virt 18985) = a (.wire 1 16) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies1 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 9481) = a (.wire 1 17) ∧ a (.virt 18985) = a (.wire 1 18) ∧ a (.virt 9481) = a (.wire 1 20) ∧ a (.virt 18986) = a (.wire 1 21) ∧ a (.virt 9481) = a (.wire 1 22) ∧ a (.wire 1 23) = a (.wire 0 20) ∧ a (.virt 18978) = a (.wire 0 21) ∧ a (.wire 0 19) = a (.wire 0 22) ∧ a (.wire 1 19) = a (.virt 18979) ∧ a (.wire 0 23) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 0 24) ∧ a (.virt 18978) = a (.wire 0 25) ∧ a (.virt 18987) = a (.wire 0 26) ∧ a (.virt 18987) = a (.wire 1 24) ∧ a (.virt 9482) = a (.wire 1 25) ∧ a (.virt 18987) = a (.wire 1 26) ∧ a (.virt 9482) = a (.wire 1 28) ∧ a (.virt 18988) = a (.wire 1 29) ∧ a (.virt 9482) = a (.wire 1 30) ∧ a (.wire 1 31) = a (.wire 0 28) ∧ a (.virt 18978) = a (.wire 0 29) ∧ a (.wire 0 27) = a (.wire 0 30) ∧ a (.wire 1 27) = a (.virt 18979) ∧ a (.wire 0 31) = a (.virt 18979) ∧ a (.virt 18981) = a (.wire 1 32) ∧ a (.virt 18983) = a (.wire 1 33) ∧ a (.virt 18981) = a (.wire 1 34) ∧ a (.virt 18985) = a (.wire 1 36) ∧ a (.virt 18987) = a (.wire 1 37) ∧ a (.virt 18985) = a (.wire 1 38) ∧ a (.wire 1 35) = a (.wire 1 40) ∧ a (.wire 1 39) = a (.wire 1 41) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies2 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 1 35) = a (.wire 1 42) ∧ a (.virt 18978) = a (.wire 0 32) ∧ a (.virt 18978) = a (.wire 0 33) ∧ a (.virt 18989) = a (.wire 0 34) ∧ a (.virt 18989) = a (.wire 1 44) ∧ a (.virt 18964) = a (.wire 1 45) ∧ a (.virt 18989) = a (.wire 1 46) ∧ a (.virt 18964) = a (.wire 1 48) ∧ a (.virt 18990) = a (.wire 1 49) ∧ a (.virt 18964) = a (.wire 1 50) ∧ a (.wire 1 51) = a (.wire 0 36) ∧ a (.virt 18978) = a (.wire 0 37) ∧ a (.wire 0 35) = a (.wire 0 38) ∧ a (.wire 1 47) = a (.virt 18979) ∧ a (.wire 0 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 0 40) ∧ a (.virt 18978) = a (.wire 0 41) ∧ a (.virt 18991) = a (.wire 0 42) ∧ a (.virt 18991) = a (.wire 1 52) ∧ a (.virt 18965) = a (.wire 1 53) ∧ a (.virt 18991) = a (.wire 1 54) ∧ a (.virt 18965) = a (.wire 1 56) ∧ a (.virt 18992) = a (.wire 1 57) ∧ a (.virt 18965) = a (.wire 1 58) ∧ a (.wire 1 59) = a (.wire 0 44) ∧ a (.virt 18978) = a (.wire 0 45) ∧ a (.wire 0 43) = a (.wire 0 46) ∧ a (.wire 1 55) = a (.virt 18979) ∧ a (.wire 0 47) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 0 48) ∧ a (.virt 18978) = a (.wire 0 49) ∧ a (.virt 18993) = a (.wire 0 50) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies3 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18993) = a (.wire 2 0) ∧ a (.virt 18966) = a (.wire 2 1) ∧ a (.virt 18993) = a (.wire 2 2) ∧ a (.virt 18966) = a (.wire 2 4) ∧ a (.virt 18994) = a (.wire 2 5) ∧ a (.virt 18966) = a (.wire 2 6) ∧ a (.wire 2 7) = a (.wire 0 52) ∧ a (.virt 18978) = a (.wire 0 53) ∧ a (.wire 0 51) = a (.wire 0 54) ∧ a (.wire 2 3) = a (.virt 18979) ∧ a (.wire 0 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 0 56) ∧ a (.virt 18978) = a (.wire 0 57) ∧ a (.virt 18995) = a (.wire 0 58) ∧ a (.virt 18995) = a (.wire 2 8) ∧ a (.virt 18967) = a (.wire 2 9) ∧ a (.virt 18995) = a (.wire 2 10) ∧ a (.virt 18967) = a (.wire 2 12) ∧ a (.virt 18996) = a (.wire 2 13) ∧ a (.virt 18967) = a (.wire 2 14) ∧ a (.wire 2 15) = a (.wire 3 0) ∧ a (.virt 18978) = a (.wire 3 1) ∧ a (.wire 0 59) = a (.wire 3 2) ∧ a (.wire 2 11) = a (.virt 18979) ∧ a (.wire 3 3) = a (.virt 18979) ∧ a (.virt 18989) = a (.wire 2 16) ∧ a (.virt 18991) = a (.wire 2 17) ∧ a (.virt 18989) = a (.wire 2 18) ∧ a (.virt 18993) = a (.wire 2 20) ∧ a (.virt 18995) = a (.wire 2 21) ∧ a (.virt 18993) = a (.wire 2 22) ∧ a (.wire 2 19) = a (.wire 2 24) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies3, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies4 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 23) = a (.wire 2 25) ∧ a (.wire 2 19) = a (.wire 2 26) ∧ a (.virt 18978) = a (.wire 3 4) ∧ a (.virt 18978) = a (.wire 3 5) ∧ a (.wire 1 43) = a (.wire 3 6) ∧ a (.wire 3 7) = a (.wire 3 8) ∧ a (.virt 9479) = a (.wire 3 9) ∧ a (.virt 18979) = a (.wire 3 10) ∧ a (.wire 3 7) = a (.wire 3 12) ∧ a (.virt 9480) = a (.wire 3 13) ∧ a (.virt 18979) = a (.wire 3 14) ∧ a (.wire 3 7) = a (.wire 3 16) ∧ a (.virt 9481) = a (.wire 3 17) ∧ a (.virt 18979) = a (.wire 3 18) ∧ a (.wire 3 7) = a (.wire 3 20) ∧ a (.virt 9482) = a (.wire 3 21) ∧ a (.virt 18979) = a (.wire 3 22) ∧ a (.wire 3 7) = a (.wire 3 24) ∧ a (.virt 9483) = a (.wire 3 25) ∧ a (.virt 18979) = a (.wire 3 26) ∧ a (.wire 3 7) = a (.wire 3 28) ∧ a (.virt 9466) = a (.wire 3 29) ∧ a (.virt 18979) = a (.wire 3 30) ∧ a (.virt 18978) = a (.wire 3 32) ∧ a (.virt 18978) = a (.wire 3 33) ∧ a (.wire 2 27) = a (.wire 3 34) ∧ a (.virt 18978) = a (.wire 3 36) ∧ a (.virt 18978) = a (.wire 3 37) ∧ a (.wire 3 7) = a (.wire 3 38) ∧ a (.wire 3 35) = a (.wire 2 28) ∧ a (.wire 3 39) = a (.wire 2 29) ∧ a (.wire 3 35) = a (.wire 2 30) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies4, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies5 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 31) = a (.wire 3 40) ∧ a (.wire 3 11) = a (.wire 3 41) ∧ a (.wire 3 11) = a (.wire 3 42) ∧ a (.wire 2 31) = a (.wire 3 44) ∧ a (.virt 18964) = a (.wire 3 45) ∧ a (.wire 3 43) = a (.wire 3 46) ∧ a (.wire 2 31) = a (.wire 3 48) ∧ a (.wire 3 15) = a (.wire 3 49) ∧ a (.wire 3 15) = a (.wire 3 50) ∧ a (.wire 2 31) = a (.wire 3 52) ∧ a (.virt 18965) = a (.wire 3 53) ∧ a (.wire 3 51) = a (.wire 3 54) ∧ a (.wire 2 31) = a (.wire 3 56) ∧ a (.wire 3 19) = a (.wire 3 57) ∧ a (.wire 3 19) = a (.wire 3 58) ∧ a (.wire 2 31) = a (.wire 4 0) ∧ a (.virt 18966) = a (.wire 4 1) ∧ a (.wire 3 59) = a (.wire 4 2) ∧ a (.wire 2 31) = a (.wire 4 4) ∧ a (.wire 3 23) = a (.wire 4 5) ∧ a (.wire 3 23) = a (.wire 4 6) ∧ a (.wire 2 31) = a (.wire 4 8) ∧ a (.virt 18967) = a (.wire 4 9) ∧ a (.wire 4 7) = a (.wire 4 10) ∧ a (.wire 2 31) = a (.wire 4 12) ∧ a (.wire 3 27) = a (.wire 4 13) ∧ a (.wire 3 27) = a (.wire 4 14) ∧ a (.wire 2 31) = a (.wire 4 16) ∧ a (.virt 18968) = a (.wire 4 17) ∧ a (.wire 4 15) = a (.wire 4 18) ∧ a (.wire 2 31) = a (.wire 4 20) ∧ a (.wire 3 31) = a (.wire 4 21) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies5, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies6 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = a (.wire 4 22) ∧ a (.wire 2 31) = a (.wire 4 24) ∧ a (.virt 18951) = a (.wire 4 25) ∧ a (.wire 4 23) = a (.wire 4 26) ∧ a (.wire 3 7) = a (.wire 5 0) ∧ a (.wire 3 35) = a (.wire 5 1) ∧ a (.wire 3 7) = a (.wire 5 2) ∧ a (.wire 5 3) = a (.wire 6 0) ∧ a (.virt 18978) = a (.wire 6 1) ∧ a (.wire 3 35) = a (.wire 6 2) ∧ a (.virt 18978) = a (.wire 4 28) ∧ a (.virt 18978) = a (.wire 4 29) ∧ a (.virt 18997) = a (.wire 4 30) ∧ a (.virt 9479) = a (.wire 4 32) ∧ a (.virt 18978) = a (.wire 4 33) ∧ a (.wire 3 47) = a (.wire 4 34) ∧ a (.virt 18997) = a (.wire 2 32) ∧ a (.wire 4 35) = a (.wire 2 33) ∧ a (.virt 18997) = a (.wire 2 34) ∧ a (.wire 4 35) = a (.wire 2 36) ∧ a (.virt 18998) = a (.wire 2 37) ∧ a (.wire 4 35) = a (.wire 2 38) ∧ a (.wire 2 39) = a (.wire 4 36) ∧ a (.virt 18978) = a (.wire 4 37) ∧ a (.wire 4 31) = a (.wire 4 38) ∧ a (.wire 2 35) = a (.virt 18979) ∧ a (.wire 4 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 4 40) ∧ a (.virt 18978) = a (.wire 4 41) ∧ a (.virt 18999) = a (.wire 4 42) ∧ a (.virt 9480) = a (.wire 4 44) ∧ a (.virt 18978) = a (.wire 4 45) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies6, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies7 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 55) = a (.wire 4 46) ∧ a (.virt 18999) = a (.wire 2 40) ∧ a (.wire 4 47) = a (.wire 2 41) ∧ a (.virt 18999) = a (.wire 2 42) ∧ a (.wire 4 47) = a (.wire 2 44) ∧ a (.virt 19000) = a (.wire 2 45) ∧ a (.wire 4 47) = a (.wire 2 46) ∧ a (.wire 2 47) = a (.wire 4 48) ∧ a (.virt 18978) = a (.wire 4 49) ∧ a (.wire 4 43) = a (.wire 4 50) ∧ a (.wire 2 43) = a (.virt 18979) ∧ a (.wire 4 51) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 4 52) ∧ a (.virt 18978) = a (.wire 4 53) ∧ a (.virt 19001) = a (.wire 4 54) ∧ a (.virt 9481) = a (.wire 4 56) ∧ a (.virt 18978) = a (.wire 4 57) ∧ a (.wire 4 3) = a (.wire 4 58) ∧ a (.virt 19001) = a (.wire 2 48) ∧ a (.wire 4 59) = a (.wire 2 49) ∧ a (.virt 19001) = a (.wire 2 50) ∧ a (.wire 4 59) = a (.wire 2 52) ∧ a (.virt 19002) = a (.wire 2 53) ∧ a (.wire 4 59) = a (.wire 2 54) ∧ a (.wire 2 55) = a (.wire 7 0) ∧ a (.virt 18978) = a (.wire 7 1) ∧ a (.wire 4 55) = a (.wire 7 2) ∧ a (.wire 2 51) = a (.virt 18979) ∧ a (.wire 7 3) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 7 4) ∧ a (.virt 18978) = a (.wire 7 5) ∧ a (.virt 19003) = a (.wire 7 6) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies7, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies8 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 9482) = a (.wire 7 8) ∧ a (.virt 18978) = a (.wire 7 9) ∧ a (.wire 4 11) = a (.wire 7 10) ∧ a (.virt 19003) = a (.wire 2 56) ∧ a (.wire 7 11) = a (.wire 2 57) ∧ a (.virt 19003) = a (.wire 2 58) ∧ a (.wire 7 11) = a (.wire 8 0) ∧ a (.virt 19004) = a (.wire 8 1) ∧ a (.wire 7 11) = a (.wire 8 2) ∧ a (.wire 8 3) = a (.wire 7 12) ∧ a (.virt 18978) = a (.wire 7 13) ∧ a (.wire 7 7) = a (.wire 7 14) ∧ a (.wire 2 59) = a (.virt 18979) ∧ a (.wire 7 15) = a (.virt 18979) ∧ a (.virt 18997) = a (.wire 8 4) ∧ a (.virt 18999) = a (.wire 8 5) ∧ a (.virt 18997) = a (.wire 8 6) ∧ a (.virt 19001) = a (.wire 8 8) ∧ a (.virt 19003) = a (.wire 8 9) ∧ a (.virt 19001) = a (.wire 8 10) ∧ a (.wire 8 7) = a (.wire 8 12) ∧ a (.wire 8 11) = a (.wire 8 13) ∧ a (.wire 8 7) = a (.wire 8 14) ∧ a (.wire 1 43) = a (.wire 5 4) ∧ a (.wire 8 15) = a (.wire 5 5) ∧ a (.wire 1 43) = a (.wire 5 6) ∧ a (.wire 5 7) = a (.wire 6 4) ∧ a (.virt 18978) = a (.wire 6 5) ∧ a (.wire 8 15) = a (.wire 6 6) ∧ a (.wire 6 7) = a (.virt 18978) ∧ True ∧ a (.virt 18978) = a (.wire 7 16) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies8, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies9 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 7 17) ∧ a (.virt 19005) = a (.wire 7 18) ∧ a (.virt 9466) = a (.wire 7 20) ∧ a (.virt 18978) = a (.wire 7 21) ∧ a (.wire 4 27) = a (.wire 7 22) ∧ a (.virt 19005) = a (.wire 8 16) ∧ a (.wire 7 23) = a (.wire 8 17) ∧ a (.virt 19005) = a (.wire 8 18) ∧ a (.wire 7 23) = a (.wire 8 20) ∧ a (.virt 19006) = a (.wire 8 21) ∧ a (.wire 7 23) = a (.wire 8 22) ∧ a (.wire 8 23) = a (.wire 7 24) ∧ a (.virt 18978) = a (.wire 7 25) ∧ a (.wire 7 19) = a (.wire 7 26) ∧ a (.wire 8 19) = a (.virt 18979) ∧ a (.wire 7 27) = a (.virt 18979) ∧ a (.wire 1 43) = a (.wire 5 8) ∧ a (.virt 19005) = a (.wire 5 9) ∧ a (.wire 1 43) = a (.wire 5 10) ∧ a (.wire 5 11) = a (.wire 6 8) ∧ a (.virt 18978) = a (.wire 6 9) ∧ a (.virt 19005) = a (.wire 6 10) ∧ a (.wire 6 11) = a (.virt 18978) ∧ a (.virt 18978) = a (.wire 7 28) ∧ a (.virt 18978) = a (.wire 7 29) ∧ a (.virt 19007) = a (.wire 7 30) ∧ a (.virt 18964) = a (.wire 7 32) ∧ a (.virt 18978) = a (.wire 7 33) ∧ a (.wire 3 47) = a (.wire 7 34) ∧ a (.virt 19007) = a (.wire 8 24) ∧ a (.wire 7 35) = a (.wire 8 25) ∧ a (.virt 19007) = a (.wire 8 26) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies9, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies10 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 35) = a (.wire 8 28) ∧ a (.virt 19008) = a (.wire 8 29) ∧ a (.wire 7 35) = a (.wire 8 30) ∧ a (.wire 8 31) = a (.wire 7 36) ∧ a (.virt 18978) = a (.wire 7 37) ∧ a (.wire 7 31) = a (.wire 7 38) ∧ a (.wire 8 27) = a (.virt 18979) ∧ a (.wire 7 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 7 40) ∧ a (.virt 18978) = a (.wire 7 41) ∧ a (.virt 19009) = a (.wire 7 42) ∧ a (.virt 18965) = a (.wire 7 44) ∧ a (.virt 18978) = a (.wire 7 45) ∧ a (.wire 3 55) = a (.wire 7 46) ∧ a (.virt 19009) = a (.wire 8 32) ∧ a (.wire 7 47) = a (.wire 8 33) ∧ a (.virt 19009) = a (.wire 8 34) ∧ a (.wire 7 47) = a (.wire 8 36) ∧ a (.virt 19010) = a (.wire 8 37) ∧ a (.wire 7 47) = a (.wire 8 38) ∧ a (.wire 8 39) = a (.wire 7 48) ∧ a (.virt 18978) = a (.wire 7 49) ∧ a (.wire 7 43) = a (.wire 7 50) ∧ a (.wire 8 35) = a (.virt 18979) ∧ a (.wire 7 51) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 7 52) ∧ a (.virt 18978) = a (.wire 7 53) ∧ a (.virt 19011) = a (.wire 7 54) ∧ a (.virt 18966) = a (.wire 7 56) ∧ a (.virt 18978) = a (.wire 7 57) ∧ a (.wire 4 3) = a (.wire 7 58) ∧ a (.virt 19011) = a (.wire 8 40) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies10, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies11 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 59) = a (.wire 8 41) ∧ a (.virt 19011) = a (.wire 8 42) ∧ a (.wire 7 59) = a (.wire 8 44) ∧ a (.virt 19012) = a (.wire 8 45) ∧ a (.wire 7 59) = a (.wire 8 46) ∧ a (.wire 8 47) = a (.wire 9 0) ∧ a (.virt 18978) = a (.wire 9 1) ∧ a (.wire 7 55) = a (.wire 9 2) ∧ a (.wire 8 43) = a (.virt 18979) ∧ a (.wire 9 3) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 9 4) ∧ a (.virt 18978) = a (.wire 9 5) ∧ a (.virt 19013) = a (.wire 9 6) ∧ a (.virt 18967) = a (.wire 9 8) ∧ a (.virt 18978) = a (.wire 9 9) ∧ a (.wire 4 11) = a (.wire 9 10) ∧ a (.virt 19013) = a (.wire 8 48) ∧ a (.wire 9 11) = a (.wire 8 49) ∧ a (.virt 19013) = a (.wire 8 50) ∧ a (.wire 9 11) = a (.wire 8 52) ∧ a (.virt 19014) = a (.wire 8 53) ∧ a (.wire 9 11) = a (.wire 8 54) ∧ a (.wire 8 55) = a (.wire 9 12) ∧ a (.virt 18978) = a (.wire 9 13) ∧ a (.wire 9 7) = a (.wire 9 14) ∧ a (.wire 8 51) = a (.virt 18979) ∧ a (.wire 9 15) = a (.virt 18979) ∧ a (.virt 19007) = a (.wire 8 56) ∧ a (.virt 19009) = a (.wire 8 57) ∧ a (.virt 19007) = a (.wire 8 58) ∧ a (.virt 19011) = a (.wire 10 0) ∧ a (.virt 19013) = a (.wire 10 1) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies11, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies12 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19011) = a (.wire 10 2) ∧ a (.wire 8 59) = a (.wire 10 4) ∧ a (.wire 10 3) = a (.wire 10 5) ∧ a (.wire 8 59) = a (.wire 10 6) ∧ a (.wire 2 27) = a (.wire 5 12) ∧ a (.wire 10 7) = a (.wire 5 13) ∧ a (.wire 2 27) = a (.wire 5 14) ∧ a (.wire 5 15) = a (.wire 6 12) ∧ a (.virt 18978) = a (.wire 6 13) ∧ a (.wire 10 7) = a (.wire 6 14) ∧ a (.wire 6 15) = a (.virt 18978) ∧ a (.virt 18948) = a (.virt 9463) ∧ a (.virt 18978) = a (.wire 9 16) ∧ a (.virt 18978) = a (.wire 9 17) ∧ a (.virt 19015) = a (.wire 9 18) ∧ a (.virt 18951) = a (.wire 9 20) ∧ a (.virt 18978) = a (.wire 9 21) ∧ a (.wire 4 27) = a (.wire 9 22) ∧ a (.virt 19015) = a (.wire 10 8) ∧ a (.wire 9 23) = a (.wire 10 9) ∧ a (.virt 19015) = a (.wire 10 10) ∧ a (.wire 9 23) = a (.wire 10 12) ∧ a (.virt 19016) = a (.wire 10 13) ∧ a (.wire 9 23) = a (.wire 10 14) ∧ a (.wire 10 15) = a (.wire 9 24) ∧ a (.virt 18978) = a (.wire 9 25) ∧ a (.wire 9 19) = a (.wire 9 26) ∧ a (.wire 10 11) = a (.virt 18979) ∧ a (.wire 9 27) = a (.virt 18979) ∧ a (.wire 2 27) = a (.wire 5 16) ∧ a (.virt 19015) = a (.wire 5 17) ∧ a (.wire 2 27) = a (.wire 5 18) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies12, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies13 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 5 19) = a (.wire 6 16) ∧ a (.virt 18978) = a (.wire 6 17) ∧ a (.virt 19015) = a (.wire 6 18) ∧ a (.wire 6 19) = a (.virt 18978) ∧ a (.wire 1 43) = a (.wire 9 28) ∧ a (.virt 9471) = a (.wire 9 29) ∧ a (.virt 9471) = a (.wire 9 30) ∧ a (.wire 1 43) = a (.wire 9 32) ∧ a (.virt 18979) = a (.wire 9 33) ∧ a (.wire 9 31) = a (.wire 9 34) ∧ a (.wire 1 43) = a (.wire 9 36) ∧ a (.virt 9472) = a (.wire 9 37) ∧ a (.virt 9472) = a (.wire 9 38) ∧ a (.wire 1 43) = a (.wire 9 40) ∧ a (.virt 18979) = a (.wire 9 41) ∧ a (.wire 9 39) = a (.wire 9 42) ∧ a (.wire 1 43) = a (.wire 9 44) ∧ a (.virt 9473) = a (.wire 9 45) ∧ a (.virt 9473) = a (.wire 9 46) ∧ a (.wire 1 43) = a (.wire 9 48) ∧ a (.virt 18979) = a (.wire 9 49) ∧ a (.wire 9 47) = a (.wire 9 50) ∧ a (.wire 1 43) = a (.wire 9 52) ∧ a (.virt 9474) = a (.wire 9 53) ∧ a (.virt 9474) = a (.wire 9 54) ∧ a (.wire 1 43) = a (.wire 9 56) ∧ a (.virt 18979) = a (.wire 9 57) ∧ a (.wire 9 55) = a (.wire 9 58) ∧ a (.wire 1 43) = a (.wire 11 0) ∧ a (.virt 9464) = a (.wire 11 1) ∧ a (.virt 9464) = a (.wire 11 2) ∧ a (.wire 1 43) = a (.wire 11 4) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies13, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies14 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18979) = a (.wire 11 5) ∧ a (.wire 11 3) = a (.wire 11 6) ∧ a (.wire 1 43) = a (.wire 11 8) ∧ a (.virt 9475) = a (.wire 11 9) ∧ a (.virt 9475) = a (.wire 11 10) ∧ a (.wire 1 43) = a (.wire 11 12) ∧ a (.virt 18979) = a (.wire 11 13) ∧ a (.wire 11 11) = a (.wire 11 14) ∧ a (.wire 1 43) = a (.wire 11 16) ∧ a (.virt 9476) = a (.wire 11 17) ∧ a (.virt 9476) = a (.wire 11 18) ∧ a (.wire 1 43) = a (.wire 11 20) ∧ a (.virt 18979) = a (.wire 11 21) ∧ a (.wire 11 19) = a (.wire 11 22) ∧ a (.wire 1 43) = a (.wire 11 24) ∧ a (.virt 9477) = a (.wire 11 25) ∧ a (.virt 9477) = a (.wire 11 26) ∧ a (.wire 1 43) = a (.wire 11 28) ∧ a (.virt 18979) = a (.wire 11 29) ∧ a (.wire 11 27) = a (.wire 11 30) ∧ a (.wire 1 43) = a (.wire 11 32) ∧ a (.virt 9478) = a (.wire 11 33) ∧ a (.virt 9478) = a (.wire 11 34) ∧ a (.wire 1 43) = a (.wire 11 36) ∧ a (.virt 18979) = a (.wire 11 37) ∧ a (.wire 11 35) = a (.wire 11 38) ∧ a (.wire 1 43) = a (.wire 11 40) ∧ a (.virt 9465) = a (.wire 11 41) ∧ a (.virt 9465) = a (.wire 11 42) ∧ a (.wire 1 43) = a (.wire 11 44) ∧ a (.virt 18979) = a (.wire 11 45) ∧ a (.wire 11 43) = a (.wire 11 46) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies14, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies15 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 27) = a (.wire 11 48) ∧ a (.virt 18956) = a (.wire 11 49) ∧ a (.virt 18956) = a (.wire 11 50) ∧ a (.wire 2 27) = a (.wire 11 52) ∧ a (.virt 18979) = a (.wire 11 53) ∧ a (.wire 11 51) = a (.wire 11 54) ∧ a (.wire 2 27) = a (.wire 11 56) ∧ a (.virt 18957) = a (.wire 11 57) ∧ a (.virt 18957) = a (.wire 11 58) ∧ a (.wire 2 27) = a (.wire 12 0) ∧ a (.virt 18979) = a (.wire 12 1) ∧ a (.wire 11 59) = a (.wire 12 2) ∧ a (.wire 2 27) = a (.wire 12 4) ∧ a (.virt 18958) = a (.wire 12 5) ∧ a (.virt 18958) = a (.wire 12 6) ∧ a (.wire 2 27) = a (.wire 12 8) ∧ a (.virt 18979) = a (.wire 12 9) ∧ a (.wire 12 7) = a (.wire 12 10) ∧ a (.wire 2 27) = a (.wire 12 12) ∧ a (.virt 18959) = a (.wire 12 13) ∧ a (.virt 18959) = a (.wire 12 14) ∧ a (.wire 2 27) = a (.wire 12 16) ∧ a (.virt 18979) = a (.wire 12 17) ∧ a (.wire 12 15) = a (.wire 12 18) ∧ a (.wire 2 27) = a (.wire 12 20) ∧ a (.virt 18949) = a (.wire 12 21) ∧ a (.virt 18949) = a (.wire 12 22) ∧ a (.wire 2 27) = a (.wire 12 24) ∧ a (.virt 18979) = a (.wire 12 25) ∧ a (.wire 12 23) = a (.wire 12 26) ∧ a (.wire 2 27) = a (.wire 12 28) ∧ a (.virt 18960) = a (.wire 12 29) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies15, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies16 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18960) = a (.wire 12 30) ∧ a (.wire 2 27) = a (.wire 12 32) ∧ a (.virt 18979) = a (.wire 12 33) ∧ a (.wire 12 31) = a (.wire 12 34) ∧ a (.wire 2 27) = a (.wire 12 36) ∧ a (.virt 18961) = a (.wire 12 37) ∧ a (.virt 18961) = a (.wire 12 38) ∧ a (.wire 2 27) = a (.wire 12 40) ∧ a (.virt 18979) = a (.wire 12 41) ∧ a (.wire 12 39) = a (.wire 12 42) ∧ a (.wire 2 27) = a (.wire 12 44) ∧ a (.virt 18962) = a (.wire 12 45) ∧ a (.virt 18962) = a (.wire 12 46) ∧ a (.wire 2 27) = a (.wire 12 48) ∧ a (.virt 18979) = a (.wire 12 49) ∧ a (.wire 12 47) = a (.wire 12 50) ∧ a (.wire 2 27) = a (.wire 12 52) ∧ a (.virt 18963) = a (.wire 12 53) ∧ a (.virt 18963) = a (.wire 12 54) ∧ a (.wire 2 27) = a (.wire 12 56) ∧ a (.virt 18979) = a (.wire 12 57) ∧ a (.wire 12 55) = a (.wire 12 58) ∧ a (.wire 2 27) = a (.wire 13 0) ∧ a (.virt 18950) = a (.wire 13 1) ∧ a (.virt 18950) = a (.wire 13 2) ∧ a (.wire 2 27) = a (.wire 13 4) ∧ a (.virt 18979) = a (.wire 13 5) ∧ a (.wire 13 3) = a (.wire 13 6) ∧ a (.wire 1 43) = a (.wire 13 8) ∧ a (.virt 9484) = a (.wire 13 9) ∧ a (.virt 9484) = a (.wire 13 10) ∧ a (.wire 1 43) = a (.wire 13 12) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies16, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies17 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18979) = a (.wire 13 13) ∧ a (.wire 13 11) = a (.wire 13 14) ∧ a (.wire 2 27) = a (.wire 13 16) ∧ a (.virt 18969) = a (.wire 13 17) ∧ a (.virt 18969) = a (.wire 13 18) ∧ a (.wire 2 27) = a (.wire 13 20) ∧ a (.virt 18979) = a (.wire 13 21) ∧ a (.wire 13 19) = a (.wire 13 22) ∧ a (.wire 13 15) = a (.wire 6 20) ∧ a (.virt 18978) = a (.wire 6 21) ∧ a (.wire 13 23) = a (.wire 6 22) ∧ a (.wire 11 7) = a (.wire 6 24) ∧ a (.virt 18978) = a (.wire 6 25) ∧ a (.wire 11 47) = a (.wire 6 26) ∧ a (.wire 6 27) = a (.wire 6 28) ∧ a (.virt 18978) = a (.wire 6 29) ∧ a (.wire 12 27) = a (.wire 6 30) ∧ a (.wire 6 31) = a (.wire 6 32) ∧ a (.virt 18978) = a (.wire 6 33) ∧ a (.wire 13 7) = a (.wire 6 34) ∧ a (.virt 19017) = a (.wire 13 24) ∧ a (.virt 18978) = a (.wire 13 25) ∧ a (.wire 4 27) = a (.wire 13 26) ∧ a (.wire 14 15) = a (.virt 18979) ∧ a (.wire 14 16) = a (.virt 18979) ∧ a (.wire 14 17) = a (.virt 18979) ∧ a (.wire 14 18) = a (.virt 18979) ∧ a (.wire 14 19) = a (.virt 18979) ∧ a (.wire 14 20) = a (.virt 18979) ∧ a (.wire 14 21) = a (.virt 18979) ∧ a (.wire 14 22) = a (.virt 18979) ∧ a (.wire 14 23) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies17, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies18 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 24) = a (.virt 18979) ∧ a (.wire 14 25) = a (.virt 18979) ∧ a (.wire 14 26) = a (.virt 18979) ∧ a (.wire 14 27) = a (.virt 18979) ∧ a (.wire 14 28) = a (.virt 18979) ∧ a (.wire 14 29) = a (.virt 18979) ∧ a (.wire 14 30) = a (.virt 18979) ∧ a (.wire 14 31) = a (.virt 18979) ∧ a (.wire 14 32) = a (.virt 18979) ∧ a (.wire 14 33) = a (.virt 18979) ∧ a (.wire 14 34) = a (.virt 18979) ∧ a (.wire 14 35) = a (.virt 18979) ∧ a (.wire 14 36) = a (.virt 18979) ∧ a (.wire 14 37) = a (.virt 18979) ∧ a (.wire 14 38) = a (.virt 18979) ∧ a (.wire 14 39) = a (.virt 18979) ∧ a (.wire 14 40) = a (.virt 18979) ∧ a (.wire 14 41) = a (.virt 18979) ∧ a (.wire 14 42) = a (.virt 18979) ∧ a (.wire 14 43) = a (.virt 18979) ∧ a (.wire 14 44) = a (.virt 18979) ∧ a (.wire 14 45) = a (.virt 18979) ∧ a (.wire 14 46) = a (.virt 18979) ∧ a (.wire 14 47) = a (.virt 18979) ∧ a (.wire 14 48) = a (.virt 18979) ∧ a (.wire 14 49) = a (.virt 18979) ∧ a (.wire 14 50) = a (.virt 18979) ∧ a (.wire 14 51) = a (.virt 18979) ∧ a (.wire 14 52) = a (.virt 18979) ∧ a (.wire 14 53) = a (.virt 18979) ∧ a (.wire 14 54) = a (.virt 18979) ∧ a (.wire 14 55) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies18, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies19 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 56) = a (.virt 18979) ∧ a (.wire 14 57) = a (.virt 18979) ∧ a (.wire 14 58) = a (.virt 18979) ∧ a (.wire 14 59) = a (.virt 18979) ∧ a (.wire 14 0) = a (.wire 13 27) ∧ a (.wire 6 35) = a (.wire 10 16) ∧ a (.virt 19017) = a (.wire 10 17) ∧ a (.wire 6 35) = a (.wire 10 18) ∧ a (.wire 6 23) = a (.wire 10 20) ∧ a (.wire 13 27) = a (.wire 10 21) ∧ a (.wire 6 23) = a (.wire 10 22) ∧ a (.wire 10 23) = a (.wire 13 28) ∧ a (.virt 18978) = a (.wire 13 29) ∧ a (.wire 10 19) = a (.wire 13 30) ∧ a (.wire 15 53) = a (.virt 18979) ∧ a (.wire 15 54) = a (.virt 18979) ∧ a (.wire 15 55) = a (.virt 18979) ∧ a (.wire 15 56) = a (.virt 18979) ∧ a (.wire 15 57) = a (.virt 18979) ∧ a (.wire 15 58) = a (.virt 18979) ∧ a (.wire 15 59) = a (.virt 18979) ∧ a (.wire 15 0) = a (.wire 13 31) ∧ a (.virt 18978) = a (.wire 13 32) ∧ a (.virt 18978) = a (.wire 13 33) ∧ a (.virt 19019) = a (.wire 13 34) ∧ a (.wire 9 35) = a (.wire 13 36) ∧ a (.virt 18978) = a (.wire 13 37) ∧ a (.wire 9 35) = a (.wire 13 38) ∧ a (.virt 19019) = a (.wire 10 24) ∧ a (.wire 13 39) = a (.wire 10 25) ∧ a (.virt 19019) = a (.wire 10 26) ∧ a (.wire 13 39) = a (.wire 10 28) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies19, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies20 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19020) = a (.wire 10 29) ∧ a (.wire 13 39) = a (.wire 10 30) ∧ a (.wire 10 31) = a (.wire 13 40) ∧ a (.virt 18978) = a (.wire 13 41) ∧ a (.wire 13 35) = a (.wire 13 42) ∧ a (.wire 10 27) = a (.virt 18979) ∧ a (.wire 13 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 13 44) ∧ a (.virt 18978) = a (.wire 13 45) ∧ a (.virt 19021) = a (.wire 13 46) ∧ a (.wire 9 43) = a (.wire 13 48) ∧ a (.virt 18978) = a (.wire 13 49) ∧ a (.wire 9 43) = a (.wire 13 50) ∧ a (.virt 19021) = a (.wire 10 32) ∧ a (.wire 13 51) = a (.wire 10 33) ∧ a (.virt 19021) = a (.wire 10 34) ∧ a (.wire 13 51) = a (.wire 10 36) ∧ a (.virt 19022) = a (.wire 10 37) ∧ a (.wire 13 51) = a (.wire 10 38) ∧ a (.wire 10 39) = a (.wire 13 52) ∧ a (.virt 18978) = a (.wire 13 53) ∧ a (.wire 13 47) = a (.wire 13 54) ∧ a (.wire 10 35) = a (.virt 18979) ∧ a (.wire 13 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 13 56) ∧ a (.virt 18978) = a (.wire 13 57) ∧ a (.virt 19023) = a (.wire 13 58) ∧ a (.wire 9 51) = a (.wire 16 0) ∧ a (.virt 18978) = a (.wire 16 1) ∧ a (.wire 9 51) = a (.wire 16 2) ∧ a (.virt 19023) = a (.wire 10 40) ∧ a (.wire 16 3) = a (.wire 10 41) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies20, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies21 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19023) = a (.wire 10 42) ∧ a (.wire 16 3) = a (.wire 10 44) ∧ a (.virt 19024) = a (.wire 10 45) ∧ a (.wire 16 3) = a (.wire 10 46) ∧ a (.wire 10 47) = a (.wire 16 4) ∧ a (.virt 18978) = a (.wire 16 5) ∧ a (.wire 13 59) = a (.wire 16 6) ∧ a (.wire 10 43) = a (.virt 18979) ∧ a (.wire 16 7) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 16 8) ∧ a (.virt 18978) = a (.wire 16 9) ∧ a (.virt 19025) = a (.wire 16 10) ∧ a (.wire 9 59) = a (.wire 16 12) ∧ a (.virt 18978) = a (.wire 16 13) ∧ a (.wire 9 59) = a (.wire 16 14) ∧ a (.virt 19025) = a (.wire 10 48) ∧ a (.wire 16 15) = a (.wire 10 49) ∧ a (.virt 19025) = a (.wire 10 50) ∧ a (.wire 16 15) = a (.wire 10 52) ∧ a (.virt 19026) = a (.wire 10 53) ∧ a (.wire 16 15) = a (.wire 10 54) ∧ a (.wire 10 55) = a (.wire 16 16) ∧ a (.virt 18978) = a (.wire 16 17) ∧ a (.wire 16 11) = a (.wire 16 18) ∧ a (.wire 10 51) = a (.virt 18979) ∧ a (.wire 16 19) = a (.virt 18979) ∧ a (.virt 19019) = a (.wire 10 56) ∧ a (.virt 19021) = a (.wire 10 57) ∧ a (.virt 19019) = a (.wire 10 58) ∧ a (.virt 19023) = a (.wire 17 0) ∧ a (.virt 19025) = a (.wire 17 1) ∧ a (.virt 19023) = a (.wire 17 2) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies21, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies22 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 59) = a (.wire 17 4) ∧ a (.wire 17 3) = a (.wire 17 5) ∧ a (.wire 10 59) = a (.wire 17 6) ∧ a (.wire 17 7) = a (.wire 16 20) ∧ a (.wire 11 7) = a (.wire 16 21) ∧ a (.virt 18979) = a (.wire 16 22) ∧ a (.virt 18978) = a (.wire 16 24) ∧ a (.virt 18978) = a (.wire 16 25) ∧ a (.virt 19027) = a (.wire 16 26) ∧ a (.wire 11 15) = a (.wire 16 28) ∧ a (.virt 18978) = a (.wire 16 29) ∧ a (.wire 9 35) = a (.wire 16 30) ∧ a (.virt 19027) = a (.wire 17 8) ∧ a (.wire 16 31) = a (.wire 17 9) ∧ a (.virt 19027) = a (.wire 17 10) ∧ a (.wire 16 31) = a (.wire 17 12) ∧ a (.virt 19028) = a (.wire 17 13) ∧ a (.wire 16 31) = a (.wire 17 14) ∧ a (.wire 17 15) = a (.wire 16 32) ∧ a (.virt 18978) = a (.wire 16 33) ∧ a (.wire 16 27) = a (.wire 16 34) ∧ a (.wire 17 11) = a (.virt 18979) ∧ a (.wire 16 35) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 16 36) ∧ a (.virt 18978) = a (.wire 16 37) ∧ a (.virt 19029) = a (.wire 16 38) ∧ a (.wire 11 23) = a (.wire 16 40) ∧ a (.virt 18978) = a (.wire 16 41) ∧ a (.wire 9 43) = a (.wire 16 42) ∧ a (.virt 19029) = a (.wire 17 16) ∧ a (.wire 16 43) = a (.wire 17 17) ∧ a (.virt 19029) = a (.wire 17 18) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies22, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies23 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 43) = a (.wire 17 20) ∧ a (.virt 19030) = a (.wire 17 21) ∧ a (.wire 16 43) = a (.wire 17 22) ∧ a (.wire 17 23) = a (.wire 16 44) ∧ a (.virt 18978) = a (.wire 16 45) ∧ a (.wire 16 39) = a (.wire 16 46) ∧ a (.wire 17 19) = a (.virt 18979) ∧ a (.wire 16 47) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 16 48) ∧ a (.virt 18978) = a (.wire 16 49) ∧ a (.virt 19031) = a (.wire 16 50) ∧ a (.wire 11 31) = a (.wire 16 52) ∧ a (.virt 18978) = a (.wire 16 53) ∧ a (.wire 9 51) = a (.wire 16 54) ∧ a (.virt 19031) = a (.wire 17 24) ∧ a (.wire 16 55) = a (.wire 17 25) ∧ a (.virt 19031) = a (.wire 17 26) ∧ a (.wire 16 55) = a (.wire 17 28) ∧ a (.virt 19032) = a (.wire 17 29) ∧ a (.wire 16 55) = a (.wire 17 30) ∧ a (.wire 17 31) = a (.wire 16 56) ∧ a (.virt 18978) = a (.wire 16 57) ∧ a (.wire 16 51) = a (.wire 16 58) ∧ a (.wire 17 27) = a (.virt 18979) ∧ a (.wire 16 59) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 18 0) ∧ a (.virt 18978) = a (.wire 18 1) ∧ a (.virt 19033) = a (.wire 18 2) ∧ a (.wire 11 39) = a (.wire 18 4) ∧ a (.virt 18978) = a (.wire 18 5) ∧ a (.wire 9 59) = a (.wire 18 6) ∧ a (.virt 19033) = a (.wire 17 32) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies23, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies23, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies24 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 7) = a (.wire 17 33) ∧ a (.virt 19033) = a (.wire 17 34) ∧ a (.wire 18 7) = a (.wire 17 36) ∧ a (.virt 19034) = a (.wire 17 37) ∧ a (.wire 18 7) = a (.wire 17 38) ∧ a (.wire 17 39) = a (.wire 18 8) ∧ a (.virt 18978) = a (.wire 18 9) ∧ a (.wire 18 3) = a (.wire 18 10) ∧ a (.wire 17 35) = a (.virt 18979) ∧ a (.wire 18 11) = a (.virt 18979) ∧ a (.virt 19027) = a (.wire 17 40) ∧ a (.virt 19029) = a (.wire 17 41) ∧ a (.virt 19027) = a (.wire 17 42) ∧ a (.virt 19031) = a (.wire 17 44) ∧ a (.virt 19033) = a (.wire 17 45) ∧ a (.virt 19031) = a (.wire 17 46) ∧ a (.wire 17 43) = a (.wire 17 48) ∧ a (.wire 17 47) = a (.wire 17 49) ∧ a (.wire 17 43) = a (.wire 17 50) ∧ a (.wire 17 51) = a (.wire 18 12) ∧ a (.wire 11 47) = a (.wire 18 13) ∧ a (.virt 18979) = a (.wire 18 14) ∧ a (.wire 16 23) = a (.wire 6 36) ∧ a (.virt 18978) = a (.wire 6 37) ∧ a (.wire 18 15) = a (.wire 6 38) ∧ a (.virt 18978) = a (.wire 18 16) ∧ a (.virt 18978) = a (.wire 18 17) ∧ a (.virt 19035) = a (.wire 18 18) ∧ a (.wire 11 55) = a (.wire 18 20) ∧ a (.virt 18978) = a (.wire 18 21) ∧ a (.wire 9 35) = a (.wire 18 22) ∧ a (.virt 19035) = a (.wire 17 52) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies24, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies24, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies25 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 23) = a (.wire 17 53) ∧ a (.virt 19035) = a (.wire 17 54) ∧ a (.wire 18 23) = a (.wire 17 56) ∧ a (.virt 19036) = a (.wire 17 57) ∧ a (.wire 18 23) = a (.wire 17 58) ∧ a (.wire 17 59) = a (.wire 18 24) ∧ a (.virt 18978) = a (.wire 18 25) ∧ a (.wire 18 19) = a (.wire 18 26) ∧ a (.wire 17 55) = a (.virt 18979) ∧ a (.wire 18 27) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 18 28) ∧ a (.virt 18978) = a (.wire 18 29) ∧ a (.virt 19037) = a (.wire 18 30) ∧ a (.wire 12 3) = a (.wire 18 32) ∧ a (.virt 18978) = a (.wire 18 33) ∧ a (.wire 9 43) = a (.wire 18 34) ∧ a (.virt 19037) = a (.wire 19 0) ∧ a (.wire 18 35) = a (.wire 19 1) ∧ a (.virt 19037) = a (.wire 19 2) ∧ a (.wire 18 35) = a (.wire 19 4) ∧ a (.virt 19038) = a (.wire 19 5) ∧ a (.wire 18 35) = a (.wire 19 6) ∧ a (.wire 19 7) = a (.wire 18 36) ∧ a (.virt 18978) = a (.wire 18 37) ∧ a (.wire 18 31) = a (.wire 18 38) ∧ a (.wire 19 3) = a (.virt 18979) ∧ a (.wire 18 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 18 40) ∧ a (.virt 18978) = a (.wire 18 41) ∧ a (.virt 19039) = a (.wire 18 42) ∧ a (.wire 12 11) = a (.wire 18 44) ∧ a (.virt 18978) = a (.wire 18 45) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies25, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies25, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies26 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 51) = a (.wire 18 46) ∧ a (.virt 19039) = a (.wire 19 8) ∧ a (.wire 18 47) = a (.wire 19 9) ∧ a (.virt 19039) = a (.wire 19 10) ∧ a (.wire 18 47) = a (.wire 19 12) ∧ a (.virt 19040) = a (.wire 19 13) ∧ a (.wire 18 47) = a (.wire 19 14) ∧ a (.wire 19 15) = a (.wire 18 48) ∧ a (.virt 18978) = a (.wire 18 49) ∧ a (.wire 18 43) = a (.wire 18 50) ∧ a (.wire 19 11) = a (.virt 18979) ∧ a (.wire 18 51) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 18 52) ∧ a (.virt 18978) = a (.wire 18 53) ∧ a (.virt 19041) = a (.wire 18 54) ∧ a (.wire 12 19) = a (.wire 18 56) ∧ a (.virt 18978) = a (.wire 18 57) ∧ a (.wire 9 59) = a (.wire 18 58) ∧ a (.virt 19041) = a (.wire 19 16) ∧ a (.wire 18 59) = a (.wire 19 17) ∧ a (.virt 19041) = a (.wire 19 18) ∧ a (.wire 18 59) = a (.wire 19 20) ∧ a (.virt 19042) = a (.wire 19 21) ∧ a (.wire 18 59) = a (.wire 19 22) ∧ a (.wire 19 23) = a (.wire 20 0) ∧ a (.virt 18978) = a (.wire 20 1) ∧ a (.wire 18 55) = a (.wire 20 2) ∧ a (.wire 19 19) = a (.virt 18979) ∧ a (.wire 20 3) = a (.virt 18979) ∧ a (.virt 19035) = a (.wire 19 24) ∧ a (.virt 19037) = a (.wire 19 25) ∧ a (.virt 19035) = a (.wire 19 26) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies26, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies26, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies27 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19039) = a (.wire 19 28) ∧ a (.virt 19041) = a (.wire 19 29) ∧ a (.virt 19039) = a (.wire 19 30) ∧ a (.wire 19 27) = a (.wire 19 32) ∧ a (.wire 19 31) = a (.wire 19 33) ∧ a (.wire 19 27) = a (.wire 19 34) ∧ a (.wire 19 35) = a (.wire 20 4) ∧ a (.wire 12 27) = a (.wire 20 5) ∧ a (.virt 18979) = a (.wire 20 6) ∧ a (.wire 6 39) = a (.wire 6 40) ∧ a (.virt 18978) = a (.wire 6 41) ∧ a (.wire 20 7) = a (.wire 6 42) ∧ a (.virt 18978) = a (.wire 20 8) ∧ a (.virt 18978) = a (.wire 20 9) ∧ a (.virt 19043) = a (.wire 20 10) ∧ a (.wire 12 35) = a (.wire 20 12) ∧ a (.virt 18978) = a (.wire 20 13) ∧ a (.wire 9 35) = a (.wire 20 14) ∧ a (.virt 19043) = a (.wire 19 36) ∧ a (.wire 20 15) = a (.wire 19 37) ∧ a (.virt 19043) = a (.wire 19 38) ∧ a (.wire 20 15) = a (.wire 19 40) ∧ a (.virt 19044) = a (.wire 19 41) ∧ a (.wire 20 15) = a (.wire 19 42) ∧ a (.wire 19 43) = a (.wire 20 16) ∧ a (.virt 18978) = a (.wire 20 17) ∧ a (.wire 20 11) = a (.wire 20 18) ∧ a (.wire 19 39) = a (.virt 18979) ∧ a (.wire 20 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 20 20) ∧ a (.virt 18978) = a (.wire 20 21) ∧ a (.virt 19045) = a (.wire 20 22) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies27, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies27, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies28 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 43) = a (.wire 20 24) ∧ a (.virt 18978) = a (.wire 20 25) ∧ a (.wire 9 43) = a (.wire 20 26) ∧ a (.virt 19045) = a (.wire 19 44) ∧ a (.wire 20 27) = a (.wire 19 45) ∧ a (.virt 19045) = a (.wire 19 46) ∧ a (.wire 20 27) = a (.wire 19 48) ∧ a (.virt 19046) = a (.wire 19 49) ∧ a (.wire 20 27) = a (.wire 19 50) ∧ a (.wire 19 51) = a (.wire 20 28) ∧ a (.virt 18978) = a (.wire 20 29) ∧ a (.wire 20 23) = a (.wire 20 30) ∧ a (.wire 19 47) = a (.virt 18979) ∧ a (.wire 20 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 20 32) ∧ a (.virt 18978) = a (.wire 20 33) ∧ a (.virt 19047) = a (.wire 20 34) ∧ a (.wire 12 51) = a (.wire 20 36) ∧ a (.virt 18978) = a (.wire 20 37) ∧ a (.wire 9 51) = a (.wire 20 38) ∧ a (.virt 19047) = a (.wire 19 52) ∧ a (.wire 20 39) = a (.wire 19 53) ∧ a (.virt 19047) = a (.wire 19 54) ∧ a (.wire 20 39) = a (.wire 19 56) ∧ a (.virt 19048) = a (.wire 19 57) ∧ a (.wire 20 39) = a (.wire 19 58) ∧ a (.wire 19 59) = a (.wire 20 40) ∧ a (.virt 18978) = a (.wire 20 41) ∧ a (.wire 20 35) = a (.wire 20 42) ∧ a (.wire 19 55) = a (.virt 18979) ∧ a (.wire 20 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 20 44) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies28, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies28, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies29 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 20 45) ∧ a (.virt 19049) = a (.wire 20 46) ∧ a (.wire 12 59) = a (.wire 20 48) ∧ a (.virt 18978) = a (.wire 20 49) ∧ a (.wire 9 59) = a (.wire 20 50) ∧ a (.virt 19049) = a (.wire 21 0) ∧ a (.wire 20 51) = a (.wire 21 1) ∧ a (.virt 19049) = a (.wire 21 2) ∧ a (.wire 20 51) = a (.wire 21 4) ∧ a (.virt 19050) = a (.wire 21 5) ∧ a (.wire 20 51) = a (.wire 21 6) ∧ a (.wire 21 7) = a (.wire 20 52) ∧ a (.virt 18978) = a (.wire 20 53) ∧ a (.wire 20 47) = a (.wire 20 54) ∧ a (.wire 21 3) = a (.virt 18979) ∧ a (.wire 20 55) = a (.virt 18979) ∧ a (.virt 19043) = a (.wire 21 8) ∧ a (.virt 19045) = a (.wire 21 9) ∧ a (.virt 19043) = a (.wire 21 10) ∧ a (.virt 19047) = a (.wire 21 12) ∧ a (.virt 19049) = a (.wire 21 13) ∧ a (.virt 19047) = a (.wire 21 14) ∧ a (.wire 21 11) = a (.wire 21 16) ∧ a (.wire 21 15) = a (.wire 21 17) ∧ a (.wire 21 11) = a (.wire 21 18) ∧ a (.wire 21 19) = a (.wire 20 56) ∧ a (.wire 13 7) = a (.wire 20 57) ∧ a (.virt 18979) = a (.wire 20 58) ∧ a (.wire 6 43) = a (.wire 6 44) ∧ a (.virt 18978) = a (.wire 6 45) ∧ a (.wire 20 59) = a (.wire 6 46) ∧ a (.virt 18979) = a (.wire 22 0) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies29, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies29, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies30 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 47) = a (.wire 22 1) ∧ a (.wire 6 47) = a (.wire 22 2) ∧ a (.virt 18979) = a (.wire 22 4) ∧ a (.virt 18979) = a (.wire 22 5) ∧ a (.wire 22 3) = a (.wire 22 6) ∧ a (.virt 18979) = a (.wire 22 8) ∧ a (.wire 9 35) = a (.wire 22 9) ∧ a (.wire 9 35) = a (.wire 22 10) ∧ a (.virt 18979) = a (.wire 22 12) ∧ a (.virt 18979) = a (.wire 22 13) ∧ a (.wire 22 11) = a (.wire 22 14) ∧ a (.virt 18979) = a (.wire 22 16) ∧ a (.wire 9 43) = a (.wire 22 17) ∧ a (.wire 9 43) = a (.wire 22 18) ∧ a (.virt 18979) = a (.wire 22 20) ∧ a (.virt 18979) = a (.wire 22 21) ∧ a (.wire 22 19) = a (.wire 22 22) ∧ a (.virt 18979) = a (.wire 22 24) ∧ a (.wire 9 51) = a (.wire 22 25) ∧ a (.wire 9 51) = a (.wire 22 26) ∧ a (.virt 18979) = a (.wire 22 28) ∧ a (.virt 18979) = a (.wire 22 29) ∧ a (.wire 22 27) = a (.wire 22 30) ∧ a (.virt 18979) = a (.wire 22 32) ∧ a (.wire 9 59) = a (.wire 22 33) ∧ a (.wire 9 59) = a (.wire 22 34) ∧ a (.virt 18979) = a (.wire 22 36) ∧ a (.virt 18979) = a (.wire 22 37) ∧ a (.wire 22 35) = a (.wire 22 38) ∧ a (.wire 23 33) = a (.virt 18979) ∧ a (.wire 23 34) = a (.virt 18979) ∧ a (.wire 23 35) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies30, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies30, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies31 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 23 36) = a (.virt 18979) ∧ a (.wire 23 37) = a (.virt 18979) ∧ a (.wire 23 38) = a (.virt 18979) ∧ a (.wire 23 39) = a (.virt 18979) ∧ a (.wire 23 40) = a (.virt 18979) ∧ a (.wire 23 41) = a (.virt 18979) ∧ a (.wire 23 42) = a (.virt 18979) ∧ a (.wire 23 43) = a (.virt 18979) ∧ a (.wire 23 44) = a (.virt 18979) ∧ a (.wire 23 45) = a (.virt 18979) ∧ a (.wire 23 46) = a (.virt 18979) ∧ a (.wire 23 47) = a (.virt 18979) ∧ a (.wire 23 48) = a (.virt 18979) ∧ a (.wire 23 49) = a (.virt 18979) ∧ a (.wire 23 50) = a (.virt 18979) ∧ a (.wire 23 51) = a (.virt 18979) ∧ a (.wire 23 52) = a (.virt 18979) ∧ a (.wire 23 53) = a (.virt 18979) ∧ a (.wire 23 54) = a (.virt 18979) ∧ a (.wire 23 55) = a (.virt 18979) ∧ a (.wire 23 56) = a (.virt 18979) ∧ a (.wire 23 57) = a (.virt 18979) ∧ a (.wire 23 58) = a (.virt 18979) ∧ a (.wire 23 59) = a (.virt 18979) ∧ a (.wire 23 0) = a (.wire 22 7) ∧ a (.virt 18978) = a (.wire 22 40) ∧ a (.virt 18978) = a (.wire 22 41) ∧ a (.virt 19051) = a (.wire 22 42) ∧ a (.wire 9 35) = a (.wire 22 44) ∧ a (.virt 18978) = a (.wire 22 45) ∧ a (.wire 11 15) = a (.wire 22 46) ∧ a (.virt 19051) = a (.wire 21 20) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies31, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies31, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies32 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 22 47) = a (.wire 21 21) ∧ a (.virt 19051) = a (.wire 21 22) ∧ a (.wire 22 47) = a (.wire 21 24) ∧ a (.virt 19052) = a (.wire 21 25) ∧ a (.wire 22 47) = a (.wire 21 26) ∧ a (.wire 21 27) = a (.wire 22 48) ∧ a (.virt 18978) = a (.wire 22 49) ∧ a (.wire 22 43) = a (.wire 22 50) ∧ a (.wire 21 23) = a (.virt 18979) ∧ a (.wire 22 51) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 22 52) ∧ a (.virt 18978) = a (.wire 22 53) ∧ a (.virt 19053) = a (.wire 22 54) ∧ a (.wire 9 43) = a (.wire 22 56) ∧ a (.virt 18978) = a (.wire 22 57) ∧ a (.wire 11 23) = a (.wire 22 58) ∧ a (.virt 19053) = a (.wire 21 28) ∧ a (.wire 22 59) = a (.wire 21 29) ∧ a (.virt 19053) = a (.wire 21 30) ∧ a (.wire 22 59) = a (.wire 21 32) ∧ a (.virt 19054) = a (.wire 21 33) ∧ a (.wire 22 59) = a (.wire 21 34) ∧ a (.wire 21 35) = a (.wire 24 0) ∧ a (.virt 18978) = a (.wire 24 1) ∧ a (.wire 22 55) = a (.wire 24 2) ∧ a (.wire 21 31) = a (.virt 18979) ∧ a (.wire 24 3) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 24 4) ∧ a (.virt 18978) = a (.wire 24 5) ∧ a (.virt 19055) = a (.wire 24 6) ∧ a (.wire 9 51) = a (.wire 24 8) ∧ a (.virt 18978) = a (.wire 24 9) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies32, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies32, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies33 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 31) = a (.wire 24 10) ∧ a (.virt 19055) = a (.wire 21 36) ∧ a (.wire 24 11) = a (.wire 21 37) ∧ a (.virt 19055) = a (.wire 21 38) ∧ a (.wire 24 11) = a (.wire 21 40) ∧ a (.virt 19056) = a (.wire 21 41) ∧ a (.wire 24 11) = a (.wire 21 42) ∧ a (.wire 21 43) = a (.wire 24 12) ∧ a (.virt 18978) = a (.wire 24 13) ∧ a (.wire 24 7) = a (.wire 24 14) ∧ a (.wire 21 39) = a (.virt 18979) ∧ a (.wire 24 15) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 24 16) ∧ a (.virt 18978) = a (.wire 24 17) ∧ a (.virt 19057) = a (.wire 24 18) ∧ a (.wire 9 59) = a (.wire 24 20) ∧ a (.virt 18978) = a (.wire 24 21) ∧ a (.wire 11 39) = a (.wire 24 22) ∧ a (.virt 19057) = a (.wire 21 44) ∧ a (.wire 24 23) = a (.wire 21 45) ∧ a (.virt 19057) = a (.wire 21 46) ∧ a (.wire 24 23) = a (.wire 21 48) ∧ a (.virt 19058) = a (.wire 21 49) ∧ a (.wire 24 23) = a (.wire 21 50) ∧ a (.wire 21 51) = a (.wire 24 24) ∧ a (.virt 18978) = a (.wire 24 25) ∧ a (.wire 24 19) = a (.wire 24 26) ∧ a (.wire 21 47) = a (.virt 18979) ∧ a (.wire 24 27) = a (.virt 18979) ∧ a (.virt 19051) = a (.wire 21 52) ∧ a (.virt 19053) = a (.wire 21 53) ∧ a (.virt 19051) = a (.wire 21 54) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies33, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies33, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies34 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19055) = a (.wire 21 56) ∧ a (.virt 19057) = a (.wire 21 57) ∧ a (.virt 19055) = a (.wire 21 58) ∧ a (.wire 21 55) = a (.wire 25 0) ∧ a (.wire 21 59) = a (.wire 25 1) ∧ a (.wire 21 55) = a (.wire 25 2) ∧ a (.virt 18978) = a (.wire 24 28) ∧ a (.virt 18978) = a (.wire 24 29) ∧ a (.virt 19059) = a (.wire 24 30) ∧ a (.virt 19059) = a (.wire 25 4) ∧ a (.wire 22 47) = a (.wire 25 5) ∧ a (.virt 19059) = a (.wire 25 6) ∧ a (.wire 22 47) = a (.wire 25 8) ∧ a (.virt 19060) = a (.wire 25 9) ∧ a (.wire 22 47) = a (.wire 25 10) ∧ a (.wire 25 11) = a (.wire 24 32) ∧ a (.virt 18978) = a (.wire 24 33) ∧ a (.wire 24 31) = a (.wire 24 34) ∧ a (.wire 25 7) = a (.virt 18979) ∧ a (.wire 24 35) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 24 36) ∧ a (.virt 18978) = a (.wire 24 37) ∧ a (.virt 19061) = a (.wire 24 38) ∧ a (.virt 19061) = a (.wire 25 12) ∧ a (.wire 22 59) = a (.wire 25 13) ∧ a (.virt 19061) = a (.wire 25 14) ∧ a (.wire 22 59) = a (.wire 25 16) ∧ a (.virt 19062) = a (.wire 25 17) ∧ a (.wire 22 59) = a (.wire 25 18) ∧ a (.wire 25 19) = a (.wire 24 40) ∧ a (.virt 18978) = a (.wire 24 41) ∧ a (.wire 24 39) = a (.wire 24 42) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies34, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies34, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies35 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 25 15) = a (.virt 18979) ∧ a (.wire 24 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 24 44) ∧ a (.virt 18978) = a (.wire 24 45) ∧ a (.virt 19063) = a (.wire 24 46) ∧ a (.virt 19063) = a (.wire 25 20) ∧ a (.wire 24 11) = a (.wire 25 21) ∧ a (.virt 19063) = a (.wire 25 22) ∧ a (.wire 24 11) = a (.wire 25 24) ∧ a (.virt 19064) = a (.wire 25 25) ∧ a (.wire 24 11) = a (.wire 25 26) ∧ a (.wire 25 27) = a (.wire 24 48) ∧ a (.virt 18978) = a (.wire 24 49) ∧ a (.wire 24 47) = a (.wire 24 50) ∧ a (.wire 25 23) = a (.virt 18979) ∧ a (.wire 24 51) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 24 52) ∧ a (.virt 18978) = a (.wire 24 53) ∧ a (.virt 19065) = a (.wire 24 54) ∧ a (.virt 19065) = a (.wire 25 28) ∧ a (.wire 24 23) = a (.wire 25 29) ∧ a (.virt 19065) = a (.wire 25 30) ∧ a (.wire 24 23) = a (.wire 25 32) ∧ a (.virt 19066) = a (.wire 25 33) ∧ a (.wire 24 23) = a (.wire 25 34) ∧ a (.wire 25 35) = a (.wire 24 56) ∧ a (.virt 18978) = a (.wire 24 57) ∧ a (.wire 24 55) = a (.wire 24 58) ∧ a (.wire 25 31) = a (.virt 18979) ∧ a (.wire 24 59) = a (.virt 18979) ∧ a (.virt 19059) = a (.wire 25 36) ∧ a (.virt 19061) = a (.wire 25 37) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies35, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies35, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies36 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19059) = a (.wire 25 38) ∧ a (.virt 19063) = a (.wire 25 40) ∧ a (.virt 19065) = a (.wire 25 41) ∧ a (.virt 19063) = a (.wire 25 42) ∧ a (.wire 25 39) = a (.wire 25 44) ∧ a (.wire 25 43) = a (.wire 25 45) ∧ a (.wire 25 39) = a (.wire 25 46) ∧ a (.wire 25 47) = a (.wire 26 0) ∧ a (.wire 11 7) = a (.wire 26 1) ∧ a (.virt 18979) = a (.wire 26 2) ∧ a (.virt 18978) = a (.wire 26 4) ∧ a (.virt 18978) = a (.wire 26 5) ∧ a (.virt 19067) = a (.wire 26 6) ∧ a (.wire 11 15) = a (.wire 26 8) ∧ a (.virt 18978) = a (.wire 26 9) ∧ a (.wire 11 15) = a (.wire 26 10) ∧ a (.virt 19067) = a (.wire 25 48) ∧ a (.wire 26 11) = a (.wire 25 49) ∧ a (.virt 19067) = a (.wire 25 50) ∧ a (.wire 26 11) = a (.wire 25 52) ∧ a (.virt 19068) = a (.wire 25 53) ∧ a (.wire 26 11) = a (.wire 25 54) ∧ a (.wire 25 55) = a (.wire 26 12) ∧ a (.virt 18978) = a (.wire 26 13) ∧ a (.wire 26 7) = a (.wire 26 14) ∧ a (.wire 25 51) = a (.virt 18979) ∧ a (.wire 26 15) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 26 16) ∧ a (.virt 18978) = a (.wire 26 17) ∧ a (.virt 19069) = a (.wire 26 18) ∧ a (.wire 11 23) = a (.wire 26 20) ∧ a (.virt 18978) = a (.wire 26 21) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies36, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies36, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies37 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 23) = a (.wire 26 22) ∧ a (.virt 19069) = a (.wire 25 56) ∧ a (.wire 26 23) = a (.wire 25 57) ∧ a (.virt 19069) = a (.wire 25 58) ∧ a (.wire 26 23) = a (.wire 27 0) ∧ a (.virt 19070) = a (.wire 27 1) ∧ a (.wire 26 23) = a (.wire 27 2) ∧ a (.wire 27 3) = a (.wire 26 24) ∧ a (.virt 18978) = a (.wire 26 25) ∧ a (.wire 26 19) = a (.wire 26 26) ∧ a (.wire 25 59) = a (.virt 18979) ∧ a (.wire 26 27) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 26 28) ∧ a (.virt 18978) = a (.wire 26 29) ∧ a (.virt 19071) = a (.wire 26 30) ∧ a (.wire 11 31) = a (.wire 26 32) ∧ a (.virt 18978) = a (.wire 26 33) ∧ a (.wire 11 31) = a (.wire 26 34) ∧ a (.virt 19071) = a (.wire 27 4) ∧ a (.wire 26 35) = a (.wire 27 5) ∧ a (.virt 19071) = a (.wire 27 6) ∧ a (.wire 26 35) = a (.wire 27 8) ∧ a (.virt 19072) = a (.wire 27 9) ∧ a (.wire 26 35) = a (.wire 27 10) ∧ a (.wire 27 11) = a (.wire 26 36) ∧ a (.virt 18978) = a (.wire 26 37) ∧ a (.wire 26 31) = a (.wire 26 38) ∧ a (.wire 27 7) = a (.virt 18979) ∧ a (.wire 26 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 26 40) ∧ a (.virt 18978) = a (.wire 26 41) ∧ a (.virt 19073) = a (.wire 26 42) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies37, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies37, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies38 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 39) = a (.wire 26 44) ∧ a (.virt 18978) = a (.wire 26 45) ∧ a (.wire 11 39) = a (.wire 26 46) ∧ a (.virt 19073) = a (.wire 27 12) ∧ a (.wire 26 47) = a (.wire 27 13) ∧ a (.virt 19073) = a (.wire 27 14) ∧ a (.wire 26 47) = a (.wire 27 16) ∧ a (.virt 19074) = a (.wire 27 17) ∧ a (.wire 26 47) = a (.wire 27 18) ∧ a (.wire 27 19) = a (.wire 26 48) ∧ a (.virt 18978) = a (.wire 26 49) ∧ a (.wire 26 43) = a (.wire 26 50) ∧ a (.wire 27 15) = a (.virt 18979) ∧ a (.wire 26 51) = a (.virt 18979) ∧ a (.virt 19067) = a (.wire 27 20) ∧ a (.virt 19069) = a (.wire 27 21) ∧ a (.virt 19067) = a (.wire 27 22) ∧ a (.virt 19071) = a (.wire 27 24) ∧ a (.virt 19073) = a (.wire 27 25) ∧ a (.virt 19071) = a (.wire 27 26) ∧ a (.wire 27 23) = a (.wire 27 28) ∧ a (.wire 27 27) = a (.wire 27 29) ∧ a (.wire 27 23) = a (.wire 27 30) ∧ a (.wire 27 31) = a (.wire 26 52) ∧ a (.wire 11 47) = a (.wire 26 53) ∧ a (.virt 18979) = a (.wire 26 54) ∧ a (.wire 26 3) = a (.wire 6 48) ∧ a (.virt 18978) = a (.wire 6 49) ∧ a (.wire 26 55) = a (.wire 6 50) ∧ a (.virt 18978) = a (.wire 26 56) ∧ a (.virt 18978) = a (.wire 26 57) ∧ a (.virt 19075) = a (.wire 26 58) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies38, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies38, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies39 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 55) = a (.wire 28 0) ∧ a (.virt 18978) = a (.wire 28 1) ∧ a (.wire 11 15) = a (.wire 28 2) ∧ a (.virt 19075) = a (.wire 27 32) ∧ a (.wire 28 3) = a (.wire 27 33) ∧ a (.virt 19075) = a (.wire 27 34) ∧ a (.wire 28 3) = a (.wire 27 36) ∧ a (.virt 19076) = a (.wire 27 37) ∧ a (.wire 28 3) = a (.wire 27 38) ∧ a (.wire 27 39) = a (.wire 28 4) ∧ a (.virt 18978) = a (.wire 28 5) ∧ a (.wire 26 59) = a (.wire 28 6) ∧ a (.wire 27 35) = a (.virt 18979) ∧ a (.wire 28 7) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 28 8) ∧ a (.virt 18978) = a (.wire 28 9) ∧ a (.virt 19077) = a (.wire 28 10) ∧ a (.wire 12 3) = a (.wire 28 12) ∧ a (.virt 18978) = a (.wire 28 13) ∧ a (.wire 11 23) = a (.wire 28 14) ∧ a (.virt 19077) = a (.wire 27 40) ∧ a (.wire 28 15) = a (.wire 27 41) ∧ a (.virt 19077) = a (.wire 27 42) ∧ a (.wire 28 15) = a (.wire 27 44) ∧ a (.virt 19078) = a (.wire 27 45) ∧ a (.wire 28 15) = a (.wire 27 46) ∧ a (.wire 27 47) = a (.wire 28 16) ∧ a (.virt 18978) = a (.wire 28 17) ∧ a (.wire 28 11) = a (.wire 28 18) ∧ a (.wire 27 43) = a (.virt 18979) ∧ a (.wire 28 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 28 20) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies39, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies39, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies40 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 28 21) ∧ a (.virt 19079) = a (.wire 28 22) ∧ a (.wire 12 11) = a (.wire 28 24) ∧ a (.virt 18978) = a (.wire 28 25) ∧ a (.wire 11 31) = a (.wire 28 26) ∧ a (.virt 19079) = a (.wire 27 48) ∧ a (.wire 28 27) = a (.wire 27 49) ∧ a (.virt 19079) = a (.wire 27 50) ∧ a (.wire 28 27) = a (.wire 27 52) ∧ a (.virt 19080) = a (.wire 27 53) ∧ a (.wire 28 27) = a (.wire 27 54) ∧ a (.wire 27 55) = a (.wire 28 28) ∧ a (.virt 18978) = a (.wire 28 29) ∧ a (.wire 28 23) = a (.wire 28 30) ∧ a (.wire 27 51) = a (.virt 18979) ∧ a (.wire 28 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 28 32) ∧ a (.virt 18978) = a (.wire 28 33) ∧ a (.virt 19081) = a (.wire 28 34) ∧ a (.wire 12 19) = a (.wire 28 36) ∧ a (.virt 18978) = a (.wire 28 37) ∧ a (.wire 11 39) = a (.wire 28 38) ∧ a (.virt 19081) = a (.wire 27 56) ∧ a (.wire 28 39) = a (.wire 27 57) ∧ a (.virt 19081) = a (.wire 27 58) ∧ a (.wire 28 39) = a (.wire 29 0) ∧ a (.virt 19082) = a (.wire 29 1) ∧ a (.wire 28 39) = a (.wire 29 2) ∧ a (.wire 29 3) = a (.wire 28 40) ∧ a (.virt 18978) = a (.wire 28 41) ∧ a (.wire 28 35) = a (.wire 28 42) ∧ a (.wire 27 59) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies40, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies40, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies41 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 28 43) = a (.virt 18979) ∧ a (.virt 19075) = a (.wire 29 4) ∧ a (.virt 19077) = a (.wire 29 5) ∧ a (.virt 19075) = a (.wire 29 6) ∧ a (.virt 19079) = a (.wire 29 8) ∧ a (.virt 19081) = a (.wire 29 9) ∧ a (.virt 19079) = a (.wire 29 10) ∧ a (.wire 29 7) = a (.wire 29 12) ∧ a (.wire 29 11) = a (.wire 29 13) ∧ a (.wire 29 7) = a (.wire 29 14) ∧ a (.wire 29 15) = a (.wire 28 44) ∧ a (.wire 12 27) = a (.wire 28 45) ∧ a (.virt 18979) = a (.wire 28 46) ∧ a (.wire 6 51) = a (.wire 6 52) ∧ a (.virt 18978) = a (.wire 6 53) ∧ a (.wire 28 47) = a (.wire 6 54) ∧ a (.virt 18978) = a (.wire 28 48) ∧ a (.virt 18978) = a (.wire 28 49) ∧ a (.virt 19083) = a (.wire 28 50) ∧ a (.wire 12 35) = a (.wire 28 52) ∧ a (.virt 18978) = a (.wire 28 53) ∧ a (.wire 11 15) = a (.wire 28 54) ∧ a (.virt 19083) = a (.wire 29 16) ∧ a (.wire 28 55) = a (.wire 29 17) ∧ a (.virt 19083) = a (.wire 29 18) ∧ a (.wire 28 55) = a (.wire 29 20) ∧ a (.virt 19084) = a (.wire 29 21) ∧ a (.wire 28 55) = a (.wire 29 22) ∧ a (.wire 29 23) = a (.wire 28 56) ∧ a (.virt 18978) = a (.wire 28 57) ∧ a (.wire 28 51) = a (.wire 28 58) ∧ a (.wire 29 19) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies41, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies41, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies42 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 28 59) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 30 0) ∧ a (.virt 18978) = a (.wire 30 1) ∧ a (.virt 19085) = a (.wire 30 2) ∧ a (.wire 12 43) = a (.wire 30 4) ∧ a (.virt 18978) = a (.wire 30 5) ∧ a (.wire 11 23) = a (.wire 30 6) ∧ a (.virt 19085) = a (.wire 29 24) ∧ a (.wire 30 7) = a (.wire 29 25) ∧ a (.virt 19085) = a (.wire 29 26) ∧ a (.wire 30 7) = a (.wire 29 28) ∧ a (.virt 19086) = a (.wire 29 29) ∧ a (.wire 30 7) = a (.wire 29 30) ∧ a (.wire 29 31) = a (.wire 30 8) ∧ a (.virt 18978) = a (.wire 30 9) ∧ a (.wire 30 3) = a (.wire 30 10) ∧ a (.wire 29 27) = a (.virt 18979) ∧ a (.wire 30 11) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 30 12) ∧ a (.virt 18978) = a (.wire 30 13) ∧ a (.virt 19087) = a (.wire 30 14) ∧ a (.wire 12 51) = a (.wire 30 16) ∧ a (.virt 18978) = a (.wire 30 17) ∧ a (.wire 11 31) = a (.wire 30 18) ∧ a (.virt 19087) = a (.wire 29 32) ∧ a (.wire 30 19) = a (.wire 29 33) ∧ a (.virt 19087) = a (.wire 29 34) ∧ a (.wire 30 19) = a (.wire 29 36) ∧ a (.virt 19088) = a (.wire 29 37) ∧ a (.wire 30 19) = a (.wire 29 38) ∧ a (.wire 29 39) = a (.wire 30 20) ∧ a (.virt 18978) = a (.wire 30 21) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies42, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies42, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies43 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 30 15) = a (.wire 30 22) ∧ a (.wire 29 35) = a (.virt 18979) ∧ a (.wire 30 23) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 30 24) ∧ a (.virt 18978) = a (.wire 30 25) ∧ a (.virt 19089) = a (.wire 30 26) ∧ a (.wire 12 59) = a (.wire 30 28) ∧ a (.virt 18978) = a (.wire 30 29) ∧ a (.wire 11 39) = a (.wire 30 30) ∧ a (.virt 19089) = a (.wire 29 40) ∧ a (.wire 30 31) = a (.wire 29 41) ∧ a (.virt 19089) = a (.wire 29 42) ∧ a (.wire 30 31) = a (.wire 29 44) ∧ a (.virt 19090) = a (.wire 29 45) ∧ a (.wire 30 31) = a (.wire 29 46) ∧ a (.wire 29 47) = a (.wire 30 32) ∧ a (.virt 18978) = a (.wire 30 33) ∧ a (.wire 30 27) = a (.wire 30 34) ∧ a (.wire 29 43) = a (.virt 18979) ∧ a (.wire 30 35) = a (.virt 18979) ∧ a (.virt 19083) = a (.wire 29 48) ∧ a (.virt 19085) = a (.wire 29 49) ∧ a (.virt 19083) = a (.wire 29 50) ∧ a (.virt 19087) = a (.wire 29 52) ∧ a (.virt 19089) = a (.wire 29 53) ∧ a (.virt 19087) = a (.wire 29 54) ∧ a (.wire 29 51) = a (.wire 29 56) ∧ a (.wire 29 55) = a (.wire 29 57) ∧ a (.wire 29 51) = a (.wire 29 58) ∧ a (.wire 29 59) = a (.wire 30 36) ∧ a (.wire 13 7) = a (.wire 30 37) ∧ a (.virt 18979) = a (.wire 30 38) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies43, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies43, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies44 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 55) = a (.wire 6 56) ∧ a (.virt 18978) = a (.wire 6 57) ∧ a (.wire 30 39) = a (.wire 6 58) ∧ a (.wire 25 3) = a (.wire 30 40) ∧ a (.wire 6 59) = a (.wire 30 41) ∧ a (.wire 6 59) = a (.wire 30 42) ∧ a (.wire 25 3) = a (.wire 30 44) ∧ a (.virt 18979) = a (.wire 30 45) ∧ a (.wire 30 43) = a (.wire 30 46) ∧ a (.wire 25 3) = a (.wire 30 48) ∧ a (.wire 11 15) = a (.wire 30 49) ∧ a (.wire 11 15) = a (.wire 30 50) ∧ a (.wire 25 3) = a (.wire 30 52) ∧ a (.virt 18979) = a (.wire 30 53) ∧ a (.wire 30 51) = a (.wire 30 54) ∧ a (.wire 25 3) = a (.wire 30 56) ∧ a (.wire 11 23) = a (.wire 30 57) ∧ a (.wire 11 23) = a (.wire 30 58) ∧ a (.wire 25 3) = a (.wire 31 0) ∧ a (.virt 18979) = a (.wire 31 1) ∧ a (.wire 30 59) = a (.wire 31 2) ∧ a (.wire 25 3) = a (.wire 31 4) ∧ a (.wire 11 31) = a (.wire 31 5) ∧ a (.wire 11 31) = a (.wire 31 6) ∧ a (.wire 25 3) = a (.wire 31 8) ∧ a (.virt 18979) = a (.wire 31 9) ∧ a (.wire 31 7) = a (.wire 31 10) ∧ a (.wire 25 3) = a (.wire 31 12) ∧ a (.wire 11 39) = a (.wire 31 13) ∧ a (.wire 11 39) = a (.wire 31 14) ∧ a (.wire 25 3) = a (.wire 31 16) ∧ a (.virt 18979) = a (.wire 31 17) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies44, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies44, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies45 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 31 15) = a (.wire 31 18) ∧ a (.wire 32 33) = a (.virt 18979) ∧ a (.wire 32 34) = a (.virt 18979) ∧ a (.wire 32 35) = a (.virt 18979) ∧ a (.wire 32 36) = a (.virt 18979) ∧ a (.wire 32 37) = a (.virt 18979) ∧ a (.wire 32 38) = a (.virt 18979) ∧ a (.wire 32 39) = a (.virt 18979) ∧ a (.wire 32 40) = a (.virt 18979) ∧ a (.wire 32 41) = a (.virt 18979) ∧ a (.wire 32 42) = a (.virt 18979) ∧ a (.wire 32 43) = a (.virt 18979) ∧ a (.wire 32 44) = a (.virt 18979) ∧ a (.wire 32 45) = a (.virt 18979) ∧ a (.wire 32 46) = a (.virt 18979) ∧ a (.wire 32 47) = a (.virt 18979) ∧ a (.wire 32 48) = a (.virt 18979) ∧ a (.wire 32 49) = a (.virt 18979) ∧ a (.wire 32 50) = a (.virt 18979) ∧ a (.wire 32 51) = a (.virt 18979) ∧ a (.wire 32 52) = a (.virt 18979) ∧ a (.wire 32 53) = a (.virt 18979) ∧ a (.wire 32 54) = a (.virt 18979) ∧ a (.wire 32 55) = a (.virt 18979) ∧ a (.wire 32 56) = a (.virt 18979) ∧ a (.wire 32 57) = a (.virt 18979) ∧ a (.wire 32 58) = a (.virt 18979) ∧ a (.wire 32 59) = a (.virt 18979) ∧ a (.wire 32 0) = a (.wire 30 47) ∧ a (.virt 18978) = a (.wire 31 20) ∧ a (.virt 18978) = a (.wire 31 21) ∧ a (.virt 19091) = a (.wire 31 22) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies45, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies45, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies46 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 35) = a (.wire 31 24) ∧ a (.virt 18978) = a (.wire 31 25) ∧ a (.wire 11 55) = a (.wire 31 26) ∧ a (.virt 19091) = a (.wire 33 0) ∧ a (.wire 31 27) = a (.wire 33 1) ∧ a (.virt 19091) = a (.wire 33 2) ∧ a (.wire 31 27) = a (.wire 33 4) ∧ a (.virt 19092) = a (.wire 33 5) ∧ a (.wire 31 27) = a (.wire 33 6) ∧ a (.wire 33 7) = a (.wire 31 28) ∧ a (.virt 18978) = a (.wire 31 29) ∧ a (.wire 31 23) = a (.wire 31 30) ∧ a (.wire 33 3) = a (.virt 18979) ∧ a (.wire 31 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 31 32) ∧ a (.virt 18978) = a (.wire 31 33) ∧ a (.virt 19093) = a (.wire 31 34) ∧ a (.wire 9 43) = a (.wire 31 36) ∧ a (.virt 18978) = a (.wire 31 37) ∧ a (.wire 12 3) = a (.wire 31 38) ∧ a (.virt 19093) = a (.wire 33 8) ∧ a (.wire 31 39) = a (.wire 33 9) ∧ a (.virt 19093) = a (.wire 33 10) ∧ a (.wire 31 39) = a (.wire 33 12) ∧ a (.virt 19094) = a (.wire 33 13) ∧ a (.wire 31 39) = a (.wire 33 14) ∧ a (.wire 33 15) = a (.wire 31 40) ∧ a (.virt 18978) = a (.wire 31 41) ∧ a (.wire 31 35) = a (.wire 31 42) ∧ a (.wire 33 11) = a (.virt 18979) ∧ a (.wire 31 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 31 44) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies46, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies46, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies47 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 31 45) ∧ a (.virt 19095) = a (.wire 31 46) ∧ a (.wire 9 51) = a (.wire 31 48) ∧ a (.virt 18978) = a (.wire 31 49) ∧ a (.wire 12 11) = a (.wire 31 50) ∧ a (.virt 19095) = a (.wire 33 16) ∧ a (.wire 31 51) = a (.wire 33 17) ∧ a (.virt 19095) = a (.wire 33 18) ∧ a (.wire 31 51) = a (.wire 33 20) ∧ a (.virt 19096) = a (.wire 33 21) ∧ a (.wire 31 51) = a (.wire 33 22) ∧ a (.wire 33 23) = a (.wire 31 52) ∧ a (.virt 18978) = a (.wire 31 53) ∧ a (.wire 31 47) = a (.wire 31 54) ∧ a (.wire 33 19) = a (.virt 18979) ∧ a (.wire 31 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 31 56) ∧ a (.virt 18978) = a (.wire 31 57) ∧ a (.virt 19097) = a (.wire 31 58) ∧ a (.wire 9 59) = a (.wire 34 0) ∧ a (.virt 18978) = a (.wire 34 1) ∧ a (.wire 12 19) = a (.wire 34 2) ∧ a (.virt 19097) = a (.wire 33 24) ∧ a (.wire 34 3) = a (.wire 33 25) ∧ a (.virt 19097) = a (.wire 33 26) ∧ a (.wire 34 3) = a (.wire 33 28) ∧ a (.virt 19098) = a (.wire 33 29) ∧ a (.wire 34 3) = a (.wire 33 30) ∧ a (.wire 33 31) = a (.wire 34 4) ∧ a (.virt 18978) = a (.wire 34 5) ∧ a (.wire 31 59) = a (.wire 34 6) ∧ a (.wire 33 27) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies47, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies47, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies48 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 34 7) = a (.virt 18979) ∧ a (.virt 19091) = a (.wire 33 32) ∧ a (.virt 19093) = a (.wire 33 33) ∧ a (.virt 19091) = a (.wire 33 34) ∧ a (.virt 19095) = a (.wire 33 36) ∧ a (.virt 19097) = a (.wire 33 37) ∧ a (.virt 19095) = a (.wire 33 38) ∧ a (.wire 33 35) = a (.wire 33 40) ∧ a (.wire 33 39) = a (.wire 33 41) ∧ a (.wire 33 35) = a (.wire 33 42) ∧ a (.virt 18978) = a (.wire 34 8) ∧ a (.virt 18978) = a (.wire 34 9) ∧ a (.virt 19099) = a (.wire 34 10) ∧ a (.wire 11 15) = a (.wire 34 12) ∧ a (.virt 18978) = a (.wire 34 13) ∧ a (.wire 11 55) = a (.wire 34 14) ∧ a (.virt 19099) = a (.wire 33 44) ∧ a (.wire 34 15) = a (.wire 33 45) ∧ a (.virt 19099) = a (.wire 33 46) ∧ a (.wire 34 15) = a (.wire 33 48) ∧ a (.virt 19100) = a (.wire 33 49) ∧ a (.wire 34 15) = a (.wire 33 50) ∧ a (.wire 33 51) = a (.wire 34 16) ∧ a (.virt 18978) = a (.wire 34 17) ∧ a (.wire 34 11) = a (.wire 34 18) ∧ a (.wire 33 47) = a (.virt 18979) ∧ a (.wire 34 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 34 20) ∧ a (.virt 18978) = a (.wire 34 21) ∧ a (.virt 19101) = a (.wire 34 22) ∧ a (.wire 11 23) = a (.wire 34 24) ∧ a (.virt 18978) = a (.wire 34 25) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies48, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies48, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies49 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 3) = a (.wire 34 26) ∧ a (.virt 19101) = a (.wire 33 52) ∧ a (.wire 34 27) = a (.wire 33 53) ∧ a (.virt 19101) = a (.wire 33 54) ∧ a (.wire 34 27) = a (.wire 33 56) ∧ a (.virt 19102) = a (.wire 33 57) ∧ a (.wire 34 27) = a (.wire 33 58) ∧ a (.wire 33 59) = a (.wire 34 28) ∧ a (.virt 18978) = a (.wire 34 29) ∧ a (.wire 34 23) = a (.wire 34 30) ∧ a (.wire 33 55) = a (.virt 18979) ∧ a (.wire 34 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 34 32) ∧ a (.virt 18978) = a (.wire 34 33) ∧ a (.virt 19103) = a (.wire 34 34) ∧ a (.wire 11 31) = a (.wire 34 36) ∧ a (.virt 18978) = a (.wire 34 37) ∧ a (.wire 12 11) = a (.wire 34 38) ∧ a (.virt 19103) = a (.wire 35 0) ∧ a (.wire 34 39) = a (.wire 35 1) ∧ a (.virt 19103) = a (.wire 35 2) ∧ a (.wire 34 39) = a (.wire 35 4) ∧ a (.virt 19104) = a (.wire 35 5) ∧ a (.wire 34 39) = a (.wire 35 6) ∧ a (.wire 35 7) = a (.wire 34 40) ∧ a (.virt 18978) = a (.wire 34 41) ∧ a (.wire 34 35) = a (.wire 34 42) ∧ a (.wire 35 3) = a (.virt 18979) ∧ a (.wire 34 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 34 44) ∧ a (.virt 18978) = a (.wire 34 45) ∧ a (.virt 19105) = a (.wire 34 46) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies49, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies49, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies50 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 39) = a (.wire 34 48) ∧ a (.virt 18978) = a (.wire 34 49) ∧ a (.wire 12 19) = a (.wire 34 50) ∧ a (.virt 19105) = a (.wire 35 8) ∧ a (.wire 34 51) = a (.wire 35 9) ∧ a (.virt 19105) = a (.wire 35 10) ∧ a (.wire 34 51) = a (.wire 35 12) ∧ a (.virt 19106) = a (.wire 35 13) ∧ a (.wire 34 51) = a (.wire 35 14) ∧ a (.wire 35 15) = a (.wire 34 52) ∧ a (.virt 18978) = a (.wire 34 53) ∧ a (.wire 34 47) = a (.wire 34 54) ∧ a (.wire 35 11) = a (.virt 18979) ∧ a (.wire 34 55) = a (.virt 18979) ∧ a (.virt 19099) = a (.wire 35 16) ∧ a (.virt 19101) = a (.wire 35 17) ∧ a (.virt 19099) = a (.wire 35 18) ∧ a (.virt 19103) = a (.wire 35 20) ∧ a (.virt 19105) = a (.wire 35 21) ∧ a (.virt 19103) = a (.wire 35 22) ∧ a (.wire 35 19) = a (.wire 35 24) ∧ a (.wire 35 23) = a (.wire 35 25) ∧ a (.wire 35 19) = a (.wire 35 26) ∧ a (.wire 33 43) = a (.wire 5 20) ∧ a (.wire 35 27) = a (.wire 5 21) ∧ a (.wire 33 43) = a (.wire 5 22) ∧ a (.wire 5 23) = a (.wire 36 0) ∧ a (.virt 18978) = a (.wire 36 1) ∧ a (.wire 35 27) = a (.wire 36 2) ∧ a (.virt 18978) = a (.wire 34 56) ∧ a (.virt 18978) = a (.wire 34 57) ∧ a (.virt 19107) = a (.wire 34 58) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies50, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies50, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies51 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19107) = a (.wire 35 28) ∧ a (.wire 31 27) = a (.wire 35 29) ∧ a (.virt 19107) = a (.wire 35 30) ∧ a (.wire 31 27) = a (.wire 35 32) ∧ a (.virt 19108) = a (.wire 35 33) ∧ a (.wire 31 27) = a (.wire 35 34) ∧ a (.wire 35 35) = a (.wire 37 0) ∧ a (.virt 18978) = a (.wire 37 1) ∧ a (.wire 34 59) = a (.wire 37 2) ∧ a (.wire 35 31) = a (.virt 18979) ∧ a (.wire 37 3) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 37 4) ∧ a (.virt 18978) = a (.wire 37 5) ∧ a (.virt 19109) = a (.wire 37 6) ∧ a (.virt 19109) = a (.wire 35 36) ∧ a (.wire 31 39) = a (.wire 35 37) ∧ a (.virt 19109) = a (.wire 35 38) ∧ a (.wire 31 39) = a (.wire 35 40) ∧ a (.virt 19110) = a (.wire 35 41) ∧ a (.wire 31 39) = a (.wire 35 42) ∧ a (.wire 35 43) = a (.wire 37 8) ∧ a (.virt 18978) = a (.wire 37 9) ∧ a (.wire 37 7) = a (.wire 37 10) ∧ a (.wire 35 39) = a (.virt 18979) ∧ a (.wire 37 11) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 37 12) ∧ a (.virt 18978) = a (.wire 37 13) ∧ a (.virt 19111) = a (.wire 37 14) ∧ a (.virt 19111) = a (.wire 35 44) ∧ a (.wire 31 51) = a (.wire 35 45) ∧ a (.virt 19111) = a (.wire 35 46) ∧ a (.wire 31 51) = a (.wire 35 48) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies51, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies51, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies52 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19112) = a (.wire 35 49) ∧ a (.wire 31 51) = a (.wire 35 50) ∧ a (.wire 35 51) = a (.wire 37 16) ∧ a (.virt 18978) = a (.wire 37 17) ∧ a (.wire 37 15) = a (.wire 37 18) ∧ a (.wire 35 47) = a (.virt 18979) ∧ a (.wire 37 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 37 20) ∧ a (.virt 18978) = a (.wire 37 21) ∧ a (.virt 19113) = a (.wire 37 22) ∧ a (.virt 19113) = a (.wire 35 52) ∧ a (.wire 34 3) = a (.wire 35 53) ∧ a (.virt 19113) = a (.wire 35 54) ∧ a (.wire 34 3) = a (.wire 35 56) ∧ a (.virt 19114) = a (.wire 35 57) ∧ a (.wire 34 3) = a (.wire 35 58) ∧ a (.wire 35 59) = a (.wire 37 24) ∧ a (.virt 18978) = a (.wire 37 25) ∧ a (.wire 37 23) = a (.wire 37 26) ∧ a (.wire 35 55) = a (.virt 18979) ∧ a (.wire 37 27) = a (.virt 18979) ∧ a (.virt 19107) = a (.wire 38 0) ∧ a (.virt 19109) = a (.wire 38 1) ∧ a (.virt 19107) = a (.wire 38 2) ∧ a (.virt 19111) = a (.wire 38 4) ∧ a (.virt 19113) = a (.wire 38 5) ∧ a (.virt 19111) = a (.wire 38 6) ∧ a (.wire 38 3) = a (.wire 38 8) ∧ a (.wire 38 7) = a (.wire 38 9) ∧ a (.wire 38 3) = a (.wire 38 10) ∧ a (.wire 38 11) = a (.wire 37 28) ∧ a (.wire 11 7) = a (.wire 37 29) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies52, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies52, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies53 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18979) = a (.wire 37 30) ∧ a (.virt 18978) = a (.wire 37 32) ∧ a (.virt 18978) = a (.wire 37 33) ∧ a (.virt 19115) = a (.wire 37 34) ∧ a (.virt 19115) = a (.wire 38 12) ∧ a (.wire 34 15) = a (.wire 38 13) ∧ a (.virt 19115) = a (.wire 38 14) ∧ a (.wire 34 15) = a (.wire 38 16) ∧ a (.virt 19116) = a (.wire 38 17) ∧ a (.wire 34 15) = a (.wire 38 18) ∧ a (.wire 38 19) = a (.wire 37 36) ∧ a (.virt 18978) = a (.wire 37 37) ∧ a (.wire 37 35) = a (.wire 37 38) ∧ a (.wire 38 15) = a (.virt 18979) ∧ a (.wire 37 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 37 40) ∧ a (.virt 18978) = a (.wire 37 41) ∧ a (.virt 19117) = a (.wire 37 42) ∧ a (.virt 19117) = a (.wire 38 20) ∧ a (.wire 34 27) = a (.wire 38 21) ∧ a (.virt 19117) = a (.wire 38 22) ∧ a (.wire 34 27) = a (.wire 38 24) ∧ a (.virt 19118) = a (.wire 38 25) ∧ a (.wire 34 27) = a (.wire 38 26) ∧ a (.wire 38 27) = a (.wire 37 44) ∧ a (.virt 18978) = a (.wire 37 45) ∧ a (.wire 37 43) = a (.wire 37 46) ∧ a (.wire 38 23) = a (.virt 18979) ∧ a (.wire 37 47) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 37 48) ∧ a (.virt 18978) = a (.wire 37 49) ∧ a (.virt 19119) = a (.wire 37 50) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies53, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies53, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies54 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19119) = a (.wire 38 28) ∧ a (.wire 34 39) = a (.wire 38 29) ∧ a (.virt 19119) = a (.wire 38 30) ∧ a (.wire 34 39) = a (.wire 38 32) ∧ a (.virt 19120) = a (.wire 38 33) ∧ a (.wire 34 39) = a (.wire 38 34) ∧ a (.wire 38 35) = a (.wire 37 52) ∧ a (.virt 18978) = a (.wire 37 53) ∧ a (.wire 37 51) = a (.wire 37 54) ∧ a (.wire 38 31) = a (.virt 18979) ∧ a (.wire 37 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 37 56) ∧ a (.virt 18978) = a (.wire 37 57) ∧ a (.virt 19121) = a (.wire 37 58) ∧ a (.virt 19121) = a (.wire 38 36) ∧ a (.wire 34 51) = a (.wire 38 37) ∧ a (.virt 19121) = a (.wire 38 38) ∧ a (.wire 34 51) = a (.wire 38 40) ∧ a (.virt 19122) = a (.wire 38 41) ∧ a (.wire 34 51) = a (.wire 38 42) ∧ a (.wire 38 43) = a (.wire 39 0) ∧ a (.virt 18978) = a (.wire 39 1) ∧ a (.wire 37 59) = a (.wire 39 2) ∧ a (.wire 38 39) = a (.virt 18979) ∧ a (.wire 39 3) = a (.virt 18979) ∧ a (.virt 19115) = a (.wire 38 44) ∧ a (.virt 19117) = a (.wire 38 45) ∧ a (.virt 19115) = a (.wire 38 46) ∧ a (.virt 19119) = a (.wire 38 48) ∧ a (.virt 19121) = a (.wire 38 49) ∧ a (.virt 19119) = a (.wire 38 50) ∧ a (.wire 38 47) = a (.wire 38 52) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies54, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies54, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies55 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 51) = a (.wire 38 53) ∧ a (.wire 38 47) = a (.wire 38 54) ∧ a (.wire 38 55) = a (.wire 39 4) ∧ a (.wire 11 47) = a (.wire 39 5) ∧ a (.virt 18979) = a (.wire 39 6) ∧ a (.wire 37 31) = a (.wire 36 4) ∧ a (.virt 18978) = a (.wire 36 5) ∧ a (.wire 39 7) = a (.wire 36 6) ∧ a (.virt 18978) = a (.wire 39 8) ∧ a (.virt 18978) = a (.wire 39 9) ∧ a (.virt 19123) = a (.wire 39 10) ∧ a (.wire 11 55) = a (.wire 39 12) ∧ a (.virt 18978) = a (.wire 39 13) ∧ a (.wire 11 55) = a (.wire 39 14) ∧ a (.virt 19123) = a (.wire 38 56) ∧ a (.wire 39 15) = a (.wire 38 57) ∧ a (.virt 19123) = a (.wire 38 58) ∧ a (.wire 39 15) = a (.wire 40 0) ∧ a (.virt 19124) = a (.wire 40 1) ∧ a (.wire 39 15) = a (.wire 40 2) ∧ a (.wire 40 3) = a (.wire 39 16) ∧ a (.virt 18978) = a (.wire 39 17) ∧ a (.wire 39 11) = a (.wire 39 18) ∧ a (.wire 38 59) = a (.virt 18979) ∧ a (.wire 39 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 39 20) ∧ a (.virt 18978) = a (.wire 39 21) ∧ a (.virt 19125) = a (.wire 39 22) ∧ a (.wire 12 3) = a (.wire 39 24) ∧ a (.virt 18978) = a (.wire 39 25) ∧ a (.wire 12 3) = a (.wire 39 26) ∧ a (.virt 19125) = a (.wire 40 4) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies55, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies55, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies56 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 39 27) = a (.wire 40 5) ∧ a (.virt 19125) = a (.wire 40 6) ∧ a (.wire 39 27) = a (.wire 40 8) ∧ a (.virt 19126) = a (.wire 40 9) ∧ a (.wire 39 27) = a (.wire 40 10) ∧ a (.wire 40 11) = a (.wire 39 28) ∧ a (.virt 18978) = a (.wire 39 29) ∧ a (.wire 39 23) = a (.wire 39 30) ∧ a (.wire 40 7) = a (.virt 18979) ∧ a (.wire 39 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 39 32) ∧ a (.virt 18978) = a (.wire 39 33) ∧ a (.virt 19127) = a (.wire 39 34) ∧ a (.wire 12 11) = a (.wire 39 36) ∧ a (.virt 18978) = a (.wire 39 37) ∧ a (.wire 12 11) = a (.wire 39 38) ∧ a (.virt 19127) = a (.wire 40 12) ∧ a (.wire 39 39) = a (.wire 40 13) ∧ a (.virt 19127) = a (.wire 40 14) ∧ a (.wire 39 39) = a (.wire 40 16) ∧ a (.virt 19128) = a (.wire 40 17) ∧ a (.wire 39 39) = a (.wire 40 18) ∧ a (.wire 40 19) = a (.wire 39 40) ∧ a (.virt 18978) = a (.wire 39 41) ∧ a (.wire 39 35) = a (.wire 39 42) ∧ a (.wire 40 15) = a (.virt 18979) ∧ a (.wire 39 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 39 44) ∧ a (.virt 18978) = a (.wire 39 45) ∧ a (.virt 19129) = a (.wire 39 46) ∧ a (.wire 12 19) = a (.wire 39 48) ∧ a (.virt 18978) = a (.wire 39 49) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies56, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies56, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies57 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 19) = a (.wire 39 50) ∧ a (.virt 19129) = a (.wire 40 20) ∧ a (.wire 39 51) = a (.wire 40 21) ∧ a (.virt 19129) = a (.wire 40 22) ∧ a (.wire 39 51) = a (.wire 40 24) ∧ a (.virt 19130) = a (.wire 40 25) ∧ a (.wire 39 51) = a (.wire 40 26) ∧ a (.wire 40 27) = a (.wire 39 52) ∧ a (.virt 18978) = a (.wire 39 53) ∧ a (.wire 39 47) = a (.wire 39 54) ∧ a (.wire 40 23) = a (.virt 18979) ∧ a (.wire 39 55) = a (.virt 18979) ∧ a (.virt 19123) = a (.wire 40 28) ∧ a (.virt 19125) = a (.wire 40 29) ∧ a (.virt 19123) = a (.wire 40 30) ∧ a (.virt 19127) = a (.wire 40 32) ∧ a (.virt 19129) = a (.wire 40 33) ∧ a (.virt 19127) = a (.wire 40 34) ∧ a (.wire 40 31) = a (.wire 40 36) ∧ a (.wire 40 35) = a (.wire 40 37) ∧ a (.wire 40 31) = a (.wire 40 38) ∧ a (.wire 40 39) = a (.wire 39 56) ∧ a (.wire 12 27) = a (.wire 39 57) ∧ a (.virt 18979) = a (.wire 39 58) ∧ a (.wire 36 7) = a (.wire 36 8) ∧ a (.virt 18978) = a (.wire 36 9) ∧ a (.wire 39 59) = a (.wire 36 10) ∧ a (.virt 18978) = a (.wire 41 0) ∧ a (.virt 18978) = a (.wire 41 1) ∧ a (.virt 19131) = a (.wire 41 2) ∧ a (.wire 12 35) = a (.wire 41 4) ∧ a (.virt 18978) = a (.wire 41 5) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies57, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies57, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies58 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 55) = a (.wire 41 6) ∧ a (.virt 19131) = a (.wire 40 40) ∧ a (.wire 41 7) = a (.wire 40 41) ∧ a (.virt 19131) = a (.wire 40 42) ∧ a (.wire 41 7) = a (.wire 40 44) ∧ a (.virt 19132) = a (.wire 40 45) ∧ a (.wire 41 7) = a (.wire 40 46) ∧ a (.wire 40 47) = a (.wire 41 8) ∧ a (.virt 18978) = a (.wire 41 9) ∧ a (.wire 41 3) = a (.wire 41 10) ∧ a (.wire 40 43) = a (.virt 18979) ∧ a (.wire 41 11) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 41 12) ∧ a (.virt 18978) = a (.wire 41 13) ∧ a (.virt 19133) = a (.wire 41 14) ∧ a (.wire 12 43) = a (.wire 41 16) ∧ a (.virt 18978) = a (.wire 41 17) ∧ a (.wire 12 3) = a (.wire 41 18) ∧ a (.virt 19133) = a (.wire 40 48) ∧ a (.wire 41 19) = a (.wire 40 49) ∧ a (.virt 19133) = a (.wire 40 50) ∧ a (.wire 41 19) = a (.wire 40 52) ∧ a (.virt 19134) = a (.wire 40 53) ∧ a (.wire 41 19) = a (.wire 40 54) ∧ a (.wire 40 55) = a (.wire 41 20) ∧ a (.virt 18978) = a (.wire 41 21) ∧ a (.wire 41 15) = a (.wire 41 22) ∧ a (.wire 40 51) = a (.virt 18979) ∧ a (.wire 41 23) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 41 24) ∧ a (.virt 18978) = a (.wire 41 25) ∧ a (.virt 19135) = a (.wire 41 26) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies58, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies58, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies59 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 51) = a (.wire 41 28) ∧ a (.virt 18978) = a (.wire 41 29) ∧ a (.wire 12 11) = a (.wire 41 30) ∧ a (.virt 19135) = a (.wire 40 56) ∧ a (.wire 41 31) = a (.wire 40 57) ∧ a (.virt 19135) = a (.wire 40 58) ∧ a (.wire 41 31) = a (.wire 42 0) ∧ a (.virt 19136) = a (.wire 42 1) ∧ a (.wire 41 31) = a (.wire 42 2) ∧ a (.wire 42 3) = a (.wire 41 32) ∧ a (.virt 18978) = a (.wire 41 33) ∧ a (.wire 41 27) = a (.wire 41 34) ∧ a (.wire 40 59) = a (.virt 18979) ∧ a (.wire 41 35) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 41 36) ∧ a (.virt 18978) = a (.wire 41 37) ∧ a (.virt 19137) = a (.wire 41 38) ∧ a (.wire 12 59) = a (.wire 41 40) ∧ a (.virt 18978) = a (.wire 41 41) ∧ a (.wire 12 19) = a (.wire 41 42) ∧ a (.virt 19137) = a (.wire 42 4) ∧ a (.wire 41 43) = a (.wire 42 5) ∧ a (.virt 19137) = a (.wire 42 6) ∧ a (.wire 41 43) = a (.wire 42 8) ∧ a (.virt 19138) = a (.wire 42 9) ∧ a (.wire 41 43) = a (.wire 42 10) ∧ a (.wire 42 11) = a (.wire 41 44) ∧ a (.virt 18978) = a (.wire 41 45) ∧ a (.wire 41 39) = a (.wire 41 46) ∧ a (.wire 42 7) = a (.virt 18979) ∧ a (.wire 41 47) = a (.virt 18979) ∧ a (.virt 19131) = a (.wire 42 12) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies59, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies59, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies60 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19133) = a (.wire 42 13) ∧ a (.virt 19131) = a (.wire 42 14) ∧ a (.virt 19135) = a (.wire 42 16) ∧ a (.virt 19137) = a (.wire 42 17) ∧ a (.virt 19135) = a (.wire 42 18) ∧ a (.wire 42 15) = a (.wire 42 20) ∧ a (.wire 42 19) = a (.wire 42 21) ∧ a (.wire 42 15) = a (.wire 42 22) ∧ a (.wire 42 23) = a (.wire 41 48) ∧ a (.wire 13 7) = a (.wire 41 49) ∧ a (.virt 18979) = a (.wire 41 50) ∧ a (.wire 36 11) = a (.wire 36 12) ∧ a (.virt 18978) = a (.wire 36 13) ∧ a (.wire 41 51) = a (.wire 36 14) ∧ a (.wire 36 3) = a (.wire 41 52) ∧ a (.wire 36 15) = a (.wire 41 53) ∧ a (.wire 36 15) = a (.wire 41 54) ∧ a (.wire 36 3) = a (.wire 41 56) ∧ a (.virt 18979) = a (.wire 41 57) ∧ a (.wire 41 55) = a (.wire 41 58) ∧ a (.wire 36 3) = a (.wire 43 0) ∧ a (.wire 11 55) = a (.wire 43 1) ∧ a (.wire 11 55) = a (.wire 43 2) ∧ a (.wire 36 3) = a (.wire 43 4) ∧ a (.virt 18979) = a (.wire 43 5) ∧ a (.wire 43 3) = a (.wire 43 6) ∧ a (.wire 36 3) = a (.wire 43 8) ∧ a (.wire 12 3) = a (.wire 43 9) ∧ a (.wire 12 3) = a (.wire 43 10) ∧ a (.wire 36 3) = a (.wire 43 12) ∧ a (.virt 18979) = a (.wire 43 13) ∧ a (.wire 43 11) = a (.wire 43 14) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies60, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies60, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies61 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 3) = a (.wire 43 16) ∧ a (.wire 12 11) = a (.wire 43 17) ∧ a (.wire 12 11) = a (.wire 43 18) ∧ a (.wire 36 3) = a (.wire 43 20) ∧ a (.virt 18979) = a (.wire 43 21) ∧ a (.wire 43 19) = a (.wire 43 22) ∧ a (.wire 36 3) = a (.wire 43 24) ∧ a (.wire 12 19) = a (.wire 43 25) ∧ a (.wire 12 19) = a (.wire 43 26) ∧ a (.wire 36 3) = a (.wire 43 28) ∧ a (.virt 18979) = a (.wire 43 29) ∧ a (.wire 43 27) = a (.wire 43 30) ∧ a (.wire 44 33) = a (.virt 18979) ∧ a (.wire 44 34) = a (.virt 18979) ∧ a (.wire 44 35) = a (.virt 18979) ∧ a (.wire 44 36) = a (.virt 18979) ∧ a (.wire 44 37) = a (.virt 18979) ∧ a (.wire 44 38) = a (.virt 18979) ∧ a (.wire 44 39) = a (.virt 18979) ∧ a (.wire 44 40) = a (.virt 18979) ∧ a (.wire 44 41) = a (.virt 18979) ∧ a (.wire 44 42) = a (.virt 18979) ∧ a (.wire 44 43) = a (.virt 18979) ∧ a (.wire 44 44) = a (.virt 18979) ∧ a (.wire 44 45) = a (.virt 18979) ∧ a (.wire 44 46) = a (.virt 18979) ∧ a (.wire 44 47) = a (.virt 18979) ∧ a (.wire 44 48) = a (.virt 18979) ∧ a (.wire 44 49) = a (.virt 18979) ∧ a (.wire 44 50) = a (.virt 18979) ∧ a (.wire 44 51) = a (.virt 18979) ∧ a (.wire 44 52) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies61, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies61, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies62 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 44 53) = a (.virt 18979) ∧ a (.wire 44 54) = a (.virt 18979) ∧ a (.wire 44 55) = a (.virt 18979) ∧ a (.wire 44 56) = a (.virt 18979) ∧ a (.wire 44 57) = a (.virt 18979) ∧ a (.wire 44 58) = a (.virt 18979) ∧ a (.wire 44 59) = a (.virt 18979) ∧ a (.wire 44 0) = a (.wire 41 59) ∧ a (.virt 18978) = a (.wire 43 32) ∧ a (.virt 18978) = a (.wire 43 33) ∧ a (.virt 19139) = a (.wire 43 34) ∧ a (.wire 9 35) = a (.wire 43 36) ∧ a (.virt 18978) = a (.wire 43 37) ∧ a (.wire 12 35) = a (.wire 43 38) ∧ a (.virt 19139) = a (.wire 42 24) ∧ a (.wire 43 39) = a (.wire 42 25) ∧ a (.virt 19139) = a (.wire 42 26) ∧ a (.wire 43 39) = a (.wire 42 28) ∧ a (.virt 19140) = a (.wire 42 29) ∧ a (.wire 43 39) = a (.wire 42 30) ∧ a (.wire 42 31) = a (.wire 43 40) ∧ a (.virt 18978) = a (.wire 43 41) ∧ a (.wire 43 35) = a (.wire 43 42) ∧ a (.wire 42 27) = a (.virt 18979) ∧ a (.wire 43 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 43 44) ∧ a (.virt 18978) = a (.wire 43 45) ∧ a (.virt 19141) = a (.wire 43 46) ∧ a (.wire 9 43) = a (.wire 43 48) ∧ a (.virt 18978) = a (.wire 43 49) ∧ a (.wire 12 43) = a (.wire 43 50) ∧ a (.virt 19141) = a (.wire 42 32) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies62, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies62, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies63 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 43 51) = a (.wire 42 33) ∧ a (.virt 19141) = a (.wire 42 34) ∧ a (.wire 43 51) = a (.wire 42 36) ∧ a (.virt 19142) = a (.wire 42 37) ∧ a (.wire 43 51) = a (.wire 42 38) ∧ a (.wire 42 39) = a (.wire 43 52) ∧ a (.virt 18978) = a (.wire 43 53) ∧ a (.wire 43 47) = a (.wire 43 54) ∧ a (.wire 42 35) = a (.virt 18979) ∧ a (.wire 43 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 43 56) ∧ a (.virt 18978) = a (.wire 43 57) ∧ a (.virt 19143) = a (.wire 43 58) ∧ a (.wire 9 51) = a (.wire 45 0) ∧ a (.virt 18978) = a (.wire 45 1) ∧ a (.wire 12 51) = a (.wire 45 2) ∧ a (.virt 19143) = a (.wire 42 40) ∧ a (.wire 45 3) = a (.wire 42 41) ∧ a (.virt 19143) = a (.wire 42 42) ∧ a (.wire 45 3) = a (.wire 42 44) ∧ a (.virt 19144) = a (.wire 42 45) ∧ a (.wire 45 3) = a (.wire 42 46) ∧ a (.wire 42 47) = a (.wire 45 4) ∧ a (.virt 18978) = a (.wire 45 5) ∧ a (.wire 43 59) = a (.wire 45 6) ∧ a (.wire 42 43) = a (.virt 18979) ∧ a (.wire 45 7) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 45 8) ∧ a (.virt 18978) = a (.wire 45 9) ∧ a (.virt 19145) = a (.wire 45 10) ∧ a (.wire 9 59) = a (.wire 45 12) ∧ a (.virt 18978) = a (.wire 45 13) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies63, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies63, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies64 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 59) = a (.wire 45 14) ∧ a (.virt 19145) = a (.wire 42 48) ∧ a (.wire 45 15) = a (.wire 42 49) ∧ a (.virt 19145) = a (.wire 42 50) ∧ a (.wire 45 15) = a (.wire 42 52) ∧ a (.virt 19146) = a (.wire 42 53) ∧ a (.wire 45 15) = a (.wire 42 54) ∧ a (.wire 42 55) = a (.wire 45 16) ∧ a (.virt 18978) = a (.wire 45 17) ∧ a (.wire 45 11) = a (.wire 45 18) ∧ a (.wire 42 51) = a (.virt 18979) ∧ a (.wire 45 19) = a (.virt 18979) ∧ a (.virt 19139) = a (.wire 42 56) ∧ a (.virt 19141) = a (.wire 42 57) ∧ a (.virt 19139) = a (.wire 42 58) ∧ a (.virt 19143) = a (.wire 46 0) ∧ a (.virt 19145) = a (.wire 46 1) ∧ a (.virt 19143) = a (.wire 46 2) ∧ a (.wire 42 59) = a (.wire 46 4) ∧ a (.wire 46 3) = a (.wire 46 5) ∧ a (.wire 42 59) = a (.wire 46 6) ∧ a (.virt 18978) = a (.wire 45 20) ∧ a (.virt 18978) = a (.wire 45 21) ∧ a (.virt 19147) = a (.wire 45 22) ∧ a (.wire 11 15) = a (.wire 45 24) ∧ a (.virt 18978) = a (.wire 45 25) ∧ a (.wire 12 35) = a (.wire 45 26) ∧ a (.virt 19147) = a (.wire 46 8) ∧ a (.wire 45 27) = a (.wire 46 9) ∧ a (.virt 19147) = a (.wire 46 10) ∧ a (.wire 45 27) = a (.wire 46 12) ∧ a (.virt 19148) = a (.wire 46 13) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies64, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies64, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies65 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 45 27) = a (.wire 46 14) ∧ a (.wire 46 15) = a (.wire 45 28) ∧ a (.virt 18978) = a (.wire 45 29) ∧ a (.wire 45 23) = a (.wire 45 30) ∧ a (.wire 46 11) = a (.virt 18979) ∧ a (.wire 45 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 45 32) ∧ a (.virt 18978) = a (.wire 45 33) ∧ a (.virt 19149) = a (.wire 45 34) ∧ a (.wire 11 23) = a (.wire 45 36) ∧ a (.virt 18978) = a (.wire 45 37) ∧ a (.wire 12 43) = a (.wire 45 38) ∧ a (.virt 19149) = a (.wire 46 16) ∧ a (.wire 45 39) = a (.wire 46 17) ∧ a (.virt 19149) = a (.wire 46 18) ∧ a (.wire 45 39) = a (.wire 46 20) ∧ a (.virt 19150) = a (.wire 46 21) ∧ a (.wire 45 39) = a (.wire 46 22) ∧ a (.wire 46 23) = a (.wire 45 40) ∧ a (.virt 18978) = a (.wire 45 41) ∧ a (.wire 45 35) = a (.wire 45 42) ∧ a (.wire 46 19) = a (.virt 18979) ∧ a (.wire 45 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 45 44) ∧ a (.virt 18978) = a (.wire 45 45) ∧ a (.virt 19151) = a (.wire 45 46) ∧ a (.wire 11 31) = a (.wire 45 48) ∧ a (.virt 18978) = a (.wire 45 49) ∧ a (.wire 12 51) = a (.wire 45 50) ∧ a (.virt 19151) = a (.wire 46 24) ∧ a (.wire 45 51) = a (.wire 46 25) ∧ a (.virt 19151) = a (.wire 46 26) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies65, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies65, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies66 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 45 51) = a (.wire 46 28) ∧ a (.virt 19152) = a (.wire 46 29) ∧ a (.wire 45 51) = a (.wire 46 30) ∧ a (.wire 46 31) = a (.wire 45 52) ∧ a (.virt 18978) = a (.wire 45 53) ∧ a (.wire 45 47) = a (.wire 45 54) ∧ a (.wire 46 27) = a (.virt 18979) ∧ a (.wire 45 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 45 56) ∧ a (.virt 18978) = a (.wire 45 57) ∧ a (.virt 19153) = a (.wire 45 58) ∧ a (.wire 11 39) = a (.wire 47 0) ∧ a (.virt 18978) = a (.wire 47 1) ∧ a (.wire 12 59) = a (.wire 47 2) ∧ a (.virt 19153) = a (.wire 46 32) ∧ a (.wire 47 3) = a (.wire 46 33) ∧ a (.virt 19153) = a (.wire 46 34) ∧ a (.wire 47 3) = a (.wire 46 36) ∧ a (.virt 19154) = a (.wire 46 37) ∧ a (.wire 47 3) = a (.wire 46 38) ∧ a (.wire 46 39) = a (.wire 47 4) ∧ a (.virt 18978) = a (.wire 47 5) ∧ a (.wire 45 59) = a (.wire 47 6) ∧ a (.wire 46 35) = a (.virt 18979) ∧ a (.wire 47 7) = a (.virt 18979) ∧ a (.virt 19147) = a (.wire 46 40) ∧ a (.virt 19149) = a (.wire 46 41) ∧ a (.virt 19147) = a (.wire 46 42) ∧ a (.virt 19151) = a (.wire 46 44) ∧ a (.virt 19153) = a (.wire 46 45) ∧ a (.virt 19151) = a (.wire 46 46) ∧ a (.wire 46 43) = a (.wire 46 48) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies66, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies66, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies67 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 47) = a (.wire 46 49) ∧ a (.wire 46 43) = a (.wire 46 50) ∧ a (.wire 46 7) = a (.wire 5 24) ∧ a (.wire 46 51) = a (.wire 5 25) ∧ a (.wire 46 7) = a (.wire 5 26) ∧ a (.wire 5 27) = a (.wire 36 16) ∧ a (.virt 18978) = a (.wire 36 17) ∧ a (.wire 46 51) = a (.wire 36 18) ∧ a (.virt 18978) = a (.wire 47 8) ∧ a (.virt 18978) = a (.wire 47 9) ∧ a (.virt 19155) = a (.wire 47 10) ∧ a (.wire 11 55) = a (.wire 47 12) ∧ a (.virt 18978) = a (.wire 47 13) ∧ a (.wire 12 35) = a (.wire 47 14) ∧ a (.virt 19155) = a (.wire 46 52) ∧ a (.wire 47 15) = a (.wire 46 53) ∧ a (.virt 19155) = a (.wire 46 54) ∧ a (.wire 47 15) = a (.wire 46 56) ∧ a (.virt 19156) = a (.wire 46 57) ∧ a (.wire 47 15) = a (.wire 46 58) ∧ a (.wire 46 59) = a (.wire 47 16) ∧ a (.virt 18978) = a (.wire 47 17) ∧ a (.wire 47 11) = a (.wire 47 18) ∧ a (.wire 46 55) = a (.virt 18979) ∧ a (.wire 47 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 47 20) ∧ a (.virt 18978) = a (.wire 47 21) ∧ a (.virt 19157) = a (.wire 47 22) ∧ a (.wire 12 3) = a (.wire 47 24) ∧ a (.virt 18978) = a (.wire 47 25) ∧ a (.wire 12 43) = a (.wire 47 26) ∧ a (.virt 19157) = a (.wire 48 0) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies67, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper2.copies67, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies68 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 47 27) = a (.wire 48 1) ∧ a (.virt 19157) = a (.wire 48 2) ∧ a (.wire 47 27) = a (.wire 48 4) ∧ a (.virt 19158) = a (.wire 48 5) ∧ a (.wire 47 27) = a (.wire 48 6) ∧ a (.wire 48 7) = a (.wire 47 28) ∧ a (.virt 18978) = a (.wire 47 29) ∧ a (.wire 47 23) = a (.wire 47 30) ∧ a (.wire 48 3) = a (.virt 18979) ∧ a (.wire 47 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 47 32) ∧ a (.virt 18978) = a (.wire 47 33) ∧ a (.virt 19159) = a (.wire 47 34) ∧ a (.wire 12 11) = a (.wire 47 36) ∧ a (.virt 18978) = a (.wire 47 37) ∧ a (.wire 12 51) = a (.wire 47 38) ∧ a (.virt 19159) = a (.wire 48 8) ∧ a (.wire 47 39) = a (.wire 48 9) ∧ a (.virt 19159) = a (.wire 48 10) ∧ a (.wire 47 39) = a (.wire 48 12) ∧ a (.virt 19160) = a (.wire 48 13) ∧ a (.wire 47 39) = a (.wire 48 14) ∧ a (.wire 48 15) = a (.wire 47 40) ∧ a (.virt 18978) = a (.wire 47 41) ∧ a (.wire 47 35) = a (.wire 47 42) ∧ a (.wire 48 11) = a (.virt 18979) ∧ a (.wire 47 43) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 47 44) ∧ a (.virt 18978) = a (.wire 47 45) ∧ a (.virt 19161) = a (.wire 47 46) ∧ a (.wire 12 19) = a (.wire 47 48) ∧ a (.virt 18978) = a (.wire 47 49) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies68, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies68, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies69 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 59) = a (.wire 47 50) ∧ a (.virt 19161) = a (.wire 48 16) ∧ a (.wire 47 51) = a (.wire 48 17) ∧ a (.virt 19161) = a (.wire 48 18) ∧ a (.wire 47 51) = a (.wire 48 20) ∧ a (.virt 19162) = a (.wire 48 21) ∧ a (.wire 47 51) = a (.wire 48 22) ∧ a (.wire 48 23) = a (.wire 47 52) ∧ a (.virt 18978) = a (.wire 47 53) ∧ a (.wire 47 47) = a (.wire 47 54) ∧ a (.wire 48 19) = a (.virt 18979) ∧ a (.wire 47 55) = a (.virt 18979) ∧ a (.virt 19155) = a (.wire 48 24) ∧ a (.virt 19157) = a (.wire 48 25) ∧ a (.virt 19155) = a (.wire 48 26) ∧ a (.virt 19159) = a (.wire 48 28) ∧ a (.virt 19161) = a (.wire 48 29) ∧ a (.virt 19159) = a (.wire 48 30) ∧ a (.wire 48 27) = a (.wire 48 32) ∧ a (.wire 48 31) = a (.wire 48 33) ∧ a (.wire 48 27) = a (.wire 48 34) ∧ a (.wire 36 19) = a (.wire 5 28) ∧ a (.wire 48 35) = a (.wire 5 29) ∧ a (.wire 36 19) = a (.wire 5 30) ∧ a (.wire 5 31) = a (.wire 36 20) ∧ a (.virt 18978) = a (.wire 36 21) ∧ a (.wire 48 35) = a (.wire 36 22) ∧ a (.virt 18978) = a (.wire 47 56) ∧ a (.virt 18978) = a (.wire 47 57) ∧ a (.virt 19163) = a (.wire 47 58) ∧ a (.virt 19163) = a (.wire 48 36) ∧ a (.wire 43 39) = a (.wire 48 37) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies69, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies69, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies70 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19163) = a (.wire 48 38) ∧ a (.wire 43 39) = a (.wire 48 40) ∧ a (.virt 19164) = a (.wire 48 41) ∧ a (.wire 43 39) = a (.wire 48 42) ∧ a (.wire 48 43) = a (.wire 49 0) ∧ a (.virt 18978) = a (.wire 49 1) ∧ a (.wire 47 59) = a (.wire 49 2) ∧ a (.wire 48 39) = a (.virt 18979) ∧ a (.wire 49 3) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 49 4) ∧ a (.virt 18978) = a (.wire 49 5) ∧ a (.virt 19165) = a (.wire 49 6) ∧ a (.virt 19165) = a (.wire 48 44) ∧ a (.wire 43 51) = a (.wire 48 45) ∧ a (.virt 19165) = a (.wire 48 46) ∧ a (.wire 43 51) = a (.wire 48 48) ∧ a (.virt 19166) = a (.wire 48 49) ∧ a (.wire 43 51) = a (.wire 48 50) ∧ a (.wire 48 51) = a (.wire 49 8) ∧ a (.virt 18978) = a (.wire 49 9) ∧ a (.wire 49 7) = a (.wire 49 10) ∧ a (.wire 48 47) = a (.virt 18979) ∧ a (.wire 49 11) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 49 12) ∧ a (.virt 18978) = a (.wire 49 13) ∧ a (.virt 19167) = a (.wire 49 14) ∧ a (.virt 19167) = a (.wire 48 52) ∧ a (.wire 45 3) = a (.wire 48 53) ∧ a (.virt 19167) = a (.wire 48 54) ∧ a (.wire 45 3) = a (.wire 48 56) ∧ a (.virt 19168) = a (.wire 48 57) ∧ a (.wire 45 3) = a (.wire 48 58) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies70, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies70, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies71 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 48 59) = a (.wire 49 16) ∧ a (.virt 18978) = a (.wire 49 17) ∧ a (.wire 49 15) = a (.wire 49 18) ∧ a (.wire 48 55) = a (.virt 18979) ∧ a (.wire 49 19) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 49 20) ∧ a (.virt 18978) = a (.wire 49 21) ∧ a (.virt 19169) = a (.wire 49 22) ∧ a (.virt 19169) = a (.wire 50 0) ∧ a (.wire 45 15) = a (.wire 50 1) ∧ a (.virt 19169) = a (.wire 50 2) ∧ a (.wire 45 15) = a (.wire 50 4) ∧ a (.virt 19170) = a (.wire 50 5) ∧ a (.wire 45 15) = a (.wire 50 6) ∧ a (.wire 50 7) = a (.wire 49 24) ∧ a (.virt 18978) = a (.wire 49 25) ∧ a (.wire 49 23) = a (.wire 49 26) ∧ a (.wire 50 3) = a (.virt 18979) ∧ a (.wire 49 27) = a (.virt 18979) ∧ a (.virt 19163) = a (.wire 50 8) ∧ a (.virt 19165) = a (.wire 50 9) ∧ a (.virt 19163) = a (.wire 50 10) ∧ a (.virt 19167) = a (.wire 50 12) ∧ a (.virt 19169) = a (.wire 50 13) ∧ a (.virt 19167) = a (.wire 50 14) ∧ a (.wire 50 11) = a (.wire 50 16) ∧ a (.wire 50 15) = a (.wire 50 17) ∧ a (.wire 50 11) = a (.wire 50 18) ∧ a (.wire 50 19) = a (.wire 49 28) ∧ a (.wire 11 7) = a (.wire 49 29) ∧ a (.virt 18979) = a (.wire 49 30) ∧ a (.virt 18978) = a (.wire 49 32) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies71, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies71, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies72 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 49 33) ∧ a (.virt 19171) = a (.wire 49 34) ∧ a (.virt 19171) = a (.wire 50 20) ∧ a (.wire 45 27) = a (.wire 50 21) ∧ a (.virt 19171) = a (.wire 50 22) ∧ a (.wire 45 27) = a (.wire 50 24) ∧ a (.virt 19172) = a (.wire 50 25) ∧ a (.wire 45 27) = a (.wire 50 26) ∧ a (.wire 50 27) = a (.wire 49 36) ∧ a (.virt 18978) = a (.wire 49 37) ∧ a (.wire 49 35) = a (.wire 49 38) ∧ a (.wire 50 23) = a (.virt 18979) ∧ a (.wire 49 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 49 40) ∧ a (.virt 18978) = a (.wire 49 41) ∧ a (.virt 19173) = a (.wire 49 42) ∧ a (.virt 19173) = a (.wire 50 28) ∧ a (.wire 45 39) = a (.wire 50 29) ∧ a (.virt 19173) = a (.wire 50 30) ∧ a (.wire 45 39) = a (.wire 50 32) ∧ a (.virt 19174) = a (.wire 50 33) ∧ a (.wire 45 39) = a (.wire 50 34) ∧ a (.wire 50 35) = a (.wire 49 44) ∧ a (.virt 18978) = a (.wire 49 45) ∧ a (.wire 49 43) = a (.wire 49 46) ∧ a (.wire 50 31) = a (.virt 18979) ∧ a (.wire 49 47) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 49 48) ∧ a (.virt 18978) = a (.wire 49 49) ∧ a (.virt 19175) = a (.wire 49 50) ∧ a (.virt 19175) = a (.wire 50 36) ∧ a (.wire 45 51) = a (.wire 50 37) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies72, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies72, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies73 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19175) = a (.wire 50 38) ∧ a (.wire 45 51) = a (.wire 50 40) ∧ a (.virt 19176) = a (.wire 50 41) ∧ a (.wire 45 51) = a (.wire 50 42) ∧ a (.wire 50 43) = a (.wire 49 52) ∧ a (.virt 18978) = a (.wire 49 53) ∧ a (.wire 49 51) = a (.wire 49 54) ∧ a (.wire 50 39) = a (.virt 18979) ∧ a (.wire 49 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 49 56) ∧ a (.virt 18978) = a (.wire 49 57) ∧ a (.virt 19177) = a (.wire 49 58) ∧ a (.virt 19177) = a (.wire 50 44) ∧ a (.wire 47 3) = a (.wire 50 45) ∧ a (.virt 19177) = a (.wire 50 46) ∧ a (.wire 47 3) = a (.wire 50 48) ∧ a (.virt 19178) = a (.wire 50 49) ∧ a (.wire 47 3) = a (.wire 50 50) ∧ a (.wire 50 51) = a (.wire 51 0) ∧ a (.virt 18978) = a (.wire 51 1) ∧ a (.wire 49 59) = a (.wire 51 2) ∧ a (.wire 50 47) = a (.virt 18979) ∧ a (.wire 51 3) = a (.virt 18979) ∧ a (.virt 19171) = a (.wire 50 52) ∧ a (.virt 19173) = a (.wire 50 53) ∧ a (.virt 19171) = a (.wire 50 54) ∧ a (.virt 19175) = a (.wire 50 56) ∧ a (.virt 19177) = a (.wire 50 57) ∧ a (.virt 19175) = a (.wire 50 58) ∧ a (.wire 50 55) = a (.wire 52 0) ∧ a (.wire 50 59) = a (.wire 52 1) ∧ a (.wire 50 55) = a (.wire 52 2) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies73, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies73, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies74 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 52 3) = a (.wire 51 4) ∧ a (.wire 11 47) = a (.wire 51 5) ∧ a (.virt 18979) = a (.wire 51 6) ∧ a (.wire 49 31) = a (.wire 36 24) ∧ a (.virt 18978) = a (.wire 36 25) ∧ a (.wire 51 7) = a (.wire 36 26) ∧ a (.virt 18978) = a (.wire 51 8) ∧ a (.virt 18978) = a (.wire 51 9) ∧ a (.virt 19179) = a (.wire 51 10) ∧ a (.virt 19179) = a (.wire 52 4) ∧ a (.wire 47 15) = a (.wire 52 5) ∧ a (.virt 19179) = a (.wire 52 6) ∧ a (.wire 47 15) = a (.wire 52 8) ∧ a (.virt 19180) = a (.wire 52 9) ∧ a (.wire 47 15) = a (.wire 52 10) ∧ a (.wire 52 11) = a (.wire 51 12) ∧ a (.virt 18978) = a (.wire 51 13) ∧ a (.wire 51 11) = a (.wire 51 14) ∧ a (.wire 52 7) = a (.virt 18979) ∧ a (.wire 51 15) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 51 16) ∧ a (.virt 18978) = a (.wire 51 17) ∧ a (.virt 19181) = a (.wire 51 18) ∧ a (.virt 19181) = a (.wire 52 12) ∧ a (.wire 47 27) = a (.wire 52 13) ∧ a (.virt 19181) = a (.wire 52 14) ∧ a (.wire 47 27) = a (.wire 52 16) ∧ a (.virt 19182) = a (.wire 52 17) ∧ a (.wire 47 27) = a (.wire 52 18) ∧ a (.wire 52 19) = a (.wire 51 20) ∧ a (.virt 18978) = a (.wire 51 21) ∧ a (.wire 51 19) = a (.wire 51 22) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies74, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies74, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies75 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 52 15) = a (.virt 18979) ∧ a (.wire 51 23) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 51 24) ∧ a (.virt 18978) = a (.wire 51 25) ∧ a (.virt 19183) = a (.wire 51 26) ∧ a (.virt 19183) = a (.wire 52 20) ∧ a (.wire 47 39) = a (.wire 52 21) ∧ a (.virt 19183) = a (.wire 52 22) ∧ a (.wire 47 39) = a (.wire 52 24) ∧ a (.virt 19184) = a (.wire 52 25) ∧ a (.wire 47 39) = a (.wire 52 26) ∧ a (.wire 52 27) = a (.wire 51 28) ∧ a (.virt 18978) = a (.wire 51 29) ∧ a (.wire 51 27) = a (.wire 51 30) ∧ a (.wire 52 23) = a (.virt 18979) ∧ a (.wire 51 31) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 51 32) ∧ a (.virt 18978) = a (.wire 51 33) ∧ a (.virt 19185) = a (.wire 51 34) ∧ a (.virt 19185) = a (.wire 52 28) ∧ a (.wire 47 51) = a (.wire 52 29) ∧ a (.virt 19185) = a (.wire 52 30) ∧ a (.wire 47 51) = a (.wire 52 32) ∧ a (.virt 19186) = a (.wire 52 33) ∧ a (.wire 47 51) = a (.wire 52 34) ∧ a (.wire 52 35) = a (.wire 51 36) ∧ a (.virt 18978) = a (.wire 51 37) ∧ a (.wire 51 35) = a (.wire 51 38) ∧ a (.wire 52 31) = a (.virt 18979) ∧ a (.wire 51 39) = a (.virt 18979) ∧ a (.virt 19179) = a (.wire 52 36) ∧ a (.virt 19181) = a (.wire 52 37) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies75, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies75, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies76 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19179) = a (.wire 52 38) ∧ a (.virt 19183) = a (.wire 52 40) ∧ a (.virt 19185) = a (.wire 52 41) ∧ a (.virt 19183) = a (.wire 52 42) ∧ a (.wire 52 39) = a (.wire 52 44) ∧ a (.wire 52 43) = a (.wire 52 45) ∧ a (.wire 52 39) = a (.wire 52 46) ∧ a (.wire 52 47) = a (.wire 51 40) ∧ a (.wire 12 27) = a (.wire 51 41) ∧ a (.virt 18979) = a (.wire 51 42) ∧ a (.wire 36 27) = a (.wire 36 28) ∧ a (.virt 18978) = a (.wire 36 29) ∧ a (.wire 51 43) = a (.wire 36 30) ∧ a (.virt 18978) = a (.wire 51 44) ∧ a (.virt 18978) = a (.wire 51 45) ∧ a (.virt 19187) = a (.wire 51 46) ∧ a (.wire 12 35) = a (.wire 51 48) ∧ a (.virt 18978) = a (.wire 51 49) ∧ a (.wire 12 35) = a (.wire 51 50) ∧ a (.virt 19187) = a (.wire 52 48) ∧ a (.wire 51 51) = a (.wire 52 49) ∧ a (.virt 19187) = a (.wire 52 50) ∧ a (.wire 51 51) = a (.wire 52 52) ∧ a (.virt 19188) = a (.wire 52 53) ∧ a (.wire 51 51) = a (.wire 52 54) ∧ a (.wire 52 55) = a (.wire 51 52) ∧ a (.virt 18978) = a (.wire 51 53) ∧ a (.wire 51 47) = a (.wire 51 54) ∧ a (.wire 52 51) = a (.virt 18979) ∧ a (.wire 51 55) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 51 56) ∧ a (.virt 18978) = a (.wire 51 57) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies76, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies76, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies77 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 19189) = a (.wire 51 58) ∧ a (.wire 12 43) = a (.wire 53 0) ∧ a (.virt 18978) = a (.wire 53 1) ∧ a (.wire 12 43) = a (.wire 53 2) ∧ a (.virt 19189) = a (.wire 52 56) ∧ a (.wire 53 3) = a (.wire 52 57) ∧ a (.virt 19189) = a (.wire 52 58) ∧ a (.wire 53 3) = a (.wire 54 0) ∧ a (.virt 19190) = a (.wire 54 1) ∧ a (.wire 53 3) = a (.wire 54 2) ∧ a (.wire 54 3) = a (.wire 53 4) ∧ a (.virt 18978) = a (.wire 53 5) ∧ a (.wire 51 59) = a (.wire 53 6) ∧ a (.wire 52 59) = a (.virt 18979) ∧ a (.wire 53 7) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 53 8) ∧ a (.virt 18978) = a (.wire 53 9) ∧ a (.virt 19191) = a (.wire 53 10) ∧ a (.wire 12 51) = a (.wire 53 12) ∧ a (.virt 18978) = a (.wire 53 13) ∧ a (.wire 12 51) = a (.wire 53 14) ∧ a (.virt 19191) = a (.wire 54 4) ∧ a (.wire 53 15) = a (.wire 54 5) ∧ a (.virt 19191) = a (.wire 54 6) ∧ a (.wire 53 15) = a (.wire 54 8) ∧ a (.virt 19192) = a (.wire 54 9) ∧ a (.wire 53 15) = a (.wire 54 10) ∧ a (.wire 54 11) = a (.wire 53 16) ∧ a (.virt 18978) = a (.wire 53 17) ∧ a (.wire 53 11) = a (.wire 53 18) ∧ a (.wire 54 7) = a (.virt 18979) ∧ a (.wire 53 19) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies77, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies77, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies78 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 53 20) ∧ a (.virt 18978) = a (.wire 53 21) ∧ a (.virt 19193) = a (.wire 53 22) ∧ a (.wire 12 59) = a (.wire 53 24) ∧ a (.virt 18978) = a (.wire 53 25) ∧ a (.wire 12 59) = a (.wire 53 26) ∧ a (.virt 19193) = a (.wire 54 12) ∧ a (.wire 53 27) = a (.wire 54 13) ∧ a (.virt 19193) = a (.wire 54 14) ∧ a (.wire 53 27) = a (.wire 54 16) ∧ a (.virt 19194) = a (.wire 54 17) ∧ a (.wire 53 27) = a (.wire 54 18) ∧ a (.wire 54 19) = a (.wire 53 28) ∧ a (.virt 18978) = a (.wire 53 29) ∧ a (.wire 53 23) = a (.wire 53 30) ∧ a (.wire 54 15) = a (.virt 18979) ∧ a (.wire 53 31) = a (.virt 18979) ∧ a (.virt 19187) = a (.wire 54 20) ∧ a (.virt 19189) = a (.wire 54 21) ∧ a (.virt 19187) = a (.wire 54 22) ∧ a (.virt 19191) = a (.wire 54 24) ∧ a (.virt 19193) = a (.wire 54 25) ∧ a (.virt 19191) = a (.wire 54 26) ∧ a (.wire 54 23) = a (.wire 54 28) ∧ a (.wire 54 27) = a (.wire 54 29) ∧ a (.wire 54 23) = a (.wire 54 30) ∧ a (.wire 54 31) = a (.wire 53 32) ∧ a (.wire 13 7) = a (.wire 53 33) ∧ a (.virt 18979) = a (.wire 53 34) ∧ a (.wire 36 31) = a (.wire 36 32) ∧ a (.virt 18978) = a (.wire 36 33) ∧ a (.wire 53 35) = a (.wire 36 34) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies78, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies78, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies79 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 23) = a (.wire 53 36) ∧ a (.wire 36 35) = a (.wire 53 37) ∧ a (.wire 36 35) = a (.wire 53 38) ∧ a (.wire 36 23) = a (.wire 53 40) ∧ a (.virt 18979) = a (.wire 53 41) ∧ a (.wire 53 39) = a (.wire 53 42) ∧ a (.wire 36 23) = a (.wire 53 44) ∧ a (.wire 12 35) = a (.wire 53 45) ∧ a (.wire 12 35) = a (.wire 53 46) ∧ a (.wire 36 23) = a (.wire 53 48) ∧ a (.virt 18979) = a (.wire 53 49) ∧ a (.wire 53 47) = a (.wire 53 50) ∧ a (.wire 36 23) = a (.wire 53 52) ∧ a (.wire 12 43) = a (.wire 53 53) ∧ a (.wire 12 43) = a (.wire 53 54) ∧ a (.wire 36 23) = a (.wire 53 56) ∧ a (.virt 18979) = a (.wire 53 57) ∧ a (.wire 53 55) = a (.wire 53 58) ∧ a (.wire 36 23) = a (.wire 55 0) ∧ a (.wire 12 51) = a (.wire 55 1) ∧ a (.wire 12 51) = a (.wire 55 2) ∧ a (.wire 36 23) = a (.wire 55 4) ∧ a (.virt 18979) = a (.wire 55 5) ∧ a (.wire 55 3) = a (.wire 55 6) ∧ a (.wire 36 23) = a (.wire 55 8) ∧ a (.wire 12 59) = a (.wire 55 9) ∧ a (.wire 12 59) = a (.wire 55 10) ∧ a (.wire 36 23) = a (.wire 55 12) ∧ a (.virt 18979) = a (.wire 55 13) ∧ a (.wire 55 11) = a (.wire 55 14) ∧ a (.wire 56 33) = a (.virt 18979) ∧ a (.wire 56 34) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies79, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies79, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies80 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 56 35) = a (.virt 18979) ∧ a (.wire 56 36) = a (.virt 18979) ∧ a (.wire 56 37) = a (.virt 18979) ∧ a (.wire 56 38) = a (.virt 18979) ∧ a (.wire 56 39) = a (.virt 18979) ∧ a (.wire 56 40) = a (.virt 18979) ∧ a (.wire 56 41) = a (.virt 18979) ∧ a (.wire 56 42) = a (.virt 18979) ∧ a (.wire 56 43) = a (.virt 18979) ∧ a (.wire 56 44) = a (.virt 18979) ∧ a (.wire 56 45) = a (.virt 18979) ∧ a (.wire 56 46) = a (.virt 18979) ∧ a (.wire 56 47) = a (.virt 18979) ∧ a (.wire 56 48) = a (.virt 18979) ∧ a (.wire 56 49) = a (.virt 18979) ∧ a (.wire 56 50) = a (.virt 18979) ∧ a (.wire 56 51) = a (.virt 18979) ∧ a (.wire 56 52) = a (.virt 18979) ∧ a (.wire 56 53) = a (.virt 18979) ∧ a (.wire 56 54) = a (.virt 18979) ∧ a (.wire 56 55) = a (.virt 18979) ∧ a (.wire 56 56) = a (.virt 18979) ∧ a (.wire 56 57) = a (.virt 18979) ∧ a (.wire 56 58) = a (.virt 18979) ∧ a (.wire 56 59) = a (.virt 18979) ∧ a (.wire 56 0) = a (.wire 53 43) ∧ a (.wire 3 7) = a (.wire 54 32) ∧ a (.wire 3 35) = a (.wire 54 33) ∧ a (.wire 3 7) = a (.wire 54 34) ∧ a (.virt 18978) = a (.wire 55 16) ∧ a (.virt 18978) = a (.wire 55 17) ∧ a (.virt 19195) = a (.wire 55 18) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies80, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies80, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies81 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 9467) = a (.wire 55 20) ∧ a (.virt 18978) = a (.wire 55 21) ∧ a (.virt 18952) = a (.wire 55 22) ∧ a (.virt 19195) = a (.wire 54 36) ∧ a (.wire 55 23) = a (.wire 54 37) ∧ a (.virt 19195) = a (.wire 54 38) ∧ a (.wire 55 23) = a (.wire 54 40) ∧ a (.virt 19196) = a (.wire 54 41) ∧ a (.wire 55 23) = a (.wire 54 42) ∧ a (.wire 54 43) = a (.wire 55 24) ∧ a (.virt 18978) = a (.wire 55 25) ∧ a (.wire 55 19) = a (.wire 55 26) ∧ a (.wire 54 39) = a (.virt 18979) ∧ a (.wire 55 27) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 55 28) ∧ a (.virt 18978) = a (.wire 55 29) ∧ a (.virt 19197) = a (.wire 55 30) ∧ a (.virt 9468) = a (.wire 55 32) ∧ a (.virt 18978) = a (.wire 55 33) ∧ a (.virt 18953) = a (.wire 55 34) ∧ a (.virt 19197) = a (.wire 54 44) ∧ a (.wire 55 35) = a (.wire 54 45) ∧ a (.virt 19197) = a (.wire 54 46) ∧ a (.wire 55 35) = a (.wire 54 48) ∧ a (.virt 19198) = a (.wire 54 49) ∧ a (.wire 55 35) = a (.wire 54 50) ∧ a (.wire 54 51) = a (.wire 55 36) ∧ a (.virt 18978) = a (.wire 55 37) ∧ a (.wire 55 31) = a (.wire 55 38) ∧ a (.wire 54 47) = a (.virt 18979) ∧ a (.wire 55 39) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 55 40) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies81, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies81, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies82 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = a (.wire 55 41) ∧ a (.virt 19199) = a (.wire 55 42) ∧ a (.virt 9469) = a (.wire 55 44) ∧ a (.virt 18978) = a (.wire 55 45) ∧ a (.virt 18954) = a (.wire 55 46) ∧ a (.virt 19199) = a (.wire 54 52) ∧ a (.wire 55 47) = a (.wire 54 53) ∧ a (.virt 19199) = a (.wire 54 54) ∧ a (.wire 55 47) = a (.wire 54 56) ∧ a (.virt 19200) = a (.wire 54 57) ∧ a (.wire 55 47) = a (.wire 54 58) ∧ a (.wire 54 59) = a (.wire 55 48) ∧ a (.virt 18978) = a (.wire 55 49) ∧ a (.wire 55 43) = a (.wire 55 50) ∧ a (.wire 54 55) = a (.virt 18979) ∧ a (.wire 55 51) = a (.virt 18979) ∧ a (.virt 18978) = a (.wire 55 52) ∧ a (.virt 18978) = a (.wire 55 53) ∧ a (.virt 19201) = a (.wire 55 54) ∧ a (.virt 9470) = a (.wire 55 56) ∧ a (.virt 18978) = a (.wire 55 57) ∧ a (.virt 18955) = a (.wire 55 58) ∧ a (.virt 19201) = a (.wire 57 0) ∧ a (.wire 55 59) = a (.wire 57 1) ∧ a (.virt 19201) = a (.wire 57 2) ∧ a (.wire 55 59) = a (.wire 57 4) ∧ a (.virt 19202) = a (.wire 57 5) ∧ a (.wire 55 59) = a (.wire 57 6) ∧ a (.wire 57 7) = a (.wire 58 0) ∧ a (.virt 18978) = a (.wire 58 1) ∧ a (.wire 55 55) = a (.wire 58 2) ∧ a (.wire 57 3) = a (.virt 18979) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies82, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies82, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies83 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 3) = a (.virt 18979) ∧ a (.virt 19195) = a (.wire 57 8) ∧ a (.virt 19197) = a (.wire 57 9) ∧ a (.virt 19195) = a (.wire 57 10) ∧ a (.virt 19199) = a (.wire 57 12) ∧ a (.virt 19201) = a (.wire 57 13) ∧ a (.virt 19199) = a (.wire 57 14) ∧ a (.wire 57 11) = a (.wire 57 16) ∧ a (.wire 57 15) = a (.wire 57 17) ∧ a (.wire 57 11) = a (.wire 57 18) ∧ a (.wire 54 35) = a (.wire 57 20) ∧ a (.wire 57 19) = a (.wire 57 21) ∧ a (.wire 54 35) = a (.wire 57 22) ∧ a (.wire 57 23) = a (.virt 18979) ∧ a (.virt 18970) = a (.wire 59 0) ∧ a (.virt 18971) = a (.wire 59 1) ∧ a (.virt 18972) = a (.wire 59 2) ∧ a (.virt 18973) = a (.wire 59 3) ∧ a (.virt 18978) = a (.wire 59 4) ∧ a (.virt 18979) = a (.wire 59 5) ∧ a (.virt 18979) = a (.wire 59 6) ∧ a (.virt 18979) = a (.wire 59 7) ∧ a (.virt 18979) = a (.wire 59 8) ∧ a (.virt 18979) = a (.wire 59 9) ∧ a (.virt 18979) = a (.wire 59 10) ∧ a (.virt 18979) = a (.wire 59 11) ∧ a (.wire 59 12) = a (.wire 60 0) ∧ a (.wire 59 13) = a (.wire 60 1) ∧ a (.wire 59 14) = a (.wire 60 2) ∧ a (.wire 59 15) = a (.wire 60 3) ∧ a (.virt 18978) = a (.wire 60 4) ∧ a (.virt 18979) = a (.wire 60 5) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies83, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies83, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies84 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18979) = a (.wire 60 6) ∧ a (.virt 18979) = a (.wire 60 7) ∧ a (.virt 18979) = a (.wire 60 8) ∧ a (.virt 18979) = a (.wire 60 9) ∧ a (.virt 18979) = a (.wire 60 10) ∧ a (.virt 18979) = a (.wire 60 11) ∧ a (.wire 1 43) = a (.wire 58 4) ∧ a (.virt 9467) = a (.wire 58 5) ∧ a (.virt 9467) = a (.wire 58 6) ∧ a (.wire 1 43) = a (.wire 58 8) ∧ a (.wire 60 12) = a (.wire 58 9) ∧ a (.wire 58 7) = a (.wire 58 10) ∧ a (.wire 1 43) = a (.wire 58 12) ∧ a (.virt 9468) = a (.wire 58 13) ∧ a (.virt 9468) = a (.wire 58 14) ∧ a (.wire 1 43) = a (.wire 58 16) ∧ a (.wire 60 13) = a (.wire 58 17) ∧ a (.wire 58 15) = a (.wire 58 18) ∧ a (.wire 1 43) = a (.wire 58 20) ∧ a (.virt 9469) = a (.wire 58 21) ∧ a (.virt 9469) = a (.wire 58 22) ∧ a (.wire 1 43) = a (.wire 58 24) ∧ a (.wire 60 14) = a (.wire 58 25) ∧ a (.wire 58 23) = a (.wire 58 26) ∧ a (.wire 1 43) = a (.wire 58 28) ∧ a (.virt 9470) = a (.wire 58 29) ∧ a (.virt 9470) = a (.wire 58 30) ∧ a (.wire 1 43) = a (.wire 58 32) ∧ a (.wire 60 15) = a (.wire 58 33) ∧ a (.wire 58 31) = a (.wire 58 34) ∧ a (.virt 18974) = a (.wire 61 0) ∧ a (.virt 18975) = a (.wire 61 1) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies84, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies84, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies85 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18976) = a (.wire 61 2) ∧ a (.virt 18977) = a (.wire 61 3) ∧ a (.virt 18978) = a (.wire 61 4) ∧ a (.virt 18979) = a (.wire 61 5) ∧ a (.virt 18979) = a (.wire 61 6) ∧ a (.virt 18979) = a (.wire 61 7) ∧ a (.virt 18979) = a (.wire 61 8) ∧ a (.virt 18979) = a (.wire 61 9) ∧ a (.virt 18979) = a (.wire 61 10) ∧ a (.virt 18979) = a (.wire 61 11) ∧ a (.wire 61 12) = a (.wire 62 0) ∧ a (.wire 61 13) = a (.wire 62 1) ∧ a (.wire 61 14) = a (.wire 62 2) ∧ a (.wire 61 15) = a (.wire 62 3) ∧ a (.virt 18978) = a (.wire 62 4) ∧ a (.virt 18979) = a (.wire 62 5) ∧ a (.virt 18979) = a (.wire 62 6) ∧ a (.virt 18979) = a (.wire 62 7) ∧ a (.virt 18979) = a (.wire 62 8) ∧ a (.virt 18979) = a (.wire 62 9) ∧ a (.virt 18979) = a (.wire 62 10) ∧ a (.virt 18979) = a (.wire 62 11) ∧ a (.wire 2 27) = a (.wire 58 36) ∧ a (.virt 18952) = a (.wire 58 37) ∧ a (.virt 18952) = a (.wire 58 38) ∧ a (.wire 2 27) = a (.wire 58 40) ∧ a (.wire 62 12) = a (.wire 58 41) ∧ a (.wire 58 39) = a (.wire 58 42) ∧ a (.wire 2 27) = a (.wire 58 44) ∧ a (.virt 18953) = a (.wire 58 45) ∧ a (.virt 18953) = a (.wire 58 46) ∧ a (.wire 2 27) = a (.wire 58 48) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies85, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies85, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies86 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 62 13) = a (.wire 58 49) ∧ a (.wire 58 47) = a (.wire 58 50) ∧ a (.wire 2 27) = a (.wire 58 52) ∧ a (.virt 18954) = a (.wire 58 53) ∧ a (.virt 18954) = a (.wire 58 54) ∧ a (.wire 2 27) = a (.wire 58 56) ∧ a (.wire 62 14) = a (.wire 58 57) ∧ a (.wire 58 55) = a (.wire 58 58) ∧ a (.wire 2 27) = a (.wire 63 0) ∧ a (.virt 18955) = a (.wire 63 1) ∧ a (.virt 18955) = a (.wire 63 2) ∧ a (.wire 2 27) = a (.wire 63 4) ∧ a (.wire 62 15) = a (.wire 63 5) ∧ a (.wire 63 3) = a (.wire 63 6) ∧ a (.virt 19203) = a (.wire 63 8) ∧ a (.virt 19203) = a (.wire 63 9) ∧ a (.virt 19203) = a (.wire 63 10) ∧ a (.wire 63 11) = a (.virt 18979) ∧ a (.virt 19203) = a (.wire 63 12) ∧ a (.wire 58 11) = a (.wire 63 13) ∧ a (.wire 58 11) = a (.wire 63 14) ∧ a (.virt 19203) = a (.wire 63 16) ∧ a (.wire 58 43) = a (.wire 63 17) ∧ a (.wire 63 15) = a (.wire 63 18) ∧ a (.virt 19203) = a (.wire 63 20) ∧ a (.wire 58 43) = a (.wire 63 21) ∧ a (.wire 58 43) = a (.wire 63 22) ∧ a (.virt 19203) = a (.wire 63 24) ∧ a (.wire 58 11) = a (.wire 63 25) ∧ a (.wire 63 23) = a (.wire 63 26) ∧ a (.virt 19203) = a (.wire 63 28) ∧ a (.wire 58 19) = a (.wire 63 29) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies86, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper2.copies86, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies87 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 19) = a (.wire 63 30) ∧ a (.virt 19203) = a (.wire 63 32) ∧ a (.wire 58 51) = a (.wire 63 33) ∧ a (.wire 63 31) = a (.wire 63 34) ∧ a (.virt 19203) = a (.wire 63 36) ∧ a (.wire 58 51) = a (.wire 63 37) ∧ a (.wire 58 51) = a (.wire 63 38) ∧ a (.virt 19203) = a (.wire 63 40) ∧ a (.wire 58 19) = a (.wire 63 41) ∧ a (.wire 63 39) = a (.wire 63 42) ∧ a (.virt 19203) = a (.wire 63 44) ∧ a (.wire 58 27) = a (.wire 63 45) ∧ a (.wire 58 27) = a (.wire 63 46) ∧ a (.virt 19203) = a (.wire 63 48) ∧ a (.wire 58 59) = a (.wire 63 49) ∧ a (.wire 63 47) = a (.wire 63 50) ∧ a (.virt 19203) = a (.wire 63 52) ∧ a (.wire 58 59) = a (.wire 63 53) ∧ a (.wire 58 59) = a (.wire 63 54) ∧ a (.virt 19203) = a (.wire 63 56) ∧ a (.wire 58 27) = a (.wire 63 57) ∧ a (.wire 63 55) = a (.wire 63 58) ∧ a (.virt 19203) = a (.wire 64 0) ∧ a (.wire 58 35) = a (.wire 64 1) ∧ a (.wire 58 35) = a (.wire 64 2) ∧ a (.virt 19203) = a (.wire 64 4) ∧ a (.wire 63 7) = a (.wire 64 5) ∧ a (.wire 64 3) = a (.wire 64 6) ∧ a (.virt 19203) = a (.wire 64 8) ∧ a (.wire 63 7) = a (.wire 64 9) ∧ a (.wire 63 7) = a (.wire 64 10) ∧ a (.virt 19203) = a (.wire 64 12) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies87, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))))
  simp only [privateBatchWrapper2.copies87, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies88 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 35) = a (.wire 64 13) ∧ a (.wire 64 11) = a (.wire 64 14) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies88, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))))
  simp only [privateBatchWrapper2.copies88, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_consts (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = 1 ∧ a (.virt 18979) = 0 ∧ a (.virt 18980) = 4 ∧ a (.virt 19017) = 10000 ∧ a (.virt 19018) = 576460752303423488 := by
  have hconst := h.2.2
  simp only [privateBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3, k4⟩ := hconst
  exact ⟨k0, k1, k2, k3, k4⟩

theorem privateBatchWrapper2_f0 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9479)) (a (.virt 18979)) (a (.virt 18981)) (a (.virt 18982)) := by
  have c0 := (privateBatchWrapper2_copies0 a h).1
  have c1 := (privateBatchWrapper2_copies0 a h).2.1
  have c2 := (privateBatchWrapper2_copies0 a h).2.2.1
  have c3 := (privateBatchWrapper2_copies0 a h).2.2.2.1
  have c4 := (privateBatchWrapper2_copies0 a h).2.2.2.2.1
  have c5 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.1
  have c6 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.1
  have c7 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
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

theorem privateBatchWrapper2_f1 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9480)) (a (.virt 18979)) (a (.virt 18983)) (a (.virt 18984)) := by
  have c14 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
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

theorem privateBatchWrapper2_f2 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9481)) (a (.virt 18979)) (a (.virt 18985)) (a (.virt 18986)) := by
  have c28 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (privateBatchWrapper2_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (privateBatchWrapper2_copies1 a h).1
  have c33 := (privateBatchWrapper2_copies1 a h).2.1
  have c34 := (privateBatchWrapper2_copies1 a h).2.2.1
  have c35 := (privateBatchWrapper2_copies1 a h).2.2.2.1
  have c36 := (privateBatchWrapper2_copies1 a h).2.2.2.2.1
  have c37 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.1
  have c38 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.1
  have c39 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
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

theorem privateBatchWrapper2_f3 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9482)) (a (.virt 18979)) (a (.virt 18987)) (a (.virt 18988)) := by
  have c42 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
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

theorem privateBatchWrapper2_f4 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 1 35) = band (a (.virt 18981)) (a (.virt 18983)) := by
  have c56 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c56, ← c57, ← c58] at e_1_8
  have hr := e_1_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f5 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 1 39) = band (a (.virt 18985)) (a (.virt 18987)) := by
  have c59 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c59, ← c60, ← c61] at e_1_9
  have hr := e_1_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f6 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) := by
  have c62 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (privateBatchWrapper2_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (privateBatchWrapper2_copies2 a h).1
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c62, ← c63, ← c64] at e_1_10
  have hr := e_1_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f7 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18964)) (a (.virt 18979)) (a (.virt 18989)) (a (.virt 18990)) := by
  have c65 := (privateBatchWrapper2_copies2 a h).2.1
  have c66 := (privateBatchWrapper2_copies2 a h).2.2.1
  have c67 := (privateBatchWrapper2_copies2 a h).2.2.2.1
  have c68 := (privateBatchWrapper2_copies2 a h).2.2.2.2.1
  have c69 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.1
  have c70 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.1
  have c71 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
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

theorem privateBatchWrapper2_f8 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18965)) (a (.virt 18979)) (a (.virt 18991)) (a (.virt 18992)) := by
  have c79 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
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

theorem privateBatchWrapper2_f9 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18966)) (a (.virt 18979)) (a (.virt 18993)) (a (.virt 18994)) := by
  have c93 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (privateBatchWrapper2_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (privateBatchWrapper2_copies3 a h).1
  have c97 := (privateBatchWrapper2_copies3 a h).2.1
  have c98 := (privateBatchWrapper2_copies3 a h).2.2.1
  have c99 := (privateBatchWrapper2_copies3 a h).2.2.2.1
  have c100 := (privateBatchWrapper2_copies3 a h).2.2.2.2.1
  have c101 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.1
  have c102 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.1
  have c103 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_0_12 := arithEq_of_rows h (row := 0) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_12
  simp only [← c93, k0, ← c94, k0, ← c95] at e_0_12
  have e_0_13 := arithEq_of_rows h (row := 0) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_13
  simp only [← c102, ← c103, k0, ← c104] at e_0_13
  have e_2_0 := arithEq_of_rows h (row := 2) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_0
  simp only [← c96, ← c97, ← c98] at e_2_0
  have e_2_1 := arithEq_of_rows h (row := 2) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_1
  simp only [← c99, ← c100, ← c101] at e_2_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_0
    linear_combination c105.trans k1 - hc
  · have hc := e_0_13
    simp only [e_0_12, e_2_1] at hc
    linear_combination c106.trans k1 - hc

theorem privateBatchWrapper2_f10 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18967)) (a (.virt 18979)) (a (.virt 18995)) (a (.virt 18996)) := by
  have c107 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_0_14 := arithEq_of_rows h (row := 0) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_0_14
  simp only [← c107, k0, ← c108, k0, ← c109] at e_0_14
  have e_2_2 := arithEq_of_rows h (row := 2) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_2
  simp only [← c110, ← c111, ← c112] at e_2_2
  have e_2_3 := arithEq_of_rows h (row := 2) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_3
  simp only [← c113, ← c114, ← c115] at e_2_3
  have e_3_0 := arithEq_of_rows h (row := 3) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_0
  simp only [← c116, ← c117, k0, ← c118] at e_3_0
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_2
    linear_combination c119.trans k1 - hc
  · have hc := e_3_0
    simp only [e_0_14, e_2_3] at hc
    linear_combination c120.trans k1 - hc

theorem privateBatchWrapper2_f11 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 19) = band (a (.virt 18989)) (a (.virt 18991)) := by
  have c121 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c121, ← c122, ← c123] at e_2_4
  have hr := e_2_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f12 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 23) = band (a (.virt 18993)) (a (.virt 18995)) := by
  have c124 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c124, ← c125, ← c126] at e_2_5
  have hr := e_2_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f13 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 27) = band (a (.wire 2 19)) (a (.wire 2 23)) := by
  have c127 := (privateBatchWrapper2_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (privateBatchWrapper2_copies4 a h).1
  have c129 := (privateBatchWrapper2_copies4 a h).2.1
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c127, ← c128, ← c129] at e_2_6
  have hr := e_2_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f14 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 7) = bnot (a (.wire 1 43)) := by
  have c130 := (privateBatchWrapper2_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper2_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper2_copies4 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c130, k0, ← c131, k0, ← c132] at e_3_1
  have hr := e_3_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f15 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = bnot (a (.virt 18979)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bnot, k0, k1]
  ring

theorem privateBatchWrapper2_f16 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 7) = band (a (.wire 3 7)) (a (.virt 18978)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [band, k0]
  ring

theorem privateBatchWrapper2_f17 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 11) = bselect (a (.wire 3 7)) (a (.virt 9479)) (a (.virt 18979)) := by
  have c133 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.1
  have c134 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.1
  have c135 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c133, ← c134, ← c135, k1] at e_3_2
  have hr := e_3_2
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f18 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 15) = bselect (a (.wire 3 7)) (a (.virt 9480)) (a (.virt 18979)) := by
  have c136 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c136, ← c137, ← c138, k1] at e_3_3
  have hr := e_3_3
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f19 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 19) = bselect (a (.wire 3 7)) (a (.virt 9481)) (a (.virt 18979)) := by
  have c139 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c139, ← c140, ← c141, k1] at e_3_4
  have hr := e_3_4
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f20 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 23) = bselect (a (.wire 3 7)) (a (.virt 9482)) (a (.virt 18979)) := by
  have c142 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c144 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c142, ← c143, ← c144, k1] at e_3_5
  have hr := e_3_5
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f21 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 27) = bselect (a (.wire 3 7)) (a (.virt 9483)) (a (.virt 18979)) := by
  have c145 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c145, ← c146, ← c147, k1] at e_3_6
  have hr := e_3_6
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f22 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = bselect (a (.wire 3 7)) (a (.virt 9466)) (a (.virt 18979)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c148, ← c149, ← c150, k1] at e_3_7
  have hr := e_3_7
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f23 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 7) = bor (a (.virt 18979)) (a (.wire 3 7)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bor, k1]
  ring

theorem privateBatchWrapper2_f24 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 35) = bnot (a (.wire 2 27)) := by
  have c151 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c151, k0, ← c152, k0, ← c153] at e_3_8
  have hr := e_3_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f25 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 39) = bnot (a (.wire 3 7)) := by
  have c154 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c154, k0, ← c155, k0, ← c156] at e_3_9
  have hr := e_3_9
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f26 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 31) = band (a (.wire 3 35)) (a (.wire 3 39)) := by
  have c157 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c157, ← c158, ← c159] at e_2_7
  have hr := e_2_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f27 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 47) = bselect (a (.wire 2 31)) (a (.virt 18964)) (a (.wire 3 11)) := by
  have c160 := (privateBatchWrapper2_copies5 a h).1
  have c161 := (privateBatchWrapper2_copies5 a h).2.1
  have c162 := (privateBatchWrapper2_copies5 a h).2.2.1
  have c163 := (privateBatchWrapper2_copies5 a h).2.2.2.1
  have c164 := (privateBatchWrapper2_copies5 a h).2.2.2.2.1
  have c165 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.1
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c160, ← c161, ← c162] at e_3_10
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c163, ← c164, ← c165] at e_3_11
  have hr := e_3_11
  simp only [e_3_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f28 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 55) = bselect (a (.wire 2 31)) (a (.virt 18965)) (a (.wire 3 15)) := by
  have c166 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.1
  have c167 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c166, ← c167, ← c168] at e_3_12
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c169, ← c170, ← c171] at e_3_13
  have hr := e_3_13
  simp only [e_3_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f29 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 3) = bselect (a (.wire 2 31)) (a (.virt 18966)) (a (.wire 3 19)) := by
  have c172 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c172, ← c173, ← c174] at e_3_14
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c175, ← c176, ← c177] at e_4_0
  have hr := e_4_0
  simp only [e_3_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f30 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 11) = bselect (a (.wire 2 31)) (a (.virt 18967)) (a (.wire 3 23)) := by
  have c178 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c178, ← c179, ← c180] at e_4_1
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c181, ← c182, ← c183] at e_4_2
  have hr := e_4_2
  simp only [e_4_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f31 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 19) = bselect (a (.wire 2 31)) (a (.virt 18968)) (a (.wire 3 27)) := by
  have c184 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c184, ← c185, ← c186] at e_4_3
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c187, ← c188, ← c189] at e_4_4
  have hr := e_4_4
  simp only [e_4_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f32 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 27) = bselect (a (.wire 2 31)) (a (.virt 18951)) (a (.wire 3 31)) := by
  have c190 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (privateBatchWrapper2_copies6 a h).1
  have c193 := (privateBatchWrapper2_copies6 a h).2.1
  have c194 := (privateBatchWrapper2_copies6 a h).2.2.1
  have c195 := (privateBatchWrapper2_copies6 a h).2.2.2.1
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c190, ← c191, ← c192] at e_4_5
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c193, ← c194, ← c195] at e_4_6
  have hr := e_4_6
  simp only [e_4_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f33 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 3) = bor (a (.wire 3 7)) (a (.wire 3 35)) := by
  have c196 := (privateBatchWrapper2_copies6 a h).2.2.2.2.1
  have c197 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.1
  have c198 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.1
  have c199 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c196, ← c197, ← c198] at e_5_0
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c199, ← c200, k0, ← c201] at e_6_0
  have hr := e_6_0
  simp only [e_5_0] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f34 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9479)) (a (.wire 3 47)) (a (.virt 18997)) (a (.virt 18998)) := by
  have c202 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c208, ← c209, ← c210] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c211, ← c212, ← c213] at e_2_9
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c202, k0, ← c203, k0, ← c204] at e_4_7
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c205, ← c206, k0, ← c207] at e_4_8
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c214, ← c215, k0, ← c216] at e_4_9
  refine ⟨?_, ?_⟩
  · have hc := e_2_8
    simp only [e_4_8] at hc
    linear_combination c217.trans k1 - hc
  · have hc := e_4_9
    simp only [e_4_7, e_2_9, e_4_8] at hc
    linear_combination c218.trans k1 - hc

theorem privateBatchWrapper2_f35 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9480)) (a (.wire 3 55)) (a (.virt 18999)) (a (.virt 19000)) := by
  have c219 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (privateBatchWrapper2_copies7 a h).1
  have c225 := (privateBatchWrapper2_copies7 a h).2.1
  have c226 := (privateBatchWrapper2_copies7 a h).2.2.1
  have c227 := (privateBatchWrapper2_copies7 a h).2.2.2.1
  have c228 := (privateBatchWrapper2_copies7 a h).2.2.2.2.1
  have c229 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.1
  have c230 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.1
  have c231 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c225, ← c226, ← c227] at e_2_10
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c228, ← c229, ← c230] at e_2_11
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c219, k0, ← c220, k0, ← c221] at e_4_10
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c222, ← c223, k0, ← c224] at e_4_11
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c231, ← c232, k0, ← c233] at e_4_12
  refine ⟨?_, ?_⟩
  · have hc := e_2_10
    simp only [e_4_11] at hc
    linear_combination c234.trans k1 - hc
  · have hc := e_4_12
    simp only [e_4_10, e_2_11, e_4_11] at hc
    linear_combination c235.trans k1 - hc

theorem privateBatchWrapper2_f36 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9481)) (a (.wire 4 3)) (a (.virt 19001)) (a (.virt 19002)) := by
  have c236 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c242, ← c243, ← c244] at e_2_12
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c245, ← c246, ← c247] at e_2_13
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c236, k0, ← c237, k0, ← c238] at e_4_13
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c239, ← c240, k0, ← c241] at e_4_14
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c248, ← c249, k0, ← c250] at e_7_0
  refine ⟨?_, ?_⟩
  · have hc := e_2_12
    simp only [e_4_14] at hc
    linear_combination c251.trans k1 - hc
  · have hc := e_7_0
    simp only [e_4_13, e_2_13, e_4_14] at hc
    linear_combination c252.trans k1 - hc

theorem privateBatchWrapper2_f37 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9482)) (a (.wire 4 11)) (a (.virt 19003)) (a (.virt 19004)) := by
  have c253 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (privateBatchWrapper2_copies8 a h).1
  have c257 := (privateBatchWrapper2_copies8 a h).2.1
  have c258 := (privateBatchWrapper2_copies8 a h).2.2.1
  have c259 := (privateBatchWrapper2_copies8 a h).2.2.2.1
  have c260 := (privateBatchWrapper2_copies8 a h).2.2.2.2.1
  have c261 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.1
  have c263 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c259, ← c260, ← c261] at e_2_14
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c253, k0, ← c254, k0, ← c255] at e_7_1
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c256, ← c257, k0, ← c258] at e_7_2
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c265, ← c266, k0, ← c267] at e_7_3
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c262, ← c263, ← c264] at e_8_0
  refine ⟨?_, ?_⟩
  · have hc := e_2_14
    simp only [e_7_2] at hc
    linear_combination c268.trans k1 - hc
  · have hc := e_7_3
    simp only [e_7_1, e_8_0, e_7_2] at hc
    linear_combination c269.trans k1 - hc

theorem privateBatchWrapper2_f38 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 7) = band (a (.virt 18997)) (a (.virt 18999)) := by
  have c270 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c270, ← c271, ← c272] at e_8_1
  have hr := e_8_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f39 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 11) = band (a (.virt 19001)) (a (.virt 19003)) := by
  have c273 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c273, ← c274, ← c275] at e_8_2
  have hr := e_8_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f40 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 15) = band (a (.wire 8 7)) (a (.wire 8 11)) := by
  have c276 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c278 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c276, ← c277, ← c278] at e_8_3
  have hr := e_8_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f41 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 7) = bor (a (.wire 1 43)) (a (.wire 8 15)) := by
  have c279 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c279, ← c280, ← c281] at e_5_1
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c282, ← c283, k0, ← c284] at e_6_1
  have hr := e_6_1
  simp only [e_5_1] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f42 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 7) = a (.virt 18978) := by
  have c285 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c285

theorem privateBatchWrapper2_f43 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 9463) = a (.virt 9463) := by
  rfl

theorem privateBatchWrapper2_f44 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9466)) (a (.wire 4 27)) (a (.virt 19005)) (a (.virt 19006)) := by
  have c287 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (privateBatchWrapper2_copies9 a h).1
  have c289 := (privateBatchWrapper2_copies9 a h).2.1
  have c290 := (privateBatchWrapper2_copies9 a h).2.2.1
  have c291 := (privateBatchWrapper2_copies9 a h).2.2.2.1
  have c292 := (privateBatchWrapper2_copies9 a h).2.2.2.2.1
  have c293 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.1
  have c294 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.1
  have c295 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c287, k0, ← c288, k0, ← c289] at e_7_4
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c290, ← c291, k0, ← c292] at e_7_5
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c299, ← c300, k0, ← c301] at e_7_6
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c293, ← c294, ← c295] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c296, ← c297, ← c298] at e_8_5
  refine ⟨?_, ?_⟩
  · have hc := e_8_4
    simp only [e_7_5] at hc
    linear_combination c302.trans k1 - hc
  · have hc := e_7_6
    simp only [e_7_4, e_8_5, e_7_5] at hc
    linear_combination c303.trans k1 - hc

theorem privateBatchWrapper2_f45 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 11) = bor (a (.wire 1 43)) (a (.virt 19005)) := by
  have c304 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c304, ← c305, ← c306] at e_5_2
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c307, ← c308, k0, ← c309] at e_6_2
  have hr := e_6_2
  simp only [e_5_2] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f46 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 11) = a (.virt 18978) := by
  have c310 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c310

theorem privateBatchWrapper2_f47 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18964)) (a (.wire 3 47)) (a (.virt 19007)) (a (.virt 19008)) := by
  have c311 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c317 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (privateBatchWrapper2_copies10 a h).1
  have c321 := (privateBatchWrapper2_copies10 a h).2.1
  have c322 := (privateBatchWrapper2_copies10 a h).2.2.1
  have c323 := (privateBatchWrapper2_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper2_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.1
  have c326 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.1
  have c327 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c311, k0, ← c312, k0, ← c313] at e_7_7
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c314, ← c315, k0, ← c316] at e_7_8
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c323, ← c324, k0, ← c325] at e_7_9
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c317, ← c318, ← c319] at e_8_6
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c320, ← c321, ← c322] at e_8_7
  refine ⟨?_, ?_⟩
  · have hc := e_8_6
    simp only [e_7_8] at hc
    linear_combination c326.trans k1 - hc
  · have hc := e_7_9
    simp only [e_7_7, e_8_7, e_7_8] at hc
    linear_combination c327.trans k1 - hc

theorem privateBatchWrapper2_f48 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18965)) (a (.wire 3 55)) (a (.virt 19009)) (a (.virt 19010)) := by
  have c328 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.1
  have c329 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c332 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c338 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c344 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c328, k0, ← c329, k0, ← c330] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c331, ← c332, k0, ← c333] at e_7_11
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c340, ← c341, k0, ← c342] at e_7_12
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c334, ← c335, ← c336] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c337, ← c338, ← c339] at e_8_9
  refine ⟨?_, ?_⟩
  · have hc := e_8_8
    simp only [e_7_11] at hc
    linear_combination c343.trans k1 - hc
  · have hc := e_7_12
    simp only [e_7_10, e_8_9, e_7_11] at hc
    linear_combination c344.trans k1 - hc

theorem privateBatchWrapper2_f49 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18966)) (a (.wire 4 3)) (a (.virt 19011)) (a (.virt 19012)) := by
  have c345 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (privateBatchWrapper2_copies11 a h).1
  have c353 := (privateBatchWrapper2_copies11 a h).2.1
  have c354 := (privateBatchWrapper2_copies11 a h).2.2.1
  have c355 := (privateBatchWrapper2_copies11 a h).2.2.2.1
  have c356 := (privateBatchWrapper2_copies11 a h).2.2.2.2.1
  have c357 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.1
  have c358 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.1
  have c359 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c345, k0, ← c346, k0, ← c347] at e_7_13
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c348, ← c349, k0, ← c350] at e_7_14
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c351, ← c352, ← c353] at e_8_10
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c354, ← c355, ← c356] at e_8_11
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c357, ← c358, k0, ← c359] at e_9_0
  refine ⟨?_, ?_⟩
  · have hc := e_8_10
    simp only [e_7_14] at hc
    linear_combination c360.trans k1 - hc
  · have hc := e_9_0
    simp only [e_7_13, e_8_11, e_7_14] at hc
    linear_combination c361.trans k1 - hc

theorem privateBatchWrapper2_f50 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18967)) (a (.wire 4 11)) (a (.virt 19013)) (a (.virt 19014)) := by
  have c362 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c368, ← c369, ← c370] at e_8_12
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c371, ← c372, ← c373] at e_8_13
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c362, k0, ← c363, k0, ← c364] at e_9_1
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c365, ← c366, k0, ← c367] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c374, ← c375, k0, ← c376] at e_9_3
  refine ⟨?_, ?_⟩
  · have hc := e_8_12
    simp only [e_9_2] at hc
    linear_combination c377.trans k1 - hc
  · have hc := e_9_3
    simp only [e_9_1, e_8_13, e_9_2] at hc
    linear_combination c378.trans k1 - hc

theorem privateBatchWrapper2_f51 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 59) = band (a (.virt 19007)) (a (.virt 19009)) := by
  have c379 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c379, ← c380, ← c381] at e_8_14
  have hr := e_8_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f52 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 3) = band (a (.virt 19011)) (a (.virt 19013)) := by
  have c382 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (privateBatchWrapper2_copies12 a h).1
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c382, ← c383, ← c384] at e_10_0
  have hr := e_10_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f53 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 7) = band (a (.wire 8 59)) (a (.wire 10 3)) := by
  have c385 := (privateBatchWrapper2_copies12 a h).2.1
  have c386 := (privateBatchWrapper2_copies12 a h).2.2.1
  have c387 := (privateBatchWrapper2_copies12 a h).2.2.2.1
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c385, ← c386, ← c387] at e_10_1
  have hr := e_10_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f54 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 15) = bor (a (.wire 2 27)) (a (.wire 10 7)) := by
  have c388 := (privateBatchWrapper2_copies12 a h).2.2.2.2.1
  have c389 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.1
  have c390 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.1
  have c391 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c388, ← c389, ← c390] at e_5_3
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c391, ← c392, k0, ← c393] at e_6_3
  have hr := e_6_3
  simp only [e_5_3] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f55 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 15) = a (.virt 18978) := by
  have c394 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c394

theorem privateBatchWrapper2_f56 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18948) = a (.virt 9463) := by
  have c395 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  exact c395

theorem privateBatchWrapper2_f57 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 18951)) (a (.wire 4 27)) (a (.virt 19015)) (a (.virt 19016)) := by
  have c396 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c401 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c405 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c396, k0, ← c397, k0, ← c398] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c399, ← c400, k0, ← c401] at e_9_5
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c408, ← c409, k0, ← c410] at e_9_6
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c402, ← c403, ← c404] at e_10_2
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c405, ← c406, ← c407] at e_10_3
  refine ⟨?_, ?_⟩
  · have hc := e_10_2
    simp only [e_9_5] at hc
    linear_combination c411.trans k1 - hc
  · have hc := e_9_6
    simp only [e_9_4, e_10_3, e_9_5] at hc
    linear_combination c412.trans k1 - hc

theorem privateBatchWrapper2_f58 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 19) = bor (a (.wire 2 27)) (a (.virt 19015)) := by
  have c413 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (privateBatchWrapper2_copies13 a h).1
  have c417 := (privateBatchWrapper2_copies13 a h).2.1
  have c418 := (privateBatchWrapper2_copies13 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c413, ← c414, ← c415] at e_5_4
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c416, ← c417, k0, ← c418] at e_6_4
  have hr := e_6_4
  simp only [e_5_4] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f59 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 19) = a (.virt 18978) := by
  have c419 := (privateBatchWrapper2_copies13 a h).2.2.2.1
  exact c419

theorem privateBatchWrapper2_f60 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 35) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9471)) := by
  have c420 := (privateBatchWrapper2_copies13 a h).2.2.2.2.1
  have c421 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.1
  have c422 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.1
  have c423 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c420, ← c421, ← c422] at e_9_7
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c423, ← c424, k1, ← c425] at e_9_8
  have hr := e_9_8
  simp only [e_9_7] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f61 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 43) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9472)) := by
  have c426 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c426, ← c427, ← c428] at e_9_9
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c429, ← c430, k1, ← c431] at e_9_10
  have hr := e_9_10
  simp only [e_9_9] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f62 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 51) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9473)) := by
  have c432 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c432, ← c433, ← c434] at e_9_11
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c435, ← c436, k1, ← c437] at e_9_12
  have hr := e_9_12
  simp only [e_9_11] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f63 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 59) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9474)) := by
  have c438 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c438, ← c439, ← c440] at e_9_13
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c441, ← c442, k1, ← c443] at e_9_14
  have hr := e_9_14
  simp only [e_9_13] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f64 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 7) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9464)) := by
  have c444 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (privateBatchWrapper2_copies14 a h).1
  have c449 := (privateBatchWrapper2_copies14 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c444, ← c445, ← c446] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c447, ← c448, k1, ← c449] at e_11_1
  have hr := e_11_1
  simp only [e_11_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f65 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 15) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9475)) := by
  have c450 := (privateBatchWrapper2_copies14 a h).2.2.1
  have c451 := (privateBatchWrapper2_copies14 a h).2.2.2.1
  have c452 := (privateBatchWrapper2_copies14 a h).2.2.2.2.1
  have c453 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.1
  have c454 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.1
  have c455 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c450, ← c451, ← c452] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c453, ← c454, k1, ← c455] at e_11_3
  have hr := e_11_3
  simp only [e_11_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f66 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 23) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9476)) := by
  have c456 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c456, ← c457, ← c458] at e_11_4
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c459, ← c460, k1, ← c461] at e_11_5
  have hr := e_11_5
  simp only [e_11_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f67 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 31) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9477)) := by
  have c462 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c462, ← c463, ← c464] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c465, ← c466, k1, ← c467] at e_11_7
  have hr := e_11_7
  simp only [e_11_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f68 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 39) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9478)) := by
  have c468 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c468, ← c469, ← c470] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c471, ← c472, k1, ← c473] at e_11_9
  have hr := e_11_9
  simp only [e_11_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f69 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 47) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9465)) := by
  have c474 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c474, ← c475, ← c476] at e_11_10
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c477, ← c478, k1, ← c479] at e_11_11
  have hr := e_11_11
  simp only [e_11_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f70 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 55) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18956)) := by
  have c480 := (privateBatchWrapper2_copies15 a h).1
  have c481 := (privateBatchWrapper2_copies15 a h).2.1
  have c482 := (privateBatchWrapper2_copies15 a h).2.2.1
  have c483 := (privateBatchWrapper2_copies15 a h).2.2.2.1
  have c484 := (privateBatchWrapper2_copies15 a h).2.2.2.2.1
  have c485 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c480, ← c481, ← c482] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c483, ← c484, k1, ← c485] at e_11_13
  have hr := e_11_13
  simp only [e_11_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f71 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 3) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18957)) := by
  have c486 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.1
  have c487 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c486, ← c487, ← c488] at e_11_14
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c489, ← c490, k1, ← c491] at e_12_0
  have hr := e_12_0
  simp only [e_11_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f72 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 11) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18958)) := by
  have c492 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c492, ← c493, ← c494] at e_12_1
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c495, ← c496, k1, ← c497] at e_12_2
  have hr := e_12_2
  simp only [e_12_1] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f73 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 19) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18959)) := by
  have c498 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c498, ← c499, ← c500] at e_12_3
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c501, ← c502, k1, ← c503] at e_12_4
  have hr := e_12_4
  simp only [e_12_3] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f74 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 27) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18949)) := by
  have c504 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c504, ← c505, ← c506] at e_12_5
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c507, ← c508, k1, ← c509] at e_12_6
  have hr := e_12_6
  simp only [e_12_5] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f75 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 35) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18960)) := by
  have c510 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (privateBatchWrapper2_copies16 a h).1
  have c513 := (privateBatchWrapper2_copies16 a h).2.1
  have c514 := (privateBatchWrapper2_copies16 a h).2.2.1
  have c515 := (privateBatchWrapper2_copies16 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c510, ← c511, ← c512] at e_12_7
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c513, ← c514, k1, ← c515] at e_12_8
  have hr := e_12_8
  simp only [e_12_7] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f76 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 43) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18961)) := by
  have c516 := (privateBatchWrapper2_copies16 a h).2.2.2.2.1
  have c517 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.1
  have c518 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.1
  have c519 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c516, ← c517, ← c518] at e_12_9
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c519, ← c520, k1, ← c521] at e_12_10
  have hr := e_12_10
  simp only [e_12_9] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f77 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 51) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18962)) := by
  have c522 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c522, ← c523, ← c524] at e_12_11
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c525, ← c526, k1, ← c527] at e_12_12
  have hr := e_12_12
  simp only [e_12_11] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f78 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 59) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18963)) := by
  have c528 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c532 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c528, ← c529, ← c530] at e_12_13
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c531, ← c532, k1, ← c533] at e_12_14
  have hr := e_12_14
  simp only [e_12_13] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f79 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 7) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18950)) := by
  have c534 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c535 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c534, ← c535, ← c536] at e_13_0
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c537, ← c538, k1, ← c539] at e_13_1
  have hr := e_13_1
  simp only [e_13_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f80 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 15) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9484)) := by
  have c540 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c541 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (privateBatchWrapper2_copies17 a h).1
  have c545 := (privateBatchWrapper2_copies17 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c540, ← c541, ← c542] at e_13_2
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c543, ← c544, k1, ← c545] at e_13_3
  have hr := e_13_3
  simp only [e_13_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f81 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 15) = a (.virt 18979) + a (.wire 13 15) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f82 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 23) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18969)) := by
  have c546 := (privateBatchWrapper2_copies17 a h).2.2.1
  have c547 := (privateBatchWrapper2_copies17 a h).2.2.2.1
  have c548 := (privateBatchWrapper2_copies17 a h).2.2.2.2.1
  have c549 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.1
  have c550 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.1
  have c551 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c546, ← c547, ← c548] at e_13_4
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c549, ← c550, k1, ← c551] at e_13_5
  have hr := e_13_5
  simp only [e_13_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f83 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 23) = a (.wire 13 15) + a (.wire 13 23) := by
  have c552 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c552, ← c553, k0, ← c554] at e_6_5
  have hr := e_6_5
  linear_combination hr

theorem privateBatchWrapper2_f84 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 7) = a (.virt 18979) + a (.wire 11 7) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f85 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 27) = a (.wire 11 7) + a (.wire 11 47) := by
  have c555 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c555, ← c556, k0, ← c557] at e_6_6
  have hr := e_6_6
  linear_combination hr

theorem privateBatchWrapper2_f86 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 31) = a (.wire 6 27) + a (.wire 12 27) := by
  have c558 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c558, ← c559, k0, ← c560] at e_6_7
  have hr := e_6_7
  linear_combination hr

theorem privateBatchWrapper2_f87 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 35) = a (.wire 6 31) + a (.wire 13 7) := by
  have c561 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c561, ← c562, k0, ← c563] at e_6_8
  have hr := e_6_8
  linear_combination hr

theorem privateBatchWrapper2_f88 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 27) = a (.virt 19017) - a (.wire 4 27) := by
  have c564 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c564, k3, ← c565, k0, ← c566] at e_13_6
  have hr := e_13_6
  simp only [k3]
  linear_combination hr

theorem privateBatchWrapper2_f89 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 13 27)) 14 := by
  have c567 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c569 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c572 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c575 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (privateBatchWrapper2_copies18 a h).1
  have c577 := (privateBatchWrapper2_copies18 a h).2.1
  have c578 := (privateBatchWrapper2_copies18 a h).2.2.1
  have c579 := (privateBatchWrapper2_copies18 a h).2.2.2.1
  have c580 := (privateBatchWrapper2_copies18 a h).2.2.2.2.1
  have c581 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.1
  have c582 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.1
  have c583 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c600 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (privateBatchWrapper2_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (privateBatchWrapper2_copies19 a h).1
  have c609 := (privateBatchWrapper2_copies19 a h).2.1
  have c610 := (privateBatchWrapper2_copies19 a h).2.2.1
  have c611 := (privateBatchWrapper2_copies19 a h).2.2.2.1
  have c612 := (privateBatchWrapper2_copies19 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 14) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c567.trans k1
    · exact c568.trans k1
    · exact c569.trans k1
    · exact c570.trans k1
    · exact c571.trans k1
    · exact c572.trans k1
    · exact c573.trans k1
    · exact c574.trans k1
    · exact c575.trans k1
    · exact c576.trans k1
    · exact c577.trans k1
    · exact c578.trans k1
    · exact c579.trans k1
    · exact c580.trans k1
    · exact c581.trans k1
    · exact c582.trans k1
    · exact c583.trans k1
    · exact c584.trans k1
    · exact c585.trans k1
    · exact c586.trans k1
    · exact c587.trans k1
    · exact c588.trans k1
    · exact c589.trans k1
    · exact c590.trans k1
    · exact c591.trans k1
    · exact c592.trans k1
    · exact c593.trans k1
    · exact c594.trans k1
    · exact c595.trans k1
    · exact c596.trans k1
    · exact c597.trans k1
    · exact c598.trans k1
    · exact c599.trans k1
    · exact c600.trans k1
    · exact c601.trans k1
    · exact c602.trans k1
    · exact c603.trans k1
    · exact c604.trans k1
    · exact c605.trans k1
    · exact c606.trans k1
    · exact c607.trans k1
    · exact c608.trans k1
    · exact c609.trans k1
    · exact c610.trans k1
    · exact c611.trans k1
    )
  rwa [c612] at hr

theorem privateBatchWrapper2_f90 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 19) = a (.wire 6 35) * a (.virt 19017) := by
  have c613 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.1
  have c614 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.1
  have c615 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c613, ← c614, k3, ← c615] at e_10_4
  have hr := e_10_4
  simp only [k3]
  linear_combination hr

theorem privateBatchWrapper2_f91 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 23) = a (.wire 6 23) * a (.wire 13 27) := by
  have c616 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c616, ← c617, ← c618] at e_10_5
  have hr := e_10_5
  linear_combination hr

theorem privateBatchWrapper2_f92 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 31) = a (.wire 10 23) - a (.wire 10 19) := by
  have c619 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c619, ← c620, k0, ← c621] at e_13_7
  have hr := e_13_7
  linear_combination hr

theorem privateBatchWrapper2_f93 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 13 31)) 52 := by
  have c622 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 15) (N := 59) (n := 52) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c622.trans k1
    · exact c623.trans k1
    · exact c624.trans k1
    · exact c625.trans k1
    · exact c626.trans k1
    · exact c627.trans k1
    · exact c628.trans k1
    )
  rwa [c629] at hr

theorem privateBatchWrapper2_f94 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 9 35)) (a (.virt 19019)) (a (.virt 19020)) := by
  have c630 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c634 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (privateBatchWrapper2_copies20 a h).1
  have c641 := (privateBatchWrapper2_copies20 a h).2.1
  have c642 := (privateBatchWrapper2_copies20 a h).2.2.1
  have c643 := (privateBatchWrapper2_copies20 a h).2.2.2.1
  have c644 := (privateBatchWrapper2_copies20 a h).2.2.2.2.1
  have c645 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.1
  have c646 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c636, ← c637, ← c638] at e_10_6
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c639, ← c640, ← c641] at e_10_7
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c630, k0, ← c631, k0, ← c632] at e_13_8
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c633, ← c634, k0, ← c635] at e_13_9
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c642, ← c643, k0, ← c644] at e_13_10
  refine ⟨?_, ?_⟩
  · have hc := e_10_6
    simp only [e_13_9] at hc
    linear_combination c645.trans k1 - hc
  · have hc := e_13_10
    simp only [e_13_8, e_10_7, e_13_9] at hc
    linear_combination c646.trans k1 - hc

theorem privateBatchWrapper2_f95 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 9 43)) (a (.virt 19021)) (a (.virt 19022)) := by
  have c647 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  have c651 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c654 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c657 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c653, ← c654, ← c655] at e_10_8
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c656, ← c657, ← c658] at e_10_9
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c647, k0, ← c648, k0, ← c649] at e_13_11
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c650, ← c651, k0, ← c652] at e_13_12
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c659, ← c660, k0, ← c661] at e_13_13
  refine ⟨?_, ?_⟩
  · have hc := e_10_8
    simp only [e_13_12] at hc
    linear_combination c662.trans k1 - hc
  · have hc := e_13_13
    simp only [e_13_11, e_10_9, e_13_12] at hc
    linear_combination c663.trans k1 - hc

theorem privateBatchWrapper2_f96 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 9 51)) (a (.virt 19023)) (a (.virt 19024)) := by
  have c664 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (privateBatchWrapper2_copies21 a h).1
  have c673 := (privateBatchWrapper2_copies21 a h).2.1
  have c674 := (privateBatchWrapper2_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper2_copies21 a h).2.2.2.1
  have c676 := (privateBatchWrapper2_copies21 a h).2.2.2.2.1
  have c677 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.1
  have c678 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.1
  have c679 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c670, ← c671, ← c672] at e_10_10
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c673, ← c674, ← c675] at e_10_11
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c664, k0, ← c665, k0, ← c666] at e_13_14
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c667, ← c668, k0, ← c669] at e_16_0
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c676, ← c677, k0, ← c678] at e_16_1
  refine ⟨?_, ?_⟩
  · have hc := e_10_10
    simp only [e_16_0] at hc
    linear_combination c679.trans k1 - hc
  · have hc := e_16_1
    simp only [e_13_14, e_10_11, e_16_0] at hc
    linear_combination c680.trans k1 - hc

theorem privateBatchWrapper2_f97 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 9 59)) (a (.virt 19025)) (a (.virt 19026)) := by
  have c681 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c685 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c687, ← c688, ← c689] at e_10_12
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c690, ← c691, ← c692] at e_10_13
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c681, k0, ← c682, k0, ← c683] at e_16_2
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c684, ← c685, k0, ← c686] at e_16_3
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c693, ← c694, k0, ← c695] at e_16_4
  refine ⟨?_, ?_⟩
  · have hc := e_10_12
    simp only [e_16_3] at hc
    linear_combination c696.trans k1 - hc
  · have hc := e_16_4
    simp only [e_16_2, e_10_13, e_16_3] at hc
    linear_combination c697.trans k1 - hc

theorem privateBatchWrapper2_f98 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 59) = band (a (.virt 19019)) (a (.virt 19021)) := by
  have c698 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c698, ← c699, ← c700] at e_10_14
  have hr := e_10_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f99 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 3) = band (a (.virt 19023)) (a (.virt 19025)) := by
  have c701 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_17_0 := arithEq_of_rows h (row := 17) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c701, ← c702, ← c703] at e_17_0
  have hr := e_17_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f100 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 7) = band (a (.wire 10 59)) (a (.wire 17 3)) := by
  have c704 := (privateBatchWrapper2_copies22 a h).1
  have c705 := (privateBatchWrapper2_copies22 a h).2.1
  have c706 := (privateBatchWrapper2_copies22 a h).2.2.1
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c704, ← c705, ← c706] at e_17_1
  have hr := e_17_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f101 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 23) = bselect (a (.wire 17 7)) (a (.wire 11 7)) (a (.virt 18979)) := by
  have c707 := (privateBatchWrapper2_copies22 a h).2.2.2.1
  have c708 := (privateBatchWrapper2_copies22 a h).2.2.2.2.1
  have c709 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c707, ← c708, ← c709, k1] at e_16_5
  have hr := e_16_5
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f102 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 23) = a (.virt 18979) + a (.wire 16 23) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f103 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 15)) (a (.wire 9 35)) (a (.virt 19027)) (a (.virt 19028)) := by
  have c710 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.1
  have c711 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c710, k0, ← c711, k0, ← c712] at e_16_6
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c713, ← c714, k0, ← c715] at e_16_7
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c722, ← c723, k0, ← c724] at e_16_8
  have e_17_2 := arithEq_of_rows h (row := 17) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c716, ← c717, ← c718] at e_17_2
  have e_17_3 := arithEq_of_rows h (row := 17) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c719, ← c720, ← c721] at e_17_3
  refine ⟨?_, ?_⟩
  · have hc := e_17_2
    simp only [e_16_7] at hc
    linear_combination c725.trans k1 - hc
  · have hc := e_16_8
    simp only [e_16_6, e_17_3, e_16_7] at hc
    linear_combination c726.trans k1 - hc

theorem privateBatchWrapper2_f104 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 23)) (a (.wire 9 43)) (a (.virt 19029)) (a (.virt 19030)) := by
  have c727 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c733 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c736 := (privateBatchWrapper2_copies23 a h).1
  have c737 := (privateBatchWrapper2_copies23 a h).2.1
  have c738 := (privateBatchWrapper2_copies23 a h).2.2.1
  have c739 := (privateBatchWrapper2_copies23 a h).2.2.2.1
  have c740 := (privateBatchWrapper2_copies23 a h).2.2.2.2.1
  have c741 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.1
  have c742 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.1
  have c743 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c727, k0, ← c728, k0, ← c729] at e_16_9
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c730, ← c731, k0, ← c732] at e_16_10
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c739, ← c740, k0, ← c741] at e_16_11
  have e_17_4 := arithEq_of_rows h (row := 17) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c733, ← c734, ← c735] at e_17_4
  have e_17_5 := arithEq_of_rows h (row := 17) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c736, ← c737, ← c738] at e_17_5
  refine ⟨?_, ?_⟩
  · have hc := e_17_4
    simp only [e_16_10] at hc
    linear_combination c742.trans k1 - hc
  · have hc := e_16_11
    simp only [e_16_9, e_17_5, e_16_10] at hc
    linear_combination c743.trans k1 - hc

theorem privateBatchWrapper2_f105 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 31)) (a (.wire 9 51)) (a (.virt 19031)) (a (.virt 19032)) := by
  have c744 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.1
  have c745 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c751 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c752 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c753 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c759 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c744, k0, ← c745, k0, ← c746] at e_16_12
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c747, ← c748, k0, ← c749] at e_16_13
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c756, ← c757, k0, ← c758] at e_16_14
  have e_17_6 := arithEq_of_rows h (row := 17) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c750, ← c751, ← c752] at e_17_6
  have e_17_7 := arithEq_of_rows h (row := 17) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c753, ← c754, ← c755] at e_17_7
  refine ⟨?_, ?_⟩
  · have hc := e_17_6
    simp only [e_16_13] at hc
    linear_combination c759.trans k1 - hc
  · have hc := e_16_14
    simp only [e_16_12, e_17_7, e_16_13] at hc
    linear_combination c760.trans k1 - hc

theorem privateBatchWrapper2_f106 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 39)) (a (.wire 9 59)) (a (.virt 19033)) (a (.virt 19034)) := by
  have c761 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c765 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (privateBatchWrapper2_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c768 := (privateBatchWrapper2_copies24 a h).1
  have c769 := (privateBatchWrapper2_copies24 a h).2.1
  have c770 := (privateBatchWrapper2_copies24 a h).2.2.1
  have c771 := (privateBatchWrapper2_copies24 a h).2.2.2.1
  have c772 := (privateBatchWrapper2_copies24 a h).2.2.2.2.1
  have c773 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.1
  have c774 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.1
  have c775 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.1
  have c777 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_17_8 := arithEq_of_rows h (row := 17) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c767, ← c768, ← c769] at e_17_8
  have e_17_9 := arithEq_of_rows h (row := 17) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c770, ← c771, ← c772] at e_17_9
  have e_18_0 := arithEq_of_rows h (row := 18) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_0
  simp only [← c761, k0, ← c762, k0, ← c763] at e_18_0
  have e_18_1 := arithEq_of_rows h (row := 18) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_1
  simp only [← c764, ← c765, k0, ← c766] at e_18_1
  have e_18_2 := arithEq_of_rows h (row := 18) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_2
  simp only [← c773, ← c774, k0, ← c775] at e_18_2
  refine ⟨?_, ?_⟩
  · have hc := e_17_8
    simp only [e_18_1] at hc
    linear_combination c776.trans k1 - hc
  · have hc := e_18_2
    simp only [e_18_0, e_17_9, e_18_1] at hc
    linear_combination c777.trans k1 - hc

theorem privateBatchWrapper2_f107 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 43) = band (a (.virt 19027)) (a (.virt 19029)) := by
  have c778 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_10 := arithEq_of_rows h (row := 17) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c778, ← c779, ← c780] at e_17_10
  have hr := e_17_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f108 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 47) = band (a (.virt 19031)) (a (.virt 19033)) := by
  have c781 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c783 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_11 := arithEq_of_rows h (row := 17) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c781, ← c782, ← c783] at e_17_11
  have hr := e_17_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f109 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 17 51) = band (a (.wire 17 43)) (a (.wire 17 47)) := by
  have c784 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c786 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_12 := arithEq_of_rows h (row := 17) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c784, ← c785, ← c786] at e_17_12
  have hr := e_17_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f110 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 18 15) = bselect (a (.wire 17 51)) (a (.wire 11 47)) (a (.virt 18979)) := by
  have c787 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_18_3 := arithEq_of_rows h (row := 18) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_3
  simp only [← c787, ← c788, ← c789, k1] at e_18_3
  have hr := e_18_3
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f111 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 39) = a (.wire 16 23) + a (.wire 18 15) := by
  have c790 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c790, ← c791, k0, ← c792] at e_6_9
  have hr := e_6_9
  linear_combination hr

theorem privateBatchWrapper2_f112 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 55)) (a (.wire 9 35)) (a (.virt 19035)) (a (.virt 19036)) := by
  have c793 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (privateBatchWrapper2_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (privateBatchWrapper2_copies25 a h).1
  have c801 := (privateBatchWrapper2_copies25 a h).2.1
  have c802 := (privateBatchWrapper2_copies25 a h).2.2.1
  have c803 := (privateBatchWrapper2_copies25 a h).2.2.2.1
  have c804 := (privateBatchWrapper2_copies25 a h).2.2.2.2.1
  have c805 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.1
  have c806 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.1
  have c807 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c799, ← c800, ← c801] at e_17_13
  have e_17_14 := arithEq_of_rows h (row := 17) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c802, ← c803, ← c804] at e_17_14
  have e_18_4 := arithEq_of_rows h (row := 18) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_4
  simp only [← c793, k0, ← c794, k0, ← c795] at e_18_4
  have e_18_5 := arithEq_of_rows h (row := 18) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_5
  simp only [← c796, ← c797, k0, ← c798] at e_18_5
  have e_18_6 := arithEq_of_rows h (row := 18) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_6
  simp only [← c805, ← c806, k0, ← c807] at e_18_6
  refine ⟨?_, ?_⟩
  · have hc := e_17_13
    simp only [e_18_5] at hc
    linear_combination c808.trans k1 - hc
  · have hc := e_18_6
    simp only [e_18_4, e_17_14, e_18_5] at hc
    linear_combination c809.trans k1 - hc

theorem privateBatchWrapper2_f113 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 3)) (a (.wire 9 43)) (a (.virt 19037)) (a (.virt 19038)) := by
  have c810 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c814 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_18_7 := arithEq_of_rows h (row := 18) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_7
  simp only [← c810, k0, ← c811, k0, ← c812] at e_18_7
  have e_18_8 := arithEq_of_rows h (row := 18) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_8
  simp only [← c813, ← c814, k0, ← c815] at e_18_8
  have e_18_9 := arithEq_of_rows h (row := 18) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_9
  simp only [← c822, ← c823, k0, ← c824] at e_18_9
  have e_19_0 := arithEq_of_rows h (row := 19) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_0
  simp only [← c816, ← c817, ← c818] at e_19_0
  have e_19_1 := arithEq_of_rows h (row := 19) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_1
  simp only [← c819, ← c820, ← c821] at e_19_1
  refine ⟨?_, ?_⟩
  · have hc := e_19_0
    simp only [e_18_8] at hc
    linear_combination c825.trans k1 - hc
  · have hc := e_18_9
    simp only [e_18_7, e_19_1, e_18_8] at hc
    linear_combination c826.trans k1 - hc

theorem privateBatchWrapper2_f114 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 11)) (a (.wire 9 51)) (a (.virt 19039)) (a (.virt 19040)) := by
  have c827 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c829 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (privateBatchWrapper2_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c832 := (privateBatchWrapper2_copies26 a h).1
  have c833 := (privateBatchWrapper2_copies26 a h).2.1
  have c834 := (privateBatchWrapper2_copies26 a h).2.2.1
  have c835 := (privateBatchWrapper2_copies26 a h).2.2.2.1
  have c836 := (privateBatchWrapper2_copies26 a h).2.2.2.2.1
  have c837 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.1
  have c838 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.1
  have c839 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.1
  have c841 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.1
  have c842 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_18_10 := arithEq_of_rows h (row := 18) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_10
  simp only [← c827, k0, ← c828, k0, ← c829] at e_18_10
  have e_18_11 := arithEq_of_rows h (row := 18) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_11
  simp only [← c830, ← c831, k0, ← c832] at e_18_11
  have e_18_12 := arithEq_of_rows h (row := 18) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_12
  simp only [← c839, ← c840, k0, ← c841] at e_18_12
  have e_19_2 := arithEq_of_rows h (row := 19) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_2
  simp only [← c833, ← c834, ← c835] at e_19_2
  have e_19_3 := arithEq_of_rows h (row := 19) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_3
  simp only [← c836, ← c837, ← c838] at e_19_3
  refine ⟨?_, ?_⟩
  · have hc := e_19_2
    simp only [e_18_11] at hc
    linear_combination c842.trans k1 - hc
  · have hc := e_18_12
    simp only [e_18_10, e_19_3, e_18_11] at hc
    linear_combination c843.trans k1 - hc

theorem privateBatchWrapper2_f115 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 19)) (a (.wire 9 59)) (a (.virt 19041)) (a (.virt 19042)) := by
  have c844 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c845 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c847 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c853 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c860 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_18_13 := arithEq_of_rows h (row := 18) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_13
  simp only [← c844, k0, ← c845, k0, ← c846] at e_18_13
  have e_18_14 := arithEq_of_rows h (row := 18) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_18_14
  simp only [← c847, ← c848, k0, ← c849] at e_18_14
  have e_19_4 := arithEq_of_rows h (row := 19) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_4
  simp only [← c850, ← c851, ← c852] at e_19_4
  have e_19_5 := arithEq_of_rows h (row := 19) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_5
  simp only [← c853, ← c854, ← c855] at e_19_5
  have e_20_0 := arithEq_of_rows h (row := 20) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_0
  simp only [← c856, ← c857, k0, ← c858] at e_20_0
  refine ⟨?_, ?_⟩
  · have hc := e_19_4
    simp only [e_18_14] at hc
    linear_combination c859.trans k1 - hc
  · have hc := e_20_0
    simp only [e_18_13, e_19_5, e_18_14] at hc
    linear_combination c860.trans k1 - hc

theorem privateBatchWrapper2_f116 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 19 27) = band (a (.virt 19035)) (a (.virt 19037)) := by
  have c861 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c863 := (privateBatchWrapper2_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_19_6 := arithEq_of_rows h (row := 19) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_6
  simp only [← c861, ← c862, ← c863] at e_19_6
  have hr := e_19_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f117 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 19 31) = band (a (.virt 19039)) (a (.virt 19041)) := by
  have c864 := (privateBatchWrapper2_copies27 a h).1
  have c865 := (privateBatchWrapper2_copies27 a h).2.1
  have c866 := (privateBatchWrapper2_copies27 a h).2.2.1
  have e_19_7 := arithEq_of_rows h (row := 19) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_7
  simp only [← c864, ← c865, ← c866] at e_19_7
  have hr := e_19_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f118 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 19 35) = band (a (.wire 19 27)) (a (.wire 19 31)) := by
  have c867 := (privateBatchWrapper2_copies27 a h).2.2.2.1
  have c868 := (privateBatchWrapper2_copies27 a h).2.2.2.2.1
  have c869 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.1
  have e_19_8 := arithEq_of_rows h (row := 19) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_8
  simp only [← c867, ← c868, ← c869] at e_19_8
  have hr := e_19_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f119 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 20 7) = bselect (a (.wire 19 35)) (a (.wire 12 27)) (a (.virt 18979)) := by
  have c870 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.1
  have c871 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_20_1 := arithEq_of_rows h (row := 20) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_1
  simp only [← c870, ← c871, ← c872, k1] at e_20_1
  have hr := e_20_1
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f120 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 43) = a (.wire 6 39) + a (.wire 20 7) := by
  have c873 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c873, ← c874, k0, ← c875] at e_6_10
  have hr := e_6_10
  linear_combination hr

theorem privateBatchWrapper2_f121 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 35)) (a (.wire 9 35)) (a (.virt 19043)) (a (.virt 19044)) := by
  have c876 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c880 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c883 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_19_9 := arithEq_of_rows h (row := 19) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_9
  simp only [← c882, ← c883, ← c884] at e_19_9
  have e_19_10 := arithEq_of_rows h (row := 19) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_10
  simp only [← c885, ← c886, ← c887] at e_19_10
  have e_20_2 := arithEq_of_rows h (row := 20) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_2
  simp only [← c876, k0, ← c877, k0, ← c878] at e_20_2
  have e_20_3 := arithEq_of_rows h (row := 20) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_3
  simp only [← c879, ← c880, k0, ← c881] at e_20_3
  have e_20_4 := arithEq_of_rows h (row := 20) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_4
  simp only [← c888, ← c889, k0, ← c890] at e_20_4
  refine ⟨?_, ?_⟩
  · have hc := e_19_9
    simp only [e_20_3] at hc
    linear_combination c891.trans k1 - hc
  · have hc := e_20_4
    simp only [e_20_2, e_19_10, e_20_3] at hc
    linear_combination c892.trans k1 - hc

theorem privateBatchWrapper2_f122 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 43)) (a (.wire 9 43)) (a (.virt 19045)) (a (.virt 19046)) := by
  have c893 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (privateBatchWrapper2_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (privateBatchWrapper2_copies28 a h).1
  have c897 := (privateBatchWrapper2_copies28 a h).2.1
  have c898 := (privateBatchWrapper2_copies28 a h).2.2.1
  have c899 := (privateBatchWrapper2_copies28 a h).2.2.2.1
  have c900 := (privateBatchWrapper2_copies28 a h).2.2.2.2.1
  have c901 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.1
  have c902 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.1
  have c903 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.1
  have c904 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  have c907 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_19_11 := arithEq_of_rows h (row := 19) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_11
  simp only [← c899, ← c900, ← c901] at e_19_11
  have e_19_12 := arithEq_of_rows h (row := 19) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_12
  simp only [← c902, ← c903, ← c904] at e_19_12
  have e_20_5 := arithEq_of_rows h (row := 20) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_5
  simp only [← c893, k0, ← c894, k0, ← c895] at e_20_5
  have e_20_6 := arithEq_of_rows h (row := 20) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_6
  simp only [← c896, ← c897, k0, ← c898] at e_20_6
  have e_20_7 := arithEq_of_rows h (row := 20) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_7
  simp only [← c905, ← c906, k0, ← c907] at e_20_7
  refine ⟨?_, ?_⟩
  · have hc := e_19_11
    simp only [e_20_6] at hc
    linear_combination c908.trans k1 - hc
  · have hc := e_20_7
    simp only [e_20_5, e_19_12, e_20_6] at hc
    linear_combination c909.trans k1 - hc

theorem privateBatchWrapper2_f123 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 51)) (a (.wire 9 51)) (a (.virt 19047)) (a (.virt 19048)) := by
  have c910 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c925 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_19_13 := arithEq_of_rows h (row := 19) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_13
  simp only [← c916, ← c917, ← c918] at e_19_13
  have e_19_14 := arithEq_of_rows h (row := 19) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_19_14
  simp only [← c919, ← c920, ← c921] at e_19_14
  have e_20_8 := arithEq_of_rows h (row := 20) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_8
  simp only [← c910, k0, ← c911, k0, ← c912] at e_20_8
  have e_20_9 := arithEq_of_rows h (row := 20) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_9
  simp only [← c913, ← c914, k0, ← c915] at e_20_9
  have e_20_10 := arithEq_of_rows h (row := 20) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_10
  simp only [← c922, ← c923, k0, ← c924] at e_20_10
  refine ⟨?_, ?_⟩
  · have hc := e_19_13
    simp only [e_20_9] at hc
    linear_combination c925.trans k1 - hc
  · have hc := e_20_10
    simp only [e_20_8, e_19_14, e_20_9] at hc
    linear_combination c926.trans k1 - hc

theorem privateBatchWrapper2_f124 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 59)) (a (.wire 9 59)) (a (.virt 19049)) (a (.virt 19050)) := by
  have c927 := (privateBatchWrapper2_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c928 := (privateBatchWrapper2_copies29 a h).1
  have c929 := (privateBatchWrapper2_copies29 a h).2.1
  have c930 := (privateBatchWrapper2_copies29 a h).2.2.1
  have c931 := (privateBatchWrapper2_copies29 a h).2.2.2.1
  have c932 := (privateBatchWrapper2_copies29 a h).2.2.2.2.1
  have c933 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.1
  have c934 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.1
  have c935 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.1
  have c937 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_20_11 := arithEq_of_rows h (row := 20) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_11
  simp only [← c927, k0, ← c928, k0, ← c929] at e_20_11
  have e_20_12 := arithEq_of_rows h (row := 20) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_12
  simp only [← c930, ← c931, k0, ← c932] at e_20_12
  have e_20_13 := arithEq_of_rows h (row := 20) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_13
  simp only [← c939, ← c940, k0, ← c941] at e_20_13
  have e_21_0 := arithEq_of_rows h (row := 21) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c933, ← c934, ← c935] at e_21_0
  have e_21_1 := arithEq_of_rows h (row := 21) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c936, ← c937, ← c938] at e_21_1
  refine ⟨?_, ?_⟩
  · have hc := e_21_0
    simp only [e_20_12] at hc
    linear_combination c942.trans k1 - hc
  · have hc := e_20_13
    simp only [e_20_11, e_21_1, e_20_12] at hc
    linear_combination c943.trans k1 - hc

theorem privateBatchWrapper2_f125 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 21 11) = band (a (.virt 19043)) (a (.virt 19045)) := by
  have c944 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_21_2 := arithEq_of_rows h (row := 21) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_2
  simp only [← c944, ← c945, ← c946] at e_21_2
  have hr := e_21_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f126 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 21 15) = band (a (.virt 19047)) (a (.virt 19049)) := by
  have c947 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_21_3 := arithEq_of_rows h (row := 21) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_3
  simp only [← c947, ← c948, ← c949] at e_21_3
  have hr := e_21_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f127 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 21 19) = band (a (.wire 21 11)) (a (.wire 21 15)) := by
  have c950 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_21_4 := arithEq_of_rows h (row := 21) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_4
  simp only [← c950, ← c951, ← c952] at e_21_4
  have hr := e_21_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f128 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 20 59) = bselect (a (.wire 21 19)) (a (.wire 13 7)) (a (.virt 18979)) := by
  have c953 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_20_14 := arithEq_of_rows h (row := 20) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_20_14
  simp only [← c953, ← c954, ← c955, k1] at e_20_14
  have hr := e_20_14
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f129 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 47) = a (.wire 6 43) + a (.wire 20 59) := by
  have c956 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c956, ← c957, k0, ← c958] at e_6_11
  have hr := e_6_11
  linear_combination hr

theorem privateBatchWrapper2_f130 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 22 7) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 6 47)) := by
  have c959 := (privateBatchWrapper2_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (privateBatchWrapper2_copies30 a h).1
  have c961 := (privateBatchWrapper2_copies30 a h).2.1
  have c962 := (privateBatchWrapper2_copies30 a h).2.2.1
  have c963 := (privateBatchWrapper2_copies30 a h).2.2.2.1
  have c964 := (privateBatchWrapper2_copies30 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_0 := arithEq_of_rows h (row := 22) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_0
  simp only [← c959, k1, ← c960, ← c961] at e_22_0
  have e_22_1 := arithEq_of_rows h (row := 22) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_1
  simp only [← c962, k1, ← c963, k1, ← c964] at e_22_1
  have hr := e_22_1
  simp only [e_22_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f131 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 22 15) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 35)) := by
  have c965 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.1
  have c966 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.1
  have c967 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_2 := arithEq_of_rows h (row := 22) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_2
  simp only [← c965, k1, ← c966, ← c967] at e_22_2
  have e_22_3 := arithEq_of_rows h (row := 22) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_3
  simp only [← c968, k1, ← c969, k1, ← c970] at e_22_3
  have hr := e_22_3
  simp only [e_22_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f132 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 22 23) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 43)) := by
  have c971 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_4 := arithEq_of_rows h (row := 22) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_4
  simp only [← c971, k1, ← c972, ← c973] at e_22_4
  have e_22_5 := arithEq_of_rows h (row := 22) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_5
  simp only [← c974, k1, ← c975, k1, ← c976] at e_22_5
  have hr := e_22_5
  simp only [e_22_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f133 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 22 31) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 51)) := by
  have c977 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_6 := arithEq_of_rows h (row := 22) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_6
  simp only [← c977, k1, ← c978, ← c979] at e_22_6
  have e_22_7 := arithEq_of_rows h (row := 22) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_7
  simp only [← c980, k1, ← c981, k1, ← c982] at e_22_7
  have hr := e_22_7
  simp only [e_22_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f134 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 22 39) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 59)) := by
  have c983 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c987 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_8 := arithEq_of_rows h (row := 22) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_8
  simp only [← c983, k1, ← c984, ← c985] at e_22_8
  have e_22_9 := arithEq_of_rows h (row := 22) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_9
  simp only [← c986, k1, ← c987, k1, ← c988] at e_22_9
  have hr := e_22_9
  simp only [e_22_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f135 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 22 7)) 32 := by
  have c989 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c990 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (privateBatchWrapper2_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (privateBatchWrapper2_copies31 a h).1
  have c993 := (privateBatchWrapper2_copies31 a h).2.1
  have c994 := (privateBatchWrapper2_copies31 a h).2.2.1
  have c995 := (privateBatchWrapper2_copies31 a h).2.2.2.1
  have c996 := (privateBatchWrapper2_copies31 a h).2.2.2.2.1
  have c997 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.1
  have c998 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.1
  have c999 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.1
  have c1002 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1005 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1008 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1011 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1012 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 23) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c989.trans k1
    · exact c990.trans k1
    · exact c991.trans k1
    · exact c992.trans k1
    · exact c993.trans k1
    · exact c994.trans k1
    · exact c995.trans k1
    · exact c996.trans k1
    · exact c997.trans k1
    · exact c998.trans k1
    · exact c999.trans k1
    · exact c1000.trans k1
    · exact c1001.trans k1
    · exact c1002.trans k1
    · exact c1003.trans k1
    · exact c1004.trans k1
    · exact c1005.trans k1
    · exact c1006.trans k1
    · exact c1007.trans k1
    · exact c1008.trans k1
    · exact c1009.trans k1
    · exact c1010.trans k1
    · exact c1011.trans k1
    · exact c1012.trans k1
    · exact c1013.trans k1
    · exact c1014.trans k1
    · exact c1015.trans k1
    )
  rwa [c1016] at hr

theorem privateBatchWrapper2_f136 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 11 15)) (a (.virt 19051)) (a (.virt 19052)) := by
  have c1017 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1019 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1022 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1023 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1024 := (privateBatchWrapper2_copies32 a h).1
  have c1025 := (privateBatchWrapper2_copies32 a h).2.1
  have c1026 := (privateBatchWrapper2_copies32 a h).2.2.1
  have c1027 := (privateBatchWrapper2_copies32 a h).2.2.2.1
  have c1028 := (privateBatchWrapper2_copies32 a h).2.2.2.2.1
  have c1029 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.1
  have c1030 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_21_5 := arithEq_of_rows h (row := 21) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_5
  simp only [← c1023, ← c1024, ← c1025] at e_21_5
  have e_21_6 := arithEq_of_rows h (row := 21) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_6
  simp only [← c1026, ← c1027, ← c1028] at e_21_6
  have e_22_10 := arithEq_of_rows h (row := 22) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_10
  simp only [← c1017, k0, ← c1018, k0, ← c1019] at e_22_10
  have e_22_11 := arithEq_of_rows h (row := 22) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c1020, ← c1021, k0, ← c1022] at e_22_11
  have e_22_12 := arithEq_of_rows h (row := 22) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_12
  simp only [← c1029, ← c1030, k0, ← c1031] at e_22_12
  refine ⟨?_, ?_⟩
  · have hc := e_21_5
    simp only [e_22_11] at hc
    linear_combination c1032.trans k1 - hc
  · have hc := e_22_12
    simp only [e_22_10, e_21_6, e_22_11] at hc
    linear_combination c1033.trans k1 - hc

theorem privateBatchWrapper2_f137 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 11 23)) (a (.virt 19053)) (a (.virt 19054)) := by
  have c1034 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1041 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_21_7 := arithEq_of_rows h (row := 21) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_7
  simp only [← c1040, ← c1041, ← c1042] at e_21_7
  have e_21_8 := arithEq_of_rows h (row := 21) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_8
  simp only [← c1043, ← c1044, ← c1045] at e_21_8
  have e_22_13 := arithEq_of_rows h (row := 22) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_13
  simp only [← c1034, k0, ← c1035, k0, ← c1036] at e_22_13
  have e_22_14 := arithEq_of_rows h (row := 22) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1037, ← c1038, k0, ← c1039] at e_22_14
  have e_24_0 := arithEq_of_rows h (row := 24) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_0
  simp only [← c1046, ← c1047, k0, ← c1048] at e_24_0
  refine ⟨?_, ?_⟩
  · have hc := e_21_7
    simp only [e_22_14] at hc
    linear_combination c1049.trans k1 - hc
  · have hc := e_24_0
    simp only [e_22_13, e_21_8, e_22_14] at hc
    linear_combination c1050.trans k1 - hc

theorem privateBatchWrapper2_f138 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 11 31)) (a (.virt 19055)) (a (.virt 19056)) := by
  have c1051 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (privateBatchWrapper2_copies33 a h).1
  have c1057 := (privateBatchWrapper2_copies33 a h).2.1
  have c1058 := (privateBatchWrapper2_copies33 a h).2.2.1
  have c1059 := (privateBatchWrapper2_copies33 a h).2.2.2.1
  have c1060 := (privateBatchWrapper2_copies33 a h).2.2.2.2.1
  have c1061 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.1
  have c1062 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_21_9 := arithEq_of_rows h (row := 21) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_9
  simp only [← c1057, ← c1058, ← c1059] at e_21_9
  have e_21_10 := arithEq_of_rows h (row := 21) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_10
  simp only [← c1060, ← c1061, ← c1062] at e_21_10
  have e_24_1 := arithEq_of_rows h (row := 24) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_1
  simp only [← c1051, k0, ← c1052, k0, ← c1053] at e_24_1
  have e_24_2 := arithEq_of_rows h (row := 24) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c1054, ← c1055, k0, ← c1056] at e_24_2
  have e_24_3 := arithEq_of_rows h (row := 24) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_3
  simp only [← c1063, ← c1064, k0, ← c1065] at e_24_3
  refine ⟨?_, ?_⟩
  · have hc := e_21_9
    simp only [e_24_2] at hc
    linear_combination c1066.trans k1 - hc
  · have hc := e_24_3
    simp only [e_24_1, e_21_10, e_24_2] at hc
    linear_combination c1067.trans k1 - hc

theorem privateBatchWrapper2_f139 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 11 39)) (a (.virt 19057)) (a (.virt 19058)) := by
  have c1068 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1078 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_21_11 := arithEq_of_rows h (row := 21) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_11
  simp only [← c1074, ← c1075, ← c1076] at e_21_11
  have e_21_12 := arithEq_of_rows h (row := 21) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_12
  simp only [← c1077, ← c1078, ← c1079] at e_21_12
  have e_24_4 := arithEq_of_rows h (row := 24) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_4
  simp only [← c1068, k0, ← c1069, k0, ← c1070] at e_24_4
  have e_24_5 := arithEq_of_rows h (row := 24) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c1071, ← c1072, k0, ← c1073] at e_24_5
  have e_24_6 := arithEq_of_rows h (row := 24) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_6
  simp only [← c1080, ← c1081, k0, ← c1082] at e_24_6
  refine ⟨?_, ?_⟩
  · have hc := e_21_11
    simp only [e_24_5] at hc
    linear_combination c1083.trans k1 - hc
  · have hc := e_24_6
    simp only [e_24_4, e_21_12, e_24_5] at hc
    linear_combination c1084.trans k1 - hc

theorem privateBatchWrapper2_f140 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 21 55) = band (a (.virt 19051)) (a (.virt 19053)) := by
  have c1085 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_21_13 := arithEq_of_rows h (row := 21) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_13
  simp only [← c1085, ← c1086, ← c1087] at e_21_13
  have hr := e_21_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f141 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 21 59) = band (a (.virt 19055)) (a (.virt 19057)) := by
  have c1088 := (privateBatchWrapper2_copies34 a h).1
  have c1089 := (privateBatchWrapper2_copies34 a h).2.1
  have c1090 := (privateBatchWrapper2_copies34 a h).2.2.1
  have e_21_14 := arithEq_of_rows h (row := 21) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_14
  simp only [← c1088, ← c1089, ← c1090] at e_21_14
  have hr := e_21_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f142 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 25 3) = band (a (.wire 21 55)) (a (.wire 21 59)) := by
  have c1091 := (privateBatchWrapper2_copies34 a h).2.2.2.1
  have c1092 := (privateBatchWrapper2_copies34 a h).2.2.2.2.1
  have c1093 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.1
  have e_25_0 := arithEq_of_rows h (row := 25) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_0
  simp only [← c1091, ← c1092, ← c1093] at e_25_0
  have hr := e_25_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f143 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 25 3) = bor (a (.virt 18979)) (a (.wire 25 3)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bor, k1]
  ring

theorem privateBatchWrapper2_f144 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 11 15)) (a (.virt 19059)) (a (.virt 19060)) := by
  have c1020 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1022 := (privateBatchWrapper2_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1094 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.1
  have c1096 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1099 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1102 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1103 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1106 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1107 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_11 := arithEq_of_rows h (row := 22) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_11
  simp only [← c1020, ← c1021, k0, ← c1022] at e_22_11
  have e_24_7 := arithEq_of_rows h (row := 24) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_7
  simp only [← c1094, k0, ← c1095, k0, ← c1096] at e_24_7
  have e_24_8 := arithEq_of_rows h (row := 24) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_8
  simp only [← c1103, ← c1104, k0, ← c1105] at e_24_8
  have e_25_1 := arithEq_of_rows h (row := 25) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_1
  simp only [← c1097, ← c1098, ← c1099] at e_25_1
  have e_25_2 := arithEq_of_rows h (row := 25) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_2
  simp only [← c1100, ← c1101, ← c1102] at e_25_2
  refine ⟨?_, ?_⟩
  · have hc := e_25_1
    simp only [e_22_11] at hc
    linear_combination c1106.trans k1 - hc
  · have hc := e_24_8
    simp only [e_24_7, e_25_2, e_22_11] at hc
    linear_combination c1107.trans k1 - hc

theorem privateBatchWrapper2_f145 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 11 23)) (a (.virt 19061)) (a (.virt 19062)) := by
  have c1037 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1039 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1108 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (privateBatchWrapper2_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1120 := (privateBatchWrapper2_copies35 a h).1
  have c1121 := (privateBatchWrapper2_copies35 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_22_14 := arithEq_of_rows h (row := 22) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_22_14
  simp only [← c1037, ← c1038, k0, ← c1039] at e_22_14
  have e_24_9 := arithEq_of_rows h (row := 24) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_9
  simp only [← c1108, k0, ← c1109, k0, ← c1110] at e_24_9
  have e_24_10 := arithEq_of_rows h (row := 24) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_10
  simp only [← c1117, ← c1118, k0, ← c1119] at e_24_10
  have e_25_3 := arithEq_of_rows h (row := 25) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_3
  simp only [← c1111, ← c1112, ← c1113] at e_25_3
  have e_25_4 := arithEq_of_rows h (row := 25) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_4
  simp only [← c1114, ← c1115, ← c1116] at e_25_4
  refine ⟨?_, ?_⟩
  · have hc := e_25_3
    simp only [e_22_14] at hc
    linear_combination c1120.trans k1 - hc
  · have hc := e_24_10
    simp only [e_24_9, e_25_4, e_22_14] at hc
    linear_combination c1121.trans k1 - hc

theorem privateBatchWrapper2_f146 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 11 31)) (a (.virt 19063)) (a (.virt 19064)) := by
  have c1054 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (privateBatchWrapper2_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1056 := (privateBatchWrapper2_copies33 a h).1
  have c1122 := (privateBatchWrapper2_copies35 a h).2.2.1
  have c1123 := (privateBatchWrapper2_copies35 a h).2.2.2.1
  have c1124 := (privateBatchWrapper2_copies35 a h).2.2.2.2.1
  have c1125 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.1
  have c1126 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_24_2 := arithEq_of_rows h (row := 24) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_2
  simp only [← c1054, ← c1055, k0, ← c1056] at e_24_2
  have e_24_11 := arithEq_of_rows h (row := 24) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_11
  simp only [← c1122, k0, ← c1123, k0, ← c1124] at e_24_11
  have e_24_12 := arithEq_of_rows h (row := 24) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_12
  simp only [← c1131, ← c1132, k0, ← c1133] at e_24_12
  have e_25_5 := arithEq_of_rows h (row := 25) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_5
  simp only [← c1125, ← c1126, ← c1127] at e_25_5
  have e_25_6 := arithEq_of_rows h (row := 25) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_6
  simp only [← c1128, ← c1129, ← c1130] at e_25_6
  refine ⟨?_, ?_⟩
  · have hc := e_25_5
    simp only [e_24_2] at hc
    linear_combination c1134.trans k1 - hc
  · have hc := e_24_12
    simp only [e_24_11, e_25_6, e_24_2] at hc
    linear_combination c1135.trans k1 - hc

theorem privateBatchWrapper2_f147 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 11 39)) (a (.virt 19065)) (a (.virt 19066)) := by
  have c1071 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1073 := (privateBatchWrapper2_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1140 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1143 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_24_5 := arithEq_of_rows h (row := 24) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_5
  simp only [← c1071, ← c1072, k0, ← c1073] at e_24_5
  have e_24_13 := arithEq_of_rows h (row := 24) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_13
  simp only [← c1136, k0, ← c1137, k0, ← c1138] at e_24_13
  have e_24_14 := arithEq_of_rows h (row := 24) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_24_14
  simp only [← c1145, ← c1146, k0, ← c1147] at e_24_14
  have e_25_7 := arithEq_of_rows h (row := 25) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_7
  simp only [← c1139, ← c1140, ← c1141] at e_25_7
  have e_25_8 := arithEq_of_rows h (row := 25) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_8
  simp only [← c1142, ← c1143, ← c1144] at e_25_8
  refine ⟨?_, ?_⟩
  · have hc := e_25_7
    simp only [e_24_5] at hc
    linear_combination c1148.trans k1 - hc
  · have hc := e_24_14
    simp only [e_24_13, e_25_8, e_24_5] at hc
    linear_combination c1149.trans k1 - hc

theorem privateBatchWrapper2_f148 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 25 39) = band (a (.virt 19059)) (a (.virt 19061)) := by
  have c1150 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (privateBatchWrapper2_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (privateBatchWrapper2_copies36 a h).1
  have e_25_9 := arithEq_of_rows h (row := 25) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_9
  simp only [← c1150, ← c1151, ← c1152] at e_25_9
  have hr := e_25_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f149 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 25 43) = band (a (.virt 19063)) (a (.virt 19065)) := by
  have c1153 := (privateBatchWrapper2_copies36 a h).2.1
  have c1154 := (privateBatchWrapper2_copies36 a h).2.2.1
  have c1155 := (privateBatchWrapper2_copies36 a h).2.2.2.1
  have e_25_10 := arithEq_of_rows h (row := 25) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_10
  simp only [← c1153, ← c1154, ← c1155] at e_25_10
  have hr := e_25_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f150 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 25 47) = band (a (.wire 25 39)) (a (.wire 25 43)) := by
  have c1156 := (privateBatchWrapper2_copies36 a h).2.2.2.2.1
  have c1157 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.1
  have c1158 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.1
  have e_25_11 := arithEq_of_rows h (row := 25) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_11
  simp only [← c1156, ← c1157, ← c1158] at e_25_11
  have hr := e_25_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f151 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 26 3) = bselect (a (.wire 25 47)) (a (.wire 11 7)) (a (.virt 18979)) := by
  have c1159 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_26_0 := arithEq_of_rows h (row := 26) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1159, ← c1160, ← c1161, k1] at e_26_0
  have hr := e_26_0
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f152 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 26 3) = a (.virt 18979) + a (.wire 26 3) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f153 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 15)) (a (.wire 11 15)) (a (.virt 19067)) (a (.virt 19068)) := by
  have c1162 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1174 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1177 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_25_12 := arithEq_of_rows h (row := 25) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_12
  simp only [← c1168, ← c1169, ← c1170] at e_25_12
  have e_25_13 := arithEq_of_rows h (row := 25) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_13
  simp only [← c1171, ← c1172, ← c1173] at e_25_13
  have e_26_1 := arithEq_of_rows h (row := 26) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1162, k0, ← c1163, k0, ← c1164] at e_26_1
  have e_26_2 := arithEq_of_rows h (row := 26) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1165, ← c1166, k0, ← c1167] at e_26_2
  have e_26_3 := arithEq_of_rows h (row := 26) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1174, ← c1175, k0, ← c1176] at e_26_3
  refine ⟨?_, ?_⟩
  · have hc := e_25_12
    simp only [e_26_2] at hc
    linear_combination c1177.trans k1 - hc
  · have hc := e_26_3
    simp only [e_26_1, e_25_13, e_26_2] at hc
    linear_combination c1178.trans k1 - hc

theorem privateBatchWrapper2_f154 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 23)) (a (.wire 11 23)) (a (.virt 19069)) (a (.virt 19070)) := by
  have c1179 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1180 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1182 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1183 := (privateBatchWrapper2_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (privateBatchWrapper2_copies37 a h).1
  have c1185 := (privateBatchWrapper2_copies37 a h).2.1
  have c1186 := (privateBatchWrapper2_copies37 a h).2.2.1
  have c1187 := (privateBatchWrapper2_copies37 a h).2.2.2.1
  have c1188 := (privateBatchWrapper2_copies37 a h).2.2.2.2.1
  have c1189 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.1
  have c1190 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.1
  have c1191 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_25_14 := arithEq_of_rows h (row := 25) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_14
  simp only [← c1185, ← c1186, ← c1187] at e_25_14
  have e_26_4 := arithEq_of_rows h (row := 26) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1179, k0, ← c1180, k0, ← c1181] at e_26_4
  have e_26_5 := arithEq_of_rows h (row := 26) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1182, ← c1183, k0, ← c1184] at e_26_5
  have e_26_6 := arithEq_of_rows h (row := 26) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1191, ← c1192, k0, ← c1193] at e_26_6
  have e_27_0 := arithEq_of_rows h (row := 27) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1188, ← c1189, ← c1190] at e_27_0
  refine ⟨?_, ?_⟩
  · have hc := e_25_14
    simp only [e_26_5] at hc
    linear_combination c1194.trans k1 - hc
  · have hc := e_26_6
    simp only [e_26_4, e_27_0, e_26_5] at hc
    linear_combination c1195.trans k1 - hc

theorem privateBatchWrapper2_f155 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 31)) (a (.wire 11 31)) (a (.virt 19071)) (a (.virt 19072)) := by
  have c1196 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1207 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_26_7 := arithEq_of_rows h (row := 26) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1196, k0, ← c1197, k0, ← c1198] at e_26_7
  have e_26_8 := arithEq_of_rows h (row := 26) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1199, ← c1200, k0, ← c1201] at e_26_8
  have e_26_9 := arithEq_of_rows h (row := 26) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1208, ← c1209, k0, ← c1210] at e_26_9
  have e_27_1 := arithEq_of_rows h (row := 27) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1202, ← c1203, ← c1204] at e_27_1
  have e_27_2 := arithEq_of_rows h (row := 27) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1205, ← c1206, ← c1207] at e_27_2
  refine ⟨?_, ?_⟩
  · have hc := e_27_1
    simp only [e_26_8] at hc
    linear_combination c1211.trans k1 - hc
  · have hc := e_26_9
    simp only [e_26_7, e_27_2, e_26_8] at hc
    linear_combination c1212.trans k1 - hc

theorem privateBatchWrapper2_f156 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 39)) (a (.wire 11 39)) (a (.virt 19073)) (a (.virt 19074)) := by
  have c1213 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (privateBatchWrapper2_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (privateBatchWrapper2_copies38 a h).1
  have c1217 := (privateBatchWrapper2_copies38 a h).2.1
  have c1218 := (privateBatchWrapper2_copies38 a h).2.2.1
  have c1219 := (privateBatchWrapper2_copies38 a h).2.2.2.1
  have c1220 := (privateBatchWrapper2_copies38 a h).2.2.2.2.1
  have c1221 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.1
  have c1222 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.1
  have c1224 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1227 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_26_10 := arithEq_of_rows h (row := 26) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1213, k0, ← c1214, k0, ← c1215] at e_26_10
  have e_26_11 := arithEq_of_rows h (row := 26) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1216, ← c1217, k0, ← c1218] at e_26_11
  have e_26_12 := arithEq_of_rows h (row := 26) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1225, ← c1226, k0, ← c1227] at e_26_12
  have e_27_3 := arithEq_of_rows h (row := 27) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1219, ← c1220, ← c1221] at e_27_3
  have e_27_4 := arithEq_of_rows h (row := 27) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_4
  simp only [← c1222, ← c1223, ← c1224] at e_27_4
  refine ⟨?_, ?_⟩
  · have hc := e_27_3
    simp only [e_26_11] at hc
    linear_combination c1228.trans k1 - hc
  · have hc := e_26_12
    simp only [e_26_10, e_27_4, e_26_11] at hc
    linear_combination c1229.trans k1 - hc

theorem privateBatchWrapper2_f157 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 27 23) = band (a (.virt 19067)) (a (.virt 19069)) := by
  have c1230 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_5 := arithEq_of_rows h (row := 27) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_5
  simp only [← c1230, ← c1231, ← c1232] at e_27_5
  have hr := e_27_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f158 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 27 27) = band (a (.virt 19071)) (a (.virt 19073)) := by
  have c1233 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_6 := arithEq_of_rows h (row := 27) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_6
  simp only [← c1233, ← c1234, ← c1235] at e_27_6
  have hr := e_27_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f159 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 27 31) = band (a (.wire 27 23)) (a (.wire 27 27)) := by
  have c1236 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_27_7 := arithEq_of_rows h (row := 27) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_7
  simp only [← c1236, ← c1237, ← c1238] at e_27_7
  have hr := e_27_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f160 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 26 55) = bselect (a (.wire 27 31)) (a (.wire 11 47)) (a (.virt 18979)) := by
  have c1239 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1241 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_26_13 := arithEq_of_rows h (row := 26) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1239, ← c1240, ← c1241, k1] at e_26_13
  have hr := e_26_13
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f161 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 51) = a (.wire 26 3) + a (.wire 26 55) := by
  have c1242 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c1242, ← c1243, k0, ← c1244] at e_6_12
  have hr := e_6_12
  linear_combination hr

theorem privateBatchWrapper2_f162 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 55)) (a (.wire 11 15)) (a (.virt 19075)) (a (.virt 19076)) := by
  have c1245 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (privateBatchWrapper2_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (privateBatchWrapper2_copies39 a h).1
  have c1249 := (privateBatchWrapper2_copies39 a h).2.1
  have c1250 := (privateBatchWrapper2_copies39 a h).2.2.1
  have c1251 := (privateBatchWrapper2_copies39 a h).2.2.2.1
  have c1252 := (privateBatchWrapper2_copies39 a h).2.2.2.2.1
  have c1253 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.1
  have c1254 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.1
  have c1256 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.1
  have c1258 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1261 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_26_14 := arithEq_of_rows h (row := 26) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1245, k0, ← c1246, k0, ← c1247] at e_26_14
  have e_27_8 := arithEq_of_rows h (row := 27) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_8
  simp only [← c1251, ← c1252, ← c1253] at e_27_8
  have e_27_9 := arithEq_of_rows h (row := 27) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_9
  simp only [← c1254, ← c1255, ← c1256] at e_27_9
  have e_28_0 := arithEq_of_rows h (row := 28) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c1248, ← c1249, k0, ← c1250] at e_28_0
  have e_28_1 := arithEq_of_rows h (row := 28) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c1257, ← c1258, k0, ← c1259] at e_28_1
  refine ⟨?_, ?_⟩
  · have hc := e_27_8
    simp only [e_28_0] at hc
    linear_combination c1260.trans k1 - hc
  · have hc := e_28_1
    simp only [e_26_14, e_27_9, e_28_0] at hc
    linear_combination c1261.trans k1 - hc

theorem privateBatchWrapper2_f163 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 3)) (a (.wire 11 23)) (a (.virt 19077)) (a (.virt 19078)) := by
  have c1262 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1264 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1267 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1270 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1271 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1274 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_27_10 := arithEq_of_rows h (row := 27) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_10
  simp only [← c1268, ← c1269, ← c1270] at e_27_10
  have e_27_11 := arithEq_of_rows h (row := 27) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_11
  simp only [← c1271, ← c1272, ← c1273] at e_27_11
  have e_28_2 := arithEq_of_rows h (row := 28) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c1262, k0, ← c1263, k0, ← c1264] at e_28_2
  have e_28_3 := arithEq_of_rows h (row := 28) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c1265, ← c1266, k0, ← c1267] at e_28_3
  have e_28_4 := arithEq_of_rows h (row := 28) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c1274, ← c1275, k0, ← c1276] at e_28_4
  refine ⟨?_, ?_⟩
  · have hc := e_27_10
    simp only [e_28_3] at hc
    linear_combination c1277.trans k1 - hc
  · have hc := e_28_4
    simp only [e_28_2, e_27_11, e_28_3] at hc
    linear_combination c1278.trans k1 - hc

theorem privateBatchWrapper2_f164 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 11)) (a (.wire 11 31)) (a (.virt 19079)) (a (.virt 19080)) := by
  have c1279 := (privateBatchWrapper2_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (privateBatchWrapper2_copies40 a h).1
  have c1281 := (privateBatchWrapper2_copies40 a h).2.1
  have c1282 := (privateBatchWrapper2_copies40 a h).2.2.1
  have c1283 := (privateBatchWrapper2_copies40 a h).2.2.2.1
  have c1284 := (privateBatchWrapper2_copies40 a h).2.2.2.2.1
  have c1285 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.1
  have c1286 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.1
  have c1287 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1291 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_27_12 := arithEq_of_rows h (row := 27) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_12
  simp only [← c1285, ← c1286, ← c1287] at e_27_12
  have e_27_13 := arithEq_of_rows h (row := 27) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_13
  simp only [← c1288, ← c1289, ← c1290] at e_27_13
  have e_28_5 := arithEq_of_rows h (row := 28) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1279, k0, ← c1280, k0, ← c1281] at e_28_5
  have e_28_6 := arithEq_of_rows h (row := 28) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1282, ← c1283, k0, ← c1284] at e_28_6
  have e_28_7 := arithEq_of_rows h (row := 28) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1291, ← c1292, k0, ← c1293] at e_28_7
  refine ⟨?_, ?_⟩
  · have hc := e_27_12
    simp only [e_28_6] at hc
    linear_combination c1294.trans k1 - hc
  · have hc := e_28_7
    simp only [e_28_5, e_27_13, e_28_6] at hc
    linear_combination c1295.trans k1 - hc

theorem privateBatchWrapper2_f165 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 19)) (a (.wire 11 39)) (a (.virt 19081)) (a (.virt 19082)) := by
  have c1296 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1308 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (privateBatchWrapper2_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1312 := (privateBatchWrapper2_copies41 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_27_14 := arithEq_of_rows h (row := 27) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_14
  simp only [← c1302, ← c1303, ← c1304] at e_27_14
  have e_28_8 := arithEq_of_rows h (row := 28) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1296, k0, ← c1297, k0, ← c1298] at e_28_8
  have e_28_9 := arithEq_of_rows h (row := 28) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1299, ← c1300, k0, ← c1301] at e_28_9
  have e_28_10 := arithEq_of_rows h (row := 28) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1308, ← c1309, k0, ← c1310] at e_28_10
  have e_29_0 := arithEq_of_rows h (row := 29) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1305, ← c1306, ← c1307] at e_29_0
  refine ⟨?_, ?_⟩
  · have hc := e_27_14
    simp only [e_28_9] at hc
    linear_combination c1311.trans k1 - hc
  · have hc := e_28_10
    simp only [e_28_8, e_29_0, e_28_9] at hc
    linear_combination c1312.trans k1 - hc

theorem privateBatchWrapper2_f166 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 29 7) = band (a (.virt 19075)) (a (.virt 19077)) := by
  have c1313 := (privateBatchWrapper2_copies41 a h).2.1
  have c1314 := (privateBatchWrapper2_copies41 a h).2.2.1
  have c1315 := (privateBatchWrapper2_copies41 a h).2.2.2.1
  have e_29_1 := arithEq_of_rows h (row := 29) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1313, ← c1314, ← c1315] at e_29_1
  have hr := e_29_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f167 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 29 11) = band (a (.virt 19079)) (a (.virt 19081)) := by
  have c1316 := (privateBatchWrapper2_copies41 a h).2.2.2.2.1
  have c1317 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.1
  have c1318 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.1
  have e_29_2 := arithEq_of_rows h (row := 29) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1316, ← c1317, ← c1318] at e_29_2
  have hr := e_29_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f168 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 29 15) = band (a (.wire 29 7)) (a (.wire 29 11)) := by
  have c1319 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.1
  have c1321 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have e_29_3 := arithEq_of_rows h (row := 29) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1319, ← c1320, ← c1321] at e_29_3
  have hr := e_29_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f169 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 28 47) = bselect (a (.wire 29 15)) (a (.wire 12 27)) (a (.virt 18979)) := by
  have c1322 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_28_11 := arithEq_of_rows h (row := 28) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1322, ← c1323, ← c1324, k1] at e_28_11
  have hr := e_28_11
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f170 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 55) = a (.wire 6 51) + a (.wire 28 47) := by
  have c1325 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c1325, ← c1326, k0, ← c1327] at e_6_13
  have hr := e_6_13
  linear_combination hr

theorem privateBatchWrapper2_f171 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 35)) (a (.wire 11 15)) (a (.virt 19083)) (a (.virt 19084)) := by
  have c1328 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1329 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1335 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1342 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (privateBatchWrapper2_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (privateBatchWrapper2_copies42 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_28_12 := arithEq_of_rows h (row := 28) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1328, k0, ← c1329, k0, ← c1330] at e_28_12
  have e_28_13 := arithEq_of_rows h (row := 28) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1331, ← c1332, k0, ← c1333] at e_28_13
  have e_28_14 := arithEq_of_rows h (row := 28) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1340, ← c1341, k0, ← c1342] at e_28_14
  have e_29_4 := arithEq_of_rows h (row := 29) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1334, ← c1335, ← c1336] at e_29_4
  have e_29_5 := arithEq_of_rows h (row := 29) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1337, ← c1338, ← c1339] at e_29_5
  refine ⟨?_, ?_⟩
  · have hc := e_29_4
    simp only [e_28_13] at hc
    linear_combination c1343.trans k1 - hc
  · have hc := e_28_14
    simp only [e_28_12, e_29_5, e_28_13] at hc
    linear_combination c1344.trans k1 - hc

theorem privateBatchWrapper2_f172 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 43)) (a (.wire 11 23)) (a (.virt 19085)) (a (.virt 19086)) := by
  have c1345 := (privateBatchWrapper2_copies42 a h).2.1
  have c1346 := (privateBatchWrapper2_copies42 a h).2.2.1
  have c1347 := (privateBatchWrapper2_copies42 a h).2.2.2.1
  have c1348 := (privateBatchWrapper2_copies42 a h).2.2.2.2.1
  have c1349 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.1
  have c1350 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.1
  have c1351 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.1
  have c1352 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have c1354 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1355 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1358 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1361 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_29_6 := arithEq_of_rows h (row := 29) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1351, ← c1352, ← c1353] at e_29_6
  have e_29_7 := arithEq_of_rows h (row := 29) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c1354, ← c1355, ← c1356] at e_29_7
  have e_30_0 := arithEq_of_rows h (row := 30) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c1345, k0, ← c1346, k0, ← c1347] at e_30_0
  have e_30_1 := arithEq_of_rows h (row := 30) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c1348, ← c1349, k0, ← c1350] at e_30_1
  have e_30_2 := arithEq_of_rows h (row := 30) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c1357, ← c1358, k0, ← c1359] at e_30_2
  refine ⟨?_, ?_⟩
  · have hc := e_29_6
    simp only [e_30_1] at hc
    linear_combination c1360.trans k1 - hc
  · have hc := e_30_2
    simp only [e_30_0, e_29_7, e_30_1] at hc
    linear_combination c1361.trans k1 - hc

theorem privateBatchWrapper2_f173 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 51)) (a (.wire 11 31)) (a (.virt 19087)) (a (.virt 19088)) := by
  have c1362 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1363 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1375 := (privateBatchWrapper2_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (privateBatchWrapper2_copies43 a h).1
  have c1377 := (privateBatchWrapper2_copies43 a h).2.1
  have c1378 := (privateBatchWrapper2_copies43 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_29_8 := arithEq_of_rows h (row := 29) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c1368, ← c1369, ← c1370] at e_29_8
  have e_29_9 := arithEq_of_rows h (row := 29) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c1371, ← c1372, ← c1373] at e_29_9
  have e_30_3 := arithEq_of_rows h (row := 30) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c1362, k0, ← c1363, k0, ← c1364] at e_30_3
  have e_30_4 := arithEq_of_rows h (row := 30) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c1365, ← c1366, k0, ← c1367] at e_30_4
  have e_30_5 := arithEq_of_rows h (row := 30) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c1374, ← c1375, k0, ← c1376] at e_30_5
  refine ⟨?_, ?_⟩
  · have hc := e_29_8
    simp only [e_30_4] at hc
    linear_combination c1377.trans k1 - hc
  · have hc := e_30_5
    simp only [e_30_3, e_29_9, e_30_4] at hc
    linear_combination c1378.trans k1 - hc

theorem privateBatchWrapper2_f174 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 59)) (a (.wire 11 39)) (a (.virt 19089)) (a (.virt 19090)) := by
  have c1379 := (privateBatchWrapper2_copies43 a h).2.2.2.1
  have c1380 := (privateBatchWrapper2_copies43 a h).2.2.2.2.1
  have c1381 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.1
  have c1382 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.1
  have c1383 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have c1386 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1389 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1392 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_29_10 := arithEq_of_rows h (row := 29) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c1385, ← c1386, ← c1387] at e_29_10
  have e_29_11 := arithEq_of_rows h (row := 29) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c1388, ← c1389, ← c1390] at e_29_11
  have e_30_6 := arithEq_of_rows h (row := 30) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c1379, k0, ← c1380, k0, ← c1381] at e_30_6
  have e_30_7 := arithEq_of_rows h (row := 30) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c1382, ← c1383, k0, ← c1384] at e_30_7
  have e_30_8 := arithEq_of_rows h (row := 30) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c1391, ← c1392, k0, ← c1393] at e_30_8
  refine ⟨?_, ?_⟩
  · have hc := e_29_10
    simp only [e_30_7] at hc
    linear_combination c1394.trans k1 - hc
  · have hc := e_30_8
    simp only [e_30_6, e_29_11, e_30_7] at hc
    linear_combination c1395.trans k1 - hc

theorem privateBatchWrapper2_f175 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 29 51) = band (a (.virt 19083)) (a (.virt 19085)) := by
  have c1396 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_12 := arithEq_of_rows h (row := 29) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c1396, ← c1397, ← c1398] at e_29_12
  have hr := e_29_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f176 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 29 55) = band (a (.virt 19087)) (a (.virt 19089)) := by
  have c1399 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_13 := arithEq_of_rows h (row := 29) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c1399, ← c1400, ← c1401] at e_29_13
  have hr := e_29_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f177 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 29 59) = band (a (.wire 29 51)) (a (.wire 29 55)) := by
  have c1402 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_29_14 := arithEq_of_rows h (row := 29) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c1402, ← c1403, ← c1404] at e_29_14
  have hr := e_29_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f178 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 30 39) = bselect (a (.wire 29 59)) (a (.wire 13 7)) (a (.virt 18979)) := by
  have c1405 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (privateBatchWrapper2_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_30_9 := arithEq_of_rows h (row := 30) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c1405, ← c1406, ← c1407, k1] at e_30_9
  have hr := e_30_9
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f179 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 59) = a (.wire 6 55) + a (.wire 30 39) := by
  have c1408 := (privateBatchWrapper2_copies44 a h).1
  have c1409 := (privateBatchWrapper2_copies44 a h).2.1
  have c1410 := (privateBatchWrapper2_copies44 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c1408, ← c1409, k0, ← c1410] at e_6_14
  have hr := e_6_14
  linear_combination hr

theorem privateBatchWrapper2_f180 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 30 47) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 6 59)) := by
  have c1411 := (privateBatchWrapper2_copies44 a h).2.2.2.1
  have c1412 := (privateBatchWrapper2_copies44 a h).2.2.2.2.1
  have c1413 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.1
  have c1414 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_30_10 := arithEq_of_rows h (row := 30) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c1411, ← c1412, ← c1413] at e_30_10
  have e_30_11 := arithEq_of_rows h (row := 30) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c1414, ← c1415, k1, ← c1416] at e_30_11
  have hr := e_30_11
  simp only [e_30_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f181 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 30 55) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 15)) := by
  have c1417 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_30_12 := arithEq_of_rows h (row := 30) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c1417, ← c1418, ← c1419] at e_30_12
  have e_30_13 := arithEq_of_rows h (row := 30) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c1420, ← c1421, k1, ← c1422] at e_30_13
  have hr := e_30_13
  simp only [e_30_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f182 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 31 3) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 23)) := by
  have c1423 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1425 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1426 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1428 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_30_14 := arithEq_of_rows h (row := 30) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c1423, ← c1424, ← c1425] at e_30_14
  have e_31_0 := arithEq_of_rows h (row := 31) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c1426, ← c1427, k1, ← c1428] at e_31_0
  have hr := e_31_0
  simp only [e_30_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f183 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 31 11) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 31)) := by
  have c1429 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1431 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1432 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1434 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_1 := arithEq_of_rows h (row := 31) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c1429, ← c1430, ← c1431] at e_31_1
  have e_31_2 := arithEq_of_rows h (row := 31) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c1432, ← c1433, k1, ← c1434] at e_31_2
  have hr := e_31_2
  simp only [e_31_1] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f184 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 31 19) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 39)) := by
  have c1435 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1437 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1438 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1439 := (privateBatchWrapper2_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1440 := (privateBatchWrapper2_copies45 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_3 := arithEq_of_rows h (row := 31) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c1435, ← c1436, ← c1437] at e_31_3
  have e_31_4 := arithEq_of_rows h (row := 31) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c1438, ← c1439, k1, ← c1440] at e_31_4
  have hr := e_31_4
  simp only [e_31_3] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f185 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 30 47)) 32 := by
  have c1441 := (privateBatchWrapper2_copies45 a h).2.1
  have c1442 := (privateBatchWrapper2_copies45 a h).2.2.1
  have c1443 := (privateBatchWrapper2_copies45 a h).2.2.2.1
  have c1444 := (privateBatchWrapper2_copies45 a h).2.2.2.2.1
  have c1445 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.1
  have c1446 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.1
  have c1447 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.1
  have c1448 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1451 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1454 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1456 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1459 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1465 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1468 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 32) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1441.trans k1
    · exact c1442.trans k1
    · exact c1443.trans k1
    · exact c1444.trans k1
    · exact c1445.trans k1
    · exact c1446.trans k1
    · exact c1447.trans k1
    · exact c1448.trans k1
    · exact c1449.trans k1
    · exact c1450.trans k1
    · exact c1451.trans k1
    · exact c1452.trans k1
    · exact c1453.trans k1
    · exact c1454.trans k1
    · exact c1455.trans k1
    · exact c1456.trans k1
    · exact c1457.trans k1
    · exact c1458.trans k1
    · exact c1459.trans k1
    · exact c1460.trans k1
    · exact c1461.trans k1
    · exact c1462.trans k1
    · exact c1463.trans k1
    · exact c1464.trans k1
    · exact c1465.trans k1
    · exact c1466.trans k1
    · exact c1467.trans k1
    )
  rwa [c1468] at hr

theorem privateBatchWrapper2_f186 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 11 55)) (a (.virt 19091)) (a (.virt 19092)) := by
  have c1469 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (privateBatchWrapper2_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (privateBatchWrapper2_copies46 a h).1
  have c1473 := (privateBatchWrapper2_copies46 a h).2.1
  have c1474 := (privateBatchWrapper2_copies46 a h).2.2.1
  have c1475 := (privateBatchWrapper2_copies46 a h).2.2.2.1
  have c1476 := (privateBatchWrapper2_copies46 a h).2.2.2.2.1
  have c1477 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.1
  have c1478 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.1
  have c1479 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.1
  have c1482 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_5 := arithEq_of_rows h (row := 31) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c1469, k0, ← c1470, k0, ← c1471] at e_31_5
  have e_31_6 := arithEq_of_rows h (row := 31) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c1472, ← c1473, k0, ← c1474] at e_31_6
  have e_31_7 := arithEq_of_rows h (row := 31) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c1481, ← c1482, k0, ← c1483] at e_31_7
  have e_33_0 := arithEq_of_rows h (row := 33) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c1475, ← c1476, ← c1477] at e_33_0
  have e_33_1 := arithEq_of_rows h (row := 33) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c1478, ← c1479, ← c1480] at e_33_1
  refine ⟨?_, ?_⟩
  · have hc := e_33_0
    simp only [e_31_6] at hc
    linear_combination c1484.trans k1 - hc
  · have hc := e_31_7
    simp only [e_31_5, e_33_1, e_31_6] at hc
    linear_combination c1485.trans k1 - hc

theorem privateBatchWrapper2_f187 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 12 3)) (a (.virt 19093)) (a (.virt 19094)) := by
  have c1486 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1493 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1496 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1499 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1502 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_8 := arithEq_of_rows h (row := 31) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c1486, k0, ← c1487, k0, ← c1488] at e_31_8
  have e_31_9 := arithEq_of_rows h (row := 31) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c1489, ← c1490, k0, ← c1491] at e_31_9
  have e_31_10 := arithEq_of_rows h (row := 31) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c1498, ← c1499, k0, ← c1500] at e_31_10
  have e_33_2 := arithEq_of_rows h (row := 33) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c1492, ← c1493, ← c1494] at e_33_2
  have e_33_3 := arithEq_of_rows h (row := 33) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c1495, ← c1496, ← c1497] at e_33_3
  refine ⟨?_, ?_⟩
  · have hc := e_33_2
    simp only [e_31_9] at hc
    linear_combination c1501.trans k1 - hc
  · have hc := e_31_10
    simp only [e_31_8, e_33_3, e_31_9] at hc
    linear_combination c1502.trans k1 - hc

theorem privateBatchWrapper2_f188 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 12 11)) (a (.virt 19095)) (a (.virt 19096)) := by
  have c1503 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (privateBatchWrapper2_copies47 a h).1
  have c1505 := (privateBatchWrapper2_copies47 a h).2.1
  have c1506 := (privateBatchWrapper2_copies47 a h).2.2.1
  have c1507 := (privateBatchWrapper2_copies47 a h).2.2.2.1
  have c1508 := (privateBatchWrapper2_copies47 a h).2.2.2.2.1
  have c1509 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.1
  have c1510 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.1
  have c1513 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have c1514 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1516 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1519 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_11 := arithEq_of_rows h (row := 31) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c1503, k0, ← c1504, k0, ← c1505] at e_31_11
  have e_31_12 := arithEq_of_rows h (row := 31) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c1506, ← c1507, k0, ← c1508] at e_31_12
  have e_31_13 := arithEq_of_rows h (row := 31) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c1515, ← c1516, k0, ← c1517] at e_31_13
  have e_33_4 := arithEq_of_rows h (row := 33) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c1509, ← c1510, ← c1511] at e_33_4
  have e_33_5 := arithEq_of_rows h (row := 33) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c1512, ← c1513, ← c1514] at e_33_5
  refine ⟨?_, ?_⟩
  · have hc := e_33_4
    simp only [e_31_12] at hc
    linear_combination c1518.trans k1 - hc
  · have hc := e_31_13
    simp only [e_31_11, e_33_5, e_31_12] at hc
    linear_combination c1519.trans k1 - hc

theorem privateBatchWrapper2_f189 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 12 19)) (a (.virt 19097)) (a (.virt 19098)) := by
  have c1520 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1522 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1523 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1526 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1530 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1531 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1532 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1533 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1535 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (privateBatchWrapper2_copies48 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_14 := arithEq_of_rows h (row := 31) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c1520, k0, ← c1521, k0, ← c1522] at e_31_14
  have e_33_6 := arithEq_of_rows h (row := 33) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c1526, ← c1527, ← c1528] at e_33_6
  have e_33_7 := arithEq_of_rows h (row := 33) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1529, ← c1530, ← c1531] at e_33_7
  have e_34_0 := arithEq_of_rows h (row := 34) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c1523, ← c1524, k0, ← c1525] at e_34_0
  have e_34_1 := arithEq_of_rows h (row := 34) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_1
  simp only [← c1532, ← c1533, k0, ← c1534] at e_34_1
  refine ⟨?_, ?_⟩
  · have hc := e_33_6
    simp only [e_34_0] at hc
    linear_combination c1535.trans k1 - hc
  · have hc := e_34_1
    simp only [e_31_14, e_33_7, e_34_0] at hc
    linear_combination c1536.trans k1 - hc

theorem privateBatchWrapper2_f190 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 33 35) = band (a (.virt 19091)) (a (.virt 19093)) := by
  have c1537 := (privateBatchWrapper2_copies48 a h).2.1
  have c1538 := (privateBatchWrapper2_copies48 a h).2.2.1
  have c1539 := (privateBatchWrapper2_copies48 a h).2.2.2.1
  have e_33_8 := arithEq_of_rows h (row := 33) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c1537, ← c1538, ← c1539] at e_33_8
  have hr := e_33_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f191 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 33 39) = band (a (.virt 19095)) (a (.virt 19097)) := by
  have c1540 := (privateBatchWrapper2_copies48 a h).2.2.2.2.1
  have c1541 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.1
  have c1542 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.1
  have e_33_9 := arithEq_of_rows h (row := 33) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c1540, ← c1541, ← c1542] at e_33_9
  have hr := e_33_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f192 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 33 43) = band (a (.wire 33 35)) (a (.wire 33 39)) := by
  have c1543 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.1
  have c1544 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.1
  have c1545 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have e_33_10 := arithEq_of_rows h (row := 33) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c1543, ← c1544, ← c1545] at e_33_10
  have hr := e_33_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f193 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 33 43) = bor (a (.virt 18979)) (a (.wire 33 43)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bor, k1]
  ring

theorem privateBatchWrapper2_f194 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 15)) (a (.wire 11 55)) (a (.virt 19099)) (a (.virt 19100)) := by
  have c1546 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1547 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1548 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1553 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1554 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1556 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1557 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1558 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1559 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1561 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1562 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_33_11 := arithEq_of_rows h (row := 33) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c1552, ← c1553, ← c1554] at e_33_11
  have e_33_12 := arithEq_of_rows h (row := 33) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c1555, ← c1556, ← c1557] at e_33_12
  have e_34_2 := arithEq_of_rows h (row := 34) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_2
  simp only [← c1546, k0, ← c1547, k0, ← c1548] at e_34_2
  have e_34_3 := arithEq_of_rows h (row := 34) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c1549, ← c1550, k0, ← c1551] at e_34_3
  have e_34_4 := arithEq_of_rows h (row := 34) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_4
  simp only [← c1558, ← c1559, k0, ← c1560] at e_34_4
  refine ⟨?_, ?_⟩
  · have hc := e_33_11
    simp only [e_34_3] at hc
    linear_combination c1561.trans k1 - hc
  · have hc := e_34_4
    simp only [e_34_2, e_33_12, e_34_3] at hc
    linear_combination c1562.trans k1 - hc

theorem privateBatchWrapper2_f195 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 23)) (a (.wire 12 3)) (a (.virt 19101)) (a (.virt 19102)) := by
  have c1563 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1565 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (privateBatchWrapper2_copies49 a h).1
  have c1569 := (privateBatchWrapper2_copies49 a h).2.1
  have c1570 := (privateBatchWrapper2_copies49 a h).2.2.1
  have c1571 := (privateBatchWrapper2_copies49 a h).2.2.2.1
  have c1572 := (privateBatchWrapper2_copies49 a h).2.2.2.2.1
  have c1573 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.1
  have c1574 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.1
  have c1575 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have c1578 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_33_13 := arithEq_of_rows h (row := 33) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c1569, ← c1570, ← c1571] at e_33_13
  have e_33_14 := arithEq_of_rows h (row := 33) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c1572, ← c1573, ← c1574] at e_33_14
  have e_34_5 := arithEq_of_rows h (row := 34) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_5
  simp only [← c1563, k0, ← c1564, k0, ← c1565] at e_34_5
  have e_34_6 := arithEq_of_rows h (row := 34) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_6
  simp only [← c1566, ← c1567, k0, ← c1568] at e_34_6
  have e_34_7 := arithEq_of_rows h (row := 34) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_7
  simp only [← c1575, ← c1576, k0, ← c1577] at e_34_7
  refine ⟨?_, ?_⟩
  · have hc := e_33_13
    simp only [e_34_6] at hc
    linear_combination c1578.trans k1 - hc
  · have hc := e_34_7
    simp only [e_34_5, e_33_14, e_34_6] at hc
    linear_combination c1579.trans k1 - hc

theorem privateBatchWrapper2_f196 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 31)) (a (.wire 12 11)) (a (.virt 19103)) (a (.virt 19104)) := by
  have c1580 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1581 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1583 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1584 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1586 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1590 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1592 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1593 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1595 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_8 := arithEq_of_rows h (row := 34) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_8
  simp only [← c1580, k0, ← c1581, k0, ← c1582] at e_34_8
  have e_34_9 := arithEq_of_rows h (row := 34) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_9
  simp only [← c1583, ← c1584, k0, ← c1585] at e_34_9
  have e_34_10 := arithEq_of_rows h (row := 34) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_10
  simp only [← c1592, ← c1593, k0, ← c1594] at e_34_10
  have e_35_0 := arithEq_of_rows h (row := 35) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_0
  simp only [← c1586, ← c1587, ← c1588] at e_35_0
  have e_35_1 := arithEq_of_rows h (row := 35) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_1
  simp only [← c1589, ← c1590, ← c1591] at e_35_1
  refine ⟨?_, ?_⟩
  · have hc := e_35_0
    simp only [e_34_9] at hc
    linear_combination c1595.trans k1 - hc
  · have hc := e_34_10
    simp only [e_34_8, e_35_1, e_34_9] at hc
    linear_combination c1596.trans k1 - hc

theorem privateBatchWrapper2_f197 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 39)) (a (.wire 12 19)) (a (.virt 19105)) (a (.virt 19106)) := by
  have c1597 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1598 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1599 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (privateBatchWrapper2_copies50 a h).1
  have c1601 := (privateBatchWrapper2_copies50 a h).2.1
  have c1602 := (privateBatchWrapper2_copies50 a h).2.2.1
  have c1603 := (privateBatchWrapper2_copies50 a h).2.2.2.1
  have c1604 := (privateBatchWrapper2_copies50 a h).2.2.2.2.1
  have c1605 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.1
  have c1606 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.1
  have c1607 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.1
  have c1609 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have c1610 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1612 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1613 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_11 := arithEq_of_rows h (row := 34) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_11
  simp only [← c1597, k0, ← c1598, k0, ← c1599] at e_34_11
  have e_34_12 := arithEq_of_rows h (row := 34) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_12
  simp only [← c1600, ← c1601, k0, ← c1602] at e_34_12
  have e_34_13 := arithEq_of_rows h (row := 34) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_13
  simp only [← c1609, ← c1610, k0, ← c1611] at e_34_13
  have e_35_2 := arithEq_of_rows h (row := 35) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_2
  simp only [← c1603, ← c1604, ← c1605] at e_35_2
  have e_35_3 := arithEq_of_rows h (row := 35) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_3
  simp only [← c1606, ← c1607, ← c1608] at e_35_3
  refine ⟨?_, ?_⟩
  · have hc := e_35_2
    simp only [e_34_12] at hc
    linear_combination c1612.trans k1 - hc
  · have hc := e_34_13
    simp only [e_34_11, e_35_3, e_34_12] at hc
    linear_combination c1613.trans k1 - hc

theorem privateBatchWrapper2_f198 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 35 19) = band (a (.virt 19099)) (a (.virt 19101)) := by
  have c1614 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1616 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_4 := arithEq_of_rows h (row := 35) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_4
  simp only [← c1614, ← c1615, ← c1616] at e_35_4
  have hr := e_35_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f199 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 35 23) = band (a (.virt 19103)) (a (.virt 19105)) := by
  have c1617 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1619 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_5 := arithEq_of_rows h (row := 35) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_5
  simp only [← c1617, ← c1618, ← c1619] at e_35_5
  have hr := e_35_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f200 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 35 27) = band (a (.wire 35 19)) (a (.wire 35 23)) := by
  have c1620 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1622 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_35_6 := arithEq_of_rows h (row := 35) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_6
  simp only [← c1620, ← c1621, ← c1622] at e_35_6
  have hr := e_35_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f201 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 3) = bor (a (.wire 33 43)) (a (.wire 35 27)) := by
  have c1623 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1626 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1628 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c1623, ← c1624, ← c1625] at e_5_5
  have e_36_0 := arithEq_of_rows h (row := 36) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_0
  simp only [← c1626, ← c1627, k0, ← c1628] at e_36_0
  have hr := e_36_0
  simp only [e_5_5] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f202 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 11 55)) (a (.virt 19107)) (a (.virt 19108)) := by
  have c1472 := (privateBatchWrapper2_copies46 a h).1
  have c1473 := (privateBatchWrapper2_copies46 a h).2.1
  have c1474 := (privateBatchWrapper2_copies46 a h).2.2.1
  have c1629 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1630 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1631 := (privateBatchWrapper2_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (privateBatchWrapper2_copies51 a h).1
  have c1633 := (privateBatchWrapper2_copies51 a h).2.1
  have c1634 := (privateBatchWrapper2_copies51 a h).2.2.1
  have c1635 := (privateBatchWrapper2_copies51 a h).2.2.2.1
  have c1636 := (privateBatchWrapper2_copies51 a h).2.2.2.2.1
  have c1637 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.1
  have c1638 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.1
  have c1640 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_6 := arithEq_of_rows h (row := 31) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c1472, ← c1473, k0, ← c1474] at e_31_6
  have e_34_14 := arithEq_of_rows h (row := 34) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_14
  simp only [← c1629, k0, ← c1630, k0, ← c1631] at e_34_14
  have e_35_7 := arithEq_of_rows h (row := 35) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_7
  simp only [← c1632, ← c1633, ← c1634] at e_35_7
  have e_35_8 := arithEq_of_rows h (row := 35) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_8
  simp only [← c1635, ← c1636, ← c1637] at e_35_8
  have e_37_0 := arithEq_of_rows h (row := 37) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_0
  simp only [← c1638, ← c1639, k0, ← c1640] at e_37_0
  refine ⟨?_, ?_⟩
  · have hc := e_35_7
    simp only [e_31_6] at hc
    linear_combination c1641.trans k1 - hc
  · have hc := e_37_0
    simp only [e_34_14, e_35_8, e_31_6] at hc
    linear_combination c1642.trans k1 - hc

theorem privateBatchWrapper2_f203 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 12 3)) (a (.virt 19109)) (a (.virt 19110)) := by
  have c1489 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (privateBatchWrapper2_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1643 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1646 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1649 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1655 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_9 := arithEq_of_rows h (row := 31) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c1489, ← c1490, k0, ← c1491] at e_31_9
  have e_35_9 := arithEq_of_rows h (row := 35) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_9
  simp only [← c1646, ← c1647, ← c1648] at e_35_9
  have e_35_10 := arithEq_of_rows h (row := 35) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_10
  simp only [← c1649, ← c1650, ← c1651] at e_35_10
  have e_37_1 := arithEq_of_rows h (row := 37) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_1
  simp only [← c1643, k0, ← c1644, k0, ← c1645] at e_37_1
  have e_37_2 := arithEq_of_rows h (row := 37) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_2
  simp only [← c1652, ← c1653, k0, ← c1654] at e_37_2
  refine ⟨?_, ?_⟩
  · have hc := e_35_9
    simp only [e_31_9] at hc
    linear_combination c1655.trans k1 - hc
  · have hc := e_37_2
    simp only [e_37_1, e_35_10, e_31_9] at hc
    linear_combination c1656.trans k1 - hc

theorem privateBatchWrapper2_f204 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 12 11)) (a (.virt 19111)) (a (.virt 19112)) := by
  have c1506 := (privateBatchWrapper2_copies47 a h).2.2.1
  have c1507 := (privateBatchWrapper2_copies47 a h).2.2.2.1
  have c1508 := (privateBatchWrapper2_copies47 a h).2.2.2.2.1
  have c1657 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1658 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1660 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1661 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1663 := (privateBatchWrapper2_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (privateBatchWrapper2_copies52 a h).1
  have c1665 := (privateBatchWrapper2_copies52 a h).2.1
  have c1666 := (privateBatchWrapper2_copies52 a h).2.2.1
  have c1667 := (privateBatchWrapper2_copies52 a h).2.2.2.1
  have c1668 := (privateBatchWrapper2_copies52 a h).2.2.2.2.1
  have c1669 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.1
  have c1670 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_31_12 := arithEq_of_rows h (row := 31) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c1506, ← c1507, k0, ← c1508] at e_31_12
  have e_35_11 := arithEq_of_rows h (row := 35) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_11
  simp only [← c1660, ← c1661, ← c1662] at e_35_11
  have e_35_12 := arithEq_of_rows h (row := 35) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_12
  simp only [← c1663, ← c1664, ← c1665] at e_35_12
  have e_37_3 := arithEq_of_rows h (row := 37) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_3
  simp only [← c1657, k0, ← c1658, k0, ← c1659] at e_37_3
  have e_37_4 := arithEq_of_rows h (row := 37) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_4
  simp only [← c1666, ← c1667, k0, ← c1668] at e_37_4
  refine ⟨?_, ?_⟩
  · have hc := e_35_11
    simp only [e_31_12] at hc
    linear_combination c1669.trans k1 - hc
  · have hc := e_37_4
    simp only [e_37_3, e_35_12, e_31_12] at hc
    linear_combination c1670.trans k1 - hc

theorem privateBatchWrapper2_f205 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 12 19)) (a (.virt 19113)) (a (.virt 19114)) := by
  have c1523 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (privateBatchWrapper2_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1671 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.1
  have c1674 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1677 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1679 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_0 := arithEq_of_rows h (row := 34) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c1523, ← c1524, k0, ← c1525] at e_34_0
  have e_35_13 := arithEq_of_rows h (row := 35) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_13
  simp only [← c1674, ← c1675, ← c1676] at e_35_13
  have e_35_14 := arithEq_of_rows h (row := 35) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_35_14
  simp only [← c1677, ← c1678, ← c1679] at e_35_14
  have e_37_5 := arithEq_of_rows h (row := 37) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_5
  simp only [← c1671, k0, ← c1672, k0, ← c1673] at e_37_5
  have e_37_6 := arithEq_of_rows h (row := 37) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_6
  simp only [← c1680, ← c1681, k0, ← c1682] at e_37_6
  refine ⟨?_, ?_⟩
  · have hc := e_35_13
    simp only [e_34_0] at hc
    linear_combination c1683.trans k1 - hc
  · have hc := e_37_6
    simp only [e_37_5, e_35_14, e_34_0] at hc
    linear_combination c1684.trans k1 - hc

theorem privateBatchWrapper2_f206 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 3) = band (a (.virt 19107)) (a (.virt 19109)) := by
  have c1685 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_0 := arithEq_of_rows h (row := 38) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_0
  simp only [← c1685, ← c1686, ← c1687] at e_38_0
  have hr := e_38_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f207 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 7) = band (a (.virt 19111)) (a (.virt 19113)) := by
  have c1688 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1689 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_1 := arithEq_of_rows h (row := 38) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_1
  simp only [← c1688, ← c1689, ← c1690] at e_38_1
  have hr := e_38_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f208 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 11) = band (a (.wire 38 3)) (a (.wire 38 7)) := by
  have c1691 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1692 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_2 := arithEq_of_rows h (row := 38) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_2
  simp only [← c1691, ← c1692, ← c1693] at e_38_2
  have hr := e_38_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f209 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 37 31) = bselect (a (.wire 38 11)) (a (.wire 11 7)) (a (.virt 18979)) := by
  have c1694 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (privateBatchWrapper2_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (privateBatchWrapper2_copies53 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_37_7 := arithEq_of_rows h (row := 37) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_7
  simp only [← c1694, ← c1695, ← c1696, k1] at e_37_7
  have hr := e_37_7
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f210 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 37 31) = a (.virt 18979) + a (.wire 37 31) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f211 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 15)) (a (.wire 11 55)) (a (.virt 19115)) (a (.virt 19116)) := by
  have c1549 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1697 := (privateBatchWrapper2_copies53 a h).2.1
  have c1698 := (privateBatchWrapper2_copies53 a h).2.2.1
  have c1699 := (privateBatchWrapper2_copies53 a h).2.2.2.1
  have c1700 := (privateBatchWrapper2_copies53 a h).2.2.2.2.1
  have c1701 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.1
  have c1702 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.1
  have c1703 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.1
  have c1705 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1708 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1709 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_3 := arithEq_of_rows h (row := 34) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c1549, ← c1550, k0, ← c1551] at e_34_3
  have e_37_8 := arithEq_of_rows h (row := 37) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_8
  simp only [← c1697, k0, ← c1698, k0, ← c1699] at e_37_8
  have e_37_9 := arithEq_of_rows h (row := 37) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_9
  simp only [← c1706, ← c1707, k0, ← c1708] at e_37_9
  have e_38_3 := arithEq_of_rows h (row := 38) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_3
  simp only [← c1700, ← c1701, ← c1702] at e_38_3
  have e_38_4 := arithEq_of_rows h (row := 38) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_4
  simp only [← c1703, ← c1704, ← c1705] at e_38_4
  refine ⟨?_, ?_⟩
  · have hc := e_38_3
    simp only [e_34_3] at hc
    linear_combination c1709.trans k1 - hc
  · have hc := e_37_9
    simp only [e_37_8, e_38_4, e_34_3] at hc
    linear_combination c1710.trans k1 - hc

theorem privateBatchWrapper2_f212 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 23)) (a (.wire 12 3)) (a (.virt 19117)) (a (.virt 19118)) := by
  have c1566 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (privateBatchWrapper2_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (privateBatchWrapper2_copies49 a h).1
  have c1711 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1714 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1715 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1716 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1717 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1718 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1719 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1720 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1721 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1722 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1723 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1724 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_6 := arithEq_of_rows h (row := 34) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_6
  simp only [← c1566, ← c1567, k0, ← c1568] at e_34_6
  have e_37_10 := arithEq_of_rows h (row := 37) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_10
  simp only [← c1711, k0, ← c1712, k0, ← c1713] at e_37_10
  have e_37_11 := arithEq_of_rows h (row := 37) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_11
  simp only [← c1720, ← c1721, k0, ← c1722] at e_37_11
  have e_38_5 := arithEq_of_rows h (row := 38) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_5
  simp only [← c1714, ← c1715, ← c1716] at e_38_5
  have e_38_6 := arithEq_of_rows h (row := 38) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_6
  simp only [← c1717, ← c1718, ← c1719] at e_38_6
  refine ⟨?_, ?_⟩
  · have hc := e_38_5
    simp only [e_34_6] at hc
    linear_combination c1723.trans k1 - hc
  · have hc := e_37_11
    simp only [e_37_10, e_38_6, e_34_6] at hc
    linear_combination c1724.trans k1 - hc

theorem privateBatchWrapper2_f213 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 31)) (a (.wire 12 11)) (a (.virt 19119)) (a (.virt 19120)) := by
  have c1583 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1584 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (privateBatchWrapper2_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1725 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1726 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1727 := (privateBatchWrapper2_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1728 := (privateBatchWrapper2_copies54 a h).1
  have c1729 := (privateBatchWrapper2_copies54 a h).2.1
  have c1730 := (privateBatchWrapper2_copies54 a h).2.2.1
  have c1731 := (privateBatchWrapper2_copies54 a h).2.2.2.1
  have c1732 := (privateBatchWrapper2_copies54 a h).2.2.2.2.1
  have c1733 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.1
  have c1734 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.1
  have c1735 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.1
  have c1736 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.1
  have c1737 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.1
  have c1738 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_9 := arithEq_of_rows h (row := 34) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_9
  simp only [← c1583, ← c1584, k0, ← c1585] at e_34_9
  have e_37_12 := arithEq_of_rows h (row := 37) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_12
  simp only [← c1725, k0, ← c1726, k0, ← c1727] at e_37_12
  have e_37_13 := arithEq_of_rows h (row := 37) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_13
  simp only [← c1734, ← c1735, k0, ← c1736] at e_37_13
  have e_38_7 := arithEq_of_rows h (row := 38) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_7
  simp only [← c1728, ← c1729, ← c1730] at e_38_7
  have e_38_8 := arithEq_of_rows h (row := 38) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_8
  simp only [← c1731, ← c1732, ← c1733] at e_38_8
  refine ⟨?_, ?_⟩
  · have hc := e_38_7
    simp only [e_34_9] at hc
    linear_combination c1737.trans k1 - hc
  · have hc := e_37_13
    simp only [e_37_12, e_38_8, e_34_9] at hc
    linear_combination c1738.trans k1 - hc

theorem privateBatchWrapper2_f214 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 39)) (a (.wire 12 19)) (a (.virt 19121)) (a (.virt 19122)) := by
  have c1600 := (privateBatchWrapper2_copies50 a h).1
  have c1601 := (privateBatchWrapper2_copies50 a h).2.1
  have c1602 := (privateBatchWrapper2_copies50 a h).2.2.1
  have c1739 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1740 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1741 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1742 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1743 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1744 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1745 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1746 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1747 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1748 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1749 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1750 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1751 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1752 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_34_12 := arithEq_of_rows h (row := 34) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_12
  simp only [← c1600, ← c1601, k0, ← c1602] at e_34_12
  have e_37_14 := arithEq_of_rows h (row := 37) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_14
  simp only [← c1739, k0, ← c1740, k0, ← c1741] at e_37_14
  have e_38_9 := arithEq_of_rows h (row := 38) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_9
  simp only [← c1742, ← c1743, ← c1744] at e_38_9
  have e_38_10 := arithEq_of_rows h (row := 38) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_10
  simp only [← c1745, ← c1746, ← c1747] at e_38_10
  have e_39_0 := arithEq_of_rows h (row := 39) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_0
  simp only [← c1748, ← c1749, k0, ← c1750] at e_39_0
  refine ⟨?_, ?_⟩
  · have hc := e_38_9
    simp only [e_34_12] at hc
    linear_combination c1751.trans k1 - hc
  · have hc := e_39_0
    simp only [e_37_14, e_38_10, e_34_12] at hc
    linear_combination c1752.trans k1 - hc

theorem privateBatchWrapper2_f215 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 47) = band (a (.virt 19115)) (a (.virt 19117)) := by
  have c1753 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1754 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1755 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_11 := arithEq_of_rows h (row := 38) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_11
  simp only [← c1753, ← c1754, ← c1755] at e_38_11
  have hr := e_38_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f216 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 51) = band (a (.virt 19119)) (a (.virt 19121)) := by
  have c1756 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1757 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1758 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_38_12 := arithEq_of_rows h (row := 38) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_12
  simp only [← c1756, ← c1757, ← c1758] at e_38_12
  have hr := e_38_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f217 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 38 55) = band (a (.wire 38 47)) (a (.wire 38 51)) := by
  have c1759 := (privateBatchWrapper2_copies54 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1760 := (privateBatchWrapper2_copies55 a h).1
  have c1761 := (privateBatchWrapper2_copies55 a h).2.1
  have e_38_13 := arithEq_of_rows h (row := 38) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_13
  simp only [← c1759, ← c1760, ← c1761] at e_38_13
  have hr := e_38_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f218 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 39 7) = bselect (a (.wire 38 55)) (a (.wire 11 47)) (a (.virt 18979)) := by
  have c1762 := (privateBatchWrapper2_copies55 a h).2.2.1
  have c1763 := (privateBatchWrapper2_copies55 a h).2.2.2.1
  have c1764 := (privateBatchWrapper2_copies55 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_39_1 := arithEq_of_rows h (row := 39) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_1
  simp only [← c1762, ← c1763, ← c1764, k1] at e_39_1
  have hr := e_39_1
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f219 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 7) = a (.wire 37 31) + a (.wire 39 7) := by
  have c1765 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.1
  have c1766 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.1
  have c1767 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_36_1 := arithEq_of_rows h (row := 36) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_1
  simp only [← c1765, ← c1766, k0, ← c1767] at e_36_1
  have hr := e_36_1
  linear_combination hr

theorem privateBatchWrapper2_f220 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 55)) (a (.wire 11 55)) (a (.virt 19123)) (a (.virt 19124)) := by
  have c1768 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.1
  have c1769 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.1
  have c1770 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1771 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1772 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1773 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1774 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1775 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1776 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1777 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1778 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1779 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1780 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1781 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1782 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1783 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1784 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_38_14 := arithEq_of_rows h (row := 38) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_38_14
  simp only [← c1774, ← c1775, ← c1776] at e_38_14
  have e_39_2 := arithEq_of_rows h (row := 39) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_2
  simp only [← c1768, k0, ← c1769, k0, ← c1770] at e_39_2
  have e_39_3 := arithEq_of_rows h (row := 39) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_3
  simp only [← c1771, ← c1772, k0, ← c1773] at e_39_3
  have e_39_4 := arithEq_of_rows h (row := 39) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_4
  simp only [← c1780, ← c1781, k0, ← c1782] at e_39_4
  have e_40_0 := arithEq_of_rows h (row := 40) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_0
  simp only [← c1777, ← c1778, ← c1779] at e_40_0
  refine ⟨?_, ?_⟩
  · have hc := e_38_14
    simp only [e_39_3] at hc
    linear_combination c1783.trans k1 - hc
  · have hc := e_39_4
    simp only [e_39_2, e_40_0, e_39_3] at hc
    linear_combination c1784.trans k1 - hc

theorem privateBatchWrapper2_f221 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 3)) (a (.wire 12 3)) (a (.virt 19125)) (a (.virt 19126)) := by
  have c1785 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1786 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1787 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1788 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1789 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1790 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1791 := (privateBatchWrapper2_copies55 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1792 := (privateBatchWrapper2_copies56 a h).1
  have c1793 := (privateBatchWrapper2_copies56 a h).2.1
  have c1794 := (privateBatchWrapper2_copies56 a h).2.2.1
  have c1795 := (privateBatchWrapper2_copies56 a h).2.2.2.1
  have c1796 := (privateBatchWrapper2_copies56 a h).2.2.2.2.1
  have c1797 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.1
  have c1798 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.1
  have c1799 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.1
  have c1800 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.1
  have c1801 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_39_5 := arithEq_of_rows h (row := 39) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_5
  simp only [← c1785, k0, ← c1786, k0, ← c1787] at e_39_5
  have e_39_6 := arithEq_of_rows h (row := 39) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_6
  simp only [← c1788, ← c1789, k0, ← c1790] at e_39_6
  have e_39_7 := arithEq_of_rows h (row := 39) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_7
  simp only [← c1797, ← c1798, k0, ← c1799] at e_39_7
  have e_40_1 := arithEq_of_rows h (row := 40) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_1
  simp only [← c1791, ← c1792, ← c1793] at e_40_1
  have e_40_2 := arithEq_of_rows h (row := 40) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_2
  simp only [← c1794, ← c1795, ← c1796] at e_40_2
  refine ⟨?_, ?_⟩
  · have hc := e_40_1
    simp only [e_39_6] at hc
    linear_combination c1800.trans k1 - hc
  · have hc := e_39_7
    simp only [e_39_5, e_40_2, e_39_6] at hc
    linear_combination c1801.trans k1 - hc

theorem privateBatchWrapper2_f222 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 11)) (a (.wire 12 11)) (a (.virt 19127)) (a (.virt 19128)) := by
  have c1802 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1803 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1804 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1805 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1806 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1807 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1808 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1809 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1810 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1811 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1812 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1813 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1814 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1815 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1816 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1817 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1818 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_39_8 := arithEq_of_rows h (row := 39) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_8
  simp only [← c1802, k0, ← c1803, k0, ← c1804] at e_39_8
  have e_39_9 := arithEq_of_rows h (row := 39) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_9
  simp only [← c1805, ← c1806, k0, ← c1807] at e_39_9
  have e_39_10 := arithEq_of_rows h (row := 39) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_10
  simp only [← c1814, ← c1815, k0, ← c1816] at e_39_10
  have e_40_3 := arithEq_of_rows h (row := 40) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_3
  simp only [← c1808, ← c1809, ← c1810] at e_40_3
  have e_40_4 := arithEq_of_rows h (row := 40) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_4
  simp only [← c1811, ← c1812, ← c1813] at e_40_4
  refine ⟨?_, ?_⟩
  · have hc := e_40_3
    simp only [e_39_9] at hc
    linear_combination c1817.trans k1 - hc
  · have hc := e_39_10
    simp only [e_39_8, e_40_4, e_39_9] at hc
    linear_combination c1818.trans k1 - hc

theorem privateBatchWrapper2_f223 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 19)) (a (.wire 12 19)) (a (.virt 19129)) (a (.virt 19130)) := by
  have c1819 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1820 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1821 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1822 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1823 := (privateBatchWrapper2_copies56 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1824 := (privateBatchWrapper2_copies57 a h).1
  have c1825 := (privateBatchWrapper2_copies57 a h).2.1
  have c1826 := (privateBatchWrapper2_copies57 a h).2.2.1
  have c1827 := (privateBatchWrapper2_copies57 a h).2.2.2.1
  have c1828 := (privateBatchWrapper2_copies57 a h).2.2.2.2.1
  have c1829 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.1
  have c1830 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.1
  have c1831 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.1
  have c1832 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.1
  have c1833 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.1
  have c1834 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1835 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_39_11 := arithEq_of_rows h (row := 39) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_11
  simp only [← c1819, k0, ← c1820, k0, ← c1821] at e_39_11
  have e_39_12 := arithEq_of_rows h (row := 39) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_12
  simp only [← c1822, ← c1823, k0, ← c1824] at e_39_12
  have e_39_13 := arithEq_of_rows h (row := 39) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_13
  simp only [← c1831, ← c1832, k0, ← c1833] at e_39_13
  have e_40_5 := arithEq_of_rows h (row := 40) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_5
  simp only [← c1825, ← c1826, ← c1827] at e_40_5
  have e_40_6 := arithEq_of_rows h (row := 40) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_6
  simp only [← c1828, ← c1829, ← c1830] at e_40_6
  refine ⟨?_, ?_⟩
  · have hc := e_40_5
    simp only [e_39_12] at hc
    linear_combination c1834.trans k1 - hc
  · have hc := e_39_13
    simp only [e_39_11, e_40_6, e_39_12] at hc
    linear_combination c1835.trans k1 - hc

theorem privateBatchWrapper2_f224 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 40 31) = band (a (.virt 19123)) (a (.virt 19125)) := by
  have c1836 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1837 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1838 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_7 := arithEq_of_rows h (row := 40) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_7
  simp only [← c1836, ← c1837, ← c1838] at e_40_7
  have hr := e_40_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f225 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 40 35) = band (a (.virt 19127)) (a (.virt 19129)) := by
  have c1839 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1840 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1841 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_8 := arithEq_of_rows h (row := 40) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_8
  simp only [← c1839, ← c1840, ← c1841] at e_40_8
  have hr := e_40_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f226 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 40 39) = band (a (.wire 40 31)) (a (.wire 40 35)) := by
  have c1842 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1843 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1844 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_40_9 := arithEq_of_rows h (row := 40) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_9
  simp only [← c1842, ← c1843, ← c1844] at e_40_9
  have hr := e_40_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f227 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 39 59) = bselect (a (.wire 40 39)) (a (.wire 12 27)) (a (.virt 18979)) := by
  have c1845 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1846 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1847 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_39_14 := arithEq_of_rows h (row := 39) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_39_14
  simp only [← c1845, ← c1846, ← c1847, k1] at e_39_14
  have hr := e_39_14
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f228 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 11) = a (.wire 36 7) + a (.wire 39 59) := by
  have c1848 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1849 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1850 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_36_2 := arithEq_of_rows h (row := 36) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_2
  simp only [← c1848, ← c1849, k0, ← c1850] at e_36_2
  have hr := e_36_2
  linear_combination hr

theorem privateBatchWrapper2_f229 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 35)) (a (.wire 11 55)) (a (.virt 19131)) (a (.virt 19132)) := by
  have c1851 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1852 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1853 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1854 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1855 := (privateBatchWrapper2_copies57 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1856 := (privateBatchWrapper2_copies58 a h).1
  have c1857 := (privateBatchWrapper2_copies58 a h).2.1
  have c1858 := (privateBatchWrapper2_copies58 a h).2.2.1
  have c1859 := (privateBatchWrapper2_copies58 a h).2.2.2.1
  have c1860 := (privateBatchWrapper2_copies58 a h).2.2.2.2.1
  have c1861 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.1
  have c1862 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.1
  have c1863 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.1
  have c1864 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.1
  have c1865 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.1
  have c1866 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1867 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_40_10 := arithEq_of_rows h (row := 40) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_10
  simp only [← c1857, ← c1858, ← c1859] at e_40_10
  have e_40_11 := arithEq_of_rows h (row := 40) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_11
  simp only [← c1860, ← c1861, ← c1862] at e_40_11
  have e_41_0 := arithEq_of_rows h (row := 41) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_0
  simp only [← c1851, k0, ← c1852, k0, ← c1853] at e_41_0
  have e_41_1 := arithEq_of_rows h (row := 41) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_1
  simp only [← c1854, ← c1855, k0, ← c1856] at e_41_1
  have e_41_2 := arithEq_of_rows h (row := 41) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_2
  simp only [← c1863, ← c1864, k0, ← c1865] at e_41_2
  refine ⟨?_, ?_⟩
  · have hc := e_40_10
    simp only [e_41_1] at hc
    linear_combination c1866.trans k1 - hc
  · have hc := e_41_2
    simp only [e_41_0, e_40_11, e_41_1] at hc
    linear_combination c1867.trans k1 - hc

theorem privateBatchWrapper2_f230 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 43)) (a (.wire 12 3)) (a (.virt 19133)) (a (.virt 19134)) := by
  have c1868 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1869 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1870 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1871 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1872 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1873 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1874 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1875 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1876 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1877 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1878 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1879 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1880 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1881 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1882 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1883 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1884 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_40_12 := arithEq_of_rows h (row := 40) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_12
  simp only [← c1874, ← c1875, ← c1876] at e_40_12
  have e_40_13 := arithEq_of_rows h (row := 40) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_13
  simp only [← c1877, ← c1878, ← c1879] at e_40_13
  have e_41_3 := arithEq_of_rows h (row := 41) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_3
  simp only [← c1868, k0, ← c1869, k0, ← c1870] at e_41_3
  have e_41_4 := arithEq_of_rows h (row := 41) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_4
  simp only [← c1871, ← c1872, k0, ← c1873] at e_41_4
  have e_41_5 := arithEq_of_rows h (row := 41) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_5
  simp only [← c1880, ← c1881, k0, ← c1882] at e_41_5
  refine ⟨?_, ?_⟩
  · have hc := e_40_12
    simp only [e_41_4] at hc
    linear_combination c1883.trans k1 - hc
  · have hc := e_41_5
    simp only [e_41_3, e_40_13, e_41_4] at hc
    linear_combination c1884.trans k1 - hc

theorem privateBatchWrapper2_f231 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 51)) (a (.wire 12 11)) (a (.virt 19135)) (a (.virt 19136)) := by
  have c1885 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1886 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1887 := (privateBatchWrapper2_copies58 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1888 := (privateBatchWrapper2_copies59 a h).1
  have c1889 := (privateBatchWrapper2_copies59 a h).2.1
  have c1890 := (privateBatchWrapper2_copies59 a h).2.2.1
  have c1891 := (privateBatchWrapper2_copies59 a h).2.2.2.1
  have c1892 := (privateBatchWrapper2_copies59 a h).2.2.2.2.1
  have c1893 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.1
  have c1894 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.1
  have c1895 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.1
  have c1896 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.1
  have c1897 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.1
  have c1898 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1899 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1900 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1901 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_40_14 := arithEq_of_rows h (row := 40) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_40_14
  simp only [← c1891, ← c1892, ← c1893] at e_40_14
  have e_41_6 := arithEq_of_rows h (row := 41) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_6
  simp only [← c1885, k0, ← c1886, k0, ← c1887] at e_41_6
  have e_41_7 := arithEq_of_rows h (row := 41) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_7
  simp only [← c1888, ← c1889, k0, ← c1890] at e_41_7
  have e_41_8 := arithEq_of_rows h (row := 41) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_8
  simp only [← c1897, ← c1898, k0, ← c1899] at e_41_8
  have e_42_0 := arithEq_of_rows h (row := 42) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c1894, ← c1895, ← c1896] at e_42_0
  refine ⟨?_, ?_⟩
  · have hc := e_40_14
    simp only [e_41_7] at hc
    linear_combination c1900.trans k1 - hc
  · have hc := e_41_8
    simp only [e_41_6, e_42_0, e_41_7] at hc
    linear_combination c1901.trans k1 - hc

theorem privateBatchWrapper2_f232 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 59)) (a (.wire 12 19)) (a (.virt 19137)) (a (.virt 19138)) := by
  have c1902 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1903 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1904 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1905 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1906 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1907 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1908 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1909 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1910 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1911 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1912 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1913 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1914 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1915 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1916 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1917 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1918 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_41_9 := arithEq_of_rows h (row := 41) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_9
  simp only [← c1902, k0, ← c1903, k0, ← c1904] at e_41_9
  have e_41_10 := arithEq_of_rows h (row := 41) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_10
  simp only [← c1905, ← c1906, k0, ← c1907] at e_41_10
  have e_41_11 := arithEq_of_rows h (row := 41) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_11
  simp only [← c1914, ← c1915, k0, ← c1916] at e_41_11
  have e_42_1 := arithEq_of_rows h (row := 42) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_1
  simp only [← c1908, ← c1909, ← c1910] at e_42_1
  have e_42_2 := arithEq_of_rows h (row := 42) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_2
  simp only [← c1911, ← c1912, ← c1913] at e_42_2
  refine ⟨?_, ?_⟩
  · have hc := e_42_1
    simp only [e_41_10] at hc
    linear_combination c1917.trans k1 - hc
  · have hc := e_41_11
    simp only [e_41_9, e_42_2, e_41_10] at hc
    linear_combination c1918.trans k1 - hc

theorem privateBatchWrapper2_f233 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 42 15) = band (a (.virt 19131)) (a (.virt 19133)) := by
  have c1919 := (privateBatchWrapper2_copies59 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1920 := (privateBatchWrapper2_copies60 a h).1
  have c1921 := (privateBatchWrapper2_copies60 a h).2.1
  have e_42_3 := arithEq_of_rows h (row := 42) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c1919, ← c1920, ← c1921] at e_42_3
  have hr := e_42_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f234 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 42 19) = band (a (.virt 19135)) (a (.virt 19137)) := by
  have c1922 := (privateBatchWrapper2_copies60 a h).2.2.1
  have c1923 := (privateBatchWrapper2_copies60 a h).2.2.2.1
  have c1924 := (privateBatchWrapper2_copies60 a h).2.2.2.2.1
  have e_42_4 := arithEq_of_rows h (row := 42) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_4
  simp only [← c1922, ← c1923, ← c1924] at e_42_4
  have hr := e_42_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f235 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 42 23) = band (a (.wire 42 15)) (a (.wire 42 19)) := by
  have c1925 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.1
  have c1926 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.1
  have c1927 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.1
  have e_42_5 := arithEq_of_rows h (row := 42) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_5
  simp only [← c1925, ← c1926, ← c1927] at e_42_5
  have hr := e_42_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f236 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 41 51) = bselect (a (.wire 42 23)) (a (.wire 13 7)) (a (.virt 18979)) := by
  have c1928 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.1
  have c1929 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.1
  have c1930 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_41_12 := arithEq_of_rows h (row := 41) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_12
  simp only [← c1928, ← c1929, ← c1930, k1] at e_41_12
  have hr := e_41_12
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f237 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 15) = a (.wire 36 11) + a (.wire 41 51) := by
  have c1931 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1932 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1933 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_36_3 := arithEq_of_rows h (row := 36) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_3
  simp only [← c1931, ← c1932, k0, ← c1933] at e_36_3
  have hr := e_36_3
  linear_combination hr

theorem privateBatchWrapper2_f238 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 41 59) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 36 15)) := by
  have c1934 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1935 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1936 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1937 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1938 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1939 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_41_13 := arithEq_of_rows h (row := 41) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_13
  simp only [← c1934, ← c1935, ← c1936] at e_41_13
  have e_41_14 := arithEq_of_rows h (row := 41) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_41_14
  simp only [← c1937, ← c1938, k1, ← c1939] at e_41_14
  have hr := e_41_14
  simp only [e_41_13] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f239 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 43 7) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 11 55)) := by
  have c1940 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1941 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1942 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1943 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1944 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1945 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_43_0 := arithEq_of_rows h (row := 43) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_0
  simp only [← c1940, ← c1941, ← c1942] at e_43_0
  have e_43_1 := arithEq_of_rows h (row := 43) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_1
  simp only [← c1943, ← c1944, k1, ← c1945] at e_43_1
  have hr := e_43_1
  simp only [e_43_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f240 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 43 15) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 12 3)) := by
  have c1946 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1947 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1948 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1949 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1950 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1951 := (privateBatchWrapper2_copies60 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_43_2 := arithEq_of_rows h (row := 43) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_2
  simp only [← c1946, ← c1947, ← c1948] at e_43_2
  have e_43_3 := arithEq_of_rows h (row := 43) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_3
  simp only [← c1949, ← c1950, k1, ← c1951] at e_43_3
  have hr := e_43_3
  simp only [e_43_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f241 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 43 23) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 12 11)) := by
  have c1952 := (privateBatchWrapper2_copies61 a h).1
  have c1953 := (privateBatchWrapper2_copies61 a h).2.1
  have c1954 := (privateBatchWrapper2_copies61 a h).2.2.1
  have c1955 := (privateBatchWrapper2_copies61 a h).2.2.2.1
  have c1956 := (privateBatchWrapper2_copies61 a h).2.2.2.2.1
  have c1957 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_43_4 := arithEq_of_rows h (row := 43) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_4
  simp only [← c1952, ← c1953, ← c1954] at e_43_4
  have e_43_5 := arithEq_of_rows h (row := 43) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_5
  simp only [← c1955, ← c1956, k1, ← c1957] at e_43_5
  have hr := e_43_5
  simp only [e_43_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f242 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 43 31) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 12 19)) := by
  have c1958 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.1
  have c1959 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.1
  have c1960 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.1
  have c1961 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.1
  have c1962 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1963 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_43_6 := arithEq_of_rows h (row := 43) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_6
  simp only [← c1958, ← c1959, ← c1960] at e_43_6
  have e_43_7 := arithEq_of_rows h (row := 43) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_7
  simp only [← c1961, ← c1962, k1, ← c1963] at e_43_7
  have hr := e_43_7
  simp only [e_43_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f243 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 41 59)) 32 := by
  have c1964 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1965 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1966 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1967 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1968 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1969 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1970 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1971 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1972 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1973 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1974 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1975 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1976 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1977 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1978 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1979 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1980 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1981 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1982 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1983 := (privateBatchWrapper2_copies61 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1984 := (privateBatchWrapper2_copies62 a h).1
  have c1985 := (privateBatchWrapper2_copies62 a h).2.1
  have c1986 := (privateBatchWrapper2_copies62 a h).2.2.1
  have c1987 := (privateBatchWrapper2_copies62 a h).2.2.2.1
  have c1988 := (privateBatchWrapper2_copies62 a h).2.2.2.2.1
  have c1989 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.1
  have c1990 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.1
  have c1991 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 44) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1964.trans k1
    · exact c1965.trans k1
    · exact c1966.trans k1
    · exact c1967.trans k1
    · exact c1968.trans k1
    · exact c1969.trans k1
    · exact c1970.trans k1
    · exact c1971.trans k1
    · exact c1972.trans k1
    · exact c1973.trans k1
    · exact c1974.trans k1
    · exact c1975.trans k1
    · exact c1976.trans k1
    · exact c1977.trans k1
    · exact c1978.trans k1
    · exact c1979.trans k1
    · exact c1980.trans k1
    · exact c1981.trans k1
    · exact c1982.trans k1
    · exact c1983.trans k1
    · exact c1984.trans k1
    · exact c1985.trans k1
    · exact c1986.trans k1
    · exact c1987.trans k1
    · exact c1988.trans k1
    · exact c1989.trans k1
    · exact c1990.trans k1
    )
  rwa [c1991] at hr

theorem privateBatchWrapper2_f244 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 12 35)) (a (.virt 19139)) (a (.virt 19140)) := by
  have c1992 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.1
  have c1993 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.1
  have c1994 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1995 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1996 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1997 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1998 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1999 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2000 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2001 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2002 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2003 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2004 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2005 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2006 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2007 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2008 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_42_6 := arithEq_of_rows h (row := 42) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_6
  simp only [← c1998, ← c1999, ← c2000] at e_42_6
  have e_42_7 := arithEq_of_rows h (row := 42) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c2001, ← c2002, ← c2003] at e_42_7
  have e_43_8 := arithEq_of_rows h (row := 43) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_8
  simp only [← c1992, k0, ← c1993, k0, ← c1994] at e_43_8
  have e_43_9 := arithEq_of_rows h (row := 43) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_9
  simp only [← c1995, ← c1996, k0, ← c1997] at e_43_9
  have e_43_10 := arithEq_of_rows h (row := 43) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_10
  simp only [← c2004, ← c2005, k0, ← c2006] at e_43_10
  refine ⟨?_, ?_⟩
  · have hc := e_42_6
    simp only [e_43_9] at hc
    linear_combination c2007.trans k1 - hc
  · have hc := e_43_10
    simp only [e_43_8, e_42_7, e_43_9] at hc
    linear_combination c2008.trans k1 - hc

theorem privateBatchWrapper2_f245 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 12 43)) (a (.virt 19141)) (a (.virt 19142)) := by
  have c2009 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2010 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2011 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2012 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2013 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2014 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2015 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2016 := (privateBatchWrapper2_copies63 a h).1
  have c2017 := (privateBatchWrapper2_copies63 a h).2.1
  have c2018 := (privateBatchWrapper2_copies63 a h).2.2.1
  have c2019 := (privateBatchWrapper2_copies63 a h).2.2.2.1
  have c2020 := (privateBatchWrapper2_copies63 a h).2.2.2.2.1
  have c2021 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.1
  have c2022 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.1
  have c2023 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.1
  have c2024 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.1
  have c2025 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_42_8 := arithEq_of_rows h (row := 42) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_8
  simp only [← c2015, ← c2016, ← c2017] at e_42_8
  have e_42_9 := arithEq_of_rows h (row := 42) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_9
  simp only [← c2018, ← c2019, ← c2020] at e_42_9
  have e_43_11 := arithEq_of_rows h (row := 43) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_11
  simp only [← c2009, k0, ← c2010, k0, ← c2011] at e_43_11
  have e_43_12 := arithEq_of_rows h (row := 43) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_12
  simp only [← c2012, ← c2013, k0, ← c2014] at e_43_12
  have e_43_13 := arithEq_of_rows h (row := 43) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_13
  simp only [← c2021, ← c2022, k0, ← c2023] at e_43_13
  refine ⟨?_, ?_⟩
  · have hc := e_42_8
    simp only [e_43_12] at hc
    linear_combination c2024.trans k1 - hc
  · have hc := e_43_13
    simp only [e_43_11, e_42_9, e_43_12] at hc
    linear_combination c2025.trans k1 - hc

theorem privateBatchWrapper2_f246 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 12 51)) (a (.virt 19143)) (a (.virt 19144)) := by
  have c2026 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2027 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2028 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2029 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2030 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2031 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2032 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2033 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2034 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2035 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2036 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2037 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2038 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2039 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2040 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2041 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2042 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_42_10 := arithEq_of_rows h (row := 42) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c2032, ← c2033, ← c2034] at e_42_10
  have e_42_11 := arithEq_of_rows h (row := 42) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_11
  simp only [← c2035, ← c2036, ← c2037] at e_42_11
  have e_43_14 := arithEq_of_rows h (row := 43) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_14
  simp only [← c2026, k0, ← c2027, k0, ← c2028] at e_43_14
  have e_45_0 := arithEq_of_rows h (row := 45) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_0
  simp only [← c2029, ← c2030, k0, ← c2031] at e_45_0
  have e_45_1 := arithEq_of_rows h (row := 45) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_1
  simp only [← c2038, ← c2039, k0, ← c2040] at e_45_1
  refine ⟨?_, ?_⟩
  · have hc := e_42_10
    simp only [e_45_0] at hc
    linear_combination c2041.trans k1 - hc
  · have hc := e_45_1
    simp only [e_43_14, e_42_11, e_45_0] at hc
    linear_combination c2042.trans k1 - hc

theorem privateBatchWrapper2_f247 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 12 59)) (a (.virt 19145)) (a (.virt 19146)) := by
  have c2043 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2044 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2045 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2046 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2047 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2048 := (privateBatchWrapper2_copies64 a h).1
  have c2049 := (privateBatchWrapper2_copies64 a h).2.1
  have c2050 := (privateBatchWrapper2_copies64 a h).2.2.1
  have c2051 := (privateBatchWrapper2_copies64 a h).2.2.2.1
  have c2052 := (privateBatchWrapper2_copies64 a h).2.2.2.2.1
  have c2053 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.1
  have c2054 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.1
  have c2055 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.1
  have c2056 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.1
  have c2057 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.1
  have c2058 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2059 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_42_12 := arithEq_of_rows h (row := 42) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_12
  simp only [← c2049, ← c2050, ← c2051] at e_42_12
  have e_42_13 := arithEq_of_rows h (row := 42) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_13
  simp only [← c2052, ← c2053, ← c2054] at e_42_13
  have e_45_2 := arithEq_of_rows h (row := 45) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_2
  simp only [← c2043, k0, ← c2044, k0, ← c2045] at e_45_2
  have e_45_3 := arithEq_of_rows h (row := 45) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_3
  simp only [← c2046, ← c2047, k0, ← c2048] at e_45_3
  have e_45_4 := arithEq_of_rows h (row := 45) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_4
  simp only [← c2055, ← c2056, k0, ← c2057] at e_45_4
  refine ⟨?_, ?_⟩
  · have hc := e_42_12
    simp only [e_45_3] at hc
    linear_combination c2058.trans k1 - hc
  · have hc := e_45_4
    simp only [e_45_2, e_42_13, e_45_3] at hc
    linear_combination c2059.trans k1 - hc

theorem privateBatchWrapper2_f248 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 42 59) = band (a (.virt 19139)) (a (.virt 19141)) := by
  have c2060 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2061 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2062 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_42_14 := arithEq_of_rows h (row := 42) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_14
  simp only [← c2060, ← c2061, ← c2062] at e_42_14
  have hr := e_42_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f249 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 3) = band (a (.virt 19143)) (a (.virt 19145)) := by
  have c2063 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2064 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2065 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_46_0 := arithEq_of_rows h (row := 46) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_0
  simp only [← c2063, ← c2064, ← c2065] at e_46_0
  have hr := e_46_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f250 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 7) = band (a (.wire 42 59)) (a (.wire 46 3)) := by
  have c2066 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2067 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2068 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_46_1 := arithEq_of_rows h (row := 46) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_1
  simp only [← c2066, ← c2067, ← c2068] at e_46_1
  have hr := e_46_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f251 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 7) = bor (a (.virt 18979)) (a (.wire 46 7)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bor, k1]
  ring

theorem privateBatchWrapper2_f252 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 15)) (a (.wire 12 35)) (a (.virt 19147)) (a (.virt 19148)) := by
  have c2069 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2070 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2071 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2072 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2073 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2074 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2075 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2076 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2077 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2078 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2079 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2080 := (privateBatchWrapper2_copies65 a h).1
  have c2081 := (privateBatchWrapper2_copies65 a h).2.1
  have c2082 := (privateBatchWrapper2_copies65 a h).2.2.1
  have c2083 := (privateBatchWrapper2_copies65 a h).2.2.2.1
  have c2084 := (privateBatchWrapper2_copies65 a h).2.2.2.2.1
  have c2085 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_5 := arithEq_of_rows h (row := 45) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_5
  simp only [← c2069, k0, ← c2070, k0, ← c2071] at e_45_5
  have e_45_6 := arithEq_of_rows h (row := 45) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_6
  simp only [← c2072, ← c2073, k0, ← c2074] at e_45_6
  have e_45_7 := arithEq_of_rows h (row := 45) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_7
  simp only [← c2081, ← c2082, k0, ← c2083] at e_45_7
  have e_46_2 := arithEq_of_rows h (row := 46) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_2
  simp only [← c2075, ← c2076, ← c2077] at e_46_2
  have e_46_3 := arithEq_of_rows h (row := 46) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_3
  simp only [← c2078, ← c2079, ← c2080] at e_46_3
  refine ⟨?_, ?_⟩
  · have hc := e_46_2
    simp only [e_45_6] at hc
    linear_combination c2084.trans k1 - hc
  · have hc := e_45_7
    simp only [e_45_5, e_46_3, e_45_6] at hc
    linear_combination c2085.trans k1 - hc

theorem privateBatchWrapper2_f253 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 23)) (a (.wire 12 43)) (a (.virt 19149)) (a (.virt 19150)) := by
  have c2086 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.1
  have c2087 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.1
  have c2088 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.1
  have c2089 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.1
  have c2090 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2091 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2092 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2093 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2094 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2095 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2096 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2097 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2098 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2099 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2100 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2101 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2102 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_8 := arithEq_of_rows h (row := 45) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_8
  simp only [← c2086, k0, ← c2087, k0, ← c2088] at e_45_8
  have e_45_9 := arithEq_of_rows h (row := 45) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_9
  simp only [← c2089, ← c2090, k0, ← c2091] at e_45_9
  have e_45_10 := arithEq_of_rows h (row := 45) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_10
  simp only [← c2098, ← c2099, k0, ← c2100] at e_45_10
  have e_46_4 := arithEq_of_rows h (row := 46) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_4
  simp only [← c2092, ← c2093, ← c2094] at e_46_4
  have e_46_5 := arithEq_of_rows h (row := 46) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_5
  simp only [← c2095, ← c2096, ← c2097] at e_46_5
  refine ⟨?_, ?_⟩
  · have hc := e_46_4
    simp only [e_45_9] at hc
    linear_combination c2101.trans k1 - hc
  · have hc := e_45_10
    simp only [e_45_8, e_46_5, e_45_9] at hc
    linear_combination c2102.trans k1 - hc

theorem privateBatchWrapper2_f254 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 31)) (a (.wire 12 51)) (a (.virt 19151)) (a (.virt 19152)) := by
  have c2103 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2104 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2105 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2106 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2107 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2108 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2109 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2110 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2111 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2112 := (privateBatchWrapper2_copies66 a h).1
  have c2113 := (privateBatchWrapper2_copies66 a h).2.1
  have c2114 := (privateBatchWrapper2_copies66 a h).2.2.1
  have c2115 := (privateBatchWrapper2_copies66 a h).2.2.2.1
  have c2116 := (privateBatchWrapper2_copies66 a h).2.2.2.2.1
  have c2117 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.1
  have c2118 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.1
  have c2119 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_11 := arithEq_of_rows h (row := 45) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_11
  simp only [← c2103, k0, ← c2104, k0, ← c2105] at e_45_11
  have e_45_12 := arithEq_of_rows h (row := 45) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_12
  simp only [← c2106, ← c2107, k0, ← c2108] at e_45_12
  have e_45_13 := arithEq_of_rows h (row := 45) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_13
  simp only [← c2115, ← c2116, k0, ← c2117] at e_45_13
  have e_46_6 := arithEq_of_rows h (row := 46) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_6
  simp only [← c2109, ← c2110, ← c2111] at e_46_6
  have e_46_7 := arithEq_of_rows h (row := 46) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_7
  simp only [← c2112, ← c2113, ← c2114] at e_46_7
  refine ⟨?_, ?_⟩
  · have hc := e_46_6
    simp only [e_45_12] at hc
    linear_combination c2118.trans k1 - hc
  · have hc := e_45_13
    simp only [e_45_11, e_46_7, e_45_12] at hc
    linear_combination c2119.trans k1 - hc

theorem privateBatchWrapper2_f255 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 39)) (a (.wire 12 59)) (a (.virt 19153)) (a (.virt 19154)) := by
  have c2120 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.1
  have c2121 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.1
  have c2122 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2123 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2124 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2125 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2126 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2127 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2128 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2129 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2130 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2131 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2132 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2133 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2134 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2135 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2136 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_14 := arithEq_of_rows h (row := 45) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_14
  simp only [← c2120, k0, ← c2121, k0, ← c2122] at e_45_14
  have e_46_8 := arithEq_of_rows h (row := 46) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_8
  simp only [← c2126, ← c2127, ← c2128] at e_46_8
  have e_46_9 := arithEq_of_rows h (row := 46) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_9
  simp only [← c2129, ← c2130, ← c2131] at e_46_9
  have e_47_0 := arithEq_of_rows h (row := 47) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_0
  simp only [← c2123, ← c2124, k0, ← c2125] at e_47_0
  have e_47_1 := arithEq_of_rows h (row := 47) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_1
  simp only [← c2132, ← c2133, k0, ← c2134] at e_47_1
  refine ⟨?_, ?_⟩
  · have hc := e_46_8
    simp only [e_47_0] at hc
    linear_combination c2135.trans k1 - hc
  · have hc := e_47_1
    simp only [e_45_14, e_46_9, e_47_0] at hc
    linear_combination c2136.trans k1 - hc

theorem privateBatchWrapper2_f256 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 43) = band (a (.virt 19147)) (a (.virt 19149)) := by
  have c2137 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2138 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2139 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_46_10 := arithEq_of_rows h (row := 46) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_10
  simp only [← c2137, ← c2138, ← c2139] at e_46_10
  have hr := e_46_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f257 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 47) = band (a (.virt 19151)) (a (.virt 19153)) := by
  have c2140 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2141 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2142 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_46_11 := arithEq_of_rows h (row := 46) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_11
  simp only [← c2140, ← c2141, ← c2142] at e_46_11
  have hr := e_46_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f258 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 46 51) = band (a (.wire 46 43)) (a (.wire 46 47)) := by
  have c2143 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2144 := (privateBatchWrapper2_copies67 a h).1
  have c2145 := (privateBatchWrapper2_copies67 a h).2.1
  have e_46_12 := arithEq_of_rows h (row := 46) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_12
  simp only [← c2143, ← c2144, ← c2145] at e_46_12
  have hr := e_46_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f259 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 19) = bor (a (.wire 46 7)) (a (.wire 46 51)) := by
  have c2146 := (privateBatchWrapper2_copies67 a h).2.2.1
  have c2147 := (privateBatchWrapper2_copies67 a h).2.2.2.1
  have c2148 := (privateBatchWrapper2_copies67 a h).2.2.2.2.1
  have c2149 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.1
  have c2150 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.1
  have c2151 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c2146, ← c2147, ← c2148] at e_5_6
  have e_36_4 := arithEq_of_rows h (row := 36) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_4
  simp only [← c2149, ← c2150, k0, ← c2151] at e_36_4
  have hr := e_36_4
  simp only [e_5_6] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f260 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 55)) (a (.wire 12 35)) (a (.virt 19155)) (a (.virt 19156)) := by
  have c2152 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.1
  have c2153 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.1
  have c2154 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2155 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2156 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2157 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2158 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2159 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2160 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2161 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2162 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2163 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2164 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2165 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2166 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2167 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2168 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_46_13 := arithEq_of_rows h (row := 46) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_13
  simp only [← c2158, ← c2159, ← c2160] at e_46_13
  have e_46_14 := arithEq_of_rows h (row := 46) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_46_14
  simp only [← c2161, ← c2162, ← c2163] at e_46_14
  have e_47_2 := arithEq_of_rows h (row := 47) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_2
  simp only [← c2152, k0, ← c2153, k0, ← c2154] at e_47_2
  have e_47_3 := arithEq_of_rows h (row := 47) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_3
  simp only [← c2155, ← c2156, k0, ← c2157] at e_47_3
  have e_47_4 := arithEq_of_rows h (row := 47) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_4
  simp only [← c2164, ← c2165, k0, ← c2166] at e_47_4
  refine ⟨?_, ?_⟩
  · have hc := e_46_13
    simp only [e_47_3] at hc
    linear_combination c2167.trans k1 - hc
  · have hc := e_47_4
    simp only [e_47_2, e_46_14, e_47_3] at hc
    linear_combination c2168.trans k1 - hc

theorem privateBatchWrapper2_f261 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 3)) (a (.wire 12 43)) (a (.virt 19157)) (a (.virt 19158)) := by
  have c2169 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2170 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2171 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2172 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2173 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2174 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2175 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2176 := (privateBatchWrapper2_copies68 a h).1
  have c2177 := (privateBatchWrapper2_copies68 a h).2.1
  have c2178 := (privateBatchWrapper2_copies68 a h).2.2.1
  have c2179 := (privateBatchWrapper2_copies68 a h).2.2.2.1
  have c2180 := (privateBatchWrapper2_copies68 a h).2.2.2.2.1
  have c2181 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.1
  have c2182 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.1
  have c2183 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.1
  have c2184 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.1
  have c2185 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_5 := arithEq_of_rows h (row := 47) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_5
  simp only [← c2169, k0, ← c2170, k0, ← c2171] at e_47_5
  have e_47_6 := arithEq_of_rows h (row := 47) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_6
  simp only [← c2172, ← c2173, k0, ← c2174] at e_47_6
  have e_47_7 := arithEq_of_rows h (row := 47) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_7
  simp only [← c2181, ← c2182, k0, ← c2183] at e_47_7
  have e_48_0 := arithEq_of_rows h (row := 48) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_0
  simp only [← c2175, ← c2176, ← c2177] at e_48_0
  have e_48_1 := arithEq_of_rows h (row := 48) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_1
  simp only [← c2178, ← c2179, ← c2180] at e_48_1
  refine ⟨?_, ?_⟩
  · have hc := e_48_0
    simp only [e_47_6] at hc
    linear_combination c2184.trans k1 - hc
  · have hc := e_47_7
    simp only [e_47_5, e_48_1, e_47_6] at hc
    linear_combination c2185.trans k1 - hc

theorem privateBatchWrapper2_f262 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 11)) (a (.wire 12 51)) (a (.virt 19159)) (a (.virt 19160)) := by
  have c2186 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2187 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2188 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2189 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2190 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2191 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2192 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2193 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2194 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2195 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2196 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2197 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2198 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2199 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2200 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2201 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2202 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_8 := arithEq_of_rows h (row := 47) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_8
  simp only [← c2186, k0, ← c2187, k0, ← c2188] at e_47_8
  have e_47_9 := arithEq_of_rows h (row := 47) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_9
  simp only [← c2189, ← c2190, k0, ← c2191] at e_47_9
  have e_47_10 := arithEq_of_rows h (row := 47) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_10
  simp only [← c2198, ← c2199, k0, ← c2200] at e_47_10
  have e_48_2 := arithEq_of_rows h (row := 48) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_2
  simp only [← c2192, ← c2193, ← c2194] at e_48_2
  have e_48_3 := arithEq_of_rows h (row := 48) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_3
  simp only [← c2195, ← c2196, ← c2197] at e_48_3
  refine ⟨?_, ?_⟩
  · have hc := e_48_2
    simp only [e_47_9] at hc
    linear_combination c2201.trans k1 - hc
  · have hc := e_47_10
    simp only [e_47_8, e_48_3, e_47_9] at hc
    linear_combination c2202.trans k1 - hc

theorem privateBatchWrapper2_f263 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 19)) (a (.wire 12 59)) (a (.virt 19161)) (a (.virt 19162)) := by
  have c2203 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2204 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2205 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2206 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2207 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2208 := (privateBatchWrapper2_copies69 a h).1
  have c2209 := (privateBatchWrapper2_copies69 a h).2.1
  have c2210 := (privateBatchWrapper2_copies69 a h).2.2.1
  have c2211 := (privateBatchWrapper2_copies69 a h).2.2.2.1
  have c2212 := (privateBatchWrapper2_copies69 a h).2.2.2.2.1
  have c2213 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.1
  have c2214 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.1
  have c2215 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.1
  have c2216 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.1
  have c2217 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.1
  have c2218 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2219 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_11 := arithEq_of_rows h (row := 47) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_11
  simp only [← c2203, k0, ← c2204, k0, ← c2205] at e_47_11
  have e_47_12 := arithEq_of_rows h (row := 47) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_12
  simp only [← c2206, ← c2207, k0, ← c2208] at e_47_12
  have e_47_13 := arithEq_of_rows h (row := 47) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_13
  simp only [← c2215, ← c2216, k0, ← c2217] at e_47_13
  have e_48_4 := arithEq_of_rows h (row := 48) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_4
  simp only [← c2209, ← c2210, ← c2211] at e_48_4
  have e_48_5 := arithEq_of_rows h (row := 48) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_5
  simp only [← c2212, ← c2213, ← c2214] at e_48_5
  refine ⟨?_, ?_⟩
  · have hc := e_48_4
    simp only [e_47_12] at hc
    linear_combination c2218.trans k1 - hc
  · have hc := e_47_13
    simp only [e_47_11, e_48_5, e_47_12] at hc
    linear_combination c2219.trans k1 - hc

theorem privateBatchWrapper2_f264 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 48 27) = band (a (.virt 19155)) (a (.virt 19157)) := by
  have c2220 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2221 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2222 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_48_6 := arithEq_of_rows h (row := 48) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_6
  simp only [← c2220, ← c2221, ← c2222] at e_48_6
  have hr := e_48_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f265 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 48 31) = band (a (.virt 19159)) (a (.virt 19161)) := by
  have c2223 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2224 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2225 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_48_7 := arithEq_of_rows h (row := 48) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_7
  simp only [← c2223, ← c2224, ← c2225] at e_48_7
  have hr := e_48_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f266 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 48 35) = band (a (.wire 48 27)) (a (.wire 48 31)) := by
  have c2226 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2227 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2228 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_48_8 := arithEq_of_rows h (row := 48) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_8
  simp only [← c2226, ← c2227, ← c2228] at e_48_8
  have hr := e_48_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f267 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 23) = bor (a (.wire 36 19)) (a (.wire 48 35)) := by
  have c2229 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2230 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2231 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2232 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2233 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2234 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_7 := arithEq_of_rows h (row := 5) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c2229, ← c2230, ← c2231] at e_5_7
  have e_36_5 := arithEq_of_rows h (row := 36) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_5
  simp only [← c2232, ← c2233, k0, ← c2234] at e_36_5
  have hr := e_36_5
  simp only [e_5_7] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f268 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 35)) (a (.wire 12 35)) (a (.virt 19163)) (a (.virt 19164)) := by
  have c1995 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1996 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1997 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2235 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2236 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2237 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2238 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2239 := (privateBatchWrapper2_copies69 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2240 := (privateBatchWrapper2_copies70 a h).1
  have c2241 := (privateBatchWrapper2_copies70 a h).2.1
  have c2242 := (privateBatchWrapper2_copies70 a h).2.2.1
  have c2243 := (privateBatchWrapper2_copies70 a h).2.2.2.1
  have c2244 := (privateBatchWrapper2_copies70 a h).2.2.2.2.1
  have c2245 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.1
  have c2246 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.1
  have c2247 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.1
  have c2248 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_43_9 := arithEq_of_rows h (row := 43) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_9
  simp only [← c1995, ← c1996, k0, ← c1997] at e_43_9
  have e_47_14 := arithEq_of_rows h (row := 47) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_14
  simp only [← c2235, k0, ← c2236, k0, ← c2237] at e_47_14
  have e_48_9 := arithEq_of_rows h (row := 48) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_9
  simp only [← c2238, ← c2239, ← c2240] at e_48_9
  have e_48_10 := arithEq_of_rows h (row := 48) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_10
  simp only [← c2241, ← c2242, ← c2243] at e_48_10
  have e_49_0 := arithEq_of_rows h (row := 49) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_0
  simp only [← c2244, ← c2245, k0, ← c2246] at e_49_0
  refine ⟨?_, ?_⟩
  · have hc := e_48_9
    simp only [e_43_9] at hc
    linear_combination c2247.trans k1 - hc
  · have hc := e_49_0
    simp only [e_47_14, e_48_10, e_43_9] at hc
    linear_combination c2248.trans k1 - hc

theorem privateBatchWrapper2_f269 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 43)) (a (.wire 12 43)) (a (.virt 19165)) (a (.virt 19166)) := by
  have c2012 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2013 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2014 := (privateBatchWrapper2_copies62 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2249 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.1
  have c2250 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2251 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2252 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2253 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2254 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2255 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2256 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2257 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2258 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2259 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2260 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2261 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2262 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_43_12 := arithEq_of_rows h (row := 43) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_43_12
  simp only [← c2012, ← c2013, k0, ← c2014] at e_43_12
  have e_48_11 := arithEq_of_rows h (row := 48) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_11
  simp only [← c2252, ← c2253, ← c2254] at e_48_11
  have e_48_12 := arithEq_of_rows h (row := 48) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_12
  simp only [← c2255, ← c2256, ← c2257] at e_48_12
  have e_49_1 := arithEq_of_rows h (row := 49) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_1
  simp only [← c2249, k0, ← c2250, k0, ← c2251] at e_49_1
  have e_49_2 := arithEq_of_rows h (row := 49) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_2
  simp only [← c2258, ← c2259, k0, ← c2260] at e_49_2
  refine ⟨?_, ?_⟩
  · have hc := e_48_11
    simp only [e_43_12] at hc
    linear_combination c2261.trans k1 - hc
  · have hc := e_49_2
    simp only [e_49_1, e_48_12, e_43_12] at hc
    linear_combination c2262.trans k1 - hc

theorem privateBatchWrapper2_f270 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 51)) (a (.wire 12 51)) (a (.virt 19167)) (a (.virt 19168)) := by
  have c2029 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2030 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2031 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2263 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2264 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2265 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2266 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2267 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2268 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2269 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2270 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2271 := (privateBatchWrapper2_copies70 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2272 := (privateBatchWrapper2_copies71 a h).1
  have c2273 := (privateBatchWrapper2_copies71 a h).2.1
  have c2274 := (privateBatchWrapper2_copies71 a h).2.2.1
  have c2275 := (privateBatchWrapper2_copies71 a h).2.2.2.1
  have c2276 := (privateBatchWrapper2_copies71 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_0 := arithEq_of_rows h (row := 45) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_0
  simp only [← c2029, ← c2030, k0, ← c2031] at e_45_0
  have e_48_13 := arithEq_of_rows h (row := 48) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_13
  simp only [← c2266, ← c2267, ← c2268] at e_48_13
  have e_48_14 := arithEq_of_rows h (row := 48) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_48_14
  simp only [← c2269, ← c2270, ← c2271] at e_48_14
  have e_49_3 := arithEq_of_rows h (row := 49) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_3
  simp only [← c2263, k0, ← c2264, k0, ← c2265] at e_49_3
  have e_49_4 := arithEq_of_rows h (row := 49) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_4
  simp only [← c2272, ← c2273, k0, ← c2274] at e_49_4
  refine ⟨?_, ?_⟩
  · have hc := e_48_13
    simp only [e_45_0] at hc
    linear_combination c2275.trans k1 - hc
  · have hc := e_49_4
    simp only [e_49_3, e_48_14, e_45_0] at hc
    linear_combination c2276.trans k1 - hc

theorem privateBatchWrapper2_f271 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 9 59)) (a (.wire 12 59)) (a (.virt 19169)) (a (.virt 19170)) := by
  have c2046 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2047 := (privateBatchWrapper2_copies63 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2048 := (privateBatchWrapper2_copies64 a h).1
  have c2277 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.1
  have c2278 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.1
  have c2279 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.1
  have c2280 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.1
  have c2281 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.1
  have c2282 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2283 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2284 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2285 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2286 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2287 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2288 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2289 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2290 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_3 := arithEq_of_rows h (row := 45) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_3
  simp only [← c2046, ← c2047, k0, ← c2048] at e_45_3
  have e_49_5 := arithEq_of_rows h (row := 49) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_5
  simp only [← c2277, k0, ← c2278, k0, ← c2279] at e_49_5
  have e_49_6 := arithEq_of_rows h (row := 49) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_6
  simp only [← c2286, ← c2287, k0, ← c2288] at e_49_6
  have e_50_0 := arithEq_of_rows h (row := 50) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_0
  simp only [← c2280, ← c2281, ← c2282] at e_50_0
  have e_50_1 := arithEq_of_rows h (row := 50) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_1
  simp only [← c2283, ← c2284, ← c2285] at e_50_1
  refine ⟨?_, ?_⟩
  · have hc := e_50_0
    simp only [e_45_3] at hc
    linear_combination c2289.trans k1 - hc
  · have hc := e_49_6
    simp only [e_49_5, e_50_1, e_45_3] at hc
    linear_combination c2290.trans k1 - hc

theorem privateBatchWrapper2_f272 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 50 11) = band (a (.virt 19163)) (a (.virt 19165)) := by
  have c2291 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2292 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2293 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_2 := arithEq_of_rows h (row := 50) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_2
  simp only [← c2291, ← c2292, ← c2293] at e_50_2
  have hr := e_50_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f273 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 50 15) = band (a (.virt 19167)) (a (.virt 19169)) := by
  have c2294 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2295 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2296 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_3 := arithEq_of_rows h (row := 50) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_3
  simp only [← c2294, ← c2295, ← c2296] at e_50_3
  have hr := e_50_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f274 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 50 19) = band (a (.wire 50 11)) (a (.wire 50 15)) := by
  have c2297 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2298 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2299 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_4 := arithEq_of_rows h (row := 50) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_4
  simp only [← c2297, ← c2298, ← c2299] at e_50_4
  have hr := e_50_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f275 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 49 31) = bselect (a (.wire 50 19)) (a (.wire 11 7)) (a (.virt 18979)) := by
  have c2300 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2301 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2302 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_49_7 := arithEq_of_rows h (row := 49) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_7
  simp only [← c2300, ← c2301, ← c2302, k1] at e_49_7
  have hr := e_49_7
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f276 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 49 31) = a (.virt 18979) + a (.wire 49 31) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f277 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 15)) (a (.wire 12 35)) (a (.virt 19171)) (a (.virt 19172)) := by
  have c2072 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2073 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2074 := (privateBatchWrapper2_copies64 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2303 := (privateBatchWrapper2_copies71 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2304 := (privateBatchWrapper2_copies72 a h).1
  have c2305 := (privateBatchWrapper2_copies72 a h).2.1
  have c2306 := (privateBatchWrapper2_copies72 a h).2.2.1
  have c2307 := (privateBatchWrapper2_copies72 a h).2.2.2.1
  have c2308 := (privateBatchWrapper2_copies72 a h).2.2.2.2.1
  have c2309 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.1
  have c2310 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.1
  have c2311 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.1
  have c2312 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.1
  have c2313 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.1
  have c2314 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2315 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2316 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_6 := arithEq_of_rows h (row := 45) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_6
  simp only [← c2072, ← c2073, k0, ← c2074] at e_45_6
  have e_49_8 := arithEq_of_rows h (row := 49) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_8
  simp only [← c2303, k0, ← c2304, k0, ← c2305] at e_49_8
  have e_49_9 := arithEq_of_rows h (row := 49) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_9
  simp only [← c2312, ← c2313, k0, ← c2314] at e_49_9
  have e_50_5 := arithEq_of_rows h (row := 50) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_5
  simp only [← c2306, ← c2307, ← c2308] at e_50_5
  have e_50_6 := arithEq_of_rows h (row := 50) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_6
  simp only [← c2309, ← c2310, ← c2311] at e_50_6
  refine ⟨?_, ?_⟩
  · have hc := e_50_5
    simp only [e_45_6] at hc
    linear_combination c2315.trans k1 - hc
  · have hc := e_49_9
    simp only [e_49_8, e_50_6, e_45_6] at hc
    linear_combination c2316.trans k1 - hc

theorem privateBatchWrapper2_f278 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 23)) (a (.wire 12 43)) (a (.virt 19173)) (a (.virt 19174)) := by
  have c2089 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.1
  have c2090 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2091 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2317 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2318 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2319 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2320 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2321 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2322 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2323 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2324 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2325 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2326 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2327 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2328 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2329 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2330 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_9 := arithEq_of_rows h (row := 45) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_9
  simp only [← c2089, ← c2090, k0, ← c2091] at e_45_9
  have e_49_10 := arithEq_of_rows h (row := 49) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_10
  simp only [← c2317, k0, ← c2318, k0, ← c2319] at e_49_10
  have e_49_11 := arithEq_of_rows h (row := 49) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_11
  simp only [← c2326, ← c2327, k0, ← c2328] at e_49_11
  have e_50_7 := arithEq_of_rows h (row := 50) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_7
  simp only [← c2320, ← c2321, ← c2322] at e_50_7
  have e_50_8 := arithEq_of_rows h (row := 50) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_8
  simp only [← c2323, ← c2324, ← c2325] at e_50_8
  refine ⟨?_, ?_⟩
  · have hc := e_50_7
    simp only [e_45_9] at hc
    linear_combination c2329.trans k1 - hc
  · have hc := e_49_11
    simp only [e_49_10, e_50_8, e_45_9] at hc
    linear_combination c2330.trans k1 - hc

theorem privateBatchWrapper2_f279 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 31)) (a (.wire 12 51)) (a (.virt 19175)) (a (.virt 19176)) := by
  have c2106 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2107 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2108 := (privateBatchWrapper2_copies65 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2331 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2332 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2333 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2334 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2335 := (privateBatchWrapper2_copies72 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2336 := (privateBatchWrapper2_copies73 a h).1
  have c2337 := (privateBatchWrapper2_copies73 a h).2.1
  have c2338 := (privateBatchWrapper2_copies73 a h).2.2.1
  have c2339 := (privateBatchWrapper2_copies73 a h).2.2.2.1
  have c2340 := (privateBatchWrapper2_copies73 a h).2.2.2.2.1
  have c2341 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.1
  have c2342 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.1
  have c2343 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.1
  have c2344 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_45_12 := arithEq_of_rows h (row := 45) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_45_12
  simp only [← c2106, ← c2107, k0, ← c2108] at e_45_12
  have e_49_12 := arithEq_of_rows h (row := 49) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_12
  simp only [← c2331, k0, ← c2332, k0, ← c2333] at e_49_12
  have e_49_13 := arithEq_of_rows h (row := 49) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_13
  simp only [← c2340, ← c2341, k0, ← c2342] at e_49_13
  have e_50_9 := arithEq_of_rows h (row := 50) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_9
  simp only [← c2334, ← c2335, ← c2336] at e_50_9
  have e_50_10 := arithEq_of_rows h (row := 50) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_10
  simp only [← c2337, ← c2338, ← c2339] at e_50_10
  refine ⟨?_, ?_⟩
  · have hc := e_50_9
    simp only [e_45_12] at hc
    linear_combination c2343.trans k1 - hc
  · have hc := e_49_13
    simp only [e_49_12, e_50_10, e_45_12] at hc
    linear_combination c2344.trans k1 - hc

theorem privateBatchWrapper2_f280 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 39)) (a (.wire 12 59)) (a (.virt 19177)) (a (.virt 19178)) := by
  have c2123 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2124 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2125 := (privateBatchWrapper2_copies66 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2345 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.1
  have c2346 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2347 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2348 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2349 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2350 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2351 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2352 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2353 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2354 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2355 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2356 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2357 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2358 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_0 := arithEq_of_rows h (row := 47) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_0
  simp only [← c2123, ← c2124, k0, ← c2125] at e_47_0
  have e_49_14 := arithEq_of_rows h (row := 49) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_49_14
  simp only [← c2345, k0, ← c2346, k0, ← c2347] at e_49_14
  have e_50_11 := arithEq_of_rows h (row := 50) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_11
  simp only [← c2348, ← c2349, ← c2350] at e_50_11
  have e_50_12 := arithEq_of_rows h (row := 50) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_12
  simp only [← c2351, ← c2352, ← c2353] at e_50_12
  have e_51_0 := arithEq_of_rows h (row := 51) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_0
  simp only [← c2354, ← c2355, k0, ← c2356] at e_51_0
  refine ⟨?_, ?_⟩
  · have hc := e_50_11
    simp only [e_47_0] at hc
    linear_combination c2357.trans k1 - hc
  · have hc := e_51_0
    simp only [e_49_14, e_50_12, e_47_0] at hc
    linear_combination c2358.trans k1 - hc

theorem privateBatchWrapper2_f281 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 50 55) = band (a (.virt 19171)) (a (.virt 19173)) := by
  have c2359 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2360 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2361 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_13 := arithEq_of_rows h (row := 50) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_13
  simp only [← c2359, ← c2360, ← c2361] at e_50_13
  have hr := e_50_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f282 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 50 59) = band (a (.virt 19175)) (a (.virt 19177)) := by
  have c2362 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2363 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2364 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_50_14 := arithEq_of_rows h (row := 50) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_50_14
  simp only [← c2362, ← c2363, ← c2364] at e_50_14
  have hr := e_50_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f283 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 52 3) = band (a (.wire 50 55)) (a (.wire 50 59)) := by
  have c2365 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2366 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2367 := (privateBatchWrapper2_copies73 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_52_0 := arithEq_of_rows h (row := 52) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_0
  simp only [← c2365, ← c2366, ← c2367] at e_52_0
  have hr := e_52_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f284 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 51 7) = bselect (a (.wire 52 3)) (a (.wire 11 47)) (a (.virt 18979)) := by
  have c2368 := (privateBatchWrapper2_copies74 a h).1
  have c2369 := (privateBatchWrapper2_copies74 a h).2.1
  have c2370 := (privateBatchWrapper2_copies74 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_51_1 := arithEq_of_rows h (row := 51) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_1
  simp only [← c2368, ← c2369, ← c2370, k1] at e_51_1
  have hr := e_51_1
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f285 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 27) = a (.wire 49 31) + a (.wire 51 7) := by
  have c2371 := (privateBatchWrapper2_copies74 a h).2.2.2.1
  have c2372 := (privateBatchWrapper2_copies74 a h).2.2.2.2.1
  have c2373 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_36_6 := arithEq_of_rows h (row := 36) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_6
  simp only [← c2371, ← c2372, k0, ← c2373] at e_36_6
  have hr := e_36_6
  linear_combination hr

theorem privateBatchWrapper2_f286 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 11 55)) (a (.wire 12 35)) (a (.virt 19179)) (a (.virt 19180)) := by
  have c2155 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2156 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2157 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2374 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.1
  have c2375 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.1
  have c2376 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.1
  have c2377 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.1
  have c2378 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2379 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2380 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2381 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2382 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2383 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2384 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2385 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2386 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2387 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_3 := arithEq_of_rows h (row := 47) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_3
  simp only [← c2155, ← c2156, k0, ← c2157] at e_47_3
  have e_51_2 := arithEq_of_rows h (row := 51) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_2
  simp only [← c2374, k0, ← c2375, k0, ← c2376] at e_51_2
  have e_51_3 := arithEq_of_rows h (row := 51) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_3
  simp only [← c2383, ← c2384, k0, ← c2385] at e_51_3
  have e_52_1 := arithEq_of_rows h (row := 52) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_1
  simp only [← c2377, ← c2378, ← c2379] at e_52_1
  have e_52_2 := arithEq_of_rows h (row := 52) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_2
  simp only [← c2380, ← c2381, ← c2382] at e_52_2
  refine ⟨?_, ?_⟩
  · have hc := e_52_1
    simp only [e_47_3] at hc
    linear_combination c2386.trans k1 - hc
  · have hc := e_51_3
    simp only [e_51_2, e_52_2, e_47_3] at hc
    linear_combination c2387.trans k1 - hc

theorem privateBatchWrapper2_f287 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 3)) (a (.wire 12 43)) (a (.virt 19181)) (a (.virt 19182)) := by
  have c2172 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2173 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2174 := (privateBatchWrapper2_copies67 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2388 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2389 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2390 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2391 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2392 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2393 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2394 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2395 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2396 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2397 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2398 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2399 := (privateBatchWrapper2_copies74 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2400 := (privateBatchWrapper2_copies75 a h).1
  have c2401 := (privateBatchWrapper2_copies75 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_6 := arithEq_of_rows h (row := 47) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_6
  simp only [← c2172, ← c2173, k0, ← c2174] at e_47_6
  have e_51_4 := arithEq_of_rows h (row := 51) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_4
  simp only [← c2388, k0, ← c2389, k0, ← c2390] at e_51_4
  have e_51_5 := arithEq_of_rows h (row := 51) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_5
  simp only [← c2397, ← c2398, k0, ← c2399] at e_51_5
  have e_52_3 := arithEq_of_rows h (row := 52) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_3
  simp only [← c2391, ← c2392, ← c2393] at e_52_3
  have e_52_4 := arithEq_of_rows h (row := 52) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_4
  simp only [← c2394, ← c2395, ← c2396] at e_52_4
  refine ⟨?_, ?_⟩
  · have hc := e_52_3
    simp only [e_47_6] at hc
    linear_combination c2400.trans k1 - hc
  · have hc := e_51_5
    simp only [e_51_4, e_52_4, e_47_6] at hc
    linear_combination c2401.trans k1 - hc

theorem privateBatchWrapper2_f288 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 11)) (a (.wire 12 51)) (a (.virt 19183)) (a (.virt 19184)) := by
  have c2189 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2190 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2191 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2402 := (privateBatchWrapper2_copies75 a h).2.2.1
  have c2403 := (privateBatchWrapper2_copies75 a h).2.2.2.1
  have c2404 := (privateBatchWrapper2_copies75 a h).2.2.2.2.1
  have c2405 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.1
  have c2406 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.1
  have c2407 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.1
  have c2408 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.1
  have c2409 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.1
  have c2410 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2411 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2412 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2413 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2414 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2415 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_9 := arithEq_of_rows h (row := 47) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_9
  simp only [← c2189, ← c2190, k0, ← c2191] at e_47_9
  have e_51_6 := arithEq_of_rows h (row := 51) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_6
  simp only [← c2402, k0, ← c2403, k0, ← c2404] at e_51_6
  have e_51_7 := arithEq_of_rows h (row := 51) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_7
  simp only [← c2411, ← c2412, k0, ← c2413] at e_51_7
  have e_52_5 := arithEq_of_rows h (row := 52) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_5
  simp only [← c2405, ← c2406, ← c2407] at e_52_5
  have e_52_6 := arithEq_of_rows h (row := 52) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_6
  simp only [← c2408, ← c2409, ← c2410] at e_52_6
  refine ⟨?_, ?_⟩
  · have hc := e_52_5
    simp only [e_47_9] at hc
    linear_combination c2414.trans k1 - hc
  · have hc := e_51_7
    simp only [e_51_6, e_52_6, e_47_9] at hc
    linear_combination c2415.trans k1 - hc

theorem privateBatchWrapper2_f289 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 19)) (a (.wire 12 59)) (a (.virt 19185)) (a (.virt 19186)) := by
  have c2206 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2207 := (privateBatchWrapper2_copies68 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2208 := (privateBatchWrapper2_copies69 a h).1
  have c2416 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2417 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2418 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2419 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2420 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2421 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2422 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2423 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2424 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2425 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2426 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2427 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2428 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2429 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_47_12 := arithEq_of_rows h (row := 47) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_47_12
  simp only [← c2206, ← c2207, k0, ← c2208] at e_47_12
  have e_51_8 := arithEq_of_rows h (row := 51) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_8
  simp only [← c2416, k0, ← c2417, k0, ← c2418] at e_51_8
  have e_51_9 := arithEq_of_rows h (row := 51) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_9
  simp only [← c2425, ← c2426, k0, ← c2427] at e_51_9
  have e_52_7 := arithEq_of_rows h (row := 52) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_7
  simp only [← c2419, ← c2420, ← c2421] at e_52_7
  have e_52_8 := arithEq_of_rows h (row := 52) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_8
  simp only [← c2422, ← c2423, ← c2424] at e_52_8
  refine ⟨?_, ?_⟩
  · have hc := e_52_7
    simp only [e_47_12] at hc
    linear_combination c2428.trans k1 - hc
  · have hc := e_51_9
    simp only [e_51_8, e_52_8, e_47_12] at hc
    linear_combination c2429.trans k1 - hc

theorem privateBatchWrapper2_f290 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 52 39) = band (a (.virt 19179)) (a (.virt 19181)) := by
  have c2430 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2431 := (privateBatchWrapper2_copies75 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2432 := (privateBatchWrapper2_copies76 a h).1
  have e_52_9 := arithEq_of_rows h (row := 52) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_9
  simp only [← c2430, ← c2431, ← c2432] at e_52_9
  have hr := e_52_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f291 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 52 43) = band (a (.virt 19183)) (a (.virt 19185)) := by
  have c2433 := (privateBatchWrapper2_copies76 a h).2.1
  have c2434 := (privateBatchWrapper2_copies76 a h).2.2.1
  have c2435 := (privateBatchWrapper2_copies76 a h).2.2.2.1
  have e_52_10 := arithEq_of_rows h (row := 52) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_10
  simp only [← c2433, ← c2434, ← c2435] at e_52_10
  have hr := e_52_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f292 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 52 47) = band (a (.wire 52 39)) (a (.wire 52 43)) := by
  have c2436 := (privateBatchWrapper2_copies76 a h).2.2.2.2.1
  have c2437 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.1
  have c2438 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.1
  have e_52_11 := arithEq_of_rows h (row := 52) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_11
  simp only [← c2436, ← c2437, ← c2438] at e_52_11
  have hr := e_52_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f293 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 51 43) = bselect (a (.wire 52 47)) (a (.wire 12 27)) (a (.virt 18979)) := by
  have c2439 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.1
  have c2440 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.1
  have c2441 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_51_10 := arithEq_of_rows h (row := 51) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_10
  simp only [← c2439, ← c2440, ← c2441, k1] at e_51_10
  have hr := e_51_10
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f294 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 31) = a (.wire 36 27) + a (.wire 51 43) := by
  have c2442 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2443 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2444 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_36_7 := arithEq_of_rows h (row := 36) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_7
  simp only [← c2442, ← c2443, k0, ← c2444] at e_36_7
  have hr := e_36_7
  linear_combination hr

theorem privateBatchWrapper2_f295 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 35)) (a (.wire 12 35)) (a (.virt 19187)) (a (.virt 19188)) := by
  have c2445 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2446 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2447 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2448 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2449 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2450 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2451 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2452 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2453 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2454 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2455 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2456 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2457 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2458 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2459 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2460 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2461 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_51_11 := arithEq_of_rows h (row := 51) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_11
  simp only [← c2445, k0, ← c2446, k0, ← c2447] at e_51_11
  have e_51_12 := arithEq_of_rows h (row := 51) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_12
  simp only [← c2448, ← c2449, k0, ← c2450] at e_51_12
  have e_51_13 := arithEq_of_rows h (row := 51) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_13
  simp only [← c2457, ← c2458, k0, ← c2459] at e_51_13
  have e_52_12 := arithEq_of_rows h (row := 52) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_12
  simp only [← c2451, ← c2452, ← c2453] at e_52_12
  have e_52_13 := arithEq_of_rows h (row := 52) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_13
  simp only [← c2454, ← c2455, ← c2456] at e_52_13
  refine ⟨?_, ?_⟩
  · have hc := e_52_12
    simp only [e_51_12] at hc
    linear_combination c2460.trans k1 - hc
  · have hc := e_51_13
    simp only [e_51_11, e_52_13, e_51_12] at hc
    linear_combination c2461.trans k1 - hc

theorem privateBatchWrapper2_f296 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 43)) (a (.wire 12 43)) (a (.virt 19189)) (a (.virt 19190)) := by
  have c2462 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2463 := (privateBatchWrapper2_copies76 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2464 := (privateBatchWrapper2_copies77 a h).1
  have c2465 := (privateBatchWrapper2_copies77 a h).2.1
  have c2466 := (privateBatchWrapper2_copies77 a h).2.2.1
  have c2467 := (privateBatchWrapper2_copies77 a h).2.2.2.1
  have c2468 := (privateBatchWrapper2_copies77 a h).2.2.2.2.1
  have c2469 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.1
  have c2470 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.1
  have c2471 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.1
  have c2472 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.1
  have c2473 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.1
  have c2474 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2475 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2476 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2477 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2478 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_51_14 := arithEq_of_rows h (row := 51) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_51_14
  simp only [← c2462, k0, ← c2463, k0, ← c2464] at e_51_14
  have e_52_14 := arithEq_of_rows h (row := 52) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_52_14
  simp only [← c2468, ← c2469, ← c2470] at e_52_14
  have e_53_0 := arithEq_of_rows h (row := 53) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_0
  simp only [← c2465, ← c2466, k0, ← c2467] at e_53_0
  have e_53_1 := arithEq_of_rows h (row := 53) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_1
  simp only [← c2474, ← c2475, k0, ← c2476] at e_53_1
  have e_54_0 := arithEq_of_rows h (row := 54) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_0
  simp only [← c2471, ← c2472, ← c2473] at e_54_0
  refine ⟨?_, ?_⟩
  · have hc := e_52_14
    simp only [e_53_0] at hc
    linear_combination c2477.trans k1 - hc
  · have hc := e_53_1
    simp only [e_51_14, e_54_0, e_53_0] at hc
    linear_combination c2478.trans k1 - hc

theorem privateBatchWrapper2_f297 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 51)) (a (.wire 12 51)) (a (.virt 19191)) (a (.virt 19192)) := by
  have c2479 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2480 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2481 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2482 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2483 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2484 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2485 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2486 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2487 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2488 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2489 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2490 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2491 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2492 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2493 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2494 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2495 := (privateBatchWrapper2_copies77 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_53_2 := arithEq_of_rows h (row := 53) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_2
  simp only [← c2479, k0, ← c2480, k0, ← c2481] at e_53_2
  have e_53_3 := arithEq_of_rows h (row := 53) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_3
  simp only [← c2482, ← c2483, k0, ← c2484] at e_53_3
  have e_53_4 := arithEq_of_rows h (row := 53) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_4
  simp only [← c2491, ← c2492, k0, ← c2493] at e_53_4
  have e_54_1 := arithEq_of_rows h (row := 54) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_1
  simp only [← c2485, ← c2486, ← c2487] at e_54_1
  have e_54_2 := arithEq_of_rows h (row := 54) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_2
  simp only [← c2488, ← c2489, ← c2490] at e_54_2
  refine ⟨?_, ?_⟩
  · have hc := e_54_1
    simp only [e_53_3] at hc
    linear_combination c2494.trans k1 - hc
  · have hc := e_53_4
    simp only [e_53_2, e_54_2, e_53_3] at hc
    linear_combination c2495.trans k1 - hc

theorem privateBatchWrapper2_f298 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.wire 12 59)) (a (.wire 12 59)) (a (.virt 19193)) (a (.virt 19194)) := by
  have c2496 := (privateBatchWrapper2_copies78 a h).1
  have c2497 := (privateBatchWrapper2_copies78 a h).2.1
  have c2498 := (privateBatchWrapper2_copies78 a h).2.2.1
  have c2499 := (privateBatchWrapper2_copies78 a h).2.2.2.1
  have c2500 := (privateBatchWrapper2_copies78 a h).2.2.2.2.1
  have c2501 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.1
  have c2502 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.1
  have c2503 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.1
  have c2504 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.1
  have c2505 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.1
  have c2506 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2507 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2508 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2509 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2510 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2511 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2512 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_53_5 := arithEq_of_rows h (row := 53) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_5
  simp only [← c2496, k0, ← c2497, k0, ← c2498] at e_53_5
  have e_53_6 := arithEq_of_rows h (row := 53) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_6
  simp only [← c2499, ← c2500, k0, ← c2501] at e_53_6
  have e_53_7 := arithEq_of_rows h (row := 53) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_7
  simp only [← c2508, ← c2509, k0, ← c2510] at e_53_7
  have e_54_3 := arithEq_of_rows h (row := 54) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_3
  simp only [← c2502, ← c2503, ← c2504] at e_54_3
  have e_54_4 := arithEq_of_rows h (row := 54) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_4
  simp only [← c2505, ← c2506, ← c2507] at e_54_4
  refine ⟨?_, ?_⟩
  · have hc := e_54_3
    simp only [e_53_6] at hc
    linear_combination c2511.trans k1 - hc
  · have hc := e_53_7
    simp only [e_53_5, e_54_4, e_53_6] at hc
    linear_combination c2512.trans k1 - hc

theorem privateBatchWrapper2_f299 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 54 23) = band (a (.virt 19187)) (a (.virt 19189)) := by
  have c2513 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2514 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2515 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_5 := arithEq_of_rows h (row := 54) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_5
  simp only [← c2513, ← c2514, ← c2515] at e_54_5
  have hr := e_54_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f300 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 54 27) = band (a (.virt 19191)) (a (.virt 19193)) := by
  have c2516 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2517 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2518 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_6 := arithEq_of_rows h (row := 54) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_6
  simp only [← c2516, ← c2517, ← c2518] at e_54_6
  have hr := e_54_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f301 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 54 31) = band (a (.wire 54 23)) (a (.wire 54 27)) := by
  have c2519 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2520 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2521 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_7 := arithEq_of_rows h (row := 54) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_7
  simp only [← c2519, ← c2520, ← c2521] at e_54_7
  have hr := e_54_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f302 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 53 35) = bselect (a (.wire 54 31)) (a (.wire 13 7)) (a (.virt 18979)) := by
  have c2522 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2523 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2524 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_53_8 := arithEq_of_rows h (row := 53) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_8
  simp only [← c2522, ← c2523, ← c2524, k1] at e_53_8
  have hr := e_53_8
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f303 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 36 35) = a (.wire 36 31) + a (.wire 53 35) := by
  have c2525 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2526 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2527 := (privateBatchWrapper2_copies78 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_36_8 := arithEq_of_rows h (row := 36) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_36_8
  simp only [← c2525, ← c2526, k0, ← c2527] at e_36_8
  have hr := e_36_8
  linear_combination hr

theorem privateBatchWrapper2_f304 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 53 43) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 36 35)) := by
  have c2528 := (privateBatchWrapper2_copies79 a h).1
  have c2529 := (privateBatchWrapper2_copies79 a h).2.1
  have c2530 := (privateBatchWrapper2_copies79 a h).2.2.1
  have c2531 := (privateBatchWrapper2_copies79 a h).2.2.2.1
  have c2532 := (privateBatchWrapper2_copies79 a h).2.2.2.2.1
  have c2533 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_53_9 := arithEq_of_rows h (row := 53) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_9
  simp only [← c2528, ← c2529, ← c2530] at e_53_9
  have e_53_10 := arithEq_of_rows h (row := 53) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_10
  simp only [← c2531, ← c2532, k1, ← c2533] at e_53_10
  have hr := e_53_10
  simp only [e_53_9] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f305 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 53 51) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 35)) := by
  have c2534 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.1
  have c2535 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.1
  have c2536 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.1
  have c2537 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.1
  have c2538 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2539 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_53_11 := arithEq_of_rows h (row := 53) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_11
  simp only [← c2534, ← c2535, ← c2536] at e_53_11
  have e_53_12 := arithEq_of_rows h (row := 53) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_12
  simp only [← c2537, ← c2538, k1, ← c2539] at e_53_12
  have hr := e_53_12
  simp only [e_53_11] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f306 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 53 59) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 43)) := by
  have c2540 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2541 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2542 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2543 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2544 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2545 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_53_13 := arithEq_of_rows h (row := 53) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_13
  simp only [← c2540, ← c2541, ← c2542] at e_53_13
  have e_53_14 := arithEq_of_rows h (row := 53) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_53_14
  simp only [← c2543, ← c2544, k1, ← c2545] at e_53_14
  have hr := e_53_14
  simp only [e_53_13] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f307 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 55 7) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 51)) := by
  have c2546 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2547 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2548 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2549 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2550 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2551 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_55_0 := arithEq_of_rows h (row := 55) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_0
  simp only [← c2546, ← c2547, ← c2548] at e_55_0
  have e_55_1 := arithEq_of_rows h (row := 55) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_1
  simp only [← c2549, ← c2550, k1, ← c2551] at e_55_1
  have hr := e_55_1
  simp only [e_55_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f308 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 55 15) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 59)) := by
  have c2552 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2553 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2554 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2555 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2556 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2557 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_55_2 := arithEq_of_rows h (row := 55) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_2
  simp only [← c2552, ← c2553, ← c2554] at e_55_2
  have e_55_3 := arithEq_of_rows h (row := 55) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_3
  simp only [← c2555, ← c2556, k1, ← c2557] at e_55_3
  have hr := e_55_3
  simp only [e_55_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f309 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 53 43)) 32 := by
  have c2558 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2559 := (privateBatchWrapper2_copies79 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2560 := (privateBatchWrapper2_copies80 a h).1
  have c2561 := (privateBatchWrapper2_copies80 a h).2.1
  have c2562 := (privateBatchWrapper2_copies80 a h).2.2.1
  have c2563 := (privateBatchWrapper2_copies80 a h).2.2.2.1
  have c2564 := (privateBatchWrapper2_copies80 a h).2.2.2.2.1
  have c2565 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.1
  have c2566 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.1
  have c2567 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.1
  have c2568 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.1
  have c2569 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.1
  have c2570 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2571 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2572 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2573 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2574 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2575 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2576 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2577 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2578 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2579 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2580 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2581 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2582 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2583 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2584 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2585 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 56) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c2558.trans k1
    · exact c2559.trans k1
    · exact c2560.trans k1
    · exact c2561.trans k1
    · exact c2562.trans k1
    · exact c2563.trans k1
    · exact c2564.trans k1
    · exact c2565.trans k1
    · exact c2566.trans k1
    · exact c2567.trans k1
    · exact c2568.trans k1
    · exact c2569.trans k1
    · exact c2570.trans k1
    · exact c2571.trans k1
    · exact c2572.trans k1
    · exact c2573.trans k1
    · exact c2574.trans k1
    · exact c2575.trans k1
    · exact c2576.trans k1
    · exact c2577.trans k1
    · exact c2578.trans k1
    · exact c2579.trans k1
    · exact c2580.trans k1
    · exact c2581.trans k1
    · exact c2582.trans k1
    · exact c2583.trans k1
    · exact c2584.trans k1
    )
  rwa [c2585] at hr

theorem privateBatchWrapper2_f310 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 7) = bnot (a (.wire 1 43)) := by
  have c130 := (privateBatchWrapper2_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper2_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper2_copies4 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c130, k0, ← c131, k0, ← c132] at e_3_1
  have hr := e_3_1
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f311 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 35) = bnot (a (.wire 2 27)) := by
  have c151 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c151, k0, ← c152, k0, ← c153] at e_3_8
  have hr := e_3_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f312 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 54 35) = band (a (.wire 3 7)) (a (.wire 3 35)) := by
  have c2586 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2587 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2588 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_54_8 := arithEq_of_rows h (row := 54) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_8
  simp only [← c2586, ← c2587, ← c2588] at e_54_8
  have hr := e_54_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f313 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9467)) (a (.virt 18952)) (a (.virt 19195)) (a (.virt 19196)) := by
  have c2589 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2590 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2591 := (privateBatchWrapper2_copies80 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2592 := (privateBatchWrapper2_copies81 a h).1
  have c2593 := (privateBatchWrapper2_copies81 a h).2.1
  have c2594 := (privateBatchWrapper2_copies81 a h).2.2.1
  have c2595 := (privateBatchWrapper2_copies81 a h).2.2.2.1
  have c2596 := (privateBatchWrapper2_copies81 a h).2.2.2.2.1
  have c2597 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.1
  have c2598 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.1
  have c2599 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.1
  have c2600 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.1
  have c2601 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.1
  have c2602 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2603 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2604 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2605 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_54_9 := arithEq_of_rows h (row := 54) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_9
  simp only [← c2595, ← c2596, ← c2597] at e_54_9
  have e_54_10 := arithEq_of_rows h (row := 54) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_10
  simp only [← c2598, ← c2599, ← c2600] at e_54_10
  have e_55_4 := arithEq_of_rows h (row := 55) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_4
  simp only [← c2589, k0, ← c2590, k0, ← c2591] at e_55_4
  have e_55_5 := arithEq_of_rows h (row := 55) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_5
  simp only [← c2592, ← c2593, k0, ← c2594] at e_55_5
  have e_55_6 := arithEq_of_rows h (row := 55) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_6
  simp only [← c2601, ← c2602, k0, ← c2603] at e_55_6
  refine ⟨?_, ?_⟩
  · have hc := e_54_9
    simp only [e_55_5] at hc
    linear_combination c2604.trans k1 - hc
  · have hc := e_55_6
    simp only [e_55_4, e_54_10, e_55_5] at hc
    linear_combination c2605.trans k1 - hc

theorem privateBatchWrapper2_f314 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9468)) (a (.virt 18953)) (a (.virt 19197)) (a (.virt 19198)) := by
  have c2606 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2607 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2608 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2609 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2610 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2611 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2612 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2613 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2614 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2615 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2616 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2617 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2618 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2619 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2620 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2621 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2622 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_54_11 := arithEq_of_rows h (row := 54) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_11
  simp only [← c2612, ← c2613, ← c2614] at e_54_11
  have e_54_12 := arithEq_of_rows h (row := 54) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_12
  simp only [← c2615, ← c2616, ← c2617] at e_54_12
  have e_55_7 := arithEq_of_rows h (row := 55) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_7
  simp only [← c2606, k0, ← c2607, k0, ← c2608] at e_55_7
  have e_55_8 := arithEq_of_rows h (row := 55) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_8
  simp only [← c2609, ← c2610, k0, ← c2611] at e_55_8
  have e_55_9 := arithEq_of_rows h (row := 55) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_9
  simp only [← c2618, ← c2619, k0, ← c2620] at e_55_9
  refine ⟨?_, ?_⟩
  · have hc := e_54_11
    simp only [e_55_8] at hc
    linear_combination c2621.trans k1 - hc
  · have hc := e_55_9
    simp only [e_55_7, e_54_12, e_55_8] at hc
    linear_combination c2622.trans k1 - hc

theorem privateBatchWrapper2_f315 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9469)) (a (.virt 18954)) (a (.virt 19199)) (a (.virt 19200)) := by
  have c2623 := (privateBatchWrapper2_copies81 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2624 := (privateBatchWrapper2_copies82 a h).1
  have c2625 := (privateBatchWrapper2_copies82 a h).2.1
  have c2626 := (privateBatchWrapper2_copies82 a h).2.2.1
  have c2627 := (privateBatchWrapper2_copies82 a h).2.2.2.1
  have c2628 := (privateBatchWrapper2_copies82 a h).2.2.2.2.1
  have c2629 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.1
  have c2630 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.1
  have c2631 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.1
  have c2632 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.1
  have c2633 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.1
  have c2634 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2635 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2636 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2637 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2638 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2639 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_54_13 := arithEq_of_rows h (row := 54) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_13
  simp only [← c2629, ← c2630, ← c2631] at e_54_13
  have e_54_14 := arithEq_of_rows h (row := 54) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_54_14
  simp only [← c2632, ← c2633, ← c2634] at e_54_14
  have e_55_10 := arithEq_of_rows h (row := 55) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_10
  simp only [← c2623, k0, ← c2624, k0, ← c2625] at e_55_10
  have e_55_11 := arithEq_of_rows h (row := 55) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_11
  simp only [← c2626, ← c2627, k0, ← c2628] at e_55_11
  have e_55_12 := arithEq_of_rows h (row := 55) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_12
  simp only [← c2635, ← c2636, k0, ← c2637] at e_55_12
  refine ⟨?_, ?_⟩
  · have hc := e_54_13
    simp only [e_55_11] at hc
    linear_combination c2638.trans k1 - hc
  · have hc := e_55_12
    simp only [e_55_10, e_54_14, e_55_11] at hc
    linear_combination c2639.trans k1 - hc

theorem privateBatchWrapper2_f316 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 9470)) (a (.virt 18955)) (a (.virt 19201)) (a (.virt 19202)) := by
  have c2640 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2641 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2642 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2643 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2644 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2645 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2646 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2647 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2648 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2649 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2650 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2651 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2652 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2653 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2654 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2655 := (privateBatchWrapper2_copies82 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2656 := (privateBatchWrapper2_copies83 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_55_13 := arithEq_of_rows h (row := 55) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_13
  simp only [← c2640, k0, ← c2641, k0, ← c2642] at e_55_13
  have e_55_14 := arithEq_of_rows h (row := 55) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_55_14
  simp only [← c2643, ← c2644, k0, ← c2645] at e_55_14
  have e_57_0 := arithEq_of_rows h (row := 57) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_0
  simp only [← c2646, ← c2647, ← c2648] at e_57_0
  have e_57_1 := arithEq_of_rows h (row := 57) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_1
  simp only [← c2649, ← c2650, ← c2651] at e_57_1
  have e_58_0 := arithEq_of_rows h (row := 58) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_0
  simp only [← c2652, ← c2653, k0, ← c2654] at e_58_0
  refine ⟨?_, ?_⟩
  · have hc := e_57_0
    simp only [e_55_14] at hc
    linear_combination c2655.trans k1 - hc
  · have hc := e_58_0
    simp only [e_55_13, e_57_1, e_55_14] at hc
    linear_combination c2656.trans k1 - hc

theorem privateBatchWrapper2_f317 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 57 11) = band (a (.virt 19195)) (a (.virt 19197)) := by
  have c2657 := (privateBatchWrapper2_copies83 a h).2.1
  have c2658 := (privateBatchWrapper2_copies83 a h).2.2.1
  have c2659 := (privateBatchWrapper2_copies83 a h).2.2.2.1
  have e_57_2 := arithEq_of_rows h (row := 57) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_2
  simp only [← c2657, ← c2658, ← c2659] at e_57_2
  have hr := e_57_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f318 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 57 15) = band (a (.virt 19199)) (a (.virt 19201)) := by
  have c2660 := (privateBatchWrapper2_copies83 a h).2.2.2.2.1
  have c2661 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.1
  have c2662 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.1
  have e_57_3 := arithEq_of_rows h (row := 57) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_3
  simp only [← c2660, ← c2661, ← c2662] at e_57_3
  have hr := e_57_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f319 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 57 19) = band (a (.wire 57 11)) (a (.wire 57 15)) := by
  have c2663 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.1
  have c2664 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.1
  have c2665 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.1
  have e_57_4 := arithEq_of_rows h (row := 57) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_4
  simp only [← c2663, ← c2664, ← c2665] at e_57_4
  have hr := e_57_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f320 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 57 23) = band (a (.wire 54 35)) (a (.wire 57 19)) := by
  have c2666 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2667 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2668 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_57_5 := arithEq_of_rows h (row := 57) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_57_5
  simp only [← c2666, ← c2667, ← c2668] at e_57_5
  have hr := e_57_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f321 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 57 23) = a (.virt 18979) := by
  have c2669 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c2669

theorem privateBatchWrapper2_f322 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 35) = bnot (a (.wire 2 27)) := by
  have c151 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c151, k0, ← c152, k0, ← c153] at e_3_8
  have hr := e_3_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f323 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 59 12) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 0 ∧ a (.wire 59 13) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 1 ∧ a (.wire 59 14) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 2 ∧ a (.wire 59 15) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 3) := by
  have c2670 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2671 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2672 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2673 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2674 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2675 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2676 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2677 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2678 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2679 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2680 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2681 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  exact poseidon2Row_hash4 perm hp (row := 59) rfl rfl
    c2670.symm c2671.symm c2672.symm c2673.symm (c2674.symm.trans k0) (c2675.symm.trans k1) (c2676.symm.trans k1) (c2677.symm.trans k1) (c2678.symm.trans k1) (c2679.symm.trans k1) (c2680.symm.trans k1) (c2681.symm.trans k1)

theorem privateBatchWrapper2_f324 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 60 12) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 0 ∧ a (.wire 60 13) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 1 ∧ a (.wire 60 14) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 2 ∧ a (.wire 60 15) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 3) := by
  have c2682 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2683 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2684 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2685 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2686 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2687 := (privateBatchWrapper2_copies83 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2688 := (privateBatchWrapper2_copies84 a h).1
  have c2689 := (privateBatchWrapper2_copies84 a h).2.1
  have c2690 := (privateBatchWrapper2_copies84 a h).2.2.1
  have c2691 := (privateBatchWrapper2_copies84 a h).2.2.2.1
  have c2692 := (privateBatchWrapper2_copies84 a h).2.2.2.2.1
  have c2693 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  exact poseidon2Row_hash4 perm hp (row := 60) rfl rfl
    c2682.symm c2683.symm c2684.symm c2685.symm (c2686.symm.trans k0) (c2687.symm.trans k1) (c2688.symm.trans k1) (c2689.symm.trans k1) (c2690.symm.trans k1) (c2691.symm.trans k1) (c2692.symm.trans k1) (c2693.symm.trans k1)

theorem privateBatchWrapper2_f325 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 11) = bselect (a (.wire 1 43)) (a (.wire 60 12)) (a (.virt 9467)) := by
  have c2694 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.1
  have c2695 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.1
  have c2696 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.1
  have c2697 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.1
  have c2698 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2699 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_1 := arithEq_of_rows h (row := 58) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_1
  simp only [← c2694, ← c2695, ← c2696] at e_58_1
  have e_58_2 := arithEq_of_rows h (row := 58) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_2
  simp only [← c2697, ← c2698, ← c2699] at e_58_2
  have hr := e_58_2
  simp only [e_58_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f326 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 19) = bselect (a (.wire 1 43)) (a (.wire 60 13)) (a (.virt 9468)) := by
  have c2700 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2701 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2702 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2703 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2704 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2705 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_3 := arithEq_of_rows h (row := 58) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_3
  simp only [← c2700, ← c2701, ← c2702] at e_58_3
  have e_58_4 := arithEq_of_rows h (row := 58) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_4
  simp only [← c2703, ← c2704, ← c2705] at e_58_4
  have hr := e_58_4
  simp only [e_58_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f327 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 27) = bselect (a (.wire 1 43)) (a (.wire 60 14)) (a (.virt 9469)) := by
  have c2706 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2707 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2708 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2709 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2710 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2711 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_5 := arithEq_of_rows h (row := 58) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_5
  simp only [← c2706, ← c2707, ← c2708] at e_58_5
  have e_58_6 := arithEq_of_rows h (row := 58) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_6
  simp only [← c2709, ← c2710, ← c2711] at e_58_6
  have hr := e_58_6
  simp only [e_58_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f328 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 35) = bselect (a (.wire 1 43)) (a (.wire 60 15)) (a (.virt 9470)) := by
  have c2712 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2713 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2714 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2715 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2716 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2717 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_7 := arithEq_of_rows h (row := 58) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_7
  simp only [← c2712, ← c2713, ← c2714] at e_58_7
  have e_58_8 := arithEq_of_rows h (row := 58) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_8
  simp only [← c2715, ← c2716, ← c2717] at e_58_8
  have hr := e_58_8
  simp only [e_58_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f329 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 61 12) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 0 ∧ a (.wire 61 13) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 1 ∧ a (.wire 61 14) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 2 ∧ a (.wire 61 15) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 3) := by
  have c2718 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2719 := (privateBatchWrapper2_copies84 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2720 := (privateBatchWrapper2_copies85 a h).1
  have c2721 := (privateBatchWrapper2_copies85 a h).2.1
  have c2722 := (privateBatchWrapper2_copies85 a h).2.2.1
  have c2723 := (privateBatchWrapper2_copies85 a h).2.2.2.1
  have c2724 := (privateBatchWrapper2_copies85 a h).2.2.2.2.1
  have c2725 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.1
  have c2726 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.1
  have c2727 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.1
  have c2728 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.1
  have c2729 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  exact poseidon2Row_hash4 perm hp (row := 61) rfl rfl
    c2718.symm c2719.symm c2720.symm c2721.symm (c2722.symm.trans k0) (c2723.symm.trans k1) (c2724.symm.trans k1) (c2725.symm.trans k1) (c2726.symm.trans k1) (c2727.symm.trans k1) (c2728.symm.trans k1) (c2729.symm.trans k1)

theorem privateBatchWrapper2_f330 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 62 12) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 0 ∧ a (.wire 62 13) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 1 ∧ a (.wire 62 14) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 2 ∧ a (.wire 62 15) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 3) := by
  have c2730 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2731 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2732 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2733 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2734 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2735 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2736 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2737 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2738 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2739 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2740 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2741 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  exact poseidon2Row_hash4 perm hp (row := 62) rfl rfl
    c2730.symm c2731.symm c2732.symm c2733.symm (c2734.symm.trans k0) (c2735.symm.trans k1) (c2736.symm.trans k1) (c2737.symm.trans k1) (c2738.symm.trans k1) (c2739.symm.trans k1) (c2740.symm.trans k1) (c2741.symm.trans k1)

theorem privateBatchWrapper2_f331 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 43) = bselect (a (.wire 2 27)) (a (.wire 62 12)) (a (.virt 18952)) := by
  have c2742 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2743 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2744 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2745 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2746 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2747 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_58_9 := arithEq_of_rows h (row := 58) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_9
  simp only [← c2742, ← c2743, ← c2744] at e_58_9
  have e_58_10 := arithEq_of_rows h (row := 58) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_10
  simp only [← c2745, ← c2746, ← c2747] at e_58_10
  have hr := e_58_10
  simp only [e_58_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f332 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 51) = bselect (a (.wire 2 27)) (a (.wire 62 13)) (a (.virt 18953)) := by
  have c2748 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2749 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2750 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2751 := (privateBatchWrapper2_copies85 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2752 := (privateBatchWrapper2_copies86 a h).1
  have c2753 := (privateBatchWrapper2_copies86 a h).2.1
  have e_58_11 := arithEq_of_rows h (row := 58) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_11
  simp only [← c2748, ← c2749, ← c2750] at e_58_11
  have e_58_12 := arithEq_of_rows h (row := 58) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_12
  simp only [← c2751, ← c2752, ← c2753] at e_58_12
  have hr := e_58_12
  simp only [e_58_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f333 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 58 59) = bselect (a (.wire 2 27)) (a (.wire 62 14)) (a (.virt 18954)) := by
  have c2754 := (privateBatchWrapper2_copies86 a h).2.2.1
  have c2755 := (privateBatchWrapper2_copies86 a h).2.2.2.1
  have c2756 := (privateBatchWrapper2_copies86 a h).2.2.2.2.1
  have c2757 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.1
  have c2758 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.1
  have c2759 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.1
  have e_58_13 := arithEq_of_rows h (row := 58) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_13
  simp only [← c2754, ← c2755, ← c2756] at e_58_13
  have e_58_14 := arithEq_of_rows h (row := 58) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_58_14
  simp only [← c2757, ← c2758, ← c2759] at e_58_14
  have hr := e_58_14
  simp only [e_58_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f334 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 7) = bselect (a (.wire 2 27)) (a (.wire 62 15)) (a (.virt 18955)) := by
  have c2760 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.1
  have c2761 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.1
  have c2762 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2763 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2764 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2765 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_0 := arithEq_of_rows h (row := 63) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_0
  simp only [← c2760, ← c2761, ← c2762] at e_63_0
  have e_63_1 := arithEq_of_rows h (row := 63) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_1
  simp only [← c2763, ← c2764, ← c2765] at e_63_1
  have hr := e_63_1
  simp only [e_63_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f335 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsBool (a (.virt 19203)) := by
  have c2766 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2767 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2768 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2769 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_63_2 := arithEq_of_rows h (row := 63) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_2
  simp only [← c2766, ← c2767, ← c2768] at e_63_2
  refine isBool_iff_assertBool.mpr ?_
  have hc := e_63_2
  linear_combination c2769.trans k1 - hc

theorem privateBatchWrapper2_f336 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 19) = bselect (a (.virt 19203)) (a (.wire 58 43)) (a (.wire 58 11)) := by
  have c2770 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2771 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2772 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2773 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2774 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2775 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_3 := arithEq_of_rows h (row := 63) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_3
  simp only [← c2770, ← c2771, ← c2772] at e_63_3
  have e_63_4 := arithEq_of_rows h (row := 63) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_4
  simp only [← c2773, ← c2774, ← c2775] at e_63_4
  have hr := e_63_4
  simp only [e_63_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f337 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 27) = bselect (a (.virt 19203)) (a (.wire 58 11)) (a (.wire 58 43)) := by
  have c2776 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2777 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2778 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2779 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2780 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2781 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_5 := arithEq_of_rows h (row := 63) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_5
  simp only [← c2776, ← c2777, ← c2778] at e_63_5
  have e_63_6 := arithEq_of_rows h (row := 63) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_6
  simp only [← c2779, ← c2780, ← c2781] at e_63_6
  have hr := e_63_6
  simp only [e_63_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f338 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 35) = bselect (a (.virt 19203)) (a (.wire 58 51)) (a (.wire 58 19)) := by
  have c2782 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2783 := (privateBatchWrapper2_copies86 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2784 := (privateBatchWrapper2_copies87 a h).1
  have c2785 := (privateBatchWrapper2_copies87 a h).2.1
  have c2786 := (privateBatchWrapper2_copies87 a h).2.2.1
  have c2787 := (privateBatchWrapper2_copies87 a h).2.2.2.1
  have e_63_7 := arithEq_of_rows h (row := 63) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_7
  simp only [← c2782, ← c2783, ← c2784] at e_63_7
  have e_63_8 := arithEq_of_rows h (row := 63) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_8
  simp only [← c2785, ← c2786, ← c2787] at e_63_8
  have hr := e_63_8
  simp only [e_63_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f339 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 43) = bselect (a (.virt 19203)) (a (.wire 58 19)) (a (.wire 58 51)) := by
  have c2788 := (privateBatchWrapper2_copies87 a h).2.2.2.2.1
  have c2789 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.1
  have c2790 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.1
  have c2791 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.1
  have c2792 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.1
  have c2793 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.1
  have e_63_9 := arithEq_of_rows h (row := 63) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_9
  simp only [← c2788, ← c2789, ← c2790] at e_63_9
  have e_63_10 := arithEq_of_rows h (row := 63) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_10
  simp only [← c2791, ← c2792, ← c2793] at e_63_10
  have hr := e_63_10
  simp only [e_63_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f340 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 51) = bselect (a (.virt 19203)) (a (.wire 58 59)) (a (.wire 58 27)) := by
  have c2794 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.1
  have c2795 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c2796 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2797 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2798 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2799 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_11 := arithEq_of_rows h (row := 63) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_11
  simp only [← c2794, ← c2795, ← c2796] at e_63_11
  have e_63_12 := arithEq_of_rows h (row := 63) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_12
  simp only [← c2797, ← c2798, ← c2799] at e_63_12
  have hr := e_63_12
  simp only [e_63_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f341 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 63 59) = bselect (a (.virt 19203)) (a (.wire 58 27)) (a (.wire 58 59)) := by
  have c2800 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2801 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2802 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2803 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2804 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2805 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_63_13 := arithEq_of_rows h (row := 63) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_13
  simp only [← c2800, ← c2801, ← c2802] at e_63_13
  have e_63_14 := arithEq_of_rows h (row := 63) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_63_14
  simp only [← c2803, ← c2804, ← c2805] at e_63_14
  have hr := e_63_14
  simp only [e_63_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f342 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 64 7) = bselect (a (.virt 19203)) (a (.wire 63 7)) (a (.wire 58 35)) := by
  have c2806 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2807 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2808 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2809 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2810 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2811 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_64_0 := arithEq_of_rows h (row := 64) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_0
  simp only [← c2806, ← c2807, ← c2808] at e_64_0
  have e_64_1 := arithEq_of_rows h (row := 64) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_1
  simp only [← c2809, ← c2810, ← c2811] at e_64_1
  have hr := e_64_1
  simp only [e_64_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f343 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 64 15) = bselect (a (.virt 19203)) (a (.wire 58 35)) (a (.wire 63 7)) := by
  have c2812 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2813 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2814 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c2815 := (privateBatchWrapper2_copies87 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c2816 := (privateBatchWrapper2_copies88 a h).1
  have c2817 := (privateBatchWrapper2_copies88 a h).2
  have e_64_2 := arithEq_of_rows h (row := 64) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_2
  simp only [← c2812, ← c2813, ← c2814] at e_64_2
  have e_64_3 := arithEq_of_rows h (row := 64) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_64_3
  simp only [← c2815, ← c2816, ← c2817] at e_64_3
  have hr := e_64_3
  simp only [e_64_2] at hr
  simp only [bselect]
  linear_combination hr

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `privateBatchWrapper2` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem privateBatchWrapper2_decode (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (IsEqual (a (.virt 9479)) (a (.virt 18979)) (a (.virt 18981)) (a (.virt 18982)) ∧
    IsEqual (a (.virt 9480)) (a (.virt 18979)) (a (.virt 18983)) (a (.virt 18984)) ∧
    IsEqual (a (.virt 9481)) (a (.virt 18979)) (a (.virt 18985)) (a (.virt 18986)) ∧
    IsEqual (a (.virt 9482)) (a (.virt 18979)) (a (.virt 18987)) (a (.virt 18988)) ∧
    a (.wire 1 35) = band (a (.virt 18981)) (a (.virt 18983)) ∧
    a (.wire 1 39) = band (a (.virt 18985)) (a (.virt 18987)) ∧
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) ∧
    IsEqual (a (.virt 18964)) (a (.virt 18979)) (a (.virt 18989)) (a (.virt 18990)) ∧
    IsEqual (a (.virt 18965)) (a (.virt 18979)) (a (.virt 18991)) (a (.virt 18992)) ∧
    IsEqual (a (.virt 18966)) (a (.virt 18979)) (a (.virt 18993)) (a (.virt 18994)) ∧
    IsEqual (a (.virt 18967)) (a (.virt 18979)) (a (.virt 18995)) (a (.virt 18996)) ∧
    a (.wire 2 19) = band (a (.virt 18989)) (a (.virt 18991)) ∧
    a (.wire 2 23) = band (a (.virt 18993)) (a (.virt 18995)) ∧
    a (.wire 2 27) = band (a (.wire 2 19)) (a (.wire 2 23)) ∧
    a (.wire 3 7) = bnot (a (.wire 1 43)) ∧
    a (.virt 18978) = bnot (a (.virt 18979)) ∧
    a (.wire 3 7) = band (a (.wire 3 7)) (a (.virt 18978)) ∧
    a (.wire 3 11) = bselect (a (.wire 3 7)) (a (.virt 9479)) (a (.virt 18979)) ∧
    a (.wire 3 15) = bselect (a (.wire 3 7)) (a (.virt 9480)) (a (.virt 18979)) ∧
    a (.wire 3 19) = bselect (a (.wire 3 7)) (a (.virt 9481)) (a (.virt 18979)) ∧
    a (.wire 3 23) = bselect (a (.wire 3 7)) (a (.virt 9482)) (a (.virt 18979)) ∧
    a (.wire 3 27) = bselect (a (.wire 3 7)) (a (.virt 9483)) (a (.virt 18979)) ∧
    a (.wire 3 31) = bselect (a (.wire 3 7)) (a (.virt 9466)) (a (.virt 18979)) ∧
    a (.wire 3 7) = bor (a (.virt 18979)) (a (.wire 3 7)) ∧
    a (.wire 3 35) = bnot (a (.wire 2 27)) ∧
    a (.wire 3 39) = bnot (a (.wire 3 7)) ∧
    a (.wire 2 31) = band (a (.wire 3 35)) (a (.wire 3 39)) ∧
    a (.wire 3 47) = bselect (a (.wire 2 31)) (a (.virt 18964)) (a (.wire 3 11)) ∧
    a (.wire 3 55) = bselect (a (.wire 2 31)) (a (.virt 18965)) (a (.wire 3 15)) ∧
    a (.wire 4 3) = bselect (a (.wire 2 31)) (a (.virt 18966)) (a (.wire 3 19)) ∧
    a (.wire 4 11) = bselect (a (.wire 2 31)) (a (.virt 18967)) (a (.wire 3 23)) ∧
    a (.wire 4 19) = bselect (a (.wire 2 31)) (a (.virt 18968)) (a (.wire 3 27))) ∧
    (a (.wire 4 27) = bselect (a (.wire 2 31)) (a (.virt 18951)) (a (.wire 3 31)) ∧
    a (.wire 6 3) = bor (a (.wire 3 7)) (a (.wire 3 35)) ∧
    IsEqual (a (.virt 9479)) (a (.wire 3 47)) (a (.virt 18997)) (a (.virt 18998)) ∧
    IsEqual (a (.virt 9480)) (a (.wire 3 55)) (a (.virt 18999)) (a (.virt 19000)) ∧
    IsEqual (a (.virt 9481)) (a (.wire 4 3)) (a (.virt 19001)) (a (.virt 19002)) ∧
    IsEqual (a (.virt 9482)) (a (.wire 4 11)) (a (.virt 19003)) (a (.virt 19004)) ∧
    a (.wire 8 7) = band (a (.virt 18997)) (a (.virt 18999)) ∧
    a (.wire 8 11) = band (a (.virt 19001)) (a (.virt 19003)) ∧
    a (.wire 8 15) = band (a (.wire 8 7)) (a (.wire 8 11)) ∧
    a (.wire 6 7) = bor (a (.wire 1 43)) (a (.wire 8 15)) ∧
    a (.wire 6 7) = a (.virt 18978) ∧
    a (.virt 9463) = a (.virt 9463) ∧
    IsEqual (a (.virt 9466)) (a (.wire 4 27)) (a (.virt 19005)) (a (.virt 19006)) ∧
    a (.wire 6 11) = bor (a (.wire 1 43)) (a (.virt 19005)) ∧
    a (.wire 6 11) = a (.virt 18978) ∧
    IsEqual (a (.virt 18964)) (a (.wire 3 47)) (a (.virt 19007)) (a (.virt 19008)) ∧
    IsEqual (a (.virt 18965)) (a (.wire 3 55)) (a (.virt 19009)) (a (.virt 19010)) ∧
    IsEqual (a (.virt 18966)) (a (.wire 4 3)) (a (.virt 19011)) (a (.virt 19012)) ∧
    IsEqual (a (.virt 18967)) (a (.wire 4 11)) (a (.virt 19013)) (a (.virt 19014)) ∧
    a (.wire 8 59) = band (a (.virt 19007)) (a (.virt 19009)) ∧
    a (.wire 10 3) = band (a (.virt 19011)) (a (.virt 19013)) ∧
    a (.wire 10 7) = band (a (.wire 8 59)) (a (.wire 10 3)) ∧
    a (.wire 6 15) = bor (a (.wire 2 27)) (a (.wire 10 7)) ∧
    a (.wire 6 15) = a (.virt 18978) ∧
    a (.virt 18948) = a (.virt 9463) ∧
    IsEqual (a (.virt 18951)) (a (.wire 4 27)) (a (.virt 19015)) (a (.virt 19016)) ∧
    a (.wire 6 19) = bor (a (.wire 2 27)) (a (.virt 19015)) ∧
    a (.wire 6 19) = a (.virt 18978) ∧
    a (.wire 9 35) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9471)) ∧
    a (.wire 9 43) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9472)) ∧
    a (.wire 9 51) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9473)) ∧
    a (.wire 9 59) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9474))) ∧
    (a (.wire 11 7) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9464)) ∧
    a (.wire 11 15) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9475)) ∧
    a (.wire 11 23) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9476)) ∧
    a (.wire 11 31) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9477)) ∧
    a (.wire 11 39) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9478)) ∧
    a (.wire 11 47) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9465)) ∧
    a (.wire 11 55) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18956)) ∧
    a (.wire 12 3) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18957)) ∧
    a (.wire 12 11) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18958)) ∧
    a (.wire 12 19) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18959)) ∧
    a (.wire 12 27) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18949)) ∧
    a (.wire 12 35) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18960)) ∧
    a (.wire 12 43) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18961)) ∧
    a (.wire 12 51) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18962)) ∧
    a (.wire 12 59) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18963)) ∧
    a (.wire 13 7) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18950)) ∧
    a (.wire 13 15) = bselect (a (.wire 1 43)) (a (.virt 18979)) (a (.virt 9484)) ∧
    a (.wire 13 15) = a (.virt 18979) + a (.wire 13 15) ∧
    a (.wire 13 23) = bselect (a (.wire 2 27)) (a (.virt 18979)) (a (.virt 18969)) ∧
    a (.wire 6 23) = a (.wire 13 15) + a (.wire 13 23) ∧
    a (.wire 11 7) = a (.virt 18979) + a (.wire 11 7) ∧
    a (.wire 6 27) = a (.wire 11 7) + a (.wire 11 47) ∧
    a (.wire 6 31) = a (.wire 6 27) + a (.wire 12 27) ∧
    a (.wire 6 35) = a (.wire 6 31) + a (.wire 13 7) ∧
    a (.wire 13 27) = a (.virt 19017) - a (.wire 4 27) ∧
    rangeCheck (a (.wire 13 27)) 14 ∧
    a (.wire 10 19) = a (.wire 6 35) * a (.virt 19017) ∧
    a (.wire 10 23) = a (.wire 6 23) * a (.wire 13 27) ∧
    a (.wire 13 31) = a (.wire 10 23) - a (.wire 10 19) ∧
    rangeCheck (a (.wire 13 31)) 52 ∧
    IsEqual (a (.wire 9 35)) (a (.wire 9 35)) (a (.virt 19019)) (a (.virt 19020)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 9 43)) (a (.virt 19021)) (a (.virt 19022))) ∧
    (IsEqual (a (.wire 9 51)) (a (.wire 9 51)) (a (.virt 19023)) (a (.virt 19024)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 9 59)) (a (.virt 19025)) (a (.virt 19026)) ∧
    a (.wire 10 59) = band (a (.virt 19019)) (a (.virt 19021)) ∧
    a (.wire 17 3) = band (a (.virt 19023)) (a (.virt 19025)) ∧
    a (.wire 17 7) = band (a (.wire 10 59)) (a (.wire 17 3)) ∧
    a (.wire 16 23) = bselect (a (.wire 17 7)) (a (.wire 11 7)) (a (.virt 18979)) ∧
    a (.wire 16 23) = a (.virt 18979) + a (.wire 16 23) ∧
    IsEqual (a (.wire 11 15)) (a (.wire 9 35)) (a (.virt 19027)) (a (.virt 19028)) ∧
    IsEqual (a (.wire 11 23)) (a (.wire 9 43)) (a (.virt 19029)) (a (.virt 19030)) ∧
    IsEqual (a (.wire 11 31)) (a (.wire 9 51)) (a (.virt 19031)) (a (.virt 19032)) ∧
    IsEqual (a (.wire 11 39)) (a (.wire 9 59)) (a (.virt 19033)) (a (.virt 19034)) ∧
    a (.wire 17 43) = band (a (.virt 19027)) (a (.virt 19029)) ∧
    a (.wire 17 47) = band (a (.virt 19031)) (a (.virt 19033)) ∧
    a (.wire 17 51) = band (a (.wire 17 43)) (a (.wire 17 47)) ∧
    a (.wire 18 15) = bselect (a (.wire 17 51)) (a (.wire 11 47)) (a (.virt 18979)) ∧
    a (.wire 6 39) = a (.wire 16 23) + a (.wire 18 15) ∧
    IsEqual (a (.wire 11 55)) (a (.wire 9 35)) (a (.virt 19035)) (a (.virt 19036)) ∧
    IsEqual (a (.wire 12 3)) (a (.wire 9 43)) (a (.virt 19037)) (a (.virt 19038)) ∧
    IsEqual (a (.wire 12 11)) (a (.wire 9 51)) (a (.virt 19039)) (a (.virt 19040)) ∧
    IsEqual (a (.wire 12 19)) (a (.wire 9 59)) (a (.virt 19041)) (a (.virt 19042)) ∧
    a (.wire 19 27) = band (a (.virt 19035)) (a (.virt 19037)) ∧
    a (.wire 19 31) = band (a (.virt 19039)) (a (.virt 19041)) ∧
    a (.wire 19 35) = band (a (.wire 19 27)) (a (.wire 19 31)) ∧
    a (.wire 20 7) = bselect (a (.wire 19 35)) (a (.wire 12 27)) (a (.virt 18979)) ∧
    a (.wire 6 43) = a (.wire 6 39) + a (.wire 20 7) ∧
    IsEqual (a (.wire 12 35)) (a (.wire 9 35)) (a (.virt 19043)) (a (.virt 19044)) ∧
    IsEqual (a (.wire 12 43)) (a (.wire 9 43)) (a (.virt 19045)) (a (.virt 19046)) ∧
    IsEqual (a (.wire 12 51)) (a (.wire 9 51)) (a (.virt 19047)) (a (.virt 19048)) ∧
    IsEqual (a (.wire 12 59)) (a (.wire 9 59)) (a (.virt 19049)) (a (.virt 19050)) ∧
    a (.wire 21 11) = band (a (.virt 19043)) (a (.virt 19045)) ∧
    a (.wire 21 15) = band (a (.virt 19047)) (a (.virt 19049)) ∧
    a (.wire 21 19) = band (a (.wire 21 11)) (a (.wire 21 15))) ∧
    (a (.wire 20 59) = bselect (a (.wire 21 19)) (a (.wire 13 7)) (a (.virt 18979)) ∧
    a (.wire 6 47) = a (.wire 6 43) + a (.wire 20 59) ∧
    a (.wire 22 7) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 6 47)) ∧
    a (.wire 22 15) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 35)) ∧
    a (.wire 22 23) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 43)) ∧
    a (.wire 22 31) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 51)) ∧
    a (.wire 22 39) = bselect (a (.virt 18979)) (a (.virt 18979)) (a (.wire 9 59)) ∧
    rangeCheck (a (.wire 22 7)) 32 ∧
    IsEqual (a (.wire 9 35)) (a (.wire 11 15)) (a (.virt 19051)) (a (.virt 19052)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 11 23)) (a (.virt 19053)) (a (.virt 19054)) ∧
    IsEqual (a (.wire 9 51)) (a (.wire 11 31)) (a (.virt 19055)) (a (.virt 19056)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 11 39)) (a (.virt 19057)) (a (.virt 19058)) ∧
    a (.wire 21 55) = band (a (.virt 19051)) (a (.virt 19053)) ∧
    a (.wire 21 59) = band (a (.virt 19055)) (a (.virt 19057)) ∧
    a (.wire 25 3) = band (a (.wire 21 55)) (a (.wire 21 59)) ∧
    a (.wire 25 3) = bor (a (.virt 18979)) (a (.wire 25 3)) ∧
    IsEqual (a (.wire 9 35)) (a (.wire 11 15)) (a (.virt 19059)) (a (.virt 19060)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 11 23)) (a (.virt 19061)) (a (.virt 19062)) ∧
    IsEqual (a (.wire 9 51)) (a (.wire 11 31)) (a (.virt 19063)) (a (.virt 19064)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 11 39)) (a (.virt 19065)) (a (.virt 19066)) ∧
    a (.wire 25 39) = band (a (.virt 19059)) (a (.virt 19061)) ∧
    a (.wire 25 43) = band (a (.virt 19063)) (a (.virt 19065)) ∧
    a (.wire 25 47) = band (a (.wire 25 39)) (a (.wire 25 43)) ∧
    a (.wire 26 3) = bselect (a (.wire 25 47)) (a (.wire 11 7)) (a (.virt 18979)) ∧
    a (.wire 26 3) = a (.virt 18979) + a (.wire 26 3) ∧
    IsEqual (a (.wire 11 15)) (a (.wire 11 15)) (a (.virt 19067)) (a (.virt 19068)) ∧
    IsEqual (a (.wire 11 23)) (a (.wire 11 23)) (a (.virt 19069)) (a (.virt 19070)) ∧
    IsEqual (a (.wire 11 31)) (a (.wire 11 31)) (a (.virt 19071)) (a (.virt 19072)) ∧
    IsEqual (a (.wire 11 39)) (a (.wire 11 39)) (a (.virt 19073)) (a (.virt 19074)) ∧
    a (.wire 27 23) = band (a (.virt 19067)) (a (.virt 19069)) ∧
    a (.wire 27 27) = band (a (.virt 19071)) (a (.virt 19073)) ∧
    a (.wire 27 31) = band (a (.wire 27 23)) (a (.wire 27 27))) ∧
    (a (.wire 26 55) = bselect (a (.wire 27 31)) (a (.wire 11 47)) (a (.virt 18979)) ∧
    a (.wire 6 51) = a (.wire 26 3) + a (.wire 26 55) ∧
    IsEqual (a (.wire 11 55)) (a (.wire 11 15)) (a (.virt 19075)) (a (.virt 19076)) ∧
    IsEqual (a (.wire 12 3)) (a (.wire 11 23)) (a (.virt 19077)) (a (.virt 19078)) ∧
    IsEqual (a (.wire 12 11)) (a (.wire 11 31)) (a (.virt 19079)) (a (.virt 19080)) ∧
    IsEqual (a (.wire 12 19)) (a (.wire 11 39)) (a (.virt 19081)) (a (.virt 19082)) ∧
    a (.wire 29 7) = band (a (.virt 19075)) (a (.virt 19077)) ∧
    a (.wire 29 11) = band (a (.virt 19079)) (a (.virt 19081)) ∧
    a (.wire 29 15) = band (a (.wire 29 7)) (a (.wire 29 11)) ∧
    a (.wire 28 47) = bselect (a (.wire 29 15)) (a (.wire 12 27)) (a (.virt 18979)) ∧
    a (.wire 6 55) = a (.wire 6 51) + a (.wire 28 47) ∧
    IsEqual (a (.wire 12 35)) (a (.wire 11 15)) (a (.virt 19083)) (a (.virt 19084)) ∧
    IsEqual (a (.wire 12 43)) (a (.wire 11 23)) (a (.virt 19085)) (a (.virt 19086)) ∧
    IsEqual (a (.wire 12 51)) (a (.wire 11 31)) (a (.virt 19087)) (a (.virt 19088)) ∧
    IsEqual (a (.wire 12 59)) (a (.wire 11 39)) (a (.virt 19089)) (a (.virt 19090)) ∧
    a (.wire 29 51) = band (a (.virt 19083)) (a (.virt 19085)) ∧
    a (.wire 29 55) = band (a (.virt 19087)) (a (.virt 19089)) ∧
    a (.wire 29 59) = band (a (.wire 29 51)) (a (.wire 29 55)) ∧
    a (.wire 30 39) = bselect (a (.wire 29 59)) (a (.wire 13 7)) (a (.virt 18979)) ∧
    a (.wire 6 59) = a (.wire 6 55) + a (.wire 30 39) ∧
    a (.wire 30 47) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 6 59)) ∧
    a (.wire 30 55) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 15)) ∧
    a (.wire 31 3) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 23)) ∧
    a (.wire 31 11) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 31)) ∧
    a (.wire 31 19) = bselect (a (.wire 25 3)) (a (.virt 18979)) (a (.wire 11 39)) ∧
    rangeCheck (a (.wire 30 47)) 32 ∧
    IsEqual (a (.wire 9 35)) (a (.wire 11 55)) (a (.virt 19091)) (a (.virt 19092)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 12 3)) (a (.virt 19093)) (a (.virt 19094)) ∧
    IsEqual (a (.wire 9 51)) (a (.wire 12 11)) (a (.virt 19095)) (a (.virt 19096)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 12 19)) (a (.virt 19097)) (a (.virt 19098)) ∧
    a (.wire 33 35) = band (a (.virt 19091)) (a (.virt 19093)) ∧
    a (.wire 33 39) = band (a (.virt 19095)) (a (.virt 19097))) ∧
    (a (.wire 33 43) = band (a (.wire 33 35)) (a (.wire 33 39)) ∧
    a (.wire 33 43) = bor (a (.virt 18979)) (a (.wire 33 43)) ∧
    IsEqual (a (.wire 11 15)) (a (.wire 11 55)) (a (.virt 19099)) (a (.virt 19100)) ∧
    IsEqual (a (.wire 11 23)) (a (.wire 12 3)) (a (.virt 19101)) (a (.virt 19102)) ∧
    IsEqual (a (.wire 11 31)) (a (.wire 12 11)) (a (.virt 19103)) (a (.virt 19104)) ∧
    IsEqual (a (.wire 11 39)) (a (.wire 12 19)) (a (.virt 19105)) (a (.virt 19106)) ∧
    a (.wire 35 19) = band (a (.virt 19099)) (a (.virt 19101)) ∧
    a (.wire 35 23) = band (a (.virt 19103)) (a (.virt 19105)) ∧
    a (.wire 35 27) = band (a (.wire 35 19)) (a (.wire 35 23)) ∧
    a (.wire 36 3) = bor (a (.wire 33 43)) (a (.wire 35 27)) ∧
    IsEqual (a (.wire 9 35)) (a (.wire 11 55)) (a (.virt 19107)) (a (.virt 19108)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 12 3)) (a (.virt 19109)) (a (.virt 19110)) ∧
    IsEqual (a (.wire 9 51)) (a (.wire 12 11)) (a (.virt 19111)) (a (.virt 19112)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 12 19)) (a (.virt 19113)) (a (.virt 19114)) ∧
    a (.wire 38 3) = band (a (.virt 19107)) (a (.virt 19109)) ∧
    a (.wire 38 7) = band (a (.virt 19111)) (a (.virt 19113)) ∧
    a (.wire 38 11) = band (a (.wire 38 3)) (a (.wire 38 7)) ∧
    a (.wire 37 31) = bselect (a (.wire 38 11)) (a (.wire 11 7)) (a (.virt 18979)) ∧
    a (.wire 37 31) = a (.virt 18979) + a (.wire 37 31) ∧
    IsEqual (a (.wire 11 15)) (a (.wire 11 55)) (a (.virt 19115)) (a (.virt 19116)) ∧
    IsEqual (a (.wire 11 23)) (a (.wire 12 3)) (a (.virt 19117)) (a (.virt 19118)) ∧
    IsEqual (a (.wire 11 31)) (a (.wire 12 11)) (a (.virt 19119)) (a (.virt 19120)) ∧
    IsEqual (a (.wire 11 39)) (a (.wire 12 19)) (a (.virt 19121)) (a (.virt 19122)) ∧
    a (.wire 38 47) = band (a (.virt 19115)) (a (.virt 19117)) ∧
    a (.wire 38 51) = band (a (.virt 19119)) (a (.virt 19121)) ∧
    a (.wire 38 55) = band (a (.wire 38 47)) (a (.wire 38 51)) ∧
    a (.wire 39 7) = bselect (a (.wire 38 55)) (a (.wire 11 47)) (a (.virt 18979)) ∧
    a (.wire 36 7) = a (.wire 37 31) + a (.wire 39 7) ∧
    IsEqual (a (.wire 11 55)) (a (.wire 11 55)) (a (.virt 19123)) (a (.virt 19124)) ∧
    IsEqual (a (.wire 12 3)) (a (.wire 12 3)) (a (.virt 19125)) (a (.virt 19126)) ∧
    IsEqual (a (.wire 12 11)) (a (.wire 12 11)) (a (.virt 19127)) (a (.virt 19128)) ∧
    IsEqual (a (.wire 12 19)) (a (.wire 12 19)) (a (.virt 19129)) (a (.virt 19130))) ∧
    (a (.wire 40 31) = band (a (.virt 19123)) (a (.virt 19125)) ∧
    a (.wire 40 35) = band (a (.virt 19127)) (a (.virt 19129)) ∧
    a (.wire 40 39) = band (a (.wire 40 31)) (a (.wire 40 35)) ∧
    a (.wire 39 59) = bselect (a (.wire 40 39)) (a (.wire 12 27)) (a (.virt 18979)) ∧
    a (.wire 36 11) = a (.wire 36 7) + a (.wire 39 59) ∧
    IsEqual (a (.wire 12 35)) (a (.wire 11 55)) (a (.virt 19131)) (a (.virt 19132)) ∧
    IsEqual (a (.wire 12 43)) (a (.wire 12 3)) (a (.virt 19133)) (a (.virt 19134)) ∧
    IsEqual (a (.wire 12 51)) (a (.wire 12 11)) (a (.virt 19135)) (a (.virt 19136)) ∧
    IsEqual (a (.wire 12 59)) (a (.wire 12 19)) (a (.virt 19137)) (a (.virt 19138)) ∧
    a (.wire 42 15) = band (a (.virt 19131)) (a (.virt 19133)) ∧
    a (.wire 42 19) = band (a (.virt 19135)) (a (.virt 19137)) ∧
    a (.wire 42 23) = band (a (.wire 42 15)) (a (.wire 42 19)) ∧
    a (.wire 41 51) = bselect (a (.wire 42 23)) (a (.wire 13 7)) (a (.virt 18979)) ∧
    a (.wire 36 15) = a (.wire 36 11) + a (.wire 41 51) ∧
    a (.wire 41 59) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 36 15)) ∧
    a (.wire 43 7) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 11 55)) ∧
    a (.wire 43 15) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 12 3)) ∧
    a (.wire 43 23) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 12 11)) ∧
    a (.wire 43 31) = bselect (a (.wire 36 3)) (a (.virt 18979)) (a (.wire 12 19)) ∧
    rangeCheck (a (.wire 41 59)) 32 ∧
    IsEqual (a (.wire 9 35)) (a (.wire 12 35)) (a (.virt 19139)) (a (.virt 19140)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 12 43)) (a (.virt 19141)) (a (.virt 19142)) ∧
    IsEqual (a (.wire 9 51)) (a (.wire 12 51)) (a (.virt 19143)) (a (.virt 19144)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 12 59)) (a (.virt 19145)) (a (.virt 19146)) ∧
    a (.wire 42 59) = band (a (.virt 19139)) (a (.virt 19141)) ∧
    a (.wire 46 3) = band (a (.virt 19143)) (a (.virt 19145)) ∧
    a (.wire 46 7) = band (a (.wire 42 59)) (a (.wire 46 3)) ∧
    a (.wire 46 7) = bor (a (.virt 18979)) (a (.wire 46 7)) ∧
    IsEqual (a (.wire 11 15)) (a (.wire 12 35)) (a (.virt 19147)) (a (.virt 19148)) ∧
    IsEqual (a (.wire 11 23)) (a (.wire 12 43)) (a (.virt 19149)) (a (.virt 19150)) ∧
    IsEqual (a (.wire 11 31)) (a (.wire 12 51)) (a (.virt 19151)) (a (.virt 19152)) ∧
    IsEqual (a (.wire 11 39)) (a (.wire 12 59)) (a (.virt 19153)) (a (.virt 19154))) ∧
    (a (.wire 46 43) = band (a (.virt 19147)) (a (.virt 19149)) ∧
    a (.wire 46 47) = band (a (.virt 19151)) (a (.virt 19153)) ∧
    a (.wire 46 51) = band (a (.wire 46 43)) (a (.wire 46 47)) ∧
    a (.wire 36 19) = bor (a (.wire 46 7)) (a (.wire 46 51)) ∧
    IsEqual (a (.wire 11 55)) (a (.wire 12 35)) (a (.virt 19155)) (a (.virt 19156)) ∧
    IsEqual (a (.wire 12 3)) (a (.wire 12 43)) (a (.virt 19157)) (a (.virt 19158)) ∧
    IsEqual (a (.wire 12 11)) (a (.wire 12 51)) (a (.virt 19159)) (a (.virt 19160)) ∧
    IsEqual (a (.wire 12 19)) (a (.wire 12 59)) (a (.virt 19161)) (a (.virt 19162)) ∧
    a (.wire 48 27) = band (a (.virt 19155)) (a (.virt 19157)) ∧
    a (.wire 48 31) = band (a (.virt 19159)) (a (.virt 19161)) ∧
    a (.wire 48 35) = band (a (.wire 48 27)) (a (.wire 48 31)) ∧
    a (.wire 36 23) = bor (a (.wire 36 19)) (a (.wire 48 35)) ∧
    IsEqual (a (.wire 9 35)) (a (.wire 12 35)) (a (.virt 19163)) (a (.virt 19164)) ∧
    IsEqual (a (.wire 9 43)) (a (.wire 12 43)) (a (.virt 19165)) (a (.virt 19166)) ∧
    IsEqual (a (.wire 9 51)) (a (.wire 12 51)) (a (.virt 19167)) (a (.virt 19168)) ∧
    IsEqual (a (.wire 9 59)) (a (.wire 12 59)) (a (.virt 19169)) (a (.virt 19170)) ∧
    a (.wire 50 11) = band (a (.virt 19163)) (a (.virt 19165)) ∧
    a (.wire 50 15) = band (a (.virt 19167)) (a (.virt 19169)) ∧
    a (.wire 50 19) = band (a (.wire 50 11)) (a (.wire 50 15)) ∧
    a (.wire 49 31) = bselect (a (.wire 50 19)) (a (.wire 11 7)) (a (.virt 18979)) ∧
    a (.wire 49 31) = a (.virt 18979) + a (.wire 49 31) ∧
    IsEqual (a (.wire 11 15)) (a (.wire 12 35)) (a (.virt 19171)) (a (.virt 19172)) ∧
    IsEqual (a (.wire 11 23)) (a (.wire 12 43)) (a (.virt 19173)) (a (.virt 19174)) ∧
    IsEqual (a (.wire 11 31)) (a (.wire 12 51)) (a (.virt 19175)) (a (.virt 19176)) ∧
    IsEqual (a (.wire 11 39)) (a (.wire 12 59)) (a (.virt 19177)) (a (.virt 19178)) ∧
    a (.wire 50 55) = band (a (.virt 19171)) (a (.virt 19173)) ∧
    a (.wire 50 59) = band (a (.virt 19175)) (a (.virt 19177)) ∧
    a (.wire 52 3) = band (a (.wire 50 55)) (a (.wire 50 59)) ∧
    a (.wire 51 7) = bselect (a (.wire 52 3)) (a (.wire 11 47)) (a (.virt 18979)) ∧
    a (.wire 36 27) = a (.wire 49 31) + a (.wire 51 7) ∧
    IsEqual (a (.wire 11 55)) (a (.wire 12 35)) (a (.virt 19179)) (a (.virt 19180)) ∧
    IsEqual (a (.wire 12 3)) (a (.wire 12 43)) (a (.virt 19181)) (a (.virt 19182))) ∧
    (IsEqual (a (.wire 12 11)) (a (.wire 12 51)) (a (.virt 19183)) (a (.virt 19184)) ∧
    IsEqual (a (.wire 12 19)) (a (.wire 12 59)) (a (.virt 19185)) (a (.virt 19186)) ∧
    a (.wire 52 39) = band (a (.virt 19179)) (a (.virt 19181)) ∧
    a (.wire 52 43) = band (a (.virt 19183)) (a (.virt 19185)) ∧
    a (.wire 52 47) = band (a (.wire 52 39)) (a (.wire 52 43)) ∧
    a (.wire 51 43) = bselect (a (.wire 52 47)) (a (.wire 12 27)) (a (.virt 18979)) ∧
    a (.wire 36 31) = a (.wire 36 27) + a (.wire 51 43) ∧
    IsEqual (a (.wire 12 35)) (a (.wire 12 35)) (a (.virt 19187)) (a (.virt 19188)) ∧
    IsEqual (a (.wire 12 43)) (a (.wire 12 43)) (a (.virt 19189)) (a (.virt 19190)) ∧
    IsEqual (a (.wire 12 51)) (a (.wire 12 51)) (a (.virt 19191)) (a (.virt 19192)) ∧
    IsEqual (a (.wire 12 59)) (a (.wire 12 59)) (a (.virt 19193)) (a (.virt 19194)) ∧
    a (.wire 54 23) = band (a (.virt 19187)) (a (.virt 19189)) ∧
    a (.wire 54 27) = band (a (.virt 19191)) (a (.virt 19193)) ∧
    a (.wire 54 31) = band (a (.wire 54 23)) (a (.wire 54 27)) ∧
    a (.wire 53 35) = bselect (a (.wire 54 31)) (a (.wire 13 7)) (a (.virt 18979)) ∧
    a (.wire 36 35) = a (.wire 36 31) + a (.wire 53 35) ∧
    a (.wire 53 43) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 36 35)) ∧
    a (.wire 53 51) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 35)) ∧
    a (.wire 53 59) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 43)) ∧
    a (.wire 55 7) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 51)) ∧
    a (.wire 55 15) = bselect (a (.wire 36 23)) (a (.virt 18979)) (a (.wire 12 59)) ∧
    rangeCheck (a (.wire 53 43)) 32 ∧
    a (.wire 3 7) = bnot (a (.wire 1 43)) ∧
    a (.wire 3 35) = bnot (a (.wire 2 27)) ∧
    a (.wire 54 35) = band (a (.wire 3 7)) (a (.wire 3 35)) ∧
    IsEqual (a (.virt 9467)) (a (.virt 18952)) (a (.virt 19195)) (a (.virt 19196)) ∧
    IsEqual (a (.virt 9468)) (a (.virt 18953)) (a (.virt 19197)) (a (.virt 19198)) ∧
    IsEqual (a (.virt 9469)) (a (.virt 18954)) (a (.virt 19199)) (a (.virt 19200)) ∧
    IsEqual (a (.virt 9470)) (a (.virt 18955)) (a (.virt 19201)) (a (.virt 19202)) ∧
    a (.wire 57 11) = band (a (.virt 19195)) (a (.virt 19197)) ∧
    a (.wire 57 15) = band (a (.virt 19199)) (a (.virt 19201)) ∧
    a (.wire 57 19) = band (a (.wire 57 11)) (a (.wire 57 15))) ∧
    (a (.wire 57 23) = band (a (.wire 54 35)) (a (.wire 57 19)) ∧
    a (.wire 57 23) = a (.virt 18979) ∧
    a (.wire 3 35) = bnot (a (.wire 2 27)) ∧
    (a (.wire 59 12) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 0 ∧ a (.wire 59 13) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 1 ∧ a (.wire 59 14) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 2 ∧ a (.wire 59 15) = spongeHash perm [a (.virt 18970), a (.virt 18971), a (.virt 18972), a (.virt 18973)] 3) ∧
    (a (.wire 60 12) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 0 ∧ a (.wire 60 13) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 1 ∧ a (.wire 60 14) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 2 ∧ a (.wire 60 15) = spongeHash perm [a (.wire 59 12), a (.wire 59 13), a (.wire 59 14), a (.wire 59 15)] 3) ∧
    a (.wire 58 11) = bselect (a (.wire 1 43)) (a (.wire 60 12)) (a (.virt 9467)) ∧
    a (.wire 58 19) = bselect (a (.wire 1 43)) (a (.wire 60 13)) (a (.virt 9468)) ∧
    a (.wire 58 27) = bselect (a (.wire 1 43)) (a (.wire 60 14)) (a (.virt 9469)) ∧
    a (.wire 58 35) = bselect (a (.wire 1 43)) (a (.wire 60 15)) (a (.virt 9470)) ∧
    (a (.wire 61 12) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 0 ∧ a (.wire 61 13) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 1 ∧ a (.wire 61 14) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 2 ∧ a (.wire 61 15) = spongeHash perm [a (.virt 18974), a (.virt 18975), a (.virt 18976), a (.virt 18977)] 3) ∧
    (a (.wire 62 12) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 0 ∧ a (.wire 62 13) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 1 ∧ a (.wire 62 14) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 2 ∧ a (.wire 62 15) = spongeHash perm [a (.wire 61 12), a (.wire 61 13), a (.wire 61 14), a (.wire 61 15)] 3) ∧
    a (.wire 58 43) = bselect (a (.wire 2 27)) (a (.wire 62 12)) (a (.virt 18952)) ∧
    a (.wire 58 51) = bselect (a (.wire 2 27)) (a (.wire 62 13)) (a (.virt 18953)) ∧
    a (.wire 58 59) = bselect (a (.wire 2 27)) (a (.wire 62 14)) (a (.virt 18954)) ∧
    a (.wire 63 7) = bselect (a (.wire 2 27)) (a (.wire 62 15)) (a (.virt 18955)) ∧
    IsBool (a (.virt 19203)) ∧
    a (.wire 63 19) = bselect (a (.virt 19203)) (a (.wire 58 43)) (a (.wire 58 11)) ∧
    a (.wire 63 27) = bselect (a (.virt 19203)) (a (.wire 58 11)) (a (.wire 58 43)) ∧
    a (.wire 63 35) = bselect (a (.virt 19203)) (a (.wire 58 51)) (a (.wire 58 19)) ∧
    a (.wire 63 43) = bselect (a (.virt 19203)) (a (.wire 58 19)) (a (.wire 58 51)) ∧
    a (.wire 63 51) = bselect (a (.virt 19203)) (a (.wire 58 59)) (a (.wire 58 27)) ∧
    a (.wire 63 59) = bselect (a (.virt 19203)) (a (.wire 58 27)) (a (.wire 58 59)) ∧
    a (.wire 64 7) = bselect (a (.virt 19203)) (a (.wire 63 7)) (a (.wire 58 35)) ∧
    a (.wire 64 15) = bselect (a (.virt 19203)) (a (.wire 58 35)) (a (.wire 63 7))) :=
  ⟨⟨privateBatchWrapper2_f0 a h, privateBatchWrapper2_f1 a h, privateBatchWrapper2_f2 a h, privateBatchWrapper2_f3 a h, privateBatchWrapper2_f4 a h, privateBatchWrapper2_f5 a h, privateBatchWrapper2_f6 a h, privateBatchWrapper2_f7 a h, privateBatchWrapper2_f8 a h, privateBatchWrapper2_f9 a h, privateBatchWrapper2_f10 a h, privateBatchWrapper2_f11 a h, privateBatchWrapper2_f12 a h, privateBatchWrapper2_f13 a h, privateBatchWrapper2_f14 a h, privateBatchWrapper2_f15 a h, privateBatchWrapper2_f16 a h, privateBatchWrapper2_f17 a h, privateBatchWrapper2_f18 a h, privateBatchWrapper2_f19 a h, privateBatchWrapper2_f20 a h, privateBatchWrapper2_f21 a h, privateBatchWrapper2_f22 a h, privateBatchWrapper2_f23 a h, privateBatchWrapper2_f24 a h, privateBatchWrapper2_f25 a h, privateBatchWrapper2_f26 a h, privateBatchWrapper2_f27 a h, privateBatchWrapper2_f28 a h, privateBatchWrapper2_f29 a h, privateBatchWrapper2_f30 a h, privateBatchWrapper2_f31 a h⟩, ⟨privateBatchWrapper2_f32 a h, privateBatchWrapper2_f33 a h, privateBatchWrapper2_f34 a h, privateBatchWrapper2_f35 a h, privateBatchWrapper2_f36 a h, privateBatchWrapper2_f37 a h, privateBatchWrapper2_f38 a h, privateBatchWrapper2_f39 a h, privateBatchWrapper2_f40 a h, privateBatchWrapper2_f41 a h, privateBatchWrapper2_f42 a h, privateBatchWrapper2_f43 a h, privateBatchWrapper2_f44 a h, privateBatchWrapper2_f45 a h, privateBatchWrapper2_f46 a h, privateBatchWrapper2_f47 a h, privateBatchWrapper2_f48 a h, privateBatchWrapper2_f49 a h, privateBatchWrapper2_f50 a h, privateBatchWrapper2_f51 a h, privateBatchWrapper2_f52 a h, privateBatchWrapper2_f53 a h, privateBatchWrapper2_f54 a h, privateBatchWrapper2_f55 a h, privateBatchWrapper2_f56 a h, privateBatchWrapper2_f57 a h, privateBatchWrapper2_f58 a h, privateBatchWrapper2_f59 a h, privateBatchWrapper2_f60 a h, privateBatchWrapper2_f61 a h, privateBatchWrapper2_f62 a h, privateBatchWrapper2_f63 a h⟩, ⟨privateBatchWrapper2_f64 a h, privateBatchWrapper2_f65 a h, privateBatchWrapper2_f66 a h, privateBatchWrapper2_f67 a h, privateBatchWrapper2_f68 a h, privateBatchWrapper2_f69 a h, privateBatchWrapper2_f70 a h, privateBatchWrapper2_f71 a h, privateBatchWrapper2_f72 a h, privateBatchWrapper2_f73 a h, privateBatchWrapper2_f74 a h, privateBatchWrapper2_f75 a h, privateBatchWrapper2_f76 a h, privateBatchWrapper2_f77 a h, privateBatchWrapper2_f78 a h, privateBatchWrapper2_f79 a h, privateBatchWrapper2_f80 a h, privateBatchWrapper2_f81 a h, privateBatchWrapper2_f82 a h, privateBatchWrapper2_f83 a h, privateBatchWrapper2_f84 a h, privateBatchWrapper2_f85 a h, privateBatchWrapper2_f86 a h, privateBatchWrapper2_f87 a h, privateBatchWrapper2_f88 a h, privateBatchWrapper2_f89 a h, privateBatchWrapper2_f90 a h, privateBatchWrapper2_f91 a h, privateBatchWrapper2_f92 a h, privateBatchWrapper2_f93 a h, privateBatchWrapper2_f94 a h, privateBatchWrapper2_f95 a h⟩, ⟨privateBatchWrapper2_f96 a h, privateBatchWrapper2_f97 a h, privateBatchWrapper2_f98 a h, privateBatchWrapper2_f99 a h, privateBatchWrapper2_f100 a h, privateBatchWrapper2_f101 a h, privateBatchWrapper2_f102 a h, privateBatchWrapper2_f103 a h, privateBatchWrapper2_f104 a h, privateBatchWrapper2_f105 a h, privateBatchWrapper2_f106 a h, privateBatchWrapper2_f107 a h, privateBatchWrapper2_f108 a h, privateBatchWrapper2_f109 a h, privateBatchWrapper2_f110 a h, privateBatchWrapper2_f111 a h, privateBatchWrapper2_f112 a h, privateBatchWrapper2_f113 a h, privateBatchWrapper2_f114 a h, privateBatchWrapper2_f115 a h, privateBatchWrapper2_f116 a h, privateBatchWrapper2_f117 a h, privateBatchWrapper2_f118 a h, privateBatchWrapper2_f119 a h, privateBatchWrapper2_f120 a h, privateBatchWrapper2_f121 a h, privateBatchWrapper2_f122 a h, privateBatchWrapper2_f123 a h, privateBatchWrapper2_f124 a h, privateBatchWrapper2_f125 a h, privateBatchWrapper2_f126 a h, privateBatchWrapper2_f127 a h⟩, ⟨privateBatchWrapper2_f128 a h, privateBatchWrapper2_f129 a h, privateBatchWrapper2_f130 a h, privateBatchWrapper2_f131 a h, privateBatchWrapper2_f132 a h, privateBatchWrapper2_f133 a h, privateBatchWrapper2_f134 a h, privateBatchWrapper2_f135 a h, privateBatchWrapper2_f136 a h, privateBatchWrapper2_f137 a h, privateBatchWrapper2_f138 a h, privateBatchWrapper2_f139 a h, privateBatchWrapper2_f140 a h, privateBatchWrapper2_f141 a h, privateBatchWrapper2_f142 a h, privateBatchWrapper2_f143 a h, privateBatchWrapper2_f144 a h, privateBatchWrapper2_f145 a h, privateBatchWrapper2_f146 a h, privateBatchWrapper2_f147 a h, privateBatchWrapper2_f148 a h, privateBatchWrapper2_f149 a h, privateBatchWrapper2_f150 a h, privateBatchWrapper2_f151 a h, privateBatchWrapper2_f152 a h, privateBatchWrapper2_f153 a h, privateBatchWrapper2_f154 a h, privateBatchWrapper2_f155 a h, privateBatchWrapper2_f156 a h, privateBatchWrapper2_f157 a h, privateBatchWrapper2_f158 a h, privateBatchWrapper2_f159 a h⟩, ⟨privateBatchWrapper2_f160 a h, privateBatchWrapper2_f161 a h, privateBatchWrapper2_f162 a h, privateBatchWrapper2_f163 a h, privateBatchWrapper2_f164 a h, privateBatchWrapper2_f165 a h, privateBatchWrapper2_f166 a h, privateBatchWrapper2_f167 a h, privateBatchWrapper2_f168 a h, privateBatchWrapper2_f169 a h, privateBatchWrapper2_f170 a h, privateBatchWrapper2_f171 a h, privateBatchWrapper2_f172 a h, privateBatchWrapper2_f173 a h, privateBatchWrapper2_f174 a h, privateBatchWrapper2_f175 a h, privateBatchWrapper2_f176 a h, privateBatchWrapper2_f177 a h, privateBatchWrapper2_f178 a h, privateBatchWrapper2_f179 a h, privateBatchWrapper2_f180 a h, privateBatchWrapper2_f181 a h, privateBatchWrapper2_f182 a h, privateBatchWrapper2_f183 a h, privateBatchWrapper2_f184 a h, privateBatchWrapper2_f185 a h, privateBatchWrapper2_f186 a h, privateBatchWrapper2_f187 a h, privateBatchWrapper2_f188 a h, privateBatchWrapper2_f189 a h, privateBatchWrapper2_f190 a h, privateBatchWrapper2_f191 a h⟩, ⟨privateBatchWrapper2_f192 a h, privateBatchWrapper2_f193 a h, privateBatchWrapper2_f194 a h, privateBatchWrapper2_f195 a h, privateBatchWrapper2_f196 a h, privateBatchWrapper2_f197 a h, privateBatchWrapper2_f198 a h, privateBatchWrapper2_f199 a h, privateBatchWrapper2_f200 a h, privateBatchWrapper2_f201 a h, privateBatchWrapper2_f202 a h, privateBatchWrapper2_f203 a h, privateBatchWrapper2_f204 a h, privateBatchWrapper2_f205 a h, privateBatchWrapper2_f206 a h, privateBatchWrapper2_f207 a h, privateBatchWrapper2_f208 a h, privateBatchWrapper2_f209 a h, privateBatchWrapper2_f210 a h, privateBatchWrapper2_f211 a h, privateBatchWrapper2_f212 a h, privateBatchWrapper2_f213 a h, privateBatchWrapper2_f214 a h, privateBatchWrapper2_f215 a h, privateBatchWrapper2_f216 a h, privateBatchWrapper2_f217 a h, privateBatchWrapper2_f218 a h, privateBatchWrapper2_f219 a h, privateBatchWrapper2_f220 a h, privateBatchWrapper2_f221 a h, privateBatchWrapper2_f222 a h, privateBatchWrapper2_f223 a h⟩, ⟨privateBatchWrapper2_f224 a h, privateBatchWrapper2_f225 a h, privateBatchWrapper2_f226 a h, privateBatchWrapper2_f227 a h, privateBatchWrapper2_f228 a h, privateBatchWrapper2_f229 a h, privateBatchWrapper2_f230 a h, privateBatchWrapper2_f231 a h, privateBatchWrapper2_f232 a h, privateBatchWrapper2_f233 a h, privateBatchWrapper2_f234 a h, privateBatchWrapper2_f235 a h, privateBatchWrapper2_f236 a h, privateBatchWrapper2_f237 a h, privateBatchWrapper2_f238 a h, privateBatchWrapper2_f239 a h, privateBatchWrapper2_f240 a h, privateBatchWrapper2_f241 a h, privateBatchWrapper2_f242 a h, privateBatchWrapper2_f243 a h, privateBatchWrapper2_f244 a h, privateBatchWrapper2_f245 a h, privateBatchWrapper2_f246 a h, privateBatchWrapper2_f247 a h, privateBatchWrapper2_f248 a h, privateBatchWrapper2_f249 a h, privateBatchWrapper2_f250 a h, privateBatchWrapper2_f251 a h, privateBatchWrapper2_f252 a h, privateBatchWrapper2_f253 a h, privateBatchWrapper2_f254 a h, privateBatchWrapper2_f255 a h⟩, ⟨privateBatchWrapper2_f256 a h, privateBatchWrapper2_f257 a h, privateBatchWrapper2_f258 a h, privateBatchWrapper2_f259 a h, privateBatchWrapper2_f260 a h, privateBatchWrapper2_f261 a h, privateBatchWrapper2_f262 a h, privateBatchWrapper2_f263 a h, privateBatchWrapper2_f264 a h, privateBatchWrapper2_f265 a h, privateBatchWrapper2_f266 a h, privateBatchWrapper2_f267 a h, privateBatchWrapper2_f268 a h, privateBatchWrapper2_f269 a h, privateBatchWrapper2_f270 a h, privateBatchWrapper2_f271 a h, privateBatchWrapper2_f272 a h, privateBatchWrapper2_f273 a h, privateBatchWrapper2_f274 a h, privateBatchWrapper2_f275 a h, privateBatchWrapper2_f276 a h, privateBatchWrapper2_f277 a h, privateBatchWrapper2_f278 a h, privateBatchWrapper2_f279 a h, privateBatchWrapper2_f280 a h, privateBatchWrapper2_f281 a h, privateBatchWrapper2_f282 a h, privateBatchWrapper2_f283 a h, privateBatchWrapper2_f284 a h, privateBatchWrapper2_f285 a h, privateBatchWrapper2_f286 a h, privateBatchWrapper2_f287 a h⟩, ⟨privateBatchWrapper2_f288 a h, privateBatchWrapper2_f289 a h, privateBatchWrapper2_f290 a h, privateBatchWrapper2_f291 a h, privateBatchWrapper2_f292 a h, privateBatchWrapper2_f293 a h, privateBatchWrapper2_f294 a h, privateBatchWrapper2_f295 a h, privateBatchWrapper2_f296 a h, privateBatchWrapper2_f297 a h, privateBatchWrapper2_f298 a h, privateBatchWrapper2_f299 a h, privateBatchWrapper2_f300 a h, privateBatchWrapper2_f301 a h, privateBatchWrapper2_f302 a h, privateBatchWrapper2_f303 a h, privateBatchWrapper2_f304 a h, privateBatchWrapper2_f305 a h, privateBatchWrapper2_f306 a h, privateBatchWrapper2_f307 a h, privateBatchWrapper2_f308 a h, privateBatchWrapper2_f309 a h, privateBatchWrapper2_f310 a h, privateBatchWrapper2_f311 a h, privateBatchWrapper2_f312 a h, privateBatchWrapper2_f313 a h, privateBatchWrapper2_f314 a h, privateBatchWrapper2_f315 a h, privateBatchWrapper2_f316 a h, privateBatchWrapper2_f317 a h, privateBatchWrapper2_f318 a h, privateBatchWrapper2_f319 a h⟩, ⟨privateBatchWrapper2_f320 a h, privateBatchWrapper2_f321 a h, privateBatchWrapper2_f322 a h, privateBatchWrapper2_f323 perm a h hp, privateBatchWrapper2_f324 perm a h hp, privateBatchWrapper2_f325 a h, privateBatchWrapper2_f326 a h, privateBatchWrapper2_f327 a h, privateBatchWrapper2_f328 a h, privateBatchWrapper2_f329 perm a h hp, privateBatchWrapper2_f330 perm a h hp, privateBatchWrapper2_f331 a h, privateBatchWrapper2_f332 a h, privateBatchWrapper2_f333 a h, privateBatchWrapper2_f334 a h, privateBatchWrapper2_f335 a h, privateBatchWrapper2_f336 a h, privateBatchWrapper2_f337 a h, privateBatchWrapper2_f338 a h, privateBatchWrapper2_f339 a h, privateBatchWrapper2_f340 a h, privateBatchWrapper2_f341 a h, privateBatchWrapper2_f342 a h, privateBatchWrapper2_f343 a h⟩⟩

end Plonky2Spec.Generated
