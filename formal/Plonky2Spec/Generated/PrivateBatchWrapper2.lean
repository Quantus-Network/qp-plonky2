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
open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

/-- `privateBatchWrapper2.copies`, items `0..32`. -/
def privateBatchWrapper2.copies0 : List (Target × Target) := [
    (.virt 13209, .wire 0 0),
    (.virt 13209, .wire 0 1),
    (.virt 13212, .wire 0 2),
    (.virt 13212, .wire 1 0),
    (.virt 6589, .wire 1 1),
    (.virt 13212, .wire 1 2),
    (.virt 6589, .wire 1 4),
    (.virt 13213, .wire 1 5),
    (.virt 6589, .wire 1 6),
    (.wire 1 7, .wire 0 4),
    (.virt 13209, .wire 0 5),
    (.wire 0 3, .wire 0 6),
    (.wire 1 3, .virt 13210),
    (.wire 0 7, .virt 13210),
    (.virt 13209, .wire 0 8),
    (.virt 13209, .wire 0 9),
    (.virt 13214, .wire 0 10),
    (.virt 13214, .wire 1 8),
    (.virt 6590, .wire 1 9),
    (.virt 13214, .wire 1 10),
    (.virt 6590, .wire 1 12),
    (.virt 13215, .wire 1 13),
    (.virt 6590, .wire 1 14),
    (.wire 1 15, .wire 0 12),
    (.virt 13209, .wire 0 13),
    (.wire 0 11, .wire 0 14),
    (.wire 1 11, .virt 13210),
    (.wire 0 15, .virt 13210),
    (.virt 13209, .wire 0 16),
    (.virt 13209, .wire 0 17),
    (.virt 13216, .wire 0 18),
    (.virt 13216, .wire 1 16)
  ]

/-- `privateBatchWrapper2.copies`, items `32..64`. -/
def privateBatchWrapper2.copies1 : List (Target × Target) := [
    (.virt 6591, .wire 1 17),
    (.virt 13216, .wire 1 18),
    (.virt 6591, .wire 1 20),
    (.virt 13217, .wire 1 21),
    (.virt 6591, .wire 1 22),
    (.wire 1 23, .wire 0 20),
    (.virt 13209, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.wire 1 19, .virt 13210),
    (.wire 0 23, .virt 13210),
    (.virt 13209, .wire 0 24),
    (.virt 13209, .wire 0 25),
    (.virt 13218, .wire 0 26),
    (.virt 13218, .wire 1 24),
    (.virt 6592, .wire 1 25),
    (.virt 13218, .wire 1 26),
    (.virt 6592, .wire 1 28),
    (.virt 13219, .wire 1 29),
    (.virt 6592, .wire 1 30),
    (.wire 1 31, .wire 0 28),
    (.virt 13209, .wire 0 29),
    (.wire 0 27, .wire 0 30),
    (.wire 1 27, .virt 13210),
    (.wire 0 31, .virt 13210),
    (.virt 13212, .wire 1 32),
    (.virt 13214, .wire 1 33),
    (.virt 13212, .wire 1 34),
    (.virt 13216, .wire 1 36),
    (.virt 13218, .wire 1 37),
    (.virt 13216, .wire 1 38),
    (.wire 1 35, .wire 1 40),
    (.wire 1 39, .wire 1 41)
  ]

/-- `privateBatchWrapper2.copies`, items `64..96`. -/
def privateBatchWrapper2.copies2 : List (Target × Target) := [
    (.wire 1 35, .wire 1 42),
    (.virt 13209, .wire 0 32),
    (.virt 13209, .wire 0 33),
    (.virt 13220, .wire 0 34),
    (.virt 13220, .wire 1 44),
    (.virt 13184, .wire 1 45),
    (.virt 13220, .wire 1 46),
    (.virt 13184, .wire 1 48),
    (.virt 13221, .wire 1 49),
    (.virt 13184, .wire 1 50),
    (.wire 1 51, .wire 0 36),
    (.virt 13209, .wire 0 37),
    (.wire 0 35, .wire 0 38),
    (.wire 1 47, .virt 13210),
    (.wire 0 39, .virt 13210),
    (.virt 13209, .wire 0 40),
    (.virt 13209, .wire 0 41),
    (.virt 13222, .wire 0 42),
    (.virt 13222, .wire 1 52),
    (.virt 13185, .wire 1 53),
    (.virt 13222, .wire 1 54),
    (.virt 13185, .wire 1 56),
    (.virt 13223, .wire 1 57),
    (.virt 13185, .wire 1 58),
    (.wire 1 59, .wire 0 44),
    (.virt 13209, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.wire 1 55, .virt 13210),
    (.wire 0 47, .virt 13210),
    (.virt 13209, .wire 0 48),
    (.virt 13209, .wire 0 49),
    (.virt 13224, .wire 0 50)
  ]

/-- `privateBatchWrapper2.copies`, items `96..128`. -/
def privateBatchWrapper2.copies3 : List (Target × Target) := [
    (.virt 13224, .wire 2 0),
    (.virt 13186, .wire 2 1),
    (.virt 13224, .wire 2 2),
    (.virt 13186, .wire 2 4),
    (.virt 13225, .wire 2 5),
    (.virt 13186, .wire 2 6),
    (.wire 2 7, .wire 0 52),
    (.virt 13209, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.wire 2 3, .virt 13210),
    (.wire 0 55, .virt 13210),
    (.virt 13209, .wire 0 56),
    (.virt 13209, .wire 0 57),
    (.virt 13226, .wire 0 58),
    (.virt 13226, .wire 2 8),
    (.virt 13187, .wire 2 9),
    (.virt 13226, .wire 2 10),
    (.virt 13187, .wire 2 12),
    (.virt 13227, .wire 2 13),
    (.virt 13187, .wire 2 14),
    (.wire 2 15, .wire 3 0),
    (.virt 13209, .wire 3 1),
    (.wire 0 59, .wire 3 2),
    (.wire 2 11, .virt 13210),
    (.wire 3 3, .virt 13210),
    (.virt 13220, .wire 2 16),
    (.virt 13222, .wire 2 17),
    (.virt 13220, .wire 2 18),
    (.virt 13224, .wire 2 20),
    (.virt 13226, .wire 2 21),
    (.virt 13224, .wire 2 22),
    (.wire 2 19, .wire 2 24)
  ]

/-- `privateBatchWrapper2.copies`, items `128..160`. -/
def privateBatchWrapper2.copies4 : List (Target × Target) := [
    (.wire 2 23, .wire 2 25),
    (.wire 2 19, .wire 2 26),
    (.virt 13209, .wire 3 4),
    (.virt 13209, .wire 3 5),
    (.wire 1 43, .wire 3 6),
    (.wire 3 7, .wire 3 8),
    (.virt 6589, .wire 3 9),
    (.virt 13210, .wire 3 10),
    (.wire 3 7, .wire 3 12),
    (.virt 6590, .wire 3 13),
    (.virt 13210, .wire 3 14),
    (.wire 3 7, .wire 3 16),
    (.virt 6591, .wire 3 17),
    (.virt 13210, .wire 3 18),
    (.wire 3 7, .wire 3 20),
    (.virt 6592, .wire 3 21),
    (.virt 13210, .wire 3 22),
    (.wire 3 7, .wire 3 24),
    (.virt 6593, .wire 3 25),
    (.virt 13210, .wire 3 26),
    (.virt 13209, .wire 3 28),
    (.virt 13209, .wire 3 29),
    (.wire 2 27, .wire 3 30),
    (.virt 13209, .wire 3 32),
    (.virt 13209, .wire 3 33),
    (.wire 3 7, .wire 3 34),
    (.wire 3 31, .wire 2 28),
    (.wire 3 35, .wire 2 29),
    (.wire 3 31, .wire 2 30),
    (.wire 2 31, .wire 3 36),
    (.wire 3 11, .wire 3 37),
    (.wire 3 11, .wire 3 38)
  ]

/-- `privateBatchWrapper2.copies`, items `160..192`. -/
def privateBatchWrapper2.copies5 : List (Target × Target) := [
    (.wire 2 31, .wire 3 40),
    (.virt 13184, .wire 3 41),
    (.wire 3 39, .wire 3 42),
    (.wire 2 31, .wire 3 44),
    (.wire 3 15, .wire 3 45),
    (.wire 3 15, .wire 3 46),
    (.wire 2 31, .wire 3 48),
    (.virt 13185, .wire 3 49),
    (.wire 3 47, .wire 3 50),
    (.wire 2 31, .wire 3 52),
    (.wire 3 19, .wire 3 53),
    (.wire 3 19, .wire 3 54),
    (.wire 2 31, .wire 3 56),
    (.virt 13186, .wire 3 57),
    (.wire 3 55, .wire 3 58),
    (.wire 2 31, .wire 4 0),
    (.wire 3 23, .wire 4 1),
    (.wire 3 23, .wire 4 2),
    (.wire 2 31, .wire 4 4),
    (.virt 13187, .wire 4 5),
    (.wire 4 3, .wire 4 6),
    (.wire 2 31, .wire 4 8),
    (.wire 3 27, .wire 4 9),
    (.wire 3 27, .wire 4 10),
    (.wire 2 31, .wire 4 12),
    (.virt 13188, .wire 4 13),
    (.wire 4 11, .wire 4 14),
    (.wire 3 7, .wire 5 0),
    (.wire 3 31, .wire 5 1),
    (.wire 3 7, .wire 5 2),
    (.wire 5 3, .wire 6 0),
    (.virt 13209, .wire 6 1)
  ]

/-- `privateBatchWrapper2.copies`, items `192..224`. -/
def privateBatchWrapper2.copies6 : List (Target × Target) := [
    (.wire 3 31, .wire 6 2),
    (.virt 13209, .wire 4 16),
    (.virt 13209, .wire 4 17),
    (.virt 13228, .wire 4 18),
    (.virt 6589, .wire 4 20),
    (.virt 13209, .wire 4 21),
    (.wire 3 43, .wire 4 22),
    (.virt 13228, .wire 2 32),
    (.wire 4 23, .wire 2 33),
    (.virt 13228, .wire 2 34),
    (.wire 4 23, .wire 2 36),
    (.virt 13229, .wire 2 37),
    (.wire 4 23, .wire 2 38),
    (.wire 2 39, .wire 4 24),
    (.virt 13209, .wire 4 25),
    (.wire 4 19, .wire 4 26),
    (.wire 2 35, .virt 13210),
    (.wire 4 27, .virt 13210),
    (.virt 13209, .wire 4 28),
    (.virt 13209, .wire 4 29),
    (.virt 13230, .wire 4 30),
    (.virt 6590, .wire 4 32),
    (.virt 13209, .wire 4 33),
    (.wire 3 51, .wire 4 34),
    (.virt 13230, .wire 2 40),
    (.wire 4 35, .wire 2 41),
    (.virt 13230, .wire 2 42),
    (.wire 4 35, .wire 2 44),
    (.virt 13231, .wire 2 45),
    (.wire 4 35, .wire 2 46),
    (.wire 2 47, .wire 4 36),
    (.virt 13209, .wire 4 37)
  ]

/-- `privateBatchWrapper2.copies`, items `224..256`. -/
def privateBatchWrapper2.copies7 : List (Target × Target) := [
    (.wire 4 31, .wire 4 38),
    (.wire 2 43, .virt 13210),
    (.wire 4 39, .virt 13210),
    (.virt 13209, .wire 4 40),
    (.virt 13209, .wire 4 41),
    (.virt 13232, .wire 4 42),
    (.virt 6591, .wire 4 44),
    (.virt 13209, .wire 4 45),
    (.wire 3 59, .wire 4 46),
    (.virt 13232, .wire 2 48),
    (.wire 4 47, .wire 2 49),
    (.virt 13232, .wire 2 50),
    (.wire 4 47, .wire 2 52),
    (.virt 13233, .wire 2 53),
    (.wire 4 47, .wire 2 54),
    (.wire 2 55, .wire 4 48),
    (.virt 13209, .wire 4 49),
    (.wire 4 43, .wire 4 50),
    (.wire 2 51, .virt 13210),
    (.wire 4 51, .virt 13210),
    (.virt 13209, .wire 4 52),
    (.virt 13209, .wire 4 53),
    (.virt 13234, .wire 4 54),
    (.virt 6592, .wire 4 56),
    (.virt 13209, .wire 4 57),
    (.wire 4 7, .wire 4 58),
    (.virt 13234, .wire 2 56),
    (.wire 4 59, .wire 2 57),
    (.virt 13234, .wire 2 58),
    (.wire 4 59, .wire 7 0),
    (.virt 13235, .wire 7 1),
    (.wire 4 59, .wire 7 2)
  ]

/-- `privateBatchWrapper2.copies`, items `256..288`. -/
def privateBatchWrapper2.copies8 : List (Target × Target) := [
    (.wire 7 3, .wire 8 0),
    (.virt 13209, .wire 8 1),
    (.wire 4 55, .wire 8 2),
    (.wire 2 59, .virt 13210),
    (.wire 8 3, .virt 13210),
    (.virt 13228, .wire 7 4),
    (.virt 13230, .wire 7 5),
    (.virt 13228, .wire 7 6),
    (.virt 13232, .wire 7 8),
    (.virt 13234, .wire 7 9),
    (.virt 13232, .wire 7 10),
    (.wire 7 7, .wire 7 12),
    (.wire 7 11, .wire 7 13),
    (.wire 7 7, .wire 7 14),
    (.wire 1 43, .wire 5 4),
    (.wire 7 15, .wire 5 5),
    (.wire 1 43, .wire 5 6),
    (.wire 5 7, .wire 6 4),
    (.virt 13209, .wire 6 5),
    (.wire 7 15, .wire 6 6),
    (.wire 6 7, .virt 13209),
    (.virt 6584, .virt 6584),
    (.virt 13209, .wire 8 4),
    (.virt 13209, .wire 8 5),
    (.virt 13236, .wire 8 6),
    (.virt 13184, .wire 8 8),
    (.virt 13209, .wire 8 9),
    (.wire 3 43, .wire 8 10),
    (.virt 13236, .wire 7 16),
    (.wire 8 11, .wire 7 17),
    (.virt 13236, .wire 7 18),
    (.wire 8 11, .wire 7 20)
  ]

/-- `privateBatchWrapper2.copies`, items `288..320`. -/
def privateBatchWrapper2.copies9 : List (Target × Target) := [
    (.virt 13237, .wire 7 21),
    (.wire 8 11, .wire 7 22),
    (.wire 7 23, .wire 8 12),
    (.virt 13209, .wire 8 13),
    (.wire 8 7, .wire 8 14),
    (.wire 7 19, .virt 13210),
    (.wire 8 15, .virt 13210),
    (.virt 13209, .wire 8 16),
    (.virt 13209, .wire 8 17),
    (.virt 13238, .wire 8 18),
    (.virt 13185, .wire 8 20),
    (.virt 13209, .wire 8 21),
    (.wire 3 51, .wire 8 22),
    (.virt 13238, .wire 7 24),
    (.wire 8 23, .wire 7 25),
    (.virt 13238, .wire 7 26),
    (.wire 8 23, .wire 7 28),
    (.virt 13239, .wire 7 29),
    (.wire 8 23, .wire 7 30),
    (.wire 7 31, .wire 8 24),
    (.virt 13209, .wire 8 25),
    (.wire 8 19, .wire 8 26),
    (.wire 7 27, .virt 13210),
    (.wire 8 27, .virt 13210),
    (.virt 13209, .wire 8 28),
    (.virt 13209, .wire 8 29),
    (.virt 13240, .wire 8 30),
    (.virt 13186, .wire 8 32),
    (.virt 13209, .wire 8 33),
    (.wire 3 59, .wire 8 34),
    (.virt 13240, .wire 7 32),
    (.wire 8 35, .wire 7 33)
  ]

/-- `privateBatchWrapper2.copies`, items `320..352`. -/
def privateBatchWrapper2.copies10 : List (Target × Target) := [
    (.virt 13240, .wire 7 34),
    (.wire 8 35, .wire 7 36),
    (.virt 13241, .wire 7 37),
    (.wire 8 35, .wire 7 38),
    (.wire 7 39, .wire 8 36),
    (.virt 13209, .wire 8 37),
    (.wire 8 31, .wire 8 38),
    (.wire 7 35, .virt 13210),
    (.wire 8 39, .virt 13210),
    (.virt 13209, .wire 8 40),
    (.virt 13209, .wire 8 41),
    (.virt 13242, .wire 8 42),
    (.virt 13187, .wire 8 44),
    (.virt 13209, .wire 8 45),
    (.wire 4 7, .wire 8 46),
    (.virt 13242, .wire 7 40),
    (.wire 8 47, .wire 7 41),
    (.virt 13242, .wire 7 42),
    (.wire 8 47, .wire 7 44),
    (.virt 13243, .wire 7 45),
    (.wire 8 47, .wire 7 46),
    (.wire 7 47, .wire 8 48),
    (.virt 13209, .wire 8 49),
    (.wire 8 43, .wire 8 50),
    (.wire 7 43, .virt 13210),
    (.wire 8 51, .virt 13210),
    (.virt 13236, .wire 7 48),
    (.virt 13238, .wire 7 49),
    (.virt 13236, .wire 7 50),
    (.virt 13240, .wire 7 52),
    (.virt 13242, .wire 7 53),
    (.virt 13240, .wire 7 54)
  ]

/-- `privateBatchWrapper2.copies`, items `352..384`. -/
def privateBatchWrapper2.copies11 : List (Target × Target) := [
    (.wire 7 51, .wire 7 56),
    (.wire 7 55, .wire 7 57),
    (.wire 7 51, .wire 7 58),
    (.wire 2 27, .wire 5 8),
    (.wire 7 59, .wire 5 9),
    (.wire 2 27, .wire 5 10),
    (.wire 5 11, .wire 6 8),
    (.virt 13209, .wire 6 9),
    (.wire 7 59, .wire 6 10),
    (.wire 6 11, .virt 13209),
    (.virt 13179, .virt 6584),
    (.wire 1 43, .wire 8 52),
    (.virt 6594, .wire 8 53),
    (.virt 6594, .wire 8 54),
    (.wire 1 43, .wire 8 56),
    (.virt 13210, .wire 8 57),
    (.wire 8 55, .wire 8 58),
    (.wire 2 27, .wire 9 0),
    (.virt 13189, .wire 9 1),
    (.virt 13189, .wire 9 2),
    (.wire 2 27, .wire 9 4),
    (.virt 13210, .wire 9 5),
    (.wire 9 3, .wire 9 6),
    (.wire 8 59, .wire 6 12),
    (.virt 13209, .wire 6 13),
    (.wire 9 7, .wire 6 14),
    (.wire 10 33, .virt 13210),
    (.wire 10 34, .virt 13210),
    (.wire 10 35, .virt 13210),
    (.wire 10 36, .virt 13210),
    (.wire 10 37, .virt 13210),
    (.wire 10 38, .virt 13210)
  ]

/-- `privateBatchWrapper2.copies`, items `384..416`. -/
def privateBatchWrapper2.copies12 : List (Target × Target) := [
    (.wire 10 39, .virt 13210),
    (.wire 10 40, .virt 13210),
    (.wire 10 41, .virt 13210),
    (.wire 10 42, .virt 13210),
    (.wire 10 43, .virt 13210),
    (.wire 10 44, .virt 13210),
    (.wire 10 45, .virt 13210),
    (.wire 10 46, .virt 13210),
    (.wire 10 47, .virt 13210),
    (.wire 10 48, .virt 13210),
    (.wire 10 49, .virt 13210),
    (.wire 10 50, .virt 13210),
    (.wire 10 51, .virt 13210),
    (.wire 10 52, .virt 13210),
    (.wire 10 53, .virt 13210),
    (.wire 10 54, .virt 13210),
    (.wire 10 55, .virt 13210),
    (.wire 10 56, .virt 13210),
    (.wire 10 57, .virt 13210),
    (.wire 10 58, .virt 13210),
    (.wire 10 59, .virt 13210),
    (.wire 10 0, .virt 13195),
    (.wire 11 33, .virt 13210),
    (.wire 11 34, .virt 13210),
    (.wire 11 35, .virt 13210),
    (.wire 11 36, .virt 13210),
    (.wire 11 37, .virt 13210),
    (.wire 11 38, .virt 13210),
    (.wire 11 39, .virt 13210),
    (.wire 11 40, .virt 13210),
    (.wire 11 41, .virt 13210),
    (.wire 11 42, .virt 13210)
  ]

/-- `privateBatchWrapper2.copies`, items `416..448`. -/
def privateBatchWrapper2.copies13 : List (Target × Target) := [
    (.wire 11 43, .virt 13210),
    (.wire 11 44, .virt 13210),
    (.wire 11 45, .virt 13210),
    (.wire 11 46, .virt 13210),
    (.wire 11 47, .virt 13210),
    (.wire 11 48, .virt 13210),
    (.wire 11 49, .virt 13210),
    (.wire 11 50, .virt 13210),
    (.wire 11 51, .virt 13210),
    (.wire 11 52, .virt 13210),
    (.wire 11 53, .virt 13210),
    (.wire 11 54, .virt 13210),
    (.wire 11 55, .virt 13210),
    (.wire 11 56, .virt 13210),
    (.wire 11 57, .virt 13210),
    (.wire 11 58, .virt 13210),
    (.wire 11 59, .virt 13210),
    (.wire 11 0, .virt 13200),
    (.virt 13195, .wire 6 16),
    (.virt 13209, .wire 6 17),
    (.virt 13200, .wire 6 18),
    (.wire 12 15, .virt 13210),
    (.wire 12 16, .virt 13210),
    (.wire 12 17, .virt 13210),
    (.wire 12 18, .virt 13210),
    (.wire 12 19, .virt 13210),
    (.wire 12 20, .virt 13210),
    (.wire 12 21, .virt 13210),
    (.wire 12 22, .virt 13210),
    (.wire 12 23, .virt 13210),
    (.wire 12 24, .virt 13210),
    (.wire 12 25, .virt 13210)
  ]

/-- `privateBatchWrapper2.copies`, items `448..480`. -/
def privateBatchWrapper2.copies14 : List (Target × Target) := [
    (.wire 12 26, .virt 13210),
    (.wire 12 27, .virt 13210),
    (.wire 12 28, .virt 13210),
    (.wire 12 29, .virt 13210),
    (.wire 12 30, .virt 13210),
    (.wire 12 31, .virt 13210),
    (.wire 12 32, .virt 13210),
    (.wire 12 33, .virt 13210),
    (.wire 12 34, .virt 13210),
    (.wire 12 35, .virt 13210),
    (.wire 12 36, .virt 13210),
    (.wire 12 37, .virt 13210),
    (.wire 12 38, .virt 13210),
    (.wire 12 39, .virt 13210),
    (.wire 12 40, .virt 13210),
    (.wire 12 41, .virt 13210),
    (.wire 12 42, .virt 13210),
    (.wire 12 43, .virt 13210),
    (.wire 12 44, .virt 13210),
    (.wire 12 45, .virt 13210),
    (.wire 12 46, .virt 13210),
    (.wire 12 47, .virt 13210),
    (.wire 12 48, .virt 13210),
    (.wire 12 49, .virt 13210),
    (.wire 12 50, .virt 13210),
    (.wire 12 51, .virt 13210),
    (.wire 12 52, .virt 13210),
    (.wire 12 53, .virt 13210),
    (.wire 12 54, .virt 13210),
    (.wire 12 55, .virt 13210),
    (.wire 12 56, .virt 13210),
    (.wire 12 57, .virt 13210)
  ]

/-- `privateBatchWrapper2.copies`, items `480..512`. -/
def privateBatchWrapper2.copies15 : List (Target × Target) := [
    (.wire 12 58, .virt 13210),
    (.wire 12 59, .virt 13210),
    (.wire 12 0, .virt 13190),
    (.virt 13245, .wire 9 8),
    (.virt 13209, .wire 9 9),
    (.virt 13190, .wire 9 10),
    (.wire 13 15, .virt 13210),
    (.wire 13 16, .virt 13210),
    (.wire 13 17, .virt 13210),
    (.wire 13 18, .virt 13210),
    (.wire 13 19, .virt 13210),
    (.wire 13 20, .virt 13210),
    (.wire 13 21, .virt 13210),
    (.wire 13 22, .virt 13210),
    (.wire 13 23, .virt 13210),
    (.wire 13 24, .virt 13210),
    (.wire 13 25, .virt 13210),
    (.wire 13 26, .virt 13210),
    (.wire 13 27, .virt 13210),
    (.wire 13 28, .virt 13210),
    (.wire 13 29, .virt 13210),
    (.wire 13 30, .virt 13210),
    (.wire 13 31, .virt 13210),
    (.wire 13 32, .virt 13210),
    (.wire 13 33, .virt 13210),
    (.wire 13 34, .virt 13210),
    (.wire 13 35, .virt 13210),
    (.wire 13 36, .virt 13210),
    (.wire 13 37, .virt 13210),
    (.wire 13 38, .virt 13210),
    (.wire 13 39, .virt 13210),
    (.wire 13 40, .virt 13210)
  ]

/-- `privateBatchWrapper2.copies`, items `512..544`. -/
def privateBatchWrapper2.copies16 : List (Target × Target) := [
    (.wire 13 41, .virt 13210),
    (.wire 13 42, .virt 13210),
    (.wire 13 43, .virt 13210),
    (.wire 13 44, .virt 13210),
    (.wire 13 45, .virt 13210),
    (.wire 13 46, .virt 13210),
    (.wire 13 47, .virt 13210),
    (.wire 13 48, .virt 13210),
    (.wire 13 49, .virt 13210),
    (.wire 13 50, .virt 13210),
    (.wire 13 51, .virt 13210),
    (.wire 13 52, .virt 13210),
    (.wire 13 53, .virt 13210),
    (.wire 13 54, .virt 13210),
    (.wire 13 55, .virt 13210),
    (.wire 13 56, .virt 13210),
    (.wire 13 57, .virt 13210),
    (.wire 13 58, .virt 13210),
    (.wire 13 59, .virt 13210),
    (.wire 13 0, .wire 9 11),
    (.wire 6 19, .wire 14 0),
    (.virt 13245, .wire 14 1),
    (.wire 6 19, .wire 14 2),
    (.wire 6 15, .wire 14 4),
    (.wire 9 11, .wire 14 5),
    (.wire 6 15, .wire 14 6),
    (.wire 14 7, .wire 9 12),
    (.virt 13209, .wire 9 13),
    (.wire 14 3, .wire 9 14),
    (.wire 15 53, .virt 13210),
    (.wire 15 54, .virt 13210),
    (.wire 15 55, .virt 13210)
  ]

/-- `privateBatchWrapper2.copies`, items `544..576`. -/
def privateBatchWrapper2.copies17 : List (Target × Target) := [
    (.wire 15 56, .virt 13210),
    (.wire 15 57, .virt 13210),
    (.wire 15 58, .virt 13210),
    (.wire 15 59, .virt 13210),
    (.wire 15 0, .wire 9 15),
    (.wire 3 7, .wire 14 8),
    (.wire 3 31, .wire 14 9),
    (.wire 3 7, .wire 14 10),
    (.virt 13209, .wire 9 16),
    (.virt 13209, .wire 9 17),
    (.virt 13246, .wire 9 18),
    (.virt 6585, .wire 9 20),
    (.virt 13209, .wire 9 21),
    (.virt 13180, .wire 9 22),
    (.virt 13246, .wire 14 12),
    (.wire 9 23, .wire 14 13),
    (.virt 13246, .wire 14 14),
    (.wire 9 23, .wire 14 16),
    (.virt 13247, .wire 14 17),
    (.wire 9 23, .wire 14 18),
    (.wire 14 19, .wire 9 24),
    (.virt 13209, .wire 9 25),
    (.wire 9 19, .wire 9 26),
    (.wire 14 15, .virt 13210),
    (.wire 9 27, .virt 13210),
    (.virt 13209, .wire 9 28),
    (.virt 13209, .wire 9 29),
    (.virt 13248, .wire 9 30),
    (.virt 6586, .wire 9 32),
    (.virt 13209, .wire 9 33),
    (.virt 13181, .wire 9 34),
    (.virt 13248, .wire 14 20)
  ]

/-- `privateBatchWrapper2.copies`, items `576..608`. -/
def privateBatchWrapper2.copies18 : List (Target × Target) := [
    (.wire 9 35, .wire 14 21),
    (.virt 13248, .wire 14 22),
    (.wire 9 35, .wire 14 24),
    (.virt 13249, .wire 14 25),
    (.wire 9 35, .wire 14 26),
    (.wire 14 27, .wire 9 36),
    (.virt 13209, .wire 9 37),
    (.wire 9 31, .wire 9 38),
    (.wire 14 23, .virt 13210),
    (.wire 9 39, .virt 13210),
    (.virt 13209, .wire 9 40),
    (.virt 13209, .wire 9 41),
    (.virt 13250, .wire 9 42),
    (.virt 6587, .wire 9 44),
    (.virt 13209, .wire 9 45),
    (.virt 13182, .wire 9 46),
    (.virt 13250, .wire 14 28),
    (.wire 9 47, .wire 14 29),
    (.virt 13250, .wire 14 30),
    (.wire 9 47, .wire 14 32),
    (.virt 13251, .wire 14 33),
    (.wire 9 47, .wire 14 34),
    (.wire 14 35, .wire 9 48),
    (.virt 13209, .wire 9 49),
    (.wire 9 43, .wire 9 50),
    (.wire 14 31, .virt 13210),
    (.wire 9 51, .virt 13210),
    (.virt 13209, .wire 9 52),
    (.virt 13209, .wire 9 53),
    (.virt 13252, .wire 9 54),
    (.virt 6588, .wire 9 56),
    (.virt 13209, .wire 9 57)
  ]

/-- `privateBatchWrapper2.copies`, items `608..640`. -/
def privateBatchWrapper2.copies19 : List (Target × Target) := [
    (.virt 13183, .wire 9 58),
    (.virt 13252, .wire 14 36),
    (.wire 9 59, .wire 14 37),
    (.virt 13252, .wire 14 38),
    (.wire 9 59, .wire 14 40),
    (.virt 13253, .wire 14 41),
    (.wire 9 59, .wire 14 42),
    (.wire 14 43, .wire 16 0),
    (.virt 13209, .wire 16 1),
    (.wire 9 55, .wire 16 2),
    (.wire 14 39, .virt 13210),
    (.wire 16 3, .virt 13210),
    (.virt 13246, .wire 14 44),
    (.virt 13248, .wire 14 45),
    (.virt 13246, .wire 14 46),
    (.virt 13250, .wire 14 48),
    (.virt 13252, .wire 14 49),
    (.virt 13250, .wire 14 50),
    (.wire 14 47, .wire 14 52),
    (.wire 14 51, .wire 14 53),
    (.wire 14 47, .wire 14 54),
    (.wire 14 11, .wire 14 56),
    (.wire 14 55, .wire 14 57),
    (.wire 14 11, .wire 14 58),
    (.wire 14 59, .virt 13210),
    (.virt 13201, .wire 17 0),
    (.virt 13202, .wire 17 1),
    (.virt 13203, .wire 17 2),
    (.virt 13204, .wire 17 3),
    (.virt 13209, .wire 17 4),
    (.virt 13210, .wire 17 5),
    (.virt 13210, .wire 17 6)
  ]

/-- `privateBatchWrapper2.copies`, items `640..672`. -/
def privateBatchWrapper2.copies20 : List (Target × Target) := [
    (.virt 13210, .wire 17 7),
    (.virt 13210, .wire 17 8),
    (.virt 13210, .wire 17 9),
    (.virt 13210, .wire 17 10),
    (.virt 13210, .wire 17 11),
    (.wire 17 12, .wire 18 0),
    (.wire 17 13, .wire 18 1),
    (.wire 17 14, .wire 18 2),
    (.wire 17 15, .wire 18 3),
    (.virt 13209, .wire 18 4),
    (.virt 13210, .wire 18 5),
    (.virt 13210, .wire 18 6),
    (.virt 13210, .wire 18 7),
    (.virt 13210, .wire 18 8),
    (.virt 13210, .wire 18 9),
    (.virt 13210, .wire 18 10),
    (.virt 13210, .wire 18 11),
    (.wire 1 43, .wire 16 4),
    (.virt 6585, .wire 16 5),
    (.virt 6585, .wire 16 6),
    (.wire 1 43, .wire 16 8),
    (.wire 18 12, .wire 16 9),
    (.wire 16 7, .wire 16 10),
    (.wire 1 43, .wire 16 12),
    (.virt 6586, .wire 16 13),
    (.virt 6586, .wire 16 14),
    (.wire 1 43, .wire 16 16),
    (.wire 18 13, .wire 16 17),
    (.wire 16 15, .wire 16 18),
    (.wire 1 43, .wire 16 20),
    (.virt 6587, .wire 16 21),
    (.virt 6587, .wire 16 22)
  ]

/-- `privateBatchWrapper2.copies`, items `672..704`. -/
def privateBatchWrapper2.copies21 : List (Target × Target) := [
    (.wire 1 43, .wire 16 24),
    (.wire 18 14, .wire 16 25),
    (.wire 16 23, .wire 16 26),
    (.wire 1 43, .wire 16 28),
    (.virt 6588, .wire 16 29),
    (.virt 6588, .wire 16 30),
    (.wire 1 43, .wire 16 32),
    (.wire 18 15, .wire 16 33),
    (.wire 16 31, .wire 16 34),
    (.virt 13205, .wire 19 0),
    (.virt 13206, .wire 19 1),
    (.virt 13207, .wire 19 2),
    (.virt 13208, .wire 19 3),
    (.virt 13209, .wire 19 4),
    (.virt 13210, .wire 19 5),
    (.virt 13210, .wire 19 6),
    (.virt 13210, .wire 19 7),
    (.virt 13210, .wire 19 8),
    (.virt 13210, .wire 19 9),
    (.virt 13210, .wire 19 10),
    (.virt 13210, .wire 19 11),
    (.wire 19 12, .wire 20 0),
    (.wire 19 13, .wire 20 1),
    (.wire 19 14, .wire 20 2),
    (.wire 19 15, .wire 20 3),
    (.virt 13209, .wire 20 4),
    (.virt 13210, .wire 20 5),
    (.virt 13210, .wire 20 6),
    (.virt 13210, .wire 20 7),
    (.virt 13210, .wire 20 8),
    (.virt 13210, .wire 20 9),
    (.virt 13210, .wire 20 10)
  ]

/-- `privateBatchWrapper2.copies`, items `704..729`. -/
def privateBatchWrapper2.copies22 : List (Target × Target) := [
    (.virt 13210, .wire 20 11),
    (.wire 2 27, .wire 16 36),
    (.virt 13180, .wire 16 37),
    (.virt 13180, .wire 16 38),
    (.wire 2 27, .wire 16 40),
    (.wire 20 12, .wire 16 41),
    (.wire 16 39, .wire 16 42),
    (.wire 2 27, .wire 16 44),
    (.virt 13181, .wire 16 45),
    (.virt 13181, .wire 16 46),
    (.wire 2 27, .wire 16 48),
    (.wire 20 13, .wire 16 49),
    (.wire 16 47, .wire 16 50),
    (.wire 2 27, .wire 16 52),
    (.virt 13182, .wire 16 53),
    (.virt 13182, .wire 16 54),
    (.wire 2 27, .wire 16 56),
    (.wire 20 14, .wire 16 57),
    (.wire 16 55, .wire 16 58),
    (.wire 2 27, .wire 21 0),
    (.virt 13183, .wire 21 1),
    (.virt 13183, .wire 21 2),
    (.wire 2 27, .wire 21 4),
    (.wire 20 15, .wire 21 5),
    (.wire 21 3, .wire 21 6)
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
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 7
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 8
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 9
    ⟨.baseSum2 59, []⟩,  -- row 10
    ⟨.baseSum2 59, []⟩,  -- row 11
    ⟨.baseSum2 59, []⟩,  -- row 12
    ⟨.baseSum2 59, []⟩,  -- row 13
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 14
    ⟨.baseSum2 59, []⟩,  -- row 15
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 16
    ⟨.poseidon2, []⟩,  -- row 17
    ⟨.poseidon2, []⟩,  -- row 18
    ⟨.poseidon2, []⟩,  -- row 19
    ⟨.poseidon2, []⟩,  -- row 20
    ⟨.arithmetic 15, [1, (-1)]⟩  -- row 21
  ]
  copies := ((((privateBatchWrapper2.copies0 ++ privateBatchWrapper2.copies1) ++ (privateBatchWrapper2.copies2 ++ (privateBatchWrapper2.copies3 ++ privateBatchWrapper2.copies4))) ++ ((privateBatchWrapper2.copies5 ++ (privateBatchWrapper2.copies6 ++ privateBatchWrapper2.copies7)) ++ (privateBatchWrapper2.copies8 ++ (privateBatchWrapper2.copies9 ++ privateBatchWrapper2.copies10)))) ++ (((privateBatchWrapper2.copies11 ++ (privateBatchWrapper2.copies12 ++ privateBatchWrapper2.copies13)) ++ (privateBatchWrapper2.copies14 ++ (privateBatchWrapper2.copies15 ++ privateBatchWrapper2.copies16))) ++ ((privateBatchWrapper2.copies17 ++ (privateBatchWrapper2.copies18 ++ privateBatchWrapper2.copies19)) ++ (privateBatchWrapper2.copies20 ++ (privateBatchWrapper2.copies21 ++ privateBatchWrapper2.copies22)))))
  constants := [
    (.virt 13209, 1),
    (.virt 13210, 0),
    (.virt 13211, 2),
    (.virt 13244, 576460752303423488 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 13245, 10000)
  ]
  publicInputs := [
    .virt 13211,
    .virt 6584,
    .virt 13190,
    .wire 3 43,
    .wire 3 51,
    .wire 3 59,
    .wire 4 7,
    .wire 4 15,
    .virt 13195,
    .virt 13191,
    .virt 13192,
    .virt 13193,
    .virt 13194,
    .virt 13200,
    .virt 13196,
    .virt 13197,
    .virt 13198,
    .virt 13199,
    .wire 16 11,
    .wire 16 19,
    .wire 16 27,
    .wire 16 35,
    .wire 16 43,
    .wire 16 51,
    .wire 16 59,
    .wire 21 7
  ]

/-- Named targets `leaf_pis_0`. -/
def privateBatchWrapper2.leaf_pis_0 : Fin 11 → Target :=
  ![.virt 6584, .virt 6585, .virt 6586, .virt 6587, .virt 6588, .virt 6589, .virt 6590, .virt 6591, .virt 6592, .virt 6593, .virt 6594]

/-- Named targets `leaf_pis_1`. -/
def privateBatchWrapper2.leaf_pis_1 : Fin 11 → Target :=
  ![.virt 13179, .virt 13180, .virt 13181, .virt 13182, .virt 13183, .virt 13184, .virt 13185, .virt 13186, .virt 13187, .virt 13188, .virt 13189]

/-- Named target `volume_fee_bps`. -/
def privateBatchWrapper2.volume_fee_bps : Target := .virt 13190

/-- Named target `exit_amount_0`. -/
def privateBatchWrapper2.exit_amount_0 : Target := .virt 13195

/-- Named targets `exit_account_0`. -/
def privateBatchWrapper2.exit_account_0 : Fin 4 → Target :=
  ![.virt 13191, .virt 13192, .virt 13193, .virt 13194]

/-- Named target `exit_amount_1`. -/
def privateBatchWrapper2.exit_amount_1 : Target := .virt 13200

/-- Named targets `exit_account_1`. -/
def privateBatchWrapper2.exit_account_1 : Fin 4 → Target :=
  ![.virt 13196, .virt 13197, .virt 13198, .virt 13199]

/-- Named targets `dummy_pre_image_0`. -/
def privateBatchWrapper2.dummy_pre_image_0 : Fin 4 → Target :=
  ![.virt 13201, .virt 13202, .virt 13203, .virt 13204]

/-- Named targets `dummy_pre_image_1`. -/
def privateBatchWrapper2.dummy_pre_image_1 : Fin 4 → Target :=
  ![.virt 13205, .virt 13206, .virt 13207, .virt 13208]

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def privateBatchWrapper2.verifiers : List (String × List Target) :=
  [
    ("leaf_circuit", [.virt 6584, .virt 6585, .virt 6586, .virt 6587, .virt 6588, .virt 6589, .virt 6590, .virt 6591, .virt 6592, .virt 6593, .virt 6594]),
    ("leaf_circuit", [.virt 13179, .virt 13180, .virt 13181, .virt 13182, .virt 13183, .virt 13184, .virt 13185, .virt 13186, .virt 13187, .virt 13188, .virt 13189])
  ]

theorem privateBatchWrapper2_copies0 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13209) = a (.wire 0 0) ∧ a (.virt 13209) = a (.wire 0 1) ∧ a (.virt 13212) = a (.wire 0 2) ∧ a (.virt 13212) = a (.wire 1 0) ∧ a (.virt 6589) = a (.wire 1 1) ∧ a (.virt 13212) = a (.wire 1 2) ∧ a (.virt 6589) = a (.wire 1 4) ∧ a (.virt 13213) = a (.wire 1 5) ∧ a (.virt 6589) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 4) ∧ a (.virt 13209) = a (.wire 0 5) ∧ a (.wire 0 3) = a (.wire 0 6) ∧ a (.wire 1 3) = a (.virt 13210) ∧ a (.wire 0 7) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 0 8) ∧ a (.virt 13209) = a (.wire 0 9) ∧ a (.virt 13214) = a (.wire 0 10) ∧ a (.virt 13214) = a (.wire 1 8) ∧ a (.virt 6590) = a (.wire 1 9) ∧ a (.virt 13214) = a (.wire 1 10) ∧ a (.virt 6590) = a (.wire 1 12) ∧ a (.virt 13215) = a (.wire 1 13) ∧ a (.virt 6590) = a (.wire 1 14) ∧ a (.wire 1 15) = a (.wire 0 12) ∧ a (.virt 13209) = a (.wire 0 13) ∧ a (.wire 0 11) = a (.wire 0 14) ∧ a (.wire 1 11) = a (.virt 13210) ∧ a (.wire 0 15) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 0 16) ∧ a (.virt 13209) = a (.wire 0 17) ∧ a (.virt 13216) = a (.wire 0 18) ∧ a (.virt 13216) = a (.wire 1 16) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies1 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 6591) = a (.wire 1 17) ∧ a (.virt 13216) = a (.wire 1 18) ∧ a (.virt 6591) = a (.wire 1 20) ∧ a (.virt 13217) = a (.wire 1 21) ∧ a (.virt 6591) = a (.wire 1 22) ∧ a (.wire 1 23) = a (.wire 0 20) ∧ a (.virt 13209) = a (.wire 0 21) ∧ a (.wire 0 19) = a (.wire 0 22) ∧ a (.wire 1 19) = a (.virt 13210) ∧ a (.wire 0 23) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 0 24) ∧ a (.virt 13209) = a (.wire 0 25) ∧ a (.virt 13218) = a (.wire 0 26) ∧ a (.virt 13218) = a (.wire 1 24) ∧ a (.virt 6592) = a (.wire 1 25) ∧ a (.virt 13218) = a (.wire 1 26) ∧ a (.virt 6592) = a (.wire 1 28) ∧ a (.virt 13219) = a (.wire 1 29) ∧ a (.virt 6592) = a (.wire 1 30) ∧ a (.wire 1 31) = a (.wire 0 28) ∧ a (.virt 13209) = a (.wire 0 29) ∧ a (.wire 0 27) = a (.wire 0 30) ∧ a (.wire 1 27) = a (.virt 13210) ∧ a (.wire 0 31) = a (.virt 13210) ∧ a (.virt 13212) = a (.wire 1 32) ∧ a (.virt 13214) = a (.wire 1 33) ∧ a (.virt 13212) = a (.wire 1 34) ∧ a (.virt 13216) = a (.wire 1 36) ∧ a (.virt 13218) = a (.wire 1 37) ∧ a (.virt 13216) = a (.wire 1 38) ∧ a (.wire 1 35) = a (.wire 1 40) ∧ a (.wire 1 39) = a (.wire 1 41) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ hq))))
  simp only [privateBatchWrapper2.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies2 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 1 35) = a (.wire 1 42) ∧ a (.virt 13209) = a (.wire 0 32) ∧ a (.virt 13209) = a (.wire 0 33) ∧ a (.virt 13220) = a (.wire 0 34) ∧ a (.virt 13220) = a (.wire 1 44) ∧ a (.virt 13184) = a (.wire 1 45) ∧ a (.virt 13220) = a (.wire 1 46) ∧ a (.virt 13184) = a (.wire 1 48) ∧ a (.virt 13221) = a (.wire 1 49) ∧ a (.virt 13184) = a (.wire 1 50) ∧ a (.wire 1 51) = a (.wire 0 36) ∧ a (.virt 13209) = a (.wire 0 37) ∧ a (.wire 0 35) = a (.wire 0 38) ∧ a (.wire 1 47) = a (.virt 13210) ∧ a (.wire 0 39) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 0 40) ∧ a (.virt 13209) = a (.wire 0 41) ∧ a (.virt 13222) = a (.wire 0 42) ∧ a (.virt 13222) = a (.wire 1 52) ∧ a (.virt 13185) = a (.wire 1 53) ∧ a (.virt 13222) = a (.wire 1 54) ∧ a (.virt 13185) = a (.wire 1 56) ∧ a (.virt 13223) = a (.wire 1 57) ∧ a (.virt 13185) = a (.wire 1 58) ∧ a (.wire 1 59) = a (.wire 0 44) ∧ a (.virt 13209) = a (.wire 0 45) ∧ a (.wire 0 43) = a (.wire 0 46) ∧ a (.wire 1 55) = a (.virt 13210) ∧ a (.wire 0 47) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 0 48) ∧ a (.virt 13209) = a (.wire 0 49) ∧ a (.virt 13224) = a (.wire 0 50) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies3 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13224) = a (.wire 2 0) ∧ a (.virt 13186) = a (.wire 2 1) ∧ a (.virt 13224) = a (.wire 2 2) ∧ a (.virt 13186) = a (.wire 2 4) ∧ a (.virt 13225) = a (.wire 2 5) ∧ a (.virt 13186) = a (.wire 2 6) ∧ a (.wire 2 7) = a (.wire 0 52) ∧ a (.virt 13209) = a (.wire 0 53) ∧ a (.wire 0 51) = a (.wire 0 54) ∧ a (.wire 2 3) = a (.virt 13210) ∧ a (.wire 0 55) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 0 56) ∧ a (.virt 13209) = a (.wire 0 57) ∧ a (.virt 13226) = a (.wire 0 58) ∧ a (.virt 13226) = a (.wire 2 8) ∧ a (.virt 13187) = a (.wire 2 9) ∧ a (.virt 13226) = a (.wire 2 10) ∧ a (.virt 13187) = a (.wire 2 12) ∧ a (.virt 13227) = a (.wire 2 13) ∧ a (.virt 13187) = a (.wire 2 14) ∧ a (.wire 2 15) = a (.wire 3 0) ∧ a (.virt 13209) = a (.wire 3 1) ∧ a (.wire 0 59) = a (.wire 3 2) ∧ a (.wire 2 11) = a (.virt 13210) ∧ a (.wire 3 3) = a (.virt 13210) ∧ a (.virt 13220) = a (.wire 2 16) ∧ a (.virt 13222) = a (.wire 2 17) ∧ a (.virt 13220) = a (.wire 2 18) ∧ a (.virt 13224) = a (.wire 2 20) ∧ a (.virt 13226) = a (.wire 2 21) ∧ a (.virt 13224) = a (.wire 2 22) ∧ a (.wire 2 19) = a (.wire 2 24) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies3, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies4 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 23) = a (.wire 2 25) ∧ a (.wire 2 19) = a (.wire 2 26) ∧ a (.virt 13209) = a (.wire 3 4) ∧ a (.virt 13209) = a (.wire 3 5) ∧ a (.wire 1 43) = a (.wire 3 6) ∧ a (.wire 3 7) = a (.wire 3 8) ∧ a (.virt 6589) = a (.wire 3 9) ∧ a (.virt 13210) = a (.wire 3 10) ∧ a (.wire 3 7) = a (.wire 3 12) ∧ a (.virt 6590) = a (.wire 3 13) ∧ a (.virt 13210) = a (.wire 3 14) ∧ a (.wire 3 7) = a (.wire 3 16) ∧ a (.virt 6591) = a (.wire 3 17) ∧ a (.virt 13210) = a (.wire 3 18) ∧ a (.wire 3 7) = a (.wire 3 20) ∧ a (.virt 6592) = a (.wire 3 21) ∧ a (.virt 13210) = a (.wire 3 22) ∧ a (.wire 3 7) = a (.wire 3 24) ∧ a (.virt 6593) = a (.wire 3 25) ∧ a (.virt 13210) = a (.wire 3 26) ∧ a (.virt 13209) = a (.wire 3 28) ∧ a (.virt 13209) = a (.wire 3 29) ∧ a (.wire 2 27) = a (.wire 3 30) ∧ a (.virt 13209) = a (.wire 3 32) ∧ a (.virt 13209) = a (.wire 3 33) ∧ a (.wire 3 7) = a (.wire 3 34) ∧ a (.wire 3 31) = a (.wire 2 28) ∧ a (.wire 3 35) = a (.wire 2 29) ∧ a (.wire 3 31) = a (.wire 2 30) ∧ a (.wire 2 31) = a (.wire 3 36) ∧ a (.wire 3 11) = a (.wire 3 37) ∧ a (.wire 3 11) = a (.wire 3 38) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies4, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies5 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 2 31) = a (.wire 3 40) ∧ a (.virt 13184) = a (.wire 3 41) ∧ a (.wire 3 39) = a (.wire 3 42) ∧ a (.wire 2 31) = a (.wire 3 44) ∧ a (.wire 3 15) = a (.wire 3 45) ∧ a (.wire 3 15) = a (.wire 3 46) ∧ a (.wire 2 31) = a (.wire 3 48) ∧ a (.virt 13185) = a (.wire 3 49) ∧ a (.wire 3 47) = a (.wire 3 50) ∧ a (.wire 2 31) = a (.wire 3 52) ∧ a (.wire 3 19) = a (.wire 3 53) ∧ a (.wire 3 19) = a (.wire 3 54) ∧ a (.wire 2 31) = a (.wire 3 56) ∧ a (.virt 13186) = a (.wire 3 57) ∧ a (.wire 3 55) = a (.wire 3 58) ∧ a (.wire 2 31) = a (.wire 4 0) ∧ a (.wire 3 23) = a (.wire 4 1) ∧ a (.wire 3 23) = a (.wire 4 2) ∧ a (.wire 2 31) = a (.wire 4 4) ∧ a (.virt 13187) = a (.wire 4 5) ∧ a (.wire 4 3) = a (.wire 4 6) ∧ a (.wire 2 31) = a (.wire 4 8) ∧ a (.wire 3 27) = a (.wire 4 9) ∧ a (.wire 3 27) = a (.wire 4 10) ∧ a (.wire 2 31) = a (.wire 4 12) ∧ a (.virt 13188) = a (.wire 4 13) ∧ a (.wire 4 11) = a (.wire 4 14) ∧ a (.wire 3 7) = a (.wire 5 0) ∧ a (.wire 3 31) = a (.wire 5 1) ∧ a (.wire 3 7) = a (.wire 5 2) ∧ a (.wire 5 3) = a (.wire 6 0) ∧ a (.virt 13209) = a (.wire 6 1) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies5, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies6 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = a (.wire 6 2) ∧ a (.virt 13209) = a (.wire 4 16) ∧ a (.virt 13209) = a (.wire 4 17) ∧ a (.virt 13228) = a (.wire 4 18) ∧ a (.virt 6589) = a (.wire 4 20) ∧ a (.virt 13209) = a (.wire 4 21) ∧ a (.wire 3 43) = a (.wire 4 22) ∧ a (.virt 13228) = a (.wire 2 32) ∧ a (.wire 4 23) = a (.wire 2 33) ∧ a (.virt 13228) = a (.wire 2 34) ∧ a (.wire 4 23) = a (.wire 2 36) ∧ a (.virt 13229) = a (.wire 2 37) ∧ a (.wire 4 23) = a (.wire 2 38) ∧ a (.wire 2 39) = a (.wire 4 24) ∧ a (.virt 13209) = a (.wire 4 25) ∧ a (.wire 4 19) = a (.wire 4 26) ∧ a (.wire 2 35) = a (.virt 13210) ∧ a (.wire 4 27) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 4 28) ∧ a (.virt 13209) = a (.wire 4 29) ∧ a (.virt 13230) = a (.wire 4 30) ∧ a (.virt 6590) = a (.wire 4 32) ∧ a (.virt 13209) = a (.wire 4 33) ∧ a (.wire 3 51) = a (.wire 4 34) ∧ a (.virt 13230) = a (.wire 2 40) ∧ a (.wire 4 35) = a (.wire 2 41) ∧ a (.virt 13230) = a (.wire 2 42) ∧ a (.wire 4 35) = a (.wire 2 44) ∧ a (.virt 13231) = a (.wire 2 45) ∧ a (.wire 4 35) = a (.wire 2 46) ∧ a (.wire 2 47) = a (.wire 4 36) ∧ a (.virt 13209) = a (.wire 4 37) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies6, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies7 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 31) = a (.wire 4 38) ∧ a (.wire 2 43) = a (.virt 13210) ∧ a (.wire 4 39) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 4 40) ∧ a (.virt 13209) = a (.wire 4 41) ∧ a (.virt 13232) = a (.wire 4 42) ∧ a (.virt 6591) = a (.wire 4 44) ∧ a (.virt 13209) = a (.wire 4 45) ∧ a (.wire 3 59) = a (.wire 4 46) ∧ a (.virt 13232) = a (.wire 2 48) ∧ a (.wire 4 47) = a (.wire 2 49) ∧ a (.virt 13232) = a (.wire 2 50) ∧ a (.wire 4 47) = a (.wire 2 52) ∧ a (.virt 13233) = a (.wire 2 53) ∧ a (.wire 4 47) = a (.wire 2 54) ∧ a (.wire 2 55) = a (.wire 4 48) ∧ a (.virt 13209) = a (.wire 4 49) ∧ a (.wire 4 43) = a (.wire 4 50) ∧ a (.wire 2 51) = a (.virt 13210) ∧ a (.wire 4 51) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 4 52) ∧ a (.virt 13209) = a (.wire 4 53) ∧ a (.virt 13234) = a (.wire 4 54) ∧ a (.virt 6592) = a (.wire 4 56) ∧ a (.virt 13209) = a (.wire 4 57) ∧ a (.wire 4 7) = a (.wire 4 58) ∧ a (.virt 13234) = a (.wire 2 56) ∧ a (.wire 4 59) = a (.wire 2 57) ∧ a (.virt 13234) = a (.wire 2 58) ∧ a (.wire 4 59) = a (.wire 7 0) ∧ a (.virt 13235) = a (.wire 7 1) ∧ a (.wire 4 59) = a (.wire 7 2) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies7, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies8 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 3) = a (.wire 8 0) ∧ a (.virt 13209) = a (.wire 8 1) ∧ a (.wire 4 55) = a (.wire 8 2) ∧ a (.wire 2 59) = a (.virt 13210) ∧ a (.wire 8 3) = a (.virt 13210) ∧ a (.virt 13228) = a (.wire 7 4) ∧ a (.virt 13230) = a (.wire 7 5) ∧ a (.virt 13228) = a (.wire 7 6) ∧ a (.virt 13232) = a (.wire 7 8) ∧ a (.virt 13234) = a (.wire 7 9) ∧ a (.virt 13232) = a (.wire 7 10) ∧ a (.wire 7 7) = a (.wire 7 12) ∧ a (.wire 7 11) = a (.wire 7 13) ∧ a (.wire 7 7) = a (.wire 7 14) ∧ a (.wire 1 43) = a (.wire 5 4) ∧ a (.wire 7 15) = a (.wire 5 5) ∧ a (.wire 1 43) = a (.wire 5 6) ∧ a (.wire 5 7) = a (.wire 6 4) ∧ a (.virt 13209) = a (.wire 6 5) ∧ a (.wire 7 15) = a (.wire 6 6) ∧ a (.wire 6 7) = a (.virt 13209) ∧ True ∧ a (.virt 13209) = a (.wire 8 4) ∧ a (.virt 13209) = a (.wire 8 5) ∧ a (.virt 13236) = a (.wire 8 6) ∧ a (.virt 13184) = a (.wire 8 8) ∧ a (.virt 13209) = a (.wire 8 9) ∧ a (.wire 3 43) = a (.wire 8 10) ∧ a (.virt 13236) = a (.wire 7 16) ∧ a (.wire 8 11) = a (.wire 7 17) ∧ a (.virt 13236) = a (.wire 7 18) ∧ a (.wire 8 11) = a (.wire 7 20) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies8, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies9 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13237) = a (.wire 7 21) ∧ a (.wire 8 11) = a (.wire 7 22) ∧ a (.wire 7 23) = a (.wire 8 12) ∧ a (.virt 13209) = a (.wire 8 13) ∧ a (.wire 8 7) = a (.wire 8 14) ∧ a (.wire 7 19) = a (.virt 13210) ∧ a (.wire 8 15) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 8 16) ∧ a (.virt 13209) = a (.wire 8 17) ∧ a (.virt 13238) = a (.wire 8 18) ∧ a (.virt 13185) = a (.wire 8 20) ∧ a (.virt 13209) = a (.wire 8 21) ∧ a (.wire 3 51) = a (.wire 8 22) ∧ a (.virt 13238) = a (.wire 7 24) ∧ a (.wire 8 23) = a (.wire 7 25) ∧ a (.virt 13238) = a (.wire 7 26) ∧ a (.wire 8 23) = a (.wire 7 28) ∧ a (.virt 13239) = a (.wire 7 29) ∧ a (.wire 8 23) = a (.wire 7 30) ∧ a (.wire 7 31) = a (.wire 8 24) ∧ a (.virt 13209) = a (.wire 8 25) ∧ a (.wire 8 19) = a (.wire 8 26) ∧ a (.wire 7 27) = a (.virt 13210) ∧ a (.wire 8 27) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 8 28) ∧ a (.virt 13209) = a (.wire 8 29) ∧ a (.virt 13240) = a (.wire 8 30) ∧ a (.virt 13186) = a (.wire 8 32) ∧ a (.virt 13209) = a (.wire 8 33) ∧ a (.wire 3 59) = a (.wire 8 34) ∧ a (.virt 13240) = a (.wire 7 32) ∧ a (.wire 8 35) = a (.wire 7 33) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies9, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies10 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13240) = a (.wire 7 34) ∧ a (.wire 8 35) = a (.wire 7 36) ∧ a (.virt 13241) = a (.wire 7 37) ∧ a (.wire 8 35) = a (.wire 7 38) ∧ a (.wire 7 39) = a (.wire 8 36) ∧ a (.virt 13209) = a (.wire 8 37) ∧ a (.wire 8 31) = a (.wire 8 38) ∧ a (.wire 7 35) = a (.virt 13210) ∧ a (.wire 8 39) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 8 40) ∧ a (.virt 13209) = a (.wire 8 41) ∧ a (.virt 13242) = a (.wire 8 42) ∧ a (.virt 13187) = a (.wire 8 44) ∧ a (.virt 13209) = a (.wire 8 45) ∧ a (.wire 4 7) = a (.wire 8 46) ∧ a (.virt 13242) = a (.wire 7 40) ∧ a (.wire 8 47) = a (.wire 7 41) ∧ a (.virt 13242) = a (.wire 7 42) ∧ a (.wire 8 47) = a (.wire 7 44) ∧ a (.virt 13243) = a (.wire 7 45) ∧ a (.wire 8 47) = a (.wire 7 46) ∧ a (.wire 7 47) = a (.wire 8 48) ∧ a (.virt 13209) = a (.wire 8 49) ∧ a (.wire 8 43) = a (.wire 8 50) ∧ a (.wire 7 43) = a (.virt 13210) ∧ a (.wire 8 51) = a (.virt 13210) ∧ a (.virt 13236) = a (.wire 7 48) ∧ a (.virt 13238) = a (.wire 7 49) ∧ a (.virt 13236) = a (.wire 7 50) ∧ a (.virt 13240) = a (.wire 7 52) ∧ a (.virt 13242) = a (.wire 7 53) ∧ a (.virt 13240) = a (.wire 7 54) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies10, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies11 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 51) = a (.wire 7 56) ∧ a (.wire 7 55) = a (.wire 7 57) ∧ a (.wire 7 51) = a (.wire 7 58) ∧ a (.wire 2 27) = a (.wire 5 8) ∧ a (.wire 7 59) = a (.wire 5 9) ∧ a (.wire 2 27) = a (.wire 5 10) ∧ a (.wire 5 11) = a (.wire 6 8) ∧ a (.virt 13209) = a (.wire 6 9) ∧ a (.wire 7 59) = a (.wire 6 10) ∧ a (.wire 6 11) = a (.virt 13209) ∧ a (.virt 13179) = a (.virt 6584) ∧ a (.wire 1 43) = a (.wire 8 52) ∧ a (.virt 6594) = a (.wire 8 53) ∧ a (.virt 6594) = a (.wire 8 54) ∧ a (.wire 1 43) = a (.wire 8 56) ∧ a (.virt 13210) = a (.wire 8 57) ∧ a (.wire 8 55) = a (.wire 8 58) ∧ a (.wire 2 27) = a (.wire 9 0) ∧ a (.virt 13189) = a (.wire 9 1) ∧ a (.virt 13189) = a (.wire 9 2) ∧ a (.wire 2 27) = a (.wire 9 4) ∧ a (.virt 13210) = a (.wire 9 5) ∧ a (.wire 9 3) = a (.wire 9 6) ∧ a (.wire 8 59) = a (.wire 6 12) ∧ a (.virt 13209) = a (.wire 6 13) ∧ a (.wire 9 7) = a (.wire 6 14) ∧ a (.wire 10 33) = a (.virt 13210) ∧ a (.wire 10 34) = a (.virt 13210) ∧ a (.wire 10 35) = a (.virt 13210) ∧ a (.wire 10 36) = a (.virt 13210) ∧ a (.wire 10 37) = a (.virt 13210) ∧ a (.wire 10 38) = a (.virt 13210) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies11, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies12 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 10 39) = a (.virt 13210) ∧ a (.wire 10 40) = a (.virt 13210) ∧ a (.wire 10 41) = a (.virt 13210) ∧ a (.wire 10 42) = a (.virt 13210) ∧ a (.wire 10 43) = a (.virt 13210) ∧ a (.wire 10 44) = a (.virt 13210) ∧ a (.wire 10 45) = a (.virt 13210) ∧ a (.wire 10 46) = a (.virt 13210) ∧ a (.wire 10 47) = a (.virt 13210) ∧ a (.wire 10 48) = a (.virt 13210) ∧ a (.wire 10 49) = a (.virt 13210) ∧ a (.wire 10 50) = a (.virt 13210) ∧ a (.wire 10 51) = a (.virt 13210) ∧ a (.wire 10 52) = a (.virt 13210) ∧ a (.wire 10 53) = a (.virt 13210) ∧ a (.wire 10 54) = a (.virt 13210) ∧ a (.wire 10 55) = a (.virt 13210) ∧ a (.wire 10 56) = a (.virt 13210) ∧ a (.wire 10 57) = a (.virt 13210) ∧ a (.wire 10 58) = a (.virt 13210) ∧ a (.wire 10 59) = a (.virt 13210) ∧ a (.wire 10 0) = a (.virt 13195) ∧ a (.wire 11 33) = a (.virt 13210) ∧ a (.wire 11 34) = a (.virt 13210) ∧ a (.wire 11 35) = a (.virt 13210) ∧ a (.wire 11 36) = a (.virt 13210) ∧ a (.wire 11 37) = a (.virt 13210) ∧ a (.wire 11 38) = a (.virt 13210) ∧ a (.wire 11 39) = a (.virt 13210) ∧ a (.wire 11 40) = a (.virt 13210) ∧ a (.wire 11 41) = a (.virt 13210) ∧ a (.wire 11 42) = a (.virt 13210) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies12, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies13 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 11 43) = a (.virt 13210) ∧ a (.wire 11 44) = a (.virt 13210) ∧ a (.wire 11 45) = a (.virt 13210) ∧ a (.wire 11 46) = a (.virt 13210) ∧ a (.wire 11 47) = a (.virt 13210) ∧ a (.wire 11 48) = a (.virt 13210) ∧ a (.wire 11 49) = a (.virt 13210) ∧ a (.wire 11 50) = a (.virt 13210) ∧ a (.wire 11 51) = a (.virt 13210) ∧ a (.wire 11 52) = a (.virt 13210) ∧ a (.wire 11 53) = a (.virt 13210) ∧ a (.wire 11 54) = a (.virt 13210) ∧ a (.wire 11 55) = a (.virt 13210) ∧ a (.wire 11 56) = a (.virt 13210) ∧ a (.wire 11 57) = a (.virt 13210) ∧ a (.wire 11 58) = a (.virt 13210) ∧ a (.wire 11 59) = a (.virt 13210) ∧ a (.wire 11 0) = a (.virt 13200) ∧ a (.virt 13195) = a (.wire 6 16) ∧ a (.virt 13209) = a (.wire 6 17) ∧ a (.virt 13200) = a (.wire 6 18) ∧ a (.wire 12 15) = a (.virt 13210) ∧ a (.wire 12 16) = a (.virt 13210) ∧ a (.wire 12 17) = a (.virt 13210) ∧ a (.wire 12 18) = a (.virt 13210) ∧ a (.wire 12 19) = a (.virt 13210) ∧ a (.wire 12 20) = a (.virt 13210) ∧ a (.wire 12 21) = a (.virt 13210) ∧ a (.wire 12 22) = a (.virt 13210) ∧ a (.wire 12 23) = a (.virt 13210) ∧ a (.wire 12 24) = a (.virt 13210) ∧ a (.wire 12 25) = a (.virt 13210) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies13, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies14 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 26) = a (.virt 13210) ∧ a (.wire 12 27) = a (.virt 13210) ∧ a (.wire 12 28) = a (.virt 13210) ∧ a (.wire 12 29) = a (.virt 13210) ∧ a (.wire 12 30) = a (.virt 13210) ∧ a (.wire 12 31) = a (.virt 13210) ∧ a (.wire 12 32) = a (.virt 13210) ∧ a (.wire 12 33) = a (.virt 13210) ∧ a (.wire 12 34) = a (.virt 13210) ∧ a (.wire 12 35) = a (.virt 13210) ∧ a (.wire 12 36) = a (.virt 13210) ∧ a (.wire 12 37) = a (.virt 13210) ∧ a (.wire 12 38) = a (.virt 13210) ∧ a (.wire 12 39) = a (.virt 13210) ∧ a (.wire 12 40) = a (.virt 13210) ∧ a (.wire 12 41) = a (.virt 13210) ∧ a (.wire 12 42) = a (.virt 13210) ∧ a (.wire 12 43) = a (.virt 13210) ∧ a (.wire 12 44) = a (.virt 13210) ∧ a (.wire 12 45) = a (.virt 13210) ∧ a (.wire 12 46) = a (.virt 13210) ∧ a (.wire 12 47) = a (.virt 13210) ∧ a (.wire 12 48) = a (.virt 13210) ∧ a (.wire 12 49) = a (.virt 13210) ∧ a (.wire 12 50) = a (.virt 13210) ∧ a (.wire 12 51) = a (.virt 13210) ∧ a (.wire 12 52) = a (.virt 13210) ∧ a (.wire 12 53) = a (.virt 13210) ∧ a (.wire 12 54) = a (.virt 13210) ∧ a (.wire 12 55) = a (.virt 13210) ∧ a (.wire 12 56) = a (.virt 13210) ∧ a (.wire 12 57) = a (.virt 13210) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies14, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies15 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 12 58) = a (.virt 13210) ∧ a (.wire 12 59) = a (.virt 13210) ∧ a (.wire 12 0) = a (.virt 13190) ∧ a (.virt 13245) = a (.wire 9 8) ∧ a (.virt 13209) = a (.wire 9 9) ∧ a (.virt 13190) = a (.wire 9 10) ∧ a (.wire 13 15) = a (.virt 13210) ∧ a (.wire 13 16) = a (.virt 13210) ∧ a (.wire 13 17) = a (.virt 13210) ∧ a (.wire 13 18) = a (.virt 13210) ∧ a (.wire 13 19) = a (.virt 13210) ∧ a (.wire 13 20) = a (.virt 13210) ∧ a (.wire 13 21) = a (.virt 13210) ∧ a (.wire 13 22) = a (.virt 13210) ∧ a (.wire 13 23) = a (.virt 13210) ∧ a (.wire 13 24) = a (.virt 13210) ∧ a (.wire 13 25) = a (.virt 13210) ∧ a (.wire 13 26) = a (.virt 13210) ∧ a (.wire 13 27) = a (.virt 13210) ∧ a (.wire 13 28) = a (.virt 13210) ∧ a (.wire 13 29) = a (.virt 13210) ∧ a (.wire 13 30) = a (.virt 13210) ∧ a (.wire 13 31) = a (.virt 13210) ∧ a (.wire 13 32) = a (.virt 13210) ∧ a (.wire 13 33) = a (.virt 13210) ∧ a (.wire 13 34) = a (.virt 13210) ∧ a (.wire 13 35) = a (.virt 13210) ∧ a (.wire 13 36) = a (.virt 13210) ∧ a (.wire 13 37) = a (.virt 13210) ∧ a (.wire 13 38) = a (.virt 13210) ∧ a (.wire 13 39) = a (.virt 13210) ∧ a (.wire 13 40) = a (.virt 13210) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies15, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies16 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 13 41) = a (.virt 13210) ∧ a (.wire 13 42) = a (.virt 13210) ∧ a (.wire 13 43) = a (.virt 13210) ∧ a (.wire 13 44) = a (.virt 13210) ∧ a (.wire 13 45) = a (.virt 13210) ∧ a (.wire 13 46) = a (.virt 13210) ∧ a (.wire 13 47) = a (.virt 13210) ∧ a (.wire 13 48) = a (.virt 13210) ∧ a (.wire 13 49) = a (.virt 13210) ∧ a (.wire 13 50) = a (.virt 13210) ∧ a (.wire 13 51) = a (.virt 13210) ∧ a (.wire 13 52) = a (.virt 13210) ∧ a (.wire 13 53) = a (.virt 13210) ∧ a (.wire 13 54) = a (.virt 13210) ∧ a (.wire 13 55) = a (.virt 13210) ∧ a (.wire 13 56) = a (.virt 13210) ∧ a (.wire 13 57) = a (.virt 13210) ∧ a (.wire 13 58) = a (.virt 13210) ∧ a (.wire 13 59) = a (.virt 13210) ∧ a (.wire 13 0) = a (.wire 9 11) ∧ a (.wire 6 19) = a (.wire 14 0) ∧ a (.virt 13245) = a (.wire 14 1) ∧ a (.wire 6 19) = a (.wire 14 2) ∧ a (.wire 6 15) = a (.wire 14 4) ∧ a (.wire 9 11) = a (.wire 14 5) ∧ a (.wire 6 15) = a (.wire 14 6) ∧ a (.wire 14 7) = a (.wire 9 12) ∧ a (.virt 13209) = a (.wire 9 13) ∧ a (.wire 14 3) = a (.wire 9 14) ∧ a (.wire 15 53) = a (.virt 13210) ∧ a (.wire 15 54) = a (.virt 13210) ∧ a (.wire 15 55) = a (.virt 13210) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies16, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies17 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 15 56) = a (.virt 13210) ∧ a (.wire 15 57) = a (.virt 13210) ∧ a (.wire 15 58) = a (.virt 13210) ∧ a (.wire 15 59) = a (.virt 13210) ∧ a (.wire 15 0) = a (.wire 9 15) ∧ a (.wire 3 7) = a (.wire 14 8) ∧ a (.wire 3 31) = a (.wire 14 9) ∧ a (.wire 3 7) = a (.wire 14 10) ∧ a (.virt 13209) = a (.wire 9 16) ∧ a (.virt 13209) = a (.wire 9 17) ∧ a (.virt 13246) = a (.wire 9 18) ∧ a (.virt 6585) = a (.wire 9 20) ∧ a (.virt 13209) = a (.wire 9 21) ∧ a (.virt 13180) = a (.wire 9 22) ∧ a (.virt 13246) = a (.wire 14 12) ∧ a (.wire 9 23) = a (.wire 14 13) ∧ a (.virt 13246) = a (.wire 14 14) ∧ a (.wire 9 23) = a (.wire 14 16) ∧ a (.virt 13247) = a (.wire 14 17) ∧ a (.wire 9 23) = a (.wire 14 18) ∧ a (.wire 14 19) = a (.wire 9 24) ∧ a (.virt 13209) = a (.wire 9 25) ∧ a (.wire 9 19) = a (.wire 9 26) ∧ a (.wire 14 15) = a (.virt 13210) ∧ a (.wire 9 27) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 9 28) ∧ a (.virt 13209) = a (.wire 9 29) ∧ a (.virt 13248) = a (.wire 9 30) ∧ a (.virt 6586) = a (.wire 9 32) ∧ a (.virt 13209) = a (.wire 9 33) ∧ a (.virt 13181) = a (.wire 9 34) ∧ a (.virt 13248) = a (.wire 14 20) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies17, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies18 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 35) = a (.wire 14 21) ∧ a (.virt 13248) = a (.wire 14 22) ∧ a (.wire 9 35) = a (.wire 14 24) ∧ a (.virt 13249) = a (.wire 14 25) ∧ a (.wire 9 35) = a (.wire 14 26) ∧ a (.wire 14 27) = a (.wire 9 36) ∧ a (.virt 13209) = a (.wire 9 37) ∧ a (.wire 9 31) = a (.wire 9 38) ∧ a (.wire 14 23) = a (.virt 13210) ∧ a (.wire 9 39) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 9 40) ∧ a (.virt 13209) = a (.wire 9 41) ∧ a (.virt 13250) = a (.wire 9 42) ∧ a (.virt 6587) = a (.wire 9 44) ∧ a (.virt 13209) = a (.wire 9 45) ∧ a (.virt 13182) = a (.wire 9 46) ∧ a (.virt 13250) = a (.wire 14 28) ∧ a (.wire 9 47) = a (.wire 14 29) ∧ a (.virt 13250) = a (.wire 14 30) ∧ a (.wire 9 47) = a (.wire 14 32) ∧ a (.virt 13251) = a (.wire 14 33) ∧ a (.wire 9 47) = a (.wire 14 34) ∧ a (.wire 14 35) = a (.wire 9 48) ∧ a (.virt 13209) = a (.wire 9 49) ∧ a (.wire 9 43) = a (.wire 9 50) ∧ a (.wire 14 31) = a (.virt 13210) ∧ a (.wire 9 51) = a (.virt 13210) ∧ a (.virt 13209) = a (.wire 9 52) ∧ a (.virt 13209) = a (.wire 9 53) ∧ a (.virt 13252) = a (.wire 9 54) ∧ a (.virt 6588) = a (.wire 9 56) ∧ a (.virt 13209) = a (.wire 9 57) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies18, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies19 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13183) = a (.wire 9 58) ∧ a (.virt 13252) = a (.wire 14 36) ∧ a (.wire 9 59) = a (.wire 14 37) ∧ a (.virt 13252) = a (.wire 14 38) ∧ a (.wire 9 59) = a (.wire 14 40) ∧ a (.virt 13253) = a (.wire 14 41) ∧ a (.wire 9 59) = a (.wire 14 42) ∧ a (.wire 14 43) = a (.wire 16 0) ∧ a (.virt 13209) = a (.wire 16 1) ∧ a (.wire 9 55) = a (.wire 16 2) ∧ a (.wire 14 39) = a (.virt 13210) ∧ a (.wire 16 3) = a (.virt 13210) ∧ a (.virt 13246) = a (.wire 14 44) ∧ a (.virt 13248) = a (.wire 14 45) ∧ a (.virt 13246) = a (.wire 14 46) ∧ a (.virt 13250) = a (.wire 14 48) ∧ a (.virt 13252) = a (.wire 14 49) ∧ a (.virt 13250) = a (.wire 14 50) ∧ a (.wire 14 47) = a (.wire 14 52) ∧ a (.wire 14 51) = a (.wire 14 53) ∧ a (.wire 14 47) = a (.wire 14 54) ∧ a (.wire 14 11) = a (.wire 14 56) ∧ a (.wire 14 55) = a (.wire 14 57) ∧ a (.wire 14 11) = a (.wire 14 58) ∧ a (.wire 14 59) = a (.virt 13210) ∧ a (.virt 13201) = a (.wire 17 0) ∧ a (.virt 13202) = a (.wire 17 1) ∧ a (.virt 13203) = a (.wire 17 2) ∧ a (.virt 13204) = a (.wire 17 3) ∧ a (.virt 13209) = a (.wire 17 4) ∧ a (.virt 13210) = a (.wire 17 5) ∧ a (.virt 13210) = a (.wire 17 6) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies19, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies20 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13210) = a (.wire 17 7) ∧ a (.virt 13210) = a (.wire 17 8) ∧ a (.virt 13210) = a (.wire 17 9) ∧ a (.virt 13210) = a (.wire 17 10) ∧ a (.virt 13210) = a (.wire 17 11) ∧ a (.wire 17 12) = a (.wire 18 0) ∧ a (.wire 17 13) = a (.wire 18 1) ∧ a (.wire 17 14) = a (.wire 18 2) ∧ a (.wire 17 15) = a (.wire 18 3) ∧ a (.virt 13209) = a (.wire 18 4) ∧ a (.virt 13210) = a (.wire 18 5) ∧ a (.virt 13210) = a (.wire 18 6) ∧ a (.virt 13210) = a (.wire 18 7) ∧ a (.virt 13210) = a (.wire 18 8) ∧ a (.virt 13210) = a (.wire 18 9) ∧ a (.virt 13210) = a (.wire 18 10) ∧ a (.virt 13210) = a (.wire 18 11) ∧ a (.wire 1 43) = a (.wire 16 4) ∧ a (.virt 6585) = a (.wire 16 5) ∧ a (.virt 6585) = a (.wire 16 6) ∧ a (.wire 1 43) = a (.wire 16 8) ∧ a (.wire 18 12) = a (.wire 16 9) ∧ a (.wire 16 7) = a (.wire 16 10) ∧ a (.wire 1 43) = a (.wire 16 12) ∧ a (.virt 6586) = a (.wire 16 13) ∧ a (.virt 6586) = a (.wire 16 14) ∧ a (.wire 1 43) = a (.wire 16 16) ∧ a (.wire 18 13) = a (.wire 16 17) ∧ a (.wire 16 15) = a (.wire 16 18) ∧ a (.wire 1 43) = a (.wire 16 20) ∧ a (.virt 6587) = a (.wire 16 21) ∧ a (.virt 6587) = a (.wire 16 22) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies20, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))
  simp only [privateBatchWrapper2.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies21 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 1 43) = a (.wire 16 24) ∧ a (.wire 18 14) = a (.wire 16 25) ∧ a (.wire 16 23) = a (.wire 16 26) ∧ a (.wire 1 43) = a (.wire 16 28) ∧ a (.virt 6588) = a (.wire 16 29) ∧ a (.virt 6588) = a (.wire 16 30) ∧ a (.wire 1 43) = a (.wire 16 32) ∧ a (.wire 18 15) = a (.wire 16 33) ∧ a (.wire 16 31) = a (.wire 16 34) ∧ a (.virt 13205) = a (.wire 19 0) ∧ a (.virt 13206) = a (.wire 19 1) ∧ a (.virt 13207) = a (.wire 19 2) ∧ a (.virt 13208) = a (.wire 19 3) ∧ a (.virt 13209) = a (.wire 19 4) ∧ a (.virt 13210) = a (.wire 19 5) ∧ a (.virt 13210) = a (.wire 19 6) ∧ a (.virt 13210) = a (.wire 19 7) ∧ a (.virt 13210) = a (.wire 19 8) ∧ a (.virt 13210) = a (.wire 19 9) ∧ a (.virt 13210) = a (.wire 19 10) ∧ a (.virt 13210) = a (.wire 19 11) ∧ a (.wire 19 12) = a (.wire 20 0) ∧ a (.wire 19 13) = a (.wire 20 1) ∧ a (.wire 19 14) = a (.wire 20 2) ∧ a (.wire 19 15) = a (.wire 20 3) ∧ a (.virt 13209) = a (.wire 20 4) ∧ a (.virt 13210) = a (.wire 20 5) ∧ a (.virt 13210) = a (.wire 20 6) ∧ a (.virt 13210) = a (.wire 20 7) ∧ a (.virt 13210) = a (.wire 20 8) ∧ a (.virt 13210) = a (.wire 20 9) ∧ a (.virt 13210) = a (.wire 20 10) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies21, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper2.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_copies22 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13210) = a (.wire 20 11) ∧ a (.wire 2 27) = a (.wire 16 36) ∧ a (.virt 13180) = a (.wire 16 37) ∧ a (.virt 13180) = a (.wire 16 38) ∧ a (.wire 2 27) = a (.wire 16 40) ∧ a (.wire 20 12) = a (.wire 16 41) ∧ a (.wire 16 39) = a (.wire 16 42) ∧ a (.wire 2 27) = a (.wire 16 44) ∧ a (.virt 13181) = a (.wire 16 45) ∧ a (.virt 13181) = a (.wire 16 46) ∧ a (.wire 2 27) = a (.wire 16 48) ∧ a (.wire 20 13) = a (.wire 16 49) ∧ a (.wire 16 47) = a (.wire 16 50) ∧ a (.wire 2 27) = a (.wire 16 52) ∧ a (.virt 13182) = a (.wire 16 53) ∧ a (.virt 13182) = a (.wire 16 54) ∧ a (.wire 2 27) = a (.wire 16 56) ∧ a (.wire 20 14) = a (.wire 16 57) ∧ a (.wire 16 55) = a (.wire 16 58) ∧ a (.wire 2 27) = a (.wire 21 0) ∧ a (.virt 13183) = a (.wire 21 1) ∧ a (.virt 13183) = a (.wire 21 2) ∧ a (.wire 2 27) = a (.wire 21 4) ∧ a (.wire 20 15) = a (.wire 21 5) ∧ a (.wire 21 3) = a (.wire 21 6) := by
  have hc : ∀ q ∈ privateBatchWrapper2.copies22, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq)))))
  simp only [privateBatchWrapper2.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper2_consts (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13209) = 1 ∧ a (.virt 13210) = 0 ∧ a (.virt 13211) = 2 ∧ a (.virt 13244) = 576460752303423488 ∧ a (.virt 13245) = 10000 := by
  have hconst := h.2.2
  simp only [privateBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3, k4⟩ := hconst
  exact ⟨k0, k1, k2, k3, k4⟩

theorem privateBatchWrapper2_f0 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6589)) (a (.virt 13210)) (a (.virt 13212)) (a (.virt 13213)) := by
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
    IsEqual (a (.virt 6590)) (a (.virt 13210)) (a (.virt 13214)) (a (.virt 13215)) := by
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
    IsEqual (a (.virt 6591)) (a (.virt 13210)) (a (.virt 13216)) (a (.virt 13217)) := by
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
    IsEqual (a (.virt 6592)) (a (.virt 13210)) (a (.virt 13218)) (a (.virt 13219)) := by
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
    a (.wire 1 35) = band (a (.virt 13212)) (a (.virt 13214)) := by
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
    a (.wire 1 39) = band (a (.virt 13216)) (a (.virt 13218)) := by
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
    IsEqual (a (.virt 13184)) (a (.virt 13210)) (a (.virt 13220)) (a (.virt 13221)) := by
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
    IsEqual (a (.virt 13185)) (a (.virt 13210)) (a (.virt 13222)) (a (.virt 13223)) := by
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
    IsEqual (a (.virt 13186)) (a (.virt 13210)) (a (.virt 13224)) (a (.virt 13225)) := by
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
    IsEqual (a (.virt 13187)) (a (.virt 13210)) (a (.virt 13226)) (a (.virt 13227)) := by
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
    a (.wire 2 19) = band (a (.virt 13220)) (a (.virt 13222)) := by
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
    a (.wire 2 23) = band (a (.virt 13224)) (a (.virt 13226)) := by
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
    a (.virt 13209) = bnot (a (.virt 13210)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bnot, k0, k1]
  ring

theorem privateBatchWrapper2_f16 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 7) = band (a (.wire 3 7)) (a (.virt 13209)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [band, k0]
  ring

theorem privateBatchWrapper2_f17 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 11) = bselect (a (.wire 3 7)) (a (.virt 6589)) (a (.virt 13210)) := by
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
    a (.wire 3 15) = bselect (a (.wire 3 7)) (a (.virt 6590)) (a (.virt 13210)) := by
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
    a (.wire 3 19) = bselect (a (.wire 3 7)) (a (.virt 6591)) (a (.virt 13210)) := by
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
    a (.wire 3 23) = bselect (a (.wire 3 7)) (a (.virt 6592)) (a (.virt 13210)) := by
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
    a (.wire 3 27) = bselect (a (.wire 3 7)) (a (.virt 6593)) (a (.virt 13210)) := by
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
    a (.wire 3 7) = bor (a (.virt 13210)) (a (.wire 3 7)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [bor, k1]
  ring

theorem privateBatchWrapper2_f23 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = bnot (a (.wire 2 27)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c148, k0, ← c149, k0, ← c150] at e_3_7
  have hr := e_3_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f24 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 35) = bnot (a (.wire 3 7)) := by
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
    a (.wire 2 31) = band (a (.wire 3 31)) (a (.wire 3 35)) := by
  have c154 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c154, ← c155, ← c156] at e_2_7
  have hr := e_2_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f26 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 43) = bselect (a (.wire 2 31)) (a (.virt 13184)) (a (.wire 3 11)) := by
  have c157 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c158 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (privateBatchWrapper2_copies5 a h).1
  have c161 := (privateBatchWrapper2_copies5 a h).2.1
  have c162 := (privateBatchWrapper2_copies5 a h).2.2.1
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c157, ← c158, ← c159] at e_3_9
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c160, ← c161, ← c162] at e_3_10
  have hr := e_3_10
  simp only [e_3_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f27 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 51) = bselect (a (.wire 2 31)) (a (.virt 13185)) (a (.wire 3 15)) := by
  have c163 := (privateBatchWrapper2_copies5 a h).2.2.2.1
  have c164 := (privateBatchWrapper2_copies5 a h).2.2.2.2.1
  have c165 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.1
  have c166 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.1
  have c167 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.1
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c163, ← c164, ← c165] at e_3_11
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c166, ← c167, ← c168] at e_3_12
  have hr := e_3_12
  simp only [e_3_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f28 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 59) = bselect (a (.wire 2 31)) (a (.virt 13186)) (a (.wire 3 19)) := by
  have c169 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c172 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c169, ← c170, ← c171] at e_3_13
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c172, ← c173, ← c174] at e_3_14
  have hr := e_3_14
  simp only [e_3_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f29 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 7) = bselect (a (.wire 2 31)) (a (.virt 13187)) (a (.wire 3 23)) := by
  have c175 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c175, ← c176, ← c177] at e_4_0
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c178, ← c179, ← c180] at e_4_1
  have hr := e_4_1
  simp only [e_4_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f30 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 4 15) = bselect (a (.wire 2 31)) (a (.virt 13188)) (a (.wire 3 27)) := by
  have c181 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c186 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c181, ← c182, ← c183] at e_4_2
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c184, ← c185, ← c186] at e_4_3
  have hr := e_4_3
  simp only [e_4_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f31 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 3) = bor (a (.wire 3 7)) (a (.wire 3 31)) := by
  have c187 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c189 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (privateBatchWrapper2_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c192 := (privateBatchWrapper2_copies6 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c187, ← c188, ← c189] at e_5_0
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c190, ← c191, k0, ← c192] at e_6_0
  have hr := e_6_0
  simp only [e_5_0] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f32 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6589)) (a (.wire 3 43)) (a (.virt 13228)) (a (.virt 13229)) := by
  have c193 := (privateBatchWrapper2_copies6 a h).2.1
  have c194 := (privateBatchWrapper2_copies6 a h).2.2.1
  have c195 := (privateBatchWrapper2_copies6 a h).2.2.2.1
  have c196 := (privateBatchWrapper2_copies6 a h).2.2.2.2.1
  have c197 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.1
  have c198 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.1
  have c199 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c209 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c199, ← c200, ← c201] at e_2_8
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c202, ← c203, ← c204] at e_2_9
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c193, k0, ← c194, k0, ← c195] at e_4_4
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c196, ← c197, k0, ← c198] at e_4_5
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c205, ← c206, k0, ← c207] at e_4_6
  refine ⟨?_, ?_⟩
  · have hc := e_2_8
    simp only [e_4_5] at hc
    linear_combination c208.trans k1 - hc
  · have hc := e_4_6
    simp only [e_4_4, e_2_9, e_4_5] at hc
    linear_combination c209.trans k1 - hc

theorem privateBatchWrapper2_f33 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6590)) (a (.wire 3 51)) (a (.virt 13230)) (a (.virt 13231)) := by
  have c210 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c223 := (privateBatchWrapper2_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (privateBatchWrapper2_copies7 a h).1
  have c225 := (privateBatchWrapper2_copies7 a h).2.1
  have c226 := (privateBatchWrapper2_copies7 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c216, ← c217, ← c218] at e_2_10
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c219, ← c220, ← c221] at e_2_11
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c210, k0, ← c211, k0, ← c212] at e_4_7
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c213, ← c214, k0, ← c215] at e_4_8
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c222, ← c223, k0, ← c224] at e_4_9
  refine ⟨?_, ?_⟩
  · have hc := e_2_10
    simp only [e_4_8] at hc
    linear_combination c225.trans k1 - hc
  · have hc := e_4_9
    simp only [e_4_7, e_2_11, e_4_8] at hc
    linear_combination c226.trans k1 - hc

theorem privateBatchWrapper2_f34 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6591)) (a (.wire 3 59)) (a (.virt 13232)) (a (.virt 13233)) := by
  have c227 := (privateBatchWrapper2_copies7 a h).2.2.2.1
  have c228 := (privateBatchWrapper2_copies7 a h).2.2.2.2.1
  have c229 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.1
  have c230 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.1
  have c231 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c237 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c233, ← c234, ← c235] at e_2_12
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c236, ← c237, ← c238] at e_2_13
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c227, k0, ← c228, k0, ← c229] at e_4_10
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c230, ← c231, k0, ← c232] at e_4_11
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c239, ← c240, k0, ← c241] at e_4_12
  refine ⟨?_, ?_⟩
  · have hc := e_2_12
    simp only [e_4_11] at hc
    linear_combination c242.trans k1 - hc
  · have hc := e_4_12
    simp only [e_4_10, e_2_13, e_4_11] at hc
    linear_combination c243.trans k1 - hc

theorem privateBatchWrapper2_f35 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6592)) (a (.wire 4 7)) (a (.virt 13234)) (a (.virt 13235)) := by
  have c244 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c251 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c254 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (privateBatchWrapper2_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (privateBatchWrapper2_copies8 a h).1
  have c257 := (privateBatchWrapper2_copies8 a h).2.1
  have c258 := (privateBatchWrapper2_copies8 a h).2.2.1
  have c259 := (privateBatchWrapper2_copies8 a h).2.2.2.1
  have c260 := (privateBatchWrapper2_copies8 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c250, ← c251, ← c252] at e_2_14
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c244, k0, ← c245, k0, ← c246] at e_4_13
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c247, ← c248, k0, ← c249] at e_4_14
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c253, ← c254, ← c255] at e_7_0
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c256, ← c257, k0, ← c258] at e_8_0
  refine ⟨?_, ?_⟩
  · have hc := e_2_14
    simp only [e_4_14] at hc
    linear_combination c259.trans k1 - hc
  · have hc := e_8_0
    simp only [e_4_13, e_7_0, e_4_14] at hc
    linear_combination c260.trans k1 - hc

theorem privateBatchWrapper2_f36 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 7) = band (a (.virt 13228)) (a (.virt 13230)) := by
  have c261 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.1
  have c263 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.1
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c261, ← c262, ← c263] at e_7_1
  have hr := e_7_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f37 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 11) = band (a (.virt 13232)) (a (.virt 13234)) := by
  have c264 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.1
  have c266 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c264, ← c265, ← c266] at e_7_2
  have hr := e_7_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f38 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 15) = band (a (.wire 7 7)) (a (.wire 7 11)) := by
  have c267 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c269 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c267, ← c268, ← c269] at e_7_3
  have hr := e_7_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f39 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 7) = bor (a (.wire 1 43)) (a (.wire 7 15)) := by
  have c270 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c272 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c275 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c270, ← c271, ← c272] at e_5_1
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c273, ← c274, k0, ← c275] at e_6_1
  have hr := e_6_1
  simp only [e_5_1] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f40 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 7) = a (.virt 13209) := by
  have c276 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c276

theorem privateBatchWrapper2_f41 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 6584) = a (.virt 6584) := by
  rfl

theorem privateBatchWrapper2_f42 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 13184)) (a (.wire 3 43)) (a (.virt 13236)) (a (.virt 13237)) := by
  have c278 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c281 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c284 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c287 := (privateBatchWrapper2_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (privateBatchWrapper2_copies9 a h).1
  have c289 := (privateBatchWrapper2_copies9 a h).2.1
  have c290 := (privateBatchWrapper2_copies9 a h).2.2.1
  have c291 := (privateBatchWrapper2_copies9 a h).2.2.2.1
  have c292 := (privateBatchWrapper2_copies9 a h).2.2.2.2.1
  have c293 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.1
  have c294 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c284, ← c285, ← c286] at e_7_4
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c287, ← c288, ← c289] at e_7_5
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c278, k0, ← c279, k0, ← c280] at e_8_1
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c281, ← c282, k0, ← c283] at e_8_2
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c290, ← c291, k0, ← c292] at e_8_3
  refine ⟨?_, ?_⟩
  · have hc := e_7_4
    simp only [e_8_2] at hc
    linear_combination c293.trans k1 - hc
  · have hc := e_8_3
    simp only [e_8_1, e_7_5, e_8_2] at hc
    linear_combination c294.trans k1 - hc

theorem privateBatchWrapper2_f43 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 13185)) (a (.wire 3 51)) (a (.virt 13238)) (a (.virt 13239)) := by
  have c295 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have c299 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c305 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c311 := (privateBatchWrapper2_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c301, ← c302, ← c303] at e_7_6
  have e_7_7 := arithEq_of_rows h (row := 7) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_7
  simp only [← c304, ← c305, ← c306] at e_7_7
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c295, k0, ← c296, k0, ← c297] at e_8_4
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c298, ← c299, k0, ← c300] at e_8_5
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c307, ← c308, k0, ← c309] at e_8_6
  refine ⟨?_, ?_⟩
  · have hc := e_7_6
    simp only [e_8_5] at hc
    linear_combination c310.trans k1 - hc
  · have hc := e_8_6
    simp only [e_8_4, e_7_7, e_8_5] at hc
    linear_combination c311.trans k1 - hc

theorem privateBatchWrapper2_f44 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 13186)) (a (.wire 3 59)) (a (.virt 13240)) (a (.virt 13241)) := by
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
  have c328 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_8 := arithEq_of_rows h (row := 7) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_8
  simp only [← c318, ← c319, ← c320] at e_7_8
  have e_7_9 := arithEq_of_rows h (row := 7) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_9
  simp only [← c321, ← c322, ← c323] at e_7_9
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c312, k0, ← c313, k0, ← c314] at e_8_7
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c315, ← c316, k0, ← c317] at e_8_8
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c324, ← c325, k0, ← c326] at e_8_9
  refine ⟨?_, ?_⟩
  · have hc := e_7_8
    simp only [e_8_8] at hc
    linear_combination c327.trans k1 - hc
  · have hc := e_8_9
    simp only [e_8_7, e_7_9, e_8_8] at hc
    linear_combination c328.trans k1 - hc

theorem privateBatchWrapper2_f45 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 13187)) (a (.wire 4 7)) (a (.virt 13242)) (a (.virt 13243)) := by
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
  have c345 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_7_10 := arithEq_of_rows h (row := 7) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_10
  simp only [← c335, ← c336, ← c337] at e_7_10
  have e_7_11 := arithEq_of_rows h (row := 7) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_11
  simp only [← c338, ← c339, ← c340] at e_7_11
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c329, k0, ← c330, k0, ← c331] at e_8_10
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c332, ← c333, k0, ← c334] at e_8_11
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c341, ← c342, k0, ← c343] at e_8_12
  refine ⟨?_, ?_⟩
  · have hc := e_7_10
    simp only [e_8_11] at hc
    linear_combination c344.trans k1 - hc
  · have hc := e_8_12
    simp only [e_8_10, e_7_11, e_8_11] at hc
    linear_combination c345.trans k1 - hc

theorem privateBatchWrapper2_f46 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 51) = band (a (.virt 13236)) (a (.virt 13238)) := by
  have c346 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_7_12 := arithEq_of_rows h (row := 7) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_12
  simp only [← c346, ← c347, ← c348] at e_7_12
  have hr := e_7_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f47 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 55) = band (a (.virt 13240)) (a (.virt 13242)) := by
  have c349 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c350 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (privateBatchWrapper2_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_7_13 := arithEq_of_rows h (row := 7) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_13
  simp only [← c349, ← c350, ← c351] at e_7_13
  have hr := e_7_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f48 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 7 59) = band (a (.wire 7 51)) (a (.wire 7 55)) := by
  have c352 := (privateBatchWrapper2_copies11 a h).1
  have c353 := (privateBatchWrapper2_copies11 a h).2.1
  have c354 := (privateBatchWrapper2_copies11 a h).2.2.1
  have e_7_14 := arithEq_of_rows h (row := 7) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_14
  simp only [← c352, ← c353, ← c354] at e_7_14
  have hr := e_7_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f49 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 11) = bor (a (.wire 2 27)) (a (.wire 7 59)) := by
  have c355 := (privateBatchWrapper2_copies11 a h).2.2.2.1
  have c356 := (privateBatchWrapper2_copies11 a h).2.2.2.2.1
  have c357 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.1
  have c358 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.1
  have c359 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c355, ← c356, ← c357] at e_5_2
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c358, ← c359, k0, ← c360] at e_6_2
  have hr := e_6_2
  simp only [e_5_2] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper2_f50 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 11) = a (.virt 13209) := by
  have c361 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.1
  exact c361

theorem privateBatchWrapper2_f51 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13179) = a (.virt 6584) := by
  have c362 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c362

theorem privateBatchWrapper2_f52 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 59) = bselect (a (.wire 1 43)) (a (.virt 13210)) (a (.virt 6594)) := by
  have c363 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c368 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_8_13 := arithEq_of_rows h (row := 8) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_13
  simp only [← c363, ← c364, ← c365] at e_8_13
  have e_8_14 := arithEq_of_rows h (row := 8) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_14
  simp only [← c366, ← c367, k1, ← c368] at e_8_14
  have hr := e_8_14
  simp only [e_8_13] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f53 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 8 59) = a (.virt 13210) + a (.wire 8 59) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f54 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 7) = bselect (a (.wire 2 27)) (a (.virt 13210)) (a (.virt 13189)) := by
  have c369 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c371 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c374 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c369, ← c370, ← c371] at e_9_0
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c372, ← c373, k1, ← c374] at e_9_1
  have hr := e_9_1
  simp only [e_9_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper2_f55 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 15) = a (.wire 8 59) + a (.wire 9 7) := by
  have c375 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c377 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c375, ← c376, k0, ← c377] at e_6_3
  have hr := e_6_3
  linear_combination hr

theorem privateBatchWrapper2_f56 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.virt 13195)) 32 := by
  have c378 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c383 := (privateBatchWrapper2_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (privateBatchWrapper2_copies12 a h).1
  have c385 := (privateBatchWrapper2_copies12 a h).2.1
  have c386 := (privateBatchWrapper2_copies12 a h).2.2.1
  have c387 := (privateBatchWrapper2_copies12 a h).2.2.2.1
  have c388 := (privateBatchWrapper2_copies12 a h).2.2.2.2.1
  have c389 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.1
  have c390 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.1
  have c391 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have c395 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 10) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c378.trans k1
    · exact c379.trans k1
    · exact c380.trans k1
    · exact c381.trans k1
    · exact c382.trans k1
    · exact c383.trans k1
    · exact c384.trans k1
    · exact c385.trans k1
    · exact c386.trans k1
    · exact c387.trans k1
    · exact c388.trans k1
    · exact c389.trans k1
    · exact c390.trans k1
    · exact c391.trans k1
    · exact c392.trans k1
    · exact c393.trans k1
    · exact c394.trans k1
    · exact c395.trans k1
    · exact c396.trans k1
    · exact c397.trans k1
    · exact c398.trans k1
    · exact c399.trans k1
    · exact c400.trans k1
    · exact c401.trans k1
    · exact c402.trans k1
    · exact c403.trans k1
    · exact c404.trans k1
    )
  rwa [c405] at hr

theorem privateBatchWrapper2_f57 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 13195) = a (.virt 13210) + a (.virt 13195) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper2_f58 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.virt 13200)) 32 := by
  have c406 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c407 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c413 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (privateBatchWrapper2_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (privateBatchWrapper2_copies13 a h).1
  have c417 := (privateBatchWrapper2_copies13 a h).2.1
  have c418 := (privateBatchWrapper2_copies13 a h).2.2.1
  have c419 := (privateBatchWrapper2_copies13 a h).2.2.2.1
  have c420 := (privateBatchWrapper2_copies13 a h).2.2.2.2.1
  have c421 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.1
  have c422 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.1
  have c423 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c430 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 11) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c406.trans k1
    · exact c407.trans k1
    · exact c408.trans k1
    · exact c409.trans k1
    · exact c410.trans k1
    · exact c411.trans k1
    · exact c412.trans k1
    · exact c413.trans k1
    · exact c414.trans k1
    · exact c415.trans k1
    · exact c416.trans k1
    · exact c417.trans k1
    · exact c418.trans k1
    · exact c419.trans k1
    · exact c420.trans k1
    · exact c421.trans k1
    · exact c422.trans k1
    · exact c423.trans k1
    · exact c424.trans k1
    · exact c425.trans k1
    · exact c426.trans k1
    · exact c427.trans k1
    · exact c428.trans k1
    · exact c429.trans k1
    · exact c430.trans k1
    · exact c431.trans k1
    · exact c432.trans k1
    )
  rwa [c433] at hr

theorem privateBatchWrapper2_f59 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 6 19) = a (.virt 13195) + a (.virt 13200) := by
  have c434 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c434, ← c435, k0, ← c436] at e_6_4
  have hr := e_6_4
  linear_combination hr

theorem privateBatchWrapper2_f60 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.virt 13190)) 14 := by
  have c437 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c447 := (privateBatchWrapper2_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (privateBatchWrapper2_copies14 a h).1
  have c449 := (privateBatchWrapper2_copies14 a h).2.1
  have c450 := (privateBatchWrapper2_copies14 a h).2.2.1
  have c451 := (privateBatchWrapper2_copies14 a h).2.2.2.1
  have c452 := (privateBatchWrapper2_copies14 a h).2.2.2.2.1
  have c453 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.1
  have c454 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.1
  have c455 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c464 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (privateBatchWrapper2_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (privateBatchWrapper2_copies15 a h).1
  have c481 := (privateBatchWrapper2_copies15 a h).2.1
  have c482 := (privateBatchWrapper2_copies15 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 12) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c437.trans k1
    · exact c438.trans k1
    · exact c439.trans k1
    · exact c440.trans k1
    · exact c441.trans k1
    · exact c442.trans k1
    · exact c443.trans k1
    · exact c444.trans k1
    · exact c445.trans k1
    · exact c446.trans k1
    · exact c447.trans k1
    · exact c448.trans k1
    · exact c449.trans k1
    · exact c450.trans k1
    · exact c451.trans k1
    · exact c452.trans k1
    · exact c453.trans k1
    · exact c454.trans k1
    · exact c455.trans k1
    · exact c456.trans k1
    · exact c457.trans k1
    · exact c458.trans k1
    · exact c459.trans k1
    · exact c460.trans k1
    · exact c461.trans k1
    · exact c462.trans k1
    · exact c463.trans k1
    · exact c464.trans k1
    · exact c465.trans k1
    · exact c466.trans k1
    · exact c467.trans k1
    · exact c468.trans k1
    · exact c469.trans k1
    · exact c470.trans k1
    · exact c471.trans k1
    · exact c472.trans k1
    · exact c473.trans k1
    · exact c474.trans k1
    · exact c475.trans k1
    · exact c476.trans k1
    · exact c477.trans k1
    · exact c478.trans k1
    · exact c479.trans k1
    · exact c480.trans k1
    · exact c481.trans k1
    )
  rwa [c482] at hr

theorem privateBatchWrapper2_f61 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 11) = a (.virt 13245) - a (.virt 13190) := by
  have c483 := (privateBatchWrapper2_copies15 a h).2.2.2.1
  have c484 := (privateBatchWrapper2_copies15 a h).2.2.2.2.1
  have c485 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c483, k4, ← c484, k0, ← c485] at e_9_2
  have hr := e_9_2
  simp only [k4]
  linear_combination hr

theorem privateBatchWrapper2_f62 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 9 11)) 14 := by
  have c486 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.1
  have c487 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have c490 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c496 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c497 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c498 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (privateBatchWrapper2_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (privateBatchWrapper2_copies16 a h).1
  have c513 := (privateBatchWrapper2_copies16 a h).2.1
  have c514 := (privateBatchWrapper2_copies16 a h).2.2.1
  have c515 := (privateBatchWrapper2_copies16 a h).2.2.2.1
  have c516 := (privateBatchWrapper2_copies16 a h).2.2.2.2.1
  have c517 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.1
  have c518 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.1
  have c519 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 13) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c486.trans k1
    · exact c487.trans k1
    · exact c488.trans k1
    · exact c489.trans k1
    · exact c490.trans k1
    · exact c491.trans k1
    · exact c492.trans k1
    · exact c493.trans k1
    · exact c494.trans k1
    · exact c495.trans k1
    · exact c496.trans k1
    · exact c497.trans k1
    · exact c498.trans k1
    · exact c499.trans k1
    · exact c500.trans k1
    · exact c501.trans k1
    · exact c502.trans k1
    · exact c503.trans k1
    · exact c504.trans k1
    · exact c505.trans k1
    · exact c506.trans k1
    · exact c507.trans k1
    · exact c508.trans k1
    · exact c509.trans k1
    · exact c510.trans k1
    · exact c511.trans k1
    · exact c512.trans k1
    · exact c513.trans k1
    · exact c514.trans k1
    · exact c515.trans k1
    · exact c516.trans k1
    · exact c517.trans k1
    · exact c518.trans k1
    · exact c519.trans k1
    · exact c520.trans k1
    · exact c521.trans k1
    · exact c522.trans k1
    · exact c523.trans k1
    · exact c524.trans k1
    · exact c525.trans k1
    · exact c526.trans k1
    · exact c527.trans k1
    · exact c528.trans k1
    · exact c529.trans k1
    · exact c530.trans k1
    )
  rwa [c531] at hr

theorem privateBatchWrapper2_f63 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 3) = a (.wire 6 19) * a (.virt 13245) := by
  have c532 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c532, ← c533, k4, ← c534] at e_14_0
  have hr := e_14_0
  simp only [k4]
  linear_combination hr

theorem privateBatchWrapper2_f64 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 7) = a (.wire 6 15) * a (.wire 9 11) := by
  have c535 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c535, ← c536, ← c537] at e_14_1
  have hr := e_14_1
  linear_combination hr

theorem privateBatchWrapper2_f65 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 9 15) = a (.wire 14 7) - a (.wire 14 3) := by
  have c538 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c538, ← c539, k0, ← c540] at e_9_3
  have hr := e_9_3
  linear_combination hr

theorem privateBatchWrapper2_f66 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    rangeCheck (a (.wire 9 15)) 52 := by
  have c541 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (privateBatchWrapper2_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (privateBatchWrapper2_copies17 a h).1
  have c545 := (privateBatchWrapper2_copies17 a h).2.1
  have c546 := (privateBatchWrapper2_copies17 a h).2.2.1
  have c547 := (privateBatchWrapper2_copies17 a h).2.2.2.1
  have c548 := (privateBatchWrapper2_copies17 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hr := rangeCheck_of_row h (row := 15) (N := 59) (n := 52) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c541.trans k1
    · exact c542.trans k1
    · exact c543.trans k1
    · exact c544.trans k1
    · exact c545.trans k1
    · exact c546.trans k1
    · exact c547.trans k1
    )
  rwa [c548] at hr

theorem privateBatchWrapper2_f67 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
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

theorem privateBatchWrapper2_f68 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = bnot (a (.wire 2 27)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c148, k0, ← c149, k0, ← c150] at e_3_7
  have hr := e_3_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f69 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 11) = band (a (.wire 3 7)) (a (.wire 3 31)) := by
  have c549 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.1
  have c550 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.1
  have c551 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.1
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c549, ← c550, ← c551] at e_14_2
  have hr := e_14_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f70 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6585)) (a (.virt 13180)) (a (.virt 13246)) (a (.virt 13247)) := by
  have c552 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c566 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (privateBatchWrapper2_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c552, k0, ← c553, k0, ← c554] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c555, ← c556, k0, ← c557] at e_9_5
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c564, ← c565, k0, ← c566] at e_9_6
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c558, ← c559, ← c560] at e_14_3
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c561, ← c562, ← c563] at e_14_4
  refine ⟨?_, ?_⟩
  · have hc := e_14_3
    simp only [e_9_5] at hc
    linear_combination c567.trans k1 - hc
  · have hc := e_9_6
    simp only [e_9_4, e_14_4, e_9_5] at hc
    linear_combination c568.trans k1 - hc

theorem privateBatchWrapper2_f71 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6586)) (a (.virt 13181)) (a (.virt 13248)) (a (.virt 13249)) := by
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
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c569, k0, ← c570, k0, ← c571] at e_9_7
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c572, ← c573, k0, ← c574] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c581, ← c582, k0, ← c583] at e_9_9
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c575, ← c576, ← c577] at e_14_5
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c578, ← c579, ← c580] at e_14_6
  refine ⟨?_, ?_⟩
  · have hc := e_14_5
    simp only [e_9_8] at hc
    linear_combination c584.trans k1 - hc
  · have hc := e_9_9
    simp only [e_9_7, e_14_6, e_9_8] at hc
    linear_combination c585.trans k1 - hc

theorem privateBatchWrapper2_f72 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6587)) (a (.virt 13182)) (a (.virt 13250)) (a (.virt 13251)) := by
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
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c586, k0, ← c587, k0, ← c588] at e_9_10
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c589, ← c590, k0, ← c591] at e_9_11
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c598, ← c599, k0, ← c600] at e_9_12
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c592, ← c593, ← c594] at e_14_7
  have e_14_8 := arithEq_of_rows h (row := 14) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c595, ← c596, ← c597] at e_14_8
  refine ⟨?_, ?_⟩
  · have hc := e_14_7
    simp only [e_9_11] at hc
    linear_combination c601.trans k1 - hc
  · have hc := e_9_12
    simp only [e_9_10, e_14_8, e_9_11] at hc
    linear_combination c602.trans k1 - hc

theorem privateBatchWrapper2_f73 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    IsEqual (a (.virt 6588)) (a (.virt 13183)) (a (.virt 13252)) (a (.virt 13253)) := by
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
  have c613 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.1
  have c614 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.1
  have c615 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.1
  have c617 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c603, k0, ← c604, k0, ← c605] at e_9_13
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c606, ← c607, k0, ← c608] at e_9_14
  have e_14_9 := arithEq_of_rows h (row := 14) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c609, ← c610, ← c611] at e_14_9
  have e_14_10 := arithEq_of_rows h (row := 14) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c612, ← c613, ← c614] at e_14_10
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c615, ← c616, k0, ← c617] at e_16_0
  refine ⟨?_, ?_⟩
  · have hc := e_14_9
    simp only [e_9_14] at hc
    linear_combination c618.trans k1 - hc
  · have hc := e_16_0
    simp only [e_9_13, e_14_10, e_9_14] at hc
    linear_combination c619.trans k1 - hc

theorem privateBatchWrapper2_f74 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 47) = band (a (.virt 13246)) (a (.virt 13248)) := by
  have c620 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_11 := arithEq_of_rows h (row := 14) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c620, ← c621, ← c622] at e_14_11
  have hr := e_14_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f75 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 51) = band (a (.virt 13250)) (a (.virt 13252)) := by
  have c623 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_12 := arithEq_of_rows h (row := 14) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c623, ← c624, ← c625] at e_14_12
  have hr := e_14_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f76 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 55) = band (a (.wire 14 47)) (a (.wire 14 51)) := by
  have c626 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_13 := arithEq_of_rows h (row := 14) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c626, ← c627, ← c628] at e_14_13
  have hr := e_14_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f77 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 59) = band (a (.wire 14 11)) (a (.wire 14 55)) := by
  have c629 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_14_14 := arithEq_of_rows h (row := 14) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c629, ← c630, ← c631] at e_14_14
  have hr := e_14_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper2_f78 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 14 59) = a (.virt 13210) := by
  have c632 := (privateBatchWrapper2_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c632

theorem privateBatchWrapper2_f79 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 3 31) = bnot (a (.wire 2 27)) := by
  have c148 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper2_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c148, k0, ← c149, k0, ← c150] at e_3_7
  have hr := e_3_7
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper2_f80 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 17 12) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 0 ∧ a (.wire 17 13) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 1 ∧ a (.wire 17 14) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 2 ∧ a (.wire 17 15) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 3) := by
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
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hin0 : poseidon2In a 17 = addBlock (fun _ => 0) [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204), 1, 0, 0, 0] :=
    poseidon2In_first c633.symm c634.symm c635.symm c636.symm (c637.symm.trans k0) (c638.symm.trans k1) (c639.symm.trans k1) (c640.symm.trans k1) (c641.symm.trans k1) (c642.symm.trans k1) (c643.symm.trans k1) (c644.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 17) rfl rfl hin0
  have hpad : pad10 [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] =
      [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f81 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 18 12) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 0 ∧ a (.wire 18 13) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 1 ∧ a (.wire 18 14) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 2 ∧ a (.wire 18 15) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 3) := by
  have c645 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.1
  have c646 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.1
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
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hin0 : poseidon2In a 18 = addBlock (fun _ => 0) [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15), 1, 0, 0, 0] :=
    poseidon2In_first c645.symm c646.symm c647.symm c648.symm (c649.symm.trans k0) (c650.symm.trans k1) (c651.symm.trans k1) (c652.symm.trans k1) (c653.symm.trans k1) (c654.symm.trans k1) (c655.symm.trans k1) (c656.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 18) rfl rfl hin0
  have hpad : pad10 [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] =
      [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f82 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 11) = bselect (a (.wire 1 43)) (a (.wire 18 12)) (a (.virt 6585)) := by
  have c657 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c660 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c657, ← c658, ← c659] at e_16_1
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c660, ← c661, ← c662] at e_16_2
  have hr := e_16_2
  simp only [e_16_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f83 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 19) = bselect (a (.wire 1 43)) (a (.wire 18 13)) (a (.virt 6586)) := by
  have c663 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c666 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c667 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c668 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c663, ← c664, ← c665] at e_16_3
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c666, ← c667, ← c668] at e_16_4
  have hr := e_16_4
  simp only [e_16_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f84 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 27) = bselect (a (.wire 1 43)) (a (.wire 18 14)) (a (.virt 6587)) := by
  have c669 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (privateBatchWrapper2_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (privateBatchWrapper2_copies21 a h).1
  have c673 := (privateBatchWrapper2_copies21 a h).2.1
  have c674 := (privateBatchWrapper2_copies21 a h).2.2.1
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c669, ← c670, ← c671] at e_16_5
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c672, ← c673, ← c674] at e_16_6
  have hr := e_16_6
  simp only [e_16_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f85 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 35) = bselect (a (.wire 1 43)) (a (.wire 18 15)) (a (.virt 6588)) := by
  have c675 := (privateBatchWrapper2_copies21 a h).2.2.2.1
  have c676 := (privateBatchWrapper2_copies21 a h).2.2.2.2.1
  have c677 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.1
  have c678 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.1
  have c679 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.1
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c675, ← c676, ← c677] at e_16_7
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c678, ← c679, ← c680] at e_16_8
  have hr := e_16_8
  simp only [e_16_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f86 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 19 12) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 0 ∧ a (.wire 19 13) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 1 ∧ a (.wire 19 14) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 2 ∧ a (.wire 19 15) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 3) := by
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
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hin0 : poseidon2In a 19 = addBlock (fun _ => 0) [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208), 1, 0, 0, 0] :=
    poseidon2In_first c681.symm c682.symm c683.symm c684.symm (c685.symm.trans k0) (c686.symm.trans k1) (c687.symm.trans k1) (c688.symm.trans k1) (c689.symm.trans k1) (c690.symm.trans k1) (c691.symm.trans k1) (c692.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 19) rfl rfl hin0
  have hpad : pad10 [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] =
      [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f87 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (a (.wire 20 12) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 0 ∧ a (.wire 20 13) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 1 ∧ a (.wire 20 14) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 2 ∧ a (.wire 20 15) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 3) := by
  have c693 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c702 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (privateBatchWrapper2_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (privateBatchWrapper2_copies22 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper2_consts a h
  have hin0 : poseidon2In a 20 = addBlock (fun _ => 0) [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15), 1, 0, 0, 0] :=
    poseidon2In_first c693.symm c694.symm c695.symm c696.symm (c697.symm.trans k0) (c698.symm.trans k1) (c699.symm.trans k1) (c700.symm.trans k1) (c701.symm.trans k1) (c702.symm.trans k1) (c703.symm.trans k1) (c704.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 20) rfl rfl hin0
  have hpad : pad10 [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] =
      [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper2_f88 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 43) = bselect (a (.wire 2 27)) (a (.wire 20 12)) (a (.virt 13180)) := by
  have c705 := (privateBatchWrapper2_copies22 a h).2.1
  have c706 := (privateBatchWrapper2_copies22 a h).2.2.1
  have c707 := (privateBatchWrapper2_copies22 a h).2.2.2.1
  have c708 := (privateBatchWrapper2_copies22 a h).2.2.2.2.1
  have c709 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.1
  have c710 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.1
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c705, ← c706, ← c707] at e_16_9
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c708, ← c709, ← c710] at e_16_10
  have hr := e_16_10
  simp only [e_16_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f89 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 51) = bselect (a (.wire 2 27)) (a (.wire 20 13)) (a (.virt 13181)) := by
  have c711 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c711, ← c712, ← c713] at e_16_11
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c714, ← c715, ← c716] at e_16_12
  have hr := e_16_12
  simp only [e_16_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f90 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 16 59) = bselect (a (.wire 2 27)) (a (.wire 20 14)) (a (.virt 13182)) := by
  have c717 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c719 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c717, ← c718, ← c719] at e_16_13
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c720, ← c721, ← c722] at e_16_14
  have hr := e_16_14
  simp only [e_16_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper2_f91 (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.wire 21 7) = bselect (a (.wire 2 27)) (a (.wire 20 15)) (a (.virt 13183)) := by
  have c723 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (privateBatchWrapper2_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_21_0 := arithEq_of_rows h (row := 21) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_0
  simp only [← c723, ← c724, ← c725] at e_21_0
  have e_21_1 := arithEq_of_rows h (row := 21) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_21_1
  simp only [← c726, ← c727, ← c728] at e_21_1
  have hr := e_21_1
  simp only [e_21_0] at hr
  simp only [bselect]
  linear_combination hr

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `privateBatchWrapper2` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem privateBatchWrapper2_decode (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (IsEqual (a (.virt 6589)) (a (.virt 13210)) (a (.virt 13212)) (a (.virt 13213)) ∧
    IsEqual (a (.virt 6590)) (a (.virt 13210)) (a (.virt 13214)) (a (.virt 13215)) ∧
    IsEqual (a (.virt 6591)) (a (.virt 13210)) (a (.virt 13216)) (a (.virt 13217)) ∧
    IsEqual (a (.virt 6592)) (a (.virt 13210)) (a (.virt 13218)) (a (.virt 13219)) ∧
    a (.wire 1 35) = band (a (.virt 13212)) (a (.virt 13214)) ∧
    a (.wire 1 39) = band (a (.virt 13216)) (a (.virt 13218)) ∧
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) ∧
    IsEqual (a (.virt 13184)) (a (.virt 13210)) (a (.virt 13220)) (a (.virt 13221)) ∧
    IsEqual (a (.virt 13185)) (a (.virt 13210)) (a (.virt 13222)) (a (.virt 13223)) ∧
    IsEqual (a (.virt 13186)) (a (.virt 13210)) (a (.virt 13224)) (a (.virt 13225)) ∧
    IsEqual (a (.virt 13187)) (a (.virt 13210)) (a (.virt 13226)) (a (.virt 13227)) ∧
    a (.wire 2 19) = band (a (.virt 13220)) (a (.virt 13222)) ∧
    a (.wire 2 23) = band (a (.virt 13224)) (a (.virt 13226)) ∧
    a (.wire 2 27) = band (a (.wire 2 19)) (a (.wire 2 23)) ∧
    a (.wire 3 7) = bnot (a (.wire 1 43)) ∧
    a (.virt 13209) = bnot (a (.virt 13210)) ∧
    a (.wire 3 7) = band (a (.wire 3 7)) (a (.virt 13209)) ∧
    a (.wire 3 11) = bselect (a (.wire 3 7)) (a (.virt 6589)) (a (.virt 13210)) ∧
    a (.wire 3 15) = bselect (a (.wire 3 7)) (a (.virt 6590)) (a (.virt 13210)) ∧
    a (.wire 3 19) = bselect (a (.wire 3 7)) (a (.virt 6591)) (a (.virt 13210)) ∧
    a (.wire 3 23) = bselect (a (.wire 3 7)) (a (.virt 6592)) (a (.virt 13210)) ∧
    a (.wire 3 27) = bselect (a (.wire 3 7)) (a (.virt 6593)) (a (.virt 13210)) ∧
    a (.wire 3 7) = bor (a (.virt 13210)) (a (.wire 3 7)) ∧
    a (.wire 3 31) = bnot (a (.wire 2 27)) ∧
    a (.wire 3 35) = bnot (a (.wire 3 7)) ∧
    a (.wire 2 31) = band (a (.wire 3 31)) (a (.wire 3 35)) ∧
    a (.wire 3 43) = bselect (a (.wire 2 31)) (a (.virt 13184)) (a (.wire 3 11)) ∧
    a (.wire 3 51) = bselect (a (.wire 2 31)) (a (.virt 13185)) (a (.wire 3 15)) ∧
    a (.wire 3 59) = bselect (a (.wire 2 31)) (a (.virt 13186)) (a (.wire 3 19)) ∧
    a (.wire 4 7) = bselect (a (.wire 2 31)) (a (.virt 13187)) (a (.wire 3 23)) ∧
    a (.wire 4 15) = bselect (a (.wire 2 31)) (a (.virt 13188)) (a (.wire 3 27)) ∧
    a (.wire 6 3) = bor (a (.wire 3 7)) (a (.wire 3 31))) ∧
    (IsEqual (a (.virt 6589)) (a (.wire 3 43)) (a (.virt 13228)) (a (.virt 13229)) ∧
    IsEqual (a (.virt 6590)) (a (.wire 3 51)) (a (.virt 13230)) (a (.virt 13231)) ∧
    IsEqual (a (.virt 6591)) (a (.wire 3 59)) (a (.virt 13232)) (a (.virt 13233)) ∧
    IsEqual (a (.virt 6592)) (a (.wire 4 7)) (a (.virt 13234)) (a (.virt 13235)) ∧
    a (.wire 7 7) = band (a (.virt 13228)) (a (.virt 13230)) ∧
    a (.wire 7 11) = band (a (.virt 13232)) (a (.virt 13234)) ∧
    a (.wire 7 15) = band (a (.wire 7 7)) (a (.wire 7 11)) ∧
    a (.wire 6 7) = bor (a (.wire 1 43)) (a (.wire 7 15)) ∧
    a (.wire 6 7) = a (.virt 13209) ∧
    a (.virt 6584) = a (.virt 6584) ∧
    IsEqual (a (.virt 13184)) (a (.wire 3 43)) (a (.virt 13236)) (a (.virt 13237)) ∧
    IsEqual (a (.virt 13185)) (a (.wire 3 51)) (a (.virt 13238)) (a (.virt 13239)) ∧
    IsEqual (a (.virt 13186)) (a (.wire 3 59)) (a (.virt 13240)) (a (.virt 13241)) ∧
    IsEqual (a (.virt 13187)) (a (.wire 4 7)) (a (.virt 13242)) (a (.virt 13243)) ∧
    a (.wire 7 51) = band (a (.virt 13236)) (a (.virt 13238)) ∧
    a (.wire 7 55) = band (a (.virt 13240)) (a (.virt 13242)) ∧
    a (.wire 7 59) = band (a (.wire 7 51)) (a (.wire 7 55)) ∧
    a (.wire 6 11) = bor (a (.wire 2 27)) (a (.wire 7 59)) ∧
    a (.wire 6 11) = a (.virt 13209) ∧
    a (.virt 13179) = a (.virt 6584) ∧
    a (.wire 8 59) = bselect (a (.wire 1 43)) (a (.virt 13210)) (a (.virt 6594)) ∧
    a (.wire 8 59) = a (.virt 13210) + a (.wire 8 59) ∧
    a (.wire 9 7) = bselect (a (.wire 2 27)) (a (.virt 13210)) (a (.virt 13189)) ∧
    a (.wire 6 15) = a (.wire 8 59) + a (.wire 9 7) ∧
    rangeCheck (a (.virt 13195)) 32 ∧
    a (.virt 13195) = a (.virt 13210) + a (.virt 13195) ∧
    rangeCheck (a (.virt 13200)) 32 ∧
    a (.wire 6 19) = a (.virt 13195) + a (.virt 13200) ∧
    rangeCheck (a (.virt 13190)) 14 ∧
    a (.wire 9 11) = a (.virt 13245) - a (.virt 13190) ∧
    rangeCheck (a (.wire 9 11)) 14 ∧
    a (.wire 14 3) = a (.wire 6 19) * a (.virt 13245)) ∧
    (a (.wire 14 7) = a (.wire 6 15) * a (.wire 9 11) ∧
    a (.wire 9 15) = a (.wire 14 7) - a (.wire 14 3) ∧
    rangeCheck (a (.wire 9 15)) 52 ∧
    a (.wire 3 7) = bnot (a (.wire 1 43)) ∧
    a (.wire 3 31) = bnot (a (.wire 2 27)) ∧
    a (.wire 14 11) = band (a (.wire 3 7)) (a (.wire 3 31)) ∧
    IsEqual (a (.virt 6585)) (a (.virt 13180)) (a (.virt 13246)) (a (.virt 13247)) ∧
    IsEqual (a (.virt 6586)) (a (.virt 13181)) (a (.virt 13248)) (a (.virt 13249)) ∧
    IsEqual (a (.virt 6587)) (a (.virt 13182)) (a (.virt 13250)) (a (.virt 13251)) ∧
    IsEqual (a (.virt 6588)) (a (.virt 13183)) (a (.virt 13252)) (a (.virt 13253)) ∧
    a (.wire 14 47) = band (a (.virt 13246)) (a (.virt 13248)) ∧
    a (.wire 14 51) = band (a (.virt 13250)) (a (.virt 13252)) ∧
    a (.wire 14 55) = band (a (.wire 14 47)) (a (.wire 14 51)) ∧
    a (.wire 14 59) = band (a (.wire 14 11)) (a (.wire 14 55)) ∧
    a (.wire 14 59) = a (.virt 13210) ∧
    a (.wire 3 31) = bnot (a (.wire 2 27)) ∧
    (a (.wire 17 12) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 0 ∧ a (.wire 17 13) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 1 ∧ a (.wire 17 14) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 2 ∧ a (.wire 17 15) = spongeHash perm [a (.virt 13201), a (.virt 13202), a (.virt 13203), a (.virt 13204)] 3) ∧
    (a (.wire 18 12) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 0 ∧ a (.wire 18 13) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 1 ∧ a (.wire 18 14) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 2 ∧ a (.wire 18 15) = spongeHash perm [a (.wire 17 12), a (.wire 17 13), a (.wire 17 14), a (.wire 17 15)] 3) ∧
    a (.wire 16 11) = bselect (a (.wire 1 43)) (a (.wire 18 12)) (a (.virt 6585)) ∧
    a (.wire 16 19) = bselect (a (.wire 1 43)) (a (.wire 18 13)) (a (.virt 6586)) ∧
    a (.wire 16 27) = bselect (a (.wire 1 43)) (a (.wire 18 14)) (a (.virt 6587)) ∧
    a (.wire 16 35) = bselect (a (.wire 1 43)) (a (.wire 18 15)) (a (.virt 6588)) ∧
    (a (.wire 19 12) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 0 ∧ a (.wire 19 13) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 1 ∧ a (.wire 19 14) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 2 ∧ a (.wire 19 15) = spongeHash perm [a (.virt 13205), a (.virt 13206), a (.virt 13207), a (.virt 13208)] 3) ∧
    (a (.wire 20 12) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 0 ∧ a (.wire 20 13) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 1 ∧ a (.wire 20 14) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 2 ∧ a (.wire 20 15) = spongeHash perm [a (.wire 19 12), a (.wire 19 13), a (.wire 19 14), a (.wire 19 15)] 3) ∧
    a (.wire 16 43) = bselect (a (.wire 2 27)) (a (.wire 20 12)) (a (.virt 13180)) ∧
    a (.wire 16 51) = bselect (a (.wire 2 27)) (a (.wire 20 13)) (a (.virt 13181)) ∧
    a (.wire 16 59) = bselect (a (.wire 2 27)) (a (.wire 20 14)) (a (.virt 13182)) ∧
    a (.wire 21 7) = bselect (a (.wire 2 27)) (a (.wire 20 15)) (a (.virt 13183))) :=
  ⟨⟨privateBatchWrapper2_f0 a h, privateBatchWrapper2_f1 a h, privateBatchWrapper2_f2 a h, privateBatchWrapper2_f3 a h, privateBatchWrapper2_f4 a h, privateBatchWrapper2_f5 a h, privateBatchWrapper2_f6 a h, privateBatchWrapper2_f7 a h, privateBatchWrapper2_f8 a h, privateBatchWrapper2_f9 a h, privateBatchWrapper2_f10 a h, privateBatchWrapper2_f11 a h, privateBatchWrapper2_f12 a h, privateBatchWrapper2_f13 a h, privateBatchWrapper2_f14 a h, privateBatchWrapper2_f15 a h, privateBatchWrapper2_f16 a h, privateBatchWrapper2_f17 a h, privateBatchWrapper2_f18 a h, privateBatchWrapper2_f19 a h, privateBatchWrapper2_f20 a h, privateBatchWrapper2_f21 a h, privateBatchWrapper2_f22 a h, privateBatchWrapper2_f23 a h, privateBatchWrapper2_f24 a h, privateBatchWrapper2_f25 a h, privateBatchWrapper2_f26 a h, privateBatchWrapper2_f27 a h, privateBatchWrapper2_f28 a h, privateBatchWrapper2_f29 a h, privateBatchWrapper2_f30 a h, privateBatchWrapper2_f31 a h⟩, ⟨privateBatchWrapper2_f32 a h, privateBatchWrapper2_f33 a h, privateBatchWrapper2_f34 a h, privateBatchWrapper2_f35 a h, privateBatchWrapper2_f36 a h, privateBatchWrapper2_f37 a h, privateBatchWrapper2_f38 a h, privateBatchWrapper2_f39 a h, privateBatchWrapper2_f40 a h, privateBatchWrapper2_f41 a h, privateBatchWrapper2_f42 a h, privateBatchWrapper2_f43 a h, privateBatchWrapper2_f44 a h, privateBatchWrapper2_f45 a h, privateBatchWrapper2_f46 a h, privateBatchWrapper2_f47 a h, privateBatchWrapper2_f48 a h, privateBatchWrapper2_f49 a h, privateBatchWrapper2_f50 a h, privateBatchWrapper2_f51 a h, privateBatchWrapper2_f52 a h, privateBatchWrapper2_f53 a h, privateBatchWrapper2_f54 a h, privateBatchWrapper2_f55 a h, privateBatchWrapper2_f56 a h, privateBatchWrapper2_f57 a h, privateBatchWrapper2_f58 a h, privateBatchWrapper2_f59 a h, privateBatchWrapper2_f60 a h, privateBatchWrapper2_f61 a h, privateBatchWrapper2_f62 a h, privateBatchWrapper2_f63 a h⟩, ⟨privateBatchWrapper2_f64 a h, privateBatchWrapper2_f65 a h, privateBatchWrapper2_f66 a h, privateBatchWrapper2_f67 a h, privateBatchWrapper2_f68 a h, privateBatchWrapper2_f69 a h, privateBatchWrapper2_f70 a h, privateBatchWrapper2_f71 a h, privateBatchWrapper2_f72 a h, privateBatchWrapper2_f73 a h, privateBatchWrapper2_f74 a h, privateBatchWrapper2_f75 a h, privateBatchWrapper2_f76 a h, privateBatchWrapper2_f77 a h, privateBatchWrapper2_f78 a h, privateBatchWrapper2_f79 a h, privateBatchWrapper2_f80 perm a h hp, privateBatchWrapper2_f81 perm a h hp, privateBatchWrapper2_f82 a h, privateBatchWrapper2_f83 a h, privateBatchWrapper2_f84 a h, privateBatchWrapper2_f85 a h, privateBatchWrapper2_f86 perm a h hp, privateBatchWrapper2_f87 perm a h hp, privateBatchWrapper2_f88 a h, privateBatchWrapper2_f89 a h, privateBatchWrapper2_f90 a h, privateBatchWrapper2_f91 a h⟩⟩

end Plonky2Spec.Generated
