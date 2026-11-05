/-
  AUTO-GENERATED — do not edit by hand.

  Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from the `n = 4` private-batch wrapper trace
  (`qp-zk-circuits/formal/traces/private_batch_wrapper_n4.json`, recorded by
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

/-- `privateBatchWrapper4.copies`, items `0..32`. -/
def privateBatchWrapper4.copies0 : List (Target × Target) := [
    (.virt 23709, .wire 0 0),
    (.virt 23709, .wire 0 1),
    (.virt 23712, .wire 0 2),
    (.virt 23712, .wire 1 0),
    (.virt 5912, .wire 1 1),
    (.virt 23712, .wire 1 2),
    (.virt 5912, .wire 1 4),
    (.virt 23713, .wire 1 5),
    (.virt 5912, .wire 1 6),
    (.wire 1 7, .wire 0 4),
    (.virt 23709, .wire 0 5),
    (.wire 0 3, .wire 0 6),
    (.wire 1 3, .virt 23710),
    (.wire 0 7, .virt 23710),
    (.virt 23709, .wire 0 8),
    (.virt 23709, .wire 0 9),
    (.virt 23714, .wire 0 10),
    (.virt 23714, .wire 1 8),
    (.virt 5913, .wire 1 9),
    (.virt 23714, .wire 1 10),
    (.virt 5913, .wire 1 12),
    (.virt 23715, .wire 1 13),
    (.virt 5913, .wire 1 14),
    (.wire 1 15, .wire 0 12),
    (.virt 23709, .wire 0 13),
    (.wire 0 11, .wire 0 14),
    (.wire 1 11, .virt 23710),
    (.wire 0 15, .virt 23710),
    (.virt 23709, .wire 0 16),
    (.virt 23709, .wire 0 17),
    (.virt 23716, .wire 0 18),
    (.virt 23716, .wire 1 16)
  ]

/-- `privateBatchWrapper4.copies`, items `32..64`. -/
def privateBatchWrapper4.copies1 : List (Target × Target) := [
    (.virt 5914, .wire 1 17),
    (.virt 23716, .wire 1 18),
    (.virt 5914, .wire 1 20),
    (.virt 23717, .wire 1 21),
    (.virt 5914, .wire 1 22),
    (.wire 1 23, .wire 0 20),
    (.virt 23709, .wire 0 21),
    (.wire 0 19, .wire 0 22),
    (.wire 1 19, .virt 23710),
    (.wire 0 23, .virt 23710),
    (.virt 23709, .wire 0 24),
    (.virt 23709, .wire 0 25),
    (.virt 23718, .wire 0 26),
    (.virt 23718, .wire 1 24),
    (.virt 5915, .wire 1 25),
    (.virt 23718, .wire 1 26),
    (.virt 5915, .wire 1 28),
    (.virt 23719, .wire 1 29),
    (.virt 5915, .wire 1 30),
    (.wire 1 31, .wire 0 28),
    (.virt 23709, .wire 0 29),
    (.wire 0 27, .wire 0 30),
    (.wire 1 27, .virt 23710),
    (.wire 0 31, .virt 23710),
    (.virt 23712, .wire 1 32),
    (.virt 23714, .wire 1 33),
    (.virt 23712, .wire 1 34),
    (.virt 23716, .wire 1 36),
    (.virt 23718, .wire 1 37),
    (.virt 23716, .wire 1 38),
    (.wire 1 35, .wire 1 40),
    (.wire 1 39, .wire 1 41)
  ]

/-- `privateBatchWrapper4.copies`, items `64..96`. -/
def privateBatchWrapper4.copies2 : List (Target × Target) := [
    (.wire 1 35, .wire 1 42),
    (.virt 23709, .wire 0 32),
    (.virt 23709, .wire 0 33),
    (.virt 23720, .wire 0 34),
    (.virt 23720, .wire 1 44),
    (.virt 11830, .wire 1 45),
    (.virt 23720, .wire 1 46),
    (.virt 11830, .wire 1 48),
    (.virt 23721, .wire 1 49),
    (.virt 11830, .wire 1 50),
    (.wire 1 51, .wire 0 36),
    (.virt 23709, .wire 0 37),
    (.wire 0 35, .wire 0 38),
    (.wire 1 47, .virt 23710),
    (.wire 0 39, .virt 23710),
    (.virt 23709, .wire 0 40),
    (.virt 23709, .wire 0 41),
    (.virt 23722, .wire 0 42),
    (.virt 23722, .wire 1 52),
    (.virt 11831, .wire 1 53),
    (.virt 23722, .wire 1 54),
    (.virt 11831, .wire 1 56),
    (.virt 23723, .wire 1 57),
    (.virt 11831, .wire 1 58),
    (.wire 1 59, .wire 0 44),
    (.virt 23709, .wire 0 45),
    (.wire 0 43, .wire 0 46),
    (.wire 1 55, .virt 23710),
    (.wire 0 47, .virt 23710),
    (.virt 23709, .wire 0 48),
    (.virt 23709, .wire 0 49),
    (.virt 23724, .wire 0 50)
  ]

/-- `privateBatchWrapper4.copies`, items `96..128`. -/
def privateBatchWrapper4.copies3 : List (Target × Target) := [
    (.virt 23724, .wire 2 0),
    (.virt 11832, .wire 2 1),
    (.virt 23724, .wire 2 2),
    (.virt 11832, .wire 2 4),
    (.virt 23725, .wire 2 5),
    (.virt 11832, .wire 2 6),
    (.wire 2 7, .wire 0 52),
    (.virt 23709, .wire 0 53),
    (.wire 0 51, .wire 0 54),
    (.wire 2 3, .virt 23710),
    (.wire 0 55, .virt 23710),
    (.virt 23709, .wire 0 56),
    (.virt 23709, .wire 0 57),
    (.virt 23726, .wire 0 58),
    (.virt 23726, .wire 2 8),
    (.virt 11833, .wire 2 9),
    (.virt 23726, .wire 2 10),
    (.virt 11833, .wire 2 12),
    (.virt 23727, .wire 2 13),
    (.virt 11833, .wire 2 14),
    (.wire 2 15, .wire 3 0),
    (.virt 23709, .wire 3 1),
    (.wire 0 59, .wire 3 2),
    (.wire 2 11, .virt 23710),
    (.wire 3 3, .virt 23710),
    (.virt 23720, .wire 2 16),
    (.virt 23722, .wire 2 17),
    (.virt 23720, .wire 2 18),
    (.virt 23724, .wire 2 20),
    (.virt 23726, .wire 2 21),
    (.virt 23724, .wire 2 22),
    (.wire 2 19, .wire 2 24)
  ]

/-- `privateBatchWrapper4.copies`, items `128..160`. -/
def privateBatchWrapper4.copies4 : List (Target × Target) := [
    (.wire 2 23, .wire 2 25),
    (.wire 2 19, .wire 2 26),
    (.virt 23709, .wire 3 4),
    (.virt 23709, .wire 3 5),
    (.virt 23728, .wire 3 6),
    (.virt 23728, .wire 2 28),
    (.virt 17748, .wire 2 29),
    (.virt 23728, .wire 2 30),
    (.virt 17748, .wire 2 32),
    (.virt 23729, .wire 2 33),
    (.virt 17748, .wire 2 34),
    (.wire 2 35, .wire 3 8),
    (.virt 23709, .wire 3 9),
    (.wire 3 7, .wire 3 10),
    (.wire 2 31, .virt 23710),
    (.wire 3 11, .virt 23710),
    (.virt 23709, .wire 3 12),
    (.virt 23709, .wire 3 13),
    (.virt 23730, .wire 3 14),
    (.virt 23730, .wire 2 36),
    (.virt 17749, .wire 2 37),
    (.virt 23730, .wire 2 38),
    (.virt 17749, .wire 2 40),
    (.virt 23731, .wire 2 41),
    (.virt 17749, .wire 2 42),
    (.wire 2 43, .wire 3 16),
    (.virt 23709, .wire 3 17),
    (.wire 3 15, .wire 3 18),
    (.wire 2 39, .virt 23710),
    (.wire 3 19, .virt 23710),
    (.virt 23709, .wire 3 20),
    (.virt 23709, .wire 3 21)
  ]

/-- `privateBatchWrapper4.copies`, items `160..192`. -/
def privateBatchWrapper4.copies5 : List (Target × Target) := [
    (.virt 23732, .wire 3 22),
    (.virt 23732, .wire 2 44),
    (.virt 17750, .wire 2 45),
    (.virt 23732, .wire 2 46),
    (.virt 17750, .wire 2 48),
    (.virt 23733, .wire 2 49),
    (.virt 17750, .wire 2 50),
    (.wire 2 51, .wire 3 24),
    (.virt 23709, .wire 3 25),
    (.wire 3 23, .wire 3 26),
    (.wire 2 47, .virt 23710),
    (.wire 3 27, .virt 23710),
    (.virt 23709, .wire 3 28),
    (.virt 23709, .wire 3 29),
    (.virt 23734, .wire 3 30),
    (.virt 23734, .wire 2 52),
    (.virt 17751, .wire 2 53),
    (.virt 23734, .wire 2 54),
    (.virt 17751, .wire 2 56),
    (.virt 23735, .wire 2 57),
    (.virt 17751, .wire 2 58),
    (.wire 2 59, .wire 3 32),
    (.virt 23709, .wire 3 33),
    (.wire 3 31, .wire 3 34),
    (.wire 2 55, .virt 23710),
    (.wire 3 35, .virt 23710),
    (.virt 23728, .wire 4 0),
    (.virt 23730, .wire 4 1),
    (.virt 23728, .wire 4 2),
    (.virt 23732, .wire 4 4),
    (.virt 23734, .wire 4 5),
    (.virt 23732, .wire 4 6)
  ]

/-- `privateBatchWrapper4.copies`, items `192..224`. -/
def privateBatchWrapper4.copies6 : List (Target × Target) := [
    (.wire 4 3, .wire 4 8),
    (.wire 4 7, .wire 4 9),
    (.wire 4 3, .wire 4 10),
    (.virt 23709, .wire 3 36),
    (.virt 23709, .wire 3 37),
    (.virt 23736, .wire 3 38),
    (.virt 23736, .wire 4 12),
    (.virt 23666, .wire 4 13),
    (.virt 23736, .wire 4 14),
    (.virt 23666, .wire 4 16),
    (.virt 23737, .wire 4 17),
    (.virt 23666, .wire 4 18),
    (.wire 4 19, .wire 3 40),
    (.virt 23709, .wire 3 41),
    (.wire 3 39, .wire 3 42),
    (.wire 4 15, .virt 23710),
    (.wire 3 43, .virt 23710),
    (.virt 23709, .wire 3 44),
    (.virt 23709, .wire 3 45),
    (.virt 23738, .wire 3 46),
    (.virt 23738, .wire 4 20),
    (.virt 23667, .wire 4 21),
    (.virt 23738, .wire 4 22),
    (.virt 23667, .wire 4 24),
    (.virt 23739, .wire 4 25),
    (.virt 23667, .wire 4 26),
    (.wire 4 27, .wire 3 48),
    (.virt 23709, .wire 3 49),
    (.wire 3 47, .wire 3 50),
    (.wire 4 23, .virt 23710),
    (.wire 3 51, .virt 23710),
    (.virt 23709, .wire 3 52)
  ]

/-- `privateBatchWrapper4.copies`, items `224..256`. -/
def privateBatchWrapper4.copies7 : List (Target × Target) := [
    (.virt 23709, .wire 3 53),
    (.virt 23740, .wire 3 54),
    (.virt 23740, .wire 4 28),
    (.virt 23668, .wire 4 29),
    (.virt 23740, .wire 4 30),
    (.virt 23668, .wire 4 32),
    (.virt 23741, .wire 4 33),
    (.virt 23668, .wire 4 34),
    (.wire 4 35, .wire 3 56),
    (.virt 23709, .wire 3 57),
    (.wire 3 55, .wire 3 58),
    (.wire 4 31, .virt 23710),
    (.wire 3 59, .virt 23710),
    (.virt 23709, .wire 5 0),
    (.virt 23709, .wire 5 1),
    (.virt 23742, .wire 5 2),
    (.virt 23742, .wire 4 36),
    (.virt 23669, .wire 4 37),
    (.virt 23742, .wire 4 38),
    (.virt 23669, .wire 4 40),
    (.virt 23743, .wire 4 41),
    (.virt 23669, .wire 4 42),
    (.wire 4 43, .wire 5 4),
    (.virt 23709, .wire 5 5),
    (.wire 5 3, .wire 5 6),
    (.wire 4 39, .virt 23710),
    (.wire 5 7, .virt 23710),
    (.virt 23736, .wire 4 44),
    (.virt 23738, .wire 4 45),
    (.virt 23736, .wire 4 46),
    (.virt 23740, .wire 4 48),
    (.virt 23742, .wire 4 49)
  ]

/-- `privateBatchWrapper4.copies`, items `256..288`. -/
def privateBatchWrapper4.copies8 : List (Target × Target) := [
    (.virt 23740, .wire 4 50),
    (.wire 4 47, .wire 4 52),
    (.wire 4 51, .wire 4 53),
    (.wire 4 47, .wire 4 54),
    (.virt 23709, .wire 5 8),
    (.virt 23709, .wire 5 9),
    (.wire 1 43, .wire 5 10),
    (.wire 5 11, .wire 5 12),
    (.virt 5912, .wire 5 13),
    (.virt 23710, .wire 5 14),
    (.wire 5 11, .wire 5 16),
    (.virt 5913, .wire 5 17),
    (.virt 23710, .wire 5 18),
    (.wire 5 11, .wire 5 20),
    (.virt 5914, .wire 5 21),
    (.virt 23710, .wire 5 22),
    (.wire 5 11, .wire 5 24),
    (.virt 5915, .wire 5 25),
    (.virt 23710, .wire 5 26),
    (.wire 5 11, .wire 5 28),
    (.virt 5916, .wire 5 29),
    (.virt 23710, .wire 5 30),
    (.virt 23709, .wire 5 32),
    (.virt 23709, .wire 5 33),
    (.wire 2 27, .wire 5 34),
    (.virt 23709, .wire 5 36),
    (.virt 23709, .wire 5 37),
    (.wire 5 11, .wire 5 38),
    (.wire 5 35, .wire 4 56),
    (.wire 5 39, .wire 4 57),
    (.wire 5 35, .wire 4 58),
    (.wire 4 59, .wire 5 40)
  ]

/-- `privateBatchWrapper4.copies`, items `288..320`. -/
def privateBatchWrapper4.copies9 : List (Target × Target) := [
    (.wire 5 15, .wire 5 41),
    (.wire 5 15, .wire 5 42),
    (.wire 4 59, .wire 5 44),
    (.virt 11830, .wire 5 45),
    (.wire 5 43, .wire 5 46),
    (.wire 4 59, .wire 5 48),
    (.wire 5 19, .wire 5 49),
    (.wire 5 19, .wire 5 50),
    (.wire 4 59, .wire 5 52),
    (.virt 11831, .wire 5 53),
    (.wire 5 51, .wire 5 54),
    (.wire 4 59, .wire 5 56),
    (.wire 5 23, .wire 5 57),
    (.wire 5 23, .wire 5 58),
    (.wire 4 59, .wire 6 0),
    (.virt 11832, .wire 6 1),
    (.wire 5 59, .wire 6 2),
    (.wire 4 59, .wire 6 4),
    (.wire 5 27, .wire 6 5),
    (.wire 5 27, .wire 6 6),
    (.wire 4 59, .wire 6 8),
    (.virt 11833, .wire 6 9),
    (.wire 6 7, .wire 6 10),
    (.wire 4 59, .wire 6 12),
    (.wire 5 31, .wire 6 13),
    (.wire 5 31, .wire 6 14),
    (.wire 4 59, .wire 6 16),
    (.virt 11834, .wire 6 17),
    (.wire 6 15, .wire 6 18),
    (.wire 5 11, .wire 7 0),
    (.wire 5 35, .wire 7 1),
    (.wire 5 11, .wire 7 2)
  ]

/-- `privateBatchWrapper4.copies`, items `320..352`. -/
def privateBatchWrapper4.copies10 : List (Target × Target) := [
    (.wire 7 3, .wire 8 0),
    (.virt 23709, .wire 8 1),
    (.wire 5 35, .wire 8 2),
    (.virt 23709, .wire 6 20),
    (.virt 23709, .wire 6 21),
    (.wire 4 11, .wire 6 22),
    (.virt 23709, .wire 6 24),
    (.virt 23709, .wire 6 25),
    (.wire 8 3, .wire 6 26),
    (.wire 6 23, .wire 9 0),
    (.wire 6 27, .wire 9 1),
    (.wire 6 23, .wire 9 2),
    (.wire 9 3, .wire 6 28),
    (.wire 5 47, .wire 6 29),
    (.wire 5 47, .wire 6 30),
    (.wire 9 3, .wire 6 32),
    (.virt 17748, .wire 6 33),
    (.wire 6 31, .wire 6 34),
    (.wire 9 3, .wire 6 36),
    (.wire 5 55, .wire 6 37),
    (.wire 5 55, .wire 6 38),
    (.wire 9 3, .wire 6 40),
    (.virt 17749, .wire 6 41),
    (.wire 6 39, .wire 6 42),
    (.wire 9 3, .wire 6 44),
    (.wire 6 3, .wire 6 45),
    (.wire 6 3, .wire 6 46),
    (.wire 9 3, .wire 6 48),
    (.virt 17750, .wire 6 49),
    (.wire 6 47, .wire 6 50),
    (.wire 9 3, .wire 6 52),
    (.wire 6 11, .wire 6 53)
  ]

/-- `privateBatchWrapper4.copies`, items `352..384`. -/
def privateBatchWrapper4.copies11 : List (Target × Target) := [
    (.wire 6 11, .wire 6 54),
    (.wire 9 3, .wire 6 56),
    (.virt 17751, .wire 6 57),
    (.wire 6 55, .wire 6 58),
    (.wire 9 3, .wire 10 0),
    (.wire 6 19, .wire 10 1),
    (.wire 6 19, .wire 10 2),
    (.wire 9 3, .wire 10 4),
    (.virt 17752, .wire 10 5),
    (.wire 10 3, .wire 10 6),
    (.wire 8 3, .wire 7 4),
    (.wire 6 23, .wire 7 5),
    (.wire 8 3, .wire 7 6),
    (.wire 7 7, .wire 8 4),
    (.virt 23709, .wire 8 5),
    (.wire 6 23, .wire 8 6),
    (.virt 23709, .wire 10 8),
    (.virt 23709, .wire 10 9),
    (.wire 4 55, .wire 10 10),
    (.virt 23709, .wire 10 12),
    (.virt 23709, .wire 10 13),
    (.wire 8 7, .wire 10 14),
    (.wire 10 11, .wire 9 4),
    (.wire 10 15, .wire 9 5),
    (.wire 10 11, .wire 9 6),
    (.wire 9 7, .wire 10 16),
    (.wire 6 35, .wire 10 17),
    (.wire 6 35, .wire 10 18),
    (.wire 9 7, .wire 10 20),
    (.virt 23666, .wire 10 21),
    (.wire 10 19, .wire 10 22),
    (.wire 9 7, .wire 10 24)
  ]

/-- `privateBatchWrapper4.copies`, items `384..416`. -/
def privateBatchWrapper4.copies12 : List (Target × Target) := [
    (.wire 6 43, .wire 10 25),
    (.wire 6 43, .wire 10 26),
    (.wire 9 7, .wire 10 28),
    (.virt 23667, .wire 10 29),
    (.wire 10 27, .wire 10 30),
    (.wire 9 7, .wire 10 32),
    (.wire 6 51, .wire 10 33),
    (.wire 6 51, .wire 10 34),
    (.wire 9 7, .wire 10 36),
    (.virt 23668, .wire 10 37),
    (.wire 10 35, .wire 10 38),
    (.wire 9 7, .wire 10 40),
    (.wire 6 59, .wire 10 41),
    (.wire 6 59, .wire 10 42),
    (.wire 9 7, .wire 10 44),
    (.virt 23669, .wire 10 45),
    (.wire 10 43, .wire 10 46),
    (.wire 9 7, .wire 10 48),
    (.wire 10 7, .wire 10 49),
    (.wire 10 7, .wire 10 50),
    (.wire 9 7, .wire 10 52),
    (.virt 23670, .wire 10 53),
    (.wire 10 51, .wire 10 54),
    (.wire 8 7, .wire 7 8),
    (.wire 10 11, .wire 7 9),
    (.wire 8 7, .wire 7 10),
    (.wire 7 11, .wire 8 8),
    (.virt 23709, .wire 8 9),
    (.wire 10 11, .wire 8 10),
    (.virt 23709, .wire 10 56),
    (.virt 23709, .wire 10 57),
    (.virt 23744, .wire 10 58)
  ]

/-- `privateBatchWrapper4.copies`, items `416..448`. -/
def privateBatchWrapper4.copies13 : List (Target × Target) := [
    (.virt 5912, .wire 11 0),
    (.virt 23709, .wire 11 1),
    (.wire 10 23, .wire 11 2),
    (.virt 23744, .wire 9 8),
    (.wire 11 3, .wire 9 9),
    (.virt 23744, .wire 9 10),
    (.wire 11 3, .wire 9 12),
    (.virt 23745, .wire 9 13),
    (.wire 11 3, .wire 9 14),
    (.wire 9 15, .wire 11 4),
    (.virt 23709, .wire 11 5),
    (.wire 10 59, .wire 11 6),
    (.wire 9 11, .virt 23710),
    (.wire 11 7, .virt 23710),
    (.virt 23709, .wire 11 8),
    (.virt 23709, .wire 11 9),
    (.virt 23746, .wire 11 10),
    (.virt 5913, .wire 11 12),
    (.virt 23709, .wire 11 13),
    (.wire 10 31, .wire 11 14),
    (.virt 23746, .wire 9 16),
    (.wire 11 15, .wire 9 17),
    (.virt 23746, .wire 9 18),
    (.wire 11 15, .wire 9 20),
    (.virt 23747, .wire 9 21),
    (.wire 11 15, .wire 9 22),
    (.wire 9 23, .wire 11 16),
    (.virt 23709, .wire 11 17),
    (.wire 11 11, .wire 11 18),
    (.wire 9 19, .virt 23710),
    (.wire 11 19, .virt 23710),
    (.virt 23709, .wire 11 20)
  ]

/-- `privateBatchWrapper4.copies`, items `448..480`. -/
def privateBatchWrapper4.copies14 : List (Target × Target) := [
    (.virt 23709, .wire 11 21),
    (.virt 23748, .wire 11 22),
    (.virt 5914, .wire 11 24),
    (.virt 23709, .wire 11 25),
    (.wire 10 39, .wire 11 26),
    (.virt 23748, .wire 9 24),
    (.wire 11 27, .wire 9 25),
    (.virt 23748, .wire 9 26),
    (.wire 11 27, .wire 9 28),
    (.virt 23749, .wire 9 29),
    (.wire 11 27, .wire 9 30),
    (.wire 9 31, .wire 11 28),
    (.virt 23709, .wire 11 29),
    (.wire 11 23, .wire 11 30),
    (.wire 9 27, .virt 23710),
    (.wire 11 31, .virt 23710),
    (.virt 23709, .wire 11 32),
    (.virt 23709, .wire 11 33),
    (.virt 23750, .wire 11 34),
    (.virt 5915, .wire 11 36),
    (.virt 23709, .wire 11 37),
    (.wire 10 47, .wire 11 38),
    (.virt 23750, .wire 9 32),
    (.wire 11 39, .wire 9 33),
    (.virt 23750, .wire 9 34),
    (.wire 11 39, .wire 9 36),
    (.virt 23751, .wire 9 37),
    (.wire 11 39, .wire 9 38),
    (.wire 9 39, .wire 11 40),
    (.virt 23709, .wire 11 41),
    (.wire 11 35, .wire 11 42),
    (.wire 9 35, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `480..512`. -/
def privateBatchWrapper4.copies15 : List (Target × Target) := [
    (.wire 11 43, .virt 23710),
    (.virt 23744, .wire 9 40),
    (.virt 23746, .wire 9 41),
    (.virt 23744, .wire 9 42),
    (.virt 23748, .wire 9 44),
    (.virt 23750, .wire 9 45),
    (.virt 23748, .wire 9 46),
    (.wire 9 43, .wire 9 48),
    (.wire 9 47, .wire 9 49),
    (.wire 9 43, .wire 9 50),
    (.wire 1 43, .wire 7 12),
    (.wire 9 51, .wire 7 13),
    (.wire 1 43, .wire 7 14),
    (.wire 7 15, .wire 8 12),
    (.virt 23709, .wire 8 13),
    (.wire 9 51, .wire 8 14),
    (.wire 8 15, .virt 23709),
    (.virt 5907, .virt 5907),
    (.virt 23709, .wire 11 44),
    (.virt 23709, .wire 11 45),
    (.virt 23752, .wire 11 46),
    (.virt 11830, .wire 11 48),
    (.virt 23709, .wire 11 49),
    (.wire 10 23, .wire 11 50),
    (.virt 23752, .wire 9 52),
    (.wire 11 51, .wire 9 53),
    (.virt 23752, .wire 9 54),
    (.wire 11 51, .wire 9 56),
    (.virt 23753, .wire 9 57),
    (.wire 11 51, .wire 9 58),
    (.wire 9 59, .wire 11 52),
    (.virt 23709, .wire 11 53)
  ]

/-- `privateBatchWrapper4.copies`, items `512..544`. -/
def privateBatchWrapper4.copies16 : List (Target × Target) := [
    (.wire 11 47, .wire 11 54),
    (.wire 9 55, .virt 23710),
    (.wire 11 55, .virt 23710),
    (.virt 23709, .wire 11 56),
    (.virt 23709, .wire 11 57),
    (.virt 23754, .wire 11 58),
    (.virt 11831, .wire 12 0),
    (.virt 23709, .wire 12 1),
    (.wire 10 31, .wire 12 2),
    (.virt 23754, .wire 13 0),
    (.wire 12 3, .wire 13 1),
    (.virt 23754, .wire 13 2),
    (.wire 12 3, .wire 13 4),
    (.virt 23755, .wire 13 5),
    (.wire 12 3, .wire 13 6),
    (.wire 13 7, .wire 12 4),
    (.virt 23709, .wire 12 5),
    (.wire 11 59, .wire 12 6),
    (.wire 13 3, .virt 23710),
    (.wire 12 7, .virt 23710),
    (.virt 23709, .wire 12 8),
    (.virt 23709, .wire 12 9),
    (.virt 23756, .wire 12 10),
    (.virt 11832, .wire 12 12),
    (.virt 23709, .wire 12 13),
    (.wire 10 39, .wire 12 14),
    (.virt 23756, .wire 13 8),
    (.wire 12 15, .wire 13 9),
    (.virt 23756, .wire 13 10),
    (.wire 12 15, .wire 13 12),
    (.virt 23757, .wire 13 13),
    (.wire 12 15, .wire 13 14)
  ]

/-- `privateBatchWrapper4.copies`, items `544..576`. -/
def privateBatchWrapper4.copies17 : List (Target × Target) := [
    (.wire 13 15, .wire 12 16),
    (.virt 23709, .wire 12 17),
    (.wire 12 11, .wire 12 18),
    (.wire 13 11, .virt 23710),
    (.wire 12 19, .virt 23710),
    (.virt 23709, .wire 12 20),
    (.virt 23709, .wire 12 21),
    (.virt 23758, .wire 12 22),
    (.virt 11833, .wire 12 24),
    (.virt 23709, .wire 12 25),
    (.wire 10 47, .wire 12 26),
    (.virt 23758, .wire 13 16),
    (.wire 12 27, .wire 13 17),
    (.virt 23758, .wire 13 18),
    (.wire 12 27, .wire 13 20),
    (.virt 23759, .wire 13 21),
    (.wire 12 27, .wire 13 22),
    (.wire 13 23, .wire 12 28),
    (.virt 23709, .wire 12 29),
    (.wire 12 23, .wire 12 30),
    (.wire 13 19, .virt 23710),
    (.wire 12 31, .virt 23710),
    (.virt 23752, .wire 13 24),
    (.virt 23754, .wire 13 25),
    (.virt 23752, .wire 13 26),
    (.virt 23756, .wire 13 28),
    (.virt 23758, .wire 13 29),
    (.virt 23756, .wire 13 30),
    (.wire 13 27, .wire 13 32),
    (.wire 13 31, .wire 13 33),
    (.wire 13 27, .wire 13 34),
    (.wire 2 27, .wire 7 16)
  ]

/-- `privateBatchWrapper4.copies`, items `576..608`. -/
def privateBatchWrapper4.copies18 : List (Target × Target) := [
    (.wire 13 35, .wire 7 17),
    (.wire 2 27, .wire 7 18),
    (.wire 7 19, .wire 8 16),
    (.virt 23709, .wire 8 17),
    (.wire 13 35, .wire 8 18),
    (.wire 8 19, .virt 23709),
    (.virt 11825, .virt 5907),
    (.virt 23709, .wire 12 32),
    (.virt 23709, .wire 12 33),
    (.virt 23760, .wire 12 34),
    (.virt 17748, .wire 12 36),
    (.virt 23709, .wire 12 37),
    (.wire 10 23, .wire 12 38),
    (.virt 23760, .wire 13 36),
    (.wire 12 39, .wire 13 37),
    (.virt 23760, .wire 13 38),
    (.wire 12 39, .wire 13 40),
    (.virt 23761, .wire 13 41),
    (.wire 12 39, .wire 13 42),
    (.wire 13 43, .wire 12 40),
    (.virt 23709, .wire 12 41),
    (.wire 12 35, .wire 12 42),
    (.wire 13 39, .virt 23710),
    (.wire 12 43, .virt 23710),
    (.virt 23709, .wire 12 44),
    (.virt 23709, .wire 12 45),
    (.virt 23762, .wire 12 46),
    (.virt 17749, .wire 12 48),
    (.virt 23709, .wire 12 49),
    (.wire 10 31, .wire 12 50),
    (.virt 23762, .wire 13 44),
    (.wire 12 51, .wire 13 45)
  ]

/-- `privateBatchWrapper4.copies`, items `608..640`. -/
def privateBatchWrapper4.copies19 : List (Target × Target) := [
    (.virt 23762, .wire 13 46),
    (.wire 12 51, .wire 13 48),
    (.virt 23763, .wire 13 49),
    (.wire 12 51, .wire 13 50),
    (.wire 13 51, .wire 12 52),
    (.virt 23709, .wire 12 53),
    (.wire 12 47, .wire 12 54),
    (.wire 13 47, .virt 23710),
    (.wire 12 55, .virt 23710),
    (.virt 23709, .wire 12 56),
    (.virt 23709, .wire 12 57),
    (.virt 23764, .wire 12 58),
    (.virt 17750, .wire 14 0),
    (.virt 23709, .wire 14 1),
    (.wire 10 39, .wire 14 2),
    (.virt 23764, .wire 13 52),
    (.wire 14 3, .wire 13 53),
    (.virt 23764, .wire 13 54),
    (.wire 14 3, .wire 13 56),
    (.virt 23765, .wire 13 57),
    (.wire 14 3, .wire 13 58),
    (.wire 13 59, .wire 14 4),
    (.virt 23709, .wire 14 5),
    (.wire 12 59, .wire 14 6),
    (.wire 13 55, .virt 23710),
    (.wire 14 7, .virt 23710),
    (.virt 23709, .wire 14 8),
    (.virt 23709, .wire 14 9),
    (.virt 23766, .wire 14 10),
    (.virt 17751, .wire 14 12),
    (.virt 23709, .wire 14 13),
    (.wire 10 47, .wire 14 14)
  ]

/-- `privateBatchWrapper4.copies`, items `640..672`. -/
def privateBatchWrapper4.copies20 : List (Target × Target) := [
    (.virt 23766, .wire 15 0),
    (.wire 14 15, .wire 15 1),
    (.virt 23766, .wire 15 2),
    (.wire 14 15, .wire 15 4),
    (.virt 23767, .wire 15 5),
    (.wire 14 15, .wire 15 6),
    (.wire 15 7, .wire 14 16),
    (.virt 23709, .wire 14 17),
    (.wire 14 11, .wire 14 18),
    (.wire 15 3, .virt 23710),
    (.wire 14 19, .virt 23710),
    (.virt 23760, .wire 15 8),
    (.virt 23762, .wire 15 9),
    (.virt 23760, .wire 15 10),
    (.virt 23764, .wire 15 12),
    (.virt 23766, .wire 15 13),
    (.virt 23764, .wire 15 14),
    (.wire 15 11, .wire 15 16),
    (.wire 15 15, .wire 15 17),
    (.wire 15 11, .wire 15 18),
    (.wire 4 11, .wire 7 20),
    (.wire 15 19, .wire 7 21),
    (.wire 4 11, .wire 7 22),
    (.wire 7 23, .wire 8 20),
    (.virt 23709, .wire 8 21),
    (.wire 15 19, .wire 8 22),
    (.wire 8 23, .virt 23709),
    (.virt 17743, .virt 5907),
    (.virt 23709, .wire 14 20),
    (.virt 23709, .wire 14 21),
    (.virt 23768, .wire 14 22),
    (.virt 23666, .wire 14 24)
  ]

/-- `privateBatchWrapper4.copies`, items `672..704`. -/
def privateBatchWrapper4.copies21 : List (Target × Target) := [
    (.virt 23709, .wire 14 25),
    (.wire 10 23, .wire 14 26),
    (.virt 23768, .wire 15 20),
    (.wire 14 27, .wire 15 21),
    (.virt 23768, .wire 15 22),
    (.wire 14 27, .wire 15 24),
    (.virt 23769, .wire 15 25),
    (.wire 14 27, .wire 15 26),
    (.wire 15 27, .wire 14 28),
    (.virt 23709, .wire 14 29),
    (.wire 14 23, .wire 14 30),
    (.wire 15 23, .virt 23710),
    (.wire 14 31, .virt 23710),
    (.virt 23709, .wire 14 32),
    (.virt 23709, .wire 14 33),
    (.virt 23770, .wire 14 34),
    (.virt 23667, .wire 14 36),
    (.virt 23709, .wire 14 37),
    (.wire 10 31, .wire 14 38),
    (.virt 23770, .wire 15 28),
    (.wire 14 39, .wire 15 29),
    (.virt 23770, .wire 15 30),
    (.wire 14 39, .wire 15 32),
    (.virt 23771, .wire 15 33),
    (.wire 14 39, .wire 15 34),
    (.wire 15 35, .wire 14 40),
    (.virt 23709, .wire 14 41),
    (.wire 14 35, .wire 14 42),
    (.wire 15 31, .virt 23710),
    (.wire 14 43, .virt 23710),
    (.virt 23709, .wire 14 44),
    (.virt 23709, .wire 14 45)
  ]

/-- `privateBatchWrapper4.copies`, items `704..736`. -/
def privateBatchWrapper4.copies22 : List (Target × Target) := [
    (.virt 23772, .wire 14 46),
    (.virt 23668, .wire 14 48),
    (.virt 23709, .wire 14 49),
    (.wire 10 39, .wire 14 50),
    (.virt 23772, .wire 15 36),
    (.wire 14 51, .wire 15 37),
    (.virt 23772, .wire 15 38),
    (.wire 14 51, .wire 15 40),
    (.virt 23773, .wire 15 41),
    (.wire 14 51, .wire 15 42),
    (.wire 15 43, .wire 14 52),
    (.virt 23709, .wire 14 53),
    (.wire 14 47, .wire 14 54),
    (.wire 15 39, .virt 23710),
    (.wire 14 55, .virt 23710),
    (.virt 23709, .wire 14 56),
    (.virt 23709, .wire 14 57),
    (.virt 23774, .wire 14 58),
    (.virt 23669, .wire 16 0),
    (.virt 23709, .wire 16 1),
    (.wire 10 47, .wire 16 2),
    (.virt 23774, .wire 15 44),
    (.wire 16 3, .wire 15 45),
    (.virt 23774, .wire 15 46),
    (.wire 16 3, .wire 15 48),
    (.virt 23775, .wire 15 49),
    (.wire 16 3, .wire 15 50),
    (.wire 15 51, .wire 16 4),
    (.virt 23709, .wire 16 5),
    (.wire 14 59, .wire 16 6),
    (.wire 15 47, .virt 23710),
    (.wire 16 7, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `736..768`. -/
def privateBatchWrapper4.copies23 : List (Target × Target) := [
    (.virt 23768, .wire 15 52),
    (.virt 23770, .wire 15 53),
    (.virt 23768, .wire 15 54),
    (.virt 23772, .wire 15 56),
    (.virt 23774, .wire 15 57),
    (.virt 23772, .wire 15 58),
    (.wire 15 55, .wire 17 0),
    (.wire 15 59, .wire 17 1),
    (.wire 15 55, .wire 17 2),
    (.wire 4 55, .wire 7 24),
    (.wire 17 3, .wire 7 25),
    (.wire 4 55, .wire 7 26),
    (.wire 7 27, .wire 8 24),
    (.virt 23709, .wire 8 25),
    (.wire 17 3, .wire 8 26),
    (.wire 8 27, .virt 23709),
    (.virt 23661, .virt 5907),
    (.wire 1 43, .wire 16 8),
    (.virt 5917, .wire 16 9),
    (.virt 5917, .wire 16 10),
    (.wire 1 43, .wire 16 12),
    (.virt 23710, .wire 16 13),
    (.wire 16 11, .wire 16 14),
    (.wire 2 27, .wire 16 16),
    (.virt 11835, .wire 16 17),
    (.virt 11835, .wire 16 18),
    (.wire 2 27, .wire 16 20),
    (.virt 23710, .wire 16 21),
    (.wire 16 19, .wire 16 22),
    (.wire 16 15, .wire 8 28),
    (.virt 23709, .wire 8 29),
    (.wire 16 23, .wire 8 30)
  ]

/-- `privateBatchWrapper4.copies`, items `768..800`. -/
def privateBatchWrapper4.copies24 : List (Target × Target) := [
    (.wire 4 11, .wire 16 24),
    (.virt 17753, .wire 16 25),
    (.virt 17753, .wire 16 26),
    (.wire 4 11, .wire 16 28),
    (.virt 23710, .wire 16 29),
    (.wire 16 27, .wire 16 30),
    (.wire 8 31, .wire 8 32),
    (.virt 23709, .wire 8 33),
    (.wire 16 31, .wire 8 34),
    (.wire 4 55, .wire 16 32),
    (.virt 23671, .wire 16 33),
    (.virt 23671, .wire 16 34),
    (.wire 4 55, .wire 16 36),
    (.virt 23710, .wire 16 37),
    (.wire 16 35, .wire 16 38),
    (.wire 8 35, .wire 8 36),
    (.virt 23709, .wire 8 37),
    (.wire 16 39, .wire 8 38),
    (.wire 18 33, .virt 23710),
    (.wire 18 34, .virt 23710),
    (.wire 18 35, .virt 23710),
    (.wire 18 36, .virt 23710),
    (.wire 18 37, .virt 23710),
    (.wire 18 38, .virt 23710),
    (.wire 18 39, .virt 23710),
    (.wire 18 40, .virt 23710),
    (.wire 18 41, .virt 23710),
    (.wire 18 42, .virt 23710),
    (.wire 18 43, .virt 23710),
    (.wire 18 44, .virt 23710),
    (.wire 18 45, .virt 23710),
    (.wire 18 46, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `800..832`. -/
def privateBatchWrapper4.copies25 : List (Target × Target) := [
    (.wire 18 47, .virt 23710),
    (.wire 18 48, .virt 23710),
    (.wire 18 49, .virt 23710),
    (.wire 18 50, .virt 23710),
    (.wire 18 51, .virt 23710),
    (.wire 18 52, .virt 23710),
    (.wire 18 53, .virt 23710),
    (.wire 18 54, .virt 23710),
    (.wire 18 55, .virt 23710),
    (.wire 18 56, .virt 23710),
    (.wire 18 57, .virt 23710),
    (.wire 18 58, .virt 23710),
    (.wire 18 59, .virt 23710),
    (.wire 18 0, .virt 23677),
    (.wire 19 33, .virt 23710),
    (.wire 19 34, .virt 23710),
    (.wire 19 35, .virt 23710),
    (.wire 19 36, .virt 23710),
    (.wire 19 37, .virt 23710),
    (.wire 19 38, .virt 23710),
    (.wire 19 39, .virt 23710),
    (.wire 19 40, .virt 23710),
    (.wire 19 41, .virt 23710),
    (.wire 19 42, .virt 23710),
    (.wire 19 43, .virt 23710),
    (.wire 19 44, .virt 23710),
    (.wire 19 45, .virt 23710),
    (.wire 19 46, .virt 23710),
    (.wire 19 47, .virt 23710),
    (.wire 19 48, .virt 23710),
    (.wire 19 49, .virt 23710),
    (.wire 19 50, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `832..864`. -/
def privateBatchWrapper4.copies26 : List (Target × Target) := [
    (.wire 19 51, .virt 23710),
    (.wire 19 52, .virt 23710),
    (.wire 19 53, .virt 23710),
    (.wire 19 54, .virt 23710),
    (.wire 19 55, .virt 23710),
    (.wire 19 56, .virt 23710),
    (.wire 19 57, .virt 23710),
    (.wire 19 58, .virt 23710),
    (.wire 19 59, .virt 23710),
    (.wire 19 0, .virt 23682),
    (.virt 23677, .wire 8 40),
    (.virt 23709, .wire 8 41),
    (.virt 23682, .wire 8 42),
    (.wire 20 33, .virt 23710),
    (.wire 20 34, .virt 23710),
    (.wire 20 35, .virt 23710),
    (.wire 20 36, .virt 23710),
    (.wire 20 37, .virt 23710),
    (.wire 20 38, .virt 23710),
    (.wire 20 39, .virt 23710),
    (.wire 20 40, .virt 23710),
    (.wire 20 41, .virt 23710),
    (.wire 20 42, .virt 23710),
    (.wire 20 43, .virt 23710),
    (.wire 20 44, .virt 23710),
    (.wire 20 45, .virt 23710),
    (.wire 20 46, .virt 23710),
    (.wire 20 47, .virt 23710),
    (.wire 20 48, .virt 23710),
    (.wire 20 49, .virt 23710),
    (.wire 20 50, .virt 23710),
    (.wire 20 51, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `864..896`. -/
def privateBatchWrapper4.copies27 : List (Target × Target) := [
    (.wire 20 52, .virt 23710),
    (.wire 20 53, .virt 23710),
    (.wire 20 54, .virt 23710),
    (.wire 20 55, .virt 23710),
    (.wire 20 56, .virt 23710),
    (.wire 20 57, .virt 23710),
    (.wire 20 58, .virt 23710),
    (.wire 20 59, .virt 23710),
    (.wire 20 0, .virt 23687),
    (.wire 8 43, .wire 8 44),
    (.virt 23709, .wire 8 45),
    (.virt 23687, .wire 8 46),
    (.wire 21 33, .virt 23710),
    (.wire 21 34, .virt 23710),
    (.wire 21 35, .virt 23710),
    (.wire 21 36, .virt 23710),
    (.wire 21 37, .virt 23710),
    (.wire 21 38, .virt 23710),
    (.wire 21 39, .virt 23710),
    (.wire 21 40, .virt 23710),
    (.wire 21 41, .virt 23710),
    (.wire 21 42, .virt 23710),
    (.wire 21 43, .virt 23710),
    (.wire 21 44, .virt 23710),
    (.wire 21 45, .virt 23710),
    (.wire 21 46, .virt 23710),
    (.wire 21 47, .virt 23710),
    (.wire 21 48, .virt 23710),
    (.wire 21 49, .virt 23710),
    (.wire 21 50, .virt 23710),
    (.wire 21 51, .virt 23710),
    (.wire 21 52, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `896..928`. -/
def privateBatchWrapper4.copies28 : List (Target × Target) := [
    (.wire 21 53, .virt 23710),
    (.wire 21 54, .virt 23710),
    (.wire 21 55, .virt 23710),
    (.wire 21 56, .virt 23710),
    (.wire 21 57, .virt 23710),
    (.wire 21 58, .virt 23710),
    (.wire 21 59, .virt 23710),
    (.wire 21 0, .virt 23692),
    (.wire 8 47, .wire 8 48),
    (.virt 23709, .wire 8 49),
    (.virt 23692, .wire 8 50),
    (.wire 22 15, .virt 23710),
    (.wire 22 16, .virt 23710),
    (.wire 22 17, .virt 23710),
    (.wire 22 18, .virt 23710),
    (.wire 22 19, .virt 23710),
    (.wire 22 20, .virt 23710),
    (.wire 22 21, .virt 23710),
    (.wire 22 22, .virt 23710),
    (.wire 22 23, .virt 23710),
    (.wire 22 24, .virt 23710),
    (.wire 22 25, .virt 23710),
    (.wire 22 26, .virt 23710),
    (.wire 22 27, .virt 23710),
    (.wire 22 28, .virt 23710),
    (.wire 22 29, .virt 23710),
    (.wire 22 30, .virt 23710),
    (.wire 22 31, .virt 23710),
    (.wire 22 32, .virt 23710),
    (.wire 22 33, .virt 23710),
    (.wire 22 34, .virt 23710),
    (.wire 22 35, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `928..960`. -/
def privateBatchWrapper4.copies29 : List (Target × Target) := [
    (.wire 22 36, .virt 23710),
    (.wire 22 37, .virt 23710),
    (.wire 22 38, .virt 23710),
    (.wire 22 39, .virt 23710),
    (.wire 22 40, .virt 23710),
    (.wire 22 41, .virt 23710),
    (.wire 22 42, .virt 23710),
    (.wire 22 43, .virt 23710),
    (.wire 22 44, .virt 23710),
    (.wire 22 45, .virt 23710),
    (.wire 22 46, .virt 23710),
    (.wire 22 47, .virt 23710),
    (.wire 22 48, .virt 23710),
    (.wire 22 49, .virt 23710),
    (.wire 22 50, .virt 23710),
    (.wire 22 51, .virt 23710),
    (.wire 22 52, .virt 23710),
    (.wire 22 53, .virt 23710),
    (.wire 22 54, .virt 23710),
    (.wire 22 55, .virt 23710),
    (.wire 22 56, .virt 23710),
    (.wire 22 57, .virt 23710),
    (.wire 22 58, .virt 23710),
    (.wire 22 59, .virt 23710),
    (.wire 22 0, .virt 23672),
    (.virt 23777, .wire 16 40),
    (.virt 23709, .wire 16 41),
    (.virt 23672, .wire 16 42),
    (.wire 23 15, .virt 23710),
    (.wire 23 16, .virt 23710),
    (.wire 23 17, .virt 23710),
    (.wire 23 18, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `960..992`. -/
def privateBatchWrapper4.copies30 : List (Target × Target) := [
    (.wire 23 19, .virt 23710),
    (.wire 23 20, .virt 23710),
    (.wire 23 21, .virt 23710),
    (.wire 23 22, .virt 23710),
    (.wire 23 23, .virt 23710),
    (.wire 23 24, .virt 23710),
    (.wire 23 25, .virt 23710),
    (.wire 23 26, .virt 23710),
    (.wire 23 27, .virt 23710),
    (.wire 23 28, .virt 23710),
    (.wire 23 29, .virt 23710),
    (.wire 23 30, .virt 23710),
    (.wire 23 31, .virt 23710),
    (.wire 23 32, .virt 23710),
    (.wire 23 33, .virt 23710),
    (.wire 23 34, .virt 23710),
    (.wire 23 35, .virt 23710),
    (.wire 23 36, .virt 23710),
    (.wire 23 37, .virt 23710),
    (.wire 23 38, .virt 23710),
    (.wire 23 39, .virt 23710),
    (.wire 23 40, .virt 23710),
    (.wire 23 41, .virt 23710),
    (.wire 23 42, .virt 23710),
    (.wire 23 43, .virt 23710),
    (.wire 23 44, .virt 23710),
    (.wire 23 45, .virt 23710),
    (.wire 23 46, .virt 23710),
    (.wire 23 47, .virt 23710),
    (.wire 23 48, .virt 23710),
    (.wire 23 49, .virt 23710),
    (.wire 23 50, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `992..1024`. -/
def privateBatchWrapper4.copies31 : List (Target × Target) := [
    (.wire 23 51, .virt 23710),
    (.wire 23 52, .virt 23710),
    (.wire 23 53, .virt 23710),
    (.wire 23 54, .virt 23710),
    (.wire 23 55, .virt 23710),
    (.wire 23 56, .virt 23710),
    (.wire 23 57, .virt 23710),
    (.wire 23 58, .virt 23710),
    (.wire 23 59, .virt 23710),
    (.wire 23 0, .wire 16 43),
    (.wire 8 51, .wire 17 4),
    (.virt 23777, .wire 17 5),
    (.wire 8 51, .wire 17 6),
    (.wire 8 39, .wire 17 8),
    (.wire 16 43, .wire 17 9),
    (.wire 8 39, .wire 17 10),
    (.wire 17 11, .wire 16 44),
    (.virt 23709, .wire 16 45),
    (.wire 17 7, .wire 16 46),
    (.wire 24 53, .virt 23710),
    (.wire 24 54, .virt 23710),
    (.wire 24 55, .virt 23710),
    (.wire 24 56, .virt 23710),
    (.wire 24 57, .virt 23710),
    (.wire 24 58, .virt 23710),
    (.wire 24 59, .virt 23710),
    (.wire 24 0, .wire 16 47),
    (.wire 5 11, .wire 17 12),
    (.wire 5 35, .wire 17 13),
    (.wire 5 11, .wire 17 14),
    (.virt 23709, .wire 16 48),
    (.virt 23709, .wire 16 49)
  ]

/-- `privateBatchWrapper4.copies`, items `1024..1056`. -/
def privateBatchWrapper4.copies32 : List (Target × Target) := [
    (.virt 23778, .wire 16 50),
    (.virt 5908, .wire 16 52),
    (.virt 23709, .wire 16 53),
    (.virt 11826, .wire 16 54),
    (.virt 23778, .wire 17 16),
    (.wire 16 55, .wire 17 17),
    (.virt 23778, .wire 17 18),
    (.wire 16 55, .wire 17 20),
    (.virt 23779, .wire 17 21),
    (.wire 16 55, .wire 17 22),
    (.wire 17 23, .wire 16 56),
    (.virt 23709, .wire 16 57),
    (.wire 16 51, .wire 16 58),
    (.wire 17 19, .virt 23710),
    (.wire 16 59, .virt 23710),
    (.virt 23709, .wire 25 0),
    (.virt 23709, .wire 25 1),
    (.virt 23780, .wire 25 2),
    (.virt 5909, .wire 25 4),
    (.virt 23709, .wire 25 5),
    (.virt 11827, .wire 25 6),
    (.virt 23780, .wire 17 24),
    (.wire 25 7, .wire 17 25),
    (.virt 23780, .wire 17 26),
    (.wire 25 7, .wire 17 28),
    (.virt 23781, .wire 17 29),
    (.wire 25 7, .wire 17 30),
    (.wire 17 31, .wire 25 8),
    (.virt 23709, .wire 25 9),
    (.wire 25 3, .wire 25 10),
    (.wire 17 27, .virt 23710),
    (.wire 25 11, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `1056..1088`. -/
def privateBatchWrapper4.copies33 : List (Target × Target) := [
    (.virt 23709, .wire 25 12),
    (.virt 23709, .wire 25 13),
    (.virt 23782, .wire 25 14),
    (.virt 5910, .wire 25 16),
    (.virt 23709, .wire 25 17),
    (.virt 11828, .wire 25 18),
    (.virt 23782, .wire 17 32),
    (.wire 25 19, .wire 17 33),
    (.virt 23782, .wire 17 34),
    (.wire 25 19, .wire 17 36),
    (.virt 23783, .wire 17 37),
    (.wire 25 19, .wire 17 38),
    (.wire 17 39, .wire 25 20),
    (.virt 23709, .wire 25 21),
    (.wire 25 15, .wire 25 22),
    (.wire 17 35, .virt 23710),
    (.wire 25 23, .virt 23710),
    (.virt 23709, .wire 25 24),
    (.virt 23709, .wire 25 25),
    (.virt 23784, .wire 25 26),
    (.virt 5911, .wire 25 28),
    (.virt 23709, .wire 25 29),
    (.virt 11829, .wire 25 30),
    (.virt 23784, .wire 17 40),
    (.wire 25 31, .wire 17 41),
    (.virt 23784, .wire 17 42),
    (.wire 25 31, .wire 17 44),
    (.virt 23785, .wire 17 45),
    (.wire 25 31, .wire 17 46),
    (.wire 17 47, .wire 25 32),
    (.virt 23709, .wire 25 33),
    (.wire 25 27, .wire 25 34)
  ]

/-- `privateBatchWrapper4.copies`, items `1088..1120`. -/
def privateBatchWrapper4.copies34 : List (Target × Target) := [
    (.wire 17 43, .virt 23710),
    (.wire 25 35, .virt 23710),
    (.virt 23778, .wire 17 48),
    (.virt 23780, .wire 17 49),
    (.virt 23778, .wire 17 50),
    (.virt 23782, .wire 17 52),
    (.virt 23784, .wire 17 53),
    (.virt 23782, .wire 17 54),
    (.wire 17 51, .wire 17 56),
    (.wire 17 55, .wire 17 57),
    (.wire 17 51, .wire 17 58),
    (.wire 17 15, .wire 26 0),
    (.wire 17 59, .wire 26 1),
    (.wire 17 15, .wire 26 2),
    (.wire 26 3, .virt 23710),
    (.wire 5 11, .wire 26 4),
    (.wire 6 23, .wire 26 5),
    (.wire 5 11, .wire 26 6),
    (.virt 23709, .wire 25 36),
    (.virt 23709, .wire 25 37),
    (.virt 23786, .wire 25 38),
    (.virt 5908, .wire 25 40),
    (.virt 23709, .wire 25 41),
    (.virt 17744, .wire 25 42),
    (.virt 23786, .wire 26 8),
    (.wire 25 43, .wire 26 9),
    (.virt 23786, .wire 26 10),
    (.wire 25 43, .wire 26 12),
    (.virt 23787, .wire 26 13),
    (.wire 25 43, .wire 26 14),
    (.wire 26 15, .wire 25 44),
    (.virt 23709, .wire 25 45)
  ]

/-- `privateBatchWrapper4.copies`, items `1120..1152`. -/
def privateBatchWrapper4.copies35 : List (Target × Target) := [
    (.wire 25 39, .wire 25 46),
    (.wire 26 11, .virt 23710),
    (.wire 25 47, .virt 23710),
    (.virt 23709, .wire 25 48),
    (.virt 23709, .wire 25 49),
    (.virt 23788, .wire 25 50),
    (.virt 5909, .wire 25 52),
    (.virt 23709, .wire 25 53),
    (.virt 17745, .wire 25 54),
    (.virt 23788, .wire 26 16),
    (.wire 25 55, .wire 26 17),
    (.virt 23788, .wire 26 18),
    (.wire 25 55, .wire 26 20),
    (.virt 23789, .wire 26 21),
    (.wire 25 55, .wire 26 22),
    (.wire 26 23, .wire 25 56),
    (.virt 23709, .wire 25 57),
    (.wire 25 51, .wire 25 58),
    (.wire 26 19, .virt 23710),
    (.wire 25 59, .virt 23710),
    (.virt 23709, .wire 27 0),
    (.virt 23709, .wire 27 1),
    (.virt 23790, .wire 27 2),
    (.virt 5910, .wire 27 4),
    (.virt 23709, .wire 27 5),
    (.virt 17746, .wire 27 6),
    (.virt 23790, .wire 26 24),
    (.wire 27 7, .wire 26 25),
    (.virt 23790, .wire 26 26),
    (.wire 27 7, .wire 26 28),
    (.virt 23791, .wire 26 29),
    (.wire 27 7, .wire 26 30)
  ]

/-- `privateBatchWrapper4.copies`, items `1152..1184`. -/
def privateBatchWrapper4.copies36 : List (Target × Target) := [
    (.wire 26 31, .wire 27 8),
    (.virt 23709, .wire 27 9),
    (.wire 27 3, .wire 27 10),
    (.wire 26 27, .virt 23710),
    (.wire 27 11, .virt 23710),
    (.virt 23709, .wire 27 12),
    (.virt 23709, .wire 27 13),
    (.virt 23792, .wire 27 14),
    (.virt 5911, .wire 27 16),
    (.virt 23709, .wire 27 17),
    (.virt 17747, .wire 27 18),
    (.virt 23792, .wire 26 32),
    (.wire 27 19, .wire 26 33),
    (.virt 23792, .wire 26 34),
    (.wire 27 19, .wire 26 36),
    (.virt 23793, .wire 26 37),
    (.wire 27 19, .wire 26 38),
    (.wire 26 39, .wire 27 20),
    (.virt 23709, .wire 27 21),
    (.wire 27 15, .wire 27 22),
    (.wire 26 35, .virt 23710),
    (.wire 27 23, .virt 23710),
    (.virt 23786, .wire 26 40),
    (.virt 23788, .wire 26 41),
    (.virt 23786, .wire 26 42),
    (.virt 23790, .wire 26 44),
    (.virt 23792, .wire 26 45),
    (.virt 23790, .wire 26 46),
    (.wire 26 43, .wire 26 48),
    (.wire 26 47, .wire 26 49),
    (.wire 26 43, .wire 26 50),
    (.wire 26 7, .wire 26 52)
  ]

/-- `privateBatchWrapper4.copies`, items `1184..1216`. -/
def privateBatchWrapper4.copies37 : List (Target × Target) := [
    (.wire 26 51, .wire 26 53),
    (.wire 26 7, .wire 26 54),
    (.wire 26 55, .virt 23710),
    (.wire 5 11, .wire 26 56),
    (.wire 10 11, .wire 26 57),
    (.wire 5 11, .wire 26 58),
    (.virt 23709, .wire 27 24),
    (.virt 23709, .wire 27 25),
    (.virt 23794, .wire 27 26),
    (.virt 5908, .wire 27 28),
    (.virt 23709, .wire 27 29),
    (.virt 23662, .wire 27 30),
    (.virt 23794, .wire 28 0),
    (.wire 27 31, .wire 28 1),
    (.virt 23794, .wire 28 2),
    (.wire 27 31, .wire 28 4),
    (.virt 23795, .wire 28 5),
    (.wire 27 31, .wire 28 6),
    (.wire 28 7, .wire 27 32),
    (.virt 23709, .wire 27 33),
    (.wire 27 27, .wire 27 34),
    (.wire 28 3, .virt 23710),
    (.wire 27 35, .virt 23710),
    (.virt 23709, .wire 27 36),
    (.virt 23709, .wire 27 37),
    (.virt 23796, .wire 27 38),
    (.virt 5909, .wire 27 40),
    (.virt 23709, .wire 27 41),
    (.virt 23663, .wire 27 42),
    (.virt 23796, .wire 28 8),
    (.wire 27 43, .wire 28 9),
    (.virt 23796, .wire 28 10)
  ]

/-- `privateBatchWrapper4.copies`, items `1216..1248`. -/
def privateBatchWrapper4.copies38 : List (Target × Target) := [
    (.wire 27 43, .wire 28 12),
    (.virt 23797, .wire 28 13),
    (.wire 27 43, .wire 28 14),
    (.wire 28 15, .wire 27 44),
    (.virt 23709, .wire 27 45),
    (.wire 27 39, .wire 27 46),
    (.wire 28 11, .virt 23710),
    (.wire 27 47, .virt 23710),
    (.virt 23709, .wire 27 48),
    (.virt 23709, .wire 27 49),
    (.virt 23798, .wire 27 50),
    (.virt 5910, .wire 27 52),
    (.virt 23709, .wire 27 53),
    (.virt 23664, .wire 27 54),
    (.virt 23798, .wire 28 16),
    (.wire 27 55, .wire 28 17),
    (.virt 23798, .wire 28 18),
    (.wire 27 55, .wire 28 20),
    (.virt 23799, .wire 28 21),
    (.wire 27 55, .wire 28 22),
    (.wire 28 23, .wire 27 56),
    (.virt 23709, .wire 27 57),
    (.wire 27 51, .wire 27 58),
    (.wire 28 19, .virt 23710),
    (.wire 27 59, .virt 23710),
    (.virt 23709, .wire 29 0),
    (.virt 23709, .wire 29 1),
    (.virt 23800, .wire 29 2),
    (.virt 5911, .wire 29 4),
    (.virt 23709, .wire 29 5),
    (.virt 23665, .wire 29 6),
    (.virt 23800, .wire 28 24)
  ]

/-- `privateBatchWrapper4.copies`, items `1248..1280`. -/
def privateBatchWrapper4.copies39 : List (Target × Target) := [
    (.wire 29 7, .wire 28 25),
    (.virt 23800, .wire 28 26),
    (.wire 29 7, .wire 28 28),
    (.virt 23801, .wire 28 29),
    (.wire 29 7, .wire 28 30),
    (.wire 28 31, .wire 29 8),
    (.virt 23709, .wire 29 9),
    (.wire 29 3, .wire 29 10),
    (.wire 28 27, .virt 23710),
    (.wire 29 11, .virt 23710),
    (.virt 23794, .wire 28 32),
    (.virt 23796, .wire 28 33),
    (.virt 23794, .wire 28 34),
    (.virt 23798, .wire 28 36),
    (.virt 23800, .wire 28 37),
    (.virt 23798, .wire 28 38),
    (.wire 28 35, .wire 28 40),
    (.wire 28 39, .wire 28 41),
    (.wire 28 35, .wire 28 42),
    (.wire 26 59, .wire 28 44),
    (.wire 28 43, .wire 28 45),
    (.wire 26 59, .wire 28 46),
    (.wire 28 47, .virt 23710),
    (.wire 5 35, .wire 28 48),
    (.wire 6 23, .wire 28 49),
    (.wire 5 35, .wire 28 50),
    (.virt 23709, .wire 29 12),
    (.virt 23709, .wire 29 13),
    (.virt 23802, .wire 29 14),
    (.virt 11826, .wire 29 16),
    (.virt 23709, .wire 29 17),
    (.virt 17744, .wire 29 18)
  ]

/-- `privateBatchWrapper4.copies`, items `1280..1312`. -/
def privateBatchWrapper4.copies40 : List (Target × Target) := [
    (.virt 23802, .wire 28 52),
    (.wire 29 19, .wire 28 53),
    (.virt 23802, .wire 28 54),
    (.wire 29 19, .wire 28 56),
    (.virt 23803, .wire 28 57),
    (.wire 29 19, .wire 28 58),
    (.wire 28 59, .wire 29 20),
    (.virt 23709, .wire 29 21),
    (.wire 29 15, .wire 29 22),
    (.wire 28 55, .virt 23710),
    (.wire 29 23, .virt 23710),
    (.virt 23709, .wire 29 24),
    (.virt 23709, .wire 29 25),
    (.virt 23804, .wire 29 26),
    (.virt 11827, .wire 29 28),
    (.virt 23709, .wire 29 29),
    (.virt 17745, .wire 29 30),
    (.virt 23804, .wire 30 0),
    (.wire 29 31, .wire 30 1),
    (.virt 23804, .wire 30 2),
    (.wire 29 31, .wire 30 4),
    (.virt 23805, .wire 30 5),
    (.wire 29 31, .wire 30 6),
    (.wire 30 7, .wire 29 32),
    (.virt 23709, .wire 29 33),
    (.wire 29 27, .wire 29 34),
    (.wire 30 3, .virt 23710),
    (.wire 29 35, .virt 23710),
    (.virt 23709, .wire 29 36),
    (.virt 23709, .wire 29 37),
    (.virt 23806, .wire 29 38),
    (.virt 11828, .wire 29 40)
  ]

/-- `privateBatchWrapper4.copies`, items `1312..1344`. -/
def privateBatchWrapper4.copies41 : List (Target × Target) := [
    (.virt 23709, .wire 29 41),
    (.virt 17746, .wire 29 42),
    (.virt 23806, .wire 30 8),
    (.wire 29 43, .wire 30 9),
    (.virt 23806, .wire 30 10),
    (.wire 29 43, .wire 30 12),
    (.virt 23807, .wire 30 13),
    (.wire 29 43, .wire 30 14),
    (.wire 30 15, .wire 29 44),
    (.virt 23709, .wire 29 45),
    (.wire 29 39, .wire 29 46),
    (.wire 30 11, .virt 23710),
    (.wire 29 47, .virt 23710),
    (.virt 23709, .wire 29 48),
    (.virt 23709, .wire 29 49),
    (.virt 23808, .wire 29 50),
    (.virt 11829, .wire 29 52),
    (.virt 23709, .wire 29 53),
    (.virt 17747, .wire 29 54),
    (.virt 23808, .wire 30 16),
    (.wire 29 55, .wire 30 17),
    (.virt 23808, .wire 30 18),
    (.wire 29 55, .wire 30 20),
    (.virt 23809, .wire 30 21),
    (.wire 29 55, .wire 30 22),
    (.wire 30 23, .wire 29 56),
    (.virt 23709, .wire 29 57),
    (.wire 29 51, .wire 29 58),
    (.wire 30 19, .virt 23710),
    (.wire 29 59, .virt 23710),
    (.virt 23802, .wire 30 24),
    (.virt 23804, .wire 30 25)
  ]

/-- `privateBatchWrapper4.copies`, items `1344..1376`. -/
def privateBatchWrapper4.copies42 : List (Target × Target) := [
    (.virt 23802, .wire 30 26),
    (.virt 23806, .wire 30 28),
    (.virt 23808, .wire 30 29),
    (.virt 23806, .wire 30 30),
    (.wire 30 27, .wire 30 32),
    (.wire 30 31, .wire 30 33),
    (.wire 30 27, .wire 30 34),
    (.wire 28 51, .wire 30 36),
    (.wire 30 35, .wire 30 37),
    (.wire 28 51, .wire 30 38),
    (.wire 30 39, .virt 23710),
    (.wire 5 35, .wire 30 40),
    (.wire 10 11, .wire 30 41),
    (.wire 5 35, .wire 30 42),
    (.virt 23709, .wire 31 0),
    (.virt 23709, .wire 31 1),
    (.virt 23810, .wire 31 2),
    (.virt 11826, .wire 31 4),
    (.virt 23709, .wire 31 5),
    (.virt 23662, .wire 31 6),
    (.virt 23810, .wire 30 44),
    (.wire 31 7, .wire 30 45),
    (.virt 23810, .wire 30 46),
    (.wire 31 7, .wire 30 48),
    (.virt 23811, .wire 30 49),
    (.wire 31 7, .wire 30 50),
    (.wire 30 51, .wire 31 8),
    (.virt 23709, .wire 31 9),
    (.wire 31 3, .wire 31 10),
    (.wire 30 47, .virt 23710),
    (.wire 31 11, .virt 23710),
    (.virt 23709, .wire 31 12)
  ]

/-- `privateBatchWrapper4.copies`, items `1376..1408`. -/
def privateBatchWrapper4.copies43 : List (Target × Target) := [
    (.virt 23709, .wire 31 13),
    (.virt 23812, .wire 31 14),
    (.virt 11827, .wire 31 16),
    (.virt 23709, .wire 31 17),
    (.virt 23663, .wire 31 18),
    (.virt 23812, .wire 30 52),
    (.wire 31 19, .wire 30 53),
    (.virt 23812, .wire 30 54),
    (.wire 31 19, .wire 30 56),
    (.virt 23813, .wire 30 57),
    (.wire 31 19, .wire 30 58),
    (.wire 30 59, .wire 31 20),
    (.virt 23709, .wire 31 21),
    (.wire 31 15, .wire 31 22),
    (.wire 30 55, .virt 23710),
    (.wire 31 23, .virt 23710),
    (.virt 23709, .wire 31 24),
    (.virt 23709, .wire 31 25),
    (.virt 23814, .wire 31 26),
    (.virt 11828, .wire 31 28),
    (.virt 23709, .wire 31 29),
    (.virt 23664, .wire 31 30),
    (.virt 23814, .wire 32 0),
    (.wire 31 31, .wire 32 1),
    (.virt 23814, .wire 32 2),
    (.wire 31 31, .wire 32 4),
    (.virt 23815, .wire 32 5),
    (.wire 31 31, .wire 32 6),
    (.wire 32 7, .wire 31 32),
    (.virt 23709, .wire 31 33),
    (.wire 31 27, .wire 31 34),
    (.wire 32 3, .virt 23710)
  ]

/-- `privateBatchWrapper4.copies`, items `1408..1440`. -/
def privateBatchWrapper4.copies44 : List (Target × Target) := [
    (.wire 31 35, .virt 23710),
    (.virt 23709, .wire 31 36),
    (.virt 23709, .wire 31 37),
    (.virt 23816, .wire 31 38),
    (.virt 11829, .wire 31 40),
    (.virt 23709, .wire 31 41),
    (.virt 23665, .wire 31 42),
    (.virt 23816, .wire 32 8),
    (.wire 31 43, .wire 32 9),
    (.virt 23816, .wire 32 10),
    (.wire 31 43, .wire 32 12),
    (.virt 23817, .wire 32 13),
    (.wire 31 43, .wire 32 14),
    (.wire 32 15, .wire 31 44),
    (.virt 23709, .wire 31 45),
    (.wire 31 39, .wire 31 46),
    (.wire 32 11, .virt 23710),
    (.wire 31 47, .virt 23710),
    (.virt 23810, .wire 32 16),
    (.virt 23812, .wire 32 17),
    (.virt 23810, .wire 32 18),
    (.virt 23814, .wire 32 20),
    (.virt 23816, .wire 32 21),
    (.virt 23814, .wire 32 22),
    (.wire 32 19, .wire 32 24),
    (.wire 32 23, .wire 32 25),
    (.wire 32 19, .wire 32 26),
    (.wire 30 43, .wire 32 28),
    (.wire 32 27, .wire 32 29),
    (.wire 30 43, .wire 32 30),
    (.wire 32 31, .virt 23710),
    (.wire 6 23, .wire 32 32)
  ]

/-- `privateBatchWrapper4.copies`, items `1440..1472`. -/
def privateBatchWrapper4.copies45 : List (Target × Target) := [
    (.wire 10 11, .wire 32 33),
    (.wire 6 23, .wire 32 34),
    (.virt 23709, .wire 31 48),
    (.virt 23709, .wire 31 49),
    (.virt 23818, .wire 31 50),
    (.virt 17744, .wire 31 52),
    (.virt 23709, .wire 31 53),
    (.virt 23662, .wire 31 54),
    (.virt 23818, .wire 32 36),
    (.wire 31 55, .wire 32 37),
    (.virt 23818, .wire 32 38),
    (.wire 31 55, .wire 32 40),
    (.virt 23819, .wire 32 41),
    (.wire 31 55, .wire 32 42),
    (.wire 32 43, .wire 31 56),
    (.virt 23709, .wire 31 57),
    (.wire 31 51, .wire 31 58),
    (.wire 32 39, .virt 23710),
    (.wire 31 59, .virt 23710),
    (.virt 23709, .wire 33 0),
    (.virt 23709, .wire 33 1),
    (.virt 23820, .wire 33 2),
    (.virt 17745, .wire 33 4),
    (.virt 23709, .wire 33 5),
    (.virt 23663, .wire 33 6),
    (.virt 23820, .wire 32 44),
    (.wire 33 7, .wire 32 45),
    (.virt 23820, .wire 32 46),
    (.wire 33 7, .wire 32 48),
    (.virt 23821, .wire 32 49),
    (.wire 33 7, .wire 32 50),
    (.wire 32 51, .wire 33 8)
  ]

/-- `privateBatchWrapper4.copies`, items `1472..1504`. -/
def privateBatchWrapper4.copies46 : List (Target × Target) := [
    (.virt 23709, .wire 33 9),
    (.wire 33 3, .wire 33 10),
    (.wire 32 47, .virt 23710),
    (.wire 33 11, .virt 23710),
    (.virt 23709, .wire 33 12),
    (.virt 23709, .wire 33 13),
    (.virt 23822, .wire 33 14),
    (.virt 17746, .wire 33 16),
    (.virt 23709, .wire 33 17),
    (.virt 23664, .wire 33 18),
    (.virt 23822, .wire 32 52),
    (.wire 33 19, .wire 32 53),
    (.virt 23822, .wire 32 54),
    (.wire 33 19, .wire 32 56),
    (.virt 23823, .wire 32 57),
    (.wire 33 19, .wire 32 58),
    (.wire 32 59, .wire 33 20),
    (.virt 23709, .wire 33 21),
    (.wire 33 15, .wire 33 22),
    (.wire 32 55, .virt 23710),
    (.wire 33 23, .virt 23710),
    (.virt 23709, .wire 33 24),
    (.virt 23709, .wire 33 25),
    (.virt 23824, .wire 33 26),
    (.virt 17747, .wire 33 28),
    (.virt 23709, .wire 33 29),
    (.virt 23665, .wire 33 30),
    (.virt 23824, .wire 34 0),
    (.wire 33 31, .wire 34 1),
    (.virt 23824, .wire 34 2),
    (.wire 33 31, .wire 34 4),
    (.virt 23825, .wire 34 5)
  ]

/-- `privateBatchWrapper4.copies`, items `1504..1536`. -/
def privateBatchWrapper4.copies47 : List (Target × Target) := [
    (.wire 33 31, .wire 34 6),
    (.wire 34 7, .wire 33 32),
    (.virt 23709, .wire 33 33),
    (.wire 33 27, .wire 33 34),
    (.wire 34 3, .virt 23710),
    (.wire 33 35, .virt 23710),
    (.virt 23818, .wire 34 8),
    (.virt 23820, .wire 34 9),
    (.virt 23818, .wire 34 10),
    (.virt 23822, .wire 34 12),
    (.virt 23824, .wire 34 13),
    (.virt 23822, .wire 34 14),
    (.wire 34 11, .wire 34 16),
    (.wire 34 15, .wire 34 17),
    (.wire 34 11, .wire 34 18),
    (.wire 32 35, .wire 34 20),
    (.wire 34 19, .wire 34 21),
    (.wire 32 35, .wire 34 22),
    (.wire 34 23, .virt 23710),
    (.virt 23693, .wire 35 0),
    (.virt 23694, .wire 35 1),
    (.virt 23695, .wire 35 2),
    (.virt 23696, .wire 35 3),
    (.virt 23709, .wire 35 4),
    (.virt 23710, .wire 35 5),
    (.virt 23710, .wire 35 6),
    (.virt 23710, .wire 35 7),
    (.virt 23710, .wire 35 8),
    (.virt 23710, .wire 35 9),
    (.virt 23710, .wire 35 10),
    (.virt 23710, .wire 35 11),
    (.wire 35 12, .wire 36 0)
  ]

/-- `privateBatchWrapper4.copies`, items `1536..1568`. -/
def privateBatchWrapper4.copies48 : List (Target × Target) := [
    (.wire 35 13, .wire 36 1),
    (.wire 35 14, .wire 36 2),
    (.wire 35 15, .wire 36 3),
    (.virt 23709, .wire 36 4),
    (.virt 23710, .wire 36 5),
    (.virt 23710, .wire 36 6),
    (.virt 23710, .wire 36 7),
    (.virt 23710, .wire 36 8),
    (.virt 23710, .wire 36 9),
    (.virt 23710, .wire 36 10),
    (.virt 23710, .wire 36 11),
    (.wire 1 43, .wire 33 36),
    (.virt 5908, .wire 33 37),
    (.virt 5908, .wire 33 38),
    (.wire 1 43, .wire 33 40),
    (.wire 36 12, .wire 33 41),
    (.wire 33 39, .wire 33 42),
    (.wire 1 43, .wire 33 44),
    (.virt 5909, .wire 33 45),
    (.virt 5909, .wire 33 46),
    (.wire 1 43, .wire 33 48),
    (.wire 36 13, .wire 33 49),
    (.wire 33 47, .wire 33 50),
    (.wire 1 43, .wire 33 52),
    (.virt 5910, .wire 33 53),
    (.virt 5910, .wire 33 54),
    (.wire 1 43, .wire 33 56),
    (.wire 36 14, .wire 33 57),
    (.wire 33 55, .wire 33 58),
    (.wire 1 43, .wire 37 0),
    (.virt 5911, .wire 37 1),
    (.virt 5911, .wire 37 2)
  ]

/-- `privateBatchWrapper4.copies`, items `1568..1600`. -/
def privateBatchWrapper4.copies49 : List (Target × Target) := [
    (.wire 1 43, .wire 37 4),
    (.wire 36 15, .wire 37 5),
    (.wire 37 3, .wire 37 6),
    (.virt 23697, .wire 38 0),
    (.virt 23698, .wire 38 1),
    (.virt 23699, .wire 38 2),
    (.virt 23700, .wire 38 3),
    (.virt 23709, .wire 38 4),
    (.virt 23710, .wire 38 5),
    (.virt 23710, .wire 38 6),
    (.virt 23710, .wire 38 7),
    (.virt 23710, .wire 38 8),
    (.virt 23710, .wire 38 9),
    (.virt 23710, .wire 38 10),
    (.virt 23710, .wire 38 11),
    (.wire 38 12, .wire 39 0),
    (.wire 38 13, .wire 39 1),
    (.wire 38 14, .wire 39 2),
    (.wire 38 15, .wire 39 3),
    (.virt 23709, .wire 39 4),
    (.virt 23710, .wire 39 5),
    (.virt 23710, .wire 39 6),
    (.virt 23710, .wire 39 7),
    (.virt 23710, .wire 39 8),
    (.virt 23710, .wire 39 9),
    (.virt 23710, .wire 39 10),
    (.virt 23710, .wire 39 11),
    (.wire 2 27, .wire 37 8),
    (.virt 11826, .wire 37 9),
    (.virt 11826, .wire 37 10),
    (.wire 2 27, .wire 37 12),
    (.wire 39 12, .wire 37 13)
  ]

/-- `privateBatchWrapper4.copies`, items `1600..1632`. -/
def privateBatchWrapper4.copies50 : List (Target × Target) := [
    (.wire 37 11, .wire 37 14),
    (.wire 2 27, .wire 37 16),
    (.virt 11827, .wire 37 17),
    (.virt 11827, .wire 37 18),
    (.wire 2 27, .wire 37 20),
    (.wire 39 13, .wire 37 21),
    (.wire 37 19, .wire 37 22),
    (.wire 2 27, .wire 37 24),
    (.virt 11828, .wire 37 25),
    (.virt 11828, .wire 37 26),
    (.wire 2 27, .wire 37 28),
    (.wire 39 14, .wire 37 29),
    (.wire 37 27, .wire 37 30),
    (.wire 2 27, .wire 37 32),
    (.virt 11829, .wire 37 33),
    (.virt 11829, .wire 37 34),
    (.wire 2 27, .wire 37 36),
    (.wire 39 15, .wire 37 37),
    (.wire 37 35, .wire 37 38),
    (.virt 23701, .wire 40 0),
    (.virt 23702, .wire 40 1),
    (.virt 23703, .wire 40 2),
    (.virt 23704, .wire 40 3),
    (.virt 23709, .wire 40 4),
    (.virt 23710, .wire 40 5),
    (.virt 23710, .wire 40 6),
    (.virt 23710, .wire 40 7),
    (.virt 23710, .wire 40 8),
    (.virt 23710, .wire 40 9),
    (.virt 23710, .wire 40 10),
    (.virt 23710, .wire 40 11),
    (.wire 40 12, .wire 41 0)
  ]

/-- `privateBatchWrapper4.copies`, items `1632..1664`. -/
def privateBatchWrapper4.copies51 : List (Target × Target) := [
    (.wire 40 13, .wire 41 1),
    (.wire 40 14, .wire 41 2),
    (.wire 40 15, .wire 41 3),
    (.virt 23709, .wire 41 4),
    (.virt 23710, .wire 41 5),
    (.virt 23710, .wire 41 6),
    (.virt 23710, .wire 41 7),
    (.virt 23710, .wire 41 8),
    (.virt 23710, .wire 41 9),
    (.virt 23710, .wire 41 10),
    (.virt 23710, .wire 41 11),
    (.wire 4 11, .wire 37 40),
    (.virt 17744, .wire 37 41),
    (.virt 17744, .wire 37 42),
    (.wire 4 11, .wire 37 44),
    (.wire 41 12, .wire 37 45),
    (.wire 37 43, .wire 37 46),
    (.wire 4 11, .wire 37 48),
    (.virt 17745, .wire 37 49),
    (.virt 17745, .wire 37 50),
    (.wire 4 11, .wire 37 52),
    (.wire 41 13, .wire 37 53),
    (.wire 37 51, .wire 37 54),
    (.wire 4 11, .wire 37 56),
    (.virt 17746, .wire 37 57),
    (.virt 17746, .wire 37 58),
    (.wire 4 11, .wire 42 0),
    (.wire 41 14, .wire 42 1),
    (.wire 37 59, .wire 42 2),
    (.wire 4 11, .wire 42 4),
    (.virt 17747, .wire 42 5),
    (.virt 17747, .wire 42 6)
  ]

/-- `privateBatchWrapper4.copies`, items `1664..1696`. -/
def privateBatchWrapper4.copies52 : List (Target × Target) := [
    (.wire 4 11, .wire 42 8),
    (.wire 41 15, .wire 42 9),
    (.wire 42 7, .wire 42 10),
    (.virt 23705, .wire 43 0),
    (.virt 23706, .wire 43 1),
    (.virt 23707, .wire 43 2),
    (.virt 23708, .wire 43 3),
    (.virt 23709, .wire 43 4),
    (.virt 23710, .wire 43 5),
    (.virt 23710, .wire 43 6),
    (.virt 23710, .wire 43 7),
    (.virt 23710, .wire 43 8),
    (.virt 23710, .wire 43 9),
    (.virt 23710, .wire 43 10),
    (.virt 23710, .wire 43 11),
    (.wire 43 12, .wire 44 0),
    (.wire 43 13, .wire 44 1),
    (.wire 43 14, .wire 44 2),
    (.wire 43 15, .wire 44 3),
    (.virt 23709, .wire 44 4),
    (.virt 23710, .wire 44 5),
    (.virt 23710, .wire 44 6),
    (.virt 23710, .wire 44 7),
    (.virt 23710, .wire 44 8),
    (.virt 23710, .wire 44 9),
    (.virt 23710, .wire 44 10),
    (.virt 23710, .wire 44 11),
    (.wire 4 55, .wire 42 12),
    (.virt 23662, .wire 42 13),
    (.virt 23662, .wire 42 14),
    (.wire 4 55, .wire 42 16),
    (.wire 44 12, .wire 42 17)
  ]

/-- `privateBatchWrapper4.copies`, items `1696..1715`. -/
def privateBatchWrapper4.copies53 : List (Target × Target) := [
    (.wire 42 15, .wire 42 18),
    (.wire 4 55, .wire 42 20),
    (.virt 23663, .wire 42 21),
    (.virt 23663, .wire 42 22),
    (.wire 4 55, .wire 42 24),
    (.wire 44 13, .wire 42 25),
    (.wire 42 23, .wire 42 26),
    (.wire 4 55, .wire 42 28),
    (.virt 23664, .wire 42 29),
    (.virt 23664, .wire 42 30),
    (.wire 4 55, .wire 42 32),
    (.wire 44 14, .wire 42 33),
    (.wire 42 31, .wire 42 34),
    (.wire 4 55, .wire 42 36),
    (.virt 23665, .wire 42 37),
    (.virt 23665, .wire 42 38),
    (.wire 4 55, .wire 42 40),
    (.wire 44 15, .wire 42 41),
    (.wire 42 39, .wire 42 42)
  ]

/-- The private-batch aggregation wrapper at `n = 4` without the leaf verifiers (`wormhole/aggregator/src/private_batch/circuit/circuit_logic.rs`): leaf public inputs `leaf_pis_0..3`, dummy-nullifier preimages `dummy_pre_image_0..3`, the permutation switches `switches`, and the aggregated public inputs. -/
def privateBatchWrapper4 (p : ℕ) : Circuit p where
  rows := [
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 0
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 1
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 2
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 3
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 4
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 5
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 6
    ⟨.arithmetic 15, [(-1), 1]⟩,  -- row 7
    ⟨.arithmetic 15, [1, 1]⟩,  -- row 8
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 9
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 10
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 11
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 12
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 13
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 14
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 15
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 16
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 17
    ⟨.baseSum2 59, []⟩,  -- row 18
    ⟨.baseSum2 59, []⟩,  -- row 19
    ⟨.baseSum2 59, []⟩,  -- row 20
    ⟨.baseSum2 59, []⟩,  -- row 21
    ⟨.baseSum2 59, []⟩,  -- row 22
    ⟨.baseSum2 59, []⟩,  -- row 23
    ⟨.baseSum2 59, []⟩,  -- row 24
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 25
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 26
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 27
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 28
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 29
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 30
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 31
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 32
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 33
    ⟨.arithmetic 15, [1, 0]⟩,  -- row 34
    ⟨.poseidon2, []⟩,  -- row 35
    ⟨.poseidon2, []⟩,  -- row 36
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 37
    ⟨.poseidon2, []⟩,  -- row 38
    ⟨.poseidon2, []⟩,  -- row 39
    ⟨.poseidon2, []⟩,  -- row 40
    ⟨.poseidon2, []⟩,  -- row 41
    ⟨.arithmetic 15, [1, (-1)]⟩,  -- row 42
    ⟨.poseidon2, []⟩,  -- row 43
    ⟨.poseidon2, []⟩  -- row 44
  ]
  copies := (((((privateBatchWrapper4.copies0 ++ (privateBatchWrapper4.copies1 ++ privateBatchWrapper4.copies2)) ++ (privateBatchWrapper4.copies3 ++ (privateBatchWrapper4.copies4 ++ privateBatchWrapper4.copies5))) ++ ((privateBatchWrapper4.copies6 ++ (privateBatchWrapper4.copies7 ++ privateBatchWrapper4.copies8)) ++ ((privateBatchWrapper4.copies9 ++ privateBatchWrapper4.copies10) ++ (privateBatchWrapper4.copies11 ++ privateBatchWrapper4.copies12)))) ++ (((privateBatchWrapper4.copies13 ++ (privateBatchWrapper4.copies14 ++ privateBatchWrapper4.copies15)) ++ ((privateBatchWrapper4.copies16 ++ privateBatchWrapper4.copies17) ++ (privateBatchWrapper4.copies18 ++ privateBatchWrapper4.copies19))) ++ ((privateBatchWrapper4.copies20 ++ (privateBatchWrapper4.copies21 ++ privateBatchWrapper4.copies22)) ++ ((privateBatchWrapper4.copies23 ++ privateBatchWrapper4.copies24) ++ (privateBatchWrapper4.copies25 ++ privateBatchWrapper4.copies26))))) ++ ((((privateBatchWrapper4.copies27 ++ (privateBatchWrapper4.copies28 ++ privateBatchWrapper4.copies29)) ++ (privateBatchWrapper4.copies30 ++ (privateBatchWrapper4.copies31 ++ privateBatchWrapper4.copies32))) ++ ((privateBatchWrapper4.copies33 ++ (privateBatchWrapper4.copies34 ++ privateBatchWrapper4.copies35)) ++ ((privateBatchWrapper4.copies36 ++ privateBatchWrapper4.copies37) ++ (privateBatchWrapper4.copies38 ++ privateBatchWrapper4.copies39)))) ++ (((privateBatchWrapper4.copies40 ++ (privateBatchWrapper4.copies41 ++ privateBatchWrapper4.copies42)) ++ ((privateBatchWrapper4.copies43 ++ privateBatchWrapper4.copies44) ++ (privateBatchWrapper4.copies45 ++ privateBatchWrapper4.copies46))) ++ ((privateBatchWrapper4.copies47 ++ (privateBatchWrapper4.copies48 ++ privateBatchWrapper4.copies49)) ++ ((privateBatchWrapper4.copies50 ++ privateBatchWrapper4.copies51) ++ (privateBatchWrapper4.copies52 ++ privateBatchWrapper4.copies53))))))
  constants := [
    (.virt 23709, 1),
    (.virt 23710, 0),
    (.virt 23711, 4),
    (.virt 23776, 576460752303423488 /- canonical u64; faithful only at p = goldilocks -/),
    (.virt 23777, 10000)
  ]
  publicInputs := [
    .virt 23711,
    .virt 5907,
    .virt 23672,
    .wire 10 23,
    .wire 10 31,
    .wire 10 39,
    .wire 10 47,
    .wire 10 55,
    .virt 23677,
    .virt 23673,
    .virt 23674,
    .virt 23675,
    .virt 23676,
    .virt 23682,
    .virt 23678,
    .virt 23679,
    .virt 23680,
    .virt 23681,
    .virt 23687,
    .virt 23683,
    .virt 23684,
    .virt 23685,
    .virt 23686,
    .virt 23692,
    .virt 23688,
    .virt 23689,
    .virt 23690,
    .virt 23691,
    .wire 33 43,
    .wire 33 51,
    .wire 33 59,
    .wire 37 7,
    .wire 37 15,
    .wire 37 23,
    .wire 37 31,
    .wire 37 39,
    .wire 37 47,
    .wire 37 55,
    .wire 42 3,
    .wire 42 11,
    .wire 42 19,
    .wire 42 27,
    .wire 42 35,
    .wire 42 43
  ]

/-- Named targets `leaf_pis_0`. -/
def privateBatchWrapper4.leaf_pis_0 : Fin 11 → Target :=
  ![.virt 5907, .virt 5908, .virt 5909, .virt 5910, .virt 5911, .virt 5912, .virt 5913, .virt 5914, .virt 5915, .virt 5916, .virt 5917]

/-- Named targets `leaf_pis_1`. -/
def privateBatchWrapper4.leaf_pis_1 : Fin 11 → Target :=
  ![.virt 11825, .virt 11826, .virt 11827, .virt 11828, .virt 11829, .virt 11830, .virt 11831, .virt 11832, .virt 11833, .virt 11834, .virt 11835]

/-- Named targets `leaf_pis_2`. -/
def privateBatchWrapper4.leaf_pis_2 : Fin 11 → Target :=
  ![.virt 17743, .virt 17744, .virt 17745, .virt 17746, .virt 17747, .virt 17748, .virt 17749, .virt 17750, .virt 17751, .virt 17752, .virt 17753]

/-- Named targets `leaf_pis_3`. -/
def privateBatchWrapper4.leaf_pis_3 : Fin 11 → Target :=
  ![.virt 23661, .virt 23662, .virt 23663, .virt 23664, .virt 23665, .virt 23666, .virt 23667, .virt 23668, .virt 23669, .virt 23670, .virt 23671]

/-- Named target `volume_fee_bps`. -/
def privateBatchWrapper4.volume_fee_bps : Target := .virt 23672

/-- Named target `exit_amount_0`. -/
def privateBatchWrapper4.exit_amount_0 : Target := .virt 23677

/-- Named targets `exit_account_0`. -/
def privateBatchWrapper4.exit_account_0 : Fin 4 → Target :=
  ![.virt 23673, .virt 23674, .virt 23675, .virt 23676]

/-- Named target `exit_amount_1`. -/
def privateBatchWrapper4.exit_amount_1 : Target := .virt 23682

/-- Named targets `exit_account_1`. -/
def privateBatchWrapper4.exit_account_1 : Fin 4 → Target :=
  ![.virt 23678, .virt 23679, .virt 23680, .virt 23681]

/-- Named target `exit_amount_2`. -/
def privateBatchWrapper4.exit_amount_2 : Target := .virt 23687

/-- Named targets `exit_account_2`. -/
def privateBatchWrapper4.exit_account_2 : Fin 4 → Target :=
  ![.virt 23683, .virt 23684, .virt 23685, .virt 23686]

/-- Named target `exit_amount_3`. -/
def privateBatchWrapper4.exit_amount_3 : Target := .virt 23692

/-- Named targets `exit_account_3`. -/
def privateBatchWrapper4.exit_account_3 : Fin 4 → Target :=
  ![.virt 23688, .virt 23689, .virt 23690, .virt 23691]

/-- Named targets `dummy_pre_image_0`. -/
def privateBatchWrapper4.dummy_pre_image_0 : Fin 4 → Target :=
  ![.virt 23693, .virt 23694, .virt 23695, .virt 23696]

/-- Named targets `dummy_pre_image_1`. -/
def privateBatchWrapper4.dummy_pre_image_1 : Fin 4 → Target :=
  ![.virt 23697, .virt 23698, .virt 23699, .virt 23700]

/-- Named targets `dummy_pre_image_2`. -/
def privateBatchWrapper4.dummy_pre_image_2 : Fin 4 → Target :=
  ![.virt 23701, .virt 23702, .virt 23703, .virt 23704]

/-- Named targets `dummy_pre_image_3`. -/
def privateBatchWrapper4.dummy_pre_image_3 : Fin 4 → Target :=
  ![.virt 23705, .virt 23706, .virt 23707, .virt 23708]

/-- The `verify_proof` gadgets the rows above do not contain: the child circuit's trace
    name and the targets carrying its public inputs (`Recursive.children`). -/
def privateBatchWrapper4.verifiers : List (String × List Target) :=
  [
    ("leaf_circuit", [.virt 5907, .virt 5908, .virt 5909, .virt 5910, .virt 5911, .virt 5912, .virt 5913, .virt 5914, .virt 5915, .virt 5916, .virt 5917]),
    ("leaf_circuit", [.virt 11825, .virt 11826, .virt 11827, .virt 11828, .virt 11829, .virt 11830, .virt 11831, .virt 11832, .virt 11833, .virt 11834, .virt 11835]),
    ("leaf_circuit", [.virt 17743, .virt 17744, .virt 17745, .virt 17746, .virt 17747, .virt 17748, .virt 17749, .virt 17750, .virt 17751, .virt 17752, .virt 17753]),
    ("leaf_circuit", [.virt 23661, .virt 23662, .virt 23663, .virt 23664, .virt 23665, .virt 23666, .virt 23667, .virt 23668, .virt 23669, .virt 23670, .virt 23671])
  ]

theorem privateBatchWrapper4_copies0 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 0 0) ∧ a (.virt 23709) = a (.wire 0 1) ∧ a (.virt 23712) = a (.wire 0 2) ∧ a (.virt 23712) = a (.wire 1 0) ∧ a (.virt 5912) = a (.wire 1 1) ∧ a (.virt 23712) = a (.wire 1 2) ∧ a (.virt 5912) = a (.wire 1 4) ∧ a (.virt 23713) = a (.wire 1 5) ∧ a (.virt 5912) = a (.wire 1 6) ∧ a (.wire 1 7) = a (.wire 0 4) ∧ a (.virt 23709) = a (.wire 0 5) ∧ a (.wire 0 3) = a (.wire 0 6) ∧ a (.wire 1 3) = a (.virt 23710) ∧ a (.wire 0 7) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 0 8) ∧ a (.virt 23709) = a (.wire 0 9) ∧ a (.virt 23714) = a (.wire 0 10) ∧ a (.virt 23714) = a (.wire 1 8) ∧ a (.virt 5913) = a (.wire 1 9) ∧ a (.virt 23714) = a (.wire 1 10) ∧ a (.virt 5913) = a (.wire 1 12) ∧ a (.virt 23715) = a (.wire 1 13) ∧ a (.virt 5913) = a (.wire 1 14) ∧ a (.wire 1 15) = a (.wire 0 12) ∧ a (.virt 23709) = a (.wire 0 13) ∧ a (.wire 0 11) = a (.wire 0 14) ∧ a (.wire 1 11) = a (.virt 23710) ∧ a (.wire 0 15) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 0 16) ∧ a (.virt 23709) = a (.wire 0 17) ∧ a (.virt 23716) = a (.wire 0 18) ∧ a (.virt 23716) = a (.wire 1 16) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies0, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies0, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies1 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 5914) = a (.wire 1 17) ∧ a (.virt 23716) = a (.wire 1 18) ∧ a (.virt 5914) = a (.wire 1 20) ∧ a (.virt 23717) = a (.wire 1 21) ∧ a (.virt 5914) = a (.wire 1 22) ∧ a (.wire 1 23) = a (.wire 0 20) ∧ a (.virt 23709) = a (.wire 0 21) ∧ a (.wire 0 19) = a (.wire 0 22) ∧ a (.wire 1 19) = a (.virt 23710) ∧ a (.wire 0 23) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 0 24) ∧ a (.virt 23709) = a (.wire 0 25) ∧ a (.virt 23718) = a (.wire 0 26) ∧ a (.virt 23718) = a (.wire 1 24) ∧ a (.virt 5915) = a (.wire 1 25) ∧ a (.virt 23718) = a (.wire 1 26) ∧ a (.virt 5915) = a (.wire 1 28) ∧ a (.virt 23719) = a (.wire 1 29) ∧ a (.virt 5915) = a (.wire 1 30) ∧ a (.wire 1 31) = a (.wire 0 28) ∧ a (.virt 23709) = a (.wire 0 29) ∧ a (.wire 0 27) = a (.wire 0 30) ∧ a (.wire 1 27) = a (.virt 23710) ∧ a (.wire 0 31) = a (.virt 23710) ∧ a (.virt 23712) = a (.wire 1 32) ∧ a (.virt 23714) = a (.wire 1 33) ∧ a (.virt 23712) = a (.wire 1 34) ∧ a (.virt 23716) = a (.wire 1 36) ∧ a (.virt 23718) = a (.wire 1 37) ∧ a (.virt 23716) = a (.wire 1 38) ∧ a (.wire 1 35) = a (.wire 1 40) ∧ a (.wire 1 39) = a (.wire 1 41) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies1, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies1, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies2 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 1 35) = a (.wire 1 42) ∧ a (.virt 23709) = a (.wire 0 32) ∧ a (.virt 23709) = a (.wire 0 33) ∧ a (.virt 23720) = a (.wire 0 34) ∧ a (.virt 23720) = a (.wire 1 44) ∧ a (.virt 11830) = a (.wire 1 45) ∧ a (.virt 23720) = a (.wire 1 46) ∧ a (.virt 11830) = a (.wire 1 48) ∧ a (.virt 23721) = a (.wire 1 49) ∧ a (.virt 11830) = a (.wire 1 50) ∧ a (.wire 1 51) = a (.wire 0 36) ∧ a (.virt 23709) = a (.wire 0 37) ∧ a (.wire 0 35) = a (.wire 0 38) ∧ a (.wire 1 47) = a (.virt 23710) ∧ a (.wire 0 39) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 0 40) ∧ a (.virt 23709) = a (.wire 0 41) ∧ a (.virt 23722) = a (.wire 0 42) ∧ a (.virt 23722) = a (.wire 1 52) ∧ a (.virt 11831) = a (.wire 1 53) ∧ a (.virt 23722) = a (.wire 1 54) ∧ a (.virt 11831) = a (.wire 1 56) ∧ a (.virt 23723) = a (.wire 1 57) ∧ a (.virt 11831) = a (.wire 1 58) ∧ a (.wire 1 59) = a (.wire 0 44) ∧ a (.virt 23709) = a (.wire 0 45) ∧ a (.wire 0 43) = a (.wire 0 46) ∧ a (.wire 1 55) = a (.virt 23710) ∧ a (.wire 0 47) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 0 48) ∧ a (.virt 23709) = a (.wire 0 49) ∧ a (.virt 23724) = a (.wire 0 50) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies2, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies2, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies3 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23724) = a (.wire 2 0) ∧ a (.virt 11832) = a (.wire 2 1) ∧ a (.virt 23724) = a (.wire 2 2) ∧ a (.virt 11832) = a (.wire 2 4) ∧ a (.virt 23725) = a (.wire 2 5) ∧ a (.virt 11832) = a (.wire 2 6) ∧ a (.wire 2 7) = a (.wire 0 52) ∧ a (.virt 23709) = a (.wire 0 53) ∧ a (.wire 0 51) = a (.wire 0 54) ∧ a (.wire 2 3) = a (.virt 23710) ∧ a (.wire 0 55) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 0 56) ∧ a (.virt 23709) = a (.wire 0 57) ∧ a (.virt 23726) = a (.wire 0 58) ∧ a (.virt 23726) = a (.wire 2 8) ∧ a (.virt 11833) = a (.wire 2 9) ∧ a (.virt 23726) = a (.wire 2 10) ∧ a (.virt 11833) = a (.wire 2 12) ∧ a (.virt 23727) = a (.wire 2 13) ∧ a (.virt 11833) = a (.wire 2 14) ∧ a (.wire 2 15) = a (.wire 3 0) ∧ a (.virt 23709) = a (.wire 3 1) ∧ a (.wire 0 59) = a (.wire 3 2) ∧ a (.wire 2 11) = a (.virt 23710) ∧ a (.wire 3 3) = a (.virt 23710) ∧ a (.virt 23720) = a (.wire 2 16) ∧ a (.virt 23722) = a (.wire 2 17) ∧ a (.virt 23720) = a (.wire 2 18) ∧ a (.virt 23724) = a (.wire 2 20) ∧ a (.virt 23726) = a (.wire 2 21) ∧ a (.virt 23724) = a (.wire 2 22) ∧ a (.wire 2 19) = a (.wire 2 24) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies3, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies3, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies4 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 2 23) = a (.wire 2 25) ∧ a (.wire 2 19) = a (.wire 2 26) ∧ a (.virt 23709) = a (.wire 3 4) ∧ a (.virt 23709) = a (.wire 3 5) ∧ a (.virt 23728) = a (.wire 3 6) ∧ a (.virt 23728) = a (.wire 2 28) ∧ a (.virt 17748) = a (.wire 2 29) ∧ a (.virt 23728) = a (.wire 2 30) ∧ a (.virt 17748) = a (.wire 2 32) ∧ a (.virt 23729) = a (.wire 2 33) ∧ a (.virt 17748) = a (.wire 2 34) ∧ a (.wire 2 35) = a (.wire 3 8) ∧ a (.virt 23709) = a (.wire 3 9) ∧ a (.wire 3 7) = a (.wire 3 10) ∧ a (.wire 2 31) = a (.virt 23710) ∧ a (.wire 3 11) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 3 12) ∧ a (.virt 23709) = a (.wire 3 13) ∧ a (.virt 23730) = a (.wire 3 14) ∧ a (.virt 23730) = a (.wire 2 36) ∧ a (.virt 17749) = a (.wire 2 37) ∧ a (.virt 23730) = a (.wire 2 38) ∧ a (.virt 17749) = a (.wire 2 40) ∧ a (.virt 23731) = a (.wire 2 41) ∧ a (.virt 17749) = a (.wire 2 42) ∧ a (.wire 2 43) = a (.wire 3 16) ∧ a (.virt 23709) = a (.wire 3 17) ∧ a (.wire 3 15) = a (.wire 3 18) ∧ a (.wire 2 39) = a (.virt 23710) ∧ a (.wire 3 19) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 3 20) ∧ a (.virt 23709) = a (.wire 3 21) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies4, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies5 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23732) = a (.wire 3 22) ∧ a (.virt 23732) = a (.wire 2 44) ∧ a (.virt 17750) = a (.wire 2 45) ∧ a (.virt 23732) = a (.wire 2 46) ∧ a (.virt 17750) = a (.wire 2 48) ∧ a (.virt 23733) = a (.wire 2 49) ∧ a (.virt 17750) = a (.wire 2 50) ∧ a (.wire 2 51) = a (.wire 3 24) ∧ a (.virt 23709) = a (.wire 3 25) ∧ a (.wire 3 23) = a (.wire 3 26) ∧ a (.wire 2 47) = a (.virt 23710) ∧ a (.wire 3 27) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 3 28) ∧ a (.virt 23709) = a (.wire 3 29) ∧ a (.virt 23734) = a (.wire 3 30) ∧ a (.virt 23734) = a (.wire 2 52) ∧ a (.virt 17751) = a (.wire 2 53) ∧ a (.virt 23734) = a (.wire 2 54) ∧ a (.virt 17751) = a (.wire 2 56) ∧ a (.virt 23735) = a (.wire 2 57) ∧ a (.virt 17751) = a (.wire 2 58) ∧ a (.wire 2 59) = a (.wire 3 32) ∧ a (.virt 23709) = a (.wire 3 33) ∧ a (.wire 3 31) = a (.wire 3 34) ∧ a (.wire 2 55) = a (.virt 23710) ∧ a (.wire 3 35) = a (.virt 23710) ∧ a (.virt 23728) = a (.wire 4 0) ∧ a (.virt 23730) = a (.wire 4 1) ∧ a (.virt 23728) = a (.wire 4 2) ∧ a (.virt 23732) = a (.wire 4 4) ∧ a (.virt 23734) = a (.wire 4 5) ∧ a (.virt 23732) = a (.wire 4 6) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies5, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies5, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies6 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 3) = a (.wire 4 8) ∧ a (.wire 4 7) = a (.wire 4 9) ∧ a (.wire 4 3) = a (.wire 4 10) ∧ a (.virt 23709) = a (.wire 3 36) ∧ a (.virt 23709) = a (.wire 3 37) ∧ a (.virt 23736) = a (.wire 3 38) ∧ a (.virt 23736) = a (.wire 4 12) ∧ a (.virt 23666) = a (.wire 4 13) ∧ a (.virt 23736) = a (.wire 4 14) ∧ a (.virt 23666) = a (.wire 4 16) ∧ a (.virt 23737) = a (.wire 4 17) ∧ a (.virt 23666) = a (.wire 4 18) ∧ a (.wire 4 19) = a (.wire 3 40) ∧ a (.virt 23709) = a (.wire 3 41) ∧ a (.wire 3 39) = a (.wire 3 42) ∧ a (.wire 4 15) = a (.virt 23710) ∧ a (.wire 3 43) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 3 44) ∧ a (.virt 23709) = a (.wire 3 45) ∧ a (.virt 23738) = a (.wire 3 46) ∧ a (.virt 23738) = a (.wire 4 20) ∧ a (.virt 23667) = a (.wire 4 21) ∧ a (.virt 23738) = a (.wire 4 22) ∧ a (.virt 23667) = a (.wire 4 24) ∧ a (.virt 23739) = a (.wire 4 25) ∧ a (.virt 23667) = a (.wire 4 26) ∧ a (.wire 4 27) = a (.wire 3 48) ∧ a (.virt 23709) = a (.wire 3 49) ∧ a (.wire 3 47) = a (.wire 3 50) ∧ a (.wire 4 23) = a (.virt 23710) ∧ a (.wire 3 51) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 3 52) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies6, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies6, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies7 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 3 53) ∧ a (.virt 23740) = a (.wire 3 54) ∧ a (.virt 23740) = a (.wire 4 28) ∧ a (.virt 23668) = a (.wire 4 29) ∧ a (.virt 23740) = a (.wire 4 30) ∧ a (.virt 23668) = a (.wire 4 32) ∧ a (.virt 23741) = a (.wire 4 33) ∧ a (.virt 23668) = a (.wire 4 34) ∧ a (.wire 4 35) = a (.wire 3 56) ∧ a (.virt 23709) = a (.wire 3 57) ∧ a (.wire 3 55) = a (.wire 3 58) ∧ a (.wire 4 31) = a (.virt 23710) ∧ a (.wire 3 59) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 5 0) ∧ a (.virt 23709) = a (.wire 5 1) ∧ a (.virt 23742) = a (.wire 5 2) ∧ a (.virt 23742) = a (.wire 4 36) ∧ a (.virt 23669) = a (.wire 4 37) ∧ a (.virt 23742) = a (.wire 4 38) ∧ a (.virt 23669) = a (.wire 4 40) ∧ a (.virt 23743) = a (.wire 4 41) ∧ a (.virt 23669) = a (.wire 4 42) ∧ a (.wire 4 43) = a (.wire 5 4) ∧ a (.virt 23709) = a (.wire 5 5) ∧ a (.wire 5 3) = a (.wire 5 6) ∧ a (.wire 4 39) = a (.virt 23710) ∧ a (.wire 5 7) = a (.virt 23710) ∧ a (.virt 23736) = a (.wire 4 44) ∧ a (.virt 23738) = a (.wire 4 45) ∧ a (.virt 23736) = a (.wire 4 46) ∧ a (.virt 23740) = a (.wire 4 48) ∧ a (.virt 23742) = a (.wire 4 49) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies7, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies7, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies8 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23740) = a (.wire 4 50) ∧ a (.wire 4 47) = a (.wire 4 52) ∧ a (.wire 4 51) = a (.wire 4 53) ∧ a (.wire 4 47) = a (.wire 4 54) ∧ a (.virt 23709) = a (.wire 5 8) ∧ a (.virt 23709) = a (.wire 5 9) ∧ a (.wire 1 43) = a (.wire 5 10) ∧ a (.wire 5 11) = a (.wire 5 12) ∧ a (.virt 5912) = a (.wire 5 13) ∧ a (.virt 23710) = a (.wire 5 14) ∧ a (.wire 5 11) = a (.wire 5 16) ∧ a (.virt 5913) = a (.wire 5 17) ∧ a (.virt 23710) = a (.wire 5 18) ∧ a (.wire 5 11) = a (.wire 5 20) ∧ a (.virt 5914) = a (.wire 5 21) ∧ a (.virt 23710) = a (.wire 5 22) ∧ a (.wire 5 11) = a (.wire 5 24) ∧ a (.virt 5915) = a (.wire 5 25) ∧ a (.virt 23710) = a (.wire 5 26) ∧ a (.wire 5 11) = a (.wire 5 28) ∧ a (.virt 5916) = a (.wire 5 29) ∧ a (.virt 23710) = a (.wire 5 30) ∧ a (.virt 23709) = a (.wire 5 32) ∧ a (.virt 23709) = a (.wire 5 33) ∧ a (.wire 2 27) = a (.wire 5 34) ∧ a (.virt 23709) = a (.wire 5 36) ∧ a (.virt 23709) = a (.wire 5 37) ∧ a (.wire 5 11) = a (.wire 5 38) ∧ a (.wire 5 35) = a (.wire 4 56) ∧ a (.wire 5 39) = a (.wire 4 57) ∧ a (.wire 5 35) = a (.wire 4 58) ∧ a (.wire 4 59) = a (.wire 5 40) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies8, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies8, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies9 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 15) = a (.wire 5 41) ∧ a (.wire 5 15) = a (.wire 5 42) ∧ a (.wire 4 59) = a (.wire 5 44) ∧ a (.virt 11830) = a (.wire 5 45) ∧ a (.wire 5 43) = a (.wire 5 46) ∧ a (.wire 4 59) = a (.wire 5 48) ∧ a (.wire 5 19) = a (.wire 5 49) ∧ a (.wire 5 19) = a (.wire 5 50) ∧ a (.wire 4 59) = a (.wire 5 52) ∧ a (.virt 11831) = a (.wire 5 53) ∧ a (.wire 5 51) = a (.wire 5 54) ∧ a (.wire 4 59) = a (.wire 5 56) ∧ a (.wire 5 23) = a (.wire 5 57) ∧ a (.wire 5 23) = a (.wire 5 58) ∧ a (.wire 4 59) = a (.wire 6 0) ∧ a (.virt 11832) = a (.wire 6 1) ∧ a (.wire 5 59) = a (.wire 6 2) ∧ a (.wire 4 59) = a (.wire 6 4) ∧ a (.wire 5 27) = a (.wire 6 5) ∧ a (.wire 5 27) = a (.wire 6 6) ∧ a (.wire 4 59) = a (.wire 6 8) ∧ a (.virt 11833) = a (.wire 6 9) ∧ a (.wire 6 7) = a (.wire 6 10) ∧ a (.wire 4 59) = a (.wire 6 12) ∧ a (.wire 5 31) = a (.wire 6 13) ∧ a (.wire 5 31) = a (.wire 6 14) ∧ a (.wire 4 59) = a (.wire 6 16) ∧ a (.virt 11834) = a (.wire 6 17) ∧ a (.wire 6 15) = a (.wire 6 18) ∧ a (.wire 5 11) = a (.wire 7 0) ∧ a (.wire 5 35) = a (.wire 7 1) ∧ a (.wire 5 11) = a (.wire 7 2) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies9, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies9, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies10 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 7 3) = a (.wire 8 0) ∧ a (.virt 23709) = a (.wire 8 1) ∧ a (.wire 5 35) = a (.wire 8 2) ∧ a (.virt 23709) = a (.wire 6 20) ∧ a (.virt 23709) = a (.wire 6 21) ∧ a (.wire 4 11) = a (.wire 6 22) ∧ a (.virt 23709) = a (.wire 6 24) ∧ a (.virt 23709) = a (.wire 6 25) ∧ a (.wire 8 3) = a (.wire 6 26) ∧ a (.wire 6 23) = a (.wire 9 0) ∧ a (.wire 6 27) = a (.wire 9 1) ∧ a (.wire 6 23) = a (.wire 9 2) ∧ a (.wire 9 3) = a (.wire 6 28) ∧ a (.wire 5 47) = a (.wire 6 29) ∧ a (.wire 5 47) = a (.wire 6 30) ∧ a (.wire 9 3) = a (.wire 6 32) ∧ a (.virt 17748) = a (.wire 6 33) ∧ a (.wire 6 31) = a (.wire 6 34) ∧ a (.wire 9 3) = a (.wire 6 36) ∧ a (.wire 5 55) = a (.wire 6 37) ∧ a (.wire 5 55) = a (.wire 6 38) ∧ a (.wire 9 3) = a (.wire 6 40) ∧ a (.virt 17749) = a (.wire 6 41) ∧ a (.wire 6 39) = a (.wire 6 42) ∧ a (.wire 9 3) = a (.wire 6 44) ∧ a (.wire 6 3) = a (.wire 6 45) ∧ a (.wire 6 3) = a (.wire 6 46) ∧ a (.wire 9 3) = a (.wire 6 48) ∧ a (.virt 17750) = a (.wire 6 49) ∧ a (.wire 6 47) = a (.wire 6 50) ∧ a (.wire 9 3) = a (.wire 6 52) ∧ a (.wire 6 11) = a (.wire 6 53) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies10, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies10, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies11 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 11) = a (.wire 6 54) ∧ a (.wire 9 3) = a (.wire 6 56) ∧ a (.virt 17751) = a (.wire 6 57) ∧ a (.wire 6 55) = a (.wire 6 58) ∧ a (.wire 9 3) = a (.wire 10 0) ∧ a (.wire 6 19) = a (.wire 10 1) ∧ a (.wire 6 19) = a (.wire 10 2) ∧ a (.wire 9 3) = a (.wire 10 4) ∧ a (.virt 17752) = a (.wire 10 5) ∧ a (.wire 10 3) = a (.wire 10 6) ∧ a (.wire 8 3) = a (.wire 7 4) ∧ a (.wire 6 23) = a (.wire 7 5) ∧ a (.wire 8 3) = a (.wire 7 6) ∧ a (.wire 7 7) = a (.wire 8 4) ∧ a (.virt 23709) = a (.wire 8 5) ∧ a (.wire 6 23) = a (.wire 8 6) ∧ a (.virt 23709) = a (.wire 10 8) ∧ a (.virt 23709) = a (.wire 10 9) ∧ a (.wire 4 55) = a (.wire 10 10) ∧ a (.virt 23709) = a (.wire 10 12) ∧ a (.virt 23709) = a (.wire 10 13) ∧ a (.wire 8 7) = a (.wire 10 14) ∧ a (.wire 10 11) = a (.wire 9 4) ∧ a (.wire 10 15) = a (.wire 9 5) ∧ a (.wire 10 11) = a (.wire 9 6) ∧ a (.wire 9 7) = a (.wire 10 16) ∧ a (.wire 6 35) = a (.wire 10 17) ∧ a (.wire 6 35) = a (.wire 10 18) ∧ a (.wire 9 7) = a (.wire 10 20) ∧ a (.virt 23666) = a (.wire 10 21) ∧ a (.wire 10 19) = a (.wire 10 22) ∧ a (.wire 9 7) = a (.wire 10 24) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies11, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies11, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies12 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 43) = a (.wire 10 25) ∧ a (.wire 6 43) = a (.wire 10 26) ∧ a (.wire 9 7) = a (.wire 10 28) ∧ a (.virt 23667) = a (.wire 10 29) ∧ a (.wire 10 27) = a (.wire 10 30) ∧ a (.wire 9 7) = a (.wire 10 32) ∧ a (.wire 6 51) = a (.wire 10 33) ∧ a (.wire 6 51) = a (.wire 10 34) ∧ a (.wire 9 7) = a (.wire 10 36) ∧ a (.virt 23668) = a (.wire 10 37) ∧ a (.wire 10 35) = a (.wire 10 38) ∧ a (.wire 9 7) = a (.wire 10 40) ∧ a (.wire 6 59) = a (.wire 10 41) ∧ a (.wire 6 59) = a (.wire 10 42) ∧ a (.wire 9 7) = a (.wire 10 44) ∧ a (.virt 23669) = a (.wire 10 45) ∧ a (.wire 10 43) = a (.wire 10 46) ∧ a (.wire 9 7) = a (.wire 10 48) ∧ a (.wire 10 7) = a (.wire 10 49) ∧ a (.wire 10 7) = a (.wire 10 50) ∧ a (.wire 9 7) = a (.wire 10 52) ∧ a (.virt 23670) = a (.wire 10 53) ∧ a (.wire 10 51) = a (.wire 10 54) ∧ a (.wire 8 7) = a (.wire 7 8) ∧ a (.wire 10 11) = a (.wire 7 9) ∧ a (.wire 8 7) = a (.wire 7 10) ∧ a (.wire 7 11) = a (.wire 8 8) ∧ a (.virt 23709) = a (.wire 8 9) ∧ a (.wire 10 11) = a (.wire 8 10) ∧ a (.virt 23709) = a (.wire 10 56) ∧ a (.virt 23709) = a (.wire 10 57) ∧ a (.virt 23744) = a (.wire 10 58) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies12, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies12, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies13 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 5912) = a (.wire 11 0) ∧ a (.virt 23709) = a (.wire 11 1) ∧ a (.wire 10 23) = a (.wire 11 2) ∧ a (.virt 23744) = a (.wire 9 8) ∧ a (.wire 11 3) = a (.wire 9 9) ∧ a (.virt 23744) = a (.wire 9 10) ∧ a (.wire 11 3) = a (.wire 9 12) ∧ a (.virt 23745) = a (.wire 9 13) ∧ a (.wire 11 3) = a (.wire 9 14) ∧ a (.wire 9 15) = a (.wire 11 4) ∧ a (.virt 23709) = a (.wire 11 5) ∧ a (.wire 10 59) = a (.wire 11 6) ∧ a (.wire 9 11) = a (.virt 23710) ∧ a (.wire 11 7) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 11 8) ∧ a (.virt 23709) = a (.wire 11 9) ∧ a (.virt 23746) = a (.wire 11 10) ∧ a (.virt 5913) = a (.wire 11 12) ∧ a (.virt 23709) = a (.wire 11 13) ∧ a (.wire 10 31) = a (.wire 11 14) ∧ a (.virt 23746) = a (.wire 9 16) ∧ a (.wire 11 15) = a (.wire 9 17) ∧ a (.virt 23746) = a (.wire 9 18) ∧ a (.wire 11 15) = a (.wire 9 20) ∧ a (.virt 23747) = a (.wire 9 21) ∧ a (.wire 11 15) = a (.wire 9 22) ∧ a (.wire 9 23) = a (.wire 11 16) ∧ a (.virt 23709) = a (.wire 11 17) ∧ a (.wire 11 11) = a (.wire 11 18) ∧ a (.wire 9 19) = a (.virt 23710) ∧ a (.wire 11 19) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 11 20) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies13, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies13, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies14 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 11 21) ∧ a (.virt 23748) = a (.wire 11 22) ∧ a (.virt 5914) = a (.wire 11 24) ∧ a (.virt 23709) = a (.wire 11 25) ∧ a (.wire 10 39) = a (.wire 11 26) ∧ a (.virt 23748) = a (.wire 9 24) ∧ a (.wire 11 27) = a (.wire 9 25) ∧ a (.virt 23748) = a (.wire 9 26) ∧ a (.wire 11 27) = a (.wire 9 28) ∧ a (.virt 23749) = a (.wire 9 29) ∧ a (.wire 11 27) = a (.wire 9 30) ∧ a (.wire 9 31) = a (.wire 11 28) ∧ a (.virt 23709) = a (.wire 11 29) ∧ a (.wire 11 23) = a (.wire 11 30) ∧ a (.wire 9 27) = a (.virt 23710) ∧ a (.wire 11 31) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 11 32) ∧ a (.virt 23709) = a (.wire 11 33) ∧ a (.virt 23750) = a (.wire 11 34) ∧ a (.virt 5915) = a (.wire 11 36) ∧ a (.virt 23709) = a (.wire 11 37) ∧ a (.wire 10 47) = a (.wire 11 38) ∧ a (.virt 23750) = a (.wire 9 32) ∧ a (.wire 11 39) = a (.wire 9 33) ∧ a (.virt 23750) = a (.wire 9 34) ∧ a (.wire 11 39) = a (.wire 9 36) ∧ a (.virt 23751) = a (.wire 9 37) ∧ a (.wire 11 39) = a (.wire 9 38) ∧ a (.wire 9 39) = a (.wire 11 40) ∧ a (.virt 23709) = a (.wire 11 41) ∧ a (.wire 11 35) = a (.wire 11 42) ∧ a (.wire 9 35) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies14, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies14, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies15 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 11 43) = a (.virt 23710) ∧ a (.virt 23744) = a (.wire 9 40) ∧ a (.virt 23746) = a (.wire 9 41) ∧ a (.virt 23744) = a (.wire 9 42) ∧ a (.virt 23748) = a (.wire 9 44) ∧ a (.virt 23750) = a (.wire 9 45) ∧ a (.virt 23748) = a (.wire 9 46) ∧ a (.wire 9 43) = a (.wire 9 48) ∧ a (.wire 9 47) = a (.wire 9 49) ∧ a (.wire 9 43) = a (.wire 9 50) ∧ a (.wire 1 43) = a (.wire 7 12) ∧ a (.wire 9 51) = a (.wire 7 13) ∧ a (.wire 1 43) = a (.wire 7 14) ∧ a (.wire 7 15) = a (.wire 8 12) ∧ a (.virt 23709) = a (.wire 8 13) ∧ a (.wire 9 51) = a (.wire 8 14) ∧ a (.wire 8 15) = a (.virt 23709) ∧ True ∧ a (.virt 23709) = a (.wire 11 44) ∧ a (.virt 23709) = a (.wire 11 45) ∧ a (.virt 23752) = a (.wire 11 46) ∧ a (.virt 11830) = a (.wire 11 48) ∧ a (.virt 23709) = a (.wire 11 49) ∧ a (.wire 10 23) = a (.wire 11 50) ∧ a (.virt 23752) = a (.wire 9 52) ∧ a (.wire 11 51) = a (.wire 9 53) ∧ a (.virt 23752) = a (.wire 9 54) ∧ a (.wire 11 51) = a (.wire 9 56) ∧ a (.virt 23753) = a (.wire 9 57) ∧ a (.wire 11 51) = a (.wire 9 58) ∧ a (.wire 9 59) = a (.wire 11 52) ∧ a (.virt 23709) = a (.wire 11 53) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies15, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies15, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies16 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 11 47) = a (.wire 11 54) ∧ a (.wire 9 55) = a (.virt 23710) ∧ a (.wire 11 55) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 11 56) ∧ a (.virt 23709) = a (.wire 11 57) ∧ a (.virt 23754) = a (.wire 11 58) ∧ a (.virt 11831) = a (.wire 12 0) ∧ a (.virt 23709) = a (.wire 12 1) ∧ a (.wire 10 31) = a (.wire 12 2) ∧ a (.virt 23754) = a (.wire 13 0) ∧ a (.wire 12 3) = a (.wire 13 1) ∧ a (.virt 23754) = a (.wire 13 2) ∧ a (.wire 12 3) = a (.wire 13 4) ∧ a (.virt 23755) = a (.wire 13 5) ∧ a (.wire 12 3) = a (.wire 13 6) ∧ a (.wire 13 7) = a (.wire 12 4) ∧ a (.virt 23709) = a (.wire 12 5) ∧ a (.wire 11 59) = a (.wire 12 6) ∧ a (.wire 13 3) = a (.virt 23710) ∧ a (.wire 12 7) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 12 8) ∧ a (.virt 23709) = a (.wire 12 9) ∧ a (.virt 23756) = a (.wire 12 10) ∧ a (.virt 11832) = a (.wire 12 12) ∧ a (.virt 23709) = a (.wire 12 13) ∧ a (.wire 10 39) = a (.wire 12 14) ∧ a (.virt 23756) = a (.wire 13 8) ∧ a (.wire 12 15) = a (.wire 13 9) ∧ a (.virt 23756) = a (.wire 13 10) ∧ a (.wire 12 15) = a (.wire 13 12) ∧ a (.virt 23757) = a (.wire 13 13) ∧ a (.wire 12 15) = a (.wire 13 14) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies16, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies16, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies17 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 13 15) = a (.wire 12 16) ∧ a (.virt 23709) = a (.wire 12 17) ∧ a (.wire 12 11) = a (.wire 12 18) ∧ a (.wire 13 11) = a (.virt 23710) ∧ a (.wire 12 19) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 12 20) ∧ a (.virt 23709) = a (.wire 12 21) ∧ a (.virt 23758) = a (.wire 12 22) ∧ a (.virt 11833) = a (.wire 12 24) ∧ a (.virt 23709) = a (.wire 12 25) ∧ a (.wire 10 47) = a (.wire 12 26) ∧ a (.virt 23758) = a (.wire 13 16) ∧ a (.wire 12 27) = a (.wire 13 17) ∧ a (.virt 23758) = a (.wire 13 18) ∧ a (.wire 12 27) = a (.wire 13 20) ∧ a (.virt 23759) = a (.wire 13 21) ∧ a (.wire 12 27) = a (.wire 13 22) ∧ a (.wire 13 23) = a (.wire 12 28) ∧ a (.virt 23709) = a (.wire 12 29) ∧ a (.wire 12 23) = a (.wire 12 30) ∧ a (.wire 13 19) = a (.virt 23710) ∧ a (.wire 12 31) = a (.virt 23710) ∧ a (.virt 23752) = a (.wire 13 24) ∧ a (.virt 23754) = a (.wire 13 25) ∧ a (.virt 23752) = a (.wire 13 26) ∧ a (.virt 23756) = a (.wire 13 28) ∧ a (.virt 23758) = a (.wire 13 29) ∧ a (.virt 23756) = a (.wire 13 30) ∧ a (.wire 13 27) = a (.wire 13 32) ∧ a (.wire 13 31) = a (.wire 13 33) ∧ a (.wire 13 27) = a (.wire 13 34) ∧ a (.wire 2 27) = a (.wire 7 16) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies17, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies17, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies18 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 13 35) = a (.wire 7 17) ∧ a (.wire 2 27) = a (.wire 7 18) ∧ a (.wire 7 19) = a (.wire 8 16) ∧ a (.virt 23709) = a (.wire 8 17) ∧ a (.wire 13 35) = a (.wire 8 18) ∧ a (.wire 8 19) = a (.virt 23709) ∧ a (.virt 11825) = a (.virt 5907) ∧ a (.virt 23709) = a (.wire 12 32) ∧ a (.virt 23709) = a (.wire 12 33) ∧ a (.virt 23760) = a (.wire 12 34) ∧ a (.virt 17748) = a (.wire 12 36) ∧ a (.virt 23709) = a (.wire 12 37) ∧ a (.wire 10 23) = a (.wire 12 38) ∧ a (.virt 23760) = a (.wire 13 36) ∧ a (.wire 12 39) = a (.wire 13 37) ∧ a (.virt 23760) = a (.wire 13 38) ∧ a (.wire 12 39) = a (.wire 13 40) ∧ a (.virt 23761) = a (.wire 13 41) ∧ a (.wire 12 39) = a (.wire 13 42) ∧ a (.wire 13 43) = a (.wire 12 40) ∧ a (.virt 23709) = a (.wire 12 41) ∧ a (.wire 12 35) = a (.wire 12 42) ∧ a (.wire 13 39) = a (.virt 23710) ∧ a (.wire 12 43) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 12 44) ∧ a (.virt 23709) = a (.wire 12 45) ∧ a (.virt 23762) = a (.wire 12 46) ∧ a (.virt 17749) = a (.wire 12 48) ∧ a (.virt 23709) = a (.wire 12 49) ∧ a (.wire 10 31) = a (.wire 12 50) ∧ a (.virt 23762) = a (.wire 13 44) ∧ a (.wire 12 51) = a (.wire 13 45) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies18, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies18, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies19 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23762) = a (.wire 13 46) ∧ a (.wire 12 51) = a (.wire 13 48) ∧ a (.virt 23763) = a (.wire 13 49) ∧ a (.wire 12 51) = a (.wire 13 50) ∧ a (.wire 13 51) = a (.wire 12 52) ∧ a (.virt 23709) = a (.wire 12 53) ∧ a (.wire 12 47) = a (.wire 12 54) ∧ a (.wire 13 47) = a (.virt 23710) ∧ a (.wire 12 55) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 12 56) ∧ a (.virt 23709) = a (.wire 12 57) ∧ a (.virt 23764) = a (.wire 12 58) ∧ a (.virt 17750) = a (.wire 14 0) ∧ a (.virt 23709) = a (.wire 14 1) ∧ a (.wire 10 39) = a (.wire 14 2) ∧ a (.virt 23764) = a (.wire 13 52) ∧ a (.wire 14 3) = a (.wire 13 53) ∧ a (.virt 23764) = a (.wire 13 54) ∧ a (.wire 14 3) = a (.wire 13 56) ∧ a (.virt 23765) = a (.wire 13 57) ∧ a (.wire 14 3) = a (.wire 13 58) ∧ a (.wire 13 59) = a (.wire 14 4) ∧ a (.virt 23709) = a (.wire 14 5) ∧ a (.wire 12 59) = a (.wire 14 6) ∧ a (.wire 13 55) = a (.virt 23710) ∧ a (.wire 14 7) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 14 8) ∧ a (.virt 23709) = a (.wire 14 9) ∧ a (.virt 23766) = a (.wire 14 10) ∧ a (.virt 17751) = a (.wire 14 12) ∧ a (.virt 23709) = a (.wire 14 13) ∧ a (.wire 10 47) = a (.wire 14 14) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies19, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies19, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies20 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23766) = a (.wire 15 0) ∧ a (.wire 14 15) = a (.wire 15 1) ∧ a (.virt 23766) = a (.wire 15 2) ∧ a (.wire 14 15) = a (.wire 15 4) ∧ a (.virt 23767) = a (.wire 15 5) ∧ a (.wire 14 15) = a (.wire 15 6) ∧ a (.wire 15 7) = a (.wire 14 16) ∧ a (.virt 23709) = a (.wire 14 17) ∧ a (.wire 14 11) = a (.wire 14 18) ∧ a (.wire 15 3) = a (.virt 23710) ∧ a (.wire 14 19) = a (.virt 23710) ∧ a (.virt 23760) = a (.wire 15 8) ∧ a (.virt 23762) = a (.wire 15 9) ∧ a (.virt 23760) = a (.wire 15 10) ∧ a (.virt 23764) = a (.wire 15 12) ∧ a (.virt 23766) = a (.wire 15 13) ∧ a (.virt 23764) = a (.wire 15 14) ∧ a (.wire 15 11) = a (.wire 15 16) ∧ a (.wire 15 15) = a (.wire 15 17) ∧ a (.wire 15 11) = a (.wire 15 18) ∧ a (.wire 4 11) = a (.wire 7 20) ∧ a (.wire 15 19) = a (.wire 7 21) ∧ a (.wire 4 11) = a (.wire 7 22) ∧ a (.wire 7 23) = a (.wire 8 20) ∧ a (.virt 23709) = a (.wire 8 21) ∧ a (.wire 15 19) = a (.wire 8 22) ∧ a (.wire 8 23) = a (.virt 23709) ∧ a (.virt 17743) = a (.virt 5907) ∧ a (.virt 23709) = a (.wire 14 20) ∧ a (.virt 23709) = a (.wire 14 21) ∧ a (.virt 23768) = a (.wire 14 22) ∧ a (.virt 23666) = a (.wire 14 24) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies20, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies20, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies21 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 14 25) ∧ a (.wire 10 23) = a (.wire 14 26) ∧ a (.virt 23768) = a (.wire 15 20) ∧ a (.wire 14 27) = a (.wire 15 21) ∧ a (.virt 23768) = a (.wire 15 22) ∧ a (.wire 14 27) = a (.wire 15 24) ∧ a (.virt 23769) = a (.wire 15 25) ∧ a (.wire 14 27) = a (.wire 15 26) ∧ a (.wire 15 27) = a (.wire 14 28) ∧ a (.virt 23709) = a (.wire 14 29) ∧ a (.wire 14 23) = a (.wire 14 30) ∧ a (.wire 15 23) = a (.virt 23710) ∧ a (.wire 14 31) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 14 32) ∧ a (.virt 23709) = a (.wire 14 33) ∧ a (.virt 23770) = a (.wire 14 34) ∧ a (.virt 23667) = a (.wire 14 36) ∧ a (.virt 23709) = a (.wire 14 37) ∧ a (.wire 10 31) = a (.wire 14 38) ∧ a (.virt 23770) = a (.wire 15 28) ∧ a (.wire 14 39) = a (.wire 15 29) ∧ a (.virt 23770) = a (.wire 15 30) ∧ a (.wire 14 39) = a (.wire 15 32) ∧ a (.virt 23771) = a (.wire 15 33) ∧ a (.wire 14 39) = a (.wire 15 34) ∧ a (.wire 15 35) = a (.wire 14 40) ∧ a (.virt 23709) = a (.wire 14 41) ∧ a (.wire 14 35) = a (.wire 14 42) ∧ a (.wire 15 31) = a (.virt 23710) ∧ a (.wire 14 43) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 14 44) ∧ a (.virt 23709) = a (.wire 14 45) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies21, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies21, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies22 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23772) = a (.wire 14 46) ∧ a (.virt 23668) = a (.wire 14 48) ∧ a (.virt 23709) = a (.wire 14 49) ∧ a (.wire 10 39) = a (.wire 14 50) ∧ a (.virt 23772) = a (.wire 15 36) ∧ a (.wire 14 51) = a (.wire 15 37) ∧ a (.virt 23772) = a (.wire 15 38) ∧ a (.wire 14 51) = a (.wire 15 40) ∧ a (.virt 23773) = a (.wire 15 41) ∧ a (.wire 14 51) = a (.wire 15 42) ∧ a (.wire 15 43) = a (.wire 14 52) ∧ a (.virt 23709) = a (.wire 14 53) ∧ a (.wire 14 47) = a (.wire 14 54) ∧ a (.wire 15 39) = a (.virt 23710) ∧ a (.wire 14 55) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 14 56) ∧ a (.virt 23709) = a (.wire 14 57) ∧ a (.virt 23774) = a (.wire 14 58) ∧ a (.virt 23669) = a (.wire 16 0) ∧ a (.virt 23709) = a (.wire 16 1) ∧ a (.wire 10 47) = a (.wire 16 2) ∧ a (.virt 23774) = a (.wire 15 44) ∧ a (.wire 16 3) = a (.wire 15 45) ∧ a (.virt 23774) = a (.wire 15 46) ∧ a (.wire 16 3) = a (.wire 15 48) ∧ a (.virt 23775) = a (.wire 15 49) ∧ a (.wire 16 3) = a (.wire 15 50) ∧ a (.wire 15 51) = a (.wire 16 4) ∧ a (.virt 23709) = a (.wire 16 5) ∧ a (.wire 14 59) = a (.wire 16 6) ∧ a (.wire 15 47) = a (.virt 23710) ∧ a (.wire 16 7) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies22, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies22, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies23 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23768) = a (.wire 15 52) ∧ a (.virt 23770) = a (.wire 15 53) ∧ a (.virt 23768) = a (.wire 15 54) ∧ a (.virt 23772) = a (.wire 15 56) ∧ a (.virt 23774) = a (.wire 15 57) ∧ a (.virt 23772) = a (.wire 15 58) ∧ a (.wire 15 55) = a (.wire 17 0) ∧ a (.wire 15 59) = a (.wire 17 1) ∧ a (.wire 15 55) = a (.wire 17 2) ∧ a (.wire 4 55) = a (.wire 7 24) ∧ a (.wire 17 3) = a (.wire 7 25) ∧ a (.wire 4 55) = a (.wire 7 26) ∧ a (.wire 7 27) = a (.wire 8 24) ∧ a (.virt 23709) = a (.wire 8 25) ∧ a (.wire 17 3) = a (.wire 8 26) ∧ a (.wire 8 27) = a (.virt 23709) ∧ a (.virt 23661) = a (.virt 5907) ∧ a (.wire 1 43) = a (.wire 16 8) ∧ a (.virt 5917) = a (.wire 16 9) ∧ a (.virt 5917) = a (.wire 16 10) ∧ a (.wire 1 43) = a (.wire 16 12) ∧ a (.virt 23710) = a (.wire 16 13) ∧ a (.wire 16 11) = a (.wire 16 14) ∧ a (.wire 2 27) = a (.wire 16 16) ∧ a (.virt 11835) = a (.wire 16 17) ∧ a (.virt 11835) = a (.wire 16 18) ∧ a (.wire 2 27) = a (.wire 16 20) ∧ a (.virt 23710) = a (.wire 16 21) ∧ a (.wire 16 19) = a (.wire 16 22) ∧ a (.wire 16 15) = a (.wire 8 28) ∧ a (.virt 23709) = a (.wire 8 29) ∧ a (.wire 16 23) = a (.wire 8 30) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies23, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies23, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies24 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 11) = a (.wire 16 24) ∧ a (.virt 17753) = a (.wire 16 25) ∧ a (.virt 17753) = a (.wire 16 26) ∧ a (.wire 4 11) = a (.wire 16 28) ∧ a (.virt 23710) = a (.wire 16 29) ∧ a (.wire 16 27) = a (.wire 16 30) ∧ a (.wire 8 31) = a (.wire 8 32) ∧ a (.virt 23709) = a (.wire 8 33) ∧ a (.wire 16 31) = a (.wire 8 34) ∧ a (.wire 4 55) = a (.wire 16 32) ∧ a (.virt 23671) = a (.wire 16 33) ∧ a (.virt 23671) = a (.wire 16 34) ∧ a (.wire 4 55) = a (.wire 16 36) ∧ a (.virt 23710) = a (.wire 16 37) ∧ a (.wire 16 35) = a (.wire 16 38) ∧ a (.wire 8 35) = a (.wire 8 36) ∧ a (.virt 23709) = a (.wire 8 37) ∧ a (.wire 16 39) = a (.wire 8 38) ∧ a (.wire 18 33) = a (.virt 23710) ∧ a (.wire 18 34) = a (.virt 23710) ∧ a (.wire 18 35) = a (.virt 23710) ∧ a (.wire 18 36) = a (.virt 23710) ∧ a (.wire 18 37) = a (.virt 23710) ∧ a (.wire 18 38) = a (.virt 23710) ∧ a (.wire 18 39) = a (.virt 23710) ∧ a (.wire 18 40) = a (.virt 23710) ∧ a (.wire 18 41) = a (.virt 23710) ∧ a (.wire 18 42) = a (.virt 23710) ∧ a (.wire 18 43) = a (.virt 23710) ∧ a (.wire 18 44) = a (.virt 23710) ∧ a (.wire 18 45) = a (.virt 23710) ∧ a (.wire 18 46) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies24, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies24, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies25 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 18 47) = a (.virt 23710) ∧ a (.wire 18 48) = a (.virt 23710) ∧ a (.wire 18 49) = a (.virt 23710) ∧ a (.wire 18 50) = a (.virt 23710) ∧ a (.wire 18 51) = a (.virt 23710) ∧ a (.wire 18 52) = a (.virt 23710) ∧ a (.wire 18 53) = a (.virt 23710) ∧ a (.wire 18 54) = a (.virt 23710) ∧ a (.wire 18 55) = a (.virt 23710) ∧ a (.wire 18 56) = a (.virt 23710) ∧ a (.wire 18 57) = a (.virt 23710) ∧ a (.wire 18 58) = a (.virt 23710) ∧ a (.wire 18 59) = a (.virt 23710) ∧ a (.wire 18 0) = a (.virt 23677) ∧ a (.wire 19 33) = a (.virt 23710) ∧ a (.wire 19 34) = a (.virt 23710) ∧ a (.wire 19 35) = a (.virt 23710) ∧ a (.wire 19 36) = a (.virt 23710) ∧ a (.wire 19 37) = a (.virt 23710) ∧ a (.wire 19 38) = a (.virt 23710) ∧ a (.wire 19 39) = a (.virt 23710) ∧ a (.wire 19 40) = a (.virt 23710) ∧ a (.wire 19 41) = a (.virt 23710) ∧ a (.wire 19 42) = a (.virt 23710) ∧ a (.wire 19 43) = a (.virt 23710) ∧ a (.wire 19 44) = a (.virt 23710) ∧ a (.wire 19 45) = a (.virt 23710) ∧ a (.wire 19 46) = a (.virt 23710) ∧ a (.wire 19 47) = a (.virt 23710) ∧ a (.wire 19 48) = a (.virt 23710) ∧ a (.wire 19 49) = a (.virt 23710) ∧ a (.wire 19 50) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies25, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies25, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies26 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 19 51) = a (.virt 23710) ∧ a (.wire 19 52) = a (.virt 23710) ∧ a (.wire 19 53) = a (.virt 23710) ∧ a (.wire 19 54) = a (.virt 23710) ∧ a (.wire 19 55) = a (.virt 23710) ∧ a (.wire 19 56) = a (.virt 23710) ∧ a (.wire 19 57) = a (.virt 23710) ∧ a (.wire 19 58) = a (.virt 23710) ∧ a (.wire 19 59) = a (.virt 23710) ∧ a (.wire 19 0) = a (.virt 23682) ∧ a (.virt 23677) = a (.wire 8 40) ∧ a (.virt 23709) = a (.wire 8 41) ∧ a (.virt 23682) = a (.wire 8 42) ∧ a (.wire 20 33) = a (.virt 23710) ∧ a (.wire 20 34) = a (.virt 23710) ∧ a (.wire 20 35) = a (.virt 23710) ∧ a (.wire 20 36) = a (.virt 23710) ∧ a (.wire 20 37) = a (.virt 23710) ∧ a (.wire 20 38) = a (.virt 23710) ∧ a (.wire 20 39) = a (.virt 23710) ∧ a (.wire 20 40) = a (.virt 23710) ∧ a (.wire 20 41) = a (.virt 23710) ∧ a (.wire 20 42) = a (.virt 23710) ∧ a (.wire 20 43) = a (.virt 23710) ∧ a (.wire 20 44) = a (.virt 23710) ∧ a (.wire 20 45) = a (.virt 23710) ∧ a (.wire 20 46) = a (.virt 23710) ∧ a (.wire 20 47) = a (.virt 23710) ∧ a (.wire 20 48) = a (.virt 23710) ∧ a (.wire 20 49) = a (.virt 23710) ∧ a (.wire 20 50) = a (.virt 23710) ∧ a (.wire 20 51) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies26, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies26, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies27 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 20 52) = a (.virt 23710) ∧ a (.wire 20 53) = a (.virt 23710) ∧ a (.wire 20 54) = a (.virt 23710) ∧ a (.wire 20 55) = a (.virt 23710) ∧ a (.wire 20 56) = a (.virt 23710) ∧ a (.wire 20 57) = a (.virt 23710) ∧ a (.wire 20 58) = a (.virt 23710) ∧ a (.wire 20 59) = a (.virt 23710) ∧ a (.wire 20 0) = a (.virt 23687) ∧ a (.wire 8 43) = a (.wire 8 44) ∧ a (.virt 23709) = a (.wire 8 45) ∧ a (.virt 23687) = a (.wire 8 46) ∧ a (.wire 21 33) = a (.virt 23710) ∧ a (.wire 21 34) = a (.virt 23710) ∧ a (.wire 21 35) = a (.virt 23710) ∧ a (.wire 21 36) = a (.virt 23710) ∧ a (.wire 21 37) = a (.virt 23710) ∧ a (.wire 21 38) = a (.virt 23710) ∧ a (.wire 21 39) = a (.virt 23710) ∧ a (.wire 21 40) = a (.virt 23710) ∧ a (.wire 21 41) = a (.virt 23710) ∧ a (.wire 21 42) = a (.virt 23710) ∧ a (.wire 21 43) = a (.virt 23710) ∧ a (.wire 21 44) = a (.virt 23710) ∧ a (.wire 21 45) = a (.virt 23710) ∧ a (.wire 21 46) = a (.virt 23710) ∧ a (.wire 21 47) = a (.virt 23710) ∧ a (.wire 21 48) = a (.virt 23710) ∧ a (.wire 21 49) = a (.virt 23710) ∧ a (.wire 21 50) = a (.virt 23710) ∧ a (.wire 21 51) = a (.virt 23710) ∧ a (.wire 21 52) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies27, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies27, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies28 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 21 53) = a (.virt 23710) ∧ a (.wire 21 54) = a (.virt 23710) ∧ a (.wire 21 55) = a (.virt 23710) ∧ a (.wire 21 56) = a (.virt 23710) ∧ a (.wire 21 57) = a (.virt 23710) ∧ a (.wire 21 58) = a (.virt 23710) ∧ a (.wire 21 59) = a (.virt 23710) ∧ a (.wire 21 0) = a (.virt 23692) ∧ a (.wire 8 47) = a (.wire 8 48) ∧ a (.virt 23709) = a (.wire 8 49) ∧ a (.virt 23692) = a (.wire 8 50) ∧ a (.wire 22 15) = a (.virt 23710) ∧ a (.wire 22 16) = a (.virt 23710) ∧ a (.wire 22 17) = a (.virt 23710) ∧ a (.wire 22 18) = a (.virt 23710) ∧ a (.wire 22 19) = a (.virt 23710) ∧ a (.wire 22 20) = a (.virt 23710) ∧ a (.wire 22 21) = a (.virt 23710) ∧ a (.wire 22 22) = a (.virt 23710) ∧ a (.wire 22 23) = a (.virt 23710) ∧ a (.wire 22 24) = a (.virt 23710) ∧ a (.wire 22 25) = a (.virt 23710) ∧ a (.wire 22 26) = a (.virt 23710) ∧ a (.wire 22 27) = a (.virt 23710) ∧ a (.wire 22 28) = a (.virt 23710) ∧ a (.wire 22 29) = a (.virt 23710) ∧ a (.wire 22 30) = a (.virt 23710) ∧ a (.wire 22 31) = a (.virt 23710) ∧ a (.wire 22 32) = a (.virt 23710) ∧ a (.wire 22 33) = a (.virt 23710) ∧ a (.wire 22 34) = a (.virt 23710) ∧ a (.wire 22 35) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies28, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies28, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies29 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 22 36) = a (.virt 23710) ∧ a (.wire 22 37) = a (.virt 23710) ∧ a (.wire 22 38) = a (.virt 23710) ∧ a (.wire 22 39) = a (.virt 23710) ∧ a (.wire 22 40) = a (.virt 23710) ∧ a (.wire 22 41) = a (.virt 23710) ∧ a (.wire 22 42) = a (.virt 23710) ∧ a (.wire 22 43) = a (.virt 23710) ∧ a (.wire 22 44) = a (.virt 23710) ∧ a (.wire 22 45) = a (.virt 23710) ∧ a (.wire 22 46) = a (.virt 23710) ∧ a (.wire 22 47) = a (.virt 23710) ∧ a (.wire 22 48) = a (.virt 23710) ∧ a (.wire 22 49) = a (.virt 23710) ∧ a (.wire 22 50) = a (.virt 23710) ∧ a (.wire 22 51) = a (.virt 23710) ∧ a (.wire 22 52) = a (.virt 23710) ∧ a (.wire 22 53) = a (.virt 23710) ∧ a (.wire 22 54) = a (.virt 23710) ∧ a (.wire 22 55) = a (.virt 23710) ∧ a (.wire 22 56) = a (.virt 23710) ∧ a (.wire 22 57) = a (.virt 23710) ∧ a (.wire 22 58) = a (.virt 23710) ∧ a (.wire 22 59) = a (.virt 23710) ∧ a (.wire 22 0) = a (.virt 23672) ∧ a (.virt 23777) = a (.wire 16 40) ∧ a (.virt 23709) = a (.wire 16 41) ∧ a (.virt 23672) = a (.wire 16 42) ∧ a (.wire 23 15) = a (.virt 23710) ∧ a (.wire 23 16) = a (.virt 23710) ∧ a (.wire 23 17) = a (.virt 23710) ∧ a (.wire 23 18) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies29, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies29, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies30 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 23 19) = a (.virt 23710) ∧ a (.wire 23 20) = a (.virt 23710) ∧ a (.wire 23 21) = a (.virt 23710) ∧ a (.wire 23 22) = a (.virt 23710) ∧ a (.wire 23 23) = a (.virt 23710) ∧ a (.wire 23 24) = a (.virt 23710) ∧ a (.wire 23 25) = a (.virt 23710) ∧ a (.wire 23 26) = a (.virt 23710) ∧ a (.wire 23 27) = a (.virt 23710) ∧ a (.wire 23 28) = a (.virt 23710) ∧ a (.wire 23 29) = a (.virt 23710) ∧ a (.wire 23 30) = a (.virt 23710) ∧ a (.wire 23 31) = a (.virt 23710) ∧ a (.wire 23 32) = a (.virt 23710) ∧ a (.wire 23 33) = a (.virt 23710) ∧ a (.wire 23 34) = a (.virt 23710) ∧ a (.wire 23 35) = a (.virt 23710) ∧ a (.wire 23 36) = a (.virt 23710) ∧ a (.wire 23 37) = a (.virt 23710) ∧ a (.wire 23 38) = a (.virt 23710) ∧ a (.wire 23 39) = a (.virt 23710) ∧ a (.wire 23 40) = a (.virt 23710) ∧ a (.wire 23 41) = a (.virt 23710) ∧ a (.wire 23 42) = a (.virt 23710) ∧ a (.wire 23 43) = a (.virt 23710) ∧ a (.wire 23 44) = a (.virt 23710) ∧ a (.wire 23 45) = a (.virt 23710) ∧ a (.wire 23 46) = a (.virt 23710) ∧ a (.wire 23 47) = a (.virt 23710) ∧ a (.wire 23 48) = a (.virt 23710) ∧ a (.wire 23 49) = a (.virt 23710) ∧ a (.wire 23 50) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies30, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies30, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies31 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 23 51) = a (.virt 23710) ∧ a (.wire 23 52) = a (.virt 23710) ∧ a (.wire 23 53) = a (.virt 23710) ∧ a (.wire 23 54) = a (.virt 23710) ∧ a (.wire 23 55) = a (.virt 23710) ∧ a (.wire 23 56) = a (.virt 23710) ∧ a (.wire 23 57) = a (.virt 23710) ∧ a (.wire 23 58) = a (.virt 23710) ∧ a (.wire 23 59) = a (.virt 23710) ∧ a (.wire 23 0) = a (.wire 16 43) ∧ a (.wire 8 51) = a (.wire 17 4) ∧ a (.virt 23777) = a (.wire 17 5) ∧ a (.wire 8 51) = a (.wire 17 6) ∧ a (.wire 8 39) = a (.wire 17 8) ∧ a (.wire 16 43) = a (.wire 17 9) ∧ a (.wire 8 39) = a (.wire 17 10) ∧ a (.wire 17 11) = a (.wire 16 44) ∧ a (.virt 23709) = a (.wire 16 45) ∧ a (.wire 17 7) = a (.wire 16 46) ∧ a (.wire 24 53) = a (.virt 23710) ∧ a (.wire 24 54) = a (.virt 23710) ∧ a (.wire 24 55) = a (.virt 23710) ∧ a (.wire 24 56) = a (.virt 23710) ∧ a (.wire 24 57) = a (.virt 23710) ∧ a (.wire 24 58) = a (.virt 23710) ∧ a (.wire 24 59) = a (.virt 23710) ∧ a (.wire 24 0) = a (.wire 16 47) ∧ a (.wire 5 11) = a (.wire 17 12) ∧ a (.wire 5 35) = a (.wire 17 13) ∧ a (.wire 5 11) = a (.wire 17 14) ∧ a (.virt 23709) = a (.wire 16 48) ∧ a (.virt 23709) = a (.wire 16 49) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies31, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies31, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies32 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23778) = a (.wire 16 50) ∧ a (.virt 5908) = a (.wire 16 52) ∧ a (.virt 23709) = a (.wire 16 53) ∧ a (.virt 11826) = a (.wire 16 54) ∧ a (.virt 23778) = a (.wire 17 16) ∧ a (.wire 16 55) = a (.wire 17 17) ∧ a (.virt 23778) = a (.wire 17 18) ∧ a (.wire 16 55) = a (.wire 17 20) ∧ a (.virt 23779) = a (.wire 17 21) ∧ a (.wire 16 55) = a (.wire 17 22) ∧ a (.wire 17 23) = a (.wire 16 56) ∧ a (.virt 23709) = a (.wire 16 57) ∧ a (.wire 16 51) = a (.wire 16 58) ∧ a (.wire 17 19) = a (.virt 23710) ∧ a (.wire 16 59) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 25 0) ∧ a (.virt 23709) = a (.wire 25 1) ∧ a (.virt 23780) = a (.wire 25 2) ∧ a (.virt 5909) = a (.wire 25 4) ∧ a (.virt 23709) = a (.wire 25 5) ∧ a (.virt 11827) = a (.wire 25 6) ∧ a (.virt 23780) = a (.wire 17 24) ∧ a (.wire 25 7) = a (.wire 17 25) ∧ a (.virt 23780) = a (.wire 17 26) ∧ a (.wire 25 7) = a (.wire 17 28) ∧ a (.virt 23781) = a (.wire 17 29) ∧ a (.wire 25 7) = a (.wire 17 30) ∧ a (.wire 17 31) = a (.wire 25 8) ∧ a (.virt 23709) = a (.wire 25 9) ∧ a (.wire 25 3) = a (.wire 25 10) ∧ a (.wire 17 27) = a (.virt 23710) ∧ a (.wire 25 11) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies32, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies32, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies33 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 25 12) ∧ a (.virt 23709) = a (.wire 25 13) ∧ a (.virt 23782) = a (.wire 25 14) ∧ a (.virt 5910) = a (.wire 25 16) ∧ a (.virt 23709) = a (.wire 25 17) ∧ a (.virt 11828) = a (.wire 25 18) ∧ a (.virt 23782) = a (.wire 17 32) ∧ a (.wire 25 19) = a (.wire 17 33) ∧ a (.virt 23782) = a (.wire 17 34) ∧ a (.wire 25 19) = a (.wire 17 36) ∧ a (.virt 23783) = a (.wire 17 37) ∧ a (.wire 25 19) = a (.wire 17 38) ∧ a (.wire 17 39) = a (.wire 25 20) ∧ a (.virt 23709) = a (.wire 25 21) ∧ a (.wire 25 15) = a (.wire 25 22) ∧ a (.wire 17 35) = a (.virt 23710) ∧ a (.wire 25 23) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 25 24) ∧ a (.virt 23709) = a (.wire 25 25) ∧ a (.virt 23784) = a (.wire 25 26) ∧ a (.virt 5911) = a (.wire 25 28) ∧ a (.virt 23709) = a (.wire 25 29) ∧ a (.virt 11829) = a (.wire 25 30) ∧ a (.virt 23784) = a (.wire 17 40) ∧ a (.wire 25 31) = a (.wire 17 41) ∧ a (.virt 23784) = a (.wire 17 42) ∧ a (.wire 25 31) = a (.wire 17 44) ∧ a (.virt 23785) = a (.wire 17 45) ∧ a (.wire 25 31) = a (.wire 17 46) ∧ a (.wire 17 47) = a (.wire 25 32) ∧ a (.virt 23709) = a (.wire 25 33) ∧ a (.wire 25 27) = a (.wire 25 34) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies33, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies33, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies34 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 43) = a (.virt 23710) ∧ a (.wire 25 35) = a (.virt 23710) ∧ a (.virt 23778) = a (.wire 17 48) ∧ a (.virt 23780) = a (.wire 17 49) ∧ a (.virt 23778) = a (.wire 17 50) ∧ a (.virt 23782) = a (.wire 17 52) ∧ a (.virt 23784) = a (.wire 17 53) ∧ a (.virt 23782) = a (.wire 17 54) ∧ a (.wire 17 51) = a (.wire 17 56) ∧ a (.wire 17 55) = a (.wire 17 57) ∧ a (.wire 17 51) = a (.wire 17 58) ∧ a (.wire 17 15) = a (.wire 26 0) ∧ a (.wire 17 59) = a (.wire 26 1) ∧ a (.wire 17 15) = a (.wire 26 2) ∧ a (.wire 26 3) = a (.virt 23710) ∧ a (.wire 5 11) = a (.wire 26 4) ∧ a (.wire 6 23) = a (.wire 26 5) ∧ a (.wire 5 11) = a (.wire 26 6) ∧ a (.virt 23709) = a (.wire 25 36) ∧ a (.virt 23709) = a (.wire 25 37) ∧ a (.virt 23786) = a (.wire 25 38) ∧ a (.virt 5908) = a (.wire 25 40) ∧ a (.virt 23709) = a (.wire 25 41) ∧ a (.virt 17744) = a (.wire 25 42) ∧ a (.virt 23786) = a (.wire 26 8) ∧ a (.wire 25 43) = a (.wire 26 9) ∧ a (.virt 23786) = a (.wire 26 10) ∧ a (.wire 25 43) = a (.wire 26 12) ∧ a (.virt 23787) = a (.wire 26 13) ∧ a (.wire 25 43) = a (.wire 26 14) ∧ a (.wire 26 15) = a (.wire 25 44) ∧ a (.virt 23709) = a (.wire 25 45) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies34, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies34, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies35 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 25 39) = a (.wire 25 46) ∧ a (.wire 26 11) = a (.virt 23710) ∧ a (.wire 25 47) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 25 48) ∧ a (.virt 23709) = a (.wire 25 49) ∧ a (.virt 23788) = a (.wire 25 50) ∧ a (.virt 5909) = a (.wire 25 52) ∧ a (.virt 23709) = a (.wire 25 53) ∧ a (.virt 17745) = a (.wire 25 54) ∧ a (.virt 23788) = a (.wire 26 16) ∧ a (.wire 25 55) = a (.wire 26 17) ∧ a (.virt 23788) = a (.wire 26 18) ∧ a (.wire 25 55) = a (.wire 26 20) ∧ a (.virt 23789) = a (.wire 26 21) ∧ a (.wire 25 55) = a (.wire 26 22) ∧ a (.wire 26 23) = a (.wire 25 56) ∧ a (.virt 23709) = a (.wire 25 57) ∧ a (.wire 25 51) = a (.wire 25 58) ∧ a (.wire 26 19) = a (.virt 23710) ∧ a (.wire 25 59) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 27 0) ∧ a (.virt 23709) = a (.wire 27 1) ∧ a (.virt 23790) = a (.wire 27 2) ∧ a (.virt 5910) = a (.wire 27 4) ∧ a (.virt 23709) = a (.wire 27 5) ∧ a (.virt 17746) = a (.wire 27 6) ∧ a (.virt 23790) = a (.wire 26 24) ∧ a (.wire 27 7) = a (.wire 26 25) ∧ a (.virt 23790) = a (.wire 26 26) ∧ a (.wire 27 7) = a (.wire 26 28) ∧ a (.virt 23791) = a (.wire 26 29) ∧ a (.wire 27 7) = a (.wire 26 30) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies35, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies35, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies36 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 31) = a (.wire 27 8) ∧ a (.virt 23709) = a (.wire 27 9) ∧ a (.wire 27 3) = a (.wire 27 10) ∧ a (.wire 26 27) = a (.virt 23710) ∧ a (.wire 27 11) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 27 12) ∧ a (.virt 23709) = a (.wire 27 13) ∧ a (.virt 23792) = a (.wire 27 14) ∧ a (.virt 5911) = a (.wire 27 16) ∧ a (.virt 23709) = a (.wire 27 17) ∧ a (.virt 17747) = a (.wire 27 18) ∧ a (.virt 23792) = a (.wire 26 32) ∧ a (.wire 27 19) = a (.wire 26 33) ∧ a (.virt 23792) = a (.wire 26 34) ∧ a (.wire 27 19) = a (.wire 26 36) ∧ a (.virt 23793) = a (.wire 26 37) ∧ a (.wire 27 19) = a (.wire 26 38) ∧ a (.wire 26 39) = a (.wire 27 20) ∧ a (.virt 23709) = a (.wire 27 21) ∧ a (.wire 27 15) = a (.wire 27 22) ∧ a (.wire 26 35) = a (.virt 23710) ∧ a (.wire 27 23) = a (.virt 23710) ∧ a (.virt 23786) = a (.wire 26 40) ∧ a (.virt 23788) = a (.wire 26 41) ∧ a (.virt 23786) = a (.wire 26 42) ∧ a (.virt 23790) = a (.wire 26 44) ∧ a (.virt 23792) = a (.wire 26 45) ∧ a (.virt 23790) = a (.wire 26 46) ∧ a (.wire 26 43) = a (.wire 26 48) ∧ a (.wire 26 47) = a (.wire 26 49) ∧ a (.wire 26 43) = a (.wire 26 50) ∧ a (.wire 26 7) = a (.wire 26 52) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies36, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies36, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies37 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 51) = a (.wire 26 53) ∧ a (.wire 26 7) = a (.wire 26 54) ∧ a (.wire 26 55) = a (.virt 23710) ∧ a (.wire 5 11) = a (.wire 26 56) ∧ a (.wire 10 11) = a (.wire 26 57) ∧ a (.wire 5 11) = a (.wire 26 58) ∧ a (.virt 23709) = a (.wire 27 24) ∧ a (.virt 23709) = a (.wire 27 25) ∧ a (.virt 23794) = a (.wire 27 26) ∧ a (.virt 5908) = a (.wire 27 28) ∧ a (.virt 23709) = a (.wire 27 29) ∧ a (.virt 23662) = a (.wire 27 30) ∧ a (.virt 23794) = a (.wire 28 0) ∧ a (.wire 27 31) = a (.wire 28 1) ∧ a (.virt 23794) = a (.wire 28 2) ∧ a (.wire 27 31) = a (.wire 28 4) ∧ a (.virt 23795) = a (.wire 28 5) ∧ a (.wire 27 31) = a (.wire 28 6) ∧ a (.wire 28 7) = a (.wire 27 32) ∧ a (.virt 23709) = a (.wire 27 33) ∧ a (.wire 27 27) = a (.wire 27 34) ∧ a (.wire 28 3) = a (.virt 23710) ∧ a (.wire 27 35) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 27 36) ∧ a (.virt 23709) = a (.wire 27 37) ∧ a (.virt 23796) = a (.wire 27 38) ∧ a (.virt 5909) = a (.wire 27 40) ∧ a (.virt 23709) = a (.wire 27 41) ∧ a (.virt 23663) = a (.wire 27 42) ∧ a (.virt 23796) = a (.wire 28 8) ∧ a (.wire 27 43) = a (.wire 28 9) ∧ a (.virt 23796) = a (.wire 28 10) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies37, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies37, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies38 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 27 43) = a (.wire 28 12) ∧ a (.virt 23797) = a (.wire 28 13) ∧ a (.wire 27 43) = a (.wire 28 14) ∧ a (.wire 28 15) = a (.wire 27 44) ∧ a (.virt 23709) = a (.wire 27 45) ∧ a (.wire 27 39) = a (.wire 27 46) ∧ a (.wire 28 11) = a (.virt 23710) ∧ a (.wire 27 47) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 27 48) ∧ a (.virt 23709) = a (.wire 27 49) ∧ a (.virt 23798) = a (.wire 27 50) ∧ a (.virt 5910) = a (.wire 27 52) ∧ a (.virt 23709) = a (.wire 27 53) ∧ a (.virt 23664) = a (.wire 27 54) ∧ a (.virt 23798) = a (.wire 28 16) ∧ a (.wire 27 55) = a (.wire 28 17) ∧ a (.virt 23798) = a (.wire 28 18) ∧ a (.wire 27 55) = a (.wire 28 20) ∧ a (.virt 23799) = a (.wire 28 21) ∧ a (.wire 27 55) = a (.wire 28 22) ∧ a (.wire 28 23) = a (.wire 27 56) ∧ a (.virt 23709) = a (.wire 27 57) ∧ a (.wire 27 51) = a (.wire 27 58) ∧ a (.wire 28 19) = a (.virt 23710) ∧ a (.wire 27 59) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 29 0) ∧ a (.virt 23709) = a (.wire 29 1) ∧ a (.virt 23800) = a (.wire 29 2) ∧ a (.virt 5911) = a (.wire 29 4) ∧ a (.virt 23709) = a (.wire 29 5) ∧ a (.virt 23665) = a (.wire 29 6) ∧ a (.virt 23800) = a (.wire 28 24) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies38, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies38, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies39 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 29 7) = a (.wire 28 25) ∧ a (.virt 23800) = a (.wire 28 26) ∧ a (.wire 29 7) = a (.wire 28 28) ∧ a (.virt 23801) = a (.wire 28 29) ∧ a (.wire 29 7) = a (.wire 28 30) ∧ a (.wire 28 31) = a (.wire 29 8) ∧ a (.virt 23709) = a (.wire 29 9) ∧ a (.wire 29 3) = a (.wire 29 10) ∧ a (.wire 28 27) = a (.virt 23710) ∧ a (.wire 29 11) = a (.virt 23710) ∧ a (.virt 23794) = a (.wire 28 32) ∧ a (.virt 23796) = a (.wire 28 33) ∧ a (.virt 23794) = a (.wire 28 34) ∧ a (.virt 23798) = a (.wire 28 36) ∧ a (.virt 23800) = a (.wire 28 37) ∧ a (.virt 23798) = a (.wire 28 38) ∧ a (.wire 28 35) = a (.wire 28 40) ∧ a (.wire 28 39) = a (.wire 28 41) ∧ a (.wire 28 35) = a (.wire 28 42) ∧ a (.wire 26 59) = a (.wire 28 44) ∧ a (.wire 28 43) = a (.wire 28 45) ∧ a (.wire 26 59) = a (.wire 28 46) ∧ a (.wire 28 47) = a (.virt 23710) ∧ a (.wire 5 35) = a (.wire 28 48) ∧ a (.wire 6 23) = a (.wire 28 49) ∧ a (.wire 5 35) = a (.wire 28 50) ∧ a (.virt 23709) = a (.wire 29 12) ∧ a (.virt 23709) = a (.wire 29 13) ∧ a (.virt 23802) = a (.wire 29 14) ∧ a (.virt 11826) = a (.wire 29 16) ∧ a (.virt 23709) = a (.wire 29 17) ∧ a (.virt 17744) = a (.wire 29 18) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies39, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies39, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies40 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23802) = a (.wire 28 52) ∧ a (.wire 29 19) = a (.wire 28 53) ∧ a (.virt 23802) = a (.wire 28 54) ∧ a (.wire 29 19) = a (.wire 28 56) ∧ a (.virt 23803) = a (.wire 28 57) ∧ a (.wire 29 19) = a (.wire 28 58) ∧ a (.wire 28 59) = a (.wire 29 20) ∧ a (.virt 23709) = a (.wire 29 21) ∧ a (.wire 29 15) = a (.wire 29 22) ∧ a (.wire 28 55) = a (.virt 23710) ∧ a (.wire 29 23) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 29 24) ∧ a (.virt 23709) = a (.wire 29 25) ∧ a (.virt 23804) = a (.wire 29 26) ∧ a (.virt 11827) = a (.wire 29 28) ∧ a (.virt 23709) = a (.wire 29 29) ∧ a (.virt 17745) = a (.wire 29 30) ∧ a (.virt 23804) = a (.wire 30 0) ∧ a (.wire 29 31) = a (.wire 30 1) ∧ a (.virt 23804) = a (.wire 30 2) ∧ a (.wire 29 31) = a (.wire 30 4) ∧ a (.virt 23805) = a (.wire 30 5) ∧ a (.wire 29 31) = a (.wire 30 6) ∧ a (.wire 30 7) = a (.wire 29 32) ∧ a (.virt 23709) = a (.wire 29 33) ∧ a (.wire 29 27) = a (.wire 29 34) ∧ a (.wire 30 3) = a (.virt 23710) ∧ a (.wire 29 35) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 29 36) ∧ a (.virt 23709) = a (.wire 29 37) ∧ a (.virt 23806) = a (.wire 29 38) ∧ a (.virt 11828) = a (.wire 29 40) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies40, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies40, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies41 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 29 41) ∧ a (.virt 17746) = a (.wire 29 42) ∧ a (.virt 23806) = a (.wire 30 8) ∧ a (.wire 29 43) = a (.wire 30 9) ∧ a (.virt 23806) = a (.wire 30 10) ∧ a (.wire 29 43) = a (.wire 30 12) ∧ a (.virt 23807) = a (.wire 30 13) ∧ a (.wire 29 43) = a (.wire 30 14) ∧ a (.wire 30 15) = a (.wire 29 44) ∧ a (.virt 23709) = a (.wire 29 45) ∧ a (.wire 29 39) = a (.wire 29 46) ∧ a (.wire 30 11) = a (.virt 23710) ∧ a (.wire 29 47) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 29 48) ∧ a (.virt 23709) = a (.wire 29 49) ∧ a (.virt 23808) = a (.wire 29 50) ∧ a (.virt 11829) = a (.wire 29 52) ∧ a (.virt 23709) = a (.wire 29 53) ∧ a (.virt 17747) = a (.wire 29 54) ∧ a (.virt 23808) = a (.wire 30 16) ∧ a (.wire 29 55) = a (.wire 30 17) ∧ a (.virt 23808) = a (.wire 30 18) ∧ a (.wire 29 55) = a (.wire 30 20) ∧ a (.virt 23809) = a (.wire 30 21) ∧ a (.wire 29 55) = a (.wire 30 22) ∧ a (.wire 30 23) = a (.wire 29 56) ∧ a (.virt 23709) = a (.wire 29 57) ∧ a (.wire 29 51) = a (.wire 29 58) ∧ a (.wire 30 19) = a (.virt 23710) ∧ a (.wire 29 59) = a (.virt 23710) ∧ a (.virt 23802) = a (.wire 30 24) ∧ a (.virt 23804) = a (.wire 30 25) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies41, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies41, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies42 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23802) = a (.wire 30 26) ∧ a (.virt 23806) = a (.wire 30 28) ∧ a (.virt 23808) = a (.wire 30 29) ∧ a (.virt 23806) = a (.wire 30 30) ∧ a (.wire 30 27) = a (.wire 30 32) ∧ a (.wire 30 31) = a (.wire 30 33) ∧ a (.wire 30 27) = a (.wire 30 34) ∧ a (.wire 28 51) = a (.wire 30 36) ∧ a (.wire 30 35) = a (.wire 30 37) ∧ a (.wire 28 51) = a (.wire 30 38) ∧ a (.wire 30 39) = a (.virt 23710) ∧ a (.wire 5 35) = a (.wire 30 40) ∧ a (.wire 10 11) = a (.wire 30 41) ∧ a (.wire 5 35) = a (.wire 30 42) ∧ a (.virt 23709) = a (.wire 31 0) ∧ a (.virt 23709) = a (.wire 31 1) ∧ a (.virt 23810) = a (.wire 31 2) ∧ a (.virt 11826) = a (.wire 31 4) ∧ a (.virt 23709) = a (.wire 31 5) ∧ a (.virt 23662) = a (.wire 31 6) ∧ a (.virt 23810) = a (.wire 30 44) ∧ a (.wire 31 7) = a (.wire 30 45) ∧ a (.virt 23810) = a (.wire 30 46) ∧ a (.wire 31 7) = a (.wire 30 48) ∧ a (.virt 23811) = a (.wire 30 49) ∧ a (.wire 31 7) = a (.wire 30 50) ∧ a (.wire 30 51) = a (.wire 31 8) ∧ a (.virt 23709) = a (.wire 31 9) ∧ a (.wire 31 3) = a (.wire 31 10) ∧ a (.wire 30 47) = a (.virt 23710) ∧ a (.wire 31 11) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 31 12) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies42, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies42, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies43 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 31 13) ∧ a (.virt 23812) = a (.wire 31 14) ∧ a (.virt 11827) = a (.wire 31 16) ∧ a (.virt 23709) = a (.wire 31 17) ∧ a (.virt 23663) = a (.wire 31 18) ∧ a (.virt 23812) = a (.wire 30 52) ∧ a (.wire 31 19) = a (.wire 30 53) ∧ a (.virt 23812) = a (.wire 30 54) ∧ a (.wire 31 19) = a (.wire 30 56) ∧ a (.virt 23813) = a (.wire 30 57) ∧ a (.wire 31 19) = a (.wire 30 58) ∧ a (.wire 30 59) = a (.wire 31 20) ∧ a (.virt 23709) = a (.wire 31 21) ∧ a (.wire 31 15) = a (.wire 31 22) ∧ a (.wire 30 55) = a (.virt 23710) ∧ a (.wire 31 23) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 31 24) ∧ a (.virt 23709) = a (.wire 31 25) ∧ a (.virt 23814) = a (.wire 31 26) ∧ a (.virt 11828) = a (.wire 31 28) ∧ a (.virt 23709) = a (.wire 31 29) ∧ a (.virt 23664) = a (.wire 31 30) ∧ a (.virt 23814) = a (.wire 32 0) ∧ a (.wire 31 31) = a (.wire 32 1) ∧ a (.virt 23814) = a (.wire 32 2) ∧ a (.wire 31 31) = a (.wire 32 4) ∧ a (.virt 23815) = a (.wire 32 5) ∧ a (.wire 31 31) = a (.wire 32 6) ∧ a (.wire 32 7) = a (.wire 31 32) ∧ a (.virt 23709) = a (.wire 31 33) ∧ a (.wire 31 27) = a (.wire 31 34) ∧ a (.wire 32 3) = a (.virt 23710) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies43, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies43, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies44 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 31 35) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 31 36) ∧ a (.virt 23709) = a (.wire 31 37) ∧ a (.virt 23816) = a (.wire 31 38) ∧ a (.virt 11829) = a (.wire 31 40) ∧ a (.virt 23709) = a (.wire 31 41) ∧ a (.virt 23665) = a (.wire 31 42) ∧ a (.virt 23816) = a (.wire 32 8) ∧ a (.wire 31 43) = a (.wire 32 9) ∧ a (.virt 23816) = a (.wire 32 10) ∧ a (.wire 31 43) = a (.wire 32 12) ∧ a (.virt 23817) = a (.wire 32 13) ∧ a (.wire 31 43) = a (.wire 32 14) ∧ a (.wire 32 15) = a (.wire 31 44) ∧ a (.virt 23709) = a (.wire 31 45) ∧ a (.wire 31 39) = a (.wire 31 46) ∧ a (.wire 32 11) = a (.virt 23710) ∧ a (.wire 31 47) = a (.virt 23710) ∧ a (.virt 23810) = a (.wire 32 16) ∧ a (.virt 23812) = a (.wire 32 17) ∧ a (.virt 23810) = a (.wire 32 18) ∧ a (.virt 23814) = a (.wire 32 20) ∧ a (.virt 23816) = a (.wire 32 21) ∧ a (.virt 23814) = a (.wire 32 22) ∧ a (.wire 32 19) = a (.wire 32 24) ∧ a (.wire 32 23) = a (.wire 32 25) ∧ a (.wire 32 19) = a (.wire 32 26) ∧ a (.wire 30 43) = a (.wire 32 28) ∧ a (.wire 32 27) = a (.wire 32 29) ∧ a (.wire 30 43) = a (.wire 32 30) ∧ a (.wire 32 31) = a (.virt 23710) ∧ a (.wire 6 23) = a (.wire 32 32) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies44, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies44, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies45 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = a (.wire 32 33) ∧ a (.wire 6 23) = a (.wire 32 34) ∧ a (.virt 23709) = a (.wire 31 48) ∧ a (.virt 23709) = a (.wire 31 49) ∧ a (.virt 23818) = a (.wire 31 50) ∧ a (.virt 17744) = a (.wire 31 52) ∧ a (.virt 23709) = a (.wire 31 53) ∧ a (.virt 23662) = a (.wire 31 54) ∧ a (.virt 23818) = a (.wire 32 36) ∧ a (.wire 31 55) = a (.wire 32 37) ∧ a (.virt 23818) = a (.wire 32 38) ∧ a (.wire 31 55) = a (.wire 32 40) ∧ a (.virt 23819) = a (.wire 32 41) ∧ a (.wire 31 55) = a (.wire 32 42) ∧ a (.wire 32 43) = a (.wire 31 56) ∧ a (.virt 23709) = a (.wire 31 57) ∧ a (.wire 31 51) = a (.wire 31 58) ∧ a (.wire 32 39) = a (.virt 23710) ∧ a (.wire 31 59) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 33 0) ∧ a (.virt 23709) = a (.wire 33 1) ∧ a (.virt 23820) = a (.wire 33 2) ∧ a (.virt 17745) = a (.wire 33 4) ∧ a (.virt 23709) = a (.wire 33 5) ∧ a (.virt 23663) = a (.wire 33 6) ∧ a (.virt 23820) = a (.wire 32 44) ∧ a (.wire 33 7) = a (.wire 32 45) ∧ a (.virt 23820) = a (.wire 32 46) ∧ a (.wire 33 7) = a (.wire 32 48) ∧ a (.virt 23821) = a (.wire 32 49) ∧ a (.wire 33 7) = a (.wire 32 50) ∧ a (.wire 32 51) = a (.wire 33 8) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies45, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies45, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies46 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = a (.wire 33 9) ∧ a (.wire 33 3) = a (.wire 33 10) ∧ a (.wire 32 47) = a (.virt 23710) ∧ a (.wire 33 11) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 33 12) ∧ a (.virt 23709) = a (.wire 33 13) ∧ a (.virt 23822) = a (.wire 33 14) ∧ a (.virt 17746) = a (.wire 33 16) ∧ a (.virt 23709) = a (.wire 33 17) ∧ a (.virt 23664) = a (.wire 33 18) ∧ a (.virt 23822) = a (.wire 32 52) ∧ a (.wire 33 19) = a (.wire 32 53) ∧ a (.virt 23822) = a (.wire 32 54) ∧ a (.wire 33 19) = a (.wire 32 56) ∧ a (.virt 23823) = a (.wire 32 57) ∧ a (.wire 33 19) = a (.wire 32 58) ∧ a (.wire 32 59) = a (.wire 33 20) ∧ a (.virt 23709) = a (.wire 33 21) ∧ a (.wire 33 15) = a (.wire 33 22) ∧ a (.wire 32 55) = a (.virt 23710) ∧ a (.wire 33 23) = a (.virt 23710) ∧ a (.virt 23709) = a (.wire 33 24) ∧ a (.virt 23709) = a (.wire 33 25) ∧ a (.virt 23824) = a (.wire 33 26) ∧ a (.virt 17747) = a (.wire 33 28) ∧ a (.virt 23709) = a (.wire 33 29) ∧ a (.virt 23665) = a (.wire 33 30) ∧ a (.virt 23824) = a (.wire 34 0) ∧ a (.wire 33 31) = a (.wire 34 1) ∧ a (.virt 23824) = a (.wire 34 2) ∧ a (.wire 33 31) = a (.wire 34 4) ∧ a (.virt 23825) = a (.wire 34 5) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies46, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies46, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies47 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 33 31) = a (.wire 34 6) ∧ a (.wire 34 7) = a (.wire 33 32) ∧ a (.virt 23709) = a (.wire 33 33) ∧ a (.wire 33 27) = a (.wire 33 34) ∧ a (.wire 34 3) = a (.virt 23710) ∧ a (.wire 33 35) = a (.virt 23710) ∧ a (.virt 23818) = a (.wire 34 8) ∧ a (.virt 23820) = a (.wire 34 9) ∧ a (.virt 23818) = a (.wire 34 10) ∧ a (.virt 23822) = a (.wire 34 12) ∧ a (.virt 23824) = a (.wire 34 13) ∧ a (.virt 23822) = a (.wire 34 14) ∧ a (.wire 34 11) = a (.wire 34 16) ∧ a (.wire 34 15) = a (.wire 34 17) ∧ a (.wire 34 11) = a (.wire 34 18) ∧ a (.wire 32 35) = a (.wire 34 20) ∧ a (.wire 34 19) = a (.wire 34 21) ∧ a (.wire 32 35) = a (.wire 34 22) ∧ a (.wire 34 23) = a (.virt 23710) ∧ a (.virt 23693) = a (.wire 35 0) ∧ a (.virt 23694) = a (.wire 35 1) ∧ a (.virt 23695) = a (.wire 35 2) ∧ a (.virt 23696) = a (.wire 35 3) ∧ a (.virt 23709) = a (.wire 35 4) ∧ a (.virt 23710) = a (.wire 35 5) ∧ a (.virt 23710) = a (.wire 35 6) ∧ a (.virt 23710) = a (.wire 35 7) ∧ a (.virt 23710) = a (.wire 35 8) ∧ a (.virt 23710) = a (.wire 35 9) ∧ a (.virt 23710) = a (.wire 35 10) ∧ a (.virt 23710) = a (.wire 35 11) ∧ a (.wire 35 12) = a (.wire 36 0) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies47, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq)))))
  simp only [privateBatchWrapper4.copies47, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies48 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 35 13) = a (.wire 36 1) ∧ a (.wire 35 14) = a (.wire 36 2) ∧ a (.wire 35 15) = a (.wire 36 3) ∧ a (.virt 23709) = a (.wire 36 4) ∧ a (.virt 23710) = a (.wire 36 5) ∧ a (.virt 23710) = a (.wire 36 6) ∧ a (.virt 23710) = a (.wire 36 7) ∧ a (.virt 23710) = a (.wire 36 8) ∧ a (.virt 23710) = a (.wire 36 9) ∧ a (.virt 23710) = a (.wire 36 10) ∧ a (.virt 23710) = a (.wire 36 11) ∧ a (.wire 1 43) = a (.wire 33 36) ∧ a (.virt 5908) = a (.wire 33 37) ∧ a (.virt 5908) = a (.wire 33 38) ∧ a (.wire 1 43) = a (.wire 33 40) ∧ a (.wire 36 12) = a (.wire 33 41) ∧ a (.wire 33 39) = a (.wire 33 42) ∧ a (.wire 1 43) = a (.wire 33 44) ∧ a (.virt 5909) = a (.wire 33 45) ∧ a (.virt 5909) = a (.wire 33 46) ∧ a (.wire 1 43) = a (.wire 33 48) ∧ a (.wire 36 13) = a (.wire 33 49) ∧ a (.wire 33 47) = a (.wire 33 50) ∧ a (.wire 1 43) = a (.wire 33 52) ∧ a (.virt 5910) = a (.wire 33 53) ∧ a (.virt 5910) = a (.wire 33 54) ∧ a (.wire 1 43) = a (.wire 33 56) ∧ a (.wire 36 14) = a (.wire 33 57) ∧ a (.wire 33 55) = a (.wire 33 58) ∧ a (.wire 1 43) = a (.wire 37 0) ∧ a (.virt 5911) = a (.wire 37 1) ∧ a (.virt 5911) = a (.wire 37 2) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies48, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies48, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies49 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 1 43) = a (.wire 37 4) ∧ a (.wire 36 15) = a (.wire 37 5) ∧ a (.wire 37 3) = a (.wire 37 6) ∧ a (.virt 23697) = a (.wire 38 0) ∧ a (.virt 23698) = a (.wire 38 1) ∧ a (.virt 23699) = a (.wire 38 2) ∧ a (.virt 23700) = a (.wire 38 3) ∧ a (.virt 23709) = a (.wire 38 4) ∧ a (.virt 23710) = a (.wire 38 5) ∧ a (.virt 23710) = a (.wire 38 6) ∧ a (.virt 23710) = a (.wire 38 7) ∧ a (.virt 23710) = a (.wire 38 8) ∧ a (.virt 23710) = a (.wire 38 9) ∧ a (.virt 23710) = a (.wire 38 10) ∧ a (.virt 23710) = a (.wire 38 11) ∧ a (.wire 38 12) = a (.wire 39 0) ∧ a (.wire 38 13) = a (.wire 39 1) ∧ a (.wire 38 14) = a (.wire 39 2) ∧ a (.wire 38 15) = a (.wire 39 3) ∧ a (.virt 23709) = a (.wire 39 4) ∧ a (.virt 23710) = a (.wire 39 5) ∧ a (.virt 23710) = a (.wire 39 6) ∧ a (.virt 23710) = a (.wire 39 7) ∧ a (.virt 23710) = a (.wire 39 8) ∧ a (.virt 23710) = a (.wire 39 9) ∧ a (.virt 23710) = a (.wire 39 10) ∧ a (.virt 23710) = a (.wire 39 11) ∧ a (.wire 2 27) = a (.wire 37 8) ∧ a (.virt 11826) = a (.wire 37 9) ∧ a (.virt 11826) = a (.wire 37 10) ∧ a (.wire 2 27) = a (.wire 37 12) ∧ a (.wire 39 12) = a (.wire 37 13) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies49, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies49, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies50 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 11) = a (.wire 37 14) ∧ a (.wire 2 27) = a (.wire 37 16) ∧ a (.virt 11827) = a (.wire 37 17) ∧ a (.virt 11827) = a (.wire 37 18) ∧ a (.wire 2 27) = a (.wire 37 20) ∧ a (.wire 39 13) = a (.wire 37 21) ∧ a (.wire 37 19) = a (.wire 37 22) ∧ a (.wire 2 27) = a (.wire 37 24) ∧ a (.virt 11828) = a (.wire 37 25) ∧ a (.virt 11828) = a (.wire 37 26) ∧ a (.wire 2 27) = a (.wire 37 28) ∧ a (.wire 39 14) = a (.wire 37 29) ∧ a (.wire 37 27) = a (.wire 37 30) ∧ a (.wire 2 27) = a (.wire 37 32) ∧ a (.virt 11829) = a (.wire 37 33) ∧ a (.virt 11829) = a (.wire 37 34) ∧ a (.wire 2 27) = a (.wire 37 36) ∧ a (.wire 39 15) = a (.wire 37 37) ∧ a (.wire 37 35) = a (.wire 37 38) ∧ a (.virt 23701) = a (.wire 40 0) ∧ a (.virt 23702) = a (.wire 40 1) ∧ a (.virt 23703) = a (.wire 40 2) ∧ a (.virt 23704) = a (.wire 40 3) ∧ a (.virt 23709) = a (.wire 40 4) ∧ a (.virt 23710) = a (.wire 40 5) ∧ a (.virt 23710) = a (.wire 40 6) ∧ a (.virt 23710) = a (.wire 40 7) ∧ a (.virt 23710) = a (.wire 40 8) ∧ a (.virt 23710) = a (.wire 40 9) ∧ a (.virt 23710) = a (.wire 40 10) ∧ a (.virt 23710) = a (.wire 40 11) ∧ a (.wire 40 12) = a (.wire 41 0) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies50, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies50, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies51 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 40 13) = a (.wire 41 1) ∧ a (.wire 40 14) = a (.wire 41 2) ∧ a (.wire 40 15) = a (.wire 41 3) ∧ a (.virt 23709) = a (.wire 41 4) ∧ a (.virt 23710) = a (.wire 41 5) ∧ a (.virt 23710) = a (.wire 41 6) ∧ a (.virt 23710) = a (.wire 41 7) ∧ a (.virt 23710) = a (.wire 41 8) ∧ a (.virt 23710) = a (.wire 41 9) ∧ a (.virt 23710) = a (.wire 41 10) ∧ a (.virt 23710) = a (.wire 41 11) ∧ a (.wire 4 11) = a (.wire 37 40) ∧ a (.virt 17744) = a (.wire 37 41) ∧ a (.virt 17744) = a (.wire 37 42) ∧ a (.wire 4 11) = a (.wire 37 44) ∧ a (.wire 41 12) = a (.wire 37 45) ∧ a (.wire 37 43) = a (.wire 37 46) ∧ a (.wire 4 11) = a (.wire 37 48) ∧ a (.virt 17745) = a (.wire 37 49) ∧ a (.virt 17745) = a (.wire 37 50) ∧ a (.wire 4 11) = a (.wire 37 52) ∧ a (.wire 41 13) = a (.wire 37 53) ∧ a (.wire 37 51) = a (.wire 37 54) ∧ a (.wire 4 11) = a (.wire 37 56) ∧ a (.virt 17746) = a (.wire 37 57) ∧ a (.virt 17746) = a (.wire 37 58) ∧ a (.wire 4 11) = a (.wire 42 0) ∧ a (.wire 41 14) = a (.wire 42 1) ∧ a (.wire 37 59) = a (.wire 42 2) ∧ a (.wire 4 11) = a (.wire 42 4) ∧ a (.virt 17747) = a (.wire 42 5) ∧ a (.virt 17747) = a (.wire 42 6) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies51, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies51, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies52 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 11) = a (.wire 42 8) ∧ a (.wire 41 15) = a (.wire 42 9) ∧ a (.wire 42 7) = a (.wire 42 10) ∧ a (.virt 23705) = a (.wire 43 0) ∧ a (.virt 23706) = a (.wire 43 1) ∧ a (.virt 23707) = a (.wire 43 2) ∧ a (.virt 23708) = a (.wire 43 3) ∧ a (.virt 23709) = a (.wire 43 4) ∧ a (.virt 23710) = a (.wire 43 5) ∧ a (.virt 23710) = a (.wire 43 6) ∧ a (.virt 23710) = a (.wire 43 7) ∧ a (.virt 23710) = a (.wire 43 8) ∧ a (.virt 23710) = a (.wire 43 9) ∧ a (.virt 23710) = a (.wire 43 10) ∧ a (.virt 23710) = a (.wire 43 11) ∧ a (.wire 43 12) = a (.wire 44 0) ∧ a (.wire 43 13) = a (.wire 44 1) ∧ a (.wire 43 14) = a (.wire 44 2) ∧ a (.wire 43 15) = a (.wire 44 3) ∧ a (.virt 23709) = a (.wire 44 4) ∧ a (.virt 23710) = a (.wire 44 5) ∧ a (.virt 23710) = a (.wire 44 6) ∧ a (.virt 23710) = a (.wire 44 7) ∧ a (.virt 23710) = a (.wire 44 8) ∧ a (.virt 23710) = a (.wire 44 9) ∧ a (.virt 23710) = a (.wire 44 10) ∧ a (.virt 23710) = a (.wire 44 11) ∧ a (.wire 4 55) = a (.wire 42 12) ∧ a (.virt 23662) = a (.wire 42 13) ∧ a (.virt 23662) = a (.wire 42 14) ∧ a (.wire 4 55) = a (.wire 42 16) ∧ a (.wire 44 12) = a (.wire 42 17) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies52, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_left _ hq))))))
  simp only [privateBatchWrapper4.copies52, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_copies53 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 15) = a (.wire 42 18) ∧ a (.wire 4 55) = a (.wire 42 20) ∧ a (.virt 23663) = a (.wire 42 21) ∧ a (.virt 23663) = a (.wire 42 22) ∧ a (.wire 4 55) = a (.wire 42 24) ∧ a (.wire 44 13) = a (.wire 42 25) ∧ a (.wire 42 23) = a (.wire 42 26) ∧ a (.wire 4 55) = a (.wire 42 28) ∧ a (.virt 23664) = a (.wire 42 29) ∧ a (.virt 23664) = a (.wire 42 30) ∧ a (.wire 4 55) = a (.wire 42 32) ∧ a (.wire 44 14) = a (.wire 42 33) ∧ a (.wire 42 31) = a (.wire 42 34) ∧ a (.wire 4 55) = a (.wire 42 36) ∧ a (.virt 23665) = a (.wire 42 37) ∧ a (.virt 23665) = a (.wire 42 38) ∧ a (.wire 4 55) = a (.wire 42 40) ∧ a (.wire 44 15) = a (.wire 42 41) ∧ a (.wire 42 39) = a (.wire 42 42) := by
  have hc : ∀ q ∈ privateBatchWrapper4.copies53, a q.1 = a q.2 := fun q hq => h.2.1 q (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ (List.mem_append_right _ hq))))))
  simp only [privateBatchWrapper4.copies53, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hc
  exact hc

theorem privateBatchWrapper4_consts (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = 1 ∧ a (.virt 23710) = 0 ∧ a (.virt 23711) = 4 ∧ a (.virt 23776) = 576460752303423488 ∧ a (.virt 23777) = 10000 := by
  have hconst := h.2.2
  simp only [privateBatchWrapper4, List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true] at hconst
  obtain ⟨k0, k1, k2, k3, k4⟩ := hconst
  exact ⟨k0, k1, k2, k3, k4⟩

theorem privateBatchWrapper4_f0 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5912)) (a (.virt 23710)) (a (.virt 23712)) (a (.virt 23713)) := by
  have c0 := (privateBatchWrapper4_copies0 a h).1
  have c1 := (privateBatchWrapper4_copies0 a h).2.1
  have c2 := (privateBatchWrapper4_copies0 a h).2.2.1
  have c3 := (privateBatchWrapper4_copies0 a h).2.2.2.1
  have c4 := (privateBatchWrapper4_copies0 a h).2.2.2.2.1
  have c5 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.1
  have c6 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.1
  have c7 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.1
  have c8 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.1
  have c9 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.1
  have c10 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.1
  have c11 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c12 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c13 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f1 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5913)) (a (.virt 23710)) (a (.virt 23714)) (a (.virt 23715)) := by
  have c14 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c15 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c16 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c17 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c18 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c19 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c20 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f2 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5914)) (a (.virt 23710)) (a (.virt 23716)) (a (.virt 23717)) := by
  have c28 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31 := (privateBatchWrapper4_copies0 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32 := (privateBatchWrapper4_copies1 a h).1
  have c33 := (privateBatchWrapper4_copies1 a h).2.1
  have c34 := (privateBatchWrapper4_copies1 a h).2.2.1
  have c35 := (privateBatchWrapper4_copies1 a h).2.2.2.1
  have c36 := (privateBatchWrapper4_copies1 a h).2.2.2.2.1
  have c37 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.1
  have c38 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.1
  have c39 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.1
  have c40 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.1
  have c41 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f3 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5915)) (a (.virt 23710)) (a (.virt 23718)) (a (.virt 23719)) := by
  have c42 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c46 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c47 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c48 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c49 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c50 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c51 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c52 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c53 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f4 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 1 35) = band (a (.virt 23712)) (a (.virt 23714)) := by
  have c56 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_8 := arithEq_of_rows h (row := 1) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_8
  simp only [← c56, ← c57, ← c58] at e_1_8
  have hr := e_1_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f5 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 1 39) = band (a (.virt 23716)) (a (.virt 23718)) := by
  have c59 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c60 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c61 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_1_9 := arithEq_of_rows h (row := 1) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_9
  simp only [← c59, ← c60, ← c61] at e_1_9
  have hr := e_1_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f6 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) := by
  have c62 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63 := (privateBatchWrapper4_copies1 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64 := (privateBatchWrapper4_copies2 a h).1
  have e_1_10 := arithEq_of_rows h (row := 1) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1_10
  simp only [← c62, ← c63, ← c64] at e_1_10
  have hr := e_1_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f7 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11830)) (a (.virt 23710)) (a (.virt 23720)) (a (.virt 23721)) := by
  have c65 := (privateBatchWrapper4_copies2 a h).2.1
  have c66 := (privateBatchWrapper4_copies2 a h).2.2.1
  have c67 := (privateBatchWrapper4_copies2 a h).2.2.2.1
  have c68 := (privateBatchWrapper4_copies2 a h).2.2.2.2.1
  have c69 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.1
  have c70 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.1
  have c71 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.1
  have c72 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.1
  have c73 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.1
  have c74 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c77 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f8 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11831)) (a (.virt 23710)) (a (.virt 23722)) (a (.virt 23723)) := by
  have c79 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c84 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c85 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f9 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11832)) (a (.virt 23710)) (a (.virt 23724)) (a (.virt 23725)) := by
  have c93 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95 := (privateBatchWrapper4_copies2 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96 := (privateBatchWrapper4_copies3 a h).1
  have c97 := (privateBatchWrapper4_copies3 a h).2.1
  have c98 := (privateBatchWrapper4_copies3 a h).2.2.1
  have c99 := (privateBatchWrapper4_copies3 a h).2.2.2.1
  have c100 := (privateBatchWrapper4_copies3 a h).2.2.2.2.1
  have c101 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.1
  have c102 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.1
  have c103 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.1
  have c104 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.1
  have c105 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.1
  have c106 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f10 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11833)) (a (.virt 23710)) (a (.virt 23726)) (a (.virt 23727)) := by
  have c107 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
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

theorem privateBatchWrapper4_f11 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 2 19) = band (a (.virt 23720)) (a (.virt 23722)) := by
  have c121 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_4 := arithEq_of_rows h (row := 2) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_4
  simp only [← c121, ← c122, ← c123] at e_2_4
  have hr := e_2_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f12 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 2 23) = band (a (.virt 23724)) (a (.virt 23726)) := by
  have c124 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c126 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_2_5 := arithEq_of_rows h (row := 2) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_5
  simp only [← c124, ← c125, ← c126] at e_2_5
  have hr := e_2_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f13 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 2 27) = band (a (.wire 2 19)) (a (.wire 2 23)) := by
  have c127 := (privateBatchWrapper4_copies3 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c128 := (privateBatchWrapper4_copies4 a h).1
  have c129 := (privateBatchWrapper4_copies4 a h).2.1
  have e_2_6 := arithEq_of_rows h (row := 2) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_6
  simp only [← c127, ← c128, ← c129] at e_2_6
  have hr := e_2_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f14 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17748)) (a (.virt 23710)) (a (.virt 23728)) (a (.virt 23729)) := by
  have c130 := (privateBatchWrapper4_copies4 a h).2.2.1
  have c131 := (privateBatchWrapper4_copies4 a h).2.2.2.1
  have c132 := (privateBatchWrapper4_copies4 a h).2.2.2.2.1
  have c133 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.1
  have c134 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.1
  have c135 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.1
  have c136 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.1
  have c137 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.1
  have c138 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.1
  have c139 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c140 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c141 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c142 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c143 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_2_7 := arithEq_of_rows h (row := 2) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_7
  simp only [← c133, ← c134, ← c135] at e_2_7
  have e_2_8 := arithEq_of_rows h (row := 2) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_8
  simp only [← c136, ← c137, ← c138] at e_2_8
  have e_3_1 := arithEq_of_rows h (row := 3) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_1
  simp only [← c130, k0, ← c131, k0, ← c132] at e_3_1
  have e_3_2 := arithEq_of_rows h (row := 3) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_2
  simp only [← c139, ← c140, k0, ← c141] at e_3_2
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_7
    linear_combination c142.trans k1 - hc
  · have hc := e_3_2
    simp only [e_3_1, e_2_8] at hc
    linear_combination c143.trans k1 - hc

theorem privateBatchWrapper4_f15 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17749)) (a (.virt 23710)) (a (.virt 23730)) (a (.virt 23731)) := by
  have c144 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c145 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c146 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c147 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c148 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c149 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c150 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c151 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c152 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c153 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c154 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c155 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c156 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c157 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_2_9 := arithEq_of_rows h (row := 2) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_9
  simp only [← c147, ← c148, ← c149] at e_2_9
  have e_2_10 := arithEq_of_rows h (row := 2) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_10
  simp only [← c150, ← c151, ← c152] at e_2_10
  have e_3_3 := arithEq_of_rows h (row := 3) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_3
  simp only [← c144, k0, ← c145, k0, ← c146] at e_3_3
  have e_3_4 := arithEq_of_rows h (row := 3) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_4
  simp only [← c153, ← c154, k0, ← c155] at e_3_4
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_9
    linear_combination c156.trans k1 - hc
  · have hc := e_3_4
    simp only [e_3_3, e_2_10] at hc
    linear_combination c157.trans k1 - hc

theorem privateBatchWrapper4_f16 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17750)) (a (.virt 23710)) (a (.virt 23732)) (a (.virt 23733)) := by
  have c158 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c159 := (privateBatchWrapper4_copies4 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c160 := (privateBatchWrapper4_copies5 a h).1
  have c161 := (privateBatchWrapper4_copies5 a h).2.1
  have c162 := (privateBatchWrapper4_copies5 a h).2.2.1
  have c163 := (privateBatchWrapper4_copies5 a h).2.2.2.1
  have c164 := (privateBatchWrapper4_copies5 a h).2.2.2.2.1
  have c165 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.1
  have c166 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.1
  have c167 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.1
  have c168 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.1
  have c169 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.1
  have c170 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.1
  have c171 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_2_11 := arithEq_of_rows h (row := 2) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_11
  simp only [← c161, ← c162, ← c163] at e_2_11
  have e_2_12 := arithEq_of_rows h (row := 2) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_12
  simp only [← c164, ← c165, ← c166] at e_2_12
  have e_3_5 := arithEq_of_rows h (row := 3) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_5
  simp only [← c158, k0, ← c159, k0, ← c160] at e_3_5
  have e_3_6 := arithEq_of_rows h (row := 3) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_6
  simp only [← c167, ← c168, k0, ← c169] at e_3_6
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_11
    linear_combination c170.trans k1 - hc
  · have hc := e_3_6
    simp only [e_3_5, e_2_12] at hc
    linear_combination c171.trans k1 - hc

theorem privateBatchWrapper4_f17 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17751)) (a (.virt 23710)) (a (.virt 23734)) (a (.virt 23735)) := by
  have c172 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c173 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c174 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c175 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c176 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c177 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c178 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c179 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c180 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c181 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c182 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c183 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c184 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c185 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_2_13 := arithEq_of_rows h (row := 2) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_13
  simp only [← c175, ← c176, ← c177] at e_2_13
  have e_2_14 := arithEq_of_rows h (row := 2) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2_14
  simp only [← c178, ← c179, ← c180] at e_2_14
  have e_3_7 := arithEq_of_rows h (row := 3) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_7
  simp only [← c172, k0, ← c173, k0, ← c174] at e_3_7
  have e_3_8 := arithEq_of_rows h (row := 3) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_8
  simp only [← c181, ← c182, k0, ← c183] at e_3_8
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_2_13
    linear_combination c184.trans k1 - hc
  · have hc := e_3_8
    simp only [e_3_7, e_2_14] at hc
    linear_combination c185.trans k1 - hc

theorem privateBatchWrapper4_f18 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 3) = band (a (.virt 23728)) (a (.virt 23730)) := by
  have c186 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c187 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c188 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_0 := arithEq_of_rows h (row := 4) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_0
  simp only [← c186, ← c187, ← c188] at e_4_0
  have hr := e_4_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f19 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 7) = band (a (.virt 23732)) (a (.virt 23734)) := by
  have c189 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c190 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c191 := (privateBatchWrapper4_copies5 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_4_1 := arithEq_of_rows h (row := 4) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_1
  simp only [← c189, ← c190, ← c191] at e_4_1
  have hr := e_4_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f20 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 11) = band (a (.wire 4 3)) (a (.wire 4 7)) := by
  have c192 := (privateBatchWrapper4_copies6 a h).1
  have c193 := (privateBatchWrapper4_copies6 a h).2.1
  have c194 := (privateBatchWrapper4_copies6 a h).2.2.1
  have e_4_2 := arithEq_of_rows h (row := 4) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_2
  simp only [← c192, ← c193, ← c194] at e_4_2
  have hr := e_4_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f21 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23666)) (a (.virt 23710)) (a (.virt 23736)) (a (.virt 23737)) := by
  have c195 := (privateBatchWrapper4_copies6 a h).2.2.2.1
  have c196 := (privateBatchWrapper4_copies6 a h).2.2.2.2.1
  have c197 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.1
  have c198 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.1
  have c199 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.1
  have c200 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.1
  have c201 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.1
  have c202 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.1
  have c203 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c204 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c205 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c206 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c207 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c208 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_3_9 := arithEq_of_rows h (row := 3) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_9
  simp only [← c195, k0, ← c196, k0, ← c197] at e_3_9
  have e_3_10 := arithEq_of_rows h (row := 3) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_10
  simp only [← c204, ← c205, k0, ← c206] at e_3_10
  have e_4_3 := arithEq_of_rows h (row := 4) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_3
  simp only [← c198, ← c199, ← c200] at e_4_3
  have e_4_4 := arithEq_of_rows h (row := 4) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_4
  simp only [← c201, ← c202, ← c203] at e_4_4
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_4_3
    linear_combination c207.trans k1 - hc
  · have hc := e_3_10
    simp only [e_3_9, e_4_4] at hc
    linear_combination c208.trans k1 - hc

theorem privateBatchWrapper4_f22 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23667)) (a (.virt 23710)) (a (.virt 23738)) (a (.virt 23739)) := by
  have c209 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c210 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c211 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c212 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c213 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c214 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c215 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c216 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c217 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c218 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c219 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c220 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c221 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c222 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_3_11 := arithEq_of_rows h (row := 3) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_11
  simp only [← c209, k0, ← c210, k0, ← c211] at e_3_11
  have e_3_12 := arithEq_of_rows h (row := 3) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_12
  simp only [← c218, ← c219, k0, ← c220] at e_3_12
  have e_4_5 := arithEq_of_rows h (row := 4) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_5
  simp only [← c212, ← c213, ← c214] at e_4_5
  have e_4_6 := arithEq_of_rows h (row := 4) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_6
  simp only [← c215, ← c216, ← c217] at e_4_6
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_4_5
    linear_combination c221.trans k1 - hc
  · have hc := e_3_12
    simp only [e_3_11, e_4_6] at hc
    linear_combination c222.trans k1 - hc

theorem privateBatchWrapper4_f23 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23668)) (a (.virt 23710)) (a (.virt 23740)) (a (.virt 23741)) := by
  have c223 := (privateBatchWrapper4_copies6 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c224 := (privateBatchWrapper4_copies7 a h).1
  have c225 := (privateBatchWrapper4_copies7 a h).2.1
  have c226 := (privateBatchWrapper4_copies7 a h).2.2.1
  have c227 := (privateBatchWrapper4_copies7 a h).2.2.2.1
  have c228 := (privateBatchWrapper4_copies7 a h).2.2.2.2.1
  have c229 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.1
  have c230 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.1
  have c231 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.1
  have c232 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.1
  have c233 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.1
  have c234 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.1
  have c235 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c236 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_3_13 := arithEq_of_rows h (row := 3) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_13
  simp only [← c223, k0, ← c224, k0, ← c225] at e_3_13
  have e_3_14 := arithEq_of_rows h (row := 3) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_3_14
  simp only [← c232, ← c233, k0, ← c234] at e_3_14
  have e_4_7 := arithEq_of_rows h (row := 4) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_7
  simp only [← c226, ← c227, ← c228] at e_4_7
  have e_4_8 := arithEq_of_rows h (row := 4) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_8
  simp only [← c229, ← c230, ← c231] at e_4_8
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_4_7
    linear_combination c235.trans k1 - hc
  · have hc := e_3_14
    simp only [e_3_13, e_4_8] at hc
    linear_combination c236.trans k1 - hc

theorem privateBatchWrapper4_f24 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23669)) (a (.virt 23710)) (a (.virt 23742)) (a (.virt 23743)) := by
  have c237 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c238 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c239 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c240 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c241 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c242 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c243 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c244 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c245 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c246 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c247 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c248 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c249 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c250 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_4_9 := arithEq_of_rows h (row := 4) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_9
  simp only [← c240, ← c241, ← c242] at e_4_9
  have e_4_10 := arithEq_of_rows h (row := 4) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_10
  simp only [← c243, ← c244, ← c245] at e_4_10
  have e_5_0 := arithEq_of_rows h (row := 5) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_0
  simp only [← c237, k0, ← c238, k0, ← c239] at e_5_0
  have e_5_1 := arithEq_of_rows h (row := 5) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_1
  simp only [← c246, ← c247, k0, ← c248] at e_5_1
  simp only [k1]
  refine ⟨?_, ?_⟩
  · have hc := e_4_9
    linear_combination c249.trans k1 - hc
  · have hc := e_5_1
    simp only [e_5_0, e_4_10] at hc
    linear_combination c250.trans k1 - hc

theorem privateBatchWrapper4_f25 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 47) = band (a (.virt 23736)) (a (.virt 23738)) := by
  have c251 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c252 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c253 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_11 := arithEq_of_rows h (row := 4) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_11
  simp only [← c251, ← c252, ← c253] at e_4_11
  have hr := e_4_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f26 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 51) = band (a (.virt 23740)) (a (.virt 23742)) := by
  have c254 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c255 := (privateBatchWrapper4_copies7 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c256 := (privateBatchWrapper4_copies8 a h).1
  have e_4_12 := arithEq_of_rows h (row := 4) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_12
  simp only [← c254, ← c255, ← c256] at e_4_12
  have hr := e_4_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f27 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 55) = band (a (.wire 4 47)) (a (.wire 4 51)) := by
  have c257 := (privateBatchWrapper4_copies8 a h).2.1
  have c258 := (privateBatchWrapper4_copies8 a h).2.2.1
  have c259 := (privateBatchWrapper4_copies8 a h).2.2.2.1
  have e_4_13 := arithEq_of_rows h (row := 4) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_13
  simp only [← c257, ← c258, ← c259] at e_4_13
  have hr := e_4_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f28 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 11) = bnot (a (.wire 1 43)) := by
  have c260 := (privateBatchWrapper4_copies8 a h).2.2.2.2.1
  have c261 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c260, k0, ← c261, k0, ← c262] at e_5_2
  have hr := e_5_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f29 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23709) = bnot (a (.virt 23710)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  simp only [bnot, k0, k1]
  ring

theorem privateBatchWrapper4_f30 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 11) = band (a (.wire 5 11)) (a (.virt 23709)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  simp only [band, k0]
  ring

theorem privateBatchWrapper4_f31 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 15) = bselect (a (.wire 5 11)) (a (.virt 5912)) (a (.virt 23710)) := by
  have c263 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.1
  have c264 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.1
  have c265 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_3 := arithEq_of_rows h (row := 5) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_3
  simp only [← c263, ← c264, ← c265, k1] at e_5_3
  have hr := e_5_3
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f32 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 19) = bselect (a (.wire 5 11)) (a (.virt 5913)) (a (.virt 23710)) := by
  have c266 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.1
  have c267 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c268 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_4 := arithEq_of_rows h (row := 5) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_4
  simp only [← c266, ← c267, ← c268, k1] at e_5_4
  have hr := e_5_4
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f33 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 23) = bselect (a (.wire 5 11)) (a (.virt 5914)) (a (.virt 23710)) := by
  have c269 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c270 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c271 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_5 := arithEq_of_rows h (row := 5) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_5
  simp only [← c269, ← c270, ← c271, k1] at e_5_5
  have hr := e_5_5
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f34 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 27) = bselect (a (.wire 5 11)) (a (.virt 5915)) (a (.virt 23710)) := by
  have c272 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c273 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c274 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_6 := arithEq_of_rows h (row := 5) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_6
  simp only [← c272, ← c273, ← c274, k1] at e_5_6
  have hr := e_5_6
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f35 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 31) = bselect (a (.wire 5 11)) (a (.virt 5916)) (a (.virt 23710)) := by
  have c275 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c276 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c277 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_7 := arithEq_of_rows h (row := 5) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_7
  simp only [← c275, ← c276, ← c277, k1] at e_5_7
  have hr := e_5_7
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f36 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 11) = bor (a (.virt 23710)) (a (.wire 5 11)) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  simp only [bor, k1]
  ring

theorem privateBatchWrapper4_f37 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 35) = bnot (a (.wire 2 27)) := by
  have c278 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c278, k0, ← c279, k0, ← c280] at e_5_8
  have hr := e_5_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f38 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 39) = bnot (a (.wire 5 11)) := by
  have c281 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c282 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c283 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_9 := arithEq_of_rows h (row := 5) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_9
  simp only [← c281, k0, ← c282, k0, ← c283] at e_5_9
  have hr := e_5_9
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f39 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 4 59) = band (a (.wire 5 35)) (a (.wire 5 39)) := by
  have c284 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c285 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c286 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_4_14 := arithEq_of_rows h (row := 4) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_4_14
  simp only [← c284, ← c285, ← c286] at e_4_14
  have hr := e_4_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f40 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 47) = bselect (a (.wire 4 59)) (a (.virt 11830)) (a (.wire 5 15)) := by
  have c287 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c288 := (privateBatchWrapper4_copies9 a h).1
  have c289 := (privateBatchWrapper4_copies9 a h).2.1
  have c290 := (privateBatchWrapper4_copies9 a h).2.2.1
  have c291 := (privateBatchWrapper4_copies9 a h).2.2.2.1
  have c292 := (privateBatchWrapper4_copies9 a h).2.2.2.2.1
  have e_5_10 := arithEq_of_rows h (row := 5) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_10
  simp only [← c287, ← c288, ← c289] at e_5_10
  have e_5_11 := arithEq_of_rows h (row := 5) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_11
  simp only [← c290, ← c291, ← c292] at e_5_11
  have hr := e_5_11
  simp only [e_5_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f41 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 55) = bselect (a (.wire 4 59)) (a (.virt 11831)) (a (.wire 5 19)) := by
  have c293 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.1
  have c294 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.1
  have c295 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.1
  have c296 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.1
  have c297 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.1
  have c298 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_5_12 := arithEq_of_rows h (row := 5) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_12
  simp only [← c293, ← c294, ← c295] at e_5_12
  have e_5_13 := arithEq_of_rows h (row := 5) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_13
  simp only [← c296, ← c297, ← c298] at e_5_13
  have hr := e_5_13
  simp only [e_5_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f42 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 3) = bselect (a (.wire 4 59)) (a (.virt 11832)) (a (.wire 5 23)) := by
  have c299 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c300 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c301 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c302 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c303 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c304 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_5_14 := arithEq_of_rows h (row := 5) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_14
  simp only [← c299, ← c300, ← c301] at e_5_14
  have e_6_0 := arithEq_of_rows h (row := 6) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_0
  simp only [← c302, ← c303, ← c304] at e_6_0
  have hr := e_6_0
  simp only [e_5_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f43 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 11) = bselect (a (.wire 4 59)) (a (.virt 11833)) (a (.wire 5 27)) := by
  have c305 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c306 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c307 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c308 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c309 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c310 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_1 := arithEq_of_rows h (row := 6) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_1
  simp only [← c305, ← c306, ← c307] at e_6_1
  have e_6_2 := arithEq_of_rows h (row := 6) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_2
  simp only [← c308, ← c309, ← c310] at e_6_2
  have hr := e_6_2
  simp only [e_6_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f44 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 19) = bselect (a (.wire 4 59)) (a (.virt 11834)) (a (.wire 5 31)) := by
  have c311 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c312 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c313 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c314 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c315 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c316 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_3 := arithEq_of_rows h (row := 6) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_3
  simp only [← c311, ← c312, ← c313] at e_6_3
  have e_6_4 := arithEq_of_rows h (row := 6) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_4
  simp only [← c314, ← c315, ← c316] at e_6_4
  have hr := e_6_4
  simp only [e_6_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f45 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 3) = bor (a (.wire 5 11)) (a (.wire 5 35)) := by
  have c317 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c318 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c319 := (privateBatchWrapper4_copies9 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c320 := (privateBatchWrapper4_copies10 a h).1
  have c321 := (privateBatchWrapper4_copies10 a h).2.1
  have c322 := (privateBatchWrapper4_copies10 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_0 := arithEq_of_rows h (row := 7) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_0
  simp only [← c317, ← c318, ← c319] at e_7_0
  have e_8_0 := arithEq_of_rows h (row := 8) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_0
  simp only [← c320, ← c321, k0, ← c322] at e_8_0
  have hr := e_8_0
  simp only [e_7_0] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f46 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 23) = bnot (a (.wire 4 11)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c323, k0, ← c324, k0, ← c325] at e_6_5
  have hr := e_6_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f47 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 27) = bnot (a (.wire 8 3)) := by
  have c326 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.1
  have c327 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.1
  have c328 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_6_6 := arithEq_of_rows h (row := 6) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_6
  simp only [← c326, k0, ← c327, k0, ← c328] at e_6_6
  have hr := e_6_6
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f48 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 9 3) = band (a (.wire 6 23)) (a (.wire 6 27)) := by
  have c329 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.1
  have c330 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.1
  have c331 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_0 := arithEq_of_rows h (row := 9) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_0
  simp only [← c329, ← c330, ← c331] at e_9_0
  have hr := e_9_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f49 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 35) = bselect (a (.wire 9 3)) (a (.virt 17748)) (a (.wire 5 47)) := by
  have c332 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c333 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c334 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c335 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c336 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c337 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_7 := arithEq_of_rows h (row := 6) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_7
  simp only [← c332, ← c333, ← c334] at e_6_7
  have e_6_8 := arithEq_of_rows h (row := 6) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_8
  simp only [← c335, ← c336, ← c337] at e_6_8
  have hr := e_6_8
  simp only [e_6_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f50 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 43) = bselect (a (.wire 9 3)) (a (.virt 17749)) (a (.wire 5 55)) := by
  have c338 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c339 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c340 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c341 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c342 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c343 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_9 := arithEq_of_rows h (row := 6) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_9
  simp only [← c338, ← c339, ← c340] at e_6_9
  have e_6_10 := arithEq_of_rows h (row := 6) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_10
  simp only [← c341, ← c342, ← c343] at e_6_10
  have hr := e_6_10
  simp only [e_6_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f51 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 51) = bselect (a (.wire 9 3)) (a (.virt 17750)) (a (.wire 6 3)) := by
  have c344 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c345 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c346 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c347 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c348 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c349 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_6_11 := arithEq_of_rows h (row := 6) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_11
  simp only [← c344, ← c345, ← c346] at e_6_11
  have e_6_12 := arithEq_of_rows h (row := 6) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_12
  simp only [← c347, ← c348, ← c349] at e_6_12
  have hr := e_6_12
  simp only [e_6_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f52 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 59) = bselect (a (.wire 9 3)) (a (.virt 17751)) (a (.wire 6 11)) := by
  have c350 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c351 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c352 := (privateBatchWrapper4_copies11 a h).1
  have c353 := (privateBatchWrapper4_copies11 a h).2.1
  have c354 := (privateBatchWrapper4_copies11 a h).2.2.1
  have c355 := (privateBatchWrapper4_copies11 a h).2.2.2.1
  have e_6_13 := arithEq_of_rows h (row := 6) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_13
  simp only [← c350, ← c351, ← c352] at e_6_13
  have e_6_14 := arithEq_of_rows h (row := 6) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_14
  simp only [← c353, ← c354, ← c355] at e_6_14
  have hr := e_6_14
  simp only [e_6_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f53 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 7) = bselect (a (.wire 9 3)) (a (.virt 17752)) (a (.wire 6 19)) := by
  have c356 := (privateBatchWrapper4_copies11 a h).2.2.2.2.1
  have c357 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.1
  have c358 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.1
  have c359 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.1
  have c360 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.1
  have c361 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.1
  have e_10_0 := arithEq_of_rows h (row := 10) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_0
  simp only [← c356, ← c357, ← c358] at e_10_0
  have e_10_1 := arithEq_of_rows h (row := 10) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_1
  simp only [← c359, ← c360, ← c361] at e_10_1
  have hr := e_10_1
  simp only [e_10_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f54 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 7) = bor (a (.wire 8 3)) (a (.wire 6 23)) := by
  have c362 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.1
  have c363 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c364 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c365 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c366 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c367 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_1 := arithEq_of_rows h (row := 7) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_1
  simp only [← c362, ← c363, ← c364] at e_7_1
  have e_8_1 := arithEq_of_rows h (row := 8) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_1
  simp only [← c365, ← c366, k0, ← c367] at e_8_1
  have hr := e_8_1
  simp only [e_7_1] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f55 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 4 55)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k0, ← c369, k0, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f56 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 15) = bnot (a (.wire 8 7)) := by
  have c371 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c372 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c373 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_10_3 := arithEq_of_rows h (row := 10) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_3
  simp only [← c371, k0, ← c372, k0, ← c373] at e_10_3
  have hr := e_10_3
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f57 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 9 7) = band (a (.wire 10 11)) (a (.wire 10 15)) := by
  have c374 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c375 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c376 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_9_1 := arithEq_of_rows h (row := 9) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_1
  simp only [← c374, ← c375, ← c376] at e_9_1
  have hr := e_9_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f58 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 23) = bselect (a (.wire 9 7)) (a (.virt 23666)) (a (.wire 6 35)) := by
  have c377 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c378 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c379 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c380 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c381 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c382 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_4 := arithEq_of_rows h (row := 10) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_4
  simp only [← c377, ← c378, ← c379] at e_10_4
  have e_10_5 := arithEq_of_rows h (row := 10) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_5
  simp only [← c380, ← c381, ← c382] at e_10_5
  have hr := e_10_5
  simp only [e_10_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f59 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 31) = bselect (a (.wire 9 7)) (a (.virt 23667)) (a (.wire 6 43)) := by
  have c383 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c384 := (privateBatchWrapper4_copies12 a h).1
  have c385 := (privateBatchWrapper4_copies12 a h).2.1
  have c386 := (privateBatchWrapper4_copies12 a h).2.2.1
  have c387 := (privateBatchWrapper4_copies12 a h).2.2.2.1
  have c388 := (privateBatchWrapper4_copies12 a h).2.2.2.2.1
  have e_10_6 := arithEq_of_rows h (row := 10) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_6
  simp only [← c383, ← c384, ← c385] at e_10_6
  have e_10_7 := arithEq_of_rows h (row := 10) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_7
  simp only [← c386, ← c387, ← c388] at e_10_7
  have hr := e_10_7
  simp only [e_10_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f60 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 39) = bselect (a (.wire 9 7)) (a (.virt 23668)) (a (.wire 6 51)) := by
  have c389 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.1
  have c390 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.1
  have c391 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.1
  have c392 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.1
  have c393 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.1
  have c394 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_10_8 := arithEq_of_rows h (row := 10) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_8
  simp only [← c389, ← c390, ← c391] at e_10_8
  have e_10_9 := arithEq_of_rows h (row := 10) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_9
  simp only [← c392, ← c393, ← c394] at e_10_9
  have hr := e_10_9
  simp only [e_10_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f61 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 47) = bselect (a (.wire 9 7)) (a (.virt 23669)) (a (.wire 6 59)) := by
  have c395 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c396 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c397 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c398 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c399 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c400 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_10 := arithEq_of_rows h (row := 10) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_10
  simp only [← c395, ← c396, ← c397] at e_10_10
  have e_10_11 := arithEq_of_rows h (row := 10) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_11
  simp only [← c398, ← c399, ← c400] at e_10_11
  have hr := e_10_11
  simp only [e_10_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f62 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 55) = bselect (a (.wire 9 7)) (a (.virt 23670)) (a (.wire 10 7)) := by
  have c401 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c402 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c403 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c404 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c405 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c406 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_10_12 := arithEq_of_rows h (row := 10) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_12
  simp only [← c401, ← c402, ← c403] at e_10_12
  have e_10_13 := arithEq_of_rows h (row := 10) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_13
  simp only [← c404, ← c405, ← c406] at e_10_13
  have hr := e_10_13
  simp only [e_10_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f63 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 11) = bor (a (.wire 8 7)) (a (.wire 10 11)) := by
  have c407 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c408 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c409 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c410 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c411 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c412 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_2 := arithEq_of_rows h (row := 7) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_2
  simp only [← c407, ← c408, ← c409] at e_7_2
  have e_8_2 := arithEq_of_rows h (row := 8) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_2
  simp only [← c410, ← c411, k0, ← c412] at e_8_2
  have hr := e_8_2
  simp only [e_7_2] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f64 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5912)) (a (.wire 10 23)) (a (.virt 23744)) (a (.virt 23745)) := by
  have c413 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c414 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c415 := (privateBatchWrapper4_copies12 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c416 := (privateBatchWrapper4_copies13 a h).1
  have c417 := (privateBatchWrapper4_copies13 a h).2.1
  have c418 := (privateBatchWrapper4_copies13 a h).2.2.1
  have c419 := (privateBatchWrapper4_copies13 a h).2.2.2.1
  have c420 := (privateBatchWrapper4_copies13 a h).2.2.2.2.1
  have c421 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.1
  have c422 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.1
  have c423 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.1
  have c424 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.1
  have c425 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.1
  have c426 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.1
  have c427 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c428 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c429 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_9_2 := arithEq_of_rows h (row := 9) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_2
  simp only [← c419, ← c420, ← c421] at e_9_2
  have e_9_3 := arithEq_of_rows h (row := 9) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_3
  simp only [← c422, ← c423, ← c424] at e_9_3
  have e_10_14 := arithEq_of_rows h (row := 10) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_14
  simp only [← c413, k0, ← c414, k0, ← c415] at e_10_14
  have e_11_0 := arithEq_of_rows h (row := 11) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_0
  simp only [← c416, ← c417, k0, ← c418] at e_11_0
  have e_11_1 := arithEq_of_rows h (row := 11) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_1
  simp only [← c425, ← c426, k0, ← c427] at e_11_1
  refine ⟨?_, ?_⟩
  · have hc := e_9_2
    simp only [e_11_0] at hc
    linear_combination c428.trans k1 - hc
  · have hc := e_11_1
    simp only [e_10_14, e_9_3, e_11_0] at hc
    linear_combination c429.trans k1 - hc

theorem privateBatchWrapper4_f65 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5913)) (a (.wire 10 31)) (a (.virt 23746)) (a (.virt 23747)) := by
  have c430 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c431 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c432 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c433 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c434 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c435 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c436 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c437 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c438 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c439 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c440 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c441 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c442 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c443 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c444 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c445 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c446 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_9_4 := arithEq_of_rows h (row := 9) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_4
  simp only [← c436, ← c437, ← c438] at e_9_4
  have e_9_5 := arithEq_of_rows h (row := 9) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_5
  simp only [← c439, ← c440, ← c441] at e_9_5
  have e_11_2 := arithEq_of_rows h (row := 11) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_2
  simp only [← c430, k0, ← c431, k0, ← c432] at e_11_2
  have e_11_3 := arithEq_of_rows h (row := 11) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_3
  simp only [← c433, ← c434, k0, ← c435] at e_11_3
  have e_11_4 := arithEq_of_rows h (row := 11) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_4
  simp only [← c442, ← c443, k0, ← c444] at e_11_4
  refine ⟨?_, ?_⟩
  · have hc := e_9_4
    simp only [e_11_3] at hc
    linear_combination c445.trans k1 - hc
  · have hc := e_11_4
    simp only [e_11_2, e_9_5, e_11_3] at hc
    linear_combination c446.trans k1 - hc

theorem privateBatchWrapper4_f66 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5914)) (a (.wire 10 39)) (a (.virt 23748)) (a (.virt 23749)) := by
  have c447 := (privateBatchWrapper4_copies13 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c448 := (privateBatchWrapper4_copies14 a h).1
  have c449 := (privateBatchWrapper4_copies14 a h).2.1
  have c450 := (privateBatchWrapper4_copies14 a h).2.2.1
  have c451 := (privateBatchWrapper4_copies14 a h).2.2.2.1
  have c452 := (privateBatchWrapper4_copies14 a h).2.2.2.2.1
  have c453 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.1
  have c454 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.1
  have c455 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.1
  have c456 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.1
  have c457 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.1
  have c458 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.1
  have c459 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c460 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c461 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c462 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c463 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_9_6 := arithEq_of_rows h (row := 9) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_6
  simp only [← c453, ← c454, ← c455] at e_9_6
  have e_9_7 := arithEq_of_rows h (row := 9) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_7
  simp only [← c456, ← c457, ← c458] at e_9_7
  have e_11_5 := arithEq_of_rows h (row := 11) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_5
  simp only [← c447, k0, ← c448, k0, ← c449] at e_11_5
  have e_11_6 := arithEq_of_rows h (row := 11) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_6
  simp only [← c450, ← c451, k0, ← c452] at e_11_6
  have e_11_7 := arithEq_of_rows h (row := 11) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_7
  simp only [← c459, ← c460, k0, ← c461] at e_11_7
  refine ⟨?_, ?_⟩
  · have hc := e_9_6
    simp only [e_11_6] at hc
    linear_combination c462.trans k1 - hc
  · have hc := e_11_7
    simp only [e_11_5, e_9_7, e_11_6] at hc
    linear_combination c463.trans k1 - hc

theorem privateBatchWrapper4_f67 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5915)) (a (.wire 10 47)) (a (.virt 23750)) (a (.virt 23751)) := by
  have c464 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c465 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c466 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c467 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c468 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c469 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c470 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c471 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c472 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c473 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c474 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c475 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c476 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c477 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c478 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c479 := (privateBatchWrapper4_copies14 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c480 := (privateBatchWrapper4_copies15 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_9_8 := arithEq_of_rows h (row := 9) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_8
  simp only [← c470, ← c471, ← c472] at e_9_8
  have e_9_9 := arithEq_of_rows h (row := 9) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_9
  simp only [← c473, ← c474, ← c475] at e_9_9
  have e_11_8 := arithEq_of_rows h (row := 11) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_8
  simp only [← c464, k0, ← c465, k0, ← c466] at e_11_8
  have e_11_9 := arithEq_of_rows h (row := 11) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_9
  simp only [← c467, ← c468, k0, ← c469] at e_11_9
  have e_11_10 := arithEq_of_rows h (row := 11) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_10
  simp only [← c476, ← c477, k0, ← c478] at e_11_10
  refine ⟨?_, ?_⟩
  · have hc := e_9_8
    simp only [e_11_9] at hc
    linear_combination c479.trans k1 - hc
  · have hc := e_11_10
    simp only [e_11_8, e_9_9, e_11_9] at hc
    linear_combination c480.trans k1 - hc

theorem privateBatchWrapper4_f68 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 9 43) = band (a (.virt 23744)) (a (.virt 23746)) := by
  have c481 := (privateBatchWrapper4_copies15 a h).2.1
  have c482 := (privateBatchWrapper4_copies15 a h).2.2.1
  have c483 := (privateBatchWrapper4_copies15 a h).2.2.2.1
  have e_9_10 := arithEq_of_rows h (row := 9) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_10
  simp only [← c481, ← c482, ← c483] at e_9_10
  have hr := e_9_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f69 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 9 47) = band (a (.virt 23748)) (a (.virt 23750)) := by
  have c484 := (privateBatchWrapper4_copies15 a h).2.2.2.2.1
  have c485 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.1
  have c486 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.1
  have e_9_11 := arithEq_of_rows h (row := 9) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_11
  simp only [← c484, ← c485, ← c486] at e_9_11
  have hr := e_9_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f70 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 9 51) = band (a (.wire 9 43)) (a (.wire 9 47)) := by
  have c487 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.1
  have c488 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.1
  have c489 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.1
  have e_9_12 := arithEq_of_rows h (row := 9) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_12
  simp only [← c487, ← c488, ← c489] at e_9_12
  have hr := e_9_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f71 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 15) = bor (a (.wire 1 43)) (a (.wire 9 51)) := by
  have c490 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.1
  have c491 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c492 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c493 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c494 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c495 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_3 := arithEq_of_rows h (row := 7) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_3
  simp only [← c490, ← c491, ← c492] at e_7_3
  have e_8_3 := arithEq_of_rows h (row := 8) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_3
  simp only [← c493, ← c494, k0, ← c495] at e_8_3
  have hr := e_8_3
  simp only [e_7_3] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f72 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 15) = a (.virt 23709) := by
  have c496 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c496

theorem privateBatchWrapper4_f73 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 5907) = a (.virt 5907) := by
  rfl

theorem privateBatchWrapper4_f74 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11830)) (a (.wire 10 23)) (a (.virt 23752)) (a (.virt 23753)) := by
  have c498 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c499 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c500 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c501 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c502 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c503 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c504 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c505 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c506 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c507 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c508 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c509 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c510 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c511 := (privateBatchWrapper4_copies15 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c512 := (privateBatchWrapper4_copies16 a h).1
  have c513 := (privateBatchWrapper4_copies16 a h).2.1
  have c514 := (privateBatchWrapper4_copies16 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_9_13 := arithEq_of_rows h (row := 9) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_13
  simp only [← c504, ← c505, ← c506] at e_9_13
  have e_9_14 := arithEq_of_rows h (row := 9) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_9_14
  simp only [← c507, ← c508, ← c509] at e_9_14
  have e_11_11 := arithEq_of_rows h (row := 11) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_11
  simp only [← c498, k0, ← c499, k0, ← c500] at e_11_11
  have e_11_12 := arithEq_of_rows h (row := 11) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_12
  simp only [← c501, ← c502, k0, ← c503] at e_11_12
  have e_11_13 := arithEq_of_rows h (row := 11) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_13
  simp only [← c510, ← c511, k0, ← c512] at e_11_13
  refine ⟨?_, ?_⟩
  · have hc := e_9_13
    simp only [e_11_12] at hc
    linear_combination c513.trans k1 - hc
  · have hc := e_11_13
    simp only [e_11_11, e_9_14, e_11_12] at hc
    linear_combination c514.trans k1 - hc

theorem privateBatchWrapper4_f75 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11831)) (a (.wire 10 31)) (a (.virt 23754)) (a (.virt 23755)) := by
  have c515 := (privateBatchWrapper4_copies16 a h).2.2.2.1
  have c516 := (privateBatchWrapper4_copies16 a h).2.2.2.2.1
  have c517 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.1
  have c518 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.1
  have c519 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.1
  have c520 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.1
  have c521 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.1
  have c522 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.1
  have c523 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c524 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c525 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c526 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c527 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c528 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c529 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c530 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c531 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_11_14 := arithEq_of_rows h (row := 11) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_11_14
  simp only [← c515, k0, ← c516, k0, ← c517] at e_11_14
  have e_12_0 := arithEq_of_rows h (row := 12) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_0
  simp only [← c518, ← c519, k0, ← c520] at e_12_0
  have e_12_1 := arithEq_of_rows h (row := 12) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_1
  simp only [← c527, ← c528, k0, ← c529] at e_12_1
  have e_13_0 := arithEq_of_rows h (row := 13) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_0
  simp only [← c521, ← c522, ← c523] at e_13_0
  have e_13_1 := arithEq_of_rows h (row := 13) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_1
  simp only [← c524, ← c525, ← c526] at e_13_1
  refine ⟨?_, ?_⟩
  · have hc := e_13_0
    simp only [e_12_0] at hc
    linear_combination c530.trans k1 - hc
  · have hc := e_12_1
    simp only [e_11_14, e_13_1, e_12_0] at hc
    linear_combination c531.trans k1 - hc

theorem privateBatchWrapper4_f76 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11832)) (a (.wire 10 39)) (a (.virt 23756)) (a (.virt 23757)) := by
  have c532 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c533 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c534 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c535 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c536 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c537 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c538 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c539 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c540 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c541 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c542 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c543 := (privateBatchWrapper4_copies16 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c544 := (privateBatchWrapper4_copies17 a h).1
  have c545 := (privateBatchWrapper4_copies17 a h).2.1
  have c546 := (privateBatchWrapper4_copies17 a h).2.2.1
  have c547 := (privateBatchWrapper4_copies17 a h).2.2.2.1
  have c548 := (privateBatchWrapper4_copies17 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_12_2 := arithEq_of_rows h (row := 12) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_2
  simp only [← c532, k0, ← c533, k0, ← c534] at e_12_2
  have e_12_3 := arithEq_of_rows h (row := 12) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_3
  simp only [← c535, ← c536, k0, ← c537] at e_12_3
  have e_12_4 := arithEq_of_rows h (row := 12) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_4
  simp only [← c544, ← c545, k0, ← c546] at e_12_4
  have e_13_2 := arithEq_of_rows h (row := 13) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_2
  simp only [← c538, ← c539, ← c540] at e_13_2
  have e_13_3 := arithEq_of_rows h (row := 13) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_3
  simp only [← c541, ← c542, ← c543] at e_13_3
  refine ⟨?_, ?_⟩
  · have hc := e_13_2
    simp only [e_12_3] at hc
    linear_combination c547.trans k1 - hc
  · have hc := e_12_4
    simp only [e_12_2, e_13_3, e_12_3] at hc
    linear_combination c548.trans k1 - hc

theorem privateBatchWrapper4_f77 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11833)) (a (.wire 10 47)) (a (.virt 23758)) (a (.virt 23759)) := by
  have c549 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.1
  have c550 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.1
  have c551 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.1
  have c552 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.1
  have c553 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.1
  have c554 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.1
  have c555 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c556 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c557 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c558 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c559 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c560 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c561 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c562 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c563 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c564 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c565 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_12_5 := arithEq_of_rows h (row := 12) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_5
  simp only [← c549, k0, ← c550, k0, ← c551] at e_12_5
  have e_12_6 := arithEq_of_rows h (row := 12) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_6
  simp only [← c552, ← c553, k0, ← c554] at e_12_6
  have e_12_7 := arithEq_of_rows h (row := 12) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_7
  simp only [← c561, ← c562, k0, ← c563] at e_12_7
  have e_13_4 := arithEq_of_rows h (row := 13) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_4
  simp only [← c555, ← c556, ← c557] at e_13_4
  have e_13_5 := arithEq_of_rows h (row := 13) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_5
  simp only [← c558, ← c559, ← c560] at e_13_5
  refine ⟨?_, ?_⟩
  · have hc := e_13_4
    simp only [e_12_6] at hc
    linear_combination c564.trans k1 - hc
  · have hc := e_12_7
    simp only [e_12_5, e_13_5, e_12_6] at hc
    linear_combination c565.trans k1 - hc

theorem privateBatchWrapper4_f78 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 13 27) = band (a (.virt 23752)) (a (.virt 23754)) := by
  have c566 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c567 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c568 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_6 := arithEq_of_rows h (row := 13) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_6
  simp only [← c566, ← c567, ← c568] at e_13_6
  have hr := e_13_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f79 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 13 31) = band (a (.virt 23756)) (a (.virt 23758)) := by
  have c569 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c570 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c571 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_7 := arithEq_of_rows h (row := 13) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_7
  simp only [← c569, ← c570, ← c571] at e_13_7
  have hr := e_13_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f80 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 13 35) = band (a (.wire 13 27)) (a (.wire 13 31)) := by
  have c572 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c573 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c574 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_13_8 := arithEq_of_rows h (row := 13) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_8
  simp only [← c572, ← c573, ← c574] at e_13_8
  have hr := e_13_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f81 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 19) = bor (a (.wire 2 27)) (a (.wire 13 35)) := by
  have c575 := (privateBatchWrapper4_copies17 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c576 := (privateBatchWrapper4_copies18 a h).1
  have c577 := (privateBatchWrapper4_copies18 a h).2.1
  have c578 := (privateBatchWrapper4_copies18 a h).2.2.1
  have c579 := (privateBatchWrapper4_copies18 a h).2.2.2.1
  have c580 := (privateBatchWrapper4_copies18 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_4 := arithEq_of_rows h (row := 7) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_4
  simp only [← c575, ← c576, ← c577] at e_7_4
  have e_8_4 := arithEq_of_rows h (row := 8) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_4
  simp only [← c578, ← c579, k0, ← c580] at e_8_4
  have hr := e_8_4
  simp only [e_7_4] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f82 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 19) = a (.virt 23709) := by
  have c581 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.1
  exact c581

theorem privateBatchWrapper4_f83 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 11825) = a (.virt 5907) := by
  have c582 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.1
  exact c582

theorem privateBatchWrapper4_f84 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17748)) (a (.wire 10 23)) (a (.virt 23760)) (a (.virt 23761)) := by
  have c583 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.1
  have c584 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.1
  have c585 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.1
  have c586 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.1
  have c587 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c588 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c589 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c590 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c591 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c592 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c593 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c594 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c595 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c596 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c597 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c598 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c599 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_12_8 := arithEq_of_rows h (row := 12) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_8
  simp only [← c583, k0, ← c584, k0, ← c585] at e_12_8
  have e_12_9 := arithEq_of_rows h (row := 12) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_9
  simp only [← c586, ← c587, k0, ← c588] at e_12_9
  have e_12_10 := arithEq_of_rows h (row := 12) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_10
  simp only [← c595, ← c596, k0, ← c597] at e_12_10
  have e_13_9 := arithEq_of_rows h (row := 13) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_9
  simp only [← c589, ← c590, ← c591] at e_13_9
  have e_13_10 := arithEq_of_rows h (row := 13) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_10
  simp only [← c592, ← c593, ← c594] at e_13_10
  refine ⟨?_, ?_⟩
  · have hc := e_13_9
    simp only [e_12_9] at hc
    linear_combination c598.trans k1 - hc
  · have hc := e_12_10
    simp only [e_12_8, e_13_10, e_12_9] at hc
    linear_combination c599.trans k1 - hc

theorem privateBatchWrapper4_f85 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17749)) (a (.wire 10 31)) (a (.virt 23762)) (a (.virt 23763)) := by
  have c600 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c601 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c602 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c603 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c604 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c605 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c606 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c607 := (privateBatchWrapper4_copies18 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c608 := (privateBatchWrapper4_copies19 a h).1
  have c609 := (privateBatchWrapper4_copies19 a h).2.1
  have c610 := (privateBatchWrapper4_copies19 a h).2.2.1
  have c611 := (privateBatchWrapper4_copies19 a h).2.2.2.1
  have c612 := (privateBatchWrapper4_copies19 a h).2.2.2.2.1
  have c613 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.1
  have c614 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.1
  have c615 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.1
  have c616 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_12_11 := arithEq_of_rows h (row := 12) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_11
  simp only [← c600, k0, ← c601, k0, ← c602] at e_12_11
  have e_12_12 := arithEq_of_rows h (row := 12) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_12
  simp only [← c603, ← c604, k0, ← c605] at e_12_12
  have e_12_13 := arithEq_of_rows h (row := 12) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_13
  simp only [← c612, ← c613, k0, ← c614] at e_12_13
  have e_13_11 := arithEq_of_rows h (row := 13) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_11
  simp only [← c606, ← c607, ← c608] at e_13_11
  have e_13_12 := arithEq_of_rows h (row := 13) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_12
  simp only [← c609, ← c610, ← c611] at e_13_12
  refine ⟨?_, ?_⟩
  · have hc := e_13_11
    simp only [e_12_12] at hc
    linear_combination c615.trans k1 - hc
  · have hc := e_12_13
    simp only [e_12_11, e_13_12, e_12_12] at hc
    linear_combination c616.trans k1 - hc

theorem privateBatchWrapper4_f86 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17750)) (a (.wire 10 39)) (a (.virt 23764)) (a (.virt 23765)) := by
  have c617 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.1
  have c618 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.1
  have c619 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c620 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c621 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c622 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c623 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c624 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c625 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c626 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c627 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c628 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c629 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c630 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c631 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c632 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c633 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_12_14 := arithEq_of_rows h (row := 12) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_12_14
  simp only [← c617, k0, ← c618, k0, ← c619] at e_12_14
  have e_13_13 := arithEq_of_rows h (row := 13) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_13
  simp only [← c623, ← c624, ← c625] at e_13_13
  have e_13_14 := arithEq_of_rows h (row := 13) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_13_14
  simp only [← c626, ← c627, ← c628] at e_13_14
  have e_14_0 := arithEq_of_rows h (row := 14) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_0
  simp only [← c620, ← c621, k0, ← c622] at e_14_0
  have e_14_1 := arithEq_of_rows h (row := 14) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_1
  simp only [← c629, ← c630, k0, ← c631] at e_14_1
  refine ⟨?_, ?_⟩
  · have hc := e_13_13
    simp only [e_14_0] at hc
    linear_combination c632.trans k1 - hc
  · have hc := e_14_1
    simp only [e_12_14, e_13_14, e_14_0] at hc
    linear_combination c633.trans k1 - hc

theorem privateBatchWrapper4_f87 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17751)) (a (.wire 10 47)) (a (.virt 23766)) (a (.virt 23767)) := by
  have c634 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c635 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c636 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c637 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c638 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c639 := (privateBatchWrapper4_copies19 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c640 := (privateBatchWrapper4_copies20 a h).1
  have c641 := (privateBatchWrapper4_copies20 a h).2.1
  have c642 := (privateBatchWrapper4_copies20 a h).2.2.1
  have c643 := (privateBatchWrapper4_copies20 a h).2.2.2.1
  have c644 := (privateBatchWrapper4_copies20 a h).2.2.2.2.1
  have c645 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.1
  have c646 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.1
  have c647 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.1
  have c648 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.1
  have c649 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.1
  have c650 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_14_2 := arithEq_of_rows h (row := 14) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_2
  simp only [← c634, k0, ← c635, k0, ← c636] at e_14_2
  have e_14_3 := arithEq_of_rows h (row := 14) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_3
  simp only [← c637, ← c638, k0, ← c639] at e_14_3
  have e_14_4 := arithEq_of_rows h (row := 14) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_4
  simp only [← c646, ← c647, k0, ← c648] at e_14_4
  have e_15_0 := arithEq_of_rows h (row := 15) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_0
  simp only [← c640, ← c641, ← c642] at e_15_0
  have e_15_1 := arithEq_of_rows h (row := 15) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_1
  simp only [← c643, ← c644, ← c645] at e_15_1
  refine ⟨?_, ?_⟩
  · have hc := e_15_0
    simp only [e_14_3] at hc
    linear_combination c649.trans k1 - hc
  · have hc := e_14_4
    simp only [e_14_2, e_15_1, e_14_3] at hc
    linear_combination c650.trans k1 - hc

theorem privateBatchWrapper4_f88 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 15 11) = band (a (.virt 23760)) (a (.virt 23762)) := by
  have c651 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c652 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c653 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_2 := arithEq_of_rows h (row := 15) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_2
  simp only [← c651, ← c652, ← c653] at e_15_2
  have hr := e_15_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f89 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 15 15) = band (a (.virt 23764)) (a (.virt 23766)) := by
  have c654 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c655 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c656 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_3 := arithEq_of_rows h (row := 15) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_3
  simp only [← c654, ← c655, ← c656] at e_15_3
  have hr := e_15_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f90 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 15 19) = band (a (.wire 15 11)) (a (.wire 15 15)) := by
  have c657 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c658 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c659 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_15_4 := arithEq_of_rows h (row := 15) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_4
  simp only [← c657, ← c658, ← c659] at e_15_4
  have hr := e_15_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f91 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 23) = bor (a (.wire 4 11)) (a (.wire 15 19)) := by
  have c660 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c661 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c662 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c663 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c664 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c665 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_5 := arithEq_of_rows h (row := 7) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_5
  simp only [← c660, ← c661, ← c662] at e_7_5
  have e_8_5 := arithEq_of_rows h (row := 8) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_5
  simp only [← c663, ← c664, k0, ← c665] at e_8_5
  have hr := e_8_5
  simp only [e_7_5] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f92 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 23) = a (.virt 23709) := by
  have c666 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c666

theorem privateBatchWrapper4_f93 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 17743) = a (.virt 5907) := by
  have c667 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c667

theorem privateBatchWrapper4_f94 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23666)) (a (.wire 10 23)) (a (.virt 23768)) (a (.virt 23769)) := by
  have c668 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c669 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c670 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c671 := (privateBatchWrapper4_copies20 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c672 := (privateBatchWrapper4_copies21 a h).1
  have c673 := (privateBatchWrapper4_copies21 a h).2.1
  have c674 := (privateBatchWrapper4_copies21 a h).2.2.1
  have c675 := (privateBatchWrapper4_copies21 a h).2.2.2.1
  have c676 := (privateBatchWrapper4_copies21 a h).2.2.2.2.1
  have c677 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.1
  have c678 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.1
  have c679 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.1
  have c680 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.1
  have c681 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.1
  have c682 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.1
  have c683 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c684 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_14_5 := arithEq_of_rows h (row := 14) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_5
  simp only [← c668, k0, ← c669, k0, ← c670] at e_14_5
  have e_14_6 := arithEq_of_rows h (row := 14) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_6
  simp only [← c671, ← c672, k0, ← c673] at e_14_6
  have e_14_7 := arithEq_of_rows h (row := 14) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_7
  simp only [← c680, ← c681, k0, ← c682] at e_14_7
  have e_15_5 := arithEq_of_rows h (row := 15) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_5
  simp only [← c674, ← c675, ← c676] at e_15_5
  have e_15_6 := arithEq_of_rows h (row := 15) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_6
  simp only [← c677, ← c678, ← c679] at e_15_6
  refine ⟨?_, ?_⟩
  · have hc := e_15_5
    simp only [e_14_6] at hc
    linear_combination c683.trans k1 - hc
  · have hc := e_14_7
    simp only [e_14_5, e_15_6, e_14_6] at hc
    linear_combination c684.trans k1 - hc

theorem privateBatchWrapper4_f95 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23667)) (a (.wire 10 31)) (a (.virt 23770)) (a (.virt 23771)) := by
  have c685 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c686 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c687 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c688 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c689 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c690 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c691 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c692 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c693 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c694 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c695 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c696 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c697 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c698 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c699 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c700 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c701 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_14_8 := arithEq_of_rows h (row := 14) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_8
  simp only [← c685, k0, ← c686, k0, ← c687] at e_14_8
  have e_14_9 := arithEq_of_rows h (row := 14) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_9
  simp only [← c688, ← c689, k0, ← c690] at e_14_9
  have e_14_10 := arithEq_of_rows h (row := 14) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_10
  simp only [← c697, ← c698, k0, ← c699] at e_14_10
  have e_15_7 := arithEq_of_rows h (row := 15) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_7
  simp only [← c691, ← c692, ← c693] at e_15_7
  have e_15_8 := arithEq_of_rows h (row := 15) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_8
  simp only [← c694, ← c695, ← c696] at e_15_8
  refine ⟨?_, ?_⟩
  · have hc := e_15_7
    simp only [e_14_9] at hc
    linear_combination c700.trans k1 - hc
  · have hc := e_14_10
    simp only [e_14_8, e_15_8, e_14_9] at hc
    linear_combination c701.trans k1 - hc

theorem privateBatchWrapper4_f96 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23668)) (a (.wire 10 39)) (a (.virt 23772)) (a (.virt 23773)) := by
  have c702 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c703 := (privateBatchWrapper4_copies21 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c704 := (privateBatchWrapper4_copies22 a h).1
  have c705 := (privateBatchWrapper4_copies22 a h).2.1
  have c706 := (privateBatchWrapper4_copies22 a h).2.2.1
  have c707 := (privateBatchWrapper4_copies22 a h).2.2.2.1
  have c708 := (privateBatchWrapper4_copies22 a h).2.2.2.2.1
  have c709 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.1
  have c710 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.1
  have c711 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.1
  have c712 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.1
  have c713 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.1
  have c714 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.1
  have c715 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c716 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c717 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c718 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_14_11 := arithEq_of_rows h (row := 14) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_11
  simp only [← c702, k0, ← c703, k0, ← c704] at e_14_11
  have e_14_12 := arithEq_of_rows h (row := 14) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_12
  simp only [← c705, ← c706, k0, ← c707] at e_14_12
  have e_14_13 := arithEq_of_rows h (row := 14) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_13
  simp only [← c714, ← c715, k0, ← c716] at e_14_13
  have e_15_9 := arithEq_of_rows h (row := 15) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_9
  simp only [← c708, ← c709, ← c710] at e_15_9
  have e_15_10 := arithEq_of_rows h (row := 15) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_10
  simp only [← c711, ← c712, ← c713] at e_15_10
  refine ⟨?_, ?_⟩
  · have hc := e_15_9
    simp only [e_14_12] at hc
    linear_combination c717.trans k1 - hc
  · have hc := e_14_13
    simp only [e_14_11, e_15_10, e_14_12] at hc
    linear_combination c718.trans k1 - hc

theorem privateBatchWrapper4_f97 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 23669)) (a (.wire 10 47)) (a (.virt 23774)) (a (.virt 23775)) := by
  have c719 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c720 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c721 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c722 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c723 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c724 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c725 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c726 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c727 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c728 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c729 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c730 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c731 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c732 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c733 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c734 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c735 := (privateBatchWrapper4_copies22 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_14_14 := arithEq_of_rows h (row := 14) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_14_14
  simp only [← c719, k0, ← c720, k0, ← c721] at e_14_14
  have e_15_11 := arithEq_of_rows h (row := 15) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_11
  simp only [← c725, ← c726, ← c727] at e_15_11
  have e_15_12 := arithEq_of_rows h (row := 15) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_12
  simp only [← c728, ← c729, ← c730] at e_15_12
  have e_16_0 := arithEq_of_rows h (row := 16) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_0
  simp only [← c722, ← c723, k0, ← c724] at e_16_0
  have e_16_1 := arithEq_of_rows h (row := 16) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_1
  simp only [← c731, ← c732, k0, ← c733] at e_16_1
  refine ⟨?_, ?_⟩
  · have hc := e_15_11
    simp only [e_16_0] at hc
    linear_combination c734.trans k1 - hc
  · have hc := e_16_1
    simp only [e_14_14, e_15_12, e_16_0] at hc
    linear_combination c735.trans k1 - hc

theorem privateBatchWrapper4_f98 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 15 55) = band (a (.virt 23768)) (a (.virt 23770)) := by
  have c736 := (privateBatchWrapper4_copies23 a h).1
  have c737 := (privateBatchWrapper4_copies23 a h).2.1
  have c738 := (privateBatchWrapper4_copies23 a h).2.2.1
  have e_15_13 := arithEq_of_rows h (row := 15) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_13
  simp only [← c736, ← c737, ← c738] at e_15_13
  have hr := e_15_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f99 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 15 59) = band (a (.virt 23772)) (a (.virt 23774)) := by
  have c739 := (privateBatchWrapper4_copies23 a h).2.2.2.1
  have c740 := (privateBatchWrapper4_copies23 a h).2.2.2.2.1
  have c741 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.1
  have e_15_14 := arithEq_of_rows h (row := 15) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_15_14
  simp only [← c739, ← c740, ← c741] at e_15_14
  have hr := e_15_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f100 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 3) = band (a (.wire 15 55)) (a (.wire 15 59)) := by
  have c742 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.1
  have c743 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.1
  have c744 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.1
  have e_17_0 := arithEq_of_rows h (row := 17) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_0
  simp only [← c742, ← c743, ← c744] at e_17_0
  have hr := e_17_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f101 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 27) = bor (a (.wire 4 55)) (a (.wire 17 3)) := by
  have c745 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.1
  have c746 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.1
  have c747 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c748 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c749 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c750 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_7_6 := arithEq_of_rows h (row := 7) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_7_6
  simp only [← c745, ← c746, ← c747] at e_7_6
  have e_8_6 := arithEq_of_rows h (row := 8) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_6
  simp only [← c748, ← c749, k0, ← c750] at e_8_6
  have hr := e_8_6
  simp only [e_7_6] at hr
  simp only [bor]
  linear_combination hr

theorem privateBatchWrapper4_f102 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 27) = a (.virt 23709) := by
  have c751 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c751

theorem privateBatchWrapper4_f103 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23661) = a (.virt 5907) := by
  have c752 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c752

theorem privateBatchWrapper4_f104 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 15) = bselect (a (.wire 1 43)) (a (.virt 23710)) (a (.virt 5917)) := by
  have c753 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c754 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c755 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c756 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c757 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c758 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_2 := arithEq_of_rows h (row := 16) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_2
  simp only [← c753, ← c754, ← c755] at e_16_2
  have e_16_3 := arithEq_of_rows h (row := 16) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_3
  simp only [← c756, ← c757, k1, ← c758] at e_16_3
  have hr := e_16_3
  simp only [e_16_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f105 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 15) = a (.virt 23710) + a (.wire 16 15) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper4_f106 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 23) = bselect (a (.wire 2 27)) (a (.virt 23710)) (a (.virt 11835)) := by
  have c759 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c760 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c761 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c762 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c763 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c764 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_4 := arithEq_of_rows h (row := 16) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_4
  simp only [← c759, ← c760, ← c761] at e_16_4
  have e_16_5 := arithEq_of_rows h (row := 16) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_5
  simp only [← c762, ← c763, k1, ← c764] at e_16_5
  have hr := e_16_5
  simp only [e_16_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f107 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 31) = a (.wire 16 15) + a (.wire 16 23) := by
  have c765 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c766 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c767 := (privateBatchWrapper4_copies23 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_8_7 := arithEq_of_rows h (row := 8) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_7
  simp only [← c765, ← c766, k0, ← c767] at e_8_7
  have hr := e_8_7
  linear_combination hr

theorem privateBatchWrapper4_f108 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 31) = bselect (a (.wire 4 11)) (a (.virt 23710)) (a (.virt 17753)) := by
  have c768 := (privateBatchWrapper4_copies24 a h).1
  have c769 := (privateBatchWrapper4_copies24 a h).2.1
  have c770 := (privateBatchWrapper4_copies24 a h).2.2.1
  have c771 := (privateBatchWrapper4_copies24 a h).2.2.2.1
  have c772 := (privateBatchWrapper4_copies24 a h).2.2.2.2.1
  have c773 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_6 := arithEq_of_rows h (row := 16) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_6
  simp only [← c768, ← c769, ← c770] at e_16_6
  have e_16_7 := arithEq_of_rows h (row := 16) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_7
  simp only [← c771, ← c772, k1, ← c773] at e_16_7
  have hr := e_16_7
  simp only [e_16_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f109 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 35) = a (.wire 8 31) + a (.wire 16 31) := by
  have c774 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.1
  have c775 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.1
  have c776 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_8_8 := arithEq_of_rows h (row := 8) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_8
  simp only [← c774, ← c775, k0, ← c776] at e_8_8
  have hr := e_8_8
  linear_combination hr

theorem privateBatchWrapper4_f110 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 39) = bselect (a (.wire 4 55)) (a (.virt 23710)) (a (.virt 23671)) := by
  have c777 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.1
  have c778 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.1
  have c779 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c780 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c781 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c782 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_8 := arithEq_of_rows h (row := 16) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_8
  simp only [← c777, ← c778, ← c779] at e_16_8
  have e_16_9 := arithEq_of_rows h (row := 16) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_9
  simp only [← c780, ← c781, k1, ← c782] at e_16_9
  have hr := e_16_9
  simp only [e_16_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem privateBatchWrapper4_f111 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 39) = a (.wire 8 35) + a (.wire 16 39) := by
  have c783 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c784 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c785 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_8_9 := arithEq_of_rows h (row := 8) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_9
  simp only [← c783, ← c784, k0, ← c785] at e_8_9
  have hr := e_8_9
  linear_combination hr

theorem privateBatchWrapper4_f112 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 23677)) 32 := by
  have c786 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c787 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c788 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c789 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c790 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c791 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c792 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c793 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c794 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c795 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c796 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c797 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c798 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c799 := (privateBatchWrapper4_copies24 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c800 := (privateBatchWrapper4_copies25 a h).1
  have c801 := (privateBatchWrapper4_copies25 a h).2.1
  have c802 := (privateBatchWrapper4_copies25 a h).2.2.1
  have c803 := (privateBatchWrapper4_copies25 a h).2.2.2.1
  have c804 := (privateBatchWrapper4_copies25 a h).2.2.2.2.1
  have c805 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.1
  have c806 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.1
  have c807 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.1
  have c808 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.1
  have c809 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.1
  have c810 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.1
  have c811 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c812 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c813 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 18) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c786.trans k1
    · exact c787.trans k1
    · exact c788.trans k1
    · exact c789.trans k1
    · exact c790.trans k1
    · exact c791.trans k1
    · exact c792.trans k1
    · exact c793.trans k1
    · exact c794.trans k1
    · exact c795.trans k1
    · exact c796.trans k1
    · exact c797.trans k1
    · exact c798.trans k1
    · exact c799.trans k1
    · exact c800.trans k1
    · exact c801.trans k1
    · exact c802.trans k1
    · exact c803.trans k1
    · exact c804.trans k1
    · exact c805.trans k1
    · exact c806.trans k1
    · exact c807.trans k1
    · exact c808.trans k1
    · exact c809.trans k1
    · exact c810.trans k1
    · exact c811.trans k1
    · exact c812.trans k1
    )
  rwa [c813] at hr

theorem privateBatchWrapper4_f113 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.virt 23677) = a (.virt 23710) + a (.virt 23677) := by
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  simp only [k1]
  ring

theorem privateBatchWrapper4_f114 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 23682)) 32 := by
  have c814 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c815 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c816 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c817 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c818 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c819 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c820 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c821 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c822 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c823 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c824 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c825 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c826 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c827 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c828 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c829 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c830 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c831 := (privateBatchWrapper4_copies25 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c832 := (privateBatchWrapper4_copies26 a h).1
  have c833 := (privateBatchWrapper4_copies26 a h).2.1
  have c834 := (privateBatchWrapper4_copies26 a h).2.2.1
  have c835 := (privateBatchWrapper4_copies26 a h).2.2.2.1
  have c836 := (privateBatchWrapper4_copies26 a h).2.2.2.2.1
  have c837 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.1
  have c838 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.1
  have c839 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.1
  have c840 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.1
  have c841 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 19) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c814.trans k1
    · exact c815.trans k1
    · exact c816.trans k1
    · exact c817.trans k1
    · exact c818.trans k1
    · exact c819.trans k1
    · exact c820.trans k1
    · exact c821.trans k1
    · exact c822.trans k1
    · exact c823.trans k1
    · exact c824.trans k1
    · exact c825.trans k1
    · exact c826.trans k1
    · exact c827.trans k1
    · exact c828.trans k1
    · exact c829.trans k1
    · exact c830.trans k1
    · exact c831.trans k1
    · exact c832.trans k1
    · exact c833.trans k1
    · exact c834.trans k1
    · exact c835.trans k1
    · exact c836.trans k1
    · exact c837.trans k1
    · exact c838.trans k1
    · exact c839.trans k1
    · exact c840.trans k1
    )
  rwa [c841] at hr

theorem privateBatchWrapper4_f115 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 43) = a (.virt 23677) + a (.virt 23682) := by
  have c842 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.1
  have c843 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c844 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_8_10 := arithEq_of_rows h (row := 8) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_10
  simp only [← c842, ← c843, k0, ← c844] at e_8_10
  have hr := e_8_10
  linear_combination hr

theorem privateBatchWrapper4_f116 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 23687)) 32 := by
  have c845 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c846 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c847 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c848 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c849 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c850 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c851 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c852 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c853 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c854 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c855 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c856 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c857 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c858 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c859 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c860 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c861 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c862 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c863 := (privateBatchWrapper4_copies26 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c864 := (privateBatchWrapper4_copies27 a h).1
  have c865 := (privateBatchWrapper4_copies27 a h).2.1
  have c866 := (privateBatchWrapper4_copies27 a h).2.2.1
  have c867 := (privateBatchWrapper4_copies27 a h).2.2.2.1
  have c868 := (privateBatchWrapper4_copies27 a h).2.2.2.2.1
  have c869 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.1
  have c870 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.1
  have c871 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.1
  have c872 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 20) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c845.trans k1
    · exact c846.trans k1
    · exact c847.trans k1
    · exact c848.trans k1
    · exact c849.trans k1
    · exact c850.trans k1
    · exact c851.trans k1
    · exact c852.trans k1
    · exact c853.trans k1
    · exact c854.trans k1
    · exact c855.trans k1
    · exact c856.trans k1
    · exact c857.trans k1
    · exact c858.trans k1
    · exact c859.trans k1
    · exact c860.trans k1
    · exact c861.trans k1
    · exact c862.trans k1
    · exact c863.trans k1
    · exact c864.trans k1
    · exact c865.trans k1
    · exact c866.trans k1
    · exact c867.trans k1
    · exact c868.trans k1
    · exact c869.trans k1
    · exact c870.trans k1
    · exact c871.trans k1
    )
  rwa [c872] at hr

theorem privateBatchWrapper4_f117 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 47) = a (.wire 8 43) + a (.virt 23687) := by
  have c873 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.1
  have c874 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.1
  have c875 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_8_11 := arithEq_of_rows h (row := 8) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_11
  simp only [← c873, ← c874, k0, ← c875] at e_8_11
  have hr := e_8_11
  linear_combination hr

theorem privateBatchWrapper4_f118 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 23692)) 32 := by
  have c876 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c877 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c878 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c879 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c880 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c881 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c882 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c883 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c884 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c885 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c886 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c887 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c888 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c889 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c890 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c891 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c892 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c893 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c894 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c895 := (privateBatchWrapper4_copies27 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c896 := (privateBatchWrapper4_copies28 a h).1
  have c897 := (privateBatchWrapper4_copies28 a h).2.1
  have c898 := (privateBatchWrapper4_copies28 a h).2.2.1
  have c899 := (privateBatchWrapper4_copies28 a h).2.2.2.1
  have c900 := (privateBatchWrapper4_copies28 a h).2.2.2.2.1
  have c901 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.1
  have c902 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.1
  have c903 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 21) (N := 59) (n := 32) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c876.trans k1
    · exact c877.trans k1
    · exact c878.trans k1
    · exact c879.trans k1
    · exact c880.trans k1
    · exact c881.trans k1
    · exact c882.trans k1
    · exact c883.trans k1
    · exact c884.trans k1
    · exact c885.trans k1
    · exact c886.trans k1
    · exact c887.trans k1
    · exact c888.trans k1
    · exact c889.trans k1
    · exact c890.trans k1
    · exact c891.trans k1
    · exact c892.trans k1
    · exact c893.trans k1
    · exact c894.trans k1
    · exact c895.trans k1
    · exact c896.trans k1
    · exact c897.trans k1
    · exact c898.trans k1
    · exact c899.trans k1
    · exact c900.trans k1
    · exact c901.trans k1
    · exact c902.trans k1
    )
  rwa [c903] at hr

theorem privateBatchWrapper4_f119 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 8 51) = a (.wire 8 47) + a (.virt 23692) := by
  have c904 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.1
  have c905 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.1
  have c906 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_8_12 := arithEq_of_rows h (row := 8) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_8_12
  simp only [← c904, ← c905, k0, ← c906] at e_8_12
  have hr := e_8_12
  linear_combination hr

theorem privateBatchWrapper4_f120 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.virt 23672)) 14 := by
  have c907 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c908 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c909 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c910 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c911 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c912 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c913 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c914 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c915 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c916 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c917 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c918 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c919 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c920 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c921 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c922 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c923 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c924 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c925 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c926 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c927 := (privateBatchWrapper4_copies28 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c928 := (privateBatchWrapper4_copies29 a h).1
  have c929 := (privateBatchWrapper4_copies29 a h).2.1
  have c930 := (privateBatchWrapper4_copies29 a h).2.2.1
  have c931 := (privateBatchWrapper4_copies29 a h).2.2.2.1
  have c932 := (privateBatchWrapper4_copies29 a h).2.2.2.2.1
  have c933 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.1
  have c934 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.1
  have c935 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.1
  have c936 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.1
  have c937 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.1
  have c938 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.1
  have c939 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c940 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c941 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c942 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c943 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c944 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c945 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c946 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c947 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c948 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c949 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c950 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c951 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c952 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 22) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c907.trans k1
    · exact c908.trans k1
    · exact c909.trans k1
    · exact c910.trans k1
    · exact c911.trans k1
    · exact c912.trans k1
    · exact c913.trans k1
    · exact c914.trans k1
    · exact c915.trans k1
    · exact c916.trans k1
    · exact c917.trans k1
    · exact c918.trans k1
    · exact c919.trans k1
    · exact c920.trans k1
    · exact c921.trans k1
    · exact c922.trans k1
    · exact c923.trans k1
    · exact c924.trans k1
    · exact c925.trans k1
    · exact c926.trans k1
    · exact c927.trans k1
    · exact c928.trans k1
    · exact c929.trans k1
    · exact c930.trans k1
    · exact c931.trans k1
    · exact c932.trans k1
    · exact c933.trans k1
    · exact c934.trans k1
    · exact c935.trans k1
    · exact c936.trans k1
    · exact c937.trans k1
    · exact c938.trans k1
    · exact c939.trans k1
    · exact c940.trans k1
    · exact c941.trans k1
    · exact c942.trans k1
    · exact c943.trans k1
    · exact c944.trans k1
    · exact c945.trans k1
    · exact c946.trans k1
    · exact c947.trans k1
    · exact c948.trans k1
    · exact c949.trans k1
    · exact c950.trans k1
    · exact c951.trans k1
    )
  rwa [c952] at hr

theorem privateBatchWrapper4_f121 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 43) = a (.virt 23777) - a (.virt 23672) := by
  have c953 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c954 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c955 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_10 := arithEq_of_rows h (row := 16) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_10
  simp only [← c953, k4, ← c954, k0, ← c955] at e_16_10
  have hr := e_16_10
  simp only [k4]
  linear_combination hr

theorem privateBatchWrapper4_f122 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.wire 16 43)) 14 := by
  have c956 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c957 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c958 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c959 := (privateBatchWrapper4_copies29 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c960 := (privateBatchWrapper4_copies30 a h).1
  have c961 := (privateBatchWrapper4_copies30 a h).2.1
  have c962 := (privateBatchWrapper4_copies30 a h).2.2.1
  have c963 := (privateBatchWrapper4_copies30 a h).2.2.2.1
  have c964 := (privateBatchWrapper4_copies30 a h).2.2.2.2.1
  have c965 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.1
  have c966 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.1
  have c967 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.1
  have c968 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.1
  have c969 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.1
  have c970 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.1
  have c971 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c972 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c973 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c974 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c975 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c976 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c977 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c978 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c979 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c980 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c981 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c982 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c983 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c984 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c985 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c986 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c987 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c988 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c989 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c990 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c991 := (privateBatchWrapper4_copies30 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c992 := (privateBatchWrapper4_copies31 a h).1
  have c993 := (privateBatchWrapper4_copies31 a h).2.1
  have c994 := (privateBatchWrapper4_copies31 a h).2.2.1
  have c995 := (privateBatchWrapper4_copies31 a h).2.2.2.1
  have c996 := (privateBatchWrapper4_copies31 a h).2.2.2.2.1
  have c997 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.1
  have c998 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.1
  have c999 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.1
  have c1000 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.1
  have c1001 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 23) (N := 59) (n := 14) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c956.trans k1
    · exact c957.trans k1
    · exact c958.trans k1
    · exact c959.trans k1
    · exact c960.trans k1
    · exact c961.trans k1
    · exact c962.trans k1
    · exact c963.trans k1
    · exact c964.trans k1
    · exact c965.trans k1
    · exact c966.trans k1
    · exact c967.trans k1
    · exact c968.trans k1
    · exact c969.trans k1
    · exact c970.trans k1
    · exact c971.trans k1
    · exact c972.trans k1
    · exact c973.trans k1
    · exact c974.trans k1
    · exact c975.trans k1
    · exact c976.trans k1
    · exact c977.trans k1
    · exact c978.trans k1
    · exact c979.trans k1
    · exact c980.trans k1
    · exact c981.trans k1
    · exact c982.trans k1
    · exact c983.trans k1
    · exact c984.trans k1
    · exact c985.trans k1
    · exact c986.trans k1
    · exact c987.trans k1
    · exact c988.trans k1
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
    )
  rwa [c1001] at hr

theorem privateBatchWrapper4_f123 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 7) = a (.wire 8 51) * a (.virt 23777) := by
  have c1002 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1003 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1004 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_17_1 := arithEq_of_rows h (row := 17) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_1
  simp only [← c1002, ← c1003, k4, ← c1004] at e_17_1
  have hr := e_17_1
  simp only [k4]
  linear_combination hr

theorem privateBatchWrapper4_f124 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 11) = a (.wire 8 39) * a (.wire 16 43) := by
  have c1005 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1006 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1007 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_2 := arithEq_of_rows h (row := 17) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_2
  simp only [← c1005, ← c1006, ← c1007] at e_17_2
  have hr := e_17_2
  linear_combination hr

theorem privateBatchWrapper4_f125 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 16 47) = a (.wire 17 11) - a (.wire 17 7) := by
  have c1008 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1009 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1010 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_11 := arithEq_of_rows h (row := 16) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_11
  simp only [← c1008, ← c1009, k0, ← c1010] at e_16_11
  have hr := e_16_11
  linear_combination hr

theorem privateBatchWrapper4_f126 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    rangeCheck (a (.wire 16 47)) 52 := by
  have c1011 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1012 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1013 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1014 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1015 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1016 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1017 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1018 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hr := rangeCheck_of_row h (row := 24) (N := 59) (n := 52) rfl rfl (by decide) (by
    intro i hi1 hi2
    interval_cases i
    · exact c1011.trans k1
    · exact c1012.trans k1
    · exact c1013.trans k1
    · exact c1014.trans k1
    · exact c1015.trans k1
    · exact c1016.trans k1
    · exact c1017.trans k1
    )
  rwa [c1018] at hr

theorem privateBatchWrapper4_f127 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 11) = bnot (a (.wire 1 43)) := by
  have c260 := (privateBatchWrapper4_copies8 a h).2.2.2.2.1
  have c261 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.1
  have c262 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_2 := arithEq_of_rows h (row := 5) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_2
  simp only [← c260, k0, ← c261, k0, ← c262] at e_5_2
  have hr := e_5_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f128 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 35) = bnot (a (.wire 2 27)) := by
  have c278 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c278, k0, ← c279, k0, ← c280] at e_5_8
  have hr := e_5_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f129 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 15) = band (a (.wire 5 11)) (a (.wire 5 35)) := by
  have c1019 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1020 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1021 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_17_3 := arithEq_of_rows h (row := 17) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_3
  simp only [← c1019, ← c1020, ← c1021] at e_17_3
  have hr := e_17_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f130 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5908)) (a (.virt 11826)) (a (.virt 23778)) (a (.virt 23779)) := by
  have c1022 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1023 := (privateBatchWrapper4_copies31 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1024 := (privateBatchWrapper4_copies32 a h).1
  have c1025 := (privateBatchWrapper4_copies32 a h).2.1
  have c1026 := (privateBatchWrapper4_copies32 a h).2.2.1
  have c1027 := (privateBatchWrapper4_copies32 a h).2.2.2.1
  have c1028 := (privateBatchWrapper4_copies32 a h).2.2.2.2.1
  have c1029 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.1
  have c1030 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.1
  have c1031 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.1
  have c1032 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.1
  have c1033 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.1
  have c1034 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1035 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1036 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1037 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1038 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_16_12 := arithEq_of_rows h (row := 16) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_12
  simp only [← c1022, k0, ← c1023, k0, ← c1024] at e_16_12
  have e_16_13 := arithEq_of_rows h (row := 16) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_13
  simp only [← c1025, ← c1026, k0, ← c1027] at e_16_13
  have e_16_14 := arithEq_of_rows h (row := 16) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_16_14
  simp only [← c1034, ← c1035, k0, ← c1036] at e_16_14
  have e_17_4 := arithEq_of_rows h (row := 17) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_4
  simp only [← c1028, ← c1029, ← c1030] at e_17_4
  have e_17_5 := arithEq_of_rows h (row := 17) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_5
  simp only [← c1031, ← c1032, ← c1033] at e_17_5
  refine ⟨?_, ?_⟩
  · have hc := e_17_4
    simp only [e_16_13] at hc
    linear_combination c1037.trans k1 - hc
  · have hc := e_16_14
    simp only [e_16_12, e_17_5, e_16_13] at hc
    linear_combination c1038.trans k1 - hc

theorem privateBatchWrapper4_f131 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5909)) (a (.virt 11827)) (a (.virt 23780)) (a (.virt 23781)) := by
  have c1039 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1040 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1041 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1042 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1043 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1044 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1045 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1046 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1047 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1048 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1049 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1050 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1051 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1052 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1053 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1054 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1055 := (privateBatchWrapper4_copies32 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_17_6 := arithEq_of_rows h (row := 17) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_6
  simp only [← c1045, ← c1046, ← c1047] at e_17_6
  have e_17_7 := arithEq_of_rows h (row := 17) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_7
  simp only [← c1048, ← c1049, ← c1050] at e_17_7
  have e_25_0 := arithEq_of_rows h (row := 25) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_0
  simp only [← c1039, k0, ← c1040, k0, ← c1041] at e_25_0
  have e_25_1 := arithEq_of_rows h (row := 25) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_1
  simp only [← c1042, ← c1043, k0, ← c1044] at e_25_1
  have e_25_2 := arithEq_of_rows h (row := 25) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_2
  simp only [← c1051, ← c1052, k0, ← c1053] at e_25_2
  refine ⟨?_, ?_⟩
  · have hc := e_17_6
    simp only [e_25_1] at hc
    linear_combination c1054.trans k1 - hc
  · have hc := e_25_2
    simp only [e_25_0, e_17_7, e_25_1] at hc
    linear_combination c1055.trans k1 - hc

theorem privateBatchWrapper4_f132 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5910)) (a (.virt 11828)) (a (.virt 23782)) (a (.virt 23783)) := by
  have c1056 := (privateBatchWrapper4_copies33 a h).1
  have c1057 := (privateBatchWrapper4_copies33 a h).2.1
  have c1058 := (privateBatchWrapper4_copies33 a h).2.2.1
  have c1059 := (privateBatchWrapper4_copies33 a h).2.2.2.1
  have c1060 := (privateBatchWrapper4_copies33 a h).2.2.2.2.1
  have c1061 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.1
  have c1062 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.1
  have c1063 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.1
  have c1064 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.1
  have c1065 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.1
  have c1066 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1067 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1068 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1069 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1070 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1071 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1072 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_17_8 := arithEq_of_rows h (row := 17) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_8
  simp only [← c1062, ← c1063, ← c1064] at e_17_8
  have e_17_9 := arithEq_of_rows h (row := 17) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_9
  simp only [← c1065, ← c1066, ← c1067] at e_17_9
  have e_25_3 := arithEq_of_rows h (row := 25) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_3
  simp only [← c1056, k0, ← c1057, k0, ← c1058] at e_25_3
  have e_25_4 := arithEq_of_rows h (row := 25) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_4
  simp only [← c1059, ← c1060, k0, ← c1061] at e_25_4
  have e_25_5 := arithEq_of_rows h (row := 25) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_5
  simp only [← c1068, ← c1069, k0, ← c1070] at e_25_5
  refine ⟨?_, ?_⟩
  · have hc := e_17_8
    simp only [e_25_4] at hc
    linear_combination c1071.trans k1 - hc
  · have hc := e_25_5
    simp only [e_25_3, e_17_9, e_25_4] at hc
    linear_combination c1072.trans k1 - hc

theorem privateBatchWrapper4_f133 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5911)) (a (.virt 11829)) (a (.virt 23784)) (a (.virt 23785)) := by
  have c1073 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1074 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1075 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1076 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1077 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1078 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1079 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1080 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1081 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1082 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1083 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1084 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1085 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1086 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1087 := (privateBatchWrapper4_copies33 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1088 := (privateBatchWrapper4_copies34 a h).1
  have c1089 := (privateBatchWrapper4_copies34 a h).2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_17_10 := arithEq_of_rows h (row := 17) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_10
  simp only [← c1079, ← c1080, ← c1081] at e_17_10
  have e_17_11 := arithEq_of_rows h (row := 17) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_11
  simp only [← c1082, ← c1083, ← c1084] at e_17_11
  have e_25_6 := arithEq_of_rows h (row := 25) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_6
  simp only [← c1073, k0, ← c1074, k0, ← c1075] at e_25_6
  have e_25_7 := arithEq_of_rows h (row := 25) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_7
  simp only [← c1076, ← c1077, k0, ← c1078] at e_25_7
  have e_25_8 := arithEq_of_rows h (row := 25) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_8
  simp only [← c1085, ← c1086, k0, ← c1087] at e_25_8
  refine ⟨?_, ?_⟩
  · have hc := e_17_10
    simp only [e_25_7] at hc
    linear_combination c1088.trans k1 - hc
  · have hc := e_25_8
    simp only [e_25_6, e_17_11, e_25_7] at hc
    linear_combination c1089.trans k1 - hc

theorem privateBatchWrapper4_f134 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 51) = band (a (.virt 23778)) (a (.virt 23780)) := by
  have c1090 := (privateBatchWrapper4_copies34 a h).2.2.1
  have c1091 := (privateBatchWrapper4_copies34 a h).2.2.2.1
  have c1092 := (privateBatchWrapper4_copies34 a h).2.2.2.2.1
  have e_17_12 := arithEq_of_rows h (row := 17) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_12
  simp only [← c1090, ← c1091, ← c1092] at e_17_12
  have hr := e_17_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f135 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 55) = band (a (.virt 23782)) (a (.virt 23784)) := by
  have c1093 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.1
  have c1094 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.1
  have c1095 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.1
  have e_17_13 := arithEq_of_rows h (row := 17) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_13
  simp only [← c1093, ← c1094, ← c1095] at e_17_13
  have hr := e_17_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f136 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 17 59) = band (a (.wire 17 51)) (a (.wire 17 55)) := by
  have c1096 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.1
  have c1097 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.1
  have c1098 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.1
  have e_17_14 := arithEq_of_rows h (row := 17) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_17_14
  simp only [← c1096, ← c1097, ← c1098] at e_17_14
  have hr := e_17_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f137 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 3) = band (a (.wire 17 15)) (a (.wire 17 59)) := by
  have c1099 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1100 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1101 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_0 := arithEq_of_rows h (row := 26) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_0
  simp only [← c1099, ← c1100, ← c1101] at e_26_0
  have hr := e_26_0
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f138 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 3) = a (.virt 23710) := by
  have c1102 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1102

theorem privateBatchWrapper4_f139 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 23) = bnot (a (.wire 4 11)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c323, k0, ← c324, k0, ← c325] at e_6_5
  have hr := e_6_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f140 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 7) = band (a (.wire 5 11)) (a (.wire 6 23)) := by
  have c1103 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1104 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1105 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_1 := arithEq_of_rows h (row := 26) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_1
  simp only [← c1103, ← c1104, ← c1105] at e_26_1
  have hr := e_26_1
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f141 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5908)) (a (.virt 17744)) (a (.virt 23786)) (a (.virt 23787)) := by
  have c1106 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1107 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1108 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1109 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1110 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1111 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1112 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1113 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1114 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1115 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1116 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1117 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1118 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1119 := (privateBatchWrapper4_copies34 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1120 := (privateBatchWrapper4_copies35 a h).1
  have c1121 := (privateBatchWrapper4_copies35 a h).2.1
  have c1122 := (privateBatchWrapper4_copies35 a h).2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_25_9 := arithEq_of_rows h (row := 25) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_9
  simp only [← c1106, k0, ← c1107, k0, ← c1108] at e_25_9
  have e_25_10 := arithEq_of_rows h (row := 25) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_10
  simp only [← c1109, ← c1110, k0, ← c1111] at e_25_10
  have e_25_11 := arithEq_of_rows h (row := 25) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_11
  simp only [← c1118, ← c1119, k0, ← c1120] at e_25_11
  have e_26_2 := arithEq_of_rows h (row := 26) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_2
  simp only [← c1112, ← c1113, ← c1114] at e_26_2
  have e_26_3 := arithEq_of_rows h (row := 26) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_3
  simp only [← c1115, ← c1116, ← c1117] at e_26_3
  refine ⟨?_, ?_⟩
  · have hc := e_26_2
    simp only [e_25_10] at hc
    linear_combination c1121.trans k1 - hc
  · have hc := e_25_11
    simp only [e_25_9, e_26_3, e_25_10] at hc
    linear_combination c1122.trans k1 - hc

theorem privateBatchWrapper4_f142 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5909)) (a (.virt 17745)) (a (.virt 23788)) (a (.virt 23789)) := by
  have c1123 := (privateBatchWrapper4_copies35 a h).2.2.2.1
  have c1124 := (privateBatchWrapper4_copies35 a h).2.2.2.2.1
  have c1125 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.1
  have c1126 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.1
  have c1127 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.1
  have c1128 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.1
  have c1129 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.1
  have c1130 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1131 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1132 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1133 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1134 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1135 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1136 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1137 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1138 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1139 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_25_12 := arithEq_of_rows h (row := 25) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_12
  simp only [← c1123, k0, ← c1124, k0, ← c1125] at e_25_12
  have e_25_13 := arithEq_of_rows h (row := 25) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_13
  simp only [← c1126, ← c1127, k0, ← c1128] at e_25_13
  have e_25_14 := arithEq_of_rows h (row := 25) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_25_14
  simp only [← c1135, ← c1136, k0, ← c1137] at e_25_14
  have e_26_4 := arithEq_of_rows h (row := 26) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_4
  simp only [← c1129, ← c1130, ← c1131] at e_26_4
  have e_26_5 := arithEq_of_rows h (row := 26) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_5
  simp only [← c1132, ← c1133, ← c1134] at e_26_5
  refine ⟨?_, ?_⟩
  · have hc := e_26_4
    simp only [e_25_13] at hc
    linear_combination c1138.trans k1 - hc
  · have hc := e_25_14
    simp only [e_25_12, e_26_5, e_25_13] at hc
    linear_combination c1139.trans k1 - hc

theorem privateBatchWrapper4_f143 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5910)) (a (.virt 17746)) (a (.virt 23790)) (a (.virt 23791)) := by
  have c1140 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1141 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1142 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1143 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1144 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1145 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1146 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1147 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1148 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1149 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1150 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1151 := (privateBatchWrapper4_copies35 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1152 := (privateBatchWrapper4_copies36 a h).1
  have c1153 := (privateBatchWrapper4_copies36 a h).2.1
  have c1154 := (privateBatchWrapper4_copies36 a h).2.2.1
  have c1155 := (privateBatchWrapper4_copies36 a h).2.2.2.1
  have c1156 := (privateBatchWrapper4_copies36 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_26_6 := arithEq_of_rows h (row := 26) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_6
  simp only [← c1146, ← c1147, ← c1148] at e_26_6
  have e_26_7 := arithEq_of_rows h (row := 26) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_7
  simp only [← c1149, ← c1150, ← c1151] at e_26_7
  have e_27_0 := arithEq_of_rows h (row := 27) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_0
  simp only [← c1140, k0, ← c1141, k0, ← c1142] at e_27_0
  have e_27_1 := arithEq_of_rows h (row := 27) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_1
  simp only [← c1143, ← c1144, k0, ← c1145] at e_27_1
  have e_27_2 := arithEq_of_rows h (row := 27) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_2
  simp only [← c1152, ← c1153, k0, ← c1154] at e_27_2
  refine ⟨?_, ?_⟩
  · have hc := e_26_6
    simp only [e_27_1] at hc
    linear_combination c1155.trans k1 - hc
  · have hc := e_27_2
    simp only [e_27_0, e_26_7, e_27_1] at hc
    linear_combination c1156.trans k1 - hc

theorem privateBatchWrapper4_f144 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5911)) (a (.virt 17747)) (a (.virt 23792)) (a (.virt 23793)) := by
  have c1157 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.1
  have c1158 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.1
  have c1159 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.1
  have c1160 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.1
  have c1161 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.1
  have c1162 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1163 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1164 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1165 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1166 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1167 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1168 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1169 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1170 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1171 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1172 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1173 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_26_8 := arithEq_of_rows h (row := 26) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_8
  simp only [← c1163, ← c1164, ← c1165] at e_26_8
  have e_26_9 := arithEq_of_rows h (row := 26) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_9
  simp only [← c1166, ← c1167, ← c1168] at e_26_9
  have e_27_3 := arithEq_of_rows h (row := 27) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_3
  simp only [← c1157, k0, ← c1158, k0, ← c1159] at e_27_3
  have e_27_4 := arithEq_of_rows h (row := 27) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_4
  simp only [← c1160, ← c1161, k0, ← c1162] at e_27_4
  have e_27_5 := arithEq_of_rows h (row := 27) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_5
  simp only [← c1169, ← c1170, k0, ← c1171] at e_27_5
  refine ⟨?_, ?_⟩
  · have hc := e_26_8
    simp only [e_27_4] at hc
    linear_combination c1172.trans k1 - hc
  · have hc := e_27_5
    simp only [e_27_3, e_26_9, e_27_4] at hc
    linear_combination c1173.trans k1 - hc

theorem privateBatchWrapper4_f145 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 43) = band (a (.virt 23786)) (a (.virt 23788)) := by
  have c1174 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1175 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1176 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_10 := arithEq_of_rows h (row := 26) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_10
  simp only [← c1174, ← c1175, ← c1176] at e_26_10
  have hr := e_26_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f146 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 47) = band (a (.virt 23790)) (a (.virt 23792)) := by
  have c1177 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1178 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1179 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_11 := arithEq_of_rows h (row := 26) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_11
  simp only [← c1177, ← c1178, ← c1179] at e_26_11
  have hr := e_26_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f147 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 51) = band (a (.wire 26 43)) (a (.wire 26 47)) := by
  have c1180 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1181 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1182 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_26_12 := arithEq_of_rows h (row := 26) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_12
  simp only [← c1180, ← c1181, ← c1182] at e_26_12
  have hr := e_26_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f148 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 55) = band (a (.wire 26 7)) (a (.wire 26 51)) := by
  have c1183 := (privateBatchWrapper4_copies36 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1184 := (privateBatchWrapper4_copies37 a h).1
  have c1185 := (privateBatchWrapper4_copies37 a h).2.1
  have e_26_13 := arithEq_of_rows h (row := 26) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_13
  simp only [← c1183, ← c1184, ← c1185] at e_26_13
  have hr := e_26_13
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f149 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 55) = a (.virt 23710) := by
  have c1186 := (privateBatchWrapper4_copies37 a h).2.2.1
  exact c1186

theorem privateBatchWrapper4_f150 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 4 55)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k0, ← c369, k0, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f151 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 26 59) = band (a (.wire 5 11)) (a (.wire 10 11)) := by
  have c1187 := (privateBatchWrapper4_copies37 a h).2.2.2.1
  have c1188 := (privateBatchWrapper4_copies37 a h).2.2.2.2.1
  have c1189 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.1
  have e_26_14 := arithEq_of_rows h (row := 26) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_26_14
  simp only [← c1187, ← c1188, ← c1189] at e_26_14
  have hr := e_26_14
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f152 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5908)) (a (.virt 23662)) (a (.virt 23794)) (a (.virt 23795)) := by
  have c1190 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.1
  have c1191 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.1
  have c1192 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.1
  have c1193 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.1
  have c1194 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1195 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1196 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1197 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1198 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1199 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1200 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1201 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1202 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1203 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1204 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1205 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1206 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_27_6 := arithEq_of_rows h (row := 27) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_6
  simp only [← c1190, k0, ← c1191, k0, ← c1192] at e_27_6
  have e_27_7 := arithEq_of_rows h (row := 27) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_7
  simp only [← c1193, ← c1194, k0, ← c1195] at e_27_7
  have e_27_8 := arithEq_of_rows h (row := 27) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_8
  simp only [← c1202, ← c1203, k0, ← c1204] at e_27_8
  have e_28_0 := arithEq_of_rows h (row := 28) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_0
  simp only [← c1196, ← c1197, ← c1198] at e_28_0
  have e_28_1 := arithEq_of_rows h (row := 28) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_1
  simp only [← c1199, ← c1200, ← c1201] at e_28_1
  refine ⟨?_, ?_⟩
  · have hc := e_28_0
    simp only [e_27_7] at hc
    linear_combination c1205.trans k1 - hc
  · have hc := e_27_8
    simp only [e_27_6, e_28_1, e_27_7] at hc
    linear_combination c1206.trans k1 - hc

theorem privateBatchWrapper4_f153 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5909)) (a (.virt 23663)) (a (.virt 23796)) (a (.virt 23797)) := by
  have c1207 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1208 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1209 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1210 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1211 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1212 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1213 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1214 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1215 := (privateBatchWrapper4_copies37 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1216 := (privateBatchWrapper4_copies38 a h).1
  have c1217 := (privateBatchWrapper4_copies38 a h).2.1
  have c1218 := (privateBatchWrapper4_copies38 a h).2.2.1
  have c1219 := (privateBatchWrapper4_copies38 a h).2.2.2.1
  have c1220 := (privateBatchWrapper4_copies38 a h).2.2.2.2.1
  have c1221 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.1
  have c1222 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.1
  have c1223 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_27_9 := arithEq_of_rows h (row := 27) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_9
  simp only [← c1207, k0, ← c1208, k0, ← c1209] at e_27_9
  have e_27_10 := arithEq_of_rows h (row := 27) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_10
  simp only [← c1210, ← c1211, k0, ← c1212] at e_27_10
  have e_27_11 := arithEq_of_rows h (row := 27) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_11
  simp only [← c1219, ← c1220, k0, ← c1221] at e_27_11
  have e_28_2 := arithEq_of_rows h (row := 28) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_2
  simp only [← c1213, ← c1214, ← c1215] at e_28_2
  have e_28_3 := arithEq_of_rows h (row := 28) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_3
  simp only [← c1216, ← c1217, ← c1218] at e_28_3
  refine ⟨?_, ?_⟩
  · have hc := e_28_2
    simp only [e_27_10] at hc
    linear_combination c1222.trans k1 - hc
  · have hc := e_27_11
    simp only [e_27_9, e_28_3, e_27_10] at hc
    linear_combination c1223.trans k1 - hc

theorem privateBatchWrapper4_f154 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5910)) (a (.virt 23664)) (a (.virt 23798)) (a (.virt 23799)) := by
  have c1224 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.1
  have c1225 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.1
  have c1226 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1227 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1228 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1229 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1230 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1231 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1232 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1233 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1234 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1235 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1236 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1237 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1238 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1239 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1240 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_27_12 := arithEq_of_rows h (row := 27) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_12
  simp only [← c1224, k0, ← c1225, k0, ← c1226] at e_27_12
  have e_27_13 := arithEq_of_rows h (row := 27) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_13
  simp only [← c1227, ← c1228, k0, ← c1229] at e_27_13
  have e_27_14 := arithEq_of_rows h (row := 27) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_27_14
  simp only [← c1236, ← c1237, k0, ← c1238] at e_27_14
  have e_28_4 := arithEq_of_rows h (row := 28) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_4
  simp only [← c1230, ← c1231, ← c1232] at e_28_4
  have e_28_5 := arithEq_of_rows h (row := 28) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_5
  simp only [← c1233, ← c1234, ← c1235] at e_28_5
  refine ⟨?_, ?_⟩
  · have hc := e_28_4
    simp only [e_27_13] at hc
    linear_combination c1239.trans k1 - hc
  · have hc := e_27_14
    simp only [e_27_12, e_28_5, e_27_13] at hc
    linear_combination c1240.trans k1 - hc

theorem privateBatchWrapper4_f155 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 5911)) (a (.virt 23665)) (a (.virt 23800)) (a (.virt 23801)) := by
  have c1241 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1242 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1243 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1244 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1245 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1246 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1247 := (privateBatchWrapper4_copies38 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1248 := (privateBatchWrapper4_copies39 a h).1
  have c1249 := (privateBatchWrapper4_copies39 a h).2.1
  have c1250 := (privateBatchWrapper4_copies39 a h).2.2.1
  have c1251 := (privateBatchWrapper4_copies39 a h).2.2.2.1
  have c1252 := (privateBatchWrapper4_copies39 a h).2.2.2.2.1
  have c1253 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.1
  have c1254 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.1
  have c1255 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.1
  have c1256 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.1
  have c1257 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_28_6 := arithEq_of_rows h (row := 28) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_6
  simp only [← c1247, ← c1248, ← c1249] at e_28_6
  have e_28_7 := arithEq_of_rows h (row := 28) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_7
  simp only [← c1250, ← c1251, ← c1252] at e_28_7
  have e_29_0 := arithEq_of_rows h (row := 29) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_0
  simp only [← c1241, k0, ← c1242, k0, ← c1243] at e_29_0
  have e_29_1 := arithEq_of_rows h (row := 29) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_1
  simp only [← c1244, ← c1245, k0, ← c1246] at e_29_1
  have e_29_2 := arithEq_of_rows h (row := 29) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_2
  simp only [← c1253, ← c1254, k0, ← c1255] at e_29_2
  refine ⟨?_, ?_⟩
  · have hc := e_28_6
    simp only [e_29_1] at hc
    linear_combination c1256.trans k1 - hc
  · have hc := e_29_2
    simp only [e_29_0, e_28_7, e_29_1] at hc
    linear_combination c1257.trans k1 - hc

theorem privateBatchWrapper4_f156 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 28 35) = band (a (.virt 23794)) (a (.virt 23796)) := by
  have c1258 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1259 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1260 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_8 := arithEq_of_rows h (row := 28) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_8
  simp only [← c1258, ← c1259, ← c1260] at e_28_8
  have hr := e_28_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f157 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 28 39) = band (a (.virt 23798)) (a (.virt 23800)) := by
  have c1261 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1262 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1263 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_9 := arithEq_of_rows h (row := 28) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_9
  simp only [← c1261, ← c1262, ← c1263] at e_28_9
  have hr := e_28_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f158 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 28 43) = band (a (.wire 28 35)) (a (.wire 28 39)) := by
  have c1264 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1265 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1266 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_10 := arithEq_of_rows h (row := 28) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_10
  simp only [← c1264, ← c1265, ← c1266] at e_28_10
  have hr := e_28_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f159 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 28 47) = band (a (.wire 26 59)) (a (.wire 28 43)) := by
  have c1267 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1268 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1269 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_11 := arithEq_of_rows h (row := 28) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_11
  simp only [← c1267, ← c1268, ← c1269] at e_28_11
  have hr := e_28_11
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f160 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 28 47) = a (.virt 23710) := by
  have c1270 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1270

theorem privateBatchWrapper4_f161 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 5 35) = bnot (a (.wire 2 27)) := by
  have c278 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c279 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c280 := (privateBatchWrapper4_copies8 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_5_8 := arithEq_of_rows h (row := 5) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_5_8
  simp only [← c278, k0, ← c279, k0, ← c280] at e_5_8
  have hr := e_5_8
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f162 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 23) = bnot (a (.wire 4 11)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c323, k0, ← c324, k0, ← c325] at e_6_5
  have hr := e_6_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f163 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 28 51) = band (a (.wire 5 35)) (a (.wire 6 23)) := by
  have c1271 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1272 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1273 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_28_12 := arithEq_of_rows h (row := 28) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_12
  simp only [← c1271, ← c1272, ← c1273] at e_28_12
  have hr := e_28_12
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f164 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11826)) (a (.virt 17744)) (a (.virt 23802)) (a (.virt 23803)) := by
  have c1274 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1275 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1276 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1277 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1278 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1279 := (privateBatchWrapper4_copies39 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1280 := (privateBatchWrapper4_copies40 a h).1
  have c1281 := (privateBatchWrapper4_copies40 a h).2.1
  have c1282 := (privateBatchWrapper4_copies40 a h).2.2.1
  have c1283 := (privateBatchWrapper4_copies40 a h).2.2.2.1
  have c1284 := (privateBatchWrapper4_copies40 a h).2.2.2.2.1
  have c1285 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.1
  have c1286 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.1
  have c1287 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.1
  have c1288 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.1
  have c1289 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.1
  have c1290 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_28_13 := arithEq_of_rows h (row := 28) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_13
  simp only [← c1280, ← c1281, ← c1282] at e_28_13
  have e_28_14 := arithEq_of_rows h (row := 28) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_28_14
  simp only [← c1283, ← c1284, ← c1285] at e_28_14
  have e_29_3 := arithEq_of_rows h (row := 29) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_3
  simp only [← c1274, k0, ← c1275, k0, ← c1276] at e_29_3
  have e_29_4 := arithEq_of_rows h (row := 29) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_4
  simp only [← c1277, ← c1278, k0, ← c1279] at e_29_4
  have e_29_5 := arithEq_of_rows h (row := 29) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_5
  simp only [← c1286, ← c1287, k0, ← c1288] at e_29_5
  refine ⟨?_, ?_⟩
  · have hc := e_28_13
    simp only [e_29_4] at hc
    linear_combination c1289.trans k1 - hc
  · have hc := e_29_5
    simp only [e_29_3, e_28_14, e_29_4] at hc
    linear_combination c1290.trans k1 - hc

theorem privateBatchWrapper4_f165 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11827)) (a (.virt 17745)) (a (.virt 23804)) (a (.virt 23805)) := by
  have c1291 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1292 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1293 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1294 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1295 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1296 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1297 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1298 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1299 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1300 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1301 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1302 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1303 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1304 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1305 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1306 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1307 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_29_6 := arithEq_of_rows h (row := 29) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_6
  simp only [← c1291, k0, ← c1292, k0, ← c1293] at e_29_6
  have e_29_7 := arithEq_of_rows h (row := 29) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_7
  simp only [← c1294, ← c1295, k0, ← c1296] at e_29_7
  have e_29_8 := arithEq_of_rows h (row := 29) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_8
  simp only [← c1303, ← c1304, k0, ← c1305] at e_29_8
  have e_30_0 := arithEq_of_rows h (row := 30) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_0
  simp only [← c1297, ← c1298, ← c1299] at e_30_0
  have e_30_1 := arithEq_of_rows h (row := 30) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_1
  simp only [← c1300, ← c1301, ← c1302] at e_30_1
  refine ⟨?_, ?_⟩
  · have hc := e_30_0
    simp only [e_29_7] at hc
    linear_combination c1306.trans k1 - hc
  · have hc := e_29_8
    simp only [e_29_6, e_30_1, e_29_7] at hc
    linear_combination c1307.trans k1 - hc

theorem privateBatchWrapper4_f166 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11828)) (a (.virt 17746)) (a (.virt 23806)) (a (.virt 23807)) := by
  have c1308 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1309 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1310 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1311 := (privateBatchWrapper4_copies40 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1312 := (privateBatchWrapper4_copies41 a h).1
  have c1313 := (privateBatchWrapper4_copies41 a h).2.1
  have c1314 := (privateBatchWrapper4_copies41 a h).2.2.1
  have c1315 := (privateBatchWrapper4_copies41 a h).2.2.2.1
  have c1316 := (privateBatchWrapper4_copies41 a h).2.2.2.2.1
  have c1317 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.1
  have c1318 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.1
  have c1319 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.1
  have c1320 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.1
  have c1321 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.1
  have c1322 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1323 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1324 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_29_9 := arithEq_of_rows h (row := 29) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_9
  simp only [← c1308, k0, ← c1309, k0, ← c1310] at e_29_9
  have e_29_10 := arithEq_of_rows h (row := 29) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_10
  simp only [← c1311, ← c1312, k0, ← c1313] at e_29_10
  have e_29_11 := arithEq_of_rows h (row := 29) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_11
  simp only [← c1320, ← c1321, k0, ← c1322] at e_29_11
  have e_30_2 := arithEq_of_rows h (row := 30) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_2
  simp only [← c1314, ← c1315, ← c1316] at e_30_2
  have e_30_3 := arithEq_of_rows h (row := 30) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_3
  simp only [← c1317, ← c1318, ← c1319] at e_30_3
  refine ⟨?_, ?_⟩
  · have hc := e_30_2
    simp only [e_29_10] at hc
    linear_combination c1323.trans k1 - hc
  · have hc := e_29_11
    simp only [e_29_9, e_30_3, e_29_10] at hc
    linear_combination c1324.trans k1 - hc

theorem privateBatchWrapper4_f167 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11829)) (a (.virt 17747)) (a (.virt 23808)) (a (.virt 23809)) := by
  have c1325 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1326 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1327 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1328 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1329 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1330 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1331 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1332 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1333 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1334 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1335 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1336 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1337 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1338 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1339 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1340 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1341 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_29_12 := arithEq_of_rows h (row := 29) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_12
  simp only [← c1325, k0, ← c1326, k0, ← c1327] at e_29_12
  have e_29_13 := arithEq_of_rows h (row := 29) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_13
  simp only [← c1328, ← c1329, k0, ← c1330] at e_29_13
  have e_29_14 := arithEq_of_rows h (row := 29) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_29_14
  simp only [← c1337, ← c1338, k0, ← c1339] at e_29_14
  have e_30_4 := arithEq_of_rows h (row := 30) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_4
  simp only [← c1331, ← c1332, ← c1333] at e_30_4
  have e_30_5 := arithEq_of_rows h (row := 30) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_5
  simp only [← c1334, ← c1335, ← c1336] at e_30_5
  refine ⟨?_, ?_⟩
  · have hc := e_30_4
    simp only [e_29_13] at hc
    linear_combination c1340.trans k1 - hc
  · have hc := e_29_14
    simp only [e_29_12, e_30_5, e_29_13] at hc
    linear_combination c1341.trans k1 - hc

theorem privateBatchWrapper4_f168 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 30 27) = band (a (.virt 23802)) (a (.virt 23804)) := by
  have c1342 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1343 := (privateBatchWrapper4_copies41 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1344 := (privateBatchWrapper4_copies42 a h).1
  have e_30_6 := arithEq_of_rows h (row := 30) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_6
  simp only [← c1342, ← c1343, ← c1344] at e_30_6
  have hr := e_30_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f169 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 30 31) = band (a (.virt 23806)) (a (.virt 23808)) := by
  have c1345 := (privateBatchWrapper4_copies42 a h).2.1
  have c1346 := (privateBatchWrapper4_copies42 a h).2.2.1
  have c1347 := (privateBatchWrapper4_copies42 a h).2.2.2.1
  have e_30_7 := arithEq_of_rows h (row := 30) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_7
  simp only [← c1345, ← c1346, ← c1347] at e_30_7
  have hr := e_30_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f170 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 30 35) = band (a (.wire 30 27)) (a (.wire 30 31)) := by
  have c1348 := (privateBatchWrapper4_copies42 a h).2.2.2.2.1
  have c1349 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.1
  have c1350 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.1
  have e_30_8 := arithEq_of_rows h (row := 30) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_8
  simp only [← c1348, ← c1349, ← c1350] at e_30_8
  have hr := e_30_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f171 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 30 39) = band (a (.wire 28 51)) (a (.wire 30 35)) := by
  have c1351 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.1
  have c1352 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.1
  have c1353 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.1
  have e_30_9 := arithEq_of_rows h (row := 30) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_9
  simp only [← c1351, ← c1352, ← c1353] at e_30_9
  have hr := e_30_9
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f172 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 30 39) = a (.virt 23710) := by
  have c1354 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.1
  exact c1354

theorem privateBatchWrapper4_f173 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 4 55)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k0, ← c369, k0, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f174 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 30 43) = band (a (.wire 5 35)) (a (.wire 10 11)) := by
  have c1355 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1356 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1357 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_30_10 := arithEq_of_rows h (row := 30) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_10
  simp only [← c1355, ← c1356, ← c1357] at e_30_10
  have hr := e_30_10
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f175 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11826)) (a (.virt 23662)) (a (.virt 23810)) (a (.virt 23811)) := by
  have c1358 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1359 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1360 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1361 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1362 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1363 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1364 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1365 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1366 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1367 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1368 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1369 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1370 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1371 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1372 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1373 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1374 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_30_11 := arithEq_of_rows h (row := 30) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_11
  simp only [← c1364, ← c1365, ← c1366] at e_30_11
  have e_30_12 := arithEq_of_rows h (row := 30) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_12
  simp only [← c1367, ← c1368, ← c1369] at e_30_12
  have e_31_0 := arithEq_of_rows h (row := 31) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_0
  simp only [← c1358, k0, ← c1359, k0, ← c1360] at e_31_0
  have e_31_1 := arithEq_of_rows h (row := 31) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_1
  simp only [← c1361, ← c1362, k0, ← c1363] at e_31_1
  have e_31_2 := arithEq_of_rows h (row := 31) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_2
  simp only [← c1370, ← c1371, k0, ← c1372] at e_31_2
  refine ⟨?_, ?_⟩
  · have hc := e_30_11
    simp only [e_31_1] at hc
    linear_combination c1373.trans k1 - hc
  · have hc := e_31_2
    simp only [e_31_0, e_30_12, e_31_1] at hc
    linear_combination c1374.trans k1 - hc

theorem privateBatchWrapper4_f176 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11827)) (a (.virt 23663)) (a (.virt 23812)) (a (.virt 23813)) := by
  have c1375 := (privateBatchWrapper4_copies42 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1376 := (privateBatchWrapper4_copies43 a h).1
  have c1377 := (privateBatchWrapper4_copies43 a h).2.1
  have c1378 := (privateBatchWrapper4_copies43 a h).2.2.1
  have c1379 := (privateBatchWrapper4_copies43 a h).2.2.2.1
  have c1380 := (privateBatchWrapper4_copies43 a h).2.2.2.2.1
  have c1381 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.1
  have c1382 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.1
  have c1383 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.1
  have c1384 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.1
  have c1385 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.1
  have c1386 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1387 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1388 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1389 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1390 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1391 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_30_13 := arithEq_of_rows h (row := 30) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_13
  simp only [← c1381, ← c1382, ← c1383] at e_30_13
  have e_30_14 := arithEq_of_rows h (row := 30) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_30_14
  simp only [← c1384, ← c1385, ← c1386] at e_30_14
  have e_31_3 := arithEq_of_rows h (row := 31) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_3
  simp only [← c1375, k0, ← c1376, k0, ← c1377] at e_31_3
  have e_31_4 := arithEq_of_rows h (row := 31) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_4
  simp only [← c1378, ← c1379, k0, ← c1380] at e_31_4
  have e_31_5 := arithEq_of_rows h (row := 31) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_5
  simp only [← c1387, ← c1388, k0, ← c1389] at e_31_5
  refine ⟨?_, ?_⟩
  · have hc := e_30_13
    simp only [e_31_4] at hc
    linear_combination c1390.trans k1 - hc
  · have hc := e_31_5
    simp only [e_31_3, e_30_14, e_31_4] at hc
    linear_combination c1391.trans k1 - hc

theorem privateBatchWrapper4_f177 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11828)) (a (.virt 23664)) (a (.virt 23814)) (a (.virt 23815)) := by
  have c1392 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1393 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1394 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1395 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1396 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1397 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1398 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1399 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1400 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1401 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1402 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1403 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1404 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1405 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1406 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1407 := (privateBatchWrapper4_copies43 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1408 := (privateBatchWrapper4_copies44 a h).1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_31_6 := arithEq_of_rows h (row := 31) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_6
  simp only [← c1392, k0, ← c1393, k0, ← c1394] at e_31_6
  have e_31_7 := arithEq_of_rows h (row := 31) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_7
  simp only [← c1395, ← c1396, k0, ← c1397] at e_31_7
  have e_31_8 := arithEq_of_rows h (row := 31) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_8
  simp only [← c1404, ← c1405, k0, ← c1406] at e_31_8
  have e_32_0 := arithEq_of_rows h (row := 32) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_0
  simp only [← c1398, ← c1399, ← c1400] at e_32_0
  have e_32_1 := arithEq_of_rows h (row := 32) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_1
  simp only [← c1401, ← c1402, ← c1403] at e_32_1
  refine ⟨?_, ?_⟩
  · have hc := e_32_0
    simp only [e_31_7] at hc
    linear_combination c1407.trans k1 - hc
  · have hc := e_31_8
    simp only [e_31_6, e_32_1, e_31_7] at hc
    linear_combination c1408.trans k1 - hc

theorem privateBatchWrapper4_f178 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 11829)) (a (.virt 23665)) (a (.virt 23816)) (a (.virt 23817)) := by
  have c1409 := (privateBatchWrapper4_copies44 a h).2.1
  have c1410 := (privateBatchWrapper4_copies44 a h).2.2.1
  have c1411 := (privateBatchWrapper4_copies44 a h).2.2.2.1
  have c1412 := (privateBatchWrapper4_copies44 a h).2.2.2.2.1
  have c1413 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.1
  have c1414 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.1
  have c1415 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.1
  have c1416 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.1
  have c1417 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.1
  have c1418 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1419 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1420 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1421 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1422 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1423 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1424 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1425 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_31_9 := arithEq_of_rows h (row := 31) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_9
  simp only [← c1409, k0, ← c1410, k0, ← c1411] at e_31_9
  have e_31_10 := arithEq_of_rows h (row := 31) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_10
  simp only [← c1412, ← c1413, k0, ← c1414] at e_31_10
  have e_31_11 := arithEq_of_rows h (row := 31) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_11
  simp only [← c1421, ← c1422, k0, ← c1423] at e_31_11
  have e_32_2 := arithEq_of_rows h (row := 32) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_2
  simp only [← c1415, ← c1416, ← c1417] at e_32_2
  have e_32_3 := arithEq_of_rows h (row := 32) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_3
  simp only [← c1418, ← c1419, ← c1420] at e_32_3
  refine ⟨?_, ?_⟩
  · have hc := e_32_2
    simp only [e_31_10] at hc
    linear_combination c1424.trans k1 - hc
  · have hc := e_31_11
    simp only [e_31_9, e_32_3, e_31_10] at hc
    linear_combination c1425.trans k1 - hc

theorem privateBatchWrapper4_f179 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 19) = band (a (.virt 23810)) (a (.virt 23812)) := by
  have c1426 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1427 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1428 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_4 := arithEq_of_rows h (row := 32) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_4
  simp only [← c1426, ← c1427, ← c1428] at e_32_4
  have hr := e_32_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f180 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 23) = band (a (.virt 23814)) (a (.virt 23816)) := by
  have c1429 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1430 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1431 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_5 := arithEq_of_rows h (row := 32) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_5
  simp only [← c1429, ← c1430, ← c1431] at e_32_5
  have hr := e_32_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f181 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 27) = band (a (.wire 32 19)) (a (.wire 32 23)) := by
  have c1432 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1433 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1434 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_6 := arithEq_of_rows h (row := 32) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_6
  simp only [← c1432, ← c1433, ← c1434] at e_32_6
  have hr := e_32_6
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f182 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 31) = band (a (.wire 30 43)) (a (.wire 32 27)) := by
  have c1435 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1436 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1437 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_32_7 := arithEq_of_rows h (row := 32) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_7
  simp only [← c1435, ← c1436, ← c1437] at e_32_7
  have hr := e_32_7
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f183 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 31) = a (.virt 23710) := by
  have c1438 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1438

theorem privateBatchWrapper4_f184 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 6 23) = bnot (a (.wire 4 11)) := by
  have c323 := (privateBatchWrapper4_copies10 a h).2.2.2.1
  have c324 := (privateBatchWrapper4_copies10 a h).2.2.2.2.1
  have c325 := (privateBatchWrapper4_copies10 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_6_5 := arithEq_of_rows h (row := 6) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_6_5
  simp only [← c323, k0, ← c324, k0, ← c325] at e_6_5
  have hr := e_6_5
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f185 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 4 55)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k0, ← c369, k0, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f186 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 32 35) = band (a (.wire 6 23)) (a (.wire 10 11)) := by
  have c1439 := (privateBatchWrapper4_copies44 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1440 := (privateBatchWrapper4_copies45 a h).1
  have c1441 := (privateBatchWrapper4_copies45 a h).2.1
  have e_32_8 := arithEq_of_rows h (row := 32) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_8
  simp only [← c1439, ← c1440, ← c1441] at e_32_8
  have hr := e_32_8
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f187 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17744)) (a (.virt 23662)) (a (.virt 23818)) (a (.virt 23819)) := by
  have c1442 := (privateBatchWrapper4_copies45 a h).2.2.1
  have c1443 := (privateBatchWrapper4_copies45 a h).2.2.2.1
  have c1444 := (privateBatchWrapper4_copies45 a h).2.2.2.2.1
  have c1445 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.1
  have c1446 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.1
  have c1447 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.1
  have c1448 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.1
  have c1449 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.1
  have c1450 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1451 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1452 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1453 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1454 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1455 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1456 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1457 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1458 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_31_12 := arithEq_of_rows h (row := 31) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_12
  simp only [← c1442, k0, ← c1443, k0, ← c1444] at e_31_12
  have e_31_13 := arithEq_of_rows h (row := 31) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_13
  simp only [← c1445, ← c1446, k0, ← c1447] at e_31_13
  have e_31_14 := arithEq_of_rows h (row := 31) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_31_14
  simp only [← c1454, ← c1455, k0, ← c1456] at e_31_14
  have e_32_9 := arithEq_of_rows h (row := 32) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_9
  simp only [← c1448, ← c1449, ← c1450] at e_32_9
  have e_32_10 := arithEq_of_rows h (row := 32) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_10
  simp only [← c1451, ← c1452, ← c1453] at e_32_10
  refine ⟨?_, ?_⟩
  · have hc := e_32_9
    simp only [e_31_13] at hc
    linear_combination c1457.trans k1 - hc
  · have hc := e_31_14
    simp only [e_31_12, e_32_10, e_31_13] at hc
    linear_combination c1458.trans k1 - hc

theorem privateBatchWrapper4_f188 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17745)) (a (.virt 23663)) (a (.virt 23820)) (a (.virt 23821)) := by
  have c1459 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1460 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1461 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1462 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1463 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1464 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1465 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1466 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1467 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1468 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1469 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1470 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1471 := (privateBatchWrapper4_copies45 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1472 := (privateBatchWrapper4_copies46 a h).1
  have c1473 := (privateBatchWrapper4_copies46 a h).2.1
  have c1474 := (privateBatchWrapper4_copies46 a h).2.2.1
  have c1475 := (privateBatchWrapper4_copies46 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_32_11 := arithEq_of_rows h (row := 32) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_11
  simp only [← c1465, ← c1466, ← c1467] at e_32_11
  have e_32_12 := arithEq_of_rows h (row := 32) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_12
  simp only [← c1468, ← c1469, ← c1470] at e_32_12
  have e_33_0 := arithEq_of_rows h (row := 33) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_0
  simp only [← c1459, k0, ← c1460, k0, ← c1461] at e_33_0
  have e_33_1 := arithEq_of_rows h (row := 33) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_1
  simp only [← c1462, ← c1463, k0, ← c1464] at e_33_1
  have e_33_2 := arithEq_of_rows h (row := 33) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_2
  simp only [← c1471, ← c1472, k0, ← c1473] at e_33_2
  refine ⟨?_, ?_⟩
  · have hc := e_32_11
    simp only [e_33_1] at hc
    linear_combination c1474.trans k1 - hc
  · have hc := e_33_2
    simp only [e_33_0, e_32_12, e_33_1] at hc
    linear_combination c1475.trans k1 - hc

theorem privateBatchWrapper4_f189 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17746)) (a (.virt 23664)) (a (.virt 23822)) (a (.virt 23823)) := by
  have c1476 := (privateBatchWrapper4_copies46 a h).2.2.2.2.1
  have c1477 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.1
  have c1478 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.1
  have c1479 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.1
  have c1480 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.1
  have c1481 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.1
  have c1482 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1483 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1484 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1485 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1486 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1487 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1488 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1489 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1490 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1491 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1492 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_32_13 := arithEq_of_rows h (row := 32) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_13
  simp only [← c1482, ← c1483, ← c1484] at e_32_13
  have e_32_14 := arithEq_of_rows h (row := 32) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_32_14
  simp only [← c1485, ← c1486, ← c1487] at e_32_14
  have e_33_3 := arithEq_of_rows h (row := 33) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_3
  simp only [← c1476, k0, ← c1477, k0, ← c1478] at e_33_3
  have e_33_4 := arithEq_of_rows h (row := 33) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_4
  simp only [← c1479, ← c1480, k0, ← c1481] at e_33_4
  have e_33_5 := arithEq_of_rows h (row := 33) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_5
  simp only [← c1488, ← c1489, k0, ← c1490] at e_33_5
  refine ⟨?_, ?_⟩
  · have hc := e_32_13
    simp only [e_33_4] at hc
    linear_combination c1491.trans k1 - hc
  · have hc := e_33_5
    simp only [e_33_3, e_32_14, e_33_4] at hc
    linear_combination c1492.trans k1 - hc

theorem privateBatchWrapper4_f190 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    IsEqual (a (.virt 17747)) (a (.virt 23665)) (a (.virt 23824)) (a (.virt 23825)) := by
  have c1493 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1494 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1495 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1496 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1497 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1498 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1499 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1500 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1501 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1502 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1503 := (privateBatchWrapper4_copies46 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1504 := (privateBatchWrapper4_copies47 a h).1
  have c1505 := (privateBatchWrapper4_copies47 a h).2.1
  have c1506 := (privateBatchWrapper4_copies47 a h).2.2.1
  have c1507 := (privateBatchWrapper4_copies47 a h).2.2.2.1
  have c1508 := (privateBatchWrapper4_copies47 a h).2.2.2.2.1
  have c1509 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_33_6 := arithEq_of_rows h (row := 33) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_6
  simp only [← c1493, k0, ← c1494, k0, ← c1495] at e_33_6
  have e_33_7 := arithEq_of_rows h (row := 33) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_7
  simp only [← c1496, ← c1497, k0, ← c1498] at e_33_7
  have e_33_8 := arithEq_of_rows h (row := 33) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_8
  simp only [← c1505, ← c1506, k0, ← c1507] at e_33_8
  have e_34_0 := arithEq_of_rows h (row := 34) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_0
  simp only [← c1499, ← c1500, ← c1501] at e_34_0
  have e_34_1 := arithEq_of_rows h (row := 34) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_1
  simp only [← c1502, ← c1503, ← c1504] at e_34_1
  refine ⟨?_, ?_⟩
  · have hc := e_34_0
    simp only [e_33_7] at hc
    linear_combination c1508.trans k1 - hc
  · have hc := e_33_8
    simp only [e_33_6, e_34_1, e_33_7] at hc
    linear_combination c1509.trans k1 - hc

theorem privateBatchWrapper4_f191 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 11) = band (a (.virt 23818)) (a (.virt 23820)) := by
  have c1510 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.1
  have c1511 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.1
  have c1512 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.1
  have e_34_2 := arithEq_of_rows h (row := 34) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_2
  simp only [← c1510, ← c1511, ← c1512] at e_34_2
  have hr := e_34_2
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f192 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 15) = band (a (.virt 23822)) (a (.virt 23824)) := by
  have c1513 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.1
  have c1514 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1515 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_3 := arithEq_of_rows h (row := 34) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_3
  simp only [← c1513, ← c1514, ← c1515] at e_34_3
  have hr := e_34_3
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f193 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 19) = band (a (.wire 34 11)) (a (.wire 34 15)) := by
  have c1516 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1517 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1518 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_4 := arithEq_of_rows h (row := 34) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_4
  simp only [← c1516, ← c1517, ← c1518] at e_34_4
  have hr := e_34_4
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f194 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 23) = band (a (.wire 32 35)) (a (.wire 34 19)) := by
  have c1519 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1520 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1521 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_34_5 := arithEq_of_rows h (row := 34) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_34_5
  simp only [← c1519, ← c1520, ← c1521] at e_34_5
  have hr := e_34_5
  simp only [band]
  linear_combination hr

theorem privateBatchWrapper4_f195 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 34 23) = a (.virt 23710) := by
  have c1522 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c1522

theorem privateBatchWrapper4_f196 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 10 11) = bnot (a (.wire 4 55)) := by
  have c368 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c369 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c370 := (privateBatchWrapper4_copies11 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have e_10_2 := arithEq_of_rows h (row := 10) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_10_2
  simp only [← c368, k0, ← c369, k0, ← c370] at e_10_2
  have hr := e_10_2
  simp only [bnot]
  linear_combination hr

theorem privateBatchWrapper4_f197 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 35 12) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 0 ∧ a (.wire 35 13) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 1 ∧ a (.wire 35 14) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 2 ∧ a (.wire 35 15) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 3) := by
  have c1523 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1524 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1525 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1526 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1527 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1528 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1529 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1530 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1531 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1532 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1533 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1534 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 35 = addBlock (fun _ => 0) [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696), 1, 0, 0, 0] :=
    poseidon2In_first c1523.symm c1524.symm c1525.symm c1526.symm (c1527.symm.trans k0) (c1528.symm.trans k1) (c1529.symm.trans k1) (c1530.symm.trans k1) (c1531.symm.trans k1) (c1532.symm.trans k1) (c1533.symm.trans k1) (c1534.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 35) rfl rfl hin0
  have hpad : pad10 [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] =
      [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f198 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 36 12) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 0 ∧ a (.wire 36 13) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 1 ∧ a (.wire 36 14) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 2 ∧ a (.wire 36 15) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 3) := by
  have c1535 := (privateBatchWrapper4_copies47 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1536 := (privateBatchWrapper4_copies48 a h).1
  have c1537 := (privateBatchWrapper4_copies48 a h).2.1
  have c1538 := (privateBatchWrapper4_copies48 a h).2.2.1
  have c1539 := (privateBatchWrapper4_copies48 a h).2.2.2.1
  have c1540 := (privateBatchWrapper4_copies48 a h).2.2.2.2.1
  have c1541 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.1
  have c1542 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.1
  have c1543 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.1
  have c1544 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.1
  have c1545 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.1
  have c1546 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 36 = addBlock (fun _ => 0) [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15), 1, 0, 0, 0] :=
    poseidon2In_first c1535.symm c1536.symm c1537.symm c1538.symm (c1539.symm.trans k0) (c1540.symm.trans k1) (c1541.symm.trans k1) (c1542.symm.trans k1) (c1543.symm.trans k1) (c1544.symm.trans k1) (c1545.symm.trans k1) (c1546.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 36) rfl rfl hin0
  have hpad : pad10 [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] =
      [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f199 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 33 43) = bselect (a (.wire 1 43)) (a (.wire 36 12)) (a (.virt 5908)) := by
  have c1547 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1548 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1549 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1550 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1551 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1552 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_9 := arithEq_of_rows h (row := 33) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_9
  simp only [← c1547, ← c1548, ← c1549] at e_33_9
  have e_33_10 := arithEq_of_rows h (row := 33) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_10
  simp only [← c1550, ← c1551, ← c1552] at e_33_10
  have hr := e_33_10
  simp only [e_33_9] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f200 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 33 51) = bselect (a (.wire 1 43)) (a (.wire 36 13)) (a (.virt 5909)) := by
  have c1553 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1554 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1555 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1556 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1557 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1558 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_11 := arithEq_of_rows h (row := 33) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_11
  simp only [← c1553, ← c1554, ← c1555] at e_33_11
  have e_33_12 := arithEq_of_rows h (row := 33) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_12
  simp only [← c1556, ← c1557, ← c1558] at e_33_12
  have hr := e_33_12
  simp only [e_33_11] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f201 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 33 59) = bselect (a (.wire 1 43)) (a (.wire 36 14)) (a (.virt 5910)) := by
  have c1559 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1560 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1561 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1562 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1563 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1564 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_33_13 := arithEq_of_rows h (row := 33) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_13
  simp only [← c1559, ← c1560, ← c1561] at e_33_13
  have e_33_14 := arithEq_of_rows h (row := 33) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_33_14
  simp only [← c1562, ← c1563, ← c1564] at e_33_14
  have hr := e_33_14
  simp only [e_33_13] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f202 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 7) = bselect (a (.wire 1 43)) (a (.wire 36 15)) (a (.virt 5911)) := by
  have c1565 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1566 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1567 := (privateBatchWrapper4_copies48 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1568 := (privateBatchWrapper4_copies49 a h).1
  have c1569 := (privateBatchWrapper4_copies49 a h).2.1
  have c1570 := (privateBatchWrapper4_copies49 a h).2.2.1
  have e_37_0 := arithEq_of_rows h (row := 37) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_0
  simp only [← c1565, ← c1566, ← c1567] at e_37_0
  have e_37_1 := arithEq_of_rows h (row := 37) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_1
  simp only [← c1568, ← c1569, ← c1570] at e_37_1
  have hr := e_37_1
  simp only [e_37_0] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f203 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 38 12) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 0 ∧ a (.wire 38 13) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 1 ∧ a (.wire 38 14) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 2 ∧ a (.wire 38 15) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 3) := by
  have c1571 := (privateBatchWrapper4_copies49 a h).2.2.2.1
  have c1572 := (privateBatchWrapper4_copies49 a h).2.2.2.2.1
  have c1573 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.1
  have c1574 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.1
  have c1575 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.1
  have c1576 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.1
  have c1577 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.1
  have c1578 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1579 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1580 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1581 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1582 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 38 = addBlock (fun _ => 0) [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700), 1, 0, 0, 0] :=
    poseidon2In_first c1571.symm c1572.symm c1573.symm c1574.symm (c1575.symm.trans k0) (c1576.symm.trans k1) (c1577.symm.trans k1) (c1578.symm.trans k1) (c1579.symm.trans k1) (c1580.symm.trans k1) (c1581.symm.trans k1) (c1582.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 38) rfl rfl hin0
  have hpad : pad10 [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] =
      [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f204 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 39 12) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 0 ∧ a (.wire 39 13) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 1 ∧ a (.wire 39 14) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 2 ∧ a (.wire 39 15) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 3) := by
  have c1583 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1584 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1585 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1586 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1587 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1588 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1589 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1590 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1591 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1592 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1593 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1594 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 39 = addBlock (fun _ => 0) [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15), 1, 0, 0, 0] :=
    poseidon2In_first c1583.symm c1584.symm c1585.symm c1586.symm (c1587.symm.trans k0) (c1588.symm.trans k1) (c1589.symm.trans k1) (c1590.symm.trans k1) (c1591.symm.trans k1) (c1592.symm.trans k1) (c1593.symm.trans k1) (c1594.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 39) rfl rfl hin0
  have hpad : pad10 [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] =
      [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f205 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 15) = bselect (a (.wire 2 27)) (a (.wire 39 12)) (a (.virt 11826)) := by
  have c1595 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1596 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1597 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1598 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1599 := (privateBatchWrapper4_copies49 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1600 := (privateBatchWrapper4_copies50 a h).1
  have e_37_2 := arithEq_of_rows h (row := 37) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_2
  simp only [← c1595, ← c1596, ← c1597] at e_37_2
  have e_37_3 := arithEq_of_rows h (row := 37) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_3
  simp only [← c1598, ← c1599, ← c1600] at e_37_3
  have hr := e_37_3
  simp only [e_37_2] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f206 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 23) = bselect (a (.wire 2 27)) (a (.wire 39 13)) (a (.virt 11827)) := by
  have c1601 := (privateBatchWrapper4_copies50 a h).2.1
  have c1602 := (privateBatchWrapper4_copies50 a h).2.2.1
  have c1603 := (privateBatchWrapper4_copies50 a h).2.2.2.1
  have c1604 := (privateBatchWrapper4_copies50 a h).2.2.2.2.1
  have c1605 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.1
  have c1606 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.1
  have e_37_4 := arithEq_of_rows h (row := 37) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_4
  simp only [← c1601, ← c1602, ← c1603] at e_37_4
  have e_37_5 := arithEq_of_rows h (row := 37) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_5
  simp only [← c1604, ← c1605, ← c1606] at e_37_5
  have hr := e_37_5
  simp only [e_37_4] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f207 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 31) = bselect (a (.wire 2 27)) (a (.wire 39 14)) (a (.virt 11828)) := by
  have c1607 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.1
  have c1608 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.1
  have c1609 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.1
  have c1610 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1611 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1612 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_6 := arithEq_of_rows h (row := 37) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_6
  simp only [← c1607, ← c1608, ← c1609] at e_37_6
  have e_37_7 := arithEq_of_rows h (row := 37) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_7
  simp only [← c1610, ← c1611, ← c1612] at e_37_7
  have hr := e_37_7
  simp only [e_37_6] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f208 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 39) = bselect (a (.wire 2 27)) (a (.wire 39 15)) (a (.virt 11829)) := by
  have c1613 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1614 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1615 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1616 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1617 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1618 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_8 := arithEq_of_rows h (row := 37) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_8
  simp only [← c1613, ← c1614, ← c1615] at e_37_8
  have e_37_9 := arithEq_of_rows h (row := 37) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_9
  simp only [← c1616, ← c1617, ← c1618] at e_37_9
  have hr := e_37_9
  simp only [e_37_8] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f209 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 40 12) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 0 ∧ a (.wire 40 13) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 1 ∧ a (.wire 40 14) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 2 ∧ a (.wire 40 15) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 3) := by
  have c1619 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1620 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1621 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1622 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1623 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1624 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1625 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1626 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1627 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1628 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1629 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1630 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 40 = addBlock (fun _ => 0) [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704), 1, 0, 0, 0] :=
    poseidon2In_first c1619.symm c1620.symm c1621.symm c1622.symm (c1623.symm.trans k0) (c1624.symm.trans k1) (c1625.symm.trans k1) (c1626.symm.trans k1) (c1627.symm.trans k1) (c1628.symm.trans k1) (c1629.symm.trans k1) (c1630.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 40) rfl rfl hin0
  have hpad : pad10 [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] =
      [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f210 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 41 12) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 0 ∧ a (.wire 41 13) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 1 ∧ a (.wire 41 14) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 2 ∧ a (.wire 41 15) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 3) := by
  have c1631 := (privateBatchWrapper4_copies50 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1632 := (privateBatchWrapper4_copies51 a h).1
  have c1633 := (privateBatchWrapper4_copies51 a h).2.1
  have c1634 := (privateBatchWrapper4_copies51 a h).2.2.1
  have c1635 := (privateBatchWrapper4_copies51 a h).2.2.2.1
  have c1636 := (privateBatchWrapper4_copies51 a h).2.2.2.2.1
  have c1637 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.1
  have c1638 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.1
  have c1639 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.1
  have c1640 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.1
  have c1641 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.1
  have c1642 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 41 = addBlock (fun _ => 0) [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15), 1, 0, 0, 0] :=
    poseidon2In_first c1631.symm c1632.symm c1633.symm c1634.symm (c1635.symm.trans k0) (c1636.symm.trans k1) (c1637.symm.trans k1) (c1638.symm.trans k1) (c1639.symm.trans k1) (c1640.symm.trans k1) (c1641.symm.trans k1) (c1642.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 41) rfl rfl hin0
  have hpad : pad10 [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] =
      [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f211 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 47) = bselect (a (.wire 4 11)) (a (.wire 41 12)) (a (.virt 17744)) := by
  have c1643 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1644 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1645 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1646 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1647 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1648 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_10 := arithEq_of_rows h (row := 37) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_10
  simp only [← c1643, ← c1644, ← c1645] at e_37_10
  have e_37_11 := arithEq_of_rows h (row := 37) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_11
  simp only [← c1646, ← c1647, ← c1648] at e_37_11
  have hr := e_37_11
  simp only [e_37_10] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f212 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 37 55) = bselect (a (.wire 4 11)) (a (.wire 41 13)) (a (.virt 17745)) := by
  have c1649 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1650 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1651 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1652 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1653 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1654 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_12 := arithEq_of_rows h (row := 37) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_12
  simp only [← c1649, ← c1650, ← c1651] at e_37_12
  have e_37_13 := arithEq_of_rows h (row := 37) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_13
  simp only [← c1652, ← c1653, ← c1654] at e_37_13
  have hr := e_37_13
  simp only [e_37_12] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f213 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 3) = bselect (a (.wire 4 11)) (a (.wire 41 14)) (a (.virt 17746)) := by
  have c1655 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1656 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1657 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1658 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1659 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1660 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_37_14 := arithEq_of_rows h (row := 37) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_37_14
  simp only [← c1655, ← c1656, ← c1657] at e_37_14
  have e_42_0 := arithEq_of_rows h (row := 42) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_0
  simp only [← c1658, ← c1659, ← c1660] at e_42_0
  have hr := e_42_0
  simp only [e_37_14] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f214 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 11) = bselect (a (.wire 4 11)) (a (.wire 41 15)) (a (.virt 17747)) := by
  have c1661 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1662 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1663 := (privateBatchWrapper4_copies51 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1664 := (privateBatchWrapper4_copies52 a h).1
  have c1665 := (privateBatchWrapper4_copies52 a h).2.1
  have c1666 := (privateBatchWrapper4_copies52 a h).2.2.1
  have e_42_1 := arithEq_of_rows h (row := 42) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_1
  simp only [← c1661, ← c1662, ← c1663] at e_42_1
  have e_42_2 := arithEq_of_rows h (row := 42) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_2
  simp only [← c1664, ← c1665, ← c1666] at e_42_2
  have hr := e_42_2
  simp only [e_42_1] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f215 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 43 12) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 0 ∧ a (.wire 43 13) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 1 ∧ a (.wire 43 14) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 2 ∧ a (.wire 43 15) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 3) := by
  have c1667 := (privateBatchWrapper4_copies52 a h).2.2.2.1
  have c1668 := (privateBatchWrapper4_copies52 a h).2.2.2.2.1
  have c1669 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.1
  have c1670 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.1
  have c1671 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.1
  have c1672 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.1
  have c1673 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.1
  have c1674 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1675 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1676 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1677 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1678 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 43 = addBlock (fun _ => 0) [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708), 1, 0, 0, 0] :=
    poseidon2In_first c1667.symm c1668.symm c1669.symm c1670.symm (c1671.symm.trans k0) (c1672.symm.trans k1) (c1673.symm.trans k1) (c1674.symm.trans k1) (c1675.symm.trans k1) (c1676.symm.trans k1) (c1677.symm.trans k1) (c1678.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 43) rfl rfl hin0
  have hpad : pad10 [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] =
      [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f216 (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (a (.wire 44 12) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 0 ∧ a (.wire 44 13) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 1 ∧ a (.wire 44 14) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 2 ∧ a (.wire 44 15) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 3) := by
  have c1679 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1680 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1681 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1682 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1683 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1684 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1685 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1686 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1687 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1688 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1689 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1690 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3, k4⟩ := privateBatchWrapper4_consts a h
  have hin0 : poseidon2In a 44 = addBlock (fun _ => 0) [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15), 1, 0, 0, 0] :=
    poseidon2In_first c1679.symm c1680.symm c1681.symm c1682.symm (c1683.symm.trans k0) (c1684.symm.trans k1) (c1685.symm.trans k1) (c1686.symm.trans k1) (c1687.symm.trans k1) (c1688.symm.trans k1) (c1689.symm.trans k1) (c1690.symm.trans k1)
  have hout0 := poseidon2Row_absorb perm hp (row := 44) rfl rfl hin0
  have hpad : pad10 [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] =
      [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15), 1, 0, 0, 0] ++ [] := by
    simp [pad10, rate, List.replicate]
  rw [spongeHash, hpad, absorbMsg_block8, absorbMsg_nil, ← hout0]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem privateBatchWrapper4_f217 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 19) = bselect (a (.wire 4 55)) (a (.wire 44 12)) (a (.virt 23662)) := by
  have c1691 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1692 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1693 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1694 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1695 := (privateBatchWrapper4_copies52 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c1696 := (privateBatchWrapper4_copies53 a h).1
  have e_42_3 := arithEq_of_rows h (row := 42) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_3
  simp only [← c1691, ← c1692, ← c1693] at e_42_3
  have e_42_4 := arithEq_of_rows h (row := 42) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_4
  simp only [← c1694, ← c1695, ← c1696] at e_42_4
  have hr := e_42_4
  simp only [e_42_3] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f218 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 27) = bselect (a (.wire 4 55)) (a (.wire 44 13)) (a (.virt 23663)) := by
  have c1697 := (privateBatchWrapper4_copies53 a h).2.1
  have c1698 := (privateBatchWrapper4_copies53 a h).2.2.1
  have c1699 := (privateBatchWrapper4_copies53 a h).2.2.2.1
  have c1700 := (privateBatchWrapper4_copies53 a h).2.2.2.2.1
  have c1701 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.1
  have c1702 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.1
  have e_42_5 := arithEq_of_rows h (row := 42) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_5
  simp only [← c1697, ← c1698, ← c1699] at e_42_5
  have e_42_6 := arithEq_of_rows h (row := 42) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_6
  simp only [← c1700, ← c1701, ← c1702] at e_42_6
  have hr := e_42_6
  simp only [e_42_5] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f219 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 35) = bselect (a (.wire 4 55)) (a (.wire 44 14)) (a (.virt 23664)) := by
  have c1703 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.1
  have c1704 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.1
  have c1705 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.1
  have c1706 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.1
  have c1707 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c1708 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_42_7 := arithEq_of_rows h (row := 42) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_7
  simp only [← c1703, ← c1704, ← c1705] at e_42_7
  have e_42_8 := arithEq_of_rows h (row := 42) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_8
  simp only [← c1706, ← c1707, ← c1708] at e_42_8
  have hr := e_42_8
  simp only [e_42_7] at hr
  simp only [bselect]
  linear_combination hr

theorem privateBatchWrapper4_f220 (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a) :
    a (.wire 42 43) = bselect (a (.wire 4 55)) (a (.wire 44 15)) (a (.virt 23665)) := by
  have c1709 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1710 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1711 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1712 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1713 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c1714 := (privateBatchWrapper4_copies53 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have e_42_9 := arithEq_of_rows h (row := 42) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_9
  simp only [← c1709, ← c1710, ← c1711] at e_42_9
  have e_42_10 := arithEq_of_rows h (row := 42) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_42_10
  simp only [← c1712, ← c1713, ← c1714] at e_42_10
  have hr := e_42_10
  simp only [e_42_9] at hr
  simp only [bselect]
  linear_combination hr

set_option maxHeartbeats 4000000 in
/-- Every satisfying assignment of `privateBatchWrapper4` has the meaning of each recorded gadget call. Generated at gadget-call granularity; see `gadget.rs`. -/
theorem privateBatchWrapper4_decode (perm : St p → St p) (a : Assignment p) (h : Satisfies (privateBatchWrapper4 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper4 p) a) :
    (IsEqual (a (.virt 5912)) (a (.virt 23710)) (a (.virt 23712)) (a (.virt 23713)) ∧
    IsEqual (a (.virt 5913)) (a (.virt 23710)) (a (.virt 23714)) (a (.virt 23715)) ∧
    IsEqual (a (.virt 5914)) (a (.virt 23710)) (a (.virt 23716)) (a (.virt 23717)) ∧
    IsEqual (a (.virt 5915)) (a (.virt 23710)) (a (.virt 23718)) (a (.virt 23719)) ∧
    a (.wire 1 35) = band (a (.virt 23712)) (a (.virt 23714)) ∧
    a (.wire 1 39) = band (a (.virt 23716)) (a (.virt 23718)) ∧
    a (.wire 1 43) = band (a (.wire 1 35)) (a (.wire 1 39)) ∧
    IsEqual (a (.virt 11830)) (a (.virt 23710)) (a (.virt 23720)) (a (.virt 23721)) ∧
    IsEqual (a (.virt 11831)) (a (.virt 23710)) (a (.virt 23722)) (a (.virt 23723)) ∧
    IsEqual (a (.virt 11832)) (a (.virt 23710)) (a (.virt 23724)) (a (.virt 23725)) ∧
    IsEqual (a (.virt 11833)) (a (.virt 23710)) (a (.virt 23726)) (a (.virt 23727)) ∧
    a (.wire 2 19) = band (a (.virt 23720)) (a (.virt 23722)) ∧
    a (.wire 2 23) = band (a (.virt 23724)) (a (.virt 23726)) ∧
    a (.wire 2 27) = band (a (.wire 2 19)) (a (.wire 2 23)) ∧
    IsEqual (a (.virt 17748)) (a (.virt 23710)) (a (.virt 23728)) (a (.virt 23729)) ∧
    IsEqual (a (.virt 17749)) (a (.virt 23710)) (a (.virt 23730)) (a (.virt 23731)) ∧
    IsEqual (a (.virt 17750)) (a (.virt 23710)) (a (.virt 23732)) (a (.virt 23733)) ∧
    IsEqual (a (.virt 17751)) (a (.virt 23710)) (a (.virt 23734)) (a (.virt 23735)) ∧
    a (.wire 4 3) = band (a (.virt 23728)) (a (.virt 23730)) ∧
    a (.wire 4 7) = band (a (.virt 23732)) (a (.virt 23734)) ∧
    a (.wire 4 11) = band (a (.wire 4 3)) (a (.wire 4 7)) ∧
    IsEqual (a (.virt 23666)) (a (.virt 23710)) (a (.virt 23736)) (a (.virt 23737)) ∧
    IsEqual (a (.virt 23667)) (a (.virt 23710)) (a (.virt 23738)) (a (.virt 23739)) ∧
    IsEqual (a (.virt 23668)) (a (.virt 23710)) (a (.virt 23740)) (a (.virt 23741)) ∧
    IsEqual (a (.virt 23669)) (a (.virt 23710)) (a (.virt 23742)) (a (.virt 23743)) ∧
    a (.wire 4 47) = band (a (.virt 23736)) (a (.virt 23738)) ∧
    a (.wire 4 51) = band (a (.virt 23740)) (a (.virt 23742)) ∧
    a (.wire 4 55) = band (a (.wire 4 47)) (a (.wire 4 51)) ∧
    a (.wire 5 11) = bnot (a (.wire 1 43)) ∧
    a (.virt 23709) = bnot (a (.virt 23710)) ∧
    a (.wire 5 11) = band (a (.wire 5 11)) (a (.virt 23709)) ∧
    a (.wire 5 15) = bselect (a (.wire 5 11)) (a (.virt 5912)) (a (.virt 23710))) ∧
    (a (.wire 5 19) = bselect (a (.wire 5 11)) (a (.virt 5913)) (a (.virt 23710)) ∧
    a (.wire 5 23) = bselect (a (.wire 5 11)) (a (.virt 5914)) (a (.virt 23710)) ∧
    a (.wire 5 27) = bselect (a (.wire 5 11)) (a (.virt 5915)) (a (.virt 23710)) ∧
    a (.wire 5 31) = bselect (a (.wire 5 11)) (a (.virt 5916)) (a (.virt 23710)) ∧
    a (.wire 5 11) = bor (a (.virt 23710)) (a (.wire 5 11)) ∧
    a (.wire 5 35) = bnot (a (.wire 2 27)) ∧
    a (.wire 5 39) = bnot (a (.wire 5 11)) ∧
    a (.wire 4 59) = band (a (.wire 5 35)) (a (.wire 5 39)) ∧
    a (.wire 5 47) = bselect (a (.wire 4 59)) (a (.virt 11830)) (a (.wire 5 15)) ∧
    a (.wire 5 55) = bselect (a (.wire 4 59)) (a (.virt 11831)) (a (.wire 5 19)) ∧
    a (.wire 6 3) = bselect (a (.wire 4 59)) (a (.virt 11832)) (a (.wire 5 23)) ∧
    a (.wire 6 11) = bselect (a (.wire 4 59)) (a (.virt 11833)) (a (.wire 5 27)) ∧
    a (.wire 6 19) = bselect (a (.wire 4 59)) (a (.virt 11834)) (a (.wire 5 31)) ∧
    a (.wire 8 3) = bor (a (.wire 5 11)) (a (.wire 5 35)) ∧
    a (.wire 6 23) = bnot (a (.wire 4 11)) ∧
    a (.wire 6 27) = bnot (a (.wire 8 3)) ∧
    a (.wire 9 3) = band (a (.wire 6 23)) (a (.wire 6 27)) ∧
    a (.wire 6 35) = bselect (a (.wire 9 3)) (a (.virt 17748)) (a (.wire 5 47)) ∧
    a (.wire 6 43) = bselect (a (.wire 9 3)) (a (.virt 17749)) (a (.wire 5 55)) ∧
    a (.wire 6 51) = bselect (a (.wire 9 3)) (a (.virt 17750)) (a (.wire 6 3)) ∧
    a (.wire 6 59) = bselect (a (.wire 9 3)) (a (.virt 17751)) (a (.wire 6 11)) ∧
    a (.wire 10 7) = bselect (a (.wire 9 3)) (a (.virt 17752)) (a (.wire 6 19)) ∧
    a (.wire 8 7) = bor (a (.wire 8 3)) (a (.wire 6 23)) ∧
    a (.wire 10 11) = bnot (a (.wire 4 55)) ∧
    a (.wire 10 15) = bnot (a (.wire 8 7)) ∧
    a (.wire 9 7) = band (a (.wire 10 11)) (a (.wire 10 15)) ∧
    a (.wire 10 23) = bselect (a (.wire 9 7)) (a (.virt 23666)) (a (.wire 6 35)) ∧
    a (.wire 10 31) = bselect (a (.wire 9 7)) (a (.virt 23667)) (a (.wire 6 43)) ∧
    a (.wire 10 39) = bselect (a (.wire 9 7)) (a (.virt 23668)) (a (.wire 6 51)) ∧
    a (.wire 10 47) = bselect (a (.wire 9 7)) (a (.virt 23669)) (a (.wire 6 59)) ∧
    a (.wire 10 55) = bselect (a (.wire 9 7)) (a (.virt 23670)) (a (.wire 10 7)) ∧
    a (.wire 8 11) = bor (a (.wire 8 7)) (a (.wire 10 11))) ∧
    (IsEqual (a (.virt 5912)) (a (.wire 10 23)) (a (.virt 23744)) (a (.virt 23745)) ∧
    IsEqual (a (.virt 5913)) (a (.wire 10 31)) (a (.virt 23746)) (a (.virt 23747)) ∧
    IsEqual (a (.virt 5914)) (a (.wire 10 39)) (a (.virt 23748)) (a (.virt 23749)) ∧
    IsEqual (a (.virt 5915)) (a (.wire 10 47)) (a (.virt 23750)) (a (.virt 23751)) ∧
    a (.wire 9 43) = band (a (.virt 23744)) (a (.virt 23746)) ∧
    a (.wire 9 47) = band (a (.virt 23748)) (a (.virt 23750)) ∧
    a (.wire 9 51) = band (a (.wire 9 43)) (a (.wire 9 47)) ∧
    a (.wire 8 15) = bor (a (.wire 1 43)) (a (.wire 9 51)) ∧
    a (.wire 8 15) = a (.virt 23709) ∧
    a (.virt 5907) = a (.virt 5907) ∧
    IsEqual (a (.virt 11830)) (a (.wire 10 23)) (a (.virt 23752)) (a (.virt 23753)) ∧
    IsEqual (a (.virt 11831)) (a (.wire 10 31)) (a (.virt 23754)) (a (.virt 23755)) ∧
    IsEqual (a (.virt 11832)) (a (.wire 10 39)) (a (.virt 23756)) (a (.virt 23757)) ∧
    IsEqual (a (.virt 11833)) (a (.wire 10 47)) (a (.virt 23758)) (a (.virt 23759)) ∧
    a (.wire 13 27) = band (a (.virt 23752)) (a (.virt 23754)) ∧
    a (.wire 13 31) = band (a (.virt 23756)) (a (.virt 23758)) ∧
    a (.wire 13 35) = band (a (.wire 13 27)) (a (.wire 13 31)) ∧
    a (.wire 8 19) = bor (a (.wire 2 27)) (a (.wire 13 35)) ∧
    a (.wire 8 19) = a (.virt 23709) ∧
    a (.virt 11825) = a (.virt 5907) ∧
    IsEqual (a (.virt 17748)) (a (.wire 10 23)) (a (.virt 23760)) (a (.virt 23761)) ∧
    IsEqual (a (.virt 17749)) (a (.wire 10 31)) (a (.virt 23762)) (a (.virt 23763)) ∧
    IsEqual (a (.virt 17750)) (a (.wire 10 39)) (a (.virt 23764)) (a (.virt 23765)) ∧
    IsEqual (a (.virt 17751)) (a (.wire 10 47)) (a (.virt 23766)) (a (.virt 23767)) ∧
    a (.wire 15 11) = band (a (.virt 23760)) (a (.virt 23762)) ∧
    a (.wire 15 15) = band (a (.virt 23764)) (a (.virt 23766)) ∧
    a (.wire 15 19) = band (a (.wire 15 11)) (a (.wire 15 15)) ∧
    a (.wire 8 23) = bor (a (.wire 4 11)) (a (.wire 15 19)) ∧
    a (.wire 8 23) = a (.virt 23709) ∧
    a (.virt 17743) = a (.virt 5907) ∧
    IsEqual (a (.virt 23666)) (a (.wire 10 23)) (a (.virt 23768)) (a (.virt 23769)) ∧
    IsEqual (a (.virt 23667)) (a (.wire 10 31)) (a (.virt 23770)) (a (.virt 23771))) ∧
    (IsEqual (a (.virt 23668)) (a (.wire 10 39)) (a (.virt 23772)) (a (.virt 23773)) ∧
    IsEqual (a (.virt 23669)) (a (.wire 10 47)) (a (.virt 23774)) (a (.virt 23775)) ∧
    a (.wire 15 55) = band (a (.virt 23768)) (a (.virt 23770)) ∧
    a (.wire 15 59) = band (a (.virt 23772)) (a (.virt 23774)) ∧
    a (.wire 17 3) = band (a (.wire 15 55)) (a (.wire 15 59)) ∧
    a (.wire 8 27) = bor (a (.wire 4 55)) (a (.wire 17 3)) ∧
    a (.wire 8 27) = a (.virt 23709) ∧
    a (.virt 23661) = a (.virt 5907) ∧
    a (.wire 16 15) = bselect (a (.wire 1 43)) (a (.virt 23710)) (a (.virt 5917)) ∧
    a (.wire 16 15) = a (.virt 23710) + a (.wire 16 15) ∧
    a (.wire 16 23) = bselect (a (.wire 2 27)) (a (.virt 23710)) (a (.virt 11835)) ∧
    a (.wire 8 31) = a (.wire 16 15) + a (.wire 16 23) ∧
    a (.wire 16 31) = bselect (a (.wire 4 11)) (a (.virt 23710)) (a (.virt 17753)) ∧
    a (.wire 8 35) = a (.wire 8 31) + a (.wire 16 31) ∧
    a (.wire 16 39) = bselect (a (.wire 4 55)) (a (.virt 23710)) (a (.virt 23671)) ∧
    a (.wire 8 39) = a (.wire 8 35) + a (.wire 16 39) ∧
    rangeCheck (a (.virt 23677)) 32 ∧
    a (.virt 23677) = a (.virt 23710) + a (.virt 23677) ∧
    rangeCheck (a (.virt 23682)) 32 ∧
    a (.wire 8 43) = a (.virt 23677) + a (.virt 23682) ∧
    rangeCheck (a (.virt 23687)) 32 ∧
    a (.wire 8 47) = a (.wire 8 43) + a (.virt 23687) ∧
    rangeCheck (a (.virt 23692)) 32 ∧
    a (.wire 8 51) = a (.wire 8 47) + a (.virt 23692) ∧
    rangeCheck (a (.virt 23672)) 14 ∧
    a (.wire 16 43) = a (.virt 23777) - a (.virt 23672) ∧
    rangeCheck (a (.wire 16 43)) 14 ∧
    a (.wire 17 7) = a (.wire 8 51) * a (.virt 23777) ∧
    a (.wire 17 11) = a (.wire 8 39) * a (.wire 16 43) ∧
    a (.wire 16 47) = a (.wire 17 11) - a (.wire 17 7) ∧
    rangeCheck (a (.wire 16 47)) 52 ∧
    a (.wire 5 11) = bnot (a (.wire 1 43))) ∧
    (a (.wire 5 35) = bnot (a (.wire 2 27)) ∧
    a (.wire 17 15) = band (a (.wire 5 11)) (a (.wire 5 35)) ∧
    IsEqual (a (.virt 5908)) (a (.virt 11826)) (a (.virt 23778)) (a (.virt 23779)) ∧
    IsEqual (a (.virt 5909)) (a (.virt 11827)) (a (.virt 23780)) (a (.virt 23781)) ∧
    IsEqual (a (.virt 5910)) (a (.virt 11828)) (a (.virt 23782)) (a (.virt 23783)) ∧
    IsEqual (a (.virt 5911)) (a (.virt 11829)) (a (.virt 23784)) (a (.virt 23785)) ∧
    a (.wire 17 51) = band (a (.virt 23778)) (a (.virt 23780)) ∧
    a (.wire 17 55) = band (a (.virt 23782)) (a (.virt 23784)) ∧
    a (.wire 17 59) = band (a (.wire 17 51)) (a (.wire 17 55)) ∧
    a (.wire 26 3) = band (a (.wire 17 15)) (a (.wire 17 59)) ∧
    a (.wire 26 3) = a (.virt 23710) ∧
    a (.wire 6 23) = bnot (a (.wire 4 11)) ∧
    a (.wire 26 7) = band (a (.wire 5 11)) (a (.wire 6 23)) ∧
    IsEqual (a (.virt 5908)) (a (.virt 17744)) (a (.virt 23786)) (a (.virt 23787)) ∧
    IsEqual (a (.virt 5909)) (a (.virt 17745)) (a (.virt 23788)) (a (.virt 23789)) ∧
    IsEqual (a (.virt 5910)) (a (.virt 17746)) (a (.virt 23790)) (a (.virt 23791)) ∧
    IsEqual (a (.virt 5911)) (a (.virt 17747)) (a (.virt 23792)) (a (.virt 23793)) ∧
    a (.wire 26 43) = band (a (.virt 23786)) (a (.virt 23788)) ∧
    a (.wire 26 47) = band (a (.virt 23790)) (a (.virt 23792)) ∧
    a (.wire 26 51) = band (a (.wire 26 43)) (a (.wire 26 47)) ∧
    a (.wire 26 55) = band (a (.wire 26 7)) (a (.wire 26 51)) ∧
    a (.wire 26 55) = a (.virt 23710) ∧
    a (.wire 10 11) = bnot (a (.wire 4 55)) ∧
    a (.wire 26 59) = band (a (.wire 5 11)) (a (.wire 10 11)) ∧
    IsEqual (a (.virt 5908)) (a (.virt 23662)) (a (.virt 23794)) (a (.virt 23795)) ∧
    IsEqual (a (.virt 5909)) (a (.virt 23663)) (a (.virt 23796)) (a (.virt 23797)) ∧
    IsEqual (a (.virt 5910)) (a (.virt 23664)) (a (.virt 23798)) (a (.virt 23799)) ∧
    IsEqual (a (.virt 5911)) (a (.virt 23665)) (a (.virt 23800)) (a (.virt 23801)) ∧
    a (.wire 28 35) = band (a (.virt 23794)) (a (.virt 23796)) ∧
    a (.wire 28 39) = band (a (.virt 23798)) (a (.virt 23800)) ∧
    a (.wire 28 43) = band (a (.wire 28 35)) (a (.wire 28 39)) ∧
    a (.wire 28 47) = band (a (.wire 26 59)) (a (.wire 28 43))) ∧
    (a (.wire 28 47) = a (.virt 23710) ∧
    a (.wire 5 35) = bnot (a (.wire 2 27)) ∧
    a (.wire 6 23) = bnot (a (.wire 4 11)) ∧
    a (.wire 28 51) = band (a (.wire 5 35)) (a (.wire 6 23)) ∧
    IsEqual (a (.virt 11826)) (a (.virt 17744)) (a (.virt 23802)) (a (.virt 23803)) ∧
    IsEqual (a (.virt 11827)) (a (.virt 17745)) (a (.virt 23804)) (a (.virt 23805)) ∧
    IsEqual (a (.virt 11828)) (a (.virt 17746)) (a (.virt 23806)) (a (.virt 23807)) ∧
    IsEqual (a (.virt 11829)) (a (.virt 17747)) (a (.virt 23808)) (a (.virt 23809)) ∧
    a (.wire 30 27) = band (a (.virt 23802)) (a (.virt 23804)) ∧
    a (.wire 30 31) = band (a (.virt 23806)) (a (.virt 23808)) ∧
    a (.wire 30 35) = band (a (.wire 30 27)) (a (.wire 30 31)) ∧
    a (.wire 30 39) = band (a (.wire 28 51)) (a (.wire 30 35)) ∧
    a (.wire 30 39) = a (.virt 23710) ∧
    a (.wire 10 11) = bnot (a (.wire 4 55)) ∧
    a (.wire 30 43) = band (a (.wire 5 35)) (a (.wire 10 11)) ∧
    IsEqual (a (.virt 11826)) (a (.virt 23662)) (a (.virt 23810)) (a (.virt 23811)) ∧
    IsEqual (a (.virt 11827)) (a (.virt 23663)) (a (.virt 23812)) (a (.virt 23813)) ∧
    IsEqual (a (.virt 11828)) (a (.virt 23664)) (a (.virt 23814)) (a (.virt 23815)) ∧
    IsEqual (a (.virt 11829)) (a (.virt 23665)) (a (.virt 23816)) (a (.virt 23817)) ∧
    a (.wire 32 19) = band (a (.virt 23810)) (a (.virt 23812)) ∧
    a (.wire 32 23) = band (a (.virt 23814)) (a (.virt 23816)) ∧
    a (.wire 32 27) = band (a (.wire 32 19)) (a (.wire 32 23)) ∧
    a (.wire 32 31) = band (a (.wire 30 43)) (a (.wire 32 27)) ∧
    a (.wire 32 31) = a (.virt 23710) ∧
    a (.wire 6 23) = bnot (a (.wire 4 11)) ∧
    a (.wire 10 11) = bnot (a (.wire 4 55)) ∧
    a (.wire 32 35) = band (a (.wire 6 23)) (a (.wire 10 11)) ∧
    IsEqual (a (.virt 17744)) (a (.virt 23662)) (a (.virt 23818)) (a (.virt 23819)) ∧
    IsEqual (a (.virt 17745)) (a (.virt 23663)) (a (.virt 23820)) (a (.virt 23821)) ∧
    IsEqual (a (.virt 17746)) (a (.virt 23664)) (a (.virt 23822)) (a (.virt 23823)) ∧
    IsEqual (a (.virt 17747)) (a (.virt 23665)) (a (.virt 23824)) (a (.virt 23825)) ∧
    a (.wire 34 11) = band (a (.virt 23818)) (a (.virt 23820))) ∧
    (a (.wire 34 15) = band (a (.virt 23822)) (a (.virt 23824)) ∧
    a (.wire 34 19) = band (a (.wire 34 11)) (a (.wire 34 15)) ∧
    a (.wire 34 23) = band (a (.wire 32 35)) (a (.wire 34 19)) ∧
    a (.wire 34 23) = a (.virt 23710) ∧
    a (.wire 10 11) = bnot (a (.wire 4 55)) ∧
    (a (.wire 35 12) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 0 ∧ a (.wire 35 13) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 1 ∧ a (.wire 35 14) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 2 ∧ a (.wire 35 15) = spongeHash perm [a (.virt 23693), a (.virt 23694), a (.virt 23695), a (.virt 23696)] 3) ∧
    (a (.wire 36 12) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 0 ∧ a (.wire 36 13) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 1 ∧ a (.wire 36 14) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 2 ∧ a (.wire 36 15) = spongeHash perm [a (.wire 35 12), a (.wire 35 13), a (.wire 35 14), a (.wire 35 15)] 3) ∧
    a (.wire 33 43) = bselect (a (.wire 1 43)) (a (.wire 36 12)) (a (.virt 5908)) ∧
    a (.wire 33 51) = bselect (a (.wire 1 43)) (a (.wire 36 13)) (a (.virt 5909)) ∧
    a (.wire 33 59) = bselect (a (.wire 1 43)) (a (.wire 36 14)) (a (.virt 5910)) ∧
    a (.wire 37 7) = bselect (a (.wire 1 43)) (a (.wire 36 15)) (a (.virt 5911)) ∧
    (a (.wire 38 12) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 0 ∧ a (.wire 38 13) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 1 ∧ a (.wire 38 14) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 2 ∧ a (.wire 38 15) = spongeHash perm [a (.virt 23697), a (.virt 23698), a (.virt 23699), a (.virt 23700)] 3) ∧
    (a (.wire 39 12) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 0 ∧ a (.wire 39 13) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 1 ∧ a (.wire 39 14) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 2 ∧ a (.wire 39 15) = spongeHash perm [a (.wire 38 12), a (.wire 38 13), a (.wire 38 14), a (.wire 38 15)] 3) ∧
    a (.wire 37 15) = bselect (a (.wire 2 27)) (a (.wire 39 12)) (a (.virt 11826)) ∧
    a (.wire 37 23) = bselect (a (.wire 2 27)) (a (.wire 39 13)) (a (.virt 11827)) ∧
    a (.wire 37 31) = bselect (a (.wire 2 27)) (a (.wire 39 14)) (a (.virt 11828)) ∧
    a (.wire 37 39) = bselect (a (.wire 2 27)) (a (.wire 39 15)) (a (.virt 11829)) ∧
    (a (.wire 40 12) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 0 ∧ a (.wire 40 13) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 1 ∧ a (.wire 40 14) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 2 ∧ a (.wire 40 15) = spongeHash perm [a (.virt 23701), a (.virt 23702), a (.virt 23703), a (.virt 23704)] 3) ∧
    (a (.wire 41 12) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 0 ∧ a (.wire 41 13) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 1 ∧ a (.wire 41 14) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 2 ∧ a (.wire 41 15) = spongeHash perm [a (.wire 40 12), a (.wire 40 13), a (.wire 40 14), a (.wire 40 15)] 3) ∧
    a (.wire 37 47) = bselect (a (.wire 4 11)) (a (.wire 41 12)) (a (.virt 17744)) ∧
    a (.wire 37 55) = bselect (a (.wire 4 11)) (a (.wire 41 13)) (a (.virt 17745)) ∧
    a (.wire 42 3) = bselect (a (.wire 4 11)) (a (.wire 41 14)) (a (.virt 17746)) ∧
    a (.wire 42 11) = bselect (a (.wire 4 11)) (a (.wire 41 15)) (a (.virt 17747)) ∧
    (a (.wire 43 12) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 0 ∧ a (.wire 43 13) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 1 ∧ a (.wire 43 14) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 2 ∧ a (.wire 43 15) = spongeHash perm [a (.virt 23705), a (.virt 23706), a (.virt 23707), a (.virt 23708)] 3) ∧
    (a (.wire 44 12) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 0 ∧ a (.wire 44 13) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 1 ∧ a (.wire 44 14) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 2 ∧ a (.wire 44 15) = spongeHash perm [a (.wire 43 12), a (.wire 43 13), a (.wire 43 14), a (.wire 43 15)] 3) ∧
    a (.wire 42 19) = bselect (a (.wire 4 55)) (a (.wire 44 12)) (a (.virt 23662)) ∧
    a (.wire 42 27) = bselect (a (.wire 4 55)) (a (.wire 44 13)) (a (.virt 23663)) ∧
    a (.wire 42 35) = bselect (a (.wire 4 55)) (a (.wire 44 14)) (a (.virt 23664)) ∧
    a (.wire 42 43) = bselect (a (.wire 4 55)) (a (.wire 44 15)) (a (.virt 23665))) :=
  ⟨⟨privateBatchWrapper4_f0 a h, privateBatchWrapper4_f1 a h, privateBatchWrapper4_f2 a h, privateBatchWrapper4_f3 a h, privateBatchWrapper4_f4 a h, privateBatchWrapper4_f5 a h, privateBatchWrapper4_f6 a h, privateBatchWrapper4_f7 a h, privateBatchWrapper4_f8 a h, privateBatchWrapper4_f9 a h, privateBatchWrapper4_f10 a h, privateBatchWrapper4_f11 a h, privateBatchWrapper4_f12 a h, privateBatchWrapper4_f13 a h, privateBatchWrapper4_f14 a h, privateBatchWrapper4_f15 a h, privateBatchWrapper4_f16 a h, privateBatchWrapper4_f17 a h, privateBatchWrapper4_f18 a h, privateBatchWrapper4_f19 a h, privateBatchWrapper4_f20 a h, privateBatchWrapper4_f21 a h, privateBatchWrapper4_f22 a h, privateBatchWrapper4_f23 a h, privateBatchWrapper4_f24 a h, privateBatchWrapper4_f25 a h, privateBatchWrapper4_f26 a h, privateBatchWrapper4_f27 a h, privateBatchWrapper4_f28 a h, privateBatchWrapper4_f29 a h, privateBatchWrapper4_f30 a h, privateBatchWrapper4_f31 a h⟩, ⟨privateBatchWrapper4_f32 a h, privateBatchWrapper4_f33 a h, privateBatchWrapper4_f34 a h, privateBatchWrapper4_f35 a h, privateBatchWrapper4_f36 a h, privateBatchWrapper4_f37 a h, privateBatchWrapper4_f38 a h, privateBatchWrapper4_f39 a h, privateBatchWrapper4_f40 a h, privateBatchWrapper4_f41 a h, privateBatchWrapper4_f42 a h, privateBatchWrapper4_f43 a h, privateBatchWrapper4_f44 a h, privateBatchWrapper4_f45 a h, privateBatchWrapper4_f46 a h, privateBatchWrapper4_f47 a h, privateBatchWrapper4_f48 a h, privateBatchWrapper4_f49 a h, privateBatchWrapper4_f50 a h, privateBatchWrapper4_f51 a h, privateBatchWrapper4_f52 a h, privateBatchWrapper4_f53 a h, privateBatchWrapper4_f54 a h, privateBatchWrapper4_f55 a h, privateBatchWrapper4_f56 a h, privateBatchWrapper4_f57 a h, privateBatchWrapper4_f58 a h, privateBatchWrapper4_f59 a h, privateBatchWrapper4_f60 a h, privateBatchWrapper4_f61 a h, privateBatchWrapper4_f62 a h, privateBatchWrapper4_f63 a h⟩, ⟨privateBatchWrapper4_f64 a h, privateBatchWrapper4_f65 a h, privateBatchWrapper4_f66 a h, privateBatchWrapper4_f67 a h, privateBatchWrapper4_f68 a h, privateBatchWrapper4_f69 a h, privateBatchWrapper4_f70 a h, privateBatchWrapper4_f71 a h, privateBatchWrapper4_f72 a h, privateBatchWrapper4_f73 a h, privateBatchWrapper4_f74 a h, privateBatchWrapper4_f75 a h, privateBatchWrapper4_f76 a h, privateBatchWrapper4_f77 a h, privateBatchWrapper4_f78 a h, privateBatchWrapper4_f79 a h, privateBatchWrapper4_f80 a h, privateBatchWrapper4_f81 a h, privateBatchWrapper4_f82 a h, privateBatchWrapper4_f83 a h, privateBatchWrapper4_f84 a h, privateBatchWrapper4_f85 a h, privateBatchWrapper4_f86 a h, privateBatchWrapper4_f87 a h, privateBatchWrapper4_f88 a h, privateBatchWrapper4_f89 a h, privateBatchWrapper4_f90 a h, privateBatchWrapper4_f91 a h, privateBatchWrapper4_f92 a h, privateBatchWrapper4_f93 a h, privateBatchWrapper4_f94 a h, privateBatchWrapper4_f95 a h⟩, ⟨privateBatchWrapper4_f96 a h, privateBatchWrapper4_f97 a h, privateBatchWrapper4_f98 a h, privateBatchWrapper4_f99 a h, privateBatchWrapper4_f100 a h, privateBatchWrapper4_f101 a h, privateBatchWrapper4_f102 a h, privateBatchWrapper4_f103 a h, privateBatchWrapper4_f104 a h, privateBatchWrapper4_f105 a h, privateBatchWrapper4_f106 a h, privateBatchWrapper4_f107 a h, privateBatchWrapper4_f108 a h, privateBatchWrapper4_f109 a h, privateBatchWrapper4_f110 a h, privateBatchWrapper4_f111 a h, privateBatchWrapper4_f112 a h, privateBatchWrapper4_f113 a h, privateBatchWrapper4_f114 a h, privateBatchWrapper4_f115 a h, privateBatchWrapper4_f116 a h, privateBatchWrapper4_f117 a h, privateBatchWrapper4_f118 a h, privateBatchWrapper4_f119 a h, privateBatchWrapper4_f120 a h, privateBatchWrapper4_f121 a h, privateBatchWrapper4_f122 a h, privateBatchWrapper4_f123 a h, privateBatchWrapper4_f124 a h, privateBatchWrapper4_f125 a h, privateBatchWrapper4_f126 a h, privateBatchWrapper4_f127 a h⟩, ⟨privateBatchWrapper4_f128 a h, privateBatchWrapper4_f129 a h, privateBatchWrapper4_f130 a h, privateBatchWrapper4_f131 a h, privateBatchWrapper4_f132 a h, privateBatchWrapper4_f133 a h, privateBatchWrapper4_f134 a h, privateBatchWrapper4_f135 a h, privateBatchWrapper4_f136 a h, privateBatchWrapper4_f137 a h, privateBatchWrapper4_f138 a h, privateBatchWrapper4_f139 a h, privateBatchWrapper4_f140 a h, privateBatchWrapper4_f141 a h, privateBatchWrapper4_f142 a h, privateBatchWrapper4_f143 a h, privateBatchWrapper4_f144 a h, privateBatchWrapper4_f145 a h, privateBatchWrapper4_f146 a h, privateBatchWrapper4_f147 a h, privateBatchWrapper4_f148 a h, privateBatchWrapper4_f149 a h, privateBatchWrapper4_f150 a h, privateBatchWrapper4_f151 a h, privateBatchWrapper4_f152 a h, privateBatchWrapper4_f153 a h, privateBatchWrapper4_f154 a h, privateBatchWrapper4_f155 a h, privateBatchWrapper4_f156 a h, privateBatchWrapper4_f157 a h, privateBatchWrapper4_f158 a h, privateBatchWrapper4_f159 a h⟩, ⟨privateBatchWrapper4_f160 a h, privateBatchWrapper4_f161 a h, privateBatchWrapper4_f162 a h, privateBatchWrapper4_f163 a h, privateBatchWrapper4_f164 a h, privateBatchWrapper4_f165 a h, privateBatchWrapper4_f166 a h, privateBatchWrapper4_f167 a h, privateBatchWrapper4_f168 a h, privateBatchWrapper4_f169 a h, privateBatchWrapper4_f170 a h, privateBatchWrapper4_f171 a h, privateBatchWrapper4_f172 a h, privateBatchWrapper4_f173 a h, privateBatchWrapper4_f174 a h, privateBatchWrapper4_f175 a h, privateBatchWrapper4_f176 a h, privateBatchWrapper4_f177 a h, privateBatchWrapper4_f178 a h, privateBatchWrapper4_f179 a h, privateBatchWrapper4_f180 a h, privateBatchWrapper4_f181 a h, privateBatchWrapper4_f182 a h, privateBatchWrapper4_f183 a h, privateBatchWrapper4_f184 a h, privateBatchWrapper4_f185 a h, privateBatchWrapper4_f186 a h, privateBatchWrapper4_f187 a h, privateBatchWrapper4_f188 a h, privateBatchWrapper4_f189 a h, privateBatchWrapper4_f190 a h, privateBatchWrapper4_f191 a h⟩, ⟨privateBatchWrapper4_f192 a h, privateBatchWrapper4_f193 a h, privateBatchWrapper4_f194 a h, privateBatchWrapper4_f195 a h, privateBatchWrapper4_f196 a h, privateBatchWrapper4_f197 perm a h hp, privateBatchWrapper4_f198 perm a h hp, privateBatchWrapper4_f199 a h, privateBatchWrapper4_f200 a h, privateBatchWrapper4_f201 a h, privateBatchWrapper4_f202 a h, privateBatchWrapper4_f203 perm a h hp, privateBatchWrapper4_f204 perm a h hp, privateBatchWrapper4_f205 a h, privateBatchWrapper4_f206 a h, privateBatchWrapper4_f207 a h, privateBatchWrapper4_f208 a h, privateBatchWrapper4_f209 perm a h hp, privateBatchWrapper4_f210 perm a h hp, privateBatchWrapper4_f211 a h, privateBatchWrapper4_f212 a h, privateBatchWrapper4_f213 a h, privateBatchWrapper4_f214 a h, privateBatchWrapper4_f215 perm a h hp, privateBatchWrapper4_f216 perm a h hp, privateBatchWrapper4_f217 a h, privateBatchWrapper4_f218 a h, privateBatchWrapper4_f219 a h, privateBatchWrapper4_f220 a h⟩⟩

end Plonky2Spec.Generated
