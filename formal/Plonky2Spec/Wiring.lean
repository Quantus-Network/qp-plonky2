/-
  Step 8 (spike) — a model of a `CircuitBuilder`'s **pre-`build` constraint system**:
  which gate sits on which row with which constants, which targets are `connect`ed,
  which virtual targets are constants, and the public-input registration order.

  This is the layer the per-gate models (`Arithmetic.lean`, `Generated/Gates.lean`) sit
  under and the wrapper bridge (`Plonky2Bridge`) sits over: `Plonky2Bridge` *assumes*
  decode hypotheses of the form "public input `k` is `bselect flag dnull real`"; a
  `Satisfies` proof over an exported `Circuit` *derives* them from the wiring the builder
  actually laid down (`Bridges/CircuitBridge.lean`).

  Fidelity. `CircuitBuilder::build` consumes exactly the data exported here: it adds a
  `ConstantGate` wire copied to each constant target (modeled as `a t = c` directly),
  turns the copy list into the permutation argument (`a x = a y`), and pads with no-ops.
  Gate kinds this model does not interpret (`.other`, `.publicInput`, `.noop`) contribute
  `True`; a theorem proved from `Satisfies` therefore holds a fortiori for the real system,
  which enforces *more*. Interpreted kinds: `ArithmeticGate`, via the extracted
  `arithmeticGate_c0` polynomial (one op) — `Bridges/CircuitBridge.lean` pins the 20-op
  row the standard config places to this per-op reading; `BaseSumGate<2>`, via the hand
  model `BaseSum 2` of `RangeCheck.lean` (pinned to the extracted gate at two limbs in
  `Bridges/Bridge.lean`), on the sum wire (column 0) and the `numLimbs` limb wires
  (columns `1..=numLimbs`, base_sum.rs:45-46). `Poseidon2Gate` rows are classified but
  their meaning — output wires `12..24` are the permutation of input wires `0..12`
  (`Poseidon2.gate_sound_complete`) — is stated separately as `Poseidon2Rows perm`, so
  `Satisfies` stays independent of the permutation.
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fin.VecNotation
import Plonky2Spec.Generated.Gates
import Plonky2Spec.RangeCheck

namespace Plonky2Spec.Wiring

/-- A witness location: a wire `(row, column)` of the gate grid, or a builder virtual
    target (`Target::VirtualTarget { index }`). -/
inductive Target
  | wire (row col : ℕ)
  | virt (idx : ℕ)
  deriving DecidableEq, Repr

/-- Gate kinds, classified from `Gate::id()` by the exporter. -/
inductive GateKind
  | arithmetic (numOps : ℕ)
  | constant (numConsts : ℕ)
  | publicInput
  | noop
  /-- `BaseSumGate { num_limbs } + Base: 2`. -/
  | baseSum2 (numLimbs : ℕ)
  /-- `Poseidon2Gate<WIDTH=12>`. -/
  | poseidon2
  | other (id : String)
  deriving DecidableEq, Repr

/-- One placed gate row: its kind and the row's `local_constants`. -/
structure Row (p : ℕ) where
  kind : GateKind
  consts : List (ZMod p)

/-- A builder's pre-`build` constraint system. -/
structure Circuit (p : ℕ) where
  rows : List (Row p)
  copies : List (Target × Target)
  constants : List (Target × ZMod p)
  publicInputs : List Target

/-- A full witness: a value for every target. -/
abbrev Assignment (p : ℕ) := Target → ZMod p

variable {p : ℕ}

/-- Op `i` of an `ArithmeticGate` row reads wires `4i, 4i+1, 4i+2` and writes `4i+3`, with
    the row's two constants; the constraint is the extracted `arithmeticGate_c0`. -/
def arithOp (a : Assignment p) (row i : ℕ) (c0 c1 : ZMod p) : Prop :=
  Generated.arithmeticGate_c0 (a (.wire row (4 * i))) (a (.wire row (4 * i + 1)))
    (a (.wire row (4 * i + 2))) (a (.wire row (4 * i + 3))) c0 c1 = 0

theorem arithOp_iff (a : Assignment p) (row i : ℕ) (c0 c1 : ZMod p) :
    arithOp a row i c0 c1 ↔
      a (.wire row (4 * i + 3)) =
        a (.wire row (4 * i)) * a (.wire row (4 * i + 1)) * c0 + a (.wire row (4 * i + 2)) * c1 := by
  unfold arithOp Generated.arithmeticGate_c0
  exact sub_eq_zero

/-- The limb wires of a `BaseSumGate` row: columns `1..=n` (base_sum.rs:46). -/
def limbWires (row n : ℕ) : List Target := (List.range n).map fun i => .wire row (i + 1)

-- `BaseSum` lives in the prime-field section of `RangeCheck.lean`, so from here on the
-- system semantics carry the instance too.
variable [Fact p.Prime]

/-- The constraints a row imposes on an assignment. -/
def rowConstraints (a : Assignment p) (row : ℕ) (r : Row p) : Prop :=
  match r.kind with
  | .arithmetic n => ∀ i < n, arithOp a row i (r.consts.getD 0 0) (r.consts.getD 1 0)
  | .constant n => ∀ i < n, a (.wire row i) = r.consts.getD i 0
  | .publicInput => True
  | .noop => True
  | .baseSum2 n => BaseSum 2 (a (.wire row 0)) ((limbWires row n).map a)
  | .poseidon2 => True
  | .other _ => True

/-- Rows `row, row+1, …` all satisfied. -/
def rowsSatisfied (a : Assignment p) : ℕ → List (Row p) → Prop
  | _, [] => True
  | row, r :: rs => rowConstraints a row r ∧ rowsSatisfied a (row + 1) rs

/-- `a` satisfies the whole system: every row, every copy, every constant target. -/
def Satisfies (c : Circuit p) (a : Assignment p) : Prop :=
  rowsSatisfied a 0 c.rows ∧
  (∀ xy ∈ c.copies, a xy.1 = a xy.2) ∧
  (∀ tc ∈ c.constants, a tc.1 = tc.2)

/-- Input / output wires of a `Poseidon2Gate` row (poseidon2.rs:452-453). -/
def poseidon2In (a : Assignment p) (row : ℕ) : Fin 12 → ZMod p := fun i => a (.wire row i)
def poseidon2Out (a : Assignment p) (row : ℕ) : Fin 12 → ZMod p := fun i => a (.wire row (12 + i))

/-- Every `Poseidon2Gate` row computes `perm`: the meaning `gate_sound_complete` gives the
    118 gate constraints, stated on the row's wires. Conjoin with `Satisfies` for the full
    system; kept separate so `Satisfies` does not depend on the permutation. -/
def Poseidon2Rows (perm : (Fin 12 → ZMod p) → Fin 12 → ZMod p) (c : Circuit p)
    (a : Assignment p) : Prop :=
  ∀ row r, c.rows[row]? = some r → r.kind = .poseidon2 →
    poseidon2Out a row = perm (poseidon2In a row)

end Plonky2Spec.Wiring
