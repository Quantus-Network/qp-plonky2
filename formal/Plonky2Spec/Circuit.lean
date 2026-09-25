/-
  Logical-circuit semantics for the public-input decode exporter (PLAN.md Step 8).

  `CircuitBuilder::formal_snapshot()` (plonky2/src/plonk/formal_snapshot.rs, behind
  `formal-export`) dumps a builder's *logical* circuit before `build()`: gate rows with
  their constants, every `connect`, the constant-pinned targets and the registered
  public inputs. This module gives that dump a meaning: a `Constraint` per gate op /
  copy / pinned constant / `BaseSumGate<2>` row, a `Witness` assigning every target a
  field element, and `Circuit.Sat`. The generated `Circuit` for a wrapper is then
  *the* hypothesis `private_batch_val` should be discharged from, replacing the
  hand-stated `PrivateBatchConstraints` fields.

  Wire layouts (the exporter's row/column → op decoding):
  * `ArithmeticGate { num_ops }` (arithmetic_base.rs:49-60): op `i` uses columns
    `4i, 4i+1, 4i+2, 4i+3` = `m0, m1, addend, out`; the row constants are `(c0, c1)`.
  * `BaseSumGate { num_limbs } + Base: 2` (base_sum.rs:45-46): column `0` is the sum,
    columns `1..=num_limbs` the little-endian limbs.
  * `range_check(x, n)` (split_join.rs:25-50): one `BaseSumGate<2>` row with
    `num_limbs = min(⌊log₂(p−1)⌋, num_routed_wires − 1)` (59 in the wormhole config),
    `x` connected to the sum, and limbs `n..num_limbs` connected to the zero constant.

  The gadget lemmas below are the per-primitive rungs: each states the exact op
  sequence the builder emits for `select` / `not` / `and` / `or` / `is_equal` /
  `range_check` and proves the Wrapper-level predicate it denotes. They are what a
  generated decode proof will chain, one lemma application per gadget call.
-/
import Plonky2Spec.Arithmetic
import Plonky2Spec.RangeCheck

namespace Plonky2Spec

variable {p : ℕ} [Fact p.Prime]

/-- A builder target (`qp_plonky2_core::Target`). -/
inductive Target where
  | wire (row column : ℕ)
  | virt (index : ℕ)
  deriving DecidableEq, Repr

/-- A full witness: a value for every target (wires and virtual targets alike). -/
abbrev Witness (p : ℕ) := Target → ZMod p

/-- One logical constraint of a snapshot. -/
inductive Constraint (p : ℕ) where
  /-- One `ArithmeticGate` op: `out = c0·m0·m1 + c1·addend`. -/
  | arith (c0 c1 : ZMod p) (m0 m1 addend out : Target)
  /-- `connect(a, b)`. -/
  | copy (a b : Target)
  /-- `builder.constant(v)` pinned to `t`. -/
  | const (t : Target) (v : ZMod p)
  /-- One `BaseSumGate<2>` row. -/
  | baseSum2 (sum : Target) (limbs : List Target)

def Constraint.Sat (w : Witness p) : Constraint p → Prop
  | .arith c0 c1 m0 m1 addend out => ArithmeticConstraint c0 c1 (w m0) (w m1) (w addend) (w out)
  | .copy a b => w a = w b
  | .const t v => w t = v
  | .baseSum2 sum limbs => BaseSum 2 (w sum) (limbs.map w)

abbrev Circuit (p : ℕ) := List (Constraint p)

def Circuit.Sat (w : Witness p) (c : Circuit p) : Prop := ∀ k ∈ c, k.Sat w

theorem Circuit.Sat.of_append {w : Witness p} {c₁ c₂ : Circuit p}
    (h : Circuit.Sat w (c₁ ++ c₂)) : Circuit.Sat w c₁ ∧ Circuit.Sat w c₂ :=
  ⟨fun k hk => h k (List.mem_append_left _ hk), fun k hk => h k (List.mem_append_right _ hk)⟩

theorem Circuit.Sat.sub {w : Witness p} {c₁ c₂ : Circuit p}
    (hsub : c₁ ⊆ c₂) (h : Circuit.Sat w c₂) : Circuit.Sat w c₁ :=
  fun k hk => h k (hsub hk)

/-! ### Builder primitives as op sequences (arithmetic.rs, select.rs)

  `add`/`sub` route the pinned `one` constant through `m1` (arithmetic.rs:193-214);
  `mul` reuses `m0` as the (zero-weighted) addend (arithmetic.rs:217-220). -/

/-- `select(b, x, y)` (select.rs:33-36): `tmp = mul_sub(b, y, y)`, `out = mul_sub(b, x, tmp)`. -/
def selectGadget (b x y tmp out : Target) : Circuit p :=
  [.arith 1 (-1) b y y tmp, .arith 1 (-1) b x tmp out]

theorem selectGadget_spec {w : Witness p} {b x y tmp out : Target}
    (hs : Circuit.Sat w (selectGadget b x y tmp out)) :
    w out = bselect (w b) (w x) (w y) := by
  have h1 : ArithmeticConstraint 1 (-1) (w b) (w y) (w y) (w tmp) :=
    hs (.arith 1 (-1) b y y tmp) (by simp [selectGadget])
  have h2 : ArithmeticConstraint 1 (-1) (w b) (w x) (w tmp) (w out) :=
    hs (.arith 1 (-1) b x tmp out) (by simp [selectGadget])
  rw [mulSub_spec h2, mulSub_spec h1, bselect]; ring

/-- `not(b)` (arithmetic.rs:345-349): `out = sub(one, b)` with `one` the pinned `1`. -/
def notGadget (one b out : Target) : Circuit p :=
  [.const one 1, .arith 1 (-1) one one b out]

theorem notGadget_spec {w : Witness p} {one b out : Target}
    (hs : Circuit.Sat w (notGadget one b out)) : w out = bnot (w b) := by
  have h1 : w one = 1 := hs (.const one 1) (by simp [notGadget])
  have h2 : ArithmeticConstraint 1 (-1) (w one) (w one) (w b) (w out) :=
    hs (.arith 1 (-1) one one b out) (by simp [notGadget])
  rw [arithmetic_iff] at h2
  rw [h2, h1, bnot]; ring

/-- `and(b1, b2)` (arithmetic.rs:352-354): `out = mul(b1, b2)`. -/
def andGadget (b1 b2 out : Target) : Circuit p :=
  [.arith 1 0 b1 b2 b1 out]

theorem andGadget_spec {w : Witness p} {b1 b2 out : Target}
    (hs : Circuit.Sat w (andGadget b1 b2 out)) : w out = band (w b1) (w b2) := by
  have h : ArithmeticConstraint 1 0 (w b1) (w b2) (w b1) (w out) :=
    hs (.arith 1 0 b1 b2 b1 out) (by simp [andGadget])
  rw [mul_spec h, band]

/-- `or(b1, b2)` (arithmetic.rs:357-360): `r = arithmetic(-1, 1, b1, b2, b1)`,
    `out = add(r, b2)`. -/
def orGadget (one b1 b2 r out : Target) : Circuit p :=
  [.const one 1, .arith (-1) 1 b1 b2 b1 r, .arith 1 1 r one b2 out]

theorem orGadget_spec {w : Witness p} {one b1 b2 r out : Target}
    (hs : Circuit.Sat w (orGadget one b1 b2 r out)) : w out = bor (w b1) (w b2) := by
  have h0 : w one = 1 := hs (.const one 1) (by simp [orGadget])
  have h1 : ArithmeticConstraint (-1) 1 (w b1) (w b2) (w b1) (w r) :=
    hs (.arith (-1) 1 b1 b2 b1 r) (by simp [orGadget])
  have h2 : ArithmeticConstraint 1 1 (w r) (w one) (w b2) (w out) :=
    hs (.arith 1 1 r one b2 out) (by simp [orGadget])
  rw [arithmetic_iff] at h1 h2
  rw [h2, h1, h0, bor]; ring

/-- `is_equal(x, y)` (arithmetic.rs:370-388): `notEqual = not(equal)`, `diff = sub(x, y)`,
    `neCheck = mul(equal, diff)`, `dn = mul(diff, inv)`, `eqCheck = sub(dn, notEqual)`,
    and `neCheck`, `eqCheck` connected to zero. -/
def isEqualGadget (zero one x y equal inv diff neCheck dn notEqual eqCheck : Target) :
    Circuit p :=
  [.const zero 0, .arith 1 (-1) x one y diff, .arith 1 0 equal diff equal neCheck,
   .arith 1 0 diff inv diff dn, .arith 1 (-1) dn one notEqual eqCheck,
   .copy neCheck zero, .copy eqCheck zero] ++ notGadget one equal notEqual

theorem isEqualGadget_spec {w : Witness p}
    {zero one x y equal inv diff neCheck dn notEqual eqCheck : Target}
    (hs : Circuit.Sat w (isEqualGadget zero one x y equal inv diff neCheck dn notEqual eqCheck)) :
    IsEqual (w x) (w y) (w equal) (w inv) := by
  obtain ⟨h, hnot⟩ := hs.of_append
  have hone : w one = 1 := hnot (.const one 1) (by simp [notGadget])
  have hz : w zero = 0 := h (.const zero 0) (by simp)
  have hdiff : ArithmeticConstraint 1 (-1) (w x) (w one) (w y) (w diff) :=
    h (.arith 1 (-1) x one y diff) (by simp)
  have hne : ArithmeticConstraint 1 0 (w equal) (w diff) (w equal) (w neCheck) :=
    h (.arith 1 0 equal diff equal neCheck) (by simp)
  have hdn : ArithmeticConstraint 1 0 (w diff) (w inv) (w diff) (w dn) :=
    h (.arith 1 0 diff inv diff dn) (by simp)
  have heq : ArithmeticConstraint 1 (-1) (w dn) (w one) (w notEqual) (w eqCheck) :=
    h (.arith 1 (-1) dn one notEqual eqCheck) (by simp)
  have hne0 : w neCheck = w zero := h (.copy neCheck zero) (by simp)
  have heq0 : w eqCheck = w zero := h (.copy eqCheck zero) (by simp)
  have hnotEq := notGadget_spec hnot
  rw [hone] at hdiff heq
  have hdiff' : w diff = w x - w y := sub_spec hdiff
  have hne' : w neCheck = w equal * w diff := mul_spec hne
  have hdn' : w dn = w diff * w inv := mul_spec hdn
  have heq' : w eqCheck = w dn - w notEqual := sub_spec heq
  refine ⟨?_, ?_⟩
  · rw [← hdiff', ← hne', hne0, hz]
  · rw [← hdiff', ← hdn', ← bnot, ← hnotEq, ← heq', heq0, hz]

/-! ### `range_check(x, n)` as a `BaseSumGate<2>` row with a zero-pinned tail -/

theorem reconstructF_append (B : ℕ) (l₁ l₂ : List (ZMod p)) :
    reconstructF B (l₁ ++ l₂) =
      reconstructF B l₁ + (B : ZMod p) ^ l₁.length * reconstructF B l₂ := by
  induction l₁ with
  | nil => simp
  | cons d ds ih =>
    rw [List.cons_append, reconstructF_cons, reconstructF_cons, ih, List.length_cons, pow_succ]
    ring

theorem reconstructF_eq_zero_of_forall_zero (B : ℕ) {l : List (ZMod p)}
    (h : ∀ x ∈ l, x = 0) : reconstructF B l = 0 := by
  induction l with
  | nil => rfl
  | cons d ds ih =>
    rw [reconstructF_cons, h d (List.mem_cons_self ..),
      ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]
    simp

/-- A satisfied `BaseSum` whose limbs beyond `n` are all zero is a satisfied
    `BaseSum` on the first `n` limbs. -/
theorem baseSum_take {B n : ℕ} {sum : ZMod p} {l : List (ZMod p)}
    (h : BaseSum B sum l) (hz : ∀ x ∈ l.drop n, x = 0) : BaseSum B sum (l.take n) := by
  refine ⟨?_, fun x hx => h.range x (List.mem_of_mem_take hx)⟩
  have hr : reconstructF B (l.take n) = reconstructF B l := by
    conv_rhs => rw [← List.take_append_drop n l]
    rw [reconstructF_append, reconstructF_eq_zero_of_forall_zero B hz, mul_zero, add_zero]
  exact hr.trans h.recon

/-- `range_check(x, n)` as emitted (split_join.rs:25-50): a `BaseSumGate<2>` row over
    `limbs` (`n ≤ limbs.length`), `x` connected to its sum wire, and every limb from
    position `n` on connected to the zero constant. -/
def rangeCheckGadget (zero x sum : Target) (limbs : List Target) (n : ℕ) : Circuit p :=
  [.const zero 0, .copy x sum, .baseSum2 sum limbs] ++ (limbs.drop n).map (.copy · zero)

theorem rangeCheckGadget_spec {w : Witness p} {zero x sum : Target} {limbs : List Target} {n : ℕ}
    (hn : n ≤ limbs.length)
    (hs : Circuit.Sat w (rangeCheckGadget zero x sum limbs n)) : rangeCheck (w x) n := by
  obtain ⟨h, htail⟩ := hs.of_append
  have hz : w zero = 0 := h (.const zero 0) (by simp)
  have hx : w x = w sum := h (.copy x sum) (by simp)
  have hb : BaseSum 2 (w sum) (limbs.map w) := h (.baseSum2 sum limbs) (by simp)
  have hzero : ∀ v ∈ (limbs.map w).drop n, v = 0 := by
    intro v hv
    rw [← List.map_drop, List.mem_map] at hv
    obtain ⟨t, ht, rfl⟩ := hv
    have : w t = w zero := htail (.copy t zero) (List.mem_map.mpr ⟨t, ht, rfl⟩)
    rw [this, hz]
  refine ⟨(limbs.map w).take n, ?_, hx ▸ baseSum_take hb hzero⟩
  rw [List.length_take, List.length_map]
  exact Nat.min_eq_left hn

end Plonky2Spec
