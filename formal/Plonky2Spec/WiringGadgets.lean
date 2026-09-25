/-
  Step 8 — the gadget-trace layer between the exported wiring and the wrapper bridge.

  `Wiring.Satisfies` is positional: row `r`, op `i`, columns `4i..4i+3`. The bridges in
  `Plonky2Bridge` want gadget-shaped facts: "this target is `bselect` of those", "these two
  wires are an `IsEqual` witness", "this wire is `rangeCheck`ed to `n` bits". This module
  is the middle: a `Constraint` is one builder op / copy / pinned constant / `BaseSumGate<2>`
  row with *named* targets, a `Trace` is the list a gadget call emits, and each
  `*Gadget_spec` lemma states the exact op sequence the builder lays down for `select` /
  `not` / `and` / `or` / `is_equal` / `range_check` (select.rs, arithmetic.rs,
  split_join.rs) and proves the predicate it denotes. The `Satisfies.*` lemmas at the end
  lift each constraint kind out of an exported `Wiring.Circuit`, so a generated decode
  proof is: locate the ops (exporter-emitted indices), lift, apply the gadget lemma.

  `range_check(x, n)` (split_join.rs:25-50) is one `BaseSumGate<2>` row with
  `num_limbs = min(⌊log₂(p−1)⌋, num_routed_wires − 1)` (59 in the wormhole config, 63 in
  the standard one), `x` connected to the sum wire (column 0) and limbs `n..num_limbs`
  (columns `1+n..`) connected to the zero constant. `Wiring.GateKind` does not yet
  interpret `BaseSumGate`, so `.baseSum2` has no lift here (PLAN.md Step 8 finding 2).
-/
import Plonky2Spec.Wiring
import Plonky2Spec.Arithmetic
import Plonky2Spec.RangeCheck

namespace Plonky2Spec.Wiring

variable {p : ℕ} [Fact p.Prime]

/-- One logical constraint with named targets. -/
inductive Constraint (p : ℕ) where
  /-- One `ArithmeticGate` op: `out = c0·m0·m1 + c1·addend`. -/
  | arith (c0 c1 : ZMod p) (m0 m1 addend out : Target)
  /-- `connect(a, b)`. -/
  | copy (a b : Target)
  /-- `builder.constant(v)` pinned to `t`. -/
  | const (t : Target) (v : ZMod p)
  /-- One `BaseSumGate<2>` row: sum wire and little-endian limb wires. -/
  | baseSum2 (sum : Target) (limbs : List Target)

def Constraint.Sat (a : Assignment p) : Constraint p → Prop
  | .arith c0 c1 m0 m1 addend out => ArithmeticConstraint c0 c1 (a m0) (a m1) (a addend) (a out)
  | .copy x y => a x = a y
  | .const t v => a t = v
  | .baseSum2 sum limbs => BaseSum 2 (a sum) (limbs.map a)

/-- The constraints one gadget call emits. -/
abbrev Trace (p : ℕ) := List (Constraint p)

def Trace.Sat (a : Assignment p) (tr : Trace p) : Prop := ∀ k ∈ tr, k.Sat a

theorem Trace.Sat.of_append {a : Assignment p} {t₁ t₂ : Trace p}
    (h : Trace.Sat a (t₁ ++ t₂)) : Trace.Sat a t₁ ∧ Trace.Sat a t₂ :=
  ⟨fun k hk => h k (List.mem_append_left _ hk), fun k hk => h k (List.mem_append_right _ hk)⟩

theorem Trace.Sat.sub {a : Assignment p} {t₁ t₂ : Trace p}
    (hsub : t₁ ⊆ t₂) (h : Trace.Sat a t₂) : Trace.Sat a t₁ :=
  fun k hk => h k (hsub hk)

/-! ### Builder primitives as op sequences (arithmetic.rs, select.rs)

  `add`/`sub` route the pinned `one` constant through `m1` (arithmetic.rs:193-214);
  `mul` reuses `m0` as the (zero-weighted) addend (arithmetic.rs:217-220). -/

/-- `select(b, x, y)` (select.rs:33-36): `tmp = mul_sub(b, y, y)`, `out = mul_sub(b, x, tmp)`. -/
def selectGadget (b x y tmp out : Target) : Trace p :=
  [.arith 1 (-1) b y y tmp, .arith 1 (-1) b x tmp out]

theorem selectGadget_spec {a : Assignment p} {b x y tmp out : Target}
    (hs : Trace.Sat a (selectGadget b x y tmp out)) :
    a out = bselect (a b) (a x) (a y) := by
  have h1 : ArithmeticConstraint 1 (-1) (a b) (a y) (a y) (a tmp) :=
    hs (.arith 1 (-1) b y y tmp) (by simp [selectGadget])
  have h2 : ArithmeticConstraint 1 (-1) (a b) (a x) (a tmp) (a out) :=
    hs (.arith 1 (-1) b x tmp out) (by simp [selectGadget])
  rw [mulSub_spec h2, mulSub_spec h1, bselect]; ring

/-- `not(b)` (arithmetic.rs:345-349): `out = sub(one, b)` with `one` the pinned `1`. -/
def notGadget (one b out : Target) : Trace p :=
  [.const one 1, .arith 1 (-1) one one b out]

theorem notGadget_spec {a : Assignment p} {one b out : Target}
    (hs : Trace.Sat a (notGadget one b out)) : a out = bnot (a b) := by
  have h1 : a one = 1 := hs (.const one 1) (by simp [notGadget])
  have h2 : ArithmeticConstraint 1 (-1) (a one) (a one) (a b) (a out) :=
    hs (.arith 1 (-1) one one b out) (by simp [notGadget])
  rw [arithmetic_iff] at h2
  rw [h2, h1, bnot]; ring

/-- `and(b1, b2)` (arithmetic.rs:352-354): `out = mul(b1, b2)`. -/
def andGadget (b1 b2 out : Target) : Trace p :=
  [.arith 1 0 b1 b2 b1 out]

theorem andGadget_spec {a : Assignment p} {b1 b2 out : Target}
    (hs : Trace.Sat a (andGadget b1 b2 out)) : a out = band (a b1) (a b2) := by
  have h : ArithmeticConstraint 1 0 (a b1) (a b2) (a b1) (a out) :=
    hs (.arith 1 0 b1 b2 b1 out) (by simp [andGadget])
  rw [mul_spec h, band]

/-- `or(b1, b2)` (arithmetic.rs:357-360): `r = arithmetic(-1, 1, b1, b2, b1)`,
    `out = add(r, b2)`. -/
def orGadget (one b1 b2 r out : Target) : Trace p :=
  [.const one 1, .arith (-1) 1 b1 b2 b1 r, .arith 1 1 r one b2 out]

theorem orGadget_spec {a : Assignment p} {one b1 b2 r out : Target}
    (hs : Trace.Sat a (orGadget one b1 b2 r out)) : a out = bor (a b1) (a b2) := by
  have h0 : a one = 1 := hs (.const one 1) (by simp [orGadget])
  have h1 : ArithmeticConstraint (-1) 1 (a b1) (a b2) (a b1) (a r) :=
    hs (.arith (-1) 1 b1 b2 b1 r) (by simp [orGadget])
  have h2 : ArithmeticConstraint 1 1 (a r) (a one) (a b2) (a out) :=
    hs (.arith 1 1 r one b2 out) (by simp [orGadget])
  rw [arithmetic_iff] at h1 h2
  rw [h2, h1, h0, bor]; ring

/-- `is_equal(x, y)` (arithmetic.rs:370-388): `notEqual = not(equal)`, `diff = sub(x, y)`,
    `neCheck = mul(equal, diff)`, `dn = mul(diff, inv)`, `eqCheck = sub(dn, notEqual)`,
    and `neCheck`, `eqCheck` connected to zero. -/
def isEqualGadget (zero one x y equal inv diff neCheck dn notEqual eqCheck : Target) :
    Trace p :=
  [.const zero 0, .arith 1 (-1) x one y diff, .arith 1 0 equal diff equal neCheck,
   .arith 1 0 diff inv diff dn, .arith 1 (-1) dn one notEqual eqCheck,
   .copy neCheck zero, .copy eqCheck zero] ++ notGadget one equal notEqual

theorem isEqualGadget_spec {a : Assignment p}
    {zero one x y equal inv diff neCheck dn notEqual eqCheck : Target}
    (hs : Trace.Sat a (isEqualGadget zero one x y equal inv diff neCheck dn notEqual eqCheck)) :
    IsEqual (a x) (a y) (a equal) (a inv) := by
  obtain ⟨h, hnot⟩ := hs.of_append
  have hone : a one = 1 := hnot (.const one 1) (by simp [notGadget])
  have hz : a zero = 0 := h (.const zero 0) (by simp)
  have hdiff : ArithmeticConstraint 1 (-1) (a x) (a one) (a y) (a diff) :=
    h (.arith 1 (-1) x one y diff) (by simp)
  have hne : ArithmeticConstraint 1 0 (a equal) (a diff) (a equal) (a neCheck) :=
    h (.arith 1 0 equal diff equal neCheck) (by simp)
  have hdn : ArithmeticConstraint 1 0 (a diff) (a inv) (a diff) (a dn) :=
    h (.arith 1 0 diff inv diff dn) (by simp)
  have heq : ArithmeticConstraint 1 (-1) (a dn) (a one) (a notEqual) (a eqCheck) :=
    h (.arith 1 (-1) dn one notEqual eqCheck) (by simp)
  have hne0 : a neCheck = a zero := h (.copy neCheck zero) (by simp)
  have heq0 : a eqCheck = a zero := h (.copy eqCheck zero) (by simp)
  have hnotEq := notGadget_spec hnot
  rw [hone] at hdiff heq
  have hdiff' : a diff = a x - a y := sub_spec hdiff
  have hne' : a neCheck = a equal * a diff := mul_spec hne
  have hdn' : a dn = a diff * a inv := mul_spec hdn
  have heq' : a eqCheck = a dn - a notEqual := sub_spec heq
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
def rangeCheckGadget (zero x sum : Target) (limbs : List Target) (n : ℕ) : Trace p :=
  [.const zero 0, .copy x sum, .baseSum2 sum limbs] ++ (limbs.drop n).map (.copy · zero)

theorem rangeCheckGadget_spec {a : Assignment p} {zero x sum : Target} {limbs : List Target}
    {n : ℕ} (hn : n ≤ limbs.length)
    (hs : Trace.Sat a (rangeCheckGadget zero x sum limbs n)) : rangeCheck (a x) n := by
  obtain ⟨h, htail⟩ := hs.of_append
  have hz : a zero = 0 := h (.const zero 0) (by simp)
  have hx : a x = a sum := h (.copy x sum) (by simp)
  have hb : BaseSum 2 (a sum) (limbs.map a) := h (.baseSum2 sum limbs) (by simp)
  have hzero : ∀ v ∈ (limbs.map a).drop n, v = 0 := by
    intro v hv
    rw [← List.map_drop, List.mem_map] at hv
    obtain ⟨t, ht, rfl⟩ := hv
    have : a t = a zero := htail (.copy t zero) (List.mem_map.mpr ⟨t, ht, rfl⟩)
    rw [this, hz]
  refine ⟨(limbs.map a).take n, ?_, hx ▸ baseSum_take hb hzero⟩
  rw [List.length_take, List.length_map]
  exact Nat.min_eq_left hn

/-! ### Lifting constraints out of an exported `Wiring.Circuit` -/

omit [Fact p.Prime] in
theorem rowsSatisfied_get {a : Assignment p} :
    ∀ {start : ℕ} {rs : List (Row p)}, rowsSatisfied a start rs →
      ∀ {i : ℕ} {r : Row p}, rs[i]? = some r → rowConstraints a (start + i) r
  | _, [], _, _, _, h => by simp at h
  | start, r :: rs, ⟨hr, hrs⟩, 0, r', h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h; simpa using hr
  | start, _ :: rs, ⟨_, hrs⟩, i + 1, r', h => by
    have := rowsSatisfied_get hrs (i := i) (r := r') (by simpa using h)
    rwa [Nat.add_assoc, Nat.add_comm 1 i] at this

/-- Op `i` of an `ArithmeticGate` row of an exported circuit, as a named constraint on its
    four wires (arithmetic_base.rs:49-60). -/
theorem Satisfies.arith {c : Circuit p} {a : Assignment p} (h : Satisfies c a)
    {row n i : ℕ} {r : Row p} (hr : c.rows[row]? = some r) (hk : r.kind = .arithmetic n)
    (hi : i < n) :
    Constraint.Sat a (.arith (r.consts.getD 0 0) (r.consts.getD 1 0)
      (.wire row (4 * i)) (.wire row (4 * i + 1)) (.wire row (4 * i + 2)) (.wire row (4 * i + 3))) := by
  have hrow := rowsSatisfied_get h.1 hr
  rw [Nat.zero_add, rowConstraints, hk] at hrow
  have := (arithOp_iff a row i _ _).mp (hrow i hi)
  show ArithmeticConstraint _ _ _ _ _ _
  rw [arithmetic_iff, this]; ring

theorem Satisfies.copy {c : Circuit p} {a : Assignment p} (h : Satisfies c a)
    {x y : Target} (hm : (x, y) ∈ c.copies) : Constraint.Sat a (.copy x y) :=
  h.2.1 (x, y) hm

theorem Satisfies.const {c : Circuit p} {a : Assignment p} (h : Satisfies c a)
    {t : Target} {v : ZMod p} (hm : (t, v) ∈ c.constants) : Constraint.Sat a (.const t v) :=
  h.2.2 (t, v) hm

end Plonky2Spec.Wiring
