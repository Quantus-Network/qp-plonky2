/-
  T2 — the range primitive: `BaseSumGate<B>` and the `range_check` gadget.

  Rust references (qp-plonky2):
  * `BaseSumGate::eval_unfiltered`            `plonky2/src/gates/base_sum.rs:68-81`
  * `range_check` → `split_le` → BaseSumGate  `plonky2/src/gadgets/range_check.rs:21`,
                                              `plonky2/src/gadgets/split_join.rs:25-62`
  * `enforce_target_less_than_const`          `qp-zk-circuits/common/src/gadgets.rs:66`

  The gate decomposes a wire `sum` into little-endian base-`B` limbs and emits:
    (1) `reduce_with_powers(limbs, B) − sum = 0`     — the reconstruction constraint
    (2) for each limb: `∏_{j<B}(limb − j) = 0`       — the limb is in `{0,…,B-1}`
  `range_check(x, n)` instantiates this at `B = 2` with `n` limbs, so `x` is forced
  to be an `n`-bit value, i.e. `x < 2^n`.
-/
import Plonky2Spec.Basic

namespace Plonky2Spec

/-! ### Little-endian base-`B` reconstruction -/

/-- `reconstruct B [d₀, d₁, …] = d₀ + B·d₁ + B²·d₂ + …`, mirroring
    `reduce_with_powers(limbs, B)` in `BaseSumGate::eval_unfiltered`
    (base_sum.rs:71). -/
def reconstruct (B : Nat) : List Felt → Felt
  | []      => 0
  | d :: ds => d + B * reconstruct B ds

@[simp] theorem reconstruct_nil (B : Nat) : reconstruct B [] = 0 := rfl

@[simp] theorem reconstruct_cons (B : Nat) (d : Felt) (ds : List Felt) :
    reconstruct B (d :: ds) = d + B * reconstruct B ds := rfl

/-- If every limb is a valid base-`B` digit (`< B`), the reconstructed value is
    bounded by `B ^ (number of limbs)`. This is the arithmetic heart of range-check
    soundness: an `L`-limb base-`B` number is `< Bᴸ`. -/
theorem reconstruct_lt {B : Nat} (_hB : 0 < B) :
    ∀ {limbs : List Felt}, (∀ d ∈ limbs, d < B) →
      reconstruct B limbs < B ^ limbs.length := by
  intro limbs
  induction limbs with
  | nil =>
    intro _
    rw [reconstruct_nil, List.length_nil, Nat.pow_zero]
    exact Nat.one_pos
  | cons d ds ih =>
    intro h
    have hd : d < B := h d (List.mem_cons_self ..)
    have hds : ∀ x ∈ ds, x < B := fun x hx => h x (List.mem_cons_of_mem _ hx)
    have hrec : reconstruct B ds < B ^ ds.length := ih hds
    have h1 : reconstruct B ds + 1 ≤ B ^ ds.length := hrec
    -- `d + B·r < B + B·r = B·(r+1) ≤ B·Bᴸ = B^(L+1)`, using `d < B` and `r+1 ≤ Bᴸ`.
    calc reconstruct B (d :: ds)
        = d + B * reconstruct B ds := reconstruct_cons B d ds
      _ < B + B * reconstruct B ds := Nat.add_lt_add_right hd _
      _ = B * (reconstruct B ds + 1) := by rw [Nat.mul_add, Nat.mul_one, Nat.add_comm]
      _ ≤ B * B ^ ds.length := Nat.mul_le_mul_left B h1
      _ = B ^ (ds.length + 1) := by rw [Nat.pow_succ, Nat.mul_comm]

/-! ### Base-`B` digit witness (for completeness) -/

/-- The little-endian base-`B` digits of `v`, padded/truncated to `L` limbs:
    the witness the gate's `BaseSplitGenerator` computes (base_sum.rs:190-219). -/
def baseDigits (B : Nat) : Nat → Felt → List Felt
  | 0,      _ => []
  | (L+1), v => (v % B) :: baseDigits B L (v / B)

@[simp] theorem baseDigits_length (B : Nat) :
    ∀ (L : Nat) (v : Felt), (baseDigits B L v).length = L := by
  intro L
  induction L with
  | zero => intro _; rfl
  | succ L ih => intro v; simp [baseDigits, ih]

theorem baseDigits_lt {B : Nat} (hB : 0 < B) :
    ∀ (L : Nat) (v : Felt), ∀ d ∈ baseDigits B L v, d < B := by
  intro L
  induction L with
  | zero => intro _ d hd; simp [baseDigits] at hd
  | succ L ih =>
    intro v d hd
    rw [baseDigits, List.mem_cons] at hd
    rcases hd with h | h
    · subst h; exact Nat.mod_lt _ hB
    · exact ih (v / B) d h

theorem baseDigits_reconstruct {B : Nat} (_hB : 0 < B) :
    ∀ (L : Nat) (v : Felt), v < B ^ L → reconstruct B (baseDigits B L v) = v := by
  intro L
  induction L with
  | zero =>
    intro v hv
    rw [Nat.pow_zero] at hv
    have hv0 : v = 0 := Nat.lt_one_iff.mp hv
    subst hv0
    rfl
  | succ L ih =>
    intro v hv
    have hdiv : v / B < B ^ L := by
      apply Nat.div_lt_of_lt_mul
      have hpow : B ^ (L + 1) = B * B ^ L := by rw [Nat.pow_succ, Nat.mul_comm]
      rw [hpow] at hv
      exact hv
    rw [baseDigits, reconstruct_cons, ih (v / B) hdiv]
    exact Nat.mod_add_div v B

/-! ### `BaseSumGate<B>` structured constraints -/

/-- Structured `BaseSumGate<B>` constraints for a single row with
    `num_limbs = limbs.length` (base_sum.rs:68-81):
    * `recon` — `reduce_with_powers(limbs, B) = sum`  (the sum wire), and
    * `range` — every limb lies in `{0,…,B-1}`.

    FIELD-FIDELITY NOTE. The gate states these over `ZMod p`: `recon` is a field
    equation and `range` is the degree-`B` product `∏_{j<B}(limbᵢ − j) = 0`. We
    model them over `Nat`:
    * modeling `recon` as an exact `Nat` equation is faithful exactly when the
      reconstructed value does not wrap mod `p` — the `NoWrap` side-condition below
      (`Bᴸ ≤ p`), which for `range_check` (`B = 2`, `L ≤ 48`) holds with huge margin;
    * modeling the product as membership `limbᵢ < B` is the *semantic content* of
      the range primitive. The product ⇔ membership equivalence is the prime-field
      integral-domain fact, deferred to the `ZMod p` phase (it is not statable in
      core Lean: `Nat.Prime`/Euclid live in mathlib). -/
structure BaseSum (B : Nat) (sum : Felt) (limbs : List Felt) : Prop where
  recon : reconstruct B limbs = sum
  range : ∀ d ∈ limbs, d < B

/-- Non-overflow side-condition under which the `Nat` model of `recon` is faithful
    to the field equation: the largest representable value `Bᴸ − 1` is below the
    field modulus, so no reconstruction wraps mod `p`. -/
def NoWrap (B : Nat) (limbs : List Felt) : Prop := B ^ limbs.length ≤ goldilocks

/-- **Soundness of `BaseSumGate<B>`**: a satisfied gate forces the sum wire to be a
    genuine `num_limbs`-digit base-`B` value, i.e. `sum < B ^ num_limbs`. -/
theorem baseSum_sound {B : Nat} (hB : 0 < B) {sum : Felt} {limbs : List Felt}
    (c : BaseSum B sum limbs) : sum < B ^ limbs.length := by
  have h := reconstruct_lt hB c.range
  rwa [c.recon] at h

/-- **Completeness of `BaseSumGate<B>`**: any value below `B ^ L` has an `L`-limb
    witness satisfying the gate constraints (the base-`B` digits). -/
theorem baseSum_complete {B : Nat} (hB : 0 < B) {sum : Felt} {L : Nat}
    (h : sum < B ^ L) : ∃ limbs : List Felt, limbs.length = L ∧ BaseSum B sum limbs :=
  ⟨baseDigits B L sum, baseDigits_length B L sum,
    { recon := baseDigits_reconstruct hB L sum h
      range := baseDigits_lt hB L sum }⟩

/-! ### `range_check(x, n)` — the `B = 2` instantiation -/

/-- `range_check(x, n)` (range_check.rs:21 → split_le → `BaseSumGate<2>`): there is
    a base-2 decomposition of `x` into `n` bits, each in `{0,1}`. -/
def rangeCheck (x : Felt) (n : Nat) : Prop :=
  ∃ limbs : List Felt, limbs.length = n ∧ BaseSum 2 x limbs

/-- **Soundness of `range_check`**: a satisfied `range_check(x, n)` proves the
    spec-level bound `x < 2^n`. -/
theorem rangeCheck_sound {x : Felt} {n : Nat} (h : rangeCheck x n) : x < 2 ^ n := by
  obtain ⟨limbs, hlen, c⟩ := h
  have hb := baseSum_sound (by omega) c
  rwa [hlen] at hb

/-- **Completeness of `range_check`**: every `n`-bit value is accepted. -/
theorem rangeCheck_complete {x : Felt} {n : Nat} (h : x < 2 ^ n) : rangeCheck x n :=
  baseSum_complete (by omega) h

/-- **Bridge lemma.** A satisfied `range_check(x, n)` *discharges* the spec's
    `inRange n x` (it is a theorem, not an assumption). In the `ZMod p` phase `x`
    becomes `x.val`; the statement is unchanged. This is the rung that lets
    `qp-zk-circuits/formal` stop assuming `inRange 32 …` for the leaf circuit. -/
theorem rangeCheck_implies_inRange {x : Felt} {n : Nat} (h : rangeCheck x n) :
    inRange n x := rangeCheck_sound h

/-! ### `enforce_target_less_than_const` -/

/-- `enforce_target_less_than_const(t, U, n_log)` (common/gadgets.rs:66): enforces
    `t < U`. Internally it `split_le`s `t` to `n_log` bits (this step's range core)
    and forces a bit-wise `is_const_less_than` comparator boolean to `0`. We record
    the intended spec here; the comparator's boolean-logic soundness rests on the
    T1 gadgets (`and`/`or`/`not`/`is_equal`) and is discharged in Step 2. -/
def enforceLessThanConst (t U : Felt) : Prop := t < U

/-- When the exclusive bound fits in `n_log` bits (`U ≤ 2^n_log`, the
    `assert_comparison_width` precondition), the enforced `t < U` also yields the
    spec range bound `inRange n_log t`. -/
theorem enforceLessThanConst_inRange {t U : Felt} {nlog : Nat}
    (h : enforceLessThanConst t U) (hU : U ≤ 2 ^ nlog) : inRange nlog t :=
  Nat.lt_of_lt_of_le h hU

end Plonky2Spec
