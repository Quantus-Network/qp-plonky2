/-
  Field-agnostic place-value core for the gadget layer.

  This module is deliberately **mathlib-free** (core Lean + `omega` only): it is the
  pure base-`B` positional-number arithmetic that the range primitive rests on,
  stated over `ℕ`. The field-level model (`RangeCheck.lean`) works over `ZMod p`
  and lifts to these `ℕ` facts through `ZMod.val`, exactly the way Zellic's gate
  proofs black-box the field via `.val`.

  Keeping this layer mathlib-free means the positional-arithmetic reasoning is
  checked independently of the (heavier) field development.
-/

namespace Plonky2Spec

/-- The Goldilocks prime `p = 2^64 − 2^32 + 1`, the native field of qp-plonky2
    (`GoldilocksField`). The generic gadget theorems are stated for an arbitrary
    prime `p` with explicit size bounds (à la Zellic); this constant records the
    concrete modulus those bounds are satisfied by (`2^48 ≪ goldilocks`). -/
def goldilocks : Nat := 0xFFFFFFFF00000001

/-! ### Little-endian base-`B` reconstruction -/

/-- `reconstruct B [d₀, d₁, …] = d₀ + B·d₁ + B²·d₂ + …`, mirroring
    `reduce_with_powers(limbs, B)` in `BaseSumGate::eval_unfiltered`
    (base_sum.rs:71). Operates on the `ℕ` limb *values*. -/
def reconstruct (B : Nat) : List Nat → Nat
  | []      => 0
  | d :: ds => d + B * reconstruct B ds

@[simp] theorem reconstruct_nil (B : Nat) : reconstruct B [] = 0 := rfl

@[simp] theorem reconstruct_cons (B : Nat) (d : Nat) (ds : List Nat) :
    reconstruct B (d :: ds) = d + B * reconstruct B ds := rfl

/-- If every limb is a valid base-`B` digit (`< B`), the reconstructed value is
    bounded by `B ^ (number of limbs)`: an `L`-limb base-`B` number is `< Bᴸ`.
    This is the arithmetic heart of range-check soundness. -/
theorem reconstruct_lt {B : Nat} (_hB : 0 < B) :
    ∀ {limbs : List Nat}, (∀ d ∈ limbs, d < B) →
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
def baseDigits (B : Nat) : Nat → Nat → List Nat
  | 0,      _ => []
  | (L+1), v => (v % B) :: baseDigits B L (v / B)

@[simp] theorem baseDigits_length (B : Nat) :
    ∀ (L : Nat) (v : Nat), (baseDigits B L v).length = L := by
  intro L
  induction L with
  | zero => intro _; rfl
  | succ L ih => intro v; simp [baseDigits, ih]

theorem baseDigits_lt {B : Nat} (hB : 0 < B) :
    ∀ (L : Nat) (v : Nat), ∀ d ∈ baseDigits B L v, d < B := by
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
    ∀ (L : Nat) (v : Nat), v < B ^ L → reconstruct B (baseDigits B L v) = v := by
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

end Plonky2Spec
