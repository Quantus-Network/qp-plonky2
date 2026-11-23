/-
  The proof-system Poseidon gate, given meaning (Step 18).

  `PoseidonGate<WIDTH=12>` (`plonky2/src/gates/poseidon.rs`) is the gate `build` lays
  down for the proof-system hash (`PoseidonHash` in `PoseidonGoldilocksConfig`): the
  public-input digest, Merkle caps, the recursive verifier's challenger. This module models
  its constraints structurally, as `Plonky2Spec.Poseidon2` models the `Poseidon2Gate`, and
  proves

      the gate's constraints are satisfiable (over its checkpoint wires)
      **iff** the output wires are the Poseidon permutation of the (possibly swapped)
      input wires,

  for any round constants and matrices (`Consts`). `Generated/PoseidonConsts.lean` gives
  the Goldilocks tables of `qp_plonky2_core::poseidon` and pins `Consts.perm` on them to
  `Poseidon::poseidon` by a kernel-evaluated test vector.

  Design (mirrors `PoseidonGate::eval_unfiltered`): a checkpointed permutation. The gate
  caps its degree at 7 by taking the input of every S-box but the first round's from a
  wire, emitting `state[i] − sbox_in = 0`, and continuing from the wire. Hence the
  existential over the checkpoint wires. The partial rounds are in the fast form of the
  Poseidon paper's appendix B (`partial_rounds`, `poseidon.rs`): one constant layer and one
  dense matrix (`partialInit`) up front, then per round an S-box on lane `0`, its round
  constant, and a sparse matrix (`partialFast`). The gate adds no constant after the last
  partial round; `FAST_PARTIAL_ROUND_CONSTANTS[21] = 0`, and the model adds `partialRc 21`
  uniformly.

  The gate also swaps its first two quadruples of inputs when its `swap` wire is `1`
  (Merkle sibling ordering), through four `delta` wires holding
  `swap · (input[i + 4] − input[i])`. The statement here carries `swap` and the deltas:
  `gate_sound_complete` pins the deltas and the output for any binary `swap`.

  Wire layout (`PoseidonGate`, width 12):
  * `w0..w11`    input                      (`wire_input i = i`)
  * `w12..w23`   output                     (`wire_output i = 12 + i`)
  * `w24`        swap                       (`WIRE_SWAP`)
  * `w25..w28`   deltas                     (`wire_delta i = 25 + i`)
  * `w29..w64`   first full rounds' S-box inputs, rounds 1..3 × 12 lanes
                 (`wire_full_sbox_0`; round 0 is not checkpointed)
  * `w65..w86`   partial rounds' S-box inputs, one per round (`wire_partial_sbox`)
  * `w87..w134`  last full rounds' S-box inputs, rounds 0..3 × 12 lanes (`wire_full_sbox_1`)
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fin.VecNotation
import Mathlib.Algebra.BigOperators.Fin
import Plonky2Spec.Poseidon2

namespace Plonky2Spec.Poseidon

open Plonky2Spec.Poseidon2 (St)

variable {p : ℕ}

/-- The tables of a `Poseidon` instance (`qp_plonky2_core::poseidon::Poseidon`), in the
    fast partial-round form the gate and the native permutation both compute.
    * `circ`, `diag`: the MDS matrix is the circulant matrix of `circ` plus the diagonal
      matrix of `diag` (`MDS_MATRIX_CIRC`, `MDS_MATRIX_DIAG`).
    * `rc r`: the constants of round `r` (`ALL_ROUND_CONSTANTS[12 r ..]`); the fast form
      uses rounds `0..3` and `26..29`.
    * `partialFirst`: the constants folded in front of the partial rounds
      (`FAST_PARTIAL_FIRST_ROUND_CONSTANT`); `partialRc r`: the lane-`0` constant of
      partial round `r` (`FAST_PARTIAL_ROUND_CONSTANTS`).
    * `initMat`: the matrix applied once before the partial rounds, with first row and
      column `e₀` (`FAST_PARTIAL_ROUND_INITIAL_MATRIX` fills the rest).
    * `w r`, `v r`: the sparse matrix of partial round `r`. `w r` is its first row,
      `w r 0 = circ 0 + diag 0` (`M₀₀`) and `w r j = FAST_PARTIAL_ROUND_W_HATS[r][j − 1]`;
      `v r j = FAST_PARTIAL_ROUND_VS[r][j − 1]` is the first column below `M₀₀`, with
      `v r 0` unused. -/
structure Consts (p : ℕ) where
  circ : St p
  diag : St p
  rc : Fin 30 → St p
  partialFirst : St p
  partialRc : Fin 22 → ZMod p
  initMat : Fin 12 → Fin 12 → ZMod p
  w : Fin 22 → Fin 12 → ZMod p
  v : Fin 22 → Fin 12 → ZMod p

/-! ### Permutation primitives

`irreducible`, as in `Plonky2Spec.Poseidon2`: the meaning proof is the checkpoint
substitution at the round layer and never unfolds a primitive. Each matrix primitive
ends in `materialize`, which lists the lanes before reading them back; the state is a
function, and without it a kernel evaluation of the permutation (the test vector in
`Generated/PoseidonConsts.lean`) would recompute each round once per read of each lane. -/

/-- The `x^7` S-box (`sbox_monomial`). -/
@[irreducible] def sbox7 (x : ZMod p) : ZMod p := x ^ 7

/-- The lanes of `s`, listed and read back. -/
def materialize (s : St p) : St p := fun i => (List.ofFn s)[i]

theorem materialize_eq (s : St p) : materialize s = s := by
  funext i; exact List.getElem_ofFn _

/-- The MDS layer (`mds_layer_field`): row `r` is `∑ i, circ i · s (i + r) + diag r · s r`. -/
@[irreducible] def Consts.mdsLayer (k : Consts p) (s : St p) : St p :=
  materialize fun r => (∑ i, k.circ i * s (i + r)) + k.diag r * s r

/-- The matrix in front of the partial rounds (`mds_partial_layer_init`). -/
@[irreducible] def Consts.partialInit (k : Consts p) (s : St p) : St p :=
  materialize fun c => ∑ r, s r * k.initMat r c

/-- The sparse matrix of partial round `r` (`mds_partial_layer_fast_field`): lane `0` is
    `w r · s`, lane `i ≥ 1` is `s 0 · v r i + s i`. -/
@[irreducible] def Consts.partialFast (k : Consts p) (r : Fin 22) (s : St p) : St p :=
  materialize fun i => if i = 0 then ∑ j, s j * k.w r j else s 0 * k.v r i + s i

/-! ### The reference permutation (checkpoint-free) -/

/-- Add a constant vector lanewise (`constant_layer_field`, `partial_first_constant_layer`). -/
def addRC (rc s : St p) : St p := fun i => s i + rc i

/-- The S-box on every lane (`sbox_layer_field`). -/
def sboxAll (s : St p) : St p := fun i => sbox7 (s i)

/-- Full round `r`: constants, S-boxes, MDS. -/
def Consts.fullRound (k : Consts p) (r : Fin 30) (s : St p) : St p :=
  k.mdsLayer (sboxAll (addRC (k.rc r) s))

/-- Partial round `r` in the fast form: the S-box and the round constant on lane `0`,
    then the sparse matrix. -/
def Consts.partialRound (k : Consts p) (r : Fin 22) (s : St p) : St p :=
  k.partialFast r (Function.update s 0 (sbox7 (s 0) + k.partialRc r))

/-- Fold a list of full rounds. -/
def Consts.fullPhase (k : Consts p) (rounds : List (Fin 30)) (s : St p) : St p :=
  rounds.foldl (fun s r => k.fullRound r s) s

/-- Fold a list of partial rounds. -/
def Consts.partialPhase (k : Consts p) (rounds : List (Fin 22)) (s : St p) : St p :=
  rounds.foldl (fun s r => k.partialRound r s) s

/-- The rounds of the first full phase, the partial phase, the last full phase. -/
def firstFull : List (Fin 30) := [0, 1, 2, 3]
def partials : List (Fin 22) := List.finRange 22
def lastFull : List (Fin 30) := [26, 27, 28, 29]

/-- The Poseidon permutation (`Poseidon::poseidon`): four full rounds, the partial
    phase's constant layer and matrix, twenty-two partial rounds, four full rounds. -/
def Consts.perm (k : Consts p) (inp : St p) : St p :=
  let s := k.fullPhase firstFull inp
  let s := k.partialPhase partials (k.partialInit (addRC k.partialFirst s))
  k.fullPhase lastFull s

/-! ### The gate's checkpointed constraints -/

/-- The input layer after the swap: `input[i] + delta i` for `i < 4`,
    `input[i] − delta (i − 4)` for `4 ≤ i < 8`, the input elsewhere. -/
def swapIn (delta : Fin 4 → ZMod p) (inp : St p) : St p := fun i =>
  if h : i.val < 4 then inp i + delta ⟨i, h⟩
  else if h' : i.val < 8 then inp i - delta ⟨i - 4, by omega⟩
  else inp i

/-- The deltas the gate's constraints force: `swap · (input[i + 4] − input[i])`. -/
def swapDelta (swap : ZMod p) (inp : St p) (i : Fin 4) : ZMod p :=
  swap * (inp ⟨i + 4, by omega⟩ - inp ⟨i, by omega⟩)

/-- A checkpointed full phase: each round emits the twelve constraints
    `state[i] + rc[i] − ck[i]` and continues from `mdsLayer (sboxAll ck)`. Returns the
    constraints and the resulting state. -/
def Consts.fullConstrs (k : Consts p) :
    List (Fin 30) → List (St p) → St p → List (ZMod p) × St p
  | r :: rs, ck :: cks, s =>
      let c := (List.finRange 12).map fun i => (s i + k.rc r i) - ck i
      let rest := k.fullConstrs rs cks (k.mdsLayer (sboxAll ck))
      (c ++ rest.1, rest.2)
  | _, _, s => ([], s)

/-- A checkpointed partial phase: each round emits `state[0] − ck` and continues from
    `partialFast r (update state 0 (sbox7 ck + rc))`. -/
def Consts.partialConstrs (k : Consts p) :
    List (Fin 22) → List (ZMod p) → St p → List (ZMod p) × St p
  | r :: rs, ck :: cks, s =>
      let c := s 0 - ck
      let rest := k.partialConstrs rs cks
        (k.partialFast r (Function.update s 0 (sbox7 ck + k.partialRc r)))
      (c :: rest.1, rest.2)
  | _, _, s => ([], s)

/-- The gate's constraint list, in the order of `eval_unfiltered`: `swap` binary, the four
    deltas, full round `0` on the swapped input without a checkpoint, the three
    checkpointed first full rounds, the partial phase, the four checkpointed last full
    rounds, and the twelve output equalities `state[i] − output[i]`. -/
def Consts.gateConstraints (k : Consts p) (inp out : St p) (swap : ZMod p)
    (delta : Fin 4 → ZMod p) (ckFirst : List (St p)) (ckPartial : List (ZMod p))
    (ckLast : List (St p)) : List (ZMod p) :=
  let cDelta := (List.finRange 4).map fun i => swapDelta swap inp i - delta i
  let s0 := k.fullRound 0 (swapIn delta inp)
  let r1 := k.fullConstrs [1, 2, 3] ckFirst s0
  let r2 := k.partialConstrs partials ckPartial (k.partialInit (addRC k.partialFirst r1.2))
  let r3 := k.fullConstrs lastFull ckLast r2.2
  let cOut := (List.finRange 12).map fun i => r3.2 i - out i
  swap * (swap - 1) :: cDelta ++ r1.1 ++ r2.1 ++ r3.1 ++ cOut

/-! ### Meaning: constraints satisfiable ⟺ output = perm (swapped input)

The primitives are never unfolded below. Each checkpoint constraint pins its wire to the
running state, so the gate's reset-and-continue is the checkpoint-free `perm`. -/

section Meaning

variable {k : Consts p}

private theorem fullPhase_cons (r : Fin 30) (rs : List (Fin 30)) (s : St p) :
    k.fullPhase (r :: rs) s = k.fullPhase rs (k.fullRound r s) := rfl

private theorem partialPhase_cons (r : Fin 22) (rs : List (Fin 22)) (s : St p) :
    k.partialPhase (r :: rs) s = k.partialPhase rs (k.partialRound r s) := rfl

private theorem forall_map_finRange {n : ℕ} {f : Fin n → ZMod p}
    (h : ∀ c ∈ (List.finRange n).map f, c = 0) (i : Fin n) : f i = 0 :=
  h _ (List.mem_map_of_mem (List.mem_finRange i))

private theorem fullConstrs_sound :
    ∀ (rs : List (Fin 30)) (cks : List (St p)) (s : St p), rs.length = cks.length →
      (∀ c ∈ (k.fullConstrs rs cks s).1, c = 0) →
      (k.fullConstrs rs cks s).2 = k.fullPhase rs s := by
  intro rs
  induction rs with
  | nil => intro cks s _ _; rfl
  | cons r rs ih =>
    intro cks s hlen h
    cases cks with
    | nil => simp at hlen
    | cons ck cks =>
      simp only [Consts.fullConstrs] at h
      rw [List.forall_mem_append] at h
      obtain ⟨hc, hr⟩ := h
      have hck : ck = addRC (k.rc r) s := by
        funext i
        exact ((sub_eq_zero.mp (forall_map_finRange hc i)).symm)
      have hstep : k.mdsLayer (sboxAll ck) = k.fullRound r s := by rw [hck]; rfl
      have hrec := ih cks (k.mdsLayer (sboxAll ck)) (by simpa using hlen) hr
      simp only [Consts.fullConstrs]
      rw [hrec, hstep]
      exact (fullPhase_cons r rs s).symm

private theorem partialConstrs_sound :
    ∀ (rs : List (Fin 22)) (cks : List (ZMod p)) (s : St p), rs.length = cks.length →
      (∀ c ∈ (k.partialConstrs rs cks s).1, c = 0) →
      (k.partialConstrs rs cks s).2 = k.partialPhase rs s := by
  intro rs
  induction rs with
  | nil => intro cks s _ _; rfl
  | cons r rs ih =>
    intro cks s hlen h
    cases cks with
    | nil => simp at hlen
    | cons ck cks =>
      simp only [Consts.partialConstrs, List.mem_cons, forall_eq_or_imp] at h
      obtain ⟨hc, hr⟩ := h
      have hck : ck = s 0 := (sub_eq_zero.mp hc).symm
      have hstep : k.partialFast r (Function.update s 0 (sbox7 ck + k.partialRc r))
          = k.partialRound r s := by rw [hck]; rfl
      have hrec := ih cks (k.partialFast r (Function.update s 0 (sbox7 ck + k.partialRc r)))
        (by simpa using hlen) hr
      simp only [Consts.partialConstrs]
      rw [hrec, hstep]
      exact (partialPhase_cons r rs s).symm

private theorem fullConstrs_complete :
    ∀ (rs : List (Fin 30)) (s : St p), ∃ cks : List (St p), cks.length = rs.length ∧
      (∀ c ∈ (k.fullConstrs rs cks s).1, c = 0) ∧
      (k.fullConstrs rs cks s).2 = k.fullPhase rs s := by
  intro rs
  induction rs with
  | nil => intro s; exact ⟨[], rfl, by simp [Consts.fullConstrs], rfl⟩
  | cons r rs ih =>
    intro s
    obtain ⟨cks, hlen, hzero, hstate⟩ := ih (k.fullRound r s)
    refine ⟨addRC (k.rc r) s :: cks, by simp [hlen], ?_, ?_⟩
    · intro c hc
      simp only [Consts.fullConstrs] at hc
      rcases List.mem_append.mp hc with h | h
      · obtain ⟨i, _, rfl⟩ := List.mem_map.mp h
        simp [addRC]
      · exact hzero c h
    · simp only [Consts.fullConstrs]
      exact hstate

private theorem partialConstrs_complete :
    ∀ (rs : List (Fin 22)) (s : St p), ∃ cks : List (ZMod p), cks.length = rs.length ∧
      (∀ c ∈ (k.partialConstrs rs cks s).1, c = 0) ∧
      (k.partialConstrs rs cks s).2 = k.partialPhase rs s := by
  intro rs
  induction rs with
  | nil => intro s; exact ⟨[], rfl, by simp [Consts.partialConstrs], rfl⟩
  | cons r rs ih =>
    intro s
    obtain ⟨cks, hlen, hzero, hstate⟩ := ih (k.partialRound r s)
    refine ⟨s 0 :: cks, by simp [hlen], ?_, ?_⟩
    · intro c hc
      simp only [Consts.partialConstrs, List.mem_cons] at hc
      rcases hc with rfl | h
      · simp
      · exact hzero c h
    · simp only [Consts.partialConstrs]
      exact hstate

/-- **The Poseidon gate computes the permutation.** The gate's constraints are
    satisfiable over its checkpoint wires — `ckFirst` (first full rounds `1..3`),
    `ckPartial` (the partial rounds), `ckLast` (the last full rounds) — iff `swap` is
    binary, the deltas are `swap · (input[i + 4] − input[i])`, and the output wires are the
    permutation of the swapped input. Soundness (→): no witness forces a wrong output.
    Completeness (←): the honest assignment is never locked out. For any constants. -/
theorem gate_sound_complete (inp out : St p) (swap : ZMod p) (delta : Fin 4 → ZMod p) :
    (∃ (ckFirst ckLast : List (St p)) (ckPartial : List (ZMod p)),
        ckFirst.length = 3 ∧ ckPartial.length = 22 ∧ ckLast.length = 4 ∧
        ∀ c ∈ k.gateConstraints inp out swap delta ckFirst ckPartial ckLast, c = 0)
      ↔ swap * (swap - 1) = 0 ∧ delta = swapDelta swap inp ∧
          out = k.perm (swapIn delta inp) := by
  constructor
  · rintro ⟨ckFirst, ckLast, ckPartial, hF, hP, hL, hzero⟩
    simp only [Consts.gateConstraints, List.cons_append, List.mem_cons, forall_eq_or_imp]
      at hzero
    rw [List.forall_mem_append, List.forall_mem_append, List.forall_mem_append,
      List.forall_mem_append] at hzero
    obtain ⟨hswap, ⟨⟨⟨hdelta, h1⟩, h2⟩, h3⟩, hout⟩ := hzero
    refine ⟨hswap, ?_, ?_⟩
    · funext i
      exact (sub_eq_zero.mp (forall_map_finRange hdelta i)).symm
    have e1 := fullConstrs_sound (k := k) _ _ _ (by simp [hF]) h1
    have e2 := partialConstrs_sound (k := k) _ _ _ (by simp [partials, hP]) h2
    have e3 := fullConstrs_sound (k := k) _ _ _ (by simp [lastFull, hL]) h3
    have hout' : out = (k.fullConstrs lastFull ckLast
        (k.partialConstrs partials ckPartial
          (k.partialInit (addRC k.partialFirst
            (k.fullConstrs [1, 2, 3] ckFirst (k.fullRound 0 (swapIn delta inp))).2))).2).2 := by
      funext i
      exact (sub_eq_zero.mp (forall_map_finRange hout i)).symm
    rw [hout', e3, e2, e1]
    rfl
  · rintro ⟨hswap, hdelta, hperm⟩
    obtain ⟨ckFirst, hFlen, hF0, hFst⟩ :=
      fullConstrs_complete (k := k) [1, 2, 3] (k.fullRound 0 (swapIn delta inp))
    obtain ⟨ckPartial, hPlen, hP0, hPst⟩ :=
      partialConstrs_complete (k := k) partials
        (k.partialInit (addRC k.partialFirst
          (k.fullConstrs [1, 2, 3] ckFirst (k.fullRound 0 (swapIn delta inp))).2))
    obtain ⟨ckLast, hLlen, hL0, hLst⟩ :=
      fullConstrs_complete (k := k) lastFull
        (k.partialConstrs partials ckPartial
          (k.partialInit (addRC k.partialFirst
            (k.fullConstrs [1, 2, 3] ckFirst (k.fullRound 0 (swapIn delta inp))).2))).2
    refine ⟨ckFirst, ckLast, ckPartial, by simpa using hFlen,
      by simpa [partials] using hPlen, by simpa [lastFull] using hLlen, ?_⟩
    simp only [Consts.gateConstraints, List.cons_append, List.mem_cons, forall_eq_or_imp]
    rw [List.forall_mem_append, List.forall_mem_append, List.forall_mem_append,
      List.forall_mem_append]
    refine ⟨hswap, ⟨⟨⟨?_, hF0⟩, hP0⟩, hL0⟩, ?_⟩
    · intro c hc
      obtain ⟨i, _, rfl⟩ := List.mem_map.mp hc
      simp [hdelta]
    · intro c hc
      obtain ⟨i, _, rfl⟩ := List.mem_map.mp hc
      have hfin : (k.fullConstrs lastFull ckLast
          (k.partialConstrs partials ckPartial
            (k.partialInit (addRC k.partialFirst
              (k.fullConstrs [1, 2, 3] ckFirst
                (k.fullRound 0 (swapIn delta inp))).2))).2).2
          = k.perm (swapIn delta inp) := by
        rw [hLst, hPst, hFst]; rfl
      rw [hfin, ← hperm]
      simp

/-- With the swap wire at `0` the deltas vanish and the input layer is the input. -/
theorem swapIn_zero (inp : St p) : swapIn (swapDelta 0 inp) inp = inp := by
  funext i
  simp only [swapIn, swapDelta, zero_mul]
  split_ifs <;> simp

end Meaning

/-! ### Rows -/

/-- The gate's constraints hold on a row's wires, read at the gate's layout. -/
def Consts.rowHolds (k : Consts p) (wires : ℕ → ZMod p) : Prop :=
  ∀ c ∈ k.gateConstraints (fun i => wires i) (fun i => wires (12 + i)) (wires 24)
      (fun i => wires (25 + i))
      ((List.range 3).map fun r i => wires (29 + 12 * r + i))
      ((List.range 22).map fun r => wires (65 + r))
      ((List.range 4).map fun r i => wires (87 + 12 * r + i)), c = 0

/-- A row whose swap wire is `0` has the permutation of its input wires on its output
    wires: the soundness half of `gate_sound_complete`. -/
theorem Consts.rowHolds.perm_eq {k : Consts p} {wires : ℕ → ZMod p} (h : k.rowHolds wires)
    (hswap : wires 24 = 0) :
    (fun i : Fin 12 => wires (12 + i)) = k.perm fun i : Fin 12 => wires i := by
  obtain ⟨-, hdelta, hout⟩ :=
    (gate_sound_complete (k := k) _ _ _ _).mp ⟨_, _, _, by simp, by simp, by simp, h⟩
  rw [hout, hdelta, hswap, swapIn_zero]

end Plonky2Spec.Poseidon
