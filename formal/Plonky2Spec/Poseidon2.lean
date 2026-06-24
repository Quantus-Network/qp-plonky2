/-
  T3 — the Poseidon2 permutation gate, given meaning.

  This module models the `Poseidon2Gate` constraint system structurally and proves
  its *meaning*:

      the gate's constraints are satisfiable (over the S-box checkpoint wires)
      **iff** the output wires are the Poseidon2 permutation of the input wires.

  `Generated/Poseidon2.lean` is the *mechanically extracted* list of the same 118
  constraints (a flat `let`-program, pinned to the live Rust by the exporter's
  differential test). Note `gateConstraints` here is a structural hand-transcription
  of `eval_unfiltered`; a `ring`/`rfl` bridge to the extracted flat form is *not*
  tractable, because the extracted DAG shares subexpressions via `let` (linear size)
  whereas expanding the structured model unfolds `internalMix`'s running sum through
  the 22 compounding internal rounds (~3²² term size). Closing that last gap cleanly
  needs the exporter to emit a *round-factored* (share-preserving) Lean model — see
  `PLAN.md` Step 3b. The faithfulness of the transcription is meanwhile carried by
  the differential test `poseidon2_hand_gate_constraints_match_real_gate`
  (constraint-exporter), which transliterates this `gateConstraints` walk and compares
  it to `eval_unfiltered` at random wire assignments, and by the line-by-line
  correspondence documented below.

  Design (mirrors `Poseidon2Gate::eval_unfiltered`, plonky2/src/gates/poseidon2.rs):
  the gate is a *checkpointed* permutation. To cap each constraint at degree 7 it
  introduces an auxiliary "S-box input" wire before every S-box, emits
  `state[i] − sbox_in = 0`, then continues from `sbox_in` (not from the degree-7
  expression). So the constraints pin `output = perm(input)` only *modulo* the
  checkpoint wires — hence the existential over them.

  The whole development is **generic in the round constants** (`eInit`, `eTerm`,
  `iRc`) and the diagonal (`diag`): the equivalence holds for any constants, so the
  permutation primitives (`mdsLight`, `sbox7`, `internalMix`) are treated *opaquely*
  and never unfolded. Instantiating at the real Goldilocks constants — and bridging
  to the spec's `RandomOracle.H` — is Step 3c.

  Wire layout (`Poseidon2Gate`, `D = 1`, width 12):
  * `w0..w11`    input        (`wire_input i = i`)
  * `w12..w23`   output       (`wire_output i = 12 + i`)
  * `w24..w107`  external S-box checkpoints, rounds 1..7 × 12 lanes
                 (`wire_ext_sbox`; round 0 of the initial phase is *elided*)
  * `w108..w129` internal S-box checkpoints, one per internal round (lane 0)
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fin.VecNotation
import Mathlib.Algebra.BigOperators.Fin

namespace Plonky2Spec.Poseidon2

variable {p : ℕ}

/-- The 12-lane Poseidon2 state over the native field. -/
abbrev St (p : ℕ) := Fin 12 → ZMod p

/-! ### Permutation primitives

These are marked `irreducible`: the meaning proof is purely structural (the
checkpoint substitution at the round/`foldl` layer) and must *never* unfold a
primitive. Crucially `internalMix` references the state 12× (`∑ j, s j`), so
unfolding it through 22 nested rounds would blow up ~12²²; `irreducible` keeps
every `rfl`/defeq at the cheap structural layer. The bridge to the extracted
constraints (Step 3b's `Poseidon2Bridge`) `unseal`s them locally instead. -/

/-- The `x^7` S-box. -/
@[irreducible] def sbox7 (x : ZMod p) : ZMod p := x ^ 7

/-- One `MDSMat4` block: `[2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]`, written exactly as
    `apply_mat4_base` (poseidon2.rs): `t = a+b+c+d`, then `t+a+b+b`, … -/
@[irreducible] def applyMat4 (a b c d : ZMod p) : Fin 4 → ZMod p :=
  let t := a + b + c + d
  ![t + a + b + b, t + b + c + c, t + c + d + d, t + a + a + d]

/-- The external light MDS: the `MDSMat4` blocks on lanes `(0..3),(4..7),(8..11)`,
    then add the per-residue-class sums (`mds_light_base`). -/
@[irreducible] def mdsLight (s : St p) : St p :=
  let a0 := applyMat4 (s 0) (s 1) (s 2) (s 3)
  let a1 := applyMat4 (s 4) (s 5) (s 6) (s 7)
  let a2 := applyMat4 (s 8) (s 9) (s 10) (s 11)
  let y0 := a0 0; let y1 := a0 1; let y2 := a0 2; let y3 := a0 3
  let y4 := a1 0; let y5 := a1 1; let y6 := a1 2; let y7 := a1 3
  let y8 := a2 0; let y9 := a2 1; let y10 := a2 2; let y11 := a2 3
  let m0 := y0 + y4 + y8
  let m1 := y1 + y5 + y9
  let m2 := y2 + y6 + y10
  let m3 := y3 + y7 + y11
  ![y0 + m0, y1 + m1, y2 + m2, y3 + m3,
    y4 + m0, y5 + m1, y6 + m2, y7 + m3,
    y8 + m0, y9 + m1, y10 + m2, y11 + m3]

/-- Internal diffusion: `y[i] = diag[i]·s[i] + Σ s` (`internal_mix_base`). -/
@[irreducible] def internalMix (diag s : St p) : St p := fun i => diag i * s i + ∑ j, s j

/-- Add a round-constant vector lanewise. -/
def addRC (rc s : St p) : St p := fun i => s i + rc i

/-- Apply the S-box on every lane. -/
def sboxAll (s : St p) : St p := fun i => sbox7 (s i)

/-- One full external round: add RC, S-box all lanes, light MDS. -/
def extRound (rc s : St p) : St p := mdsLight (sboxAll (addRC rc s))

/-- One internal round: add RC to lane 0, S-box lane 0 only, internal diffusion. -/
def intRound (diag : St p) (rc : ZMod p) (s : St p) : St p :=
  internalMix diag (Function.update s 0 (sbox7 (s 0 + rc)))

/-! ### The reference permutation (checkpoint-free) -/

/-- Fold a list of external rounds. -/
def extPhase (rcs : List (St p)) (s : St p) : St p :=
  rcs.foldl (fun s rc => extRound rc s) s

/-- Fold a list of internal rounds. -/
def intPhase (diag : St p) (rcs : List (ZMod p)) (s : St p) : St p :=
  rcs.foldl (fun s rc => intRound diag rc s) s

/-- The Poseidon2 permutation: light-MDS preamble, then 4 initial external rounds,
    22 internal rounds, 4 terminal external rounds (`eval`/witness order). -/
def permState (eInit eTerm : Fin 4 → St p) (iRc : Fin 22 → ZMod p) (diag : St p)
    (inp : St p) : St p :=
  let s := extPhase [eInit 0, eInit 1, eInit 2, eInit 3] (mdsLight inp)
  let s := intPhase diag ((List.finRange 22).map iRc) s
  extPhase [eTerm 0, eTerm 1, eTerm 2, eTerm 3] s

/-! ### The gate's checkpointed constraints -/

/-- A checkpointed external phase: walking `rcs` against checkpoint vectors `cks`,
    each round emits the 12 constraints `state[i] + rc[i] − ck[i]` (pre-S-box) and
    then continues from `mdsLight (sboxAll ck)` (reset to the checkpoint, then S-box,
    then MDS). Returns the constraints and the resulting state. -/
def extConstrs : List (St p) → List (St p) → St p → List (ZMod p) × St p
  | rc :: rcs, ck :: cks, s =>
      let c := (List.finRange 12).map (fun i => (s i + rc i) - ck i)
      let r := extConstrs rcs cks (mdsLight (sboxAll ck))
      (c ++ r.1, r.2)
  | _, _, s => ([], s)

/-- A checkpointed internal phase: each round emits `state[0] + rc − ck` and
    continues from `internalMix diag (update state 0 (sbox7 ck))`. -/
def intConstrs (diag : St p) : List (ZMod p) → List (ZMod p) → St p → List (ZMod p) × St p
  | rc :: rcs, ck :: cks, s =>
      let c := (s 0 + rc) - ck
      let r := intConstrs diag rcs cks (internalMix diag (Function.update s 0 (sbox7 ck)))
      (c :: r.1, r.2)
  | _, _, s => ([], s)

/-- The full gate constraint list, mirroring `eval_unfiltered`:
    preamble + initial round 0 (no checkpoint), then the 3 checkpointed initial
    rounds, 22 internal rounds, 4 terminal rounds, and 12 output equalities.
    `ckInit`/`ckTerm`/`ckI` are the S-box checkpoint wires. -/
def gateConstraints (eInit eTerm : Fin 4 → St p) (iRc : Fin 22 → ZMod p) (diag : St p)
    (inp out : St p) (ckInit ckTerm : List (St p)) (ckI : List (ZMod p)) : List (ZMod p) :=
  let s0 := extRound (eInit 0) (mdsLight inp)
  let r1 := extConstrs [eInit 1, eInit 2, eInit 3] ckInit s0
  let r2 := intConstrs diag ((List.finRange 22).map iRc) ckI r1.2
  let r3 := extConstrs [eTerm 0, eTerm 1, eTerm 2, eTerm 3] ckTerm r2.2
  let c4 := (List.finRange 12).map (fun i => out i - r3.2 i)
  r1.1 ++ r2.1 ++ r3.1 ++ c4

/-! ### Meaning: constraints satisfiable ⟺ output = perm(input)

The permutation primitives are never unfolded below; the argument is purely the
*checkpoint substitution* — each S-box constraint pins its checkpoint wire to the
running state, so the gate's reset-and-continue equals the checkpoint-free `perm`. -/

section Meaning

variable {eInit eTerm : Fin 4 → St p} {iRc : Fin 22 → ZMod p} {diag inp out : St p}

/-- Peel one external round off a phase fold (one `List.foldl` step; cheap because
    it never touches the round's interior). Used to fold the elided round-0 back in
    without ever asking the kernel to compare two full permutation states. -/
private theorem extPhase_cons (a : St p) (l : List (St p)) (s : St p) :
    extPhase (a :: l) s = extPhase l (extRound a s) := rfl

/-- Peel one internal round off a phase fold. -/
private theorem intPhase_cons (diag : St p) (a : ZMod p) (l : List (ZMod p)) (s : St p) :
    intPhase diag (a :: l) s = intPhase diag l (intRound diag a s) := rfl

/-- From "every constraint in `(finRange 12).map f` vanishes", read off `f i = 0`. -/
private theorem forall_map_finRange12 {f : Fin 12 → ZMod p}
    (h : ∀ c ∈ (List.finRange 12).map f, c = 0) (i : Fin 12) : f i = 0 :=
  h _ (List.mem_map_of_mem (List.mem_finRange i))

/-- Soundness of a checkpointed external phase: if all its constraints vanish, the
    resulting (checkpoint-read) state equals the checkpoint-free `extPhase`. -/
private theorem extConstrs_sound :
    ∀ (rcs cks : List (St p)) (s : St p), rcs.length = cks.length →
      (∀ c ∈ (extConstrs rcs cks s).1, c = 0) →
      (extConstrs rcs cks s).2 = extPhase rcs s := by
  intro rcs
  induction rcs with
  | nil => intro cks s _ _; rfl
  | cons rc rcs ih =>
    intro cks s hlen h
    cases cks with
    | nil => simp at hlen
    | cons ck cks =>
      simp only [extConstrs] at h
      rw [List.forall_mem_append] at h
      obtain ⟨hc, hr⟩ := h
      -- the head constraints force `ck = addRC rc s`
      have hck : ck = addRC rc s := by
        funext i
        have := forall_map_finRange12 hc i
        have h2 : s i + rc i = ck i := sub_eq_zero.mp this
        simpa [addRC] using h2.symm
      have hstep : mdsLight (sboxAll ck) = extRound rc s := by
        rw [hck]; rfl
      -- recurse on the tail
      have hrec := ih cks (mdsLight (sboxAll ck)) (by simpa using hlen) hr
      simp only [extConstrs]
      rw [hrec, hstep]
      exact (extPhase_cons rc rcs s).symm

/-- Soundness of the checkpointed internal phase. -/
private theorem intConstrs_sound :
    ∀ (rcs : List (ZMod p)) (cks : List (ZMod p)) (s : St p), rcs.length = cks.length →
      (∀ c ∈ (intConstrs diag rcs cks s).1, c = 0) →
      (intConstrs diag rcs cks s).2 = intPhase diag rcs s := by
  intro rcs
  induction rcs with
  | nil => intro cks s _ _; rfl
  | cons rc rcs ih =>
    intro cks s hlen h
    cases cks with
    | nil => simp at hlen
    | cons ck cks =>
      simp only [intConstrs, List.mem_cons, forall_eq_or_imp] at h
      obtain ⟨hc, hr⟩ := h
      have hck : ck = s 0 + rc := (sub_eq_zero.mp hc).symm
      have hstep : internalMix diag (Function.update s 0 (sbox7 ck))
          = intRound diag rc s := by rw [hck]; rfl
      have hrec := ih cks (internalMix diag (Function.update s 0 (sbox7 ck)))
        (by simpa using hlen) hr
      simp only [intConstrs]
      rw [hrec, hstep]
      exact (intPhase_cons diag rc rcs s).symm

/-- Completeness of a checkpointed external phase: there is a choice of checkpoint
    wires making every constraint vanish, with the same resulting state. -/
private theorem extConstrs_complete :
    ∀ (rcs : List (St p)) (s : St p), ∃ cks : List (St p), cks.length = rcs.length ∧
      (∀ c ∈ (extConstrs rcs cks s).1, c = 0) ∧
      (extConstrs rcs cks s).2 = extPhase rcs s := by
  intro rcs
  induction rcs with
  | nil => intro s; exact ⟨[], rfl, by simp [extConstrs], rfl⟩
  | cons rc rcs ih =>
    intro s
    obtain ⟨cks, hlen, hzero, hstate⟩ := ih (extRound rc s)
    refine ⟨addRC rc s :: cks, by simp [hlen], ?_, ?_⟩
    · intro c hc
      simp only [extConstrs] at hc
      rcases List.mem_append.mp hc with h | h
      · obtain ⟨i, _, rfl⟩ := List.mem_map.mp h
        simp [addRC]
      · exact hzero c h
    · simp only [extConstrs]
      exact hstate

/-- Completeness of the checkpointed internal phase. -/
private theorem intConstrs_complete :
    ∀ (rcs : List (ZMod p)) (s : St p), ∃ cks : List (ZMod p), cks.length = rcs.length ∧
      (∀ c ∈ (intConstrs diag rcs cks s).1, c = 0) ∧
      (intConstrs diag rcs cks s).2 = intPhase diag rcs s := by
  intro rcs
  induction rcs with
  | nil => intro s; exact ⟨[], rfl, by simp [intConstrs], rfl⟩
  | cons rc rcs ih =>
    intro s
    obtain ⟨cks, hlen, hzero, hstate⟩ := ih (intRound diag rc s)
    refine ⟨(s 0 + rc) :: cks, by simp [hlen], ?_, ?_⟩
    · intro c hc
      simp only [intConstrs, List.mem_cons] at hc
      rcases hc with rfl | h
      · simp
      · exact hzero c h
    · simp only [intConstrs]
      exact hstate

/-- **Main (3b): the Poseidon2 gate computes the permutation.**

    The 118 gate constraints are satisfiable — over the S-box checkpoint wires
    `ckInit` (initial rounds 1–3), `ckTerm` (terminal rounds), `ckI` (internal
    rounds) — **iff** the output wires are the Poseidon2 permutation of the inputs.

    *Soundness* (→): no witness can force a wrong output. *Completeness* (←): the
    honest assignment is never locked out. Holds for *any* round constants. -/
theorem gate_sound_complete (inp out : St p) :
    (∃ (ckInit ckTerm : List (St p)) (ckI : List (ZMod p)),
        ckInit.length = 3 ∧ ckTerm.length = 4 ∧ ckI.length = 22 ∧
        ∀ c ∈ gateConstraints eInit eTerm iRc diag inp out ckInit ckTerm ckI, c = 0)
      ↔ out = permState eInit eTerm iRc diag inp := by
  constructor
  · rintro ⟨ckInit, ckTerm, ckI, hI, hT, hII, hzero⟩
    simp only [gateConstraints] at hzero
    rw [List.forall_mem_append, List.forall_mem_append, List.forall_mem_append] at hzero
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := hzero
    -- chain the three phase-soundness lemmas, each fed the constraint set verbatim
    have e1 := extConstrs_sound _ _ _ (by simp [hI]) h1
    have e2 := intConstrs_sound _ _ _ (by simp [hII]) h2
    have e3 := extConstrs_sound _ _ _ (by simp [hT]) h3
    -- the output constraints force `out = final state`
    have hout : out = (extConstrs [eTerm 0, eTerm 1, eTerm 2, eTerm 3] ckTerm
        (intConstrs diag ((List.finRange 22).map iRc) ckI
          (extConstrs [eInit 1, eInit 2, eInit 3] ckInit
            (extRound (eInit 0) (mdsLight inp))).2).2).2 := by
      funext i
      exact sub_eq_zero.mp (forall_map_finRange12 h4 i)
    rw [hout, e3, e2, e1, ← extPhase_cons]
    rfl
  · intro hperm
    obtain ⟨ckInit, hIlen, hI0, hIst⟩ :=
      extConstrs_complete [eInit 1, eInit 2, eInit 3] (extRound (eInit 0) (mdsLight inp))
    obtain ⟨ckI, hIIlen, hII0, hIIst⟩ :=
      intConstrs_complete ((List.finRange 22).map iRc)
        (extConstrs [eInit 1, eInit 2, eInit 3] ckInit
          (extRound (eInit 0) (mdsLight inp))).2
    obtain ⟨ckTerm, hTlen, hT0, hTst⟩ :=
      extConstrs_complete [eTerm 0, eTerm 1, eTerm 2, eTerm 3]
        (intConstrs diag ((List.finRange 22).map iRc) ckI
          (extConstrs [eInit 1, eInit 2, eInit 3] ckInit
            (extRound (eInit 0) (mdsLight inp))).2).2
    refine ⟨ckInit, ckTerm, ckI, by simpa using hIlen, by simpa using hTlen,
      by simpa using hIIlen, ?_⟩
    simp only [gateConstraints]
    rw [List.forall_mem_append, List.forall_mem_append, List.forall_mem_append]
    refine ⟨⟨⟨hI0, hII0⟩, hT0⟩, ?_⟩
    -- output constraints: `out i - finalState i = 0`, and `finalState = permState`
    intro c hc
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp hc
    have hfin : (extConstrs [eTerm 0, eTerm 1, eTerm 2, eTerm 3] ckTerm
        (intConstrs diag ((List.finRange 22).map iRc) ckI
          (extConstrs [eInit 1, eInit 2, eInit 3] ckInit
            (extRound (eInit 0) (mdsLight inp))).2).2).2 = permState eInit eTerm iRc diag inp := by
      rw [hTst, hIIst, hIst, ← extPhase_cons]; rfl
    rw [hfin, ← hperm]
    simp

end Meaning

end Plonky2Spec.Poseidon2
