/-
  2c — the layer-0 wrapper logic, bridged to the spec's `RL0` building blocks.

  `build_layer0_wrapper_constraints` (qp-zk-circuits, layer0/circuit/circuit_logic.rs)
  is built *entirely* from the T0/T1 gadgets verified in `Arithmetic.lean`/`Boolean.lean`
  (`select`, `and`, `or`, `not`, `is_equal`) plus the Poseidon2 hash (the dummy
  nullifier `H(H(u))`, deferred to Step 3). This module proves each wrapper primitive
  computes *exactly* the conditional/selection that the corresponding `RL0` definition
  in `qp-zk-circuits/formal/WormholeSpec/Aggregation.lean` is built from:

  | circuit primitive (circuit_logic.rs)            | `RL0` definition it realizes        |
  |-------------------------------------------------|-------------------------------------|
  | `select(is_dummy, dummy_null, real)` (328-331)  | `nullifiersReplaced` per-slot `if`  |
  | `select(exit_j==key, amount_j, 0)` (294-296)    | `matchSum` per-element `if`         |
  | `select(is_duplicate, 0, acc)` (300)            | `groupAux` dedup `if k ∈ seen`      |
  | `bytes_digest_eq(block, 0)` (171)               | `isDummyL0 p ↔ blockHash = 0`       |
  | `or(is_dummy, matches) = 1` (224-225)           | `metadataConsistent` block clause   |
  | first-real prefix scan (183-199)                | `referenceFromFirstReal`            |

  The values here are `ZMod p`; the spec is `Felt = Nat` over `RandomOracle`. The full
  cross-package composition (`circuit ⟹ RL0`) additionally needs the `.val` bridge and
  the Poseidon `RandomOracle` instantiation (Steps 3–4), but the *selection logic* — the
  part built from T0/T1 — is discharged here.
-/
import Plonky2Spec.Boolean

namespace Plonky2Spec

variable {p : ℕ} [Fact p.Prime]

/-! ### Per-slot nullifier replacement (`nullifiersReplaced`) -/

/-- `output = select(is_dummy, dummy_null, real)` (circuit_logic.rs:328-331). With
    `is_dummy` boolean this is exactly the per-slot equation of `nullifiersReplaced`:
    `n = if isDummyL0 then dummyNull u else nullifier`. -/
theorem nullifier_replacement {is_dummy dnull real : ZMod p} (h : IsBool is_dummy) :
    bselect is_dummy dnull real = if is_dummy = 1 then dnull else real := by
  rcases h with h | h <;> subst h
  · rw [bselect_false rfl]; simp [zero_ne_one]
  · rw [bselect_true rfl]; simp

/-! ### Per-slot exit-match contribution (`matchSum`) and dedup (`groupAux`) -/

/-- `conditional_amount = select(exit_j == key, amount_j, 0)` (circuit_logic.rs:294-296).
    With the equality flag correct (`eq = 1 ↔ keys match`), this is `matchSum`'s
    per-element `if k = key then amount else 0`. -/
theorem match_contribution {eq amount : ZMod p} {P : Prop} [Decidable P]
    (hb : IsBool eq) (hP : eq = 1 ↔ P) :
    bselect eq amount 0 = if P then amount else 0 := by
  rcases hb with h | h <;> subst h
  · rw [bselect_false rfl, if_neg (fun hp => zero_ne_one (hP.mpr hp))]
  · rw [bselect_true rfl, if_pos (hP.mp rfl)]

/-- `final = select(is_duplicate, 0, acc)` (circuit_logic.rs:300). With `is_duplicate`
    boolean and correct (`= 1 ↔ key already seen`), this is `groupAux`'s dedup branch:
    `if k ∈ seen then 0 else acc`. -/
theorem dedup_select {is_dup acc : ZMod p} {P : Prop} [Decidable P]
    (hb : IsBool is_dup) (hP : is_dup = 1 ↔ P) :
    bselect is_dup 0 acc = if P then 0 else acc := by
  rcases hb with h | h <;> subst h
  · rw [bselect_false rfl, if_neg (fun hp => zero_ne_one (hP.mpr hp))]
  · rw [bselect_true rfl, if_pos (hP.mp rfl)]

/-! ### Digest equality as a conjunction of per-limb equalities (`bytes_digest_eq`) -/

/-- AND-fold of a list of boolean flags — how `bytes_digest_eq` combines the four
    per-limb `is_equal` results into a single digest-equality flag. -/
def andAll : List (ZMod p) → ZMod p
  | []      => 1
  | f :: fs => band f (andAll fs)

theorem andAll_isBool : ∀ {fs : List (ZMod p)}, (∀ f ∈ fs, IsBool f) → IsBool (andAll fs)
  | [],       _ => Or.inr rfl
  | _ :: _fs, h => band_isBool (h _ List.mem_cons_self)
      (andAll_isBool (fun g hg => h g (List.mem_cons_of_mem _ hg)))

/-- The combined digest-equality flag is `1` iff every per-limb flag is `1`. Composed
    with `isEqual_iff` per limb, this gives `digest_eq = 1 ↔ digests equal limbwise`. -/
theorem andAll_eq_one_iff : ∀ {fs : List (ZMod p)}, (∀ f ∈ fs, IsBool f) →
    (andAll fs = 1 ↔ ∀ f ∈ fs, f = 1)
  | [],      _ => by simp [andAll]
  | f :: fs, h => by
      have hf : IsBool f := h f List.mem_cons_self
      have hrest : ∀ g ∈ fs, IsBool g := fun g hg => h g (List.mem_cons_of_mem _ hg)
      rw [andAll, band_eq_one hf (andAll_isBool hrest), andAll_eq_one_iff hrest]
      constructor
      · rintro ⟨hf1, hall⟩ g hg
        rcases List.mem_cons.1 hg with h' | h'
        · exact h' ▸ hf1
        · exact hall g h'
      · intro hall
        exact ⟨hall f List.mem_cons_self, fun g hg => hall g (List.mem_cons_of_mem _ hg)⟩

/-! ### Block / metadata consistency (`metadataConsistent`) -/

/-- `valid = or(is_dummy, matches_ref)` is connected to `one` (circuit_logic.rs:224-225).
    With both flags boolean this forces `is_dummy = 1 ∨ matches_ref = 1`. -/
theorem block_consistency {is_dummy matches_ref : ZMod p}
    (hd : IsBool is_dummy) (hm : IsBool matches_ref)
    (h : bor is_dummy matches_ref = 1) :
    is_dummy = 1 ∨ matches_ref = 1 :=
  (bor_eq_one hd hm).mp h

/-- The `metadataConsistent` conclusion for one child: a non-dummy child's block
    matches the reference. `hmEq` is the per-field `is_equal` correctness (`Boolean`
    + `andAll`), `hcons` the connected-to-one constraint. -/
theorem real_block_matches {α : Type} {blk ref : α} {is_dummy matches_ref : ZMod p}
    (hd : IsBool is_dummy) (hm : IsBool matches_ref)
    (hcons : bor is_dummy matches_ref = 1)
    (hmEq : matches_ref = 1 ↔ blk = ref)
    (hnotdummy : is_dummy ≠ 1) : blk = ref := by
  rcases block_consistency hd hm hcons with h | h
  · exact absurd h hnotdummy
  · exact hmEq.mp h

/-! ### First-real reference selection (`referenceFromFirstReal`) -/

/-- One step of the prefix scan (circuit_logic.rs:186-199), as a state transition on
    `(found, ref)`: `take = and(is_real, not found)`, `ref' = select(take, v, ref)`,
    `found' = or(found, is_real)`. -/
def scanStep (st : ZMod p × ZMod p) (rv : ZMod p × ZMod p) : ZMod p × ZMod p :=
  (bor st.1 rv.1, bselect (band rv.1 (bnot st.1)) rv.2 st.2)

/-- The reference the scan *should* select: the value `v` of the first slot whose
    real-flag is `1`, else the initial reference (an all-dummy batch keeps `init`,
    which the circuit sets to zero — matching `referenceFromFirstReal`'s `none` case). -/
def firstRealVal (init : ZMod p) : List (ZMod p × ZMod p) → ZMod p
  | []      => init
  | rv :: rest => if rv.1 = 1 then rv.2 else firstRealVal init rest

/-- Once a real slot has been found (`found = 1`), the scan is *locked*: no later slot
    overwrites the reference. (No booleanity needed.) -/
theorem scan_locked : ∀ (xs : List (ZMod p × ZMod p)) (ref : ZMod p),
    List.foldl scanStep (1, ref) xs = (1, ref) := by
  intro xs
  induction xs with
  | nil => intro ref; rfl
  | cons rv rest ih =>
      intro ref
      obtain ⟨r, v⟩ := rv
      have hstep : scanStep ((1 : ZMod p), ref) (r, v) = (1, ref) := by
        simp only [scanStep, Prod.mk.injEq]
        refine ⟨?_, ?_⟩
        · simp only [bor]; ring
        · simp only [bselect, band, bnot]; ring
      rw [List.foldl_cons, hstep, ih]

/-- **`referenceFromFirstReal` soundness.** The prefix scan selects exactly the first
    real slot's value (with all flags boolean). Applied per block-hash limb and to the
    block number, this is the position-independent reference selection (the privacy fix
    that makes real and dummy slots indistinguishable). -/
theorem scanFirst_correct : ∀ (xs : List (ZMod p × ZMod p)) (init : ZMod p),
    (∀ rv ∈ xs, IsBool rv.1) →
    (List.foldl scanStep (0, init) xs).2 = firstRealVal init xs := by
  intro xs
  induction xs with
  | nil => intro init _; rfl
  | cons rv rest ih =>
      intro init hb
      obtain ⟨r, v⟩ := rv
      have hr : IsBool r := hb (r, v) List.mem_cons_self
      have hstep : scanStep ((0 : ZMod p), init) (r, v) = (r, bselect r v init) := by
        simp only [scanStep, Prod.mk.injEq]
        refine ⟨?_, ?_⟩
        · simp only [bor]; ring
        · simp only [bselect, band, bnot]; ring
      rw [List.foldl_cons, hstep]
      rcases hr with h0 | h1
      · subst h0
        rw [bselect_false rfl, ih init (fun q hq => hb q (List.mem_cons_of_mem _ hq))]
        simp [firstRealVal, zero_ne_one]
      · subst h1
        rw [bselect_true rfl, scan_locked rest v]
        simp [firstRealVal]

end Plonky2Spec
