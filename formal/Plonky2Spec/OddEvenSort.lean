/-
  Odd–even transposition sort: `n` alternating layers of adjacent comparators sort any
  list of length `n`.

  This is the sorting-network theorem behind the private-batch permutation gadget
  (`permute_digests4`, `Permutation.lean`): the circuit runs `n` rounds of switches on
  pairs `(0,1),(2,3),…` then `(1,2),(3,4),…`, and the honest prover sets each switch to
  the comparator's decision on the target positions (`permutation_switches`). That this
  realizes *every* permutation is exactly that the comparator network sorts.

  PROOF. The 0-1 principle reduces sorting to lists of booleans (comparator layers commute
  with monotone maps, `run_map`). For booleans we track every `true` by its rank from the
  right, `k = count true (drop i bs)`: after round `t`, a `true` of rank `k ≤ t` at
  position `i` is either final (`i + k = n`) or sits at a comparator-left position for the
  next round (`i % 2 = t % 2`) having advanced every round since round `k`
  (`t ≤ i + k`). The step lemma `inv_step` is a case split on the comparator the `true`
  came through; after `n` rounds every `true` is final, i.e. the list is sorted.
-/
import Mathlib.Data.List.Sort
import Mathlib.Order.Monotone.Basic

namespace Plonky2Spec.OddEven

variable {κ : Type*} [LinearOrder κ]

/-- Comparators on `(0,1), (2,3), …`; an odd tail passes through. -/
def layerEven : List κ → List κ
  | a :: b :: rest => if b < a then b :: a :: layerEven rest else a :: b :: layerEven rest
  | xs => xs

/-- The head passes through, then comparators on `(1,2), (3,4), …`. -/
def layerOdd : List κ → List κ
  | a :: rest => a :: layerEven rest
  | [] => []

/-- Round `t`: the even layer for even `t`, the odd layer for odd `t`. -/
def layer (t : ℕ) (xs : List κ) : List κ := if t % 2 = 0 then layerEven xs else layerOdd xs

/-- Rounds `t, t+1, …, t+r-1`. -/
def run : ℕ → ℕ → List κ → List κ
  | _, 0, xs => xs
  | t, r + 1, xs => run (t + 1) r (layer t xs)

/-! ### Structure: permutation, length, alignment of the comparator pairs -/

theorem layerEven_perm : ∀ xs : List κ, (layerEven xs).Perm xs
  | [] => .refl _
  | [_] => .refl _
  | a :: b :: rest => by
      simp only [layerEven]
      split
      · exact (List.Perm.swap a b _).trans (((layerEven_perm rest).cons b).cons a)
      · exact ((layerEven_perm rest).cons b).cons a

theorem layerOdd_perm (xs : List κ) : (layerOdd xs).Perm xs := by
  cases xs with
  | nil => exact .refl _
  | cons a rest => exact (layerEven_perm rest).cons a

theorem layer_perm (t : ℕ) (xs : List κ) : (layer t xs).Perm xs := by
  unfold layer; split
  · exact layerEven_perm xs
  · exact layerOdd_perm xs

theorem run_perm : ∀ (t r : ℕ) (xs : List κ), (run t r xs).Perm xs
  | _, 0, _ => .refl _
  | t, r + 1, xs => (run_perm (t + 1) r _).trans (layer_perm t xs)

theorem layer_length (t : ℕ) (xs : List κ) : (layer t xs).length = xs.length :=
  (layer_perm t xs).length_eq

theorem run_length (t r : ℕ) (xs : List κ) : (run t r xs).length = xs.length :=
  (run_perm t r xs).length_eq

/-- Comparator pairs of the even layer are aligned with even offsets. -/
theorem layerEven_drop : ∀ (m : ℕ) (xs : List κ),
    (layerEven xs).drop (2 * m) = layerEven (xs.drop (2 * m))
  | 0, _ => by simp
  | m + 1, xs => by
      rw [show 2 * (m + 1) = 2 * m + 1 + 1 by omega]
      match xs with
      | [] => rfl
      | [_] => rfl
      | a :: b :: rest =>
        simp only [layerEven, List.drop_succ_cons]
        split <;> simpa [List.drop_succ_cons] using layerEven_drop m rest

/-- Dropping a prefix aligned with round `t`'s comparators commutes with the layer. -/
theorem layer_drop (t : ℕ) (xs : List κ) (j : ℕ) (hj : j % 2 = t % 2) :
    (layer t xs).drop j = layerEven (xs.drop j) := by
  unfold layer
  split
  · obtain ⟨m, rfl⟩ : ∃ m, j = 2 * m := ⟨j / 2, by omega⟩
    exact layerEven_drop m xs
  · obtain ⟨m, rfl⟩ : ∃ m, j = 2 * m + 1 := ⟨j / 2, by omega⟩
    cases xs with
    | nil => rfl
    | cons a rest =>
      simp only [layerOdd, List.drop_succ_cons]
      exact layerEven_drop m rest

/-! ### 0-1 principle: comparator layers commute with monotone maps -/

variable {κ' : Type*} [LinearOrder κ']

theorem layerEven_map (f : κ → κ') (hf : Monotone f) :
    ∀ xs : List κ, layerEven (xs.map f) = (layerEven xs).map f
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
      simp only [List.map_cons, layerEven]
      by_cases hba : b < a
      · rw [if_pos hba]
        by_cases hfba : f b < f a
        · rw [if_pos hfba, layerEven_map f hf rest, List.map_cons, List.map_cons]
        · rw [if_neg hfba, layerEven_map f hf rest, List.map_cons, List.map_cons,
            le_antisymm (not_lt.mp hfba) (hf hba.le)]
      · have : ¬ f b < f a := not_lt.mpr (hf (not_lt.mp hba))
        rw [if_neg hba, if_neg this, layerEven_map f hf rest, List.map_cons, List.map_cons]

theorem layer_map (f : κ → κ') (hf : Monotone f) (t : ℕ) (xs : List κ) :
    layer t (xs.map f) = (layer t xs).map f := by
  unfold layer
  split
  · exact layerEven_map f hf xs
  · cases xs with
    | nil => rfl
    | cons a rest => simp only [List.map_cons, layerOdd, layerEven_map f hf rest]

theorem run_map (f : κ → κ') (hf : Monotone f) :
    ∀ (t r : ℕ) (xs : List κ), run t r (xs.map f) = (run t r xs).map f
  | _, 0, _ => rfl
  | t, r + 1, xs => by
      simp only [run, layer_map f hf]
      exact run_map f hf (t + 1) r _

/-! ### Booleans: rank tracking -/

theorem layerEven_bool_cons_cons (a b : Bool) (rest : List Bool) :
    layerEven (a :: b :: rest) = (a && b) :: (a || b) :: layerEven rest := by
  cases a <;> cases b <;> simp [layerEven]

/-- After round `t` (i.e. rounds `0..t-1` have run), every `true` of rank `k ≤ t` from the
    right is final or advancing: see the module header. -/
def Inv (t : ℕ) (bs : List Bool) : Prop :=
  ∀ i, i < bs.length → bs[i]? = some true → (bs.drop i).count true ≤ t →
    i + (bs.drop i).count true = bs.length ∨
      (i % 2 = t % 2 ∧ t ≤ i + (bs.drop i).count true)

theorem inv_zero (bs : List Bool) : Inv 0 bs := by
  intro i hi hbi hk
  exfalso
  have : true ∈ bs.drop i := by
    rw [List.drop_eq_getElem_cons hi]
    rw [List.getElem?_eq_getElem hi] at hbi
    rw [Option.some.inj hbi]; exact List.mem_cons_self
  have := List.count_pos_iff.mpr this
  omega

theorem count_drop_le (bs : List Bool) (i : ℕ) : (bs.drop i).count true ≤ bs.length - i := by
  have := List.count_le_length (a := true) (l := bs.drop i)
  simpa using this

theorem inv_step (t : ℕ) (bs : List Bool) (h : Inv t bs) : Inv (t + 1) (layer t bs) := by
  intro i hi hbi hk
  have hlen : (layer t bs).length = bs.length := layer_length t bs
  rw [hlen] at hi ⊢
  rcases eq_or_ne (i % 2) (t % 2) with hpar | hpar
  · -- `i` is a comparator-left position of round `t` (or the unpaired last element).
    have hdrop : (layer t bs).drop i = layerEven (bs.drop i) := layer_drop t bs i hpar
    have hcnt : ((layer t bs).drop i).count true = (bs.drop i).count true := by
      rw [hdrop]; exact (layerEven_perm _).count_eq _
    have hi' : (layer t bs)[i]? = (layerEven (bs.drop i))[0]? := by
      rw [← hdrop, List.getElem?_drop, Nat.add_zero]
    have hlenD : (bs.drop i).length = bs.length - i := List.length_drop
    rw [hbi] at hi'
    rcases hshape : bs.drop i with _ | ⟨a, _ | ⟨b, rest⟩⟩
    · exact absurd (List.drop_eq_nil_iff.mp hshape) (by omega)
    · left
      rw [hshape] at hlenD hi'
      simp only [List.length_singleton] at hlenD
      have ha : a = true := (Option.some.inj hi').symm
      rw [hcnt, hshape, ha, List.count_singleton_self]
      omega
    · rw [hshape] at hlenD hi'
      simp only [List.length_cons] at hlenD
      rw [layerEven_bool_cons_cons] at hi'
      have hab : a = true ∧ b = true := by simpa using hi'
      have hk' : (bs.drop i).count true = 2 + rest.count true := by
        rw [hshape, hab.1, hab.2, List.count_cons_self, List.count_cons_self]; omega
      have hi1 : i + 1 < bs.length := by omega
      have hdrop1 : bs.drop (i + 1) = b :: rest := by
        rw [← List.drop_drop, hshape, List.drop_succ_cons, List.drop_zero]
      have hb1 : bs[i + 1]? = some true := by
        rw [← Nat.add_zero (i + 1), ← List.getElem?_drop, hdrop1, hab.2]; rfl
      have hcnt1 : (bs.drop (i + 1)).count true = 1 + rest.count true := by
        rw [hdrop1, hab.2, List.count_cons_self]; omega
      rcases h (i + 1) hi1 hb1 (by rw [hcnt1]; omega) with hfin | ⟨hp, _⟩
      · left; rw [hcnt, hk']; rw [hcnt1] at hfin; omega
      · exfalso; omega
  · -- `i` is a comparator-right position of round `t`, or the unpaired head.
    rcases Nat.eq_zero_or_pos i with hi0 | hipos
    · subst hi0
      have ht : t % 2 = 1 := by omega
      have hcnt : ((layer t bs).drop 0).count true = (bs.drop 0).count true := by
        simp only [List.drop_zero]; exact (layer_perm t bs).count_eq _
      have hb0 : bs[0]? = some true := by
        have e : layer t bs = layerOdd bs := by unfold layer; rw [if_neg (by omega)]
        rw [e] at hbi
        cases bs with
        | nil => simp at hi
        | cons a rest => simpa [layerOdd] using hbi
      rcases Nat.lt_or_ge ((bs.drop 0).count true) (t + 1) with hlt | hge
      · rcases h 0 (by omega) hb0 (by omega) with hfin | ⟨hp, _⟩
        · left; rwa [hcnt]
        · exfalso; omega
      · right; exact ⟨by omega, by omega⟩
    · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
      have hpj : j % 2 = t % 2 := by omega
      have hdrop : (layer t bs).drop j = layerEven (bs.drop j) := layer_drop t bs j hpj
      have hlenD : (bs.drop j).length = bs.length - j := List.length_drop
      rcases hshape : bs.drop j with _ | ⟨a, _ | ⟨b, rest⟩⟩
      · exact absurd (List.drop_eq_nil_iff.mp hshape) (by omega)
      · rw [hshape] at hlenD; simp only [List.length_singleton] at hlenD; omega
      · rw [hshape] at hlenD
        simp only [List.length_cons] at hlenD
        have hi' : (layer t bs)[j + 1]? = some (a || b) := by
          rw [← List.getElem?_drop, hdrop, hshape, layerEven_bool_cons_cons]; rfl
        rw [hi'] at hbi
        have hab : (a || b) = true := Option.some.inj hbi
        have hk' : ((layer t bs).drop (j + 1)).count true = 1 + rest.count true := by
          have e : (layer t bs).drop (j + 1) = (a || b) :: layerEven rest := by
            rw [← List.drop_drop, hdrop, hshape, layerEven_bool_cons_cons, List.drop_succ_cons,
              List.drop_zero]
          rw [e, hab, List.count_cons_self, (layerEven_perm rest).count_eq]; omega
        have hrest_le : rest.count true ≤ rest.length := List.count_le_length
        rw [hk'] at hk ⊢
        cases b with
        | false =>
          -- the `true` moved here from `j`.
          have ha : a = true := by simpa using hab
          have hbj : bs[j]? = some true := by
            rw [← Nat.add_zero j, ← List.getElem?_drop, hshape, ha]; rfl
          have hkj : (bs.drop j).count true = 1 + rest.count true := by
            rw [hshape, ha, List.count_cons_self, List.count_cons_of_ne (by decide)]; omega
          by_cases hkt : 1 + rest.count true ≤ t
          · rcases h j (by omega) hbj (by rw [hkj]; exact hkt) with hfin | ⟨_, hle⟩
            · exfalso; rw [hkj] at hfin; omega
            · right; rw [hkj] at hle; exact ⟨by omega, by omega⟩
          · right; exact ⟨by omega, by omega⟩
        | true =>
          -- the `true` at `j + 1` stayed (right of its comparator).
          have hbj1 : bs[j + 1]? = some true := by
            rw [← List.getElem?_drop, hshape]; rfl
          have hkj1 : (bs.drop (j + 1)).count true = 1 + rest.count true := by
            rw [← List.drop_drop, hshape, List.drop_succ_cons, List.drop_zero,
              List.count_cons_self]; omega
          by_cases hkt : 1 + rest.count true ≤ t
          · rcases h (j + 1) (by omega) hbj1 (by rw [hkj1]; exact hkt) with hfin | ⟨hp, _⟩
            · left; rw [hkj1] at hfin; omega
            · exfalso; omega
          · right; exact ⟨by omega, by omega⟩

theorem inv_run (bs : List Bool) : ∀ (t r : ℕ), Inv t bs → Inv (t + r) (run t r bs)
  | _, 0, h => h
  | t, r + 1, h => by
      have := inv_run (layer t bs) (t + 1) r (inv_step t bs h)
      rw [show t + (r + 1) = t + 1 + r by omega]
      exact this

/-- **Odd-even transposition sort on booleans**: `n` rounds sort a boolean list of
    length `n`. -/
theorem run_sorted_bool (bs : List Bool) : (run 0 bs.length bs).Pairwise (· ≤ ·) := by
  have hinv : Inv bs.length (run 0 bs.length bs) := by
    have := inv_run bs 0 bs.length (inv_zero bs)
    simpa using this
  set r := run 0 bs.length bs with hr
  have hlen : r.length = bs.length := run_length _ _ _
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  cases hri : r[i] with
  | false => exact Bool.false_le _
  | true =>
    have hle := count_drop_le r i
    have hk := hinv i hi (by rw [List.getElem?_eq_getElem hi, hri]) (by omega)
    have hfin : i + (r.drop i).count true = r.length := by
      rcases hk with hk | ⟨_, hk⟩ <;> omega
    have hall : ∀ b ∈ r.drop i, true = b := by
      rw [← List.count_eq_length]
      simp only [List.length_drop]
      omega
    have hj' : r[j] ∈ r.drop i := by
      have e : i + (j - i) = j := by omega
      have : (r.drop i)[j - i]'(by simp only [List.length_drop]; omega) = r[j] := by
        rw [List.getElem_drop]; simp only [e]
      rw [← this]; exact List.getElem_mem _
    rw [← hall _ hj']

/-- **Odd-even transposition sort**: `n` rounds sort any list of length `n`. -/
theorem run_sorted (xs : List κ) : (run 0 xs.length xs).Pairwise (· ≤ ·) := by
  set r := run 0 xs.length xs with hr
  have hlen : r.length = xs.length := run_length _ _ _
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  by_contra hlt
  rw [not_le] at hlt
  have hf : Monotone (fun x : κ => decide (r[i] ≤ x)) := by
    intro x y hxy
    by_cases hx : r[i] ≤ x
    · simp [hx, le_trans hx hxy]
    · simp [hx]
  have hrun : run 0 xs.length (xs.map fun x : κ => decide (r[i] ≤ x))
      = r.map fun x : κ => decide (r[i] ≤ x) := run_map _ hf _ _ _
  have hs := run_sorted_bool (xs.map fun x : κ => decide (r[i] ≤ x))
  rw [List.length_map, hrun, List.pairwise_iff_getElem] at hs
  have := hs i j (by rw [List.length_map]; exact hi) (by rw [List.length_map]; exact hj) hij
  simp [not_le.mpr hlt] at this
  exact absurd this (by decide)

end Plonky2Spec.OddEven
