/-
  The private nullifier permutation network (`permute_digests4`,
  qp-zk-circuits `common/src/gadgets.rs`).

  The private-batch wrapper routes the per-slot selected nullifiers through an
  odd-even adjacent-swap network before registering them:

      for round in 0..n:
        i = round % 2
        while i + 1 < n:
          swap = add_virtual_bool_target_safe()          -- b·b − b = 0
          routed[i]   = select(swap, routed[i+1], routed[i])   (lanewise, 4 limbs)
          routed[i+1] = select(swap, routed[i],   routed[i+1])
          i += 2

  Each switch either passes its pair through or exchanges the two *complete* digests,
  so for every boolean witness the output is a permutation of the input — nothing is
  modified, dropped, or duplicated. That structural fact is what `RPrivateBatch`'s
  `out.nullifiers.Perm raw` conjunct (and `PrivateBatchCircuit.nullsPerm`) rests on;
  here it is a theorem about the gadget rather than a hypothesis.

  MODEL. A digest is `Fin 4 → ZMod p`. One switch is `digestSelect s hi lo`, the
  lanewise `select(s, hi, lo)` from `Boolean.lean`. A *layer* pairs adjacent elements
  starting at index 0 (`layerEven`) or 1 (`layerOdd`, which passes the head through),
  consuming one switch per pair; the network alternates layers starting even. The
  circuit's flat `Vec<BoolTarget>` is the concatenation of the per-round switch lists.
  The number of rounds is immaterial to the permutation property. The circuit uses
  `n` rounds, and that is enough for the network to realize *every* permutation: this
  is the completeness side, `network_routable` below. The honest witness is the one
  `permutation_switches` computes in Rust — tag each input digest with its target
  position, run the comparator network on the tags (`OddEvenSort.lean`), and set each
  switch to the comparator's decision. Sorting the tags puts the digests in target order.
-/
import Mathlib.Algebra.Field.ZMod
import Plonky2Spec.Boolean
import Plonky2Spec.OddEvenSort

namespace Plonky2Spec

variable {p : ℕ} [Fact p.Prime]

/-- A four-limb digest as the circuit carries it. -/
abbrev Digest4 (p : ℕ) := Fin 4 → ZMod p

/-- One switch output: `select(s, x, y)` on every limb. -/
def digestSelect (s : ZMod p) (x y : Digest4 p) : Digest4 p :=
  fun j => bselect s (x j) (y j)

theorem digestSelect_true {s : ZMod p} {x y : Digest4 p} (h : s = 1) :
    digestSelect s x y = x := by
  funext j; exact bselect_true h

theorem digestSelect_false {s : ZMod p} {x y : Digest4 p} (h : s = 0) :
    digestSelect s x y = y := by
  funext j; exact bselect_false h

/-- A layer starting at index 0: each switch `s` acts on the next adjacent pair
    `(a, b)`, emitting `(select(s, b, a), select(s, a, b))`. Leftovers (an odd tail or
    exhausted switches) pass through. -/
def layerEven : List (ZMod p) → List (Digest4 p) → List (Digest4 p)
  | s :: ss, a :: b :: rest => digestSelect s b a :: digestSelect s a b :: layerEven ss rest
  | _, xs => xs

/-- A layer starting at index 1: the head passes through, then an even layer. -/
def layerOdd (ss : List (ZMod p)) : List (Digest4 p) → List (Digest4 p)
  | a :: rest => a :: layerEven ss rest
  | [] => []

/-- The network: rounds alternate even/odd, starting even (`round % 2`). -/
def networkAux : Bool → List (List (ZMod p)) → List (Digest4 p) → List (Digest4 p)
  | _, [], xs => xs
  | odd, ss :: rounds, xs =>
      networkAux (!odd) rounds (if odd then layerOdd ss xs else layerEven ss xs)

/-- `permute_digests4` with per-round switch lists `rounds`. -/
def network (rounds : List (List (ZMod p))) (xs : List (Digest4 p)) : List (Digest4 p) :=
  networkAux false rounds xs

/-- **One layer is a permutation** (boolean switches). -/
theorem layerEven_perm : ∀ (ss : List (ZMod p)) (xs : List (Digest4 p)),
    (∀ s ∈ ss, IsBool s) → (layerEven ss xs).Perm xs
  | [], xs, _ => by
      cases xs with
      | nil => exact List.Perm.refl _
      | cons a rest =>
          cases rest with
          | nil => exact List.Perm.refl _
          | cons b rest => exact List.Perm.refl _
  | _ :: _, [], _ => List.Perm.refl _
  | _ :: _, [a], _ => List.Perm.refl _
  | s :: ss, a :: b :: rest, hb => by
      have hs : IsBool s := hb s List.mem_cons_self
      have ih := layerEven_perm ss rest (fun t ht => hb t (List.mem_cons_of_mem _ ht))
      show (digestSelect s b a :: digestSelect s a b :: layerEven ss rest).Perm (a :: b :: rest)
      rcases hs with h0 | h1
      · rw [digestSelect_false h0, digestSelect_false h0]
        exact (ih.cons b).cons a
      · rw [digestSelect_true h1, digestSelect_true h1]
        exact (List.Perm.swap a b _).trans ((ih.cons b).cons a)

theorem layerOdd_perm (ss : List (ZMod p)) (xs : List (Digest4 p))
    (hb : ∀ s ∈ ss, IsBool s) : (layerOdd ss xs).Perm xs := by
  cases xs with
  | nil => exact List.Perm.refl _
  | cons a rest => exact (layerEven_perm ss rest hb).cons a

theorem networkAux_perm : ∀ (odd : Bool) (rounds : List (List (ZMod p))) (xs : List (Digest4 p)),
    (∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s) → (networkAux odd rounds xs).Perm xs
  | _, [], _, _ => List.Perm.refl _
  | odd, ss :: rounds, xs, hb => by
      have hss : ∀ s ∈ ss, IsBool s := hb ss List.mem_cons_self
      have hrest : ∀ ss' ∈ rounds, ∀ s ∈ ss', IsBool s :=
        fun ss' h => hb ss' (List.mem_cons_of_mem _ h)
      show (networkAux (!odd) rounds (if odd then layerOdd ss xs else layerEven ss xs)).Perm xs
      refine (networkAux_perm (!odd) rounds _ hrest).trans ?_
      cases odd
      · exact layerEven_perm ss xs hss
      · exact layerOdd_perm ss xs hss

/-- **`permute_digests4` is a permutation.** For every witness whose switches are all
    boolean (the `add_virtual_bool_target_safe` constraint `b·b − b = 0`, i.e. `IsBool`
    via `isBool_iff_assertBool`), the routed output is a permutation of the input
    digests. This discharges `RPrivateBatch`'s `Perm` conjunct at the gadget level. -/
theorem network_perm (rounds : List (List (ZMod p))) (xs : List (Digest4 p))
    (hb : ∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s) : (network rounds xs).Perm xs :=
  networkAux_perm false rounds xs hb

/-- The same, phrased on the raw `assert_bool` constraints the circuit emits. -/
theorem network_perm_of_assertBool (rounds : List (List (ZMod p))) (xs : List (Digest4 p))
    (hb : ∀ ss ∈ rounds, ∀ s ∈ ss, s * s - s = 0) : (network rounds xs).Perm xs :=
  network_perm rounds xs (fun ss hss s hs => isBool_iff_assertBool.mpr (hb ss hss s hs))

/-- Length preservation (a permutation), the fact `RPrivateBatch`'s
    `out.nullifiers.length = leaves.length` conjunct uses. -/
theorem network_length (rounds : List (List (ZMod p))) (xs : List (Digest4 p))
    (hb : ∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s) : (network rounds xs).length = xs.length :=
  (network_perm rounds xs hb).length_eq

/-! ### Completeness: the `n`-round network realizes every permutation

Digests are tagged with a target position (`ℕ × Digest4 p`); `keyedLayerEven` is the
comparator layer on tags and `switchesEven` the switch bits it decides, so that the
digest network driven by those bits computes exactly the tag-sorting network's digest
column (`layerEven_switchesEven`). -/

/-- The comparator layer on tagged digests: swap when the right tag is smaller. -/
def keyedLayerEven : List (ℕ × Digest4 p) → List (ℕ × Digest4 p)
  | a :: b :: rest =>
      if b.1 < a.1 then b :: a :: keyedLayerEven rest else a :: b :: keyedLayerEven rest
  | xs => xs

def keyedLayerOdd : List (ℕ × Digest4 p) → List (ℕ × Digest4 p)
  | a :: rest => a :: keyedLayerEven rest
  | [] => []

/-- The switch bits the even layer decides, one per adjacent pair. -/
def switchesEven : List (ℕ × Digest4 p) → List (ZMod p)
  | a :: b :: rest => (if b.1 < a.1 then 1 else 0) :: switchesEven rest
  | _ => []

def switchesOdd : List (ℕ × Digest4 p) → List (ZMod p)
  | _ :: rest => switchesEven rest
  | [] => []

def keyedLayer (t : ℕ) (xs : List (ℕ × Digest4 p)) : List (ℕ × Digest4 p) :=
  if t % 2 = 0 then keyedLayerEven xs else keyedLayerOdd xs

def switches (t : ℕ) (xs : List (ℕ × Digest4 p)) : List (ZMod p) :=
  if t % 2 = 0 then switchesEven xs else switchesOdd xs

/-- Rounds `t, …, t+r-1` on tagged digests. -/
def keyedRun : ℕ → ℕ → List (ℕ × Digest4 p) → List (ℕ × Digest4 p)
  | _, 0, xs => xs
  | t, r + 1, xs => keyedRun (t + 1) r (keyedLayer t xs)

/-- The per-round switch lists of rounds `t, …, t+r-1` (the honest routing witness). -/
def switchRounds : ℕ → ℕ → List (ℕ × Digest4 p) → List (List (ZMod p))
  | _, 0, _ => []
  | t, r + 1, xs => switches t xs :: switchRounds (t + 1) r (keyedLayer t xs)

omit [Fact p.Prime] in
theorem keyedLayerEven_map_fst : ∀ xs : List (ℕ × Digest4 p),
    (keyedLayerEven xs).map Prod.fst = OddEven.layerEven (xs.map Prod.fst)
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
      simp only [keyedLayerEven, OddEven.layerEven, List.map_cons]
      split <;> simp [keyedLayerEven_map_fst rest]

omit [Fact p.Prime] in
theorem keyedLayer_map_fst (t : ℕ) (xs : List (ℕ × Digest4 p)) :
    (keyedLayer t xs).map Prod.fst = OddEven.layer t (xs.map Prod.fst) := by
  unfold keyedLayer OddEven.layer
  split
  · exact keyedLayerEven_map_fst xs
  · cases xs with
    | nil => rfl
    | cons a rest => simp [keyedLayerOdd, OddEven.layerOdd, keyedLayerEven_map_fst rest]

omit [Fact p.Prime] in
theorem keyedRun_map_fst : ∀ (t r : ℕ) (xs : List (ℕ × Digest4 p)),
    (keyedRun t r xs).map Prod.fst = OddEven.run t r (xs.map Prod.fst)
  | _, 0, _ => rfl
  | t, r + 1, xs => by
      simp only [keyedRun, OddEven.run, ← keyedLayer_map_fst]
      exact keyedRun_map_fst (t + 1) r _

omit [Fact p.Prime] in
theorem keyedLayerEven_perm : ∀ xs : List (ℕ × Digest4 p), (keyedLayerEven xs).Perm xs
  | [] => .refl _
  | [_] => .refl _
  | a :: b :: rest => by
      simp only [keyedLayerEven]
      split
      · exact (List.Perm.swap a b _).trans (((keyedLayerEven_perm rest).cons b).cons a)
      · exact ((keyedLayerEven_perm rest).cons b).cons a

omit [Fact p.Prime] in
theorem keyedLayer_perm (t : ℕ) (xs : List (ℕ × Digest4 p)) : (keyedLayer t xs).Perm xs := by
  unfold keyedLayer
  split
  · exact keyedLayerEven_perm xs
  · cases xs with
    | nil => exact .refl _
    | cons a rest => exact (keyedLayerEven_perm rest).cons a

omit [Fact p.Prime] in
theorem keyedRun_perm : ∀ (t r : ℕ) (xs : List (ℕ × Digest4 p)), (keyedRun t r xs).Perm xs
  | _, 0, _ => .refl _
  | t, r + 1, xs => (keyedRun_perm (t + 1) r _).trans (keyedLayer_perm t xs)

theorem switchesEven_bool : ∀ xs : List (ℕ × Digest4 p), ∀ s ∈ switchesEven xs, IsBool s
  | [], _, h => absurd h List.not_mem_nil
  | [_], _, h => absurd h List.not_mem_nil
  | a :: b :: rest, s, h => by
      simp only [switchesEven, List.mem_cons] at h
      rcases h with rfl | h
      · split
        · exact Or.inr rfl
        · exact Or.inl rfl
      · exact switchesEven_bool rest s h

theorem switches_bool (t : ℕ) (xs : List (ℕ × Digest4 p)) : ∀ s ∈ switches t xs, IsBool s := by
  unfold switches
  split
  · exact switchesEven_bool xs
  · cases xs with
    | nil => intro s h; exact absurd h List.not_mem_nil
    | cons a rest => exact switchesEven_bool rest

theorem switchRounds_bool : ∀ (t r : ℕ) (xs : List (ℕ × Digest4 p)),
    ∀ ss ∈ switchRounds t r xs, ∀ s ∈ ss, IsBool s
  | _, 0, _, _, h => absurd h List.not_mem_nil
  | t, r + 1, xs, ss, h => by
      simp only [switchRounds, List.mem_cons] at h
      rcases h with rfl | h
      · exact switches_bool t xs
      · exact switchRounds_bool (t + 1) r _ ss h

theorem switchRounds_length : ∀ (t r : ℕ) (xs : List (ℕ × Digest4 p)),
    (switchRounds t r xs).length = r
  | _, 0, _ => rfl
  | t, r + 1, xs => by simp [switchRounds, switchRounds_length (t + 1) r]

/-- The digest network driven by the decided switches is the tag-sorting layer's digest
    column. -/
theorem layerEven_switchesEven : ∀ xs : List (ℕ × Digest4 p),
    layerEven (switchesEven xs) (xs.map Prod.snd) = (keyedLayerEven xs).map Prod.snd
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
      simp only [switchesEven, keyedLayerEven, List.map_cons, layerEven]
      split
      · rw [digestSelect_true rfl, digestSelect_true rfl, layerEven_switchesEven rest]
        simp
      · rw [digestSelect_false rfl, digestSelect_false rfl, layerEven_switchesEven rest]
        simp

theorem layer_switches (t : ℕ) (xs : List (ℕ × Digest4 p)) :
    (if t % 2 = 1 then layerOdd (switches t xs) (xs.map Prod.snd)
      else layerEven (switches t xs) (xs.map Prod.snd)) = (keyedLayer t xs).map Prod.snd := by
  unfold switches keyedLayer
  rcases Nat.mod_two_eq_zero_or_one t with h | h
  · simp [h, layerEven_switchesEven]
  · simp only [h, if_true, one_ne_zero, if_false]
    cases xs with
    | nil => rfl
    | cons a rest =>
      simp only [switchesOdd, keyedLayerOdd, List.map_cons, layerOdd, layerEven_switchesEven rest]

theorem networkAux_switchRounds : ∀ (t r : ℕ) (xs : List (ℕ × Digest4 p)),
    networkAux (decide (t % 2 = 1)) (switchRounds t r xs) (xs.map Prod.snd)
      = (keyedRun t r xs).map Prod.snd
  | _, 0, _ => rfl
  | t, r + 1, xs => by
      have hpar : (!decide (t % 2 = 1)) = decide ((t + 1) % 2 = 1) := by
        rcases Nat.mod_two_eq_zero_or_one t with h | h <;> simp [h, Nat.add_mod]
      simp only [switchRounds, keyedRun, networkAux, hpar]
      rw [← networkAux_switchRounds (t + 1) r (keyedLayer t xs), ← layer_switches]
      congr 1
      split <;> simp_all

omit [Fact p.Prime] in
/-- A permutation of a mapped list lifts to a permutation of the source list. -/
theorem exists_perm_of_perm_map {α β : Type*} (f : α → β) :
    ∀ (l : List α) (l₂ : List β), l₂.Perm (l.map f) → ∃ l' : List α, l'.Perm l ∧ l'.map f = l₂
  | l, [], h => ⟨[], by
      have : l.map f = [] := h.symm.eq_nil
      rw [List.map_eq_nil_iff.mp this], rfl⟩
  | l, y :: l₂, h => by
      have hy : y ∈ l.map f := h.subset List.mem_cons_self
      obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hy
      obtain ⟨s, u, rfl⟩ := List.append_of_mem ha
      have h' : l₂.Perm ((s ++ u).map f) := by
        have : (f a :: l₂).Perm (f a :: (s ++ u).map f) := by
          refine h.trans ?_
          simp only [List.map_append, List.map_cons]
          exact List.perm_middle
        exact this.cons_inv
      obtain ⟨l', hl', hmap⟩ := exists_perm_of_perm_map f (s ++ u) l₂ h'
      exact ⟨a :: l', (hl'.cons a).trans List.perm_middle.symm, by simp [hmap]⟩

/-- **`permute_digests4` realizes every permutation in `n` rounds.** For any
    rearrangement `ys` of `xs` there is a boolean switch witness with exactly
    `xs.length` rounds (the circuit's `for round in 0..n`) routing `xs` to `ys`. -/
theorem network_routable (xs ys : List (Digest4 p)) (h : ys.Perm xs) :
    ∃ rounds : List (List (ZMod p)), rounds.length = xs.length ∧
      (∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s) ∧ network rounds xs = ys := by
  set n := xs.length with hn
  have hylen : ys.length = n := h.length_eq
  -- Tag the target list with its positions.
  set ys' : List (ℕ × Digest4 p) := (List.range n).zip ys with hys'
  have hys'_snd : ys'.map Prod.snd = ys := List.map_snd_zip (by simp [hylen])
  have hys'_fst : ys'.map Prod.fst = List.range n := List.map_fst_zip (by simp [hylen])
  -- The tagged source: a permutation of the tagged target whose digest column is `xs`.
  obtain ⟨xs', hperm, hsnd⟩ := exists_perm_of_perm_map Prod.snd ys' xs (by rw [hys'_snd]; exact h.symm)
  have hxs'len : xs'.length = n := by rw [hn, ← hsnd, List.length_map]
  refine ⟨switchRounds 0 n xs', switchRounds_length 0 n xs', switchRounds_bool 0 n xs', ?_⟩
  -- The network computes the tag-sorted digest column …
  have hnet : network (switchRounds 0 n xs') xs = (keyedRun 0 n xs').map Prod.snd := by
    rw [← hsnd, network]
    exact networkAux_switchRounds 0 n xs'
  -- … and sorting the tags (a permutation of `range n`) recovers `ys'`.
  have hsorted : (keyedRun 0 n xs').Pairwise (fun a b => a.1 < b.1) := by
    have hle : ((keyedRun 0 n xs').map Prod.fst).Pairwise (· ≤ ·) := by
      rw [keyedRun_map_fst, ← hxs'len, ← List.length_map]
      exact OddEven.run_sorted _
    have hnd : ((keyedRun 0 n xs').map Prod.fst).Nodup := by
      have : ((keyedRun 0 n xs').map Prod.fst).Perm (List.range n) := by
        rw [← hys'_fst]
        exact ((keyedRun_perm 0 n xs').trans hperm).map Prod.fst
      exact this.nodup_iff.mpr (List.nodup_range)
    rw [List.pairwise_map] at hle
    rw [List.Nodup, List.pairwise_map] at hnd
    exact (hle.and hnd).imp fun ⟨h1, h2⟩ => lt_of_le_of_ne h1 h2
  have hys'sorted : ys'.Pairwise (fun a b => a.1 < b.1) := by
    rw [← List.pairwise_map, hys'_fst]
    exact List.pairwise_lt_range
  have heq : keyedRun 0 n xs' = ys' :=
    List.Perm.eq_of_pairwise (fun _ _ _ _ h1 h2 => absurd h2 (lt_asymm h1)) hsorted hys'sorted
      ((keyedRun_perm 0 n xs').trans hperm)
  rw [hnet, heq, hys'_snd]

end Plonky2Spec
