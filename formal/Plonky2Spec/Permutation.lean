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
  The number of rounds is immaterial to the permutation property (the circuit uses
  `n`, enough for the odd-even network to realize every permutation, which is the
  completeness side — `permutation_switches` in Rust computes the witness).
-/
import Mathlib.Algebra.Field.ZMod
import Plonky2Spec.Boolean

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

end Plonky2Spec
