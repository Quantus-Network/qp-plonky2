/-
  The trusted base (PLAN.md §7, Steps 12 to 18).

  `proof_sound` is a theorem. It is the composition of three assumptions and
  `verifier_algebra`, and the capstones depend on nothing else outside the standard Lean
  axioms. Each assumption is an implication on a good event of the execution
  (`GoodEvents`, Step 15): the statements are not true of every proof, and the events
  are where their probabilities live.

  * `poseidon_transcript_oracle` — the proof-system Poseidon is a random oracle on
    the verifier transcript (QROM for the post-quantum reading). Merkle binding is a
    corollary. This is `PoseidonGoldilocksConfig`'s Poseidon, not the wormhole Poseidon2.
  * `fri_proximity_gap` — FRI proximity gaps. The query error is statistical
    (`2^{-84}` for the default config). Grinding is a separate work factor.
  * `algebraic_sampling` — the point checks imply identical constraints only on a
    good event of probability `1 − ε`. `ε` adds the base-field challenge failures
    (`α`, `β`, `γ`, repeated `num_challenges` times) to the extension-field opening
    at `ζ`. Both remain after ideal hashes and perfect FRI.
  * `verifier_algebra` — the remainder: identical constraints are a satisfying
    assignment of the exported circuit. It is a theorem. `IdenticalConstraints` is the
    gate, copy, and permutation identities of the built system on its whole domain, and
    `Built.Refines.satisfies` proves they give `Satisfies` for an export that system
    refines, and `poseidon2_gate_rows` proves the Poseidon2 rows compute `perm` from
    `Poseidon2.gate_sound_complete`. Four obligations remain as declarations:
    `build_refines_export`, `public_inputs_wired`, `child_gadget_satisfied`, and
    `recursive_verifier_transcript`. They are deterministic. The public inputs
    additionally need `public_input_binding`: on its good event, the proof-system hash
    does not collide them with the list the verifier was given.
  * Certified builds (Step 16). `BuildCert` is what the exporter reads off
    `CircuitBuilder::build`, `Built.Agrees` says a built system has it, and
    `Built.Agrees.refines` proves `Built.Refines` from it and kernel-evaluated checks of
    the certificate against the export. `verifier_algebra_of` and `proof_sound_of` take
    the refinement as a hypothesis; `proof_sound_leaf` is their instance for a certified
    circuit with no children, with neither `build_refines_export` nor the two child
    obligations in its footprint. The leaf is certified in `Plonky2Bridge/LeafBuild.lean`.
  * The public-input hash (Steps 17, 18). `Built.HashesPublicInputs` is the in-circuit
    sponge `build` lays down over `PoseidonGate` rows and copies into the
    `PublicInputGate`, `Built.HashesPublicInputs.hash_eq` proves the carried public
    inputs hash to `pis` from it, and `Built.Agrees.hashesPublicInputs` proves it from a
    certificate. `poseidonPerm` is the proof-system permutation on the Goldilocks tables
    (`Plonky2Spec.Poseidon`), and `poseidon_gate_rows` proves a `PoseidonGate` row
    computes it from `Poseidon.gate_sound_complete`. Nothing of the hash is trusted;
    `public_inputs_wired` is the obligation for an uncertified build.

  Poseidon2 collision resistance is not an assumption here. A forged proof need not
  exhibit a collision. The wormhole relations that do reduce to a collision are
  `*_or_collision` against `spongeRO` in the spec package.
-/
import Plonky2Spec.Wiring
import Plonky2Spec.Poseidon2
import Plonky2Spec.Generated.PoseidonConsts
import Plonky2Spec.Sponge

namespace Plonky2Bridge

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (rate squeeze4)

/-! ### FRI parameters the in-repo default config is chosen against

  `CircuitConfig.standard_recursion_config` uses blowup `2^rateBits`, `numQueries`
  FRI queries and `proofOfWorkBits` of grinding. These are different quantities.
  `friQueryBits` is the statistical query exponent: each query fails with
  probability at most the rate, so the default is `(1/8)^28 = 2^{-84}`.
  `friConjecturedBits` adds the grinding bits. That sum is the work factor stored in
  `CircuitConfig.security_bits`. It is not a statistical error of `2^{-100}`: the
  grinding term depends on how many times the adversary queries the oracle, as
  separated in Appendix A.3.2 of *Fiat-Shamir Security of FRI and Related SNARKs*.
  The `rfl` checks certify only the arithmetic. The deployed wormhole circuit may
  use a different `FriConfig`. -/

/-- The FRI query parameters that determine the conjectured soundness error. -/
structure FriQueryParams where
  rateBits : ℕ
  numQueries : ℕ
  proofOfWorkBits : ℕ

/-- `CircuitConfig.standard_recursion_config`'s FRI query parameters. -/
def standardRecursionFri : FriQueryParams where
  rateBits := 3
  numQueries := 28
  proofOfWorkBits := 16

/-- Statistical query exponent: `numQueries * rateBits`, i.e. `rate^{numQueries} = 2^{-this}`. -/
def friQueryBits (q : FriQueryParams) : ℕ :=
  q.numQueries * q.rateBits

/-- Query exponent plus grinding bits. This is `CircuitConfig.security_bits`'s accounting,
    not a single statistical probability. -/
def friConjecturedBits (q : FriQueryParams) : ℕ :=
  friQueryBits q + q.proofOfWorkBits

example : friQueryBits standardRecursionFri = 84 := rfl
example : friConjecturedBits standardRecursionFri = 100 := rfl

/-! ### The built system and its domain

  `Plonky2Spec.Wiring.Circuit` is a builder's pre-`build` state. A verifier key commits to
  what `build` makes of it, and the identities a transcript checks are identities of that
  system, not of the export. `Built` is that system as the identities read it, `Opened` the
  values on its domain of the polynomials a transcript opens, and `Built.Refines` the
  relation to the export under which those identities give `Satisfies`. This section is
  definitions and theorems; it assumes nothing. -/

section Built

variable {p : ℕ}

/-- What `CircuitBuilder::build` produces, as far as the verifier's identities read it,
    with the place of the export's targets in it. Row `i` is the domain point `g^i`;
    `build` pads the rows to a power of two. -/
structure Built (p : ℕ) where
  /-- The gate and the local constants on each row of the domain. -/
  rows : List (Row p)
  /-- `num_routed_wires`: the columns the permutation argument covers. -/
  routed : ℕ
  /-- `config.num_challenges`: the rounds of the permutation argument, each with its own
      `β`, `γ` and grand product. -/
  numChallenges : ℕ
  /-- `num_public_inputs`: the length the verifier requires of the public-input list
      (`validate_proof_with_pis_shape`). -/
  numPublicInputs : ℕ
  /-- The wire permutation `σ` on positions `(row, column)`: `get_sigma_map` sends a routed
      wire to the next one of its copy class. -/
  sigma : ℕ × ℕ → ℕ × ℕ
  /-- The label `k_column · g^row` of a position. `σ` is committed as `label ∘ sigma`. -/
  label : ℕ × ℕ → ZMod p
  /-- The routed constants: the wires a gate constrains to one of its constants
      (`ConstantGate`, and the spare constant wires of `Gate::extra_constant_wires`), each
      with the constant `build` gives it. -/
  fixed : List ((ℕ × ℕ) × ZMod p)
  /-- Where a target of the export sits in this system: the wire a wire target became, a
      routed wire of its copy class for a virtual target. `none` when there is no such
      wire: a virtual target that nothing routes, or a wire of an arithmetic op slot
      that is not in use. -/
  loc : Target → Option (ℕ × ℕ)

/-- One round of the permutation argument: its challenges and the grand product `Z` on
    the domain. -/
structure GrandProduct (p : ℕ) where
  beta : ZMod p
  gamma : ZMod p
  z : ℕ → ZMod p

/-- The values on the domain of the polynomials a transcript opens: `wire i j` is wire
    polynomial `j` at `g^i`, and `products` has, for each challenge round in order, that
    round's `β`, `γ` and grand product. -/
structure Opened (p : ℕ) where
  wire : ℕ → ℕ → ZMod p
  products : List (GrandProduct p)

/-- The wire value at a position. -/
def Opened.val (o : Opened p) (u : ℕ × ℕ) : ZMod p := o.wire u.1 u.2

/-- The wire values as an assignment of the built system's own rows. -/
def Opened.assignment (o : Opened p) : Assignment p
  | .wire i j => o.wire i j
  | .virt _ => 0

/-- **The witness.** The assignment of the export's targets that the opened values are:
    each target reads the wire it sits on, and an unplaced target reads `0`. -/
def Built.witness (b : Built p) (o : Opened p) : Assignment p := fun t =>
  match b.loc t with
  | some u => o.val u
  | none => 0

/-- One step of `σ` from a routed position of the domain. -/
def Built.Step (b : Built p) (u v : ℕ × ℕ) : Prop :=
  u.1 < b.rows.length ∧ u.2 < b.routed ∧ b.sigma u = v

/-- `σ` leads from `u` to `v` through routed positions of the domain. -/
def Built.Chain (b : Built p) : ℕ × ℕ → ℕ × ℕ → Prop := Relation.ReflTransGen b.Step

/-- **Copy identities.** The wire values are invariant under `σ` at every routed position
    of the domain. -/
def CopyIdentities (b : Built p) (o : Opened p) : Prop :=
  ∀ u : ℕ × ℕ, u.1 < b.rows.length → u.2 < b.routed → o.val (b.sigma u) = o.val u

/-- **Permutation identities.** There is one grand product per challenge round of the
    built system. Each starts at `1` and steps by the ratio of the row's `wire + β·id + γ`
    to its `wire + β·σ + γ`, on every row of the domain. The verifier checks this through
    partial products (`check_partial_products`); multiplying its chunk equations out
    gives the step stated here.

    The `β`, `γ` of a round are not constrained here. For `β = 0` and `Z = 1` the step
    holds of any wires, so the identities mean something only for the challenges a
    transcript derived, which is what `TranscriptOpens` says of `o`. -/
def PermutationIdentities (b : Built p) (o : Opened p) : Prop :=
  o.products.length = b.numChallenges ∧
    ∀ g ∈ o.products, g.z 0 = 1 ∧ ∀ i < b.rows.length,
      g.z i * ∏ j ∈ Finset.range b.routed, (o.wire i j + g.beta * b.label (i, j) + g.gamma) =
        g.z ((i + 1) % b.rows.length) *
          ∏ j ∈ Finset.range b.routed,
            (o.wire i j + g.beta * b.label (b.sigma (i, j)) + g.gamma)

/-- The built system forces the export's targets `x` and `y` equal: both are unplaced, or
    `σ` leads from the wire of `x` to the wire of `y`. -/
def Built.Linked (b : Built p) (x y : Target) : Prop :=
  b.loc x = b.loc y ∨ ∃ u v, b.loc x = some u ∧ b.loc y = some v ∧ b.Chain u v

/-- The built system forces the export's target `t` to the constant `k`: `σ` leads from
    its wire to a routed constant `k`. -/
def Built.Pins (b : Built p) (t : Target) (k : ZMod p) : Prop :=
  ∃ u v, b.loc t = some u ∧ b.Chain u v ∧ (v, k) ∈ b.fixed

/-- Op `k` of the export's arithmetic row `i`, with constants `c0`, `c1`, is an op of
    the built system with the same constants, or none of its four wires is placed. -/
def Built.HostsOp (b : Built p) (i k : ℕ) (c0 c1 : ZMod p) : Prop :=
  (∃ i' n cs k', b.rows[i']? = some ⟨.arithmetic n, cs⟩ ∧ k' < n ∧
      cs.getD 0 0 = c0 ∧ cs.getD 1 0 = c1 ∧
      ∀ m < 4, b.loc (.wire i (4 * k + m)) = some (i', 4 * k' + m)) ∨
    ∀ m < 4, b.loc (.wire i (4 * k + m)) = none

/-- The export's row `i` is a row of the built system with the same gate, column for
    column on its first `cols` columns. -/
def Built.RowAt (b : Built p) (i : ℕ) (kind : GateKind) (cols : ℕ) : Prop :=
  ∃ i' cs, b.rows[i']? = some ⟨kind, cs⟩ ∧ ∀ j < cols, b.loc (.wire i j) = some (i', j)

/-- The built system carries the export's row `i`. Arithmetic rows are carried op by op,
    since the builder packs ops into whatever row has a free slot. Rows of the gates
    `rowConstraints` reads as `True` need not be carried, except `Poseidon2Gate` rows,
    which `Poseidon2Rows` reads. A `ConstantGate` row is read as pinned wires; the
    exports here have none, since `build` is what adds them.

    A base-sum or Poseidon2 row is placed on the columns its constraints here read: the
    sum and the `n` limbs, and the `12` inputs and `12` outputs. The other columns of the
    row, and the columns past the circuit's width, have no place to be given. -/
def Built.Hosts (b : Built p) (i : ℕ) (r : Row p) : Prop :=
  match r.kind with
  | .arithmetic n => ∀ k < n, b.HostsOp i k (r.consts.getD 0 0) (r.consts.getD 1 0)
  | .constant n => ∀ j < n, b.Pins (.wire i j) (r.consts.getD j 0)
  | .baseSum2 n => b.RowAt i (.baseSum2 n) (n + 1)
  | .poseidon2 => b.RowAt i .poseidon2 24
  | .poseidon => True
  | .publicInput => True
  | .noop => True
  | .other _ => True

/-- **The built system refines the export `c`.** Every row, copy and constant target of
    `c` is carried by the built system, at the places `loc` gives, and the two have the
    same number of public inputs. It is not an inclusion of rows: the built system may
    hold the export's rows at other indices, between rows of its own. -/
structure Built.Refines (b : Built p) (c : Circuit p) : Prop where
  rows : ∀ i r, c.rows[i]? = some r → b.Hosts i r
  copies : ∀ xy ∈ c.copies, b.Linked xy.1 xy.2
  constants : ∀ tc ∈ c.constants, b.Pins tc.1 tc.2
  publicInputs : b.numPublicInputs = c.publicInputs.length

theorem Built.witness_of_loc {b : Built p} {t : Target} {u : ℕ × ℕ} (o : Opened p)
    (h : b.loc t = some u) : b.witness o t = o.val u := by
  simp only [Built.witness, h]

theorem Built.witness_of_unplaced {b : Built p} {t : Target} (o : Opened p)
    (h : b.loc t = none) : b.witness o t = 0 := by
  simp only [Built.witness, h]

/-- Under the copy identities the wire values agree along `σ`. -/
theorem Built.Chain.val_eq {b : Built p} {o : Opened p} (hc : CopyIdentities b o)
    {u v : ℕ × ℕ} (h : b.Chain u v) : o.val v = o.val u := by
  induction h with
  | refl => rfl
  | tail _ hs ih =>
    obtain ⟨h1, h2, rfl⟩ := hs
    exact (hc _ h1 h2).trans ih

/-- Under the copy identities the witness agrees on linked targets. -/
theorem Built.Linked.witness_eq {b : Built p} {o : Opened p} (hc : CopyIdentities b o)
    {x y : Target} (h : b.Linked x y) : b.witness o x = b.witness o y := by
  rcases h with h | ⟨u, v, hx, hy, huv⟩
  · simp only [Built.witness, h]
  · rw [Built.witness_of_loc o hx, Built.witness_of_loc o hy, huv.val_eq hc]

/-! ### Certified builds

  `Built.Refines` is about the whole built system. What the exporter reads off
  `CircuitBuilder::build`, and what a proof of `Refines` needs, is less: the export's rows
  at their indices, the `σ`-cycles the export's copies and constants run along, the place
  of each virtual target, and the wires `build` fixes to the export's constants.
  `BuildCert` is that data, `Built.Agrees` says a built system has it, and
  `Built.Agrees.refines` proves `Refines` from it and from checks of the certificate
  against the export that the kernel evaluates. What is left to trust of a certified
  circuit is `Agrees`: that the exporter read the deployed build correctly. -/

/-- A list as a balanced binary tree, for lookups the kernel evaluates cheaply: `get? i`
    walks `log` nodes where a list walks `i` cells. `node n l r` carries `n`, the size
    of `l`; the lemmas below do not depend on it being right. -/
inductive Tbl (α : Type) where
  | nil
  | leaf (a : α)
  | node (n : ℕ) (l r : Tbl α)

/-- The list a table stands for. -/
def Tbl.toList {α : Type} : Tbl α → List α
  | .nil => []
  | .leaf a => [a]
  | .node _ l r => l.toList ++ r.toList

/-- Lookup by index in the list a table stands for. -/
def Tbl.get? {α : Type} : Tbl α → ℕ → Option α
  | .nil, _ => none
  | .leaf a, i => if i = 0 then some a else none
  | .node n l r, i => if i < n then l.get? i else r.get? (i - n)

theorem Tbl.get?_mem {α : Type} {t : Tbl α} {i : ℕ} {a : α} (h : t.get? i = some a) :
    a ∈ t.toList := by
  induction t generalizing i with
  | nil => exact nomatch h
  | leaf b =>
    simp only [Tbl.get?] at h
    split at h
    · exact (Option.some.inj h) ▸ List.mem_singleton_self _
    · exact nomatch h
  | node n l r ihl ihr =>
    simp only [Tbl.get?] at h
    split at h
    · exact List.mem_append_left _ (ihl h)
    · exact List.mem_append_right _ (ihr h)

/-- The parts of a built system that `Built.Refines` reads, as the exporter reads them
    off `build`. Natural numbers only: a constant is named by its index in the export's
    constant list, so the certificate does not depend on the field. The cycles and the
    virtual targets are tables (`Tbl`) so that the checks against the export, which look
    them up once per copy, are cheap for the kernel. -/
structure BuildCert where
  /-- The export's rows are rows `0..exportRows` of the built system, and the export's
      wire targets sit at themselves. -/
  exportRows : ℕ
  routed : ℕ
  numPublicInputs : ℕ
  /-- Cycles of `σ`, each in `σ` order: the copy classes the export's copies and
      constants use. -/
  cycles : Tbl (Tbl (ℕ × ℕ))
  /-- A routed wire of each virtual target's copy class, by index; `none` when the class
      has none. -/
  virt : Tbl (Option (ℕ × ℕ))
  /-- `(m, v)`: wire `v` is fixed to the constant of the export's constant target `m`. -/
  pins : List (ℕ × (ℕ × ℕ))
  /-- Wires fixed to `0`: the `zero` target's class after `build`. -/
  zeros : List (ℕ × ℕ)
  /-- The `PublicInputGate` row. -/
  piRow : ℕ
  /-- The `PoseidonGate` rows of the public-input hash, in absorb order. -/
  hashRows : List ℕ

/-- The certificate's cycles, as lists. -/
def BuildCert.cycleList (k : BuildCert) : List (List (ℕ × ℕ)) :=
  k.cycles.toList.map Tbl.toList

/-- Where the certificate places a target of the export. -/
def BuildCert.loc (k : BuildCert) : Target → Option (ℕ × ℕ)
  | .wire i j => if i < k.exportRows then some (i, j) else none
  | .virt n => (k.virt.get? n).join

/-- Cycle `ci` of the certificate, looked up and listed, is one of its cycles. -/
theorem BuildCert.mem_cycleList {k : BuildCert} {ci : ℕ} {C : Tbl (ℕ × ℕ)}
    (h : k.cycles.get? ci = some C) : C.toList ∈ k.cycleList :=
  List.mem_map_of_mem (Tbl.get?_mem h)

/-- The built system `b` is as the certificate `k` describes it, against the export `c`:
    the export's rows at their indices, `σ` along the listed cycles, which lie in the
    domain, the listed wires fixed to the export's constants, the targets `k.loc`
    places, where it places them, the listed wires fixed to `0`, and the
    `PublicInputGate` and `PoseidonGate` rows it names. Nothing is said of targets it
    does not place. -/
structure Built.Agrees (b : Built p) (c : Circuit p) (k : BuildCert) : Prop where
  rows : ∀ (i : ℕ) (r : Row p), c.rows[i]? = some r → b.rows[i]? = some r
  routed : b.routed = k.routed
  numPublicInputs : b.numPublicInputs = k.numPublicInputs
  domain : ∀ L ∈ k.cycleList, ∀ u ∈ L, u.1 < b.rows.length ∧ u.2 < b.routed
  sigma : ∀ L ∈ k.cycleList, ∀ i < L.length,
    b.sigma (L.getD i (0, 0)) = L.getD ((i + 1) % L.length) (0, 0)
  fixed : ∀ mv ∈ k.pins, ∀ tc, c.constants[mv.1]? = some tc → (mv.2, tc.2) ∈ b.fixed
  loc : ∀ t u, k.loc t = some u → b.loc t = some u
  zeros : ∀ v ∈ k.zeros, (v, 0) ∈ b.fixed
  piRow : ∃ cs, b.rows[k.piRow]? = some ⟨.publicInput, cs⟩
  hashRows : ∀ g ∈ k.hashRows, ∃ cs, b.rows[g]? = some ⟨.poseidon, cs⟩

private theorem getD_eq_getElem {L : List (ℕ × ℕ)} {i : ℕ} (hi : i < L.length) :
    L.getD i (0, 0) = L[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]

/-- Along a listed cycle, `σ` leads from any member to any member. -/
theorem Built.Agrees.chain {b : Built p} {c : Circuit p} {k : BuildCert} (h : b.Agrees c k)
    {L : List (ℕ × ℕ)} (hL : L ∈ k.cycleList) {u v : ℕ × ℕ} (hu : u ∈ L) (hv : v ∈ L) :
    b.Chain u v := by
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hu
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hv
  rw [← getD_eq_getElem hi, ← getD_eq_getElem hj]
  have hn : 0 < L.length := Nat.zero_lt_of_lt hi
  have step : ∀ i < L.length,
      b.Step (L.getD i (0, 0)) (L.getD ((i + 1) % L.length) (0, 0)) := fun i hi => by
    have hm : L.getD i (0, 0) ∈ L := by
      rw [getD_eq_getElem hi]; exact List.getElem_mem hi
    exact ⟨(h.domain L hL _ hm).1, (h.domain L hL _ hm).2, h.sigma L hL i hi⟩
  have walk : ∀ m i, i < L.length →
      b.Chain (L.getD i (0, 0)) (L.getD ((i + m) % L.length) (0, 0)) := by
    intro m
    induction m with
    | zero =>
      intro i hi
      rw [Nat.add_zero, Nat.mod_eq_of_lt hi]
      exact Relation.ReflTransGen.refl
    | succ m ih =>
      intro i hi
      refine (ih i hi).tail ?_
      rw [← Nat.add_assoc, Nat.add_mod (i + m) 1, Nat.add_mod_mod]
      exact step _ (Nat.mod_lt _ hn)
  have e : (i + (L.length + j - i)) % L.length = j := by
    rw [show i + (L.length + j - i) = L.length + j by omega, Nat.add_mod_left,
      Nat.mod_eq_of_lt hj]
  have := walk (L.length + j - i) i hi
  rwa [e] at this

/-- Check a copy against its witness `(ci, iu, iv)`: cycle `ci` carries the two targets
    at `iu` and `iv`. -/
def BuildCert.copyOk (k : BuildCert) (xy : Target × Target) : ℕ × ℕ × ℕ → Bool
  | (ci, iu, iv) =>
    match k.cycles.get? ci with
    | some C =>
      match C.get? iu, C.get? iv with
      | some u, some v => k.loc xy.1 == some u && k.loc xy.2 == some v
      | _, _ => false
    | none => false

/-- Check a copy list against its witnesses, one each. -/
def BuildCert.copiesOk (k : BuildCert) : List (Target × Target) → List (ℕ × ℕ × ℕ) → Bool
  | [], [] => true
  | xy :: xys, w :: ws => k.copyOk xy w && k.copiesOk xys ws
  | _, _ => false

/-- What `copyOk` certifies: the targets are placed on one listed cycle. -/
def BuildCert.Links (k : BuildCert) (x y : Target) : Prop :=
  ∃ L ∈ k.cycleList, ∃ u v, k.loc x = some u ∧ k.loc y = some v ∧ u ∈ L ∧ v ∈ L

theorem BuildCert.copyOk_sound {k : BuildCert} {xy : Target × Target} {w : ℕ × ℕ × ℕ}
    (h : k.copyOk xy w = true) : k.Links xy.1 xy.2 := by
  obtain ⟨ci, iu, iv⟩ := w
  simp only [copyOk] at h
  split at h
  · rename_i C hC
    split at h
    · rename_i u v hu hv
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      exact ⟨C.toList, BuildCert.mem_cycleList hC, u, v, h.1, h.2,
        Tbl.get?_mem hu, Tbl.get?_mem hv⟩
    · exact absurd h Bool.false_ne_true
  · exact absurd h Bool.false_ne_true

theorem BuildCert.copiesOk_sound {k : BuildCert} :
    ∀ {xys : List (Target × Target)} {ws : List (ℕ × ℕ × ℕ)}, k.copiesOk xys ws = true →
      ∀ xy ∈ xys, k.Links xy.1 xy.2
  | [], _, _, _, h => nomatch h
  | _ :: _, [], h, _, _ => absurd h Bool.false_ne_true
  | xy :: xys, w :: ws, h, z, hz => by
    simp only [copiesOk, Bool.and_eq_true] at h
    rcases List.mem_cons.mp hz with rfl | hz
    · exact copyOk_sound h.1
    · exact copiesOk_sound h.2 z hz

/-- Check the export's constant target number `m`, `t`, against its witness: cycle `ci`
    carries `t` at `iu` and, at `iv`, a wire pinned to constant `m`. -/
def BuildCert.constOk (k : BuildCert) (m : ℕ) (t : Target) : ℕ × ℕ × ℕ → Bool
  | (ci, iu, iv) =>
    match k.cycles.get? ci with
    | some C =>
      match C.get? iu, C.get? iv with
      | some u, some v => k.loc t == some u && decide ((m, v) ∈ k.pins)
      | _, _ => false
    | none => false

/-- Check the constant targets from number `m` on against their witnesses, one each. -/
def BuildCert.constsOk (k : BuildCert) (m : ℕ) : List Target → List (ℕ × ℕ × ℕ) → Bool
  | [], [] => true
  | t :: ts, w :: ws => k.constOk m t w && k.constsOk (m + 1) ts ws
  | _, _ => false

/-- What `constOk` certifies of constant target `m`, `t`. -/
def BuildCert.PinsAt (k : BuildCert) (m : ℕ) (t : Target) : Prop :=
  ∃ L ∈ k.cycleList, ∃ u v, k.loc t = some u ∧ u ∈ L ∧ v ∈ L ∧ (m, v) ∈ k.pins

theorem BuildCert.constOk_sound {k : BuildCert} {m : ℕ} {t : Target} {w : ℕ × ℕ × ℕ}
    (h : k.constOk m t w = true) : k.PinsAt m t := by
  obtain ⟨ci, iu, iv⟩ := w
  simp only [constOk] at h
  split at h
  · rename_i C hC
    split at h
    · rename_i u v hu hv
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      exact ⟨C.toList, BuildCert.mem_cycleList hC, u, v, h.1, Tbl.get?_mem hu,
        Tbl.get?_mem hv, h.2⟩
    · exact absurd h Bool.false_ne_true
  · exact absurd h Bool.false_ne_true

theorem BuildCert.constsOk_sound {k : BuildCert} :
    ∀ {m : ℕ} {ts : List Target} {ws : List (ℕ × ℕ × ℕ)}, k.constsOk m ts ws = true →
      ∀ i t, ts[i]? = some t → k.PinsAt (m + i) t
  | _, [], _, _, _, _, h => by simp at h
  | _, _ :: _, [], h, _, _, _ => absurd h Bool.false_ne_true
  | m, t :: ts, w :: ws, h, i, t', ht => by
    simp only [constsOk, Bool.and_eq_true] at h
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at ht
      subst ht
      exact h.1 |> constOk_sound
    | succ i =>
      simp only [List.getElem?_cons_succ] at ht
      have := constsOk_sound h.2 i t' ht
      rwa [Nat.add_assoc, Nat.add_comm 1 i] at this

/-- Exports have no `ConstantGate` rows: `build` is what adds them. -/
def noConstantRow : GateKind → Bool
  | .constant _ => false
  | _ => true

/-- The export's wire targets are placed at themselves. -/
theorem Built.Agrees.loc_wire {b : Built p} {c : Circuit p} {k : BuildCert}
    (h : b.Agrees c k) {i : ℕ} (hi : i < k.exportRows) (j : ℕ) :
    b.loc (.wire i j) = some (i, j) :=
  h.loc _ _ (by simp [BuildCert.loc, hi])

/-- Each export row is hosted at its own index. -/
theorem Built.Agrees.hosts {b : Built p} {c : Circuit p} {k : BuildCert} (h : b.Agrees c k)
    (hlen : c.rows.length ≤ k.exportRows) {i : ℕ} {r : Row p} (hi : c.rows[i]? = some r)
    (hnc : noConstantRow r.kind = true) : b.Hosts i r := by
  have hlt : i < k.exportRows :=
    lt_of_lt_of_le (List.getElem?_eq_some_iff.mp hi).1 hlen
  have hb : b.rows[i]? = some ⟨r.kind, r.consts⟩ := h.rows i r hi
  unfold Built.Hosts
  split
  · rename_i n hk
    intro k' hk'
    exact Or.inl ⟨i, n, r.consts, k', hk ▸ hb, hk', rfl, rfl, fun m _ => h.loc_wire hlt _⟩
  · rename_i n hk
    exact absurd hnc (by simp [hk, noConstantRow])
  · rename_i n hk
    exact ⟨i, r.consts, hk ▸ hb, fun j _ => h.loc_wire hlt j⟩
  · rename_i hk
    exact ⟨i, r.consts, hk ▸ hb, fun j _ => h.loc_wire hlt j⟩
  · trivial
  · trivial
  · trivial
  · trivial

/-- A certified link is a `Linked` pair of the built system. -/
theorem Built.Agrees.linked {b : Built p} {c : Circuit p} {k : BuildCert} (h : b.Agrees c k)
    {x y : Target} (hl : k.Links x y) : b.Linked x y :=
  let ⟨_, hL, u, v, hx, hy, hu, hv⟩ := hl
  Or.inr ⟨u, v, h.loc x u hx, h.loc y v hy, h.chain hL hu hv⟩

/-- A certified pin of constant target `m` pins the export's constant `m`. -/
theorem Built.Agrees.pins {b : Built p} {c : Circuit p} {k : BuildCert} (h : b.Agrees c k)
    {m : ℕ} {tc : Target × ZMod p} (hm : c.constants[m]? = some tc)
    (hp : k.PinsAt m tc.1) : b.Pins tc.1 tc.2 := by
  obtain ⟨L, hL, u, v, hx, hu, hv, hmv⟩ := hp
  exact ⟨u, v, h.loc _ _ hx, h.chain hL hu hv, h.fixed (m, v) hmv tc hm⟩

/-- **A certified build refines its export.** From `Agrees` and the checks of the
    certificate against the export, which the kernel evaluates: no `ConstantGate` rows,
    every copy linked, every constant pinned, the public-input count. -/
theorem Built.Agrees.refines {b : Built p} {c : Circuit p} {k : BuildCert} (h : b.Agrees c k)
    {ws cws : List (ℕ × ℕ × ℕ)}
    (hlen : c.rows.length ≤ k.exportRows)
    (hnc : (c.rows.map Row.kind).all noConstantRow = true)
    (hcopies : k.copiesOk c.copies ws = true)
    (hconsts : k.constsOk 0 (c.constants.map Prod.fst) cws = true)
    (hpi : k.numPublicInputs = c.publicInputs.length) : b.Refines c where
  rows i r hi := h.hosts hlen hi (by
    have := List.all_eq_true.mp hnc r.kind
    exact this (List.mem_map_of_mem (List.mem_of_getElem? hi)))
  copies xy hxy := h.linked (BuildCert.copiesOk_sound hcopies xy hxy)
  constants tc htc := by
    obtain ⟨m, hm, rfl⟩ := List.getElem_of_mem htc
    have hm' : c.constants[m]? = some c.constants[m] := List.getElem?_eq_getElem hm
    have := BuildCert.constsOk_sound hconsts m c.constants[m].1
      (by rw [List.getElem?_map, hm']; rfl)
    rw [Nat.zero_add] at this
    exact h.pins hm' this
  publicInputs := h.numPublicInputs.trans hpi

/-! ### The public-input hash

  `build` hashes the public-input targets in circuit with the proof-system hash
  (`hash_n_to_hash_no_pad::<C::InnerHasher>`; `PoseidonHash` in
  `PoseidonGoldilocksConfig`) and copies the digest to the `PublicInputGate`, whose
  constraint is that its four wires are the digest the verifier computes of `pis`
  (`get_public_inputs_hash`). The sponge is `hash_n_to_m_no_pad`: the all-zero state,
  each rate block *overwriting* the rate lanes, a permutation per block, the first four
  lanes squeezed. No padding, so `[1]` and `[1, 0]` share a digest. The permutation is
  the proof-system Poseidon, not the wormhole Poseidon2: `Plonky2Spec.Poseidon.Consts.perm`
  on the Goldilocks tables. -/

/-- The proof-system Poseidon permutation (`Poseidon::poseidon`, width `12`): the model's
    permutation on the Goldilocks tables, pinned to the native one by
    `Poseidon.goldilocks_perm_test`. -/
def poseidonPerm : St p → St p := Plonky2Spec.Poseidon.goldilocks.perm

/-- Overwrite the first `block.length` lanes with the block (`set_from_slice(block, 0)`). -/
def setBlock (s : St p) (block : List (ZMod p)) : St p :=
  fun i => if h : i.val < block.length then block[i.val] else s i

/-- The absorb loop of `hash_n_to_m_no_pad`: a rate block overwrites the rate lanes, then
    the permutation, until the input is consumed. -/
def absorbNoPad (perm : St p → St p) (s : St p) : List (ZMod p) → St p
  | [] => s
  | hd :: tl =>
    absorbNoPad perm (perm (setBlock s ((hd :: tl).take rate)))
      ((hd :: tl).drop rate)
  termination_by xs => xs.length
  decreasing_by
    simp only [List.length_drop, List.length_cons, rate]
    omega

/-- `InnerHasher::hash_no_pad`, the proof-system hash of a public-input list: the verifier
    computes it of the list it is given (`get_public_inputs_hash`) and `build` lays it
    down in circuit on the public-input targets. `absorbNoPad` from the all-zero state,
    the first four lanes squeezed (`NUM_HASH_OUT_ELTS = 4 ≤ RATE`, so one squeeze). -/
def publicInputsHash (xs : List (ZMod p)) : Fin 4 → ZMod p :=
  squeeze4 (absorbNoPad poseidonPerm (fun _ => 0) xs)

/-! ### The public-input hash in circuit

  `CircuitBuilder::hash_n_to_m_no_pad` runs the same loop on targets: the state starts
  as twelve copies of the `zero` target, each block's targets overwrite its lanes, and
  `PoseidonHash::permute_swapped` lays down a `PoseidonGate` row, copies the state to its
  input wires and `zero` to its swap wire, and takes its output wires as the new state.
  `build` then copies the first four lanes to the `PublicInputGate`. `Built.Sponge`
  states that layout of a built system; `Built.Sponge.val` reads the hash off it. -/

/-- A lane of the in-circuit sponge's state: the `zero` target, or a wire of the built
    system. -/
inductive Lane
  | zero
  | wire (u : ℕ × ℕ)

/-- Position `u` holds the constant `k`: `σ` leads from it to a wire fixed to `k`. -/
def Built.Holds (b : Built p) (u : ℕ × ℕ) (k : ZMod p) : Prop :=
  ∃ v, b.Chain u v ∧ (v, k) ∈ b.fixed

/-- The lane feeds position `u`: `u` holds `0`, or `σ` leads from the lane's wire to `u`. -/
def Built.Feeds (b : Built p) : Lane → ℕ × ℕ → Prop
  | .zero, u => b.Holds u 0
  | .wire w, u => b.Chain w u

/-- The export's target `t` feeds position `u`: `σ` leads from its wire to `u`. -/
def Built.FeedsTarget (b : Built p) (t : Target) (u : ℕ × ℕ) : Prop :=
  ∃ w, b.loc t = some w ∧ b.Chain w u

/-- What feeds input lane `i` of the gate absorbing `block` on the state `s`: the block's
    target on the lanes it overwrites, the state's lane on the rest. -/
def spongeInput (s : Fin 12 → Lane) (block : List Target) (i : Fin 12) : Target ⊕ Lane :=
  match block[i.val]? with
  | some t => .inl t
  | none => .inr (s i)

/-- A target or a lane feeds position `u`. -/
def Built.FeedsInput (b : Built p) : Target ⊕ Lane → ℕ × ℕ → Prop
  | .inl t, u => b.FeedsTarget t u
  | .inr l, u => b.Feeds l u

/-- The output wires `12..24` of the `PoseidonGate` row `g`, as the state they become. -/
def outLanes (g : ℕ) : Fin 12 → Lane := fun i => .wire (g, 12 + i)

/-- **The in-circuit sponge.** From the state `s`, the targets `ts` go through the
    `PoseidonGate` rows `gs`, a rate block each: the block's targets feed the lanes it
    overwrites, the state the rest, `zero` the swap wire `24`; the row's outputs are the
    next state. The state ends as `s'`. -/
inductive Built.Sponge (b : Built p) :
    (Fin 12 → Lane) → List Target → List ℕ → (Fin 12 → Lane) → Prop
  | nil (s : Fin 12 → Lane) : b.Sponge s [] [] s
  | cons (s : Fin 12 → Lane) (t : Target) (ts : List Target) (g : ℕ) (gs : List ℕ)
      (s' : Fin 12 → Lane) (cs : List (ZMod p)) :
      b.rows[g]? = some ⟨.poseidon, cs⟩ → b.Holds (g, 24) 0 →
      (∀ i : Fin 12, b.FeedsInput (spongeInput s ((t :: ts).take rate) i) (g, i)) →
      b.Sponge (outLanes g) ((t :: ts).drop rate) gs s' →
      b.Sponge s (t :: ts) (g :: gs) s'

/-- **The built system hashes `c`'s public inputs into its `PublicInputGate`.** The
    sponge over `c.publicInputs` from the all-`zero` state, and its end state's first
    four lanes feeding the gate's wires `0..4`. -/
def Built.HashesPublicInputs (b : Built p) (c : Circuit p) : Prop :=
  ∃ (r : ℕ) (cs : List (ZMod p)) (gs : List ℕ) (s : Fin 12 → Lane),
    b.rows[r]? = some ⟨.publicInput, cs⟩ ∧ b.Sponge (fun _ => .zero) c.publicInputs gs s ∧
      ∀ i : Fin 4, b.Feeds (s ⟨i.val, by have := i.isLt; omega⟩) (r, i)

/-! #### Certifying the sponge

  The checks below, on a certificate and the export's public-input targets, are what the
  kernel evaluates for a certified circuit; `Built.Agrees.hashesPublicInputs` proves
  `HashesPublicInputs` from them. -/

/-- Check that `src` feeds position `u`, by the witness `(ci, iu, iv)`: cycle `ci`
    carries the source's wire at `iu` — the export target's place, or the lane's wire —
    and `u` at `iv`; for the `zero` lane, `u` at `iu` and a wire fixed to `0` at `iv`. -/
def BuildCert.feedOk (k : BuildCert) (src : Target ⊕ Lane) (u : ℕ × ℕ) : ℕ × ℕ × ℕ → Bool
  | (ci, iu, iv) =>
    match k.cycles.get? ci with
    | some C =>
      match C.get? iu, C.get? iv with
      | some x, some y =>
        match src with
        | .inl t => k.loc t == some x && y == u
        | .inr .zero => x == u && decide (y ∈ k.zeros)
        | .inr (.wire w) => x == w && y == u
      | _, _ => false
    | none => false

theorem BuildCert.feedOk_sound {b : Built p} {c : Circuit p} {k : BuildCert}
    (h : b.Agrees c k) {src : Target ⊕ Lane} {u : ℕ × ℕ} {w : ℕ × ℕ × ℕ}
    (hw : k.feedOk src u w = true) : b.FeedsInput src u := by
  obtain ⟨ci, iu, iv⟩ := w
  simp only [feedOk] at hw
  split at hw
  · rename_i C hC
    split at hw
    · rename_i x y hx hy
      have chain : b.Chain x y :=
        h.chain (BuildCert.mem_cycleList hC) (Tbl.get?_mem hx) (Tbl.get?_mem hy)
      rcases src with t | l
      · simp only [Bool.and_eq_true, beq_iff_eq] at hw
        exact ⟨x, h.loc t x hw.1, hw.2 ▸ chain⟩
      · cases l with
        | zero =>
          simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hw
          exact ⟨y, hw.1 ▸ chain, h.zeros y hw.2⟩
        | wire v =>
          simp only [Bool.and_eq_true, beq_iff_eq] at hw
          exact hw.1 ▸ hw.2 ▸ chain
    · exact absurd hw Bool.false_ne_true
  · exact absurd hw Bool.false_ne_true

/-- Check the inputs of one sponge gate from lane `i` on, one witness each. -/
def BuildCert.lanesOk (k : BuildCert) (s : Fin 12 → Lane) (block : List Target) (g : ℕ) :
    List (ℕ × ℕ × ℕ) → ℕ → Bool
  | [], i => i == 12
  | w :: ws, i =>
    if h : i < 12 then
      k.feedOk (spongeInput s block ⟨i, h⟩) (g, i) w && k.lanesOk s block g ws (i + 1)
    else false

theorem BuildCert.lanesOk_sound {b : Built p} {c : Circuit p} {k : BuildCert}
    (h : b.Agrees c k) {s : Fin 12 → Lane} {block : List Target} {g : ℕ} :
    ∀ {ws : List (ℕ × ℕ × ℕ)} {i : ℕ}, k.lanesOk s block g ws i = true →
      ∀ j : Fin 12, i ≤ j.val → b.FeedsInput (spongeInput s block j) (g, j)
  | [], i, hok, j, hij => by
    simp only [lanesOk, beq_iff_eq] at hok
    exact absurd (hok ▸ hij) (by have := j.isLt; omega)
  | w :: ws, i, hok, j, hij => by
    simp only [lanesOk] at hok
    split at hok
    · rename_i hi
      simp only [Bool.and_eq_true] at hok
      rcases Nat.eq_or_lt_of_le hij with hji | hlt
      · have : j = ⟨i, hi⟩ := Fin.ext hji.symm
        subst this
        exact feedOk_sound h hok.1
      · exact lanesOk_sound h hok.2 j hlt
    · exact absurd hok Bool.false_ne_true

/-- The state a sponge run ends in: the last listed row's outputs, or `s` with no row. -/
def spongeEnd (s : Fin 12 → Lane) : List ℕ → Fin 12 → Lane
  | [] => s
  | g :: gs => spongeEnd (outLanes g) gs

/-- The witnesses of one sponge gate: its swap wire's `zero`, and its twelve inputs. -/
structure GateWit where
  swap : ℕ × ℕ × ℕ
  lanes : List (ℕ × ℕ × ℕ)

/-- Check a sponge run: for each block, a listed `PoseidonGate` row, its swap wire fed
    by `zero`, its inputs fed (`lanesOk`), and the run on from its outputs. -/
def BuildCert.spongeOk (k : BuildCert) (s : Fin 12 → Lane) :
    List Target → List ℕ → List GateWit → Bool
  | [], [], [] => true
  | t :: ts, g :: gs, w :: ws =>
    decide (g ∈ k.hashRows) && k.feedOk (.inr .zero) (g, 24) w.swap &&
      k.lanesOk s ((t :: ts).take rate) g w.lanes 0 &&
      k.spongeOk (outLanes g) ((t :: ts).drop rate) gs ws
  | _, _, _ => false

theorem BuildCert.spongeOk_sound {b : Built p} {c : Circuit p} {k : BuildCert}
    (h : b.Agrees c k) :
    ∀ {gs : List ℕ} {ts : List Target} {ws : List GateWit} {s : Fin 12 → Lane},
      k.spongeOk s ts gs ws = true → b.Sponge s ts gs (spongeEnd s gs)
  | [], [], [], s, _ => .nil s
  | [], _ :: _, _, _, hok => absurd hok Bool.false_ne_true
  | _ :: _, [], _, _, hok => absurd hok Bool.false_ne_true
  | _ :: _, _ :: _, [], _, hok => absurd hok Bool.false_ne_true
  | g :: gs, t :: ts, w :: ws, s, hok => by
    simp only [spongeOk, Bool.and_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨⟨hg, hsw⟩, hl⟩, hrest⟩ := hok
    obtain ⟨cs, hrow⟩ := h.hashRows g hg
    exact .cons s t ts g gs _ cs hrow (feedOk_sound h hsw)
      (fun i => lanesOk_sound h hl i (Nat.zero_le _)) (spongeOk_sound h hrest)

/-- Check the end state's lanes from `i` on feeding the `PublicInputGate`'s wires. -/
def BuildCert.outsOk (k : BuildCert) (s : Fin 12 → Lane) : List (ℕ × ℕ × ℕ) → ℕ → Bool
  | [], i => i == 4
  | w :: ws, i =>
    if h : i < 4 then
      k.feedOk (.inr (s ⟨i, by omega⟩)) (k.piRow, i) w && k.outsOk s ws (i + 1)
    else false

theorem BuildCert.outsOk_sound {b : Built p} {c : Circuit p} {k : BuildCert}
    (h : b.Agrees c k) {s : Fin 12 → Lane} :
    ∀ {ws : List (ℕ × ℕ × ℕ)} {i : ℕ}, k.outsOk s ws i = true →
      ∀ j : Fin 4, i ≤ j.val → b.Feeds (s ⟨j.val, by have := j.isLt; omega⟩) (k.piRow, j)
  | [], i, hok, j, hij => by
    simp only [outsOk, beq_iff_eq] at hok
    exact absurd (hok ▸ hij) (by have := j.isLt; omega)
  | w :: ws, i, hok, j, hij => by
    simp only [outsOk] at hok
    split at hok
    · rename_i hi
      simp only [Bool.and_eq_true] at hok
      rcases Nat.eq_or_lt_of_le hij with hji | hlt
      · have : j = ⟨i, hi⟩ := Fin.ext hji.symm
        subst this
        exact feedOk_sound h hok.1
      · exact outsOk_sound h hok.2 j hlt
    · exact absurd hok Bool.false_ne_true

/-- Check the public-input hash: the sponge over `ts` from the all-`zero` state through
    the rows `gs`, and its end state feeding the `PublicInputGate`. -/
def BuildCert.piOk (k : BuildCert) (ts : List Target) (gs : List ℕ) (ws : List GateWit)
    (outs : List (ℕ × ℕ × ℕ)) : Bool :=
  k.spongeOk (fun _ => .zero) ts gs ws && k.outsOk (spongeEnd (fun _ => .zero) gs) outs 0

/-- **A certified build hashes its export's public inputs.** From `Agrees` and the check
    of the certificate against the export's public-input targets. -/
theorem Built.Agrees.hashesPublicInputs {b : Built p} {c : Circuit p} {k : BuildCert}
    (h : b.Agrees c k) {gs : List ℕ} {ws : List GateWit} {outs : List (ℕ × ℕ × ℕ)}
    (hok : k.piOk c.publicInputs gs ws outs = true) : b.HashesPublicInputs c := by
  simp only [BuildCert.piOk, Bool.and_eq_true] at hok
  obtain ⟨cs, hrow⟩ := h.piRow
  exact ⟨k.piRow, cs, gs, _, hrow, BuildCert.spongeOk_sound h hok.1,
    fun j => BuildCert.outsOk_sound h hok.2 j (Nat.zero_le _)⟩

end Built

variable {p : ℕ} [Fact p.Prime]

/-- The recursive verifier gadget under `r`'s verifier key accepted a proof whose public
    inputs are `pis`, in a system whose Poseidon2 gate computes `perm`. `opaque`, not
    `axiom`: an abstract predicate whose truth is never assumed. -/
opaque ProofAccepted (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop

/-- An accepting verifier transcript for `r` on `pis`: the Fiat–Shamir challenges are
    hashes of the proof messages under the proof-system Poseidon, the Merkle openings
    authenticate those messages, and the algebraic checks the verifier evaluates
    (vanishing identity, quotient relation, permutation partial products) hold at the
    challenged point. `opaque`: only `poseidon_transcript_oracle` produces it. -/
opaque TranscriptAccepted (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop

/-- The polynomials opened in an accepting transcript are within the FRI proximity
    radius of codewords of degree below the circuit's claimed degree, and the opened
    values are the evaluations of those codewords. `opaque`: only `fri_proximity_gap`
    produces it. -/
opaque LowDegreeOpenings (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop

/-- `b` is the system `r`'s verifier key commits to: what `CircuitBuilder::build` makes
    of the builder that holds `r`'s export and the `verify_proof` gadgets of its children,
    with `b.loc` the place of the export's targets in it. `opaque`: `build` is not
    modelled, so only the obligations below read it. -/
opaque BuildOf (r : Recursive p) (b : Built p) : Prop

/-- `o` is what an accepting transcript for `r` on `pis` opens: `o.wire` is the values on
    the domain of the low-degree wire polynomials of `LowDegreeOpenings`, and `o.products`
    is, round by round, the `β`, `γ` the transcript derives for that round and the values
    of that round's `Z`. `opaque`: the transcript is not modelled, so only
    `algebraic_sampling` produces this. -/
opaque TranscriptOpens (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p))
    (o : Opened p) : Prop

/-- **The public-input binding good event.** The proof-system hash does not collide `xs`
    with `pis`: with one length and one digest they are one list. The lengths are a
    hypothesis because the hash does not pad. -/
def PublicInputBinding (xs pis : List (ZMod p)) : Prop :=
  xs.length = pis.length → publicInputsHash xs = publicInputsHash pis → xs = pis

/-! ### The Poseidon2 gate's rows

  `Poseidon2Gate<WIDTH=12>` (`plonky2/src/gates/poseidon2.rs`) lays its wires out as the
  `12` inputs, the `12` outputs, the S-box checkpoints of external rounds `1..8`
  (`wire_ext_sbox`, round `0` elided, `12` lanes each) and the `22` internal checkpoints
  (`wire_int_sbox`): `130` wires. Its constraints are `Poseidon2.gateConstraints`, pinned
  to `eval_unfiltered` by the exporter's differential tests. -/

/-- The round constants of a `Poseidon2Gate`: `Poseidon2Params` in the Rust gate. -/
structure Poseidon2Consts (p : ℕ) where
  eInit : Fin 4 → St p
  eTerm : Fin 4 → St p
  iRc : Fin 22 → ZMod p
  diag : St p

/-- The permutation the gate with constants `k` computes. -/
def Poseidon2Consts.perm (k : Poseidon2Consts p) : St p → St p :=
  Plonky2Spec.Poseidon2.permState k.eInit k.eTerm k.iRc k.diag

/-- Wire `W_EXT_SBOX + (round − 1) · 12 + lane`: the checkpoint of external round
    `round ≥ 1`. Rounds `1..3` are the initial rounds after the elided round `0`, rounds
    `4..7` the terminal ones. -/
def poseidon2ExtCk (wires : ℕ → ZMod p) (round : ℕ) : St p :=
  fun lane => wires (24 + (round - 1) * 12 + lane)

/-- Wire `W_INT_SBOX + round`: the lane-`0` checkpoint of internal round `round`. -/
def poseidon2IntCk (wires : ℕ → ZMod p) (round : ℕ) : ZMod p := wires (108 + round)

/-- The gate's constraints with constants `k`, read off a row's wires. -/
def Poseidon2Consts.rowHolds (k : Poseidon2Consts p) (wires : ℕ → ZMod p) : Prop :=
  ∀ c ∈ Plonky2Spec.Poseidon2.gateConstraints k.eInit k.eTerm k.iRc k.diag
      (fun i : Fin 12 => wires i) (fun i : Fin 12 => wires (12 + i))
      ((List.range 3).map fun r => poseidon2ExtCk wires (1 + r))
      ((List.range 4).map fun r => poseidon2ExtCk wires (4 + r))
      ((List.range 22).map fun r => poseidon2IntCk wires r), c = 0

/-- A `Poseidon2Gate` row in a system whose gate computes `perm`: the constraints of the
    gate hold on its wires, for the round constants whose permutation is `perm`. -/
def Poseidon2Row (perm : St p → St p) (wires : ℕ → ZMod p) : Prop :=
  ∃ k : Poseidon2Consts p, k.perm = perm ∧ k.rowHolds wires

/-- The `PublicInputGate`: its constraints are `wires[i] - public_inputs_hash[i]` for
    `i < 4` (`eval_unfiltered` in `plonky2/src/gates/public_input.rs`), where
    `public_inputs_hash` is the verifier's `get_public_inputs_hash` of `pis`. -/
def PublicInputRow (pis : List (ZMod p)) (wires : ℕ → ZMod p) : Prop :=
  ∀ i : Fin 4, wires i = publicInputsHash pis i

/-- The constraints of a `PoseidonGate` row: `Poseidon.gateConstraints` on the Goldilocks
    tables, read at the gate's layout (`Poseidon.Consts.rowHolds`). -/
def PoseidonRow (wires : ℕ → ZMod p) : Prop :=
  Plonky2Spec.Poseidon.goldilocks.rowHolds wires

/-- **Poseidon rows.** The constraints of a `PoseidonGate` row whose swap wire `24` is `0`
    make its output wires `12..24` the proof-system permutation of its input wires
    `0..12`: the soundness half of `Poseidon.gate_sound_complete`, with the deltas at `0`.
    The gate and the native hash are one algorithm over one table of constants
    (`qp_plonky2_core::poseidon`), which is why one `poseidonPerm` names both. -/
theorem poseidon_gate_rows (wires : ℕ → ZMod p) :
    PoseidonRow wires → wires 24 = 0 →
      (fun i : Fin 12 => wires (12 + i)) = poseidonPerm fun i : Fin 12 => wires i :=
  fun h hswap => h.perm_eq hswap

/-- The constraints of a gate the export names only by `id`. `opaque`: not modelled. -/
opaque OtherRow (id : String) (consts : List (ZMod p)) (wires : ℕ → ZMod p) : Prop

/-- What `rowConstraints` leaves out of a row's gate: the constraints of a `Poseidon2Gate`
    in a system where it computes `perm` (`Poseidon2Row`), of a `PoseidonGate`
    (`PoseidonRow`), of the `PublicInputGate` (`PublicInputRow`), and of the `.other`
    gates (`OtherRow`). The arithmetic, constant, base-sum and no-op gates have nothing
    left over. -/
def UninterpretedRow (perm : St p → St p) (pis : List (ZMod p)) (r : Row p)
    (wires : ℕ → ZMod p) : Prop :=
  match r.kind with
  | .poseidon2 => Poseidon2Row perm wires
  | .poseidon => PoseidonRow wires
  | .publicInput => PublicInputRow pis wires
  | .other id => OtherRow id r.consts wires
  | .arithmetic _ => True
  | .constant _ => True
  | .baseSum2 _ => True
  | .noop => True

/-- **Gate identities.** Every row of the domain satisfies its gate, and every routed
    constant holds its constant. -/
def GateIdentities (perm : St p → St p) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) : Prop :=
  (∀ i r, b.rows[i]? = some r →
      rowConstraints o.assignment i r ∧ UninterpretedRow perm pis r (o.wire i)) ∧
    ∀ uk ∈ b.fixed, o.val uk.1 = uk.2

/-- The gate, copy, and permutation identities of `b` hold of `o` on the whole domain. -/
structure DomainIdentities (perm : St p → St p) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) : Prop where
  gate : GateIdentities perm pis b o
  copy : CopyIdentities b o
  permutation : PermutationIdentities b o

/-- The gate, copy, and permutation identities hold on the whole evaluation domain,
    not merely at the sampled challenges, and the opened values are the corresponding
    witness: the identities are those of the system `b` that `r`'s verifier key commits
    to, they hold of the values `o` the transcript opens, under its own challenges
    (`TranscriptOpens`), and the witness is `b.witness o`. `pis` has the length the
    verifier checks before anything else (`validate_proof_with_pis_shape`).

    Both the copy and the permutation identities are stated. The transcript checks the
    second; it gives the first on the good event for `β`, `γ`, which is part of what
    `algebraic_sampling` conditions on. Only `algebraic_sampling` produces this. -/
def IdenticalConstraints (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop :=
  ∃ (b : Built p) (o : Opened p), BuildOf r b ∧ TranscriptOpens perm r pis o ∧
    pis.length = b.numPublicInputs ∧ DomainIdentities perm pis b o

/-! ### The good events

  Each trusted implication below is false of some proof: a prover may find a Merkle or
  public-input collision, a FRI query may miss a far word, a challenge may land on a
  root. The implications hold on the good events of the execution, and those events are
  what carry the probabilities. They are stated per verifier key: `OracleGood perm r`
  is the event for the proofs of `r` the adversary produced, not for one of them, since
  the children's proofs are only named after extraction. `opaque`: the execution is not
  modelled. The capstones take `GoodEvents` as a hypothesis, and nothing produces it. -/

/-- The random-oracle good event for the proofs of `r`: the Fiat–Shamir extractor
    succeeds and no Merkle opening collides. Probability `1 − negl` against a bounded
    adversary in the (quantum) random-oracle model. -/
opaque OracleGood (perm : St p → St p) (r : Recursive p) : Prop

/-- The proximity good event for the proofs of `r`: every FRI query of an accepting
    transcript sees a word within the proximity radius. Probability
    `1 − 2^{-friQueryBits}` under the proximity-gap conjecture. -/
opaque ProximityGood (perm : St p → St p) (r : Recursive p) : Prop

/-- The sampling good event for the proofs of `r`: the base-field challenges `α`, `β`,
    `γ` and the opening point `ζ` miss the roots of the nonzero discrepancies.
    Probability `1 − ε`, as `algebraic_sampling` describes. -/
opaque SamplingGood (perm : St p → St p) (r : Recursive p) : Prop

/-- The binding good event for the proofs of `r`: the prover did not find two
    public-input lists of one length with one `InnerHasher` digest. Computational. -/
opaque BindingGood (perm : St p → St p) (r : Recursive p) : Prop

/-- All four good events for the proofs of `r`. The hypothesis of `proof_sound` and of
    every capstone that consumes an accepted proof, once per tree whose proofs it reads. -/
structure GoodEvents (perm : St p → St p) (r : Recursive p) : Prop where
  oracle : OracleGood perm r
  proximity : ProximityGood perm r
  sampling : SamplingGood perm r
  binding : BindingGood perm r

/-- **TRUSTED — random oracle.** On `OracleGood`, acceptance of a proof is an accepting
    interactive transcript (`TranscriptAccepted`) whose challenges are derived by the
    proof-system Poseidon (`PoseidonHash` in `PoseidonGoldilocksConfig`: Merkle caps and
    the Fiat–Shamir challenger). The post-quantum reading is the quantum random oracle.

    Collision resistance of that Poseidon is not this assumption. Collision resistance
    binds a Merkle opening; it does not make the challenges unpredictable, and the
    knowledge extractor rewinds the oracle. The wormhole Poseidon2 sponge is a
    different function and is not mentioned here.

    Discharging this is a Fiat–Shamir theorem for this IOP in the (quantum) random
    oracle model, instantiated by this Poseidon. -/
axiom poseidon_transcript_oracle (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    OracleGood perm (.node tag c children) →
      ProofAccepted perm (.node tag c children) pis →
        TranscriptAccepted perm (.node tag c children) pis

/-- **TRUSTED — FRI proximity gaps.** On `ProximityGood`, an accepting transcript means
    the opened polynomials are low degree in the sense of `LowDegreeOpenings`.

    This is the ethSTARK proximity-gap conjecture at the FRI parameters of the circuit
    whose verifier accepted, not a property of either hash. For `standardRecursionFri`
    the statistical query error is `2^{-friQueryBits standardRecursionFri} = 2^{-84}`:
    each of the 28 queries is bounded by the rate `1/8`. The further 16 bits in
    `friConjecturedBits standardRecursionFri = 100` are grinding. They are adversary
    work against the oracle, not another `2^{-16}` in this probability. This
    implication is the proximity good event only. It does not remove the algebraic
    sampling error (`algebraic_sampling`).

    Discharging this is a proof of proximity gaps out to the list-decoding radius
    this blowup uses. The unique-decoding regime is proved in the literature; the
    parameters above sit past it. -/
axiom fri_proximity_gap (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) :
    ProximityGood perm r → TranscriptAccepted perm r pis → LowDegreeOpenings perm r pis

/-- **TRUSTED — algebraic sampling.** On `SamplingGood`, low-degree openings whose
    vanishing, quotient, and permutation checks hold at the Fiat–Shamir challenges are
    identical constraints.

    Two different fields are sampled (`core/src/proof.rs`, `get_challenges` in
    `plonky2/src/plonk/get_challenges.rs`). The permutation challenges `β`, `γ` and
    the constraint-combination challenges `α` are base-field elements, each drawn
    `num_challenges` times by `get_n_challenges`. That repetition count does not
    depend on the extension degree `D`. Lookup deltas, when the circuit has lookups,
    are further base-field coins of the same kind. Only the opening point `ζ` is an
    extension element (`get_extension_challenge`).

    A bad base-field challenge is already a failure, before any opening. With
    `β = 0` the verifier's numerator `wire + β·s_id + γ` and denominator
    `wire + β·s_sigma + γ` (`verifier/src/plonk/vanishing_poly.rs`) coincide even
    when the copy constraints are violated. One such value has probability `1/|F|`
    per draw, about `2^{-64}` on Goldilocks, and `CircuitBuilder` accepts
    `num_challenges = 1`. The extension-field term is separate: the vanishing
    identity is checked at `ζ` (`eval_vanishing_poly` against the quotient in
    `verifier/src/plonk/verifier.rs`), so a nonzero discrepancy of degree below the
    FRI claim can still vanish there. Lemma 4.9 of *Fiat-Shamir Security of FRI
    and Related SNARKs* keeps the repeated base-field terms apart from that
    extension-field evaluation.

    `SamplingGood` is the conjunction of those good events. Write `ε` for its
    failure probability, the sum of the base-field terms and the extension-field
    term. The good-event probability is `1 − ε`. No closed form is stated: a single
    factor `(degree · number of checks) / |F^D|` drops the base-field terms and
    understates `ε`. `ε` is independent of `friQueryBits` and of grinding, and it
    is not itself the `Prop`.

    Discharging this is a concrete bound that states the base-field terms and the
    extension-field term separately. -/
axiom algebraic_sampling (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) :
    SamplingGood perm r → LowDegreeOpenings perm r pis → IdenticalConstraints perm r pis

/-! ### Identical constraints satisfy the export

  The first piece of `verifier_algebra`, proved: the identities on the whole domain of a
  system that refines the export make its witness a satisfying assignment of the export. -/

/-- Under the gate and copy identities a pinned target reads its constant. -/
theorem Built.Pins.witness_eq {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} (hg : GateIdentities perm pis b o) (hc : CopyIdentities b o)
    {t : Target} {k : ZMod p} (h : b.Pins t k) : b.witness o t = k := by
  obtain ⟨u, v, ht, huv, hk⟩ := h
  rw [Built.witness_of_loc o ht, ← huv.val_eq hc]
  exact hg.2 _ hk

/-- The value of a lane: `0` for `zero`, the wire's value for a wire. -/
def Lane.val (o : Opened p) : Lane → ZMod p
  | .zero => 0
  | .wire u => o.val u

/-- Under the gate and copy identities a position that holds `k` reads `k`. -/
theorem Built.Holds.val_eq {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} (hg : GateIdentities perm pis b o) (hc : CopyIdentities b o)
    {u : ℕ × ℕ} {k : ZMod p} (h : b.Holds u k) : o.val u = k := by
  obtain ⟨v, huv, hk⟩ := h
  rw [← huv.val_eq hc]
  exact hg.2 _ hk

/-- Under the identities a fed position reads its lane. -/
theorem Built.Feeds.val_eq {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} (hg : GateIdentities perm pis b o) (hc : CopyIdentities b o)
    {l : Lane} {u : ℕ × ℕ} (h : b.Feeds l u) : o.val u = l.val o := by
  cases l with
  | zero => exact Built.Holds.val_eq hg hc h
  | wire w => exact Built.Chain.val_eq hc h

omit [Fact p.Prime] in
/-- Under the copy identities a position fed by a target reads the witness on it. -/
theorem Built.FeedsTarget.val_eq {b : Built p} {o : Opened p} (hc : CopyIdentities b o)
    {t : Target} {u : ℕ × ℕ} (h : b.FeedsTarget t u) : o.val u = b.witness o t := by
  obtain ⟨w, hw, hwu⟩ := h
  rw [hwu.val_eq hc, Built.witness_of_loc o hw]

/-- Under the identities the input lanes of a sponge gate read the block written over
    the state: `setBlock` of the state's values and the witness on the block. -/
theorem Built.FeedsInput.val_eq {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} (hg : GateIdentities perm pis b o) (hc : CopyIdentities b o)
    {s : Fin 12 → Lane} {block : List Target} {i : Fin 12} {u : ℕ × ℕ}
    (h : b.FeedsInput (spongeInput s block i) u) :
    o.val u = setBlock (fun j => (s j).val o) (block.map (b.witness o)) i := by
  unfold spongeInput at h
  unfold setBlock
  split
  · rename_i hlt
    rw [List.length_map] at hlt
    rw [List.getElem?_eq_getElem hlt] at h
    rw [List.getElem_map]
    exact Built.FeedsTarget.val_eq hc h
  · rename_i hge
    rw [List.length_map] at hge
    rw [List.getElem?_eq_none (by omega)] at h
    exact Built.Feeds.val_eq hg hc h

/-- **The in-circuit sponge computes `absorbNoPad`.** Under the domain identities, the
    end state of a sponge run reads `absorbNoPad poseidonPerm` of the start state's
    values on the witness of the absorbed targets: block by block, the gate's inputs read
    the block over the state (`FeedsInput.val_eq`), its swap wire reads `0`, and its
    outputs are then the permutation of its inputs (`poseidon_gate_rows`). -/
theorem Built.Sponge.val {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} (hid : DomainIdentities perm pis b o) {s : Fin 12 → Lane}
    {ts : List Target} {gs : List ℕ} {s' : Fin 12 → Lane} (h : b.Sponge s ts gs s') :
    (fun i => (s' i).val o) =
      absorbNoPad poseidonPerm (fun i => (s i).val o) (ts.map (b.witness o)) := by
  induction h with
  | nil s => rw [List.map_nil, absorbNoPad]
  | cons s t ts g gs s' cs hrow hswap hin _ ih =>
    rw [List.map_cons, absorbNoPad, ← List.map_cons, ← List.map_take, ← List.map_drop, ih]
    congr 1
    have hsw : o.wire g 24 = 0 := hswap.val_eq hid.gate hid.copy
    have key := poseidon_gate_rows (o.wire g) (hid.gate.1 g _ hrow).2 hsw
    show (fun i : Fin 12 => o.wire g (12 + i)) = _
    rw [key]
    congr 1
    funext i
    exact (hin i).val_eq hid.gate hid.copy

/-- **The hash of the carried public inputs is the hash of `pis`.** The sponge rows
    compute the hash of the witness on the public-input targets (`Sponge.val`), the
    `PublicInputGate` reads it (`Feeds.val_eq`), and the gate's constraint makes it the
    hash of `pis` (`PublicInputRow`). This was the obligation `public_input_wires`. -/
theorem Built.HashesPublicInputs.hash_eq {perm : St p → St p} {pis : List (ZMod p)}
    {b : Built p} {o : Opened p} {c : Circuit p} (h : b.HashesPublicInputs c)
    (hid : DomainIdentities perm pis b o) :
    publicInputsHash (c.publicInputs.map (b.witness o)) = publicInputsHash pis := by
  obtain ⟨r, cs, gs, s, hrow, hsp, hout⟩ := h
  have hpi : PublicInputRow pis (o.wire r) := (hid.gate.1 r _ hrow).2
  have hval : (fun i => (s i).val o) =
      absorbNoPad poseidonPerm (fun _ => 0) (c.publicInputs.map (b.witness o)) :=
    hsp.val hid
  funext i
  rw [← hpi i]
  show squeeze4 _ i = o.val (r, i)
  rw [(hout i).val_eq hid.gate hid.copy]
  unfold squeeze4
  rw [← hval]

/-- Under the gate and copy identities a carried row holds on the witness. -/
theorem Built.Hosts.holds {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} (hg : GateIdentities perm pis b o) (hc : CopyIdentities b o)
    {i : ℕ} {r : Row p} (h : b.Hosts i r) : rowConstraints (b.witness o) i r := by
  obtain ⟨kind, consts⟩ := r
  cases kind with
  | arithmetic n =>
    intro k hk
    rcases h k hk with ⟨i', n', cs, k', hrow, hk', h0, h1, hloc⟩ | hnone
    · have key : arithOp o.assignment i' k' (cs.getD 0 0) (cs.getD 1 0) :=
        (hg.1 i' _ hrow).1 k' hk'
      have e0 : b.witness o (.wire i (4 * k)) = o.wire i' (4 * k') :=
        Built.witness_of_loc o (hloc 0 (by omega))
      have e1 : b.witness o (.wire i (4 * k + 1)) = o.wire i' (4 * k' + 1) :=
        Built.witness_of_loc o (hloc 1 (by omega))
      have e2 : b.witness o (.wire i (4 * k + 2)) = o.wire i' (4 * k' + 2) :=
        Built.witness_of_loc o (hloc 2 (by omega))
      have e3 : b.witness o (.wire i (4 * k + 3)) = o.wire i' (4 * k' + 3) :=
        Built.witness_of_loc o (hloc 3 (by omega))
      unfold arithOp at key ⊢
      rw [e0, e1, e2, e3, ← h0, ← h1]
      exact key
    · have e0 : b.witness o (.wire i (4 * k)) = 0 :=
        Built.witness_of_unplaced o (hnone 0 (by omega))
      have e1 : b.witness o (.wire i (4 * k + 1)) = 0 :=
        Built.witness_of_unplaced o (hnone 1 (by omega))
      have e2 : b.witness o (.wire i (4 * k + 2)) = 0 :=
        Built.witness_of_unplaced o (hnone 2 (by omega))
      have e3 : b.witness o (.wire i (4 * k + 3)) = 0 :=
        Built.witness_of_unplaced o (hnone 3 (by omega))
      unfold arithOp
      rw [e0, e1, e2, e3]
      simp [Plonky2Spec.Generated.arithmeticGate_c0]
  | constant n => exact fun j hj => (h j hj).witness_eq hg hc
  | baseSum2 n =>
    obtain ⟨i', cs, hrow, hloc⟩ := h
    have key : Plonky2Spec.BaseSum 2 (o.assignment (.wire i' 0))
        ((limbWires i' n).map o.assignment) := (hg.1 i' _ hrow).1
    have e0 : b.witness o (.wire i 0) = o.assignment (.wire i' 0) :=
      Built.witness_of_loc o (hloc 0 (by omega))
    have el : (limbWires i n).map (b.witness o) = (limbWires i' n).map o.assignment := by
      simp only [limbWires, List.map_map]
      exact List.map_congr_left fun j hj =>
        Built.witness_of_loc o (hloc (j + 1) (by have := List.mem_range.mp hj; omega))
    show Plonky2Spec.BaseSum 2 (b.witness o (.wire i 0)) ((limbWires i n).map (b.witness o))
    rw [e0, el]
    exact key
  | publicInput => trivial
  | noop => trivial
  | poseidon2 => trivial
  | poseidon => trivial
  | other _ => trivial

theorem rowsSatisfied_of_forall {a : Assignment p} :
    ∀ (rs : List (Row p)) (k : ℕ),
      (∀ i r, rs[i]? = some r → rowConstraints a (k + i) r) → rowsSatisfied a k rs
  | [], _, _ => trivial
  | r :: rs, k, h =>
    ⟨h 0 r rfl, rowsSatisfied_of_forall rs (k + 1) fun i r' hi => by
      have := h (i + 1) r' hi
      rwa [show k + (i + 1) = k + 1 + i by omega] at this⟩

/-- **Identical constraints satisfy the export.** If the built system refines `c` and its
    gate, copy and permutation identities hold of `o` on the whole domain, the witness
    satisfies `c`: rows by the gate identities, copies by the copy identities along `σ`,
    constant targets by both. -/
theorem Built.Refines.satisfies {perm : St p → St p} {pis : List (ZMod p)} {b : Built p}
    {o : Opened p} {c : Circuit p} (href : b.Refines c)
    (hid : DomainIdentities perm pis b o) : Satisfies c (b.witness o) :=
  ⟨rowsSatisfied_of_forall c.rows 0 fun i r hi => by
      rw [Nat.zero_add]
      exact (href.rows i r hi).holds hid.gate hid.copy,
    fun xy hxy => (href.copies xy hxy).witness_eq hid.copy,
    fun tc htc => (href.constants tc htc).witness_eq hid.gate hid.copy⟩

/-! ### The obligations that remain

  `verifier_algebra` is these four, `public_input_binding`, and the theorems
  `Built.Refines.satisfies` and `poseidon2_gate_rows`. None is a proximity conjecture or
  a sampling error. The four are deterministic. `public_input_binding` is not: it is the
  binding of the proof-system hash on the public inputs. -/

/-- **OBLIGATION — `build` versus the pre-`build` export.** The system a verifier key
    commits to refines the export of its circuit.

    For the leaf the export is the builder's whole pre-`build` state. `build` adds the
    public-input hash and its `PublicInputGate`, the routed constants, the no-op padding,
    and `σ` from the copy list. For the wrappers the export is less than that state:
    their traces are recorded on a fresh builder that lays down no `verify_proof` rows
    (`TracingBuilder::verify_proof` in qp-zk-circuits), and production runs
    `add_recursive_verifiers` before the wrapper logic. The deployed wrapper rows and
    virtual targets are numbered differently from the export, the wrapper's arithmetic ops
    can share rows with the gadgets', and its constants can sit on `RandomAccessGate`
    wires. `Built.Refines` is stated up to `loc` for that reason.

    Discharging this is a model of `build` on the deployed builder, or a check of
    `Built.Refines` against an export of the built system. The second is
    `Built.Agrees.refines`: a circuit whose built system is certified (`BuildCert`) proves
    its refinement from `Agrees`, and `proof_sound_of` takes that proof instead of this
    declaration. The leaf is certified (`Plonky2Bridge.LeafBuild`); the wrappers are not,
    since their traces are not the deployed builder's, and use this declaration. -/
axiom build_refines_export (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (b : Built p) :
    BuildOf (.node tag c children) b → b.Refines c

/-- **Poseidon2 rows.** The constraints of a `Poseidon2Gate` row make its output wires
    `12..24` the image under `perm` of its input wires `0..12`. This is the soundness
    half of `Poseidon2.gate_sound_complete` on the row's checkpoint wires. -/
theorem poseidon2_gate_rows (perm : St p → St p) (pis : List (ZMod p))
    (consts : List (ZMod p)) (wires : ℕ → ZMod p) :
    UninterpretedRow perm pis ⟨.poseidon2, consts⟩ wires →
      (fun i : Fin 12 => wires (12 + i)) = perm fun i : Fin 12 => wires i := by
  rintro ⟨k, rfl, h⟩
  exact (Plonky2Spec.Poseidon2.gate_sound_complete _ _).mp
    ⟨_, _, _, by simp, by simp, by simp, h⟩

/-- **OBLIGATION — the public-input hash wiring.** The system a verifier key commits to
    hashes its circuit's public inputs into its `PublicInputGate`
    (`Built.HashesPublicInputs`): the `hash_n_to_hash_no_pad` rows `build` lays down and
    the copies from their outputs to the gate.

    The same kind of fact as `build_refines_export`, about the part of `build` that is
    not the export. A circuit with a certified build proves it from `Agrees`
    (`Built.Agrees.hashesPublicInputs`), and `proof_sound_of` takes that proof instead of
    this declaration. The leaf is certified (`Plonky2Bridge.LeafBuild`); the wrappers use
    this declaration. -/
axiom public_inputs_wired (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (b : Built p) :
    BuildOf (.node tag c children) b → b.HashesPublicInputs c

/-- **TRUSTED — public-input binding.** On `BindingGood`, the public inputs carried by
    the values a transcript for `pis` opens do not collide with `pis` under the
    proof-system hash (`PublicInputBinding`).

    This is the good event for the binding of `InnerHasher`, `PoseidonHash` in
    `PoseidonGoldilocksConfig`, on two lists of one length. It is computational, not
    statistical: a list longer than a digest has collisions, and the assumption is that
    the prover did not find one. In the random-oracle reading of
    `poseidon_transcript_oracle` it is a corollary, as Merkle binding is. It is declared
    because that implication does not carry it to the witness. It is not the wormhole
    Poseidon2.

    `TranscriptOpens` is a premise so that the statement is about a proof: without it it
    would quantify over every witness of the built system, and a colliding pair of lists
    exists for any hash into four elements.

    The length is not part of this event. `[1]` and `[1, 0]` have one digest, and the
    verifier tells them apart by `num_public_inputs`, which `IdenticalConstraints`
    carries.

    Discharging this is collision resistance of that sponge at a fixed input length. -/
axiom public_input_binding (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) :
    BindingGood perm (.node tag c children) → BuildOf (.node tag c children) b →
      TranscriptOpens perm (.node tag c children) pis o → DomainIdentities perm pis b o →
        PublicInputBinding (c.publicInputs.map (b.witness o)) pis

/-- The witness carries `pis` on the export's public-input targets: the lists have one
    digest (`Built.HashesPublicInputs.hash_eq`) and one length (`href`), and do not
    collide (`public_input_binding`). -/
theorem public_inputs_eq (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) (hev : BindingGood perm (.node tag c children))
    (hb : BuildOf (.node tag c children) b) (href : b.Refines c)
    (hpi : b.HashesPublicInputs c)
    (hto : TranscriptOpens perm (.node tag c children) pis o)
    (hlen : pis.length = b.numPublicInputs) (hid : DomainIdentities perm pis b o) :
    c.publicInputs.map (b.witness o) = pis :=
  public_input_binding perm tag c children pis b o hev hb hto hid
    (by rw [List.length_map, hlen, href.publicInputs]) (hpi.hash_eq hid)

/-- The rows a `verify_proof` gadget lays down under `r`'s verifier key hold, on targets
    whose public inputs carry `pis`. `opaque`: the gadget's rows are not modelled. -/
opaque GadgetSatisfied (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop

/-- **OBLIGATION — child `ProofAccepted`, the parent's half.** The built system holds,
    for each child of the tree, the `verify_proof` gadget the export omits, under that
    child's verifier key and with its public inputs on those targets. The identities
    therefore hold on the gadget's rows.

    This is the fidelity of `children` to the deployed circuit: the exporter's
    `verifiers` list, which each generated `tree_verifiers` checks against the tree.

    Discharging this is the same model of the deployed builder as `build_refines_export`,
    extended to the gadget rows. -/
axiom child_gadget_satisfied (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) :
    BuildOf (.node tag c children) b → DomainIdentities perm pis b o →
      ∀ v ∈ children, GadgetSatisfied perm v.1 (v.2.map (b.witness o))

/-- **OBLIGATION — native versus recursive `TranscriptAccepted`.** A satisfied
    `verify_proof` gadget is an accepted proof.

    `ProofAccepted` is what `poseidon_transcript_oracle` reads as `TranscriptAccepted`.
    At the root of a tree it comes from the native verifier
    (`verifier/src/plonk/verifier.rs`), at a child from the gadget
    (`plonky2/src/recursion/recursive_verifier.rs`). Both have to implement the one
    `TranscriptAccepted`: the same challenges from the same transcript, the same Merkle
    and FRI checks, the same vanishing identity. This declaration is the recursive half.
    The native half is how a capstone's `ProofAccepted` hypothesis is read, and no
    declaration consumes it.

    Discharging this is a model of both verifiers and a proof that the gadget's rows
    compute the native checks. -/
axiom recursive_verifier_transcript (perm : St p → St p) (r : Recursive p)
    (pis : List (ZMod p)) :
    GadgetSatisfied perm r pis → ProofAccepted perm r pis

/-- Under the gate identities of a system that refines `c`, the witness computes `perm`
    on `c`'s Poseidon2 rows. -/
theorem Built.Refines.poseidon2Rows {perm : St p → St p} {pis : List (ZMod p)}
    {b : Built p} {o : Opened p} {c : Circuit p} (href : b.Refines c)
    (hg : GateIdentities perm pis b o) : Poseidon2Rows perm c (b.witness o) := by
  intro i r hi hk
  have h := href.rows i r hi
  obtain ⟨kind, consts⟩ := r
  subst hk
  obtain ⟨i', cs, hrow, hloc⟩ := h
  have key := poseidon2_gate_rows perm pis cs (o.wire i') (hg.1 i' _ hrow).2
  have hin : poseidon2In (b.witness o) i = fun j : Fin 12 => o.wire i' j :=
    funext fun j => Built.witness_of_loc o (hloc j (by have := j.isLt; omega))
  have hout : poseidon2Out (b.witness o) i = fun j : Fin 12 => o.wire i' (12 + j) :=
    funext fun j => Built.witness_of_loc o (hloc (12 + j) (by have := j.isLt; omega))
  rw [hin, hout]
  exact key

/-- **Identical constraints satisfy the exported circuit.** The first piece of
    `verifier_algebra`: it uses `build_refines_export` and no other obligation. -/
theorem identical_constraints_satisfies (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    IdenticalConstraints perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a := by
  rintro ⟨b, o, hb, -, -, hid⟩
  exact ⟨b.witness o, (build_refines_export tag c children b hb).satisfies hid⟩

/-- **Verifier algebra.** Identical constraints are a satisfying assignment of the
    exported circuit: `build` agrees with the pre-`build` export (`Satisfies`), the
    Poseidon2 rows compute `perm`, the public-input wires match `pis`, and each omitted
    `verify_proof` gadget is `ProofAccepted` on the child inputs it carries.

    `Satisfies` is `Built.Refines.satisfies`, the Poseidon2 rows are
    `poseidon2_gate_rows`, and the public inputs are `Built.HashesPublicInputs.hash_eq`
    with `public_input_binding` on its good event. Three
    hypotheses are about the deployed builder: `href`, that the system the verifier key
    commits to refines the export; `hpi`, that it hashes the export's public inputs into
    its `PublicInputGate`; and `hch`, that the children's proofs were accepted on the
    targets the tree names. `verifier_algebra` supplies them from
    `build_refines_export`, `public_inputs_wired`, and `child_gadget_satisfied` with
    `recursive_verifier_transcript`; a circuit with a certified build supplies `href` and
    `hpi` from `Built.Agrees.refines` and `Built.Agrees.hashesPublicInputs`, and a circuit
    without children supplies `hch` vacuously. -/
theorem verifier_algebra_of (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p))
    (hev : BindingGood perm (.node tag c children))
    (href : ∀ b : Built p, BuildOf (.node tag c children) b → b.Refines c)
    (hpi : ∀ b : Built p, BuildOf (.node tag c children) b → b.HashesPublicInputs c)
    (hch : ∀ (b : Built p) (o : Opened p), BuildOf (.node tag c children) b →
      DomainIdentities perm pis b o →
        ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map (b.witness o))) :
    IdenticalConstraints perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
        c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) := by
  rintro ⟨b, o, hb, hto, hlen, hid⟩
  have href := href b hb
  exact ⟨b.witness o, href.satisfies hid, href.poseidon2Rows hid.gate,
    public_inputs_eq perm tag c children pis b o hev hb href (hpi b hb) hto hlen hid,
    hch b o hb hid⟩

/-- The children's `ProofAccepted` from the two declarations: the built system holds
    each child's gadget (`child_gadget_satisfied`), and a held gadget is an accepted proof
    (`recursive_verifier_transcript`). -/
theorem children_accepted (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) (hb : BuildOf (.node tag c children) b)
    (hid : DomainIdentities perm pis b o) :
    ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map (b.witness o)) := fun v hv =>
  recursive_verifier_transcript perm v.1 _
    (child_gadget_satisfied perm tag c children pis b o hb hid v hv)

/-- `verifier_algebra_of` with the three builder hypotheses supplied by the declarations.

    A proof of `proof_sound` may use only `poseidon_transcript_oracle`,
    `fri_proximity_gap`, `algebraic_sampling`, and the declarations above. -/
theorem verifier_algebra (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p))
    (hev : BindingGood perm (.node tag c children)) :
    IdenticalConstraints perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
        c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) :=
  verifier_algebra_of perm tag c children pis hev (build_refines_export tag c children)
    (public_inputs_wired tag c children) (children_accepted perm tag c children pis)

/-- **Proof-system soundness, composed.** On the good events for `node tag c children`,
    an accepted proof of it comes from an assignment that satisfies `c`'s exported
    wiring, computes `perm` on its Poseidon2 rows, carries `pis` on `c`'s public inputs,
    and — for every `verify_proof` gadget `(child, targets)` of `c` — had the child's
    proof accepted on the values those targets carry.

    This is `verifier_algebra_of` after `algebraic_sampling` after `fri_proximity_gap`
    after `poseidon_transcript_oracle`, and nothing else, with the three builder
    obligations `href`, `hpi` and `hch` as in `verifier_algebra_of`. The children's
    `ProofAccepted` is unconditional; reading it through `proof_sound` again needs the
    children's good events, which is how the wrapper capstones take one `GoodEvents` per
    tree. -/
theorem proof_sound_of (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p))
    (hg : GoodEvents perm (.node tag c children))
    (href : ∀ b : Built p, BuildOf (.node tag c children) b → b.Refines c)
    (hpi : ∀ b : Built p, BuildOf (.node tag c children) b → b.HashesPublicInputs c)
    (hch : ∀ (b : Built p) (o : Opened p), BuildOf (.node tag c children) b →
      DomainIdentities perm pis b o →
        ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map (b.witness o)))
    (h : ProofAccepted perm (.node tag c children) pis) :
    ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
      c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) :=
  verifier_algebra_of perm tag c children pis hg.binding href hpi hch
    (algebraic_sampling perm (.node tag c children) pis hg.sampling
      (fri_proximity_gap perm (.node tag c children) pis hg.proximity
        (poseidon_transcript_oracle perm tag c children pis hg.oracle h)))

/-- `proof_sound_of` with the builder obligations supplied by `build_refines_export`,
    `public_inputs_wired` and `child_gadget_satisfied`: the reading for a circuit whose
    build is not certified. -/
theorem proof_sound (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p))
    (hg : GoodEvents perm (.node tag c children))
    (h : ProofAccepted perm (.node tag c children) pis) :
    ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
      c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) :=
  proof_sound_of perm tag c children pis hg (build_refines_export tag c children)
    (public_inputs_wired tag c children) (children_accepted perm tag c children pis) h

/-- `proof_sound_of` for a circuit with a certified build and no children: `href` and
    `hpi` from `Built.Agrees.refines` and `Built.Agrees.hashesPublicInputs`, and nothing
    to say of children. Its footprint has neither `build_refines_export` nor
    `public_inputs_wired` nor `child_gadget_satisfied` nor
    `recursive_verifier_transcript`. -/
theorem proof_sound_leaf (perm : St p → St p) (tag : String) (c : Circuit p)
    (pis : List (ZMod p)) (hg : GoodEvents perm (.node tag c []))
    (href : ∀ b : Built p, BuildOf (.node tag c []) b → b.Refines c)
    (hpi : ∀ b : Built p, BuildOf (.node tag c []) b → b.HashesPublicInputs c)
    (h : ProofAccepted perm (.node tag c []) pis) :
    ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧ c.publicInputs.map a = pis :=
  let ⟨a, hs, hp, hpis, _⟩ :=
    proof_sound_of perm tag c [] pis hg href hpi (fun _ _ _ _ _ hv => nomatch hv) h
  ⟨a, hs, hp, hpis⟩

/-- `proof_sound` on a tree given by its projections. -/
theorem proof_sound' (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p))
    (hg : GoodEvents perm r) (h : ProofAccepted perm r pis) :
    ∃ a : Assignment p, Satisfies r.circuit a ∧ Poseidon2Rows perm r.circuit a ∧
      r.circuit.publicInputs.map a = pis ∧ ∀ v ∈ r.children, ProofAccepted perm v.1 (v.2.map a) := by
  obtain ⟨tag, c, children⟩ := r
  exact proof_sound perm tag c children pis hg h

omit [Fact p.Prime] in
/-- Two lists of public-input targets read the same values: componentwise. -/
theorem pis_eq {n : ℕ} {a a' : Assignment p} {ts ts' : Fin n → Target}
    (h : (List.ofFn ts').map a' = (List.ofFn ts).map a) : a' ∘ ts' = a ∘ ts := by
  rw [List.map_ofFn, List.map_ofFn] at h
  exact List.ofFn_injective h

end Plonky2Bridge
