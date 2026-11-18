/-
  The trusted base (PLAN.md §7, Steps 12 and 13).

  `proof_sound` is a theorem. It is the composition of three assumptions and
  `verifier_algebra`, and the capstones depend on nothing else outside the standard Lean
  axioms:

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
    refines. Five obligations remain as declarations: `build_refines_export`,
    `poseidon2_gate_rows`, `public_input_wires`, `child_gadget_satisfied`, and
    `recursive_verifier_transcript`.

  Poseidon2 collision resistance is not an assumption here. A forged proof need not
  exhibit a collision. The wormhole relations that do reduce to a collision are
  `*_or_collision` against `spongeRO` in the spec package.
-/
import Plonky2Spec.Wiring
import Plonky2Spec.Poseidon2

namespace Plonky2Bridge

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)

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
    polynomial `j` at `g^i`, and `products` has one grand product per challenge round
    (`num_challenges`). -/
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

/-- **Permutation identities.** Each grand product starts at `1` and steps by the ratio
    of the row's `wire + β·id + γ` to its `wire + β·σ + γ`, on every row of the domain.
    The verifier checks this through partial products (`check_partial_products`);
    multiplying its chunk equations out gives the step stated here. -/
def PermutationIdentities (b : Built p) (o : Opened p) : Prop :=
  ∀ g ∈ o.products, g.z 0 = 1 ∧ ∀ i < b.rows.length,
    g.z i * ∏ j ∈ Finset.range b.routed, (o.wire i j + g.beta * b.label (i, j) + g.gamma) =
      g.z ((i + 1) % b.rows.length) *
        ∏ j ∈ Finset.range b.routed, (o.wire i j + g.beta * b.label (b.sigma (i, j)) + g.gamma)

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
    column. -/
def Built.RowAt (b : Built p) (i : ℕ) (kind : GateKind) : Prop :=
  ∃ i' cs, b.rows[i']? = some ⟨kind, cs⟩ ∧ ∀ j, b.loc (.wire i j) = some (i', j)

/-- The built system carries the export's row `i`. Arithmetic rows are carried op by op,
    since the builder packs ops into whatever row has a free slot. Rows of the gates
    `rowConstraints` reads as `True` need not be carried, except `Poseidon2Gate` rows,
    which `Poseidon2Rows` reads. A `ConstantGate` row is read as pinned wires; the
    exports here have none, since `build` is what adds them. -/
def Built.Hosts (b : Built p) (i : ℕ) (r : Row p) : Prop :=
  match r.kind with
  | .arithmetic n => ∀ k < n, b.HostsOp i k (r.consts.getD 0 0) (r.consts.getD 1 0)
  | .constant n => ∀ j < n, b.Pins (.wire i j) (r.consts.getD j 0)
  | .baseSum2 n => b.RowAt i (.baseSum2 n)
  | .poseidon2 => b.RowAt i .poseidon2
  | .publicInput => True
  | .noop => True
  | .other _ => True

/-- **The built system refines the export `c`.** Every row, copy and constant target of
    `c` is carried by the built system, at the places `loc` gives. It is not an inclusion
    of rows: the built system may hold the export's rows at other indices, between rows
    of its own. -/
structure Built.Refines (b : Built p) (c : Circuit p) : Prop where
  rows : ∀ i r, c.rows[i]? = some r → b.Hosts i r
  copies : ∀ xy ∈ c.copies, b.Linked xy.1 xy.2
  constants : ∀ tc ∈ c.constants, b.Pins tc.1 tc.2

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

/-- What `rowConstraints` leaves out of a row's gate: the constraints of a `Poseidon2Gate`
    in a system where it computes `perm`, of the `PublicInputGate`, whose four hash wires
    are the proof-system hash of `pis`, and of the `.other` gates. The arithmetic,
    constant, base-sum and no-op gates have nothing left over. `opaque`: these gate
    polynomials are not modelled here. -/
opaque UninterpretedRow (perm : St p → St p) (pis : List (ZMod p)) (r : Row p)
    (wires : ℕ → ZMod p) : Prop

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
    to, they hold of the opened values `o`, and the witness is `b.witness o`.

    Both the copy and the permutation identities are stated. The transcript checks the
    second; it gives the first on the good event for `β`, `γ`, which is part of what
    `algebraic_sampling` conditions on. Only `algebraic_sampling` produces this. -/
def IdenticalConstraints (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop :=
  ∃ (b : Built p) (o : Opened p), BuildOf r b ∧ DomainIdentities perm pis b o

/-- **TRUSTED — random oracle.** Acceptance of a proof is an accepting interactive
    transcript (`TranscriptAccepted`) whose challenges are derived by the proof-system
    Poseidon (`PoseidonHash` in `PoseidonGoldilocksConfig`: Merkle caps and the
    Fiat–Shamir challenger). The post-quantum reading is the quantum random oracle.

    Collision resistance of that Poseidon is not this assumption. Collision resistance
    binds a Merkle opening; it does not make the challenges unpredictable, and the
    knowledge extractor rewinds the oracle. The wormhole Poseidon2 sponge is a
    different function and is not mentioned here.

    Discharging this is a Fiat–Shamir theorem for this IOP in the (quantum) random
    oracle model, instantiated by this Poseidon. -/
axiom poseidon_transcript_oracle (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    ProofAccepted perm (.node tag c children) pis →
      TranscriptAccepted perm (.node tag c children) pis

/-- **TRUSTED — FRI proximity gaps.** An accepting transcript means the opened
    polynomials are low degree in the sense of `LowDegreeOpenings`.

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
    TranscriptAccepted perm r pis → LowDegreeOpenings perm r pis

/-- **TRUSTED — algebraic sampling.** Low-degree openings whose vanishing, quotient,
    and permutation checks hold at the Fiat–Shamir challenges are identical constraints.

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

    This implication is the conjunction of those good events. Write `ε` for its
    failure probability, the sum of the base-field terms and the extension-field
    term. The good-event probability is `1 − ε`. No closed form is stated: a single
    factor `(degree · number of checks) / |F^D|` drops the base-field terms and
    understates `ε`. `ε` is independent of `friQueryBits` and of grinding, and it
    is not itself the `Prop`.

    Discharging this is a concrete bound that states the base-field terms and the
    extension-field term separately. -/
axiom algebraic_sampling (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) :
    LowDegreeOpenings perm r pis → IdenticalConstraints perm r pis

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
      Built.witness_of_loc o (hloc 0)
    have el : (limbWires i n).map (b.witness o) = (limbWires i' n).map o.assignment := by
      simp only [limbWires, List.map_map]
      exact List.map_congr_left fun j _ => Built.witness_of_loc o (hloc (j + 1))
    show Plonky2Spec.BaseSum 2 (b.witness o (.wire i 0)) ((limbWires i n).map (b.witness o))
    rw [e0, el]
    exact key
  | publicInput => trivial
  | noop => trivial
  | poseidon2 => trivial
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

  `verifier_algebra` is these five and the theorem above. None is a proximity conjecture
  or a sampling error. Four are deterministic. `public_input_wires` is not: it rests on
  the binding of the proof-system hash, as its docstring says. -/

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
    `Built.Refines` against an export of the built system. -/
axiom build_refines_export (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (b : Built p) :
    BuildOf (.node tag c children) b → b.Refines c

/-- **OBLIGATION — Poseidon2 rows.** The constraints of a `Poseidon2Gate` row make its
    output wires `12..24` the image under `perm` of its input wires `0..12`.

    `Poseidon2.gate_sound_complete` proves this of the extracted gate constraints, for
    the permutation they define. Discharging this is stating `UninterpretedRow` on these
    rows as those constraints, with `perm` that permutation. -/
axiom poseidon2_gate_rows (perm : St p → St p) (pis : List (ZMod p))
    (consts : List (ZMod p)) (wires : ℕ → ZMod p) :
    UninterpretedRow perm pis ⟨.poseidon2, consts⟩ wires →
      (fun i : Fin 12 => wires (12 + i)) = perm fun i : Fin 12 => wires i

/-- **OBLIGATION — public-input wires.** The witness carries `pis` on the export's
    public-input targets.

    This does not follow from the identities alone. `build` hashes the public-input
    targets in circuit and copies the digest to the `PublicInputGate`, whose constraint
    is that those four wires are the hash of `pis` the verifier computes
    (`get_public_inputs_hash`, the config's `InnerHasher`). The identities therefore give
    equal digests. Equal lists follow only where that hash does not collide, which is a
    property of the proof-system hash and not a field identity. The conclusion is kept
    as `verifier_algebra` had it.

    Discharging this needs the in-circuit hash rows to compute that hash, and its binding
    on the public inputs, which is an assumption on the proof-system hash. -/
axiom public_input_wires (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) (b : Built p)
    (o : Opened p) :
    BuildOf (.node tag c children) b → DomainIdentities perm pis b o →
      c.publicInputs.map (b.witness o) = pis

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
    funext fun j => Built.witness_of_loc o (hloc j)
  have hout : poseidon2Out (b.witness o) i = fun j : Fin 12 => o.wire i' (12 + j) :=
    funext fun j => Built.witness_of_loc o (hloc (12 + j))
  rw [hin, hout]
  exact key

/-- **Identical constraints satisfy the exported circuit.** The first piece of
    `verifier_algebra`: it uses `build_refines_export` and no other obligation. -/
theorem identical_constraints_satisfies (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    IdenticalConstraints perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a := by
  rintro ⟨b, o, hb, hid⟩
  exact ⟨b.witness o, (build_refines_export tag c children b hb).satisfies hid⟩

/-- **Verifier algebra.** Identical constraints are a satisfying assignment of the
    exported circuit: `build` agrees with the pre-`build` export (`Satisfies`), the
    Poseidon2 rows compute `perm`, the public-input wires match `pis`, and each omitted
    `verify_proof` gadget is `ProofAccepted` on the child inputs it carries.

    `Satisfies` is `Built.Refines.satisfies`. The rest is the five obligations above.

    A proof of `proof_sound` may use only `poseidon_transcript_oracle`,
    `fri_proximity_gap`, `algebraic_sampling`, and those five. -/
theorem verifier_algebra (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    IdenticalConstraints perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
        c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) := by
  rintro ⟨b, o, hb, hid⟩
  have href := build_refines_export tag c children b hb
  exact ⟨b.witness o, href.satisfies hid, href.poseidon2Rows hid.gate,
    public_input_wires perm tag c children pis b o hb hid, fun v hv =>
      recursive_verifier_transcript perm v.1 _
        (child_gadget_satisfied perm tag c children pis b o hb hid v hv)⟩

/-- **Proof-system soundness, composed.** An accepted proof of `node tag c children`
    comes from an assignment that satisfies `c`'s exported wiring, computes `perm` on
    its Poseidon2 rows, carries `pis` on `c`'s public inputs, and — for every
    `verify_proof` gadget `(child, targets)` of `c` — had the child's proof accepted
    on the values those targets carry.

    This is `verifier_algebra` after `algebraic_sampling` after `fri_proximity_gap`
    after `poseidon_transcript_oracle`, and nothing else. -/
theorem proof_sound (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p))
    (h : ProofAccepted perm (.node tag c children) pis) :
    ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
      c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) :=
  verifier_algebra perm tag c children pis
    (algebraic_sampling perm (.node tag c children) pis
      (fri_proximity_gap perm (.node tag c children) pis
        (poseidon_transcript_oracle perm tag c children pis h)))

/-- `proof_sound` on a tree given by its projections. -/
theorem proof_sound' (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p))
    (h : ProofAccepted perm r pis) :
    ∃ a : Assignment p, Satisfies r.circuit a ∧ Poseidon2Rows perm r.circuit a ∧
      r.circuit.publicInputs.map a = pis ∧ ∀ v ∈ r.children, ProofAccepted perm v.1 (v.2.map a) := by
  obtain ⟨tag, c, children⟩ := r
  exact proof_sound perm tag c children pis h

omit [Fact p.Prime] in
/-- Two lists of public-input targets read the same values: componentwise. -/
theorem pis_eq {n : ℕ} {a a' : Assignment p} {ts ts' : Fin n → Target}
    (h : (List.ofFn ts').map a' = (List.ofFn ts).map a) : a' ∘ ts' = a ∘ ts := by
  rw [List.map_ofFn, List.map_ofFn] at h
  exact List.ofFn_injective h

end Plonky2Bridge
