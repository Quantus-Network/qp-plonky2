/-
  The trusted base (PLAN.md §7, Step 12).

  `proof_sound` is a theorem. It is the composition of three declarations, and the
  capstones depend on nothing else outside the standard Lean axioms:

  * `poseidon_transcript_oracle` — the proof-system Poseidon is a random oracle on
    the verifier transcript (QROM for the post-quantum reading). Merkle binding is a
    corollary. This is `PoseidonGoldilocksConfig`'s Poseidon, not the wormhole Poseidon2.
  * `fri_proximity_gap` — FRI proximity gaps at the circuit's FRI parameters. The
    soundness error is statistical and is not a `Prop`.
  * `verifier_algebra` — the unproved mathematics: vanishing and permutation checks
    on low-degree openings are a satisfying assignment of the exported circuit, the
    native and recursive verifiers implement those checks, and `build` agrees with
    the export. Not a hash assumption and not a coding conjecture.

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
  FRI queries and `proofOfWorkBits` of grinding. The ethSTARK accounting this
  repository targets is `numQueries * rateBits + proofOfWorkBits`. The deployed
  wormhole circuit may use a different `FriConfig`; `fri_proximity_gap` is the
  conjecture at whatever parameters that circuit was built with, and `ε` is a
  function of those parameters. -/

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

/-- ethSTARK bit-accounting for `CircuitConfig.security_bits`. -/
def friConjecturedBits (q : FriQueryParams) : ℕ :=
  q.numQueries * q.rateBits + q.proofOfWorkBits

example : friConjecturedBits standardRecursionFri = 100 := rfl

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
    whose verifier accepted, not a property of either hash. For
    `standardRecursionFri` the in-repo accounting is
    `friConjecturedBits standardRecursionFri = 100`, i.e. a statistical error of
    about `2^{-100}` on top of whatever the oracle contributes. That probability is
    not a `Prop`: this implication is the `1 − ε` event. A perfect hash leaves `ε`.

    Discharging this is a proof of proximity gaps out to the list-decoding radius
    this blowup uses. The unique-decoding regime is proved in the literature; the
    parameters above sit past it. -/
axiom fri_proximity_gap (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) :
    TranscriptAccepted perm r pis → LowDegreeOpenings perm r pis

/-- **OBLIGATION — verifier algebra.** Not a cryptographic assumption. Low-degree
    openings that passed the transcript checks are a satisfying assignment of the
    exported circuit: the vanishing and permutation identities, `build` agreeing
    with the pre-`build` export (`Satisfies`), the Poseidon2 rows computing `perm`,
    the public-input wires matching `pis`, and each omitted `verify_proof` gadget
    being `ProofAccepted` on the child inputs it carries. The native verifier and
    the recursive verifier both have to implement `TranscriptAccepted`.

    A proof of `proof_sound` may use only `poseidon_transcript_oracle`,
    `fri_proximity_gap`, and this obligation. -/
axiom verifier_algebra (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    LowDegreeOpenings perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
        c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a)

/-- **Proof-system soundness, composed.** An accepted proof of `node tag c children`
    comes from an assignment that satisfies `c`'s exported wiring, computes `perm` on
    its Poseidon2 rows, carries `pis` on `c`'s public inputs, and — for every
    `verify_proof` gadget `(child, targets)` of `c` — had the child's proof accepted
    on the values those targets carry.

    This is `verifier_algebra` after `fri_proximity_gap` after
    `poseidon_transcript_oracle`, and nothing else. -/
theorem proof_sound (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p))
    (h : ProofAccepted perm (.node tag c children) pis) :
    ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
      c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a) :=
  verifier_algebra perm tag c children pis
    (fri_proximity_gap perm (.node tag c children) pis
      (poseidon_transcript_oracle perm tag c children pis h))

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
