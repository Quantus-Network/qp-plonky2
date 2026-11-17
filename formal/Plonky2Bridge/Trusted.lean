/-
  The trusted base (PLAN.md §7, Step 12).

  `proof_sound` is a theorem. It is the composition of four declarations, and the
  capstones depend on nothing else outside the standard Lean axioms:

  * `poseidon_transcript_oracle` — the proof-system Poseidon is a random oracle on
    the verifier transcript (QROM for the post-quantum reading). Merkle binding is a
    corollary. This is `PoseidonGoldilocksConfig`'s Poseidon, not the wormhole Poseidon2.
  * `fri_proximity_gap` — FRI proximity gaps. The query error is statistical
    (`2^{-84}` for the default config). Grinding is a separate work factor.
  * `algebraic_sampling` — point checks at the Fiat–Shamir challenges imply identical
    constraints only on a good event. Schwartz–Zippel and permutation-challenge
    failures remain after ideal hashes and perfect FRI.
  * `verifier_algebra` — the deterministic remainder: identical constraints are a
    satisfying assignment of the exported circuit.

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

/-- The gate, copy, and permutation identities hold on the whole evaluation domain,
    not merely at the sampled challenges, and the opened values are the corresponding
    witness. `opaque`: only `algebraic_sampling` produces it. -/
opaque IdenticalConstraints (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop

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

    The native verifier checks those identities at the sampled point `ζ`
    (`eval_vanishing_poly` against the quotient in `verifier/src/plonk/verifier.rs`),
    not on the whole domain. A nonzero discrepancy of degree below the FRI claim can
    still vanish there. Schwartz–Zippel and a bad permutation challenge remain when
    the hash is ideal and FRI proximity is perfect; the low-degree-only IOP keeps a
    nonzero error (Lemma 4.9 of *Fiat-Shamir Security of FRI and Related SNARKs*).
    This implication is that good event. Its probability is on the order of
    `(degree · number of checks) / |F^D|`, independent of `friQueryBits` and of grinding.
    It is not a `Prop`.

    Discharging this is the concrete Schwartz–Zippel bound for these checks. -/
axiom algebraic_sampling (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) :
    LowDegreeOpenings perm r pis → IdenticalConstraints perm r pis

/-- **OBLIGATION — verifier algebra.** Deterministic given `IdenticalConstraints`.
    Not a hash assumption, not a proximity conjecture, and not a sampling error:
    those are the three axioms above. Identical constraints are a satisfying assignment
    of the exported circuit: `build` agrees with the pre-`build` export (`Satisfies`),
    the Poseidon2 rows compute `perm`, the public-input wires match `pis`, and each
    omitted `verify_proof` gadget is `ProofAccepted` on the child inputs it carries.
    The native verifier and the recursive verifier both have to implement
    `TranscriptAccepted`.

    A proof of `proof_sound` may use only `poseidon_transcript_oracle`,
    `fri_proximity_gap`, `algebraic_sampling`, and this obligation. -/
axiom verifier_algebra (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    IdenticalConstraints perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
        c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a)

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
