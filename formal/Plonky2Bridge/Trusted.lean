/-
  The trusted base (PLAN.md Step 10): one axiom, proof-system soundness for an exported
  recursion tree.

  Everything else in `Plonky2Spec` and `Plonky2Bridge` is a theorem. This module is the
  only place an `axiom` is declared, and it names no relation and no particular circuit:
  it says that when the recursive verifier gadget accepts a proof, the proved circuit's
  exported constraint system is satisfiable on the accepted public inputs, and the child
  proofs its own gadgets verified were accepted in turn. The circuit-specific content the
  aggregation soundness argument needs — that a satisfying leaf assignment is an `Rleaf`
  instance, that a satisfying wrapper assignment is an `RPrivateBatch` instance — is proved
  from the exported wiring (`leaf_wired`, `Wrapper{2,4}.sound`, `PublicWrapper{2,4}.sound`),
  so `WormholeSpec`'s former `leaf_proof_sound` / `private_batch_proof_sound` are derived here
  (`LeafCircuit.accepted_sound`, `Wrapper{2,4}.accepted_sound`) rather than assumed.

  WHAT `proof_sound` STANDS FOR. `add_recursive_verifiers → builder.verify_proof` lays down
  the plonky2 verifier for the child's verifier key as ordinary gates in the parent. The
  traces do not contain those rows (`TracingBuilder::verify_proof` records the gadget as a
  `Recursive` child instead), so `Satisfies parent a` says nothing about the child proof;
  `ProofAccepted` is the meaning of those omitted rows, and turning it into a satisfying
  child assignment is exactly proof-system soundness: FRI query soundness, the Plonk/AIR
  arithmetization, Fiat–Shamir (QROM Fiat–Shamir for the post-quantum claims) and the
  recursion composition. That is rung (1) of the trust stack (PLAN.md §1); formalizing it
  is a verified plonky2 verifier, out of scope.

  FIDELITY CLAIMS FOLDED INTO THE AXIOM. (i) The verifier key baked into each gadget is
  that of the circuit the trace names as the child (`<name>.verifiers`, checked against the
  Lean tree by `rfl` in each generated bridge): production passes the built child's
  `VerifierCircuitData` into `add_recursive_verifiers`, so this holds by construction.
  (ii) The exported pre-`build` system is what `build` compiles (Wiring.lean header), and
  the Poseidon2 gate computes `perm` — carried as a parameter, as `RandomOracle` was —
  so a satisfying real witness restricts to an assignment satisfying `Satisfies ∧
  Poseidon2Rows perm`.
-/
import Plonky2Spec.Wiring
import Plonky2Spec.Poseidon2

namespace Plonky2Bridge

open Plonky2Spec.Wiring
open Plonky2Spec.Poseidon2 (St)

variable {p : ℕ} [Fact p.Prime]

/-- The recursive verifier gadget under `r`'s verifier key accepted a proof whose public
    inputs are `pis`, in a system whose Poseidon2 gate computes `perm`. `opaque`, not
    `axiom`: an abstract predicate whose truth is never assumed; only `proof_sound` below
    says what its satisfaction attests. -/
opaque ProofAccepted (perm : St p → St p) (r : Recursive p) (pis : List (ZMod p)) : Prop

/-- **TRUSTED — proof-system soundness.** An accepted proof of the tree `node tag c children`
    comes from an assignment that satisfies `c`'s exported wiring, computes `perm` on its
    Poseidon2 rows, carries `pis` on `c`'s public inputs, and — for every `verify_proof`
    gadget `(child, targets)` of `c` — had the child's proof accepted on the values those
    targets carry. -/
axiom proof_sound (perm : St p → St p) (tag : String) (c : Circuit p)
    (children : List (Recursive p × List Target)) (pis : List (ZMod p)) :
    ProofAccepted perm (.node tag c children) pis →
      ∃ a : Assignment p, Satisfies c a ∧ Poseidon2Rows perm c a ∧
        c.publicInputs.map a = pis ∧ ∀ v ∈ children, ProofAccepted perm v.1 (v.2.map a)

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
