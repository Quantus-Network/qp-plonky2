# Formal verification plan — the gadget→spec bridge (layer 2)

This package formalizes the **constraint semantics** of the plonky2 gadgets the
wormhole circuits actually use, so that the spec relations in
`qp-zk-circuits/formal` (`R_leaf`, `R_L0`, `R_L1`) can be proved to be *what the
circuit constraints actually enforce* — not merely asserted.

## 1. The trust stack (why this layer)

```
(4) Security properties     ← qp-zk-circuits/formal  (DONE: conservation, exclusivity, …)
      ⇧
(3) Spec relations R_leaf…  ← qp-zk-circuits/formal  (DONE: as Lean Props)
      ⇧  ===  THIS PACKAGE: the bridge  ===
(2) Circuit constraints     ← gadget semantics here + a faithful model of each circuit()
      ⇧
(1) Proof-system soundness  ← FRI + Plonk arithmetization + Fiat–Shamir + recursion  (TRUSTED, see §7)
```

- **(1)** says "a valid proof ⟹ *some* witness satisfies *this* constraint system."
  It is agnostic to what the constraints *mean*. We treat it as an explicit,
  documented assumption (§7); fully verifying a recursive SNARK prover is a
  multi-year research program and is out of scope.
- **(3)/(4)** reason about hand-written relations and their consequences. They say
  nothing about the emitted gates.
- **(2) — this package — is the missing rung.** It is where under/over-constraint
  bugs live (a missing constraint lets a forged witness through, and (1) will
  happily prove the under-constrained system). Highest bug-value per unit effort.

The end-to-end theorem we are building toward:

```
(1) plonky2 soundness  ∧  (2) circuit constraints ⟺ R_leaf  ⟹  (valid proof ⟹ R_leaf on public inputs)  ⟹ (4)
```

## 2. Scope

**In scope:** a Lean model of each gadget's emitted constraint as a predicate over
field assignments, with **both** lemmas relating the predicate to its arithmetic
meaning:
- **soundness** — constraints ⟹ spec (the security direction: no forged witness),
- **completeness** — spec ⟹ ∃ witness satisfying constraints (no honest prover is
  locked out; this is the direction that surfaces *hidden assumptions* — in the
  Zellic engagement completeness exposed an undocumented `carryIn ≤ 15` precondition
  that callers could violate).

Each gadget separates an **`Assumptions`** record (preconditions the *surrounding*
circuit must enforce — e.g. "inputs are canonical u32") from a **`Spec`** record
(what *this* gadget enforces). The `Assumptions` become explicit wiring obligations
the circuit model must discharge. We then model each `circuit()`'s constraint
conjunction and prove it implies the corresponding spec relation. Surfacing latent
under/over-constraint assumptions is an explicit goal, not just confirmation.

**Out of scope:** FRI/Plonk/Fiat–Shamir soundness, recursion-gadget internals
(`verify_proof`), and Poseidon2 *collision resistance* (that stays the
`RandomOracle` idealization — we prove only that the gadget *computes* the hash).

## 3. Fidelity — mechanically export gate constraints (prior art)

Methodology follows Zellic's *Formal Verification of a Plonky2 Gate* (Lighter
engagement). The faithfulness of the constraint model to the Rust is the crux, so
we do **not** hand-transcribe gate constraints. Instead:

1. **Export.** A small Rust tool symbolically evaluates each gate's
   `eval_unfiltered` (run with a symbolic field that builds an expression AST) and
   emits the polynomial constraints as Lean definitions. The Lean we reason about
   is then mechanically derived from the real gate code, not transcribed.
2. **Generalize.** Export is per-parameter (e.g. a fixed `num_ops`); we write
   generic constraint-generating functions and let Lean check they agree with each
   concrete export, then prove the generic statement once.
3. **Structured constraints.** Package the raw exported constraint vector into a
   named record (with an equivalence lemma) so proofs read by purpose, not index.

Residual fidelity gap (smaller than hand-modeling, but real): the exporter covers
**gate** constraints per row; the full circuit also has **wiring/copy
constraints** (which gates are instantiated and how their wires are connected by
`connect`). That routing is still modeled by hand at the circuit level, mitigated
by 1:1 structural correspondence with `circuit()` + the existing differential
tests (`spec_differential.rs`). Full circuit export (gates + copy constraints) is
future work; the Zellic scope was gate-level too.

## 4. Gadget inventory (derived from the live circuit code)

Constraint-emitting gadgets used by the leaf + aggregator circuits (witness
allocation and infra calls excluded):

| Tier | Gadget(s) | Constraint meaning | Discharges |
|---|---|---|---|
| **T0** arithmetic/wiring | `constant` `zero` `one`, `add` `sub` `mul`, `connect`/`connect_hashes`/`connect_shared_targets`, `register_public_input(s)` | field equalities; `connect a b` ⟺ `a = b` | equality clauses; `feeOk` arithmetic; shared-target wiring |
| **T1** boolean + selection | `is_equal`, `and` `or` `not` `_false`, `add_virtual_bool_target_safe` (b²=b), `select`, `bytes_digest_eq` (=`is_equal`×4 + `and`) | boolean algebra + conditional mux | `R_L0` dummy flags, dedup `is_duplicate`, block-ref prefix scan, nullifier `select` |
| **T2** range/comparison | `range_check(x,n)` (14/32/48), `enforce_target_less_than_const` | `range_check(x,n)` ⟺ `x < 2ⁿ` | all `R_leaf` `inRange 32`, `feeOk` 48-bit non-overflow, 14-bit fee complement, depth ≤ `MAX_DEPTH`; ties to `Encoding.lean` |
| **T3** hash | `hash_n_to_hash_no_pad_p2::<Poseidon2Hash>` | gadget output = Poseidon2 sponge of input | realizes `RandomOracle.H`: `WA`, `Null`, `leafHash`, `nodeHash`/`stepUp`, `dummyNull`, block-hash |
| **T4** recursion | `verify_proof` (via `add_recursive_verifiers`) | "π attests stmt x under vk" | `R_L0`/`R_L1` child attestation — **TRUSTED (§7)** |

Coverage map:
- **Bridge `R_leaf`** needs **T0 + T1 + T2 + T3** (no recursion at the leaf).
- **Bridge `R_L0`/`R_L1`** needs **T0 + T1 + T2** for the wrapper logic, plus
  **T4 axiomatized**.

**Two-level structure (gate vs gadget).** The exporter operates on *gates*; the
builder *gadgets* above lower to them. We verify the gate once (exported,
sound+complete) and compose up to the gadget. The gates our gadgets emit
(confirmed in `plonky2/src/gates/`):
- T0/T1 arithmetic + select + boolean → `ArithmeticBaseGate` (`arithmetic_base.rs`,
  computes `c₀·xy + c₁·z`; `select`, `add`, `sub`, `mul`, `b²=b` are all instances).
- T2 `range_check` → `BaseSumGate` (`base_sum.rs`, base-B limb split + range).
- T3 hash → the Poseidon2 gate(s) (`poseidon2.rs`, `poseidon2_mds.rs`,
  `poseidon2_int_mix.rs`) — the case where hand-modeling is infeasible and the
  exporter is required.

## 5. Package & tooling conventions

- Location: `qp-plonky2/formal/` (Lean package `Plonky2Spec`), co-located with the
  gadgets it models so it is reusable across circuits.
- **Field, not `Nat`. Uses mathlib.** Unlike `qp-zk-circuits/formal` (which is
  `Nat`-based and deliberately mathlib-free), the gadget layer reasons about real
  field behavior — `ZMod goldilocks`, `Fact (Nat.Prime p)`, `.val`, base-B digit
  decomposition, Horner reconstruction. That is mathlib territory; hand-rolling it
  would be enormous and is the wrong call. So this package **depends on mathlib**
  (matching the Zellic setup: `import Mathlib.Data.ZMod.Defs`, etc.).
- **Why this doesn't pull mathlib into the fast spec build.** Three layers, meeting
  at the `.val` interface (see §6):
  1. `qp-zk-circuits/formal` — Spec: `Nat`, RO-abstract, **mathlib-free**, builds
     in seconds. *Unchanged.*
  2. `qp-plonky2/formal` — GadgetSemantics: `ZMod p`, **mathlib**, exporter-backed.
  3. Bridge: imports both; proves `circuitConstraints ⟹ R_leaf`. Lives **with this
     package** (depends on mathlib) and imports the spec as a light dep — so the
     hermetic spec build stays mathlib-free.
- Pinned `lean-toolchain` to the **Lean version mathlib's release pins** (mathlib
  is the binding constraint; reconcile with qp-zk-circuits' v4.30.0 when wiring the
  bridge — bump the spec toolchain if needed).
- CI mirrors the qp-zk-circuits `formal-spec` job: `leanprover/lean-action`
  (pinned SHA) running `lake build` + `leanchecker: true`, plus the grep-based
  **no-`sorry`/`admit`** gate. **Add `lake exe cache get`** (mathlib prebuilt
  `.olean` cache) so CI doesn't compile mathlib from source. No `nanoda` (crashes
  on v4.30 export format).

## 6. Composition with qp-zk-circuits

The gadget model is field-level (`ZMod p`, predicates over assignments). `R_leaf`
et al. are felt-level (`Felt = Nat`) over the abstract `RandomOracle`. The
**`.val` map is the interface** between the two — exactly the abstraction Zellic
use to lift field reasoning to u32/Nat: a gadget proven to enforce
`x.val < 2^32` discharges the spec's `inRange 32 x`; a gadget proven to compute
Poseidon2 instantiates `RandomOracle.H` (its felt outputs are `digest.val`s).

1. This package exports gadget predicates + `sound`/`complete` lemmas and a model
   `LeafCircuit.constraints` of the leaf `circuit()`.
2. The **bridge target** (in this package, mathlib-enabled) adds
   `qp-zk-circuits/formal` as a light `lake` dep and proves
   `LeafCircuit.constraints assignment ⟹ Rleaf ro p w` over `.val`, with the
   `RandomOracle` instantiated by the T3 "gadget computes Poseidon2" result.
   Keeping the bridge here (not in the spec package) preserves the spec's
   mathlib-free, seconds-long build.

## 7. Trusted assumptions (the seam to layer 1)

Captured as explicit Lean `axiom`s in a dedicated `Trusted.lean`, mirroring how
`RandomOracle` makes collision-resistance explicit:

- **`verify_proof` soundness:** `verify_proof(vk, x, π)` constrains that `π` is a
  valid proof of statement `x` under `vk`; given (1), a satisfied recursion gadget
  implies the child's public-input relation holds.
- **Proof-system soundness (1):** FRI query soundness, Plonk/AIR arithmetization,
  Fiat–Shamir (and **QROM** Fiat–Shamir for the post-quantum claims), recursion.

Each axiom is documented with what it would take to discharge it, so the
trusted base is auditable and individually replaceable later.

## 8. Execution plan (commit checkpoint between each step)

**Exporter sequencing.** The `BaseSumGate` (range) is small enough to hand-model in
Step 1 to validate the full Lean pipeline (`Assumptions`/`Spec`, sound, complete)
cheaply. The exporter is *built in Step 2* and is *mandatory by Step 3* — the
Poseidon2 gate has far too many constraints to transcribe reliably. Each gadget
gets an `Assumptions` record + a `Spec` record and **both** `sound` and `complete`
lemmas (per §2).

### Step 1 — T2 `range_check` (+ package scaffold), hand-modeled  ✅ DONE
- Scaffolded `formal/` (`lakefile.toml`, `lean-toolchain` v4.30.0,
  `Plonky2Spec.lean`, `.gitignore`, CI job).
- Hand-modeled the `BaseSumGate<B>` constraints (`BaseSum` structured record:
  `reconstruct B limbs = sum` + per-limb range), with `NoWrap` side-condition;
  proved **soundness** (`baseSum_sound : BaseSum ⟹ sum < Bᴸ`) and **completeness**
  (`baseSum_complete : sum < Bᴸ ⟹ ∃ base-B digit witness`). Specialized to
  `rangeCheck (x,n)` (`B = 2`): `rangeCheck_sound`/`rangeCheck_complete`, the
  `rangeCheck_implies_inRange` bridge lemma, and modeled
  `enforce_target_less_than_const`.
- **Acceptance met:** `lake build` + `leanchecker` clean, no `sorry`; `#print
  axioms` shows only `propext`/`Quot.sound`/`Classical.choice`;
  `rangeCheck_implies_inRange` upgrades the spec's `inRange n x` from assumption to
  consequence of `range_check`.
- *Checkpoint: reviewed + committed (initial `Nat`/`.val` model).*

### Step 1.5 — `ZMod p` field-native port (mathlib)  ✅ DONE
The initial Step-1 commit modeled the per-limb range as its semantic content
(`limbᵢ < B`) over `Nat`, deferring the one genuinely field-algebraic fact —
"degree-`B` product `∏_{j<B}(limbᵢ − j) = 0` ⟹ `limbᵢ ∈ {0,…,B-1}`" — because that
needs `ZMod p` to be an integral domain (`Nat.Prime`/Euclid are not in core Lean).
With mathlib now wired in (pinned `v4.30.0`), this is **discharged**:
- `Basic.lean` keeps the field-agnostic positional-arithmetic core over `ℕ`
  (mathlib-free); `RangeCheck.lean` is now **field-native over `ZMod p`**.
- The gate's per-limb constraint is modeled faithfully as
  `limbRangeProduct B x = ∏_{j ∈ range B} (x − j)`, and
  `limb_val_lt_of_product` proves `product = 0 ⟹ x.val < B` via
  `Finset.prod_eq_zero_iff` + `mul_eq_zero` (the integral-domain step) — exactly the
  rung that was previously a modeling assumption.
- `reconstructF` (field) is bridged to the `ℕ` `reconstruct` through `ZMod.val`
  (`reconstructF_eq_cast`), so the place-value bounds in `Basic.lean` govern the
  field computation. `baseSum_sound`/`baseSum_complete` and the
  `rangeCheck_*`/`inRange` lemmas are all restated and proved over `ZMod p`.
- Generality follows Zellic: theorems are over an arbitrary `[Fact p.Prime]` with
  explicit no-wrap bounds (`Bᴸ ≤ p`); soundness needs the bound, completeness does
  not (the reconstructed value is the canonical `sum.val < p`).
- CI `formal-spec` job flipped to `use-mathlib-cache: "true"`.
- **Acceptance:** `lake build` + `leanchecker` clean, no `sorry`; `#print axioms`
  shows only `propext`/`Classical.choice`/`Quot.sound`.
- *Checkpoint: review + commit.*

### Step 2 — T0/T1 gates + constraint exporter + wrapper bridge
Sliced into three checkpoints. The T0/T1 gates are small algebraic identities, so
they are hand-modeled and machine-checked **first** (2a); those Lean models then
serve as the oracle the Rust exporter must reproduce (2b); the wrapper bridge (2c)
consumes both.

#### Step 2a — T0/T1 gadget semantics (hand-modeled)  ✅ DONE
- `Arithmetic.lean` (T0): `ArithmeticConstraint c0 c1 m0 m1 addend output`
  (`output = c0·m0·m1 + c1·addend`, arithmetic_base.rs:87-89) with `arithmetic_iff`,
  and the `add/sub/mul/mul_add/mul_sub` specs as instantiations (arithmetic.rs).
- `Boolean.lean` (T1): `IsBool` with `isBool_iff_assertBool` (the `b²−b=0`
  constraint ⇔ booleanity, integral-domain); `bnot/band/bor` with closure +
  truth-table lemmas; `bmux` (`_if`) and `bselect` (`select`) with `=`-equivalence
  and the `b=1→x`, `b=0→y` reductions; and the **`is_equal`** gadget (`IsEqual`
  two-constraint record) with full soundness (`isEqual_isBool`, `isEqual_iff :
  equal=1 ↔ x=y`) and completeness (`isEqual_complete`).
- **Acceptance met:** `lake build` + `leanchecker` clean, no `sorry`; `#print
  axioms` standard-only.
- *Checkpoint: review + commit.*

#### Step 2b — Constraint exporter (Rust)
- Build the Rust exporter (symbolic `eval_unfiltered` → Lean), validated by
  re-deriving the Step-1 range constraints **and** the 2a arithmetic constraint and
  checking they match the hand models. Generalize the export.
- **Acceptance:** exporter reproduces Step-1 + 2a constraints verbatim.
- *Checkpoint: review + commit.*

#### Step 2c — `R_L0` wrapper-logic bridge  ✅ DONE (field-level)
`Wrapper.lean` proves each layer-0 wrapper primitive
(`build_layer0_wrapper_constraints`, circuit_logic.rs) computes *exactly* the
conditional/selection that the corresponding `RL0` definition is built from, all on
the 2a lemmas:
- `nullifier_replacement` — `select(is_dummy, dnull, real) = if is_dummy then …`
  (the `nullifiersReplaced` per-slot equation).
- `match_contribution` / `dedup_select` — `select(eq, a, 0)` / `select(dup, 0, acc)`
  as `matchSum` / `groupAux`'s `if`s (via `isEqual_iff`).
- `andAll` + `andAll_eq_one_iff` — `bytes_digest_eq` as the AND of per-limb
  equalities (`= 1 ↔ all limbs equal`), the digest-equality used by metadata/dedup.
- `block_consistency` / `real_block_matches` — `or(is_dummy, matches) = 1` ⟹ a
  non-dummy child matches the reference (`metadataConsistent`).
- `scanStep` + `scan_locked` + `scanFirst_correct` — the prefix scan selects the
  **first real slot's** value (`referenceFromFirstReal`; the position-independent
  privacy fix), with the all-dummy batch keeping the zero initial reference.
- **Acceptance met:** `lake build` + `leanchecker` clean, no `sorry`; `#print axioms`
  standard-only.
- **Remaining seam (the cross-package composition `circuit ⟹ RL0`):** wiring
  `qp-zk-circuits/formal` in as a `lake` dep and reconciling `Felt = Nat` ↔ `ZMod p`
  via `.val`, plus instantiating `RandomOracle.dummyNull` with the Poseidon
  `H(H(u))` result — i.e. it depends on Step 3 and the cross-repo dep. The
  selection logic (everything built from T0/T1) is discharged here.
- *Checkpoint: review + commit.*

### Step 3 — T3 Poseidon2 hash "computes H" (exporter-driven)
- Export the Poseidon2 gate constraints (permutation: round function, MDS, round
  constants); model the `hash_no_pad` sponge over them; prove gadget output equals
  the native Poseidon2 function. Instantiate the spec's `RandomOracle.H` with it.
- **Acceptance:** `gadgetHash = nativePoseidon2` theorem; the existing
  differential tests become a checked corollary rather than only empirical.
- *Checkpoint: review + commit.*

### Step 4 — T4 axiomatize `verify_proof` + aggregation bridge
- Add `Trusted.lean` (§7). Model `R_L0`/`R_L1` circuit constraints over the
  wrapper gadgets + the trusted `verify_proof`, and prove they imply `RL0`/`RL1`.
- **Acceptance:** aggregation bridge lemmas build clean; trusted base fully
  enumerated in `Trusted.lean`.
- *Checkpoint: review + commit.*

## 9. Definition of done

`R_leaf` fully bridged (T0–T3), `R_L0`/`R_L1` bridged modulo the enumerated
trusted assumptions (T4 + layer 1), each gate proven **sound and complete** with
its `Assumptions`/`Spec` split, CI green on all gates, and the trusted base
documented in `Trusted.lean`. The remaining gap is exactly (a) the residual
**wiring/copy-constraint** model fidelity (§3 — gate constraints are exporter-
backed; routing is still hand-modeled) and (b) the layer-1 assumptions (§7) — both
explicit.
