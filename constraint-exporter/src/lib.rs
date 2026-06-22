//! qp-plonky2 constraint exporter.
//!
//! Symbolically executes plonky2 gates' real `Gate::eval_unfiltered` and emits
//! the resulting constraint polynomials as Lean definitions, for the formal
//! spec under `../formal`. See `../formal/PLAN.md` Step 2b and `symbolic.rs`.

pub mod extract;
pub mod render;
pub mod symbolic;

use core::fmt::Write as _;

use extract::Extracted;

fn emit(out: &mut String, e: &Extracted) {
    // Parameter list: every gate wire then every gate constant.
    let mut params = String::new();
    for i in 0..e.num_wires {
        let _ = write!(params, "w{i} ");
    }
    for i in 0..e.num_consts {
        let _ = write!(params, "c{i} ");
    }
    debug_assert!(!params.is_empty(), "a gate with no wires/constants?");

    for (idx, &c) in e.constraints.iter().enumerate() {
        let _ = writeln!(
            out,
            "/-- `{name}` constraint #{idx}, extracted verbatim from \
             `{name_rust}::eval_unfiltered`. -/",
            name = e.name,
            name_rust = e.name,
        );
        let _ = writeln!(
            out,
            "def {}_c{idx} ({}: ZMod p) : ZMod p :=\n  {}\n",
            e.name,
            params,
            render::to_lean(c),
        );
    }
}

/// Build the full contents of `formal/Plonky2Spec/Generated/Gates.lean`.
pub fn generate_lean() -> String {
    let mut out = String::new();
    out.push_str(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Produced by the `qp-plonky2-constraint-exporter` dev tool, which symbolically\n\
         \x20 executes each gate's real `Gate::eval_unfiltered` (over a symbolic field) and\n\
         \x20 prints the constraint polynomials it emits. Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\n\
         \x20 Each `def …_c{i}` is the i-th constraint the gate forces to zero, with `w{j}`\n\
         \x20 the j-th `local_wires` entry and `c{j}` the j-th `local_constants` entry.\n\
         \x20 `Generated/Bridge.lean` proves each of these equals the corresponding\n\
         \x20 hand-written model in `Arithmetic.lean` / `RangeCheck.lean` (by `ring`), so a\n\
         \x20 drift between the gate code and the spec breaks `lake build`.\n\
         -/\n\
         import Mathlib.Algebra.Field.ZMod\n\n\
         namespace Plonky2Spec.Generated\n\n\
         -- Extracted defs carry every gate wire/constant as a parameter, so some are\n\
         -- unused in a given constraint; that is intentional and not a code smell.\n\
         set_option linter.unusedVariables false\n\n\
         variable {p : ℕ}\n\n",
    );
    // Render each gate immediately after extracting it. The symbolic arena is a
    // single thread-local cleared at the start of every extraction, so a handle
    // is only valid until the *next* extraction — never hold handles from two
    // gates at once. These gates are exactly those hand-modeled in
    // `Arithmetic.lean` / `RangeCheck.lean`; `Generated/Bridge.lean` pins them.
    emit(&mut out, &extract::arithmetic_gate());
    // BaseSumGate<2>, 2 limbs: reconstruction `(w1 + 2·w2) − w0` plus one
    // degree-2 range product `wi·(wi − 1)` per limb.
    emit(&mut out, &extract::base_sum_gate::<2>(2, "baseSum2"));
    out.push_str("end Plonky2Spec.Generated\n");
    out
}

/// Build the contents of `formal/Plonky2Spec/Generated/Poseidon2.lean`.
///
/// The Poseidon2 permutation gate emits 118 constraints sharing a large common
/// sub-DAG (the round state), so unlike `generate_lean` this emits a **single**
/// `def` returning the constraint *list*, with one `let` binding per shared
/// arithmetic node (see `render::emit_lets`). One inlined-tree def per constraint
/// would be exponential.
pub fn generate_poseidon2_lean() -> String {
    // Extract first; the arena then holds exactly this gate's nodes, which
    // `emit_lets` / `ref_str` read. Do not extract anything else before rendering.
    let ex = extract::poseidon2_gate();

    let mut out = String::new();
    out.push_str(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Produced by `qp-plonky2-constraint-exporter` from the real\n\
         \x20 `Poseidon2Gate::eval_unfiltered` (plonky2/src/gates/poseidon2.rs), symbolically\n\
         \x20 executed over a symbolic field. Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\n\
         \x20 `poseidon2Gate w0 … w{n}` returns the list of constraint polynomials the gate\n\
         \x20 forces to zero (`wᵢ` = `local_wires[i]`: `w0..w11` input, `w12..w23` output, the\n\
         \x20 rest per-round S-box-input checkpoints). The round constants are the gate's own\n\
         \x20 `self.params` values (independent of the field), so they appear verbatim here.\n\
         \x20 Emitted as a straight-line `let`-program: one binding per shared subexpression,\n\
         \x20 since the 22 internal rounds reuse the running sum heavily.\n\n\
         \x20 Step 3b proves this list is all-zero iff `output = Poseidon2_perm(input)`.\n\
         -/\n\
         import Mathlib.Algebra.Field.ZMod\n\n\
         namespace Plonky2Spec.Generated\n\n\
         set_option linter.unusedVariables false\n\
         -- The straight-line `let`-program nests ~2.4k bindings deep; raise the\n\
         -- elaboration recursion limit above that depth.\n\
         set_option maxRecDepth 8000\n\n\
         variable {p : ℕ}\n\n",
    );

    let mut params = String::new();
    for i in 0..ex.num_wires {
        let _ = write!(params, "w{i} ");
    }
    let _ = writeln!(
        out,
        "/-- Poseidon2 permutation gate: the {n} constraints extracted verbatim from \
         `Poseidon2Gate::eval_unfiltered`. -/",
        n = ex.constraints.len(),
    );
    let _ = writeln!(
        out,
        "def poseidon2Gate ({params}: ZMod p) : List (ZMod p) :="
    );
    out.push_str(&render::emit_lets("  "));
    let roots: Vec<String> = ex.constraints.iter().map(|&r| render::ref_str(r)).collect();
    let _ = writeln!(out, "  [{}]\n", roots.join(",\n   "));
    out.push_str("end Plonky2Spec.Generated\n");
    out
}

/// Build `formal/Plonky2Spec/Generated/Poseidon2Prims.lean`: the three Poseidon2
/// permutation *primitives* (`sbox7`, `mdsLight`, `internalMix`), each extracted by
/// running the **real** helper (`sbox7_base` / `mds_light_base` / `internal_mix_base`,
/// poseidon2.rs) over the symbolic field. They are small (linear MDS / mix; one
/// `x^7`), so they are emitted inline. `Generated/Poseidon2Bridge.lean` proves each
/// equals the hand model `Plonky2Spec.Poseidon2.{…}` by `ring`, so the arithmetic of
/// the permutation is machine-checked against the live Rust — the structural round
/// composition that `Plonky2Spec.Poseidon2.gateConstraints` builds from these
/// primitives is the only remaining reviewed/diff-tested piece.
pub fn generate_poseidon2_prims_lean() -> String {
    // Each extractor calls `reset()`, so render its result before the next call.
    let sbox = render::to_lean(extract::sbox7_prim());
    let mds: Vec<String> = extract::mds_light_prim()
        .iter()
        .map(|&s| render::to_lean(s))
        .collect();
    let imix: Vec<String> = extract::internal_mix_prim()
        .iter()
        .map(|&s| render::to_lean(s))
        .collect();

    let mut out = String::new();
    out.push_str(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 The three Poseidon2 permutation primitives, each extracted by running the real\n\
         \x20 `plonky2/src/gates/poseidon2.rs` helper over the symbolic field. Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\n\
         \x20 * `sbox7 w0`            = `sbox7_base(w0)`               (the `x^7` S-box)\n\
         \x20 * `mdsLight w0..w11`    = `mds_light_base([w0..w11])`    (external light MDS)\n\
         \x20 * `internalMix w0..w23` = `internal_mix_base([w0..w11], diag=[w12..w23])`\n\n\
         \x20 `Generated/Poseidon2Bridge.lean` proves each equals the opaque hand model\n\
         \x20 `Plonky2Spec.Poseidon2.{sbox7,mdsLight,internalMix}` by `ring`.\n\
         -/\n\
         import Mathlib.Algebra.Field.ZMod\n\
         import Mathlib.Data.Fin.VecNotation\n\n\
         namespace Plonky2Spec.Generated\n\n\
         set_option linter.unusedVariables false\n\n\
         variable {p : ℕ}\n\n",
    );

    let _ = writeln!(
        out,
        "/-- `sbox7_base(w0) = w0^7`, extracted verbatim. -/\n\
         def sbox7 (w0 : ZMod p) : ZMod p :=\n  {sbox}\n"
    );

    let mut p12 = String::new();
    for i in 0..12 {
        let _ = write!(p12, "w{i} ");
    }
    let _ = writeln!(
        out,
        "/-- `mds_light_base([w0..w11])`, lane outputs, extracted verbatim. -/\n\
         def mdsLight ({p12}: ZMod p) : Fin 12 → ZMod p :=\n  ![{}]\n",
        mds.join(",\n    ")
    );

    let mut p24 = String::new();
    for i in 0..24 {
        let _ = write!(p24, "w{i} ");
    }
    let _ = writeln!(
        out,
        "/-- `internal_mix_base([w0..w11], diag=[w12..w23])`, lane outputs, extracted. -/\n\
         def internalMix ({p24}: ZMod p) : Fin 12 → ZMod p :=\n  ![{}]\n",
        imix.join(",\n    ")
    );

    out.push_str("end Plonky2Spec.Generated\n");
    out
}

#[cfg(test)]
mod tests {
    use plonky2::field::extension::{Extendable, FieldExtension};
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::gates::arithmetic_base::ArithmeticGate;
    use plonky2::gates::base_sum::BaseSumGate;
    use plonky2::gates::gate::Gate;
    use plonky2::gates::poseidon2::Poseidon2Gate;
    use plonky2::hash::hash_types::HashOut;
    use plonky2::plonk::vars::EvaluationVars;
    use symbolic::GOLDILOCKS_ORDER;

    use super::*;

    type GF = GoldilocksField;
    type Ext2 = <GoldilocksField as Extendable<2>>::Extension;

    /// Simple deterministic LCG so the differential test needs no `rand` plumbing.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> GF {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            GF::from_canonical_u64(self.0 % GOLDILOCKS_ORDER)
        }
    }

    /// Run the *real* gate over the degree-2 Goldilocks extension at a base-field
    /// point (imaginary parts zero) and return the real component of each
    /// constraint. Because the gate constraints are polynomials with base-field
    /// coefficients, this equals the gate's constraint polynomial evaluated over
    /// the base field — the ground truth the symbolic extraction must match.
    fn real_eval(out_constraints: Vec<Ext2>) -> Vec<GF> {
        out_constraints
            .iter()
            .map(|e| <Ext2 as FieldExtension<2>>::to_basefield_array(e)[0])
            .collect()
    }

    fn embed(xs: &[GF]) -> Vec<Ext2> {
        xs.iter().map(|&x| Ext2::from(x)).collect()
    }

    #[test]
    fn arithmetic_extraction_matches_real_gate() {
        let ex = extract::arithmetic_gate();
        let mut rng = Lcg(0x0123_4567_89ab_cdef);
        for _ in 0..200 {
            let wires: Vec<GF> = (0..ex.num_wires).map(|_| rng.next()).collect();
            let consts: Vec<GF> = (0..ex.num_consts).map(|_| rng.next()).collect();

            let sym_vals: Vec<GF> = ex
                .constraints
                .iter()
                .map(|&c| render::eval(c, &wires, &consts))
                .collect();

            let gate = ArithmeticGate { num_ops: 1 };
            let we = embed(&wires);
            let ce = embed(&consts);
            let pih = HashOut::<GF>::ZERO;
            let vars = EvaluationVars {
                local_constants: &ce,
                local_wires: &we,
                public_inputs_hash: &pih,
            };
            let real = real_eval(<ArithmeticGate as Gate<GF, 2>>::eval_unfiltered(
                &gate, vars,
            ));

            assert_eq!(sym_vals, real, "arithmetic gate constraint mismatch");
        }
    }

    #[test]
    fn base_sum_extraction_matches_real_gate() {
        const NUM_LIMBS: usize = 2;
        let ex = extract::base_sum_gate::<2>(NUM_LIMBS, "baseSum2");
        let mut rng = Lcg(0xfeed_face_dead_beef);
        for _ in 0..200 {
            let wires: Vec<GF> = (0..ex.num_wires).map(|_| rng.next()).collect();

            let sym_vals: Vec<GF> = ex
                .constraints
                .iter()
                .map(|&c| render::eval(c, &wires, &[]))
                .collect();

            let gate = BaseSumGate::<2>::new(NUM_LIMBS);
            let we = embed(&wires);
            let pih = HashOut::<GF>::ZERO;
            let vars = EvaluationVars {
                local_constants: &[],
                local_wires: &we,
                public_inputs_hash: &pih,
            };
            let real = real_eval(<BaseSumGate<2> as Gate<GF, 2>>::eval_unfiltered(
                &gate, vars,
            ));

            assert_eq!(sym_vals, real, "base sum gate constraint mismatch");
        }
    }

    #[test]
    fn poseidon2_extraction_matches_real_gate() {
        // Extract once; evaluate the shared DAG with the memoized evaluator
        // (recursive `eval` would blow up on the internal rounds).
        let ex = extract::poseidon2_gate();
        assert_eq!(
            ex.constraints.len(),
            118,
            "expected 118 Poseidon2 constraints"
        );
        let mut rng = Lcg(0x00c0_ffee_1234_5678);
        for _ in 0..16 {
            let wires: Vec<GF> = (0..ex.num_wires).map(|_| rng.next()).collect();

            let vals = render::eval_all(&wires, &[]);
            let sym_vals: Vec<GF> = ex
                .constraints
                .iter()
                .map(|&r| render::eval_root(r, &vals))
                .collect();

            let gate = Poseidon2Gate::<GF, 2>::new();
            let we = embed(&wires);
            let pih = HashOut::<GF>::ZERO;
            let vars = EvaluationVars {
                local_constants: &[],
                local_wires: &we,
                public_inputs_hash: &pih,
            };
            let real = real_eval(<Poseidon2Gate<GF, 2> as Gate<GF, 2>>::eval_unfiltered(
                &gate, vars,
            ));

            assert_eq!(sym_vals, real, "poseidon2 gate constraint mismatch");
        }
    }

    /// The three extracted Poseidon2 primitives (`Generated/Poseidon2Prims.lean`)
    /// evaluate to exactly what the *real* `plonky2` helpers compute. Together with
    /// the Lean `Poseidon2Bridge` (`= Plonky2Spec.Poseidon2` primitives by `ring`),
    /// this anchors the hand model's S-box / light-MDS / internal-diffusion arithmetic
    /// to the live Rust code at random base-field points.
    #[test]
    fn poseidon2_primitives_match_real_helpers() {
        use plonky2::gates::poseidon2::{
            internal_mix_base, mds_light_base, sbox7_base, SPONGE_WIDTH,
        };

        let mut rng = Lcg(0xdead_d00d_face_b00c);

        // sbox7: one wire in, one out. (Each `extract::*` resets the shared arena,
        // so extract once per primitive and evaluate its handles before the next.)
        let sbox = extract::sbox7_prim();
        for _ in 0..200 {
            let x = rng.next();
            assert_eq!(
                render::eval(sbox, &[x], &[]),
                sbox7_base(x),
                "sbox7 mismatch"
            );
        }

        // mds_light_base: state w0..w11.
        let mds = extract::mds_light_prim();
        for _ in 0..200 {
            let state: Vec<GF> = (0..SPONGE_WIDTH).map(|_| rng.next()).collect();
            let mut real_state: [GF; SPONGE_WIDTH] = state.clone().try_into().unwrap();
            mds_light_base(&mut real_state);
            for (lane, &c) in mds.iter().enumerate() {
                assert_eq!(
                    render::eval(c, &state, &[]),
                    real_state[lane],
                    "mds_light lane {lane} mismatch",
                );
            }
        }

        // internal_mix_base: state w0..w11, diagonal w12..w23.
        let imix = extract::internal_mix_prim();
        for _ in 0..200 {
            let s: [GF; SPONGE_WIDTH] = core::array::from_fn(|_| rng.next());
            let diag: [GF; SPONGE_WIDTH] = core::array::from_fn(|_| rng.next());
            let mut wires: Vec<GF> = s.to_vec();
            wires.extend_from_slice(&diag);
            let real_imix = internal_mix_base(&s, &diag);
            for (lane, &c) in imix.iter().enumerate() {
                assert_eq!(
                    render::eval(c, &wires, &[]),
                    real_imix[lane],
                    "internal_mix lane {lane} mismatch",
                );
            }
        }
    }

    /// Differential test of the Step-3c Lean sponge model
    /// (`formal/Plonky2Spec/Sponge.lean`) against the real `Poseidon2Hash::hash_no_pad`.
    ///
    /// `lean_sponge` below is a *transliteration* of the Lean `pad10` / `addBlock` /
    /// `absorbMsg` / `squeeze4`, run over the **same** width-12 permutation the real hash
    /// uses (`<GoldilocksField as P2Permuter>::permute`). Agreement at random inputs of
    /// every length around the rate boundary pins the Lean wrapper's structure — the
    /// `10*` padding, additive absorption on the 8 rate lanes (capacity untouched), the
    /// per-block permute, and the 4-lane squeeze with no trailing permute — to the live
    /// Rust. (The permutation itself is the Step-3b exporter-backed object; this test
    /// targets the sponge wrapper that Step 3c adds on top of it.)
    #[test]
    fn sponge_structure_matches_lean_model() {
        use plonky2::gates::poseidon2::SPONGE_WIDTH;
        use plonky2::hash::poseidon2::{P2Permuter, Poseidon2Hash};
        use plonky2::plonk::config::Hasher;

        const RATE: usize = 8; // Sponge.lean `rate`

        // `pad10` (Sponge.lean): append a single `1`, zero-fill to a multiple of RATE.
        fn pad10(inputs: &[GF]) -> Vec<GF> {
            let n = inputs.len();
            let padded = ((n + 1 + (RATE - 1)) / RATE) * RATE;
            let mut msg = Vec::with_capacity(padded);
            msg.extend_from_slice(inputs);
            msg.push(GF::ONE);
            msg.resize(padded, GF::ZERO);
            msg
        }

        // `spongeHash` (Sponge.lean): zero state, additive absorb on the rate lanes,
        // permute per block, squeeze the first 4 lanes (no trailing permute).
        fn lean_sponge(inputs: &[GF]) -> [GF; 4] {
            let msg = pad10(inputs);
            let mut state = [GF::ZERO; SPONGE_WIDTH];
            for block in msg.chunks(RATE) {
                // addBlock: state[i] += block[i] for i < block.len(); capacity lanes stay.
                for (i, &x) in block.iter().enumerate() {
                    state[i] = state[i] + x;
                }
                state = <GF as P2Permuter>::permute(state);
            }
            [state[0], state[1], state[2], state[3]] // squeeze4
        }

        let mut rng = Lcg(0xa5a5_5a5a_1234_abcd);
        // Lengths 0..=20 cover every padding boundary: empty (full pad block), < rate,
        // == rate (forces a whole extra `[1,0,…]` block), and multi-block messages.
        for len in 0..=20usize {
            for _ in 0..20 {
                let inputs: Vec<GF> = (0..len).map(|_| rng.next()).collect();
                let lean = lean_sponge(&inputs);
                let real = Poseidon2Hash::hash_no_pad(&inputs).elements;
                assert_eq!(lean, real, "sponge structure mismatch at len {len}");
            }
        }
    }
}
