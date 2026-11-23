//! The proof-system Poseidon's tables and test vector for Lean (Step 18).
//!
//! `formal/Plonky2Spec/Poseidon.lean` models the `PoseidonGate` and the permutation it
//! computes for any `Consts`. This module writes the Goldilocks tables of
//! `impl Poseidon for GoldilocksField` as `Plonky2Spec.Poseidon.goldilocks`, in the
//! padded shape the Lean model indexes (`initMat` with `e₀` as its first row and column,
//! `w r` with `M₀₀` in front of the `w^` row, `v r` with an unused lane `0`), and one
//! input/output pair of `Poseidon::poseidon`, which the generated module checks against
//! `Consts.perm` by kernel evaluation. It also writes an input/output pair of
//! `PoseidonHash::hash_no_pad`, the proof-system public-input hash, which
//! `Plonky2Bridge/Generated/PublicInputsHashVector.lean` checks against
//! `Plonky2Bridge.publicInputsHash` the same way (Step 19).

use core::fmt::Write as _;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64, PrimeField64};
use plonky2::hash::poseidon::{
    Poseidon, PoseidonHash, ALL_ROUND_CONSTANTS, N_PARTIAL_ROUNDS, N_ROUNDS, SPONGE_RATE,
    SPONGE_WIDTH,
};
use plonky2::plonk::config::Hasher;

/// The tables of `Plonky2Spec.Poseidon.Consts`, as the Lean model indexes them.
pub struct Tables {
    pub circ: [u64; SPONGE_WIDTH],
    pub diag: [u64; SPONGE_WIDTH],
    pub rc: [[u64; SPONGE_WIDTH]; N_ROUNDS],
    pub partial_first: [u64; SPONGE_WIDTH],
    pub partial_rc: [u64; N_PARTIAL_ROUNDS],
    pub init_mat: [[u64; SPONGE_WIDTH]; SPONGE_WIDTH],
    pub w: [[u64; SPONGE_WIDTH]; N_PARTIAL_ROUNDS],
    pub v: [[u64; SPONGE_WIDTH]; N_PARTIAL_ROUNDS],
}

type GF = GoldilocksField;

/// The Goldilocks tables (`core/src/poseidon_goldilocks.rs`).
pub fn goldilocks_tables() -> Tables {
    let circ = <GF as Poseidon>::MDS_MATRIX_CIRC;
    let diag = <GF as Poseidon>::MDS_MATRIX_DIAG;
    // `0` in front of each row of an `11 × …` table, so that the Lean model indexes
    // lanes `1..12` where the Rust tables index `0..11`.
    let padded = |row: &[u64; SPONGE_WIDTH - 1], first: u64| -> [u64; SPONGE_WIDTH] {
        core::array::from_fn(|j| if j == 0 { first } else { row[j - 1] })
    };
    Tables {
        circ,
        diag,
        rc: core::array::from_fn(|r| {
            core::array::from_fn(|i| ALL_ROUND_CONSTANTS[r * SPONGE_WIDTH + i])
        }),
        partial_first: <GF as Poseidon>::FAST_PARTIAL_FIRST_ROUND_CONSTANT,
        partial_rc: <GF as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS,
        init_mat: core::array::from_fn(|r| {
            if r == 0 {
                padded(&[0; SPONGE_WIDTH - 1], 1)
            } else {
                padded(
                    &<GF as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1],
                    0,
                )
            }
        }),
        w: core::array::from_fn(|r| {
            padded(
                &<GF as Poseidon>::FAST_PARTIAL_ROUND_W_HATS[r],
                circ[0] + diag[0],
            )
        }),
        v: core::array::from_fn(|r| padded(&<GF as Poseidon>::FAST_PARTIAL_ROUND_VS[r], 0)),
    }
}

/// The test vector: `Poseidon::poseidon` of the lanes `0, 1, …, 11`.
pub fn test_vector() -> ([u64; SPONGE_WIDTH], [u64; SPONGE_WIDTH]) {
    let input: [GF; SPONGE_WIDTH] = core::array::from_fn(GF::from_canonical_usize);
    let output = <GF as Poseidon>::poseidon(input);
    (
        input.map(|x| x.to_canonical_u64()),
        output.map(|x| x.to_canonical_u64()),
    )
}

/// `PoseidonHash::hash_no_pad` on `1, 2, …, 11` — a leaf's public-input count: a full rate
/// block and a partial one, so the vector exercises the overwrite absorption of a second
/// block and the no-padding of a short one.
pub fn hash_no_pad_vector() -> (Vec<u64>, [u64; 4]) {
    let input: Vec<GF> = (1..=SPONGE_RATE as u64 + 3)
        .map(GF::from_canonical_u64)
        .collect();
    let output = PoseidonHash::hash_no_pad(&input).elements;
    (
        input.iter().map(|x| x.to_canonical_u64()).collect(),
        output.map(|x| x.to_canonical_u64()),
    )
}

/// `![a, b, …]`: small values in decimal on one line, `u64`-sized ones in zero-padded hex
/// four a line, continuation lines indented by `indent`.
fn vec_lean(xs: &[u64], indent: &str) -> String {
    if xs.iter().all(|&x| x < 1 << 32) {
        let lanes: Vec<String> = xs.iter().map(u64::to_string).collect();
        return format!("![{}]", lanes.join(", "));
    }
    let lanes: Vec<String> = xs.iter().map(|x| format!("{x:#018x}")).collect();
    let lines: Vec<String> = lanes.chunks(4).map(|c| c.join(", ")).collect();
    format!("![{}]", lines.join(&format!(",\n{indent}")))
}

/// `![![…], …]`: one inner vector per row.
fn rows_lean(rows: &[[u64; SPONGE_WIDTH]]) -> String {
    let inner: Vec<String> = rows.iter().map(|r| vec_lean(r, "      ")).collect();
    format!("![{}]", inner.join(",\n    "))
}

/// Build `formal/Plonky2Spec/Generated/PoseidonConsts.lean`.
pub fn generate_poseidon_consts_lean() -> String {
    let t = goldilocks_tables();
    let (input, output) = test_vector();
    let mut out = String::new();
    out.push_str(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 The Goldilocks tables of the proof-system Poseidon (`impl Poseidon for\n\
         \x20 GoldilocksField`, core/src/poseidon_goldilocks.rs) as a `Plonky2Spec.Poseidon.Consts`,\n\
         \x20 and a test vector of `Poseidon::poseidon` on them. Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\n\
         \x20 `initMat`, `w` and `v` are the Rust tables in the shape the model indexes:\n\
         \x20 `FAST_PARTIAL_ROUND_INITIAL_MATRIX` below and right of `e₀`, `MDS_MATRIX_CIRC[0] +\n\
         \x20 MDS_MATRIX_DIAG[0]` in front of each `FAST_PARTIAL_ROUND_W_HATS` row, an unused `0` in\n\
         \x20 front of each `FAST_PARTIAL_ROUND_VS` row.\n\n\
         \x20 `goldilocks_perm_test` evaluates `Consts.perm` on the tables in the kernel and is what\n\
         \x20 pins the model's arithmetic to the native permutation: the matrices, the constants,\n\
         \x20 the round structure. `poseidon_hand_gate_constraints_match_real_gate` (the exporter's\n\
         \x20 tests) pins the model's constraint walk to `PoseidonGate::eval_unfiltered`.\n\
         -/\n\
         import Plonky2Spec.Poseidon\n\n\
         namespace Plonky2Spec.Poseidon\n\n\
         open Plonky2Spec.Poseidon2 (St)\n\n\
         variable {p : ℕ}\n\n",
    );
    let _ = writeln!(
        out,
        "/-- The tables of `impl Poseidon for GoldilocksField`. -/\n\
         def goldilocks : Consts p where\n\
         \x20 circ := {}\n\
         \x20 diag := {}\n\
         \x20 rc := {}\n\
         \x20 partialFirst := {}\n\
         \x20 partialRc := {}\n\
         \x20 initMat := {}\n\
         \x20 w := {}\n\
         \x20 v := {}\n",
        vec_lean(&t.circ, "    "),
        vec_lean(&t.diag, "    "),
        rows_lean(&t.rc),
        vec_lean(&t.partial_first, "    "),
        vec_lean(&t.partial_rc, "    "),
        rows_lean(&t.init_mat),
        rows_lean(&t.w),
        rows_lean(&t.v),
    );
    let _ = writeln!(
        out,
        "/-- The Goldilocks prime, `2^64 − 2^32 + 1`. -/\n\
         def goldilocksPrime : ℕ := {prime:#x}\n\n\
         /-- The lanes `0, 1, …, 11`. -/\n\
         def testInput : St goldilocksPrime := {input}\n\n\
         /-- `Poseidon::poseidon` of `testInput`. -/\n\
         def testOutput : St goldilocksPrime :=\n\
         \x20 {output}\n\n\
         /-- The model's permutation on the Goldilocks tables is the native one, on the test\n\
         \x20   vector. Kernel-evaluated. -/\n\
         theorem goldilocks_perm_test : goldilocks.perm testInput = testOutput := by\n\
         \x20 decide +kernel\n",
        prime = GF::ORDER,
        input = vec_lean(&input, "  "),
        output = vec_lean(&output, "    "),
    );
    out.push_str("end Plonky2Spec.Poseidon\n");
    out
}

/// `formal/Plonky2Bridge/Generated/PublicInputsHashVector.lean`: the no-pad sponge model
/// of the public-input hash evaluated in the kernel on a vector of the real
/// `PoseidonHash::hash_no_pad`.
pub fn generate_public_inputs_hash_vector_lean() -> String {
    let (input, output) = hash_no_pad_vector();
    let input: Vec<String> = input.iter().map(u64::to_string).collect();
    format!(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 A test vector of `PoseidonHash::hash_no_pad`, the proof-system hash of a public-input\n\
         \x20 list (`get_public_inputs_hash`), checked against its model `publicInputsHash`\n\
         \x20 (`Plonky2Bridge/Trusted.lean`) by kernel evaluation. Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\n\
         \x20 The input is eleven lanes, a leaf's public-input count: a full rate block and a\n\
         \x20 partial one. The vector pins what the differential test\n\
         \x20 `public_inputs_hash_matches_lean_model` (the exporter's tests) checks at random\n\
         \x20 inputs — no padding, overwrite absorption, a permutation per block, the four-lane\n\
         \x20 squeeze — on the permutation `goldilocks_perm_test` pins.\n\
         -/\n\
         import Plonky2Bridge.Trusted\n\n\
         namespace Plonky2Bridge\n\n\
         open Plonky2Spec.Poseidon (goldilocksPrime)\n\n\
         /-- The lanes `1, 2, …, {}`. -/\n\
         def publicInputsHashTestInput : List (ZMod goldilocksPrime) := [{}]\n\n\
         /-- `PoseidonHash::hash_no_pad` of `publicInputsHashTestInput`. -/\n\
         def publicInputsHashTestOutput : Fin 4 → ZMod goldilocksPrime :=\n\
         \x20 {}\n\n\
         set_option maxRecDepth 8192 in\n\
         /-- The model of the public-input hash is the native one, on the test vector.\n\
         \x20   Kernel-evaluated. -/\n\
         theorem publicInputsHash_test :\n\
         \x20   publicInputsHash publicInputsHashTestInput = publicInputsHashTestOutput := by\n\
         \x20 decide +kernel\n\n\
         end Plonky2Bridge\n",
        input.len(),
        input.join(", "),
        vec_lean(&output, "    "),
    )
}

#[cfg(test)]
mod tests {
    //! Differential tests of the Lean model against the live Rust: the transliterations
    //! below follow `formal/Plonky2Spec/Poseidon.lean` definition by definition, on the
    //! padded tables the generator writes.

    use plonky2::field::extension::Extendable;
    use plonky2::field::types::Field;
    use plonky2::gates::gate::Gate;
    use plonky2::gates::poseidon::PoseidonGate;
    use plonky2::hash::hash_types::HashOut;
    use plonky2::plonk::vars::EvaluationVars;

    use super::*;

    type St = [GF; SPONGE_WIDTH];

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> GF {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            GF::from_noncanonical_u64(self.0)
        }
        fn state(&mut self) -> St {
            core::array::from_fn(|_| self.next())
        }
    }

    fn f(x: u64) -> GF {
        GF::from_canonical_u64(x)
    }

    fn sbox7(x: GF) -> GF {
        x.exp_u64(7)
    }

    fn mds_layer(t: &Tables, s: &St) -> St {
        core::array::from_fn(|r| {
            (0..SPONGE_WIDTH)
                .map(|i| f(t.circ[i]) * s[(i + r) % SPONGE_WIDTH])
                .sum::<GF>()
                + f(t.diag[r]) * s[r]
        })
    }

    fn partial_init(t: &Tables, s: &St) -> St {
        core::array::from_fn(|c| (0..SPONGE_WIDTH).map(|r| s[r] * f(t.init_mat[r][c])).sum())
    }

    fn partial_fast(t: &Tables, r: usize, s: &St) -> St {
        core::array::from_fn(|i| {
            if i == 0 {
                (0..SPONGE_WIDTH).map(|j| s[j] * f(t.w[r][j])).sum()
            } else {
                s[0] * f(t.v[r][i]) + s[i]
            }
        })
    }

    fn add_rc(rc: &[u64; SPONGE_WIDTH], s: &St) -> St {
        core::array::from_fn(|i| s[i] + f(rc[i]))
    }

    fn full_round(t: &Tables, r: usize, s: &St) -> St {
        mds_layer(t, &add_rc(&t.rc[r], s).map(sbox7))
    }

    fn partial_round(t: &Tables, r: usize, s: &St) -> St {
        let mut s = *s;
        s[0] = sbox7(s[0]) + f(t.partial_rc[r]);
        partial_fast(t, r, &s)
    }

    const FIRST_FULL: [usize; 4] = [0, 1, 2, 3];
    const LAST_FULL: [usize; 4] = [26, 27, 28, 29];

    /// `Consts.perm`.
    fn perm(t: &Tables, inp: &St) -> St {
        let mut s = *inp;
        for r in FIRST_FULL {
            s = full_round(t, r, &s);
        }
        s = partial_init(t, &add_rc(&t.partial_first, &s));
        for r in 0..N_PARTIAL_ROUNDS {
            s = partial_round(t, r, &s);
        }
        for r in LAST_FULL {
            s = full_round(t, r, &s);
        }
        s
    }

    fn swap_in(delta: &[GF; 4], inp: &St) -> St {
        core::array::from_fn(|i| match i {
            0..4 => inp[i] + delta[i],
            4..8 => inp[i] - delta[i - 4],
            _ => inp[i],
        })
    }

    /// `Consts.fullConstrs`.
    fn full_constrs(t: &Tables, rounds: &[usize], cks: &[St], s: &St) -> (Vec<GF>, St) {
        let mut constrs = Vec::new();
        let mut state = *s;
        for (&r, ck) in rounds.iter().zip(cks) {
            constrs.extend((0..SPONGE_WIDTH).map(|i| state[i] + f(t.rc[r][i]) - ck[i]));
            state = mds_layer(t, &ck.map(sbox7));
        }
        (constrs, state)
    }

    /// `Consts.partialConstrs`.
    fn partial_constrs(t: &Tables, cks: &[GF], s: &St) -> (Vec<GF>, St) {
        let mut constrs = Vec::new();
        let mut state = *s;
        for (r, &ck) in cks.iter().enumerate() {
            constrs.push(state[0] - ck);
            state[0] = sbox7(ck) + f(t.partial_rc[r]);
            state = partial_fast(t, r, &state);
        }
        (constrs, state)
    }

    /// `Consts.gateConstraints`.
    #[allow(clippy::too_many_arguments)]
    fn gate_constraints(
        t: &Tables,
        inp: &St,
        out: &St,
        swap: GF,
        delta: &[GF; 4],
        ck_first: &[St],
        ck_partial: &[GF],
        ck_last: &[St],
    ) -> Vec<GF> {
        let mut all = vec![swap * (swap - GF::ONE)];
        all.extend((0..4).map(|i| swap * (inp[i + 4] - inp[i]) - delta[i]));
        let s0 = full_round(t, 0, &swap_in(delta, inp));
        let (c1, s1) = full_constrs(t, &[1, 2, 3], ck_first, &s0);
        let (c2, s2) = partial_constrs(
            t,
            ck_partial,
            &partial_init(t, &add_rc(&t.partial_first, &s1)),
        );
        let (c3, s3) = full_constrs(t, &LAST_FULL, ck_last, &s2);
        all.extend(c1);
        all.extend(c2);
        all.extend(c3);
        all.extend((0..SPONGE_WIDTH).map(|i| s3[i] - out[i]));
        all
    }

    #[test]
    fn poseidon_hand_perm_matches_native() {
        let t = goldilocks_tables();
        let mut rng = Lcg(0x5eed_0f9015e1d0);
        for _ in 0..64 {
            let inp = rng.state();
            assert_eq!(perm(&t, &inp), <GF as Poseidon>::poseidon(inp));
        }
        let (input, output) = test_vector();
        assert_eq!(perm(&t, &input.map(f)), output.map(f));
    }

    /// The constraint walk of `Consts.gateConstraints`, on the generated tables, against
    /// `PoseidonGate::eval_unfiltered` at random wire assignments (so every constraint is
    /// generically non-zero and compared as a polynomial value, not as `0 = 0`).
    #[test]
    fn poseidon_hand_gate_constraints_match_real_gate() {
        type Ext2 = <GF as Extendable<2>>::Extension;
        let t = goldilocks_tables();
        let gate = PoseidonGate::<GF, 2>::new();
        let mut rng = Lcg(0xface_feed_cafe_babe);
        for _ in 0..16 {
            let wires: Vec<GF> = (0..gate.num_wires()).map(|_| rng.next()).collect();
            let lane = |start: usize| -> St { core::array::from_fn(|i| wires[start + i]) };
            let inp = lane(0);
            let out = lane(12);
            let swap = wires[24];
            let delta: [GF; 4] = core::array::from_fn(|i| wires[25 + i]);
            let ck_first: Vec<St> = (0..3).map(|r| lane(29 + 12 * r)).collect();
            let ck_partial: Vec<GF> = wires[65..87].to_vec();
            let ck_last: Vec<St> = (0..4).map(|r| lane(87 + 12 * r)).collect();
            assert_eq!(wires.len(), 135);

            let hand = gate_constraints(
                &t,
                &inp,
                &out,
                swap,
                &delta,
                &ck_first,
                &ck_partial,
                &ck_last,
            );

            let embedded: Vec<Ext2> = wires.iter().map(|&x| Ext2::from(x)).collect();
            let pih = HashOut::<GF>::ZERO;
            let vars = EvaluationVars {
                local_constants: &[],
                local_wires: &embedded,
                public_inputs_hash: &pih,
            };
            let real: Vec<GF> = <PoseidonGate<GF, 2> as Gate<GF, 2>>::eval_unfiltered(&gate, vars)
                .iter()
                .map(|e| {
                    <Ext2 as plonky2::field::extension::FieldExtension<2>>::to_basefield_array(e)[0]
                })
                .collect();
            assert_eq!(hand, real, "hand gateConstraints transliteration mismatch");
        }
    }
}
