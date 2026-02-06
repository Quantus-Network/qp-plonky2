#![allow(clippy::needless_range_loop)]

extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use core::fmt::Debug;
use core::marker::PhantomData;

use plonky2_field::extension::Extendable;
use qp_poseidon_constants::{
    POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW, POSEIDON2_INTERNAL_CONSTANTS_RAW,
    POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
};

use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::EvaluationVars;
use crate::util::serialization::{Buffer, IoResult};

/// WIDTH=12, RATE=4 (capacity 8).
pub const P2_WIDTH: usize = 12;
pub const P2_RATE: usize = 4;
pub const P2_INTERNAL_ROUNDS: usize = 22;
pub const P2_EXT_ROUNDS: usize = 8; // 4 init + 4 terminal

/// Poseidon2 over Goldilocks with `WIDTH = 12`, `RATE = 4` (capacity 8).
///
/// ## Structure (matches p3-goldilocks)
///
/// - **Initial preamble (once):**
///   - Apply the external light MDS (`mds_light_permutation`) to the 12-lane state.
///     This consists of applying the 4×4 matrix
///     ```text
///     [2 3 1 1]
///     [1 2 3 1]
///     [1 1 2 3]
///     [3 1 1 2]
///     ```
///     blockwise to lanes `(0..3), (4..7), (8..11)`, then adding the per-residue-class sums:
///     for each lane `i`, add `sum(state[j])` over all `j` with `j ≡ i (mod 4)`.
///
/// - **External rounds (initial and terminal, 4 each):**
///   For each round `r` and lane `i`:
///   1. Add round constant: ``state[i] = state[i] + rc_ext[phase][r][i]``.
///   2. Apply the S-box on **all lanes**: ``state[i] = state[i]^7``.
///   3. Apply the **light MDS** as in the preamble (same matrix + outer sums).
///
/// - **Internal rounds (22 total):**
///   For each round `r`:
///   1. Add round constant to **lane 0 only**, then apply the S-box on **lane 0 only**:
///      ```text
///      state[0] = (state[0] + rc_internal[r])^7
///      ```
///      Other lanes **do not** go through the S-box in internal rounds.
///   2. Apply the internal diffusion:
///      for each lane `i`,
///      ```text
///      state[i] = diag[i] * state[i] + sum(state)
///      ```
///      where `diag` is the fixed Goldilocks diagonal for `WIDTH=12`, and `sum(state)`
///      is the sum over all 12 lanes in the current state.
///
/// ## Notes
/// - All round constants are derived from the fixed seed used in p3-goldilocks, and the
///   diagonal matches `MATRIX_DIAG_12_GOLDILOCKS`.
/// - The only nonlinearity per internal round is the single `x^7` on lane 0, so each
///   internal-round gate’s constraint degree is 7.
/// - External rounds are also degree 7 (S-box on all lanes).

/// Constants (shared with CPU hashing).
#[derive(Clone, Debug, Default)]
pub struct Poseidon2Params<F: RichField + Extendable<D>, const D: usize> {
    /// 4 external rounds (initial phase), each a WIDTH-sized RC vector.
    pub ext_init: [[F; P2_WIDTH]; 4],
    /// 4 external rounds (terminal phase), each a WIDTH-sized RC vector.
    pub ext_term: [[F; P2_WIDTH]; 4],
    /// 22 internal round constants added to lane 0.
    pub int_rc: [F; P2_INTERNAL_ROUNDS],
    /// Fixed GL diagonal used in the internal mixing.
    pub diag: [F; P2_WIDTH],
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Params<F, D> {
    /// Create params from p3-style raw constants (as u64), exactly like your dump.
    pub fn from_p3_constants_u64(
        initial: [[u64; P2_WIDTH]; 4],
        terminal: [[u64; P2_WIDTH]; 4],
        internal: [u64; P2_INTERNAL_ROUNDS],
    ) -> Self {
        // map helpers
        let map_u = |x: u64| F::from_canonical_u64(x);
        let map_rounds = |src: [[u64; P2_WIDTH]; 4]| {
            core::array::from_fn::<[F; P2_WIDTH], 4, _>(|r| {
                core::array::from_fn(|i| map_u(src[r][i]))
            })
        };

        let ext_init = map_rounds(initial);
        let ext_term = map_rounds(terminal);

        let mut int_rc = [F::ZERO; P2_INTERNAL_ROUNDS];
        for i in 0..P2_INTERNAL_ROUNDS {
            int_rc[i] = map_u(internal[i]);
        }

        // Goldilocks Poseidon2 diag for WIDTH=12 (matches p3-goldilocks MATRIX_DIAG_12_GOLDILOCKS)
        let diag = [
            F::from_canonical_u64(0xc3b6c08e23ba9300),
            F::from_canonical_u64(0xd84b5de94a324fb6),
            F::from_canonical_u64(0x0d0c371c5b35b84f),
            F::from_canonical_u64(0x7964f570e7188037),
            F::from_canonical_u64(0x5daf18bbd996604b),
            F::from_canonical_u64(0x6743bc47b9595257),
            F::from_canonical_u64(0x5528b9362c59bb70),
            F::from_canonical_u64(0xac45e25b7127b68b),
            F::from_canonical_u64(0xa2077d7dfbb606b5),
            F::from_canonical_u64(0xf3faac6faee378ae),
            F::from_canonical_u64(0x0c6388b51545e883),
            F::from_canonical_u64(0xd27dbb6944917b60),
        ];

        Self {
            ext_init,
            ext_term,
            int_rc,
            diag,
        }
    }
}

#[inline(always)]
fn apply_mat4<F: Field>(a: F, x: F, c: F, d: F) -> [F; 4] {
    let two = F::from_canonical_u64(2);
    let three = F::from_canonical_u64(3);
    let y0 = a * two + x * three + c + d;
    let y1 = a + x * two + c * three + d;
    let y2 = a + x + c * two + d * three;
    let y3 = a * three + x + c + d * two;
    [y0, y1, y2, y3]
}

#[inline(always)]
fn mds_light<F: Field>(s: &mut [F; P2_WIDTH]) {
    // 1) 4×4 per block
    for k in (0..P2_WIDTH).step_by(4) {
        let [y0, y1, y2, y3] = apply_mat4(s[k], s[k + 1], s[k + 2], s[k + 3]);
        s[k] = y0;
        s[k + 1] = y1;
        s[k + 2] = y2;
        s[k + 3] = y3;
    }
    // 2) sums per residue class
    let mut sums = [F::ZERO; 4];
    for k in 0..4 {
        sums[k] = s[k] + s[4 + k] + s[8 + k];
    }
    // 3) add sums[i%4]
    for i in 0..P2_WIDTH {
        s[i] += sums[i % 4];
    }
}

#[inline(always)]
fn ext_c<Fx: RichField + Extendable<D>, const D: usize>(x: Fx) -> <Fx as Extendable<D>>::Extension {
    <<Fx as Extendable<D>>::Extension as Field>::from_canonical_u64(x.to_canonical_u64())
}

#[inline(always)]
fn sbox7<F: Field>(x: F) -> F {
    let x2 = x * x;
    let x4 = x2 * x2;
    (x * x2) * x4
}

#[derive(Clone, Debug)]
pub struct Poseidon2Gate<F: RichField + Extendable<D>, const D: usize> {
    params: Poseidon2Params<F, D>,
    _pd: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Gate<F, D> {
    pub const W_IN: usize = 0;
    pub const W_OUT: usize = P2_WIDTH;

    // S-box input wires for external (full) rounds
    pub const W_EXT_SBOX: usize = 2 * P2_WIDTH;

    // S-box input wires for internal rounds (lane 0 only)
    pub const W_INT_SBOX: usize = Self::W_EXT_SBOX + P2_EXT_ROUNDS * P2_WIDTH;

    pub const fn wire_input(i: usize) -> usize {
        Self::W_IN + i
    }
    pub const fn wire_output(i: usize) -> usize {
        Self::W_OUT + i
    }

    #[inline]
    pub const fn wire_ext_sbox(round: usize, lane: usize) -> usize {
        debug_assert!(round < P2_EXT_ROUNDS);
        debug_assert!(lane < P2_WIDTH);
        Self::W_EXT_SBOX + round * P2_WIDTH + lane
    }

    #[inline]
    pub const fn wire_int_sbox(round: usize) -> usize {
        debug_assert!(round < P2_INTERNAL_ROUNDS);
        Self::W_INT_SBOX + round
    }

    pub const fn end() -> usize {
        // 12 in + 12 out + 8*12 external s-box inputs + 22 internal s-box inputs = 142
        Self::W_INT_SBOX + P2_INTERNAL_ROUNDS
    }
    pub fn new() -> Self {
        Self {
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            _pd: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for Poseidon2Gate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={P2_WIDTH}>")
    }

    fn serialize(&self, _dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::new())
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn num_constraints(&self) -> usize {
        // One constraint per S-box input (external + internal), plus output equality.
        P2_EXT_ROUNDS * P2_WIDTH + P2_INTERNAL_ROUNDS + P2_WIDTH
        // = 8*12 + 22 + 12 = 130
    }

    fn degree(&self) -> usize {
        7
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let lw = vars.local_wires;
        let mut constr = Vec::with_capacity(self.num_constraints());

        // 0) load inputs into extension state
        let mut state: [F::Extension; P2_WIDTH] = core::array::from_fn(|i| lw[Self::wire_input(i)]);

        // 1) initial preamble (light MDS)
        mds_light::<F::Extension>(&mut state);

        let ext_init = &self.params.ext_init;
        let ext_term = &self.params.ext_term;
        let int_rc = &self.params.int_rc;
        let diag = &self.params.diag;

        let mut ext_round_idx = 0usize;

        // 2) 4 initial external rounds
        for r in 0..4 {
            // add RCs
            for i in 0..P2_WIDTH {
                state[i] += ext_c::<F, D>(ext_init[r][i]);
            }
            // constrain S-box inputs and update state = sbox_in
            for i in 0..P2_WIDTH {
                let sbox_in = lw[Self::wire_ext_sbox(ext_round_idx, i)];
                constr.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            // apply S-box x^7 on all lanes
            for i in 0..P2_WIDTH {
                state[i] = sbox7(state[i]);
            }
            // light MDS
            mds_light::<F::Extension>(&mut state);
            ext_round_idx += 1;
        }

        // 3) 22 internal rounds (lane 0 sbox + internal mix)
        for r in 0..P2_INTERNAL_ROUNDS {
            // lane 0: add RC
            state[0] += ext_c::<F, D>(int_rc[r]);

            // constrain S-box input for lane 0 and update
            let sbox_in = lw[Self::wire_int_sbox(r)];
            constr.push(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = sbox7(state[0]);

            // internal mixing: y[i] = diag[i]*x[i] + sum(x)
            let mut sum = state[0];
            for i in 1..P2_WIDTH {
                sum += state[i];
            }
            for i in 0..P2_WIDTH {
                let d_i = ext_c::<F, D>(diag[i]);
                state[i] = d_i * state[i] + sum;
            }
        }

        // 4) 4 terminal external rounds
        for r in 0..4 {
            // add RCs
            for i in 0..P2_WIDTH {
                state[i] += ext_c::<F, D>(ext_term[r][i]);
            }
            // constrain S-box inputs and update state = sbox_in
            for i in 0..P2_WIDTH {
                let sbox_in = lw[Self::wire_ext_sbox(ext_round_idx, i)];
                constr.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            // apply S-box x^7 on all lanes
            for i in 0..P2_WIDTH {
                state[i] = sbox7(state[i]);
            }
            mds_light::<F::Extension>(&mut state);
            ext_round_idx += 1;
        }

        // 5) outputs equal final state
        for i in 0..P2_WIDTH {
            let out = lw[Self::wire_output(i)];
            constr.push(out - state[i]);
        }

        constr
    }
}
