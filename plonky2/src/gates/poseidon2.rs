#![allow(clippy::needless_range_loop)]

extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use core::fmt::Debug;
use core::marker::PhantomData;

use plonky2_field::extension::Extendable;
use qp_poseidon_constants::{
    POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW, POSEIDON2_INTERNAL_CONSTANTS_RAW,
    POSEIDON2_MATRIX_DIAG_12_RAW, POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
};
// Re-export sponge parameters from the canonical source (qp-poseidon-constants)
pub use qp_poseidon_constants::{
    POSEIDON2_EXTERNAL_ROUNDS, POSEIDON2_INTERNAL_ROUNDS, POSEIDON2_OUTPUT, SPONGE_CAPACITY,
    SPONGE_RATE, SPONGE_WIDTH,
};

use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Poseidon2 over Goldilocks with `WIDTH = 12`, `RATE = 8` (capacity 4).
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
    pub ext_init: [[F; SPONGE_WIDTH]; 4],
    /// 4 external rounds (terminal phase), each a WIDTH-sized RC vector.
    pub ext_term: [[F; SPONGE_WIDTH]; 4],
    /// 22 internal round constants added to lane 0.
    pub int_rc: [F; POSEIDON2_INTERNAL_ROUNDS],
    /// Fixed GL diagonal used in the internal mixing.
    pub diag: [F; SPONGE_WIDTH],
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Params<F, D> {
    /// Create params from p3-style raw constants (as u64), exactly like your dump.
    pub fn from_p3_constants_u64(
        initial: [[u64; SPONGE_WIDTH]; 4],
        terminal: [[u64; SPONGE_WIDTH]; 4],
        internal: [u64; POSEIDON2_INTERNAL_ROUNDS],
    ) -> Self {
        // map helpers
        let map_u = |x: u64| F::from_canonical_u64(x);
        let map_rounds = |src: [[u64; SPONGE_WIDTH]; 4]| {
            core::array::from_fn::<[F; SPONGE_WIDTH], 4, _>(|r| {
                core::array::from_fn(|i| map_u(src[r][i]))
            })
        };

        let ext_init = map_rounds(initial);
        let ext_term = map_rounds(terminal);

        let mut int_rc = [F::ZERO; POSEIDON2_INTERNAL_ROUNDS];
        for i in 0..POSEIDON2_INTERNAL_ROUNDS {
            int_rc[i] = map_u(internal[i]);
        }

        // Goldilocks Poseidon2 diagonal from canonical constants
        let diag = core::array::from_fn(|i| F::from_canonical_u64(POSEIDON2_MATRIX_DIAG_12_RAW[i]));

        Self {
            ext_init,
            ext_term,
            int_rc,
            diag,
        }
    }
}

#[inline(always)]
fn sbox7_base<F: Field>(x: F) -> F {
    let x2 = x * x;
    let x4 = x2 * x2;
    (x * x2) * x4
}
#[inline(always)]
fn apply_mat4_base<F: Field>(a: F, x: F, c: F, d: F) -> [F; 4] {
    let two = F::from_canonical_u64(2);
    let three = F::from_canonical_u64(3);
    let y0 = a * two + x * three + c + d;
    let y1 = a + x * two + c * three + d;
    let y2 = a + x + c * two + d * three;
    let y3 = a * three + x + c + d * two;
    [y0, y1, y2, y3]
}

#[inline(always)]
fn mds_light_base<F: Field>(s: &mut [F; SPONGE_WIDTH]) {
    // 1) 4×4 per block
    for k in (0..SPONGE_WIDTH).step_by(4) {
        let [y0, y1, y2, y3] = apply_mat4_base(s[k], s[k + 1], s[k + 2], s[k + 3]);
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
    for i in 0..SPONGE_WIDTH {
        s[i] += sums[i % 4];
    }
}

fn mds_light_any<Fx: Field>(s: &mut [Fx; SPONGE_WIDTH]) {
    // identical to mds_light_base, just generic
    mds_light_base::<Fx>(s);
}

#[inline(always)]
fn apply_mat4_ext<F: RichField + Extendable<D>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    a: ExtensionTarget<D>,
    x: ExtensionTarget<D>,
    c: ExtensionTarget<D>,
    d: ExtensionTarget<D>,
) -> [ExtensionTarget<D>; 4] {
    // y0 = 2a + 3x + 1c + 1d
    let two_a = b.add_extension(a, a);
    let two_x = b.add_extension(x, x);
    let three_x = b.add_extension(two_x, x);
    let t = b.add_extension(two_a, three_x);
    let t = b.add_extension(t, c);
    let y0 = b.add_extension(t, d);

    // y1 = 1a + 2x + 3c + 1d
    let two_c = b.add_extension(c, c);
    let three_c = b.add_extension(two_c, c);
    let two_x = b.add_extension(x, x);
    let two_x_plus_three_c = b.add_extension(two_x, three_c);
    let t = b.add_extension(a, two_x_plus_three_c);
    let y1 = b.add_extension(t, d);

    // y2 = 1a + 1x + 2c + 3d
    let two_d = b.add_extension(d, d);
    let three_d = b.add_extension(two_d, d);
    let two_c = b.add_extension(c, c);
    let two_c_plus_one_x = b.add_extension(two_c, x);
    let t = b.add_extension(a, two_c_plus_one_x);
    let y2 = b.add_extension(t, three_d);

    // y3 = 3a + 1x + 1c + 2d
    let three_a = b.add_extension(two_a, a);
    let one_x_plus_one_c = b.add_extension(x, c);
    let t = b.add_extension(three_a, one_x_plus_one_c);
    let two_d = b.add_extension(d, d);
    let y3 = b.add_extension(t, two_d);

    [y0, y1, y2, y3]
}

#[inline(always)]
fn mds_light_ext<F: RichField + Extendable<D>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    s: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
) {
    // 1) 4×4 per block with MDSMat4
    for k in (0..SPONGE_WIDTH).step_by(4) {
        let [y0, y1, y2, y3] = apply_mat4_ext::<F, D>(b, s[k], s[k + 1], s[k + 2], s[k + 3]);
        s[k] = y0;
        s[k + 1] = y1;
        s[k + 2] = y2;
        s[k + 3] = y3;
    }
    // 2) sums per residue class mod 4
    let mut sums = [b.zero_extension(); 4];
    for k in 0..4 {
        let t = b.add_extension(s[k], s[4 + k]);
        sums[k] = b.add_extension(t, s[8 + k]);
    }
    // 3) add sums[i%4]
    for i in 0..SPONGE_WIDTH {
        s[i] = b.add_extension(s[i], sums[i % 4]);
    }
}

#[inline(always)]
fn internal_mix_base<F: Field>(
    x: &[F; SPONGE_WIDTH],
    diag: &[F; SPONGE_WIDTH],
) -> [F; SPONGE_WIDTH] {
    let mut sum = x[0];
    for i in 1..SPONGE_WIDTH {
        sum += x[i];
    }
    let mut y = [F::ZERO; SPONGE_WIDTH];
    for i in 0..SPONGE_WIDTH {
        y[i] = diag[i] * x[i] + sum;
    }
    y
}

#[inline(always)]
fn ext_c<Fx: RichField + Extendable<D>, const D: usize>(x: Fx) -> <Fx as Extendable<D>>::Extension {
    <<Fx as Extendable<D>>::Extension as Field>::from_canonical_u64(x.to_canonical_u64())
}

#[inline(always)]
fn sbox7_ext<Fx: RichField + Extendable<D>, const D: usize>(
    x: <Fx as Extendable<D>>::Extension,
) -> <Fx as Extendable<D>>::Extension {
    let x2 = x * x;
    let x4 = x2 * x2;
    (x * x2) * x4
}
fn sbox7_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    b: &mut CircuitBuilder<F, D>,
    x: ExtensionTarget<D>,
) -> ExtensionTarget<D> {
    let x2 = b.mul_extension(x, x);
    let x4 = b.mul_extension(x2, x2);
    let x3 = b.mul_extension(x, x2);
    b.mul_extension(x3, x4) // x^7
}

fn mds_light_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
) where
    F: RichField + Extendable<D>,
{
    // Note: Always use inline computation here instead of adding a gate.
    // Adding a Poseidon2MdsGate during eval_unfiltered_circuit causes issues:
    // the gate's generator conflicts with other generators when used in
    // recursive verification circuits.
    mds_light_ext::<F, D>(builder, state);
}

fn internal_mix_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &[ExtensionTarget<D>; SPONGE_WIDTH],
    diag: &[F; SPONGE_WIDTH],
) -> [ExtensionTarget<D>; SPONGE_WIDTH] {
    // Note: Always use inline computation here instead of adding a gate.
    // Adding a Poseidon2IntMixGate during eval_unfiltered_circuit causes issues:
    // the gate's generator conflicts with other generators when used in
    // recursive verification circuits.
    let mut s = *state;

    // diag as extension constants
    let diag_ext: [ExtensionTarget<D>; SPONGE_WIDTH] = core::array::from_fn(|i| {
        let val = ext_c::<F, D>(diag[i]);
        builder.constant_extension(val)
    });

    // sum = sum_j s[j]
    let mut sum = s[0];
    for i in 1..SPONGE_WIDTH {
        sum = builder.add_extension(sum, s[i]);
    }

    // y[i] = diag[i] * s[i] + sum
    for i in 0..SPONGE_WIDTH {
        s[i] = builder.mul_add_extension(diag_ext[i], s[i], sum);
    }

    s
}

#[derive(Clone, Debug)]
pub struct Poseidon2Gate<F: RichField + Extendable<D>, const D: usize> {
    params: Poseidon2Params<F, D>,
    _pd: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Gate<F, D> {
    pub const W_IN: usize = 0;
    pub const W_OUT: usize = SPONGE_WIDTH;

    // S-box input wires for external (full) rounds
    pub const W_EXT_SBOX: usize = 2 * SPONGE_WIDTH;

    // S-box input wires for internal rounds (lane 0 only)
    pub const W_INT_SBOX: usize = Self::W_EXT_SBOX + POSEIDON2_EXTERNAL_ROUNDS * SPONGE_WIDTH;

    pub const fn wire_input(i: usize) -> usize {
        Self::W_IN + i
    }
    pub const fn wire_output(i: usize) -> usize {
        Self::W_OUT + i
    }

    #[inline]
    pub const fn wire_ext_sbox(round: usize, lane: usize) -> usize {
        debug_assert!(round < POSEIDON2_EXTERNAL_ROUNDS);
        debug_assert!(lane < SPONGE_WIDTH);
        Self::W_EXT_SBOX + round * SPONGE_WIDTH + lane
    }

    #[inline]
    pub const fn wire_int_sbox(round: usize) -> usize {
        debug_assert!(round < POSEIDON2_INTERNAL_ROUNDS);
        Self::W_INT_SBOX + round
    }

    pub const fn end() -> usize {
        // 12 in + 12 out + 8*12 external s-box inputs + 22 internal s-box inputs = 142
        Self::W_INT_SBOX + POSEIDON2_INTERNAL_ROUNDS
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
        format!("Poseidon2Gate<WIDTH={SPONGE_WIDTH}>")
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
        POSEIDON2_EXTERNAL_ROUNDS * SPONGE_WIDTH + POSEIDON2_INTERNAL_ROUNDS + SPONGE_WIDTH
        // = 8*12 + 22 + 12 = 130
    }

    fn degree(&self) -> usize {
        7
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let lw = vars.local_wires;
        let mut constr = Vec::with_capacity(self.num_constraints());

        // 0) load inputs into extension state
        let mut state: [F::Extension; SPONGE_WIDTH] =
            core::array::from_fn(|i| lw[Self::wire_input(i)]);

        // 1) initial preamble (light MDS)
        mds_light_any::<F::Extension>(&mut state);

        let ext_init = &self.params.ext_init;
        let ext_term = &self.params.ext_term;
        let int_rc = &self.params.int_rc;
        let diag = &self.params.diag;

        let mut ext_round_idx = 0usize;

        // 2) 4 initial external rounds
        for r in 0..4 {
            // add RCs
            for i in 0..SPONGE_WIDTH {
                state[i] += ext_c::<F, D>(ext_init[r][i]);
            }
            // constrain S-box inputs and update state = sbox_in
            for i in 0..SPONGE_WIDTH {
                let sbox_in = lw[Self::wire_ext_sbox(ext_round_idx, i)];
                constr.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            // apply S-box x^7 on all lanes
            for i in 0..SPONGE_WIDTH {
                state[i] = sbox7_ext::<F, D>(state[i]);
            }
            // light MDS
            mds_light_any::<F::Extension>(&mut state);
            ext_round_idx += 1;
        }

        // 3) 22 internal rounds (lane 0 sbox + internal mix)
        for r in 0..POSEIDON2_INTERNAL_ROUNDS {
            // lane 0: add RC
            state[0] += ext_c::<F, D>(int_rc[r]);

            // constrain S-box input for lane 0 and update
            let sbox_in = lw[Self::wire_int_sbox(r)];
            constr.push(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = sbox7_ext::<F, D>(state[0]);

            // internal mixing: y[i] = diag[i]*x[i] + sum(x)
            let mut sum = state[0];
            for i in 1..SPONGE_WIDTH {
                sum += state[i];
            }
            for i in 0..SPONGE_WIDTH {
                let d_i = ext_c::<F, D>(diag[i]);
                state[i] = d_i * state[i] + sum;
            }
        }

        // 4) 4 terminal external rounds
        for r in 0..4 {
            // add RCs
            for i in 0..SPONGE_WIDTH {
                state[i] += ext_c::<F, D>(ext_term[r][i]);
            }
            // constrain S-box inputs and update state = sbox_in
            for i in 0..SPONGE_WIDTH {
                let sbox_in = lw[Self::wire_ext_sbox(ext_round_idx, i)];
                constr.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            // apply S-box x^7 on all lanes
            for i in 0..SPONGE_WIDTH {
                state[i] = sbox7_ext::<F, D>(state[i]);
            }
            mds_light_any::<F::Extension>(&mut state);
            ext_round_idx += 1;
        }

        // 5) outputs equal final state
        for i in 0..SPONGE_WIDTH {
            let out = lw[Self::wire_output(i)];
            constr.push(out - state[i]);
        }

        constr
    }

    fn eval_unfiltered_circuit(
        &self,
        b: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let lw = &vars.local_wires;
        let mut constr = Vec::with_capacity(self.num_constraints());

        // 0) load inputs
        let mut state: [ExtensionTarget<D>; SPONGE_WIDTH] =
            core::array::from_fn(|i| lw[Self::wire_input(i)]);

        // 1) preamble light MDS
        mds_light_circuit::<F, D>(b, &mut state);

        let ext_init = &self.params.ext_init;
        let ext_term = &self.params.ext_term;
        let int_rc = &self.params.int_rc;
        let diag = &self.params.diag;

        let mut ext_round_idx = 0usize;

        // 2) initial external rounds
        for r in 0..4 {
            // Hoist this round's RCs: one constant per lane
            let rc_row: [ExtensionTarget<D>; SPONGE_WIDTH] = core::array::from_fn(|i| {
                let val = ext_c::<F, D>(ext_init[r][i]);
                b.constant_extension(val)
            });

            // add RCs
            for i in 0..SPONGE_WIDTH {
                state[i] = b.add_extension(state[i], rc_row[i]);
            }

            // constrain S-box inputs and update state = sbox_in
            for i in 0..SPONGE_WIDTH {
                let sbox_in = lw[Self::wire_ext_sbox(ext_round_idx, i)];
                constr.push(b.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }

            // S-box x^7
            for i in 0..SPONGE_WIDTH {
                state[i] = sbox7_ext_circuit::<F, D>(b, state[i]);
            }

            mds_light_circuit::<F, D>(b, &mut state);
            ext_round_idx += 1;
        }

        // 3) internal rounds
        for r in 0..POSEIDON2_INTERNAL_ROUNDS {
            // lane 0 RC, hoisted once per round
            let rc0 = {
                let val = ext_c::<F, D>(int_rc[r]);
                b.constant_extension(val)
            };
            state[0] = b.add_extension(state[0], rc0);

            let sbox_in = lw[Self::wire_int_sbox(r)];
            constr.push(b.sub_extension(state[0], sbox_in));
            state[0] = sbox_in;
            state[0] = sbox7_ext_circuit::<F, D>(b, state[0]);

            // internal mixing via dedicated gate (or inline fallback)
            state = internal_mix_circuit::<F, D>(b, &state, diag);
        }

        // 4) terminal external rounds
        for r in 0..4 {
            // Hoist terminal RCs for this round
            let rc_row: [ExtensionTarget<D>; SPONGE_WIDTH] = core::array::from_fn(|i| {
                let val = ext_c::<F, D>(ext_term[r][i]);
                b.constant_extension(val)
            });

            // add RCs
            for i in 0..SPONGE_WIDTH {
                state[i] = b.add_extension(state[i], rc_row[i]);
            }

            // constrain S-box inputs
            for i in 0..SPONGE_WIDTH {
                let sbox_in = lw[Self::wire_ext_sbox(ext_round_idx, i)];
                constr.push(b.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }

            // S-box x^7
            for i in 0..SPONGE_WIDTH {
                state[i] = sbox7_ext_circuit::<F, D>(b, state[i]);
            }

            mds_light_circuit::<F, D>(b, &mut state);
            ext_round_idx += 1;
        }

        // 5) outputs == state
        for i in 0..SPONGE_WIDTH {
            let out = lw[Self::wire_output(i)];
            constr.push(b.sub_extension(out, state[i]));
        }

        constr
    }

    fn generators(&self, row: usize, _lc: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            Poseidon2FullGen::<F, D> {
                row,
                params: self.params.clone(),
                _pd: PhantomData,
            }
            .adapter(),
        )]
    }
}

#[derive(Clone, Debug, Default)]
pub struct Poseidon2FullGen<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    params: Poseidon2Params<F, D>,
    _pd: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for Poseidon2FullGen<F, D>
{
    fn id(&self) -> String {
        "Poseidon2FullGen".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        // Only depends on the 12 inputs.
        (0..SPONGE_WIDTH)
            .map(|i| Target::wire(self.row, Poseidon2Gate::<F, D>::wire_input(i)))
            .collect()
    }

    fn run_once(
        &self,
        pw: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> anyhow::Result<()> {
        let mut state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = pw.get_wire(Wire {
                row: self.row,
                column: Poseidon2Gate::<F, D>::wire_input(i),
            });
        }

        // 0) preamble MDS
        mds_light_base(&mut state);

        let ext_init = &self.params.ext_init;
        let ext_term = &self.params.ext_term;
        let int_rc = &self.params.int_rc;
        let diag = &self.params.diag;

        let mut ext_round_idx = 0usize;

        // 1) initial external rounds
        for r in 0..4 {
            for i in 0..SPONGE_WIDTH {
                state[i] = state[i] + ext_init[r][i];
                let s_in = state[i];
                let idx = Poseidon2Gate::<F, D>::wire_ext_sbox(ext_round_idx, i);
                out.set_wire(
                    Wire {
                        row: self.row,
                        column: idx,
                    },
                    s_in,
                )?;
                state[i] = sbox7_base(s_in);
            }
            mds_light_base(&mut state);
            ext_round_idx += 1;
        }

        // 2) internal rounds
        for r in 0..POSEIDON2_INTERNAL_ROUNDS {
            state[0] = state[0] + int_rc[r];
            let s_in = state[0];
            let idx = Poseidon2Gate::<F, D>::wire_int_sbox(r);
            out.set_wire(
                Wire {
                    row: self.row,
                    column: idx,
                },
                s_in,
            )?;
            state[0] = sbox7_base(s_in);
            state = internal_mix_base(&state, diag);
        }

        // 3) terminal external rounds
        for r in 0..4 {
            for i in 0..SPONGE_WIDTH {
                state[i] = state[i] + ext_term[r][i];
                let s_in = state[i];
                let idx = Poseidon2Gate::<F, D>::wire_ext_sbox(ext_round_idx, i);
                out.set_wire(
                    Wire {
                        row: self.row,
                        column: idx,
                    },
                    s_in,
                )?;
                state[i] = sbox7_base(s_in);
            }
            mds_light_base(&mut state);
            ext_round_idx += 1;
        }

        // 4) outputs
        for i in 0..SPONGE_WIDTH {
            out.set_wire(
                Wire {
                    row: self.row,
                    column: Poseidon2Gate::<F, D>::wire_output(i),
                },
                state[i],
            )?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            row: src.read_usize()?,
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            _pd: PhantomData,
        })
    }
}
