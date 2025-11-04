#![allow(clippy::needless_range_loop)]

extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use core::fmt::Debug;

use core::marker::PhantomData;
use plonky2_field::extension::Extendable;

use crate::iop::wire::Wire;
use crate::iop::witness::Witness;

use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::PartitionWitness;
use crate::iop::witness::WitnessWrite;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars};
use crate::util::serialization::{Buffer, IoResult, Read, Write};
use qp_poseidon_constants::{
    POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW, POSEIDON2_INTERNAL_CONSTANTS_RAW,
    POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
};

/// WIDTH=12, RATE=4 (capacity 8).
pub const P2_WIDTH: usize = 12;
pub const P2_RATE: usize = 4;

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

pub const P2_INTERNAL_ROUNDS: usize = 22;

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
fn mds_light_base<F: Field>(s: &mut [F; P2_WIDTH]) {
    // 1) 4×4 per block
    for k in (0..P2_WIDTH).step_by(4) {
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
    for i in 0..P2_WIDTH {
        s[i] += sums[i % 4];
    }
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
    s: &mut [ExtensionTarget<D>; P2_WIDTH],
) {
    // 1) 4×4 per block with MDSMat4
    for k in (0..P2_WIDTH).step_by(4) {
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
    for i in 0..P2_WIDTH {
        s[i] = b.add_extension(s[i], sums[i % 4]);
    }
}

#[inline(always)]
fn internal_mix_base<F: Field>(x: &[F; P2_WIDTH], diag: &[F; P2_WIDTH]) -> [F; P2_WIDTH] {
    let mut sum = x[0];
    for i in 1..P2_WIDTH {
        sum += x[i];
    }
    let mut y = [F::ZERO; P2_WIDTH];
    for i in 0..P2_WIDTH {
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

#[derive(Clone, Debug)]
pub struct Poseidon2ExtInitPreambleGate<F: RichField + Extendable<D>, const D: usize>(
    PhantomData<F>,
);

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2ExtInitPreambleGate<F, D> {
    pub const W_IN: usize = 0;
    pub const W_OUT: usize = P2_WIDTH;
    #[inline]
    pub fn wire_input(i: usize) -> usize {
        Self::W_IN + i
    }
    #[inline]
    pub fn wire_output(i: usize) -> usize {
        Self::W_OUT + i
    }
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D>
    for Poseidon2ExtInitPreambleGate<F, D>
{
    fn id(&self) -> String {
        "Poseidon2ExtInitPreamble".into()
    }
    fn serialize(&self, _dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        Ok(())
    }
    fn deserialize(_src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::new())
    }

    fn num_wires(&self) -> usize {
        2 * P2_WIDTH
    }
    fn num_constants(&self) -> usize {
        0
    }
    fn num_constraints(&self) -> usize {
        P2_WIDTH
    }
    fn degree(&self) -> usize {
        1
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let lw = vars.local_wires;
        let mut s: [F::Extension; P2_WIDTH] = core::array::from_fn(|i| lw[Self::wire_input(i)]);
        // light MDS only
        // 4×4 per block
        for k in (0..P2_WIDTH).step_by(4) {
            let a = s[k];
            let x = s[k + 1];
            let c = s[k + 2];
            let d = s[k + 3];
            let two = F::Extension::from_canonical_u64(2);
            let three = F::Extension::from_canonical_u64(3);
            let y0 = a * two + x * three + c + d;
            let y1 = a + x * two + c * three + d;
            let y2 = a + x + c * two + d * three;
            let y3 = a * three + x + c + d * two;
            s[k] = y0;
            s[k + 1] = y1;
            s[k + 2] = y2;
            s[k + 3] = y3;
        }
        // sums per residue
        let mut sums = [F::Extension::ZERO; 4];
        for k in 0..4 {
            sums[k] = s[k] + s[4 + k] + s[8 + k];
        }
        for i in 0..P2_WIDTH {
            s[i] = s[i] + sums[i % 4];
        }

        let mut v = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            v.push(lw[Self::wire_output(i)] - s[i]);
        }
        v
    }

    fn eval_unfiltered_circuit(
        &self,
        b: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let lw = &vars.local_wires;
        let mut s: [ExtensionTarget<D>; P2_WIDTH] =
            core::array::from_fn(|i| lw[Self::wire_input(i)]);
        mds_light_ext::<F, D>(b, &mut s);
        let mut v = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            v.push(b.sub_extension(lw[Self::wire_output(i)], s[i]));
        }
        v
    }

    fn generators(&self, row: usize, _lc: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            Poseidon2ExtInitPreambleGen::<F, D> {
                row,
                _pd: PhantomData,
            }
            .adapter(),
        )]
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2ExtInitPreambleGen<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    _pd: PhantomData<F>,
}
impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for Poseidon2ExtInitPreambleGen<F, D>
{
    fn id(&self) -> String {
        "Poseidon2ExtInitPreambleGen".into()
    }
    fn dependencies(&self) -> Vec<Target> {
        (0..P2_WIDTH)
            .map(|i| {
                Target::wire(
                    self.row,
                    Poseidon2ExtInitPreambleGate::<F, D>::wire_input(i),
                )
            })
            .collect()
    }
    fn run_once(
        &self,
        pw: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> anyhow::Result<()> {
        let mut s = [F::ZERO; P2_WIDTH];
        for i in 0..P2_WIDTH {
            s[i] = pw.get_wire(Wire {
                row: self.row,
                column: Poseidon2ExtInitPreambleGate::<F, D>::wire_input(i),
            });
        }
        mds_light_base(&mut s);
        for i in 0..P2_WIDTH {
            out.set_wire(
                Wire {
                    row: self.row,
                    column: Poseidon2ExtInitPreambleGate::<F, D>::wire_output(i),
                },
                s[i],
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
            _pd: PhantomData,
        })
    }
}

// ---------- External (full) round gate: add RCs, x^7 on all lanes, block MDS ----------
#[derive(Clone, Debug)]
pub struct Poseidon2ExtRoundGate<F: RichField + Extendable<D>, const D: usize> {
    params: Poseidon2Params<F, D>,
    round_idx: usize, // 0..4
    is_initial: bool, // true => use ext_init; false => ext_term
    _pd: PhantomData<F>,
}
impl<F: RichField + Extendable<D>, const D: usize> Poseidon2ExtRoundGate<F, D> {
    pub const W_IN: usize = 0;
    pub const W_OUT: usize = P2_WIDTH;
    #[inline]
    pub fn wire_input(i: usize) -> usize {
        Self::W_IN + i
    }
    #[inline]
    pub fn wire_output(i: usize) -> usize {
        Self::W_OUT + i
    }

    pub fn new_initial(round_idx: usize) -> Self {
        Self {
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            round_idx,
            is_initial: true,
            _pd: PhantomData,
        }
    }
    pub fn new_terminal(round_idx: usize) -> Self {
        Self {
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            round_idx,
            is_initial: false,
            _pd: PhantomData,
        }
    }
}
impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for Poseidon2ExtRoundGate<F, D> {
    fn id(&self) -> String {
        format!(
            "Poseidon2ExtRound<r={},init={}>",
            self.round_idx, self.is_initial
        )
    }
    fn serialize(&self, dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.round_idx)?;
        dst.write_bool(self.is_initial)
    }
    fn deserialize(src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let r = src.read_usize()?;
        let is_init = src.read_bool()?;
        Ok(if is_init {
            Self::new_initial(r)
        } else {
            Self::new_terminal(r)
        })
    }

    fn num_wires(&self) -> usize {
        2 * P2_WIDTH
    } // 12 in + 12 out
    fn num_constants(&self) -> usize {
        0
    }
    fn num_constraints(&self) -> usize {
        P2_WIDTH
    } // output equality only
    fn degree(&self) -> usize {
        7
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let lw = vars.local_wires;
        let mut s: [F::Extension; P2_WIDTH] = core::array::from_fn(|i| lw[Self::wire_input(i)]);
        // add RCs + sbox
        let rcs = if self.is_initial {
            &self.params.ext_init[self.round_idx]
        } else {
            &self.params.ext_term[self.round_idx]
        };
        for i in 0..P2_WIDTH {
            s[i] = s[i] + ext_c::<F, D>(rcs[i]);
            s[i] = sbox7_ext::<F, D>(s[i]);
        }
        // apply light MDS
        for k in (0..P2_WIDTH).step_by(4) {
            let a = s[k];
            let x = s[k + 1];
            let c = s[k + 2];
            let d = s[k + 3];
            let two = F::Extension::from_canonical_u64(2);
            let three = F::Extension::from_canonical_u64(3);
            let y0 = a * two + x * three + c + d;
            let y1 = a + x * two + c * three + d;
            let y2 = a + x + c * two + d * three;
            let y3 = a * three + x + c + d * two;
            s[k] = y0;
            s[k + 1] = y1;
            s[k + 2] = y2;
            s[k + 3] = y3;
        }
        let mut sums = [F::Extension::ZERO; 4];
        for k in 0..4 {
            sums[k] = s[k] + s[4 + k] + s[8 + k];
        }
        for i in 0..P2_WIDTH {
            s[i] = s[i] + sums[i % 4];
        }

        // outputs equal
        let mut v = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            v.push(lw[Self::wire_output(i)] - s[i]);
        }
        v
    }

    fn eval_unfiltered_circuit(
        &self,
        b: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let lw = &vars.local_wires;
        // load inputs
        let mut s: [ExtensionTarget<D>; P2_WIDTH] =
            core::array::from_fn(|i| lw[Self::wire_input(i)]);
        // add RCs + sbox
        for i in 0..P2_WIDTH {
            let rc = b.constant_extension(ext_c::<F, D>(if self.is_initial {
                self.params.ext_init[self.round_idx][i]
            } else {
                self.params.ext_term[self.round_idx][i]
            }));
            s[i] = b.add_extension(s[i], rc);
            s[i] = b.exp_u64_extension(s[i], 7);
        }
        // apply light MDS
        mds_light_ext::<F, D>(b, &mut s);
        // constraints: out - s
        let mut v = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            v.push(b.sub_extension(lw[Self::wire_output(i)], s[i]));
        }
        v
    }

    fn generators(&self, row: usize, _lc: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            Poseidon2ExtRoundGen::<F, D> {
                row,
                params: self.params.clone(),
                round_idx: self.round_idx,
                is_initial: self.is_initial,
            }
            .adapter(),
        )]
    }
}
#[derive(Debug, Default)]
pub struct Poseidon2ExtRoundGen<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    params: Poseidon2Params<F, D>,
    round_idx: usize,
    is_initial: bool,
}
impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for Poseidon2ExtRoundGen<F, D>
{
    fn id(&self) -> String {
        String::from("Poseidon2ExtRoundGen")
    }
    fn dependencies(&self) -> Vec<Target> {
        (0..P2_WIDTH)
            .map(|i| Target::wire(self.row, Poseidon2ExtRoundGate::<F, D>::wire_input(i)))
            .collect()
    }
    fn run_once(
        &self,
        pw: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> anyhow::Result<()> {
        let mut s = [F::ZERO; P2_WIDTH];
        for i in 0..P2_WIDTH {
            s[i] = pw.get_wire(Wire {
                row: self.row,
                column: Poseidon2ExtRoundGate::<F, D>::wire_input(i),
            });
        }
        let rcs = if self.is_initial {
            self.params.ext_init[self.round_idx]
        } else {
            self.params.ext_term[self.round_idx]
        };
        for i in 0..P2_WIDTH {
            s[i] = sbox7_base(s[i] + rcs[i]);
        }
        mds_light_base(&mut s);
        for i in 0..P2_WIDTH {
            out.set_wire(
                Wire {
                    row: self.row,
                    column: Poseidon2ExtRoundGate::<F, D>::wire_output(i),
                },
                s[i],
            )?;
        }
        Ok(())
    }
    fn serialize(&self, dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.round_idx)?;
        dst.write_bool(self.is_initial)
    }
    fn deserialize(src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let r = src.read_usize()?;
        let is_init = src.read_bool()?;
        Ok(Self {
            row,
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            round_idx: r,
            is_initial: is_init,
        })
    }
}

// ---------- Internal round gate: add rc0, x^7 on all, then internal mixing ----------
#[derive(Clone, Debug)]
pub struct Poseidon2IntRoundGate<F: RichField + Extendable<D>, const D: usize> {
    params: Poseidon2Params<F, D>,
    round_idx: usize, // 0..P2_INTERNAL_ROUNDS
    _pd: PhantomData<F>,
}
impl<F: RichField + Extendable<D>, const D: usize> Poseidon2IntRoundGate<F, D> {
    pub const W_IN: usize = 0;
    pub const W_OUT: usize = P2_WIDTH;
    #[inline]
    pub fn wire_input(i: usize) -> usize {
        Self::W_IN + i
    }
    #[inline]
    pub fn wire_output(i: usize) -> usize {
        Self::W_OUT + i
    }
    pub fn new(round_idx: usize) -> Self {
        Self {
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            round_idx,
            _pd: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for Poseidon2IntRoundGate<F, D> {
    fn id(&self) -> String {
        format!("Poseidon2IntRound<r={}>", self.round_idx)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.round_idx)
    }
    fn deserialize(src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let r = src.read_usize()?;
        Ok(Self::new(r))
    }

    fn num_wires(&self) -> usize {
        2 * P2_WIDTH
    }
    fn num_constants(&self) -> usize {
        0
    }
    fn num_constraints(&self) -> usize {
        P2_WIDTH
    }
    fn degree(&self) -> usize {
        7
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let lw = vars.local_wires;
        let mut s: [F::Extension; P2_WIDTH] = core::array::from_fn(|i| lw[Self::wire_input(i)]);

        // 1) lane 0: add RC then S-box; other lanes: no S-box
        s[0] = s[0] + ext_c::<F, D>(self.params.int_rc[self.round_idx]);
        s[0] = sbox7_ext::<F, D>(s[0]);

        // 2) internal mix: y[i] = diag[i] * x[i] + sum(x)
        let mut sum = s[0];
        for i in 1..P2_WIDTH {
            sum = sum + s[i];
        }
        for i in 0..P2_WIDTH {
            s[i] = s[i] * ext_c::<F, D>(self.params.diag[i]) + sum;
        }

        // out == s
        let mut v = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            v.push(lw[Self::wire_output(i)] - s[i]);
        }
        v
    }

    fn eval_unfiltered_circuit(
        &self,
        b: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let lw = &vars.local_wires;
        let mut s: [ExtensionTarget<D>; P2_WIDTH] =
            core::array::from_fn(|i| lw[Self::wire_input(i)]);

        // 1) lane 0: add RC then S-box; other lanes: no S-box
        let rc0 = b.constant_extension(ext_c::<F, D>(self.params.int_rc[self.round_idx]));
        s[0] = b.add_extension(s[0], rc0);
        s[0] = b.exp_u64_extension(s[0], 7);

        // 2) internal mix: y[i] = diag[i] * x[i] + sum(x)
        let mut sum = s[0];
        for i in 1..P2_WIDTH {
            sum = b.add_extension(sum, s[i]);
        }
        for i in 0..P2_WIDTH {
            let d = b.constant_extension(ext_c::<F, D>(self.params.diag[i]));
            let t = b.mul_extension(s[i], d);
            s[i] = b.add_extension(t, sum);
        }

        // out == s
        let mut v = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            v.push(b.sub_extension(lw[Self::wire_output(i)], s[i]));
        }
        v
    }

    fn generators(&self, row: usize, _lc: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            Poseidon2IntRoundGen::<F, D> {
                row,
                params: self.params.clone(),
                round_idx: self.round_idx,
            }
            .adapter(),
        )]
    }
}

#[derive(Default, Debug)]
pub struct Poseidon2IntRoundGen<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    params: Poseidon2Params<F, D>,
    round_idx: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for Poseidon2IntRoundGen<F, D>
{
    fn id(&self) -> String {
        String::from("Poseidon2IntRoundGen")
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..P2_WIDTH)
            .map(|i| Target::wire(self.row, Poseidon2IntRoundGate::<F, D>::wire_input(i)))
            .collect()
    }

    fn run_once(
        &self,
        pw: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> anyhow::Result<()> {
        let mut s = [F::ZERO; P2_WIDTH];
        for i in 0..P2_WIDTH {
            s[i] = pw.get_wire(Wire {
                row: self.row,
                column: Poseidon2IntRoundGate::<F, D>::wire_input(i),
            });
        }

        // 1) lane 0: add RC then S-box; other lanes: no S-box
        s[0] = sbox7_base(s[0] + self.params.int_rc[self.round_idx]);

        // 2) internal mix
        s = internal_mix_base(&s, &self.params.diag);

        // write outputs
        for i in 0..P2_WIDTH {
            out.set_wire(
                Wire {
                    row: self.row,
                    column: Poseidon2IntRoundGate::<F, D>::wire_output(i),
                },
                s[i],
            )?;
        }
        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _cd: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.round_idx)
    }
    fn deserialize(src: &mut Buffer, _cd: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let r = src.read_usize()?;
        Ok(Self {
            row,
            params: Poseidon2Params::from_p3_constants_u64(
                POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
                POSEIDON2_INTERNAL_CONSTANTS_RAW,
            ),
            round_idx: r,
        })
    }
}
