#![allow(clippy::needless_range_loop)]

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use plonky2_field::extension::Extendable;
use core::marker::PhantomData;

use crate::iop::wire::Wire;
use crate::iop::witness::Witness;

use crate::gates::gate::{Gate};
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::field::types::{Field};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::PartitionWitness;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars};
use crate::util::serialization::{Buffer, IoResult, Read, Write};
use crate::field::extension::FieldExtension;
use crate::iop::witness::WitnessWrite;

/// WIDTH=12, RATE=4 (capacity 8).
pub const P2_WIDTH: usize = 12;
pub const P2_RATE: usize  = 4;

/// We follow the GL Poseidon2 design:
/// - External (full) rounds at beginning and end:
///     state[i] = (state[i] + rc_ext[phase][i])^7; then blockwise MDS4x4
/// - Internal rounds (R=22):
///     state[0] += rc_internal[r];
///     state[i] = state[i]^7 for all i;
///     then internal mixing: y[i] = diag[i] * state[i] + sum(state)
///
/// All constants are deterministically derived from a fixed seed,
/// reusing the same seed used in CPU hashing.
const P2_INTERNAL_ROUNDS: usize = 22;
pub const EXT_INIT_U64: [[u64; P2_WIDTH]; 4] = [
    [
        16014189612424237997, 3234130550329388769, 12770310677236056740, 2522059831407870648,
        16516262146806714622, 14346955448063414307, 18098338222873569217, 16506915715127403909,
        1740142456686137078, 12590467223883536996, 6078974114340264836, 394831167838043051,
    ],
    [
        14292589812259157005, 8460209114181982917, 15011214242444884325, 13462791225860855588,
        10627487433335459888, 1624755420897328351, 2063209653571085939, 18067308934647734526,
        6538629857251435881, 3217577208297444599, 15338949981999486846, 12864825732577819657,
    ],
    [
        7706595502408214652, 16738738787192632900, 10982098878410176042, 14268282552721501435,
        11454941221469237933, 14617692430364504137, 13916571882331192922, 6814962505004916576,
        14028125575994250062, 13225657213032524375, 4626384406055707720, 14990940063728124690,
    ],
    [
        1911457633639084210, 7163368907680515062, 8413980300210324228, 1675047486134189732,
        16041698939438607870, 11228094700653369233, 6501995543359590331, 9685480985245200466,
        12679244446883551807, 8464058752967496937, 7741578890911679009, 3283703336442212930,
    ],
];

pub const EXT_TERM_U64: [[u64; P2_WIDTH]; 4] = [
    [
        17351355999406684364, 16137439874414729127, 12055514998018621195, 594679093829729354,
        16737512074876758192, 2737574840832795298, 3465446135797481753, 4292409154763056687,
        11224437497275816114, 4023045499853201921, 10779624707907978147, 14278665975259142116,
    ],
    [
        11903321889543839706, 5408672356853253133, 11722650664643554577, 15832064295836011218,
        12196513835950390987, 4557876614450441262, 4514423607167246200, 13179281050059116604,
        13816175314507993453, 16904171011585020664, 9501194326912928839, 16115508209786346411,
    ],
    [
        14980304976005993651, 13949798916377127336, 8391339834223394752, 2486734703723642889,
        16495720201386686142, 17027676271156490018, 9836722774726345255, 1598759041324173985,
        9904883565707568937, 7841704489011952451, 588114624806878733, 673913537142101185,
    ],
    [
        4599835491909300182, 12688992728618520237, 14877058244946658134, 10918174468110998885,
        10255536904610753386, 2590325024884566512, 13607037626913552939, 9862747855710264415,
        376886820764458257, 9236712289059050564, 7363125273922585709, 6340039608049649771,
    ],
];

pub const INT_RC_U64: [u64; P2_INTERNAL_ROUNDS] = [
    3164763325237167292, 4510474569205763846, 10020902063516359798, 8069563132531746417,
    3254592259677490479, 11985549796265474924, 10987927494624206223, 12015039453665918149,
    4575586241449602538, 10824249772622471957, 9852067153475416880, 18282006677946798315,
    17127667785536367426, 9262743637454041195, 7842676173661650237, 6586650076667080425,
    3357942524992632948, 13653200854074857022, 5944505826517591163, 1374723928025097978,
    360175609930259452, 18390393266911553461,
];
/// Constants (shared with CPU hashing).
#[derive(Clone, Debug, Default)]
pub struct Poseidon2Params<F: RichField + Extendable<D>, const D: usize> {
    /// 4 external rounds (initial phase), each a WIDTH-sized RC vector.
    pub ext_init: [[F; P2_WIDTH]; 4],
    /// 4 external rounds (terminal phase), each a WIDTH-sized RC vector.
    pub ext_term: [[F; P2_WIDTH]; 4],
    /// 22 internal round constants added to lane 0.
    pub int_rc:   [F; P2_INTERNAL_ROUNDS],
    /// Fixed GL diagonal used in the internal mixing.
    pub diag:     [F; P2_WIDTH],
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Params<F, D> {
    /// Create params from p3-style raw constants (as u64), exactly like your dump.
    pub fn from_p3_constants_u64(
        initial:  [[u64; P2_WIDTH]; 4],
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
            F::from_canonical_u64(0xc3b6c08e23ba9300), F::from_canonical_u64(0xd84b5de94a324fb6),
            F::from_canonical_u64(0x0d0c371c5b35b84f), F::from_canonical_u64(0x7964f570e7188037),
            F::from_canonical_u64(0x5daf18bbd996604b), F::from_canonical_u64(0x6743bc47b9595257),
            F::from_canonical_u64(0x5528b9362c59bb70), F::from_canonical_u64(0xac45e25b7127b68b),
            F::from_canonical_u64(0xa2077d7dfbb606b5), F::from_canonical_u64(0xf3faac6faee378ae),
            F::from_canonical_u64(0x0c6388b51545e883), F::from_canonical_u64(0xd27dbb6944917b60),
        ];

        Self { ext_init, ext_term, int_rc, diag }
    }
}

/// Wires layout — mirror PoseidonGate ergonomics.
/// One row does:   outputs = Poseidon2( inputs )
/// with an extra BOOL wire for swap control used by `permute_swapped`.
#[derive(Default, Debug)]
pub struct Poseidon2Gate<F: RichField + Extendable<D>, const D: usize> {
    pub params: Poseidon2Params<F, D>,
    _pd: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Gate<F, D> {
    pub const WIRE_SWAP: usize = 0;
    pub const WIRE_INPUT_START: usize = 1;                         // 1..=12
    pub const WIRE_OUTPUT_START: usize = 1 + P2_WIDTH;             // 13..=24

    pub fn wire_input(i: usize) -> usize  { Self::WIRE_INPUT_START  + i }
    pub fn wire_output(i: usize) -> usize { Self::WIRE_OUTPUT_START + i }

    pub fn num_wires() -> usize { 1 + P2_WIDTH + P2_WIDTH }        // swap + 12 in + 12 out

    pub fn new() -> Self {

        Self {
            params: Poseidon2Params::from_p3_constants_u64(EXT_INIT_U64, EXT_TERM_U64, INT_RC_U64),
            _pd: PhantomData,
        }
    }

    #[inline(always)]
    fn mds4x4_block(
        builder: &mut CircuitBuilder<F, D>,
        a: ExtensionTarget<D>,
        b: ExtensionTarget<D>,
        c: ExtensionTarget<D>,
        d: ExtensionTarget<D>,
    ) -> [ExtensionTarget<D>; 4] {
        // [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]

        // m0
        let t3 = builder.mul_const_extension(F::from_canonical_u64(3), d);
        let t2 = builder.mul_const_add_extension(F::from_canonical_u64(1), c, t3);
        let t1 = builder.mul_const_add_extension(F::from_canonical_u64(7), b, t2);
        let m0 = builder.mul_const_add_extension(F::from_canonical_u64(5), a, t1);

        // m1
        let t3 = builder.mul_const_extension(F::from_canonical_u64(1), d);
        let t2 = builder.mul_const_add_extension(F::from_canonical_u64(1), c, t3);
        let t1 = builder.mul_const_add_extension(F::from_canonical_u64(6), b, t2);
        let m1 = builder.mul_const_add_extension(F::from_canonical_u64(4), a, t1);

        // m2
        let t3 = builder.mul_const_extension(F::from_canonical_u64(7), d);
        let t2 = builder.mul_const_add_extension(F::from_canonical_u64(5), c, t3);
        let t1 = builder.mul_const_add_extension(F::from_canonical_u64(3), b, t2);
        let m2 = builder.mul_const_add_extension(F::from_canonical_u64(1), a, t1);

        // m3
        let t3 = builder.mul_const_extension(F::from_canonical_u64(6), d);
        let t2 = builder.mul_const_add_extension(F::from_canonical_u64(4), c, t3);
        let t1 = builder.mul_const_add_extension(F::from_canonical_u64(1), b, t2);
        let m3 = builder.mul_const_add_extension(F::from_canonical_u64(1), a, t1);

        [m0, m1, m2, m3]
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for Poseidon2Gate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={P2_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Poseidon2Gate::new())
    }

    fn eval_unfiltered(&self,vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        type EF<F, const D: usize> = <F as Extendable<D>>::Extension;


        // Helpers
        #[inline(always)]
        fn fe<Fx: crate::hash::hash_types::RichField + Extendable<Dx>, const Dx: usize>(x: Fx) -> EF<Fx, Dx> {
            // Lift base-field constant to extension
            EF::<Fx, Dx>::from_canonical_u64(x.to_canonical_u64())
        }
        #[inline(always)]
        fn u<Fx: crate::hash::hash_types::RichField + Extendable<Dx>, const Dx: usize>(x: u64) -> EF<Fx, Dx> {
            EF::<Fx, Dx>::from_canonical_u64(x)
        }
        #[inline(always)]
        fn sbox7<Fx: crate::hash::hash_types::RichField + Extendable<Dx>, const Dx: usize>(x: EF<Fx, Dx>) -> EF<Fx, Dx> {
            let x2 = x * x;
            let x4 = x2 * x2;
            (x * x2) * x4
        }

        let one = u::<F, D>(1);

        // Load inputs (as extension elements)
        let lw = vars.local_wires;
        let swap = lw[Self::WIRE_SWAP];

        // Build initial state WITH swap, and keep it.
        let mut s: [EF<F, D>; P2_WIDTH] =
            [EF::<F, D>::ZERO; P2_WIDTH];

        for i in 0..4 {
            let a = lw[Self::wire_input(i)];
            let b = lw[Self::wire_input(i + 4)];
            let one = F::Extension::ONE;
            let s1 = a * (one - swap) + b * swap;
            let s2 = b * (one - swap) + a * swap;
            s[i] = s1;
            s[i + 4] = s2;
        }
        for i in 8..P2_WIDTH {
            s[i] = lw[Self::wire_input(i)];
        }

        // ---- External initial: 4 rounds ----
        for r in 0..4 {
            for i in 0..P2_WIDTH {
                s[i] = s[i] + fe::<F, D>(self.params.ext_init[r][i]);
                s[i] = sbox7::<F, D>(s[i]);
            }
            // blockwise MDS inside the round
            for k in (0..P2_WIDTH).step_by(4) {
                let a = s[k]; let b = s[k+1]; let c = s[k+2]; let d = s[k+3];
                let m0 = a*u::<F,D>(5) + b*u::<F,D>(7) + c*u::<F,D>(1) + d*u::<F,D>(3);
                let m1 = a*u::<F,D>(4) + b*u::<F,D>(6) + c*u::<F,D>(1) + d*u::<F,D>(1);
                let m2 = a*u::<F,D>(1) + b*u::<F,D>(3) + c*u::<F,D>(5) + d*u::<F,D>(7);
                let m3 = a*u::<F,D>(1) + b*u::<F,D>(1) + c*u::<F,D>(4) + d*u::<F,D>(6);
                s[k]=m0; s[k+1]=m1; s[k+2]=m2; s[k+3]=m3;
            }
        }

        // ------------------------ Internal rounds ------------------------
        for r in 0..P2_INTERNAL_ROUNDS {
            // add rc to lane 0
            s[0] = s[0] + fe::<F, D>(self.params.int_rc[r]);

            // S-box on all lanes
            for i in 0..P2_WIDTH {
                s[i] = sbox7::<F, D>(s[i]);
            }

            // Internal mixing: y[i] = diag[i] * x[i] + sum(x)
            let mut sum = s[0];
            for i in 1..P2_WIDTH {
                sum = sum + s[i];
            }
            for i in 0..P2_WIDTH {
                s[i] = s[i] * fe::<F, D>(self.params.diag[i]) + sum;
            }
        }

        // ---- External terminal: 4 rounds ----
        for r in 0..4 {
            for i in 0..P2_WIDTH {
                s[i] = s[i] + fe::<F, D>(self.params.ext_term[r][i]);
                s[i] = sbox7::<F, D>(s[i]);
            }
            for k in (0..P2_WIDTH).step_by(4) {
                let a = s[k]; let b = s[k+1]; let c = s[k+2]; let d = s[k+3];
                let m0 = a*u::<F,D>(5) + b*u::<F,D>(7) + c*u::<F,D>(1) + d*u::<F,D>(3);
                let m1 = a*u::<F,D>(4) + b*u::<F,D>(6) + c*u::<F,D>(1) + d*u::<F,D>(1);
                let m2 = a*u::<F,D>(1) + b*u::<F,D>(3) + c*u::<F,D>(5) + d*u::<F,D>(7);
                let m3 = a*u::<F,D>(1) + b*u::<F,D>(1) + c*u::<F,D>(4) + d*u::<F,D>(6);
                s[k]=m0; s[k+1]=m1; s[k+2]=m2; s[k+3]=m3;
            }
        }

        // -------- Build constraints --------
        let mut constrs = Vec::with_capacity(P2_WIDTH + 1);

        // Output equality constraints
        for i in 0..P2_WIDTH {
            let got = lw[self::Poseidon2Gate::<F, D>::wire_output(i)];
            constrs.push(got - s[i]);
        }

        // swap is boolean: swap * (swap - 1) == 0
        let swap = lw[self::Poseidon2Gate::<F, D>::WIRE_SWAP];
        constrs.push(swap * (swap - one));

        constrs
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        type FE<Fx, const Dx: usize> = <Fx as Extendable<Dx>>::Extension;

        // Helper: lift base-field const to extension const target.
        #[inline(always)]
        fn cst<Fx: RichField + Extendable<Dx>, const Dx: usize>(
            builder: &mut CircuitBuilder<Fx, Dx>,
            x: Fx,
        ) -> ExtensionTarget<Dx> {
            builder.constant_extension(FE::<Fx, Dx>::from_canonical_u64(x.to_canonical_u64()))
        }

        // S-box x^7 over extension.
        #[inline(always)]
        fn sbox7<Fx: RichField + Extendable<Dx>, const Dx: usize>(
            builder: &mut CircuitBuilder<Fx, Dx>,
            x: ExtensionTarget<Dx>,
        ) -> ExtensionTarget<Dx> {
            builder.exp_u64_extension(x, 7)
        }

        let lw = &vars.local_wires;

        let swap = vars.local_wires[Self::WIRE_SWAP]; // ExtensionTarget<D>, constrained to {0,1}

        let mut s: [ExtensionTarget<D>; P2_WIDTH] =
            core::array::from_fn(|_| builder.zero_extension());

        for i in 0..4 {
            let a = lw[Self::wire_input(i)];       // left  half
            let b = lw[Self::wire_input(i + 4)];   // right half

            // s[i]     = a + swap*(b - a)
            let diff_ab  = builder.sub_extension(b, a);
            let delta_ab = builder.mul_extension(swap, diff_ab);
            s[i]         = builder.add_extension(a, delta_ab);

            // s[i+4]   = b + swap*(a - b)  = (1 - swap)*b + swap*a
            let diff_ba  = builder.sub_extension(a, b);
            let delta_ba = builder.mul_extension(swap, diff_ba);
            s[i + 4]     = builder.add_extension(b, delta_ba);
        }
        for i in 8..P2_WIDTH {
            s[i] = vars.local_wires[Self::wire_input(i)];
        }

        // External initial (4 rounds)
        for r in 0..4 {
            for i in 0..P2_WIDTH {
                let rc = cst::<F, D>(builder, self.params.ext_init[r][i]);
                s[i] = builder.add_extension(s[i], rc);
                s[i] = sbox7::<F, D>(builder, s[i]);
            }
            for k in (0..P2_WIDTH).step_by(4) {
                let [m0, m1, m2, m3] = Self::mds4x4_block(builder, s[k], s[k+1], s[k+2], s[k+3]);
                s[k]=m0; s[k+1]=m1; s[k+2]=m2; s[k+3]=m3;
            }
        }
        // ---------------------- Internal rounds (R=22) ----------------------
        for r in 0..P2_INTERNAL_ROUNDS {
            // Add round constant to lane 0.
            let rc0 = cst::<F, D>(builder, self.params.int_rc[r]);
            s[0] = builder.add_extension(s[0], rc0);

            // S-box on all lanes.
            for i in 0..P2_WIDTH {
                s[i] = sbox7::<F, D>(builder, s[i]);
            }

            // Internal mixing: y[i] = diag[i] * x[i] + sum(x)
            let mut sum = s[0];
            for i in 1..P2_WIDTH {
                sum = builder.add_extension(sum, s[i]);
            }
            for i in 0..P2_WIDTH {
                let d = self.params.diag[i];
                let t = builder.mul_const_extension(d, s[i]);
                s[i] = builder.add_extension(t, sum);
            }
        }
        // External terminal (4 rounds)
        for r in 0..4 {
            for i in 0..P2_WIDTH {
                let rc = cst::<F, D>(builder, self.params.ext_term[r][i]);
                s[i] = builder.add_extension(s[i], rc);
                s[i] = sbox7::<F, D>(builder, s[i]);
            }
            for k in (0..P2_WIDTH).step_by(4) {
                let [m0, m1, m2, m3] = Self::mds4x4_block(builder, s[k], s[k+1], s[k+2], s[k+3]);
                s[k]=m0; s[k+1]=m1; s[k+2]=m2; s[k+3]=m3;
            }
        }

        // --------------------- Build and return constraints -----------------
        let mut constrs = Vec::with_capacity(P2_WIDTH + 1);

        // Constrain outputs equal to permuted state.
        for i in 0..P2_WIDTH {
            let got = lw[Self::wire_output(i)];
            constrs.push(builder.sub_extension(got, s[i]));
        }

        // Enforce swap is boolean: swap * (swap - 1) == 0
        let swap = lw[Self::WIRE_SWAP];
        let one = builder.one_extension();
        let swap_minus_one = builder.sub_extension(swap, one);
        constrs.push(builder.mul_extension(swap, swap_minus_one));

        constrs
    }
    
    fn generators(&self, row: usize, _lc: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            Poseidon2Generator::<F, D> { row, params: self.params.clone() }.adapter()
        )]
    }

    fn num_wires(&self) -> usize {
        // swap (1) + inputs (P2_WIDTH) + outputs (P2_WIDTH)
        1 + P2_WIDTH + P2_WIDTH
        // equivalently: 2 * P2_WIDTH + 1
    }

    fn num_constants(&self) -> usize {
        // All constants are serialized in `self.params`; no per-row local constants.
        0
    }

    fn degree(&self) -> usize {
        // Highest algebraic degree in constraints comes from x^7 S-Box.
        7
    }

    fn num_constraints(&self) -> usize {
        // 12 output-equality constraints + 1 boolean constraint for `swap`.
        P2_WIDTH + 1 // 12 + 1 = 13
    }
    
    fn eval_unfiltered_base_one(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBase<F>,
        mut yield_constr: super::util::StridedConstraintConsumer<F>,
    ) {
        // Note that this method uses `yield_constr` instead of returning its constraints.
        // `yield_constr` abstracts out the underlying memory layout.
        let local_constants = &vars_base
            .local_constants
            .iter()
            .map(|c| <F as plonky2_field::extension::Extendable<D>>::Extension::from_basefield(*c))
            .collect::<Vec<_>>();
        let local_wires = &vars_base
            .local_wires
            .iter()
            .map(|w| <F as plonky2_field::extension::Extendable<D>>::Extension::from_basefield(*w))
            .collect::<Vec<_>>();
        let public_inputs_hash = &vars_base.public_inputs_hash;
        let vars = crate::plonk::vars::EvaluationVars {
            local_constants,
            local_wires,
            public_inputs_hash,
        };
        let values = self.eval_unfiltered(vars);
    
        // Each value should be in the base field, i.e. only the degree-zero part should be nonzero.
        values.into_iter().for_each(|value| {
            core::debug_assert!(F::Extension::is_in_basefield(&value));
            yield_constr.one(value.to_basefield_array()[0])
        })
    }

    fn eval_unfiltered_base_batch(&self, vars_base: crate::plonk::vars::EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let mut res = std::vec![F::ZERO; vars_base.len() * self.num_constraints()];
        for (i, vars_base_one) in vars_base.iter().enumerate() {
            self.eval_unfiltered_base_one(
                vars_base_one,
                super::util::StridedConstraintConsumer::new(&mut res, vars_base.len(), i),
            );
        }
        res
    }
}

// -------- base-field helpers used by the generator --------
#[inline(always)]
fn sbox7_base<F: Field>(x: F) -> F {
    let x2 = x * x;
    let x4 = x2 * x2;
    (x * x2) * x4
}

#[inline(always)]
fn mds4x4_base<F: Field>(v: [F; 4]) -> [F; 4] {
    let one  = F::from_canonical_u64(1);
    let three= F::from_canonical_u64(3);
    let four = F::from_canonical_u64(4);
    let five = F::from_canonical_u64(5);
    let six  = F::from_canonical_u64(6);
    let seven= F::from_canonical_u64(7);
    let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
    [
        a * five + b * seven + c * one + d * three,
        a * four + b * six   + c * one + d * one,
        a * one  + b * three + c * five + d * seven,
        a * one  + b * one   + c * four + d * six,
    ]
}

#[inline(always)]
fn internal_mix_base<F: Field>(x: &[F; P2_WIDTH], diag: &[F; P2_WIDTH]) -> [F; P2_WIDTH] {
    let mut sum = x[0];
    for i in 1..P2_WIDTH { sum += x[i]; }
    let mut y = [F::ZERO; P2_WIDTH];
    for i in 0..P2_WIDTH {
        y[i] = diag[i] * x[i] + sum;
    }
    y
}

#[derive(Debug)]
struct Poseidon2Generator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    params: Poseidon2Params<F, D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for Poseidon2Generator<F, D> {
    
    fn id(&self) -> String { "Poseidon2Generator".to_string() }

    fn dependencies(&self) -> Vec<Target> {
        // Only the 12 inputs are actual data deps.
        let mut deps = Vec::with_capacity(P2_WIDTH);
        for i in 0..P2_WIDTH {
            deps.push(Target::wire(self.row, Poseidon2Gate::<F, D>::wire_input(i)));
        }
        deps
    }

    fn run_once(
        &self,
        pw: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> anyhow::Result<()> {
        // swap is hard-wired to 0 in the circuit; don’t read it here.
        let mut s = [F::ZERO; P2_WIDTH];

        // First 8 lanes: no swap (left stays left, right stays right).
        for i in 0..4 {
            let a = pw.get_wire(Wire { row: self.row, column: Poseidon2Gate::<F, D>::wire_input(i) });
            let b = pw.get_wire(Wire { row: self.row, column: Poseidon2Gate::<F, D>::wire_input(i + 4) });
            s[i]     = a;
            s[i + 4] = b;
        }
        for i in 8..P2_WIDTH {
            s[i] = pw.get_wire(Wire { row: self.row, column: Poseidon2Gate::<F, D>::wire_input(i) });
        }

        // ---- external initial: 4 rounds ----
        for r in 0..4 {
            for i in 0..P2_WIDTH {
                s[i] = s[i] + self.params.ext_init[r][i];
                s[i] = sbox7_base(s[i]);
            }
            for k in (0..P2_WIDTH).step_by(4) {
                let [m0, m1, m2, m3] = mds4x4_base([s[k], s[k + 1], s[k + 2], s[k + 3]]);
                s[k] = m0; s[k + 1] = m1; s[k + 2] = m2; s[k + 3] = m3;
            }
        }

        // ---- internal rounds ----
        for r in 0..P2_INTERNAL_ROUNDS {
            s[0] = s[0] + self.params.int_rc[r];
            for i in 0..P2_WIDTH {
                s[i] = sbox7_base(s[i]);
            }
            s = internal_mix_base(&s, &self.params.diag);
        }

        // ---- external terminal: 4 rounds ----
        for r in 0..4 {
            for i in 0..P2_WIDTH {
                s[i] = s[i] + self.params.ext_term[r][i];
                s[i] = sbox7_base(s[i]);
            }
            for k in (0..P2_WIDTH).step_by(4) {
                let [m0, m1, m2, m3] = mds4x4_base([s[k], s[k + 1], s[k + 2], s[k + 3]]);
                s[k] = m0; s[k + 1] = m1; s[k + 2] = m2; s[k + 3] = m3;
            }
        }

        for i in 0..P2_WIDTH {
            let w = Wire { row: self.row, column: Poseidon2Gate::<F, D>::wire_output(i) };
            out.set_wire(w, s[i])?;
        }
        Ok(())
    }


    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            params: Poseidon2Params::from_p3_constants_u64(EXT_INIT_U64, EXT_TERM_U64, INT_RC_U64)
        })
    }
}
