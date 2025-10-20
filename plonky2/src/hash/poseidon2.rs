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

use p3_goldilocks::{Goldilocks as P3G, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

use p3_field::integers::QuotientMap;
use p3_field::{PrimeCharacteristicRing, PrimeField64 as P3PrimeField64};

use crate::field::types::{Field, PrimeField64};
use crate::gates::poseidon2::{
    Poseidon2ExtInitPreambleGate, Poseidon2ExtRoundGate, Poseidon2IntRoundGate, EXT_INIT_U64,
    EXT_TERM_U64, INT_RC_U64, P2_INTERNAL_ROUNDS, P2_WIDTH,
};
use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::{hash_n_to_hash_no_pad_p2, PlonkyPermutation};
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};
use plonky2_field::extension::Extendable;

// We only support Goldilocks for now, which matches your Poseidon2Core.
use plonky2_field::goldilocks_field::GoldilocksField as GL;

// ---------- Params (match your Poseidon2Core exactly) ----------
const SPONGE_WIDTH: usize = 12;
const SPONGE_RATE: usize = 4; // 4-felt output, 4-felt rate, 8-felt capacity

// ---------- Internal helper: p3 permutation on Goldilocks ----------
#[inline(always)]
fn p2_permute_gl(mut state: [GL; SPONGE_WIDTH]) -> [GL; SPONGE_WIDTH] {
    // Convert to p3 Goldilocks.
    let mut s_p3 = [P3G::ZERO; SPONGE_WIDTH];
    for i in 0..SPONGE_WIDTH {
        // GL -> u64 -> P3G (both mod 2^64 - 2^32 + 1)
        s_p3[i] = unsafe { P3G::from_canonical_unchecked(state[i].to_canonical_u64()) };
    }

    // Create a version of p3 gl poseidon2 from the ext_init/ext_term/int_rc constants.
    // These constants match what are generated from the poseidon-resonance core
    // using the rand chacha rng and the seed 0x0189_1891_8918_9189;
    let poseidon2_from_const: Poseidon2Goldilocks<SPONGE_WIDTH> = Poseidon2Goldilocks::new(
        p3_poseidon2::ExternalLayerConstants::<P3G, SPONGE_WIDTH>::new(
            EXT_INIT_U64
                .map(|row| row.map(|x| unsafe { P3G::from_canonical_unchecked(x) }))
                .to_vec(),
            EXT_TERM_U64
                .map(|row| row.map(|x| unsafe { P3G::from_canonical_unchecked(x) }))
                .to_vec(),
        ),
        INT_RC_U64
            .map(|x| unsafe { P3G::from_canonical_unchecked(x) })
            .to_vec(),
    );

    let mut st = s_p3;
    poseidon2_from_const.permute_mut(&mut st);

    // Back to plonky2 GL
    for i in 0..SPONGE_WIDTH {
        state[i] = GL::from_noncanonical_u64(st[i].as_canonical_u64());
    }
    state
}

// ---------- Permuter wiring ----------
trait P2Permuter: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

// CPU: use the canonical p3 Poseidon2 GL permutation.
impl P2Permuter for GL {
    #[inline(always)]
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        p2_permute_gl(input)
    }
}

// Targets: must go through AlgebraicHasher::permute_swapped (the gate); never call permute().
impl P2Permuter for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}

// ---------- PlonkyPermutation wrapper ----------
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Poseidon2Permutation<T> {
    state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for Poseidon2Permutation<T> {}

impl<T> AsRef<[T]> for Poseidon2Permutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

impl<T: Default + Copy> Poseidon2Permutation<T> {
    #[inline(always)]
    fn new_blank() -> Self {
        Self {
            state: [T::default(); SPONGE_WIDTH],
        }
    }
}

impl<T: Copy + core::fmt::Debug + Default + Eq + P2Permuter + Send + Sync> PlonkyPermutation<T>
    for Poseidon2Permutation<T>
{
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self::new_blank();
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let end = start_idx + elts.len();
        self.state[start_idx..end].copy_from_slice(elts);
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

// ---------- Hasher (CPU) ----------
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2Hash;

impl<F: RichField + P2Permuter> Hasher<F> for Poseidon2Hash {
    const HASH_SIZE: usize = NUM_HASH_OUT_ELTS * 8; // 4 * 8 = 32 bytes
    type Hash = HashOut<F>;
    type Permutation = Poseidon2Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad_p2::<F, Self::Permutation>(input)
    }

    /// Keep CPU equivalence: concatenate 8 felts and call the same `hash_no_pad`.
    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let mut input = [F::ZERO; 2 * NUM_HASH_OUT_ELTS];
        input[..NUM_HASH_OUT_ELTS].copy_from_slice(&left.elements);
        input[NUM_HASH_OUT_ELTS..].copy_from_slice(&right.elements);
        Self::hash_no_pad(&input)
    }
}

#[allow(dead_code)]
pub fn hash_no_pad_bytes(input: &[GL]) -> [u8; 32] {
    let h = Poseidon2Hash::hash_no_pad(input);
    let mut out = [0u8; 32];
    // Little-endian u64 per felt, concatenated.
    for (i, elt) in h.elements.iter().enumerate() {
        let w = elt.to_canonical_u64().to_le_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&w);
    }
    out
}

// ---------- AlgebraicHasher (in-circuit) ----------
impl<F: RichField + P2Permuter> AlgebraicHasher<F> for Poseidon2Hash {
    type AlgebraicPermutation = Poseidon2Permutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        _swap: BoolTarget, // ignored (assumed 0)
        b: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        // Start from inputs as-is (no swap).
        let mut state: [Target; SPONGE_WIDTH] = inputs.as_ref().try_into().unwrap();
        // 0) initial pre-mds_light
        {
            let gate = Poseidon2ExtInitPreambleGate::<F, D>::new();
            let row = b.add_gate(gate, vec![]);
            for i in 0..P2_WIDTH {
                b.connect(
                    state[i],
                    Target::wire(row, Poseidon2ExtInitPreambleGate::<F, D>::wire_input(i)),
                );
                state[i] = Target::wire(row, Poseidon2ExtInitPreambleGate::<F, D>::wire_output(i));
            }
        }

        // 1) 4 external initial rounds (rc+sbox on all lanes, then light-MDS)
        for r in 0..4 {
            let row = b.add_gate(Poseidon2ExtRoundGate::<F, D>::new_initial(r), vec![]);
            for i in 0..P2_WIDTH {
                b.connect(
                    state[i],
                    Target::wire(row, Poseidon2ExtRoundGate::<F, D>::wire_input(i)),
                );
                state[i] = Target::wire(row, Poseidon2ExtRoundGate::<F, D>::wire_output(i));
            }
        }

        // 2) 22 internal rounds (rc+sbox on lane 0 only, then internal mix)
        for r in 0..P2_INTERNAL_ROUNDS {
            let row = b.add_gate(Poseidon2IntRoundGate::<F, D>::new(r), vec![]);
            for i in 0..P2_WIDTH {
                b.connect(
                    state[i],
                    Target::wire(row, Poseidon2IntRoundGate::<F, D>::wire_input(i)),
                );
                state[i] = Target::wire(row, Poseidon2IntRoundGate::<F, D>::wire_output(i));
            }
        }

        // 3) 4 external terminal rounds (rc+sbox all lanes, then light-MDS)
        for r in 0..4 {
            let row = b.add_gate(Poseidon2ExtRoundGate::<F, D>::new_terminal(r), vec![]);
            for i in 0..P2_WIDTH {
                b.connect(
                    state[i],
                    Target::wire(row, Poseidon2ExtRoundGate::<F, D>::wire_input(i)),
                );
                state[i] = Target::wire(row, Poseidon2ExtRoundGate::<F, D>::wire_output(i));
            }
        }

        Poseidon2Permutation { state }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use plonky2_field::goldilocks_field::GoldilocksField as F;
    use rand_chacha::rand_core::RngCore;

    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    /// Helper: build a circuit that hashes `inputs` in-circuit with Poseidon2,
    /// and constrain it to match the CPU result.
    fn assert_hash_matches(inputs: Vec<F>) {
        // --- CPU reference ---
        let cpu = Poseidon2Hash::hash_no_pad(&inputs);

        // --- Build circuit ---
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());

        // Add virtual targets for all input elements.
        let ts: Vec<Target> = inputs
            .iter()
            .map(|_| builder.add_virtual_target())
            .collect();

        // In-circuit Poseidon2 additive-absorption hash.
        let out_t = builder.hash_n_to_hash_no_pad_p2::<Poseidon2Hash>(ts.clone());

        // Constrain circuit outputs to equal the CPU digest.
        for i in 0..NUM_HASH_OUT_ELTS {
            let c = builder.constant(cpu.elements[i]);
            builder.connect(out_t.elements[i], c);
        }

        let data = builder.build::<C>();

        // Set the witness for the inputs and prove+verify.
        let mut pw = PartialWitness::new();
        for (t, v) in ts.into_iter().zip(inputs.into_iter()) {
            pw.set_target(t, v).expect("setting target failed");
        }
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
    }

    #[test]
    fn poseidon2_hash_matches_cpu_edge_lengths() {
        // Exercise the padding logic carefully: empty, short, full blocks, and beyond.
        // RATE = 4, so we hit: 0,1,2,3,4,5,7,8,9,12,16,17
        let lens: [usize; 12] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17];
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);

        for &len in &lens {
            let inputs: Vec<F> = (0..len)
                .map(|_| F::from_canonical_u64(rng.next_u64()))
                .collect();
            assert_hash_matches(inputs);
        }
    }

    #[test]
    fn poseidon2_hash_matches_cpu_random_cases() {
        // A bunch of random lengths & values.
        let mut rng = ChaCha8Rng::seed_from_u64(0xFACEFEED);
        for _ in 0..20 {
            let len = (rng.next_u32() as usize) % 64; // up to 63 elements
            let inputs: Vec<F> = (0..len)
                .map(|_| F::from_canonical_u64(rng.next_u64()))
                .collect();
            assert_hash_matches(inputs);
        }
    }
}
