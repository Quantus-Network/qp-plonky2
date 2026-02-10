#![allow(clippy::needless_range_loop)]

extern crate alloc;

#[cfg(not(feature = "std"))]
use core::fmt::Debug;

use p3_field::integers::QuotientMap;
use p3_field::{PrimeCharacteristicRing, PrimeField64 as P3PrimeField64};
use p3_goldilocks::Goldilocks as P3G;
use p3_symmetric::Permutation;
// We only support Goldilocks for now, which matches your Poseidon2Core.
use plonky2_field::goldilocks_field::GoldilocksField as GL;
use qp_poseidon_constants::create_poseidon;

use crate::field::types::{Field, PrimeField64};
use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::plonk::config::Hasher;
use qp_plonky2_core::hashing::{hash_n_to_hash_no_pad_p2, PlonkyPermutation};

// ---------- Params (match your Poseidon2Core exactly) ----------
const SPONGE_WIDTH: usize = 12;
const SPONGE_RATE: usize = 4; // 4-felt output, 4-felt rate, 8-felt capacity

/// ---------- Internal helper: p3 permutation on Goldilocks ----------
#[inline(always)]
fn p2_permute_gl(mut state: [GL; SPONGE_WIDTH]) -> [GL; SPONGE_WIDTH] {
    // Convert to p3 Goldilocks.
    let mut s_p3 = [P3G::ZERO; SPONGE_WIDTH];
    for i in 0..SPONGE_WIDTH {
        // GL -> u64 -> P3G (both mod 2^64 - 2^32 + 1)
        s_p3[i] = unsafe { P3G::from_canonical_unchecked(state[i].to_canonical_u64()) };
    }

    let poseidon2 = create_poseidon();
    let mut st = s_p3;
    poseidon2.permute_mut(&mut st);

    // Back to plonky2 GL
    for i in 0..SPONGE_WIDTH {
        state[i] = GL::from_noncanonical_u64(st[i].as_canonical_u64());
    }
    state
}

// ---------- Permuter wiring ----------
pub trait P2Permuter: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

// CPU: use the canonical p3 Poseidon2 GL permutation.
impl P2Permuter for GL {
    #[inline(always)]
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        p2_permute_gl(input)
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
