//! Concrete instantiation of a hash function.
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::field::types::Field;
use crate::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};

/// Permutation that can be used in the sponge construction for an algebraic hash.
pub trait PlonkyPermutation<T: Copy + Default>:
    AsRef<[T]> + Copy + Debug + Default + Eq + Sync + Send
{
    const RATE: usize;
    const WIDTH: usize;

    /// Initialises internal state with values from `iter` until
    /// `iter` is exhausted or `Self::WIDTH` values have been
    /// received; remaining state (if any) initialised with
    /// `T::default()`. To initialise remaining elements with a
    /// different value, instead of your original `iter` pass
    /// `iter.chain(core::iter::repeat(F::from_canonical_u64(12345)))`
    /// or similar.
    fn new<I: IntoIterator<Item = T>>(iter: I) -> Self;

    /// Set idx-th state element to be `elt`. Panics if `idx >= WIDTH`.
    fn set_elt(&mut self, elt: T, idx: usize);

    /// Set state element `i` to be `elts[i] for i =
    /// start_idx..start_idx + n` where `n = min(elts.len(),
    /// WIDTH-start_idx)`. Panics if `start_idx > WIDTH`.
    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize);

    /// Same semantics as for `set_from_iter` but probably faster than
    /// just calling `set_from_iter(elts.iter())`.
    fn set_from_slice(&mut self, elts: &[T], start_idx: usize);

    /// Apply permutation to internal state
    fn permute(&mut self);

    /// Return a slice of `RATE` elements
    fn squeeze(&self) -> &[T];
}

/// A one-way compression function which takes two ~256 bit inputs and returns a ~256 bit output.
pub fn compress<F: Field, P: PlonkyPermutation<F>>(x: HashOut<F>, y: HashOut<F>) -> HashOut<F> {
    // TODO: With some refactoring, this function could be implemented as
    // hash_n_to_m_no_pad(chain(x.elements, y.elements), NUM_HASH_OUT_ELTS).

    debug_assert_eq!(x.elements.len(), NUM_HASH_OUT_ELTS);
    debug_assert_eq!(y.elements.len(), NUM_HASH_OUT_ELTS);
    debug_assert!(P::RATE >= NUM_HASH_OUT_ELTS);

    let mut perm = P::new(core::iter::repeat(F::ZERO));
    perm.set_from_slice(&x.elements, 0);
    perm.set_from_slice(&y.elements, NUM_HASH_OUT_ELTS);

    perm.permute();

    HashOut {
        elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
    }
}

/// Hash a message without any padding step. Note that this can enable length-extension attacks.
/// However, it is still collision-resistant in cases where the input has a fixed length.
pub fn hash_n_to_m_no_pad<F: RichField, P: PlonkyPermutation<F>>(
    inputs: &[F],
    num_outputs: usize,
) -> Vec<F> {
    let mut perm = P::new(core::iter::repeat(F::ZERO));

    // Absorb all input chunks.
    for input_chunk in inputs.chunks(P::RATE) {
        perm.set_from_slice(input_chunk, 0);
        perm.permute();
    }

    // Squeeze until we have the desired number of outputs.
    let mut outputs = Vec::new();
    loop {
        for &item in perm.squeeze() {
            outputs.push(item);
            if outputs.len() == num_outputs {
                return outputs;
            }
        }
        perm.permute();
    }
}

pub fn hash_n_to_hash_no_pad<F: RichField, P: PlonkyPermutation<F>>(inputs: &[F]) -> HashOut<F> {
    HashOut::from_vec(hash_n_to_m_no_pad::<F, P>(inputs, NUM_HASH_OUT_ELTS))
}

/// Poseidon2 variable length padding (…||1||0* to RATE)
pub fn hash_n_to_hash_no_pad_p2<F: RichField, P: PlonkyPermutation<F>>(inputs: &[F]) -> HashOut<F> {
    let rate = P::RATE;

    // Defensive: RATE should never be 0 for a real permutation.
    if rate == 0 {
        return HashOut::from_vec(vec![F::ZERO; NUM_HASH_OUT_ELTS]);
    }

    // Append one '1' and zero-pad to a rate-aligned length.
    // This automatically adds a whole [1,0,...,0] block when inputs.len() % rate == 0,
    // including the empty-input case.
    let padded_len = ((inputs.len() + 1 + rate - 1) / rate) * rate;

    let mut msg = vec![F::ZERO; padded_len];
    msg[..inputs.len()].copy_from_slice(inputs);
    msg[inputs.len()] = F::ONE;

    // Absorb/additively and permute per block.
    let mut perm = P::new(core::iter::repeat(F::ZERO));
    for block in msg.chunks(rate) {
        for (i, &x) in block.iter().enumerate() {
            let si = perm.as_ref()[i];
            perm.set_elt(si + x, i);
        }
        perm.permute();
    }

    // Squeeze without an extra permute.
    HashOut::from_vec(perm.squeeze()[..NUM_HASH_OUT_ELTS].to_vec())
}
