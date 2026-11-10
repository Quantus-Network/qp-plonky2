#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::mem::size_of;

use itertools::Itertools;
use tiny_keccak::{Hasher as _, Keccak};

use crate::config::Hasher;
use crate::hash_types::{BytesHash, RichField};
use crate::hashing::PlonkyPermutation;

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

/// Keccak-256 (the original Keccak padding, as used by Ethereum; not SHA3-256).
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(data);
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    output
}

/// Keccak-256 pseudo-permutation (not necessarily one-to-one) used in the challenger.
/// A state `input: [F; 12]` is sent to the field representation of `H(input) || H(H(input)) || H(H(H(input)))`
/// where `H` is the Keccak-256 hash.
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct KeccakPermutation<F: RichField> {
    state: [F; SPONGE_WIDTH],
}

impl<F: RichField> Eq for KeccakPermutation<F> {}

impl<F: RichField> AsRef<[F]> for KeccakPermutation<F> {
    fn as_ref(&self) -> &[F] {
        &self.state
    }
}

// TODO: Several implementations here are copied from
// PoseidonPermutation; they should be refactored.
impl<F: RichField> PlonkyPermutation<F> for KeccakPermutation<F> {
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = F>>(elts: I) -> Self {
        let mut perm = Self {
            state: [F::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: F, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[F], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = F>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        let mut state_bytes = vec![0u8; SPONGE_WIDTH * size_of::<u64>()];
        for i in 0..SPONGE_WIDTH {
            state_bytes[i * size_of::<u64>()..(i + 1) * size_of::<u64>()]
                .copy_from_slice(&self.state[i].to_canonical_u64().to_le_bytes());
        }

        let hash_onion = core::iter::repeat_with(|| {
            let output = keccak256(&state_bytes);
            state_bytes = output.to_vec();
            output
        });

        let hash_onion_u64s = hash_onion.flat_map(|output| {
            output
                .chunks_exact(size_of::<u64>())
                .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
                .collect_vec()
        });

        // Parse field elements from u64 stream, using rejection sampling such that words that don't
        // fit in F are ignored.
        let hash_onion_elems = hash_onion_u64s
            .filter(|&word| word < F::ORDER)
            .map(F::from_canonical_u64);

        self.state = hash_onion_elems
            .take(SPONGE_WIDTH)
            .collect_vec()
            .try_into()
            .unwrap();
    }

    fn squeeze(&self) -> &[F] {
        &self.state[..Self::RATE]
    }
}

/// Keccak-256 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KeccakHash<const N: usize>;
impl<F: RichField, const N: usize> Hasher<F> for KeccakHash<N> {
    const HASH_SIZE: usize = N;
    type Hash = BytesHash<N>;
    type Permutation = KeccakPermutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        // Serialize field elements to bytes
        let mut buffer = Vec::with_capacity(input.len() * 8);
        for f in input {
            buffer.extend_from_slice(&f.to_canonical_u64().to_le_bytes());
        }
        let mut arr = [0; N];
        let hash_bytes = keccak256(&buffer);
        arr.copy_from_slice(&hash_bytes[..N]);
        BytesHash(arr)
    }

    fn hash_leaf(input: &[F]) -> Self::Hash {
        // Domain-separated leaf hash: prefix with 0x00 byte to distinguish from
        // internal nodes (two_to_one concatenates raw hash bytes without prefix).
        let mut buffer = Vec::with_capacity(1 + input.len() * 8);
        buffer.push(0x00); // Leaf domain separator
        for f in input {
            buffer.extend_from_slice(&f.to_canonical_u64().to_le_bytes());
        }
        let mut arr = [0; N];
        let hash_bytes = keccak256(&buffer);
        arr.copy_from_slice(&hash_bytes[..N]);
        BytesHash(arr)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let mut v = vec![0; N * 2];
        v[0..N].copy_from_slice(&left.0);
        v[N..].copy_from_slice(&right.0);
        let mut arr = [0; N];
        arr.copy_from_slice(&keccak256(&v)[..N]);
        BytesHash(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::keccak256;

    /// Pins the primitive to Keccak-256 (Ethereum padding), not SHA3-256.
    #[test]
    fn keccak256_known_answers() {
        const EMPTY: [u8; 32] = [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7,
            0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04,
            0x5d, 0x85, 0xa4, 0x70,
        ];
        const ABC: [u8; 32] = [
            0x4e, 0x03, 0x65, 0x7a, 0xea, 0x45, 0xa9, 0x4f, 0xc7, 0xd4, 0x7b, 0xa8, 0x26, 0xc8,
            0xd6, 0x67, 0xc0, 0xd1, 0xe6, 0xe3, 0x3a, 0x64, 0xa0, 0x36, 0xec, 0x44, 0xf5, 0x8f,
            0xa1, 0x2d, 0x6c, 0x45,
        ];
        assert_eq!(keccak256(b""), EMPTY);
        assert_eq!(keccak256(b"abc"), ABC);
    }
}
