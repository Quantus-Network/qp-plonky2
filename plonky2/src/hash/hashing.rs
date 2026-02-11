//! Concrete instantiation of a hash function.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOutTarget, RichField, NUM_HASH_OUT_ELTS};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::AlgebraicHasher;

// Re-export core hashing types and functions
pub use qp_plonky2_core::hashing::{
    compress, hash_n_to_hash_no_pad, hash_n_to_hash_no_pad_p2, hash_n_to_m_no_pad,
    PlonkyPermutation,
};

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub fn hash_or_noop<H: AlgebraicHasher<F>>(&mut self, inputs: Vec<Target>) -> HashOutTarget {
        let zero = self.zero();
        if inputs.len() <= NUM_HASH_OUT_ELTS {
            HashOutTarget::from_partial(&inputs, zero)
        } else {
            self.hash_n_to_hash_no_pad::<H>(inputs)
        }
    }

    pub fn hash_n_to_hash_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
    ) -> HashOutTarget {
        HashOutTarget::from_vec(self.hash_n_to_m_no_pad::<H>(inputs, NUM_HASH_OUT_ELTS))
    }

    pub fn hash_n_to_m_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
        num_outputs: usize,
    ) -> Vec<Target> {
        let zero = self.zero();
        let mut state = H::AlgebraicPermutation::new(core::iter::repeat(zero));

        // Absorb all input chunks.
        for input_chunk in inputs.chunks(H::AlgebraicPermutation::RATE) {
            // Overwrite the first r elements with the inputs. This differs from a standard sponge,
            // where we would xor or add in the inputs. This is a well-known variant, though,
            // sometimes called "overwrite mode".
            state.set_from_slice(input_chunk, 0);
            state = self.permute::<H>(state);
        }

        // Squeeze until we have the desired number of outputs.
        let mut outputs = Vec::with_capacity(num_outputs);
        loop {
            for &s in state.squeeze() {
                outputs.push(s);
                if outputs.len() == num_outputs {
                    return outputs;
                }
            }
            state = self.permute::<H>(state);
        }
    }

    /// Poseidon2 variable length hashing (identical to CPU `hash_n_to_hash_no_pad_p2`).
    pub fn hash_n_to_hash_no_pad_p2<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
    ) -> HashOutTarget {
        const RATE: usize = 4; // must match your gate
        let zero = self.zero();
        let one = self.one();

        // Start from all-zero state.
        let mut st = H::AlgebraicPermutation::new(core::iter::repeat(zero));

        if RATE == 0 {
            // Defensive
            return HashOutTarget::from_vec(vec![zero; NUM_HASH_OUT_ELTS]);
        }

        // Absorb input in RATE-sized chunks with additive absorption.
        let mut idx = 0usize;
        while idx < inputs.len() {
            let remaining = inputs.len() - idx;
            let take = remaining.min(RATE);

            // Build one block of length RATE.
            let mut blk = vec![zero; RATE];
            for i in 0..take {
                blk[i] = inputs[idx + i];
            }

            // If this is the final (possibly partial) block and it's not full,
            // append the single '1' delimiter then zero-fill the rest.
            if idx + take == inputs.len() && take < RATE {
                blk[take] = one;
            }

            // Additive absorption then permute.
            for i in 0..RATE {
                let sum = self.add(st.as_ref()[i], blk[i]);
                st.set_elt(sum, i);
            }
            st = self.permute::<H>(st);

            idx += take;
        }

        // If inputs were an exact multiple of RATE (including empty),
        // absorb one full padding block [1, 0, 0, 0].
        if inputs.len() % RATE == 0 {
            let mut blk = vec![zero; RATE];
            blk[0] = one;
            for i in 0..RATE {
                let sum = self.add(st.as_ref()[i], blk[i]);
                st.set_elt(sum, i);
            }
            st = self.permute::<H>(st);
        }

        // Squeeze NUM_HASH_OUT_ELTS elements from the current state (no extra permute).
        let outs = st.squeeze()[..NUM_HASH_OUT_ELTS].to_vec();
        HashOutTarget::from_vec(outs)
    }
}
