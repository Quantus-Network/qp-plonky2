//! Driving `Gate::eval_unfiltered` with the symbolic field to extract each
//! gate's constraint polynomials.
//!
//! Each extractor builds an `EvaluationVars` whose wires/constants are *named
//! symbolic variables* (`Sym::wire(i)`, `Sym::local_const(i)`), instantiates the
//! gate at extension degree `D = 1`, and returns the constraint DAGs the gate
//! emits. These are the *real* gate constraints — the same `eval_unfiltered`
//! code the prover runs, just over `Sym` instead of `GoldilocksField`.

use plonky2::gates::arithmetic_base::ArithmeticGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::gate::Gate;
use plonky2::gates::poseidon2::formal_export::{internal_mix_base, mds_light_base, sbox7_base};
use plonky2::gates::poseidon2::{Poseidon2Gate, SPONGE_WIDTH};
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::vars::EvaluationVars;

use crate::symbolic::{reset, Sym};

/// A named, extracted gate constraint set.
pub struct Extracted {
    /// Human / Lean identifier stem for the gate.
    pub name: &'static str,
    /// Number of `local_wires` variables referenced (`w0..w{num_wires-1}`).
    pub num_wires: usize,
    /// Number of `local_constants` variables referenced (`c0..c{num_consts-1}`).
    pub num_consts: usize,
    /// One symbolic polynomial per emitted constraint (each `= 0`).
    pub constraints: Vec<Sym>,
}

/// `ArithmeticGate` with a single op: `output = c0·m0·m1 + c1·addend`.
/// Wires: `w0=m0, w1=m1, w2=addend, w3=output`; constants `c0, c1`.
pub fn arithmetic_gate() -> Extracted {
    reset();
    let gate = ArithmeticGate { num_ops: 1 };
    let consts = [Sym::local_const(0), Sym::local_const(1)];
    let wires = [Sym::wire(0), Sym::wire(1), Sym::wire(2), Sym::wire(3)];
    let pih = HashOut::<Sym>::ZERO;
    let vars = EvaluationVars {
        local_constants: &consts,
        local_wires: &wires,
        public_inputs_hash: &pih,
    };
    let constraints = <ArithmeticGate as Gate<Sym, 1>>::eval_unfiltered(&gate, vars);
    Extracted {
        name: "arithmeticGate",
        num_wires: 4,
        num_consts: 2,
        constraints,
    }
}

/// `BaseSumGate<B>` with `num_limbs` limbs. Wires: `w0 = sum`, `w1..w{num_limbs}`
/// the little-endian base-`B` limbs. Emits the reconstruction constraint plus one
/// degree-`B` range product per limb.
pub fn base_sum_gate<const B: usize>(num_limbs: usize, name: &'static str) -> Extracted {
    reset();
    let gate = BaseSumGate::<B>::new(num_limbs);
    let num_wires = 1 + num_limbs;
    let wires: Vec<Sym> = (0..num_wires).map(Sym::wire).collect();
    let consts: [Sym; 0] = [];
    let pih = HashOut::<Sym>::ZERO;
    let vars = EvaluationVars {
        local_constants: &consts,
        local_wires: &wires,
        public_inputs_hash: &pih,
    };
    let constraints = <BaseSumGate<B> as Gate<Sym, 1>>::eval_unfiltered(&gate, vars);
    Extracted {
        name,
        num_wires,
        num_consts: 0,
        constraints,
    }
}

/// `Poseidon2Gate` (width 12, `x^7` S-box, 4 initial + 22 internal + 4 terminal
/// rounds). Wires: `w0..w11` input, `w12..w23` output, then the per-round S-box
/// input checkpoints. The gate carries its own real round constants in
/// `self.params` (independent of `F`), so the extracted polynomials contain the
/// genuine Poseidon2 constants as `Const(n)` nodes. Emits 118 constraints.
pub fn poseidon2_gate() -> Extracted {
    reset();
    let gate = Poseidon2Gate::<Sym, 1>::new();
    let num_wires = <Poseidon2Gate<Sym, 1> as Gate<Sym, 1>>::num_wires(&gate);
    let wires: Vec<Sym> = (0..num_wires).map(Sym::wire).collect();
    let pih = HashOut::<Sym>::ZERO;
    let vars = EvaluationVars {
        local_constants: &[],
        local_wires: &wires,
        public_inputs_hash: &pih,
    };
    let constraints = <Poseidon2Gate<Sym, 1> as Gate<Sym, 1>>::eval_unfiltered(&gate, vars);
    Extracted {
        name: "poseidon2Gate",
        num_wires,
        num_consts: 0,
        constraints,
    }
}

// --- Poseidon2 permutation *primitives* -------------------------------------
//
// These run the *real* helper functions (`plonky2/src/gates/poseidon2.rs`) over
// the symbolic field, so each returns the exact polynomial the primitive computes.
// They are tiny (linear MDS / mix; univariate degree-7 S-box), so the rendered
// Lean is inlined and the `ring` bridge to `Plonky2Spec.Poseidon2.{mdsLight,sbox7,
// internalMix}` is cheap. Each calls `reset()`, so render its result *before* the
// next call. `Wire(i)` names the i-th argument (`w{i}`).

/// `sbox7_base(w0) = w0^7`. One output.
pub fn sbox7_prim() -> Sym {
    reset();
    sbox7_base(Sym::wire(0))
}

/// `mds_light_base` on the state `w0..w11`. 12 outputs.
pub fn mds_light_prim() -> Vec<Sym> {
    reset();
    let mut s: [Sym; SPONGE_WIDTH] = core::array::from_fn(Sym::wire);
    mds_light_base(&mut s);
    s.to_vec()
}

/// `internal_mix_base` with state `w0..w11` and diagonal `w12..w23`. 12 outputs.
pub fn internal_mix_prim() -> Vec<Sym> {
    reset();
    let x: [Sym; SPONGE_WIDTH] = core::array::from_fn(Sym::wire);
    let diag: [Sym; SPONGE_WIDTH] = core::array::from_fn(|i| Sym::wire(SPONGE_WIDTH + i));
    internal_mix_base(&x, &diag).to_vec()
}
