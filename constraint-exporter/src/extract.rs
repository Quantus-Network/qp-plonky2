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
