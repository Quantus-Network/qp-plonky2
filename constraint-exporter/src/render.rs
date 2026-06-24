//! Rendering a symbolic constraint DAG to (a) a Lean expression string and
//! (b) a concrete `GoldilocksField` value (used by the differential test).
//!
//! Two rendering strategies:
//! * **Inline tree** (`to_lean`) — fully parenthesized, no sharing. Fine for the
//!   tiny T0/T2 gates whose DAGs have no meaningful sharing.
//! * **Straight-line `let`-program** (`emit_lets`) — one `let` binding per
//!   arithmetic arena node, so each shared subexpression is written exactly once.
//!   Required for Poseidon2, whose internal rounds reuse `sum(state)` ~13×/round
//!   over 22 rounds; inlining that would be ~13²² nodes.
//!
//! The evaluator has the matching split: `eval` recurses (fine for small DAGs)
//! and `eval_all` is the memoized, linear-time version used for Poseidon2.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;

use crate::symbolic::{self, Node, Sym};

/// Render a symbolic expression as a fully-parenthesized Lean term over a field
/// `F`, naming `Wire(i)` as `w{i}` and `LocalConst(i)` as `c{i}`. Numeric
/// constants are emitted as bare numerals (they elaborate in the `ZMod p`
/// context of the generated definitions).
pub fn to_lean(s: Sym) -> String {
    match s.node() {
        Node::Const(n) => n.to_string(),
        Node::NegOne => "(-1)".to_string(),
        Node::Wire(i) => format!("w{i}"),
        Node::LocalConst(i) => format!("c{i}"),
        Node::Add(a, b) => format!("({} + {})", to_lean(a), to_lean(b)),
        Node::Sub(a, b) => format!("({} - {})", to_lean(a), to_lean(b)),
        Node::Mul(a, b) => format!("({} * {})", to_lean(a), to_lean(b)),
        Node::Neg(a) => format!("(-{})", to_lean(a)),
    }
}

/// How to *refer* to a symbolic handle inside a straight-line program: leaves
/// (constants, wires, local constants) are inlined; an arithmetic node is named
/// `n{id}` after its `let` binding.
pub fn ref_str(s: Sym) -> String {
    match s.node() {
        Node::Const(n) => n.to_string(),
        Node::NegOne => "(-1)".to_string(),
        Node::Wire(i) => format!("w{i}"),
        Node::LocalConst(i) => format!("c{i}"),
        // Add/Sub/Mul/Neg are bound to `n{id}` by `emit_lets`.
        Node::Add(_, _) | Node::Sub(_, _) | Node::Mul(_, _) | Node::Neg(_) => {
            format!("n{}", s.0)
        }
    }
}

/// Emit one `let n{id} := …` binding per arithmetic node in the current arena,
/// in topological (id) order, each indented by `indent`. Leaf nodes
/// (wires/constants) are inlined at their use sites by [`ref_str`], so they get
/// no binding. The returned block is meant to precede a body that refers to the
/// constraint roots via [`ref_str`].
pub fn emit_lets(indent: &str) -> String {
    let nodes = symbolic::arena_snapshot();
    let mut out = String::new();
    for (id, node) in nodes.iter().enumerate() {
        let rhs = match node {
            Node::Add(a, b) => format!("({} + {})", ref_str(*a), ref_str(*b)),
            Node::Sub(a, b) => format!("({} - {})", ref_str(*a), ref_str(*b)),
            Node::Mul(a, b) => format!("({} * {})", ref_str(*a), ref_str(*b)),
            Node::Neg(a) => format!("(-{})", ref_str(*a)),
            // Leaves are inlined, not bound.
            Node::Const(_) | Node::NegOne | Node::Wire(_) | Node::LocalConst(_) => continue,
        };
        out.push_str(&format!("{indent}let n{id} := {rhs}\n"));
    }
    out
}

/// Evaluate a symbolic expression at a concrete assignment, over Goldilocks.
/// `wires[i]` binds `Wire(i)`, `consts[i]` binds `LocalConst(i)`.
///
/// Recursive and unmemoized: use only on small DAGs (T0/T2). For Poseidon2 use
/// [`eval_all`] + [`eval_root`], which are linear in the arena size.
pub fn eval(s: Sym, wires: &[GoldilocksField], consts: &[GoldilocksField]) -> GoldilocksField {
    match s.node() {
        Node::Const(n) => GoldilocksField::from_canonical_u64(n),
        Node::NegOne => GoldilocksField::NEG_ONE,
        Node::Wire(i) => wires[i],
        Node::LocalConst(i) => consts[i],
        Node::Add(a, b) => eval(a, wires, consts) + eval(b, wires, consts),
        Node::Sub(a, b) => eval(a, wires, consts) - eval(b, wires, consts),
        Node::Mul(a, b) => eval(a, wires, consts) * eval(b, wires, consts),
        Node::Neg(a) => -eval(a, wires, consts),
    }
}

/// Resolve a handle's value given the per-node value table (sentinels handled
/// inline; arena handles index the already-computed `vals`).
fn resolve(s: Sym, vals: &[GoldilocksField]) -> GoldilocksField {
    match s.node() {
        Node::Const(n) => GoldilocksField::from_canonical_u64(n),
        Node::NegOne => GoldilocksField::NEG_ONE,
        // Leaves are computed into `vals` too, but sentinels are not in the arena
        // — match on the resolved node to cover both uniformly.
        _ => vals[s.0 as usize],
    }
}

/// Memoized evaluation of *every* arena node at a concrete assignment, returning
/// a table indexed by handle id. Linear in the arena size (each node is computed
/// once from its already-computed children). Pair with [`eval_root`].
pub fn eval_all(wires: &[GoldilocksField], consts: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let nodes = symbolic::arena_snapshot();
    let mut vals = vec![GoldilocksField::ZERO; nodes.len()];
    for (id, node) in nodes.iter().enumerate() {
        vals[id] = match node {
            Node::Const(n) => GoldilocksField::from_canonical_u64(*n),
            Node::NegOne => GoldilocksField::NEG_ONE,
            Node::Wire(i) => wires[*i],
            Node::LocalConst(i) => consts[*i],
            Node::Add(a, b) => resolve(*a, &vals) + resolve(*b, &vals),
            Node::Sub(a, b) => resolve(*a, &vals) - resolve(*b, &vals),
            Node::Mul(a, b) => resolve(*a, &vals) * resolve(*b, &vals),
            Node::Neg(a) => -resolve(*a, &vals),
        };
    }
    vals
}

/// The value of a constraint root, given the table from [`eval_all`].
pub fn eval_root(root: Sym, vals: &[GoldilocksField]) -> GoldilocksField {
    resolve(root, vals)
}
