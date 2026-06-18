//! Rendering a symbolic constraint DAG to (a) a Lean expression string and
//! (b) a concrete `GoldilocksField` value (used by the differential test).

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;

use crate::symbolic::{Node, Sym};

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

/// Evaluate a symbolic expression at a concrete assignment, over Goldilocks.
/// `wires[i]` binds `Wire(i)`, `consts[i]` binds `LocalConst(i)`.
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
