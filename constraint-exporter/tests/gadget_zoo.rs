//! Exports the gadget mix the wormhole private-batch wrapper uses (`is_equal`, `select`,
//! `or`, `range_check`, `connect`, public inputs) and checks the export carries the
//! structure `Plonky2Spec/WiringGadgets.lean` reads off it. See `../formal/PLAN.md` Step 8.

use constraint_exporter::circuit::{export, GateKind};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;

type F = GoldilocksField;
const D: usize = 2;

#[test]
fn gadget_zoo_export_shape() {
    let config = CircuitConfig::standard_recursion_config();
    let mut b = CircuitBuilder::<F, D>::new(config.clone());
    let x = b.add_virtual_target();
    let y = b.add_virtual_target();
    let fee = b.add_virtual_target();
    let flag = b.add_virtual_bool_target_safe();

    let eq = b.is_equal(x, y);
    let sel = b.select(flag, x, y);
    let either = b.or(eq, flag);
    let ten_thousand = b.constant(F::from_canonical_u64(10_000));
    let head = b.sub(ten_thousand, fee);
    b.range_check(head, 14);
    b.connect(sel, either.target);
    b.register_public_input(x);
    b.register_public_input(sel);

    let ex = export(&b, vec![]).unwrap();

    // ArithmeticGate rows are keyed by their (c0, c1) pair, so the zoo's `mul`/`sub`/`or`/
    // `mul_sub` land on separate rows; ConstantGate rows only appear after `build()`.
    assert!(ex
        .rows
        .iter()
        .any(|(k, _)| matches!(k, GateKind::Arithmetic { .. })));
    assert!(!ex
        .rows
        .iter()
        .any(|(k, _)| matches!(k, GateKind::Constant { .. })));

    // `range_check(_, 14)`: one `BaseSumGate<2>` row with
    // `num_limbs = min(⌊log₂(p − 1)⌋, num_routed_wires − 1)` (split_join.rs:29).
    let base_sum: Vec<&str> = ex
        .rows
        .iter()
        .filter_map(|(k, _)| match k {
            GateKind::Other(id) if id.starts_with("BaseSumGate") && id.ends_with("+ Base: 2") => {
                Some(id.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(base_sum.len(), 1, "{:?}", ex.rows);
    let num_limbs: usize = base_sum[0]
        .trim_start_matches("BaseSumGate { num_limbs: ")
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(num_limbs, 63.min(config.num_routed_wires - 1));

    // Limbs `14..num_limbs` are connected to the zero constant, as are `is_equal`'s two
    // product checks.
    let zero = ex
        .constants
        .iter()
        .find(|(_, c)| *c == F::ZERO)
        .map(|(t, _)| *t)
        .expect("zero constant is pinned");
    let to_zero = ex
        .copies
        .iter()
        .filter(|(a, b)| *a == zero || *b == zero)
        .count();
    assert!(to_zero >= (num_limbs - 14) + 2, "copies to zero: {to_zero}");

    assert!(ex
        .constants
        .iter()
        .any(|(_, c)| *c == F::from_canonical_u64(10_000)));
    assert_eq!(ex.public_inputs, vec![x, sel]);
    assert!(matches!(flag.target, Target::VirtualTarget { .. }));
}
