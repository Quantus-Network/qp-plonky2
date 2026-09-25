//! Exercises `CircuitBuilder::formal_snapshot` on the gadget mix the wormhole
//! private-batch wrapper uses (`is_equal`, `select`, `or`, `range_check`,
//! `connect`, public inputs) and checks the snapshot captures the structure
//! the Lean decode model needs. See `../formal/PLAN.md` Step 8.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::formal_snapshot::{FormalSnapshot, TargetSnapshot};

type F = GoldilocksField;
const D: usize = 2;

fn snap(t: Target) -> TargetSnapshot {
    t.into()
}

fn build_gadget_zoo() -> (FormalSnapshot, [Target; 4]) {
    let mut b = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
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

    (b.formal_snapshot(), [x, y, fee, flag.target])
}

#[test]
fn snapshot_captures_gadget_structure() {
    let (s, [x, _y, _fee, flag]) = build_gadget_zoo();

    let ids: Vec<&str> = s
        .gate_instances
        .iter()
        .map(|g| g.gate_id.as_str())
        .collect();
    // ArithmeticGate rows are keyed by their (const_0, const_1) pair, so the
    // zoo's `mul`/`sub`/`or`/`mul_sub` land on separate rows.
    assert!(
        ids.iter().any(|id| id.starts_with("ArithmeticGate")),
        "{ids:?}"
    );
    // `builder.constant` only pins a target in `constants`; the ConstantGate rows
    // are placed by `build()`, so they are absent before it.
    assert!(
        !ids.iter().any(|id| id.starts_with("ConstantGate")),
        "{ids:?}"
    );
    let base_sum: Vec<&&str> = ids
        .iter()
        .filter(|id| id.starts_with("BaseSumGate") && id.ends_with("+ Base: 2"))
        .collect();
    assert_eq!(
        base_sum.len(),
        1,
        "one binary BaseSum row per range_check: {ids:?}"
    );
    let num_limbs: usize = base_sum[0]
        .trim_start_matches("BaseSumGate { num_limbs: ")
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    // `BaseSumGate::<2>::new_from_config`: min(⌊log₂(p − 1)⌋, num_routed_wires − 1).
    assert_eq!(num_limbs, 63.min(s.num_routed_wires - 1));
    for (i, g) in s.gate_instances.iter().enumerate() {
        assert_eq!(g.row, i);
    }

    let zero = s
        .constants
        .iter()
        .find(|(_, c)| *c == 0)
        .map(|(t, _)| *t)
        .expect("zero constant is pinned");
    let to_zero = s
        .copy_constraints
        .iter()
        .filter(|(a, b)| *a == zero || *b == zero)
        .count();
    // range_check(_, 14) pins limbs 14..num_limbs to zero and is_equal pins two
    // products to zero.
    assert!(to_zero >= (num_limbs - 14) + 2, "copies to zero: {to_zero}");

    assert!(s.constants.iter().any(|(_, c)| *c == 10_000));
    assert_eq!(s.public_inputs.len(), 2);
    assert_eq!(s.public_inputs[0], snap(x));
    assert!(matches!(snap(flag), TargetSnapshot::Virtual { .. }));
    assert!(s.num_virtual_targets >= 4);

    let json = serde_json::to_string(&s).unwrap();
    let back: FormalSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.copy_constraints.len(), s.copy_constraints.len());
    assert_eq!(back.gate_instances.len(), s.gate_instances.len());
    println!(
        "gadget zoo: rows={} copies={} consts={} virtuals={} json_bytes={}",
        s.gate_instances.len(),
        s.copy_constraints.len(),
        s.constants.len(),
        s.num_virtual_targets,
        json.len()
    );
}
