//! The gadget-zoo spike circuit (`gadget.rs`): the gadget mix the wormhole private-batch
//! wrapper uses, built through the recording builder. Checks the export carries the
//! structure `Plonky2Spec/WiringGadgets.lean` reads off it, that a real witness satisfies
//! the exported system (including the `BaseSumGate<2>` semantics), and that the recorded
//! facts hold on that witness. See `../formal/PLAN.md` Step 8.

use constraint_exporter::circuit::{check_satisfied, GateKind};
use constraint_exporter::gadget::{build_gadget_zoo, generate_gadget_zoo_lean, Fact};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::PoseidonGoldilocksConfig;

type F = GoldilocksField;

#[test]
fn gadget_zoo_export_shape() {
    let config = CircuitConfig::standard_recursion_config();
    let (r, t) = build_gadget_zoo();
    let ex = r.export(vec![]).unwrap();

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
    let base_sum: Vec<usize> = ex
        .rows
        .iter()
        .filter_map(|(k, _)| match k {
            GateKind::BaseSum2 { num_limbs } => Some(*num_limbs),
            _ => None,
        })
        .collect();
    assert_eq!(
        base_sum,
        vec![63.min(config.num_routed_wires - 1)],
        "{:?}",
        ex.rows
    );
    let num_limbs = base_sum[0];

    // Limbs `14..num_limbs` are connected to the zero constant, as are `is_equal`'s two
    // product checks and `assert_bool`'s check.
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
    assert_eq!(to_zero, (num_limbs - 14) + 3, "copies to zero: {to_zero}");

    assert!(ex
        .constants
        .iter()
        .any(|(_, c)| *c == F::from_canonical_u64(10_000)));
    assert_eq!(ex.public_inputs[..2], [t.x, t.sel]);
    assert!(matches!(t.flag, Target::VirtualTarget { .. }));

    // The recorder saw one call per gadget, and attributed the `BaseSumGate<2>` row and
    // both auxiliary `is_equal` targets correctly.
    let kinds: Vec<&str> = r
        .calls
        .iter()
        .map(|c| match c.fact {
            Fact::AssertBool { .. } => "assert_bool",
            Fact::IsEqual { .. } => "is_equal",
            Fact::Select { .. } => "select",
            Fact::Or { .. } => "or",
            Fact::Not { .. } => "not",
            Fact::And { .. } => "and",
            Fact::Sub { .. } => "sub",
            Fact::RangeCheck { .. } => "range_check",
            Fact::Connect { .. } => "connect",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "assert_bool",
            "is_equal",
            "select",
            "or",
            "not",
            "and",
            "sub",
            "range_check",
            "connect"
        ]
    );
    for c in &r.calls {
        match c.fact {
            Fact::IsEqual { equal, inv, .. } => {
                assert_eq!(equal, t.eq);
                assert_ne!(inv, equal);
                assert!(!ex.constants.iter().any(|(k, _)| *k == inv));
            }
            Fact::RangeCheck { x, bits } => {
                assert_eq!((x, bits), (t.head, 14));
                assert_eq!(c.rows.len(), 1);
                assert!(matches!(ex.rows[c.rows.start].0, GateKind::BaseSum2 { .. }));
            }
            Fact::Connect { .. } => assert!(c.rows.is_empty() && c.copies.len() == 1),
            _ => {}
        }
    }
}

/// A real prover witness satisfies the exported system (exercising the `BaseSumGate<2>`
/// arm of `check_satisfied`), and each recorded fact holds on it.
#[test]
fn gadget_zoo_export_satisfied_by_real_witness() {
    let (r, t) = build_gadget_zoo();
    let ex = r.export(vec![]).unwrap();
    let data = r.builder.build::<PoseidonGoldilocksConfig>();

    let mut seed = 0x5eed_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let mut checked = 0;
    for _ in 0..50 {
        let flag = next() & 1 == 1;
        let x = F::from_canonical_u64(next() % 3);
        let y = F::from_canonical_u64(next() % 3);
        let fee = F::from_canonical_u64(next() % 10_001);
        // `connect(sel, either)` must hold, so pick `flag` consistently.
        let eq = x == y;
        let sel = if flag { x } else { y };
        let either = eq || flag;
        if sel != F::from_bool(either) {
            continue;
        }
        checked += 1;

        let mut pw = PartialWitness::new();
        pw.set_target(t.flag, F::from_bool(flag)).unwrap();
        pw.set_target(t.x, x).unwrap();
        pw.set_target(t.y, y).unwrap();
        pw.set_target(t.fee, fee).unwrap();
        let w =
            plonky2::iop::generator::generate_partial_witness(pw, &data.prover_only, &data.common)
                .unwrap();
        // Unused ops on partially filled arithmetic rows have no witness value; they are
        // unconstrained, so any value (here zero) satisfies them.
        let val = |tg: Target| w.try_get_target(tg).unwrap_or(F::ZERO);
        check_satisfied(&ex, val).unwrap();

        for c in &r.calls {
            match c.fact {
                Fact::Select { b, x, y, out } => {
                    assert_eq!(val(out), val(b) * (val(x) - val(y)) + val(y))
                }
                Fact::Not { b, out } => assert_eq!(val(out), F::ONE - val(b)),
                Fact::And { b1, b2, out } => assert_eq!(val(out), val(b1) * val(b2)),
                Fact::Or { b1, b2, out } => {
                    assert_eq!(val(out), val(b1) + val(b2) - val(b1) * val(b2))
                }
                Fact::Add { x, y, out } => assert_eq!(val(out), val(x) + val(y)),
                Fact::Sub { x, y, out } => assert_eq!(val(out), val(x) - val(y)),
                Fact::Mul { x, y, out } => assert_eq!(val(out), val(x) * val(y)),
                Fact::AssertBool { b } => assert!(val(b) == F::ZERO || val(b) == F::ONE),
                Fact::IsEqual { x, y, equal, inv } => {
                    let d = val(x) - val(y);
                    assert_eq!(val(equal) * d, F::ZERO);
                    assert_eq!(d * val(inv) - (F::ONE - val(equal)), F::ZERO);
                    assert_eq!(val(equal), F::from_bool(eq));
                }
                Fact::RangeCheck { x, bits } => assert!(val(x).to_canonical_u64() < 1 << bits),
                Fact::Connect { x, y } => assert_eq!(val(x), val(y)),
            }
        }

        // Not vacuous: flipping `eq` breaks the `is_equal` checks.
        let bad = |tg: Target| {
            if tg == t.eq {
                F::ONE - val(tg)
            } else {
                val(tg)
            }
        };
        assert!(check_satisfied(&ex, bad).is_err());
    }
    assert!(checked >= 10, "only {checked} consistent samples");
}

/// The generated Lean is what `formal/Plonky2Spec/Generated/GadgetZooCircuit.lean` holds
/// (CI also diffs the whole `Generated/` tree against a fresh export).
#[test]
fn gadget_zoo_lean_is_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../formal/Plonky2Spec/Generated/GadgetZooCircuit.lean"
    );
    let on_disk = std::fs::read_to_string(path).unwrap();
    assert_eq!(
        on_disk,
        generate_gadget_zoo_lean(),
        "run export-constraints"
    );
}
