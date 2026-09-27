//! Builder lowerings the plain gadget zoo does not exercise (`build_gadget_edge_cases`):
//! a gadget output pinned to a constant, `is_equal` checks that constant-fold away or onto
//! the `equal` target itself, and one output wire pinned twice through op memoization.
//! Each was a generator failure (unprovable block or panic) before the checks were derived
//! from each call's own copy range.

use constraint_exporter::circuit::check_satisfied;
use constraint_exporter::gadget::{build_gadget_edge_cases, generate_gadget_edge_cases_lean, Fact};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
use plonky2::plonk::config::PoseidonGoldilocksConfig;

mod common;

type F = GoldilocksField;

#[test]
fn gadget_edge_cases_export_shape() {
    let (r, t) = build_gadget_edge_cases();
    let ex = r.export(vec![]).unwrap();
    let copies_of = |fact: &dyn Fn(&Fact) -> bool| -> Vec<(Target, Target)> {
        let c = r
            .calls
            .iter()
            .find(|c| fact(&c.fact))
            .expect("call recorded");
        ex.copies[c.copies.clone()].to_vec()
    };
    let is_eq_of =
        |equal: Target| move |f: &Fact| matches!(f, Fact::IsEqual { equal: e, .. } if *e == equal);
    let pins = |copies: &[(Target, Target)], tg: Target| {
        copies
            .iter()
            .filter(|(a, b)| (*a == tg && *b == t.zero) || (*a == t.zero && *b == tg))
            .count()
    };

    // `sum = add x y` is a wire output that `connect(sum, zero)` pins.
    assert!(matches!(t.sum, Target::Wire(_)));
    assert_eq!(pins(&ex.copies, t.sum), 1);

    // `is_equal zero zero`: `mul(equal, diff)` folds to the zero constant, so the first
    // check is a constant-to-constant copy.
    let zz = copies_of(&is_eq_of(t.eq_zz));
    assert_eq!(pins(&zz, t.zero), 1, "{zz:?}");

    // `is_equal one zero`: `diff = one`, so `mul(equal, diff)` folds to `equal` and the
    // first check pins the virtual `equal` target itself.
    let oz = copies_of(&is_eq_of(t.eq_oz));
    assert!(matches!(t.eq_oz, Target::VirtualTarget { .. }));
    assert_eq!(pins(&oz, t.eq_oz), 1, "{oz:?}");

    // `check = mul eq_xy diff` is memoized onto `is_equal`'s own `not_equal_check`, so
    // `connect(check, zero)` pins a wire `is_equal x y` already pinned.
    let xy = copies_of(&is_eq_of(t.eq_xy));
    assert_eq!(pins(&xy, t.check), 1, "{xy:?}");
    let last = copies_of(&|f| matches!(f, Fact::Connect { x, .. } if *x == t.check));
    assert_eq!(pins(&last, t.check), 1);
    assert_eq!(pins(&ex.copies, t.check), 2);

    // All of which the generator now handles.
    let lean = generate_gadget_edge_cases_lean();
    assert!(lean.contains("theorem gadgetEdgeCases_decode"));
}

#[test]
fn gadget_edge_cases_satisfied_by_real_witness() {
    let (r, t) = build_gadget_edge_cases();
    let ex = r.export(vec![]).unwrap();
    let data = r.builder.build::<PoseidonGoldilocksConfig>();

    for x in [0_u64, 1, 7, 12345] {
        let x = F::from_canonical_u64(x);
        // `connect(sum, zero)` forces `y = -x`.
        let y = -x;
        let mut pw = PartialWitness::new();
        pw.set_target(t.x, x).unwrap();
        pw.set_target(t.y, y).unwrap();
        let w =
            plonky2::iop::generator::generate_partial_witness(pw, &data.prover_only, &data.common)
                .unwrap();
        let val = |tg: Target| w.try_get_target(tg).unwrap_or(F::ZERO);
        check_satisfied(&ex, val).unwrap();
        common::assert_facts_hold(&r.calls, val);
        assert_eq!(val(t.eq_zz), F::ONE);
        assert_eq!(val(t.eq_oz), F::ZERO);
        assert_eq!(val(t.eq_xx), F::ONE);
        assert_eq!(val(t.eq_xy), F::from_bool(x == y));

        // Not vacuous: each folded or re-pinned check still constrains its `equal`.
        for flipped in [t.eq_zz, t.eq_oz, t.eq_xy] {
            let bad = |tg: Target| {
                if tg == flipped {
                    F::ONE - val(tg)
                } else {
                    val(tg)
                }
            };
            assert!(check_satisfied(&ex, bad).is_err(), "{flipped:?}");
        }
    }
}

/// The generated Lean is what `formal/Plonky2Spec/Generated/GadgetEdgeCasesCircuit.lean`
/// holds.
#[test]
fn gadget_edge_cases_lean_is_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../formal/Plonky2Spec/Generated/GadgetEdgeCasesCircuit.lean"
    );
    let on_disk = std::fs::read_to_string(path).unwrap();
    assert_eq!(
        on_disk,
        generate_gadget_edge_cases_lean(),
        "run export-constraints"
    );
}
