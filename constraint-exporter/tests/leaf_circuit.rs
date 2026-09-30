//! The recorded leaf circuit (`trace.rs`): the vendored trace parses into the export + calls
//! the decode generator expects, it matches the `wormholeSpec` lake package when that has
//! been fetched, its hash calls span the multi-block sponge and `split_le` shapes the
//! generator handles, the checked-in `Plonky2Spec/Generated/LeafCircuit.lean` is what it
//! generates, and the bridge walker (`leaf.rs`) recognises the call structure of
//! `build_leaf_constraints` and emits the checked-in `Plonky2Bridge/Generated/Leaf.lean`.
//! See `../formal/PLAN.md` Step 9.

use std::collections::BTreeMap;

use constraint_exporter::circuit::GateKind;
use constraint_exporter::gadget::{Fact, FACT_GROUP};
use constraint_exporter::leaf::{generate_leaf_bridge_lean, Shape};
use constraint_exporter::trace::{
    assert_vendored_trace_matches_pinned, generate_leaf_circuit_lean, load, traces_dir,
};
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::Field;
use plonky2::iop::target::Target;

const TRACE: &str = "leaf_circuit.json";

fn leaf() -> constraint_exporter::trace::LoadedTrace {
    load(&traces_dir().join(TRACE)).expect("vendored leaf trace")
}

#[test]
fn vendored_trace_matches_pinned_package() {
    assert_vendored_trace_matches_pinned(TRACE);
}

#[test]
fn leaf_trace_shape() {
    let t = leaf();
    assert_eq!(t.circuit, "leaf_circuit");
    let kinds = |k: &GateKind| t.ex.rows.iter().filter(|r| r.0 == *k).count();
    // Hashes: 2 (unspendable account) + 2 (nullifier) + 1 (leaf) + 16 (levels) + 1 (header).
    assert_eq!(kinds(&GateKind::Poseidon2), 61);
    assert_eq!(kinds(&GateKind::BaseSum2 { num_limbs: 63 }), 41);
    assert!(t.ex.rows.iter().all(|r| !matches!(r.0, GateKind::Other(_))));
    assert_eq!(t.ex.public_inputs.len(), 22);
    let named: BTreeMap<&str, usize> =
        t.ex.named
            .iter()
            .map(|(n, v)| (n.as_str(), v.len()))
            .collect();
    assert_eq!(named["secret"], 4);
    assert_eq!(named["transfer_count"], 2);
    assert_eq!(named["to_account"], 4);
    assert_eq!(named["account_id"], 4);
    assert_eq!(named["root_hash"], 4);
    assert_eq!(named["depth"], 1);
    assert_eq!(named["positions"], 16);
    assert_eq!(named["is_not_dummy"], 1);
    for level in 0..16 {
        assert_eq!(named[format!("siblings_{level}").as_str()], 12);
    }
    assert_eq!(named["header_digest"], 28);

    // Input lengths of the sponge calls and the rows each spans.
    let mut hashes: BTreeMap<usize, usize> = BTreeMap::new();
    for c in &t.calls {
        if let Fact::Poseidon2 { rows, inputs } = &c.fact {
            assert_eq!(rows.len(), inputs.len() / 8 + 1);
            assert!(rows.windows(2).all(|w| w[0] < w[1]));
            *hashes.entry(inputs.len()).or_default() += 1;
        }
    }
    assert_eq!(
        hashes,
        BTreeMap::from([(4, 2), (7, 1), (8, 1), (9, 1), (16, 16), (45, 1)])
    );
    let count = |f: fn(&Fact) -> bool| t.calls.iter().filter(|c| f(&c.fact)).count();
    // 7 amounts/ids/fee + the block number + 16 positions.
    assert_eq!(count(|f| matches!(f, Fact::RangeCheck { .. })), 24);
    // `is_const_less_than(level, depth, 5)` per level, and `enforce_target_less_than_const`.
    assert_eq!(count(|f| matches!(f, Fact::SplitLe { bits: 5, .. })), 17);
    assert_eq!(count(|f| matches!(f, Fact::AssertBool { .. })), 1);
    assert_eq!(t.calls.len(), 1566);
    assert!(t.calls.len() > FACT_GROUP, "decode theorem is grouped");
    let split = t
        .calls
        .iter()
        .find_map(|c| match c.fact {
            Fact::SplitLe { x, row, bits } => Some((x, row, bits)),
            _ => None,
        })
        .unwrap();
    let depth = &t.ex.named.iter().find(|(n, _)| n == "depth").unwrap().1;
    assert_eq!(split.0, depth[0]);
    assert!(matches!(t.ex.rows[split.1].0, GateKind::BaseSum2 { .. }));
    assert_eq!(split.2, 5);
    // The header hash chains six rows; its digest is the last row's output.
    let header = t
        .calls
        .iter()
        .find_map(|c| match &c.fact {
            Fact::Poseidon2 { rows, inputs } if inputs.len() == 45 => Some(rows.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(header.len(), 6);
    let digest = Target::wire(*header.last().unwrap(), 12);
    assert!(t
        .calls
        .iter()
        .any(|c| matches!(c.fact, Fact::Sub { y, .. } if y == digest)));
}

#[test]
fn leaf_lean_is_current() {
    let path = format!(
        "{}/../formal/Plonky2Spec/Generated/LeafCircuit.lean",
        env!("CARGO_MANIFEST_DIR")
    );
    let on_disk = std::fs::read_to_string(&path).unwrap();
    let generated = generate_leaf_circuit_lean().unwrap();
    assert_eq!(on_disk, generated, "run export-constraints ({path})");
    assert!(generated.contains(
        "theorem leafCircuit_decode (perm : St p → St p) (a : Assignment p) \
         (h : Satisfies (leafCircuit p) a)\n    \
         (hp : Poseidon2Rows perm (leafCircuit p) a) :"
    ));
    assert_eq!(
        generated
            .matches("have hout0 := poseidon2Row_absorb")
            .count(),
        22
    );
    assert_eq!(
        generated
            .matches("have hout5 := poseidon2Row_absorb")
            .count(),
        1
    );
    assert_eq!(generated.matches("have hr := baseSum_of_row").count(), 17);
    assert_eq!(
        generated.matches("have hr := rangeCheck_of_row").count(),
        24
    );
}

/// `Shape::read` walks the calls positionally, so it must notice when the structure it
/// assumes is not what was recorded — otherwise a changed leaf circuit would yield a bridge
/// that talks about the wrong wires (and only fails, less legibly, in Lean).
#[test]
fn shape_rejects_perturbed_traces() {
    Shape::read(&leaf()).expect("the vendored trace has the leaf's call structure");
    // The wormhole-address hash must come right after `assert_bool(is_not_dummy)`.
    let mut swapped = leaf();
    swapped.calls.swap(0, 1);
    let err = Shape::read(&swapped).unwrap_err();
    assert!(
        err.contains("call 0: expected assert_bool for is_not_dummy"),
        "{err}"
    );
    // Drop the last call: the zk-root binding is incomplete.
    let mut short = leaf();
    short.calls.pop();
    let err = Shape::read(&short).unwrap_err();
    assert!(
        err.contains("call 1565: expected connect for zk_tree_root binding, but the trace ends"),
        "{err}"
    );
    // An extra trailing call.
    let mut long = leaf();
    let last = long.calls.last().unwrap().clone();
    long.calls.push(last);
    let err = Shape::read(&long).unwrap_err();
    assert!(err.contains("1 trailing calls"), "{err}");
    // A salt constant that is not the spec's `string_to_felts("wormhole")` encoding.
    let mut salted = leaf();
    let wa_inputs = salted
        .calls
        .iter()
        .find_map(|c| match &c.fact {
            Fact::Poseidon2 { inputs, .. } if inputs.len() == 7 => Some(inputs.clone()),
            _ => None,
        })
        .unwrap();
    for (t, v) in &mut salted.ex.constants {
        if *t == wa_inputs[1] {
            *v = F::from_canonical_u64(u64::from(u32::from_le_bytes(*b"hola")));
        }
    }
    let err = Shape::read(&salted).unwrap_err();
    assert!(
        err.contains(&format!(
            "wormhole-address salt is [1836216183, {}, 1], but WormholeSpec defines it as \
             [1836216183, 1701605224, 1]",
            u32::from_le_bytes(*b"hola")
        )),
        "{err}"
    );
    // A comparator op reading the wrong bit: `is_const_less_than` is checked bit by bit.
    let mut bit = leaf();
    let k = bit
        .calls
        .iter()
        .position(|c| matches!(c.fact, Fact::And { .. }))
        .unwrap();
    if let Fact::And { b1, out, .. } = bit.calls[k].fact {
        bit.calls[k].fact = Fact::And { b1, b2: b1, out };
    }
    let err = Shape::read(&bit).unwrap_err();
    assert!(
        err.contains(&format!(
            "call {k}: expected and for is_const_less_than(16, depth) bit 4"
        )),
        "{err}"
    );
    // A Merkle level whose child select reads a sibling of the wrong level.
    let mut sib = leaf();
    let k = sib
        .calls
        .iter()
        .position(|c| matches!(c.fact, Fact::Select { .. }))
        .unwrap();
    if let Fact::Select { b, x, out, .. } = sib.calls[k].fact {
        sib.calls[k].fact = Fact::Select { b, x, y: x, out };
    }
    let err = Shape::read(&sib).unwrap_err();
    assert!(
        err.contains(&format!(
            "call {k}: expected select for level 0 child 0 limb 0"
        )),
        "{err}"
    );
    // A dummy-detection `is_equal` against something other than zero.
    let mut dummy = leaf();
    let k = dummy
        .calls
        .iter()
        .rposition(|c| matches!(c.fact, Fact::IsEqual { .. }))
        .unwrap();
    if let Fact::IsEqual { x, equal, inv, .. } = dummy.calls[k].fact {
        dummy.calls[k].fact = Fact::IsEqual {
            x,
            y: x,
            equal,
            inv,
        };
    }
    let err = Shape::read(&dummy).unwrap_err();
    assert!(
        err.contains(&format!(
            "call {k}: expected is_equal for output_amount == 0"
        )),
        "{err}"
    );
}

#[test]
fn leaf_bridge_lean_is_current() {
    let path = format!(
        "{}/../formal/Plonky2Bridge/Generated/Leaf.lean",
        env!("CARGO_MANIFEST_DIR")
    );
    let on_disk = std::fs::read_to_string(&path).unwrap();
    let generated = generate_leaf_bridge_lean().unwrap();
    assert_eq!(on_disk, generated, "run export-constraints ({path})");
    assert!(generated.contains("namespace Plonky2Bridge.LeafCircuit\n"));
    // No salt hypotheses: the spec's `wormholeSalt` / `nullifierSalt` are concrete and the
    // exporter checks the trace's constants against them.
    assert!(generated.contains(
        "theorem sound (perm : St p → St p) (hpg : goldilocks ≤ p)\n\
         \x20   (a : Assignment p) (h : Satisfies (leafCircuit p) a)\n\
         \x20   (hp : Poseidon2Rows perm (leafCircuit p) a) :\n\
         \x20   Rleaf (spongeRO perm) (pub a) (wit a) := by\n"
    ));
    // The depth bound and the sixteen `is_active` comparators, levels and walk steps.
    assert!(generated.contains("have hdloop : (ltLoop [(cb false, "));
    assert_eq!(generated.matches("have hloop").count(), 16);
    assert_eq!(
        generated.matches(" := stepUp_of_level perm hpg ").count(),
        16
    );
    assert_eq!(
        generated.matches("gatedWalk_step (spongeRO perm)").count(),
        16
    );
    assert_eq!(
        generated
            .matches("    gatedStep (spongeRO perm) hact")
            .count(),
        16
    );
    assert_eq!(generated.matches("· exact pos_lt_four hpg f").count(), 16);
    assert!(generated.contains("have hnd := notDummy_spec "));
    assert!(generated.contains("have hWA := WA_of_hashes perm hpg f1 f2\n"));
    assert!(generated.contains("have hnull := Null_of_hashes perm hpg f1527 f1528\n"));
    assert!(generated.contains("have hhdr := H_of_hash perm f1541\n"));
    // Every recorded fact but the re-emitted `or(e0, e1)`s is consumed.
    let used: std::collections::BTreeSet<usize> = generated
        .match_indices("have f")
        .map(|(i, _)| {
            generated[i + 6..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap()
        })
        .collect();
    assert_eq!(used.len(), 1566 - 16 * 3);
}
