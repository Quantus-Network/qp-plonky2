//! Post-`build` export (PLAN.md Step 16): the system a verifier key commits to, read off
//! the real `CircuitBuilder::build`, and a certificate that it refines the pre-`build`
//! export.
//!
//! A trace is the pre-`build` state of the builder that recorded it. `build` is a function
//! of that state and the config: it hashes the public inputs, adds the `PublicInputGate`,
//! the `ConstantGate`s and the padding, and resolves the copy list into `σ`. So a trace
//! whose recording builder was the deployed one — the leaf's, whose trace test runs the
//! production constructor's steps — can be *replayed* into a fresh builder here and built,
//! and the result is the deployed system. `FormalBuiltView` reads it: the rows of the
//! padded domain, `σ`, the copy-class representatives, the fixed constant wires.
//!
//! From the view, `certify` writes the `BuildCert` the Lean side consumes
//! (`Plonky2Bridge.BuildCert`): the `σ`-cycles the export's copies and constants run
//! along, the place of every virtual target, the wires pinned to the export's constants,
//! and per-copy witnesses; and (Step 17) the public-input hash `build` laid down — the
//! `PublicInputGate` row, the `PoseidonGate` rows of `hash_n_to_m_no_pad`, the wires
//! fixed to `0` — with a witness per copy into those rows. `Built.Agrees.refines` then
//! proves `Built.Refines`, and `Built.Agrees.hashesPublicInputs` proves
//! `Built.HashesPublicInputs`, from the certificate, with the checks of certificate
//! against export evaluated by the kernel. What is trusted of a certified circuit is
//! `Built.Agrees`: that the deployed build has the rows, `σ`-cycles, fixed wires and
//! placements the certificate lists. `verify_agrees` is that statement checked in Rust
//! against the built view, and the test `leaf_build_matches_deployed_digest` ties the
//! replayed build to the deployed one by `circuit_digest`.

use core::fmt::Write as _;
use std::collections::HashMap;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::fri::{FriConfig, FriReductionStrategy};
use plonky2::gates::arithmetic_base::ArithmeticGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::poseidon2::Poseidon2Gate;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::{CircuitBuilder, FormalBuiltView};
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::PoseidonGoldilocksConfig;

use crate::circuit::{classify, export, lean_target, CircuitExport, GateKind};
use crate::trace::{load, traces_dir, LoadedTrace};

type F = GoldilocksField;
const D: usize = 2;
type C = PoseidonGoldilocksConfig;

/// The deployed leaf's config, `wormhole_leaf_circuit_config()` in
/// `qp-zk-circuits/common/src/circuit.rs` over `qp_wormhole_inputs::leaf_config`. Only
/// `num_wires`, `num_routed_wires`, `num_constants`, `num_challenges` and
/// `zero_knowledge` shape the built system; the FRI parameters shape the proof. The
/// digest test checks the whole config against the deployed build.
pub fn leaf_config() -> CircuitConfig {
    CircuitConfig {
        num_wires: 135,
        num_routed_wires: 60,
        security_bits: 128,
        zero_knowledge: false,
        fri_config: FriConfig {
            rate_bits: 6,
            cap_height: 4,
            proof_of_work_bits: 16,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
            num_query_rounds: 19,
        },
        ..CircuitConfig::standard_recursion_config()
    }
}

/// The deployed leaf's `circuit_digest`: `WormholeCircuit::new(wormhole_leaf_circuit_config())
/// .build_circuit()` at the `wormholeSpec` revision `formal/lakefile.toml` pins, over the
/// published `qp-plonky2 1.6.0` that revision depends on. `leaf_build_matches_deployed_digest`
/// checks the replayed build against it.
pub const LEAF_CIRCUIT_DIGEST: [u64; 4] = [
    15101204307867406391,
    4984110629781477994,
    15964602978973519077,
    1260699525966399192,
];

/// Replay a trace into a fresh builder with `config`: its rows as gate instances with
/// their constants, its virtual targets in index order with the constant targets made by
/// `constant`, its copies and its public inputs. The result's `formal_export_view` is
/// checked to be the trace again.
pub fn replay(t: &LoadedTrace, config: CircuitConfig) -> Result<CircuitBuilder<F, D>, String> {
    let routed = t
        .num_routed_wires
        .ok_or("trace records no num_routed_wires")?;
    if routed != config.num_routed_wires {
        return Err(format!(
            "trace has num_routed_wires {routed}, config {}",
            config.num_routed_wires
        ));
    }
    let num_virt = t
        .num_virtual_targets
        .ok_or("trace records no num_virtual_targets")?;
    let mut b = CircuitBuilder::<F, D>::new(config);
    for (i, (kind, consts)) in t.ex.rows.iter().enumerate() {
        let row = match kind {
            GateKind::Arithmetic { num_ops } => {
                b.add_gate(ArithmeticGate { num_ops: *num_ops }, consts.clone())
            }
            GateKind::BaseSum2 { num_limbs } => {
                b.add_gate(BaseSumGate::<2>::new(*num_limbs), consts.clone())
            }
            GateKind::Poseidon2 => b.add_gate(Poseidon2Gate::<F, D>::new(), consts.clone()),
            other => return Err(format!("row {i}: cannot replay a {other:?} row")),
        };
        if row != i {
            return Err(format!("row {i} was placed at {row}"));
        }
    }
    let consts: HashMap<Target, F> = t.ex.constants.iter().copied().collect();
    for index in 0..num_virt {
        let want = Target::VirtualTarget { index };
        let got = match consts.get(&want) {
            Some(&c) => b.constant(c),
            None => b.add_virtual_target(),
        };
        if got != want {
            return Err(format!("virtual target {index} was allocated as {got:?}"));
        }
    }
    for &(x, y) in &t.ex.copies {
        b.connect(x, y);
    }
    b.register_public_inputs(&t.ex.public_inputs);

    let again = export(&b, Vec::new())?;
    let sorted = |cs: &[(Target, F)]| {
        let mut cs = cs.to_vec();
        cs.sort_by_key(|(t, c)| (c.to_canonical_u64(), format!("{t:?}")));
        cs
    };
    if again.rows != t.ex.rows {
        return Err("replayed rows differ from the trace".into());
    }
    if again.copies != t.ex.copies {
        return Err("replayed copies differ from the trace".into());
    }
    if sorted(&again.constants) != sorted(&t.ex.constants) {
        return Err("replayed constants differ from the trace".into());
    }
    if again.public_inputs != t.ex.public_inputs {
        return Err("replayed public inputs differ from the trace".into());
    }
    Ok(b)
}

/// `build`, with the built view.
pub fn build(
    b: CircuitBuilder<F, D>,
) -> Result<(CircuitData<F, C, D>, FormalBuiltView<F, D>), String> {
    let (data, ok, view) = b.try_build_with_formal_view::<C>(true);
    if !ok {
        return Err("build failed".into());
    }
    Ok((data, view))
}

/// The circuit digest as canonical `u64`s.
pub fn digest(data: &CircuitData<F, C, D>) -> [u64; 4] {
    let e = &data.verifier_only.circuit_digest.elements;
    [
        e[0].to_canonical_u64(),
        e[1].to_canonical_u64(),
        e[2].to_canonical_u64(),
        e[3].to_canonical_u64(),
    ]
}

/// `Plonky2Bridge.BuildCert` with the witnesses of its checks.
#[derive(Debug, Clone)]
pub struct Cert {
    pub export_rows: usize,
    pub routed: usize,
    pub num_public_inputs: usize,
    /// Cycles of `σ`, each in `σ` order: the copy classes the export's copies and constants
    /// use.
    pub cycles: Vec<Vec<(usize, usize)>>,
    /// A routed wire of each virtual target's copy class, by index.
    pub virt: Vec<Option<(usize, usize)>>,
    /// `(m, v)`: wire `v` is fixed to the export's constant number `m`.
    pub pins: Vec<(usize, (usize, usize))>,
    /// One `(ci, iu, iv)` per export copy: cycle `ci` carries its two targets at `iu` and
    /// `iv`.
    pub copy_wits: Vec<(usize, usize, usize)>,
    /// One `(ci, iu, iv)` per export constant: cycle `ci` carries the constant target at
    /// `iu` and a wire pinned to it at `iv`.
    pub const_wits: Vec<(usize, usize, usize)>,
    /// The wires `build` fixed to `0`: the `zero` target's class.
    pub zeros: Vec<(usize, usize)>,
    /// The `PublicInputGate` row.
    pub pi_row: usize,
    /// The `PoseidonGate` rows of the public-input hash, in absorb order.
    pub hash_rows: Vec<usize>,
    /// Per hash row, its witnesses (`Plonky2Bridge.GateWit`): the swap wire's `zero`, then
    /// its twelve input lanes.
    pub hash_wits: Vec<GateWit>,
    /// The four end-state lanes into the `PublicInputGate`'s wires `0..4`.
    pub out_wits: Vec<(usize, usize, usize)>,
}

/// The witnesses of one sponge gate. Each is `(ci, iu, iv)` as `BuildCert.feedOk` reads
/// it: for a wire or export target feeding a position, cycle `ci` carries the source at
/// `iu` and the position at `iv`; for `zero`, the position at `iu` and a wire fixed to
/// `0` at `iv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateWit {
    pub swap: (usize, usize, usize),
    pub lanes: Vec<(usize, usize, usize)>,
}

/// `SPONGE_WIDTH`, `SPONGE_RATE`, `NUM_HASH_OUT_ELTS` of the proof-system Poseidon.
const WIDTH: usize = 12;
const RATE: usize = 8;
const HASH_OUT: usize = 4;

/// `PoseidonGate::WIRE_SWAP`; its inputs are wires `0..12` and its outputs `12..24`.
const WIRE_SWAP: usize = 24;

/// A lane of the in-circuit sponge's state, as `Plonky2Bridge.Lane`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    Zero,
    Wire(usize, usize),
}

/// `Target::index` with the view's dimensions.
fn index(view: &FormalBuiltView<F, D>, t: Target) -> usize {
    t.index(view.num_wires, view.gate_instances.len())
}

/// The copy classes of the routed wires, by representative, each in the row-major order
/// `WirePartition` uses, which is the `σ` order.
fn classes(view: &FormalBuiltView<F, D>) -> HashMap<usize, Vec<(usize, usize)>> {
    let degree = view.gate_instances.len();
    let mut classes: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for row in 0..degree {
        for column in 0..view.num_routed_wires {
            let rep = view.representatives[index(view, Target::wire(row, column))];
            classes.entry(rep).or_default().push((row, column));
        }
    }
    classes
}

/// Where the certificate places a target: an export wire at itself, a virtual target at
/// the routed wire of its class.
fn loc(export_rows: usize, virt: &[Option<(usize, usize)>], t: Target) -> Option<(usize, usize)> {
    match t {
        Target::Wire(w) => (w.row < export_rows).then_some((w.row, w.column)),
        Target::VirtualTarget { index } => virt.get(index).copied().flatten(),
    }
}

/// Write the certificate of `view` against `ex`.
pub fn certify(ex: &CircuitExport, view: &FormalBuiltView<F, D>) -> Result<Cert, String> {
    let degree = view.gate_instances.len();
    let routed = view.num_routed_wires;
    let export_rows = ex.rows.len();
    if export_rows > degree {
        return Err(format!("export has {export_rows} rows, the build {degree}"));
    }
    for (i, (kind, consts)) in ex.rows.iter().enumerate() {
        let gi = &view.gate_instances[i];
        if classify(&gi.gate_ref.0.id()) != *kind || gi.constants != *consts {
            return Err(format!("row {i} differs after build"));
        }
    }
    if view.num_public_inputs != ex.public_inputs.len() {
        return Err(format!(
            "build has {} public inputs, the export {}",
            view.num_public_inputs,
            ex.public_inputs.len()
        ));
    }

    let classes = classes(view);
    for cls in classes.values() {
        for (n, &(row, column)) in cls.iter().enumerate() {
            if view.sigma[row * routed + column] != cls[(n + 1) % cls.len()] {
                return Err(format!(
                    "σ at ({row}, {column}) is not the next of its class"
                ));
            }
        }
    }
    let num_virt = view.representatives.len() - degree * view.num_wires;
    let rep = |t: Target| view.representatives[index(view, t)];
    let virt: Vec<Option<(usize, usize)>> = (0..num_virt)
        .map(|index| {
            classes
                .get(&rep(Target::VirtualTarget { index }))
                .map(|cls| cls[0])
        })
        .collect();

    let mut cycle_index: HashMap<usize, usize> = HashMap::new();
    let mut cycles: Vec<Vec<(usize, usize)>> = Vec::new();
    fn cycle_of(
        cycle_index: &mut HashMap<usize, usize>,
        cycles: &mut Vec<Vec<(usize, usize)>>,
        classes: &HashMap<usize, Vec<(usize, usize)>>,
        r: usize,
    ) -> usize {
        *cycle_index.entry(r).or_insert_with(|| {
            cycles.push(classes[&r].clone());
            cycles.len() - 1
        })
    }
    let position = |cyc: &[(usize, usize)], u: (usize, usize)| -> Result<usize, String> {
        cyc.iter()
            .position(|&w| w == u)
            .ok_or_else(|| format!("{u:?} is not on its cycle"))
    };

    let mut copy_wits = Vec::with_capacity(ex.copies.len());
    for (n, &(x, y)) in ex.copies.iter().enumerate() {
        let (Some(u), Some(v)) = (loc(export_rows, &virt, x), loc(export_rows, &virt, y)) else {
            return Err(format!("copy {n} {x:?} = {y:?} has an unplaced target"));
        };
        if rep(x) != rep(y) {
            return Err(format!("copy {n} {x:?} = {y:?} was not merged by build"));
        }
        let ci = cycle_of(&mut cycle_index, &mut cycles, &classes, rep(x));
        let cyc = &cycles[ci];
        copy_wits.push((ci, position(cyc, u)?, position(cyc, v)?));
    }

    let mut pins = Vec::with_capacity(ex.constants.len());
    let mut const_wits = Vec::with_capacity(ex.constants.len());
    for (m, &(t, c)) in ex.constants.iter().enumerate() {
        let u = loc(export_rows, &virt, t)
            .ok_or_else(|| format!("constant target {t:?} is unplaced"))?;
        let ci = cycle_of(&mut cycle_index, &mut cycles, &classes, rep(t));
        let cyc = &cycles[ci];
        let iu = position(cyc, u)?;
        let (iv, &v) = cyc
            .iter()
            .enumerate()
            .find(|(_, &w)| {
                view.fixed
                    .iter()
                    .any(|&(fr, fc, fv)| (fr, fc) == w && fv == c)
            })
            .ok_or_else(|| format!("no wire of constant {m}'s class is fixed to {c}"))?;
        pins.push((m, v));
        const_wits.push((ci, iu, iv));
    }

    // The public-input hash. `build` lays down `hash_n_to_m_no_pad` on the public inputs
    // right after the export's rows — one `PoseidonGate` per rate block — then the
    // `PublicInputGate`; so the rows between the export and the gate are the hash rows.
    let pi_row = (export_rows..degree)
        .find(|&i| classify(&view.gate_instances[i].gate_ref.0.id()) == GateKind::PublicInput)
        .ok_or("build has no PublicInputGate row")?;
    let hash_rows: Vec<usize> = (export_rows..pi_row).collect();
    let blocks = ex.public_inputs.len().div_ceil(RATE);
    if hash_rows.len() != blocks {
        return Err(format!(
            "{} rows between the export and the PublicInputGate, {blocks} rate blocks",
            hash_rows.len()
        ));
    }
    let zeros: Vec<(usize, usize)> = view
        .fixed
        .iter()
        .filter(|&&(_, _, c)| c == F::ZERO)
        .map(|&(r, c, _)| (r, c))
        .collect();
    let zero_rep = {
        let &(r, c) = zeros.first().ok_or("build fixes no wire to 0")?;
        rep(Target::wire(r, c))
    };
    // A witness that `src` feeds the routed wire `u`: a cycle carrying both.
    let mut feed =
        |src: Result<Target, Lane>, u: (usize, usize)| -> Result<(usize, usize, usize), String> {
            let (from, r) = match src {
                Ok(t) => (
                    loc(export_rows, &virt, t).ok_or_else(|| format!("{t:?} is unplaced"))?,
                    rep(t),
                ),
                Err(Lane::Wire(r, c)) => ((r, c), rep(Target::wire(r, c))),
                Err(Lane::Zero) => ((0, 0), zero_rep),
            };
            if r != rep(Target::wire(u.0, u.1)) {
                return Err(format!("{src:?} is not copied to {u:?}"));
            }
            let ci = cycle_of(&mut cycle_index, &mut cycles, &classes, r);
            let cyc = &cycles[ci];
            let iu = position(cyc, u)?;
            match src {
                Err(Lane::Zero) => {
                    let iv = cyc
                        .iter()
                        .position(|w| zeros.contains(w))
                        .ok_or_else(|| format!("no wire of {u:?}'s class is fixed to 0"))?;
                    Ok((ci, iu, iv))
                }
                _ => Ok((ci, position(cyc, from)?, iu)),
            }
        };
    let mut state = [Lane::Zero; WIDTH];
    let mut hash_wits = Vec::with_capacity(blocks);
    for (g, block) in hash_rows.iter().zip(ex.public_inputs.chunks(RATE)) {
        let swap = feed(Err(Lane::Zero), (*g, WIRE_SWAP))?;
        let mut lanes = Vec::with_capacity(WIDTH);
        for (i, &lane) in state.iter().enumerate() {
            let src = block.get(i).copied().ok_or(lane);
            lanes.push(feed(src, (*g, i))?);
        }
        hash_wits.push(GateWit { swap, lanes });
        state = core::array::from_fn(|i| Lane::Wire(*g, WIDTH + i));
    }
    let out_wits = (0..HASH_OUT)
        .map(|i| feed(Err(state[i]), (pi_row, i)))
        .collect::<Result<Vec<_>, _>>()?;

    let cert = Cert {
        export_rows,
        routed,
        num_public_inputs: view.num_public_inputs,
        cycles,
        virt,
        pins,
        copy_wits,
        const_wits,
        zeros,
        pi_row,
        hash_rows,
        hash_wits,
        out_wits,
    };
    verify_agrees(&cert, ex, view)?;
    Ok(cert)
}

/// `Plonky2Bridge.Built.Agrees` of the built view, against the export, checked in Rust:
/// what the Lean side trusts of a certified circuit. The export's rows are rows
/// `0..exportRows` of the build; `routed` and `numPublicInputs` are the build's; every
/// listed cycle lies in the domain and is a cycle of `σ`; every pin is a fixed wire of
/// the build with the export's constant; and every target is placed where `loc` says:
/// virtual target `n` at a routed wire of its copy class, or nowhere when the class has
/// none; every listed zero is a wire fixed to `0`; and the `PublicInputGate` row and the
/// `PoseidonGate` rows are where the certificate names them.
pub fn verify_agrees(
    cert: &Cert,
    ex: &CircuitExport,
    view: &FormalBuiltView<F, D>,
) -> Result<(), String> {
    let degree = view.gate_instances.len();
    let routed = view.num_routed_wires;
    if cert.export_rows != ex.rows.len() || cert.export_rows > degree {
        return Err("exportRows".into());
    }
    for (i, (kind, consts)) in ex.rows.iter().enumerate() {
        let gi = &view.gate_instances[i];
        if classify(&gi.gate_ref.0.id()) != *kind || gi.constants != *consts {
            return Err(format!("Agrees.rows at {i}"));
        }
    }
    if cert.routed != routed {
        return Err("Agrees.routed".into());
    }
    if cert.num_public_inputs != view.num_public_inputs {
        return Err("Agrees.numPublicInputs".into());
    }
    for (ci, cyc) in cert.cycles.iter().enumerate() {
        if cyc.is_empty() {
            return Err(format!("cycle {ci} is empty"));
        }
        for (n, &(row, column)) in cyc.iter().enumerate() {
            if row >= degree || column >= routed {
                return Err(format!("Agrees.domain: cycle {ci} leaves the domain"));
            }
            if view.sigma[row * routed + column] != cyc[(n + 1) % cyc.len()] {
                return Err(format!("Agrees.sigma: cycle {ci} at {n}"));
            }
        }
    }
    for &(m, (row, column)) in &cert.pins {
        let c = ex
            .constants
            .get(m)
            .ok_or_else(|| format!("pin names constant {m}"))?
            .1;
        if !view
            .fixed
            .iter()
            .any(|&(fr, fc, fv)| (fr, fc) == (row, column) && fv == c)
        {
            return Err(format!(
                "Agrees.fixed: ({row}, {column}) is not fixed to constant {m}"
            ));
        }
    }
    let num_virt = view.representatives.len() - degree * view.num_wires;
    if cert.virt.len() != num_virt {
        return Err("Agrees.loc: virtual target count".into());
    }
    let classes = classes(view);
    for (n, &place) in cert.virt.iter().enumerate() {
        let rep = view.representatives[index(view, Target::VirtualTarget { index: n })];
        match (place, classes.get(&rep)) {
            (None, None) => {}
            (Some((row, column)), Some(cls)) => {
                if column >= routed || !cls.contains(&(row, column)) {
                    return Err(format!("Agrees.loc: virtual target {n}"));
                }
            }
            _ => return Err(format!("Agrees.loc: virtual target {n}")),
        }
    }
    for &(row, column) in &cert.zeros {
        if !view
            .fixed
            .iter()
            .any(|&(fr, fc, fv)| (fr, fc) == (row, column) && fv == F::ZERO)
        {
            return Err(format!("Agrees.zeros: ({row}, {column}) is not fixed to 0"));
        }
    }
    let kind_at = |i: usize| -> Option<GateKind> {
        view.gate_instances
            .get(i)
            .map(|gi| classify(&gi.gate_ref.0.id()))
    };
    if kind_at(cert.pi_row) != Some(GateKind::PublicInput) {
        return Err("Agrees.piRow".into());
    }
    for &g in &cert.hash_rows {
        if kind_at(g) != Some(GateKind::Poseidon) {
            return Err(format!("Agrees.hashRows: row {g}"));
        }
    }
    Ok(())
}

// --- Lean rendering -----------------------------------------------------------------------

/// Items per chunk definition of a long list.
const CHUNK: usize = 128;

/// Render `items` as `def {name}{k} : List ({ty}) := [...]` chunks into `defs` and return
/// the balanced `++` tree over them (or the flat literal when short).
fn render_list(defs: &mut String, name: &str, ty: &str, items: &[String]) -> String {
    fn flat(items: &[String]) -> String {
        let mut s = String::from("[");
        for (i, it) in items.iter().enumerate() {
            if i % 8 == 0 {
                s.push_str("\n   ");
            }
            let _ = write!(s, " {it}{}", if i + 1 == items.len() { "" } else { "," });
        }
        s.push(']');
        s
    }
    fn tree(name: &str, lo: usize, hi: usize) -> String {
        if hi - lo == 1 {
            format!("{name}{lo}")
        } else {
            let mid = lo + (hi - lo) / 2;
            format!("({} ++ {})", tree(name, lo, mid), tree(name, mid, hi))
        }
    }
    if items.len() <= CHUNK {
        return flat(items);
    }
    let ranges: Vec<_> = (0..items.len())
        .step_by(CHUNK)
        .map(|s| s..(s + CHUNK).min(items.len()))
        .collect();
    for (k, r) in ranges.iter().enumerate() {
        let _ = writeln!(
            defs,
            "/-- `{name}`, items `{}..{}`. -/\ndef {name}{k} : List ({ty}) := {}\n",
            r.start,
            r.end,
            flat(&items[r.clone()])
        );
    }
    tree(name, 0, ranges.len())
}

fn pos(u: (usize, usize)) -> String {
    format!("({}, {})", u.0, u.1)
}

/// Positions per chunk definition of a table.
const TBL_CHUNK: usize = 256;

/// A balanced `Tbl` literal over `leaves`.
fn tbl_lit(leaves: &[String]) -> String {
    match leaves {
        [] => "Tbl.nil".into(),
        [x] => format!("(Tbl.leaf {x})"),
        _ => {
            let mid = leaves.len() / 2;
            format!(
                "(Tbl.node {mid} {} {})",
                tbl_lit(&leaves[..mid]),
                tbl_lit(&leaves[mid..])
            )
        }
    }
}

/// Render the table whose leaves are `leaves` (expression, weight) as
/// `def {name}{k} : Tbl ({ty}) := ...` chunks of about `TBL_CHUNK` weight into `defs`, and
/// return the balanced `Tbl.node` tree over them (or the literal when short).
fn render_tbl(defs: &mut String, name: &str, ty: &str, leaves: &[(String, usize)]) -> String {
    let total: usize = leaves.iter().map(|(_, w)| w).sum();
    if total <= TBL_CHUNK {
        return tbl_lit(&leaves.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>());
    }
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let (mut start, mut weight) = (0, 0);
    for (i, (_, w)) in leaves.iter().enumerate() {
        if weight > 0 && weight + w > TBL_CHUNK {
            ranges.push(start..i);
            start = i;
            weight = 0;
        }
        weight += w;
    }
    ranges.push(start..leaves.len());
    for (k, r) in ranges.iter().enumerate() {
        let _ = writeln!(
            defs,
            "/-- `{name}`, items `{}..{}`. -/
def {name}{k} : Tbl ({ty}) :=
  {}
",
            r.start,
            r.end,
            tbl_lit(
                &leaves[r.clone()]
                    .iter()
                    .map(|(e, _)| e.clone())
                    .collect::<Vec<_>>()
            )
        );
    }
    fn tree(name: &str, ranges: &[std::ops::Range<usize>], lo: usize, hi: usize) -> String {
        if hi - lo == 1 {
            format!("{name}{lo}")
        } else {
            let mid = lo + (hi - lo) / 2;
            let left: usize = ranges[lo..mid].iter().map(ExactSizeIterator::len).sum();
            format!(
                "(Tbl.node {left} {} {})",
                tree(name, ranges, lo, mid),
                tree(name, ranges, mid, hi)
            )
        }
    }
    tree(name, &ranges, 0, ranges.len())
}

/// The `++` tree the export renderer wrote for the export's copies (`render_lean` in
/// `circuit.rs`), so `copies_eq` is `rfl`.
fn export_copies_expr(name: &str, len: usize) -> String {
    match crate::circuit::chunks(len) {
        None => format!("({name} p).copies"),
        Some(ranges) => crate::circuit::chunk_tree(&format!("{name}.copies"), 0, ranges.len()),
    }
}

/// Render the certificate as `Plonky2Bridge/Generated/{Module}.lean`: the `BuildCert`,
/// its witnesses, the kernel-evaluated checks against the export, and
/// `build_refines_of_agrees`.
pub fn render_lean(
    module: &str,
    namespace: &str,
    circuit_name: &str,
    spec_module: &str,
    doc: &str,
    ex: &CircuitExport,
    cert: &Cert,
) -> String {
    let positions: usize = cert.cycles.iter().map(Vec::len).sum();
    let mut o = String::new();
    let _ = writeln!(
        o,
        "/-\n  GENERATED by `constraint-exporter` (`built.rs`) — do not edit.\n\n  {doc}\n\n  \
         `buildCert` is what the exporter read off `CircuitBuilder::build` of the replayed trace:\n  \
         the export's {} rows at their indices, {} positions on {} `σ`-cycles the export's {} copies\n  \
         and {} constants run along, over {} routed columns, the place of {} virtual targets, and\n  \
         the public-input hash: `PoseidonGate` rows {:?} absorbing the {} public inputs into the\n  \
         `PublicInputGate` at row {}.\n  \
         The checks of the certificate against the export are kernel evaluations\n  \
         (`decide +kernel`); `build_refines_of_agrees` is `Built.Agrees.refines` on them and\n  \
         `build_hashes_of_agrees` is `Built.Agrees.hashesPublicInputs`. Nothing here is assumed:\n  \
         that the deployed build `Agrees` with this certificate is `{namespace}.build_agrees` in\n  \
         `Plonky2Bridge/{module}.lean`.\n-/",
        cert.export_rows,
        positions,
        cert.cycles.len(),
        ex.copies.len(),
        ex.constants.len(),
        cert.routed,
        cert.virt.len(),
        cert.hash_rows,
        ex.public_inputs.len(),
        cert.pi_row,
    );
    let _ = writeln!(o, "import Plonky2Bridge.Trusted");
    let _ = writeln!(o, "import {spec_module}\n");
    let _ = writeln!(o, "namespace {namespace}\n");
    let _ = writeln!(o, "open Plonky2Spec.Wiring");
    let _ = writeln!(o, "open Plonky2Spec.Generated\n");
    let _ = writeln!(o, "variable {{p : ℕ}}\n");
    let _ = writeln!(o, "set_option maxRecDepth 8192\n");

    let mut defs = String::new();
    let cycles: Vec<(String, usize)> = cert
        .cycles
        .iter()
        .enumerate()
        .map(|(ci, cyc)| {
            let leaves: Vec<(String, usize)> = cyc.iter().map(|&u| (pos(u), 1)).collect();
            if cyc.len() <= TBL_CHUNK {
                (
                    tbl_lit(&leaves.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>()),
                    cyc.len(),
                )
            } else {
                let name = format!("buildCert.cycle{ci}");
                let expr = render_tbl(&mut defs, &format!("{name}_"), "ℕ × ℕ", &leaves);
                let _ = writeln!(
                    defs,
                    "/-- Cycle `{ci}` of `buildCert`, {} positions. -/
def {name} : Tbl (ℕ × ℕ) := {expr}
",
                    cyc.len()
                );
                (name, 1)
            }
        })
        .collect();
    let cycles = render_tbl(&mut defs, "buildCert.cycles", "Tbl (ℕ × ℕ)", &cycles);
    let virt: Vec<(String, usize)> = cert
        .virt
        .iter()
        .map(|v| match v {
            Some(u) => (format!("(some {})", pos(*u)), 1),
            None => ("none".into(), 1),
        })
        .collect();
    let virt = render_tbl(&mut defs, "buildCert.virt", "Option (ℕ × ℕ)", &virt);
    let pins: Vec<String> = cert
        .pins
        .iter()
        .map(|(m, v)| format!("({m}, {})", pos(*v)))
        .collect();
    let pins = render_list(&mut defs, "buildCert.pins", "ℕ × (ℕ × ℕ)", &pins);
    let copy_wits: Vec<String> = cert
        .copy_wits
        .iter()
        .map(|(ci, iu, iv)| format!("({ci}, {iu}, {iv})"))
        .collect();
    let copy_wits = render_list(&mut defs, "copyWits", "ℕ × ℕ × ℕ", &copy_wits);
    let const_wits: Vec<String> = cert
        .const_wits
        .iter()
        .map(|(ci, iu, iv)| format!("({ci}, {iu}, {iv})"))
        .collect();
    let const_wits = render_list(&mut defs, "constWits", "ℕ × ℕ × ℕ", &const_wits);
    o.push_str(&defs);

    let wit = |(ci, iu, iv): &(usize, usize, usize)| format!("({ci}, {iu}, {iv})");
    let zeros: Vec<String> = cert.zeros.iter().map(|&u| pos(u)).collect();
    let hash_rows: Vec<String> = cert.hash_rows.iter().map(usize::to_string).collect();
    let _ = writeln!(
        o,
        "/-- The certificate of the deployed build against `{circuit_name}`. -/\n\
         def buildCert : BuildCert where\n  exportRows := {}\n  routed := {}\n  numPublicInputs := {}\n  \
         cycles := {cycles}\n  virt := {virt}\n  pins := {pins}\n  zeros := [{}]\n  piRow := {}\n  \
         hashRows := [{}]\n",
        cert.export_rows,
        cert.routed,
        cert.num_public_inputs,
        zeros.join(", "),
        cert.pi_row,
        hash_rows.join(", "),
    );
    let _ = writeln!(
        o,
        "/-- One witness per copy of the export. -/\ndef copyWits : List (ℕ × ℕ × ℕ) := {copy_wits}\n"
    );
    let _ = writeln!(
        o,
        "/-- One witness per constant target of the export. -/\ndef constWits : List (ℕ × ℕ × ℕ) := {const_wits}\n"
    );
    let hash_wits: Vec<String> = cert
        .hash_wits
        .iter()
        .map(|w| {
            format!(
                "⟨{}, [{}]⟩",
                wit(&w.swap),
                w.lanes.iter().map(wit).collect::<Vec<_>>().join(", ")
            )
        })
        .collect();
    let _ = writeln!(
        o,
        "/-- The witnesses of the hash rows: the swap wire's `zero` and the twelve input lanes\n    \
         of each. -/\ndef hashWits : List GateWit :=\n  [{}]\n",
        hash_wits.join(",\n   ")
    );
    let _ = writeln!(
        o,
        "/-- The end state's four lanes into the `PublicInputGate`. -/\n\
         def outWits : List (ℕ × ℕ × ℕ) := [{}]\n",
        cert.out_wits.iter().map(wit).collect::<Vec<_>>().join(", ")
    );

    let kinds: Vec<String> = ex
        .rows
        .iter()
        .map(|(k, _)| match k {
            GateKind::Arithmetic { num_ops } => format!(".arithmetic {num_ops}"),
            GateKind::Constant { num_consts } => format!(".constant {num_consts}"),
            GateKind::PublicInput => ".publicInput".into(),
            GateKind::Noop => ".noop".into(),
            GateKind::BaseSum2 { num_limbs } => format!(".baseSum2 {num_limbs}"),
            GateKind::Poseidon2 => ".poseidon2".into(),
            GateKind::Poseidon => ".poseidon".into(),
            GateKind::Other(id) => format!(".other {id:?}"),
        })
        .collect();
    let mut kind_defs = String::new();
    let kinds = render_list(&mut kind_defs, "rowKinds", "GateKind", &kinds);
    o.push_str(&kind_defs);
    let _ = writeln!(
        o,
        "/-- The export's gate kinds, row by row. -/\ndef rowKinds : List GateKind := {kinds}\n"
    );
    let const_targets: Vec<String> = ex.constants.iter().map(|(t, _)| lean_target(*t)).collect();
    let _ = writeln!(
        o,
        "/-- The export's constant targets, in its order. -/\ndef constTargets : List Target := [{}]\n",
        const_targets.join(", ")
    );
    let pi_targets: Vec<String> = ex.public_inputs.iter().map(|&t| lean_target(t)).collect();
    let _ = writeln!(
        o,
        "/-- The export's public-input targets, in its order. -/\ndef piTargets : List Target := [{}]\n",
        pi_targets.join(", ")
    );
    let copies_expr = export_copies_expr(circuit_name, ex.copies.len());

    let _ = writeln!(
        o,
        "theorem rowKinds_eq : ({circuit_name} p).rows.map Row.kind = rowKinds := rfl\n"
    );
    let _ = writeln!(
        o,
        "theorem rows_length : ({circuit_name} p).rows.length = {} := rfl\n",
        ex.rows.len()
    );
    let _ = writeln!(
        o,
        "theorem copies_eq : ({circuit_name} p).copies = {copies_expr} := rfl\n"
    );
    let _ = writeln!(
        o,
        "theorem constants_fst : ({circuit_name} p).constants.map Prod.fst = constTargets := rfl\n"
    );
    let _ = writeln!(
        o,
        "theorem publicInputs_length : ({circuit_name} p).publicInputs.length = {} := rfl\n",
        ex.public_inputs.len()
    );
    let _ = writeln!(
        o,
        "theorem publicInputs_eq : ({circuit_name} p).publicInputs = piTargets := rfl\n"
    );
    let _ = writeln!(
        o,
        "/-- The export has no `ConstantGate` row. -/\n\
         theorem checks_rows : rowKinds.all noConstantRow = true := by decide +kernel\n"
    );
    let _ = writeln!(
        o,
        "/-- Every copy of the export is linked in the certificate. -/\n\
         theorem checks_copies : buildCert.copiesOk {copies_expr} copyWits = true := by\n  decide +kernel\n"
    );
    let _ = writeln!(
        o,
        "/-- Every constant target of the export is pinned in the certificate. -/\n\
         theorem checks_constants : buildCert.constsOk 0 constTargets constWits = true := by\n  decide +kernel\n"
    );
    let _ = writeln!(
        o,
        "/-- **A build the certificate describes refines the export.** `Built.Agrees.refines`\n    \
         on the checks above. -/\n\
         theorem build_refines_of_agrees (b : Built p) (h : b.Agrees ({circuit_name} p) buildCert) :\n    \
         b.Refines ({circuit_name} p) :=\n  \
         h.refines (le_of_eq rows_length) (by rw [rowKinds_eq]; exact checks_rows)\n    \
         (by rw [copies_eq]; exact checks_copies) (by rw [constants_fst]; exact checks_constants)\n    \
         publicInputs_length.symm\n"
    );
    let _ = writeln!(
        o,
        "/-- The public-input hash is wired in the certificate: the sponge over the export's\n    \
         public inputs through the hash rows into the `PublicInputGate`. -/\n\
         theorem checks_pi : buildCert.piOk piTargets buildCert.hashRows hashWits outWits = true := by\n  \
         decide +kernel\n"
    );
    let _ = writeln!(
        o,
        "/-- **A build the certificate describes hashes the export's public inputs.**\n    \
         `Built.Agrees.hashesPublicInputs` on `checks_pi`. -/\n\
         theorem build_hashes_of_agrees (b : Built p) (h : b.Agrees ({circuit_name} p) buildCert) :\n    \
         b.HashesPublicInputs ({circuit_name} p) :=\n  \
         h.hashesPublicInputs (by rw [publicInputs_eq]; exact checks_pi)\n"
    );
    let _ = writeln!(o, "end {namespace}");
    o
}

/// A certified build: the trace, what `build` made of its replay, the built view and the
/// certificate.
pub type Certified = (
    LoadedTrace,
    CircuitData<F, C, D>,
    FormalBuiltView<F, D>,
    Cert,
);

/// The leaf's certificate from its vendored trace: replayed with `leaf_config`, built, read.
pub fn leaf_cert() -> Result<Certified, String> {
    let t = load(&traces_dir().join("leaf_circuit.json"))?;
    if t.circuit != "leaf_circuit" {
        return Err(format!("trace is for {:?}", t.circuit));
    }
    let b = replay(&t, leaf_config())?;
    let (data, view) = build(b)?;
    let cert = certify(&t.ex, &view)?;
    Ok((t, data, view, cert))
}

/// `Plonky2Bridge/Generated/LeafBuilt.lean`.
pub fn generate_leaf_built_lean() -> Result<String, String> {
    let (t, _, _, cert) = leaf_cert()?;
    Ok(render_lean(
        "LeafBuild",
        "Plonky2Bridge.LeafCircuit",
        "leafCircuit",
        "Plonky2Spec.Generated.LeafCircuit",
        "The certified build of the wormhole leaf circuit: `leaf_circuit.json` replayed into\n  \
         `CircuitBuilder` with `wormhole_leaf_circuit_config()` and built, as the production\n  \
         constructor builds it (`WormholeCircuit::new_internal`: `CircuitTargets::new`, then\n  \
         `build_leaf_constraints`, the steps the trace test runs).",
        &t.ex,
        &cert,
    ))
}
