//! Post-`build` export (PLAN.md Steps 16, 17 and 19): the system a verifier key commits
//! to, read off the real `CircuitBuilder::build`, and a certificate that it refines the
//! pre-`build` export.
//!
//! A trace is the pre-`build` state of the production builder: for a wrapper, the state
//! after `add_recursive_verifiers`, of which the trace records the wrapper's own part —
//! the rows and copies its gadgets added, the rows whose free slots they used, the
//! constants and public inputs — plus the builder's counts when tracing began and a
//! digest of the whole state (`state_digest`, the same function in
//! `qp-zk-circuits/common/src/formal_trace.rs`). `replay` runs the recursive verifiers
//! here over the child circuits built the same way, lays the trace's part down on top,
//! and checks the digest: the result is the production builder. `build` is a function of
//! that state and the config, so building it gives the deployed system, and
//! `FormalBuiltView` reads it: the rows of the padded domain, `σ`, the copy-class
//! representatives, the fixed constant wires.
//!
//! `complete` then extends the trace's export to the whole builder (PLAN.md Step 21):
//! the verifiers' rows, copies, constants and virtual targets are appended after the
//! wrapper's own, so the export a certified circuit's Lean modules are generated from
//! holds its `verify_proof` gadgets, and `recursive_verifier_transcript` is stated on
//! their rows.
//!
//! From the view, `certify` writes the `BuildCert` the Lean side consumes
//! (`Plonky2Bridge.BuildCert`): where each export row sits in the build, the `σ`-cycles
//! the export's copies and constants run along, the place of every virtual target, the
//! wires pinned to the export's constants, per-copy witnesses, and the public-input hash
//! `build` laid down — the `PublicInputGate` row, the `PoseidonGate` rows of
//! `hash_n_to_m_no_pad`, the wires fixed to `0` — with a witness per copy into those
//! rows. `Built.Agrees.refines` then proves `Built.Refines`, and
//! `Built.Agrees.hashesPublicInputs` proves `Built.HashesPublicInputs`, from the
//! certificate, with the checks of certificate against export evaluated by the kernel.
//! What is trusted of a certified circuit is `Built.Agrees`: that the deployed build has
//! the rows, `σ`-cycles, fixed wires and placements the certificate lists, and that the
//! export is the whole pre-`build` system (`Built.Complete`). `verify_agrees` is that
//! statement checked in Rust against the built view.

use core::fmt::Write as _;
use std::collections::{BTreeMap, HashMap, HashSet};

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::gates::arithmetic_base::ArithmeticGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::poseidon2::Poseidon2Gate;
use plonky2::hash::hash_types::BytesHash;
use plonky2::hash::keccak::KeccakHash;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::{CircuitBuilder, FormalBuiltView};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::config::{Hasher, PoseidonGoldilocksConfig};

use crate::circuit::{classify, lean_target, GateKind};
use crate::trace::{load, traces_dir, Counts, LoadedTrace};

type F = GoldilocksField;
const D: usize = 2;
type C = PoseidonGoldilocksConfig;

/// Keccak-256, hex, of the builder's pre-`build` constraint system: every gate row with
/// its constants, the copy constraints, constant targets, public inputs and the
/// virtual-target count. The same function as `formal_trace::state_digest` in
/// qp-zk-circuits, whose trace tests check it against the production constructor's
/// builder; two builders with one digest build one circuit.
pub fn state_digest(b: &CircuitBuilder<F, D>) -> String {
    let view = b.formal_export_view();
    assert!(
        view.lookups.iter().all(Vec::is_empty),
        "lookups are not modeled"
    );
    let felt = |n: usize| F::from_canonical_u64(n as u64);
    let target = |t: &Target| match *t {
        Target::Wire(w) => [felt(0), felt(w.row), felt(w.column)],
        Target::VirtualTarget { index } => [felt(1), felt(index), felt(0)],
    };
    let mut input = Vec::new();
    input.push(felt(view.gate_instances.len()));
    for gi in view.gate_instances {
        let id = gi.gate_ref.0.id();
        input.push(felt(id.len()));
        input.extend(id.bytes().map(|b| felt(b as usize)));
        input.push(felt(gi.constants.len()));
        input.extend(&gi.constants);
    }
    input.push(felt(view.copy_constraints.len()));
    for c in view.copy_constraints {
        input.extend(target(&c.pair.0));
        input.extend(target(&c.pair.1));
    }
    let constants: BTreeMap<String, (&Target, &F)> = view
        .constant_targets
        .iter()
        .map(|(t, v)| (crate::trace::encode_target(*t), (t, v)))
        .collect();
    input.push(felt(constants.len()));
    for (t, v) in constants.values() {
        input.extend(target(t));
        input.push(**v);
    }
    input.push(felt(view.public_inputs.len()));
    for t in view.public_inputs {
        input.extend(target(t));
    }
    input.push(felt(view.num_virtual_targets));
    let BytesHash(bytes) = KeccakHash::<32>::hash_no_pad(&input);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn counts(b: &CircuitBuilder<F, D>) -> Counts {
    let view = b.formal_export_view();
    Counts {
        rows: view.gate_instances.len(),
        copies: view.copy_constraints.len(),
        virtual_targets: view.num_virtual_targets,
    }
}

/// Replay a trace into a fresh builder with its config: first the recursive verifiers,
/// as `add_recursive_verifiers` lays them down — the child's verifier key as constants,
/// then a virtual proof and `verify_proof` per recorded verifier — on the built `child`
/// the trace's verifiers name; then the trace's rows at their recorded indices, its
/// virtual targets in index order with the constant targets made by `constant`, its
/// copies and its public inputs. The result's `state_digest` must be the trace's.
pub fn replay(
    t: &LoadedTrace,
    child: Option<(&str, &CircuitData<F, C, D>)>,
) -> Result<CircuitBuilder<F, D>, String> {
    let mut b = CircuitBuilder::<F, D>::new(t.config.clone());
    match (child, t.ex.verifiers.is_empty()) {
        (None, true) => {}
        (None, false) => return Err("trace has verifiers but no child circuit was given".into()),
        (Some(_), true) => return Err("trace has no verifiers for the child circuit given".into()),
        (Some((name, data)), false) => {
            let key = b.constant_verifier_data::<C>(&data.verifier_only);
            for (i, (tag, pis)) in t.ex.verifiers.iter().enumerate() {
                if tag != name {
                    return Err(format!(
                        "verifier {i} is for {tag:?}, child given is {name:?}"
                    ));
                }
                let proof = b.add_virtual_proof_with_pis(&data.common);
                b.verify_proof::<C>(&proof, &key, &data.common);
                let recorded: Vec<Target> = pis.iter().map(|&t_| t.production(t_)).collect();
                if proof.public_inputs != recorded {
                    return Err(format!("verifier {i}'s public inputs are not the trace's"));
                }
            }
        }
    }
    let start = counts(&b);
    if start != t.start {
        return Err(format!(
            "the verifiers left {start:?}, the trace began at {:?}",
            t.start
        ));
    }
    for (i, (kind, consts)) in t.ex.rows.iter().enumerate() {
        let at = t.row_at[i];
        if at < t.start.rows {
            let gi = &b.formal_export_view().gate_instances[at];
            if classify(&gi.gate_ref.0.id()) != *kind || gi.constants != *consts {
                return Err(format!("row {at} of the verifiers is not export row {i}"));
            }
            continue;
        }
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
        if row != at {
            return Err(format!(
                "export row {i} was placed at {row}, the trace has {at}"
            ));
        }
    }
    let consts: HashMap<Target, F> =
        t.ex.constants
            .iter()
            .map(|&(t_, c)| (t.production(t_), c))
            .collect();
    for index in t.start.virtual_targets..t.num_virtual_targets {
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
        b.connect(t.production(x), t.production(y));
    }
    let pis: Vec<Target> =
        t.ex.public_inputs
            .iter()
            .map(|&t_| t.production(t_))
            .collect();
    b.register_public_inputs(&pis);
    let digest = state_digest(&b);
    if digest != t.digest {
        return Err(format!(
            "the replayed builder's state digest is {digest}, the trace's {}",
            t.digest
        ));
    }
    Ok(b)
}

/// `build`, with the built view. The certificate reads only the view, so the commitment
/// to `σ` — most of the memory of a large build — is optional; the circuit digest is only
/// meaningful with it.
pub fn build(
    b: CircuitBuilder<F, D>,
    commit_to_sigma: bool,
) -> Result<(CircuitData<F, C, D>, FormalBuiltView<F, D>), String> {
    let (data, ok, view) = b.try_build_with_formal_view::<C>(commit_to_sigma);
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

/// What `complete` reads of the builder before `build` consumes it: its row count, its
/// copies in order, and its constant targets.
pub struct Prebuild {
    pub rows: usize,
    pub copies: Vec<(Target, Target)>,
    pub constants: Vec<(Target, F)>,
}

impl Prebuild {
    pub fn of(b: &CircuitBuilder<F, D>) -> Self {
        let view = b.formal_export_view();
        let mut constants: Vec<(Target, F)> = view
            .constant_targets
            .iter()
            .map(|(&t, &c)| (t, c))
            .collect();
        constants.sort_by_key(|&(t, _)| crate::trace::encode_target(t));
        Prebuild {
            rows: view.gate_instances.len(),
            copies: view.copy_constraints.iter().map(|c| c.pair).collect(),
            constants,
        }
    }
}

/// The trace's export completed to the whole builder it was recorded in (PLAN.md Step
/// 21). The trace lists the wrapper's own part; the builder's other rows, copies,
/// constants and virtual targets — the recursive verifiers' — are appended after it in
/// builder order, so the trace's numbering, which its decode theorems use, is a prefix
/// of the export's. Rows carry the constants `build` leaves them: a gate's spare
/// constant wires (`Gate::extra_constant_wires`, the `RandomAccessGate`'s) get their
/// constants, and the wires fixed to them, at `build`.
pub fn complete(
    t: LoadedTrace,
    pre: &Prebuild,
    view: &FormalBuiltView<F, D>,
) -> Result<LoadedTrace, String> {
    if pre.copies.len() != t.start.copies + t.ex.copies.len() {
        return Err(format!(
            "the builder has {} copies, the verifiers' {} and the trace's {}",
            pre.copies.len(),
            t.start.copies,
            t.ex.copies.len()
        ));
    }
    let traced: HashSet<Target> =
        t.ex.constants
            .iter()
            .map(|&(t_, _)| t.production(t_))
            .collect();
    let LoadedTrace {
        circuit,
        mut ex,
        calls,
        config,
        digest,
        start,
        num_virtual_targets,
        mut row_at,
        mut virt_at,
    } = t;
    let listed: HashSet<usize> = row_at.iter().copied().collect();
    row_at.extend((0..pre.rows).filter(|r| !listed.contains(r)));
    let listed: HashSet<usize> = virt_at.iter().copied().collect();
    virt_at.extend((0..num_virtual_targets).filter(|n| !listed.contains(n)));
    let row_local: HashMap<usize, usize> =
        row_at.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    let virt_local: HashMap<usize, usize> =
        virt_at.iter().enumerate().map(|(n, &v)| (v, n)).collect();
    let local = |t_: Target| match t_ {
        Target::Wire(w) => Target::wire(row_local[&w.row], w.column),
        Target::VirtualTarget { index } => Target::VirtualTarget {
            index: virt_local[&index],
        },
    };
    ex.rows.extend(row_at[ex.rows.len()..].iter().map(|&r| {
        let gi = &view.gate_instances[r];
        (classify(&gi.gate_ref.0.id()), gi.constants.clone())
    }));
    ex.copies.extend(
        pre.copies[..start.copies]
            .iter()
            .map(|&(x, y)| (local(x), local(y))),
    );
    ex.constants.extend(
        pre.constants
            .iter()
            .filter(|(t_, _)| !traced.contains(t_))
            .map(|&(t_, c)| (local(t_), c)),
    );
    Ok(LoadedTrace {
        circuit,
        ex,
        calls,
        config,
        digest,
        start,
        num_virtual_targets,
        row_at,
        virt_at,
    })
}

/// `Plonky2Bridge.BuildCert` with the witnesses of its checks.
#[derive(Debug, Clone)]
pub struct Cert {
    /// Export row `i` is row `row_at[i]` of the build.
    pub row_at: Vec<usize>,
    pub routed: usize,
    pub num_public_inputs: usize,
    /// Cycles of `σ`, each in `σ` order: the copy classes the export's copies and constants
    /// use.
    pub cycles: Vec<Vec<(usize, usize)>>,
    /// A routed wire of each export virtual target's copy class, by index.
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

/// Where the certificate places an export target: a wire target at its row's place in
/// the build, a virtual target at the routed wire of its class.
fn loc(row_at: &[usize], virt: &[Option<(usize, usize)>], t: Target) -> Option<(usize, usize)> {
    match t {
        Target::Wire(w) => row_at.get(w.row).map(|&r| (r, w.column)),
        Target::VirtualTarget { index } => virt.get(index).copied().flatten(),
    }
}

/// Write the certificate of `view`, the build of `t` replayed, against `t`'s export.
pub fn certify(t: &LoadedTrace, view: &FormalBuiltView<F, D>) -> Result<Cert, String> {
    let ex = &t.ex;
    let degree = view.gate_instances.len();
    let routed = view.num_routed_wires;
    let row_at = t.row_at.clone();
    for (i, (kind, consts)) in ex.rows.iter().enumerate() {
        let gi = view
            .gate_instances
            .get(row_at[i])
            .ok_or_else(|| format!("export row {i} is past the build"))?;
        if classify(&gi.gate_ref.0.id()) != *kind || gi.constants != *consts {
            return Err(format!("export row {i} differs after build"));
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
    let rep = |t_: Target| view.representatives[index(view, t.production(t_))];
    let virt: Vec<Option<(usize, usize)>> = (0..t.virt_at.len())
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
        let (Some(u), Some(v)) = (loc(&row_at, &virt, x), loc(&row_at, &virt, y)) else {
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
    for (m, &(t_, c)) in ex.constants.iter().enumerate() {
        let u =
            loc(&row_at, &virt, t_).ok_or_else(|| format!("constant target {t_:?} is unplaced"))?;
        let ci = cycle_of(&mut cycle_index, &mut cycles, &classes, rep(t_));
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
    // right after the builder's rows — one `PoseidonGate` per rate block — then the
    // `PublicInputGate`; so the rows just before the gate are the hash rows.
    let pi_row = (0..degree)
        .find(|&i| classify(&view.gate_instances[i].gate_ref.0.id()) == GateKind::PublicInput)
        .ok_or("build has no PublicInputGate row")?;
    let blocks = ex.public_inputs.len().div_ceil(RATE);
    if pi_row < blocks {
        return Err(format!(
            "the PublicInputGate is at row {pi_row}, before {blocks} hash rows"
        ));
    }
    let hash_rows: Vec<usize> = (pi_row - blocks..pi_row).collect();
    let zeros: Vec<(usize, usize)> = view
        .fixed
        .iter()
        .filter(|&&(_, _, c)| c == F::ZERO)
        .map(|&(r, c, _)| (r, c))
        .collect();
    let zero_rep = {
        let &(r, c) = zeros.first().ok_or("build fixes no wire to 0")?;
        view.representatives[index(view, Target::wire(r, c))]
    };
    let wire_rep = |r: usize, c: usize| view.representatives[index(view, Target::wire(r, c))];
    // A witness that `src` feeds the routed wire `u`: a cycle carrying both.
    let mut feed =
        |src: Result<Target, Lane>, u: (usize, usize)| -> Result<(usize, usize, usize), String> {
            let (from, r) = match src {
                Ok(t_) => (
                    loc(&row_at, &virt, t_).ok_or_else(|| format!("{t_:?} is unplaced"))?,
                    rep(t_),
                ),
                Err(Lane::Wire(r, c)) => ((r, c), wire_rep(r, c)),
                Err(Lane::Zero) => ((0, 0), zero_rep),
            };
            if r != wire_rep(u.0, u.1) {
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
        row_at,
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
    verify_agrees(&cert, t, view)?;
    Ok(cert)
}

/// `Plonky2Bridge.Built.Agrees` of the built view, against the export, checked in Rust:
/// what the Lean side trusts of a certified circuit. Export row `i` is row `rowAt i` of
/// the build; `routed` and `numPublicInputs` are the build's; every listed cycle lies in
/// the domain and is a cycle of `σ`; every pin is a fixed wire of the build with the
/// export's constant; every target is placed where `loc` says: virtual target `n` at a
/// routed wire of its copy class, or nowhere when the class has none; every listed zero
/// is a wire fixed to `0`; the `PublicInputGate` row and the `PoseidonGate` rows are
/// where the certificate names them; and the export is the whole pre-`build` system
/// (`Built.Complete`): every build row before the export's length hosts an export row,
/// export targets on one `σ`-cycle are joined by the export's copies, and every fixed
/// wire is on the cycle of an export constant target of its constant.
pub fn verify_agrees(
    cert: &Cert,
    t: &LoadedTrace,
    view: &FormalBuiltView<F, D>,
) -> Result<(), String> {
    let ex = &t.ex;
    let degree = view.gate_instances.len();
    let routed = view.num_routed_wires;
    if cert.row_at.len() != ex.rows.len() {
        return Err("rowAt".into());
    }
    for (i, (kind, consts)) in ex.rows.iter().enumerate() {
        let gi = view
            .gate_instances
            .get(cert.row_at[i])
            .ok_or_else(|| format!("Agrees.rows at {i}"))?;
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
    if cert.virt.len() != t.virt_at.len() {
        return Err("Agrees.loc: virtual target count".into());
    }
    let classes = classes(view);
    for (n, &place) in cert.virt.iter().enumerate() {
        let rep =
            view.representatives[index(view, t.production(Target::VirtualTarget { index: n }))];
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

    let mut hosted = vec![false; ex.rows.len()];
    for &r in &cert.row_at {
        if let Some(h) = hosted.get_mut(r) {
            *h = true;
        }
    }
    if let Some(r) = hosted.iter().position(|h| !h) {
        return Err(format!("Complete.rows: build row {r} hosts no export row"));
    }
    for i in ex.rows.len()..degree {
        if !matches!(
            kind_at(i),
            Some(
                GateKind::Poseidon
                    | GateKind::PublicInput
                    | GateKind::Constant { .. }
                    | GateKind::Noop
            )
        ) {
            return Err(format!(
                "Complete.tail: build row {i} is not one `build` adds"
            ));
        }
    }
    // Export targets placed on one `σ`-cycle must be joined by the export's copies, or
    // by a wire `build` fixed to the constant target of its value: a union-find over the
    // export's targets, the wires of the routed columns and the virtual targets, against
    // the build's classes.
    let wires = ex.rows.len() * routed;
    let slot = |t_: Target| match t_ {
        Target::Wire(w) => (w.column < routed).then(|| w.row * routed + w.column),
        Target::VirtualTarget { index } => cert.virt[index].map(|_| wires + index),
    };
    let mut parent: Vec<usize> = (0..wires + cert.virt.len()).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut join = |x: Target, y: Target| {
        if let (Some(a), Some(b)) = (slot(x), slot(y)) {
            let (a, b) = (root(&mut parent, a), root(&mut parent, b));
            parent[a] = b;
        }
    };
    for &(x, y) in &ex.copies {
        join(x, y);
    }
    let export_row: HashMap<usize, usize> = cert
        .row_at
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i))
        .collect();
    let constant_of: HashMap<u64, Target> = ex
        .constants
        .iter()
        .map(|&(t_, c)| (c.to_canonical_u64(), t_))
        .collect();
    for &(r, c, v) in &view.fixed {
        if let (Some(&i), Some(&t_)) = (export_row.get(&r), constant_of.get(&v.to_canonical_u64()))
        {
            join(Target::wire(i, c), t_);
        }
    }
    let rep = |t_: Target| view.representatives[index(view, t.production(t_))];
    let mut class_root: HashMap<usize, usize> = HashMap::new();
    let targets = (0..ex.rows.len())
        .flat_map(|i| (0..routed).map(move |j| Target::wire(i, j)))
        .chain((0..cert.virt.len()).map(|n| Target::VirtualTarget { index: n }));
    for t_ in targets {
        let Some(s) = slot(t_) else { continue };
        let r = root(&mut parent, s);
        if *class_root.entry(rep(t_)).or_insert(r) != r {
            return Err(format!(
                "Complete.copies: {t_:?} is on a cycle its copies do not join"
            ));
        }
    }
    let mut pinned: HashMap<usize, Vec<F>> = HashMap::new();
    for &(t_, c) in &ex.constants {
        pinned.entry(rep(t_)).or_default().push(c);
    }
    for &(r, c, v) in &view.fixed {
        let class = view.representatives[index(view, Target::wire(r, c))];
        if !pinned.get(&class).is_some_and(|cs| cs.contains(&v)) {
            return Err(format!(
                "Complete.fixed: ({r}, {c}) is fixed to {v}, which no export constant on its cycle is"
            ));
        }
    }
    Ok(())
}

// --- Lean rendering -----------------------------------------------------------------------

/// Items per chunk definition of a long list.
const CHUNK: usize = 128;

/// A flat Lean list literal, eight items to a line.
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

/// Render `items` as `def {name}{k} : List ({ty}) := [...]` chunks into `defs` and return
/// the balanced `++` tree over them (or the flat literal when short).
fn render_list(defs: &mut String, name: &str, ty: &str, items: &[String]) -> String {
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
            "/-- `{name}`, items `{}..{}`. -/\nnoncomputable def {name}{k} : List ({ty}) := {}\n",
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

/// `copies_linked` is combined from lemmas over at most this many copy chunks each.
const LINKED_GROUP: usize = 512;

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
noncomputable def {name}{k} : Tbl ({ty}) :=
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

/// The Lean names of a traced circuit: its generated module under
/// `Plonky2Bridge/Generated/`, its bridge namespace, its export's definition and the
/// spec module defining it.
pub struct Names {
    pub module: String,
    pub namespace: String,
    pub circuit: String,
    pub spec_module: String,
}

/// The names of the circuit whose trace is `tag`.
pub fn names(tag: &str) -> Result<Names, String> {
    let (module, namespace, circuit, spec) = if tag == "leaf_circuit" {
        (
            "LeafBuilt".to_string(),
            "LeafCircuit".to_string(),
            "leafCircuit".to_string(),
            "LeafCircuit".to_string(),
        )
    } else if let Some(n) = tag.strip_prefix("private_batch_wrapper_n") {
        (
            format!("Wrapper{n}Built"),
            format!("Wrapper{n}"),
            format!("privateBatchWrapper{n}"),
            format!("PrivateBatchWrapper{n}"),
        )
    } else if let Some(n) = tag.strip_prefix("public_batch_wrapper_n") {
        (
            format!("PublicWrapper{n}Built"),
            format!("PublicWrapper{n}"),
            format!("publicBatchWrapper{n}"),
            format!("PublicBatchWrapper{n}"),
        )
    } else {
        return Err(format!("no Lean names for circuit {tag:?}"));
    };
    Ok(Names {
        module,
        namespace: format!("Plonky2Bridge.{namespace}"),
        circuit,
        spec_module: format!("Plonky2Spec.Generated.{spec}"),
    })
}

/// The children of a circuit's tree, as the bridge names them: the leaf's public inputs
/// are `leafPis` over the `leaf_pis_i` targets, an inner wrapper's `innerPis` over
/// `inner_pis_i`.
fn child_pis(child: &str) -> (&'static str, &'static str) {
    if child == "leaf_circuit" {
        ("leafPis", "leaf_pis_")
    } else {
        ("innerPis", "inner_pis_")
    }
}

/// Render the certificate as `Plonky2Bridge/Generated/{Module}.lean`: the circuit as a
/// recursion tree over its children, the `BuildCert`, its witnesses, the
/// kernel-evaluated checks against the export, `build_refines_of_agrees` and
/// `build_hashes_of_agrees`.
pub fn render_lean(t: &LoadedTrace, cert: &Cert) -> Result<String, String> {
    let ex = &t.ex;
    let tag = &t.circuit;
    let Names {
        namespace,
        circuit: circuit_name,
        spec_module,
        ..
    } = names(tag)?;
    let child = ex.verifiers.first().map(|(c, _)| names(c)).transpose()?;
    let positions: usize = cert.cycles.iter().map(Vec::len).sum();
    let mut o = String::new();
    let prelude = match &ex.verifiers[..] {
        [] => "with the recorded config".to_string(),
        [(c, _), ..] => format!(
            "with the recorded config, after\n  \
             `verify_proof` gadgets under the built `{c}` on {} proofs, as\n  \
             `add_recursive_verifiers` lays them down,",
            ex.verifiers.len()
        ),
    };
    let _ = writeln!(
        o,
        "/-\n  GENERATED by `constraint-exporter` (`built.rs`) — do not edit.\n\n  \
         The certified build of `{tag}`: its trace replayed into `CircuitBuilder` {prelude} and built,\n  \
         to the state digest the trace records of the production builder. The export is the\n  \
         whole of that builder (`built::complete`): the trace's part first, then the rest.\n\n  \
         `buildCert` is what the exporter read off `CircuitBuilder::build` of that builder:\n  \
         the export's {} rows at their places, {} positions on {} `σ`-cycles the export's {} copies\n  \
         and {} constants run along, over {} routed columns, the place of {} virtual targets, and\n  \
         the public-input hash: `PoseidonGate` rows {}..{} absorbing the {} public inputs into the\n  \
         `PublicInputGate` at row {}.\n  \
         The checks of the certificate against the export are kernel evaluations\n  \
         (`decide +kernel`); `build_refines_of_agrees` is `Built.Agrees.refines` on them and\n  \
         `build_hashes_of_agrees` is `Built.Agrees.hashesPublicInputs`. Nothing here is assumed:\n  \
         that the deployed build `Agrees` with this certificate, and that the export is all of\n  \
         it (`Built.Complete`), is `{namespace}.build_agrees`.\n-/",
        ex.rows.len(),
        positions,
        cert.cycles.len(),
        ex.copies.len(),
        ex.constants.len(),
        cert.routed,
        cert.virt.len(),
        cert.hash_rows.first().copied().unwrap_or(cert.pi_row),
        cert.pi_row,
        ex.public_inputs.len(),
        cert.pi_row,
    );
    let _ = writeln!(o, "import Plonky2Bridge.Trusted");
    if let Some(child) = &child {
        let _ = writeln!(o, "import Plonky2Bridge.Generated.{}", child.module);
    }
    let _ = writeln!(o, "import {spec_module}\n");
    let _ = writeln!(o, "namespace {namespace}\n");
    let _ = writeln!(o, "open Plonky2Spec.Wiring");
    let _ = writeln!(o, "open Plonky2Spec.Generated\n");
    let _ = writeln!(o, "variable {{p : ℕ}}\n");
    let _ = writeln!(o, "set_option maxRecDepth 8192\n");
    // Elaborating the checks as tasks retains some 12 MB per check until the file is done.
    let _ = writeln!(o, "set_option Elab.async false\n");

    // The tree.
    let _ = writeln!(
        o,
        "/-! ### The circuit as a recursion tree (PLAN.md Step 10) -/\n"
    );
    let n = ex.verifiers.len();
    if let Some(child) = &child {
        let child_tag = &ex.verifiers[0].0;
        let pi_len = ex.verifiers[0].1.len();
        let (pis, prefix) = child_pis(child_tag);
        let _ = writeln!(
            o,
            "/-- The {pi_len} public inputs of child proof `i` (`{prefix}i`): the targets its\n    \
             `verify_proof` gadget reads. -/\n\
             noncomputable def {pis} : Fin {n} → Fin {pi_len} → Target"
        );
        for i in 0..n {
            let _ = writeln!(o, "  | {i} => {circuit_name}.{prefix}{i}");
        }
        if child_tag != "leaf_circuit" {
            let _ = writeln!(o, "  | _ => {circuit_name}.{prefix}0");
        }
        let _ = writeln!(
            o,
            "\n/-- The circuit with its recursion gadgets: `verify_proof` under the `{child_tag}`\n    \
             verifier key on each `{prefix}i`. -/\n\
             noncomputable def tree : Recursive p :=\n  .node \"{tag}\" ({circuit_name} p)"
        );
        let children: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    "({}.tree, List.ofFn ({pis} {i}))",
                    child.namespace.trim_start_matches("Plonky2Bridge.")
                )
            })
            .collect();
        let _ = writeln!(o, "    [{}]\n", children.join(",\n     "));
        let child_ns = child.namespace.trim_start_matches("Plonky2Bridge.");
        if child_tag == "leaf_circuit" {
            let _ = writeln!(
                o,
                "/-- The tree's gadgets are the recorded ones. -/\n\
                 theorem tree_verifiers : (tree (p := p)).verifiers = {circuit_name}.verifiers := rfl\n"
            );
        } else {
            for i in 0..n {
                let _ = writeln!(
                    o,
                    "theorem pisEq_{i} :\n    \
                     List.ofFn ({pis} {i}) = ({circuit_name}.verifiers[{i}]'(by decide)).2 := by\n  rfl\n"
                );
            }
            let _ = writeln!(
                o,
                "/-- The tree's gadgets are the recorded ones. -/\n\
                 theorem tree_verifiers : (tree (p := p)).verifiers = {circuit_name}.verifiers := by\n  \
                 have htag : {child_ns}.tree.tag (p := p) = \"{child_tag}\" := by\n    \
                 simp only [Recursive.tag, {child_ns}.tree]\n  \
                 rw [Recursive.verifiers, tree, Recursive.children]"
            );
            for i in 0..n {
                let _ = writeln!(o, "  rw [List.map_cons, htag, pisEq_{i}]");
            }
            let _ = writeln!(o, "  rw [List.map_nil]\n  rfl\n");
        }
    } else {
        let _ = writeln!(
            o,
            "/-- The circuit as a recursion tree: no gadgets. -/\n\
             noncomputable def tree : Recursive p := .node \"{tag}\" ({circuit_name} p) []\n\n\
             /-- The tree's gadgets are the recorded ones: none. -/\n\
             theorem tree_verifiers : (tree (p := p)).verifiers = {circuit_name}.verifiers := rfl\n"
        );
    }

    // The certificate.
    let _ = writeln!(o, "/-! ### The certificate -/\n");
    let mut defs = String::new();
    let row_at: Vec<(String, usize)> = cert.row_at.iter().map(|r| (r.to_string(), 1)).collect();
    let row_at = render_tbl(&mut defs, "buildCert.rowAt", "ℕ", &row_at);
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
noncomputable def {name} : Tbl (ℕ × ℕ) := {expr}
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
    let wit = |(ci, iu, iv): &(usize, usize, usize)| format!("({ci}, {iu}, {iv})");
    let copy_wits: Vec<String> = cert.copy_wits.iter().map(wit).collect();
    let const_wits: Vec<String> = cert.const_wits.iter().map(wit).collect();
    let const_wits = render_list(&mut defs, "constWits", "ℕ × ℕ × ℕ", &const_wits);
    o.push_str(&defs);

    let zeros: Vec<String> = cert.zeros.iter().map(|&u| pos(u)).collect();
    let hash_rows: Vec<String> = cert.hash_rows.iter().map(usize::to_string).collect();
    let _ = writeln!(
        o,
        "/-- The certificate of the deployed build against `{circuit_name}`. -/\n\
         noncomputable def buildCert : BuildCert where\n  rowAt := {row_at}\n  routed := {}\n  numPublicInputs := {}\n  \
         cycles := {cycles}\n  virt := {virt}\n  pins := {pins}\n  zeros := [{}]\n  piRow := {}\n  \
         hashRows := [{}]\n",
        cert.routed,
        cert.num_public_inputs,
        zeros.join(", "),
        cert.pi_row,
        hash_rows.join(", "),
    );
    let copy_chunks = crate::circuit::chunks(ex.copies.len());
    match &copy_chunks {
        None => {
            let _ = writeln!(
                o,
                "/-- One witness per copy of the export. -/\nnoncomputable def copyWits : List (ℕ × ℕ × ℕ) := {}\n",
                flat(&copy_wits)
            );
        }
        Some(ranges) => {
            for (k, r) in ranges.iter().enumerate() {
                let _ = writeln!(
                    o,
                    "/-- One witness per copy of `{circuit_name}.copies{k}`. -/\n\
                     noncomputable def copyWits{k} : List (ℕ × ℕ × ℕ) := {}\n",
                    flat(&copy_wits[r.clone()])
                );
            }
        }
    }
    let _ = writeln!(
        o,
        "/-- One witness per constant target of the export. -/\nnoncomputable def constWits : List (ℕ × ℕ × ℕ) := {const_wits}\n"
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
         of each. -/\nnoncomputable def hashWits : List GateWit :=\n  [{}]\n",
        hash_wits.join(",\n   ")
    );
    let _ = writeln!(
        o,
        "/-- The end state's four lanes into the `PublicInputGate`. -/\n\
         noncomputable def outWits : List (ℕ × ℕ × ℕ) := [{}]\n",
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
        "/-- The export's gate kinds, row by row. -/\nnoncomputable def rowKinds : List GateKind := {kinds}\n"
    );
    let const_targets: Vec<String> = ex.constants.iter().map(|(t, _)| lean_target(*t)).collect();
    let _ = writeln!(
        o,
        "/-- The export's constant targets, in its order. -/\nnoncomputable def constTargets : List Target := [{}]\n",
        const_targets.join(", ")
    );
    let pi_targets: Vec<String> = ex.public_inputs.iter().map(|&t| lean_target(t)).collect();
    let _ = writeln!(
        o,
        "/-- The export's public-input targets, in its order. -/\nnoncomputable def piTargets : List Target := {}\n",
        flat(&pi_targets)
    );

    let _ = writeln!(
        o,
        "theorem rowKinds_eq : ({circuit_name} p).rows.map Row.kind = rowKinds := rfl\n"
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
        "theorem publicInputs_targets : ({circuit_name} p).publicInputs = piTargets := rfl\n"
    );
    let _ = writeln!(
        o,
        "/-- The export has no `ConstantGate` row. -/\n\
         theorem checks_rows : rowKinds.all noConstantRow = true := by decide +kernel\n"
    );
    match &copy_chunks {
        None => {
            let _ = writeln!(
                o,
                "theorem checks_copies : buildCert.copiesOk ({circuit_name} p).copies copyWits = true := by\n  \
                 decide +kernel\n\n\
                 /-- Every copy of the export is linked in the certificate. -/\n\
                 theorem copies_linked : ∀ xy ∈ ({circuit_name} p).copies, buildCert.Links xy.1 xy.2 :=\n  \
                 BuildCert.copiesOk_sound checks_copies\n"
            );
        }
        Some(ranges) => {
            for k in 0..ranges.len() {
                let _ = writeln!(
                    o,
                    "theorem checks_copies_{k} : buildCert.copiesOk {circuit_name}.copies{k} copyWits{k} = true := by\n  \
                     decide +kernel\n"
                );
            }
            fn linked(lo: usize, hi: usize) -> String {
                if hi - lo == 1 {
                    format!("(BuildCert.copiesOk_sound checks_copies_{lo})")
                } else {
                    let mid = lo + (hi - lo) / 2;
                    format!(
                        "(BuildCert.links_append {} {})",
                        linked(lo, mid),
                        linked(mid, hi)
                    )
                }
            }
            // The tree over the chunks, as a lemma per subtree of at most `LINKED_GROUP`
            // chunks: a single declaration naming every chunk's check takes minutes to
            // index (`.ilean`), where the same references over many take seconds.
            fn linked_groups(o: &mut String, chunks: &str, lo: usize, hi: usize) -> String {
                if hi - lo <= LINKED_GROUP {
                    let _ = writeln!(
                        o,
                        "theorem linked_{lo} : ∀ xy ∈ {}, buildCert.Links xy.1 xy.2 :=\n  {}\n",
                        crate::circuit::chunk_tree(chunks, "", lo, hi),
                        linked(lo, hi)
                    );
                    return format!("linked_{lo}");
                }
                let mid = lo + (hi - lo) / 2;
                format!(
                    "(BuildCert.links_append {} {})",
                    linked_groups(o, chunks, lo, mid),
                    linked_groups(o, chunks, mid, hi)
                )
            }
            let chunks = format!("{circuit_name}.copies");
            let all_linked = linked_groups(&mut o, &chunks, 0, ranges.len());
            let _ = writeln!(
                o,
                "theorem copies_eq : ({circuit_name} p).copies = {} := rfl\n\n\
                 /-- Every copy of the export is linked in the certificate. -/\n\
                 theorem copies_linked : ∀ xy ∈ ({circuit_name} p).copies, buildCert.Links xy.1 xy.2 := by\n  \
                 rw [copies_eq]\n  exact {all_linked}\n",
                crate::circuit::chunk_tree(&chunks, "", 0, ranges.len())
            );
        }
    }
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
         h.refines (by rw [rowKinds_eq]; exact checks_rows) copies_linked\n    \
         (by rw [constants_fst]; exact checks_constants) publicInputs_length.symm\n"
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
         h.hashesPublicInputs (by rw [publicInputs_targets]; exact checks_pi)\n"
    );
    let _ = writeln!(o, "end {namespace}");
    Ok(o)
}

/// A certified build: the trace completed to the whole builder, the built view of its
/// replay and the certificate.
pub struct Certified {
    pub trace: LoadedTrace,
    pub view: FormalBuiltView<F, D>,
    pub cert: Cert,
}

/// The named circuit's vendored trace, replayed over its child built the same way.
pub fn replay_circuit(name: &str) -> Result<(LoadedTrace, CircuitBuilder<F, D>), String> {
    let t = load(&traces_dir().join(format!("{name}.json")))?;
    if t.circuit != name {
        return Err(format!("trace is for {:?}", t.circuit));
    }
    let child = match t.ex.verifiers.first() {
        Some((child, _)) => {
            let (_, b) = replay_circuit(child)?;
            Some((child.clone(), b.build::<C>()))
        }
        None => None,
    };
    let b = replay(&t, child.as_ref().map(|(name, data)| (name.as_str(), data)))?;
    Ok((t, b))
}

/// The named circuit's certificate from its vendored trace: replayed, built, completed,
/// read.
pub fn certify_circuit(name: &str) -> Result<Certified, String> {
    let (trace, b) = replay_circuit(name)?;
    let pre = Prebuild::of(&b);
    let (_, view) = build(b, false)?;
    let trace = complete(trace, &pre, &view)?;
    let cert = certify(&trace, &view)?;
    Ok(Certified { trace, view, cert })
}

/// `Plonky2Bridge/Generated/{Module}.lean` for the named circuit, split across
/// `{Module}/` when the certificate is too large for one Lean module.
pub fn generate_built_modules(name: &str) -> Result<Vec<crate::lean_split::LeanModule>, String> {
    let Certified { trace, cert, .. } = certify_circuit(name)?;
    built_modules(&trace, &cert)
}

pub fn built_modules(
    trace: &LoadedTrace,
    cert: &Cert,
) -> Result<Vec<crate::lean_split::LeanModule>, String> {
    let module = names(&trace.circuit)?.module;
    Ok(crate::lean_split::split_module(
        &crate::lean_split::BUILT,
        &module,
        &render_lean(trace, cert)?,
    ))
}
