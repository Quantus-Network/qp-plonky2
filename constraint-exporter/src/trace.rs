//! Gadget-call traces recorded in `qp-zk-circuits` (`common/src/formal_trace.rs`,
//! PLAN.md Steps 8c and 19).
//!
//! The wormhole wrappers live in another workspace whose `plonky2` is the published crate,
//! so they cannot be built through this crate's `Recorder`. Instead that workspace records
//! the same information (`formal_export_view` before and after each `GadgetBuilder` call)
//! into a JSON trace checked in under its `formal/traces/`, which the pinned `wormholeSpec`
//! lake package brings here. `load` turns a trace into the `CircuitExport` + `Call`s the
//! decode-theorem generator consumes.
//!
//! A trace is recorded on the production builder, after its recursive verifiers: it names
//! rows by their index in that builder and virtual targets by theirs, and lists only the
//! rows and virtual targets the wrapper gadgets touch. The export renumbers both densely
//! — export row `i` is builder row `row_at[i]`, export virtual target `n` is builder
//! virtual target `virt_at[n]` — so the Lean statement is about the wrapper alone, and
//! `built::replay` maps back to rebuild the circuit.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_data::CircuitConfig;
use serde::Deserialize;

use crate::circuit::{classify, CircuitExport, GateKind};
use crate::gadget::{render_module, Call, Fact, GeneratedCircuit};

type F = GoldilocksField;

#[derive(Deserialize)]
struct TraceRow {
    row: usize,
    gate: String,
    constants: Vec<String>,
}

#[derive(Deserialize)]
struct TraceCall {
    kind: String,
    args: Vec<String>,
    outs: Vec<String>,
    fresh: Vec<String>,
    #[serde(default)]
    bits: Option<usize>,
    rows: [usize; 2],
    copies: [usize; 2],
}

/// The recording builder's counts of rows, copy constraints and virtual targets at a
/// moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Counts {
    pub rows: usize,
    pub copies: usize,
    pub virtual_targets: usize,
}

#[derive(Deserialize)]
struct Trace {
    circuit: String,
    config: CircuitConfig,
    digest: String,
    start: Counts,
    num_virtual_targets: usize,
    rows: Vec<TraceRow>,
    copies: Vec<[String; 2]>,
    constants: Vec<(String, String)>,
    public_inputs: Vec<String>,
    named: Vec<(String, Vec<String>)>,
    verifiers: Vec<TraceVerifier>,
    calls: Vec<TraceCall>,
}

#[derive(Deserialize)]
struct TraceVerifier {
    child: String,
    public_inputs: Vec<String>,
}

/// `w{row}:{column}` for wires, `v{index}` for virtual targets: the trace's spelling.
pub fn encode_target(t: Target) -> String {
    match t {
        Target::Wire(w) => format!("w{}:{}", w.row, w.column),
        Target::VirtualTarget { index } => format!("v{index}"),
    }
}

/// `w{row}:{column}` or `v{index}`.
fn target(s: &str) -> Result<Target, String> {
    if let Some(rest) = s.strip_prefix('w') {
        let (row, col) = rest
            .split_once(':')
            .ok_or_else(|| format!("bad wire target {s:?}"))?;
        let row = row.parse().map_err(|_| format!("bad wire target {s:?}"))?;
        let col = col.parse().map_err(|_| format!("bad wire target {s:?}"))?;
        Ok(Target::wire(row, col))
    } else if let Some(idx) = s.strip_prefix('v') {
        let index = idx
            .parse()
            .map_err(|_| format!("bad virtual target {s:?}"))?;
        Ok(Target::VirtualTarget { index })
    } else {
        Err(format!("bad target {s:?}"))
    }
}

fn felt(s: &str) -> Result<F, String> {
    let n: u64 = s.parse().map_err(|_| format!("bad field element {s:?}"))?;
    Ok(F::from_canonical_u64(n))
}

fn fixed<const N: usize>(ts: Vec<Target>, what: &str) -> Result<[Target; N], String> {
    let n = ts.len();
    ts.try_into()
        .map_err(|_| format!("{what}: expected {N} targets, got {n}"))
}

/// A call in the export's numbering.
struct LocalCall {
    kind: String,
    args: Vec<Target>,
    outs: Vec<Target>,
    fresh: Vec<Target>,
    bits: Option<usize>,
    rows: [usize; 2],
    copies: [usize; 2],
}

fn call(ex: &CircuitExport, c: LocalCall) -> Result<Call, String> {
    let LocalCall {
        args, outs, fresh, ..
    } = c;
    let what = format!("{} call", c.kind);
    if c.rows[0] > c.rows[1] || c.rows[1] > ex.rows.len() {
        return Err(format!("{what}: row range {:?} is out of bounds", c.rows));
    }
    if c.copies[0] > c.copies[1] || c.copies[1] > ex.copies.len() {
        return Err(format!(
            "{what}: copy range {:?} is out of bounds",
            c.copies
        ));
    }
    let out1 = |outs: &[Target]| -> Result<Target, String> {
        match outs {
            [o] => Ok(*o),
            _ => Err(format!("{what}: expected one output, got {}", outs.len())),
        }
    };
    let fact = match c.kind.as_str() {
        "select" => {
            let [b, x, y] = fixed(args, &what)?;
            Fact::Select {
                b,
                x,
                y,
                out: out1(&outs)?,
            }
        }
        "not" => {
            let [b] = fixed(args, &what)?;
            Fact::Not {
                b,
                out: out1(&outs)?,
            }
        }
        "and" => {
            let [b1, b2] = fixed(args, &what)?;
            Fact::And {
                b1,
                b2,
                out: out1(&outs)?,
            }
        }
        "or" => {
            let [b1, b2] = fixed(args, &what)?;
            Fact::Or {
                b1,
                b2,
                out: out1(&outs)?,
            }
        }
        "add" => {
            let [x, y] = fixed(args, &what)?;
            Fact::Add {
                x,
                y,
                out: out1(&outs)?,
            }
        }
        "sub" => {
            let [x, y] = fixed(args, &what)?;
            Fact::Sub {
                x,
                y,
                out: out1(&outs)?,
            }
        }
        "mul" => {
            let [x, y] = fixed(args, &what)?;
            Fact::Mul {
                x,
                y,
                out: out1(&outs)?,
            }
        }
        "is_equal" => {
            let [x, y] = fixed(args, &what)?;
            let equal = out1(&outs)?;
            let inv: Vec<Target> = fresh.into_iter().filter(|t| *t != equal).collect();
            let [inv] = fixed(inv, "is_equal call: auxiliary target")?;
            Fact::IsEqual { x, y, equal, inv }
        }
        "range_check" => {
            let [x] = fixed(args, &what)?;
            let bits = c.bits.ok_or("range_check call without bits")?;
            Fact::RangeCheck { x, bits }
        }
        "connect" => {
            let [x, y] = fixed(args, &what)?;
            Fact::Connect { x, y }
        }
        "assert_bool" => {
            let b = out1(&outs)?;
            Fact::AssertBool { b }
        }
        "split_le" => {
            let [x] = fixed(args, &what)?;
            let bits = c.bits.ok_or("split_le call without bits")?;
            let rows: Vec<usize> = (c.rows[0]..c.rows[1])
                .filter(|&r| matches!(ex.rows[r].0, GateKind::BaseSum2 { .. }))
                .collect();
            let [row] = rows[..] else {
                return Err(format!(
                    "{what}: expected one BaseSumGate<2> row, got {rows:?}"
                ));
            };
            let expect: Vec<Target> = (1..=bits).map(|col| Target::wire(row, col)).collect();
            if outs != expect {
                return Err(format!(
                    "{what}: outputs {outs:?} are not the row's first {bits} limb wires"
                ));
            }
            Fact::SplitLe { x, row, bits }
        }
        "poseidon2_hash" => {
            let blocks = args.len() / 8 + 1;
            let rows: Vec<usize> = (c.rows[0]..c.rows[1])
                .filter(|&r| ex.rows[r].0 == GateKind::Poseidon2)
                .collect();
            if rows.len() != blocks {
                return Err(format!(
                    "{what}: expected {blocks} Poseidon2Gate rows for {} inputs, got {rows:?}",
                    args.len()
                ));
            }
            let last = rows[blocks - 1];
            let expect: Vec<Target> = (12..16).map(|col| Target::wire(last, col)).collect();
            if outs != expect {
                return Err(format!(
                    "{what}: outputs {outs:?} are not the last row's first four output wires"
                ));
            }
            Fact::Poseidon2 { rows, inputs: args }
        }
        "split_low_high" => {
            return Err(format!("{what}: not modelled by the decode generator"));
        }
        other => return Err(format!("unknown gadget call kind {other:?}")),
    };
    Ok(Call {
        fact,
        rows: c.rows[0]..c.rows[1],
        copies: c.copies[0]..c.copies[1],
    })
}

/// A parsed trace: the circuit's name, its export and calls in the export's numbering,
/// and what `built::replay` needs to rebuild the recording builder.
#[derive(Debug)]
pub struct LoadedTrace {
    pub circuit: String,
    pub ex: CircuitExport,
    pub calls: Vec<Call>,
    pub config: CircuitConfig,
    /// `state_digest` of the recording builder.
    pub digest: String,
    /// The recording builder's counts when tracing began: what its verifiers laid down.
    pub start: Counts,
    pub num_virtual_targets: usize,
    /// Export row `i` is the builder's row `row_at[i]`; increasing.
    pub row_at: Vec<usize>,
    /// Export virtual target `n` is the builder's virtual target `virt_at[n]`; increasing.
    pub virt_at: Vec<usize>,
}

impl LoadedTrace {
    /// The builder's target an export target stands for.
    pub fn production(&self, t: Target) -> Target {
        match t {
            Target::Wire(w) => Target::wire(self.row_at[w.row], w.column),
            Target::VirtualTarget { index } => Target::VirtualTarget {
                index: self.virt_at[index],
            },
        }
    }
}

pub fn parse(json: &str) -> Result<LoadedTrace, String> {
    let t: Trace = serde_json::from_str(json).map_err(|e| format!("bad trace JSON: {e}"))?;
    let row_at: Vec<usize> = t.rows.iter().map(|r| r.row).collect();
    if !row_at.windows(2).all(|w| w[0] < w[1]) {
        return Err("trace rows are not in increasing order".into());
    }
    let call_targets = t
        .calls
        .iter()
        .flat_map(|c| c.args.iter().chain(&c.outs).chain(&c.fresh));
    let referenced = t
        .copies
        .iter()
        .flatten()
        .chain(t.constants.iter().map(|(x, _)| x))
        .chain(&t.public_inputs)
        .chain(t.named.iter().flat_map(|(_, ts)| ts))
        .chain(t.verifiers.iter().flat_map(|v| &v.public_inputs))
        .chain(call_targets)
        .map(|s| target(s))
        .collect::<Result<Vec<_>, _>>()?;
    let virt_at: Vec<usize> = referenced
        .iter()
        .filter_map(|t| match t {
            Target::VirtualTarget { index } => Some(*index),
            Target::Wire(_) => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let local = |s: &str| -> Result<Target, String> {
        match target(s)? {
            Target::Wire(w) => {
                let i = row_at
                    .binary_search(&w.row)
                    .map_err(|_| format!("target {s:?} is on a row the trace does not list"))?;
                Ok(Target::wire(i, w.column))
            }
            Target::VirtualTarget { index } => Ok(Target::VirtualTarget {
                index: virt_at.binary_search(&index).expect("referenced"),
            }),
        }
    };
    let locals = |ss: &[String]| ss.iter().map(|s| local(s)).collect::<Result<Vec<_>, _>>();
    let local_row = |r: usize| row_at.partition_point(|&at| at < r);

    let rows = t
        .rows
        .iter()
        .map(|r| {
            Ok((
                classify(&r.gate),
                r.constants
                    .iter()
                    .map(|c| felt(c))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let copies = t
        .copies
        .iter()
        .map(|[x, y]| Ok((local(x)?, local(y)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let constants = t
        .constants
        .iter()
        .map(|(x, v)| Ok((local(x)?, felt(v)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let public_inputs = locals(&t.public_inputs)?;
    let named = t
        .named
        .iter()
        .map(|(n, ts)| Ok((n.clone(), locals(ts)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let verifiers = t
        .verifiers
        .iter()
        .map(|v| Ok((v.child.clone(), locals(&v.public_inputs)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let ex = CircuitExport {
        rows,
        copies,
        constants,
        public_inputs,
        named,
        verifiers,
    };
    let calls = t
        .calls
        .iter()
        .map(|c| {
            call(
                &ex,
                LocalCall {
                    kind: c.kind.clone(),
                    args: locals(&c.args)?,
                    outs: locals(&c.outs)?,
                    fresh: locals(&c.fresh)?,
                    bits: c.bits,
                    rows: c.rows.map(local_row),
                    copies: c.copies.map(|n| n - t.start.copies),
                },
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(LoadedTrace {
        circuit: t.circuit,
        ex,
        calls,
        config: t.config,
        digest: t.digest,
        start: t.start,
        num_virtual_targets: t.num_virtual_targets,
        row_at,
        virt_at,
    })
}

/// The trace's `verify_proof` gadgets are one per child slot, under `child`, on exactly the
/// slot's public-input targets — what the generated `tree` states, so its `tree_verifiers`
/// check against `<name>.verifiers` holds.
pub fn check_verifiers(t: &LoadedTrace, child: &str, pis: &[Vec<Target>]) -> Result<(), String> {
    if t.ex.verifiers.len() != pis.len() {
        return Err(format!(
            "expected {} verify_proof gadgets, trace has {}",
            pis.len(),
            t.ex.verifiers.len()
        ));
    }
    for (i, ((tag, ts), slot)) in t.ex.verifiers.iter().zip(pis).enumerate() {
        if tag != child {
            return Err(format!("verifier {i} is for {tag:?}, expected {child:?}"));
        }
        if ts != slot {
            return Err(format!(
                "verifier {i} public inputs are not the slot's pis_{i} targets"
            ));
        }
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<LoadedTrace, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read trace {}: {e}", path.display()))?;
    parse(&json)
}

/// The traces the exporter reads: `constraint-exporter/traces/`, a checked-in copy of
/// `qp-zk-circuits/formal/traces/` at the `wormholeSpec` revision pinned in
/// `formal/lakefile.toml`, so `cargo test` needs neither Lake nor the network.
/// `WORMHOLE_TRACES` overrides the directory (for developing against a local
/// `qp-zk-circuits` checkout). `vendored_trace_matches_pinned_package` checks the copy
/// against the lake package whenever the latter has been fetched.
pub fn traces_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("WORMHOLE_TRACES") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("traces")
}

/// The same traces inside the `wormholeSpec` lake package, present only after
/// `lake build`/`lake update` in `formal/`.
pub fn pinned_package_traces_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../formal/.lake/packages/wormholeSpec/formal/traces")
}

/// Assert the vendored `constraint-exporter/traces/{name}` is byte-identical to the copy in
/// the pinned `wormholeSpec` package. When the package has not been fetched this skips,
/// unless `FORMAL_TRACES_STRICT` is set, in which case a missing file is a failure (the
/// `formal` CI job, which has run `lake build`, sets it).
pub fn assert_vendored_trace_matches_pinned(name: &str) {
    let strict = std::env::var_os("FORMAL_TRACES_STRICT").is_some();
    let pinned = pinned_package_traces_dir().join(name);
    let expected = match std::fs::read(&pinned) {
        Ok(bytes) => bytes,
        Err(e) if strict => panic!("cannot read pinned trace {}: {e}", pinned.display()),
        Err(_) => {
            eprintln!("skipping: {} not fetched", pinned.display());
            return;
        }
    };
    let vendored = std::fs::read(traces_dir().join(name)).unwrap();
    assert!(
        vendored == expected,
        "constraint-exporter/traces/{name} differs from the pinned wormholeSpec copy; \
         `cp {} constraint-exporter/traces/` and rerun export-constraints",
        pinned.display()
    );
}

/// Build `formal/Plonky2Spec/Generated/PrivateBatchWrapper{n}.lean` from the recorded `n`-leaf
/// private-batch wrapper.
pub fn generate_private_batch_wrapper_lean(n: usize) -> Result<String, String> {
    generate_wrapper_lean(
        &format!("private_batch_wrapper_n{n}"),
        &format!(
            "the `n = {n}` private-batch wrapper trace\n\
             \x20 (`qp-zk-circuits/formal/traces/private_batch_wrapper_n{n}.json`, recorded by\n\
             \x20 `TracingBuilder` while `build_private_batch_constraints` ran on the real builder)"
        ),
        &format!("privateBatchWrapper{n}"),
        &format!(
            "The private-batch aggregation wrapper at `n = {n}` without the leaf verifiers \
             (`wormhole/aggregator/src/private_batch/circuit/circuit_logic.rs`): leaf \
             public inputs `leaf_pis_0..{}`, dummy-nullifier preimages \
             `dummy_pre_image_0..{}`, the permutation switches `switches`, and the aggregated \
             public inputs.",
            n - 1,
            n - 1
        ),
    )
}

/// Build `formal/Plonky2Spec/Generated/PublicBatchWrapper{n}.lean` from the recorded
/// `n_inner = n` public-batch wrapper over `2`-leaf private batches.
pub fn generate_public_batch_wrapper_lean(n: usize) -> Result<String, String> {
    generate_wrapper_lean(
        &format!("public_batch_wrapper_n{n}"),
        &format!(
            "the `n_inner = {n}` public-batch wrapper trace\n\
             \x20 (`qp-zk-circuits/formal/traces/public_batch_wrapper_n{n}.json`, recorded by\n\
             \x20 `TracingBuilder` while `build_public_batch_constraints` ran on the real builder)"
        ),
        &format!("publicBatchWrapper{n}"),
        &format!(
            "The public-batch aggregation wrapper at `n_inner = {n}` over the inner \
             private-batch shape named by its recursion gadgets, without the inner verifiers \
             (`wormhole/aggregator/src/public_batch/circuit/circuit_logic.rs`): inner \
             public inputs `inner_pis_0..{}`, the `aggregator_address` witness, and the \
             aggregated public inputs.",
            n - 1
        ),
    )
}

/// The public-batch decode module, split into several Lean files when it is too large
/// to elaborate at once. Each entry is a path relative to `Plonky2Spec/Generated/`.
pub fn generate_public_batch_wrapper_modules(
    n: usize,
) -> Result<Vec<crate::lean_split::LeanModule>, String> {
    let src = generate_public_batch_wrapper_lean(n)?;
    Ok(crate::lean_split::split_module(
        &format!("PublicBatchWrapper{n}"),
        &src,
    ))
}

/// Build `formal/Plonky2Spec/Generated/LeafCircuit.lean` from the recorded leaf circuit.
pub fn generate_leaf_circuit_lean() -> Result<String, String> {
    generate_wrapper_lean(
        "leaf_circuit",
        "the leaf circuit trace\n\
         \x20 (`qp-zk-circuits/formal/traces/leaf_circuit.json`, recorded by `TracingBuilder`\n\
         \x20 while `build_leaf_constraints` ran on the real builder)",
        "leafCircuit",
        "The wormhole leaf circuit (`wormhole/circuit/src/circuit.rs`): the unspendable \
         account, the zk-tree Merkle proof, the block header and the shared-target wiring \
         with the nullifier and block-hash bindings, on the witness targets `secret`, \
         `transfer_count`, `to_account`, `account_id`, `root_hash`, `depth`, `positions`, \
         `is_not_dummy`, `siblings_0..15` and the header fields.",
    )
}

fn generate_wrapper_lean(
    circuit: &str,
    source: &str,
    name: &str,
    doc: &str,
) -> Result<String, String> {
    let t = load(&traces_dir().join(format!("{circuit}.json")))?;
    if t.circuit != circuit {
        return Err(format!("unexpected trace circuit {:?}", t.circuit));
    }
    Ok(render_module(
        source,
        &[GeneratedCircuit {
            name: name.into(),
            doc: doc.into(),
            ex: t.ex,
            calls: t.calls,
        }],
    ))
}
