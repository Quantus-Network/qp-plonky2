//! Gadget-call traces recorded in `qp-zk-circuits` (`common/src/formal_trace.rs`,
//! PLAN.md Step 8c).
//!
//! The wormhole wrappers live in another workspace whose `plonky2` is the published crate,
//! so they cannot be built through this crate's `Recorder`. Instead that workspace records
//! the same information (`formal_export_view` before and after each `GadgetBuilder` call)
//! into a JSON trace checked in under its `formal/traces/`, which the pinned `wormholeSpec`
//! lake package brings here. `load` turns a trace into the `CircuitExport` + `Call`s the
//! decode-theorem generator consumes.

use std::path::{Path, PathBuf};

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::Target;
use serde::Deserialize;

use crate::circuit::{classify, CircuitExport, GateKind};
use crate::gadget::{render_module, Call, Fact, GeneratedCircuit};

type F = GoldilocksField;

#[derive(Deserialize)]
struct TraceRow {
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

#[derive(Deserialize)]
struct Trace {
    circuit: String,
    rows: Vec<TraceRow>,
    copies: Vec<[String; 2]>,
    constants: Vec<(String, String)>,
    public_inputs: Vec<String>,
    named: Vec<(String, Vec<String>)>,
    calls: Vec<TraceCall>,
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

fn targets(ss: &[String]) -> Result<Vec<Target>, String> {
    ss.iter().map(|s| target(s)).collect()
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

fn call(ex: &CircuitExport, c: &TraceCall) -> Result<Call, String> {
    let args = targets(&c.args)?;
    let outs = targets(&c.outs)?;
    let fresh = targets(&c.fresh)?;
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
        "poseidon2_hash" => {
            let inputs = fixed(args, &what)?;
            let rows: Vec<usize> = (c.rows[0]..c.rows[1])
                .filter(|&r| ex.rows[r].0 == GateKind::Poseidon2)
                .collect();
            let [row] = rows[..] else {
                return Err(format!(
                    "{what}: expected one Poseidon2Gate row, got {rows:?}"
                ));
            };
            let expect: Vec<Target> = (12..16).map(|col| Target::wire(row, col)).collect();
            if outs != expect {
                return Err(format!(
                    "{what}: outputs {outs:?} are not the row's first four output wires"
                ));
            }
            Fact::Poseidon2 { row, inputs }
        }
        other => return Err(format!("unknown gadget call kind {other:?}")),
    };
    Ok(Call {
        fact,
        rows: c.rows[0]..c.rows[1],
        copies: c.copies[0]..c.copies[1],
    })
}

/// A parsed trace: the circuit's name, its export, and its calls.
#[derive(Debug)]
pub struct LoadedTrace {
    pub circuit: String,
    pub ex: CircuitExport,
    pub calls: Vec<Call>,
}

pub fn parse(json: &str) -> Result<LoadedTrace, String> {
    let t: Trace = serde_json::from_str(json).map_err(|e| format!("bad trace JSON: {e}"))?;
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
        .map(|[x, y]| Ok((target(x)?, target(y)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let constants = t
        .constants
        .iter()
        .map(|(x, v)| Ok((target(x)?, felt(v)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let public_inputs = targets(&t.public_inputs)?;
    let named = t
        .named
        .iter()
        .map(|(n, ts)| Ok((n.clone(), targets(ts)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let ex = CircuitExport {
        rows,
        copies,
        constants,
        public_inputs,
        named,
    };
    let calls = t
        .calls
        .iter()
        .map(|c| call(&ex, c))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(LoadedTrace {
        circuit: t.circuit,
        ex,
        calls,
    })
}

pub fn load(path: &Path) -> Result<LoadedTrace, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read trace {}: {e}", path.display()))?;
    parse(&json)
}

/// The trace directory of the pinned `wormholeSpec` lake package, or `WORMHOLE_TRACES`
/// when set (for developing against a local `qp-zk-circuits` checkout).
pub fn traces_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("WORMHOLE_TRACES") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../formal/.lake/packages/wormholeSpec/formal/traces")
}

/// Build `formal/Plonky2Spec/Generated/PrivateBatchWrapper2.lean` from the recorded
/// `n = 2` private-batch wrapper.
pub fn generate_private_batch_wrapper_lean() -> Result<String, String> {
    let t = load(&traces_dir().join("private_batch_wrapper_n2.json"))?;
    if t.circuit != "private_batch_wrapper_n2" {
        return Err(format!("unexpected trace circuit {:?}", t.circuit));
    }
    Ok(render_module(
        "the `n = 2` private-batch wrapper trace\n\
         \x20 (`qp-zk-circuits/formal/traces/private_batch_wrapper_n2.json`, recorded by\n\
         \x20 `TracingBuilder` while `build_private_batch_constraints` ran on the real builder)",
        &[GeneratedCircuit {
            name: "privateBatchWrapper2".into(),
            doc: "The private-batch aggregation wrapper at `n = 2` without the leaf verifiers \
                  (`wormhole/aggregator/src/private_batch/circuit/circuit_logic.rs`): leaf \
                  public inputs `leaf_pis_0/1`, dummy-nullifier preimages `dummy_pre_image_0/1`, \
                  the permutation switch `switches`, and the aggregated public inputs."
                .into(),
            ex: t.ex,
            calls: t.calls,
        }],
    ))
}
