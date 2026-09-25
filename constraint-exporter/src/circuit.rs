//! Circuit-level export (PLAN.md Step 8 spike): walk a `CircuitBuilder`'s *pre-`build`*
//! constraint system — placed gate rows, copy constraints, constant targets, and the
//! public-input registration order — and render it as a Lean `Plonky2Spec.Wiring.Circuit`
//! value. This is the wiring/copy-constraint layer the per-gate exporter (`extract.rs`)
//! does not cover: which gate sits on which row with which constants, and how `connect`
//! ties targets together.
//!
//! What is exported is exactly what `CircuitBuilder::build` consumes: `build` only adds
//! `ConstantGate` rows for the constant targets (modeled here as `a t = c` directly),
//! resolves the copy constraints into the permutation argument, and pads. Everything the
//! wrapper *logic* imposes is already present in the builder state.

use core::fmt::Write as _;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::PrimeField64;
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::symbolic::GOLDILOCKS_ORDER;

type F = GoldilocksField;
const D: usize = 2;

/// A gate row as placed by the builder, classified by gate kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateKind {
    Arithmetic { num_ops: usize },
    Constant { num_consts: usize },
    PublicInput,
    Noop,
    Other(String),
}

/// The exported pre-`build` constraint system.
#[derive(Debug, Clone)]
pub struct CircuitExport {
    pub rows: Vec<(GateKind, Vec<F>)>,
    pub copies: Vec<(Target, Target)>,
    pub constants: Vec<(Target, F)>,
    pub public_inputs: Vec<Target>,
    /// Named targets of interest, so Lean statements can refer to them by role.
    pub named: Vec<(String, Vec<Target>)>,
}

/// Classify a gate by its `Gate::id()` string (the `Debug` rendering of the gate struct).
fn classify(id: &str) -> GateKind {
    fn field(id: &str, key: &str) -> Option<usize> {
        let start = id.find(key)? + key.len();
        let rest = &id[start..];
        let digits: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse().ok()
    }
    if id.starts_with("ArithmeticGate") {
        if let Some(n) = field(id, "num_ops:") {
            return GateKind::Arithmetic { num_ops: n };
        }
    }
    if id.starts_with("ConstantGate") {
        if let Some(n) = field(id, "num_consts:") {
            return GateKind::Constant { num_consts: n };
        }
    }
    if id.starts_with("PublicInputGate") {
        return GateKind::PublicInput;
    }
    if id.starts_with("NoopGate") {
        return GateKind::Noop;
    }
    GateKind::Other(id.to_string())
}

/// Snapshot the builder's constraint system.
pub fn export(builder: &CircuitBuilder<F, D>, named: Vec<(String, Vec<Target>)>) -> CircuitExport {
    let view = builder.formal_export_view();
    let rows = view
        .gate_instances
        .iter()
        .map(|gi| (classify(&gi.gate_ref.0.id()), gi.constants.clone()))
        .collect();
    let copies = view.copy_constraints.iter().map(|c| c.pair).collect();
    let mut constants: Vec<(Target, F)> = view
        .constant_targets
        .iter()
        .map(|(t, c)| (*t, *c))
        .collect();
    // Deterministic order (the builder's map is a hash map).
    constants.sort_by_key(|(_, c)| c.to_canonical_u64());
    CircuitExport {
        rows,
        copies,
        constants,
        public_inputs: view.public_inputs.to_vec(),
        named,
    }
}

// --- Lean rendering -----------------------------------------------------------------------

fn lean_target(t: Target) -> String {
    match t {
        Target::Wire(w) => format!(".wire {} {}", w.row, w.column),
        Target::VirtualTarget { index } => format!(".virt {index}"),
    }
}

/// Render a Goldilocks constant generically over `ZMod p`: small values as numerals,
/// small negatives as `-(numeral)`. Anything else is emitted as its canonical `u64`,
/// which is only faithful at `p = goldilocks` (flagged in a comment).
fn lean_const(c: F) -> String {
    let n = c.to_canonical_u64();
    if n <= 1 << 32 {
        n.to_string()
    } else if GOLDILOCKS_ORDER - n <= 1 << 32 {
        format!("(-{})", GOLDILOCKS_ORDER - n)
    } else {
        format!("{n} /- canonical u64; faithful only at p = goldilocks -/")
    }
}

fn lean_kind(k: &GateKind) -> String {
    match k {
        GateKind::Arithmetic { num_ops } => format!(".arithmetic {num_ops}"),
        GateKind::Constant { num_consts } => format!(".constant {num_consts}"),
        GateKind::PublicInput => ".publicInput".to_string(),
        GateKind::Noop => ".noop".to_string(),
        GateKind::Other(id) => format!(".other {id:?}"),
    }
}

/// Render the export as a Lean `def <name> (p : ℕ) : Circuit p` plus one `def <name>.<role>`
/// per named target group (a single `Target`, or a `Fin n → Target` vector).
pub fn render_lean(name: &str, doc: &str, ex: &CircuitExport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "/-- {doc} -/");
    let _ = writeln!(out, "def {name} (p : ℕ) : Circuit p where");
    out.push_str("  rows := [\n");
    for (i, (k, consts)) in ex.rows.iter().enumerate() {
        let cs: Vec<String> = consts.iter().map(|&c| lean_const(c)).collect();
        let sep = if i + 1 == ex.rows.len() { "" } else { "," };
        let _ = writeln!(
            out,
            "    ⟨{}, [{}]⟩{sep}  -- row {i}",
            lean_kind(k),
            cs.join(", ")
        );
    }
    out.push_str("  ]\n  copies := [\n");
    for (i, (x, y)) in ex.copies.iter().enumerate() {
        let sep = if i + 1 == ex.copies.len() { "" } else { "," };
        let _ = writeln!(out, "    ({}, {}){sep}", lean_target(*x), lean_target(*y));
    }
    out.push_str("  ]\n  constants := [\n");
    for (i, (t, c)) in ex.constants.iter().enumerate() {
        let sep = if i + 1 == ex.constants.len() { "" } else { "," };
        let _ = writeln!(out, "    ({}, {}){sep}", lean_target(*t), lean_const(*c));
    }
    out.push_str("  ]\n  publicInputs := [\n");
    for (i, t) in ex.public_inputs.iter().enumerate() {
        let sep = if i + 1 == ex.public_inputs.len() {
            ""
        } else {
            ","
        };
        let _ = writeln!(out, "    {}{sep}", lean_target(*t));
    }
    out.push_str("  ]\n\n");
    for (role, ts) in &ex.named {
        if ts.len() == 1 {
            let _ = writeln!(
                out,
                "/-- Named target `{role}`. -/\ndef {name}.{role} : Target := {}\n",
                lean_target(ts[0])
            );
        } else {
            let items: Vec<String> = ts.iter().map(|&t| lean_target(t)).collect();
            let _ = writeln!(
                out,
                "/-- Named targets `{role}`. -/\ndef {name}.{role} : Fin {} → Target :=\n  ![{}]\n",
                ts.len(),
                items.join(", ")
            );
        }
    }
    out
}

// --- The spike circuit ------------------------------------------------------------------

/// Check an assignment against the exported system, mirroring the Lean `Satisfies`:
/// arithmetic rows op-by-op (`out = c0·m0·m1 + c1·addend`), constant rows wire-by-wire,
/// every copy pair equal, every constant target at its constant. Gate kinds the Lean
/// model leaves unconstrained are skipped here too.
pub fn check_satisfied(ex: &CircuitExport, a: impl Fn(Target) -> F) -> Result<(), String> {
    let w = |row: usize, col: usize| a(Target::wire(row, col));
    for (row, (kind, consts)) in ex.rows.iter().enumerate() {
        match kind {
            GateKind::Arithmetic { num_ops } => {
                let (c0, c1) = (consts[0], consts[1]);
                for i in 0..*num_ops {
                    let expect = c0 * w(row, 4 * i) * w(row, 4 * i + 1) + c1 * w(row, 4 * i + 2);
                    if w(row, 4 * i + 3) != expect {
                        return Err(format!("arithmetic row {row} op {i} violated"));
                    }
                }
            }
            GateKind::Constant { num_consts } => {
                for (i, &c) in consts.iter().enumerate().take(*num_consts) {
                    if w(row, i) != c {
                        return Err(format!("constant row {row} wire {i} violated"));
                    }
                }
            }
            GateKind::PublicInput | GateKind::Noop | GateKind::Other(_) => {}
        }
    }
    for (x, y) in &ex.copies {
        if a(*x) != a(*y) {
            return Err(format!("copy {x:?} = {y:?} violated"));
        }
    }
    for (t, c) in &ex.constants {
        if a(*t) != *c {
            return Err(format!("constant target {t:?} = {c} violated"));
        }
    }
    Ok(())
}

// --- The spike circuit ------------------------------------------------------------------

/// Targets of the nullifier-select circuit, in slot-major order (`4*s + j`).
#[derive(Debug, Clone)]
pub struct NullifierSelectTargets {
    pub is_dummy: Vec<Target>,
    pub dnull: Vec<Target>,
    pub real: Vec<Target>,
    pub out: Vec<Target>,
}

impl NullifierSelectTargets {
    fn named(&self) -> Vec<(String, Vec<Target>)> {
        vec![
            ("isDummy".to_string(), self.is_dummy.clone()),
            ("dnull".to_string(), self.dnull.clone()),
            ("real".to_string(), self.real.clone()),
            ("out".to_string(), self.out.clone()),
        ]
    }
}

/// The private-batch nullifier-selection path for `n` slots, built with the **same builder
/// operations** the wrapper uses (`add_virtual_bool_target_safe`, `select`,
/// `register_public_inputs`): per slot, `out_j = select(is_dummy, dnull_j, real_j)` for the
/// four limbs, the outputs registered as public inputs in slot order.
pub fn build_nullifier_select(n: usize) -> (CircuitBuilder<F, D>, NullifierSelectTargets) {
    use plonky2::plonk::circuit_data::CircuitConfig;

    let config = CircuitConfig::standard_recursion_config();
    let mut b = CircuitBuilder::<F, D>::new(config);

    let mut t = NullifierSelectTargets {
        is_dummy: Vec::with_capacity(n),
        dnull: Vec::with_capacity(4 * n),
        real: Vec::with_capacity(4 * n),
        out: Vec::with_capacity(4 * n),
    };
    for _ in 0..n {
        let is_dummy = b.add_virtual_bool_target_safe();
        let dnull: [Target; 4] = core::array::from_fn(|_| b.add_virtual_target());
        let real: [Target; 4] = core::array::from_fn(|_| b.add_virtual_target());
        let out: [Target; 4] = core::array::from_fn(|j| b.select(is_dummy, dnull[j], real[j]));
        b.register_public_inputs(&out);
        t.is_dummy.push(is_dummy.target);
        t.dnull.extend_from_slice(&dnull);
        t.real.extend_from_slice(&real);
        t.out.extend_from_slice(&out);
    }
    (b, t)
}

/// Export of [`build_nullifier_select`].
pub fn nullifier_select_circuit(n: usize) -> CircuitExport {
    let (b, t) = build_nullifier_select(n);
    export(&b, t.named())
}

/// Build `formal/Plonky2Spec/Generated/NullifierSelectCircuit.lean`.
pub fn generate_nullifier_select_lean() -> String {
    let ex = nullifier_select_circuit(2);
    let mut out = String::new();
    out.push_str(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Produced by `qp-plonky2-constraint-exporter` by building the private-batch\n\
         \x20 nullifier-selection path for 2 slots with the real `CircuitBuilder` and walking its\n\
         \x20 pre-`build` constraint system (`CircuitBuilder::formal_export_view`): placed gate\n\
         \x20 rows with their constants, copy constraints, constant targets, and the public-input\n\
         \x20 registration order. Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\n\
         \x20 `Bridges/CircuitBridge.lean` proves that every assignment satisfying this system\n\
         \x20 decodes each public input to `bselect is_dummy dnull real` — the wiring-level\n\
         \x20 counterpart of the hand-stated decode hypotheses in `Plonky2Bridge`.\n\
         -/\n\
         import Plonky2Spec.Wiring\n\n\
         namespace Plonky2Spec.Generated\n\n\
         open Plonky2Spec.Wiring\n\n",
    );
    out.push_str(&render_lean(
        "nullifierSelect2",
        "Nullifier selection for 2 slots: `out[4s+j] = select(isDummy[s], dnull[4s+j], real[4s+j])`, \
         each `isDummy[s]` an `assert_bool`ed virtual target, outputs registered as public inputs.",
        &ex,
    ));
    out.push_str("end Plonky2Spec.Generated\n");
    out
}

#[cfg(test)]
mod tests {
    use plonky2::field::types::{Field, Sample};
    use plonky2::iop::generator::generate_partial_witness;
    use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
    use plonky2::plonk::config::PoseidonGoldilocksConfig;

    use super::*;

    /// Wires no generator touched are zero-filled by the prover (`PartitionWitness::full_witness`).
    fn val(w: &impl Witness<F>, t: Target) -> F {
        w.try_get_target(t).unwrap_or(F::ZERO)
    }

    #[test]
    fn classify_parses_gate_ids() {
        assert_eq!(
            classify("ArithmeticGate { num_ops: 20 }"),
            GateKind::Arithmetic { num_ops: 20 }
        );
        assert_eq!(
            classify("ConstantGate { num_consts: 2 }"),
            GateKind::Constant { num_consts: 2 }
        );
        assert_eq!(classify("PublicInputGate"), GateKind::PublicInput);
        assert_eq!(classify("NoopGate"), GateKind::Noop);
        assert!(matches!(
            classify("BaseSumGate { num_limbs: 32 } + Base: 2"),
            GateKind::Other(_)
        ));
    }

    #[test]
    fn nullifier_select_export_shape() {
        let ex = nullifier_select_circuit(2);
        // 2 assert_bool ops + 16 select ops (2 arithmetic ops each) = 18 ops, all with
        // constants (1, -1), so they share a single 20-op row.
        assert_eq!(ex.rows.len(), 1);
        assert_eq!(ex.rows[0].0, GateKind::Arithmetic { num_ops: 20 });
        assert_eq!(ex.rows[0].1, vec![F::ONE, F::NEG_ONE]);
        assert_eq!(ex.public_inputs.len(), 8);
        assert_eq!(ex.constants, vec![(ex.constants[0].0, F::ZERO)]);
        assert!(ex.copies.len() >= 18 * 3);
    }

    /// The exported system is satisfied by every witness the *real* prover generates for
    /// the circuit, and the public inputs decode as `select`. This pins the export to the
    /// live builder: a missed copy constraint, a wrong wire index, or a misread constant
    /// would fail on a random witness.
    #[test]
    fn nullifier_select_export_satisfied_by_real_witness() {
        let (b, t) = build_nullifier_select(2);
        let ex = export(&b, t.named());
        let data = b.build::<PoseidonGoldilocksConfig>();

        let mut seed = 0x5eed_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        for _ in 0..50 {
            let mut pw = PartialWitness::new();
            let flags: Vec<bool> = t.is_dummy.iter().map(|_| next() & 1 == 1).collect();
            let dn: Vec<F> = t.dnull.iter().map(|_| F::rand()).collect();
            let re: Vec<F> = t.real.iter().map(|_| F::rand()).collect();
            for (s, &f) in flags.iter().enumerate() {
                pw.set_target(t.is_dummy[s], F::from_bool(f)).unwrap();
            }
            pw.set_target_arr(&t.dnull, &dn).unwrap();
            pw.set_target_arr(&t.real, &re).unwrap();

            let w = generate_partial_witness(pw, &data.prover_only, &data.common).unwrap();
            check_satisfied(&ex, |tg| val(&w, tg)).unwrap();

            for (k, &o) in t.out.iter().enumerate() {
                let expect = if flags[k / 4] { dn[k] } else { re[k] };
                assert_eq!(w.get_target(o), expect, "output {k}");
                assert_eq!(ex.public_inputs[k], o);
            }
        }
    }

    /// `check_satisfied` is not vacuous: perturbing a public input breaks it.
    #[test]
    fn check_satisfied_rejects_bad_assignment() {
        let (b, t) = build_nullifier_select(1);
        let ex = export(&b, t.named());
        let data = b.build::<PoseidonGoldilocksConfig>();
        let mut pw = PartialWitness::new();
        pw.set_target(t.is_dummy[0], F::ONE).unwrap();
        pw.set_target_arr(&t.dnull, &[F::ONE; 4]).unwrap();
        pw.set_target_arr(&t.real, &[F::TWO; 4]).unwrap();
        let w = generate_partial_witness(pw, &data.prover_only, &data.common).unwrap();
        check_satisfied(&ex, |tg| val(&w, tg)).unwrap();
        let bad = t.out[2];
        assert!(check_satisfied(&ex, |tg| if tg == bad { F::TWO } else { val(&w, tg) }).is_err());
    }
}
