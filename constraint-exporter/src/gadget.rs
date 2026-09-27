//! Gadget-call recording and decode-proof skeleton generation (PLAN.md Step 8b).
//!
//! `Recorder` wraps a `CircuitBuilder` and mirrors the builder gadgets the wormhole wrapper
//! uses (`select`, `is_equal`, `range_check`, `not`/`and`/`or`, `add`/`sub`/`mul`,
//! `add_virtual_bool_target_safe`, `connect`). Each call records the *fact* it denotes
//! (`Fact`) together with the rows and copy constraints the builder actually emitted for
//! it, read off `formal_export_view` before and after the call — so folding and
//! memoisation inside `CircuitBuilder::arithmetic` are observed, not assumed.
//!
//! `render_decode_theorem` then emits a Lean theorem `∀ a, Satisfies circuit a → fact₁ ∧ …`
//! whose proof is generated at *gadget-call* granularity: every used arithmetic op is
//! instantiated once (`arithEq_of_rows`), copy constraints are oriented so gate input wires
//! rewrite to the targets connected to them, and each fact is closed by one fixed
//! tactic block per fact kind using only the ops internal to that call
//! (`Plonky2Spec/WiringGadgets.lean`).

use core::fmt::Write as _;
use core::ops::Range;
use std::collections::{BTreeSet, HashMap, HashSet};

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;

use crate::circuit::{export, lean_target, CircuitExport, GateKind};

type F = GoldilocksField;
const D: usize = 2;

/// The Lean fact a recorded gadget call establishes about the assignment `a`.
#[derive(Debug, Clone)]
pub enum Fact {
    /// `a out = bselect (a b) (a x) (a y)`
    Select {
        b: Target,
        x: Target,
        y: Target,
        out: Target,
    },
    /// `a out = bnot (a b)`
    Not { b: Target, out: Target },
    /// `a out = band (a b1) (a b2)`
    And { b1: Target, b2: Target, out: Target },
    /// `a out = bor (a b1) (a b2)`
    Or { b1: Target, b2: Target, out: Target },
    /// `a out = a x + a y`
    Add { x: Target, y: Target, out: Target },
    /// `a out = a x - a y`
    Sub { x: Target, y: Target, out: Target },
    /// `a out = a x * a y`
    Mul { x: Target, y: Target, out: Target },
    /// `IsBool (a b)`
    AssertBool { b: Target },
    /// `IsEqual (a x) (a y) (a equal) (a inv)`
    IsEqual {
        x: Target,
        y: Target,
        equal: Target,
        inv: Target,
    },
    /// `rangeCheck (a x) bits`
    RangeCheck { x: Target, bits: usize },
    /// `a x = a y`
    Connect { x: Target, y: Target },
}

impl Fact {
    /// Targets the fact is stated on; the generated proof never rewrites these away.
    fn named(&self) -> Vec<Target> {
        match *self {
            Fact::Select { b, x, y, out } => vec![b, x, y, out],
            Fact::Not { b, out } => vec![b, out],
            Fact::And { b1, b2, out } | Fact::Or { b1, b2, out } => vec![b1, b2, out],
            Fact::Add { x, y, out } | Fact::Sub { x, y, out } | Fact::Mul { x, y, out } => {
                vec![x, y, out]
            }
            Fact::AssertBool { b } => vec![b],
            Fact::IsEqual { x, y, equal, inv } => vec![x, y, equal, inv],
            Fact::RangeCheck { x, .. } => vec![x],
            Fact::Connect { x, y } => vec![x, y],
        }
    }

    fn lean(&self) -> String {
        let a = |t: Target| format!("a ({})", lean_target(t));
        match *self {
            Fact::Select { b, x, y, out } => {
                format!("{} = bselect ({}) ({}) ({})", a(out), a(b), a(x), a(y))
            }
            Fact::Not { b, out } => format!("{} = bnot ({})", a(out), a(b)),
            Fact::And { b1, b2, out } => format!("{} = band ({}) ({})", a(out), a(b1), a(b2)),
            Fact::Or { b1, b2, out } => format!("{} = bor ({}) ({})", a(out), a(b1), a(b2)),
            Fact::Add { x, y, out } => format!("{} = {} + {}", a(out), a(x), a(y)),
            Fact::Sub { x, y, out } => format!("{} = {} - {}", a(out), a(x), a(y)),
            Fact::Mul { x, y, out } => format!("{} = {} * {}", a(out), a(x), a(y)),
            Fact::AssertBool { b } => format!("IsBool ({})", a(b)),
            Fact::IsEqual { x, y, equal, inv } => {
                format!("IsEqual ({}) ({}) ({}) ({})", a(x), a(y), a(equal), a(inv))
            }
            Fact::RangeCheck { x, bits } => format!("rangeCheck ({}) {bits}", a(x)),
            Fact::Connect { x, y } => format!("{} = {}", a(x), a(y)),
        }
    }
}

/// One recorded gadget call.
#[derive(Debug, Clone)]
pub struct Call {
    pub fact: Fact,
    /// Rows the call added (`range_check` adds its `BaseSumGate<2>` row here).
    pub rows: Range<usize>,
    /// Copy constraints the call added.
    pub copies: Range<usize>,
}

/// A `CircuitBuilder` that records the gadget calls made through it.
pub struct Recorder {
    pub builder: CircuitBuilder<F, D>,
    pub calls: Vec<Call>,
}

impl Recorder {
    pub fn new(config: CircuitConfig) -> Self {
        Recorder {
            builder: CircuitBuilder::new(config),
            calls: Vec::new(),
        }
    }

    fn counts(&self) -> (usize, usize, usize) {
        let v = self.builder.formal_export_view();
        (
            v.gate_instances.len(),
            v.copy_constraints.len(),
            v.num_virtual_targets,
        )
    }

    /// Run `f` on the builder and record `fact(result, fresh)`, where `fresh` lists the
    /// virtual targets the call allocated that are not constants (constants such as `one`
    /// and `zero` are allocated lazily, so their position among the fresh targets varies).
    fn record<T>(
        &mut self,
        f: impl FnOnce(&mut CircuitBuilder<F, D>) -> T,
        fact: impl FnOnce(&T, &[Target]) -> Fact,
    ) -> T {
        let (rows0, copies0, virt0) = self.counts();
        let r = f(&mut self.builder);
        let (rows1, copies1, virt1) = self.counts();
        let view = self.builder.formal_export_view();
        let fresh: Vec<Target> = (virt0..virt1)
            .map(|index| Target::VirtualTarget { index })
            .filter(|t| !view.constant_targets.contains_key(t))
            .collect();
        self.calls.push(Call {
            fact: fact(&r, &fresh),
            rows: rows0..rows1,
            copies: copies0..copies1,
        });
        r
    }

    pub fn add_virtual_target(&mut self) -> Target {
        self.builder.add_virtual_target()
    }

    pub fn constant(&mut self, c: F) -> Target {
        self.builder.constant(c)
    }

    pub fn register_public_input(&mut self, t: Target) {
        self.builder.register_public_input(t);
    }

    pub fn add_virtual_bool_target_safe(&mut self) -> BoolTarget {
        self.record(
            |b| b.add_virtual_bool_target_safe(),
            |r, _| Fact::AssertBool { b: r.target },
        )
    }

    pub fn select(&mut self, b: BoolTarget, x: Target, y: Target) -> Target {
        self.record(
            |bd| bd.select(b, x, y),
            |&out, _| Fact::Select {
                b: b.target,
                x,
                y,
                out,
            },
        )
    }

    pub fn not(&mut self, b: BoolTarget) -> BoolTarget {
        self.record(
            |bd| bd.not(b),
            |r, _| Fact::Not {
                b: b.target,
                out: r.target,
            },
        )
    }

    pub fn and(&mut self, b1: BoolTarget, b2: BoolTarget) -> BoolTarget {
        self.record(
            |bd| bd.and(b1, b2),
            |r, _| Fact::And {
                b1: b1.target,
                b2: b2.target,
                out: r.target,
            },
        )
    }

    pub fn or(&mut self, b1: BoolTarget, b2: BoolTarget) -> BoolTarget {
        self.record(
            |bd| bd.or(b1, b2),
            |r, _| Fact::Or {
                b1: b1.target,
                b2: b2.target,
                out: r.target,
            },
        )
    }

    pub fn add(&mut self, x: Target, y: Target) -> Target {
        self.record(|bd| bd.add(x, y), |&out, _| Fact::Add { x, y, out })
    }

    pub fn sub(&mut self, x: Target, y: Target) -> Target {
        self.record(|bd| bd.sub(x, y), |&out, _| Fact::Sub { x, y, out })
    }

    pub fn mul(&mut self, x: Target, y: Target) -> Target {
        self.record(|bd| bd.mul(x, y), |&out, _| Fact::Mul { x, y, out })
    }

    /// `is_equal` allocates two non-constant virtual targets, `equal` and `inv`
    /// (arithmetic.rs:373-375).
    pub fn is_equal(&mut self, x: Target, y: Target) -> BoolTarget {
        self.record(
            |bd| bd.is_equal(x, y),
            |r, fresh| {
                let inv: Vec<Target> = fresh.iter().copied().filter(|t| *t != r.target).collect();
                assert_eq!(
                    inv.len(),
                    1,
                    "is_equal allocates exactly one auxiliary target"
                );
                Fact::IsEqual {
                    x,
                    y,
                    equal: r.target,
                    inv: inv[0],
                }
            },
        )
    }

    pub fn range_check(&mut self, x: Target, bits: usize) {
        self.record(
            |bd| bd.range_check(x, bits),
            |_, _| Fact::RangeCheck { x, bits },
        );
    }

    pub fn connect(&mut self, x: Target, y: Target) {
        self.record(|bd| bd.connect(x, y), |_, _| Fact::Connect { x, y });
    }

    pub fn export(&self, named: Vec<(String, Vec<Target>)>) -> Result<CircuitExport, String> {
        export(&self.builder, named)
    }
}

// --- Proof skeleton -------------------------------------------------------------------------

fn wire(t: Target) -> Option<(usize, usize)> {
    match t {
        Target::Wire(w) => Some((w.row, w.column)),
        Target::VirtualTarget { .. } => None,
    }
}

/// Op-level view of an export: which `(row, op)` slots are in use, what each input wire is
/// connected to, and which output wires are pinned to a constant.
struct Ops<'a> {
    ex: &'a CircuitExport,
    /// Used ops, in first-copy order.
    used: Vec<(usize, usize)>,
    /// Input wire `(row, col)` → the target `connect`ed to it (`copies` index).
    input_source: HashMap<(usize, usize), (Target, usize)>,
    /// Output wire `(row, col)` → every copy pinning it to a constant target, in copy order.
    pinned: HashMap<(usize, usize), Vec<usize>>,
    /// Constant target → its `constants` index.
    const_idx: HashMap<Target, usize>,
}

/// A copy in a call's range that pins a target to a constant.
#[derive(Debug, Clone, Copy)]
enum Check {
    /// A pinned arithmetic op output: after orientation its equation reads `0 = RHS`
    /// (more precisely `v = RHS` for the constant's value `v`).
    Op((usize, usize)),
    /// A non-constant virtual target pinned by copy `ci` (`fwd`: the copy is
    /// `(target, constant)`), to constant number `k`.
    Virt { ci: usize, fwd: bool, k: usize },
    /// A constant pinned to a constant: carries nothing.
    Const,
}

impl<'a> Ops<'a> {
    fn new(ex: &'a CircuitExport) -> Self {
        let const_idx: HashMap<Target, usize> = ex
            .constants
            .iter()
            .enumerate()
            .map(|(i, (t, _))| (*t, i))
            .collect();
        let is_arith_wire = |t: Target| -> Option<(usize, usize)> {
            let (row, col) = wire(t)?;
            matches!(ex.rows.get(row), Some((GateKind::Arithmetic { .. }, _))).then_some((row, col))
        };
        let mut used = Vec::new();
        let mut seen = HashSet::new();
        let mut input_source = HashMap::new();
        let mut pinned: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (ci, &(x, y)) in ex.copies.iter().enumerate() {
            for (t, other) in [(x, y), (y, x)] {
                if let Some((row, col)) = is_arith_wire(t) {
                    if seen.insert((row, col / 4)) {
                        used.push((row, col / 4));
                    }
                    if col % 4 != 3 {
                        input_source.insert((row, col), (other, ci));
                    } else if const_idx.contains_key(&other) {
                        pinned.entry((row, col)).or_default().push(ci);
                    }
                }
            }
        }
        Ops {
            ex,
            used,
            input_source,
            pinned,
            const_idx,
        }
    }

    fn op_of_output(&self, t: Target) -> Option<(usize, usize)> {
        let (row, col) = wire(t)?;
        (col % 4 == 3
            && matches!(
                self.ex.rows.get(row),
                Some((GateKind::Arithmetic { .. }, _))
            ))
        .then_some((row, col / 4))
    }

    fn is_pinned(&self, op: (usize, usize)) -> bool {
        self.pinned.contains_key(&(op.0, 4 * op.1 + 3))
    }

    /// Ops whose outputs are reachable from `roots` through op inputs, not crossing `stop`
    /// targets and not entering ops whose output is pinned to a constant: the global
    /// rewrite already replaced such outputs by the constant's value everywhere, so their
    /// equations are checks, not definitions.
    fn internal_defs(&self, roots: &[Target], stop: &[Target]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        let mut stack: Vec<Target> = roots.to_vec();
        while let Some(t) = stack.pop() {
            if stop.contains(&t) {
                continue;
            }
            let Some((row, i)) = self.op_of_output(t) else {
                continue;
            };
            if self.is_pinned((row, i)) || !seen.insert((row, i)) {
                continue;
            }
            out.push((row, i));
            self.push_inputs((row, i), &mut stack);
        }
        out
    }

    fn push_inputs(&self, (row, i): (usize, usize), stack: &mut Vec<Target>) {
        for col in [4 * i, 4 * i + 1, 4 * i + 2] {
            if let Some((src, _)) = self.input_source.get(&(row, col)) {
                stack.push(*src);
            }
        }
    }

    /// Input sources of an op.
    fn inputs(&self, op: (usize, usize)) -> Vec<Target> {
        let mut v = Vec::new();
        self.push_inputs(op, &mut v);
        v
    }

    /// The constant-pinning copies in a call's copy range, in copy order. Discovered from
    /// the copies themselves, so repeated pins of one output by later calls cannot hide
    /// earlier evidence, and checks that constant-folded to a virtual target or to a
    /// constant are still seen.
    fn checks_in(&self, copies: &Range<usize>) -> Vec<Check> {
        let mut v = Vec::new();
        for ci in copies.clone() {
            let (x, y) = self.ex.copies[ci];
            let (t, fwd, k) = match (self.const_idx.get(&x), self.const_idx.get(&y)) {
                (Some(_), Some(_)) => {
                    v.push(Check::Const);
                    continue;
                }
                (None, Some(&k)) => (x, true, k),
                (Some(&k), None) => (y, false, k),
                (None, None) => continue,
            };
            if let Some(op) = self.op_of_output(t) {
                v.push(Check::Op(op));
            } else if matches!(t, Target::VirtualTarget { .. }) {
                v.push(Check::Virt { ci, fwd, k });
            }
            // A gate input wire connected to a constant is an operand feed, not a check.
        }
        v
    }

    /// Rewrites that bring the goal's named targets into the form the oriented equations
    /// use: constants to their values, pinned outputs to the constant then its value.
    fn goal_rules(&self, named: &[Target]) -> Vec<String> {
        let mut rules = Vec::new();
        for &t in named {
            if let Some(&k) = self.const_idx.get(&t) {
                rules.push(format!("k{k}"));
            } else if let Some(op) = self.op_of_output(t) {
                if let Some(cis) = self.pinned.get(&(op.0, 4 * op.1 + 3)) {
                    let ci = cis[0];
                    let (x, y) = self.ex.copies[ci];
                    let (rule, konst) = if self.const_idx.contains_key(&x) {
                        (format!("← c{ci}"), x)
                    } else {
                        (format!("c{ci}"), y)
                    };
                    rules.push(rule);
                    rules.push(format!("k{}", self.const_idx[&konst]));
                }
            }
        }
        rules.dedup();
        rules
    }
}

fn e(op: (usize, usize)) -> String {
    format!("e_{}_{}", op.0, op.1)
}

fn simp_only(rules: &[String], at: Option<&str>) -> String {
    if rules.is_empty() {
        return String::new();
    }
    match at {
        Some(h) => format!("simp only [{}] at {h}\n", rules.join(", ")),
        None => format!("simp only [{}]\n", rules.join(", ")),
    }
}

/// Tactic script proving one zero-check from a `Check`: for an op, its oriented equation
/// with the call-internal definitions substituted, closed by `linear_combination`; for a
/// pinned virtual target, rewrite it to the constant's value and `ring`; for a
/// constant-to-constant copy, `ring` alone.
fn check_script(ops: &Ops, chk: Check, stop: &[Target], indent: &str) -> String {
    let mut s = String::new();
    match chk {
        Check::Op(op) => {
            let defs: Vec<String> = ops
                .internal_defs(&ops.inputs(op), stop)
                .into_iter()
                .map(e)
                .collect();
            let _ = writeln!(s, "{indent}have hc := {}", e(op));
            s.push_str(&indent_lines(&simp_only(&defs, Some("hc")), indent));
            let _ = writeln!(s, "{indent}linear_combination -hc");
        }
        Check::Virt { ci, fwd, k } => {
            let c = if fwd {
                format!("c{ci}")
            } else {
                format!("c{ci}.symm")
            };
            let _ = writeln!(s, "{indent}rw [{c}.trans k{k}]");
            let _ = writeln!(s, "{indent}ring");
        }
        Check::Const => {
            let _ = writeln!(s, "{indent}ring");
        }
    }
    s
}

fn indent_lines(block: &str, indent: &str) -> String {
    block.lines().map(|l| format!("{indent}{l}\n")).collect()
}

/// Render `theorem <name>_decode (a) (h : Satisfies (<name> p) a) : fact₁ ∧ … := by …`.
pub fn render_decode_theorem(name: &str, ex: &CircuitExport, calls: &[Call]) -> String {
    let ops = Ops::new(ex);
    let mut out = String::new();
    let facts: Vec<String> = calls.iter().map(|c| c.fact.lean()).collect();
    let _ = writeln!(
        out,
        "/-- Every satisfying assignment of `{name}` has the meaning of each recorded gadget \
         call. Generated at gadget-call granularity; see `gadget.rs`. -/"
    );
    let _ = writeln!(
        out,
        "theorem {name}_decode (a : Assignment p) (h : Satisfies ({name} p) a) :"
    );
    for (i, f) in facts.iter().enumerate() {
        let sep = if i + 1 == facts.len() {
            " := by"
        } else {
            " ∧"
        };
        let _ = writeln!(out, "    {f}{sep}");
    }
    // Copies and constants, by position.
    out.push_str("  have hcopy := h.2.1\n  have hconst := h.2.2\n");
    let _ = writeln!(
        out,
        "  simp only [{name}, List.forall_mem_cons, List.not_mem_nil, false_implies, \
         implies_true, and_true] at hcopy hconst"
    );
    let names = |prefix: &str, n: usize| -> String {
        (0..n)
            .map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if ex.copies.len() > 1 {
        let _ = writeln!(out, "  obtain ⟨{}⟩ := hcopy", names("c", ex.copies.len()));
    } else if ex.copies.len() == 1 {
        out.push_str("  have c0 := hcopy\n");
    }
    if ex.constants.len() > 1 {
        let _ = writeln!(
            out,
            "  obtain ⟨{}⟩ := hconst",
            names("k", ex.constants.len())
        );
    } else if ex.constants.len() == 1 {
        out.push_str("  have k0 := hconst\n");
    }
    // Every used op, as its wire equation.
    for &(row, i) in &ops.used {
        let _ = writeln!(
            out,
            "  have {} := arithEq_of_rows h (row := {row}) (i := {i}) rfl (by norm_num)",
            e((row, i))
        );
    }
    let all_e: Vec<String> = ops.used.iter().map(|&op| e(op)).collect();
    if !all_e.is_empty() {
        let _ = writeln!(
            out,
            "  norm_num only [Nat.reduceMul, Nat.reduceAdd] at {}",
            all_e.join(" ")
        );
        // Orient: gate input wires → their sources; pinned outputs → the constant; constants
        // → their values.
        let mut rules: Vec<String> = Vec::new();
        let mut in_copies: Vec<usize> = ops.input_source.values().map(|(_, ci)| *ci).collect();
        in_copies.sort();
        in_copies.dedup();
        for ci in in_copies {
            let (x, _) = ex.copies[ci];
            // `connect(source, wire)`: rewrite `a wire` into `a source`.
            rules.push(
                if wire(x).is_some_and(|(r, c)| {
                    matches!(ex.rows.get(r), Some((GateKind::Arithmetic { .. }, _))) && c % 4 != 3
                }) {
                    format!("c{ci}")
                } else {
                    format!("← c{ci}")
                },
            );
        }
        let mut pin_copies: Vec<usize> = ops.pinned.values().flatten().copied().collect();
        pin_copies.sort();
        for ci in pin_copies {
            let (x, _) = ex.copies[ci];
            rules.push(if ops.const_idx.contains_key(&x) {
                format!("← c{ci}")
            } else {
                format!("c{ci}")
            });
        }
        for i in 0..ex.constants.len() {
            rules.push(format!("k{i}"));
        }
        let _ = writeln!(
            out,
            "  simp only [{}] at {}",
            rules.join(", "),
            all_e.join(" ")
        );
    }
    // One block per call.
    for (n, call) in calls.iter().enumerate() {
        let f = &call.fact;
        let stop = f.named();
        let _ = writeln!(out, "  have f{n} : {} := by", f.lean());
        match *f {
            Fact::Select { out: o, .. }
            | Fact::Not { out: o, .. }
            | Fact::And { out: o, .. }
            | Fact::Or { out: o, .. }
            | Fact::Add { out: o, .. }
            | Fact::Sub { out: o, .. }
            | Fact::Mul { out: o, .. } => {
                let spec = match f {
                    Fact::Select { .. } => Some("bselect"),
                    Fact::Not { .. } => Some("bnot"),
                    Fact::And { .. } => Some("band"),
                    Fact::Or { .. } => Some("bor"),
                    _ => None,
                };
                let mut goal: Vec<String> = spec.map(|s| s.to_string()).into_iter().collect();
                goal.extend(ops.goal_rules(&stop));
                // The output's own equation (if the builder placed an op for it; a folded
                // call has none and the fact is an identity after the goal rewrites),
                // with the definitions internal to this call substituted.
                match ops.op_of_output(o) {
                    Some(root) => {
                        let inner: Vec<Target> = ops.inputs(root);
                        let defs: Vec<String> = ops
                            .internal_defs(&inner, &stop)
                            .into_iter()
                            .map(e)
                            .collect();
                        let _ = writeln!(out, "    have hr := {}", e(root));
                        out.push_str(&indent_lines(&simp_only(&defs, Some("hr")), "    "));
                        out.push_str(&indent_lines(&simp_only(&goal, None), "    "));
                        out.push_str("    linear_combination hr\n");
                    }
                    None => {
                        out.push_str(&indent_lines(&simp_only(&goal, None), "    "));
                        out.push_str("    ring\n");
                    }
                }
            }
            Fact::AssertBool { .. } => {
                let checks = ops.checks_in(&call.copies);
                let [chk] = checks[..] else {
                    panic!("assert_bool pins exactly one target: {checks:?}")
                };
                out.push_str(&indent_lines(
                    &simp_only(&ops.goal_rules(&stop), None),
                    "    ",
                ));
                out.push_str("    refine isBool_iff_assertBool.mpr ?_\n");
                out.push_str(&check_script(&ops, chk, &stop, "    "));
            }
            Fact::IsEqual { .. } => {
                let checks = ops.checks_in(&call.copies);
                // `connect(not_equal_check, zero)` then `connect(equal_check, zero)`
                // (arithmetic.rs), each possibly constant-folded.
                let [ne, eq] = checks[..] else {
                    panic!("is_equal pins exactly two targets: {checks:?}")
                };
                out.push_str(&indent_lines(
                    &simp_only(&ops.goal_rules(&stop), None),
                    "    ",
                ));
                out.push_str("    refine ⟨?_, ?_⟩\n");
                for chk in [ne, eq] {
                    let script = check_script(&ops, chk, &stop, "      ");
                    let mut lines = script.lines();
                    if let Some(first) = lines.next() {
                        let _ = writeln!(out, "    · {}", first.trim_start());
                    }
                    for l in lines {
                        let _ = writeln!(out, "{l}");
                    }
                }
            }
            Fact::RangeCheck { bits, .. } => {
                assert_eq!(
                    call.rows.len(),
                    1,
                    "range_check ≤ num_limbs bits places one row"
                );
                let row = call.rows.start;
                let GateKind::BaseSum2 { num_limbs } = ex.rows[row].0 else {
                    panic!(
                        "range_check row {row} is not BaseSumGate<2>: {:?}",
                        ex.rows[row].0
                    )
                };
                // Zero-pinned limbs `bits..num_limbs`, and the sum-wire copy, from this call.
                let zero_t = ex
                    .constants
                    .iter()
                    .find(|(_, c)| *c == F::ZERO)
                    .map(|(t, _)| *t)
                    .expect("range_check pins limbs to the zero constant");
                let kz = ops.const_idx[&zero_t];
                let mut limb_copy: HashMap<usize, (usize, bool)> = HashMap::new();
                let mut sum_copy: Option<(usize, bool)> = None;
                for ci in call.copies.clone() {
                    let (x, y) = ex.copies[ci];
                    for (t, other, fwd) in [(x, y, true), (y, x, false)] {
                        if let Some((r, col)) = wire(t) {
                            if r == row {
                                if col == 0 {
                                    sum_copy = Some((ci, fwd));
                                } else if other == zero_t {
                                    limb_copy.insert(col - 1, (ci, fwd));
                                }
                            }
                        }
                    }
                }
                let (sci, sfwd) = sum_copy.expect("range_check connects the sum wire");
                let _ = writeln!(
                    out,
                    "    have hr := rangeCheck_of_row h (row := {row}) (N := {num_limbs}) \
                     (n := {bits}) rfl rfl (by norm_num) (by"
                );
                out.push_str("      intro i hi1 hi2\n      interval_cases i\n");
                for i in bits..num_limbs {
                    let (ci, fwd) = limb_copy
                        .get(&i)
                        .unwrap_or_else(|| panic!("limb {i} of row {row} is not pinned to zero"));
                    let _ = writeln!(
                        out,
                        "      · exact {}.trans k{kz}",
                        if *fwd {
                            format!("c{ci}")
                        } else {
                            format!("c{ci}.symm")
                        }
                    );
                }
                out.push_str("      )\n");
                let _ = writeln!(
                    out,
                    "    rwa [{}] at hr",
                    if sfwd {
                        format!("c{sci}")
                    } else {
                        format!("← c{sci}")
                    }
                );
            }
            Fact::Connect { .. } => {
                assert_eq!(call.copies.len(), 1, "connect adds one copy");
                let _ = writeln!(out, "    exact c{}", call.copies.start);
            }
        }
    }
    let fs: Vec<String> = (0..calls.len()).map(|i| format!("f{i}")).collect();
    let _ = writeln!(out, "  exact ⟨{}⟩", fs.join(", "));
    out
}

// --- The gadget-zoo spike circuit -----------------------------------------------------------

/// Named targets of the gadget zoo.
#[derive(Debug, Clone)]
pub struct GadgetZooTargets {
    pub x: Target,
    pub y: Target,
    pub fee: Target,
    pub flag: Target,
    pub eq: Target,
    pub sel: Target,
    pub head: Target,
}

/// The wrapper's gadget mix on a handful of targets, recorded through `Recorder`:
/// `assert_bool`, `is_equal`, `select`, `or`, `not`, `and`, `sub` against a constant,
/// `range_check(·, 14)`, a wire-to-wire `connect`, and two public inputs.
pub fn build_gadget_zoo() -> (Recorder, GadgetZooTargets) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    let fee = r.add_virtual_target();
    let flag = r.add_virtual_bool_target_safe();
    let eq = r.is_equal(x, y);
    let sel = r.select(flag, x, y);
    let either = r.or(eq, flag);
    let nflag = r.not(flag);
    let both = r.and(eq, nflag);
    let ten_thousand = r.constant(F::from_canonical_u64(10_000));
    let head = r.sub(ten_thousand, fee);
    r.range_check(head, 14);
    r.connect(sel, either.target);
    r.register_public_input(x);
    r.register_public_input(sel);
    r.register_public_input(both.target);
    let t = GadgetZooTargets {
        x,
        y,
        fee,
        flag: flag.target,
        eq: eq.target,
        sel,
        head,
    };
    (r, t)
}

impl GadgetZooTargets {
    fn named(&self) -> Vec<(String, Vec<Target>)> {
        vec![
            ("x".to_string(), vec![self.x]),
            ("y".to_string(), vec![self.y]),
            ("fee".to_string(), vec![self.fee]),
            ("flag".to_string(), vec![self.flag]),
            ("eq".to_string(), vec![self.eq]),
            ("sel".to_string(), vec![self.sel]),
            ("head".to_string(), vec![self.head]),
        ]
    }
}

/// Build `formal/Plonky2Spec/Generated/GadgetZooCircuit.lean`: the export plus the
/// generated decode theorem.
pub fn generate_gadget_zoo_lean() -> String {
    let (r, t) = build_gadget_zoo();
    let ex = r.export(t.named()).expect("gadget zoo has no lookups");
    render_module(
        "gadget-zoo",
        "gadgetZoo",
        "The gadget zoo: `assert_bool flag`, `eq = is_equal x y`, `sel = select flag x y`, \
         `either = or eq flag`, `nflag = not flag`, `both = and eq nflag`, \
         `head = sub 10000 fee`, `range_check head 14`, `connect sel either`; \
         public inputs `x`, `sel`, `both`.",
        &ex,
        &r.calls,
    )
}

// --- Builder folding and re-pinning edge cases ----------------------------------------------

/// Named targets of the edge-case circuit.
#[derive(Debug, Clone)]
pub struct GadgetEdgeCasesTargets {
    pub x: Target,
    pub y: Target,
    pub zero: Target,
    pub one: Target,
    pub sum: Target,
    pub eq_zz: Target,
    pub eq_oz: Target,
    pub eq_xx: Target,
    pub eq_xy: Target,
    pub diff: Target,
    pub check: Target,
}

/// Gadget calls whose lowering differs from the plain case:
/// `sum = add x y` pinned to zero (a gadget output that is also a check),
/// `is_equal zero zero` (both checks fold: one to a constant copy, one to a real op with
/// constant operands), `is_equal one zero` (the first check folds to the `equal` target
/// itself), `is_equal x x`, and `eq_xy = is_equal x y; diff = sub x y; check = mul eq_xy
/// diff; connect check zero` (the memoized `check` op is the one `is_equal` already pinned,
/// so the same output wire is pinned twice).
pub fn build_gadget_edge_cases() -> (Recorder, GadgetEdgeCasesTargets) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    let zero = r.constant(F::ZERO);
    let one = r.constant(F::ONE);
    let sum = r.add(x, y);
    r.connect(sum, zero);
    let eq_zz = r.is_equal(zero, zero);
    let eq_oz = r.is_equal(one, zero);
    let eq_xx = r.is_equal(x, x);
    let eq_xy = r.is_equal(x, y);
    let diff = r.sub(x, y);
    let check = r.mul(eq_xy.target, diff);
    r.connect(check, zero);
    r.register_public_input(eq_zz.target);
    r.register_public_input(eq_oz.target);
    r.register_public_input(eq_xx.target);
    r.register_public_input(eq_xy.target);
    let t = GadgetEdgeCasesTargets {
        x,
        y,
        zero,
        one,
        sum,
        eq_zz: eq_zz.target,
        eq_oz: eq_oz.target,
        eq_xx: eq_xx.target,
        eq_xy: eq_xy.target,
        diff,
        check,
    };
    (r, t)
}

impl GadgetEdgeCasesTargets {
    fn named(&self) -> Vec<(String, Vec<Target>)> {
        vec![
            ("x".to_string(), vec![self.x]),
            ("y".to_string(), vec![self.y]),
            ("zero".to_string(), vec![self.zero]),
            ("one".to_string(), vec![self.one]),
            ("sum".to_string(), vec![self.sum]),
            ("eq_zz".to_string(), vec![self.eq_zz]),
            ("eq_oz".to_string(), vec![self.eq_oz]),
            ("eq_xx".to_string(), vec![self.eq_xx]),
            ("eq_xy".to_string(), vec![self.eq_xy]),
            ("diff".to_string(), vec![self.diff]),
            ("check".to_string(), vec![self.check]),
        ]
    }
}

/// Build `formal/Plonky2Spec/Generated/GadgetEdgeCasesCircuit.lean`.
pub fn generate_gadget_edge_cases_lean() -> String {
    let (r, t) = build_gadget_edge_cases();
    let ex = r.export(t.named()).expect("edge cases have no lookups");
    render_module(
        "gadget edge-case",
        "gadgetEdgeCases",
        "Builder folding and re-pinning edge cases: `sum = add x y; connect sum zero`, \
         `is_equal zero zero`, `is_equal one zero`, `is_equal x x`, \
         `eq_xy = is_equal x y; diff = sub x y; check = mul eq_xy diff; connect check zero`; \
         public inputs the four `equal` targets.",
        &ex,
        &r.calls,
    )
}

/// A complete `Plonky2Spec.Generated` module: header, the export, and the decode theorem.
fn render_module(what: &str, name: &str, doc: &str, ex: &CircuitExport, calls: &[Call]) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) by building the\n\
         \x20 {what} circuit through the recording builder and walking its pre-`build`\n\
         \x20 constraint system. The theorem's proof is generated too, one block per recorded\n\
         \x20 gadget call, from the ops and copy constraints the builder emitted for it.\n\
         \x20 Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\
         -/\n\
         import Mathlib.Tactic.IntervalCases\n\
         import Mathlib.Tactic.LinearCombination\n\
         import Plonky2Spec.WiringGadgets\n\n\
         namespace Plonky2Spec.Generated\n\n\
         open Plonky2Spec.Wiring\n\n\
         set_option linter.unusedVariables false\n\
         set_option linter.unusedSimpArgs false\n\n",
    );
    out.push_str(&crate::circuit::render_lean(name, doc, ex));
    out.push_str("variable {p : ℕ} [Fact p.Prime]\n\n");
    out.push_str(&render_decode_theorem(name, ex, calls));
    out.push_str("\nend Plonky2Spec.Generated\n");
    out
}
