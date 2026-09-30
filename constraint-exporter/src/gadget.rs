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
//! assembled from one lemma per recorded call: each lemma instantiates the arithmetic ops
//! internal to its call (`arithEq_of_rows`), with copy constraints oriented so gate input
//! wires rewrite to the targets connected to them, and closes its fact by one fixed tactic
//! block per fact kind (`Plonky2Spec/WiringGadgets.lean`).

use core::fmt::Write as _;
use core::ops::Range;
use std::collections::{BTreeSet, HashMap};

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;

use crate::circuit::{export, lean_const_int, lean_target, CircuitExport, GateKind};
use crate::symbolic::GOLDILOCKS_ORDER;

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
    /// `split_le(x, bits)`: `BaseSum 2 (a x) [a (.wire row 1), …, a (.wire row bits)]`, the
    /// first `bits` limb wires of the `BaseSumGate<2>` row at `row` being the bits.
    SplitLe { x: Target, row: usize, bits: usize },
    /// `hash_n_to_hash_no_pad_p2` on `inputs`: one `Poseidon2Gate` row per block of
    /// `pad10 inputs`, in absorption order, the last row's first four output wires being
    /// `spongeHash perm [a x0, …]`.
    Poseidon2 {
        rows: Vec<usize>,
        inputs: Vec<Target>,
    },
}

impl Fact {
    /// The row whose output wires carry a hash's digest.
    pub fn digest_row(&self) -> Option<usize> {
        match self {
            Fact::Poseidon2 { rows, .. } => rows.last().copied(),
            _ => None,
        }
    }
}

impl Fact {
    /// Targets the fact is stated on; the generated proof never rewrites these away.
    pub(crate) fn named(&self) -> Vec<Target> {
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
            Fact::SplitLe { x, row, bits } => {
                let mut v = vec![x];
                v.extend((1..=bits).map(|col| Target::wire(row, col)));
                v
            }
            Fact::Poseidon2 {
                ref rows,
                ref inputs,
            } => {
                let mut v = inputs.clone();
                let last = *rows.last().expect("a hash has a row");
                v.extend((12..16).map(|col| Target::wire(last, col)));
                v
            }
        }
    }

    pub(crate) fn is_poseidon2(&self) -> bool {
        matches!(self, Fact::Poseidon2 { .. })
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
            Fact::SplitLe { x, row, bits } => format!(
                "BaseSum 2 ({}) [{}]",
                a(x),
                (1..=bits)
                    .map(|col| format!("a (.wire {row} {col})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Fact::Poseidon2 {
                ref rows,
                ref inputs,
            } => {
                let digest = format!(
                    "spongeHash perm [{}]",
                    inputs.iter().map(|&t| a(t)).collect::<Vec<_>>().join(", ")
                );
                let last = rows.last().expect("a hash has a row");
                format!(
                    "({})",
                    (0..4)
                        .map(|i| format!("a (.wire {last} {}) = {digest} {i}", 12 + i))
                        .collect::<Vec<_>>()
                        .join(" ∧ ")
                )
            }
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

/// Op-level view of an export: which `(row, op)` slots are in use and what each input wire
/// is connected to.
struct Ops<'a> {
    ex: &'a CircuitExport,
    /// Input wire `(row, col)` → the target `connect`ed to it (`copies` index).
    input_source: HashMap<(usize, usize), (Target, usize)>,
    /// Constant target → its `constants` index.
    const_idx: HashMap<Target, usize>,
}

/// Copy `ci` pins a target to constant number `k`; `fwd` if the copy is `(target, const)`.
#[derive(Debug, Clone, Copy)]
struct Pin {
    ci: usize,
    fwd: bool,
    k: usize,
}

impl Pin {
    /// A proof of `a target = v` from the copy and the constant's value.
    fn lean(&self) -> String {
        let Pin { ci, fwd, k } = *self;
        if fwd {
            format!("c{ci}.trans k{k}")
        } else {
            format!("c{ci}.symm.trans k{k}")
        }
    }
}

/// A copy in a call's range that pins a target to a constant.
#[derive(Debug, Clone, Copy)]
enum Check {
    /// A pinned arithmetic op output: its equation and its pin together give `v = RHS`.
    Op { op: (usize, usize), pin: Pin },
    /// A non-constant virtual target pinned to a constant (a check that folded onto one of
    /// its operands).
    Virt { t: Target, pin: Pin },
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
        let mut input_source = HashMap::new();
        for (ci, &(x, y)) in ex.copies.iter().enumerate() {
            for (t, other) in [(x, y), (y, x)] {
                if let Some((row, col)) = is_arith_wire(t) {
                    if col % 4 != 3 {
                        input_source.insert((row, col), (other, ci));
                    }
                }
            }
        }
        Ops {
            ex,
            input_source,
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

    fn row_consts(&self, row: usize) -> (F, F) {
        let c = &self.ex.rows[row].1;
        (c[0], c[1])
    }

    /// What an op's input wire reads after the global orientation: its connected source,
    /// or the wire itself if nothing was connected.
    fn src(&self, row: usize, col: usize) -> Target {
        match self.input_source.get(&(row, col)) {
            Some((t, _)) => *t,
            None => Target::wire(row, col),
        }
    }

    /// Input sources of an op.
    fn inputs(&self, (row, i): (usize, usize)) -> Vec<Target> {
        (0..3).map(|j| self.src(row, 4 * i + j)).collect()
    }

    /// Ops whose outputs are reachable from `roots` through op inputs, not crossing `stop`
    /// targets. Every op is a definition of its output wire, pinned or not; a pin is a
    /// separate equation the check that owns it uses.
    fn internal_defs(&self, roots: &[Target], stop: &[Target]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        let mut stack: Vec<Target> = roots.to_vec();
        while let Some(t) = stack.pop() {
            if stop.contains(&t) {
                continue;
            }
            let Some(op) = self.op_of_output(t) else {
                continue;
            };
            if !seen.insert(op) {
                continue;
            }
            out.push(op);
            stack.extend(self.inputs(op));
        }
        out
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
            let pin = Pin { ci, fwd, k };
            if let Some(op) = self.op_of_output(t) {
                v.push(Check::Op { op, pin });
            } else if matches!(t, Target::VirtualTarget { .. }) {
                v.push(Check::Virt { t, pin });
            }
            // A gate input wire connected to a constant is an operand feed, not a check.
        }
        v
    }

    /// `k` rewrites for the named targets that are constants, so the goal reads constants
    /// as values like the oriented equations do.
    fn const_rules(&self, named: &[Target]) -> Vec<String> {
        let mut ks: Vec<usize> = named
            .iter()
            .filter_map(|t| self.const_idx.get(t).copied())
            .collect();
        ks.sort();
        ks.dedup();
        ks.into_iter().map(|k| format!("k{k}")).collect()
    }
}

/// A residue modulo the prime `m` (`m < 2^64`), with the modulus carried so the polynomial
/// closures below read like field arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct V {
    v: u64,
    m: u64,
}

impl V {
    fn of_int(n: i128, m: u64) -> V {
        V {
            v: n.rem_euclid(m as i128) as u64,
            m,
        }
    }

    fn is_zero(self) -> bool {
        self.v == 0
    }
}

impl core::ops::Add for V {
    type Output = V;
    fn add(self, o: V) -> V {
        debug_assert_eq!(self.m, o.m);
        V {
            v: ((self.v as u128 + o.v as u128) % self.m as u128) as u64,
            m: self.m,
        }
    }
}

impl core::ops::Sub for V {
    type Output = V;
    fn sub(self, o: V) -> V {
        debug_assert_eq!(self.m, o.m);
        V {
            v: ((self.v as u128 + self.m as u128 - o.v as u128) % self.m as u128) as u64,
            m: self.m,
        }
    }
}

impl core::ops::Mul for V {
    type Output = V;
    fn mul(self, o: V) -> V {
        debug_assert_eq!(self.m, o.m);
        V {
            v: ((self.v as u128 * o.v as u128) % self.m as u128) as u64,
            m: self.m,
        }
    }
}

/// Random-point evaluation of the polynomials the generated tactics equate. A tactic is
/// emitted only if the identity it relies on (`linear_combination` closes a goal iff
/// `goal.lhs - goal.rhs - (h.lhs - h.rhs)` is zero by `ring`) holds at every sampled
/// point.
///
/// The theorem is stated over `ZMod p` for an arbitrary prime `p`, with every constant
/// written as the integer `lean_const` renders it, so the identity has to hold over `ℤ`,
/// i.e. modulo every prime — not just modulo the Goldilocks order the builder folded
/// constants in. Each identity is therefore sampled modulo several unrelated primes
/// (`MODULI`); one that holds only modulo Goldilocks is a characteristic-dependent constant
/// fold and is rejected. A false positive on these low-degree polynomials is negligible,
/// and Lean re-checks the result anyway.
struct Eval<'o, 'a> {
    ops: &'o Ops<'a>,
    m: u64,
    rho: HashMap<Target, V>,
    seed: u64,
}

/// Goldilocks (the builder's field), Mersenne-61, and Baby Bear.
const MODULI: [u64; 3] = [GOLDILOCKS_ORDER, (1 << 61) - 1, 2_013_265_921];

impl<'o, 'a> Eval<'o, 'a> {
    fn new(ops: &'o Ops<'a>, m: u64, seed: u64) -> Self {
        Eval {
            ops,
            m,
            rho: HashMap::new(),
            seed,
        }
    }

    fn fresh(&mut self) -> V {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        V::of_int(self.seed as i128, self.m)
    }

    /// A Goldilocks constant, as the integer Lean sees.
    fn konst(&self, c: F) -> V {
        V::of_int(lean_const_int(c), self.m)
    }

    fn one(&self) -> V {
        V::of_int(1, self.m)
    }

    /// A target as the goal sees it: a constant's value, otherwise an atom.
    fn atom(&mut self, t: Target) -> V {
        if let Some(&k) = self.ops.const_idx.get(&t) {
            return self.konst(self.ops.ex.constants[k].1);
        }
        if let Some(&v) = self.rho.get(&t) {
            return v;
        }
        let v = self.fresh();
        self.rho.insert(t, v);
        v
    }

    /// A target after `simp only [defs]`: op outputs in `defs` unfold to their equations.
    fn expand(&mut self, t: Target, defs: &[(usize, usize)]) -> V {
        match self.ops.op_of_output(t) {
            Some(op) if defs.contains(&op) => self.op_rhs(op, defs),
            _ => self.atom(t),
        }
    }

    /// `c0 * m0 * m1 + c1 * addend` of an op, operands unfolded through `defs`.
    fn op_rhs(&mut self, (row, i): (usize, usize), defs: &[(usize, usize)]) -> V {
        let (c0, c1) = self.ops.row_consts(row);
        let (c0, c1) = (self.konst(c0), self.konst(c1));
        let m0 = self.expand(self.ops.src(row, 4 * i), defs);
        let m1 = self.expand(self.ops.src(row, 4 * i + 1), defs);
        let ad = self.expand(self.ops.src(row, 4 * i + 2), defs);
        c0 * m0 * m1 + c1 * ad
    }

    /// `lhs - rhs` of a value fact, on atoms.
    fn fact_poly(&mut self, f: &Fact) -> V {
        let one = self.one();
        let mut a = |t: Target| self.atom(t);
        match *f {
            Fact::Select { b, x, y, out } => a(out) - (a(b) * (a(x) - a(y)) + a(y)),
            Fact::Not { b, out } => a(out) - (one - a(b)),
            Fact::And { b1, b2, out } => a(out) - a(b1) * a(b2),
            Fact::Or { b1, b2, out } => a(out) - (a(b1) + a(b2) - a(b1) * a(b2)),
            Fact::Add { x, y, out } => a(out) - (a(x) + a(y)),
            Fact::Sub { x, y, out } => a(out) - (a(x) - a(y)),
            Fact::Mul { x, y, out } => a(out) - a(x) * a(y),
            Fact::Connect { x, y } => a(x) - a(y),
            Fact::AssertBool { .. }
            | Fact::IsEqual { .. }
            | Fact::RangeCheck { .. }
            | Fact::SplitLe { .. }
            | Fact::Poseidon2 { .. } => unreachable!("not a value fact"),
        }
    }
}

/// Whether an identity holds at random points modulo each of `MODULI`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Identity {
    /// Holds modulo every sampled prime: an identity over `ℤ`, provable for generic `p`.
    Holds,
    /// Holds modulo the Goldilocks order only: a constant fold the builder reduced modulo
    /// its field, not an identity of the rendered integers.
    GoldilocksOnly,
    /// Fails modulo Goldilocks too.
    Fails,
}

fn identity(ops: &Ops, mut f: impl FnMut(&mut Eval) -> bool) -> Identity {
    let per_modulus = MODULI.map(|m| {
        (1..=3u64).all(|seed| {
            f(&mut Eval::new(
                ops,
                m,
                seed.wrapping_mul(0x9e37_79b9_7f4a_7c15),
            ))
        })
    });
    match (per_modulus[0], per_modulus[1..].iter().all(|ok| *ok)) {
        (true, true) => Identity::Holds,
        (true, false) => Identity::GoldilocksOnly,
        (false, _) => Identity::Fails,
    }
}

/// Panics unless `id` is `Holds`; `fail` names the evidence in the `Fails` message.
fn reject_unless_holds(what: &str, fail: &str, id: Identity) {
    match id {
        Identity::Holds => {}
        Identity::GoldilocksOnly => panic!(
            "{what}: holds only modulo the Goldilocks order (a constant fold reduced in the \
             builder's field); the theorem is stated for a generic prime, so this circuit \
             cannot be decoded generically"
        ),
        Identity::Fails => panic!("{what}: {fail}"),
    }
}

/// Panics unless `f` is an identity over `ℤ`.
fn require_identity(ops: &Ops, what: &str, fail: &str, f: impl FnMut(&mut Eval) -> bool) {
    reject_unless_holds(what, fail, identity(ops, f));
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

fn indent_lines(block: &str, indent: &str) -> String {
    block.lines().map(|l| format!("{indent}{l}\n")).collect()
}

/// Tactic script closing a goal `g = 0` (its `lhs - rhs` computed by `goal`) from one
/// `Check`. For a pinned op, its equation with the call-internal definitions substituted,
/// combined with the pin; for a pinned virtual target, rewrite it to the constant's value
/// and `ring`; for a constant-to-constant copy, `ring` alone. Panics if the identity the
/// script relies on does not hold, naming the fact.
fn check_script(
    ops: &Ops,
    chk: Check,
    stop: &[Target],
    goal: &dyn Fn(&mut Eval) -> V,
    what: &str,
    indent: &str,
) -> String {
    let mut s = String::new();
    match chk {
        Check::Op { op, pin } => {
            let defs = ops.internal_defs(&ops.inputs(op), stop);
            let c = ops.ex.constants[pin.k].1;
            require_identity(
                ops,
                what,
                &format!("pinned op {op:?} does not establish the check"),
                |ev| goal(ev) == ev.op_rhs(op, &defs) - ev.konst(c),
            );
            let defs: Vec<String> = defs.into_iter().map(e).collect();
            let _ = writeln!(s, "{indent}have hc := {}", e(op));
            s.push_str(&indent_lines(&simp_only(&defs, Some("hc")), indent));
            let _ = writeln!(s, "{indent}linear_combination {} - hc", pin.lean());
        }
        Check::Virt { t, pin } => {
            let c = ops.ex.constants[pin.k].1;
            require_identity(
                ops,
                what,
                &format!("pinning {t:?} does not establish the check"),
                |ev| {
                    let v = ev.konst(c);
                    ev.rho.insert(t, v);
                    goal(ev).is_zero()
                },
            );
            let _ = writeln!(s, "{indent}rw [{}]", pin.lean());
            let _ = writeln!(s, "{indent}ring");
        }
        Check::Const => {
            require_identity(
                ops,
                what,
                "constant-folded check is not an identity",
                |ev| goal(ev).is_zero(),
            );
            let _ = writeln!(s, "{indent}ring");
        }
    }
    s
}

/// Bind `prefix<i>` to item `i` of a rendered list field (`hyp := proj` is the
/// `∀ x ∈ list, …` hypothesis), flat or chunked as `render_lean` laid it out.
fn destructure(
    out: &mut String,
    name: &str,
    field: &str,
    hyp: &str,
    proj: &str,
    prefix: &str,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let names = |r: Range<usize>| -> String {
        r.map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let cons = "List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true";
    let bind = |out: &mut String, h: &str, r: Range<usize>| {
        if r.len() > 1 {
            let _ = writeln!(out, "  obtain ⟨{}⟩ := {h}", names(r));
        } else {
            let _ = writeln!(out, "  have {}{} := {h}", prefix, r.start);
        }
    };
    let _ = writeln!(out, "  have {hyp} := {proj}");
    match crate::circuit::chunks(len) {
        None => {
            let _ = writeln!(out, "  simp only [{name}, {cons}] at {hyp}");
            bind(out, hyp, 0..len);
        }
        Some(ranges) => {
            let _ = writeln!(out, "  simp only [{name}, List.forall_mem_append] at {hyp}");
            for (k, r) in ranges.iter().enumerate() {
                let m = format!("{hyp}{k}");
                let _ = writeln!(
                    out,
                    "  have {m} := {hyp}{}",
                    crate::circuit::chunk_path(ranges.len(), k)
                );
                let _ = writeln!(out, "  simp only [{name}.{field}{k}, {cons}] at {m}");
                bind(out, &m, r.clone());
            }
        }
    }
}

/// Identifier-like tokens `prefix<digits>[…]` in a tactic script (`c12`, `k0`, `e_3_1`),
/// not preceded by an identifier character.
fn tokens<'s>(script: &'s str, prefix: &str) -> Vec<&'s str> {
    let bytes = script.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut found = Vec::new();
    let mut i = 0;
    while let Some(off) = script[i..].find(prefix) {
        let start = i + off;
        let end0 = start + prefix.len();
        let boundary = start == 0 || !is_ident(bytes[start - 1]);
        let mut end = end0;
        while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'_') {
            end += 1;
        }
        if boundary && end > end0 && bytes[end0].is_ascii_digit() {
            found.push(&script[start..end]);
        }
        i = end0;
    }
    found
}

/// Copy indices `c<i>` a script mentions.
fn copy_refs(script: &str) -> BTreeSet<usize> {
    tokens(script, "c")
        .into_iter()
        .filter_map(|t| t[1..].parse().ok())
        .collect()
}

/// Ops `e_<row>_<i>` a script mentions.
fn op_refs(script: &str) -> BTreeSet<(usize, usize)> {
    tokens(script, "e_")
        .into_iter()
        .filter_map(|t| {
            let mut it = t[2..].split('_');
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect()
}

/// `List.mem_append_left/right` path from membership in chunk `k` to membership in the
/// `++` tree over `n` chunks (`circuit.rs::chunk_tree`), applied to `hq`.
fn chunk_mem(n: usize, k: usize) -> String {
    fn go(lo: usize, hi: usize, k: usize) -> String {
        if hi - lo == 1 {
            return "hq".into();
        }
        let mid = lo + (hi - lo) / 2;
        if k < mid {
            format!("(List.mem_append_left _ {})", go(lo, mid, k))
        } else {
            format!("(List.mem_append_right _ {})", go(mid, hi, k))
        }
    }
    go(0, n, k)
}

/// Facts per parenthesised group in a decode theorem's conclusion.
pub const FACT_GROUP: usize = 32;
/// Heartbeat budget for decode theorems with more than one fact group.
const LARGE_HEARTBEATS: usize = 4_000_000;

/// Render `theorem <name>_decode (a) (h : Satisfies (<name> p) a) : fact₁ ∧ …`, assembled
/// from one lemma `<name>_f<k>` per recorded call. Each lemma destructures only the copy
/// chunks, constants and op equations its own tactic block mentions, so elaboration time is
/// linear in the number of calls (a single tactic block over every copy and op is
/// superlinear and unusable past a few hundred calls).
pub fn render_decode_theorem(name: &str, ex: &CircuitExport, calls: &[Call]) -> String {
    let ops = Ops::new(ex);
    let mut out = String::new();
    let facts: Vec<String> = calls.iter().map(|c| c.fact.lean()).collect();
    let doc = format!(
        "/-- Every satisfying assignment of `{name}` has the meaning of each recorded gadget \
         call. Generated at gadget-call granularity; see `gadget.rs`. -/"
    );
    if facts.is_empty() {
        let _ = writeln!(out, "{doc}");
        let _ = writeln!(
            out,
            "theorem {name}_decode (a : Assignment p) (h : Satisfies ({name} p) a) : True :=\n  \
             trivial"
        );
        return out;
    }
    let sponge = calls.iter().any(|c| c.fact.is_poseidon2());
    let hyps = |poseidon: bool| -> String {
        format!(
            "{}(a : Assignment p) (h : Satisfies ({name} p) a){}",
            if poseidon {
                "(perm : St p → St p) "
            } else {
                ""
            },
            if poseidon {
                format!("\n    (hp : Poseidon2Rows perm ({name} p) a)")
            } else {
                String::new()
            }
        )
    };
    let cons = "List.forall_mem_cons, List.not_mem_nil, false_implies, implies_true, and_true";

    // Copies: one lemma per chunk (or one for a flat list) stating the chunk's copies as a
    // conjunction, in the normal form `simp only [List.forall_mem_cons, …]` leaves: a copy
    // `(x, x)` is `True`, and trailing `True`s are absorbed. Fact lemmas project out of it.
    let chunked = crate::circuit::chunks(ex.copies.len());
    let chunk_ranges: Vec<Range<usize>> = match &chunked {
        Some(ranges) => ranges.clone(),
        None => vec![0..ex.copies.len()],
    };
    let conj = |r: &Range<usize>| -> Vec<Option<String>> {
        let mut v: Vec<Option<String>> = r
            .clone()
            .map(|ci| {
                let (x, y) = ex.copies[ci];
                (x != y).then(|| format!("a ({}) = a ({})", lean_target(x), lean_target(y)))
            })
            .collect();
        while matches!(v.last(), Some(None)) {
            v.pop();
        }
        v
    };
    let chunk_lemma = |k: usize| -> String {
        match &chunked {
            Some(_) => format!("{name}_copies{k}"),
            None => format!("{name}_copies"),
        }
    };
    if !ex.copies.is_empty() {
        for (k, r) in chunk_ranges.iter().enumerate() {
            let cj = conj(r);
            let stmt: Vec<String> = cj
                .iter()
                .map(|c| c.clone().unwrap_or_else(|| "True".into()))
                .collect();
            let stmt = if stmt.is_empty() {
                "True".to_string()
            } else {
                stmt.join(" ∧ ")
            };
            let _ = writeln!(
                out,
                "theorem {} (a : Assignment p) (h : Satisfies ({name} p) a) :\n    {stmt} := by",
                chunk_lemma(k)
            );
            match &chunked {
                Some(ranges) => {
                    let _ = writeln!(
                        out,
                        "  have hc : ∀ q ∈ {name}.copies{k}, a q.1 = a q.2 := fun q hq => h.2.1 q {}",
                        chunk_mem(ranges.len(), k)
                    );
                    let _ = writeln!(out, "  simp only [{name}.copies{k}, {cons}] at hc");
                }
                None => {
                    let _ = writeln!(out, "  have hc := h.2.1");
                    let _ = writeln!(out, "  simp only [{name}, {cons}] at hc");
                }
            }
            let _ = writeln!(out, "  exact hc\n");
        }
    }
    // `have c<i> := <chunk lemma>.2.….1` for copy `i`.
    let copy_have = |ci: usize| -> String {
        let (k, r) = chunk_ranges
            .iter()
            .enumerate()
            .find(|(_, r)| r.contains(&ci))
            .expect("copy index in range");
        let cj = conj(r);
        let j = ci - r.start;
        assert!(
            cj.get(j).is_some_and(|c| c.is_some()),
            "copy {ci} is a self-copy and carries nothing"
        );
        let mut path = String::new();
        if cj.len() > 1 {
            path.push_str(&".2".repeat(j));
            if j + 1 < cj.len() {
                path.push_str(".1");
            }
        }
        format!("  have c{ci} := ({} a h){path}\n", chunk_lemma(k))
    };
    // Constants, once.
    if !ex.constants.is_empty() {
        let stmt: Vec<String> = ex
            .constants
            .iter()
            .map(|(t, v)| format!("a ({}) = {}", lean_target(*t), lean_const_int(*v)))
            .collect();
        let _ = writeln!(
            out,
            "theorem {name}_consts (a : Assignment p) (h : Satisfies ({name} p) a) :\n    {} := by",
            stmt.join(" ∧ ")
        );
        destructure(
            &mut out,
            name,
            "constants",
            "hconst",
            "h.2.2",
            "k",
            ex.constants.len(),
        );
        let ks: Vec<String> = (0..ex.constants.len()).map(|k| format!("k{k}")).collect();
        if let [k0] = &ks[..] {
            let _ = writeln!(out, "  exact {k0}\n");
        } else {
            let _ = writeln!(out, "  exact ⟨{}⟩\n", ks.join(", "));
        }
    }

    // The op equation of `(row, i)`: its input wires oriented to their sources and constant
    // sources to their values. Output wires stay atoms; a pin on one is used only by the
    // check that owns it.
    let op_block = |(row, i): (usize, usize)| -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "  have {} := arithEq_of_rows h (row := {row}) (i := {i}) rfl (by decide)",
            e((row, i))
        );
        let mut rules: Vec<String> = Vec::new();
        for col in [4 * i, 4 * i + 1, 4 * i + 2] {
            let Some(&(src, ci)) = ops.input_source.get(&(row, col)) else {
                continue;
            };
            let (x, _) = ex.copies[ci];
            // `connect(source, wire)`: rewrite `a wire` into `a source`.
            rules.push(if x == Target::wire(row, col) {
                format!("c{ci}")
            } else {
                format!("← c{ci}")
            });
            if let Some(&k) = ops.const_idx.get(&src) {
                rules.push(format!("k{k}"));
            }
        }
        rules.dedup();
        let _ = writeln!(
            s,
            "  simp only [Nat.reduceMul, Nat.reduceAdd] at {}",
            e((row, i))
        );
        s.push_str(&indent_lines(&simp_only(&rules, Some(&e((row, i)))), "  "));
        s
    };

    // One lemma per call.
    for (n, call) in calls.iter().enumerate() {
        let f = &call.fact;
        let what = f.lean();
        let script = fact_script(&ops, ex, call);
        let mut body = String::new();
        for op in op_refs(&script) {
            body.push_str(&op_block(op));
        }
        body.push_str(&script);
        let mut pre = String::new();
        for ci in copy_refs(&body) {
            pre.push_str(&copy_have(ci));
        }
        if !tokens(&body, "k").is_empty() {
            let ks: Vec<String> = (0..ex.constants.len()).map(|k| format!("k{k}")).collect();
            if let [k0] = &ks[..] {
                let _ = writeln!(pre, "  have {k0} := {name}_consts a h");
            } else {
                let _ = writeln!(pre, "  obtain ⟨{}⟩ := {name}_consts a h", ks.join(", "));
            }
        }
        let _ = writeln!(
            out,
            "theorem {name}_f{n} {} :\n    {what} := by",
            hyps(f.is_poseidon2())
        );
        out.push_str(&pre);
        out.push_str(&body);
        out.push('\n');
    }

    // The conjunction, grouped `FACT_GROUP` to a parenthesised conjunction: a flat `∧`
    // chain hundreds deep is superlinear to elaborate.
    if facts.len() > FACT_GROUP {
        let _ = writeln!(out, "set_option maxHeartbeats {LARGE_HEARTBEATS} in");
    }
    let _ = writeln!(out, "{doc}");
    let groups: Vec<&[String]> = facts.chunks(FACT_GROUP).collect();
    let _ = writeln!(out, "theorem {name}_decode {} :", hyps(sponge));
    for (g, group) in groups.iter().enumerate() {
        let (open, close) = if groups.len() > 1 {
            ("(", ")")
        } else {
            ("", "")
        };
        for (i, f) in group.iter().enumerate() {
            let pre = if i == 0 { open } else { "" };
            let sep = match (i + 1 == group.len(), g + 1 == groups.len()) {
                (false, _) => " ∧",
                (true, false) => &*format!("{close} ∧"),
                (true, true) => &*format!("{close} :="),
            };
            let _ = writeln!(out, "    {pre}{f}{sep}");
        }
    }
    let fs: Vec<String> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if c.fact.is_poseidon2() {
                format!("{name}_f{i} perm a h hp")
            } else {
                format!("{name}_f{i} a h")
            }
        })
        .collect();
    if let [f0] = &fs[..] {
        let _ = writeln!(out, "  {f0}");
    } else if fs.len() <= FACT_GROUP {
        let _ = writeln!(out, "  ⟨{}⟩", fs.join(", "));
    } else {
        let gs: Vec<String> = fs
            .chunks(FACT_GROUP)
            .map(|g| match g {
                [f] => f.clone(),
                _ => format!("⟨{}⟩", g.join(", ")),
            })
            .collect();
        let _ = writeln!(out, "  ⟨{}⟩", gs.join(", "));
    }
    out
}

/// The tactic block closing one recorded call's fact, at two-space indentation, referring
/// to `c<i>`, `k<i>` and `e_<row>_<i>` by name.
fn fact_script(ops: &Ops, ex: &CircuitExport, call: &Call) -> String {
    let mut out = String::new();
    let f = &call.fact;
    let stop = f.named();
    let what = f.lean();
    let ks = ops.const_rules(&stop);
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
            goal.extend(ks);
            let id = identity(ops, |ev| ev.fact_poly(f).is_zero());
            match id {
                Identity::Holds => {
                    // The builder folded the call onto an operand or a constant: the
                    // fact is an identity of the rendered integers.
                    out.push_str(&indent_lines(&simp_only(&goal, None), "  "));
                    out.push_str("  ring\n");
                }
                Identity::GoldilocksOnly => reject_unless_holds(&what, "", id),
                Identity::Fails => {
                    // The output's own equation, with the definitions internal to
                    // this call substituted, is the fact. An output wire whose op
                    // belongs to another call (an identity fold onto it) fails this
                    // test and is reported rather than emitted.
                    let root = ops
                        .op_of_output(o)
                        .unwrap_or_else(|| panic!("{what}: output has no op and is no identity"));
                    let defs = ops.internal_defs(&ops.inputs(root), &stop);
                    require_identity(
                        ops,
                        &what,
                        &format!("op {root:?} does not establish the fact"),
                        |ev| ev.fact_poly(f) == ev.atom(o) - ev.op_rhs(root, &defs),
                    );
                    let defs: Vec<String> = defs.into_iter().map(e).collect();
                    let _ = writeln!(out, "  have hr := {}", e(root));
                    out.push_str(&indent_lines(&simp_only(&defs, Some("hr")), "  "));
                    out.push_str(&indent_lines(&simp_only(&goal, None), "  "));
                    out.push_str("  linear_combination hr\n");
                }
            }
        }
        Fact::AssertBool { b } => {
            let checks = ops.checks_in(&call.copies);
            let [chk] = checks[..] else {
                panic!("{what}: assert_bool pins exactly one target: {checks:?}")
            };
            out.push_str(&indent_lines(&simp_only(&ks, None), "  "));
            out.push_str("  refine isBool_iff_assertBool.mpr ?_\n");
            let goal = |ev: &mut Eval| ev.atom(b) * ev.atom(b) - ev.atom(b);
            out.push_str(&check_script(ops, chk, &stop, &goal, &what, "  "));
        }
        Fact::IsEqual { x, y, equal, inv } => {
            let checks = ops.checks_in(&call.copies);
            // `connect(not_equal_check, zero)` then `connect(equal_check, zero)`
            // (arithmetic.rs), each possibly constant-folded.
            let [ne, eq] = checks[..] else {
                panic!("{what}: is_equal pins exactly two targets: {checks:?}")
            };
            out.push_str(&indent_lines(&simp_only(&ks, None), "  "));
            out.push_str("  refine ⟨?_, ?_⟩\n");
            let c1 = |ev: &mut Eval| ev.atom(equal) * (ev.atom(x) - ev.atom(y));
            let c2 = |ev: &mut Eval| {
                (ev.atom(x) - ev.atom(y)) * ev.atom(inv) - (ev.one() - ev.atom(equal))
            };
            let goals: [&dyn Fn(&mut Eval) -> V; 2] = [&c1, &c2];
            for (chk, goal) in [ne, eq].into_iter().zip(goals) {
                let script = check_script(ops, chk, &stop, goal, &what, "    ");
                let mut lines = script.lines();
                if let Some(first) = lines.next() {
                    let _ = writeln!(out, "  · {}", first.trim_start());
                }
                for l in lines {
                    let _ = writeln!(out, "{l}");
                }
            }
        }
        Fact::RangeCheck { bits, .. } => {
            let (row, hr) = base_sum_row(ops, ex, call, "rangeCheck_of_row", bits, &what);
            let _ = write!(out, "{hr}");
            let _ = writeln!(out, "  rwa [{}] at hr", sum_wire_rule(ex, call, row, &what));
        }
        Fact::SplitLe { row, bits, .. } => {
            let (r, hr) = base_sum_row(ops, ex, call, "baseSum_of_row", bits, &what);
            assert_eq!(r, row, "{what}: the call's row is not the recorded one");
            let _ = write!(out, "{hr}");
            let _ = writeln!(out, "  rwa [{}] at hr", sum_wire_rule(ex, call, row, &what));
        }
        Fact::Connect { x, y } => {
            assert_eq!(call.copies.len(), 1, "connect adds one copy");
            if x == y {
                // The copy `(x, x)` destructures to `True`.
                out.push_str("  rfl\n");
            } else {
                let _ = writeln!(out, "  exact c{}", call.copies.start);
            }
        }
        Fact::Poseidon2 {
            ref rows,
            ref inputs,
        } => {
            out.push_str(&sponge_script(ops, ex, call, rows, inputs, &what));
        }
    }
    out
}

/// `have hr := <lemma> h (row := …) (N := …) (n := bits) rfl rfl (by decide) (by …)` for the
/// `BaseSumGate<2>` row a `range_check`/`split_le` call placed, the tail limbs `bits..N`
/// discharged from this call's copies to the zero constant. Returns the row too.
fn base_sum_row(
    ops: &Ops,
    ex: &CircuitExport,
    call: &Call,
    lemma: &str,
    bits: usize,
    what: &str,
) -> (usize, String) {
    assert_eq!(
        call.rows.len(),
        1,
        "{what}: a split of ≤ num_limbs bits places one row"
    );
    let row = call.rows.start;
    let GateKind::BaseSum2 { num_limbs } = ex.rows[row].0 else {
        panic!(
            "{what}: row {row} is not BaseSumGate<2>: {:?}",
            ex.rows[row].0
        )
    };
    let zero_t = ex
        .constants
        .iter()
        .find(|(_, c)| *c == F::ZERO)
        .map(|(t, _)| *t)
        .unwrap_or_else(|| panic!("{what}: limbs are pinned to the zero constant"));
    let kz = ops.const_idx[&zero_t];
    let mut limb_copy: HashMap<usize, (usize, bool)> = HashMap::new();
    for ci in call.copies.clone() {
        let (x, y) = ex.copies[ci];
        for (t, other, fwd) in [(x, y, true), (y, x, false)] {
            if let Some((r, col)) = wire(t) {
                if r == row && col > 0 && other == zero_t {
                    limb_copy.insert(col - 1, (ci, fwd));
                }
            }
        }
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "  have hr := {lemma} h (row := {row}) (N := {num_limbs}) (n := {bits}) rfl rfl \
         (by decide) (by"
    );
    out.push_str("    intro i hi1 hi2\n    interval_cases i\n");
    for i in bits..num_limbs {
        let (ci, fwd) = limb_copy
            .get(&i)
            .unwrap_or_else(|| panic!("{what}: limb {i} of row {row} is not pinned to zero"));
        let _ = writeln!(
            out,
            "    · exact {}.trans k{kz}",
            if *fwd {
                format!("c{ci}")
            } else {
                format!("c{ci}.symm")
            }
        );
    }
    out.push_str("    )\n");
    (row, out)
}

/// The rewrite moving a fact off a `BaseSumGate<2>` row's sum wire onto the target this
/// call connected to it.
fn sum_wire_rule(ex: &CircuitExport, call: &Call, row: usize, what: &str) -> String {
    let sum = Target::wire(row, 0);
    for ci in call.copies.clone() {
        let (x, y) = ex.copies[ci];
        if x == sum {
            return format!("c{ci}");
        }
        if y == sum {
            return format!("← c{ci}");
        }
    }
    panic!("{what}: the sum wire of row {row} is not connected")
}

/// An element of the padded sponge message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Elem {
    Input(Target),
    One,
    Zero,
}

impl Elem {
    fn lean(self) -> String {
        match self {
            Elem::Input(t) => format!("a ({})", lean_target(t)),
            Elem::One => "1".into(),
            Elem::Zero => "0".into(),
        }
    }

    fn value(self, ev: &mut Eval) -> V {
        match self {
            Elem::Input(t) => ev.atom(t),
            Elem::One => ev.one(),
            Elem::Zero => ev.one() - ev.one(),
        }
    }
}

/// The copy in this call connecting `w` (a row input wire) to its source, as the source and
/// a proof of `a w = a source`.
fn feed(ex: &CircuitExport, call: &Call, w: Target, what: &str) -> (Target, usize, bool) {
    call.copies
        .clone()
        .find_map(|ci| match ex.copies[ci] {
            (x, y) if x == w => Some((y, ci, true)),
            (x, y) if y == w => Some((x, ci, false)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{what}: input wire {w:?} is not connected"))
}

fn copy_term(ci: usize, fwd: bool) -> String {
    if fwd {
        format!("c{ci}")
    } else {
        format!("c{ci}.symm")
    }
}

/// The tactic block for a `poseidon2_hash` call: per row, the twelve input-wire facts
/// packaged by `poseidon2In_first`/`poseidon2In_chain` and permuted by
/// `poseidon2Row_absorb`; then `spongeHash` unfolded block by block onto the row outputs.
fn sponge_script(
    ops: &Ops,
    ex: &CircuitExport,
    call: &Call,
    rows: &[usize],
    inputs: &[Target],
    what: &str,
) -> String {
    let mut out = String::new();
    let mut msg: Vec<Elem> = inputs.iter().map(|&t| Elem::Input(t)).collect();
    msg.push(Elem::One);
    while msg.len() % 8 != 0 {
        msg.push(Elem::Zero);
    }
    assert_eq!(
        msg.len(),
        8 * rows.len(),
        "{what}: {} padded elements need {} Poseidon2Gate rows, got {rows:?}",
        msg.len(),
        msg.len() / 8
    );
    let const_val = |t: Target| ops.const_idx.get(&t).map(|&k| (k, ex.constants[k].1));
    let expect_const = |src: Target, ci: usize, fwd: bool, v: F, lane: &str| -> String {
        let (k, actual) =
            const_val(src).unwrap_or_else(|| panic!("{what}: {lane} is not fed by a constant"));
        assert_eq!(actual, v, "{what}: {lane} is not the {v} constant");
        format!("({}.trans k{k})", copy_term(ci, fwd))
    };
    for (k, &row) in rows.iter().enumerate() {
        assert_eq!(
            ex.rows.get(row).map(|r| &r.0),
            Some(&GateKind::Poseidon2),
            "{what}: row {row} is not a Poseidon2Gate"
        );
        let block = &msg[8 * k..8 * k + 8];
        let mut terms: Vec<String> = Vec::with_capacity(12);
        for j in 0..12 {
            let w = Target::wire(row, j);
            let (src, ci, fwd) = feed(ex, call, w, what);
            let lane = format!("input wire {j} of row {row}");
            let term = if k == 0 {
                match if j < 8 { block[j] } else { Elem::Zero } {
                    Elem::Input(t) if src == t => copy_term(ci, fwd),
                    Elem::Input(t) => match (const_val(src), const_val(t)) {
                        (Some((ks, vs)), Some((kt, vt))) if vs == vt => {
                            format!("({}.trans (k{ks}.trans k{kt}.symm))", copy_term(ci, fwd))
                        }
                        _ => panic!("{what}: {lane} is not input {j}"),
                    },
                    Elem::One => expect_const(src, ci, fwd, F::ONE, &lane),
                    Elem::Zero => expect_const(src, ci, fwd, F::ZERO, &lane),
                }
            } else {
                let prev = Target::wire(rows[k - 1], 12 + j);
                if j >= 8 || src == prev {
                    assert_eq!(src, prev, "{what}: {lane} is not the previous output");
                    if j >= 8 {
                        copy_term(ci, fwd)
                    } else {
                        assert_eq!(
                            block[j],
                            Elem::Zero,
                            "{what}: {lane} skips a nonzero block element"
                        );
                        format!("({}.trans (add_zero _).symm)", copy_term(ci, fwd))
                    }
                } else {
                    // `add(state, element)` placed an op: `a src = 1 * a prev * 1 + 1 * elem`.
                    let op = ops.op_of_output(src).unwrap_or_else(|| {
                        panic!("{what}: {lane} is fed by neither the previous output nor an op")
                    });
                    let b = block[j];
                    require_identity(
                        ops,
                        what,
                        &format!("op {op:?} does not add the block element onto {lane}"),
                        |ev| ev.op_rhs(op, &[]) == ev.atom(prev) + b.value(ev),
                    );
                    let sign = if fwd { "" } else { "-" };
                    let konst = match b {
                        Elem::Input(t) => const_val(t)
                            .map(|(kt, _)| format!(" - k{kt}"))
                            .unwrap_or_default(),
                        _ => String::new(),
                    };
                    format!("(by linear_combination {sign}c{ci} + {}{konst})", e(op))
                }
            };
            terms.push(term);
        }
        let blk: Vec<String> = block.iter().map(|b| b.lean()).collect();
        let (state, lemma) = if k == 0 {
            ("(fun _ => 0)".to_string(), "poseidon2In_first")
        } else {
            (
                format!("(poseidon2Out a {})", rows[k - 1]),
                "poseidon2In_chain",
            )
        };
        let _ = writeln!(
            out,
            "  have hin{k} : poseidon2In a {row} = addBlock {state} [{}] :=\n    {lemma} {}",
            blk.join(", "),
            terms.join(" ")
        );
        let _ = writeln!(
            out,
            "  have hout{k} := poseidon2Row_absorb perm hp (row := {row}) rfl rfl hin{k}"
        );
    }
    let blocks: Vec<String> = msg
        .chunks(8)
        .map(|b| {
            format!(
                "[{}]",
                b.iter().map(|e| e.lean()).collect::<Vec<_>>().join(", ")
            )
        })
        .collect();
    let mut padded = String::new();
    for b in blocks.iter().rev() {
        padded = if padded.is_empty() {
            format!("{b} ++ []")
        } else {
            format!("{b} ++ ({padded})")
        };
    }
    let _ =
        writeln!(
        out,
        "  have hpad : pad10 [{}] =\n      {padded} := by\n    simp [pad10, rate, List.replicate]",
        inputs.iter().map(|&t| format!("a ({})", lean_target(t))).collect::<Vec<_>>().join(", ")
    );
    let mut rules = vec!["spongeHash".to_string(), "hpad".to_string()];
    rules.extend(rows.iter().map(|_| "absorbMsg_block8".to_string()));
    rules.push("absorbMsg_nil".into());
    rules.extend((0..rows.len()).map(|k| format!("← hout{k}")));
    let _ = writeln!(out, "  rw [{}]", rules.join(", "));
    out.push_str("  exact ⟨rfl, rfl, rfl, rfl⟩\n");
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

/// One exported circuit with its recorded calls, ready to render.
pub struct GeneratedCircuit {
    pub name: String,
    pub doc: String,
    pub ex: CircuitExport,
    pub calls: Vec<Call>,
}

impl GeneratedCircuit {
    fn new(name: &str, doc: &str, r: &Recorder, named: Vec<(String, Vec<Target>)>) -> Self {
        GeneratedCircuit {
            name: name.to_string(),
            doc: doc.to_string(),
            ex: r.export(named).expect("no lookups"),
            calls: r.calls.clone(),
        }
    }
}

/// Build `formal/Plonky2Spec/Generated/GadgetZooCircuit.lean`: the export plus the
/// generated decode theorem.
pub fn generate_gadget_zoo_lean() -> String {
    let (r, t) = build_gadget_zoo();
    render_module(
        "the gadget-zoo circuit built through the recording builder",
        &[GeneratedCircuit::new(
            "gadgetZoo",
            "The gadget zoo: `assert_bool flag`, `eq = is_equal x y`, `sel = select flag x y`, \
             `either = or eq flag`, `nflag = not flag`, `both = and eq nflag`, \
             `head = sub 10000 fee`, `range_check head 14`, `connect sel either`; \
             public inputs `x`, `sel`, `both`.",
            &r,
            t.named(),
        )],
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

/// `sum = add x y; prod = mul sum one`: the multiplication folds onto `sum`, whose op is
/// the addition, so the recorded `mul` fact is an identity and not an op equation.
/// Returns `(recorder, x, y, one, sum, prod)`.
pub fn build_identity_fold() -> (Recorder, [Target; 5]) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    let one = r.constant(F::ONE);
    let sum = r.add(x, y);
    let prod = r.mul(sum, one);
    (r, [x, y, one, sum, prod])
}

/// `diff = sub x y; eq = is_equal x y; connect diff zero`: `is_equal` reuses the memoized
/// subtraction, which a later call pins to zero, so the equality checks must still unfold
/// `diff` through its op rather than through the pin. Returns `(recorder, x, y, zero, diff,
/// equal)`.
pub fn build_pinned_intermediate() -> (Recorder, [Target; 5]) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    let zero = r.constant(F::ZERO);
    let diff = r.sub(x, y);
    let eq = r.is_equal(x, y);
    r.connect(diff, zero);
    (r, [x, y, zero, diff, eq.target])
}

/// Constant folds that are identities of the rendered integers: `nine = mul three three`
/// (`9 = 3 * 3`), `eight = add three five`, and `neg5 = sub zero five` (`(-5) = 0 - 5`, the
/// renderer's negative form). Returns `(recorder, zero, three, five, nine, eight, neg5)`.
pub fn build_constant_fold() -> (Recorder, [Target; 6]) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let zero = r.constant(F::ZERO);
    let three = r.constant(F::from_canonical_u64(3));
    let five = r.constant(F::from_canonical_u64(5));
    let nine = r.mul(three, three);
    let eight = r.add(three, five);
    let neg5 = r.sub(zero, five);
    (r, [zero, three, five, nine, eight, neg5])
}

/// A constant fold the builder reduced modulo the Goldilocks order: `mul c c` for
/// `c = 2^32` folds to `2^32 - 1`, which `9 = 3 * 3`-style generic reasoning cannot see.
/// The generator rejects it.
pub fn build_goldilocks_fold_mul() -> Recorder {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let c = r.constant(F::from_canonical_u64(1 << 32));
    r.mul(c, c);
    r
}

/// `sub zero c` for `c = 2^32 + 1` folds to `p - 2^32 - 1`, past the renderer's negative
/// cutoff, so it is rendered as a large positive canonical value: also Goldilocks-only.
pub fn build_goldilocks_fold_sub() -> Recorder {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let zero = r.constant(F::ZERO);
    let c = r.constant(F::from_canonical_u64((1 << 32) + 1));
    r.sub(zero, c);
    r
}

/// A single recorded call: `sum = add x y`. Returns `(recorder, x, y, sum)`.
pub fn build_single_fact() -> (Recorder, [Target; 3]) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    let sum = r.add(x, y);
    (r, [x, y, sum])
}

/// No recorded calls at all. Returns `(recorder, x, y)`.
pub fn build_no_facts() -> (Recorder, [Target; 2]) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    (r, [x, y])
}

/// `FACT_GROUP + 1` recorded calls: `connect x y` repeated, so the decode theorem's last
/// fact group holds a single fact. Returns `(recorder, x, y)`.
pub fn build_fact_group_boundary() -> (Recorder, [Target; 2]) {
    let mut r = Recorder::new(CircuitConfig::standard_recursion_config());
    let x = r.add_virtual_target();
    let y = r.add_virtual_target();
    for _ in 0..=FACT_GROUP {
        r.connect(x, y);
    }
    (r, [x, y])
}

fn named(names: &[&str], targets: &[Target]) -> Vec<(String, Vec<Target>)> {
    names
        .iter()
        .zip(targets)
        .map(|(n, t)| (n.to_string(), vec![*t]))
        .collect()
}

/// Build `formal/Plonky2Spec/Generated/GadgetEdgeCasesCircuit.lean`: the edge-case circuit
/// and the smaller reproductions, each with its generated decode theorem.
pub fn generate_gadget_edge_cases_lean() -> String {
    let (r, t) = build_gadget_edge_cases();
    let edge = GeneratedCircuit::new(
        "gadgetEdgeCases",
        "Builder folding and re-pinning edge cases: `sum = add x y; connect sum zero`, \
         `is_equal zero zero`, `is_equal one zero`, `is_equal x x`, \
         `eq_xy = is_equal x y; diff = sub x y; check = mul eq_xy diff; connect check zero`; \
         public inputs the four `equal` targets.",
        &r,
        t.named(),
    );
    let (r, t) = build_identity_fold();
    let fold = GeneratedCircuit::new(
        "gadgetIdentityFold",
        "`sum = add x y; prod = mul sum one`: the product folds onto `sum`.",
        &r,
        named(&["x", "y", "one", "sum", "prod"], &t),
    );
    let (r, t) = build_pinned_intermediate();
    let pinned = GeneratedCircuit::new(
        "gadgetPinnedIntermediate",
        "`diff = sub x y; eq = is_equal x y; connect diff zero`: `is_equal` reuses `diff`, \
         which is then pinned to zero.",
        &r,
        named(&["x", "y", "zero", "diff", "equal"], &t),
    );
    let (r, t) = build_constant_fold();
    let folds = GeneratedCircuit::new(
        "gadgetConstantFold",
        "Constant folds that hold over `ℤ`: `nine = mul three three`, `eight = add three \
         five`, `neg5 = sub zero five`.",
        &r,
        named(&["zero", "three", "five", "nine", "eight", "neg5"], &t),
    );
    let (r, t) = build_single_fact();
    let single = GeneratedCircuit::new(
        "gadgetSingleFact",
        "One recorded call, `sum = add x y`.",
        &r,
        named(&["x", "y", "sum"], &t),
    );
    let (r, t) = build_no_facts();
    let none = GeneratedCircuit::new(
        "gadgetNoFacts",
        "No recorded calls.",
        &r,
        named(&["x", "y"], &t),
    );
    let (r, t) = build_fact_group_boundary();
    let boundary = GeneratedCircuit::new(
        "gadgetFactGroupBoundary",
        "`FACT_GROUP + 1` recorded calls (`connect x y` repeated): the last fact group of the \
         decode theorem is a single fact.",
        &r,
        named(&["x", "y"], &t),
    );
    render_module(
        "the gadget edge-case circuits built through the recording builder",
        &[edge, fold, pinned, folds, single, none, boundary],
    )
}

/// A complete `Plonky2Spec.Generated` module: header, then each circuit's export and
/// decode theorem.
pub fn render_module(what: &str, circuits: &[GeneratedCircuit]) -> String {
    let sponge = circuits
        .iter()
        .any(|c| c.calls.iter().any(|call| call.fact.is_poseidon2()));
    let mut out = String::new();
    let _ = write!(
        out,
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Produced by `qp-plonky2-constraint-exporter` (`gadget.rs`) from {what}: the\n\
         \x20 pre-`build` constraint system and the gadget calls recorded while building it.\n\
         \x20 Each theorem's proof is generated too, one block per recorded gadget call, from\n\
         \x20 the ops and copy constraints the builder emitted for it.\n\
         \x20 Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\
         -/\n\
         import Mathlib.Tactic.IntervalCases\n\
         import Mathlib.Tactic.LinearCombination\n\
         import Plonky2Spec.WiringGadgets\n{}\n\
         namespace Plonky2Spec.Generated\n\n\
         open Plonky2Spec.Wiring\n{}\n\
         set_option linter.all false\n\n\
         variable {{p : ℕ}} [Fact p.Prime]\n\n",
        if sponge {
            "import Plonky2Spec.WiringSponge\n"
        } else {
            ""
        },
        if sponge {
            "open Plonky2Spec.Poseidon2 (St)\n\
             open Plonky2Spec.Sponge (spongeHash pad10 addBlock rate absorbMsg_block8 absorbMsg_nil)\n"
        } else {
            ""
        },
    );
    for c in circuits {
        out.push_str(&crate::circuit::render_lean(&c.name, &c.doc, &c.ex));
        out.push_str(&render_decode_theorem(&c.name, &c.ex, &c.calls));
        out.push('\n');
    }
    out.push_str("end Plonky2Spec.Generated\n");
    out
}
