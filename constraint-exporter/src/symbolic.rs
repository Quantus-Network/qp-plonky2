//! A symbolic field element (`Sym`) that records the arithmetic performed on it
//! as an expression DAG, instead of computing a concrete value.
//!
//! The point: a gate's `Gate::eval_unfiltered` is generic over the field type
//! and works entirely through the `Field` arithmetic ops. If we instantiate it
//! with `Sym` instead of `GoldilocksField`, the `Vec<F::Extension>` it returns
//! *is* the list of constraint polynomials, captured symbolically — there is no
//! hand-transcription step that could disagree with the gate code.
//!
//! `Field: Copy`, so `Sym` cannot be a `Box`-based tree; it is a `Copy` handle
//! (`u32`) into a thread-local arena of [`Node`]s. The four field constants
//! (`ZERO/ONE/TWO/NEG_ONE`) and the two generators must be `const`-evaluable, so
//! they are reserved *sentinel* ids resolved without touching the arena.
//!
//! We instantiate gates at extension degree `D = 1`, where plonky2 blanket-
//! implements `OEF<1>`, `FieldExtension<1>`, `Extendable<1>`, and `PackedField`
//! for any `Field` (+ a trivial `Frobenius<1>`). The constraint *polynomials* are
//! independent of `D` — `D` only changes the extension arithmetic used while
//! proving — so extracting at `D = 1` yields exactly the gate's real constraints.

use core::cell::RefCell;
use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::BigUint;
use plonky2::field::extension::Frobenius;
use plonky2::field::types::{Field, Field64, PrimeField, PrimeField64, Sample};
use plonky2::hash::hash_types::RichField;
use qp_plonky2_core::poseidon::{Poseidon, N_PARTIAL_ROUNDS, SPONGE_WIDTH};
use serde::{Deserialize, Serialize};

/// Goldilocks order `2^64 - 2^32 + 1`. The symbolic type advertises this so that
/// any constant a gate produces via `from_canonical_u64` reduces the same way a
/// real Goldilocks element would (only relevant for the differential test).
pub const GOLDILOCKS_ORDER: u64 = 0xFFFF_FFFF_0000_0001;

// Sentinel ids for the `const` field elements. Real arena handles are small
// (`0..SENTINEL_MIN`); ids at/above the sentinel range are resolved specially.
const ID_ZERO: u32 = u32::MAX;
const ID_ONE: u32 = u32::MAX - 1;
const ID_TWO: u32 = u32::MAX - 2;
const ID_NEG_ONE: u32 = u32::MAX - 3;
const ID_MULT_GEN: u32 = u32::MAX - 4;
const ID_POW2_GEN: u32 = u32::MAX - 5;
const SENTINEL_MIN: u32 = u32::MAX - 7;

/// A node in the symbolic expression DAG. `Add/Sub/Mul/Neg` hold child handles.
#[derive(Clone, Debug)]
pub enum Node {
    /// A canonical field constant `n` (`0 <= n < ORDER`).
    Const(u64),
    /// The field element `-1`.
    NegOne,
    /// A named local-wire variable, `local_wires[i]`.
    Wire(usize),
    /// A named local-constant variable, `local_constants[i]`.
    LocalConst(usize),
    Add(Sym, Sym),
    Sub(Sym, Sym),
    Mul(Sym, Sym),
    Neg(Sym),
}

thread_local! {
    static ARENA: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
}

/// Clear the arena. Call between independent extractions to keep ids small and
/// output deterministic. (Handles from a previous extraction become invalid.)
pub fn reset() {
    ARENA.with(|a| a.borrow_mut().clear());
}

/// A snapshot of the current arena, indexed by handle id. Node `id` only
/// references children with strictly smaller ids (a node is interned after its
/// children), so iterating `0..len` is a topological order — exactly what the
/// straight-line `let`-renderer and the memoized evaluator rely on. Sentinel
/// handles (the `const` field elements) are *not* in the arena and must be
/// resolved separately.
pub fn arena_snapshot() -> Vec<Node> {
    ARENA.with(|a| a.borrow().clone())
}

/// `true` iff `s` is one of the reserved sentinel handles (`ZERO/ONE/TWO/…`),
/// i.e. not an index into the arena.
pub fn is_sentinel(s: Sym) -> bool {
    s.0 >= SENTINEL_MIN
}

fn intern(node: Node) -> Sym {
    ARENA.with(|a| {
        let mut v = a.borrow_mut();
        let id = v.len() as u32;
        assert!(id < SENTINEL_MIN, "symbolic arena overflow");
        v.push(node);
        Sym(id)
    })
}

/// A `Copy` handle to a symbolic expression.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Sym(pub u32);

impl Sym {
    /// The variable `local_wires[i]`.
    pub fn wire(i: usize) -> Self {
        intern(Node::Wire(i))
    }

    /// The variable `local_constants[i]`.
    pub fn local_const(i: usize) -> Self {
        intern(Node::LocalConst(i))
    }

    /// Resolve this handle to its (cloned) node, interpreting sentinels.
    pub fn node(self) -> Node {
        match self.0 {
            ID_ZERO => Node::Const(0),
            ID_ONE => Node::Const(1),
            ID_TWO => Node::Const(2),
            ID_NEG_ONE => Node::NegOne,
            // The generators never appear in arithmetic-gate / base-sum
            // constraints; give them harmless distinct constants so misuse is at
            // least visible rather than silently aliasing a real value.
            ID_MULT_GEN => Node::Const(7),
            ID_POW2_GEN => Node::Const(7),
            id => ARENA.with(|a| a.borrow()[id as usize].clone()),
        }
    }

    fn is_zero_expr(self) -> bool {
        matches!(self.node(), Node::Const(0))
    }

    fn is_one_expr(self) -> bool {
        matches!(self.node(), Node::Const(1))
    }
}

// ----------------------------------------------------------------------------
// Arithmetic ops. Light identity folding (`x+0`, `x*1`, `x*0`, `x-0`, `0-x`)
// keeps the extracted DAG in the same shape as the hand-written Lean models
// (e.g. Horner form for `reduce_with_powers`, `x*(x-1)` for the B=2 range
// product), which makes the `ring`-equivalence lemmas trivial.
// ----------------------------------------------------------------------------

impl Add for Sym {
    type Output = Sym;
    fn add(self, rhs: Sym) -> Sym {
        if self.is_zero_expr() {
            return rhs;
        }
        if rhs.is_zero_expr() {
            return self;
        }
        intern(Node::Add(self, rhs))
    }
}

impl Sub for Sym {
    type Output = Sym;
    fn sub(self, rhs: Sym) -> Sym {
        if rhs.is_zero_expr() {
            return self;
        }
        if self.is_zero_expr() {
            return -rhs;
        }
        intern(Node::Sub(self, rhs))
    }
}

impl Mul for Sym {
    type Output = Sym;
    fn mul(self, rhs: Sym) -> Sym {
        if self.is_zero_expr() || rhs.is_zero_expr() {
            return Sym::ZERO;
        }
        if self.is_one_expr() {
            return rhs;
        }
        if rhs.is_one_expr() {
            return self;
        }
        intern(Node::Mul(self, rhs))
    }
}

impl Neg for Sym {
    type Output = Sym;
    fn neg(self) -> Sym {
        if self.is_zero_expr() {
            return Sym::ZERO;
        }
        intern(Node::Neg(self))
    }
}

impl Div for Sym {
    type Output = Sym;
    fn div(self, _rhs: Sym) -> Sym {
        panic!("symbolic division is not supported (no gate constraint divides)")
    }
}

impl AddAssign for Sym {
    fn add_assign(&mut self, rhs: Sym) {
        *self = *self + rhs;
    }
}
impl SubAssign for Sym {
    fn sub_assign(&mut self, rhs: Sym) {
        *self = *self - rhs;
    }
}
impl MulAssign for Sym {
    fn mul_assign(&mut self, rhs: Sym) {
        *self = *self * rhs;
    }
}
impl DivAssign for Sym {
    fn div_assign(&mut self, rhs: Sym) {
        *self = *self / rhs;
    }
}

impl Sum for Sym {
    fn sum<I: Iterator<Item = Sym>>(iter: I) -> Sym {
        iter.fold(Sym::ZERO, |acc, x| acc + x)
    }
}
impl<'a> Sum<&'a Sym> for Sym {
    fn sum<I: Iterator<Item = &'a Sym>>(iter: I) -> Sym {
        iter.fold(Sym::ZERO, |acc, &x| acc + x)
    }
}
impl Product for Sym {
    fn product<I: Iterator<Item = Sym>>(iter: I) -> Sym {
        iter.fold(Sym::ONE, |acc, x| acc * x)
    }
}
impl<'a> Product<&'a Sym> for Sym {
    fn product<I: Iterator<Item = &'a Sym>>(iter: I) -> Sym {
        iter.fold(Sym::ONE, |acc, &x| acc * x)
    }
}

impl Default for Sym {
    fn default() -> Self {
        Sym::ZERO
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sym({})", self.0)
    }
}

// `Field` requires `Eq + Hash` for use as map keys etc.; the structural derive on
// the `u32` handle is fine because no gate's `eval_unfiltered` we target branches
// on field equality. `PartialOrd/Ord` are not required by `Field` but cheap.
impl PartialOrd for Sym {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Sym {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Sample for Sym {
    fn sample<R>(_rng: &mut R) -> Self
    where
        R: rand::Rng + ?Sized,
    {
        // Sampling is meaningless for a symbolic element; never called during
        // constraint extraction.
        Sym::ZERO
    }
}

impl Field for Sym {
    const ZERO: Self = Sym(ID_ZERO);
    const ONE: Self = Sym(ID_ONE);
    const TWO: Self = Sym(ID_TWO);
    const NEG_ONE: Self = Sym(ID_NEG_ONE);

    const TWO_ADICITY: usize = 32;
    const CHARACTERISTIC_TWO_ADICITY: usize = 32;

    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Sym(ID_MULT_GEN);
    const POWER_OF_TWO_GENERATOR: Self = Sym(ID_POW2_GEN);

    const BITS: usize = 64;

    fn order() -> BigUint {
        BigUint::from(GOLDILOCKS_ORDER)
    }
    fn characteristic() -> BigUint {
        Self::order()
    }

    fn try_inverse(&self) -> Option<Self> {
        panic!("symbolic inverse is not supported (no gate constraint inverts)")
    }

    fn from_noncanonical_biguint(n: BigUint) -> Self {
        Self::from_canonical_u64(
            (n % BigUint::from(GOLDILOCKS_ORDER))
                .try_into()
                .expect("reduced value fits in u64"),
        )
    }

    fn from_canonical_u64(n: u64) -> Self {
        match n {
            0 => Sym::ZERO,
            1 => Sym::ONE,
            2 => Sym::TWO,
            _ => intern(Node::Const(n)),
        }
    }

    fn from_noncanonical_u128(n: u128) -> Self {
        Self::from_canonical_u64((n % (GOLDILOCKS_ORDER as u128)) as u64)
    }

    fn from_noncanonical_u64(n: u64) -> Self {
        Self::from_canonical_u64(n % GOLDILOCKS_ORDER)
    }

    fn from_noncanonical_i64(n: i64) -> Self {
        if n >= 0 {
            Self::from_noncanonical_u64(n as u64)
        } else {
            -Self::from_noncanonical_u64((-n) as u64)
        }
    }
}

impl PrimeField for Sym {
    fn to_canonical_biguint(&self) -> BigUint {
        panic!("symbolic element has no canonical value")
    }
}

impl Field64 for Sym {
    const ORDER: u64 = GOLDILOCKS_ORDER;
}

impl PrimeField64 for Sym {
    // Constant nodes have a genuine canonical value; this is how a gate's round
    // constants round-trip through `ext_c` (`from_canonical_u64(x.to_canonical_u64())`,
    // poseidon2.rs). A non-constant symbol has no value, so that genuinely panics.
    fn to_canonical_u64(&self) -> u64 {
        match self.node() {
            Node::Const(n) => n,
            _ => panic!("to_canonical_u64 on a non-constant symbolic element"),
        }
    }
    fn to_noncanonical_u64(&self) -> u64 {
        match self.node() {
            Node::Const(n) => n,
            _ => panic!("to_noncanonical_u64 on a non-constant symbolic element"),
        }
    }
}

// `Frobenius<1>` is the only thing needed (beyond `Field`) for the blanket
// `Extendable<1>` impl to apply; all of its methods are trivial defaults.
impl Frobenius<1> for Sym {}

// Poseidon is a supertrait of `RichField`. The arithmetic gate and base-sum gate
// never invoke the hash, so the round-constant tables are dummy zeros and
// `permute` panics. Implementing it just satisfies the `RichField` bound.
impl Poseidon for Sym {
    const MDS_MATRIX_CIRC: [u64; SPONGE_WIDTH] = [0; SPONGE_WIDTH];
    const MDS_MATRIX_DIAG: [u64; SPONGE_WIDTH] = [0; SPONGE_WIDTH];
    const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [u64; SPONGE_WIDTH] = [0; SPONGE_WIDTH];
    const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS] = [0; N_PARTIAL_ROUNDS];
    const FAST_PARTIAL_ROUND_VS: [[u64; SPONGE_WIDTH - 1]; N_PARTIAL_ROUNDS] =
        [[0; SPONGE_WIDTH - 1]; N_PARTIAL_ROUNDS];
    const FAST_PARTIAL_ROUND_W_HATS: [[u64; SPONGE_WIDTH - 1]; N_PARTIAL_ROUNDS] =
        [[0; SPONGE_WIDTH - 1]; N_PARTIAL_ROUNDS];
    const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[u64; SPONGE_WIDTH - 1]; SPONGE_WIDTH - 1] =
        [[0; SPONGE_WIDTH - 1]; SPONGE_WIDTH - 1];
}

impl RichField for Sym {}
