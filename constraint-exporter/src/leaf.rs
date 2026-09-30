//! Bridge generator for the recorded leaf circuit (PLAN.md Step 9d).
//!
//! `Generated/LeafCircuit.lean` (Step 9b) proves the meaning of each of the gadget calls
//! `build_leaf_constraints` made. This module walks the recorded call sequence once more,
//! checking each call against the role it must play in `wormhole/circuit/src/circuit.rs`
//! (which named targets it reads, which constants, how the outputs chain), and renders
//! `formal/Plonky2Bridge/Generated/Leaf.lean`: the `LeafPublic` / `LeafWitness` read off the
//! wiring and `sound : Satisfies → Poseidon2Rows perm → Rleaf (spongeRO perm) (pub a) (wit a)`,
//! assembled from the lemmas of `Plonky2Bridge/Leaf.lean`.
//!
//! The walk is a structural check as much as a generator: a leaf circuit whose gadget
//! sequence drifts from the shape below fails here with the offending call.

use core::fmt::Write as _;
use std::collections::BTreeSet;

use plonky2::field::types::PrimeField64;
use plonky2::iop::target::Target;

use crate::circuit::lean_target;
use crate::gadget::{Call, Fact};
use crate::trace::{load, traces_dir, LoadedTrace};

/// The leaf public-input layout (`qp-wormhole-inputs`).
mod layout {
    pub const LEAF_PI_LEN: usize = 22;
    pub const NULLIFIER: usize = 4;
    pub const BLOCK_HASH: usize = 16;
    pub const BLOCK_NUMBER: usize = 20;
    pub const INPUT_AMOUNT: usize = 21;
}

const MAX_DEPTH: usize = 16;
const DEPTH_BITS: usize = 5;

/// A `(sub, mul, connect-to-zero)` triple: `(x - y) * flag = 0`.
#[derive(Clone, Copy, Debug)]
struct Gate {
    sub: usize,
    mul: usize,
    zero: usize,
}

/// An `is_const_less_than(c, depth, 5)` loop: its 50 op facts in call order, the constant's
/// bits (LSB first) and the output target.
#[derive(Debug)]
struct Loop {
    facts: Vec<usize>,
    bits: [bool; DEPTH_BITS],
    out: Target,
}

#[derive(Debug)]
struct Level {
    split: usize,
    bits: [Target; DEPTH_BITS],
    lt: Loop,
    range: usize,
    eq: [usize; 4],
    /// `a_j = select(e0, cur_j, s0_j)`.
    sel_a: [usize; 4],
    /// `(m_j = select(e0, s0_j, s1_j), b_j = select(e1, cur_j, m_j))`.
    sel_b: [(usize, usize); 4],
    /// The first `or(e0, e1)` (the circuit re-emits it per limb, same output wire).
    or: usize,
    /// `(n_j = select(or, s1_j, s2_j), c_j = select(e2, cur_j, n_j))`.
    sel_c: [(usize, usize); 4],
    /// `d_j = select(e3, cur_j, s2_j)`.
    sel_d: [usize; 4],
    hash: usize,
    /// `next_j = select(lt, hash_j, cur_j)`.
    next: [usize; 4],
    cur: [Target; 4],
    next_out: [Target; 4],
}

#[derive(Debug)]
pub struct Shape {
    calls: Vec<Call>,
    /// `(target, value)` in the order of the export's `constants` (the `k{i}` of
    /// `leafCircuit_consts`).
    constants: Vec<(Target, u64)>,
    pis: Vec<Target>,
    secret: [Target; 4],
    transfer_count: [Target; 2],
    to_account: [Target; 4],
    root_hash: [Target; 4],
    depth: Target,
    is_not_dummy: Target,
    header_parent_hash: [Target; 4],
    header_state_root: [Target; 4],
    header_extrinsics_root: [Target; 4],
    header_zk_tree_root: [Target; 4],
    header_digest: Vec<Target>,
    assert_bool: usize,
    wa_hash: (usize, usize),
    wa_connect: [usize; 4],
    /// `range_check 32` on `tc0, tc1, asset_id, input_amount, out1, out2, fee`.
    range32: [usize; 7],
    leaf_hash: usize,
    leaf_out: [Target; 4],
    depth_split: usize,
    depth_bits: [Target; DEPTH_BITS],
    depth_lt: Loop,
    depth_zero: usize,
    levels: Vec<Level>,
    root_bind: [Gate; 4],
    bn_range: usize,
    secret_connect: [usize; 4],
    tc_connect: [usize; 2],
    account_connect: [usize; 4],
    dummy_eq: [usize; 6],
    /// `and(e0, e1)`, `and(e2, e3)`, `and(·, ·)`, `and(e4, e5)`, `and(·, ·)`.
    dummy_and: [usize; 5],
    dummy_sub: usize,
    dummy_connect: usize,
    null_hash: (usize, usize),
    null_bind: [Gate; 4],
    header_hash: usize,
    header_bind: [Gate; 4],
    zk_bind: [Gate; 4],
}

struct Cursor<'a> {
    calls: &'a [Call],
    k: usize,
}

impl Cursor<'_> {
    fn expect<T>(
        &mut self,
        what: &str,
        f: impl Fn(&Fact) -> Option<T>,
    ) -> Result<(usize, T), String> {
        let k = self.k;
        let call = self
            .calls
            .get(k)
            .ok_or_else(|| format!("call {k}: expected {what}, but the trace ends"))?;
        self.k += 1;
        f(&call.fact)
            .map(|v| (k, v))
            .ok_or_else(|| format!("call {k}: expected {what}, got {:?}", call.fact))
    }

    fn select(
        &mut self,
        b: Target,
        x: Target,
        y: Target,
        what: &str,
    ) -> Result<(usize, Target), String> {
        self.expect(&format!("select for {what}"), |f| match *f {
            Fact::Select {
                b: b1,
                x: x1,
                y: y1,
                out,
            } if b1 == b && x1 == x && y1 == y => Some(out),
            _ => None,
        })
    }

    fn not(&mut self, b: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("not for {what}"), |f| match *f {
            Fact::Not { b: b1, out } if b1 == b => Some(out),
            _ => None,
        })
    }

    fn and(&mut self, b1: Target, b2: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("and for {what}"), |f| match *f {
            Fact::And { b1: x, b2: y, out } if x == b1 && y == b2 => Some(out),
            _ => None,
        })
    }

    fn or(&mut self, b1: Target, b2: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("or for {what}"), |f| match *f {
            Fact::Or { b1: x, b2: y, out } if x == b1 && y == b2 => Some(out),
            _ => None,
        })
    }

    fn add(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("add for {what}"), |f| match *f {
            Fact::Add { x: x1, y: y1, out } if x1 == x && y1 == y => Some(out),
            _ => None,
        })
    }

    fn sub(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("sub for {what}"), |f| match *f {
            Fact::Sub { x: x1, y: y1, out } if x1 == x && y1 == y => Some(out),
            _ => None,
        })
    }

    fn mul(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("mul for {what}"), |f| match *f {
            Fact::Mul { x: x1, y: y1, out } if x1 == x && y1 == y => Some(out),
            _ => None,
        })
    }

    fn is_equal(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        self.expect(&format!("is_equal for {what}"), |f| match *f {
            Fact::IsEqual {
                x: x1,
                y: y1,
                equal,
                ..
            } if x1 == x && y1 == y => Some(equal),
            _ => None,
        })
    }

    fn connect(&mut self, x: Target, y: Target, what: &str) -> Result<usize, String> {
        self.expect(&format!("connect for {what}"), |f| match *f {
            Fact::Connect { x: x1, y: y1 } if x1 == x && y1 == y => Some(()),
            _ => None,
        })
        .map(|(k, ())| k)
    }

    fn range_check(&mut self, x: Target, bits: usize, what: &str) -> Result<usize, String> {
        self.expect(&format!("range_check {bits} for {what}"), |f| match *f {
            Fact::RangeCheck { x: x1, bits: b } if x1 == x && b == bits => Some(()),
            _ => None,
        })
        .map(|(k, ())| k)
    }

    fn assert_bool(&mut self, b: Target, what: &str) -> Result<usize, String> {
        self.expect(&format!("assert_bool for {what}"), |f| match *f {
            Fact::AssertBool { b: b1 } if b1 == b => Some(()),
            _ => None,
        })
        .map(|(k, ())| k)
    }

    fn split_le(
        &mut self,
        x: Target,
        bits: usize,
        what: &str,
    ) -> Result<(usize, Vec<Target>), String> {
        self.expect(&format!("split_le {bits} for {what}"), |f| match *f {
            Fact::SplitLe {
                x: x1,
                row,
                bits: b,
            } if x1 == x && b == bits => {
                Some((1..=bits).map(|col| Target::wire(row, col)).collect())
            }
            _ => None,
        })
    }

    fn poseidon2(&mut self, inputs: &[Target], what: &str) -> Result<(usize, [Target; 4]), String> {
        self.expect(&format!("poseidon2 for {what}"), |f| match f {
            Fact::Poseidon2 { rows, inputs: i } if i == inputs => {
                let last = *rows.last()?;
                Some([12, 13, 14, 15].map(|col| Target::wire(last, col)))
            }
            _ => None,
        })
    }

    /// `(x - y) * flag` connected to zero.
    fn gate(
        &mut self,
        x: Target,
        y: Target,
        flag: Target,
        zero: Target,
        what: &str,
    ) -> Result<Gate, String> {
        let (sub, diff) = self.sub(x, y, what)?;
        let (mul, prod) = self.mul(diff, flag, what)?;
        let zero = self.connect(prod, zero, what)?;
        Ok(Gate { sub, mul, zero })
    }

    /// `is_const_less_than(c, x, 5)` over the bits `bits` of `x` (common/gadgets.rs): per
    /// bit, most significant first, `not(c)`, `and(¬c, b)`, `and(·, eq)`, `or(lt, ·)`,
    /// `mul(c, b)`, `mul(2, ·)`, `add(c, b)`, `sub(·, ·)`, `not(·)`, `and(eq, ·)`.
    fn lt_loop(
        &mut self,
        c: usize,
        bits: &[Target],
        k: &Consts,
        what: &str,
    ) -> Result<Loop, String> {
        let mut facts = Vec::new();
        let mut lt = k.zero;
        let mut eq = k.one;
        let mut cbits = [false; DEPTH_BITS];
        for i in (0..DEPTH_BITS).rev() {
            let cbit = (c >> i) & 1 == 1;
            cbits[i] = cbit;
            let ct = if cbit { k.one } else { k.zero };
            let b = bits[i];
            let w = format!("{what} bit {i}");
            let (k1, not_c) = self.not(ct, &w)?;
            let (k2, t1) = self.and(not_c, b, &w)?;
            let (k3, t2) = self.and(t1, eq, &w)?;
            let (k4, lt1) = self.or(lt, t2, &w)?;
            let (k5, cb) = self.mul(ct, b, &w)?;
            let (k6, two_cb) = self.mul(k.two, cb, &w)?;
            let (k7, s) = self.add(ct, b, &w)?;
            let (k8, x) = self.sub(s, two_cb, &w)?;
            let (k9, nx) = self.not(x, &w)?;
            let (k10, eq1) = self.and(eq, nx, &w)?;
            facts.extend([k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]);
            lt = lt1;
            eq = eq1;
        }
        Ok(Loop {
            facts,
            bits: cbits,
            out: lt,
        })
    }
}

struct Consts {
    zero: Target,
    one: Target,
    two: Target,
    three: Target,
}

fn arr4(v: &[Target]) -> Result<[Target; 4], String> {
    <[Target; 4]>::try_from(v).map_err(|_| format!("expected 4 targets, found {}", v.len()))
}

impl Shape {
    pub fn read(t: &LoadedTrace) -> Result<Shape, String> {
        let named = |name: &str| -> Result<&Vec<Target>, String> {
            t.ex.named
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, ts)| ts)
                .ok_or_else(|| format!("trace has no named targets {name:?}"))
        };
        let named4 = |name: &str| -> Result<[Target; 4], String> { arr4(named(name)?) };
        let pis = t.ex.public_inputs.clone();
        if pis.len() != layout::LEAF_PI_LEN {
            return Err(format!("expected 22 public inputs, found {}", pis.len()));
        }
        let secret = named4("secret")?;
        let transfer_count = <[Target; 2]>::try_from(named("transfer_count")?.as_slice())
            .map_err(|_| "transfer_count is not 2 targets".to_string())?;
        let to_account = named4("to_account")?;
        let account_id = named4("account_id")?;
        let root_hash = named4("root_hash")?;
        let depth = match named("depth")?.as_slice() {
            [d] => *d,
            _ => return Err("depth is not one target".into()),
        };
        let positions = named("positions")?.clone();
        if positions.len() != MAX_DEPTH {
            return Err(format!("expected 16 positions, found {}", positions.len()));
        }
        let is_not_dummy = match named("is_not_dummy")?.as_slice() {
            [d] => *d,
            _ => return Err("is_not_dummy is not one target".into()),
        };
        let siblings: Vec<Vec<Target>> = (0..MAX_DEPTH)
            .map(|i| named(&format!("siblings_{i}")).cloned())
            .collect::<Result<_, _>>()?;
        if siblings.iter().any(|s| s.len() != 12) {
            return Err("siblings_i is not 12 targets".into());
        }
        let header_parent_hash = named4("header_parent_hash")?;
        let header_state_root = named4("header_state_root")?;
        let header_extrinsics_root = named4("header_extrinsics_root")?;
        let header_zk_tree_root = named4("header_zk_tree_root")?;
        let header_digest = named("header_digest")?.clone();

        let constants: Vec<(Target, u64)> =
            t.ex.constants
                .iter()
                .map(|(x, v)| (*x, v.to_canonical_u64()))
                .collect();
        let const_target = |v: u64| -> Result<Target, String> {
            let hits: Vec<Target> = constants
                .iter()
                .filter(|(_, c)| *c == v)
                .map(|(x, _)| *x)
                .collect();
            match hits.as_slice() {
                [x] => Ok(*x),
                _ => Err(format!(
                    "expected exactly one constant target = {v}, found {}",
                    hits.len()
                )),
            }
        };
        let const_value = |x: Target| -> Result<u64, String> {
            constants
                .iter()
                .find(|(c, _)| *c == x)
                .map(|(_, v)| *v)
                .ok_or_else(|| format!("{} is not a constant target", lean_target(x)))
        };
        // The salts must be the `string_to_felts` encodings `WormholeSpec.wormholeSalt` /
        // `nullifierSalt` are defined as; `WA_of_hashes` / `Null_of_hashes` are stated on them.
        let check_salt = |name: &str, salt: &[u8; 8], inputs: &[Target]| -> Result<(), String> {
            let expected = [
                u64::from(u32::from_le_bytes([salt[0], salt[1], salt[2], salt[3]])),
                u64::from(u32::from_le_bytes([salt[4], salt[5], salt[6], salt[7]])),
                1,
            ];
            let found = [
                const_value(inputs[0])?,
                const_value(inputs[1])?,
                const_value(inputs[2])?,
            ];
            if found != expected {
                return Err(format!(
                    "{name} salt is {found:?}, but WormholeSpec defines it as {expected:?}"
                ));
            }
            Ok(())
        };
        let k = Consts {
            zero: const_target(0)?,
            one: const_target(1)?,
            two: const_target(2)?,
            three: const_target(3)?,
        };
        let eq_consts = [k.zero, k.one, k.two, k.three];

        let mut c = Cursor {
            calls: &t.calls,
            k: 0,
        };

        let assert_bool = c.assert_bool(is_not_dummy, "is_not_dummy")?;

        // UnspendableAccount: `H(H(salt ‖ secret))` connected to `account_id`.
        let (wa1, wa_mid) = c.expect("the wormhole-address salt hash", |f| match f {
            Fact::Poseidon2 { rows, inputs } if inputs.len() == 7 => {
                let last = *rows.last()?;
                Some((
                    inputs.clone(),
                    [12, 13, 14, 15].map(|col| Target::wire(last, col)),
                ))
            }
            _ => None,
        })?;
        let (wa_inputs, wa_mid) = wa_mid;
        check_salt("wormhole-address", b"wormhole", &wa_inputs)?;
        let wa_secret = arr4(&wa_inputs[3..])?;
        let (wa2, wa_out) = c.poseidon2(&wa_mid, "the wormhole-address outer hash")?;
        let mut wa_connect = [0; 4];
        for j in 0..4 {
            wa_connect[j] = c.connect(wa_out[j], account_id[j], "account_id")?;
        }

        // ZkLeaf::collect_32_bit_targets, then the leaf hash.
        let (leaf_tc, range32, leaf_hash, leaf_out) = {
            let (r0, tc0) = c.expect("range_check 32 of transfer_count[0]", |f| match *f {
                Fact::RangeCheck { x, bits: 32 } => Some(x),
                _ => None,
            })?;
            let (r1, tc1) = c.expect("range_check 32 of transfer_count[1]", |f| match *f {
                Fact::RangeCheck { x, bits: 32 } => Some(x),
                _ => None,
            })?;
            let r2 = c.range_check(pis[0], 32, "asset_id")?;
            let r3 = c.range_check(pis[layout::INPUT_AMOUNT], 32, "input_amount")?;
            let r4 = c.range_check(pis[1], 32, "output_amount_1")?;
            let r5 = c.range_check(pis[2], 32, "output_amount_2")?;
            let r6 = c.range_check(pis[3], 32, "volume_fee_bps")?;
            let mut inputs = to_account.to_vec();
            inputs.extend([tc0, tc1, pis[0], pis[layout::INPUT_AMOUNT]]);
            let (kh, out) = c.poseidon2(&inputs, "the leaf hash")?;
            ([tc0, tc1], [r0, r1, r2, r3, r4, r5, r6], kh, out)
        };

        // enforce_target_less_than_const(depth, MAX_DEPTH + 1, 5).
        let (depth_split, depth_bits) = c.split_le(depth, DEPTH_BITS, "depth")?;
        let depth_bits = <[Target; DEPTH_BITS]>::try_from(depth_bits).unwrap();
        let depth_lt = c.lt_loop(MAX_DEPTH, &depth_bits, &k, "is_const_less_than(16, depth)")?;
        let depth_zero = c.connect(depth_lt.out, k.zero, "depth ≤ MAX_DEPTH")?;

        // ZkMerkleProofData::constraints.
        let mut levels = Vec::with_capacity(MAX_DEPTH);
        let mut cur = leaf_out;
        for i in 0..MAX_DEPTH {
            let w = format!("level {i}");
            let (split, bits) = c.split_le(depth, DEPTH_BITS, &w)?;
            let bits = <[Target; DEPTH_BITS]>::try_from(bits).unwrap();
            let lt = c.lt_loop(i, &bits, &k, &format!("is_const_less_than({i}, depth)"))?;
            let pos = positions[i];
            let range = c.range_check(pos, 2, &w)?;
            let mut eq = [0; 4];
            let mut e = [pos; 4];
            for (q, kc) in eq_consts.iter().enumerate() {
                let (kk, flag) = c.is_equal(pos, *kc, &format!("{w} is_equal(pos, {q})"))?;
                eq[q] = kk;
                e[q] = flag;
            }
            let s = &siblings[i];
            let (s0, s1, s2) = (&s[0..4], &s[4..8], &s[8..12]);
            let mut sel_a = [0; 4];
            let mut a_out = [pos; 4];
            for j in 0..4 {
                let (kk, o) = c.select(e[0], cur[j], s0[j], &format!("{w} child 0 limb {j}"))?;
                sel_a[j] = kk;
                a_out[j] = o;
            }
            let mut sel_b = [(0, 0); 4];
            let mut b_out = [pos; 4];
            for j in 0..4 {
                let (km, m) =
                    c.select(e[0], s0[j], s1[j], &format!("{w} child 1 inner limb {j}"))?;
                let (kb, o) = c.select(e[1], cur[j], m, &format!("{w} child 1 limb {j}"))?;
                sel_b[j] = (km, kb);
                b_out[j] = o;
            }
            let mut or = None;
            let mut or_out = pos;
            let mut sel_c = [(0, 0); 4];
            let mut c_out = [pos; 4];
            for j in 0..4 {
                let (ko, o01) = c.or(e[0], e[1], &format!("{w} or(e0, e1) limb {j}"))?;
                if or.is_some() {
                    if o01 != or_out {
                        return Err(format!(
                            "call {ko}: {w} re-emitted or(e0, e1) on a new wire"
                        ));
                    }
                } else {
                    or = Some(ko);
                    or_out = o01;
                }
                let (kn, n) =
                    c.select(o01, s1[j], s2[j], &format!("{w} child 2 inner limb {j}"))?;
                let (kc, o) = c.select(e[2], cur[j], n, &format!("{w} child 2 limb {j}"))?;
                sel_c[j] = (kn, kc);
                c_out[j] = o;
            }
            let mut sel_d = [0; 4];
            let mut d_out = [pos; 4];
            for j in 0..4 {
                let (kk, o) = c.select(e[3], cur[j], s2[j], &format!("{w} child 3 limb {j}"))?;
                sel_d[j] = kk;
                d_out[j] = o;
            }
            let mut inputs = Vec::with_capacity(16);
            inputs.extend(a_out);
            inputs.extend(b_out);
            inputs.extend(c_out);
            inputs.extend(d_out);
            let (hash, hout) = c.poseidon2(&inputs, &format!("{w} node hash"))?;
            let mut next = [0; 4];
            let mut next_out = [pos; 4];
            for j in 0..4 {
                let (kk, o) =
                    c.select(lt.out, hout[j], cur[j], &format!("{w} is_active limb {j}"))?;
                next[j] = kk;
                next_out[j] = o;
            }
            levels.push(Level {
                split,
                bits,
                lt,
                range,
                eq,
                sel_a,
                sel_b,
                or: or.unwrap(),
                sel_c,
                sel_d,
                hash,
                next,
                cur,
                next_out,
            });
            cur = next_out;
        }
        let mut root_bind = [Gate {
            sub: 0,
            mul: 0,
            zero: 0,
        }; 4];
        for j in 0..4 {
            root_bind[j] = c.gate(cur[j], root_hash[j], is_not_dummy, k.zero, "root binding")?;
        }

        // BlockHeader::circuit_without_hash_binding.
        let bn_range = c.range_check(pis[layout::BLOCK_NUMBER], 32, "block_number")?;

        // connect_shared_targets.
        let mut secret_connect = [0; 4];
        for j in 0..4 {
            secret_connect[j] = c.connect(secret[j], wa_secret[j], "secret")?;
        }
        let mut tc_connect = [0; 2];
        for j in 0..2 {
            tc_connect[j] = c.connect(transfer_count[j], leaf_tc[j], "transfer_count")?;
        }
        let mut account_connect = [0; 4];
        for j in 0..4 {
            account_connect[j] = c.connect(account_id[j], to_account[j], "to_account")?;
        }
        let mut dummy_eq = [0; 6];
        let mut de = [k.zero; 6];
        for j in 0..4 {
            let (kk, flag) = c.is_equal(pis[layout::BLOCK_HASH + j], k.zero, "block_hash == 0")?;
            dummy_eq[j] = kk;
            de[j] = flag;
        }
        let (ka, fa) = c.and(de[0], de[1], "dummy and(e0, e1)")?;
        let (kb, fb) = c.and(de[2], de[3], "dummy and(e2, e3)")?;
        let (kc, fc) = c.and(fa, fb, "dummy and-root of block_hash")?;
        for j in 0..2 {
            let (kk, flag) = c.is_equal(pis[1 + j], k.zero, "output_amount == 0")?;
            dummy_eq[4 + j] = kk;
            de[4 + j] = flag;
        }
        let (kd, fd) = c.and(de[4], de[5], "dummy and(e4, e5)")?;
        let (ke, is_dummy) = c.and(fc, fd, "is_dummy")?;
        let (dummy_sub, not_dummy) = c.sub(k.one, is_dummy, "is_not_dummy = 1 - is_dummy")?;
        let dummy_connect = c.connect(is_not_dummy, not_dummy, "is_not_dummy")?;

        let (n1, (null_salt, null_mid)) = c.expect("the nullifier salt hash", |f| match f {
            Fact::Poseidon2 { rows, inputs }
                if inputs.len() == 9
                    && inputs[3..7] == secret[..]
                    && inputs[7..9] == transfer_count[..] =>
            {
                let last = *rows.last()?;
                Some((
                    [inputs[0], inputs[1], inputs[2]],
                    [12, 13, 14, 15].map(|col| Target::wire(last, col)),
                ))
            }
            _ => None,
        })?;
        check_salt("nullifier", b"~nullif~", &null_salt)?;
        let (n2, null_out) = c.poseidon2(&null_mid, "the nullifier outer hash")?;
        let mut null_bind = [Gate {
            sub: 0,
            mul: 0,
            zero: 0,
        }; 4];
        for j in 0..4 {
            null_bind[j] = c.gate(
                pis[layout::NULLIFIER + j],
                null_out[j],
                not_dummy,
                k.zero,
                "nullifier binding",
            )?;
        }

        let mut header_inputs = Vec::with_capacity(45);
        header_inputs.extend(header_parent_hash);
        header_inputs.push(pis[layout::BLOCK_NUMBER]);
        header_inputs.extend(header_state_root);
        header_inputs.extend(header_extrinsics_root);
        header_inputs.extend(header_zk_tree_root);
        header_inputs.extend(header_digest.iter().copied());
        let (header_hash, header_out) = c.poseidon2(&header_inputs, "the block-header hash")?;
        let mut header_bind = [Gate {
            sub: 0,
            mul: 0,
            zero: 0,
        }; 4];
        for j in 0..4 {
            header_bind[j] = c.gate(
                pis[layout::BLOCK_HASH + j],
                header_out[j],
                not_dummy,
                k.zero,
                "block_hash binding",
            )?;
        }
        let mut zk_bind = [Gate {
            sub: 0,
            mul: 0,
            zero: 0,
        }; 4];
        for j in 0..4 {
            zk_bind[j] = c.gate(
                header_zk_tree_root[j],
                root_hash[j],
                not_dummy,
                k.zero,
                "zk_tree_root binding",
            )?;
        }
        if c.k != t.calls.len() {
            return Err(format!(
                "{} trailing calls after the zk_tree_root binding",
                t.calls.len() - c.k
            ));
        }

        Ok(Shape {
            calls: t.calls.clone(),
            constants,
            pis,
            secret,
            transfer_count,
            to_account,
            root_hash,
            depth,
            is_not_dummy,
            header_parent_hash,
            header_state_root,
            header_extrinsics_root,
            header_zk_tree_root,
            header_digest,
            assert_bool,
            wa_hash: (wa1, wa2),
            wa_connect,
            range32,
            leaf_hash,
            leaf_out,
            depth_split,
            depth_bits,
            depth_lt,
            depth_zero,
            levels,
            root_bind,
            bn_range,
            secret_connect,
            tc_connect,
            account_connect,
            dummy_eq,
            dummy_and: [ka, kb, kc, kd, ke],
            dummy_sub,
            dummy_connect,
            null_hash: (n1, n2),
            null_bind,
            header_hash,
            header_bind,
            zk_bind,
        })
    }

    fn fact(&self, k: usize) -> &Fact {
        &self.calls[k].fact
    }

    /// The `k{i}` constant equations rewriting every constant target the fact mentions.
    fn const_rewrites(&self, k: usize) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for t in self.fact(k).named() {
            if let Some(i) = self.constants.iter().position(|(c, _)| *c == t) {
                if seen.insert(i) {
                    out.push(format!("k{i}"));
                }
            }
        }
        out
    }

    /// `have f{k} := leafCircuit_f{k} …` with its constants rewritten in.
    fn have_fact(&self, o: &mut String, k: usize) {
        let args = if self.fact(k).is_poseidon2() {
            "perm a h hp"
        } else {
            "a h"
        };
        let _ = writeln!(o, "  have f{k} := leafCircuit_f{k} {args}");
        let rw = self.const_rewrites(k);
        if !rw.is_empty() {
            let _ = writeln!(o, "  rw [{}] at f{k}", rw.join(", "));
        }
    }
}

fn a(t: Target) -> String {
    format!("a ({})", lean_target(t))
}

fn d4(ts: &[Target]) -> String {
    format!(
        "D4 ({})",
        ts.iter().map(|t| a(*t)).collect::<Vec<_>>().join(") (")
    )
}

fn val(t: Target) -> String {
    format!("({}).val", a(t))
}

impl Shape {
    /// `⟨(a pos).val, D4 s0, D4 s1, D4 s2⟩` of level `i`, as a `MerkleLevel`.
    fn level_term(&self, i: usize, positions: &[Target], siblings: &[Vec<Target>]) -> String {
        let s = &siblings[i];
        format!(
            "⟨{}, {}, {}, {}⟩",
            val(positions[i]),
            d4(&s[0..4]),
            d4(&s[4..8]),
            d4(&s[8..12])
        )
    }

    /// `(ltLoop [(cb c0, a b0), …]).1 = a out`, closed by the reverse-oriented op facts.
    fn loop_proof(&self, o: &mut String, name: &str, lp: &Loop, bits: &[Target; DEPTH_BITS]) {
        let pairs: Vec<String> = (0..DEPTH_BITS)
            .map(|i| format!("(cb {}, {})", lp.bits[i], a(bits[i])))
            .collect();
        let mut seen = BTreeSet::new();
        let mut rules = Vec::new();
        for &k in &lp.facts {
            self.have_fact(o, k);
            if seen.insert(format!("{:?}", self.fact(k))) {
                rules.push(format!("← f{k}"));
            }
        }
        let _ = writeln!(
            o,
            "  have {name} : (ltLoop [{}]).1 = {} := by",
            pairs.join(", "),
            a(lp.out)
        );
        let _ = writeln!(
            o,
            "    simp only [ltLoop, cb, Bool.cond_true, Bool.cond_false, {}]",
            rules.join(", ")
        );
    }
}

/// Render the bridge module for the recorded leaf circuit.
pub fn render(shape: &Shape, t: &LoadedTrace) -> String {
    let named = |name: &str| -> &Vec<Target> {
        &t.ex.named.iter().find(|(n, _)| n == name).expect("named").1
    };
    let positions = named("positions").clone();
    let siblings: Vec<Vec<Target>> = (0..MAX_DEPTH)
        .map(|i| named(&format!("siblings_{i}")).clone())
        .collect();
    let pis = &shape.pis;
    let mut o = String::new();
    macro_rules! w {
        ($($arg:tt)*) => { writeln!(o, $($arg)*).unwrap() }
    }

    w!("/-");
    w!("  AUTO-GENERATED by `constraint-exporter` (`src/leaf.rs`) from the recorded leaf circuit");
    w!("  trace `leaf_circuit.json`. Do not edit; regenerate with");
    w!("  `cargo run -p qp-plonky2-constraint-exporter --bin export-constraints`.");
    w!("");
    w!("  Step 9d — `Generated/LeafCircuit.lean` proves, from `Satisfies` on the wiring the real");
    w!(
        "  `CircuitBuilder` emitted plus `Poseidon2Rows perm`, the meaning of each of the {} gadget",
        shape.calls.len()
    );
    w!("  calls `build_leaf_constraints` made. This module reads `LeafPublic` off the 22 public");
    w!("  inputs and `LeafWitness` off the named witness targets, and assembles `Rleaf` from the");
    w!("  lemmas of `Plonky2Bridge/Leaf.lean`: the wormhole address and nullifier double hashes,");
    w!("  the leaf hash, the depth bound, the sixteen gated Merkle levels (`gatedWalk` =");
    w!("  `computeRoot` over the first `depth` levels), the dummy flag and the `is_not_dummy`-gated");
    w!("  bindings. The salts the circuit bakes in are checked by the exporter against the");
    w!("  spec's `wormholeSalt` / `nullifierSalt` encodings, which the hash lemmas are stated on.");
    w!("-/");
    w!("import Plonky2Bridge.Leaf");
    w!("import Plonky2Spec.Generated.LeafCircuit");
    w!("");
    w!("namespace Plonky2Bridge.LeafCircuit");
    w!("");
    w!("open Plonky2Spec (IsBool IsEqual bselect band bnot bor rangeCheck BaseSum)");
    w!("open Plonky2Spec.Wiring");
    w!("open Plonky2Spec.Generated");
    w!("open Plonky2Spec.Poseidon2 (St)");
    w!("open Plonky2Spec.Sponge (spongeHash)");
    w!("open Plonky2Bridge.Leaf");
    w!("open WormholeSpec (Digest LeafPublic LeafWitness MerkleLevel Rleaf goldilocks stepUp");
    w!("  computeRoot headerPreimage)");
    w!("");
    w!("variable {{p : ℕ}} [Fact p.Prime]");
    w!("");
    w!("set_option linter.unusedSimpArgs false");
    w!("");
    w!("/-! ### Reading the spec objects off the wiring -/");
    w!("");
    w!("/-- The leaf public inputs, decoded through `.val`. -/");
    w!("def pub (a : Assignment p) : LeafPublic :=");
    w!(
        "  {{ assetId := {}, outputAmount1 := {}, outputAmount2 := {},",
        val(pis[0]),
        val(pis[1]),
        val(pis[2])
    );
    w!("    volumeFeeBps := {},", val(pis[3]));
    w!("    nullifier := {},", d4(&pis[4..8]));
    w!("    exitAccount1 := {},", d4(&pis[8..12]));
    w!("    exitAccount2 := {},", d4(&pis[12..16]));
    w!("    blockHash := {},", d4(&pis[16..20]));
    w!(
        "    blockNumber := {}, inputAmount := {} }}",
        val(pis[20]),
        val(pis[21])
    );
    w!("");
    w!("/-- The sixteen recorded Merkle levels: position hint and the three sibling digests");
    w!("    (`positions`, `siblings_i`). -/");
    w!("def levels (a : Assignment p) : List MerkleLevel :=");
    w!(
        "  [{}]",
        (0..MAX_DEPTH)
            .map(|i| shape.level_term(i, &positions, &siblings))
            .collect::<Vec<_>>()
            .join(",\n   ")
    );
    w!("");
    w!("/-- The leaf witness, decoded through `.val`: the named witness targets, the block-header");
    w!("    fields and the first `depth` recorded levels. -/");
    w!("def wit (a : Assignment p) : LeafWitness :=");
    w!("  {{ secret := {},", d4(&shape.secret));
    w!(
        "    transferCount := [{}, {}],",
        val(shape.transfer_count[0]),
        val(shape.transfer_count[1])
    );
    w!("    toAccount := {},", d4(&shape.to_account));
    w!("    parentHash := {},", d4(&shape.header_parent_hash));
    w!("    stateRoot := {},", d4(&shape.header_state_root));
    w!(
        "    extrinsicsRoot := {},",
        d4(&shape.header_extrinsics_root)
    );
    w!("    zkTreeRoot := {},", d4(&shape.header_zk_tree_root));
    w!(
        "    digestLogs := [{}],",
        shape
            .header_digest
            .iter()
            .map(|t| val(*t))
            .collect::<Vec<_>>()
            .join(", ")
    );
    w!("    depth := {},", val(shape.depth));
    w!("    levels := (levels a).take {},", val(shape.depth));
    w!("    rootHash := {} }}", d4(&shape.root_hash));
    w!("");
    w!("/-! ### The relation -/");
    w!("");
    w!("set_option maxHeartbeats 4000000 in");
    w!("/-- **The recorded leaf circuit satisfies `Rleaf`.** Every satisfying assignment whose");
    w!("    Poseidon2 rows compute `perm` decodes to an `Rleaf` instance on its public inputs and");
    w!("    witness, for the realized oracle `spongeRO perm`. -/");
    w!("theorem sound (perm : St p → St p) (hpg : goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies (leafCircuit p) a)");
    w!("    (hp : Poseidon2Rows perm (leafCircuit p) a) :");
    w!("    Rleaf (spongeRO perm) (pub a) (wit a) := by");
    w!(
        "  obtain ⟨{}⟩ := leafCircuit_consts a h",
        (0..shape.constants.len())
            .map(|i| format!("k{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Wormhole address.
    let (wa1, wa2) = shape.wa_hash;
    shape.have_fact(&mut o, wa1);
    shape.have_fact(&mut o, wa2);
    for k in shape.wa_connect {
        shape.have_fact(&mut o, k);
    }
    for k in shape.secret_connect {
        shape.have_fact(&mut o, k);
    }
    for k in shape.account_connect {
        shape.have_fact(&mut o, k);
    }
    w!("  have hWA := WA_of_hashes perm hpg f{wa1} f{wa2}");
    w!(
        "  rw [{}, {}] at hWA",
        shape
            .wa_connect
            .iter()
            .zip(shape.account_connect)
            .map(|(kc, ka)| format!("f{kc}, f{ka}"))
            .collect::<Vec<_>>()
            .join(", "),
        shape
            .secret_connect
            .iter()
            .map(|k| format!("← f{k}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Range checks and the leaf hash.
    for k in shape.range32 {
        shape.have_fact(&mut o, k);
    }
    shape.have_fact(&mut o, shape.bn_range);
    for k in shape.tc_connect {
        shape.have_fact(&mut o, k);
    }
    let tc_rw: Vec<String> = shape.tc_connect.iter().map(|k| format!("← f{k}")).collect();
    w!("  rw [{}] at f{}", tc_rw[0], shape.range32[0]);
    w!("  rw [{}] at f{}", tc_rw[1], shape.range32[1]);
    shape.have_fact(&mut o, shape.leaf_hash);
    w!("  have hleaf := leafHash_of_hash perm f{}", shape.leaf_hash);
    w!("  rw [{}] at hleaf", tc_rw.join(", "));

    // Depth bound.
    shape.have_fact(&mut o, shape.depth_split);
    shape.loop_proof(&mut o, "hdloop", &shape.depth_lt, &shape.depth_bits);
    shape.have_fact(&mut o, shape.depth_zero);
    w!(
        "  have hdepth : {} ≤ 16 := depth_le_of_loop hpg f{} (hdloop.trans f{})",
        val(shape.depth),
        shape.depth_split,
        shape.depth_zero
    );

    // Merkle levels.
    for (i, lv) in shape.levels.iter().enumerate() {
        w!("  -- level {i}");
        shape.have_fact(&mut o, lv.split);
        shape.loop_proof(&mut o, &format!("hloop{i}"), &lv.lt, &lv.bits);
        let c: usize = (0..DEPTH_BITS).map(|b| (lv.lt.bits[b] as usize) << b).sum();
        w!(
            "  have hact{i} : {} = if {i} < {} then 1 else 0 := by",
            a(lv.lt.out),
            val(shape.depth)
        );
        w!(
            "    rw [← hloop{i}, isActive5 {c} {} {} {} {} {} (by decide) hpg f{}]",
            lv.lt.bits[0],
            lv.lt.bits[1],
            lv.lt.bits[2],
            lv.lt.bits[3],
            lv.lt.bits[4],
            lv.split
        );
        shape.have_fact(&mut o, lv.range);
        for k in lv.eq {
            shape.have_fact(&mut o, k);
        }
        for k in lv.sel_a {
            shape.have_fact(&mut o, k);
        }
        for (km, kb) in lv.sel_b {
            shape.have_fact(&mut o, km);
            shape.have_fact(&mut o, kb);
        }
        shape.have_fact(&mut o, lv.or);
        for (kn, kc) in lv.sel_c {
            shape.have_fact(&mut o, kn);
            shape.have_fact(&mut o, kc);
        }
        for k in lv.sel_d {
            shape.have_fact(&mut o, k);
        }
        shape.have_fact(&mut o, lv.hash);
        for k in lv.next {
            shape.have_fact(&mut o, k);
        }
        let mut args = vec![format!("f{}", lv.range)];
        args.extend(lv.eq.iter().map(|k| format!("f{k}")));
        args.extend(lv.sel_a.iter().map(|k| format!("f{k}")));
        args.extend(
            lv.sel_b
                .iter()
                .flat_map(|(m, b)| [format!("f{m}"), format!("f{b}")]),
        );
        args.push(format!("f{}", lv.or));
        args.extend(
            lv.sel_c
                .iter()
                .flat_map(|(n, c)| [format!("f{n}"), format!("f{c}")]),
        );
        args.extend(lv.sel_d.iter().map(|k| format!("f{k}")));
        args.push(format!("f{}", lv.hash));
        w!(
            "  have hstep{i} := stepUp_of_level perm hpg {}",
            args.join(" ")
        );
        w!(
            "  have hlvl{i} : {} = (if {i} < {} then stepUp (spongeRO perm) ({}) {} else {}) :=",
            d4(&lv.next_out),
            val(shape.depth),
            d4(&lv.cur),
            shape.level_term(i, &positions, &siblings),
            d4(&lv.cur)
        );
        w!(
            "    gatedStep (spongeRO perm) hact{i} rfl rfl hstep{i} {}",
            lv.next
                .iter()
                .map(|k| format!("f{k}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    let last = shape.levels.last().unwrap();
    w!(
        "  have hwalk : gatedWalk (spongeRO perm) {} 0 ({}) (levels a) = {} :=",
        val(shape.depth),
        d4(&shape.leaf_out),
        d4(&last.next_out)
    );
    let mut chain = format!(
        "gatedWalk_nil (spongeRO perm) {} {MAX_DEPTH} _",
        val(shape.depth)
    );
    for i in (0..MAX_DEPTH).rev() {
        chain = format!(
            "(gatedWalk_step (spongeRO perm) {} {i} _ _ hlvl{i}).trans ({chain})",
            val(shape.depth)
        );
    }
    w!("    {chain}");
    w!("  rw [gatedWalk_eq, Nat.sub_zero] at hwalk");

    // Root binding, dummy flag, gated bindings.
    shape.have_fact(&mut o, shape.assert_bool);
    for g in shape.root_bind {
        shape.have_fact(&mut o, g.sub);
        shape.have_fact(&mut o, g.mul);
        shape.have_fact(&mut o, g.zero);
    }
    for k in shape.dummy_eq {
        shape.have_fact(&mut o, k);
    }
    for k in shape.dummy_and {
        shape.have_fact(&mut o, k);
    }
    shape.have_fact(&mut o, shape.dummy_sub);
    shape.have_fact(&mut o, shape.dummy_connect);
    let de = shape.dummy_eq;
    let da = shape.dummy_and;
    w!(
        "  have hnd := notDummy_spec f{} f{} f{} f{} f{} f{} f{} f{} f{} f{} f{} f{}",
        de[0],
        de[1],
        de[2],
        de[3],
        de[4],
        de[5],
        da[0],
        da[1],
        da[2],
        da[3],
        da[4],
        shape.dummy_sub
    );
    let (n1, n2) = shape.null_hash;
    shape.have_fact(&mut o, n1);
    shape.have_fact(&mut o, n2);
    w!("  have hnull := Null_of_hashes perm hpg f{n1} f{n2}");
    for g in shape
        .null_bind
        .iter()
        .chain(&shape.header_bind)
        .chain(&shape.zk_bind)
    {
        shape.have_fact(&mut o, g.sub);
        shape.have_fact(&mut o, g.mul);
        shape.have_fact(&mut o, g.zero);
    }
    shape.have_fact(&mut o, shape.header_hash);
    w!("  have hhdr := H_of_hash perm f{}", shape.header_hash);
    w!("  simp only [List.map_cons, List.map_nil] at hhdr");

    // Assemble Rleaf.
    w!("  simp only [Rleaf, pub, wit, LeafPublic.isDummy, WormholeSpec.maxDepth]");
    w!(
        "  refine ⟨inRange32 hpg f{}, inRange32 hpg f{}, inRange32 hpg f{}, inRange32 hpg f{},",
        shape.range32[4],
        shape.range32[5],
        shape.range32[3],
        shape.range32[2]
    );
    w!(
        "    inRange32 hpg f{}, inRange32 hpg f{}, rfl, ?_, hWA, levels_length rfl hdepth, hdepth,",
        shape.range32[6],
        shape.bn_range
    );
    w!("    levels_pos ?_ _, ?_⟩");
    w!("  · intro t ht");
    w!("    simp only [List.mem_cons, List.not_mem_nil, or_false] at ht");
    w!("    rcases ht with rfl | rfl");
    w!("    · exact inRange32 hpg f{}", shape.range32[0]);
    w!("    · exact inRange32 hpg f{}", shape.range32[1]);
    w!("  · intro lvl hl");
    w!("    simp only [levels, List.mem_cons, List.not_mem_nil, or_false] at hl");
    w!(
        "    rcases hl with {}",
        (0..MAX_DEPTH)
            .map(|_| "rfl")
            .collect::<Vec<_>>()
            .join(" | ")
    );
    for lv in &shape.levels {
        w!("    · exact pos_lt_four hpg f{}", lv.range);
    }
    w!("  · intro hreal");
    w!(
        "    have hflag : {} = 1 := hnd.2.mpr hreal",
        a_of_sub_out(shape)
    );
    w!(
        "    have hflag' : {} = 1 := f{}.trans hflag",
        a(shape.is_not_dummy),
        shape.dummy_connect
    );
    let bind = |o: &mut String, name: &str, g: &Gate, flag: &str| {
        let _ = writeln!(
            o,
            "    have {name} := bind_of_gate f{} f{} f{} {flag}",
            g.sub, g.mul, g.zero
        );
    };
    for (j, g) in shape.root_bind.iter().enumerate() {
        bind(&mut o, &format!("hroot{j}"), g, "hflag'");
    }
    for (j, g) in shape.null_bind.iter().enumerate() {
        bind(&mut o, &format!("hnb{j}"), g, "hflag");
    }
    for (j, g) in shape.header_bind.iter().enumerate() {
        bind(&mut o, &format!("hhb{j}"), g, "hflag");
    }
    for (j, g) in shape.zk_bind.iter().enumerate() {
        bind(&mut o, &format!("hzb{j}"), g, "hflag");
    }
    w!("    refine ⟨?_, ?_, ?_, ?_⟩");
    w!("    · rw [hnb0, hnb1, hnb2, hnb3]; exact hnull");
    w!("    · rw [hhb0, hhb1, hhb2, hhb3]; exact hhdr");
    w!("    · rw [hzb0, hzb1, hzb2, hzb3]");
    w!("    · rw [← hleaf, hwalk, hroot0, hroot1, hroot2, hroot3]");
    w!("");
    w!("end Plonky2Bridge.LeafCircuit");
    o
}

/// `a (is_not_dummy wire)` as computed by `sub(1, is_dummy)`.
fn a_of_sub_out(shape: &Shape) -> String {
    match *shape.fact(shape.dummy_sub) {
        Fact::Sub { out, .. } => a(out),
        _ => unreachable!(),
    }
}

/// Build `formal/Plonky2Bridge/Generated/Leaf.lean` from the recorded leaf trace.
pub fn generate_leaf_bridge_lean() -> Result<String, String> {
    let t = load(&traces_dir().join("leaf_circuit.json"))?;
    if t.circuit != "leaf_circuit" {
        return Err(format!("unexpected trace circuit {:?}", t.circuit));
    }
    let shape = Shape::read(&t)?;
    Ok(render(&shape, &t))
}
