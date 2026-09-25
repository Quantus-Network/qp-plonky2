//! A serializable snapshot of a `CircuitBuilder`'s *logical* circuit, for the
//! formal-verification public-input decode exporter (`../formal/PLAN.md` Step 8).
//!
//! The snapshot is taken before `build()`: it records the gate instances with
//! their constants, every copy constraint, the constant targets, the registered
//! public inputs and the virtual-target count. Together these determine the
//! constraint system a satisfying witness must meet — each gate's
//! `eval_unfiltered` on its row's wires, plus equality along copy constraints —
//! without any of `build()`'s compilation (selectors, padding, blinding, the
//! public-input hash), which stays in the trusted proof-system layer.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use serde::{Deserialize, Serialize};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;

/// A target, flattened for serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetSnapshot {
    Wire { row: usize, column: usize },
    Virtual { index: usize },
}

impl From<Target> for TargetSnapshot {
    fn from(t: Target) -> Self {
        match t {
            Target::Wire(w) => TargetSnapshot::Wire {
                row: w.row,
                column: w.column,
            },
            Target::VirtualTarget { index } => TargetSnapshot::Virtual { index },
        }
    }
}

/// One placed gate: its row, the gate's `id()` string, and its constants
/// (canonical `u64`s).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateInstanceSnapshot {
    pub row: usize,
    pub gate_id: String,
    pub constants: Vec<u64>,
}

/// The logical circuit a `CircuitBuilder` holds before `build()`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormalSnapshot {
    pub num_wires: usize,
    pub num_routed_wires: usize,
    pub num_virtual_targets: usize,
    pub gate_instances: Vec<GateInstanceSnapshot>,
    /// Every `connect(a, b)`, in insertion order.
    pub copy_constraints: Vec<(TargetSnapshot, TargetSnapshot)>,
    /// Targets pinned to constants (`builder.constant(c)`), as `(target, c)`.
    pub constants: Vec<(TargetSnapshot, u64)>,
    /// `register_public_input` targets, in registration order.
    pub public_inputs: Vec<TargetSnapshot>,
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Snapshot the logical circuit built so far. See the module docs.
    pub fn formal_snapshot(&self) -> FormalSnapshot {
        let gate_instances = self
            .gate_instances
            .iter()
            .enumerate()
            .map(|(row, gi)| GateInstanceSnapshot {
                row,
                gate_id: gi.gate_ref.0.id(),
                constants: gi.constants.iter().map(|c| c.to_canonical_u64()).collect(),
            })
            .collect();
        let copy_constraints = self
            .copy_constraints
            .iter()
            .map(|cc| (cc.pair.0.into(), cc.pair.1.into()))
            .collect();
        let mut constants: Vec<(TargetSnapshot, u64)> = self
            .targets_to_constants
            .iter()
            .map(|(t, c)| ((*t).into(), c.to_canonical_u64()))
            .collect();
        constants.sort_by_key(|(_, c)| *c);
        FormalSnapshot {
            num_wires: self.config.num_wires,
            num_routed_wires: self.config.num_routed_wires,
            num_virtual_targets: self.virtual_target_index,
            gate_instances,
            copy_constraints,
            constants,
            public_inputs: self.public_inputs.iter().map(|t| (*t).into()).collect(),
        }
    }
}
