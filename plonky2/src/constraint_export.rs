//! Backend-neutral arithmetic expressions for selector-filtered gate constraints.
//!
//! Export uses the existing recursive gate evaluators at base-field inputs.
//! It does not evaluate the full vanishing polynomial, compute permutation
//! products, generate shaders, or select a proving backend. The initial export
//! supports Goldilocks with extension degree two and no lookups.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec, vec::Vec};
use core::fmt;

use hashbrown::HashMap;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::EvaluationTargets;

#[cfg(test)]
mod tests;

/// One base-field input to a gate constraint expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintInput {
    /// A constant/selector column, before removing the selector prefix.
    Constant(usize),
    /// A wire column in the circuit's wire order.
    Wire(usize),
    /// An element of the four-element public-input hash.
    PublicInputHash(usize),
}

/// An arithmetic node. Child indices refer to earlier nodes in the same graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticNode<F: Field> {
    Constant(F),
    Input(ConstraintInput),
    Add(usize, usize),
    Mul(usize, usize),
}

/// The ordered, selector-filtered constraints for one gate type.
#[derive(Debug, Clone)]
pub struct GateConstraintExpression<F: Field> {
    pub gate_id: String,
    /// Nodes in topological order. Only nodes needed by the outputs are retained.
    pub nodes: Vec<ArithmeticNode<F>>,
    /// One node per common gate-constraint slot, including zero padding.
    pub outputs: Vec<usize>,
}

/// Circuit gate constraints without backend code or device-specific layouts.
#[derive(Debug, Clone)]
pub struct GateConstraintProgram<F: Field> {
    pub num_constants: usize,
    pub num_wires: usize,
    pub num_constraints: usize,
    /// Same gate order as `CommonCircuitData::gates`. Sum corresponding outputs
    /// across gates to obtain the CPU gate-constraint vector.
    pub gates: Vec<GateConstraintExpression<F>>,
}

/// A configuration or gate operation the expression exporter cannot represent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintExportError {
    UnsupportedField,
    UnsupportedExtensionDegree(usize),
    UnsupportedLookups,
    InvalidSelectorMetadata,
    UnsupportedGate {
        gate_id: String,
        reason: &'static str,
    },
}

impl fmt::Display for ConstraintExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedField => write!(f, "constraint export requires Goldilocks"),
            Self::UnsupportedExtensionDegree(d) => {
                write!(
                    f,
                    "constraint export requires extension degree two, got {d}"
                )
            }
            Self::UnsupportedLookups => write!(f, "constraint export does not support lookups"),
            Self::InvalidSelectorMetadata => write!(f, "invalid constraint selector metadata"),
            Self::UnsupportedGate { gate_id, reason } => {
                write!(f, "cannot export {gate_id}: {reason}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConstraintExportError {}

/// Export gate constraints using their existing recursive definitions.
///
/// This evaluates gate code symbolically, not on a witness. Unsupported operations
/// detected by the tracer return an error; no partial program is returned.
/// Custom gate evaluators must obey the same contracts as normal recursive
/// evaluation, including valid wire and constant indices. Panics in arbitrary
/// custom gate code are not caught.
pub fn export_gate_constraints<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
) -> Result<GateConstraintProgram<F>, ConstraintExportError> {
    if F::ORDER != 0xffff_ffff_0000_0001 {
        return Err(ConstraintExportError::UnsupportedField);
    }
    if D != 2 {
        return Err(ConstraintExportError::UnsupportedExtensionDegree(D));
    }
    if common_data.num_lookup_selectors != 0
        || common_data.num_lookup_polys != 0
        || !common_data.luts.is_empty()
    {
        return Err(ConstraintExportError::UnsupportedLookups);
    }
    let selectors = &common_data.selectors_info;
    if selectors.selector_indices.len() != common_data.gates.len()
        || selectors.num_selectors() > common_data.num_constants
        || selectors
            .selector_indices
            .iter()
            .enumerate()
            .any(|(gate_index, &index)| {
                index >= selectors.groups.len() || !selectors.groups[index].contains(&gate_index)
            })
    {
        return Err(ConstraintExportError::InvalidSelectorMetadata);
    }
    let mut gates = Vec::with_capacity(common_data.gates.len());
    for (gate_index, gate) in common_data.gates.iter().enumerate() {
        let gate_id = gate.0.id();
        let unsupported = |reason| ConstraintExportError::UnsupportedGate {
            gate_id: gate_id.clone(),
            reason,
        };
        if gate.0.num_constraints() > common_data.num_gate_constraints {
            return Err(unsupported("constraint count exceeds common metadata"));
        }
        if gate.0.num_wires() > common_data.config.num_wires
            || gate.0.num_constants() > common_data.num_constants - selectors.num_selectors()
        {
            return Err(unsupported("gate inputs exceed common metadata"));
        }
        let mut config = CircuitConfig::standard_recursion_config();
        config.use_base_arithmetic_gate = false;
        let mut builder = CircuitBuilder::<F, D>::new(config);
        builder.constraint_expression = Some(Expression::default());
        let mut input = |kind| {
            let target = builder.add_virtual_target();
            builder
                .constraint_expression
                .as_mut()
                .unwrap()
                .input(target, kind);
            target
        };
        let constants: Vec<_> = (0..common_data.num_constants)
            .map(|i| input(ConstraintInput::Constant(i)))
            .collect();
        let wires: Vec<_> = (0..common_data.config.num_wires)
            .map(|i| input(ConstraintInput::Wire(i)))
            .collect();
        let hash = HashOutTarget {
            elements: core::array::from_fn(|i| input(ConstraintInput::PublicInputHash(i))),
        };
        let constants: Vec<_> = constants
            .iter()
            .map(|&t| builder.convert_to_ext(t))
            .collect();
        let wires: Vec<_> = wires.iter().map(|&t| builder.convert_to_ext(t)).collect();
        let vars = EvaluationTargets {
            local_constants: &constants,
            local_wires: &wires,
            public_inputs_hash: &hash,
        };
        let mut constraints = vec![builder.zero_extension(); common_data.num_gate_constraints];
        let selector = selectors.selector_indices[gate_index];
        gate.0.eval_filtered_circuit(
            &mut builder,
            vars,
            gate_index,
            selector,
            selectors.groups[selector].clone(),
            selectors.num_selectors(),
            0,
            &mut constraints,
        );
        if !builder.gate_instances.is_empty() {
            return Err(unsupported("untraced gate instance"));
        }
        let expression = builder.constraint_expression.take().unwrap();
        let outputs = expression.outputs(&constraints).map_err(unsupported)?;
        let (nodes, outputs) = expression.compact(outputs);
        gates.push(GateConstraintExpression {
            gate_id,
            nodes,
            outputs,
        });
    }
    Ok(GateConstraintProgram {
        num_constants: common_data.num_constants,
        num_wires: common_data.config.num_wires,
        num_constraints: common_data.num_gate_constraints,
        gates,
    })
}

/// Private arithmetic capture used only by the exporter, never by normal builds.
#[derive(Debug)]
pub(crate) struct Expression<F: Field> {
    nodes: Vec<ArithmeticNode<F>>,
    targets: HashMap<Target, usize>,
    error: Option<&'static str>,
}

impl<F: Field> Default for Expression<F> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            targets: HashMap::new(),
            error: None,
        }
    }
}

impl<F: Field> Expression<F> {
    fn push(&mut self, node: ArithmeticNode<F>) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    pub(crate) fn reject(&mut self, reason: &'static str) {
        self.error.get_or_insert(reason);
    }

    pub(crate) fn constant(&mut self, target: Target, value: F) {
        let node = self.push(ArithmeticNode::Constant(value));
        self.targets.insert(target, node);
    }

    fn input(&mut self, target: Target, kind: ConstraintInput) {
        let node = self.push(ArithmeticNode::Input(kind));
        self.targets.insert(target, node);
    }

    fn value(&self, node: usize) -> Option<F> {
        match self.nodes[node] {
            ArithmeticNode::Constant(value) => Some(value),
            _ => None,
        }
    }

    fn target(&mut self, target: Target) -> usize {
        if let Some(&node) = self.targets.get(&target) {
            return node;
        }
        self.reject("untraced target");
        // Keep tracing to reach the error boundary; this graph is never returned.
        self.push(ArithmeticNode::Constant(F::ZERO))
    }

    fn binary(&mut self, a: usize, b: usize, multiply: bool) -> usize {
        let (x, y) = (self.value(a), self.value(b));
        if let (Some(x), Some(y)) = (x, y) {
            return self.push(ArithmeticNode::Constant(if multiply {
                x * y
            } else {
                x + y
            }));
        }
        if multiply && (x == Some(F::ZERO) || y == Some(F::ZERO)) {
            return self.push(ArithmeticNode::Constant(F::ZERO));
        }
        let identity = if multiply { F::ONE } else { F::ZERO };
        if x == Some(identity) {
            return b;
        }
        if y == Some(identity) {
            return a;
        }
        self.push(if multiply {
            ArithmeticNode::Mul(a, b)
        } else {
            ArithmeticNode::Add(a, b)
        })
    }

    fn base_component(&mut self, targets: &[Target]) -> usize {
        for &target in &targets[1..] {
            let node = self.target(target);
            if self.value(node) != Some(F::ZERO) {
                self.reject("non-base expression");
            }
        }
        self.target(targets[0])
    }

    pub(crate) fn arithmetic(
        &mut self,
        out: Target,
        c0: F,
        c1: F,
        a: &[Target],
        b: &[Target],
        c: &[Target],
    ) {
        let a = self.base_component(a);
        let b = self.base_component(b);
        let c = self.base_component(c);
        let product = self.binary(a, b, true);
        let c0 = self.push(ArithmeticNode::Constant(c0));
        let product = self.binary(product, c0, true);
        let c1 = self.push(ArithmeticNode::Constant(c1));
        let addend = self.binary(c, c1, true);
        let result = self.binary(product, addend, false);
        self.targets.insert(out, result);
    }

    fn outputs<const D: usize>(
        &self,
        targets: &[crate::iop::ext_target::ExtensionTarget<D>],
    ) -> Result<Vec<usize>, &'static str> {
        if let Some(error) = self.error {
            return Err(error);
        }
        targets
            .iter()
            .map(|target| {
                for t in &target.0[1..] {
                    let node = *self.targets.get(t).ok_or("untraced output target")?;
                    if self.value(node) != Some(F::ZERO) {
                        return Err("non-base output expression");
                    }
                }
                self.targets
                    .get(&target.0[0])
                    .copied()
                    .ok_or("untraced output target")
            })
            .collect()
    }

    fn compact(self, outputs: Vec<usize>) -> (Vec<ArithmeticNode<F>>, Vec<usize>) {
        let mut used = vec![false; self.nodes.len()];
        for &output in &outputs {
            used[output] = true;
        }
        for i in (0..self.nodes.len()).rev() {
            if used[i] {
                if let ArithmeticNode::Add(a, b) | ArithmeticNode::Mul(a, b) = self.nodes[i] {
                    used[a] = true;
                    used[b] = true;
                }
            }
        }
        let mut remap = vec![0; self.nodes.len()];
        let mut nodes = Vec::new();
        for (i, node) in self.nodes.into_iter().enumerate() {
            if used[i] {
                remap[i] = nodes.len();
                nodes.push(match node {
                    ArithmeticNode::Add(a, b) => ArithmeticNode::Add(remap[a], remap[b]),
                    ArithmeticNode::Mul(a, b) => ArithmeticNode::Mul(remap[a], remap[b]),
                    node => node,
                });
            }
        }
        (nodes, outputs.into_iter().map(|i| remap[i]).collect())
    }
}
