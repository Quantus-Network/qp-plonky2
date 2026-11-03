use anyhow::Result;

use crate::field::types::Field;
use crate::iop::dense_storage::DenseStorage;
use crate::iop::target::Target;
use crate::iop::witness::{Witness, WitnessWrite};

/// A dense witness whose targets resolve through a fixed representative map.
#[derive(Debug)]
pub(crate) struct DenseWitness<'a, F: Field> {
    storage: DenseStorage<F>,
    representative_map: &'a [usize],
    num_wires: usize,
    degree: usize,
}

impl<'a, F: Field> DenseWitness<'a, F> {
    pub(crate) fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
        Self {
            storage: DenseStorage::new(representative_map.len()),
            representative_map,
            num_wires,
            degree,
        }
    }

    fn slot(&self, target: Target) -> usize {
        self.representative_map[target.index(self.num_wires, self.degree)]
    }
}

impl<F: Field> WitnessWrite<F> for DenseWitness<'_, F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        let slot = self.slot(target);
        self.storage.assign(slot, value)
    }
}

impl<F: Field> Witness<F> for DenseWitness<'_, F> {
    /// The caller must guarantee that the target is assigned.
    fn get_target(&self, target: Target) -> F {
        self.storage.get(self.slot(target))
    }

    fn try_get_target(&self, target: Target) -> Option<F> {
        self.storage.try_get(self.slot(target))
    }

    fn contains(&self, target: Target) -> bool {
        self.storage.is_set(self.slot(target))
    }
}
