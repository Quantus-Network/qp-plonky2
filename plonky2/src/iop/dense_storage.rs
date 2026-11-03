#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{ensure, Result};

use crate::field::types::Field;

/// Field values and their assignment status, indexed by witness slot.
#[derive(Clone, Debug)]
pub(crate) struct DenseStorage<F: Field> {
    values: Vec<F>,
    assigned: Vec<u8>,
}

impl<F: Field> DenseStorage<F> {
    pub(crate) fn new(num_slots: usize) -> Self {
        Self {
            values: vec![F::ZERO; num_slots],
            assigned: vec![0; num_slots],
        }
    }

    /// Assigns an unset slot or checks that an existing assignment agrees.
    /// A conflicting assignment returns an error without changing storage.
    pub(crate) fn assign(&mut self, slot: usize, value: F) -> Result<()> {
        if self.is_set(slot) {
            let old = self.values[slot];
            ensure!(
                old == value,
                "Witness slot {slot} was set twice with different values: {old} != {value}"
            );
        } else {
            self.values[slot] = value;
            self.assigned[slot] = 1;
        }
        Ok(())
    }

    /// Reads a slot that the caller guarantees is assigned.
    pub(crate) fn get(&self, slot: usize) -> F {
        assert!(self.is_set(slot), "read of unassigned witness slot {slot}");
        self.values[slot]
    }

    pub(crate) fn try_get(&self, slot: usize) -> Option<F> {
        self.is_set(slot).then(|| self.values[slot])
    }

    /// Marks a slot unassigned without clearing its stored value.
    pub(crate) fn clear(&mut self, slot: usize) {
        self.assigned[slot] = 0;
    }

    pub(crate) fn is_set(&self, slot: usize) -> bool {
        self.assigned[slot] != 0
    }
}
