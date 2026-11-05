#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::cell::UnsafeCell;
use core::fmt;

use anyhow::{ensure, Result};

use crate::field::types::Field;

/// Field values and assignment bytes, indexed by witness slot.
/// Shared mutation is restricted to the scheduled executor's unsafe entry point.
pub(super) struct DenseStorage<F: Field> {
    values: Vec<UnsafeCell<F>>,
    assigned: Vec<UnsafeCell<u8>>,
    parallel: bool,
}

// SAFETY: Safe mutation requires exclusive access. Shared mutation is unsafe and
// requires exclusive access to each unset slot until its value and byte have
// been published. Assigned slots are immutable while workers run. The executor
// orders readers and other writers after the selected writer's completion.
unsafe impl<F: Field> Sync for DenseStorage<F> {}

impl<F: Field> Clone for DenseStorage<F> {
    fn clone(&self) -> Self {
        assert!(
            !self.parallel,
            "cannot clone dense storage during parallel generation"
        );
        Self {
            // SAFETY: Shared writes are only permitted during parallel execution,
            // when cloning is rejected above.
            values: self
                .values
                .iter()
                .map(|v| UnsafeCell::new(unsafe { *v.get() }))
                .collect(),
            assigned: self
                .assigned
                .iter()
                .map(|v| UnsafeCell::new(unsafe { *v.get() }))
                .collect(),
            parallel: false,
        }
    }
}

impl<F: Field> fmt::Debug for DenseStorage<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Do not inspect cells: Debug may be called by a running generator.
        f.debug_struct("DenseStorage")
            .field("num_slots", &self.values.len())
            .field("parallel", &self.parallel)
            .finish()
    }
}

impl<F: Field> DenseStorage<F> {
    pub(super) fn new(num_slots: usize) -> Self {
        Self {
            values: (0..num_slots).map(|_| UnsafeCell::new(F::ZERO)).collect(),
            assigned: (0..num_slots).map(|_| UnsafeCell::new(0)).collect(),
            parallel: false,
        }
    }

    /// Assigns an unset slot or checks that an existing assignment agrees.
    /// A conflicting assignment returns an error without changing storage.
    pub(super) fn assign(&mut self, slot: usize, value: F) -> Result<()> {
        let assigned = self.assigned[slot].get_mut();
        let stored = self.values[slot].get_mut();
        if *assigned != 0 {
            ensure!(
                *stored == value,
                "Witness slot {slot} was set twice with different values: {stored} != {value}"
            );
        } else {
            *stored = value;
            *assigned = 1;
        }
        Ok(())
    }

    /// Reads an assigned slot; concurrent access must obey `assign_shared`'s contract.
    pub(super) fn get(&self, slot: usize) -> F {
        assert!(self.is_set(slot), "read of unassigned witness slot {slot}");
        // SAFETY: Safe APIs cannot mutate through shared access. The parallel
        // executor's unsafe contract orders all reads after slot publication.
        unsafe { *self.values[slot].get() }
    }

    pub(super) fn try_get(&self, slot: usize) -> Option<F> {
        self.is_set(slot).then(|| {
            // SAFETY: As for get(), callers of shared writes must order readers
            // after publication; the assignment byte was checked above.
            unsafe { *self.values[slot].get() }
        })
    }

    /// Marks a slot unassigned without clearing its stored value.
    pub(super) fn clear(&mut self, slot: usize) {
        *self.assigned[slot].get_mut() = 0;
    }

    pub(super) fn is_set(&self, slot: usize) -> bool {
        // SAFETY: As for get(), presence reads must not overlap a slot's writer.
        unsafe { *self.assigned[slot].get() != 0 }
    }

    #[cfg(feature = "std")]
    pub(super) fn set_parallel(&mut self, active: bool) {
        self.parallel = active;
    }

    /// Shared equivalent of `assign`, for the scheduled executor only.
    ///
    /// # Safety
    /// No access to an unset slot may overlap this call. After assignment, all
    /// accesses must be read-only and acquire publication of both stores. The
    /// caller must keep storage alive and must not obtain exclusive access while
    /// workers run. `parallel` must be true for the entire shared-write phase.
    #[cfg(feature = "std")]
    pub(super) unsafe fn assign_shared(&self, slot: usize, value: F) -> Result<()> {
        debug_assert!(self.parallel);
        if self.is_set(slot) {
            let old = self.get(slot);
            ensure!(
                old == value,
                "Witness slot {slot} was set twice with different values: {old} != {value}"
            );
        } else {
            // SAFETY: The caller grants exclusive slot access until publication.
            unsafe {
                *self.values[slot].get() = value;
                *self.assigned[slot].get() = 1;
            }
        }
        Ok(())
    }
}
