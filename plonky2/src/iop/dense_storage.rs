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
    assigned_flags: Vec<UnsafeCell<u8>>,
}

// SAFETY: Safe mutation requires exclusive access. Shared mutation is unsafe and
// requires exclusive access to each unset slot until its value and byte have
// been published. Assigned slots are immutable while workers run. The executor
// orders readers and other writers after the selected writer's completion.
unsafe impl<F: Field> Sync for DenseStorage<F> {}

impl<F: Field> fmt::Debug for DenseStorage<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Do not inspect cells: Debug may be called by a running generator.
        f.debug_struct("DenseStorage")
            .field("num_targets", &self.values.len())
            .finish()
    }
}

impl<F: Field> DenseStorage<F> {
    pub(super) fn new(num_targets: usize) -> Self {
        Self {
            values: (0..num_targets).map(|_| UnsafeCell::new(F::ZERO)).collect(),
            assigned_flags: (0..num_targets).map(|_| UnsafeCell::new(0)).collect(),
        }
    }

    pub(super) fn values_and_assignment_flags(&mut self) -> (&[F], &[u8]) {
        // SAFETY: UnsafeCell<T> has the layout of T. Both vectors contain
        // initialized elements, and the exclusive borrow prevents mutation for
        // the lifetime of the returned slices.
        unsafe {
            (
                core::slice::from_raw_parts(self.values.as_ptr().cast::<F>(), self.values.len()),
                core::slice::from_raw_parts(
                    self.assigned_flags.as_ptr().cast::<u8>(),
                    self.assigned_flags.len(),
                ),
            )
        }
    }

    /// Assigns an unset slot or checks that an existing assignment agrees.
    /// A conflicting assignment returns an error without changing storage.
    pub(super) fn set(&mut self, rep_index: usize, value: F) -> Result<()> {
        let assigned_flag = self.assigned_flags[rep_index].get_mut();
        let stored = self.values[rep_index].get_mut();
        if *assigned_flag != 0 {
            ensure!(
                *stored == value,
                "Witness slot {rep_index} was set twice with different values: {stored} != {value}"
            );
        } else {
            *stored = value;
            *assigned_flag = 1;
        }
        Ok(())
    }

    pub(super) fn try_get(&self, rep_index: usize) -> Option<F> {
        self.is_assigned(rep_index).then(|| {
            // SAFETY: Safe APIs cannot mutate through shared access. The parallel
            // executor orders reads after publication; the assignment byte
            // was checked above.
            unsafe { *self.values[rep_index].get() }
        })
    }

    /// Clears assignment flags while retaining stored values and allocations.
    pub(super) fn clear(&mut self) {
        for assigned_flag in &mut self.assigned_flags {
            *assigned_flag.get_mut() = 0;
        }
    }

    /// Marks a slot unassigned without clearing its stored value.
    pub(super) fn clear_assignment(&mut self, rep_index: usize) {
        *self.assigned_flags[rep_index].get_mut() = 0;
    }

    pub(super) fn is_assigned(&self, rep_index: usize) -> bool {
        // SAFETY: Safe APIs cannot mutate through shared access. The parallel
        // executor orders presence reads after the slot writer completes.
        unsafe { *self.assigned_flags[rep_index].get() != 0 }
    }

    /// Shared equivalent of `set`, for the scheduled executor only.
    ///
    /// # Safety
    /// No access to an unset slot may overlap this call. After assignment, all
    /// accesses must be read-only and acquire publication of both stores. The
    /// caller must keep storage alive and must not obtain exclusive access while
    /// workers run.
    #[cfg(feature = "std")]
    pub(super) unsafe fn set_shared(&self, rep_index: usize, value: F) -> Result<()> {
        if let Some(old) = self.try_get(rep_index) {
            ensure!(
                old == value,
                "Witness slot {rep_index} was set twice with different values: {old} != {value}"
            );
        } else {
            // SAFETY: The caller grants exclusive slot access until publication.
            unsafe {
                *self.values[rep_index].get() = value;
                *self.assigned_flags[rep_index].get() = 1;
            }
        }
        Ok(())
    }
}
