#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::cell::UnsafeCell;
use core::fmt;

use anyhow::{ensure, Result};

use crate::field::types::Field;

/// Field values and assignment flags indexed by representative index.
pub(super) struct DenseStorage<F: Field> {
    values: Vec<UnsafeCell<F>>,
    assigned_flags: Vec<UnsafeCell<u8>>,
}

impl<F: Field> DenseStorage<F> {
    pub(super) fn new(num_targets: usize) -> Self {
        Self {
            values: (0..num_targets).map(|_| UnsafeCell::new(F::ZERO)).collect(),
            assigned_flags: (0..num_targets).map(|_| UnsafeCell::new(0)).collect(),
        }
    }

    /// Sets a representative value or compares it with its existing assignment.
    /// Returns an error for conflicting values.
    pub(super) fn set(&mut self, rep_index: usize, value: F) -> Result<()> {
        let assigned_flag = self.assigned_flags[rep_index].get_mut();
        let stored = self.values[rep_index].get_mut();
        if *assigned_flag != 0 {
            ensure!(
                *stored == value,
                "representative index {rep_index} was assigned different values: {stored} != {value}"
            );
        } else {
            *stored = value;
            *assigned_flag = 1;
        }
        Ok(())
    }

    /// Clears assignment flags while retaining stored values and allocations.
    pub(super) fn clear(&mut self) {
        for assigned_flag in &mut self.assigned_flags {
            *assigned_flag.get_mut() = 0;
        }
    }

    /// Clears the assignment flag for one representative, retaining its value.
    pub(super) fn clear_assignment(&mut self, rep_index: usize) {
        *self.assigned_flags[rep_index].get_mut() = 0;
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

    pub(super) fn try_get(&self, rep_index: usize) -> Option<F> {
        self.is_assigned(rep_index).then(|| {
            // SAFETY: Exclusive mutation requires &mut self. During parallel
            // execution, the executor publishes assignments before these reads
            // and keeps assigned values immutable.
            unsafe { *self.values[rep_index].get() }
        })
    }

    pub(super) fn is_assigned(&self, rep_index: usize) -> bool {
        // SAFETY: Exclusive mutation requires &mut self. During parallel
        // execution, the executor orders this read after initial assignment.
        unsafe { *self.assigned_flags[rep_index].get() != 0 }
    }
}

#[cfg(all(feature = "std", feature = "parallel"))]
impl<F: Field> DenseStorage<F> {
    /// Sets or compares a representative value through shared access.
    ///
    /// # Safety
    /// First assignment requires exclusive access to the representative.
    /// Subsequent reads and comparisons must acquire publication of both the
    /// value and its assignment flag. Assigned values and flags must remain
    /// immutable while workers run.
    pub(super) unsafe fn set_shared(&self, rep_index: usize, value: F) -> Result<()> {
        if let Some(old) = self.try_get(rep_index) {
            ensure!(
                old == value,
                "representative index {rep_index} was assigned different values: {old} != {value}"
            );
        } else {
            // SAFETY: The caller grants exclusive representative access until publication.
            unsafe {
                *self.values[rep_index].get() = value;
                *self.assigned_flags[rep_index].get() = 1;
            }
        }
        Ok(())
    }
}

// SAFETY: Safe setters require &mut self. Shared assignment requires exclusive
// access to each unassigned representative until both stores are published.
// Assigned values and flags are immutable while workers run. The executor orders
// readers and other writers after the selected writer's completion.
unsafe impl<F: Field> Sync for DenseStorage<F> {}

impl<F: Field> fmt::Debug for DenseStorage<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Read immutable metadata while generators may be assigning values.
        f.debug_struct("DenseStorage")
            .field("num_targets", &self.values.len())
            .finish()
    }
}
