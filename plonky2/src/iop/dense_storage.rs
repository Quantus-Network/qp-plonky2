#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use anyhow::{ensure, Result};

use crate::field::types::PrimeField64;

const EMPTY: u8 = 0;
const ASSIGNED: u8 = 1;

/// Field values and assignment flags indexed by representative index.
///
/// Values are stored as their noncanonical `u64` representation. A value is
/// only read through shared access after an acquire load of its flag observed
/// `ASSIGNED`, and a flag is only published with a release store after the
/// value was written, so relaxed accesses to the value itself suffice.
pub(super) struct DenseStorage<F> {
    values: Vec<AtomicU64>,
    assigned_flags: Vec<AtomicU8>,
    _field: PhantomData<F>,
}

impl<F: PrimeField64> DenseStorage<F> {
    pub(super) fn new(num_targets: usize) -> Self {
        Self {
            values: (0..num_targets).map(|_| AtomicU64::new(0)).collect(),
            assigned_flags: (0..num_targets).map(|_| AtomicU8::new(EMPTY)).collect(),
            _field: PhantomData,
        }
    }

    /// Sets a representative value or compares it with its existing assignment.
    /// Returns an error for conflicting values.
    pub(super) fn set(&mut self, rep_index: usize, value: F) -> Result<()> {
        let assigned_flag = self.assigned_flags[rep_index].get_mut();
        let stored = self.values[rep_index].get_mut();
        if *assigned_flag == ASSIGNED {
            let old = F::from_noncanonical_u64(*stored);
            ensure!(
                old == value,
                "representative index {rep_index} was assigned different values: {old} != {value}"
            );
        } else {
            *stored = value.to_noncanonical_u64();
            *assigned_flag = ASSIGNED;
        }
        Ok(())
    }

    /// Sets or compares a representative value through shared access.
    ///
    /// The executor only calls this for a generator's planned outputs, which the
    /// schedule gives a single first writer per stage. If that ordering were ever
    /// violated, concurrent writers could publish different values for one slot,
    /// which a later comparison or the prover's constraint checks would reject;
    /// no memory unsafety is possible either way.
    pub(super) fn set_shared(&self, rep_index: usize, value: F) -> Result<()> {
        if self.is_assigned(rep_index) {
            let old = F::from_noncanonical_u64(self.values[rep_index].load(Ordering::Relaxed));
            ensure!(
                old == value,
                "representative index {rep_index} was assigned different values: {old} != {value}"
            );
        } else {
            self.values[rep_index].store(value.to_noncanonical_u64(), Ordering::Relaxed);
            self.assigned_flags[rep_index].store(ASSIGNED, Ordering::Release);
        }
        Ok(())
    }

    /// Clears assignment flags while retaining stored values and allocations.
    pub(super) fn clear(&mut self) {
        for assigned_flag in &mut self.assigned_flags {
            *assigned_flag.get_mut() = EMPTY;
        }
    }

    /// Clears the assignment flag for one representative, retaining its value.
    pub(super) fn clear_assignment(&mut self, rep_index: usize) {
        *self.assigned_flags[rep_index].get_mut() = EMPTY;
    }

    /// Borrows the stored noncanonical values and assignment flags.
    /// A nonzero flag marks an assigned slot.
    pub(super) fn values_and_assignment_flags(&self) -> (&[AtomicU64], &[AtomicU8]) {
        (&self.values, &self.assigned_flags)
    }

    pub(super) fn try_get(&self, rep_index: usize) -> Option<F> {
        self.is_assigned(rep_index).then(|| {
            // The acquire load in `is_assigned` orders this after the publishing write.
            F::from_noncanonical_u64(self.values[rep_index].load(Ordering::Relaxed))
        })
    }

    pub(super) fn is_assigned(&self, rep_index: usize) -> bool {
        self.assigned_flags[rep_index].load(Ordering::Acquire) == ASSIGNED
    }
}

impl<F> fmt::Debug for DenseStorage<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseStorage")
            .field("num_targets", &self.values.len())
            .finish()
    }
}
