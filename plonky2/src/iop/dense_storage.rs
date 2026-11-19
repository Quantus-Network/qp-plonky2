#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU8, Ordering};

use anyhow::{ensure, Result};

use self::slot::ValueSlot;
use crate::field::types::{Field, PrimeField64};

const EMPTY: u8 = 0;
const ASSIGNED: u8 = 1;

/// Storage for one noncanonical `u64` value. Targets without 64-bit atomics
/// cannot run the parallel executor, so a plain cell suffices there.
#[cfg(target_has_atomic = "64")]
mod slot {
    use core::sync::atomic::{AtomicU64, Ordering};

    pub(crate) type ValueSlot = AtomicU64;

    #[inline]
    pub(super) fn load(slot: &ValueSlot) -> u64 {
        slot.load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn store(slot: &ValueSlot, value: u64) {
        slot.store(value, Ordering::Relaxed);
    }
}

#[cfg(not(target_has_atomic = "64"))]
mod slot {
    use core::cell::Cell;

    pub(crate) type ValueSlot = Cell<u64>;

    #[inline]
    pub(super) fn load(slot: &ValueSlot) -> u64 {
        slot.get()
    }

    #[inline]
    pub(super) fn store(slot: &ValueSlot, value: u64) {
        slot.set(value);
    }
}

/// Field values and assignment flags indexed by representative index.
///
/// Values are stored as their noncanonical `u64` representation. A value is
/// only read through shared access after an acquire load of its flag observed
/// `ASSIGNED`, and a flag is only published with a release store after the
/// value was written, so relaxed accesses to the value itself suffice.
///
/// Decoding uses `Field::from_noncanonical_u64`. Encoding needs `PrimeField64`,
/// so it is fixed at construction, the only operation with that bound; the
/// rest is available for any `Field` so `PartitionWitness` keeps its `Field`
/// bounds.
pub(super) struct DenseStorage<F> {
    values: Vec<ValueSlot>,
    assigned_flags: Vec<AtomicU8>,
    encode: fn(F) -> u64,
    _field: PhantomData<F>,
}

impl<F: PrimeField64> DenseStorage<F> {
    pub(super) fn new(num_targets: usize) -> Self {
        Self {
            values: (0..num_targets).map(|_| ValueSlot::new(0)).collect(),
            assigned_flags: (0..num_targets).map(|_| AtomicU8::new(EMPTY)).collect(),
            encode: encode::<F>,
            _field: PhantomData,
        }
    }
}

fn encode<F: PrimeField64>(value: F) -> u64 {
    value.to_noncanonical_u64()
}

impl<F: Field> DenseStorage<F> {
    /// Sets a representative value or compares it with its existing assignment.
    /// Returns an error for conflicting values.
    pub(super) fn set(&mut self, rep_index: usize, value: F) -> Result<()> {
        let encoded = (self.encode)(value);
        self.set_encoded(rep_index, value, encoded)
    }

    /// Like [`Self::set`], with the value's encoding supplied by the caller so
    /// that hot paths with a `PrimeField64` bound avoid the indirect call.
    pub(super) fn set_encoded(&mut self, rep_index: usize, value: F, encoded: u64) -> Result<()> {
        let assigned_flag = self.assigned_flags[rep_index].get_mut();
        let stored = self.values[rep_index].get_mut();
        if *assigned_flag == ASSIGNED {
            let old = F::from_noncanonical_u64(*stored);
            ensure!(
                old == value,
                "representative index {rep_index} was assigned different values: {old} != {value}"
            );
        } else {
            *stored = encoded;
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
    pub(super) fn set_shared_encoded(
        &self,
        rep_index: usize,
        value: F,
        encoded: u64,
    ) -> Result<()> {
        if self.is_assigned(rep_index) {
            let old = F::from_noncanonical_u64(slot::load(&self.values[rep_index]));
            ensure!(
                old == value,
                "representative index {rep_index} was assigned different values: {old} != {value}"
            );
        } else {
            slot::store(&self.values[rep_index], encoded);
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
    #[cfg(target_has_atomic = "64")]
    pub(super) fn values_and_assignment_flags(&self) -> (&[ValueSlot], &[AtomicU8]) {
        (&self.values, &self.assigned_flags)
    }

    pub(super) fn try_get(&self, rep_index: usize) -> Option<F> {
        self.is_assigned(rep_index).then(|| {
            // The acquire load in `is_assigned` orders this after the publishing write.
            F::from_noncanonical_u64(slot::load(&self.values[rep_index]))
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
