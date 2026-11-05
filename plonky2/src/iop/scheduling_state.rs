#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Assumed availability used to construct a sequence of schedules.
/// These flags describe planned work, independently of witness execution.
#[derive(Debug)]
pub struct SchedulingState {
    classes_set: Vec<u8>,
    generators_scheduled: Vec<u8>,
}

impl SchedulingState {
    pub(super) fn new(num_representatives: usize, num_generators: usize) -> Self {
        Self {
            classes_set: vec![0; num_representatives],
            generators_scheduled: vec![0; num_generators],
        }
    }

    /// Flags indexed by representative index; nonzero means assumed assigned.
    pub fn classes_set(&self) -> &[u8] {
        &self.classes_set
    }

    pub fn classes_set_mut(&mut self) -> &mut [u8] {
        &mut self.classes_set
    }

    /// Flags indexed by circuit generator index; nonzero means assumed scheduled.
    pub fn generators_scheduled(&self) -> &[u8] {
        &self.generators_scheduled
    }

    pub fn generators_scheduled_mut(&mut self) -> &mut [u8] {
        &mut self.generators_scheduled
    }

    pub fn remaining_generators(&self) -> usize {
        self.generators_scheduled
            .iter()
            .filter(|&&flag| flag == 0)
            .count()
    }

    pub fn is_fully_scheduled(&self) -> bool {
        self.generators_scheduled.iter().all(|&flag| flag != 0)
    }

    /// Counts flagged positions; callers mark only representative indices.
    pub fn num_classes_set(&self) -> usize {
        self.classes_set.iter().filter(|&&flag| flag != 0).count()
    }

    /// Clears planned availability while retaining allocations.
    pub fn reset(&mut self) {
        self.classes_set.fill(0);
        self.generators_scheduled.fill(0);
    }
}
