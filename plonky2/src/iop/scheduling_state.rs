#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use anyhow::{anyhow, Result};

/// Identifies schedules planned together by one scheduler.
#[derive(Debug)]
pub(super) struct ScheduleChainId {
    scheduler: Arc<()>,
}

impl ScheduleChainId {
    pub(super) fn matches_scheduler(&self, scheduler: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.scheduler, scheduler)
    }
}

/// Assumed availability used to construct a sequence of schedules.
/// These flags describe planned work, independently of witness execution.
#[derive(Debug)]
pub struct SchedulingState {
    classes_set: Vec<u8>,
    generators_scheduled: Vec<u8>,
    chain_id: Arc<ScheduleChainId>,
    next_schedule: usize,
    newly_supplied: Vec<usize>,
}

impl SchedulingState {
    pub(super) fn new(scheduler: Arc<()>, num_targets: usize, num_generators: usize) -> Self {
        Self {
            classes_set: vec![0; num_targets],
            generators_scheduled: vec![0; num_generators],
            chain_id: Arc::new(ScheduleChainId { scheduler }),
            next_schedule: 0,
            newly_supplied: Vec::new(),
        }
    }

    pub(super) fn with_supplied(
        scheduler: Arc<()>,
        classes_set: &[u8],
        num_generators: usize,
    ) -> Self {
        Self {
            classes_set: classes_set.to_vec(),
            generators_scheduled: vec![0; num_generators],
            chain_id: Arc::new(ScheduleChainId { scheduler }),
            next_schedule: 0,
            newly_supplied: classes_set
                .iter()
                .enumerate()
                .filter_map(|(rep_index, &flag)| (flag != 0).then_some(rep_index))
                .collect(),
        }
    }

    /// Promises that this representative will be assigned before the next schedule.
    /// Repeated promises, including values produced by preceding schedules, do nothing.
    pub fn assume_set(&mut self, rep_index: usize) -> Result<()> {
        let flag = self.classes_set.get_mut(rep_index).ok_or_else(|| {
            anyhow!("representative index {rep_index} is outside the scheduling state")
        })?;
        if *flag == 0 {
            *flag = 1;
            self.newly_supplied.push(rep_index);
        }
        Ok(())
    }

    pub(super) fn chain_id(&self) -> &Arc<ScheduleChainId> {
        &self.chain_id
    }

    pub(super) fn finish_schedule(&mut self) -> (Arc<ScheduleChainId>, usize, Vec<usize>) {
        let index = self.next_schedule;
        self.next_schedule = index.checked_add(1).expect("schedule index overflow");
        (
            Arc::clone(&self.chain_id),
            index,
            core::mem::take(&mut self.newly_supplied),
        )
    }

    pub(super) fn mark_produced(&mut self, rep_index: usize) {
        self.classes_set[rep_index] = 1;
    }

    pub(super) fn mark_scheduled(&mut self, generator_index: usize) {
        self.generators_scheduled[generator_index] = 1;
    }

    /// Flags indexed by representative index; nonzero means assumed assigned.
    pub fn classes_set(&self) -> &[u8] {
        &self.classes_set
    }

    /// Flags indexed by circuit generator index; nonzero means assumed scheduled.
    pub fn generators_scheduled(&self) -> &[u8] {
        &self.generators_scheduled
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

    pub fn num_classes_set(&self) -> usize {
        self.classes_set.iter().filter(|&&flag| flag != 0).count()
    }

    /// Clears planning flags and starts a new chain, retaining buffer allocations.
    pub fn reset(&mut self) {
        self.classes_set.fill(0);
        self.generators_scheduled.fill(0);
        self.chain_id = Arc::new(ScheduleChainId {
            scheduler: Arc::clone(&self.chain_id.scheduler),
        });
        self.next_schedule = 0;
        self.newly_supplied.clear();
    }
}
