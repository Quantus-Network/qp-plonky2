#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec, vec::Vec};

use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crate::iop::scheduling_state::{ScheduleChainId, SchedulingState};

/// Execution order and dependencies for a runnable subset of circuit generators.
/// Graph indices are local; `generator_indices` maps them to circuit generator indices.
#[derive(Debug)]
pub struct GenerationSchedule {
    generator_indices: Vec<usize>,
    dependency_counts: Vec<usize>,
    successor_offsets: Vec<usize>,
    successors: Vec<usize>,
    initial_ready: Vec<usize>,
    chain_id: Arc<ScheduleChainId>,
    index: usize,
    newly_promised: Vec<usize>,
}

impl GenerationSchedule {
    fn new(
        state: &mut SchedulingState,
        generator_indices: Vec<usize>,
        dependency_counts: Vec<usize>,
        successor_offsets: Vec<usize>,
        successors: Vec<usize>,
        initial_ready: Vec<usize>,
    ) -> Self {
        let (chain_id, index, newly_promised) = state.finish_schedule();
        Self {
            generator_indices,
            dependency_counts,
            successor_offsets,
            successors,
            initial_ready,
            chain_id,
            index,
            newly_promised,
        }
    }

    /// Position in the planning chain, starting at zero.
    pub fn index(&self) -> usize {
        self.index
    }

    pub(super) fn chain_id(&self) -> &Arc<ScheduleChainId> {
        &self.chain_id
    }

    pub(super) fn newly_promised(&self) -> &[usize] {
        &self.newly_promised
    }

    pub fn generator_indices(&self) -> &[usize] {
        &self.generator_indices
    }

    /// Distinct generator-predecessor counts, indexed locally.
    pub fn dependency_counts(&self) -> &[usize] {
        &self.dependency_counts
    }

    /// Successors of a schedule-local generator, in circuit-builder order.
    pub fn successors(&self, local_index: usize) -> &[usize] {
        &self.successors
            [self.successor_offsets[local_index]..self.successor_offsets[local_index + 1]]
    }

    /// Schedule-local roots, in circuit-builder order.
    pub fn initial_ready(&self) -> &[usize] {
        &self.initial_ready
    }

    #[cfg(feature = "std")]
    pub(super) fn release_successors(
        &self,
        local_index: usize,
        pending_dependencies: &[AtomicUsize],
        mut enqueue: impl FnMut(usize),
    ) -> bool {
        let mut pushed = false;
        for &successor_local_index in self.successors(local_index) {
            let previous =
                pending_dependencies[successor_local_index].fetch_sub(1, Ordering::AcqRel);
            assert!(
                previous > 0,
                "generator {} dependency released twice",
                self.generator_indices[successor_local_index]
            );
            if previous == 1 {
                enqueue(successor_local_index);
                pushed = true;
            }
        }
        pushed
    }
}

/// Extends planned availability with a depth-first runnable closure.
/// Both roots and newly runnable consumers are visited in builder-index order.
pub(super) fn make_schedule(
    reads: &[Vec<usize>],
    writes: &[Vec<usize>],
    consumers: &HashMap<usize, Vec<usize>>,
    state: &mut SchedulingState,
) -> GenerationSchedule {
    const NONE: usize = usize::MAX;
    let mut unavailable_read_counts = vec![0usize; reads.len()];
    let mut ready = Vec::new();
    for (generator_index, read_rep_indices) in reads.iter().enumerate() {
        if state.generators_scheduled()[generator_index] != 0 {
            continue;
        }
        unavailable_read_counts[generator_index] = read_rep_indices
            .iter()
            .filter(|&&rep_index| state.assumed_assigned()[rep_index] == 0)
            .count();
        if unavailable_read_counts[generator_index] == 0 {
            ready.push(generator_index);
        }
    }
    if ready.is_empty() {
        return GenerationSchedule::new(
            state,
            Vec::new(),
            Vec::new(),
            vec![0],
            Vec::new(),
            Vec::new(),
        );
    }
    ready.reverse();
    let mut producer_local_indices = vec![NONE; state.assumed_assigned().len()];
    let mut local_indices = vec![NONE; reads.len()];
    let mut generator_indices = Vec::new();
    let mut edges = Vec::new();
    let mut unlocked = Vec::new();
    while let Some(generator_index) = ready.pop() {
        let local_index = generator_indices.len();
        local_indices[generator_index] = local_index;
        generator_indices.push(generator_index);
        state.mark_scheduled(generator_index);
        unlocked.clear();
        for &rep_index in &writes[generator_index] {
            let producer_local_index = producer_local_indices[rep_index];
            if producer_local_index != NONE {
                // Other writers compare only after the first assignment.
                edges.push((producer_local_index, generator_index));
            }
            if state.assumed_assigned()[rep_index] != 0 {
                continue;
            }
            state.mark_produced(rep_index);
            producer_local_indices[rep_index] = local_index;
            if let Some(readers) = consumers.get(&rep_index) {
                for &consumer_generator_index in readers {
                    if state.generators_scheduled()[consumer_generator_index] != 0 {
                        continue;
                    }
                    unavailable_read_counts[consumer_generator_index] -= 1;
                    edges.push((local_index, consumer_generator_index));
                    if unavailable_read_counts[consumer_generator_index] == 0 {
                        unlocked.push(consumer_generator_index);
                    }
                }
            }
        }
        unlocked.sort_unstable();
        ready.extend(unlocked.iter().rev().copied());
    }

    // Readers still blocked on another class are not members of this schedule.
    edges.retain(|&(_, consumer_generator_index)| local_indices[consumer_generator_index] != NONE);
    edges.sort_unstable();
    edges.dedup();
    let mut dependency_counts = vec![0; generator_indices.len()];
    let mut successor_offsets = vec![0; generator_indices.len() + 1];
    let mut successors = Vec::with_capacity(edges.len());
    for (producer_local_index, consumer_generator_index) in edges {
        let successor_local_index = local_indices[consumer_generator_index];
        debug_assert!(producer_local_index < successor_local_index);
        dependency_counts[successor_local_index] += 1;
        successor_offsets[producer_local_index + 1] += 1;
        successors.push(successor_local_index);
    }
    for local_index in 0..generator_indices.len() {
        successor_offsets[local_index + 1] += successor_offsets[local_index];
    }
    let mut initial_ready: Vec<_> = (0..generator_indices.len())
        .filter(|&local_index| dependency_counts[local_index] == 0)
        .collect();
    initial_ready.sort_unstable_by_key(|&local_index| generator_indices[local_index]);
    GenerationSchedule::new(
        state,
        generator_indices,
        dependency_counts,
        successor_offsets,
        successors,
        initial_ready,
    )
}
