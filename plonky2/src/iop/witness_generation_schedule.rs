//! Immutable generator schedules and dependency-graph construction.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hashbrown::HashMap;

use crate::iop::witness_generation_plan::WitnessGenerationState;

/// Execution order and dependencies for a runnable subset of circuit generators.
/// Graph indices are local; `generator_indices` maps them to circuit generator indices.
#[derive(Debug)]
pub struct WitnessGenerationSchedule {
    generator_indices: Vec<usize>,
    dependency_counts: Vec<usize>,
    successors: Vec<usize>,
    successor_offsets: Vec<usize>,
    initial_ready: Vec<usize>,
}

impl WitnessGenerationSchedule {
    /// Circuit generator indices in scheduled order; all graph indices are local.
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
}

/// Builds a depth-first schedule and updates assumed generation state.
/// Returns the schedule and the representatives it first makes available.
/// Both roots and newly runnable consumers are visited in builder-index order.
pub(super) fn make_schedule(
    reads: &[Vec<usize>],
    writes: &[Vec<usize>],
    consumers: &HashMap<usize, Vec<usize>>,
    state: &mut WitnessGenerationState,
) -> (WitnessGenerationSchedule, Vec<usize>) {
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
        return (
            WitnessGenerationSchedule {
                generator_indices: Vec::new(),
                dependency_counts: Vec::new(),
                successors: Vec::new(),
                successor_offsets: vec![0],
                initial_ready: Vec::new(),
            },
            Vec::new(),
        );
    }
    ready.reverse();
    let mut producer_local_indices = vec![NONE; state.assumed_assigned().len()];
    let mut local_indices = vec![NONE; reads.len()];
    let mut generator_indices = Vec::new();
    let mut edges = Vec::new();
    let mut unlocked = Vec::new();
    let mut newly_assigned = Vec::new();
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
            state.mark_assigned(rep_index);
            newly_assigned.push(rep_index);
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
    let mut successors = Vec::with_capacity(edges.len());
    let mut successor_offsets = vec![0; generator_indices.len() + 1];
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
    (
        WitnessGenerationSchedule {
            generator_indices,
            dependency_counts,
            successors,
            successor_offsets,
            initial_ready,
        },
        newly_assigned,
    )
}
