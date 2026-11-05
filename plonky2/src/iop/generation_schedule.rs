#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{ensure, Result};
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::iop::scheduling_state::SchedulingState;

/// Execution order and dependencies for a runnable subset of circuit generators.
/// Graph indices are local; `order` maps them to circuit generator indices.
#[derive(Debug)]
pub struct GenerationSchedule {
    order: Vec<usize>,
    dependency_counts: Vec<usize>,
    successor_offsets: Vec<usize>,
    successors: Vec<usize>,
    initial_ready: Vec<usize>,
}

impl GenerationSchedule {
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Distinct generator-predecessor counts, indexed locally.
    pub fn dependency_counts(&self) -> &[usize] {
        &self.dependency_counts
    }

    /// Successors of a schedule-local generator, in circuit-builder order.
    pub fn successors(&self, generator: usize) -> &[usize] {
        &self.successors[self.successor_offsets[generator]..self.successor_offsets[generator + 1]]
    }

    /// Schedule-local roots, in circuit-builder order.
    pub fn initial_ready(&self) -> &[usize] {
        &self.initial_ready
    }

    #[cfg(feature = "std")]
    pub(super) fn release_successors(
        &self,
        local: usize,
        pending_dependencies: &[AtomicUsize],
        mut enqueue: impl FnMut(usize),
    ) -> bool {
        let mut pushed = false;
        for &consumer in self.successors(local) {
            let previous = pending_dependencies[consumer].fetch_sub(1, Ordering::AcqRel);
            assert!(
                previous > 0,
                "generator {} dependency released twice",
                self.order[consumer]
            );
            if previous == 1 {
                enqueue(consumer);
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
) -> Result<GenerationSchedule> {
    const NONE: usize = usize::MAX;
    ensure!(
        state.generators_scheduled().len() == reads.len(),
        "scheduling state generator count does not match circuit"
    );
    let mut remaining = vec![0usize; reads.len()];
    let mut ready = Vec::new();
    for (generator, inputs) in reads.iter().enumerate() {
        if state.generators_scheduled()[generator] != 0 {
            continue;
        }
        remaining[generator] = inputs
            .iter()
            .filter(|&&rep_index| state.classes_set()[rep_index] == 0)
            .count();
        if remaining[generator] == 0 {
            ready.push(generator);
        }
    }
    if ready.is_empty() {
        return Ok(GenerationSchedule {
            order: Vec::new(),
            dependency_counts: Vec::new(),
            successor_offsets: vec![0],
            successors: Vec::new(),
            initial_ready: Vec::new(),
        });
    }
    ready.reverse();
    let mut producers = vec![NONE; state.classes_set().len()];
    let mut position = vec![NONE; reads.len()];
    let mut order = Vec::new();
    let mut edges = Vec::new();
    let mut unlocked = Vec::new();
    while let Some(generator) = ready.pop() {
        let local = order.len();
        position[generator] = local;
        order.push(generator);
        state.generators_scheduled_mut()[generator] = 1;
        unlocked.clear();
        for &rep_index in &writes[generator] {
            let producer = producers[rep_index];
            if producer != NONE {
                // Other writers compare only after the first assignment.
                edges.push((producer, generator));
            }
            if state.classes_set()[rep_index] != 0 {
                continue;
            }
            state.classes_set_mut()[rep_index] = 1;
            producers[rep_index] = local;
            if let Some(readers) = consumers.get(&rep_index) {
                for &consumer in readers {
                    if state.generators_scheduled()[consumer] != 0 {
                        continue;
                    }
                    remaining[consumer] -= 1;
                    edges.push((local, consumer));
                    if remaining[consumer] == 0 {
                        unlocked.push(consumer);
                    }
                }
            }
        }
        unlocked.sort_unstable();
        ready.extend(unlocked.iter().rev().copied());
    }

    // Readers still blocked on another class are not members of this schedule.
    edges.retain(|&(_, consumer)| position[consumer] != NONE);
    edges.sort_unstable();
    edges.dedup();
    let mut dependency_counts = vec![0; order.len()];
    let mut successor_offsets = vec![0; order.len() + 1];
    let mut successors = Vec::with_capacity(edges.len());
    for (producer, consumer) in edges {
        let consumer = position[consumer];
        debug_assert!(producer < consumer);
        dependency_counts[consumer] += 1;
        successor_offsets[producer + 1] += 1;
        successors.push(consumer);
    }
    for local in 0..order.len() {
        successor_offsets[local + 1] += successor_offsets[local];
    }
    let mut initial_ready: Vec<_> = (0..order.len())
        .filter(|&local| dependency_counts[local] == 0)
        .collect();
    initial_ready.sort_unstable_by_key(|&local| order[local]);
    Ok(GenerationSchedule {
        order,
        dependency_counts,
        successor_offsets,
        successors,
        initial_ready,
    })
}
