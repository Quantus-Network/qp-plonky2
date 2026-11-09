//! Immutable generator schedules and dependency-graph construction.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{ensure, Result};
use hashbrown::HashMap;

/// Tracks assignments and generator completion assumed during planning.
#[derive(Debug)]
pub(super) struct WitnessGenerationState {
    assumed_assigned_flags: Vec<u8>,
    scheduled_generator_flags: Vec<u8>,
}

impl WitnessGenerationState {
    pub(super) fn new(num_targets: usize, num_generators: usize) -> Self {
        Self {
            assumed_assigned_flags: vec![0; num_targets],
            scheduled_generator_flags: vec![0; num_generators],
        }
    }

    pub(super) fn mark_assigned(&mut self, rep_index: usize) {
        self.assumed_assigned_flags[rep_index] = 1;
    }

    pub(super) fn mark_scheduled(&mut self, generator_index: usize) {
        self.scheduled_generator_flags[generator_index] = 1;
    }

    pub(super) fn assumed_assigned_flags(&self) -> &[u8] {
        &self.assumed_assigned_flags
    }

    pub(super) fn scheduled_generator_flags(&self) -> &[u8] {
        &self.scheduled_generator_flags
    }

    pub(super) fn is_done(&self) -> bool {
        self.scheduled_generator_flags.iter().all(|&flag| flag != 0)
    }
}

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

    /// Schedule-local indices of generators with zero dependencies, in circuit-builder order.
    pub fn initial_ready(&self) -> &[usize] {
        &self.initial_ready
    }
}

/// Validated representative reads, writes, and consumers for circuit generators.
#[derive(Debug)]
pub(super) struct GeneratorDependencies {
    reads: Vec<Vec<usize>>,
    writes: Vec<Vec<usize>>,
    consumers: HashMap<usize, Vec<usize>>,
    num_targets: usize,
}

impl GeneratorDependencies {
    pub(super) fn new(
        mut reads: Vec<Vec<usize>>,
        mut writes: Vec<Vec<usize>>,
        num_targets: usize,
    ) -> Result<Self> {
        ensure!(
            reads.len() == writes.len(),
            "generator read and write counts do not match"
        );
        for rep_indices in reads.iter_mut().chain(&mut writes) {
            for &rep_index in rep_indices.iter() {
                ensure!(
                    rep_index < num_targets,
                    "invalid representative index {rep_index}"
                );
            }
            rep_indices.sort_unstable();
            rep_indices.dedup();
        }
        let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
        for (generator_index, rep_indices) in reads.iter().enumerate() {
            for &rep_index in rep_indices {
                consumers
                    .entry(rep_index)
                    .or_default()
                    .push(generator_index);
            }
        }
        Ok(Self {
            reads,
            writes,
            consumers,
            num_targets,
        })
    }

    pub(super) fn reads(&self) -> &[Vec<usize>] {
        &self.reads
    }

    fn writes(&self) -> &[Vec<usize>] {
        &self.writes
    }

    fn consumers(&self, rep_index: usize) -> &[usize] {
        self.consumers.get(&rep_index).map_or(&[], Vec::as_slice)
    }

    fn num_targets(&self) -> usize {
        self.num_targets
    }
}

/// Builds a depth-first schedule from assumed generation state.
/// Returns the schedule and representative indices first assigned by its generators.
/// Initial and newly enabled generators are visited in circuit-builder order.
pub(super) fn make_schedule(
    dependencies: &GeneratorDependencies,
    state: &WitnessGenerationState,
) -> (WitnessGenerationSchedule, Vec<usize>) {
    const NONE: usize = usize::MAX;
    let reads = dependencies.reads();
    let writes = dependencies.writes();
    assert_eq!(
        state.scheduled_generator_flags().len(),
        reads.len(),
        "scheduling state generator count does not match dependencies"
    );
    assert_eq!(
        state.assumed_assigned_flags().len(),
        dependencies.num_targets(),
        "scheduling state target count does not match dependencies"
    );
    let mut unavailable_read_counts = vec![0usize; reads.len()];
    let mut ready = Vec::new();
    for (generator_index, read_rep_indices) in reads.iter().enumerate() {
        if state.scheduled_generator_flags()[generator_index] != 0 {
            continue;
        }
        unavailable_read_counts[generator_index] = read_rep_indices
            .iter()
            .filter(|&&rep_index| state.assumed_assigned_flags()[rep_index] == 0)
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
    let mut producer_local_indices = vec![NONE; state.assumed_assigned_flags().len()];
    let mut local_indices = vec![NONE; reads.len()];
    let mut generator_indices = Vec::new();
    let mut edges = Vec::new();
    let mut unlocked = Vec::new();
    let mut newly_assigned = Vec::new();
    while let Some(generator_index) = ready.pop() {
        let local_index = generator_indices.len();
        local_indices[generator_index] = local_index;
        generator_indices.push(generator_index);
        unlocked.clear();
        for &rep_index in &writes[generator_index] {
            let producer_local_index = producer_local_indices[rep_index];
            if producer_local_index != NONE {
                // Order additional writers after the generator assigning the value.
                edges.push((producer_local_index, generator_index));
            }
            if state.assumed_assigned_flags()[rep_index] != 0 || producer_local_index != NONE {
                continue;
            }
            newly_assigned.push(rep_index);
            producer_local_indices[rep_index] = local_index;
            for &consumer_generator_index in dependencies.consumers(rep_index) {
                if state.scheduled_generator_flags()[consumer_generator_index] != 0
                    || local_indices[consumer_generator_index] != NONE
                {
                    continue;
                }
                unavailable_read_counts[consumer_generator_index] -= 1;
                edges.push((local_index, consumer_generator_index));
                if unavailable_read_counts[consumer_generator_index] == 0 {
                    unlocked.push(consumer_generator_index);
                }
            }
        }
        unlocked.sort_unstable();
        ready.extend(unlocked.iter().rev().copied());
    }

    // Retain dependencies between generators included in this schedule.
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
