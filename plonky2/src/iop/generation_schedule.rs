#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{anyhow, ensure, Result};
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// An execution order and dependency graph for a subset of circuit generators.
/// Graph indices are local to this schedule; `order` maps them to the circuit.
#[derive(Debug)]
pub struct GenerationSchedule {
    pub(super) order: Vec<usize>,
    pub(super) pending: Vec<usize>,
    successor_offsets: Vec<usize>,
    successors: Vec<usize>,
    pub(super) initial_ready: Vec<usize>,
    pub(super) input_slots: Vec<usize>,
    input_runs: Vec<InputRun>,
    input_successor_offsets: Vec<usize>,
    input_successors: Vec<usize>,
}

/// A contiguous representative range mapped to consecutive input-class indices.
#[derive(Debug)]
struct InputRun {
    start: usize,
    end: usize,
    first_input: usize,
}

impl GenerationSchedule {
    /// Circuit generator indices in serial execution order. Each position is a
    /// schedule-local index used by the dependency counts and successor lists.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Initial generator-predecessor and required-input counts, indexed locally.
    pub fn dependency_counts(&self) -> &[usize] {
        &self.pending
    }

    /// Distinct successors of a schedule-local generator, in circuit-builder order.
    pub fn successors(&self, generator: usize) -> &[usize] {
        &self.successors[self.successor_offsets[generator]..self.successor_offsets[generator + 1]]
    }

    /// Required caller-supplied classes, ordered by representative slot.
    pub fn input_slots(&self) -> &[usize] {
        &self.input_slots
    }

    /// Finds an input-class index from its representative slot, returning None
    /// for slots outside the required input classes, including gaps between runs.
    pub fn input_index(&self, slot: usize) -> Option<usize> {
        let index = self.input_runs.partition_point(|run| run.start <= slot);
        let run = self.input_runs.get(index.checked_sub(1)?)?;
        (slot < run.end).then(|| run.first_input + (slot - run.start))
    }

    /// Consumers of the input class at `input`, using schedule-local indices.
    pub fn input_successors(&self, input: usize) -> &[usize] {
        &self.input_successors
            [self.input_successor_offsets[input]..self.input_successor_offsets[input + 1]]
    }

    #[cfg(feature = "std")]
    pub(super) fn release_successors(
        &self,
        local: usize,
        pending: &[AtomicUsize],
        enqueue: impl FnMut(usize),
    ) -> bool {
        self.release(
            self.successors(local),
            |consumer| {
                pending[consumer]
                    .fetch_sub(1, Ordering::AcqRel)
                    .checked_sub(1)
            },
            enqueue,
        )
    }

    #[cfg(feature = "std")]
    pub(super) fn release_input(
        &self,
        input: usize,
        pending: &mut [AtomicUsize],
        enqueue: impl FnMut(usize),
    ) -> bool {
        self.release(
            self.input_successors(input),
            |consumer| {
                let count = pending[consumer].get_mut();
                let remaining = count.checked_sub(1)?;
                *count = remaining;
                Some(remaining)
            },
            enqueue,
        )
    }

    #[cfg(feature = "std")]
    fn release(
        &self,
        successors: &[usize],
        mut decrement: impl FnMut(usize) -> Option<usize>,
        mut enqueue: impl FnMut(usize),
    ) -> bool {
        let mut pushed = false;
        for &consumer in successors {
            let remaining = decrement(consumer).unwrap_or_else(|| {
                panic!(
                    "generator {} dependency released twice",
                    self.order[consumer]
                )
            });
            if remaining == 0 {
                enqueue(consumer);
                pushed = true;
            }
        }
        pushed
    }
}

#[derive(Default)]
pub(super) struct ReadClass {
    // Appended in builder order, with each generator appearing once per class.
    pub(super) readers: Vec<usize>,
    pub(super) has_writer: bool,
}

/// Finds the closure runnable without caller inputs, including random generators.
/// Every distinct read class must first be supplied by a generator in the closure.
pub(super) fn prepopulation_members(
    classes: &HashMap<usize, ReadClass>,
    writes: &[Vec<usize>],
) -> Result<Vec<usize>> {
    let mut remaining = vec![0usize; writes.len()];
    for class in classes.values() {
        for &reader in &class.readers {
            remaining[reader] += 1;
        }
    }
    let mut members: Vec<_> = (0..writes.len()).filter(|&g| remaining[g] == 0).collect();
    let mut available = HashSet::new();
    let mut head = 0;
    let mut unlocked = Vec::new();
    while head < members.len() {
        let generator = members[head];
        head += 1;
        unlocked.clear();
        for slot in &writes[generator] {
            let Some(class) = classes.get(slot) else {
                continue;
            };
            if !available.insert(*slot) {
                continue;
            }
            for &reader in &class.readers {
                remaining[reader] = remaining[reader].checked_sub(1).ok_or_else(|| {
                    anyhow!("readiness count released twice for generator {reader}")
                })?;
                if remaining[reader] == 0 {
                    unlocked.push(reader);
                }
            }
        }
        unlocked.sort_unstable();
        members.extend_from_slice(&unlocked);
    }
    Ok(members)
}

/// Orders members with a depth-first readiness walk and builds their local graph.
/// Read classes without a member writer, and explicitly supplied classes, are
/// external inputs.
/// Supplied classes retain no local producer even when members emit to them.
pub(super) fn prepare_schedule(
    members: &[usize],
    classes: &HashMap<usize, ReadClass>,
    writes: &[Vec<usize>],
    supplied_classes: &[usize],
) -> Result<GenerationSchedule> {
    const EXTERNAL: usize = usize::MAX;
    const UNASSIGNED: usize = usize::MAX - 1;

    let mut local = vec![EXTERNAL; writes.len()];
    for (index, &generator) in members.iter().enumerate() {
        ensure!(
            generator < writes.len(),
            "invalid schedule generator {generator}"
        );
        ensure!(
            local[generator] == EXTERNAL,
            "generator {generator} occurs twice in a schedule"
        );
        local[generator] = index;
    }

    // Include output classes with no readers: their other writers must still
    // wait for the first assignment. Slots beyond this span have no local writer.
    let num_slots = members
        .iter()
        .flat_map(|&g| writes[g].iter())
        .max()
        .map_or(0, |&slot| slot + 1);
    let mut producers = vec![EXTERNAL; num_slots];
    for &generator in members {
        for &slot in &writes[generator] {
            producers[slot] = UNASSIGNED;
        }
    }
    for &slot in supplied_classes {
        if let Some(producer) = producers.get_mut(slot) {
            *producer = EXTERNAL;
        }
    }

    let mut remaining = vec![0usize; members.len()];
    for (&slot, class) in classes {
        if producers.get(slot) != Some(&UNASSIGNED) {
            continue;
        }
        for &reader in &class.readers {
            if local[reader] != EXTERNAL {
                remaining[local[reader]] += 1;
            }
        }
    }
    // Reverse the root pushes so the first ready circuit generator is visited
    // first. Newly unlocked generators take priority over roots still waiting.
    let mut ready: Vec<_> = (0..members.len()).filter(|&g| remaining[g] == 0).collect();
    ready.sort_unstable_by_key(|&g| core::cmp::Reverse(members[g]));
    let mut order = Vec::with_capacity(members.len());
    let mut unlocked = Vec::new();
    let mut edges = Vec::new();
    while let Some(generator) = ready.pop() {
        order.push(generator);
        unlocked.clear();
        for slot in &writes[members[generator]] {
            let producer = &mut producers[*slot];
            if *producer == EXTERNAL {
                continue;
            }
            if *producer != UNASSIGNED {
                if *producer != generator {
                    edges.push((*producer, generator));
                }
                continue;
            }
            *producer = generator;
            let Some(class) = classes.get(slot) else {
                continue;
            };
            for &reader in &class.readers {
                let reader = local[reader];
                if reader == EXTERNAL {
                    continue;
                }
                remaining[reader] = remaining[reader].checked_sub(1).ok_or_else(|| {
                    anyhow!(
                        "readiness count released twice for generator {}",
                        members[reader]
                    )
                })?;
                if reader != generator {
                    edges.push((generator, reader));
                }
                if remaining[reader] == 0 {
                    unlocked.push(reader);
                }
            }
        }
        // Match runtime successor pushes: builder order, then LIFO selection.
        unlocked.sort_unstable_by_key(|&g| members[g]);
        ready.extend_from_slice(&unlocked);
    }
    if order.len() != members.len() {
        let generator = remaining
            .iter()
            .position(|&count| count != 0)
            .expect("unscheduled generator has an unresolved read");
        let mut slots: Vec<_> = classes
            .iter()
            .filter(|(slot, class)| {
                producers.get(**slot) == Some(&UNASSIGNED)
                    && class.readers.binary_search(&members[generator]).is_ok()
            })
            .map(|(&slot, _)| slot)
            .collect();
        slots.sort_unstable();
        let blocked_classes = slots.len();
        slots.truncate(8);
        return Err(anyhow!(
            "witness preparation blocked: {} generators could not be ordered; generator {} waits on {blocked_classes} read classes (first eight: {slots:?})",
            members.len() - order.len(), members[generator]
        ));
    }

    let mut position = vec![0; members.len()];
    for (index, &generator) in order.iter().enumerate() {
        position[generator] = index;
    }
    for (producer, consumer) in &mut edges {
        *producer = position[*producer];
        *consumer = position[*consumer];
    }
    // Local indices follow the readiness walk; release order follows the
    // circuit builder so reindexing does not change successor-push priority.
    edges.sort_unstable_by_key(|&(producer, consumer)| (producer, members[order[consumer]]));
    edges.dedup();
    let mut pending = vec![0; members.len()];
    let mut successor_offsets = vec![0; members.len() + 1];
    let mut successors = Vec::with_capacity(edges.len());
    for (producer, consumer) in edges {
        ensure!(
            producer < consumer,
            "generator {} depends on later generator {}",
            members[order[consumer]],
            members[order[producer]]
        );
        pending[consumer] += 1;
        successor_offsets[producer + 1] += 1;
        successors.push(consumer);
    }
    for generator in 0..members.len() {
        successor_offsets[generator + 1] += successor_offsets[generator];
    }
    // Caller-only classes affect runtime readiness, but not the readiness walk
    // that selects generator writers and assigns schedule-local indices.
    let mut input_slots: Vec<_> = classes
        .iter()
        .filter(|(_, class)| {
            !class.has_writer
                && class
                    .readers
                    .iter()
                    .any(|&reader| local[reader] != EXTERNAL)
        })
        .map(|(&slot, _)| slot)
        .collect();
    input_slots.sort_unstable();
    let mut input_runs: Vec<InputRun> = Vec::new();
    for (input, &slot) in input_slots.iter().enumerate() {
        if let Some(run) = input_runs.last_mut().filter(|run| run.end == slot) {
            run.end = slot + 1;
        } else {
            input_runs.push(InputRun {
                start: slot,
                end: slot + 1,
                first_input: input,
            });
        }
    }
    let mut input_successor_offsets = Vec::with_capacity(input_slots.len() + 1);
    let mut input_successors = Vec::new();
    input_successor_offsets.push(0);
    for slot in &input_slots {
        for &reader in &classes[slot].readers {
            if local[reader] != EXTERNAL {
                let consumer = position[local[reader]];
                pending[consumer] += 1;
                input_successors.push(consumer);
            }
        }
        input_successor_offsets.push(input_successors.len());
    }
    let mut initial_ready: Vec<_> = (0..members.len()).filter(|&g| pending[g] == 0).collect();
    initial_ready.sort_unstable_by_key(|&g| members[order[g]]);
    Ok(GenerationSchedule {
        order: order.into_iter().map(|g| members[g]).collect(),
        pending,
        successor_offsets,
        successors,
        initial_ready,
        input_slots,
        input_runs,
        input_successor_offsets,
        input_successors,
    })
}
