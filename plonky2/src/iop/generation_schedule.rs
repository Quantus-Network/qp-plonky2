#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{anyhow, ensure, Result};
use hashbrown::{HashMap, HashSet};

/// An execution order and dependency graph for a subset of circuit generators.
/// Graph indices are local to this schedule; `order` maps them to the circuit.
#[derive(Debug)]
pub struct GenerationSchedule {
    pub(super) order: Vec<usize>,
    pub(super) pending: Vec<usize>,
    successor_offsets: Vec<usize>,
    successors: Vec<usize>,
    pub(super) initial_ready: Vec<usize>,
}

impl GenerationSchedule {
    /// Circuit generator indices in serial execution order. Each position is a
    /// schedule-local index used by the dependency counts and successor lists.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Initial distinct-predecessor counts, indexed by schedule-local index.
    pub fn dependency_counts(&self) -> &[usize] {
        &self.pending
    }

    /// Distinct successors of a schedule-local generator, in local-index order.
    pub fn successors(&self, generator: usize) -> &[usize] {
        &self.successors[self.successor_offsets[generator]..self.successor_offsets[generator + 1]]
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

/// Orders the given members and builds their local dependency graph. Read classes
/// without a member writer, and explicitly supplied classes, are external inputs.
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
    // The order doubles as a FIFO queue until it is remapped to circuit indices.
    let mut order: Vec<_> = (0..members.len()).filter(|&g| remaining[g] == 0).collect();
    let mut head = 0;
    let mut unlocked = Vec::new();
    let mut edges = Vec::new();
    while head < order.len() {
        let generator = order[head];
        head += 1;
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
        // Members need not be supplied in builder order.
        unlocked.sort_unstable_by_key(|&g| members[g]);
        order.extend_from_slice(&unlocked);
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
    edges.sort_unstable();
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
    let initial_ready = (0..members.len()).filter(|&g| pending[g] == 0).collect();
    Ok(GenerationSchedule {
        order: order.into_iter().map(|g| members[g]).collect(),
        pending,
        successor_offsets,
        successors,
        initial_ready,
    })
}
