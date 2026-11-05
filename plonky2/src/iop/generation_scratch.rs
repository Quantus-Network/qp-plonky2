//! Reusable temporary allocations for executing a generation schedule.

use crossbeam_deque::Worker;
use std::sync::atomic::AtomicUsize;

use crate::field::types::Field;
use crate::iop::generator::GeneratedValues;

/// Executor-owned allocations reset for each schedule. The witness borrow,
/// failure state and schedule-chain progress are held separately by the executor.
#[derive(Debug)]
pub(super) struct GenerationScratch<F: Field> {
    pub(super) pending_dependencies: Vec<AtomicUsize>,
    pub(super) workers: Vec<WorkerScratch<F>>,
}

impl<F: Field> Default for GenerationScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> GenerationScratch<F> {
    pub(super) fn new() -> Self {
        Self {
            pending_dependencies: Vec::new(),
            workers: Vec::new(),
        }
    }

    pub(super) fn reset(&mut self, counts: &[usize], workers: usize) {
        assert!(
            workers > 0,
            "witness generation requires at least one worker"
        );
        self.pending_dependencies
            .resize_with(counts.len(), || AtomicUsize::new(0));
        for (pending_dependencies, &count) in self.pending_dependencies.iter_mut().zip(counts) {
            *pending_dependencies.get_mut() = count;
        }
        if self.workers.len() < workers {
            self.workers.resize_with(workers, WorkerScratch::new);
        }
        for worker in &mut self.workers {
            worker.outputs.target_values.clear();
            while worker.ready.pop().is_some() {}
        }
    }

    /// Seeds balanced worker slices from roots already in circuit-builder order.
    pub(super) fn seed_ready(&mut self, initial_ready: &[usize], workers: usize) {
        debug_assert!(self.workers[..workers]
            .iter()
            .all(|worker| worker.ready.is_empty()));
        // Owners start at the slice's beginning; thieves take from its end.
        let per_worker = initial_ready.len() / workers;
        let extra = initial_ready.len() % workers;
        let mut start = 0;
        for (index, worker) in self.workers.iter().take(workers).enumerate() {
            let end = start + per_worker + usize::from(index < extra);
            for &generator in initial_ready[start..end].iter().rev() {
                worker.ready.push(generator);
            }
            start = end;
        }
    }
}

#[derive(Debug)]
pub(super) struct WorkerScratch<F: Field> {
    pub(super) outputs: GeneratedValues<F>,
    pub(super) ready: Worker<usize>,
}

impl<F: Field> WorkerScratch<F> {
    fn new() -> Self {
        Self {
            outputs: GeneratedValues::empty(),
            ready: Worker::new_lifo(),
        }
    }
}
