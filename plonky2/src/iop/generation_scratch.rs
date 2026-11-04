//! Reusable temporary state for prepared witness execution.

use crossbeam_deque::Worker;
use std::sync::atomic::AtomicUsize;

use crate::field::types::Field;
use crate::iop::generator::GeneratedValues;

/// Caller-owned scratch for one prepared generation at a time.
/// Each call resets its state while retaining allocations. Concurrent calls
/// require separate scratch objects; witness values are stored separately.
#[derive(Debug)]
pub struct GenerationScratch<F: Field> {
    pub(super) pending: Vec<AtomicUsize>,
    pub(super) workers: Vec<WorkerScratch<F>>,
}

impl<F: Field> Default for GenerationScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> GenerationScratch<F> {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            workers: Vec::new(),
        }
    }

    pub(super) fn reset(&mut self, counts: &[usize], order: &[usize], workers: usize) {
        self.pending
            .resize_with(counts.len(), || AtomicUsize::new(0));
        for (pending, &count) in self.pending.iter_mut().zip(counts) {
            *pending.get_mut() = count;
        }
        if self.workers.len() < workers {
            self.workers.resize_with(workers, WorkerScratch::new);
        }
        for worker in &mut self.workers {
            worker.outputs.target_values.clear();
            while worker.ready.pop().is_some() {}
        }
        // Seed in prepared order, round-robin. Each owner pops its last push.
        for (index, &generator) in order.iter().filter(|&&g| counts[g] == 0).enumerate() {
            self.workers[index % workers].ready.push(generator);
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
