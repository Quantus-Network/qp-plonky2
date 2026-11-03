//! Reusable temporary state for prepared witness execution.

use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::{Condvar, Mutex};

use crate::field::types::Field;
use crate::iop::generator::GeneratedValues;

/// Caller-owned scratch for one prepared generation at a time.
/// Each call resets its state while retaining allocations. Concurrent calls
/// require separate scratch objects; witness values are stored separately.
#[derive(Debug)]
pub struct GenerationScratch<F: Field> {
    pub(super) pending: Vec<AtomicUsize>,
    pub(super) workers: Vec<WorkerScratch<F>>,
    pub(super) queue: Mutex<QueueState>,
    pub(super) wake: Condvar,
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
            queue: Mutex::new(QueueState::default()),
            wake: Condvar::new(),
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
            worker.ready_successors.clear();
        }
        // A worker panic may poison this mutex; reset is exclusive and restores
        // all state before the scratch can be used by another call.
        let queue = self.queue.get_mut().unwrap_or_else(|e| e.into_inner());
        queue.ready.clear();
        queue
            .ready
            .extend(order.iter().copied().filter(|&g| counts[g] == 0));
        queue.remaining = counts.len();
        queue.failure = None;
        queue.panicked = false;
    }
}

#[derive(Debug)]
pub(super) struct WorkerScratch<F: Field> {
    pub(super) outputs: GeneratedValues<F>,
    pub(super) ready_successors: Vec<usize>,
}

impl<F: Field> WorkerScratch<F> {
    fn new() -> Self {
        Self {
            outputs: GeneratedValues::empty(),
            ready_successors: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct QueueState {
    pub(super) ready: VecDeque<usize>,
    pub(super) remaining: usize,
    pub(super) failure: Option<anyhow::Error>,
    pub(super) panicked: bool,
}
