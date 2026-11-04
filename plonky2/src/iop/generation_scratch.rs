//! Reusable temporary state for prepared witness execution.

use crossbeam_deque::Worker;
use std::sync::atomic::AtomicUsize;

use crate::field::types::Field;
use crate::iop::generator::GeneratedValues;

/// Caller-owned scratch for one prepared generation at a time.
/// Initialization resets progress while retaining allocations. Input application
/// and execution preserve progress until the witness is complete. Concurrent
/// witnesses require separate scratch objects; witness values are stored separately.
#[derive(Debug)]
pub struct GenerationScratch<F: Field> {
    pub(super) pending: Vec<AtomicUsize>,
    pub(super) ready: Vec<usize>,
    pub(super) waiting_for_inputs: Vec<usize>,
    /// None before the first batch; true selects generator-driven input checks.
    pub(super) inputs_by_generator: Option<bool>,
    pub(super) workers: Vec<WorkerScratch<F>>,
    pub(super) num_workers: usize,
    pub(super) remaining: usize,
    pub(super) initialized: bool,
    pub(super) failed: bool,
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
            ready: Vec::new(),
            waiting_for_inputs: Vec::new(),
            inputs_by_generator: None,
            workers: Vec::new(),
            num_workers: 0,
            remaining: 0,
            initialized: false,
            failed: false,
        }
    }

    /// Whether every generator in the initialized schedule has completed.
    pub fn is_complete(&self) -> bool {
        self.initialized && !self.failed && self.remaining == 0
    }

    pub(super) fn reset(&mut self, counts: &[usize], initial_ready: &[usize], workers: usize) {
        assert!(
            workers > 0,
            "witness generation requires at least one worker"
        );
        self.num_workers = workers;
        self.remaining = counts.len();
        self.initialized = true;
        self.failed = false;
        self.waiting_for_inputs.clear();
        self.inputs_by_generator = None;
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
        self.ready.clear();
        self.ready.extend_from_slice(initial_ready);
    }

    /// Distributes pending ready entries in circuit-builder order without
    /// changing dependency counts or execution progress.
    pub(super) fn distribute_ready(&mut self, order: &[usize]) {
        debug_assert!(self.workers[..self.num_workers]
            .iter()
            .all(|worker| worker.ready.is_empty()));
        self.ready.sort_unstable_by_key(|&local| order[local]);
        // Reverse pushes so owners start at the slice's beginning and thieves
        // take from its end.
        let per_worker = self.ready.len() / self.num_workers;
        let extra = self.ready.len() % self.num_workers;
        let mut start = 0;
        for (index, worker) in self.workers.iter().take(self.num_workers).enumerate() {
            let end = start + per_worker + usize::from(index < extra);
            for &generator in self.ready[start..end].iter().rev() {
                worker.ready.push(generator);
            }
            start = end;
        }
        self.ready.clear();
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
