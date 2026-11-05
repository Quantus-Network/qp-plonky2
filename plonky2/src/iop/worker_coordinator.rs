//! Work stealing and sleeping-worker coordination for witness generation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use anyhow::{ensure, Error, Result};
use crossbeam_deque::{Steal, Stealer, Worker};

/// Queues have one owner each. The coordinator holds only their stealing handles.
pub(super) struct WorkerCoordinator {
    stealers: Vec<Stealer<usize>>,
    stopped: AtomicBool,
    state: Mutex<CoordinationState>,
    wake: Condvar,
}

#[derive(Default)]
struct CoordinationState {
    sleepers: usize,
    exhausted: bool,
    failure: Option<Error>,
}

impl WorkerCoordinator {
    pub(super) fn new(stealers: Vec<Stealer<usize>>) -> Self {
        assert!(
            !stealers.is_empty(),
            "worker coordinator requires at least one queue"
        );
        Self {
            stealers,
            stopped: AtomicBool::new(false),
            state: Mutex::new(CoordinationState::default()),
            wake: Condvar::new(),
        }
    }

    fn finished(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    fn find(
        &self,
        local_queue: &Worker<usize>,
        worker_index: usize,
        next_victim_index: &mut usize,
    ) -> Option<usize> {
        loop {
            if self.finished() {
                return None;
            }
            if let Some(local_index) = local_queue.pop() {
                return Some(local_index);
            }
            let start = *next_victim_index;
            *next_victim_index = (start + 1) % self.stealers.len();
            let mut retry = false;
            for offset in 0..self.stealers.len() {
                let victim_index = (start + offset) % self.stealers.len();
                if victim_index == worker_index {
                    continue;
                }
                match self.stealers[victim_index].steal_batch_and_pop(local_queue) {
                    Steal::Success(local_index) => return Some(local_index),
                    Steal::Retry => retry = true,
                    Steal::Empty => {}
                }
            }
            if !retry {
                return None;
            }
            core::hint::spin_loop();
        }
    }

    pub(super) fn next(
        &self,
        local_queue: &Worker<usize>,
        worker_index: usize,
        next_victim_index: &mut usize,
    ) -> Option<usize> {
        if let Some(local_index) = self.find(local_queue, worker_index, next_victim_index) {
            return Some(local_index);
        }
        // Publishers use this same mutex after pushing. Register, then recheck
        // work under the mutex so a notification cannot be lost before wait.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.sleepers += 1;
        let local_index = loop {
            if self.finished() {
                break None;
            }
            if let Some(local_index) = self.find(local_queue, worker_index, next_victim_index) {
                break Some(local_index);
            }
            // A worker remains active until it registers under this mutex.
            // If every worker is idle after a full scan, nobody can publish
            // more work. A worker finding work unregisters before unlocking.
            if state.sleepers == self.stealers.len() {
                state.exhausted = true;
                self.stopped.store(true, Ordering::Release);
                self.wake.notify_all();
                break None;
            }
            state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
        };
        state.sleepers -= 1;
        local_index
    }

    pub(super) fn notify_work(&self) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.sleepers != 0 {
            // A release can expose a wide fan-out on one deque. Wake every
            // sleeper so available work is not left waiting for thieves.
            self.wake.notify_all();
        }
    }

    pub(super) fn fail(&self, error: Error) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.failure.is_none() {
            state.failure = Some(error);
        }
        self.stopped.store(true, Ordering::Release);
        self.wake.notify_all();
    }

    pub(super) fn abort(&self) {
        let _state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.stopped.store(true, Ordering::Release);
        self.wake.notify_all();
    }

    pub(super) fn finish(self) -> Result<()> {
        let state = self.state.into_inner().unwrap_or_else(|e| e.into_inner());
        if let Some(error) = state.failure {
            return Err(error);
        }
        ensure!(state.exhausted, "runnable witness work did not complete");
        Ok(())
    }
}
