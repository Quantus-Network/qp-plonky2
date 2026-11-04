//! Work stealing and sleeping-worker coordination for witness generation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use anyhow::{ensure, Error, Result};
use crossbeam_deque::{Steal, Stealer, Worker};

/// Queues have one owner each. This pool holds only their stealing handles.
pub(super) struct GenerationPool {
    stealers: Vec<Stealer<usize>>,
    stopped: AtomicBool,
    parking: Mutex<Parking>,
    wake: Condvar,
}

#[derive(Default)]
struct Parking {
    sleepers: usize,
    exhausted: bool,
    failure: Option<Error>,
}

impl GenerationPool {
    pub(super) fn new(stealers: Vec<Stealer<usize>>) -> Self {
        assert!(
            !stealers.is_empty(),
            "worker pool requires at least one queue"
        );
        Self {
            stealers,
            stopped: AtomicBool::new(false),
            parking: Mutex::new(Parking::default()),
            wake: Condvar::new(),
        }
    }

    fn finished(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    fn find(&self, local: &Worker<usize>, me: usize, victim: &mut usize) -> Option<usize> {
        loop {
            if self.finished() {
                return None;
            }
            if let Some(task) = local.pop() {
                return Some(task);
            }
            let start = *victim;
            *victim = (start + 1) % self.stealers.len();
            let mut retry = false;
            for offset in 0..self.stealers.len() {
                let index = (start + offset) % self.stealers.len();
                if index == me {
                    continue;
                }
                match self.stealers[index].steal_batch_and_pop(local) {
                    Steal::Success(task) => return Some(task),
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
        local: &Worker<usize>,
        me: usize,
        victim: &mut usize,
    ) -> Option<usize> {
        if let Some(task) = self.find(local, me, victim) {
            return Some(task);
        }
        // Publishers use this same mutex after pushing. Register, then recheck
        // work under the mutex so a notification cannot be lost before wait.
        let mut parking = self.parking.lock().unwrap_or_else(|e| e.into_inner());
        parking.sleepers += 1;
        let task = loop {
            if self.finished() {
                break None;
            }
            if let Some(task) = self.find(local, me, victim) {
                break Some(task);
            }
            // A worker remains active until it registers under this mutex.
            // If every worker is idle after a full scan, nobody can publish
            // more work. A worker finding work unregisters before unlocking.
            if parking.sleepers == self.stealers.len() {
                parking.exhausted = true;
                self.stopped.store(true, Ordering::Release);
                self.wake.notify_all();
                break None;
            }
            parking = self.wake.wait(parking).unwrap_or_else(|e| e.into_inner());
        };
        parking.sleepers -= 1;
        task
    }

    pub(super) fn notify_work(&self) {
        let parking = self.parking.lock().unwrap_or_else(|e| e.into_inner());
        if parking.sleepers != 0 {
            // A release can expose a wide fan-out on one deque. Wake every
            // sleeper so available work is not left waiting for thieves.
            self.wake.notify_all();
        }
    }

    pub(super) fn fail(&self, error: Error) {
        let mut parking = self.parking.lock().unwrap_or_else(|e| e.into_inner());
        if parking.failure.is_none() {
            parking.failure = Some(error);
        }
        self.stopped.store(true, Ordering::Release);
        self.wake.notify_all();
    }

    pub(super) fn abort(&self) {
        let _parking = self.parking.lock().unwrap_or_else(|e| e.into_inner());
        self.stopped.store(true, Ordering::Release);
        self.wake.notify_all();
    }

    pub(super) fn finish(self) -> Result<()> {
        let parking = self.parking.into_inner().unwrap_or_else(|e| e.into_inner());
        if let Some(error) = parking.failure {
            return Err(error);
        }
        ensure!(parking.exhausted, "runnable witness work did not complete");
        Ok(())
    }
}
