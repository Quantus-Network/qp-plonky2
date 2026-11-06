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
    idle_workers: usize,
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

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    fn find(
        &self,
        local_queue: &Worker<usize>,
        worker_index: usize,
        next_victim_index: &mut usize,
    ) -> Option<usize> {
        loop {
            if self.is_stopped() {
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
        state.idle_workers += 1;
        let local_index = loop {
            if self.is_stopped() {
                break None;
            }
            if let Some(local_index) = self.find(local_queue, worker_index, next_victim_index) {
                break Some(local_index);
            }
            // A worker remains active until it registers under this mutex.
            // If every worker is idle after a full scan, nobody can publish
            // more work. A worker finding work unregisters before unlocking.
            if state.idle_workers == self.stealers.len() {
                state.exhausted = true;
                self.stopped.store(true, Ordering::Release);
                self.wake.notify_all();
                break None;
            }
            state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
        };
        state.idle_workers -= 1;
        local_index
    }

    pub(super) fn notify_work(&self) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.idle_workers != 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    mod subprocess {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/subprocess.rs"
        ));
    }

    const TIMEOUT: Duration = Duration::from_secs(5);

    fn wait_for_sleepers(coordinator: &WorkerCoordinator, count: usize) {
        let start = Instant::now();
        loop {
            // Taking the mutex ensures registered workers have reached wait.
            if coordinator.state.lock().unwrap().idle_workers == count {
                return;
            }
            assert!(start.elapsed() < TIMEOUT, "workers did not become idle");
            std::thread::yield_now();
        }
    }

    #[test]
    fn empty_queues_and_successor_chains_terminate() -> Result<()> {
        subprocess::run(
            "iop::worker_coordinator::tests::empty_queues_and_successor_chains_terminate",
            || {
                for num_workers in [1, 2, 4] {
                    for num_tasks in [0, 256] {
                        let queues: Vec<_> = (0..num_workers).map(|_| Worker::new_lifo()).collect();
                        if num_tasks > 0 {
                            queues[0].push(0);
                        }
                        let coordinator =
                            WorkerCoordinator::new(queues.iter().map(Worker::stealer).collect());
                        let counts: Vec<_> = (0..num_tasks).map(|_| AtomicUsize::new(0)).collect();
                        std::thread::scope(|scope| {
                            for (worker_index, local) in queues.into_iter().enumerate() {
                                let coordinator = &coordinator;
                                let counts = &counts;
                                scope.spawn(move || {
                                    let mut victim = (worker_index + 1) % num_workers;
                                    while let Some(task) =
                                        coordinator.next(&local, worker_index, &mut victim)
                                    {
                                        assert_eq!(counts[task].fetch_add(1, Ordering::Relaxed), 0);
                                        if task + 1 < num_tasks {
                                            local.push(task + 1);
                                            coordinator.notify_work();
                                        }
                                    }
                                });
                            }
                        });
                        coordinator.finish()?;
                        assert!(counts
                            .iter()
                            .all(|count| count.load(Ordering::Relaxed) == 1));
                    }
                }
                Ok(())
            },
        )
    }

    #[test]
    fn fanout_wakes_all_sleepers_and_runs_each_task_once() -> Result<()> {
        subprocess::run(
            "iop::worker_coordinator::tests::fanout_wakes_all_sleepers_and_runs_each_task_once",
            || {
                const NUM_WORKERS: usize = 4;
                const NUM_TASKS: usize = 64;
                let mut queues: Vec<_> = (0..NUM_WORKERS).map(|_| Worker::new_lifo()).collect();
                let coordinator =
                    WorkerCoordinator::new(queues.iter().map(Worker::stealer).collect());
                let publisher = queues.remove(0);
                let counts: Vec<_> = (0..NUM_TASKS).map(|_| AtomicUsize::new(0)).collect();
                let (sender, receiver) = mpsc::channel();
                std::thread::scope(|scope| {
                    let mut releases = Vec::new();
                    for (offset, local) in queues.into_iter().enumerate() {
                        let worker_index = offset + 1;
                        let coordinator = &coordinator;
                        let counts = &counts;
                        let (release, resume) = mpsc::channel();
                        releases.push(release);
                        let sender = sender.clone();
                        scope.spawn(move || {
                            let mut victim = 0;
                            let first =
                                coordinator.next(&local, worker_index, &mut victim).unwrap();
                            assert_eq!(counts[first].fetch_add(1, Ordering::Relaxed), 0);
                            sender.send(()).unwrap();
                            // Each sleeper must obtain work before any proceeds to exhaustion.
                            resume
                                .recv_timeout(TIMEOUT)
                                .expect("publisher did not release worker");
                            while let Some(task) =
                                coordinator.next(&local, worker_index, &mut victim)
                            {
                                assert_eq!(counts[task].fetch_add(1, Ordering::Relaxed), 0);
                            }
                        });
                    }
                    wait_for_sleepers(&coordinator, NUM_WORKERS - 1);
                    for task in 0..NUM_TASKS {
                        publisher.push(task);
                    }
                    coordinator.notify_work();
                    for _ in 1..NUM_WORKERS {
                        receiver
                            .recv_timeout(TIMEOUT)
                            .expect("fanout did not wake every sleeper");
                    }
                    for release in releases {
                        release.send(()).unwrap();
                    }
                    let mut victim = 1;
                    while let Some(task) = coordinator.next(&publisher, 0, &mut victim) {
                        assert_eq!(counts[task].fetch_add(1, Ordering::Relaxed), 0);
                    }
                });
                coordinator.finish()?;
                assert!(counts
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1));
                Ok(())
            },
        )
    }

    #[test]
    fn failure_and_abort_wake_parked_workers() -> Result<()> {
        subprocess::run(
            "iop::worker_coordinator::tests::failure_and_abort_wake_parked_workers",
            || {
                for abort in [false, true] {
                    let mut queues: Vec<_> = (0..4).map(|_| Worker::new_lifo()).collect();
                    let coordinator =
                        WorkerCoordinator::new(queues.iter().map(Worker::stealer).collect());
                    let _publisher = queues.remove(0);
                    std::thread::scope(|scope| {
                        for (offset, local) in queues.into_iter().enumerate() {
                            let coordinator = &coordinator;
                            scope.spawn(move || {
                                assert_eq!(coordinator.next(&local, offset + 1, &mut 0), None);
                            });
                        }
                        wait_for_sleepers(&coordinator, 3);
                        if abort {
                            coordinator.abort();
                        } else {
                            coordinator.fail(anyhow::anyhow!("injected failure"));
                            coordinator.fail(anyhow::anyhow!("later failure"));
                        }
                    });
                    let error = coordinator.finish().unwrap_err().to_string();
                    assert!(
                        error.contains(if abort {
                            "did not complete"
                        } else {
                            "injected failure"
                        }),
                        "{error}"
                    );
                }
                Ok(())
            },
        )
    }
}
