use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};

/// Runs just this test in a child process so blocked worker joins cannot hang the suite.
pub fn run(test_name: &str, test: impl FnOnce() -> Result<()>) -> Result<()> {
    run_with_timeout(test_name, Duration::from_secs(30), test)
}

pub fn run_with_timeout(
    test_name: &str,
    timeout: Duration,
    test: impl FnOnce() -> Result<()>,
) -> Result<()> {
    const CHILD_TEST: &str = "PLONKY2_TEST_CHILD";
    if std::env::var(CHILD_TEST).as_deref() == Ok(test_name) {
        return test();
    }

    // Inherit output instead of piping it: verbose failures must not fill a pipe
    // and block the child before the parent can observe its exit.
    let mut child = Command::new(std::env::current_exe()?)
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(CHILD_TEST, test_name)
        .spawn()
        .with_context(|| format!("could not spawn test {test_name}"))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                ensure!(status.success(), "test {test_name} failed: {status}");
                return Ok(());
            }
            Ok(None) if start.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(10));
            }
            result => {
                // Always reap the child, including on a wait error. Killing the
                // process also stops threads blocked in scope cleanup.
                let kill = child.kill();
                let wait = child.wait();
                if let Err(error) = result {
                    bail!("could not wait for test {test_name}: {error}; kill={kill:?}, wait={wait:?}");
                }
                bail!("test {test_name} timed out after {timeout:?}; kill={kill:?}, wait={wait:?}");
            }
        }
    }
}
