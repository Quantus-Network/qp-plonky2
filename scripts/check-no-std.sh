#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "${REPO_ROOT}"

# Ensure target is available
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true

# Build the minimal no_std check crate which depends on qp-plonky2 without std
cargo build -p no-std-check --target wasm32v1-none

echo "no_std check succeeded (compiled for wasm32v1-none)."


