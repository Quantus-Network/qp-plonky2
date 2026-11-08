//! Logic common to multiple IOPs.

pub mod challenger;
mod dense_storage;
pub mod ext_target;
pub mod generator;
pub mod target;
pub mod wire;
pub mod witness;
pub mod witness_generation_executor;
pub mod witness_generation_plan;
pub mod witness_generation_planner;
pub mod witness_generation_schedule;
#[cfg(feature = "std")]
mod worker_coordinator;
