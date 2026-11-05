//! Logic common to multiple IOPs.

pub mod challenger;
mod dense_storage;
pub mod ext_target;
pub mod generation_schedule;
pub mod generator;
pub mod scheduling_state;
pub mod target;
pub mod wire;
pub mod witness;
pub mod witness_generator_executor;
pub mod witness_generator_scheduler;
#[cfg(feature = "std")]
mod worker_coordinator;
