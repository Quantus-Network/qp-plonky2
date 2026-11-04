//! Logic common to multiple IOPs.

pub mod challenger;
pub(crate) mod dense_storage;
pub mod ext_target;
#[cfg(feature = "std")]
pub(crate) mod generation_pool;
pub mod generation_schedule;
#[cfg(feature = "std")]
pub mod generation_scratch;
pub mod generator;
pub(crate) mod generator_outputs;
pub mod prepared_witness;
pub mod target;
pub mod wire;
pub mod witness;
