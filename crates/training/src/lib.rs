//! Training infrastructure: trainer, optimizers, schedulers, and checkpointing.
//!
//! Reproducible-init utilities live in [`sparam_core::determinism`]
//! (single source of truth — `sparam-models` tests pull from the
//! same place, which is what broke the prior dev-dep cycle).  No
//! re-export here: callers go directly through `sparam_core::determinism::*`.

pub mod batch_source;
pub mod checkpoint;
pub mod logger;
pub mod losses;
pub mod optimizers;
pub mod schedulers;
pub mod trainer;
