//! Re-export of [`BatchSource`] from `sparam_core`.
//!
//! The trait lives in `sparam-core` so that downstream crates (e.g. `sparam-data`)
//! can implement it without depending on `sparam-training`.

pub use sparam_core::batch_source::BatchSource;
