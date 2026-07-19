//! Core types, error handling, and utilities for S-parameter inversion.

#[cfg(feature = "tensor")]
pub mod batch_source;
pub mod complex;
#[cfg(feature = "tensor")]
pub mod complex_tensor;
pub mod config;
pub mod constants;
/// Reproducible-init utilities for `candle_nn::VarMap`s.  Lives here
/// (rather than `sparam-training`) because `sparam-models`' tests
/// also need it — hosting it in the lower crate breaks the dev-dep
/// cycle `models -> training -> models`.  See the `varmap` feature
/// in `Cargo.toml`.
#[cfg(feature = "varmap")]
pub mod determinism;
pub mod error;
pub mod grid;
pub mod io;
pub mod math;
#[cfg(feature = "tensor")]
pub mod metrics;
pub mod rng;
pub mod s_params;
#[cfg(feature = "tensor")]
pub mod tensor_ops;
pub mod validation;
