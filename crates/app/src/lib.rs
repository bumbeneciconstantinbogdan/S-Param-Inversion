//! Application orchestration, CLI commands, and end-to-end workflows.

#[cfg(feature = "bench-utils")]
pub mod bench_support;
#[cfg(feature = "cli")]
pub mod cli;
pub mod workflows;
