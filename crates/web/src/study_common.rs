//! Shared CLI scaffolding for the methodology and migration
//! binaries (the `m0N_*` family plus `migration_benchmark`,
//! `sqrt_branch_impact`, `extract_bests`, `gen_dense_eval`).
//! Each binary historically reimplemented
//! its own hand-rolled `arg(...)` helper plus the same four flags
//! (`--epochs`, `--seed`, `--only`, `--results-dir`); this module
//! consolidates them so a new study binary lands at ~10 lines of
//! boilerplate instead of ~50.
//!
//! Hand-rolled (no clap) so the binary surface stays consistent with
//! the historical pattern from `debug-train` (now retired).

use std::path::PathBuf;
use std::str::FromStr;

/// The four flags every `studyN` binary accepts; binary-specific
/// flags are parsed alongside via [`arg`] / [`arg_parsed`].
#[derive(Debug, Clone)]
pub struct CommonStudyArgs {
    pub epochs: usize,
    pub seed: u64,
    /// Comma-separated allowlist; `None` = run every cell.
    pub only: Option<String>,
    pub results_dir: PathBuf,
    /// Path to `sparam.db`.  Defaults to `sparam_data/sparam.db` (the
    /// same default the web server uses).
    pub db: PathBuf,
}

impl Default for CommonStudyArgs {
    fn default() -> Self {
        Self {
            epochs: 100,
            seed: 42,
            only: None,
            results_dir: PathBuf::from("poster/studies/methodology/results"),
            db: PathBuf::from("sparam_data/sparam.db"),
        }
    }
}

/// Parse the four common flags from `argv` (typically
/// `std::env::args().collect::<Vec<_>>()`).  Anything not on the list
/// stays in `argv` for binary-specific parsing via [`arg`] /
/// [`arg_parsed`].  Defaults match historical study-binary behaviour.
pub fn parse_common_args(argv: &[String]) -> CommonStudyArgs {
    let mut out = CommonStudyArgs::default();
    if let Some(v) = arg_parsed::<usize>(argv, "--epochs") {
        out.epochs = v;
    }
    if let Some(v) = arg_parsed::<u64>(argv, "--seed") {
        out.seed = v;
    }
    if let Some(v) = arg(argv, "--only") {
        out.only = Some(v.to_string());
    }
    if let Some(v) = arg(argv, "--results-dir") {
        out.results_dir = PathBuf::from(v);
    }
    if let Some(v) = arg(argv, "--db") {
        out.db = PathBuf::from(v);
    }
    out
}

/// Look up `--flag VALUE` and return the value as `&str` (zero-copy).
/// Returns `None` if the flag is missing or has no value after it.
pub fn arg<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
    let i = argv.iter().position(|a| a == flag)?;
    argv.get(i + 1).map(|s| s.as_str())
}

/// `arg` but returns an owned `String`.  Matches the historical
/// signature used across the methodology / migration binaries so
/// they can drop their local helper without touching call-site
/// bindings.
pub fn arg_owned(argv: &[String], flag: &str) -> Option<String> {
    arg(argv, flag).map(String::from)
}

/// Look up `--flag VALUE` and parse the value as `T`.  Returns `None`
/// on either missing flag or parse failure (so callers can chain
/// `.unwrap_or(default)`).
pub fn arg_parsed<T: FromStr>(argv: &[String], flag: &str) -> Option<T> {
    arg(argv, flag).and_then(|v| v.parse().ok())
}

/// `true` when `argv` contains the bare flag (no value follows).
/// E.g. `has_flag(&argv, "--dry-run")`.
pub fn has_flag(argv: &[String], flag: &str) -> bool {
    argv.iter().any(|a| a == flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn arg_returns_value_after_flag() {
        let v = argv(&["bin", "--epochs", "200", "--seed", "7"]);
        assert_eq!(arg(&v, "--epochs"), Some("200"));
        assert_eq!(arg(&v, "--seed"), Some("7"));
        assert_eq!(arg(&v, "--missing"), None);
    }

    #[test]
    fn arg_parsed_typed() {
        let v = argv(&["bin", "--epochs", "200"]);
        assert_eq!(arg_parsed::<usize>(&v, "--epochs"), Some(200));
        assert_eq!(arg_parsed::<usize>(&v, "--epochs-bad"), None);
    }

    #[test]
    fn parse_common_args_picks_up_all_four() {
        let v = argv(&[
            "bin",
            "--epochs", "50",
            "--seed", "99",
            "--only", "h8,easy",
            "--results-dir", "/tmp/r",
        ]);
        let p = parse_common_args(&v);
        assert_eq!(p.epochs, 50);
        assert_eq!(p.seed, 99);
        assert_eq!(p.only.as_deref(), Some("h8,easy"));
        assert_eq!(p.results_dir, PathBuf::from("/tmp/r"));
    }
}
