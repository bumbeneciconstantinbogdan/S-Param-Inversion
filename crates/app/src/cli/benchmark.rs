//! US-10.5: `benchmark` — parse CLI args and delegate to workflows.

use std::path::PathBuf;

use candle_core::Error as CandleError;
use clap::Args;

use crate::workflows::benchmark::{
    BenchmarkConfig, run_benchmark, write_benchmark_report,
};

use super::shared::OutputFormat;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Benchmark suite to run: all, nrw, model, training.
    #[arg(long, short = 's', default_value = "all")]
    pub suite: String,

    /// Number of iterations for quick in-process benchmarks.
    #[arg(long, short = 'n', default_value_t = 100)]
    pub iterations: usize,

    /// Output directory for results.
    #[arg(long, short = 'o', default_value = "./artifacts/benchmarks")]
    pub output_dir: PathBuf,

    /// Run Criterion benchmarks via `cargo bench`.
    #[arg(long)]
    pub criterion: bool,

    /// Output format: text, json, markdown.
    #[arg(long, short = 'f', default_value = "text")]
    pub format: String,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: &BenchmarkArgs, seed: u64) -> Result<(), CandleError> {
    let format = OutputFormat::from_name(&args.format)?;

    let config = BenchmarkConfig {
        suite: args.suite.clone(),
        iterations: args.iterations,
        output_dir: args.output_dir.clone(),
        criterion: args.criterion,
        seed,
    };

    let report = run_benchmark(&config)?;
    write_benchmark_report(&report, &args.output_dir, &format.to_string())?;
    Ok(())
}
