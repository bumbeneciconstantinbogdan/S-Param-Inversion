//! End-to-end benchmark workflow.

use std::process::Command;
use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use serde::Serialize;

use sparam_data::generation::{DataGenerationConfig, generate_non_magnetic_data};
use sparam_data::scaling::{Scaler, StandardScaler};
use sparam_models::{Activation, MLPConfig, MLPRegressor};
use sparam_physics::nrw::{WaveguideConfig, nrw_direct_with_config};
use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::io::ensure_dir;

use super::internal::writers::{write_json_artifact, write_text_artifact};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for the benchmark workflow.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Benchmark suite to run: "all", "nrw", "model", "training".
    pub suite: String,
    /// Number of iterations for quick in-process benchmarks.
    pub iterations: usize,
    /// Output directory for results.
    pub output_dir: std::path::PathBuf,
    /// Run Criterion benchmarks via `cargo bench`.
    pub criterion: bool,
    /// Random seed.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Complete benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub suites: Vec<SuiteResult>,
    pub total_time_secs: f64,
}

/// Results from a single benchmark suite.
#[derive(Debug, Clone, Serialize)]
pub struct SuiteResult {
    pub name: String,
    pub benchmarks: Vec<BenchEntry>,
}

/// A single benchmark measurement.
#[derive(Debug, Clone, Serialize)]
pub struct BenchEntry {
    pub name: String,
    pub iterations: usize,
    pub total_ms: f64,
    pub mean_us: f64,
    pub throughput: Option<f64>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the benchmark workflow.
pub fn run_benchmark(config: &BenchmarkConfig) -> Result<BenchmarkReport> {
    ensure_dir(&config.output_dir)?;

    let suite_lower = config.suite.to_ascii_lowercase();
    let suites: Vec<&str> = match suite_lower.as_str() {
        "all" => vec!["nrw", "model", "training"],
        other => vec![match other {
            "nrw" | "model" | "training" => other,
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown suite '{other}', expected: all, nrw, model, training"
                )));
            }
        }],
    };

    // Optionally run Criterion
    if config.criterion {
        eprintln!("Running Criterion benchmarks via `cargo bench`...");
        let status = Command::new("cargo")
            .args(["bench", "--bench", "performance"])
            .status()
            .map_err(|e| candle_core::Error::Msg(format!("failed to run cargo bench: {e}")))?;
        if !status.success() {
            eprintln!("Warning: cargo bench exited with {status}");
        }
    }

    let start = Instant::now();
    let mut results = Vec::new();

    for suite in &suites {
        let suite_result = match *suite {
            "nrw" => bench_nrw(config.iterations)?,
            "model" => bench_model(config.iterations, config.seed)?,
            "training" => bench_training(config.iterations, config.seed)?,
            _ => unreachable!(),
        };
        results.push(suite_result);
    }

    let elapsed = start.elapsed();
    let report = BenchmarkReport {
        suites: results,
        total_time_secs: elapsed.as_secs_f64(),
    };

    eprintln!("Total benchmark time: {:.2}s", elapsed.as_secs_f64());
    Ok(report)
}

/// Write a benchmark report to disk in the given format.
pub fn write_benchmark_report(
    report: &BenchmarkReport,
    output_dir: &std::path::Path,
    format: &str,
) -> Result<()> {
    match format {
        "json" => {
            write_json_artifact(report, &output_dir.join("benchmarks.json"), "benchmark results")?;
            eprintln!(
                "Results written to {}",
                output_dir.join("benchmarks.json").display()
            );
        }
        "markdown" | "md" => {
            let md = format_markdown(report);
            write_text_artifact(&output_dir.join("benchmarks.md"), &md, "benchmark markdown")?;
            println!("{md}");
        }
        "text" => {
            print_text(report);
        }
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "unknown benchmark format '{format}'"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// NRW benchmarks
// ---------------------------------------------------------------------------

fn bench_nrw(iterations: usize) -> Result<SuiteResult> {
    eprintln!("Benchmarking NRW...");
    let device = Device::Cpu;
    let config = WaveguideConfig::new(1.5e-3, 22.86e-3)
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

    let mut entries = Vec::new();

    for batch_size in [1, 64, 256, 1024, 4096] {
        let (freqs, eps_r, mu_r) = make_nrw_inputs(batch_size, &device)?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = nrw_direct_with_config(&config, &freqs, &eps_r, &mu_r)?;
        }
        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let mean_us = total_ms * 1000.0 / iterations as f64;
        let throughput = batch_size as f64 * iterations as f64 / elapsed.as_secs_f64();

        entries.push(BenchEntry {
            name: format!("nrw_batch_{batch_size}"),
            iterations,
            total_ms,
            mean_us,
            throughput: Some(throughput),
        });
    }

    Ok(SuiteResult {
        name: "nrw".to_string(),
        benchmarks: entries,
    })
}

fn make_nrw_inputs(
    batch_size: usize,
    device: &Device,
) -> Result<(Tensor, ComplexTensor, ComplexTensor)> {
    let freq_span = 12.4e9 - 8.2e9;
    let freqs: Vec<f64> = if batch_size == 1 {
        vec![8.2e9]
    } else {
        (0..batch_size)
            .map(|i| 8.2e9 + freq_span * i as f64 / (batch_size - 1) as f64)
            .collect()
    };

    let frequencies = Tensor::new(freqs.as_slice(), device)?;
    let eps_r = ComplexTensor::new(
        Tensor::new(vec![2.5f64; batch_size].as_slice(), device)?,
        Tensor::new(vec![-0.15f64; batch_size].as_slice(), device)?,
    )?;
    let mu_r = ComplexTensor::new(
        Tensor::new(vec![1.0f64; batch_size].as_slice(), device)?,
        Tensor::new(vec![0.0f64; batch_size].as_slice(), device)?,
    )?;
    Ok((frequencies, eps_r, mu_r))
}

// ---------------------------------------------------------------------------
// Model forward-pass benchmarks
// ---------------------------------------------------------------------------

fn bench_model(iterations: usize, _seed: u64) -> Result<SuiteResult> {
    eprintln!("Benchmarking model forward pass...");
    let device = Device::Cpu;

    let mut entries = Vec::new();

    for hidden in [32, 64, 128, 256] {
        let config = MLPConfig::new(
            4,
            hidden,
            2,
            Activation::GELU,
        );
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F64, &device);
        let model = MLPRegressor::new(vb.pp("model"), &config)?;

        let input = Tensor::randn(0f64, 1.0, &[256, 4], &device)?;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&input)?;
        }
        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let mean_us = total_ms * 1000.0 / iterations as f64;

        entries.push(BenchEntry {
            name: format!("forward_h{hidden}x3_b256"),
            iterations,
            total_ms,
            mean_us,
            throughput: Some(256.0 * iterations as f64 / elapsed.as_secs_f64()),
        });
    }

    Ok(SuiteResult {
        name: "model".to_string(),
        benchmarks: entries,
    })
}

// ---------------------------------------------------------------------------
// Training loop benchmarks (single epoch)
// ---------------------------------------------------------------------------

fn bench_training(iterations: usize, seed: u64) -> Result<SuiteResult> {
    eprintln!("Benchmarking training (1 epoch)...");
    let device = Device::Cpu;

    let data_config = DataGenerationConfig {
        seed,
        n_eps_prime_train: 10,
        n_eps_double_prime_train: 10,
        ..Default::default()
    };
    let dataset = generate_non_magnetic_data(&data_config)?;

    let n = dataset.train.len();
    let mut feat = Vec::with_capacity(n * 4);
    let mut targ = Vec::with_capacity(n * 2);
    for s in &dataset.train {
        feat.extend_from_slice(&[s.s11_real, s.s11_imag, s.s21_real, s.s21_imag]);
        targ.extend_from_slice(&[s.eps_prime, s.eps_double_prime]);
    }
    let train_x = Tensor::from_vec(feat, &[n, 4], &device)?;
    let train_y = Tensor::from_vec(targ, &[n, 2], &device)?;

    let mut scaler = StandardScaler::new();
    scaler.fit(&train_x)?;
    let train_x_scaled = scaler.transform(&train_x)?;

    let mut entries = Vec::new();

    for hidden in [32, 128] {
        let config = MLPConfig::new(
            4,
            hidden,
            2,
            Activation::GELU,
        );

        let iters = iterations.min(20);
        let start = Instant::now();
        for _ in 0..iters {
            let varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F64, &device);
            let model = MLPRegressor::new(vb.pp("model"), &config)?;

            let pred = model.forward(&train_x_scaled)?;
            let loss = candle_nn::loss::mse(&pred, &train_y)?;
            loss.backward()?;
        }
        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let mean_us = total_ms * 1000.0 / iters as f64;

        entries.push(BenchEntry {
            name: format!("train_1ep_h{hidden}x2_n{n}"),
            iterations: iters,
            total_ms,
            mean_us,
            throughput: None,
        });
    }

    Ok(SuiteResult {
        name: "training".to_string(),
        benchmarks: entries,
    })
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Format a benchmark report as plaintext.
pub fn print_text(report: &BenchmarkReport) {
    for suite in &report.suites {
        println!("\n=== {} ===", suite.name.to_uppercase());
        println!(
            "{:<35} {:>8} {:>12} {:>14}",
            "Benchmark", "Iters", "Mean (µs)", "Throughput"
        );
        println!("{}", "-".repeat(75));
        for b in &suite.benchmarks {
            let tp = b
                .throughput
                .map(|t| format!("{:.0}/s", t))
                .unwrap_or_default();
            println!(
                "{:<35} {:>8} {:>12.1} {:>14}",
                b.name, b.iterations, b.mean_us, tp
            );
        }
    }
    println!("\nTotal: {:.2}s", report.total_time_secs);
}

/// Format a benchmark report as markdown.
pub fn format_markdown(report: &BenchmarkReport) -> String {
    let mut md = String::from("# Benchmark Results\n\n");
    for suite in &report.suites {
        md.push_str(&format!("## {}\n\n", suite.name));
        md.push_str("| Benchmark | Iters | Mean (µs) | Throughput |\n");
        md.push_str("|-----------|-------|-----------|------------|\n");
        for b in &suite.benchmarks {
            let tp = b
                .throughput
                .map(|t| format!("{:.0}/s", t))
                .unwrap_or_else(|| "—".to_string());
            md.push_str(&format!(
                "| {} | {} | {:.1} | {} |\n",
                b.name, b.iterations, b.mean_us, tp
            ));
        }
        md.push('\n');
    }
    md.push_str(&format!("**Total time:** {:.2}s\n", report.total_time_secs));
    md
}
