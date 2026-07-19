//! US-10.3: `evaluate` — parse CLI flags and delegate evaluation to workflows.

use std::path::PathBuf;

use candle_core::Error as CandleError;
use clap::Args;

use crate::workflows::config::EvaluateConfig as WorkflowEvaluateConfig;
use crate::workflows::evaluate::run_evaluation;

use super::shared::OutputFormat;

#[derive(Debug, Args)]
pub struct EvaluateArgs {
    /// Path to model checkpoint (.safetensors).
    #[arg(long, short = 'm')]
    pub model: PathBuf,

    /// Model config JSON (auto-detected from checkpoint dir by default).
    #[arg(long, short = 'c')]
    pub config: Option<PathBuf>,

    /// Path to test CSV file.
    #[arg(long, short = 'd')]
    pub data: PathBuf,

    /// Path to training CSV (for fitting scalers).
    #[arg(long)]
    pub train_data: Option<PathBuf>,

    /// Directory for outputs (defaults to same directory as model).
    #[arg(long, short = 'o')]
    pub output_dir: Option<PathBuf>,

    /// Error thresholds for classification (comma-separated percentages).
    #[arg(long, short = 't', value_parser = super::shared::parse_range, default_value = "1.0,10.0")]
    pub thresholds: (f64, f64),

    /// Generate visualization plots.
    #[arg(long, default_value_t = true)]
    pub generate_plots: bool,

    /// Output format: text, json, markdown.
    #[arg(long, default_value = "text")]
    pub format: String,

    /// Only output metrics, suppress progress.
    #[arg(long, short = 'q', default_value_t = false)]
    pub quiet: bool,
}

pub fn run(args: &EvaluateArgs) -> Result<(), CandleError> {
    let output_format = OutputFormat::from_name(&args.format)?;

    // Load model config JSON from disk (CLI convention: <model_dir>/config.json).
    let config_path = args.config.clone().unwrap_or_else(|| {
        args.model
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("config.json")
    });
    let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
        CandleError::Msg(format!("failed to read {}: {e}", config_path.display()))
    })?;
    #[derive(serde::Deserialize)]
    struct Persisted {
        #[serde(flatten)]
        model: crate::workflows::config::ModelConfig,
        parameter_count: usize,
    }
    let saved: Persisted = serde_json::from_str(&config_text)
        .map_err(|e| CandleError::Msg(format!("invalid model config JSON: {e}")))?;

    // Load samples from CSV.
    let test_samples = sparam_data::generation::load_samples_from_csv(&args.data)?;
    let scaler_samples = match &args.train_data {
        Some(p) if p.exists() => sparam_data::generation::load_samples_from_csv(p)?,
        _ => Vec::new(),
    };

    let result = run_evaluation(&WorkflowEvaluateConfig {
        // CLI passes a safetensors file path. The web UI passes
        // `CheckpointSource::Bytes` loaded from a DB BLOB instead.
        model: crate::workflows::config::CheckpointSource::Path(args.model.clone()),
        model_config: saved.model,
        parameter_count: saved.parameter_count,
        test_samples: test_samples.into(),
        scaler_samples: scaler_samples.into(),
        thresholds: args.thresholds,
        generate_plots: args.generate_plots,
        quiet: args.quiet,
        stack: crate::workflows::config::StackedInferenceConfig::default(),
        pre_fitted_scalers: None,
    })?;

    match output_format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result.output)
                .map_err(|error| CandleError::Msg(format!("JSON error: {error}")))?;
            println!("{json}");
        }
        OutputFormat::Text => {
            println!("Evaluation Results: {}", args.model.display());
            println!("{}", "=".repeat(50));
            println!("Test samples: {}", result.output.test_samples);
            println!();
            println!("Accuracy Metrics:");
            for classification in &result.classifications {
                println!(
                    "  OK@{:.0}%: {:.2}% ({}/{})",
                    classification.threshold,
                    classification.ok_percent,
                    classification.ok_count,
                    result.output.test_samples
                );
            }
            println!();
            println!("Error Statistics:");
            println!("  Mean:  {:.3}%", result.output.metrics.mean_error);
            println!("  Max:   {:.3}%", result.output.metrics.max_error);
            println!("  Min:   {:.3}%", result.output.metrics.min_error);
            println!("  Std:   {:.3}%", result.output.metrics.std_error);
            println!();
            println!("R² Scores:");
            println!("  ε' (real):  {:.5}", result.output.metrics.r2_real);
            println!("  ε'' (imag): {:.5}", result.output.metrics.r2_imag);
        }
        OutputFormat::Markdown => {
            println!("# Evaluation Results\n");
            println!("**Model:** `{}`\n", args.model.display());
            println!("| Metric | Value |");
            println!("|--------|-------|");
            for classification in &result.classifications {
                println!(
                    "| OK@{:.0}% | {:.2}% ({}/{}) |",
                    classification.threshold,
                    classification.ok_percent,
                    classification.ok_count,
                    result.output.test_samples
                );
            }
            println!("| Mean Error | {:.3}% |", result.output.metrics.mean_error);
            println!("| Max Error | {:.3}% |", result.output.metrics.max_error);
            println!("| R²(ε') | {:.5} |", result.output.metrics.r2_real);
            println!("| R²(ε'') | {:.5} |", result.output.metrics.r2_imag);
        }
    }

    Ok(())
}
