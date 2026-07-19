//! Thin CLI entry point for S-parameter inversion.

use clap::{Parser, Subcommand};
use sparam_app::cli::benchmark::BenchmarkArgs;
use sparam_app::cli::evaluate::EvaluateArgs;
use sparam_app::cli::generate::GenerateDataArgs;
use sparam_app::cli::hpo::HpoArgs;
use sparam_app::cli::train::TrainArgs;
use sparam_core::rng::SeedArgs;

#[derive(Parser)]
#[command(
    name = "sparam",
    about = "Complex-Valued Neural Network for S-parameter inversion"
)]
struct Cli {
    #[command(flatten)]
    seed_args: SeedArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Generate synthetic permittivity dataset via NRW forward model.
    GenerateData(GenerateDataArgs),

    /// Train a single MLP model (real or complex-valued).
    Train(TrainArgs),

    /// Evaluate a trained model on test data.
    Evaluate(EvaluateArgs),

    /// Run multi-objective hyperparameter optimization.
    Hpo(HpoArgs),

    /// Run performance benchmarks.
    Benchmark(BenchmarkArgs),
}

fn main() {
    let cli = Cli::parse();
    sparam_core::rng::set_global_seed(cli.seed_args.seed);

    let result = match cli.command {
        Some(Command::GenerateData(ref args)) => {
            sparam_app::cli::generate::run(args, cli.seed_args.seed).map_err(|e| e.to_string())
        }
        Some(Command::Train(ref args)) => {
            sparam_app::cli::train::run(args, cli.seed_args.seed).map_err(|e| e.to_string())
        }
        Some(Command::Evaluate(ref args)) => {
            sparam_app::cli::evaluate::run(args).map_err(|e| e.to_string())
        }
        Some(Command::Hpo(ref args)) => {
            sparam_app::cli::hpo::run(args, cli.seed_args.seed).map_err(|e| e.to_string())
        }
        Some(Command::Benchmark(ref args)) => {
            sparam_app::cli::benchmark::run(args, cli.seed_args.seed).map_err(|e| e.to_string())
        }
        None => {
            eprintln!("No command specified. Use --help for usage.");
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
