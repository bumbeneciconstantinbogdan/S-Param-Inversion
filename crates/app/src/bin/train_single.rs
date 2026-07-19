//! Train a single model variant with proper scaling
//!
//! Usage: cargo run --release --bin train_single --model <model_name> [options]
//!
//! Model variants:
//!   - rvnn_base: Real-valued MLP, base configuration
//!   - rvnn_pi: Real-valued MLP with physics-informed loss
//!   - rvnn_noise: Real-valued MLP with input noise
//!   - rvnn_both: Real-valued MLP with both PI and noise
//!   - cvnn_base: Complex-valued MLP, base configuration
//!   - cvnn_pi: Complex-valued MLP with physics-informed loss
//!   - cvnn_noise: Complex-valued MLP with input noise
//!   - cvnn_both: Complex-valued MLP with both PI and noise
//!
//! All training uses StandardScaler for input/output normalization.

use std::path::Path;
use clap::{Arg, Command};
use sparam_app::workflows::config::{ModelConfig, TrainConfig};
use sparam_app::workflows::train::run_training;
use sparam_data::generation::load_samples_from_csv;
use sparam_models::{Activation, ComplexActivation, ComplexNormChoice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("train_single")
        .about("Train a single model variant with scaling")
        .arg(Arg::new("model").required(true).help("Model variant name"))
        .arg(Arg::new("data_dir").short('d').long("data-dir").default_value("data").help("Data directory"))
        .arg(Arg::new("artifacts_dir").short('a').long("artifacts-dir").default_value("artifacts").help("Artifacts directory"))
        .arg(Arg::new("physics_lambda").long("physics-lambda").value_parser(clap::value_parser!(f64)).help("Physics-informed loss weight"))
        .arg(Arg::new("input_noise").long("input-noise-std").value_parser(clap::value_parser!(f64)).help("Input noise standard deviation"))
        .arg(Arg::new("loss_type").long("loss").value_parser(clap::value_parser!(String)).help("Loss function type (mse, physics_forward, etc.)"))
        .arg(Arg::new("seed").short('s').long("seed").default_value("42").value_parser(clap::value_parser!(u64)).help("Random seed"))
        .arg(Arg::new("epochs").short('e').long("epochs").default_value("200").value_parser(clap::value_parser!(usize)).help("Max epochs"))
        .arg(Arg::new("verbose").short('v').long("verbose").action(clap::ArgAction::SetTrue).help("Verbose output"))
        .get_matches();

    let model_name = matches.get_one::<String>("model").unwrap();
    let data_dir = matches.get_one::<String>("data_dir").unwrap();
    let artifacts_dir = matches.get_one::<String>("artifacts_dir").unwrap();
    let physics_lambda = matches.get_one::<f64>("physics_lambda").copied();
    let input_noise_std = matches.get_one::<f64>("input_noise").copied();
    let loss_type = matches.get_one::<String>("loss_type").cloned();
    let mut seed = *matches.get_one::<u64>("seed").unwrap();
    let max_epochs = *matches.get_one::<usize>("epochs").unwrap();
    let verbose = matches.get_flag("verbose");
    
    // Use model-specific seed if default seed (42) is used
    // This ensures different models get different initializations
    if seed == 42 {
        seed = match model_name.as_str() {
            "rvnn_base" => 42,
            "rvnn_pi" => 43,
            "rvnn_noise" => 44,
            "rvnn_both" => 45,
            "cvnn_base" => 46,
            "cvnn_pi" => 47,
            "cvnn_noise" => 48,
            "cvnn_both" => 49,
            _ => 42,
        };
    }

    // Determine model type and configuration based on name
    let (model_type, model_config, use_physics, use_noise) = match model_name.as_str() {
        "rvnn_base" => (
            "Real",
            ModelConfig::Real {
                hidden_size: 64,
                activation: Activation::ReLU,
                dropout: 0.0,
                norm: String::from("layernorm"),
            },
            false,
            false,
        ),
        "rvnn_pi" => (
            "Real",
            ModelConfig::Real {
                hidden_size: 64,
                activation: Activation::ReLU,
                dropout: 0.0,
                norm: String::from("layernorm"),
            },
            false,
            false,
        ),
        "rvnn_noise" => (
            "Real",
            ModelConfig::Real {
                hidden_size: 64,
                activation: Activation::ReLU,
                dropout: 0.0,
                norm: String::from("layernorm"),
            },
            false,
            true,
        ),
        "rvnn_both" => (
            "Real",
            ModelConfig::Real {
                hidden_size: 64,
                activation: Activation::ReLU,
                dropout: 0.0,
                norm: String::from("layernorm"),
            },
            false,
            true,
        ),
        "cvnn_base" => (
            "Complex",
            ModelConfig::Complex {
                hidden_size: 48,
                complex_activation: ComplexActivation::Worelu,
                dropout: 0.0004071689035540895,
                norm: ComplexNormChoice::LayerNorm,
            },
            false,
            false,
        ),
        "cvnn_pi" => (
            "Complex",
            ModelConfig::Complex {
                hidden_size: 48,
                complex_activation: ComplexActivation::Worelu,
                dropout: 0.0004071689035540895,
                norm: ComplexNormChoice::LayerNorm,
            },
            false,  // PI models now use eps_rel_mse, not physics_forward
            false,
        ),
        "cvnn_noise" => (
            "Complex",
            ModelConfig::Complex {
                hidden_size: 48,
                complex_activation: ComplexActivation::Worelu,
                dropout: 0.0004071689035540895,
                norm: ComplexNormChoice::LayerNorm,
            },
            false,
            true,
        ),
        "cvnn_both" => (
            "Complex",
            ModelConfig::Complex {
                hidden_size: 48,
                complex_activation: ComplexActivation::Worelu,
                dropout: 0.0004071689035540895,
                norm: ComplexNormChoice::LayerNorm,
            },
            false,
            true,
        ),
        _ => return Err(format!("Unknown model variant: {}", model_name).into()),
    };

    // Override with command line arguments if provided
    let effective_physics_lambda = physics_lambda.or(if use_physics { Some(0.1) } else { None });
    let effective_input_noise = input_noise_std.or(if use_noise { Some(0.007933403759394204) } else { None });
    
    // Use eps_rel_mse loss for all models (winning recipe from M10 study)
    // Command line --loss overrides the automatic selection
    let effective_loss = loss_type.unwrap_or_else(|| {
        "eps_rel_mse".to_string()
    });

    println!("{}", "=".repeat(80));
    println!("TRAINING: {} ({} model)", model_name, model_type);
    println!("{}", "-".repeat(80));

    // Load datasets
    let train_path = Path::new(data_dir).join("data_train.csv");
    let val_path = Path::new(data_dir).join("data_val.csv");
    let test_path = Path::new(data_dir).join("data_test.csv");
    
    if !train_path.exists() || !val_path.exists() || !test_path.exists() {
        return Err(format!("Data files not found in {}", data_dir).into());
    }
    
    if verbose {
        println!("Loading datasets...");
    }
    let train_samples = load_samples_from_csv(&train_path)?;
    let val_samples = load_samples_from_csv(&val_path)?;
    let test_samples = load_samples_from_csv(&test_path)?;
    
    if verbose {
        println!("  Train: {} samples", train_samples.len());
        println!("  Val:   {} samples", val_samples.len());
        println!("  Test:  {} samples", test_samples.len());
    }

    // Create checkpoint path
    let checkpoint_path = Path::new(artifacts_dir).join(format!("{}/model.safetensors", model_name));
    
    // Build training configuration
    // Use max_epochs from command line, but override with winning recipe if not specified
    let actual_max_epochs = if max_epochs == 100 { 100 } else { max_epochs };
    
    let config = TrainConfig {
        train_samples: train_samples.into(),
        val_samples: val_samples.into(),
        test_samples: test_samples.into(),
        model: model_config,
        checkpoint_path: Some(checkpoint_path.clone()),
        seed,
        max_epochs: actual_max_epochs,
        patience: actual_max_epochs / 10,  // 10% of max epochs
        batch_size: "16".to_string(),  // Winning recipe uses batch_size=16
        lr: 0.012305018392414956,  // Winning recipe learning rate
        weight_decay: 9.940853574579717e-7,  // Winning recipe weight decay
        optimizer: "adamw".to_string(),
        scheduler: "cosine".to_string(),
        loss: effective_loss.clone(),
        checkpoint: None,
        warmup_epochs: 0,
        shuffle_seed: None,
        log_progress: verbose,
        // PI and noise configurations
        grad_clip_norm: Some(0.5),  // Winning recipe gradient clip
        input_noise_std: effective_input_noise,
        cosine_eta_min: Some(4.34544151503091e-6),  // Winning recipe cosine eta min
        plateau_factor: None,
        plateau_patience: None,
        plateau_min_lr: None,
        sgd_momentum: None,
        sgd_nesterov: None,
        rmsprop_momentum: None,
        rmsprop_alpha: None,
        pi_mape_beta: None,
        physics_lambda: effective_physics_lambda,
    };

    if verbose {
        println!("\nTraining configuration:");
        println!("  Model type: {}", model_type);
        println!("  Physics-informed: {}", effective_physics_lambda.is_some());
        println!("  Input noise: {}", effective_input_noise.is_some());
        println!("  Max epochs: {}", actual_max_epochs);
        println!("  Batch size: 16");
        println!("  Optimizer: adam (lr=0.012305, wd=9.94e-7)");
        println!("  Scheduler: cosine (eta_min=4.35e-6)");
        println!("  Loss: {}", effective_loss);
        println!("  Grad clip: 0.5");
        println!("  Scaling: StandardScaler (automatic)");
    }

    // Run training
    if verbose {
        println!("\nStarting training...");
    }
    let result = run_training(&config)?;

    println!("\n{}", "=".repeat(80));
    println!("TRAINING COMPLETE: {}", model_name);
    println!("{}", "-".repeat(80));
    println!("  Best epoch: {}", result.metrics.best_epoch);
    println!("  Mean error: {:.4}%", result.metrics.mean_error);
    println!("  Max error: {:.4}%", result.metrics.max_error);
    println!("  Min error: {:.4}%", result.metrics.min_error);
    println!("  Median error: {:.4}%", result.metrics.median_error);
    println!("  R²(ε'): {:.5}", result.metrics.r2_real);
    println!("  R²(ε''): {:.5}", result.metrics.r2_imag);
    println!("  Model saved to: {}", checkpoint_path.display());

    Ok(())
}
