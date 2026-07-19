//! Debug script to check CVNN predictions

use std::path::Path;
use candle_core::{Device, Tensor, DType};
use sparam_models::{MlpModel, ModelType, Activation, ComplexActivation, ComplexNormChoice};
use sparam_data::generation::{PermittivitySample, load_samples_from_csv};
use sparam_data::tensor_bridges::samples_to_tensors;
use sparam_data::scaling::StandardScaler;
use sparam_training::checkpoint::load_model_checkpoint;
use sparam_app::workflows::internal::model_factory::build_model;
use sparam_app::workflows::config::ModelConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test input: epsilon = 1 + 0j (vacuum)
    let d = 0.0015;
    let a = 0.02286;
    let freq = 8.2e9;
    let eps_prime = 1.0;
    let eps_double_prime = 0.0;
    
    let (s11, s21) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
        d, a, freq, eps_prime, eps_double_prime
    );
    
    println!("Test input (vacuum):");
    println!("  eps' = {}, eps'' = {}", eps_prime, eps_double_prime);
    println!("  S11 = ({:.6}, {:.6})", s11.re, s11.im);
    println!("  S21 = ({:.6}, {:.6})", s21.re, s21.im);
    
    // Create sample
    let sample = PermittivitySample {
        s11_real: s11.re,
        s11_imag: s11.im,
        s21_real: s21.re,
        s21_imag: s21.im,
        eps_prime: 1.0,
        eps_double_prime: 0.0,
        is_dense_patch: false,
    };
    
    // Load training data for scaler
    let train_path = Path::new("data/data_train.csv");
    let train_samples = load_samples_from_csv(train_path)?;
    println!("\nLoaded {} training samples", train_samples.len());
    
    // Check first few training targets for CVNN
    let model_type = ModelType::Complex;
    let train_ds = samples_to_tensors(model_type, &train_samples[..5], DType::F64, false)?;
    println!("\nCVNN Training targets (first 5):");
    let train_targets_vec = train_ds.targets.to_vec2::<f64>()?;
    for (i, row) in train_targets_vec.iter().enumerate() {
        println!("  [{}] col0={:.4}, col1={:.4}", i, row[0], row[1]);
    }
    
    // Fit scaler on ALL training targets
    let mut target_scaler = StandardScaler::new();
    let all_train_ds = samples_to_tensors(model_type, &train_samples, DType::F64, false)?;
    target_scaler.fit(&all_train_ds.targets)?;
    println!("\nTarget scaler statistics:");
    println!("  Mean: {:?}", target_scaler.mean()?);
    println!("  Std: {:?}", target_scaler.std()?);
    
    // Create test tensor
    let test_ds = samples_to_tensors(model_type, &[sample.clone()], DType::F64, false)?;
    println!("\nTest sample:");
    println!("  Features shape: {:?}", test_ds.features.dims());
    println!("  Features: {:?}", test_ds.features.to_vec2::<f64>()?);
    println!("  Targets: {:?}", test_ds.targets.to_vec2::<f64>()?);
    
    // Scale features
    let scaled_features = target_scaler.transform(&test_ds.features)?;
    println!("\nScaled features: {:?}", scaled_features.to_vec2::<f64>()?);
    
    // Load CVNN model
    let model_config = ModelConfig::Complex {
        hidden_size: 48,
        complex_activation: ComplexActivation::Worelu,
        dropout: 0.0,
        norm: ComplexNormChoice::None,
    };
    let (model, mut varmap) = build_model(&model_config, DType::F64)?;
    let model_path = Path::new("artifacts/cvnn_base/model.safetensors");
    if !model_path.exists() {
        return Err(format!("Model not found at {}", model_path.display()).into());
    }
    load_model_checkpoint(&mut varmap, model_path)?;
    println!("\nLoaded CVNN model");
    
    // Run prediction
    let output = model.forward_t(&scaled_features, false)?;
    println!("\nModel raw output (scaled space):");
    println!("  Shape: {:?}", output.dims());
    println!("  Values: {:?}", output.to_vec2::<f64>()?);
    
    // Inverse scale
    let output_physical = target_scaler.inverse_transform(&output)?;
    println!("\nModel output (physical space, before clamp):");
    let output_vec = output_physical.to_vec2::<f64>()?;
    println!("  Shape: {:?}", output_physical.dims());
    println!("  Values: {:?}", output_vec);
    println!("  Column 0 (eps' or Re): {}", output_vec[0][0]);
    println!("  Column 1 (-eps'' or Im): {}", output_vec[0][1]);
    println!("\n  Interpretation for Complex:");
    println!("    eps' = {}", output_vec[0][0]);
    println!("    -eps'' = {} => eps'' = {}", output_vec[0][1], -output_vec[0][1]);
    
    Ok(())
}
