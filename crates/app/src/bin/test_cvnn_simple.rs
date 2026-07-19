//! Simple test to understand CVNN predictions

use std::path::Path;
use candle_core::{DType, Tensor};
use sparam_data::generation::{PermittivitySample, load_samples_from_csv};
use sparam_data::tensor_bridges::samples_to_tensors;
use sparam_data::scaling::StandardScaler;
use sparam_models::ModelType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load training data
    let train_path = Path::new("data/data_train.csv");
    let train_samples = load_samples_from_csv(train_path)?;
    println!("Loaded {} training samples", train_samples.len());
    
    // Get CVNN tensors
    let train_ds = samples_to_tensors(ModelType::Complex, &train_samples, DType::F64, false)?;
    println!("\nCVNN Training tensors:");
    println!("  Features shape: {:?}", train_ds.features.dims());
    println!("  Targets shape: {:?}", train_ds.targets.dims());
    
    // Fit scaler on targets
    let mut target_scaler = StandardScaler::new();
    target_scaler.fit(&train_ds.targets)?;
    
    // Print scaler stats
    let mean = target_scaler.mean()?;
    let std = target_scaler.std()?;
    println!("\nCVNN Target Scaler:");
    println!("  Mean: [{:.4}, {:.4}]", mean.get(0)?, mean.get(1)?);
    println!("  Std:  [{:.4}, {:.4}]", std.get(0)?, std.get(1)?);
    
    // Test with vacuum: ε' = 1, ε'' = 0
    let d = 0.0015;
    let a = 0.02286;
    let freq = 8.2e9;
    let (s11, s21) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
        d, a, freq, 1.0, 0.0
    );
    
    let test_sample = PermittivitySample {
        s11_real: s11.re,
        s11_imag: s11.im,
        s21_real: s21.re,
        s21_imag: s21.im,
        eps_prime: 1.0,
        eps_double_prime: 0.0,
        is_dense_patch: false,
    };
    
    let test_ds = samples_to_tensors(ModelType::Complex, &[test_sample], DType::F64, false)?;
    println!("\nVacuum test sample:");
    println!("  ε' = 1.0, ε'' = 0.0");
    println!("  Features: {:?}", test_ds.features.to_vec2::<f64>()?);
    println!("  Targets: {:?}", test_ds.targets.to_vec2::<f64>()?);
    
    // Scale features and targets
    let scaled_features = target_scaler.transform(&test_ds.features)?;
    let scaled_targets = target_scaler.transform(&test_ds.targets)?;
    println!("\nScaled:");
    println!("  Features: {:?}", scaled_features.to_vec2::<f64>()?);
    println!("  Targets: {:?}", scaled_targets.to_vec2::<f64>()?);
    
    // Inverse transform targets
    let inv_targets = target_scaler.inverse_transform(&scaled_targets)?;
    println!("\nInverse transformed targets:");
    println!("  Values: {:?}", inv_targets.to_vec2::<f64>()?);
    println!("  Interpretation:");
    println!("    ε' = {}", inv_targets.get(0, 0)?);
    println!("    -ε'' = {} => ε'' = {}", inv_targets.get(0, 1)?, -inv_targets.get(0, 1)?);
    
    // What if model predicts all zeros?
    let zero_pred = Tensor::zeros_like(&scaled_targets)?;
    let zero_inv = target_scaler.inverse_transform(&zero_pred)?;
    println!("\nIf model predicts zeros (mean):");
    println!("  Inverse: {:?}", zero_inv.to_vec2::<f64>()?);
    println!("  ε' = {}, ε'' = {}", zero_inv.get(0, 0)?, -zero_inv.get(0, 1)?);
    
    Ok(())
}
