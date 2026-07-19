//! Comprehensive comparison of trained models vs NRW direct method with proper scaling
//!
//! This script:
//! 1. Loads training data and fits StandardScaler on it
//! 2. Computes S-parameters using NRW direct method for test points
//! 3. Scales test inputs using the same scaler fit on training data
//! 4. Runs all 8 models on scaled inputs
//! 5. Descales model outputs back to original ε range
//! 6. Compares predicted ε with original ε values
//!
//! Test points include:
//!   - Resonance region: ε' ∈ [148.8, 149.2] (near βₛ·d = π for n=1 half-wavelength)
//!   - Vacuum region: ε' ∈ [1.0, 1.01] (near free space)
//!
//! Usage: cargo run --release --bin compare_models

use std::collections::HashMap;
use std::path::Path;
use candle_core::DType;
use sparam_core::complex::Complex64;
use sparam_models::{Activation, ComplexActivation, ComplexNormChoice, MlpModel, ModelType as ModelsModelType};
use sparam_app::workflows::config::ModelConfig;
use sparam_app::workflows::internal::model_factory::build_model;
use sparam_app::workflows::internal::data_prep::{prepare_evaluation_data, PreparedEvaluationData};
use sparam_app::workflows::internal::evaluation::evaluate_model_predictions;
use sparam_data::generation::{load_samples_from_csv, PermittivitySample};
use sparam_training::checkpoint::load_model_checkpoint;

/// Prediction result structure
#[derive(Debug, Clone)]
struct PredictionResult {
    model_name: String,
    point_idx: usize,
    eps_prime_true: f64,
    eps_double_prime_true: f64,
    eps_prime_pred: f64,
    eps_double_prime_pred: f64,
    error_prime: f64,
    error_double_prime: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("COMPREHENSIVE MODEL COMPARISON WITH PROPER SCALING");
    println!("Trained Models vs NRW Direct Method vs Original Values");
    println!("{}", "=".repeat(80));

    // Waveguide configuration (WR-90 at 8.2 GHz)
    let d = 0.0015;      // sample thickness in meters
    let a = 0.02286;     // waveguide width in meters
    let freq = 8.2e9;    // frequency in Hz

    println!("\nConfiguration:");
    println!("  Frequency:       {:.1} GHz", freq / 1e9);
    println!("  Waveguide width: {:.4} mm (WR-90)", a * 1000.0);
    println!("  Sample thickness: {:.4} mm", d * 1000.0);
    println!("  Material:        non-magnetic (mu_r = 1)");
    println!("  Scaling:         StandardScaler (fit on training data)");
    println!("  Important:       Inputs are SCALED before model, outputs are DE-SCALED");

    // ========================================================================
    // SECTION 0: LOAD TRAINING DATA FOR SCALING
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 0: LOADING TRAINING DATA FOR SCALING");
    println!("{}", "=".repeat(80));

    let train_path = Path::new("data/data_train.csv");
    if !train_path.exists() {
        return Err(format!("Training data not found at {}", train_path.display()).into());
    }
    
    println!("Loading training data from {}...", train_path.display());
    let train_samples = load_samples_from_csv(train_path)?;
    println!("Loaded {} training samples for scaler fitting.", train_samples.len());
    
    // Print training data statistics
    let mut eps_prime_min = f64::INFINITY;
    let mut eps_prime_max = f64::NEG_INFINITY;
    let mut eps_double_prime_min = f64::INFINITY;
    let mut eps_double_prime_max = f64::NEG_INFINITY;
    for sample in &train_samples {
        eps_prime_min = eps_prime_min.min(sample.eps_prime);
        eps_prime_max = eps_prime_max.max(sample.eps_prime);
        eps_double_prime_min = eps_double_prime_min.min(sample.eps_double_prime);
        eps_double_prime_max = eps_double_prime_max.max(sample.eps_double_prime);
    }
    println!("\nTraining data ε' range: [{:.2}, {:.2}]", eps_prime_min, eps_prime_max);
    println!("Training data ε'' range: [{:.2}, {:.2}]", eps_double_prime_min, eps_double_prime_max);

    // ========================================================================
    // SECTION 1: DEFINE TEST POINTS AND CREATE SAMPLES
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 1: TEST POINTS DEFINITION");
    println!("{}", "=".repeat(80));

    // Define test epsilon values: resonance region + vacuum region
    let test_points_data: Vec<(f64, f64, &str)> = vec![
        // Resonance region (near βₛ·d = π for n=1 half-wavelength at er ≈ 149.1547)
        (148.8, 0.0, "148.8 - j0.0"),
        (148.9, 0.0, "148.9 - j0.0"),
        (149.0, 0.0, "149.0 - j0.0"),
        (149.1, 0.0, "149.1 - j0.0"),
        (149.15, 0.0, "149.15 - j0.0 (near resonance)"),
        (149.1547, 0.0, "149.1547 - j0.0 (resonance: βₛ·d=π)"),
        (149.2, 0.0, "149.2 - j0.0"),
        // With losses
        (149.1547, 0.05, "149.1547 - j0.05"),
        (149.1547, 0.1, "149.1547 - j0.1"),
        (149.1547, 0.2, "149.1547 - j0.2"),
        // Vacuum region
        (1.0, 0.0, "1.0 - j0.0 (vacuum)"),
        (1.0025, 0.0, "1.0025 - j0.0"),
        (1.005, 0.0, "1.005 - j0.0"),
        (1.01, 0.0, "1.01 - j0.0"),
        // With small losses
        (1.0, 0.001, "1.0 - j0.001"),
        (1.0, 0.005, "1.0 - j0.005"),
    ];

    // Compute S-parameters using NRW and create PermittivitySample structs
    let mut test_samples: Vec<PermittivitySample> = Vec::new();
    let mut ground_truth: Vec<(f64, f64, Complex64, Complex64)> = Vec::new();
    
    println!("\nComputing S-parameters using NRW direct method...");
    for (eps_prime, eps_double_prime, label) in &test_points_data {
        let (s11, s21) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
            d, a, freq, *eps_prime, *eps_double_prime
        );
        
        // Create a PermittivitySample with S-parameters
        let sample = PermittivitySample {
            s11_real: s11.re,
            s11_imag: s11.im,
            s21_real: s21.re,
            s21_imag: s21.im,
            eps_prime: *eps_prime,
            eps_double_prime: *eps_double_prime,
            is_dense_patch: false,
        };
        
        test_samples.push(sample);
        ground_truth.push((*eps_prime, *eps_double_prime, s11, s21));
    }

    println!("\nTest points ({} total):", test_samples.len());
    for (idx, (eps_prime, eps_double_prime, label)) in test_points_data.iter().enumerate() {
        let (_, _, s11, s21) = ground_truth[idx];
        println!("  {}: ε' = {:.4}, ε'' = {:.4} -> S11 = ({:+.6}, {:+.6}), S21 = ({:+.6}, {:+.6})",
                 label, eps_prime, eps_double_prime, s11.re, s11.im, s21.re, s21.im);
    }

    // ========================================================================
    // SECTION 2: LOAD ALL TRAINED MODELS
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 2: LOADING TRAINED MODELS");
    println!("{}", "=".repeat(80));

    // Model variants with their configurations
    let base_path = Path::new("artifacts");
    let model_configs: Vec<(&str, String, ModelsModelType)> = vec![
        // RVNN models (Real-valued)
        ("RVNN - Base", base_path.join("rvnn_base/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Real),
        ("RVNN - PI", base_path.join("rvnn_pi/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Real),
        ("RVNN - Noise", base_path.join("rvnn_noise/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Real),
        ("RVNN - Both", base_path.join("rvnn_both/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Real),
        // CVNN models (Complex-valued)
        ("CVNN - Base", base_path.join("cvnn_base/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Complex),
        ("CVNN - PI", base_path.join("cvnn_pi/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Complex),
        ("CVNN - Noise", base_path.join("cvnn_noise/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Complex),
        ("CVNN - Both", base_path.join("cvnn_both/model.safetensors").to_string_lossy().into_owned(), ModelsModelType::Complex),
    ];

    let mut loaded_models: Vec<(String, MlpModel, ModelsModelType)> = Vec::new();

    for (name, path, model_type) in &model_configs {
        let model_path = Path::new(&path);
        if !model_path.exists() {
            println!("  [SKIP] {}: Model not found at {}", name, path);
            continue;
        }

        // Build model with appropriate architecture
        let model_config = match model_type {
            ModelsModelType::Complex => ModelConfig::Complex {
                hidden_size: 48,
                complex_activation: ComplexActivation::from_name("worelu").unwrap(),
                dropout: 0.0,
                norm: ComplexNormChoice::None,
            },
            ModelsModelType::Real => ModelConfig::Real {
                hidden_size: 64,
                activation: Activation::from_name("gelu").unwrap(),
                dropout: 0.0,
                norm: String::from("layernorm"),
            },
        };
        
        let (model, mut varmap) = build_model(&model_config, DType::F64)?;
        load_model_checkpoint(&mut varmap, model_path)?;
        println!("  [LOADED] {}", name);
        loaded_models.push((name.to_string(), model, *model_type));
    }

    if loaded_models.is_empty() {
        return Err("No models loaded. Please run train_all_models.sh first.".into());
    }

    println!("\nLoaded {} models.", loaded_models.len());

    // ========================================================================
    // SECTION 3: PREPARE EVALUATION DATA WITH SCALING
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 3: PREPARING EVALUATION DATA WITH SCALING");
    println!("{}", "=".repeat(80));

    // Prepare evaluation data for each model
    // This fits scalers on training data and scales test inputs
    let mut model_prepared_data: Vec<(String, ModelsModelType, PreparedEvaluationData)> = Vec::new();

    for (model_name, _model, model_type) in &loaded_models {
        println!("\nPreparing data for {} ({:?})...", model_name, model_type);
        
        let prepared = prepare_evaluation_data(
            &test_samples,
            &train_samples,
            *model_type,
            DType::F64,
            false,  // negate_real_imag
            None,   // pre_fitted scalers
        )?;
        
        println!("  Features shape: {:?}", prepared.features.dims());
        println!("  Targets shape: {:?}", prepared.targets.dims());
        println!("  Feature scaler fitted: {}", prepared.feature_scaler.is_fitted());
        println!("  Target scaler fitted: {}", prepared.target_scaler.is_fitted());
        
        model_prepared_data.push((model_name.clone(), *model_type, prepared));
    }

    // ========================================================================
    // SECTION 4: MODEL PREDICTIONS WITH SCALING AND COMPARISON
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 4: MODEL PREDICTIONS WITH SCALING vs GROUND TRUTH");
    println!("{}", "=".repeat(80));

    let mut all_results: Vec<PredictionResult> = Vec::new();

    for (model_name, _model_type, prepared) in &model_prepared_data {
        // Find the corresponding model
        let model = loaded_models.iter()
            .find(|(name, _, _)| *name == *model_name)
            .map(|(_, model, _)| model)
            .unwrap();
        
        // Run model predictions with scaling
        // evaluate_model_predictions applies:
        // 1. Feature scaling (input normalization)
        // 2. Model forward pass
        // 3. Target inverse scaling (output denormalization)
        let evaluation = evaluate_model_predictions(
            model,
            &prepared.features,
            &prepared.targets,
            &prepared.feature_scaler,
            &prepared.target_scaler,
            prepared.encoding,
        )?;
        
        println!("\n--- {} ---", model_name);
        println!("{:<5} | {:<20} | {:<12} | {:<12} | {:<12}", 
                 "#", "Point", "ε' pred", "ε'' pred", "Error");
        println!("{}", "-".repeat(70));

        // Get predictions (already descaled by evaluate_model_predictions)
        let pred_real = evaluation.artifacts.pred_real;
        let pred_imag = evaluation.artifacts.pred_imag;
        
        for (idx, (eps_prime_true, eps_double_prime_true, _s11, _s21)) in ground_truth.iter().enumerate() {
            let eps_prime_pred = pred_real[idx];
            let eps_double_prime_pred = pred_imag[idx];

            // Compute errors
            let error_prime = (eps_prime_pred - eps_prime_true).abs();
            let error_double_prime = (eps_double_prime_pred - eps_double_prime_true).abs();
            let total_error = (error_prime.powi(2) + error_double_prime.powi(2)).sqrt();

            // Store results
            all_results.push(PredictionResult {
                model_name: model_name.clone(),
                point_idx: idx,
                eps_prime_true: *eps_prime_true,
                eps_double_prime_true: *eps_double_prime_true,
                eps_prime_pred,
                eps_double_prime_pred,
                error_prime,
                error_double_prime,
            });

            let label = test_points_data[idx].2;
            println!("{:<5} | {:<20} | {:<12.6} | {:<12.6} | {:<12.6e}", 
                     idx+1, label, eps_prime_pred, eps_double_prime_pred, total_error);
        }
    }

    // ========================================================================
    // SECTION 5: COMPREHENSIVE ERROR ANALYSIS
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 5: COMPREHENSIVE ERROR ANALYSIS");
    println!("{}", "=".repeat(80));

    // Group results by model
    let mut model_errors: HashMap<String, Vec<&PredictionResult>> = HashMap::new();
    for result in &all_results {
        model_errors.entry(result.model_name.clone()).or_default().push(result);
    }

    // Calculate statistics per model
    println!("\n--- ERROR STATISTICS BY MODEL ---\n");
    println!("{:<25} | {:<12} | {:<12} | {:<12} | {:<12}", 
             "Model", "Mean ε'", "Mean ε''", "Max ε'", "Max ε''");
    println!("{}", "-".repeat(85));

    for (model_name, results) in &model_errors {
        let n = results.len() as f64;
        let sum_prime: f64 = results.iter().map(|r| r.error_prime).sum();
        let sum_double_prime: f64 = results.iter().map(|r| r.error_double_prime).sum();
        let max_prime = results.iter().map(|r| r.error_prime).fold(0.0_f64, f64::max);
        let max_double_prime = results.iter().map(|r| r.error_double_prime).fold(0.0_f64, f64::max);
        
        let mean_prime = sum_prime / n;
        let mean_double_prime = sum_double_prime / n;

        println!("{:<25} | {:<12.6e} | {:<12.6e} | {:<12.6e} | {:<12.6e}", 
                 model_name, mean_prime, mean_double_prime, max_prime, max_double_prime);
    }

    // ========================================================================
    // SECTION 6: RESONANCE AND VACUUM REGION ANALYSIS
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 6: CRITICAL REGION ANALYSIS");
    println!("{}", "=".repeat(80));

    // Identify resonance and vacuum points
    let resonance_indices: Vec<usize> = test_points_data.iter().enumerate()
        .filter(|(_, (eps_prime, _, _))| *eps_prime >= 148.8 && *eps_prime <= 149.2)
        .map(|(idx, _)| idx)
        .collect();
    
    let vacuum_indices: Vec<usize> = test_points_data.iter().enumerate()
        .filter(|(_, (eps_prime, _, _))| *eps_prime >= 1.0 && *eps_prime <= 1.01)
        .map(|(idx, _)| idx)
        .collect();

    println!("\n--- RESONANCE REGION (er' = [148.8, 149.2]) ---");
    for model_name in model_errors.keys() {
        let results: Vec<_> = model_errors[model_name].iter()
            .filter(|r| resonance_indices.contains(&r.point_idx))
            .collect();
        
        if results.is_empty() { continue; }
        
        let n = results.len() as f64;
        let sum_prime: f64 = results.iter().map(|r| r.error_prime).sum();
        let mean_error = sum_prime / n;
        let max_error = results.iter().map(|r| r.error_prime).fold(0.0_f64, f64::max);
        
        println!("  {}: Mean ε' error = {:.6e}, Max ε' error = {:.6e}", 
                 model_name, mean_error, max_error);
    }

    println!("\n--- VACUUM REGION (er' = [1.0, 1.01]) ---");
    for model_name in model_errors.keys() {
        let results: Vec<_> = model_errors[model_name].iter()
            .filter(|r| vacuum_indices.contains(&r.point_idx))
            .collect();
        
        if results.is_empty() { continue; }
        
        let n = results.len() as f64;
        let sum_prime: f64 = results.iter().map(|r| r.error_prime).sum();
        let mean_error = sum_prime / n;
        let max_error = results.iter().map(|r| r.error_prime).fold(0.0_f64, f64::max);
        
        println!("  {}: Mean ε' error = {:.6e}, Max ε' error = {:.6e}", 
                 model_name, mean_error, max_error);
    }

    // ========================================================================
    // SECTION 7: NRW METHOD COMPARISON
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 7: METHODOLOGY");
    println!("{}", "=".repeat(80));

    println!("\nNRW Direct Method (Forward): ε -> S-parameters");
    println!("  - Uses: sparam_physics::nrw::nrw_direct_scalar_non_magnetic()");
    println!("  - This is the GROUND TRUTH for S-parameters");
    println!("\nTrained Models (Inverse): S-parameters -> ε");
    println!("  - CVNN: Complex-valued neural network (2c-48c-1c)");
    println!("  - RVNN: Real-valued neural network (4-64-2)");
    println!("  - Variants: base, +PI (physics-informed), +noise (input noise), +both");
    println!("\nScaling:");
    println!("  - Input: StandardScaler fit on training data, applied to test data");
    println!("  - Output: StandardScaler inverse transform applied to predictions");
    println!("\nComparison Method:");
    println!("  1. Generate S-parameters from ε using NRW forward");
    println!("  2. Create PermittivitySample structs");
    println!("  3. Use prepare_evaluation_data to fit scalers on training data");
    println!("  4. Run models on SCALED inputs");
    println!("  5. evaluate_model_predictions applies INVERSE scaling to outputs");
    println!("  6. Compare predicted ε with original ε");

    // ========================================================================
    // SECTION 8: DETAILED COMPARISON TABLE
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 8: DETAILED COMPARISON TABLE");
    println!("{}", "=".repeat(80));

    println!("\nFormat: Model | Point | ε' true | ε'' true | ε' pred | ε'' pred | |Δε'| | |Δε''|");
    println!("{}", "-".repeat(100));

    for result in &all_results {
        println!("{:<25} | {:<5} | {:<9.4} | {:<9.4} | {:<9.4} | {:<9.4} | {:<9.6e} | {:<9.6e}",
                 result.model_name, 
                 result.point_idx + 1,
                 result.eps_prime_true,
                 result.eps_double_prime_true,
                 result.eps_prime_pred,
                 result.eps_double_prime_pred,
                 result.error_prime,
                 result.error_double_prime);
    }

    // ========================================================================
    // SECTION 9: FINAL SUMMARY
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SECTION 9: FINAL SUMMARY");
    println!("{}", "=".repeat(80));

    println!("\n✓ Computed S-parameters for {} test points using NRW direct method", test_samples.len());
    println!("✓ Loaded {} trained models with proper scaling", loaded_models.len());
    println!("✓ Collected {} predictions for comparison", all_results.len());
    println!("\nKey findings:");
    println!("  - Models predict ε from S-parameters (inverse problem)");
    println!("  - NRW direct provides ground truth S-parameters (forward problem)");
    println!("  - Proper scaling applied: inputs normalized, outputs denormalized");
    println!("  - Errors represent |predicted ε - original ε| for each test point");
    println!("  - Resonance region (er' ~ 149) may be challenging if outside training range");
    println!("  - Vacuum region (er' ~ 1) should be well-covered if in training data");

    Ok(())
}
