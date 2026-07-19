//! Custom evaluation binary to compute separate error metrics for epsilon' and epsilon'',
//! and perform a concrete numerical stability test using Monte Carlo noise injection.

use std::path::Path;
use candle_core::{DType, Tensor, Device};
use candle_nn::ModuleT;
use sparam_core::complex::Complex64;
use sparam_core::constants::{EPSILON_0, MU_0};
use sparam_data::generation::load_samples_from_csv;
use sparam_training::checkpoint::load_model_checkpoint;
use sparam_data::scaling::{Scaler, ScalerRef};
use sparam_data::physical_constraint::apply_physical_softplus_clamp;
use sparam_app::workflows::config::ModelConfig;
use sparam_app::workflows::internal::data_prep::prepare_evaluation_data;
use sparam_app::workflows::internal::evaluation::evaluate_model_predictions;

/// A lightweight deterministic PRNG using XORShift64
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 42 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

fn rand_gaussian(rng: &mut SimpleRng, std_dev: f64) -> f64 {
    let mut u1 = rng.next_f64();
    let mut u2 = rng.next_f64();
    // Guard against 0.0 for log
    if u1 == 0.0 { u1 = 1e-30; }
    if u2 == 0.0 { u2 = 1e-30; }
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    z0 * std_dev
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Separated Error Evaluation & Stability Test Workflow...\n");

    // 1. Load data
    let train_path = Path::new("data/data_train.csv");
    let test_path = Path::new("data/data_test.csv");
    if !train_path.exists() || !test_path.exists() {
        return Err("Train or test data not found. Please run generate-data first.".into());
    }

    println!("Loading datasets...");
    let train_samples = load_samples_from_csv(train_path)?;
    let test_samples = load_samples_from_csv(test_path)?;
    println!("Loaded {} train samples, {} test samples.", train_samples.len(), test_samples.len());

    // 2. Build and load RVNN model
    println!("\nLoading RVNN model...");
    let rvnn_config = ModelConfig::Real {
        hidden_size: 64,
        activation: sparam_models::Activation::GELU,
        dropout: 0.0,
        norm: String::from("layernorm"),
    };

    let (rvnn_model, mut rvnn_varmap) = sparam_app::workflows::internal::model_factory::build_model(&rvnn_config, DType::F64)?;
    let binding1 = std::env::var("RVNN_PATH").unwrap_or_else(|_| "artifacts/best_real_run/model.safetensors".to_string()); let rvnn_checkpoint_path = binding1.as_str();
    if !Path::new(rvnn_checkpoint_path).exists() {
        return Err(format!("RVNN checkpoint not found at {}", rvnn_checkpoint_path).into());
    }
    load_model_checkpoint(&mut rvnn_varmap, rvnn_checkpoint_path)?;
    println!("RVNN model successfully loaded from {}", rvnn_checkpoint_path);

    // 3. Prepare features, targets, and scalers for RVNN
    println!("Preparing and scaling evaluation data for RVNN...");
    let rvnn_prepared = prepare_evaluation_data(
        &test_samples,
        &train_samples,
        rvnn_config.model_type(),
        DType::F64,
        false,
        None,
    )?;

    // 4. Run RVNN evaluation to extract raw predictions
    println!("Running RVNN predictions on test grid... (using fitted scalers)");
    let rvnn_evaluation = evaluate_model_predictions(
        &rvnn_model,
        &rvnn_prepared.features,
        &rvnn_prepared.targets,
        &rvnn_prepared.feature_scaler,
        &rvnn_prepared.target_scaler,
        rvnn_prepared.encoding,
    )?;

    let true_real = &rvnn_evaluation.artifacts.true_real;
    let true_imag = &rvnn_evaluation.artifacts.true_imag;
    let rvnn_pred_real = &rvnn_evaluation.artifacts.pred_real;
    let rvnn_pred_imag = &rvnn_evaluation.artifacts.pred_imag;
    let n_samples = true_real.len();

    // 5. Build and load CVNN model
    println!("\nLoading CVNN model...");
    let cvnn_config = ModelConfig::Complex {
        hidden_size: 48,
        complex_activation: sparam_models::ComplexActivation::from_name("worelu")?,
        dropout: 0.0,
        norm: sparam_models::ComplexNormChoice::LayerNorm,
    };

    let (cvnn_model, mut cvnn_varmap) = sparam_app::workflows::internal::model_factory::build_model(&cvnn_config, DType::F64)?;
    let binding2 = std::env::var("CVNN_PATH").unwrap_or_else(|_| "artifacts/best_cvnn_run/model.safetensors".to_string()); let cvnn_checkpoint_path = binding2.as_str();
    if !Path::new(cvnn_checkpoint_path).exists() {
        return Err(format!("CVNN checkpoint not found at {}", cvnn_checkpoint_path).into());
    }
    load_model_checkpoint(&mut cvnn_varmap, cvnn_checkpoint_path)?;
    println!("CVNN model successfully loaded from {}", cvnn_checkpoint_path);

    // 6. Prepare features, targets, and scalers for CVNN
    println!("Preparing and scaling evaluation data for CVNN...");
    let cvnn_prepared = prepare_evaluation_data(
        &test_samples,
        &train_samples,
        cvnn_config.model_type(),
        DType::F64,
        false,
        None,
    )?;

    // 7. Run CVNN evaluation to extract raw predictions
    println!("Running CVNN predictions on test grid... (using fitted scalers)");
    let cvnn_evaluation = evaluate_model_predictions(
        &cvnn_model,
        &cvnn_prepared.features,
        &cvnn_prepared.targets,
        &cvnn_prepared.feature_scaler,
        &cvnn_prepared.target_scaler,
        cvnn_prepared.encoding,
    )?;

    let cvnn_pred_real = &cvnn_evaluation.artifacts.pred_real;
    let cvnn_pred_imag = &cvnn_evaluation.artifacts.pred_imag;

    // 8. Evaluate standard analytical NRW on same test grid
    println!("\nRunning analytical NRW inverse on test grid S-parameters...");
    let mut nrw_pred_real = Vec::with_capacity(n_samples);
    let mut nrw_pred_imag = Vec::with_capacity(n_samples);
    
    // Test geometry
    let d = 0.0015; // sample thickness in meters
    let a = 0.02286; // waveguide width in meters
    let freq = std::env::var("MC_FREQ").unwrap_or_else(|_| "8.2e9".to_string()).parse::<f64>().unwrap(); // frequency in Hz

    for sample in &test_samples {
        let s11 = Complex64::new(sample.s11_real, sample.s11_imag);
        let s21 = Complex64::new(sample.s21_real, sample.s21_imag);
        let (eps_r, _) = nrw_inverse_reference_single(d, a, freq, s11, s21);
        nrw_pred_real.push(eps_r.re);
        // NRW returns negative imaginary part for physical lossy samples, flip sign to positive convention
        nrw_pred_imag.push(-eps_r.im);
    }

    // 9. Compute separate error metrics
    println!("\n==================================================");
    println!("         SEPARATE ACCURACY METRICS REPORT");
    println!("==================================================");

    println!("\n--- 1. Real Permittivity (\u{03b5}_r') Relative Error ---");
    report_relative_errors("RVNN", true_real, rvnn_pred_real);
    report_relative_errors("CVNN", true_real, cvnn_pred_real);
    report_relative_errors("NRW ", true_real, &nrw_pred_real);

    println!("\n--- 2. Imaginary Permittivity (\u{03b5}_r'') Relative Error (for \u{03b5}_r'' > 0) ---");
    // Filter out sample points where true eps'' = 0 (about 1% of uniform grid is 0)
    let mut true_imag_nonzero = Vec::new();
    let mut rvnn_pred_imag_nonzero = Vec::new();
    let mut cvnn_pred_imag_nonzero = Vec::new();
    for ((&t, &p_rvnn), &p_cvnn) in true_imag.iter().zip(rvnn_pred_imag.iter()).zip(cvnn_pred_imag.iter()) {
        if t > 0.0 {
            true_imag_nonzero.push(t);
            rvnn_pred_imag_nonzero.push(p_rvnn);
            cvnn_pred_imag_nonzero.push(p_cvnn);
        }
    }

    let mut nrw_imag_nonzero = Vec::new();
    for (&t, &p) in true_imag.iter().zip(nrw_pred_imag.iter()) {
        if t > 0.0 {
            nrw_imag_nonzero.push(p);
        }
    }
    
    report_relative_errors("RVNN", &true_imag_nonzero, &rvnn_pred_imag_nonzero);
    report_relative_errors("CVNN", &true_imag_nonzero, &cvnn_pred_imag_nonzero);
    report_relative_errors("NRW ", &true_imag_nonzero, &nrw_imag_nonzero);

    println!("\n--- 3. Imaginary Permittivity (\u{03b5}_r'') Absolute Error (all samples) ---");
    report_absolute_errors("RVNN", true_imag, rvnn_pred_imag);
    report_absolute_errors("CVNN", true_imag, cvnn_pred_imag);
    report_absolute_errors("NRW ", true_imag, &nrw_pred_imag);

    println!("\nDumping RVNN data to plot_rvnn_data.json...");
    let mut file = std::fs::File::create("plot_rvnn_data.json").unwrap();
    use std::io::Write;
    writeln!(file, "{{")?;
    writeln!(file, "\"true_re\": {:?},", true_real)?;
    writeln!(file, "\"true_im\": {:?},", true_imag)?;
    writeln!(file, "\"rvnn_pred_re\": {:?},", rvnn_pred_real)?;
    writeln!(file, "\"rvnn_pred_im\": {:?},", rvnn_pred_imag)?;
    writeln!(file, "\"dummy\": 0\n}}")?;
    
    println!("Dumping CVNN data to plot_cvnn_data.json...");
    let mut file = std::fs::File::create("plot_cvnn_data.json").unwrap();
    writeln!(file, "{{")?;
    writeln!(file, "\"true_re\": {:?},", true_real)?;
    writeln!(file, "\"true_im\": {:?},", true_imag)?;
    writeln!(file, "\"cvnn_pred_re\": {:?},", cvnn_pred_real)?;
    writeln!(file, "\"cvnn_pred_im\": {:?},", cvnn_pred_imag)?;
    writeln!(file, "\"nrw_pred_re\": {:?},", nrw_pred_real)?;
    writeln!(file, "\"nrw_pred_im\": {:?},", nrw_pred_imag)?;
    writeln!(file, "\"dummy\": 0\n}}")?;

    // 10. Concrete Numerical Stability Test (Monte Carlo Perturbation Analysis)
    println!("\n==================================================");
    println!("        CONCRETE NUMERICAL STABILITY TEST");
    println!("==================================================");
    println!("Material under test: FR-4 (nominal eps = 4.4 - j 0.08)");
    println!("Noise level: 40 dB SNR (Gaussian noise std = 0.01 added to S-params)");
    println!("Operating frequency: f = 8.2 GHz");
    println!("Waveguide geometry: WR-90 (a = 22.86 mm), d = 1.50 mm");
    println!("Number of Monte Carlo trials: 1000");

    let fr4_eps_prime = 4.4;
    let fr4_eps_double_prime = 0.08;

    // Direct S-parameters (noise-free)
    let (s11_true, s21_true) = nrw_direct_scalar_non_magnetic(d, a, freq, fr4_eps_prime, fr4_eps_double_prime);
    println!("\nNoise-free simulated S-parameters:");
    println!("  S11: {:.6} + j({:.6})", s11_true.re, s11_true.im);
    println!("  S21: {:.6} + j({:.6})", s21_true.re, s21_true.im);

    // Use CVNN's prepared data for scaling (both models use same test data)
    let prepared = &cvnn_prepared;

    let mut rng = SimpleRng::new(12345);
    let mut rvnn_preds_prime = Vec::with_capacity(1000);
    let mut rvnn_preds_double_prime = Vec::with_capacity(1000);
    let mut cvnn_preds_prime = Vec::with_capacity(1000);
    let mut cvnn_preds_double_prime = Vec::with_capacity(1000);
    let mut nrw_preds_prime = Vec::with_capacity(1000);
    let mut nrw_preds_double_prime = Vec::with_capacity(1000);

    for _ in 0..1000 {
        // Add 40 dB Gaussian noise to S11 and S21
        let s11_noise_re = rand_gaussian(&mut rng, 0.01);
        let s11_noise_im = rand_gaussian(&mut rng, 0.01);
        let s21_noise_re = rand_gaussian(&mut rng, 0.01);
        let s21_noise_im = rand_gaussian(&mut rng, 0.01);

        let s11_noisy = Complex64::new(s11_true.re + s11_noise_re, s11_true.im + s11_noise_im);
        let s21_noisy = Complex64::new(s21_true.re + s21_noise_re, s21_true.im + s21_noise_im);

        // A. NRW Inverse
        let (nrw_eps, _) = nrw_inverse_reference_single(d, a, freq, s11_noisy, s21_noisy);
        nrw_preds_prime.push(nrw_eps.re);
        nrw_preds_double_prime.push(-nrw_eps.im); // positive convention

        // B. RVNN Inference
        let raw_feat_rvnn = vec![s11_noisy.re, s21_noisy.re, s11_noisy.im, s21_noisy.im];
        let feat_tensor_rvnn = Tensor::new(raw_feat_rvnn.as_slice(), &Device::Cpu)?.reshape((1, 4))?;
        let scaled_feat_rvnn = rvnn_prepared.feature_scaler.transform(&feat_tensor_rvnn)?;
        let raw_pred_rvnn = rvnn_model.forward_t(&scaled_feat_rvnn, false)?;
        let clamped_pred_rvnn = apply_physical_softplus_clamp(&raw_pred_rvnn, ScalerRef::Standard(&rvnn_prepared.target_scaler), true)?;
        let pred_physical_rvnn = rvnn_prepared.target_scaler.inverse_transform(&clamped_pred_rvnn)?;
        let pred_flat_rvnn = pred_physical_rvnn.flatten_all()?.to_vec1::<f64>()?;
        rvnn_preds_prime.push(pred_flat_rvnn[0]);
        rvnn_preds_double_prime.push(-pred_flat_rvnn[1]); // positive convention

        // C. CVNN Inference
        let raw_feat_cvnn = vec![s11_noisy.re, s21_noisy.re, s11_noisy.im, s21_noisy.im];
        let feat_tensor_cvnn = Tensor::new(raw_feat_cvnn.as_slice(), &Device::Cpu)?.reshape((1, 4))?;
        let scaled_feat_cvnn = prepared.feature_scaler.transform(&feat_tensor_cvnn)?;
        let raw_pred_cvnn = cvnn_model.forward_t(&scaled_feat_cvnn, false)?;
        let clamped_pred_cvnn = apply_physical_softplus_clamp(&raw_pred_cvnn, ScalerRef::Standard(&prepared.target_scaler), true)?;
        let pred_physical_cvnn = prepared.target_scaler.inverse_transform(&clamped_pred_cvnn)?;
        let pred_flat_cvnn = pred_physical_cvnn.flatten_all()?.to_vec1::<f64>()?;
        cvnn_preds_prime.push(pred_flat_cvnn[0]);
        cvnn_preds_double_prime.push(-pred_flat_cvnn[1]); // positive convention
    }

    println!("\nStability Test Results:");
    println!("--------------------------------------------------");
    report_stability_results("NRW Baseline   ", fr4_eps_prime, fr4_eps_double_prime, &nrw_preds_prime, &nrw_preds_double_prime);
    println!();
    report_stability_results("RVNN Real-MLP  ", fr4_eps_prime, fr4_eps_double_prime, &rvnn_preds_prime, &rvnn_preds_double_prime);
    println!();
    report_stability_results("CVNN Complex-MLP", fr4_eps_prime, fr4_eps_double_prime, &cvnn_preds_prime, &cvnn_preds_double_prime);
    println!("--------------------------------------------------");

    // Save Monte Carlo results for plotting
    println!("\nSaving Monte Carlo results to mc_stability_results.json...");
    let mut mc_file = std::fs::File::create("mc_stability_results.json")?;
    writeln!(mc_file, "{{")?;
    writeln!(mc_file, "  \"n_trials\": 1000,")?;
    writeln!(mc_file, "  \"true_eps_prime\": {},", fr4_eps_prime)?;
    writeln!(mc_file, "  \"true_eps_double_prime\": {},", fr4_eps_double_prime)?;
    
    // NRW results
    let nrw_mean_p: f64 = nrw_preds_prime.iter().sum::<f64>() / nrw_preds_prime.len() as f64;
    let nrw_mean_pp: f64 = nrw_preds_double_prime.iter().sum::<f64>() / nrw_preds_double_prime.len() as f64;
    let nrw_std_p: f64 = (nrw_preds_prime.iter().map(|&x| (x - nrw_mean_p).powi(2)).sum::<f64>() / nrw_preds_prime.len() as f64).sqrt();
    let nrw_std_pp: f64 = (nrw_preds_double_prime.iter().map(|&x| (x - nrw_mean_pp).powi(2)).sum::<f64>() / nrw_preds_double_prime.len() as f64).sqrt();
    writeln!(mc_file, "  \"nrw_mean_prime\": {:.6},", nrw_mean_p)?;
    writeln!(mc_file, "  \"nrw_mean_double_prime\": {:.6},", nrw_mean_pp)?;
    writeln!(mc_file, "  \"nrw_std_prime\": {:.6},", nrw_std_p)?;
    writeln!(mc_file, "  \"nrw_std_double_prime\": {:.6},", nrw_std_pp)?;
    
    // RVNN results
    let rvnn_mean_p: f64 = rvnn_preds_prime.iter().sum::<f64>() / rvnn_preds_prime.len() as f64;
    let rvnn_mean_pp: f64 = rvnn_preds_double_prime.iter().sum::<f64>() / rvnn_preds_double_prime.len() as f64;
    let rvnn_std_p: f64 = (rvnn_preds_prime.iter().map(|&x| (x - rvnn_mean_p).powi(2)).sum::<f64>() / rvnn_preds_prime.len() as f64).sqrt();
    let rvnn_std_pp: f64 = (rvnn_preds_double_prime.iter().map(|&x| (x - rvnn_mean_pp).powi(2)).sum::<f64>() / rvnn_preds_double_prime.len() as f64).sqrt();
    writeln!(mc_file, "  \"rvnn_mean_prime\": {:.6},", rvnn_mean_p)?;
    writeln!(mc_file, "  \"rvnn_mean_double_prime\": {:.6},", rvnn_mean_pp)?;
    writeln!(mc_file, "  \"rvnn_std_prime\": {:.6},", rvnn_std_p)?;
    writeln!(mc_file, "  \"rvnn_std_double_prime\": {:.6},", rvnn_std_pp)?;
    
    // CVNN results
    let cvnn_mean_p: f64 = cvnn_preds_prime.iter().sum::<f64>() / cvnn_preds_prime.len() as f64;
    let cvnn_mean_pp: f64 = cvnn_preds_double_prime.iter().sum::<f64>() / cvnn_preds_double_prime.len() as f64;
    let cvnn_std_p: f64 = (cvnn_preds_prime.iter().map(|&x| (x - cvnn_mean_p).powi(2)).sum::<f64>() / cvnn_preds_prime.len() as f64).sqrt();
    let cvnn_std_pp: f64 = (cvnn_preds_double_prime.iter().map(|&x| (x - cvnn_mean_pp).powi(2)).sum::<f64>() / cvnn_preds_double_prime.len() as f64).sqrt();
    writeln!(mc_file, "  \"cvnn_mean_prime\": {:.6},", cvnn_mean_p)?;
    writeln!(mc_file, "  \"cvnn_mean_double_prime\": {:.6},", cvnn_mean_pp)?;
    writeln!(mc_file, "  \"cvnn_std_prime\": {:.6},", cvnn_std_p)?;
    writeln!(mc_file, "  \"cvnn_std_double_prime\": {:.6}", cvnn_std_pp)?;
    writeln!(mc_file, "}}")?;
    
    Ok(())
}

fn report_relative_errors(model_name: &str, true_vals: &[f64], pred_vals: &[f64]) {
    let errors: Vec<f64> = true_vals.iter().zip(pred_vals.iter())
        .map(|(&t, &p)| (p - t).abs() / t * 100.0)
        .collect();
    
    let n = errors.len() as f64;
    let mean = errors.iter().sum::<f64>() / n;
    
    let variance = errors.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    
    let max = errors.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
    
    let ok_1 = errors.iter().filter(|&&e| e <= 1.0).count() as f64 * 100.0 / n;
    let ok_10 = errors.iter().filter(|&&e| e <= 10.0).count() as f64 * 100.0 / n;
    
    println!("  {}: Mean={:.3}%, Std={:.3}%, Max={:.3}%, OK@1%={:.2}%, OK@10%={:.2}%", 
             model_name, mean, std, max, ok_1, ok_10);
}

fn report_absolute_errors(model_name: &str, true_vals: &[f64], pred_vals: &[f64]) {
    let errors: Vec<f64> = true_vals.iter().zip(pred_vals.iter())
        .map(|(&t, &p)| (p - t).abs())
        .collect();
    
    let n = errors.len() as f64;
    let mean = errors.iter().sum::<f64>() / n;
    
    let variance = errors.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    
    let max = errors.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
    
    let ok_0_1 = errors.iter().filter(|&&e| e <= 0.1).count() as f64 * 100.0 / n;
    let ok_0_5 = errors.iter().filter(|&&e| e <= 0.5).count() as f64 * 100.0 / n;
    let ok_1_0 = errors.iter().filter(|&&e| e <= 1.0).count() as f64 * 100.0 / n;
    
    println!("  {}: Mean={:.3}, Std={:.3}, Max={:.3}, OK@0.1={:.2}%, OK@0.5={:.2}%, OK@1.0={:.2}%", 
             model_name, mean, std, max, ok_0_1, ok_0_5, ok_1_0);
}

fn report_stability_results(name: &str, true_prime: f64, true_double_prime: f64, preds_prime: &[f64], preds_double_prime: &[f64]) {
    let n = preds_prime.len() as f64;
    
    let mean_p = preds_prime.iter().sum::<f64>() / n;
    let var_p = preds_prime.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>() / n;
    let std_p = var_p.sqrt();
    let max_err_p = preds_prime.iter().map(|&p| (p - true_prime).abs() / true_prime * 100.0).max_by(f64::total_cmp).unwrap_or(0.0);

    let mean_pp = preds_double_prime.iter().sum::<f64>() / n;
    let var_pp = preds_double_prime.iter().map(|&p| (p - mean_pp).powi(2)).sum::<f64>() / n;
    let std_pp = var_pp.sqrt();
    let max_err_pp = preds_double_prime.iter().map(|&p| (p - true_double_prime).abs() / true_double_prime * 100.0).max_by(f64::total_cmp).unwrap_or(0.0);

    println!("{}{} Results:", name, " ".repeat(20_usize.saturating_sub(name.len())));
    println!("  Real Part (eps'_r = {}):", true_prime);
    println!("    Extracted Mean: {:.4}", mean_p);
    println!("    Extracted Std : {:.4}", std_p);
    println!("    Worst-case Rel-Err: {:.2}%", max_err_p);
    println!("  Imaginary Part (eps\"_r = {}):", true_double_prime);
    println!("    Extracted Mean: {:.4}", mean_pp);
    println!("    Extracted Std : {:.4}", std_pp);
    println!("    Worst-case Rel-Err: {:.2}%", max_err_pp);
}

fn nrw_inverse_reference_single(
    d: f64,
    a: f64,
    frequency: f64,
    s11: Complex64,
    s21: Complex64,
) -> (Complex64, Complex64) {
    let one = Complex64::new(1.0, 0.0);
    let v1 = s21.add(s11);
    let v2 = s21.sub(s11);
    let x = one
        .sub(v1.mul(v2))
        .div(v1.sub(v2).add(Complex64::new(1e-16, 0.0)));
    let sqrt_term = x.mul(x).sub(one).sqrt();
    let gamma_plus = x.add(sqrt_term);
    let gamma = if gamma_plus.abs() > 1.0 {
        one.div(gamma_plus)
    } else {
        gamma_plus
    };
    let propagation = v1.sub(gamma).div(one.sub(gamma.mul(v1)));
    let beta1_s = Complex64::new(0.0, 0.0)
        .sub(propagation.ln())
        .div(Complex64::new(0.0, d));

    let omega = 2.0 * std::f64::consts::PI * frequency;
    let k0_sq = omega.powi(2) * EPSILON_0 * MU_0;
    let kt_sq = (std::f64::consts::PI / a).powi(2);
    let beta1_e = Complex64::new(k0_sq - kt_sq, 0.0).sqrt();
    let mu_r = one.add(gamma).div(one.sub(gamma)).mul(beta1_s).div(beta1_e);
    let eps_r = beta1_s
        .mul(beta1_s)
        .add(Complex64::new(kt_sq, 0.0))
        .div(Complex64::new(k0_sq, 0.0).mul(mu_r));

    (eps_r, mu_r)
}

fn nrw_direct_scalar_non_magnetic(
    d: f64,
    a: f64,
    frequency: f64,
    eps_prime: f64,
    eps_double_prime: f64,
) -> (Complex64, Complex64) {
    let one = Complex64::new(1.0, 0.0);
    let omega = 2.0 * std::f64::consts::PI * frequency;
    let k0_sq = omega.powi(2) * EPSILON_0 * MU_0;
    let kt_sq = (std::f64::consts::PI / a).powi(2);

    let beta1_e = Complex64::new(k0_sq - kt_sq, 0.0).sqrt();
    
    // eps = eps' - j * eps''
    let eps_r = Complex64::new(eps_prime, -eps_double_prime);
    let mu_r = one; // non-magnetic

    let beta1_s = eps_r
        .mul(mu_r)
        .mul(Complex64::new(k0_sq, 0.0))
        .sub(Complex64::new(kt_sq, 0.0))
        .sqrt();

    let impedance_ratio = mu_r.mul(beta1_e).div(beta1_s);
    let gamma = impedance_ratio.sub(one).div(impedance_ratio.add(one));

    let propagation_term = Complex64::new(0.0, -d).mul(beta1_s).exp();
    let propagation_term_sq = propagation_term.mul(propagation_term);

    let denom = one.sub(gamma.mul(gamma).mul(propagation_term_sq));
    let s11 = gamma.mul(one.sub(propagation_term_sq)).div(denom);
    let s21 = propagation_term.mul(one.sub(gamma.mul(gamma))).div(denom);

    (s11, s21)
}
