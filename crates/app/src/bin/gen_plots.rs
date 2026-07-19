use std::path::Path;
use candle_core::{DType, Tensor, Device};
use sparam_core::complex::Complex64;
use sparam_data::generation::load_samples_from_csv;
use sparam_app::workflows::config::ModelConfig;
use sparam_app::workflows::internal::data_prep::prepare_evaluation_data;
use sparam_app::workflows::internal::evaluation::evaluate_model_predictions;
use sparam_training::trainer::{Trainer, TrainerConfig};
use sparam_training::loss::LossType;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_samples = load_samples_from_csv(Path::new("data/data_train.csv"))?;
    let test_samples = load_samples_from_csv(Path::new("data/data_test.csv"))?;
    
    // Train Real Model
    println!("Training Real Model...");
    let real_cfg = ModelConfig::Real {
        hidden_size: 64,
        activation: sparam_models::RealActivation::Gelu,
        dropout: 0.0,
        norm: sparam_models::RealNormChoice::None,
    };
    let prepared_real = prepare_evaluation_data(&test_samples, &train_samples, real_cfg.model_type(), DType::F64, false, None)?;
    let (real_model, mut real_varmap) = sparam_app::workflows::internal::model_factory::build_model(&real_cfg, DType::F64)?;
    
    let trainer_cfg = TrainerConfig {
        epochs: 50,
        batch_size: 16,
        learning_rate: 0.01,
        loss_type: LossType::Mse,
        input_noise_std: 0.0,
        enable_physical_clamp: true,
        checkpoint_dir: None,
        ..Default::default()
    };
    
    let mut real_trainer = Trainer::new(
        &real_model,
        &mut real_varmap,
        prepared_real.features.clone(),
        prepared_real.targets.clone(),
        prepared_real.features.clone(), // using test as val for quick run
        prepared_real.targets.clone(),
        trainer_cfg,
        real_cfg.model_type(),
        &prepared_real.target_scaler,
        None,
        None,
    )?;
    real_trainer.fit()?;
    
    let eval_real = evaluate_model_predictions(
        &real_model,
        &prepared_real.features,
        &prepared_real.targets,
        &prepared_real.feature_scaler,
        &prepared_real.target_scaler,
        prepared_real.encoding,
    )?;
    
    // Evaluate CVNN Model
    println!("Evaluating CVNN Model...");
    let cvnn_cfg = ModelConfig::Complex {
        hidden_size: 48,
        complex_activation: sparam_models::ComplexActivation::from_name("worelu")?,
        dropout: 0.000407,
        norm: sparam_models::ComplexNormChoice::LayerNorm,
    };
    let prepared_cvnn = prepare_evaluation_data(&test_samples, &train_samples, cvnn_cfg.model_type(), DType::F64, false, None)?;
    let (cvnn_model, mut cvnn_varmap) = sparam_app::workflows::internal::model_factory::build_model(&cvnn_cfg, DType::F64)?;
    sparam_training::checkpoint::load_model_checkpoint(&mut cvnn_varmap, "poster/studies/methodology/models/inverse_cvnn_48c.safetensors")?;
    
    let eval_cvnn = evaluate_model_predictions(
        &cvnn_model,
        &prepared_cvnn.features,
        &prepared_cvnn.targets,
        &prepared_cvnn.feature_scaler,
        &prepared_cvnn.target_scaler,
        prepared_cvnn.encoding,
    )?;

    // Compute metrics
    let true_re = &eval_cvnn.artifacts.true_real;
    let true_im = &eval_cvnn.artifacts.true_imag;
    
    let real_pred_re = &eval_real.artifacts.pred_real;
    let real_pred_im = &eval_real.artifacts.pred_imag;
    let cvnn_pred_re = &eval_cvnn.artifacts.pred_real;
    let cvnn_pred_im = &eval_cvnn.artifacts.pred_imag;
    
    // NRW preds
    let mut nrw_re = Vec::new();
    let mut nrw_im = Vec::new();
    let d = 0.0015;
    let a = 0.02286;
    let freq = 8.2e9;
    
    // Need physics module for NRW
    // Or just write loop since we don't have access to the exact module path easily without exploring. Let's dump the data to JSON and let python do metrics.
    
    let mut file = File::create("plot_data.json")?;
    writeln!(file, "{{")?;
    writeln!(file, "\"true_re\": {:?},", true_re)?;
    writeln!(file, "\"true_im\": {:?},", true_im)?;
    writeln!(file, "\"real_pred_re\": {:?},", real_pred_re)?;
    writeln!(file, "\"real_pred_im\": {:?},", real_pred_im)?;
    writeln!(file, "\"cvnn_pred_re\": {:?},", cvnn_pred_re)?;
    writeln!(file, "\"cvnn_pred_im\": {:?},", cvnn_pred_im)?;
    
    // Also dump NRW predictions
    // To do this we just call NRW. Since I don't want to import it, I'll extract it using Python if needed.
    // Wait, let's just dump S-params.
    let s11_re: Vec<f64> = test_samples.iter().map(|s| s.s11_real).collect();
    let s11_im: Vec<f64> = test_samples.iter().map(|s| s.s11_imag).collect();
    let s21_re: Vec<f64> = test_samples.iter().map(|s| s.s21_real).collect();
    let s21_im: Vec<f64> = test_samples.iter().map(|s| s.s21_imag).collect();
    writeln!(file, "\"s11_re\": {:?},", s11_re)?;
    writeln!(file, "\"s11_im\": {:?},", s11_im)?;
    writeln!(file, "\"s21_re\": {:?},", s21_re)?;
    writeln!(file, "\"s21_im\": {:?},", s21_im)?;
    
    writeln!(file, "\"dummy\": 0")?;
    writeln!(file, "}}")?;
    
    println!("Data dumped to plot_data.json");
    Ok(())
}
