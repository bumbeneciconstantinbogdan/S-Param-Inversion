use std::path::Path;
use candle_core::{DType, Device};
use sparam_app::workflows::config::ModelConfig;
use sparam_app::workflows::internal::data_prep::prepare_evaluation_data;
use sparam_app::workflows::internal::evaluation::evaluate_model_predictions;
use sparam_data::generation::load_samples_from_csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading datasets...");
    let train_samples = load_samples_from_csv(Path::new("data/data_train.csv"))?;
    let test_samples = load_samples_from_csv(Path::new("data/data_test.csv"))?;

    println!("Evaluating CVNN...");
    let cvnn_cfg = ModelConfig::Complex {
        hidden_size: 48,
        complex_activation: sparam_models::ComplexActivation::from_name("worelu")?,
        dropout: 0.0004,
        norm: sparam_models::ComplexNormChoice::None, // we used default for cvnn which has no norm config or what? Wait, the cvnn default is None?
    };
    // CVNN evaluation
    let (cvnn_model, mut cvnn_varmap) = sparam_app::workflows::internal::model_factory::build_model(&cvnn_cfg, DType::F64)?;
    sparam_training::checkpoint::load_model_checkpoint(&mut cvnn_varmap, "artifacts/best_cvnn_run/model.safetensors")?;
    let prep_cvnn = prepare_evaluation_data(&test_samples, &train_samples, cvnn_cfg.model_type(), DType::F64, false, None)?;
    let eval_cvnn = evaluate_model_predictions(
        &cvnn_model, &prep_cvnn.features, &prep_cvnn.targets, &prep_cvnn.feature_scaler, &prep_cvnn.target_scaler, prep_cvnn.encoding,
    )?;

    println!("Evaluating RVNN...");
    let rvnn_cfg = ModelConfig::Real {
        hidden_size: 64,
        activation: sparam_models::RealActivation::Gelu,
        dropout: 0.0,
        norm: sparam_models::RealNormChoice::LayerNorm, // default
    };
    let (rvnn_model, mut rvnn_varmap) = sparam_app::workflows::internal::model_factory::build_model(&rvnn_cfg, DType::F64)?;
    sparam_training::checkpoint::load_model_checkpoint(&mut rvnn_varmap, "artifacts/best_real_run/model.safetensors")?;
    let prep_rvnn = prepare_evaluation_data(&test_samples, &train_samples, rvnn_cfg.model_type(), DType::F64, false, None)?;
    let eval_rvnn = evaluate_model_predictions(
        &rvnn_model, &prep_rvnn.features, &prep_rvnn.targets, &prep_rvnn.feature_scaler, &prep_rvnn.target_scaler, prep_rvnn.encoding,
    )?;

    let mut file = std::fs::File::create("plot_comparison_data.json").unwrap();
    use std::io::Write;
    writeln!(file, "{{").unwrap();
    writeln!(file, "\"true_re\": {:?},", eval_cvnn.artifacts.true_real).unwrap();
    writeln!(file, "\"true_im\": {:?},", eval_cvnn.artifacts.true_imag).unwrap();
    writeln!(file, "\"cvnn_pred_re\": {:?},", eval_cvnn.artifacts.pred_real).unwrap();
    writeln!(file, "\"cvnn_pred_im\": {:?},", eval_cvnn.artifacts.pred_imag).unwrap();
    writeln!(file, "\"rvnn_pred_re\": {:?},", eval_rvnn.artifacts.pred_real).unwrap();
    writeln!(file, "\"rvnn_pred_im\": {:?},", eval_rvnn.artifacts.pred_imag).unwrap();
    writeln!(file, "\"dummy\": 0\n}}").unwrap();
    
    Ok(())
}
