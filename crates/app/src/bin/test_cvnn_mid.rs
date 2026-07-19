
use std::path::Path;
use sparam_data::generation::PermittivitySample;
use sparam_models::{MlpModel, ModelType, Activation, ComplexActivation, ComplexNormChoice};
use sparam_data::tensor_bridges::samples_to_tensors;
use sparam_data::scaling::StandardScaler;
use sparam_data::generation::load_samples_from_csv;
use sparam_training::checkpoint::load_model_checkpoint;
use sparam_app::workflows::internal::model_factory::build_model;
use sparam_app::workflows::config::ModelConfig;
use candle_core::DType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = 0.0015;
    let a = 0.02286;
    let freq = 8.2e9;
    let eps_prime = 100.0;
    let eps_double_prime = 50.0;
    
    let (s11, s21) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
        d, a, freq, eps_prime, eps_double_prime
    );
    
    let sample = PermittivitySample {
        s11_real: s11.re,
        s11_imag: s11.im,
        s21_real: s21.re,
        s21_imag: s21.im,
        eps_prime,
        eps_double_prime,
        is_dense_patch: false,
    };
    
    let train_path = Path::new("data/data_train.csv");
    let train_samples = load_samples_from_csv(train_path)?;
    
    let model_type = ModelType::Complex;
    let train_ds = samples_to_tensors(model_type, &train_samples, DType::F64, false)?;
    
    let mut feature_scaler = StandardScaler::new();
    let mut target_scaler = StandardScaler::new();
    feature_scaler.fit(&train_ds.features)?;
    target_scaler.fit(&train_ds.targets)?;
    
    let test_ds = samples_to_tensors(model_type, &[sample], DType::F64, false)?;
    let scaled_features = feature_scaler.transform(&test_ds.features)?;
    
    let model_config = ModelConfig::Complex {
        hidden_size: 48,
        complex_activation: ComplexActivation::Worelu,
        dropout: 0.0,
        norm: ComplexNormChoice::None,
    };
    let (model, mut varmap) = build_model(&model_config, DType::F64)?;
    let model_path = Path::new("artifacts/cvnn_base/model.safetensors");
    load_model_checkpoint(&mut varmap, model_path)?;
    
    let output = model.forward_t(&scaled_features, false)?;
    let output_physical = target_scaler.inverse_transform(&output)?;
    let output_vec = output_physical.to_vec2::<f64>()?;
    
    println!("Input: eps'={}, eps'’={}", eps_prime, eps_double_prime);
    println!("Prediction (physical): eps'={:.4}, -eps'’={:.4}", output_vec[0][0], output_vec[0][1]);
    println!("After negation: eps'={:.4}, eps'’={:.4}", output_vec[0][0], -output_vec[0][1]);
    
    Ok(())
}
