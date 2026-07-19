//! Model construction helpers for canonical workflow entry points.
//!
//! Just a thin VarMap/device wrapper around [`ModelConfig::build_model`]
//! — the Real vs Complex dispatch lives on `ModelConfig` itself.

use candle_core::{DType, Result};
use candle_nn::{VarBuilder, VarMap};

use sparam_models::MlpModel;

use crate::workflows::config::ModelConfig;

pub fn build_model(model_config: &ModelConfig, dtype: DType) -> Result<(MlpModel, VarMap)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &candle_core::Device::Cpu);
    let model = model_config.build_model(vb.pp("model"))?;
    Ok((model, varmap))
}
