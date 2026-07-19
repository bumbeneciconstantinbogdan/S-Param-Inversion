//! Real-valued multilayer perceptron (MLP) regressor.
//!
//! [`MLPRegressor`] is the standard fully-connected architecture used for
//! S-parameter inversion.  Supports configurable hidden sizes, activation,
//! dropout, and optional layer/batch normalization.

use std::fmt;

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Linear, ModuleT, VarBuilder,
    batch_norm, layer_norm, linear,
};
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use sparam_core::config::normalize_config_name;
use sparam_core::error::candle_msg;

/// LayerNorm supporting F32 and F64 (Candle's built-in doesn't support F64).
///
/// Uses only differentiable Candle ops — a manual `Tensor::from_vec` fast path
/// was removed because it produced a leaf tensor that silently broke backprop.
#[derive(Clone, Debug)]
pub(crate) struct FusedLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl FusedLayerNorm {
    pub(crate) fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(hidden_size, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Self { weight, bias, eps: 1e-5 })
    }

    /// LayerNorm over the last dim. Keep differentiable — see struct docs.
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match xs.dtype() {
            DType::F32 | DType::F64 => {}
            dt => {
                return Err(candle_core::Error::Msg(format!(
                    "FusedLayerNorm: unsupported dtype {dt:?}"
                )));
            }
        }
        let hidden_size = xs.dim(candle_core::D::Minus1)? as f64;
        let mean = (xs.sum_keepdim(candle_core::D::Minus1)? / hidden_size)?;
        let x_centered = xs.broadcast_sub(&mean)?;
        let var = (x_centered.sqr()?.sum_keepdim(candle_core::D::Minus1)? / hidden_size)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}


/// Supported normalization layers for the hidden representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Normalization {
    #[default]
    #[serde(rename = "none")]
    None,
    #[serde(rename = "batchnorm", alias = "batch_norm", alias = "batch")]
    BatchNorm,
    #[serde(rename = "layernorm", alias = "layer_norm", alias = "layer")]
    LayerNorm,
    #[serde(
        rename = "fused_layernorm",
        alias = "fused_layer_norm",
        alias = "fused_layer",
        alias = "fused"
    )]
    FusedLayerNorm,
}

impl Normalization {
    /// Parse a normalization configuration string.
    pub fn from_name(name: &str) -> Result<Self> {
        let normalized = normalize_config_name(name);
        match normalized.as_str() {
            "" | "none" => Ok(Self::None),
            "batch" | "batchnorm" | "batch_norm" => Ok(Self::BatchNorm),
            "layer" | "layernorm" | "layer_norm" => Ok(Self::LayerNorm),
            "fused" | "fused_layer" | "fused_layernorm" | "fused_layer_norm" => {
                Ok(Self::FusedLayerNorm)
            }
            _ => Err(candle_msg(format!("unknown normalization: {name}"))),
        }
    }

    /// Number of learnable normalization parameters for the hidden width.
    #[must_use]
    pub fn parameter_count(self, hidden_size: usize) -> usize {
        match self {
            Self::None => 0,
            Self::BatchNorm | Self::LayerNorm | Self::FusedLayerNorm => hidden_size * 2,
        }
    }
}

impl fmt::Display for Normalization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::BatchNorm => write!(f, "batchnorm"),
            Self::LayerNorm => write!(f, "layernorm"),
            Self::FusedLayerNorm => write!(f, "fused_layernorm"),
        }
    }
}

impl std::str::FromStr for Normalization {
    type Err = candle_core::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_name(s)
    }
}

/// Configuration for a single-hidden-layer real-valued MLP regressor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MLPConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    #[serde(default)]
    pub activation: Activation,
    #[serde(default)]
    pub dropout_p: f32,
    #[serde(default)]
    pub norm: Normalization,
}

impl MLPConfig {
    /// Build a config with explicit sizes and activation.
    #[must_use]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            activation,
            dropout_p: 0.0,
            norm: Normalization::None,
        }
    }

    /// Build the standard 4-H-2 regressor config for permittivity prediction.
    #[must_use]
    pub fn permittivity(hidden_size: usize, activation: Activation) -> Self {
        Self::new(4, hidden_size, 2, activation)
    }

    /// Build the standard 4-H-4 regressor config for magneto-dielectric prediction.
    #[must_use]
    pub fn magneto_dielectric(hidden_size: usize, activation: Activation) -> Self {
        Self::new(4, hidden_size, 4, activation)
    }

    /// Override the dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, dropout_p: f32) -> Self {
        self.dropout_p = dropout_p;
        self
    }

    /// Override the hidden normalization mode.
    #[must_use]
    pub fn with_norm(mut self, norm: Normalization) -> Self {
        self.norm = norm;
        self
    }

    /// Count learnable parameters using the theoretical MLP formula.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        (self.input_size * self.hidden_size + self.hidden_size)
            + (self.hidden_size * self.output_size + self.output_size)
            + self.norm.parameter_count(self.hidden_size)
    }

    fn validate(&self) -> Result<()> {
        sparam_core::validation::validate_positive_usize("MLPConfig input_size", self.input_size)?;
        sparam_core::validation::validate_positive_usize("MLPConfig hidden_size", self.hidden_size)?;
        sparam_core::validation::validate_positive_usize("MLPConfig output_size", self.output_size)?;
        if !(0.0..1.0).contains(&self.dropout_p) {
            return Err(candle_msg(format!(
                "MLPConfig dropout_p must be in [0, 1), got {}",
                self.dropout_p
            )));
        }

        Ok(())
    }
}

impl fmt::Display for MLPConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MLP({}->{}->{}, {}, norm={}, dropout={})",
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.activation,
            self.norm,
            self.dropout_p
        )
    }
}

#[derive(Clone, Debug)]
enum NormalizationLayer {
    None,
    BatchNorm(BatchNorm),
    LayerNorm(LayerNorm),
    FusedLayerNorm(FusedLayerNorm),
}

impl NormalizationLayer {
    fn new(kind: Normalization, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        match kind {
            Normalization::None => Ok(Self::None),
            Normalization::BatchNorm => Ok(Self::BatchNorm(batch_norm(
                hidden_size,
                BatchNormConfig::default(),
                vb,
            )?)),
            Normalization::LayerNorm if vb.dtype() == DType::F64 => {
                Ok(Self::FusedLayerNorm(FusedLayerNorm::new(hidden_size, vb)?))
            }
            Normalization::LayerNorm => Ok(Self::LayerNorm(layer_norm(
                hidden_size,
                LayerNormConfig::default(),
                vb,
            )?)),
            Normalization::FusedLayerNorm => {
                Ok(Self::FusedLayerNorm(FusedLayerNorm::new(hidden_size, vb)?))
            }
        }
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        match self {
            Self::None => Ok(xs.clone()),
            Self::BatchNorm(layer) => layer.forward_t(xs, train),
            Self::LayerNorm(layer) => layer.forward(xs),
            Self::FusedLayerNorm(layer) => layer.forward(xs),
        }
    }
}

/// Seeded drop-in replacement for `candle_nn::Dropout` — draws its mask
/// from our seeded ChaCha8 so the forward pass stays reproducible. Same
/// inverted-dropout scaling (`* 1/(1-p)` on kept elements).
#[derive(Clone, Debug)]
struct SeededDropout {
    drop_p: f32,
}

impl SeededDropout {
    fn new(drop_p: f32) -> Self {
        Self { drop_p }
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if !train || self.drop_p == 0.0 {
            return Ok(xs.clone());
        }
        let mask = sparam_core::rng::seeded_rand(
            xs.shape(), 0.0, 1.0, xs.dtype(), xs.device(),
        )?
        .ge(self.drop_p as f64)?
        .to_dtype(xs.dtype())?;
        let scale = 1.0 / (1.0 - self.drop_p as f64);
        xs.mul(&mask)?.affine(scale, 0.0)
    }
}

/// A configurable single-hidden-layer MLP regressor.
///
/// The layer order matches the Python reference implementation:
/// `Linear -> [Norm] -> Activation -> [Dropout] -> Linear`.
#[derive(Clone, Debug)]
pub struct MLPRegressor {
    input_layer: Linear,
    normalization: NormalizationLayer,
    activation: Activation,
    dropout: Option<SeededDropout>,
    output_layer: Linear,
    config: MLPConfig,
}

impl MLPRegressor {
    /// Construct a regressor from a Candle [`VarBuilder`] and typed config.
    pub fn new(vb: VarBuilder, config: &MLPConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            input_layer: linear(config.input_size, config.hidden_size, vb.pp("feature.0"))?,
            normalization: NormalizationLayer::new(
                config.norm,
                config.hidden_size,
                vb.pp("feature.1"),
            )?,
            activation: config.activation,
            dropout: (config.dropout_p > 0.0).then(|| SeededDropout::new(config.dropout_p)),
            output_layer: linear(config.hidden_size, config.output_size, vb.pp("out"))?,
            config: config.clone(),
        })
    }

    /// Run the model in training or inference mode.
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.validate_input(xs)?;

        let hidden = self.input_layer.forward(xs)?;
        let hidden = self.normalization.forward_t(&hidden, train)?;
        let hidden = self.activation.forward(&hidden)?;
        let hidden = match &self.dropout {
            Some(dropout) => dropout.forward_t(&hidden, train)?,
            None => hidden,
        };

        self.output_layer.forward(&hidden)
    }

    /// Run the model in inference mode.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, false)
    }

    /// Access the validated configuration used to build the network.
    #[must_use]
    pub fn config(&self) -> &MLPConfig {
        &self.config
    }

    /// Count learnable parameters using the stored configuration.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.config.parameter_count()
    }

    /// Per-batch shape check. Gated behind `debug_assertions` so
    /// release builds skip it — the cost is ~5 ns per forward (two
    /// `usize` compares on metadata) but it compounds over an HPO
    /// campaign. Shape bugs in release still surface via Candle's
    /// internal errors from the first matmul or norm kernel.
    fn validate_input(&self, xs: &Tensor) -> Result<()> {
        #[cfg(debug_assertions)]
        {
            let dims = xs.dims();
            if dims.len() != 2 {
                return Err(candle_msg(format!(
                    "MLPRegressor expects a 2D input tensor shaped (batch_size, {}), got {:?}",
                    self.config.input_size, dims
                )));
            }
            if dims[1] != self.config.input_size {
                return Err(candle_msg(format!(
                    "MLPRegressor expected {} input features, got {}",
                    self.config.input_size, dims[1]
                )));
            }
        }
        let _ = xs;
        Ok(())
    }
}

impl ModuleT for MLPRegressor {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        MLPRegressor::forward_t(self, xs, train)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device};
    use candle_nn::{ModuleT, VarBuilder, VarMap};

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_tensor_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                assert_close(*actual_value, *expected_value);
            }
        }
    }

    fn assert_tensor_close_f32(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                assert!(
                    (actual_value - expected_value).abs() <= tolerance,
                    "expected {expected_value}, got {actual_value}"
                );
            }
        }
    }

    fn model_from_varmap(config: &MLPConfig) -> Result<MLPRegressor> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        MLPRegressor::new(vb, config)
    }

    fn tensor_map(entries: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
        entries
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect()
    }

    #[test]
    fn test_normalization_aliases_and_display_are_config_friendly() -> Result<()> {
        assert_eq!(Normalization::default(), Normalization::None);
        assert_eq!(Normalization::from_name("none")?, Normalization::None);
        assert_eq!(Normalization::from_name("batch")?, Normalization::BatchNorm);
        assert_eq!(
            Normalization::from_name("batch_norm")?,
            Normalization::BatchNorm
        );
        assert_eq!(
            Normalization::from_name("layernorm")?,
            Normalization::LayerNorm
        );
        assert_eq!(Normalization::from_name("layer")?, Normalization::LayerNorm);
        assert_eq!(
            Normalization::from_name("fused_layer")?,
            Normalization::FusedLayerNorm
        );
        assert_eq!(Normalization::BatchNorm.to_string(), "batchnorm");
        assert_eq!(Normalization::LayerNorm.to_string(), "layernorm");
        assert_eq!(Normalization::FusedLayerNorm.to_string(), "fused_layernorm");

        let error = Normalization::from_name("groupnorm")
            .err()
            .expect("unknown normalization should be rejected");
        assert!(error.to_string().contains("unknown normalization"));

        Ok(())
    }

    #[test]
    fn test_mlp_config_display_and_permittivity_constructor_are_log_friendly() {
        let config = MLPConfig::permittivity(32, Activation::ReLU)
            .with_norm(Normalization::LayerNorm)
            .with_dropout(0.25);

        assert_eq!(config.input_size, 4);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.output_size, 2);
        assert_eq!(config.activation, Activation::ReLU);
        assert_eq!(config.norm, Normalization::LayerNorm);
        assert_eq!(config.dropout_p, 0.25);
        assert_eq!(
            config.to_string(),
            "MLP(4->32->2, relu, norm=layernorm, dropout=0.25)"
        );
    }

    #[test]
    fn test_mlp_magneto_dielectric_constructor_is_explicit_and_reuses_mlp_shape() {
        let config = MLPConfig::magneto_dielectric(32, Activation::GELU);

        assert_eq!(config.input_size, 4);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.output_size, 4);
        assert_eq!(config.activation, Activation::GELU);
        assert_eq!(config.norm, Normalization::None);
        assert_eq!(config.dropout_p, 0.0);
        assert_eq!(config.parameter_count(), 292);
        assert_eq!(
            config.to_string(),
            "MLP(4->32->4, gelu, norm=none, dropout=0)"
        );
    }

    #[test]
    fn test_mlp_forward_produces_expected_output_shape() -> Result<()> {
        let config = MLPConfig::new(4, 32, 2, Activation::GELU);
        let model = model_from_varmap(&config)?;
        let xs = Tensor::zeros((5, 4), DType::F64, &Device::Cpu)?;

        let ys = model.forward(&xs)?;

        assert_eq!(ys.dims(), &[5, 2]);
        assert_eq!(model.config(), &config);
        assert_eq!(model.parameter_count(), 226);

        Ok(())
    }

    #[test]
    fn test_mlp_magneto_dielectric_forward_produces_expected_output_shape() -> Result<()> {
        let config = MLPConfig::magneto_dielectric(32, Activation::GELU);
        let model = model_from_varmap(&config)?;
        let xs = Tensor::zeros((5, 4), DType::F64, &Device::Cpu)?;

        let ys = model.forward(&xs)?;

        assert_eq!(ys.dims(), &[5, 4]);
        assert_eq!(model.config(), &config);
        assert_eq!(model.parameter_count(), 292);

        Ok(())
    }

    #[test]
    fn test_mlp_parameter_count_matches_theoretical_formula_for_norm_variants() -> Result<()> {
        let plain = MLPConfig::new(4, 32, 2, Activation::ReLU);
        let batch_norm =
            MLPConfig::new(4, 32, 2, Activation::ReLU).with_norm(Normalization::BatchNorm);
        let layer_norm =
            MLPConfig::new(4, 64, 2, Activation::ReLU).with_norm(Normalization::LayerNorm);
        let fused_layer_norm =
            MLPConfig::new(4, 64, 2, Activation::ReLU).with_norm(Normalization::FusedLayerNorm);

        assert_eq!(plain.parameter_count(), 226);
        assert_eq!(batch_norm.parameter_count(), 290);
        assert_eq!(layer_norm.parameter_count(), 578);
        assert_eq!(fused_layer_norm.parameter_count(), 578);

        let batch_norm_model = model_from_varmap(&batch_norm)?;
        let layer_norm_model = model_from_varmap(&layer_norm)?;
        let fused_layer_norm_model = model_from_varmap(&fused_layer_norm)?;

        assert_eq!(batch_norm_model.parameter_count(), 290);
        assert_eq!(layer_norm_model.parameter_count(), 578);
        assert_eq!(fused_layer_norm_model.parameter_count(), 578);

        Ok(())
    }

    #[test]
    fn test_mlp_parameter_count_matches_actual_varmap_elem_sum() -> Result<()> {
        let norms = [
            Normalization::None,
            Normalization::LayerNorm,
            Normalization::FusedLayerNorm,
        ];
        let hidden_sizes = [8_usize, 16, 32, 64];

        for &h in &hidden_sizes {
            for &norm in &norms {
                let cfg = MLPConfig::permittivity(h, Activation::ReLU).with_norm(norm);
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
                let _model = MLPRegressor::new(vb, &cfg)?;
                let actual: usize = varmap
                    .all_vars()
                    .iter()
                    .map(|v| v.as_tensor().elem_count())
                    .sum();
                assert_eq!(
                    cfg.parameter_count(),
                    actual,
                    "param-count mismatch: H={h} norm={norm:?} \
                     formula={} actual={actual}",
                    cfg.parameter_count(),
                );
            }
        }
        Ok(())
    }

    fn build_layer_norm_reference_model(norm: Normalization) -> Result<(MLPRegressor, Tensor)> {
        let config = MLPConfig::new(4, 3, 2, Activation::SiLU).with_norm(norm);
        let tensors = tensor_map(vec![
            (
                "feature.0.weight",
                Tensor::from_vec(
                    vec![
                        0.15f32, -0.25, 0.35, 0.45, -0.55, 0.65, 0.05, -0.15, 0.75, -0.05, -0.45,
                        0.25,
                    ],
                    (3, 4),
                    &Device::Cpu,
                )?,
            ),
            (
                "feature.0.bias",
                Tensor::from_vec(vec![0.02f32, -0.08, 0.12], 3, &Device::Cpu)?,
            ),
            (
                "feature.1.weight",
                Tensor::from_vec(vec![1.2f32, 0.85, 1.05], 3, &Device::Cpu)?,
            ),
            (
                "feature.1.bias",
                Tensor::from_vec(vec![-0.03f32, 0.04, 0.09], 3, &Device::Cpu)?,
            ),
            (
                "out.weight",
                Tensor::from_vec(
                    vec![0.45f32, -0.35, 0.25, -0.2, 0.55, 0.15],
                    (2, 3),
                    &Device::Cpu,
                )?,
            ),
            (
                "out.bias",
                Tensor::from_vec(vec![0.03f32, -0.01], 2, &Device::Cpu)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let model = MLPRegressor::new(vb, &config)?;
        let xs = Tensor::from_vec(
            vec![0.6f32, -0.1, 0.4, 0.9, -0.7, 0.2, 0.8, -0.4],
            (2, 4),
            &Device::Cpu,
        )?;

        Ok((model, xs))
    }

    #[test]
    fn test_mlp_rejects_invalid_configs_and_input_shapes() -> Result<()> {
        let zero_hidden = MLPRegressor::new(
            VarBuilder::zeros(DType::F64, &Device::Cpu),
            &MLPConfig::new(4, 0, 2, Activation::ReLU),
        )
        .err()
        .expect("zero hidden size should be rejected");
        let bad_dropout = MLPRegressor::new(
            VarBuilder::zeros(DType::F64, &Device::Cpu),
            &MLPConfig::new(4, 8, 2, Activation::ReLU).with_dropout(1.0),
        )
        .err()
        .expect("dropout=1 should be rejected");

        assert!(zero_hidden.to_string().contains("hidden_size"));
        assert!(bad_dropout.to_string().contains("dropout_p"));

        let model = model_from_varmap(&MLPConfig::new(4, 8, 2, Activation::ReLU))?;
        let wrong_rank = Tensor::zeros(4, DType::F64, &Device::Cpu)?;
        let wrong_feature_count = Tensor::zeros((3, 5), DType::F64, &Device::Cpu)?;

        let rank_error = model
            .forward(&wrong_rank)
            .err()
            .expect("rank-1 inputs should be rejected");
        let feature_error = model
            .forward(&wrong_feature_count)
            .err()
            .expect("wrong feature count should be rejected");

        assert!(rank_error.to_string().contains("2D input tensor"));
        assert!(
            feature_error
                .to_string()
                .contains("expected 4 input features")
        );

        Ok(())
    }

    #[test]
    fn test_mlp_dropout_is_only_applied_in_train_mode() -> Result<()> {
        let config = MLPConfig::new(4, 16, 2, Activation::ReLU).with_dropout(0.5);
        let model = model_from_varmap(&config)?;
        let xs = Tensor::ones((8, 4), DType::F64, &Device::Cpu)?;

        let eval_a = model.forward_t(&xs, false)?.to_vec2::<f64>()?;
        let eval_b = model.forward_t(&xs, false)?.to_vec2::<f64>()?;
        let train = ModuleT::forward_t(&model, &xs, true)?;

        assert_eq!(eval_a, eval_b);
        assert_eq!(train.dims(), &[8, 2]);

        Ok(())
    }

    #[test]
    fn test_mlp_forward_matches_pytorch_reference_without_normalization() -> Result<()> {
        let config = MLPConfig::new(4, 3, 2, Activation::GELU);
        let tensors = tensor_map(vec![
            (
                "feature.0.weight",
                Tensor::from_vec(
                    vec![
                        0.2f64, -0.1, 0.3, 0.5, -0.4, 0.6, 0.1, -0.2, 0.7, 0.2, -0.5, 0.4,
                    ],
                    (3, 4),
                    &Device::Cpu,
                )?,
            ),
            (
                "feature.0.bias",
                Tensor::from_vec(vec![0.1f64, -0.2, 0.05], 3, &Device::Cpu)?,
            ),
            (
                "out.weight",
                Tensor::from_vec(
                    vec![0.5f64, -0.3, 0.2, -0.1, 0.4, 0.6],
                    (2, 3),
                    &Device::Cpu,
                )?,
            ),
            (
                "out.bias",
                Tensor::from_vec(vec![0.05f64, -0.02], 2, &Device::Cpu)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let model = MLPRegressor::new(vb, &config)?;
        let xs = Tensor::from_vec(
            vec![0.25f64, -0.5, 1.0, 0.75, -1.2, 0.3, 0.5, -0.8],
            (2, 4),
            &Device::Cpu,
        )?;

        let ys = model.forward(&xs)?.to_vec2::<f64>()?;

        assert_tensor_close(
            &ys,
            &vec![
                vec![0.447_259_089_394_963_7, -0.179_000_961_452_549_87],
                vec![-0.196_451_890_716_233_74, 0.119_276_872_665_307_34],
            ],
        );

        Ok(())
    }

    #[test]
    fn test_mlp_batch_norm_eval_matches_pytorch_reference() -> Result<()> {
        let config = MLPConfig::new(4, 3, 2, Activation::ELU { alpha: 1.0 })
            .with_norm(Normalization::BatchNorm);
        let tensors = tensor_map(vec![
            (
                "feature.0.weight",
                Tensor::from_vec(
                    vec![
                        0.3f64, -0.4, 0.2, 0.1, -0.5, 0.2, 0.6, -0.3, 0.7, 0.1, -0.2, 0.4,
                    ],
                    (3, 4),
                    &Device::Cpu,
                )?,
            ),
            (
                "feature.0.bias",
                Tensor::from_vec(vec![0.05f64, -0.15, 0.2], 3, &Device::Cpu)?,
            ),
            (
                "feature.1.running_mean",
                Tensor::from_vec(vec![0.1f64, -0.2, 0.3], 3, &Device::Cpu)?,
            ),
            (
                "feature.1.running_var",
                Tensor::from_vec(vec![1.5f64, 0.5, 2.0], 3, &Device::Cpu)?,
            ),
            (
                "feature.1.weight",
                Tensor::from_vec(vec![1.1f64, 0.9, 1.05], 3, &Device::Cpu)?,
            ),
            (
                "feature.1.bias",
                Tensor::from_vec(vec![-0.05f64, 0.02, 0.1], 3, &Device::Cpu)?,
            ),
            (
                "out.weight",
                Tensor::from_vec(
                    vec![0.4f64, -0.2, 0.3, -0.6, 0.5, 0.1],
                    (2, 3),
                    &Device::Cpu,
                )?,
            ),
            (
                "out.bias",
                Tensor::from_vec(vec![0.01f64, -0.03], 2, &Device::Cpu)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let model = MLPRegressor::new(vb, &config)?;
        let xs = Tensor::from_vec(
            vec![0.4f64, -0.2, 0.8, 0.1, -0.5, 0.7, 0.3, -0.9],
            (2, 4),
            &Device::Cpu,
        )?;

        let ys = model.forward_t(&xs, false)?.to_vec2::<f64>()?;

        assert_tensor_close(
            &ys,
            &vec![
                vec![0.073_687_325_679_607_04, 0.015_987_670_167_776_494],
                vec![-0.496_828_353_277_005, 0.746_407_376_217_971],
            ],
        );

        Ok(())
    }

    #[test]
    fn test_mlp_layer_norm_eval_matches_pytorch_reference() -> Result<()> {
        let (model, xs) = build_layer_norm_reference_model(Normalization::LayerNorm)?;

        let ys = model.forward(&xs)?.to_vec2::<f32>()?;

        assert_tensor_close_f32(
            &ys,
            &vec![
                vec![0.539_804_16, -0.206_598_62],
                vec![-0.260_345_37, 0.337_872_48],
            ],
            1e-6,
        );

        Ok(())
    }

    #[test]
    fn test_mlp_fused_layer_norm_eval_matches_pytorch_reference() -> Result<()> {
        let (model, xs) = build_layer_norm_reference_model(Normalization::FusedLayerNorm)?;

        let ys = model.forward(&xs)?.to_vec2::<f32>()?;

        assert_tensor_close_f32(
            &ys,
            &vec![
                vec![0.539_804_16, -0.206_598_62],
                vec![-0.260_345_37, 0.337_872_48],
            ],
            1e-6,
        );

        Ok(())
    }

    /// Asserts gradients flow through `FusedLayerNorm` to the input Linear.
    /// Guards against reintroducing a `Tensor::from_vec` leaf-tensor fast path.
    #[test]
    fn fused_layer_norm_propagates_gradients_through_input_layer() -> Result<()> {
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);

        let cfg = MLPConfig::new(4, 8, 2, Activation::ReLU)
            .with_norm(Normalization::FusedLayerNorm);
        let model = MLPRegressor::new(vb, &cfg)?;

        let xs = Tensor::randn(0f64, 1.0, (16, 4), &device)?;
        let ys = Tensor::randn(0f64, 1.0, (16, 2), &device)?;
        let pred = model.forward_t(&xs, true)?;
        let loss = pred.sub(&ys)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;

        let input_weight = {
            let guard = varmap.data().lock().expect("poisoned");
            guard
                .iter()
                .find(|(k, _)| k.ends_with("feature.0.weight"))
                .map(|(_, v)| v.clone())
                .expect("input Linear weight must exist")
        };
        let grad = grads
            .get(input_weight.as_tensor())
            .expect("input Linear weight must receive a gradient through LayerNorm");
        let grad_flat = grad.flatten_all()?.to_vec1::<f64>()?;
        let grad_norm: f64 = grad_flat.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            grad_norm > 0.0 && grad_norm.is_finite(),
            "input weight gradient norm must be positive and finite, got {grad_norm}"
        );
        Ok(())
    }
}
