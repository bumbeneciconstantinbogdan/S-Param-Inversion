//! Complex-valued (CVNN) MLP regressor.

use std::fmt;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use crate::complex_activation::{
    CSwishPhase, ComplexActivation, HybridCardioidGelu, Lpma, ModReLU, Worelu,
};
use crate::complex_linear::{ComplexInit, ComplexLinear};
use sparam_core::complex_tensor::ComplexTensor;
#[cfg(debug_assertions)]
use sparam_core::error::candle_msg;

/// `LayerNorm` runs per-half (γ/β shared across re/im halves).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComplexNormChoice {
    #[default]
    #[serde(rename = "none")]
    None,
    #[serde(rename = "layernorm", alias = "layer_norm", alias = "layer")]
    LayerNorm,
}

impl ComplexNormChoice {
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "" | "none" => Ok(Self::None),
            "layer" | "layernorm" | "layer_norm" => Ok(Self::LayerNorm),
            _ => Err(candle_core::Error::Msg(format!(
                "unknown complex normalization: {name}"
            ))),
        }
    }
}

/// `ComplexLinear(in→H) → [LN] → Activation → [dropout] → ComplexLinear(H→out)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexMLPConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    #[serde(default)]
    pub activation: ComplexActivation,
    /// Pair-wise dropout (drops whole `(re, im)` pairs).
    #[serde(default)]
    pub dropout_p: f32,
    #[serde(default)]
    pub norm: ComplexNormChoice,
}

impl ComplexMLPConfig {
    #[must_use]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        activation: ComplexActivation,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            activation,
            dropout_p: 0.0,
            norm: ComplexNormChoice::None,
        }
    }

    #[must_use]
    pub fn with_dropout(mut self, dropout_p: f32) -> Self {
        self.dropout_p = dropout_p;
        self
    }

    #[must_use]
    pub fn with_norm(mut self, norm: ComplexNormChoice) -> Self {
        self.norm = norm;
        self
    }

    /// Standard 2c-Hc-1c config: 2 S-params → 1 ε.
    #[must_use]
    pub fn permittivity(hidden_size: usize, activation: ComplexActivation) -> Self {
        Self::new(2, hidden_size, 1, activation)
    }

    /// Standard 2c-Hc-2c config: 2 S-params → (ε, μ).
    #[must_use]
    pub fn magneto_dielectric(hidden_size: usize, activation: ComplexActivation) -> Self {
        Self::new(2, hidden_size, 2, activation)
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let input_layer = 2 * (self.input_size * self.hidden_size + self.hidden_size);
        let output_layer = 2 * (self.hidden_size * self.output_size + self.output_size);
        let activation_params = match self.activation {
            ComplexActivation::CReLU
            | ComplexActivation::CGELU
            | ComplexActivation::Cardioid => 0,
            ComplexActivation::ModReLU => self.hidden_size,
            ComplexActivation::HybridCardioidGelu => 1,
            ComplexActivation::Lpma => self.hidden_size * 2,
            ComplexActivation::CSwishPhase => 1,
            ComplexActivation::Worelu => 3,
        };
        let norm_params = match self.norm {
            ComplexNormChoice::None => 0,
            ComplexNormChoice::LayerNorm => 4 * self.hidden_size,
        };
        input_layer + output_layer + activation_params + norm_params
    }

    fn validate(&self) -> Result<()> {
        fn validate_usize(name: &str, value: usize) -> candle_core::Result<()> {
            sparam_core::validation::validate_positive_usize(name, value)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        }
        validate_usize("ComplexMLPConfig input_size", self.input_size)?;
        validate_usize("ComplexMLPConfig hidden_size", self.hidden_size)?;
        validate_usize("ComplexMLPConfig output_size", self.output_size)?;
        Ok(())
    }
}

impl fmt::Display for ComplexMLPConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplexMLP({}c->{}c->{}c, {})",
            self.input_size, self.hidden_size, self.output_size, self.activation)
    }
}

/// One-to-one with `ComplexActivation`. Stateful variants own params.
#[derive(Clone, Debug)]
enum ActivationLayer {
    CReLU,
    CGELU,
    Cardioid,
    ModReLU(ModReLU),
    HybridCardioidGelu(HybridCardioidGelu),
    Lpma(Lpma),
    CSwishPhase(CSwishPhase),
    Worelu(Worelu),
}

impl ActivationLayer {
    fn new(choice: ComplexActivation, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        match choice {
            ComplexActivation::CReLU => Ok(Self::CReLU),
            ComplexActivation::CGELU => Ok(Self::CGELU),
            ComplexActivation::Cardioid => Ok(Self::Cardioid),
            ComplexActivation::ModReLU => {
                Ok(Self::ModReLU(ModReLU::new(hidden_size, vb)?))
            }
            ComplexActivation::HybridCardioidGelu => {
                Ok(Self::HybridCardioidGelu(HybridCardioidGelu::new(vb)?))
            }
            ComplexActivation::Lpma => Ok(Self::Lpma(Lpma::new(hidden_size, vb)?)),
            ComplexActivation::CSwishPhase => {
                Ok(Self::CSwishPhase(CSwishPhase::new(vb)?))
            }
            ComplexActivation::Worelu => Ok(Self::Worelu(Worelu::new(vb)?)),
        }
    }

    fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        match self {
            Self::CReLU => ComplexActivation::CReLU.forward(xs),
            Self::CGELU => ComplexActivation::CGELU.forward(xs),
            Self::Cardioid => ComplexActivation::Cardioid.forward(xs),
            Self::ModReLU(l) => l.forward(xs),
            Self::HybridCardioidGelu(l) => l.forward(xs),
            Self::Lpma(l) => l.forward(xs),
            Self::CSwishPhase(l) => l.forward(xs),
            Self::Worelu(l) => l.forward(xs),
        }
    }
}

/// Per-half LN with independent γ/β. Real-side `FusedLayerNorm` math.
#[derive(Clone, Debug)]
pub(crate) struct ComplexLayerNorm {
    weight_real: Tensor,
    bias_real: Tensor,
    weight_imag: Tensor,
    bias_imag: Tensor,
    eps: f64,
}

impl ComplexLayerNorm {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight_real =
            vb.get_with_hints(hidden_size, "weight_real", candle_nn::Init::Const(1.0))?;
        let bias_real =
            vb.get_with_hints(hidden_size, "bias_real", candle_nn::Init::Const(0.0))?;
        let weight_imag =
            vb.get_with_hints(hidden_size, "weight_imag", candle_nn::Init::Const(1.0))?;
        let bias_imag =
            vb.get_with_hints(hidden_size, "bias_imag", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            weight_real,
            bias_real,
            weight_imag,
            bias_imag,
            eps: 1e-5,
        })
    }

    fn ln_one(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let hidden_size = x.dim(candle_core::D::Minus1)? as f64;
        let mean = (x.sum_keepdim(candle_core::D::Minus1)? / hidden_size)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = (centered.sqr()?.sum_keepdim(candle_core::D::Minus1)? / hidden_size)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(weight)?.broadcast_add(bias)
    }

    fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        let real = self.ln_one(&xs.real, &self.weight_real, &self.bias_real)?;
        let imag = self.ln_one(&xs.imag, &self.weight_imag, &self.bias_imag)?;
        Ok(ComplexTensor::new_unchecked(real, imag))
    }
}

/// Pair-wise dropout: one mask drives both halves. ChaCha8-seeded so
/// the forward pass is bit-reproducible under a fixed seed.
#[derive(Clone, Debug)]
struct SeededComplexDropout {
    drop_p: f32,
}

impl SeededComplexDropout {
    fn new(drop_p: f32) -> Self {
        Self { drop_p }
    }

    fn forward_t(&self, xs: &ComplexTensor, train: bool) -> Result<ComplexTensor> {
        if !train || self.drop_p == 0.0 {
            return Ok(xs.clone());
        }
        let mask = sparam_core::rng::seeded_rand(
            xs.real.shape(), 0.0, 1.0, xs.real.dtype(), xs.real.device(),
        )?
        .ge(self.drop_p as f64)?
        .to_dtype(xs.real.dtype())?;
        let scale = 1.0 / (1.0 - self.drop_p as f64);
        let real = xs.real.mul(&mask)?.affine(scale, 0.0)?;
        let imag = xs.imag.mul(&mask)?.affine(scale, 0.0)?;
        Ok(ComplexTensor::new_unchecked(real, imag))
    }
}

#[derive(Clone, Debug)]
pub struct ComplexMLPRegressor {
    input_layer: ComplexLinear,
    norm: Option<ComplexLayerNorm>,
    activation: ActivationLayer,
    dropout: Option<SeededComplexDropout>,
    output_layer: ComplexLinear,
    config: ComplexMLPConfig,
}

impl ComplexMLPRegressor {
    /// Trabelsi-He on hidden, Glorot on output.
    pub fn new(vb: VarBuilder, config: &ComplexMLPConfig) -> Result<Self> {
        config.validate()?;

        let input_layer = ComplexLinear::new_with_init(
            config.input_size, config.hidden_size, ComplexInit::He, vb.pp("input"),
        )?;
        let norm = match config.norm {
            ComplexNormChoice::None => None,
            ComplexNormChoice::LayerNorm => {
                Some(ComplexLayerNorm::new(config.hidden_size, vb.pp("norm"))?)
            }
        };
        let activation = ActivationLayer::new(config.activation, config.hidden_size, vb.pp("act"))?;
        let output_layer = ComplexLinear::new_with_init(
            config.hidden_size, config.output_size, ComplexInit::Glorot, vb.pp("output"),
        )?;
        let dropout =
            (config.dropout_p > 0.0).then(|| SeededComplexDropout::new(config.dropout_p));

        Ok(Self {
            input_layer,
            norm,
            activation,
            dropout,
            output_layer,
            config: config.clone(),
        })
    }

    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        self.validate_input(xs)?;
        self.forward_t_unchecked(xs, false)
    }

    /// Half-half packed `[re_block | im_block]` of shape `(batch, 2·input_size)`.
    pub fn forward_packed(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        #[cfg(debug_assertions)]
        {
            let dims = xs.dims();
            if dims.len() != 2 {
                return Err(candle_msg(format!(
                    "ComplexMLPRegressor expects packed 2D input (batch, {}), got shape {:?}",
                    self.config.input_size * 2,
                    dims
                )));
            }
            let expected_scalar_features = self.config.input_size * 2;
            if dims[1] != expected_scalar_features {
                return Err(candle_msg(format!(
                    "ComplexMLPRegressor expected {expected_scalar_features} packed scalar features, got {}",
                    dims[1]
                )));
            }
        }

        let f = self.config.input_size;
        let real = xs.narrow(1, 0, f)?;
        let imag = xs.narrow(1, f, f)?;
        let ys = self.forward_t_unchecked(&ComplexTensor::new_unchecked(real, imag), train)?;
        Tensor::cat(&[&ys.real, &ys.imag], 1)
    }

    pub fn forward_t_unchecked(&self, xs: &ComplexTensor, train: bool) -> Result<ComplexTensor> {
        let hidden = self.input_layer.forward(xs)?;
        let hidden = match &self.norm {
            Some(norm) => norm.forward(&hidden)?,
            None => hidden,
        };
        let hidden = self.activation.forward(&hidden)?;
        let hidden = match &self.dropout {
            Some(dropout) => dropout.forward_t(&hidden, train)?,
            None => hidden,
        };
        self.output_layer.forward(&hidden)
    }

    /// Hot-path: skip shape validation.
    pub fn forward_unchecked(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        self.forward_t_unchecked(xs, false)
    }

    #[must_use]
    pub fn config(&self) -> &ComplexMLPConfig {
        &self.config
    }

    #[must_use]
    pub fn input_size(&self) -> usize {
        self.config.input_size
    }

    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[must_use]
    pub fn output_size(&self) -> usize {
        self.config.output_size
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.config.parameter_count()
    }

    fn validate_input(&self, xs: &ComplexTensor) -> Result<()> {
        #[cfg(debug_assertions)]
        {
            let dims = xs.real.dims();
            if dims.len() != 2 {
                return Err(candle_msg(format!(
                    "ComplexMLPRegressor expects 2D input (batch, {}), got shape {:?}",
                    self.config.input_size, dims
                )));
            }
            if dims[1] != self.config.input_size {
                return Err(candle_msg(format!(
                    "ComplexMLPRegressor expected {} complex input features, got {}",
                    self.config.input_size, dims[1]
                )));
            }
        }
        let _ = xs;
        Ok(())
    }
}

impl fmt::Display for ComplexMLPRegressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.config)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor, Var};
    use candle_nn::{VarBuilder, VarMap};

    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_tensor_close(values: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(values.len(), expected.len());
        for (a, e) in values.iter().zip(expected.iter()) {
            assert_close(*a, *e, tol);
        }
    }

    fn tensor_map(entries: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
        entries
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect()
    }

    #[test]
    fn complex_mlp_two_forward_backward_runs_match_exactly() {
        use sparam_core::complex_tensor::ComplexTensor;
        use sparam_core::rng::{set_global_seed, test_seed_lock};
        use sparam_core::determinism::deterministic_reinit_varmap;

        let _guard = test_seed_lock().lock().unwrap_or_else(|e| e.into_inner());
        let seed = 20260417u64;

        fn once(seed: u64) -> Vec<f64> {
            set_global_seed(seed);
            let dev = Device::Cpu;
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F64, &dev);
            // Use ModReLU (learnable) + dropout so the forward pass
            // exercises the most randomness-sensitive ops at once.
            let cfg = ComplexMLPConfig::permittivity(16, ComplexActivation::ModReLU);
            let model = ComplexMLPRegressor::new(vb, &cfg).unwrap();
            deterministic_reinit_varmap(&vm, seed).unwrap();
            set_global_seed(seed);

            let batch = 8;
            let xr = Tensor::arange(0.0_f64, batch as f64 * 2.0, &dev)
                .unwrap().reshape((batch, 2)).unwrap();
            let xi = xr.affine(-1.0, 0.5).unwrap();
            let x = ComplexTensor::new(xr, xi).unwrap();

            let y = model.forward_t_unchecked(&x, true).unwrap();

            let mut out = y.real.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            out.extend(y.imag.flatten_all().unwrap().to_vec1::<f64>().unwrap());
            out
        }

        let a = once(seed);
        let b = once(seed);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(), y.to_bits(),
                "complex MLP output element {i} diverged: {x} vs {y}"
            );
        }
    }

    #[test]
    fn complex_mlp_many_training_steps_match_exactly() {
        use candle_nn::Optimizer;
        use sparam_core::complex_tensor::ComplexTensor;
        use sparam_core::rng::{set_global_seed, test_seed_lock};
        use sparam_core::determinism::{
            clip_grad_norm_stable, deterministic_reinit_varmap, flatten_vars_f64,
            seeded_randn_like, stable_all_vars,
        };

        let _guard = test_seed_lock().lock().unwrap_or_else(|e| e.into_inner());
        let seed = 20260418u64;

        fn once(seed: u64) -> Vec<f64> {
            set_global_seed(seed);
            let dev = Device::Cpu;
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F64, &dev);
            let cfg = ComplexMLPConfig::permittivity(12, ComplexActivation::ModReLU);
            let model = ComplexMLPRegressor::new(vb, &cfg).unwrap();
            deterministic_reinit_varmap(&vm, seed).unwrap();
            set_global_seed(seed);

            let mut opt = candle_nn::AdamW::new_lr(vm.all_vars(), 1e-2).unwrap();

            let batch = 16;
            let n_steps = 40;

            for step in 0..n_steps {
                let xr = Tensor::arange(
                    step as f64 * 0.1, step as f64 * 0.1 + batch as f64 * 2.0, &dev,
                ).unwrap().reshape((batch, 2)).unwrap();
                let xi = xr.affine(0.5, -0.25).unwrap();
                let yr_vec: Vec<f64> = (0..batch).map(|i| i as f64 * 0.5 + step as f64 * 0.01).collect();
                let yr = Tensor::from_vec(yr_vec, (batch, 1), &dev).unwrap();
                let yi = yr.affine(-0.3, 0.05).unwrap();

                let xr_noise = seeded_randn_like(&xr, 0.0, 0.01).unwrap();
                let xi_noise = seeded_randn_like(&xi, 0.0, 0.01).unwrap();
                let xr_noisy = xr.add(&xr_noise).unwrap();
                let xi_noisy = xi.add(&xi_noise).unwrap();
                let x = ComplexTensor::new(xr_noisy, xi_noisy).unwrap();

                let pred = model.forward_t_unchecked(&x, true).unwrap();
                let target = ComplexTensor::new(yr, yi).unwrap();

                let dr = (&pred.real - &target.real).unwrap().sqr().unwrap().sum_all().unwrap();
                let di = (&pred.imag - &target.imag).unwrap().sqr().unwrap().sum_all().unwrap();
                let loss = (&dr + &di).unwrap();

                let mut grads = loss.backward().unwrap();
                let vars = stable_all_vars(&vm);
                clip_grad_norm_stable(&mut grads, &vars, 1.0).unwrap();
                opt.step(&grads).unwrap();
            }

            flatten_vars_f64(&vm)
        }

        let a = once(seed);
        let b = once(seed);
        assert_eq!(a.len(), b.len(), "parameter count drifted between runs");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(), y.to_bits(),
                "complex MLP parameter {i} diverged after training: {x} vs {y}"
            );
        }
    }

    #[test]
    fn complex_mlp_hpo_scale_training_matches_exactly() {
        // Larger matmul (batch=2048, hidden=24) crosses Accelerate's
        // dgemm parallelism threshold.
        use candle_nn::Optimizer;
        use sparam_core::complex_tensor::ComplexTensor;
        use sparam_core::rng::{set_global_seed, test_seed_lock};
        use sparam_core::determinism::{
            clip_grad_norm_stable, deterministic_reinit_varmap, flatten_vars_f64,
            seeded_randn_like, stable_all_vars,
        };

        let _guard = test_seed_lock().lock().unwrap_or_else(|e| e.into_inner());
        let seed = 20260419u64;

        fn once(seed: u64) -> Vec<f64> {
            set_global_seed(seed);
            let dev = Device::Cpu;
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F64, &dev);
            let cfg = ComplexMLPConfig::permittivity(24, ComplexActivation::ModReLU);
            let model = ComplexMLPRegressor::new(vb, &cfg).unwrap();
            deterministic_reinit_varmap(&vm, seed).unwrap();
            set_global_seed(seed);

            let mut opt = candle_nn::AdamW::new_lr(stable_all_vars(&vm), 1e-2).unwrap();

            let batch = 2048;
            let n_steps = 8;

            for step in 0..n_steps {
                let xr_vec: Vec<f64> =
                    (0..(batch * 2)).map(|i| (i as f64 + step as f64 * 0.01) * 0.001).collect();
                let xr = Tensor::from_vec(xr_vec, (batch, 2), &dev).unwrap();
                let xi = xr.affine(0.5, -0.25).unwrap();

                let yr_vec: Vec<f64> = (0..batch).map(|i| i as f64 * 0.001 + step as f64 * 0.01).collect();
                let yr = Tensor::from_vec(yr_vec, (batch, 1), &dev).unwrap();
                let yi = yr.affine(-0.3, 0.05).unwrap();

                let xr_noise = seeded_randn_like(&xr, 0.0, 0.01).unwrap();
                let xi_noise = seeded_randn_like(&xi, 0.0, 0.01).unwrap();
                let xr_noisy = xr.add(&xr_noise).unwrap();
                let xi_noisy = xi.add(&xi_noise).unwrap();
                let x = ComplexTensor::new(xr_noisy, xi_noisy).unwrap();

                let pred = model.forward_t_unchecked(&x, true).unwrap();
                let target = ComplexTensor::new(yr, yi).unwrap();
                let dr = (&pred.real - &target.real).unwrap().sqr().unwrap().sum_all().unwrap();
                let di = (&pred.imag - &target.imag).unwrap().sqr().unwrap().sum_all().unwrap();
                let loss = (&dr + &di).unwrap();

                let mut grads = loss.backward().unwrap();
                let vars = stable_all_vars(&vm);
                clip_grad_norm_stable(&mut grads, &vars, 1.0).unwrap();
                opt.step(&grads).unwrap();
            }

            flatten_vars_f64(&vm)
        }

        let a = once(seed);
        let b = once(seed);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(), y.to_bits(),
                "HPO-scale complex MLP parameter {i} diverged: {x} vs {y}"
            );
        }
    }

    #[test]
    fn test_config_permittivity_and_magneto_dielectric() {
        let cfg = ComplexMLPConfig::permittivity(32, ComplexActivation::default());
        assert_eq!(cfg.input_size, 2);
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.output_size, 1);

        let cfg_md = ComplexMLPConfig::magneto_dielectric(32, ComplexActivation::ModReLU);
        assert_eq!(cfg_md.input_size, 2);
        assert_eq!(cfg_md.output_size, 2);
    }

    #[test]
    fn test_config_parameter_count_formula_without_modrelu() {
        // P = 2(I*H + H) + 2(H*O + O) = 2(2*32 + 32) + 2(32*1 + 1) = 192 + 66 = 258
        let cfg = ComplexMLPConfig::permittivity(32, ComplexActivation::default());
        assert_eq!(cfg.parameter_count(), 258);

        // P = 2(2*16 + 16) + 2(16*1 + 1) = 96 + 34 = 130
        let cfg16 = ComplexMLPConfig::permittivity(16, ComplexActivation::default());
        assert_eq!(cfg16.parameter_count(), 130);

        // P = 2(2*64 + 64) + 2(64*1 + 1) = 384 + 130 = 514
        let cfg64 = ComplexMLPConfig::permittivity(64, ComplexActivation::default());
        assert_eq!(cfg64.parameter_count(), 514);
    }

    #[test]
    fn test_config_parameter_count_formula_with_modrelu() {
        // P = 258 + 32 = 290
        let cfg = ComplexMLPConfig::permittivity(32, ComplexActivation::ModReLU);
        assert_eq!(cfg.parameter_count(), 290);

        // P = 130 + 16 = 146
        let cfg16 = ComplexMLPConfig::permittivity(16, ComplexActivation::ModReLU);
        assert_eq!(cfg16.parameter_count(), 146);
    }

    #[test]
    fn test_config_magneto_dielectric_parameter_count() {
        // P = 2(2*32 + 32) + 2(32*2 + 2) = 192 + 132 = 324
        let cfg = ComplexMLPConfig::magneto_dielectric(32, ComplexActivation::default());
        assert_eq!(cfg.parameter_count(), 324);
    }

    #[test]
    fn test_config_parameter_count_matches_actual_varmap_elem_sum() -> Result<()> {
        let activations = [
            ComplexActivation::CReLU,
            ComplexActivation::CGELU,
            ComplexActivation::Cardioid,
            ComplexActivation::ModReLU,
            ComplexActivation::HybridCardioidGelu,
            ComplexActivation::Lpma,
            ComplexActivation::CSwishPhase,
            ComplexActivation::Worelu,
        ];
        let norms = [ComplexNormChoice::None, ComplexNormChoice::LayerNorm];
        let hidden_sizes = [6_usize, 12, 24, 48];

        for &h in &hidden_sizes {
            for act in &activations {
                for &norm in &norms {
                    let cfg = ComplexMLPConfig::permittivity(h, *act).with_norm(norm);
                    let varmap = VarMap::new();
                    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
                    let _model = ComplexMLPRegressor::new(vb, &cfg)?;

                    let actual: usize = varmap
                        .all_vars()
                        .iter()
                        .map(|v| v.as_tensor().elem_count())
                        .sum();
                    let formula = cfg.parameter_count();
                    assert_eq!(
                        formula, actual,
                        "param-count mismatch: H={h} act={act:?} norm={norm:?} \
                         formula={formula} actual={actual}",
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_config_display_is_log_friendly() {
        let cfg = ComplexMLPConfig::permittivity(
            32,
            ComplexActivation::CGELU,
        );
        assert_eq!(cfg.to_string(), "ComplexMLP(2c->32c->1c, cgelu)");

        let cfg_lr = ComplexMLPConfig::permittivity(32, ComplexActivation::ModReLU);
        assert_eq!(cfg_lr.to_string(), "ComplexMLP(2c->32c->1c, modrelu)");
    }

    #[test]
    fn test_config_rejects_zero_sizes() {
        let zero_in = ComplexMLPConfig::new(0, 32, 1, ComplexActivation::default());
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        assert!(ComplexMLPRegressor::new(vb.clone(), &zero_in).is_err());

        let zero_hidden = ComplexMLPConfig::new(2, 0, 1, ComplexActivation::default());
        assert!(ComplexMLPRegressor::new(vb.clone(), &zero_hidden).is_err());

        let zero_out = ComplexMLPConfig::new(2, 32, 0, ComplexActivation::default());
        assert!(ComplexMLPRegressor::new(vb, &zero_out).is_err());
    }

    #[test]
    fn test_config_serde_roundtrip() -> Result<()> {
        let cfg = ComplexMLPConfig::permittivity(
            32,
            ComplexActivation::CGELU,
        );
        let json =
            serde_json::to_string(&cfg).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let restored: ComplexMLPConfig =
            serde_json::from_str(&json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        assert_eq!(cfg, restored);

        let cfg_lr = ComplexMLPConfig::permittivity(16, ComplexActivation::ModReLU);
        let json_lr =
            serde_json::to_string(&cfg_lr).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let restored_lr: ComplexMLPConfig =
            serde_json::from_str(&json_lr).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        assert_eq!(cfg_lr, restored_lr);

        Ok(())
    }

    #[test]
    fn test_config_serde_roundtrip_magneto_dielectric() -> Result<()> {
        let cfg = ComplexMLPConfig::magneto_dielectric(16, ComplexActivation::ModReLU);
        let json =
            serde_json::to_string(&cfg).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let restored: ComplexMLPConfig =
            serde_json::from_str(&json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        assert_eq!(cfg, restored);

        Ok(())
    }

    #[test]
    fn test_forward_output_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(16, ComplexActivation::default());
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (8, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (8, 2), &Device::Cpu)?,
        )?;
        let ys = model.forward(&xs)?;

        assert_eq!(ys.real.dims(), &[8, 1]);
        assert_eq!(ys.imag.dims(), &[8, 1]);

        Ok(())
    }

    #[test]
    fn test_forward_output_shape_magneto_dielectric() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::magneto_dielectric(16, ComplexActivation::ModReLU);
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
        )?;
        let ys = model.forward(&xs)?;

        assert_eq!(ys.real.dims(), &[4, 2]);
        assert_eq!(ys.imag.dims(), &[4, 2]);

        Ok(())
    }

    #[test]
    fn test_forward_rejects_wrong_input_dims() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(8, ComplexActivation::default());
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let wrong_features = ComplexTensor::new(
            Tensor::zeros((4, 3), DType::F64, &Device::Cpu)?,
            Tensor::zeros((4, 3), DType::F64, &Device::Cpu)?,
        )?;
        assert!(model.forward(&wrong_features).is_err());

        let one_d = ComplexTensor::new(
            Tensor::zeros(2, DType::F64, &Device::Cpu)?,
            Tensor::zeros(2, DType::F64, &Device::Cpu)?,
        )?;
        assert!(model.forward(&one_d).is_err());

        Ok(())
    }

    #[test]
    fn test_forward_with_each_stateless_activation() -> Result<()> {
        let activations = [
            ComplexActivation::CReLU,
            ComplexActivation::CGELU,
            ComplexActivation::Cardioid,
        ];
        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
        )?;

        for act in &activations {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
            let cfg = ComplexMLPConfig::permittivity(8, *act);
            let model = ComplexMLPRegressor::new(vb, &cfg)?;
            let ys = model.forward(&xs)?;
            assert_eq!(ys.real.dims(), &[4, 1], "failed for {act}");
        }

        Ok(())
    }

    #[test]
    fn test_forward_with_modrelu_activation() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(8, ComplexActivation::ModReLU);
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
        )?;
        let ys = model.forward(&xs)?;

        assert_eq!(ys.real.dims(), &[4, 1]);
        assert_eq!(ys.imag.dims(), &[4, 1]);

        Ok(())
    }

    #[test]
    fn test_forward_deterministic_with_fixed_weights() -> Result<()> {
        // 2c-2c-1c with CReLU. Hand-calculated: y_re = 0.3, y_im = 0.0
        // for input x_re=[1,0], x_im=[0,1] under the weights below.
        let dev = Device::Cpu;
        let tensors = tensor_map(vec![
            (
                "input.weight_re",
                Tensor::from_vec(vec![1.0f64, 0., 0., 1.], (2, 2), &dev)?,
            ),
            (
                "input.weight_im",
                Tensor::from_vec(vec![0.0f64, 1., -1., 0.], (2, 2), &dev)?,
            ),
            (
                "input.bias_re",
                Tensor::from_vec(vec![0.1f64, 0.2], 2, &dev)?,
            ),
            (
                "input.bias_im",
                Tensor::from_vec(vec![0.0f64, 0.0], 2, &dev)?,
            ),
            (
                "output.weight_re",
                Tensor::from_vec(vec![1.0f64, 1.], (1, 2), &dev)?,
            ),
            (
                "output.weight_im",
                Tensor::from_vec(vec![0.0f64, 0.], (1, 2), &dev)?,
            ),
            ("output.bias_re", Tensor::from_vec(vec![0.0f64], 1, &dev)?),
            ("output.bias_im", Tensor::from_vec(vec![0.0f64], 1, &dev)?),
        ]);

        let vb = VarBuilder::from_tensors(tensors, DType::F64, &dev);
        let cfg = ComplexMLPConfig::new(
            2,
            2,
            1,
            ComplexActivation::CReLU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 0.], (1, 2), &dev)?,
            Tensor::from_vec(vec![0.0f64, 1.], (1, 2), &dev)?,
        )?;
        let ys = model.forward(&xs)?;

        assert_tensor_close(&ys.real.to_vec2::<f64>()?[0], &[0.3], 1e-12);
        assert_tensor_close(&ys.imag.to_vec2::<f64>()?[0], &[0.0], 1e-12);

        Ok(())
    }

    #[test]
    fn test_forward_deterministic_magneto_dielectric_with_fixed_weights() -> Result<()> {
        // Same first layer as the scalar test → h = [0.1+j0, 0.2+j0].
        // Hand-calc: y_re = [0.8, -0.1], y_im = [0.05, 0.1].
        let dev = Device::Cpu;
        let tensors = tensor_map(vec![
            (
                "input.weight_re",
                Tensor::from_vec(vec![1.0f64, 0., 0., 1.], (2, 2), &dev)?,
            ),
            (
                "input.weight_im",
                Tensor::from_vec(vec![0.0f64, 1., -1., 0.], (2, 2), &dev)?,
            ),
            (
                "input.bias_re",
                Tensor::from_vec(vec![0.1f64, 0.2], 2, &dev)?,
            ),
            (
                "input.bias_im",
                Tensor::from_vec(vec![0.0f64, 0.0], 2, &dev)?,
            ),
            (
                "output.weight_re",
                Tensor::from_vec(vec![1.0f64, 1.0, 2.0, -1.0], (2, 2), &dev)?,
            ),
            (
                "output.weight_im",
                Tensor::from_vec(vec![0.5f64, -0.5, 1.0, 1.0], (2, 2), &dev)?,
            ),
            (
                "output.bias_re",
                Tensor::from_vec(vec![0.5f64, -0.1], 2, &dev)?,
            ),
            (
                "output.bias_im",
                Tensor::from_vec(vec![0.1f64, -0.2], 2, &dev)?,
            ),
        ]);

        let vb = VarBuilder::from_tensors(tensors, DType::F64, &dev);
        let cfg = ComplexMLPConfig::new(
            2,
            2,
            2,
            ComplexActivation::CReLU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 0.0], (1, 2), &dev)?,
            Tensor::from_vec(vec![0.0f64, 1.0], (1, 2), &dev)?,
        )?;
        let ys = model.forward(&xs)?;

        assert_tensor_close(&ys.real.to_vec2::<f64>()?[0], &[0.8, -0.1], 1e-12);
        assert_tensor_close(&ys.imag.to_vec2::<f64>()?[0], &[0.05, 0.1], 1e-12);

        Ok(())
    }

    #[test]
    fn test_gradient_flow_with_stateless_activation() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(
            8,
            ComplexActivation::CGELU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
        )?;
        let ys = model.forward(&xs)?;

        let loss = ys.mag_sq()?.sum_all()?;
        let grads = loss.backward()?;

        let all_vars = varmap.all_vars();
        assert!(!all_vars.is_empty());
        for var in &all_vars {
            assert!(grads.get(var).is_some());
        }

        // 2 ComplexLinear × 4 tensors = 8.
        assert_eq!(all_vars.len(), 8);
        assert_eq!(cfg.parameter_count(), model.parameter_count());

        Ok(())
    }

    #[test]
    fn test_gradient_flow_with_modrelu_activation() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(8, ComplexActivation::ModReLU);
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
        )?;
        let ys = model.forward(&xs)?;
        let loss = ys.mag_sq()?.sum_all()?;
        let grads = loss.backward()?;

        let all_vars = varmap.all_vars();
        // 2 ComplexLinear × 4 + ModReLU bias = 9.
        assert_eq!(all_vars.len(), 9);

        for var in &all_vars {
            assert!(grads.get(var).is_some());
        }

        assert_eq!(cfg.parameter_count(), model.parameter_count());

        Ok(())
    }

    #[test]
    fn test_gradient_flow_with_magneto_dielectric_output() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::magneto_dielectric(
            8,
            ComplexActivation::CGELU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (4, 2), &Device::Cpu)?,
        )?;
        let ys = model.forward(&xs)?;
        let loss = ys.mag_sq()?.sum_all()?;
        let grads = loss.backward()?;

        let all_vars = varmap.all_vars();
        assert_eq!(ys.real.dims(), &[4, 2]);
        assert_eq!(ys.imag.dims(), &[4, 2]);
        assert_eq!(all_vars.len(), 8);

        for var in &all_vars {
            assert!(grads.get(var).is_some());
        }

        assert_eq!(cfg.parameter_count(), model.parameter_count());

        Ok(())
    }

    #[test]
    fn test_end_to_end_finite_difference_gradient_check() -> Result<()> {
        let dev = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &dev);
        let cfg = ComplexMLPConfig::permittivity(
            4,
            ComplexActivation::CGELU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        // Pinned weights — random init lands too close to the FD tolerance.
        varmap.set(
            [
                (
                    "input.weight_re",
                    Tensor::from_vec(
                        vec![0.25f64, -0.10, -0.30, 0.40, 0.15, 0.35, -0.20, 0.50],
                        (4, 2),
                        &dev,
                    )?,
                ),
                (
                    "input.weight_im",
                    Tensor::from_vec(
                        vec![0.05f64, 0.30, -0.25, 0.10, 0.20, -0.15, 0.45, -0.05],
                        (4, 2),
                        &dev,
                    )?,
                ),
                (
                    "input.bias_re",
                    Tensor::from_vec(vec![0.02f64, -0.03, 0.07, 0.01], 4, &dev)?,
                ),
                (
                    "input.bias_im",
                    Tensor::from_vec(vec![-0.04f64, 0.05, -0.02, 0.06], 4, &dev)?,
                ),
                (
                    "output.weight_re",
                    Tensor::from_vec(vec![0.40f64, -0.15, 0.25, 0.10], (1, 4), &dev)?,
                ),
                (
                    "output.weight_im",
                    Tensor::from_vec(vec![-0.20f64, 0.35, 0.05, -0.30], (1, 4), &dev)?,
                ),
                ("output.bias_re", Tensor::from_vec(vec![0.03f64], 1, &dev)?),
                ("output.bias_im", Tensor::from_vec(vec![-0.01f64], 1, &dev)?),
            ]
            .into_iter(),
        )?;

        let xr = Var::from_tensor(&Tensor::from_vec(
            vec![0.5f64, -0.3, 1.0, 0.2],
            (2, 2),
            &dev,
        )?)?;
        let xi = Var::from_tensor(&Tensor::from_vec(
            vec![0.1f64, 0.7, -0.5, 0.4],
            (2, 2),
            &dev,
        )?)?;
        let xs = ComplexTensor::new(xr.as_tensor().clone(), xi.as_tensor().clone())?;

        let ys = model.forward(&xs)?;
        let loss = ys.mag_sq()?.sum_all()?;
        let grads = loss.backward()?;

        let eps = 1e-5;
        let tol = 1e-4;

        let tensor_data = varmap.data().lock().unwrap();
        let vars = [
            tensor_data.get("input.weight_re").unwrap().clone(),
            tensor_data.get("input.weight_im").unwrap().clone(),
            tensor_data.get("output.weight_re").unwrap().clone(),
            tensor_data.get("output.weight_im").unwrap().clone(),
        ];
        drop(tensor_data);

        for var in vars.iter() {
            let analytical = grads.get(var).unwrap().flatten_all()?.to_vec1::<f64>()?;
            let param_data = var.flatten_all()?.to_vec1::<f64>()?;

            for idx in 0..param_data.len().min(4) {
                let mut p_plus = param_data.clone();
                p_plus[idx] += eps;
                var.set(&Tensor::from_vec(p_plus, var.shape(), &dev)?)?;
                let xs_c = ComplexTensor::new(xr.as_tensor().clone(), xi.as_tensor().clone())?;
                let l_plus = model
                    .forward(&xs_c)?
                    .mag_sq()?
                    .sum_all()?
                    .to_scalar::<f64>()?;

                let mut p_minus = param_data.clone();
                p_minus[idx] -= eps;
                var.set(&Tensor::from_vec(p_minus, var.shape(), &dev)?)?;
                let xs_c = ComplexTensor::new(xr.as_tensor().clone(), xi.as_tensor().clone())?;
                let l_minus = model
                    .forward(&xs_c)?
                    .mag_sq()?
                    .sum_all()?
                    .to_scalar::<f64>()?;

                var.set(&Tensor::from_vec(param_data.clone(), var.shape(), &dev)?)?;

                let numerical = (l_plus - l_minus) / (2.0 * eps);
                let a = analytical[idx];
                let rel_err = (a - numerical).abs() / a.abs().max(numerical.abs()).max(1e-8);
                assert!(
                    rel_err < tol,
                    "finite-diff gradient mismatch: analytical={a:.8}, numerical={numerical:.8}, rel_err={rel_err:.2e}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_model_display() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(
            32,
            ComplexActivation::CReLU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;
        assert_eq!(model.to_string(), "ComplexMLP(2c->32c->1c, crelu)");

        Ok(())
    }

    #[test]
    fn test_model_display_magneto_dielectric() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::magneto_dielectric(
            32,
            ComplexActivation::CReLU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;
        assert_eq!(model.to_string(), "ComplexMLP(2c->32c->2c, crelu)");

        Ok(())
    }

    #[test]
    fn test_accessors_and_forward_unchecked_match_validated_forward() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(
            16,
            ComplexActivation::CReLU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        assert_eq!(model.input_size(), 2);
        assert_eq!(model.hidden_size(), 16);
        assert_eq!(model.output_size(), 1);

        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1., (3, 2), &Device::Cpu)?,
            Tensor::randn(0f64, 1., (3, 2), &Device::Cpu)?,
        )?;

        let validated = model.forward(&xs)?;
        let unchecked = model.forward_unchecked(&xs)?;

        assert_tensor_close(
            &validated.real.flatten_all()?.to_vec1::<f64>()?,
            &unchecked.real.flatten_all()?.to_vec1::<f64>()?,
            1e-12,
        );
        assert_tensor_close(
            &validated.imag.flatten_all()?.to_vec1::<f64>()?,
            &unchecked.imag.flatten_all()?.to_vec1::<f64>()?,
            1e-12,
        );

        Ok(())
    }

    #[test]
    fn forward_packed_uses_half_half_layout_not_interleaved() -> Result<()> {
        use sparam_core::complex_tensor::ComplexTensor;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let cfg = ComplexMLPConfig::permittivity(
            8,
            ComplexActivation::CReLU,
        );
        let model = ComplexMLPRegressor::new(vb, &cfg)?;

        // Packed `[s11_re, s21_re, s11_im, s21_im]`.
        let packed = Tensor::from_vec(
            vec![0.1f64, 0.3, 0.2, 0.4],
            (1, 4),
            &Device::Cpu,
        )?;
        let packed_out = model.forward_packed(&packed, false)?;
        let packed_vec = packed_out.flatten_all()?.to_vec1::<f64>()?;

        let good_real = packed.narrow(1, 0, 2)?;
        let good_imag = packed.narrow(1, 2, 2)?;
        let good_cx = ComplexTensor::new(good_real, good_imag)?;
        let good_out = model.forward(&good_cx)?;
        let good_re = good_out.real.flatten_all()?.to_vec1::<f64>()?[0];
        let good_im = good_out.imag.flatten_all()?.to_vec1::<f64>()?[0];
        assert_tensor_close(&packed_vec, &[good_re, good_im], 1e-12);

        // Old interleaved layout (must differ from packed result).
        let reshaped = packed.reshape((1usize, 2usize, 2usize))?;
        let bad_real = reshaped.narrow(2, 0, 1)?.reshape((1usize, 2usize))?;
        let bad_imag = reshaped.narrow(2, 1, 1)?.reshape((1usize, 2usize))?;
        let bad_cx = ComplexTensor::new(bad_real, bad_imag)?;
        let bad_out = model.forward(&bad_cx)?;
        let bad_re = bad_out.real.flatten_all()?.to_vec1::<f64>()?[0];
        let bad_im = bad_out.imag.flatten_all()?.to_vec1::<f64>()?[0];
        let diff = (bad_re - packed_vec[0]).abs() + (bad_im - packed_vec[1]).abs();
        assert!(
            diff > 1e-6,
            "interleaved layout coincidentally matched packed layout (diff={diff:.3e})",
        );
        Ok(())
    }

    #[test]
    fn all_hpo_complex_activations_parse_and_build() -> Result<()> {
        let names = [
            "crelu",
            "cgelu",
            "cardioid",
            "modrelu",
            "hybrid_cardioid_gelu",
            "lpma",
            "cswish_phase",
            "worelu",
        ];
        for name in names {
            let choice = ComplexActivation::from_name(name)
                .unwrap_or_else(|e| panic!("parse {name}: {e}"));
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
            let cfg = ComplexMLPConfig::permittivity(8, choice);
            ComplexMLPRegressor::new(vb, &cfg)
                .unwrap_or_else(|e| panic!("build {name}: {e}"));
        }
        Ok(())
    }

    #[test]
    fn all_hpo_complex_activation_names_round_trip() {
        let names = [
            "crelu",
            "cgelu",
            "cardioid",
            "modrelu",
            "hybrid_cardioid_gelu",
            "lpma",
            "cswish_phase",
            "worelu",
        ];
        for name in names {
            let choice = ComplexActivation::from_name(name)
                .unwrap_or_else(|e| panic!("from_name({name}): {e}"));
            let round_tripped = choice.name();
            assert_eq!(
                name, round_tripped,
                "round-trip mismatch for '{name}': from_name → {:?} → name() → '{round_tripped}'",
                choice,
            );
        }
    }

}
