//! Unified optimizer surface wrapping Candle's built-in AdamW and providing
//! full SGD (with momentum / Nesterov / weight-decay) and RMSprop.

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};
use std::fmt;

use sparam_core::error::candle_msg;

// ---------------------------------------------------------------------------
// AdamW configuration  (US-5.4)
// ---------------------------------------------------------------------------

/// Configuration for the AdamW optimizer (decoupled weight decay).
///
/// Wraps [`candle_nn::optim::AdamW`] — no custom implementation needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl From<AdamWConfig> for ParamsAdamW {
    fn from(c: AdamWConfig) -> Self {
        ParamsAdamW {
            lr: c.lr,
            beta1: c.beta1,
            beta2: c.beta2,
            eps: c.eps,
            weight_decay: c.weight_decay,
        }
    }
}

impl fmt::Display for AdamWConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AdamW(lr={}, β1={}, β2={}, eps={}, wd={})",
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay
        )
    }
}

// ---------------------------------------------------------------------------
// Adam configuration  (US-5.5)
// ---------------------------------------------------------------------------

/// Configuration for the standard Adam optimizer.
///
/// Implemented as AdamW with `weight_decay = 0`. For L2 regularization,
/// add the penalty directly to the loss using [`l2_regularization`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

impl fmt::Display for AdamConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Adam(lr={}, β1={}, β2={}, eps={})",
            self.lr, self.beta1, self.beta2, self.eps
        )
    }
}

// ---------------------------------------------------------------------------
// SGD configuration  (US-5.6)
// ---------------------------------------------------------------------------

/// Full Stochastic Gradient Descent with optional momentum, Nesterov
/// acceleration, and L2 weight decay.
///
/// Candle's built-in SGD is bare-bones (learning rate only), so this is a
/// custom implementation following the PyTorch formulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SgdConfig {
    pub lr: f64,
    /// Momentum factor (0 = disabled).
    pub momentum: f64,
    /// Dampening for momentum accumulation (default 0.0).
    pub dampening: f64,
    /// Enable Nesterov momentum (requires `momentum > 0`).
    pub nesterov: bool,
    /// L2 weight decay (added to gradients).
    pub weight_decay: f64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            weight_decay: 0.0,
        }
    }
}

impl fmt::Display for SgdConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SGD(lr={}, mom={}, damp={}, nesterov={}, wd={})",
            self.lr, self.momentum, self.dampening, self.nesterov, self.weight_decay
        )
    }
}

// ---------------------------------------------------------------------------
// RMSprop configuration  (US-5.7)
// ---------------------------------------------------------------------------

/// RMSprop optimizer with optional momentum, centering, and weight decay.
///
/// Follows the PyTorch formulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSpropConfig {
    pub lr: f64,
    /// Smoothing constant (default 0.99).
    pub alpha: f64,
    pub eps: f64,
    /// Momentum factor (0 = disabled).
    pub momentum: f64,
    /// L2 weight decay.
    pub weight_decay: f64,
    /// If `true`, compute centred RMSprop (variance rather than second moment).
    pub centered: bool,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            centered: false,
        }
    }
}

impl fmt::Display for RMSpropConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RMSprop(lr={}, α={}, eps={}, mom={}, wd={}, centered={})",
            self.lr, self.alpha, self.eps, self.momentum, self.weight_decay, self.centered
        )
    }
}

/// A tagged, serializable optimizer configuration for config-driven training.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "params", rename_all = "snake_case")]
pub enum OptimizerConfig {
    AdamW(AdamWConfig),
    Adam(AdamConfig),
    Sgd(SgdConfig),
    RMSprop(RMSpropConfig),
}

impl OptimizerConfig {
    #[must_use]
    pub fn adamw(lr: f64, weight_decay: f64) -> Self {
        Self::AdamW(AdamWConfig {
            lr,
            weight_decay,
            ..Default::default()
        })
    }

    #[must_use]
    pub fn adam(lr: f64) -> Self {
        Self::Adam(AdamConfig {
            lr,
            ..Default::default()
        })
    }

    #[must_use]
    pub fn sgd(lr: f64, weight_decay: f64) -> Self {
        Self::sgd_with(lr, weight_decay, 0.0, false)
    }

    #[must_use]
    pub fn sgd_with(lr: f64, weight_decay: f64, momentum: f64, nesterov: bool) -> Self {
        Self::Sgd(SgdConfig {
            lr,
            weight_decay,
            momentum,
            nesterov,
            ..Default::default()
        })
    }

    #[must_use]
    pub fn rmsprop(lr: f64, weight_decay: f64) -> Self {
        Self::rmsprop_with(lr, weight_decay, 0.0, 0.99)
    }

    #[must_use]
    pub fn rmsprop_with(lr: f64, weight_decay: f64, momentum: f64, alpha: f64) -> Self {
        Self::RMSprop(RMSpropConfig {
            lr,
            weight_decay,
            momentum,
            alpha,
            ..Default::default()
        })
    }

    pub fn from_name(name: &str, lr: f64, weight_decay: f64) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "adamw" => Ok(Self::adamw(lr, weight_decay)),
            "adam" => Ok(Self::adam(lr)),
            "sgd" => Ok(Self::sgd(lr, weight_decay)),
            "rmsprop" => Ok(Self::rmsprop(lr, weight_decay)),
            other => Err(candle_msg(format!(
                "unknown optimizer '{other}', expected: adam, adamw, sgd, rmsprop"
            ))),
        }
    }

    pub fn build(self, vars: Vec<Var>) -> Result<OptimizerKind> {
        OptimizerKind::from_config(vars, self)
    }
}

impl fmt::Display for OptimizerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdamW(config) => write!(f, "{config}"),
            Self::Adam(config) => write!(f, "{config}"),
            Self::Sgd(config) => write!(f, "{config}"),
            Self::RMSprop(config) => write!(f, "{config}"),
        }
    }
}

// ---------------------------------------------------------------------------
// SGD state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SgdVar {
    var: Var,
    /// Momentum buffer (lazily initialised on the first step that sees a gradient).
    velocity: Option<Tensor>,
}

#[derive(Debug)]
pub(crate) struct SgdInner {
    vars: Vec<SgdVar>,
    config: SgdConfig,
}

fn gradient_with_weight_decay(var: &Var, grad: &Tensor, weight_decay: f64) -> Result<Tensor> {
    if weight_decay != 0.0 {
        grad + var.as_tensor().affine(weight_decay, 0.0)?
    } else {
        Ok(grad.clone())
    }
}

fn apply_scaled_update(var: &Var, update: &Tensor, lr: f64) -> Result<()> {
    var.set(&var.as_tensor().sub(&update.affine(lr, 0.0)?)?)?;
    Ok(())
}

fn sgd_update_with_momentum(
    velocity: &mut Option<Tensor>,
    grad: &Tensor,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
) -> Result<Tensor> {
    if momentum == 0.0 {
        return Ok(grad.clone());
    }

    let buf = match velocity {
        Some(previous) => (previous.affine(momentum, 0.0)? + grad.affine(1.0 - dampening, 0.0)?)?,
        None => grad.clone(),
    };
    let update = if nesterov {
        (grad + buf.affine(momentum, 0.0)?)?
    } else {
        buf.clone()
    };
    *velocity = Some(buf);
    Ok(update)
}

impl SgdInner {
    fn new(vars: Vec<Var>, config: SgdConfig) -> Result<Self> {
        validate_non_negative(config.momentum, "SGD: momentum")?;
        validate_non_negative(config.dampening, "SGD: dampening")?;
        validate_non_negative(config.weight_decay, "SGD: weight_decay")?;
        if config.nesterov && config.momentum == 0.0 {
            return Err(candle_msg("SGD: nesterov=true requires momentum > 0"));
        }
        if config.nesterov && config.dampening != 0.0 {
            return Err(candle_msg("SGD: nesterov=true requires dampening == 0"));
        }
        let vars = vars
            .into_iter()
            .filter(|v| v.dtype().is_float())
            .map(|v| SgdVar {
                var: v,
                velocity: None,
            })
            .collect();
        Ok(Self { vars, config })
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let mom = self.config.momentum;
        let dampening = self.config.dampening;
        let nesterov = self.config.nesterov;

        for sv in &mut self.vars {
            let grad = match grads.get(&sv.var) {
                Some(grad) => grad,
                None => continue,
            };

            let grad = gradient_with_weight_decay(&sv.var, grad, wd)?;
            let update =
                sgd_update_with_momentum(&mut sv.velocity, &grad, mom, dampening, nesterov)?;
            apply_scaled_update(&sv.var, &update, lr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RMSprop state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RMSpropVar {
    var: Var,
    square_avg: Option<Tensor>,
    grad_avg: Option<Tensor>,
    momentum_buf: Option<Tensor>,
}

#[derive(Debug)]
pub(crate) struct RMSpropInner {
    vars: Vec<RMSpropVar>,
    config: RMSpropConfig,
}

fn update_square_average(
    square_avg: &mut Option<Tensor>,
    grad: &Tensor,
    alpha: f64,
) -> Result<Tensor> {
    let next = match square_avg {
        Some(previous) => (previous.affine(alpha, 0.0)? + grad.sqr()?.affine(1.0 - alpha, 0.0)?)?,
        None => grad.sqr()?.affine(1.0 - alpha, 0.0)?,
    };
    *square_avg = Some(next.clone());
    Ok(next)
}

fn update_centered_average(
    grad_avg: &mut Option<Tensor>,
    grad: &Tensor,
    square_avg: Tensor,
    alpha: f64,
    centered: bool,
) -> Result<Tensor> {
    if !centered {
        return Ok(square_avg);
    }

    let next = match grad_avg {
        Some(previous) => (previous.affine(alpha, 0.0)? + grad.affine(1.0 - alpha, 0.0)?)?,
        None => grad.affine(1.0 - alpha, 0.0)?,
    };
    *grad_avg = Some(next.clone());
    square_avg - next.sqr()?
}

fn update_momentum_buffer(
    momentum_buf: &mut Option<Tensor>,
    update: Tensor,
    momentum: f64,
) -> Result<Tensor> {
    if momentum == 0.0 {
        return Ok(update);
    }

    let buf = match momentum_buf {
        Some(previous) => (previous.affine(momentum, 0.0)? + &update)?,
        None => update.clone(),
    };
    *momentum_buf = Some(buf.clone());
    Ok(buf)
}

impl RMSpropInner {
    fn new(vars: Vec<Var>, config: RMSpropConfig) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|v| v.dtype().is_float())
            .map(|v| RMSpropVar {
                var: v,
                square_avg: None,
                grad_avg: None,
                momentum_buf: None,
            })
            .collect();
        Ok(Self { vars, config })
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let alpha = self.config.alpha;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let mom = self.config.momentum;
        let centered = self.config.centered;

        for rv in &mut self.vars {
            let grad = match grads.get(&rv.var) {
                Some(grad) => grad,
                None => continue,
            };

            let grad = gradient_with_weight_decay(&rv.var, grad, wd)?;
            let square_avg = update_square_average(&mut rv.square_avg, &grad, alpha)?;
            let avg =
                update_centered_average(&mut rv.grad_avg, &grad, square_avg, alpha, centered)?;
            let denom = (avg.sqrt()? + eps)?;
            let update = grad.div(&denom)?;
            let update = update_momentum_buffer(&mut rv.momentum_buf, update, mom)?;
            apply_scaled_update(&rv.var, &update, lr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unified optimizer enum  (US-5.4 – US-5.7)
// ---------------------------------------------------------------------------

/// A unified optimizer covering AdamW, Adam, SGD, and RMSprop.
///
/// `OptimizerKind` delegates to Candle's built-in `AdamW` where possible and
/// provides full-featured custom implementations for SGD and RMSprop.
#[allow(private_interfaces)]
#[derive(Debug)]
pub enum OptimizerKind {
    /// AdamW with decoupled weight decay (Candle built-in).
    AdamW(AdamW),
    /// Standard Adam (AdamW with weight_decay = 0).
    Adam(AdamW),
    /// Full SGD with momentum, Nesterov, and L2 weight decay.
    Sgd(SgdInner),
    /// RMSprop with momentum, centering, and weight decay.
    RMSprop(RMSpropInner),
}

impl OptimizerKind {
    // -- Constructors -------------------------------------------------------

    /// Create an optimizer from a tagged configuration enum.
    pub fn from_config(vars: Vec<Var>, config: OptimizerConfig) -> Result<Self> {
        match config {
            OptimizerConfig::AdamW(config) => Self::adamw(vars, config),
            OptimizerConfig::Adam(config) => Self::adam(vars, config),
            OptimizerConfig::Sgd(config) => Self::sgd(vars, config),
            OptimizerConfig::RMSprop(config) => Self::rmsprop(vars, config),
        }
    }

    /// Create an AdamW optimizer.
    pub fn adamw(vars: Vec<Var>, config: AdamWConfig) -> Result<Self> {
        validate_lr(config.lr, "AdamW")?;
        validate_betas(config.beta1, config.beta2, "AdamW")?;
        validate_eps(config.eps, "AdamW")?;
        Ok(Self::AdamW(AdamW::new(vars, config.into())?))
    }

    /// Create a standard Adam optimizer (AdamW with weight_decay = 0).
    pub fn adam(vars: Vec<Var>, config: AdamConfig) -> Result<Self> {
        validate_lr(config.lr, "Adam")?;
        validate_betas(config.beta1, config.beta2, "Adam")?;
        validate_eps(config.eps, "Adam")?;
        let params = ParamsAdamW {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: 0.0,
        };
        Ok(Self::Adam(AdamW::new(vars, params)?))
    }

    /// Create an SGD optimizer.
    pub fn sgd(vars: Vec<Var>, config: SgdConfig) -> Result<Self> {
        validate_lr(config.lr, "SGD")?;
        Ok(Self::Sgd(SgdInner::new(vars, config)?))
    }

    /// Create an RMSprop optimizer.
    pub fn rmsprop(vars: Vec<Var>, config: RMSpropConfig) -> Result<Self> {
        validate_lr(config.lr, "RMSprop")?;
        validate_eps(config.eps, "RMSprop")?;
        if !(0.0..=1.0).contains(&config.alpha) {
            return Err(candle_msg(format!(
                "RMSprop: alpha must be in [0, 1], got {}",
                config.alpha
            )));
        }
        Ok(Self::RMSprop(RMSpropInner::new(vars, config)?))
    }

    // -- Unified interface --------------------------------------------------

    /// Perform one parameter update given computed gradients.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        match self {
            Self::AdamW(o) | Self::Adam(o) => Optimizer::step(o, grads),
            Self::Sgd(o) => o.step(grads),
            Self::RMSprop(o) => o.step(grads),
        }
    }

    /// Convenience: compute gradients from `loss` and step.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    /// Current learning rate.
    pub fn learning_rate(&self) -> f64 {
        match self {
            Self::AdamW(o) | Self::Adam(o) => o.learning_rate(),
            Self::Sgd(o) => o.config.lr,
            Self::RMSprop(o) => o.config.lr,
        }
    }

    /// Update the learning rate (e.g. from a scheduler).
    pub fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::AdamW(o) | Self::Adam(o) => o.set_learning_rate(lr),
            Self::Sgd(o) => o.config.lr = lr,
            Self::RMSprop(o) => o.config.lr = lr,
        }
    }

    /// Human-readable name of the active variant.
    pub fn name(&self) -> &'static str {
        match self {
            Self::AdamW(_) => "AdamW",
            Self::Adam(_) => "Adam",
            Self::Sgd(_) => "SGD",
            Self::RMSprop(_) => "RMSprop",
        }
    }
}

impl fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdamW(o) => {
                let p = o.params();
                write!(
                    f,
                    "AdamW(lr={}, β1={}, β2={}, eps={}, wd={})",
                    p.lr, p.beta1, p.beta2, p.eps, p.weight_decay
                )
            }
            Self::Adam(o) => {
                let p = o.params();
                write!(
                    f,
                    "Adam(lr={}, β1={}, β2={}, eps={})",
                    p.lr, p.beta1, p.beta2, p.eps
                )
            }
            Self::Sgd(o) => write!(f, "{}", o.config),
            Self::RMSprop(o) => write!(f, "{}", o.config),
        }
    }
}

// ---------------------------------------------------------------------------
// L2 regularisation helper  (for Adam / SGD without built-in decoupled WD)
// ---------------------------------------------------------------------------

/// Compute an L2 regularisation term: `0.5 * lambda * Σ‖θ‖²`.
///
/// Add the returned scalar to your loss before calling `backward()` when using
/// Adam (which has no decoupled weight decay) or any optimizer where you want
/// explicit L2 regularisation in the loss.
///
/// # When to use
/// - **Adam**: Has no built-in weight decay; add this to the loss.
/// - **SGD/RMSprop**: Already apply L2 via `weight_decay` in config — use one or the other, not both.
/// - **AdamW**: Uses *decoupled* weight decay (applied after gradient update), which is different from L2 in the loss.
pub fn l2_regularization(vars: &[Var], lambda: f64) -> Result<Tensor> {
    if vars.is_empty() {
        return Err(candle_msg("l2_regularization: no variables provided"));
    }
    let first = &vars[0];
    let device = first.as_tensor().device();
    let dtype = first.dtype();
    let mut acc = Tensor::zeros((), dtype, device)?;
    for v in vars {
        acc = (acc + v.as_tensor().sqr()?.sum_all()?)?;
    }
    acc.affine(lambda * 0.5, 0.0)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_lr(lr: f64, name: &str) -> Result<()> {
    sparam_core::validation::validate_positive_f64(&format!("{name}: lr"), lr)?;
    Ok(())
}

fn validate_betas(beta1: f64, beta2: f64, name: &str) -> Result<()> {
    if !(0.0..1.0).contains(&beta1) {
        return Err(candle_msg(format!(
            "{name}: beta1 must be in [0, 1), got {beta1}"
        )));
    }
    if !(0.0..1.0).contains(&beta2) {
        return Err(candle_msg(format!(
            "{name}: beta2 must be in [0, 1), got {beta2}"
        )));
    }
    Ok(())
}

fn validate_eps(eps: f64, name: &str) -> Result<()> {
    sparam_core::validation::validate_positive_f64(&format!("{name}: eps"), eps)?;
    Ok(())
}

fn validate_non_negative(value: f64, name: &str) -> Result<()> {
    sparam_core::validation::validate_non_negative_f64(name, value)?;
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Helper: create a single trainable variable with a given value.
    fn make_var(val: f64) -> Var {
        Var::from_tensor(&Tensor::new(&[val], &Device::Cpu).unwrap()).unwrap()
    }

    /// Helper: read the scalar value from a Var.
    fn var_val(v: &Var) -> f64 {
        v.as_tensor()
            .to_dtype(DType::F64)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()[0]
    }

    /// Helper: compute ∂(x²)/∂x = 2x by calling backward on x².
    fn grad_x_squared(x: &Var) -> GradStore {
        let loss = x.as_tensor().sqr().unwrap().sum_all().unwrap();
        loss.backward().unwrap()
    }

    // -----------------------------------------------------------------------
    // Config defaults & Display
    // -----------------------------------------------------------------------

    #[test]
    fn adamw_config_defaults() {
        let c = AdamWConfig::default();
        assert_eq!(c.lr, 1e-3);
        assert_eq!(c.weight_decay, 0.01);
        assert!(c.to_string().contains("AdamW"));
    }

    #[test]
    fn adam_config_defaults() {
        let c = AdamConfig::default();
        assert_eq!(c.lr, 1e-3);
        assert!(c.to_string().contains("Adam"));
    }

    #[test]
    fn sgd_config_defaults() {
        let c = SgdConfig::default();
        assert_eq!(c.lr, 1e-2);
        assert_eq!(c.momentum, 0.0);
        assert_eq!(c.dampening, 0.0);
        assert!(!c.nesterov);
        assert!(c.to_string().contains("SGD"));
    }

    #[test]
    fn rmsprop_config_defaults() {
        let c = RMSpropConfig::default();
        assert_eq!(c.lr, 1e-2);
        assert_eq!(c.alpha, 0.99);
        assert!(!c.centered);
        assert!(c.to_string().contains("RMSprop"));
    }

    // -----------------------------------------------------------------------
    // Config serde round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn adamw_config_serde() {
        let c = AdamWConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: AdamWConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c.lr, c2.lr);
        assert_eq!(c.weight_decay, c2.weight_decay);
    }

    #[test]
    fn sgd_config_serde() {
        let c = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            dampening: 0.1,
            nesterov: false,
            weight_decay: 1e-4,
        };
        let json = serde_json::to_string(&c).unwrap();
        let c2: SgdConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.momentum, 0.9);
        assert_eq!(c2.dampening, 0.1);
        assert!(!c2.nesterov);
    }

    #[test]
    fn optimizer_config_serde() {
        let config = OptimizerConfig::Sgd(SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            dampening: 0.0,
            nesterov: true,
            weight_decay: 1e-4,
        });
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"sgd\""));
        let round_trip: OptimizerConfig = serde_json::from_str(&json).unwrap();
        match round_trip {
            OptimizerConfig::Sgd(config) => {
                assert_eq!(config.lr, 0.05);
                assert_eq!(config.momentum, 0.9);
                assert!(config.nesterov);
            }
            other => panic!("unexpected config variant: {other:?}"),
        }
    }

    #[test]
    fn rmsprop_config_serde() {
        let c = RMSpropConfig {
            centered: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&c).unwrap();
        let c2: RMSpropConfig = serde_json::from_str(&json).unwrap();
        assert!(c2.centered);
    }

    #[test]
    fn optimizer_config_helpers_construct_expected_values() {
        match OptimizerConfig::adamw(1e-3, 1e-4) {
            OptimizerConfig::AdamW(config) => {
                assert_eq!(config.lr, 1e-3);
                assert_eq!(config.weight_decay, 1e-4);
            }
            other => panic!("unexpected config variant: {other:?}"),
        }

        match OptimizerConfig::sgd_with(1e-2, 1e-3, 0.9, true) {
            OptimizerConfig::Sgd(config) => {
                assert_eq!(config.lr, 1e-2);
                assert_eq!(config.weight_decay, 1e-3);
                assert_eq!(config.momentum, 0.9);
                assert!(config.nesterov);
            }
            other => panic!("unexpected config variant: {other:?}"),
        }
    }

    #[test]
    fn optimizer_config_from_name_parses_supported_values() {
        assert!(matches!(
            OptimizerConfig::from_name("adamw", 1e-3, 1e-4).unwrap(),
            OptimizerConfig::AdamW(_)
        ));
        assert!(matches!(
            OptimizerConfig::from_name("ADAM", 1e-3, 1e-4).unwrap(),
            OptimizerConfig::Adam(_)
        ));
        assert!(matches!(
            OptimizerConfig::from_name("sgd", 1e-3, 1e-4).unwrap(),
            OptimizerConfig::Sgd(_)
        ));
        assert!(matches!(
            OptimizerConfig::from_name("rmsprop", 1e-3, 1e-4).unwrap(),
            OptimizerConfig::RMSprop(_)
        ));
    }

    #[test]
    fn optimizer_config_from_name_rejects_unknown_values() {
        let error = OptimizerConfig::from_name("adagrad", 1e-3, 1e-4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown optimizer"));
    }

    #[test]
    fn optimizer_config_builds_expected_variant() {
        let v = make_var(1.0);
        let opt = OptimizerConfig::adamw(1e-3, 1e-4).build(vec![v]).unwrap();
        assert_eq!(opt.name(), "AdamW");
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn adamw_rejects_bad_lr() {
        let v = make_var(1.0);
        let r = OptimizerKind::adamw(
            vec![v],
            AdamWConfig {
                lr: -1.0,
                ..Default::default()
            },
        );
        assert!(r.is_err());
    }

    #[test]
    fn adam_rejects_bad_betas() {
        let v = make_var(1.0);
        let r = OptimizerKind::adam(
            vec![v],
            AdamConfig {
                beta1: 1.0,
                ..Default::default()
            },
        );
        assert!(r.is_err());
    }

    #[test]
    fn sgd_rejects_nesterov_without_momentum() {
        let v = make_var(1.0);
        let r = OptimizerKind::sgd(
            vec![v],
            SgdConfig {
                nesterov: true,
                momentum: 0.0,
                ..Default::default()
            },
        );
        assert!(r.is_err());
    }

    #[test]
    fn sgd_rejects_nesterov_with_dampening() {
        let v = make_var(1.0);
        let r = OptimizerKind::sgd(
            vec![v],
            SgdConfig {
                momentum: 0.9,
                dampening: 0.1,
                nesterov: true,
                ..Default::default()
            },
        );
        assert!(r.is_err());
    }

    #[test]
    fn rmsprop_rejects_bad_alpha() {
        let v = make_var(1.0);
        let r = OptimizerKind::rmsprop(
            vec![v],
            RMSpropConfig {
                alpha: 1.5,
                ..Default::default()
            },
        );
        assert!(r.is_err());
    }

    // -----------------------------------------------------------------------
    // AdamW step  (US-5.4)
    // -----------------------------------------------------------------------

    #[test]
    fn adamw_step_decreases_loss() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::adamw(vec![x.clone()], AdamWConfig::default()).unwrap();
        for _ in 0..10 {
            let grads = grad_x_squared(&x);
            opt.step(&grads).unwrap();
        }
        assert!(var_val(&x).abs() < 3.0, "x should move towards 0");
    }

    #[test]
    fn adamw_weight_decay_applied() {
        let x_hi = make_var(5.0);
        let x_lo = make_var(5.0);
        let mut opt_hi = OptimizerKind::adamw(
            vec![x_hi.clone()],
            AdamWConfig {
                weight_decay: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        let mut opt_lo = OptimizerKind::adamw(
            vec![x_lo.clone()],
            AdamWConfig {
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..20 {
            opt_hi.step(&grad_x_squared(&x_hi)).unwrap();
            opt_lo.step(&grad_x_squared(&x_lo)).unwrap();
        }
        assert!(
            var_val(&x_hi).abs() < var_val(&x_lo).abs(),
            "high WD should shrink parameter more"
        );
    }

    // -----------------------------------------------------------------------
    // Adam step  (US-5.5)
    // -----------------------------------------------------------------------

    #[test]
    fn adam_step_decreases_loss() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::adam(vec![x.clone()], AdamConfig::default()).unwrap();
        for _ in 0..10 {
            opt.step(&grad_x_squared(&x)).unwrap();
        }
        assert!(var_val(&x).abs() < 3.0);
    }

    #[test]
    fn adam_has_no_weight_decay() {
        let x1 = make_var(4.0);
        let x2 = make_var(4.0);
        let mut adam = OptimizerKind::adam(vec![x1.clone()], AdamConfig::default()).unwrap();
        let mut adamw = OptimizerKind::adamw(
            vec![x2.clone()],
            AdamWConfig {
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..10 {
            adam.step(&grad_x_squared(&x1)).unwrap();
            adamw.step(&grad_x_squared(&x2)).unwrap();
        }
        let diff = (var_val(&x1) - var_val(&x2)).abs();
        assert!(
            diff < 1e-12,
            "Adam and AdamW(wd=0) should match: diff={diff}"
        );
    }

    // -----------------------------------------------------------------------
    // SGD step  (US-5.6)
    // -----------------------------------------------------------------------

    #[test]
    fn sgd_vanilla_step() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::sgd(
            vec![x.clone()],
            SgdConfig {
                lr: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        opt.step(&grad_x_squared(&x)).unwrap();
        let v = var_val(&x);
        assert!((v - 2.4).abs() < 1e-10, "expected 2.4, got {v}");
    }

    #[test]
    fn sgd_with_momentum() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::sgd(
            vec![x.clone()],
            SgdConfig {
                lr: 0.01,
                momentum: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..50 {
            opt.step(&grad_x_squared(&x)).unwrap();
        }
        assert!(
            var_val(&x).abs() < 1.0,
            "momentum SGD should converge towards 0"
        );
    }

    #[test]
    fn sgd_dampening_slows_momentum_updates() {
        let x_no_damp = make_var(3.0);
        let x_damped = make_var(3.0);
        let mut opt_no_damp = OptimizerKind::sgd(
            vec![x_no_damp.clone()],
            SgdConfig {
                lr: 0.01,
                momentum: 0.9,
                dampening: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut opt_damped = OptimizerKind::sgd(
            vec![x_damped.clone()],
            SgdConfig {
                lr: 0.01,
                momentum: 0.9,
                dampening: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..2 {
            opt_no_damp.step(&grad_x_squared(&x_no_damp)).unwrap();
            opt_damped.step(&grad_x_squared(&x_damped)).unwrap();
        }
        let undamped = var_val(&x_no_damp);
        let damped = var_val(&x_damped);
        assert!(
            (undamped - 2.8272).abs() < 1e-10,
            "expected 2.8272, got {undamped}"
        );
        assert!(
            (damped - 2.8566).abs() < 1e-10,
            "expected 2.8566, got {damped}"
        );
        assert!(
            undamped < damped,
            "dampening should reduce the second momentum update"
        );
    }

    #[test]
    fn sgd_nesterov_converges_faster() {
        let x_mom = make_var(5.0);
        let x_nes = make_var(5.0);
        let cfg_base = SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            ..Default::default()
        };
        let mut opt_mom = OptimizerKind::sgd(vec![x_mom.clone()], cfg_base.clone()).unwrap();
        let mut opt_nes = OptimizerKind::sgd(
            vec![x_nes.clone()],
            SgdConfig {
                nesterov: true,
                ..cfg_base
            },
        )
        .unwrap();
        for _ in 0..30 {
            opt_mom.step(&grad_x_squared(&x_mom)).unwrap();
            opt_nes.step(&grad_x_squared(&x_nes)).unwrap();
        }
        assert!(
            var_val(&x_nes).abs() <= var_val(&x_mom).abs() + 0.05,
            "Nesterov should converge at least as fast"
        );
    }

    #[test]
    fn sgd_weight_decay() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::sgd(
            vec![x.clone()],
            SgdConfig {
                lr: 0.1,
                weight_decay: 0.01,
                ..Default::default()
            },
        )
        .unwrap();
        opt.step(&grad_x_squared(&x)).unwrap();
        let v = var_val(&x);
        let expected = 3.0 - 0.1 * (6.0 + 0.01 * 3.0);
        assert!((v - expected).abs() < 1e-10, "expected {expected}, got {v}");
    }

    // -----------------------------------------------------------------------
    // RMSprop step  (US-5.7)
    // -----------------------------------------------------------------------

    #[test]
    fn rmsprop_step_decreases_loss() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::rmsprop(vec![x.clone()], RMSpropConfig::default()).unwrap();
        for _ in 0..20 {
            opt.step(&grad_x_squared(&x)).unwrap();
        }
        assert!(var_val(&x).abs() < 3.0, "RMSprop should reduce x");
    }

    #[test]
    fn rmsprop_centered() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::rmsprop(
            vec![x.clone()],
            RMSpropConfig {
                centered: true,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..20 {
            opt.step(&grad_x_squared(&x)).unwrap();
        }
        assert!(var_val(&x).abs() < 3.0, "centred RMSprop should reduce x");
    }

    #[test]
    fn rmsprop_with_momentum() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::rmsprop(
            vec![x.clone()],
            RMSpropConfig {
                momentum: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..20 {
            opt.step(&grad_x_squared(&x)).unwrap();
        }
        assert!(var_val(&x).abs() < 3.0);
    }

    #[test]
    fn rmsprop_weight_decay() {
        let x_wd = make_var(5.0);
        let x_no = make_var(5.0);
        let mut opt_wd = OptimizerKind::rmsprop(
            vec![x_wd.clone()],
            RMSpropConfig {
                weight_decay: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        let mut opt_no =
            OptimizerKind::rmsprop(vec![x_no.clone()], RMSpropConfig::default()).unwrap();
        for _ in 0..20 {
            opt_wd.step(&grad_x_squared(&x_wd)).unwrap();
            opt_no.step(&grad_x_squared(&x_no)).unwrap();
        }
        assert!(var_val(&x_wd).abs() < var_val(&x_no).abs());
    }

    // -----------------------------------------------------------------------
    // Unified interface
    // -----------------------------------------------------------------------

    #[test]
    fn name_returns_correct_variant() {
        let v = make_var(1.0);
        assert_eq!(
            OptimizerKind::adamw(vec![v.clone()], AdamWConfig::default())
                .unwrap()
                .name(),
            "AdamW"
        );
        assert_eq!(
            OptimizerKind::adam(vec![v.clone()], AdamConfig::default())
                .unwrap()
                .name(),
            "Adam"
        );
        assert_eq!(
            OptimizerKind::sgd(vec![v.clone()], SgdConfig::default())
                .unwrap()
                .name(),
            "SGD"
        );
        assert_eq!(
            OptimizerKind::rmsprop(vec![v.clone()], RMSpropConfig::default())
                .unwrap()
                .name(),
            "RMSprop"
        );
    }

    #[test]
    fn from_config_builds_expected_variant() {
        let v = make_var(1.0);
        let opt =
            OptimizerKind::from_config(vec![v], OptimizerConfig::AdamW(AdamWConfig::default()))
                .unwrap();
        assert_eq!(opt.name(), "AdamW");
    }

    #[test]
    fn display_includes_params() {
        let v = make_var(1.0);
        let opt = OptimizerKind::adamw(vec![v], AdamWConfig::default()).unwrap();
        let s = opt.to_string();
        assert!(s.contains("AdamW"));
        assert!(s.contains("0.001")); // lr
    }

    #[test]
    fn set_learning_rate_works() {
        let v = make_var(1.0);
        let mut opt = OptimizerKind::sgd(vec![v], SgdConfig::default()).unwrap();
        assert_eq!(opt.learning_rate(), 0.01);
        opt.set_learning_rate(0.05);
        assert_eq!(opt.learning_rate(), 0.05);
    }

    #[test]
    fn backward_step_convenience() {
        let x = make_var(3.0);
        let mut opt = OptimizerKind::adam(vec![x.clone()], AdamConfig::default()).unwrap();
        let loss = x.as_tensor().sqr().unwrap().sum_all().unwrap();
        opt.backward_step(&loss).unwrap();
        assert!(var_val(&x).abs() < 3.0);
    }

    // -----------------------------------------------------------------------
    // L2 regularisation helper
    // -----------------------------------------------------------------------

    #[test]
    fn l2_regularization_value() {
        let x = make_var(3.0);
        let reg = l2_regularization(&[x], 0.01).unwrap();
        let v: f64 = reg.to_vec0().unwrap();
        assert!((v - 0.045).abs() < 1e-10);
    }

    #[test]
    fn l2_regularization_multiple_vars() {
        let x = make_var(3.0); // 9
        let y = make_var(4.0); // 16
        // 0.5 * 0.1 * (9 + 16) = 1.25
        let reg = l2_regularization(&[x, y], 0.1).unwrap();
        let v: f64 = reg.to_vec0().unwrap();
        assert!((v - 1.25).abs() < 1e-10);
    }

    #[test]
    fn l2_regularization_empty_vars() {
        let r = l2_regularization(&[], 0.01);
        assert!(r.is_err());
    }

    // -----------------------------------------------------------------------
    // Multi-variable optimisation scenario
    // -----------------------------------------------------------------------

    #[test]
    fn multi_var_optimisation() {
        let x = make_var(5.0);
        let y = make_var(-3.0);
        let mut opt = OptimizerKind::adamw(
            vec![x.clone(), y.clone()],
            AdamWConfig {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..200 {
            let loss = (x.as_tensor().sqr().unwrap() + y.as_tensor().sqr().unwrap()).unwrap();
            let loss = loss.sum_all().unwrap();
            opt.backward_step(&loss).unwrap();
        }
        assert!(var_val(&x).abs() < 0.5, "x should be near 0");
        assert!(var_val(&y).abs() < 0.5, "y should be near 0");
    }
}
