//! Web/CLI-agnostic training form schema.
//!
//! `TrainForm` is the canonical deserialization target for a training
//! submission — HTTP POST bodies, TOML configs, and future FFI callers
//! all deserialize into the same struct. The web route used to define
//! this locally, which drifted from [`TrainConfig`] every time a field
//! was added; centralizing both the form shape AND the
//! `TrainForm → TrainConfig` mapping here eliminates that drift.

use serde::{Deserialize, Deserializer};

use sparam_data::generation::PermittivitySample;
use sparam_models::ModelType;

use super::config::{ModelConfig, StackedTrainConfig, TrainConfig};

/// Treat empty-string form values as `None` — HTML hidden inputs
/// submit `name=` when unset, which `serde_urlencoded` otherwise
/// rejects when deserialising into `Option<f64>` etc.
pub fn empty_string_as_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    match opt.as_deref() {
        None | Some("") => Ok(None),
        Some(s) => T::from_str(s).map(Some).map_err(serde::de::Error::custom),
    }
}

/// Error returned when `TrainForm::build` can't map user-submitted
/// strings to a valid [`TrainConfig`] (bad activation name, unknown
/// `model_type`, etc.).
#[derive(Debug)]
pub struct TrainFormError(pub String);

impl std::fmt::Display for TrainFormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrainFormError {}

/// Training submission form — shared between the web `/train` POST
/// handler and anywhere else that accepts a user-submitted training
/// spec. Deserializes cleanly from HTTP form bodies thanks to the
/// `empty_string_as_none` helper; the HPO-parity hidden inputs send
/// empty strings when no HPO prefill is active.
#[derive(Deserialize)]
pub struct TrainForm {
    pub dataset_id: i64,
    // Model
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_activation")]
    pub activation: String,
    #[serde(default = "default_complex_activation")]
    pub complex_activation: String,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default = "default_norm")]
    pub norm: String,
    // Optimizer
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    // Training
    #[serde(default = "default_max_epochs")]
    pub max_epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: String,
    #[serde(default = "default_patience")]
    pub patience: usize,
    #[serde(default)]
    pub warmup_epochs: usize,
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_loss")]
    pub loss: String,
    #[serde(default = "default_seed")]
    pub seed: u64,

    // HPO-parity fields submitted as hidden inputs, populated by
    // `?prefill=` on "Retrain from trial". Manual submissions leave
    // them `None`. `empty_string_as_none` handles the empty-value
    // form bodies the hidden inputs send when no prefill is present.
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub shuffle_seed: Option<u64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub grad_clip_norm: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub input_noise_std: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub cosine_eta_min: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub plateau_factor: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub plateau_patience: Option<usize>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub plateau_min_lr: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub sgd_momentum: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub sgd_nesterov: Option<bool>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub rmsprop_momentum: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub rmsprop_alpha: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub pi_mape_beta: Option<f64>,
    /// HPO trial's sampled λ for the hybrid PhysicsForward loss.
    /// Only meaningful when `loss = "physics_forward"`; absent for
    /// MSE / SmoothL1 trials. Empty string from the form falls back to
    /// the workflow default (`DEFAULT_PHYSICS_LAMBDA = 0.001`).
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub physics_lambda: Option<f64>,

    // ─── stacked pipeline toggles (each opt-in, default-off) ───────────────────
    //
    // Match the [`StackedTrainConfig`] fields one-for-one. The web UI
    // surfaces these as checkboxes/numeric inputs labelled "Enable
    // train-overlay anchor", "Median ensemble size", "NRW refinement
    // steps", and "Refinement learning rate". When all are at their
    // defaults the form builds a passthrough stacked config (which
    // delegates to plain `run_training`), so existing submissions
    // behave identically.
    /// Train-overlay-only density. `0` = no overlay. winning sweet spot: `8`.
    #[serde(default)]
    pub m10_train_overlay_density: usize,
    /// Median-ensemble size. `1` = single model. winning sweet spot: `3`.
    #[serde(default = "default_ensemble_size")]
    pub m10_ensemble_size: usize,
    /// Inference-time NRW refinement step count. `0` = no refinement.
    /// winning sweet spot: `100`.
    #[serde(default)]
    pub m10_refine_steps: usize,
    /// Adam learning rate for the inference-time refinement loop.
    /// Ignored when `m10_refine_steps == 0`. winning sweet spot: `1e-3`.
    #[serde(default = "default_refine_lr")]
    pub m10_refine_lr: f64,
}

fn default_model_type() -> String { "real".into() }
fn default_hidden_size() -> usize { 64 }
fn default_activation() -> String { "gelu".into() }
fn default_complex_activation() -> String { "crelu".into() }
fn default_norm() -> String { "none".into() }
fn default_optimizer() -> String { "adamw".into() }
fn default_lr() -> f64 { 1e-3 }
fn default_weight_decay() -> f64 { 0.01 }
fn default_max_epochs() -> usize { 200 }
fn default_batch_size() -> String { "64".into() }
fn default_patience() -> usize { 20 }
fn default_scheduler() -> String { "cosine".into() }
fn default_seed() -> u64 { 42 }
fn default_loss() -> String { "mse".into() }
fn default_ensemble_size() -> usize { 1 }
fn default_refine_lr() -> f64 { 1e-3 }

/// Samples carried alongside the form, loaded from whichever source
/// the caller uses (DB for the web, CSV for the CLI). `Arc<[_]>` so
/// cloning the splits into a workflow config is an atomic bump, not
/// an O(n) copy — a "both"-model HPO study shares a single backing
/// allocation across its Real and Complex sub-studies.
pub struct TrainSplits {
    pub train: std::sync::Arc<[PermittivitySample]>,
    pub val: std::sync::Arc<[PermittivitySample]>,
    pub test: std::sync::Arc<[PermittivitySample]>,
}

/// Output of `TrainForm::build` — carries both the workflow-ready
/// [`TrainConfig`] and the same payload pre-serialized as JSON, so
/// the caller can persist it verbatim to DB without re-mapping.
pub struct BuiltTrainSubmission {
    pub model_name: String,
    pub config: TrainConfig,
    pub config_json: String,
}

/// Output of [`TrainForm::build_stacked`] — same as
/// [`BuiltTrainSubmission`] but with the stacked pipeline toggles attached
/// via [`StackedTrainConfig`]. When all the toggles are at their
/// defaults the wrapper is a passthrough and downstream calls to
/// [`crate::workflows::stacked_training::run_stacked_training`] delegate to plain
/// [`crate::workflows::train::run_training`] — so existing callers that
/// don't care about the stack see no behaviour change.
pub struct BuiltStackedTrainSubmission {
    pub model_name: String,
    pub config: StackedTrainConfig,
    pub config_json: String,
}

impl TrainForm {
    /// Validate the form, build the [`TrainConfig`], and emit the
    /// matching JSON payload for DB persistence. Single source of
    /// truth for mapping user-submitted strings to workflow types —
    /// adding a field to `TrainConfig` requires updating this method
    /// (not duplicating the logic per caller).
    pub fn build(self, splits: TrainSplits) -> Result<BuiltTrainSubmission, TrainFormError> {
        let parsed_model_type: ModelType = self
            .model_type
            .parse()
            .map_err(TrainFormError)?;

        let model_config = match parsed_model_type {
            // Complex and Real both accept `dropout` from the form —
            // the Complex path applies pair-wise dropout after the
            // activation (whole (real, imag) pairs dropped together).
            // `norm` and the real-only activations stay Real-only.
            ModelType::Complex => ModelConfig::complex(
                self.hidden_size,
                &self.complex_activation,
                self.dropout,
            ),
            ModelType::Real => ModelConfig::real(
                self.hidden_size,
                &self.activation,
                self.dropout,
                &self.norm,
            ),
        }
        .map_err(|e| TrainFormError(format!("invalid activation: {e}")))?;

        let model_name = model_config.arch_description();

        // Emit the persisted JSON from the `ModelConfig` enum itself —
        // Complex runs carry only `{model_type, hidden_size,
        // complex_activation}`, Real carries dropout / norm. Training
        // hyperparams are appended verbatim from the form.
        let model_json = serde_json::to_value(&model_config)
            .map_err(|e| TrainFormError(format!("serialize model config: {e}")))?;
        let mut cfg = serde_json::Map::new();
        if let serde_json::Value::Object(model_obj) = model_json {
            cfg.extend(model_obj);
        }
        cfg.insert("optimizer".into(), self.optimizer.clone().into());
        cfg.insert("lr".into(), self.lr.into());
        cfg.insert("weight_decay".into(), self.weight_decay.into());
        cfg.insert("max_epochs".into(), self.max_epochs.into());
        cfg.insert("batch_size".into(), self.batch_size.clone().into());
        cfg.insert("patience".into(), self.patience.into());
        cfg.insert("warmup_epochs".into(), self.warmup_epochs.into());
        cfg.insert("scheduler".into(), self.scheduler.clone().into());
        cfg.insert("loss".into(), self.loss.clone().into());
        cfg.insert("seed".into(), self.seed.into());
        cfg.insert("shuffle_seed".into(), to_json(&self.shuffle_seed)?);
        cfg.insert("grad_clip_norm".into(), to_json(&self.grad_clip_norm)?);
        cfg.insert("input_noise_std".into(), to_json(&self.input_noise_std)?);
        cfg.insert("cosine_eta_min".into(), to_json(&self.cosine_eta_min)?);
        cfg.insert("plateau_factor".into(), to_json(&self.plateau_factor)?);
        cfg.insert("plateau_patience".into(), to_json(&self.plateau_patience)?);
        cfg.insert("plateau_min_lr".into(), to_json(&self.plateau_min_lr)?);
        cfg.insert("sgd_momentum".into(), to_json(&self.sgd_momentum)?);
        cfg.insert("sgd_nesterov".into(), to_json(&self.sgd_nesterov)?);
        cfg.insert("rmsprop_momentum".into(), to_json(&self.rmsprop_momentum)?);
        cfg.insert("rmsprop_alpha".into(), to_json(&self.rmsprop_alpha)?);
        cfg.insert("pi_mape_beta".into(), to_json(&self.pi_mape_beta)?);
        cfg.insert("physics_lambda".into(), to_json(&self.physics_lambda)?);
        // stacked pipeline toggles — persisted in the config JSON so a
        // training run is fully reproducible from its DB row, including
        // the stack interventions chosen at submission time.
        cfg.insert("m10_train_overlay_density".into(), self.m10_train_overlay_density.into());
        cfg.insert("m10_ensemble_size".into(), self.m10_ensemble_size.into());
        cfg.insert("m10_refine_steps".into(), self.m10_refine_steps.into());
        cfg.insert("m10_refine_lr".into(), self.m10_refine_lr.into());
        let config_json = serde_json::to_string(&serde_json::Value::Object(cfg))
            .map_err(|e| TrainFormError(format!("serialize config json: {e}")))?;

        let config = TrainConfig {
            train_samples: splits.train,
            val_samples: splits.val,
            test_samples: splits.test,
            checkpoint_path: None,
            model: model_config,
            optimizer: self.optimizer,
            lr: self.lr,
            weight_decay: self.weight_decay,
            batch_size: self.batch_size,
            max_epochs: self.max_epochs,
            patience: self.patience,
            warmup_epochs: self.warmup_epochs,
            scheduler: self.scheduler,
            loss: self.loss,
            checkpoint: None,
            seed: self.seed,
            log_progress: false,
            shuffle_seed: self.shuffle_seed,
            grad_clip_norm: self.grad_clip_norm,
            input_noise_std: self.input_noise_std,
            cosine_eta_min: self.cosine_eta_min,
            plateau_factor: self.plateau_factor,
            plateau_patience: self.plateau_patience,
            plateau_min_lr: self.plateau_min_lr,
            sgd_momentum: self.sgd_momentum,
            sgd_nesterov: self.sgd_nesterov,
            rmsprop_momentum: self.rmsprop_momentum,
            rmsprop_alpha: self.rmsprop_alpha,
            pi_mape_beta: self.pi_mape_beta,
            physics_lambda: self.physics_lambda,
        };

        Ok(BuiltTrainSubmission { model_name, config, config_json })
    }

    /// Same as [`Self::build`] but also assembles a
    /// [`StackedTrainConfig`] using the stack toggles on the form. Use
    /// this when the caller is ready to run the stacked pipeline
    /// (`crate::workflows::stacked_training::run_stacked_training`); use [`Self::build`]
    /// for the legacy path that bypasses the stack entirely.
    ///
    /// The persisted `config_json` is the same in both cases (already
    /// includes the stack toggle fields), so the DB row records the
    /// chosen stack regardless of which entry point ran the training.
    pub fn build_stacked(
        self,
        splits: TrainSplits,
    ) -> Result<BuiltStackedTrainSubmission, TrainFormError> {
        let m10_train_overlay_density = self.m10_train_overlay_density;
        let m10_ensemble_size = self.m10_ensemble_size;
        let m10_refine_steps = self.m10_refine_steps;
        let m10_refine_lr = self.m10_refine_lr;
        let BuiltTrainSubmission { model_name, config, config_json } = self.build(splits)?;
        let stacked = StackedTrainConfig {
            base: config,
            train_overlay_density: m10_train_overlay_density,
            ensemble_size: m10_ensemble_size.max(1),
            refine_steps: m10_refine_steps,
            refine_lr: m10_refine_lr,
        };
        Ok(BuiltStackedTrainSubmission {
            model_name,
            config: stacked,
            config_json,
        })
    }
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<serde_json::Value, TrainFormError> {
    serde_json::to_value(value)
        .map_err(|e| TrainFormError(format!("serialize field: {e}")))
}

#[cfg(test)]
mod tests {
    use super::TrainForm;

    /// No-prefill form submits `name=` for every hidden HPO-parity
    /// input; all must deserialise to `None` instead of 422.
    #[test]
    fn form_with_empty_hpo_parity_fields_deserialises_to_none() {
        let body = [
            ("dataset_id", "1"),
            ("model_type", "real"),
            ("hidden_size", "32"),
            ("activation", "gelu"),
            ("norm", "none"),
            ("optimizer", "adamw"),
            ("lr", "0.001"),
            ("weight_decay", "0.01"),
            ("batch_size", "64"),
            ("max_epochs", "10"),
            ("patience", "20"),
            ("warmup_epochs", "0"),
            ("scheduler", "cosine"),
            ("loss", "mse"),
            ("seed", "42"),
            ("grad_clip_norm", ""),
            ("input_noise_std", ""),
            ("cosine_eta_min", ""),
            ("plateau_factor", ""),
            ("plateau_patience", ""),
            ("plateau_min_lr", ""),
            ("sgd_momentum", ""),
            ("sgd_nesterov", ""),
            ("rmsprop_momentum", ""),
            ("rmsprop_alpha", ""),
        ];
        let encoded = body
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let form: TrainForm = serde_urlencoded::from_str(&encoded)
            .expect("empty HPO-parity fields must deserialise as None, not error");

        assert!(form.grad_clip_norm.is_none());
        assert!(form.input_noise_std.is_none());
        assert!(form.cosine_eta_min.is_none());
        assert!(form.plateau_factor.is_none());
        assert!(form.plateau_patience.is_none());
        assert!(form.plateau_min_lr.is_none());
        assert!(form.sgd_momentum.is_none());
        assert!(form.sgd_nesterov.is_none());
        assert!(form.rmsprop_momentum.is_none());
        assert!(form.rmsprop_alpha.is_none());
    }

    /// Prefill-populated values round-trip correctly.
    #[test]
    fn form_with_populated_hpo_parity_fields_deserialises_correctly() {
        let body = [
            ("dataset_id", "1"),
            ("model_type", "complex"),
            ("hidden_size", "48"),
            ("complex_activation", "cswish_phase"),
            ("optimizer", "adamw"),
            ("lr", "0.0284"),
            ("weight_decay", "0.000459"),
            ("batch_size", "64"),
            ("max_epochs", "25"),
            ("patience", "10"),
            ("warmup_epochs", "10"),
            ("scheduler", "none"),
            ("loss", "mse"),
            ("seed", "270"),
            ("grad_clip_norm", "1.0"),
            ("input_noise_std", "0"),
            ("cosine_eta_min", ""),
            ("plateau_factor", ""),
            ("plateau_patience", ""),
            ("plateau_min_lr", ""),
            ("sgd_momentum", ""),
            ("sgd_nesterov", ""),
            ("rmsprop_momentum", ""),
            ("rmsprop_alpha", ""),
        ];
        let encoded = body
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let form: TrainForm = serde_urlencoded::from_str(&encoded).unwrap();

        assert_eq!(form.grad_clip_norm, Some(1.0));
        assert_eq!(form.input_noise_std, Some(0.0));
    }
}
