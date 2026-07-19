//! Concrete MLP evaluation pipeline for HPO trials.
//!
//! Provides concrete model builders, tensor data sources, and
//! [`RegressionEvaluator`] implementations that wire sampled [`HyperParams`]
//! into the existing Candle-based trainer.

use std::sync::OnceLock;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ModuleT, VarBuilder, VarMap};

use sparam_data::loader::{BatchSize, DataLoader};
use sparam_data::physical_constraint::apply_physical_softplus_clamp;
use sparam_data::scaling::ScalerRef;
use sparam_data::tensor_bridges::{pack_complex_features, unpack_complex_features};
use sparam_models::{MLPRegressor, MlpModel, ModelType};
use sparam_training::losses::PhysicsContext;
use sparam_training::losses::{
    DEFAULT_SMOOTH_L1_BETA, Reduction, complex_mse_loss, complex_relative_error_elements,
    complex_smooth_l1_loss, mse_loss, physics_forward_loss, smooth_l1_loss,
    validate_same_complex_shape, validate_same_shape,
};
// Post-training `(OK@1%, max_error)` shares the same scalar
// `f64::hypot` path as `/train` so retrains reproduce the HPO
// value bit-for-bit. See `compute_validation_objectives` below.
use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::error::candle_msg;
use sparam_core::metrics::{classify_predictions, relative_error_metrics_from_components};
use sparam_training::logger::LogSender;

use crate::evaluation::{
    DataSource, Evaluator, ModelBuilder, TrialMetrics, TrialOutcome, TrialStatus,
};
use crate::{HyperParams, LossChoice};

type PackedLossFn<'a> = Box<dyn Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + 'a>;

/// Default weight of the physics term in the hybrid PhysicsForward
/// loss when the trial doesn't carry an explicit
/// [`LossHyperParams::physics_lambda`]. Used as the fallback for
/// legacy trials persisted before `physics_lambda` became a sampled
/// hyperparameter and for retrains where the field is omitted from
/// the bridge JSON.
const DEFAULT_PHYSICS_LAMBDA: f64 = 0.001;

/// Relative-error loss — inverse-scales both sides then divides by
/// `|target|` directly. The dataset enforces `ε' ≥ 1` so the
/// denominator is always ≥ 1; no `max(|target|, ε)` floor.
#[derive(Clone, Copy)]
pub struct RelativeErrorLoss<'a> {
    scaler: ScalerRef<'a>,
    reduction: Reduction,
}

impl<'a> RelativeErrorLoss<'a> {
    #[must_use]
    pub fn new<S>(scaler: S) -> Self
    where
        S: Into<ScalerRef<'a>>,
    {
        Self {
            scaler: scaler.into(),
            reduction: Reduction::Mean,
        }
    }

    /// Access the captured target scaler — needed by the
    /// `packed_loss_fn` PhysicsForward branch so it can run the
    /// scaler inverse-transform itself (the NRW forward model lives
    /// outside `RelativeErrorLoss`, so it can't route through
    /// `forward_complex` / `forward`).
    #[must_use]
    pub fn scaler_ref(&self) -> ScalerRef<'a> {
        self.scaler
    }

    #[cfg(any(test, doctest))]
    #[allow(dead_code)]
    #[must_use]
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    /// Squared relative error on 2-column real predictions, treating the
    /// columns as `(Re, Im)` of a complex permittivity (matches Python's
    /// `eps_rel_mse`: magnitude-based denominator so near-zero `ε''` doesn't
    /// explode the loss).
    pub fn forward(&self, pred_scaled: &Tensor, target_scaled: &Tensor) -> Result<Tensor> {
        validate_same_shape(pred_scaled, target_scaled, "RelativeErrorLoss::forward")?;

        let pred_unscaled = self.scaler.inverse_transform(pred_scaled)?;
        let target_unscaled = self.scaler.inverse_transform(target_scaled)?;

        let complex_feature_count =
            packed_complex_feature_count(&target_unscaled, "RelativeErrorLoss::forward")?;
        let pred = unpack_complex_features(&pred_unscaled, complex_feature_count)?;
        let target = unpack_complex_features(&target_unscaled, complex_feature_count)?;
        let rel_err = complex_relative_error_elements(&pred, &target)?;
        let squared = rel_err.sqr()?;
        self.reduction.reduce(&squared)
    }

    /// Compute squared relative error for complex-valued predictions after
    /// inverse scaling. See [`forward`](Self::forward) for the scalar case.
    pub fn forward_complex(
        &self,
        pred_scaled: &ComplexTensor,
        target_scaled: &ComplexTensor,
    ) -> Result<Tensor> {
        validate_same_complex_shape(
            pred_scaled,
            target_scaled,
            "RelativeErrorLoss::forward_complex",
        )?;

        let dims = pred_scaled.real.dims();
        if dims.len() != 2 {
            return Err(candle_msg(format!(
                "RelativeErrorLoss::forward_complex expects 2D complex tensors shaped (batch, features), got {:?}",
                dims
            )));
        }
        let complex_feature_count = dims[1];
        let pred_packed = pack_complex_features(pred_scaled)?;
        let target_packed = pack_complex_features(target_scaled)?;

        let pred = unpack_complex_features(
            &self.scaler.inverse_transform(&pred_packed)?,
            complex_feature_count,
        )?;
        let target = unpack_complex_features(
            &self.scaler.inverse_transform(&target_packed)?,
            complex_feature_count,
        )?;
        let rel_err = complex_relative_error_elements(&pred, &target)?;
        let squared = rel_err.sqr()?;
        self.reduction.reduce(&squared)
    }

    /// Physics-informed loss against pre-computed target S-parameters.
    /// Inverse-scales the packed `pred` via the target scaler, unpacks
    /// to a complex `ε̂`, and delegates to [`physics_forward_loss`] —
    /// one pack-friendly path shared by both Real and Complex MLPs.
    pub fn forward_physics(
        &self,
        pred_scaled: &Tensor,
        s_target: &ComplexTensor,
        ctx: &PhysicsContext,
    ) -> Result<Tensor> {
        let pred_unscaled = self.scaler.inverse_transform(pred_scaled)?;
        let n = packed_complex_feature_count(&pred_unscaled, "RelativeErrorLoss::forward_physics")?;
        let pred = unpack_complex_features(&pred_unscaled, n)?;
        physics_forward_loss(
            &pred,
            s_target,
            &ctx.waveguide,
            &ctx.frequencies,
            self.reduction,
        )
    }
}

fn packed_complex_feature_count(values: &Tensor, context: &str) -> Result<usize> {
    let dims = values.dims();
    if dims.len() != 2 {
        return Err(candle_msg(format!(
            "{context} expects a 2D packed tensor shaped (batch, scalar_features), got {:?}",
            dims
        )));
    }

    if dims[1] % 2 != 0 {
        return Err(candle_msg(format!(
            "{context} expects an even number of scalar features so they can be unpacked into complex pairs, got {}",
            dims[1]
        )));
    }

    Ok(dims[1] / 2)
}

fn concat_batches(mut tensors: Vec<Tensor>, context: &str) -> Result<Tensor> {
    match tensors.len() {
        0 => Err(candle_msg(format!("{context} did not produce any batches"))),
        1 => Ok(tensors.pop().expect("single tensor present")),
        _ => {
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Tensor::cat(&refs, 0)
        }
    }
}

fn collect_validation_predictions<M>(model: &M, loader: &DataLoader) -> Result<(Tensor, Tensor)>
where
    M: ModuleT,
{
    let n = loader.num_batches();
    let mut predictions = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);

    for batch in loader.iter() {
        let (inputs, target) = batch?;
        predictions.push(model.forward_t(&inputs, false)?);
        targets.push(target);
    }

    Ok((
        concat_batches(predictions, "validation predictions")?,
        concat_batches(targets, "validation targets")?,
    ))
}

fn compute_validation_objectives<M>(
    model: &M,
    loader: &DataLoader,
    target_scaler: Option<ScalerRef<'_>>,
    is_complex: bool,
) -> Result<(f64, f64)>
where
    M: ModuleT,
{
    let (predictions, targets) = collect_validation_predictions(model, loader)?;
    // Apply the same architectural softplus clamp the model trained
    // under (matches the loss closure in `packed_loss_factory`), then
    // inverse-transform for the metric kernels. Without this, the
    // recorded objectives would be measured on a different output
    // distribution than the model was optimised against.
    let predictions = match target_scaler {
        Some(scaler) => {
            let clamped = apply_physical_softplus_clamp(&predictions, scaler, is_complex)?;
            scaler.inverse_transform(&clamped)?
        }
        None => predictions,
    };
    let targets = match target_scaler {
        Some(scaler) => scaler.inverse_transform(&targets)?,
        None => targets,
    };

    // Scalar `f64::hypot` path so `max_error` matches `/train`'s
    // `evaluate_model_predictions` bit-for-bit.
    let complex_feature_count = packed_complex_feature_count(&targets, "validation targets")?;
    let pred_complex = unpack_complex_features(&predictions, complex_feature_count)?;
    let target_complex = unpack_complex_features(&targets, complex_feature_count)?;

    let true_real = target_complex.real.flatten_all()?.to_vec1::<f64>()?;
    let true_imag = target_complex.imag.flatten_all()?.to_vec1::<f64>()?;
    let pred_real = pred_complex.real.flatten_all()?.to_vec1::<f64>()?;
    let pred_imag = pred_complex.imag.flatten_all()?.to_vec1::<f64>()?;

    let relative_error =
        relative_error_metrics_from_components(&true_real, &true_imag, &pred_real, &pred_imag)?;
    let classification = classify_predictions(&relative_error.errors, 1.0);
    Ok((classification.ok_percent, relative_error.max_error))
}

/// Compose a packed-tensor loss closure from a pair of kernels —
/// `real_fn` applies directly to packed tensors; `complex_fn` is
/// applied after complex unpacking. `complex_output_size` selects
/// between them: `None` → Real (no unpack), `Some(n)` → Complex
/// (unpack `n` features first). The composed closure takes the
/// batch `input` as its first argument but the standard residual
/// losses ignore it — only `LossChoice::PhysicsForward` consumes it
/// (via its own dedicated branch in `packed_loss_factory`).
fn packed_loss_from_pair<'a, R, C>(
    complex_output_size: Option<usize>,
    real_fn: R,
    complex_fn: C,
) -> PackedLossFn<'a>
where
    R: Fn(&Tensor, &Tensor) -> Result<Tensor> + 'a,
    C: Fn(
            &sparam_core::complex_tensor::ComplexTensor,
            &sparam_core::complex_tensor::ComplexTensor,
        ) -> Result<Tensor>
        + 'a,
{
    match complex_output_size {
        None => Box::new(move |_input, pred, target| real_fn(pred, target)),
        Some(n) => Box::new(move |_input, pred, target| {
            let pred = unpack_complex_features(pred, n)?;
            let target = unpack_complex_features(target, n)?;
            complex_fn(&pred, &target)
        }),
    }
}

/// Single source of truth for packed-loss construction. Takes
/// `complex_output_size = None` for Real models and `Some(n)` for
/// Complex models (where `n` is the Complex MLP's complex-feature
/// count, i.e. its `output_size()`). `feature_scaler` and `physics`
/// together enable `LossChoice::PhysicsForward` (the former recovers
/// the batch's raw S-parameters from the scaled model input; the
/// latter supplies the NRW waveguide + frequency context). Missing
/// either falls back to `complex_mse_loss`.
fn packed_loss_factory<'a>(
    complex_output_size: Option<usize>,
    loss: LossChoice,
    physics_lambda: f64,
    feature_scaler: Option<ScalerRef<'a>>,
    target_scaler: Option<ScalerRef<'a>>,
    physics: Option<&'a PhysicsContext>,
) -> PackedLossFn<'a> {
    let inner = packed_loss_factory_inner(
        complex_output_size,
        loss,
        physics_lambda,
        feature_scaler,
        target_scaler,
        physics,
    );
    // Always-on architectural softplus clamp on the model output.
    // Wraps the inner loss closure so every loss kernel sees a
    // predicted ε that satisfies ε′ ≥ 1 and the right ε″ sign for the
    // model family. Falls back to the raw `pred` only when no target
    // scaler was wired through (programmatic test paths).
    match target_scaler {
        Some(scaler) => {
            let is_complex = complex_output_size.is_some();
            Box::new(move |input, pred, target| {
                let pred = apply_physical_softplus_clamp(pred, scaler, is_complex)?;
                inner(input, &pred, target)
            })
        }
        None => inner,
    }
}

fn packed_loss_factory_inner<'a>(
    complex_output_size: Option<usize>,
    loss: LossChoice,
    physics_lambda: f64,
    feature_scaler: Option<ScalerRef<'a>>,
    target_scaler: Option<ScalerRef<'a>>,
    physics: Option<&'a PhysicsContext>,
) -> PackedLossFn<'a> {
    match loss {
        LossChoice::Mse => packed_loss_from_pair(
            complex_output_size,
            |p, t| mse_loss(p, t, Reduction::Mean),
            |p, t| complex_mse_loss(p, t, Reduction::Mean),
        ),
        LossChoice::SmoothL1 => packed_loss_from_pair(
            complex_output_size,
            |p, t| smooth_l1_loss(p, t, DEFAULT_SMOOTH_L1_BETA, Reduction::Mean),
            |p, t| complex_smooth_l1_loss(p, t, DEFAULT_SMOOTH_L1_BETA, Reduction::Mean),
        ),
        LossChoice::EpsRelMse => match target_scaler {
            Some(scaler) => {
                // `RelativeErrorLoss` is `Copy` (holds just a `ScalerRef`),
                // so the same `loss` value moves freely into both closures.
                let loss = RelativeErrorLoss::new(scaler);
                packed_loss_from_pair(
                    complex_output_size,
                    move |p, t| loss.forward(p, t),
                    move |p, t| loss.forward_complex(p, t),
                )
            }
            // Graceful fallback — EpsRelMse without a fitted target
            // scaler degrades to plain MSE instead of failing, so an
            // HPO trial that hasn't wired up scalers still produces a
            // meaningful training signal.
            None => packed_loss_from_pair(
                complex_output_size,
                |p, t| mse_loss(p, t, Reduction::Mean),
                |p, t| complex_mse_loss(p, t, Reduction::Mean),
            ),
        },
        LossChoice::PhysicsForward => match (target_scaler, feature_scaler, physics) {
            (Some(t_scaler), Some(f_scaler), Some(ctx)) => {
                let rel = RelativeErrorLoss::new(t_scaler);
                // Hybrid PINN loss, matches
                // `sparam_app::workflows::internal::training`:
                //   L = MSE(ε̂, ε)  +  λ · physics_forward(NRW(ε̂), S_true)
                // Real and Complex paths share the same closure —
                // MSE runs in scaled space (cheap direct comparison
                // of pred/target), then `forward_physics` handles
                // the target-scaler round-trip + NRW for the
                // physics term. `HYBRID_PHYSICS_LAMBDA = 0.1` keeps
                // training stable; pure physics_forward was prone to
                // mid-training basin swaps due to NRW's
                // non-injectivity.
                match complex_output_size {
                    None => Box::new(move |input, pred, target| {
                        let mse = mse_loss(pred, target, Reduction::Mean)?;
                        let s_raw = f_scaler.inverse_transform(input)?;
                        let s_target = unpack_complex_features(&s_raw, 2)?;
                        let phys = rel.forward_physics(pred, &s_target, ctx)?;
                        mse.broadcast_add(&phys.affine(physics_lambda, 0.0)?)
                    }),
                    Some(n) => Box::new(move |input, pred, target| {
                        let p = unpack_complex_features(pred, n)?;
                        let t = unpack_complex_features(target, n)?;
                        let mse = complex_mse_loss(&p, &t, Reduction::Mean)?;
                        let s_raw = f_scaler.inverse_transform(input)?;
                        let s_target = unpack_complex_features(&s_raw, 2)?;
                        let phys = rel.forward_physics(pred, &s_target, ctx)?;
                        mse.broadcast_add(&phys.affine(physics_lambda, 0.0)?)
                    }),
                }
            }
            // Same fallback policy as EpsRelMse: missing scaler or
            // physics context degrades to MSE so the trial still
            // returns a finite loss instead of failing hard.
            _ => packed_loss_from_pair(
                complex_output_size,
                |p, t| mse_loss(p, t, Reduction::Mean),
                |p, t| complex_mse_loss(p, t, Reduction::Mean),
            ),
        },
    }
}

/// Thin wrapper that pulls the complex-output-size from an
/// [`MlpModel`] and delegates to [`packed_loss_factory`].
fn mlp_model_packed_loss<'a>(
    model: &MlpModel,
    params: &HyperParams,
    feature_scaler: Option<ScalerRef<'a>>,
    target_scaler: Option<ScalerRef<'a>>,
    physics: Option<&'a PhysicsContext>,
) -> PackedLossFn<'a> {
    let complex_n = match model {
        MlpModel::Real(_) => None,
        MlpModel::Complex(m) => Some(m.output_size()),
    };
    // PhysicsForward picks the trial's sampled λ; legacy or non-PINN
    // trials fall back to the default and never use the value.
    let physics_lambda = params
        .loss_params
        .physics_lambda
        .unwrap_or(DEFAULT_PHYSICS_LAMBDA);
    packed_loss_factory(
        complex_n,
        params.loss,
        physics_lambda,
        feature_scaler,
        target_scaler,
        physics,
    )
}

trait HpoTrainableModel: ModuleT {
    fn parameter_count(&self) -> usize;

    fn packed_loss<'a>(
        &self,
        params: &HyperParams,
        feature_scaler: Option<ScalerRef<'a>>,
        target_scaler: Option<ScalerRef<'a>>,
        physics: Option<&'a PhysicsContext>,
    ) -> PackedLossFn<'a>;
}

impl HpoTrainableModel for MLPRegressor {
    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }

    fn packed_loss<'a>(
        &self,
        params: &HyperParams,
        feature_scaler: Option<ScalerRef<'a>>,
        target_scaler: Option<ScalerRef<'a>>,
        physics: Option<&'a PhysicsContext>,
    ) -> PackedLossFn<'a> {
        // Real-only builder → `None` selects the no-unpack branch.
        let physics_lambda = params
            .loss_params
            .physics_lambda
            .unwrap_or(DEFAULT_PHYSICS_LAMBDA);
        packed_loss_factory(
            None,
            params.loss,
            physics_lambda,
            feature_scaler,
            target_scaler,
            physics,
        )
    }
}

impl HpoTrainableModel for MlpModel {
    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }

    fn packed_loss<'a>(
        &self,
        params: &HyperParams,
        feature_scaler: Option<ScalerRef<'a>>,
        target_scaler: Option<ScalerRef<'a>>,
        physics: Option<&'a PhysicsContext>,
    ) -> PackedLossFn<'a> {
        mlp_model_packed_loss(self, params, feature_scaler, target_scaler, physics)
    }
}

// ---------------------------------------------------------------------------
// Model builders
// ---------------------------------------------------------------------------

/// Builds an [`MLPRegressor`] from sampled [`HyperParams`].
pub struct MlpModelBuilder {
    input_size: usize,
    output_size: usize,
    device: Device,
}

impl MlpModelBuilder {
    pub fn new(input_size: usize, output_size: usize, device: Device) -> Self {
        Self {
            input_size,
            output_size,
            device,
        }
    }
}

impl ModelBuilder for MlpModelBuilder {
    type Model = MLPRegressor;

    fn build(&self, params: &HyperParams) -> Result<(MLPRegressor, VarMap)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &self.device);
        // Use the unified `build_mlp` dispatch and unwrap the Real
        // variant — this builder is Real-only by contract, so a
        // Complex `HyperParams` here is a caller bug.
        match params.build_mlp(self.input_size, self.output_size, vb.pp("model"))? {
            MlpModel::Real(model) => Ok((model, varmap)),
            MlpModel::Complex(_) => Err(candle_core::Error::Msg(
                "MlpModelBuilder is Real-only but HyperParams carries ModelKind::Complex — \
                 use UnifiedMlpBuilder for Complex trials"
                    .into(),
            )),
        }
    }
}

/// Builds either a real or complex MLP depending on the trial's
/// [`ModelKind`]. All Real/Complex dispatch happens inside
/// [`HyperParams::build_mlp`]; this wrapper just threads the
/// VarMap/device plumbing.
pub struct UnifiedMlpBuilder {
    input_size: usize,
    output_size: usize,
    device: Device,
    model_type: ModelType,
}

impl UnifiedMlpBuilder {
    pub fn new(
        input_size: usize,
        output_size: usize,
        device: Device,
        model_type: ModelType,
    ) -> Self {
        Self {
            input_size,
            output_size,
            device,
            model_type,
        }
    }
}

impl ModelBuilder for UnifiedMlpBuilder {
    type Model = MlpModel;

    fn build(&self, params: &HyperParams) -> Result<(MlpModel, VarMap)> {
        // Defence-in-depth: the caller tells us the model family via
        // `self.model_type` and the search space tells us via
        // `params.kind`. If they ever disagree, we'd silently train
        // the wrong model for that family's data tensors — fail loudly
        // instead.
        let kind_family = if params.is_complex() {
            ModelType::Complex
        } else {
            ModelType::Real
        };
        if kind_family != self.model_type {
            return Err(candle_core::Error::Msg(format!(
                "model type mismatch: builder=`{}`, params.kind=`{}`",
                self.model_type, kind_family,
            )));
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &self.device);
        let model = params.build_mlp(self.input_size, self.output_size, vb.pp("model"))?;
        Ok((model, varmap))
    }
}

// ---------------------------------------------------------------------------
// TensorDataSource
// ---------------------------------------------------------------------------

/// Pre-split train/val/test tensors exposed as [`DataLoader`]s. Scaling is
/// applied once (cached in `OnceLock`) and shared across all trials.
pub struct TensorDataSource<'a> {
    pub train_features: &'a Tensor,
    pub train_targets: &'a Tensor,
    pub val_features: &'a Tensor,
    pub val_targets: &'a Tensor,
    pub test_features: Option<&'a Tensor>,
    pub test_targets: Option<&'a Tensor>,
    pub feature_scaler: Option<ScalerRef<'a>>,
    pub target_scaler: Option<ScalerRef<'a>>,
    pub shuffle: bool,
    pub shuffle_seed: u64,
    scaled_train_features: OnceLock<Tensor>,
    scaled_train_targets: OnceLock<Tensor>,
    scaled_val_features: OnceLock<Tensor>,
    scaled_val_targets: OnceLock<Tensor>,
    scaled_test_features: OnceLock<Tensor>,
    scaled_test_targets: OnceLock<Tensor>,
}

impl<'a> TensorDataSource<'a> {
    pub fn new(
        train_features: &'a Tensor,
        train_targets: &'a Tensor,
        val_features: &'a Tensor,
        val_targets: &'a Tensor,
    ) -> Self {
        Self {
            train_features,
            train_targets,
            val_features,
            val_targets,
            test_features: None,
            test_targets: None,
            feature_scaler: None,
            target_scaler: None,
            shuffle: true,
            shuffle_seed: 42,
            scaled_train_features: OnceLock::new(),
            scaled_train_targets: OnceLock::new(),
            scaled_val_features: OnceLock::new(),
            scaled_val_targets: OnceLock::new(),
            scaled_test_features: OnceLock::new(),
            scaled_test_targets: OnceLock::new(),
        }
    }

    /// Attach the perturbed test split used for HPO Pareto objectives.
    pub fn with_test(mut self, test_features: &'a Tensor, test_targets: &'a Tensor) -> Self {
        self.test_features = Some(test_features);
        self.test_targets = Some(test_targets);
        self
    }

    pub fn with_feature_scaler<S>(mut self, scaler: S) -> Self
    where
        S: Into<ScalerRef<'a>>,
    {
        self.feature_scaler = Some(scaler.into());
        self
    }

    pub fn with_target_scaler<S>(mut self, scaler: S) -> Self
    where
        S: Into<ScalerRef<'a>>,
    {
        self.target_scaler = Some(scaler.into());
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.shuffle_seed = seed;
        self
    }

    /// Scaled features, cached in `slot` on first call. See
    /// [`get_or_pack`] for the single-clone cache-miss pattern.
    fn scaled_features(&self, slot: &OnceLock<Tensor>, source: &Tensor) -> Result<Tensor> {
        if let Some(t) = slot.get() {
            return Ok(t.clone());
        }
        let computed = match self.feature_scaler {
            Some(scaler) => scaler.transform(source)?,
            None => source.clone(),
        };
        let for_caller = computed.clone();
        let _ = slot.set(computed);
        Ok(for_caller)
    }

    /// Scaled targets, cached in `slot` on first call.
    fn scaled_targets(&self, slot: &OnceLock<Tensor>, source: &Tensor) -> Result<Tensor> {
        if let Some(t) = slot.get() {
            return Ok(t.clone());
        }
        let computed = match self.target_scaler {
            Some(scaler) => scaler.transform(source)?,
            None => source.clone(),
        };
        let for_caller = computed.clone();
        let _ = slot.set(computed);
        Ok(for_caller)
    }
}

impl<'a> DataSource<'a> for TensorDataSource<'a> {
    fn loaders(&'a self, params: &HyperParams) -> Result<(DataLoader, DataLoader)> {
        let batch_size = BatchSize::from(params.train_batch_size);

        // Pulls from the OnceLock cache — first call in the study
        // transforms, every subsequent trial reuses the Arc-shared
        // tensor via `.clone()`.
        let train_features =
            self.scaled_features(&self.scaled_train_features, self.train_features)?;
        let train_targets = self.scaled_targets(&self.scaled_train_targets, self.train_targets)?;
        let val_features = self.scaled_features(&self.scaled_val_features, self.val_features)?;
        let val_targets = self.scaled_targets(&self.scaled_val_targets, self.val_targets)?;

        let train = DataLoader::new(train_features, train_targets, batch_size)?
            .with_shuffle(self.shuffle)
            .with_seed(self.shuffle_seed);
        let val = DataLoader::new(val_features, val_targets, BatchSize::All)?;
        Ok((train, val))
    }

    fn test_loader(&'a self, _params: &HyperParams) -> Result<Option<DataLoader>> {
        let (Some(feat), Some(tgt)) = (self.test_features, self.test_targets) else {
            return Ok(None);
        };
        let features = self.scaled_features(&self.scaled_test_features, feat)?;
        let targets = self.scaled_targets(&self.scaled_test_targets, tgt)?;
        Ok(Some(DataLoader::new(features, targets, BatchSize::All)?))
    }

    fn target_scaler(&self) -> Option<ScalerRef<'a>> {
        self.target_scaler
    }

    fn feature_scaler(&self) -> Option<ScalerRef<'a>> {
        self.feature_scaler
    }
}

/// Pre-split complex train/val/test tensors exposed as packed real loaders.
/// The scaled form is cached once per study; the packed form is a transient.
pub struct ComplexTensorDataSource<'a> {
    pub train_features: &'a ComplexTensor,
    pub train_targets: &'a ComplexTensor,
    pub val_features: &'a ComplexTensor,
    pub val_targets: &'a ComplexTensor,
    pub test_features: Option<&'a ComplexTensor>,
    pub test_targets: Option<&'a ComplexTensor>,
    pub feature_scaler: Option<ScalerRef<'a>>,
    pub target_scaler: Option<ScalerRef<'a>>,
    pub shuffle: bool,
    pub shuffle_seed: u64,
    scaled_train_features: OnceLock<Tensor>,
    scaled_train_targets: OnceLock<Tensor>,
    scaled_val_features: OnceLock<Tensor>,
    scaled_val_targets: OnceLock<Tensor>,
    scaled_test_features: OnceLock<Tensor>,
    scaled_test_targets: OnceLock<Tensor>,
}

impl<'a> ComplexTensorDataSource<'a> {
    pub fn new(
        train_features: &'a ComplexTensor,
        train_targets: &'a ComplexTensor,
        val_features: &'a ComplexTensor,
        val_targets: &'a ComplexTensor,
    ) -> Self {
        Self {
            train_features,
            train_targets,
            val_features,
            val_targets,
            test_features: None,
            test_targets: None,
            feature_scaler: None,
            target_scaler: None,
            shuffle: true,
            shuffle_seed: 42,
            scaled_train_features: OnceLock::new(),
            scaled_train_targets: OnceLock::new(),
            scaled_val_features: OnceLock::new(),
            scaled_val_targets: OnceLock::new(),
            scaled_test_features: OnceLock::new(),
            scaled_test_targets: OnceLock::new(),
        }
    }

    /// Attach the perturbed test split used for HPO Pareto objectives.
    pub fn with_test(
        mut self,
        test_features: &'a ComplexTensor,
        test_targets: &'a ComplexTensor,
    ) -> Self {
        self.test_features = Some(test_features);
        self.test_targets = Some(test_targets);
        self
    }

    pub fn with_feature_scaler<S>(mut self, scaler: S) -> Self
    where
        S: Into<ScalerRef<'a>>,
    {
        self.feature_scaler = Some(scaler.into());
        self
    }

    pub fn with_target_scaler<S>(mut self, scaler: S) -> Self
    where
        S: Into<ScalerRef<'a>>,
    {
        self.target_scaler = Some(scaler.into());
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.shuffle_seed = seed;
        self
    }

    /// Packed-then-feature-scaled tensor, cached on first call. The
    /// packed intermediate is dropped — scaler output is the only
    /// long-lived form kept.
    fn scaled_features(&self, scaled: &OnceLock<Tensor>, source: &ComplexTensor) -> Result<Tensor> {
        if let Some(t) = scaled.get() {
            return Ok(t.clone());
        }
        let packed = pack_complex_features(source)?;
        let computed = match self.feature_scaler {
            Some(scaler) => scaler.transform(&packed)?,
            None => packed,
        };
        let for_caller = computed.clone();
        let _ = scaled.set(computed);
        Ok(for_caller)
    }

    /// Packed-then-target-scaled tensor, cached on first call.
    fn scaled_targets(&self, scaled: &OnceLock<Tensor>, source: &ComplexTensor) -> Result<Tensor> {
        if let Some(t) = scaled.get() {
            return Ok(t.clone());
        }
        let packed = pack_complex_features(source)?;
        let computed = match self.target_scaler {
            Some(scaler) => scaler.transform(&packed)?,
            None => packed,
        };
        let for_caller = computed.clone();
        let _ = scaled.set(computed);
        Ok(for_caller)
    }
}

impl<'a> DataSource<'a> for ComplexTensorDataSource<'a> {
    fn loaders(&'a self, params: &HyperParams) -> Result<(DataLoader, DataLoader)> {
        let batch_size = BatchSize::from(params.train_batch_size);
        let train_features =
            self.scaled_features(&self.scaled_train_features, self.train_features)?;
        let train_targets = self.scaled_targets(&self.scaled_train_targets, self.train_targets)?;
        let val_features = self.scaled_features(&self.scaled_val_features, self.val_features)?;
        let val_targets = self.scaled_targets(&self.scaled_val_targets, self.val_targets)?;

        let train = DataLoader::new(train_features, train_targets, batch_size)?
            .with_shuffle(self.shuffle)
            .with_seed(self.shuffle_seed);
        let val = DataLoader::new(val_features, val_targets, BatchSize::All)?;
        Ok((train, val))
    }

    fn test_loader(&'a self, _params: &HyperParams) -> Result<Option<DataLoader>> {
        let (Some(feat), Some(tgt)) = (self.test_features, self.test_targets) else {
            return Ok(None);
        };
        let features = self.scaled_features(&self.scaled_test_features, feat)?;
        let targets = self.scaled_targets(&self.scaled_test_targets, tgt)?;
        Ok(Some(DataLoader::new(features, targets, BatchSize::All)?))
    }

    fn target_scaler(&self) -> Option<ScalerRef<'a>> {
        self.target_scaler
    }

    fn feature_scaler(&self) -> Option<ScalerRef<'a>> {
        self.feature_scaler
    }
}

// ---------------------------------------------------------------------------
// RegressionEvaluator
// ---------------------------------------------------------------------------

/// Trains an MLP regressor on any [`DataSource`] and returns a [`TrialOutcome`].
pub struct RegressionEvaluator<'a, D> {
    data: &'a D,
    max_epochs: usize,
    patience: usize,
    warmup_epochs: usize,
    multi_objective: bool,
    log: LogSender,
    physics: Option<&'a PhysicsContext>,
}

impl<'a, D> RegressionEvaluator<'a, D> {
    pub fn new(data: &'a D, max_epochs: usize) -> Self {
        Self {
            data,
            max_epochs,
            patience: 10,
            warmup_epochs: 0,
            multi_objective: false,
            log: LogSender::null(),
            physics: None,
        }
    }

    /// Set early stopping patience.
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set warmup epochs before early stopping monitoring begins.
    pub fn with_warmup_epochs(mut self, warmup_epochs: usize) -> Self {
        self.warmup_epochs = warmup_epochs;
        self
    }

    /// Enable dual-objective mode: `[OK@1%, hidden_size]`.
    pub fn with_multi_objective(mut self, enabled: bool) -> Self {
        self.multi_objective = enabled;
        self
    }

    /// Attach a [`LogSender`] for per-trial output.
    pub fn with_log(mut self, log: LogSender) -> Self {
        self.log = log;
        self
    }

    /// Attach a [`PhysicsContext`] so trials selecting
    /// `LossChoice::PhysicsForward` can evaluate the NRW forward
    /// model with the dataset's waveguide geometry + operating
    /// frequency. Absent context falls back to complex MSE for that
    /// loss variant.
    pub fn with_physics_context(mut self, ctx: &'a PhysicsContext) -> Self {
        self.physics = Some(ctx);
        self
    }
}

impl<'a, D, M> Evaluator<M> for RegressionEvaluator<'a, D>
where
    D: DataSource<'a>,
    M: HpoTrainableModel,
{
    fn evaluate(
        &self,
        model: &M,
        varmap: &mut VarMap,
        params: &HyperParams,
    ) -> Result<TrialOutcome> {
        // Time every stage of the per-trial pipeline. Previously only
        // `train_ms` / `restore_ms` / `objectives_ms` were measured —
        // any regression in the setup phase (loaders, optimizer build,
        // trainer config, var enumeration, best-buf allocation) was
        // invisible. Now every slice is timed so a trial-speed drop
        // can be attributed to a specific stage.
        let t_setup = std::time::Instant::now();

        // Deterministic, name-based weight init — same protocol as
        // `sparam_app::workflows::run_training`. Without this, HPO
        // weights depend on however the global RNG was advanced
        // between `set_global_seed(trial_seed)` and the `VarBuilder`
        // init inside `build_mlp` (sampler internals, varmap
        // iteration order, …). Calling `deterministic_reinit_varmap`
        // here makes the trial's starting weights bit-equivalent to
        // a `/train` retrain that uses `seed = base_seed + trial_number`,
        // which is what the "Retrain from trial" UI relies on for
        // bit-for-bit reproduction of the recorded objectives.
        sparam_core::determinism::deterministic_reinit_varmap(
            varmap,
            sparam_core::rng::get_global_seed(),
        )?;

        let t_loaders = std::time::Instant::now();
        let (train_loader, val_loader) = self.data.loaders(params)?;
        let loaders_ms = t_loaders.elapsed().as_secs_f64() * 1000.0;

        // `stable_all_vars` locks the VarMap and sorts; do it once and
        // hand a clone to the optimizer (Vars are Arc — clone is a
        // refcount bump per element).
        let t_opt = std::time::Instant::now();
        let vars = sparam_core::determinism::stable_all_vars(varmap);
        let optimizer = params.optimizer_config().build(vars.clone())?;
        let opt_build_ms = t_opt.elapsed().as_secs_f64() * 1000.0;

        // `trainer_config` dispatches on `params.kind` and returns no
        // grad-clip / input-noise for Complex trials (those fields live
        // only on `ModelKind::Real`), so we don't need a separate
        // model.is_complex() override here.
        let t_trainer = std::time::Instant::now();
        let trainer_builder = params
            .trainer_builder(self.max_epochs, self.patience, self.warmup_epochs)?
            .with_log(self.log.clone());
        let mut trainer = trainer_builder.build(optimizer)?;
        let trainer_build_ms = t_trainer.elapsed().as_secs_f64() * 1000.0;

        let t_loss = std::time::Instant::now();
        let loss_fn = model.packed_loss(
            params,
            self.data.feature_scaler(),
            self.data.target_scaler(),
            self.physics,
        );
        let loss_build_ms = t_loss.elapsed().as_secs_f64() * 1000.0;

        // Best-epoch weight snapshot — raw memcpy, no graph nodes.
        let t_buf = std::time::Instant::now();
        let var_sizes: Vec<usize> = vars.iter().map(|v| v.as_tensor().elem_count()).collect();
        let total_params: usize = var_sizes.iter().sum();
        let mut best_buf = vec![0.0f64; total_params];
        let mut has_best = false;
        let buf_alloc_ms = t_buf.elapsed().as_secs_f64() * 1000.0;

        sparam_core::determinism::emit_varmap_hash(format_args!("hpo-evaluator pre-fit"), varmap);

        let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;

        let t_train = std::time::Instant::now();
        let result = trainer.fit_for_hpo(
            model,
            varmap,
            move |input, pred, target| loss_fn(input, pred, target),
            &train_loader,
            &val_loader,
            || {
                // Raw memcpy snapshot — no Tensor API, no graph nodes.
                let mut off = 0;
                for (var, &n) in vars.iter().zip(&var_sizes) {
                    let (storage, layout) = var.as_tensor().storage_and_layout();
                    if let candle_core::Storage::Cpu(cpu) = &*storage {
                        let src: &[f64] = cpu.as_slice()?;
                        let s = layout.start_offset();
                        best_buf[off..off + n].copy_from_slice(&src[s..s + n]);
                    }
                    off += n;
                }
                has_best = true;
                Ok(())
            },
        )?;
        let train_ms = t_train.elapsed().as_secs_f64() * 1000.0;

        // Restore best-epoch weights.
        let t_restore = std::time::Instant::now();
        if has_best && result.best_epoch < result.final_epoch {
            let mut off = 0;
            for (var, &n) in vars.iter().zip(&var_sizes) {
                let t = Tensor::from_slice(
                    &best_buf[off..off + n],
                    var.as_tensor().shape(),
                    var.as_tensor().device(),
                )?;
                var.set(&t)?;
                off += n;
            }
        }
        let restore_ms = t_restore.elapsed().as_secs_f64() * 1000.0;

        sparam_core::determinism::emit_varmap_hash(
            format_args!(
                "hpo-evaluator pre-objectives (best_epoch={}, final_epoch={})",
                result.best_epoch, result.final_epoch
            ),
            varmap,
        );

        let t_obj = std::time::Instant::now();
        let (ok_at_1pct, max_error) = if self.multi_objective {
            // Use the perturbed test grid when available (matches Python).
            let test_opt = self.data.test_loader(params)?;
            let eval_loader = test_opt.as_ref().unwrap_or(&val_loader);
            compute_validation_objectives(
                model,
                eval_loader,
                self.data.target_scaler(),
                params.is_complex(),
            )?
        } else {
            (f64::NAN, f64::INFINITY)
        };
        let obj_ms = t_obj.elapsed().as_secs_f64() * 1000.0;

        // Full per-stage timing — emitted every trial so a sudden
        // trial-speed regression can be pinpointed to a stage (setup
        // subphase, fit loop, best-weight restore, or objectives
        // evaluation) by comparing numbers across runs.
        //
        // Setup subphases (loaders, opt_build, trainer_build,
        // loss_build, buf_alloc) together sum to `setup_ms`. A
        // regression in any one shows up as its individual field
        // growing.
        //
        // The first 3 trials and every 50th are `Info`; the rest are
        // sent at the same level but expect the log worker to swallow
        // them when no active sink is attached (`LogSender::null`).
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static TRIAL_CTR: AtomicUsize = AtomicUsize::new(0);
            let n = TRIAL_CTR.fetch_add(1, Ordering::Relaxed) + 1;
            let total_ms = setup_ms + train_ms + restore_ms + obj_ms;
            if n <= 3 || n % 10 == 0 {
                self.log
                    .send(sparam_training::logger::LogMessage::Info(format!(
                        "[eval #{n}] setup={setup_ms:.1}ms (loaders={loaders_ms:.1} \
                     opt={opt_build_ms:.2} trainer={trainer_build_ms:.2} \
                     loss={loss_build_ms:.2} buf={buf_alloc_ms:.2})  \
                     train={train_ms:.1}ms  restore={restore_ms:.2}ms  \
                     objectives={obj_ms:.1}ms  total={total_ms:.1}ms",
                    )));
            }
        }

        let metrics = TrialMetrics {
            best_val_loss: result.best_val_loss,
            param_count: model.parameter_count(),
            stopped_early: result.stopped_early,
            best_epoch: result.best_epoch,
            final_epoch: result.final_epoch,
            training_time_secs: result.training_time_secs,
            ok_at_1pct,
            max_error,
        };

        let status = if result.best_val_loss.is_finite() {
            TrialStatus::Completed
        } else {
            TrialStatus::Failed
        };

        if self.multi_objective {
            // 3 objectives: [OK@1% max, param_count min, max_error min].
            Ok(TrialOutcome::accuracy_complexity_error(
                ok_at_1pct,
                metrics.param_count,
                max_error,
                metrics,
                status,
            ))
        } else {
            Ok(TrialOutcome::single(result.best_val_loss, metrics, status))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_space::{
        BatchSizeChoice, GradClipChoice, LossChoice, LossHyperParams, ModelKind, NormChoice,
        OptimizerChoice, OptimizerHyperParams, SchedulerChoice, SchedulerHyperParams,
    };
    use sparam_data::loader::BatchSize;
    use sparam_data::scaling::{Scaler, StandardScaler};
    use sparam_models::Normalization;
    use sparam_training::optimizers::OptimizerConfig;
    use sparam_training::schedulers::{CosineAnnealingConfig, LRScheduler, SchedulerConfig};

    fn rand_complex(batch: usize, features: usize, device: &Device) -> ComplexTensor {
        ComplexTensor::new(
            Tensor::randn(0f64, 1.0, (batch, features), device).unwrap(),
            Tensor::randn(0f64, 1.0, (batch, features), device).unwrap(),
        )
        .unwrap()
    }

    fn default_hyper_params() -> HyperParams {
        HyperParams {
            hidden_size: 32,
            optimizer: OptimizerChoice::AdamW,
            lr: 1e-3,
            train_batch_size: BatchSizeChoice::All,
            weight_decay: 0.01,
            loss: LossChoice::Mse,
            loss_params: LossHyperParams::default(),
            scheduler: SchedulerChoice::None,
            scheduler_params: SchedulerHyperParams::default(),
            optimizer_params: OptimizerHyperParams::default(),
            kind: ModelKind::Real {
                activation: sparam_models::Activation::ReLU,
                dropout_p: 0.0,
                norm: NormChoice::None,
                grad_clip_norm: GradClipChoice::None,
                input_noise_std: 0.0,
            },
        }
    }

    fn default_complex_hyper_params() -> HyperParams {
        HyperParams {
            hidden_size: 24,
            optimizer: OptimizerChoice::AdamW,
            lr: 1e-3,
            train_batch_size: BatchSizeChoice::All,
            weight_decay: 0.01,
            loss: LossChoice::Mse,
            loss_params: LossHyperParams::default(),
            scheduler: SchedulerChoice::None,
            scheduler_params: SchedulerHyperParams::default(),
            optimizer_params: OptimizerHyperParams::default(),
            kind: ModelKind::Complex {
                activation: sparam_models::ComplexActivation::CReLU,
                dropout_p: 0.0,
                grad_clip_norm: crate::GradClipChoice::None,
                input_noise_std: 0.0,
            },
        }
    }

    fn fit_standard_scaler(data: &Tensor) -> StandardScaler {
        let mut scaler = StandardScaler::new();
        scaler.fit(data).unwrap();
        scaler
    }

    // -- Conversion tests ---------------------------------------------------

    #[test]
    fn norm_choice_round_trips() {
        assert_eq!(Normalization::from(NormChoice::None), Normalization::None);
        assert_eq!(
            Normalization::from(NormChoice::LayerNorm),
            Normalization::LayerNorm
        );
        assert_eq!(
            Normalization::from(NormChoice::BatchNorm),
            Normalization::BatchNorm
        );
    }

    #[test]
    fn batch_size_choice_converts() {
        assert_eq!(BatchSize::from(BatchSizeChoice::B32), BatchSize::Fixed(32));
        assert_eq!(BatchSize::from(BatchSizeChoice::All), BatchSize::All);
    }

    // -- Optimizer config tests ---------------------------------------------

    #[test]
    fn optimizer_config_maps_each_choice_to_typed_config() {
        // AdamW — default hp already has AdamW selected with lr/wd set.
        match default_hyper_params().optimizer_config() {
            OptimizerConfig::AdamW(c) => {
                assert!((c.lr - 1e-3).abs() < f64::EPSILON);
                assert!((c.weight_decay - 0.01).abs() < f64::EPSILON);
            }
            other => panic!("expected AdamW, got {other:?}"),
        }

        // SGD — carries sgd_momentum + sgd_nesterov.
        let mut hp = default_hyper_params();
        hp.optimizer = OptimizerChoice::SGD;
        hp.optimizer_params.sgd_momentum = Some(0.9);
        hp.optimizer_params.sgd_nesterov = Some(true);
        match hp.optimizer_config() {
            OptimizerConfig::Sgd(c) => {
                assert!((c.momentum - 0.9).abs() < f64::EPSILON);
                assert!(c.nesterov);
            }
            other => panic!("expected Sgd, got {other:?}"),
        }

        // RMSprop — carries rmsprop_momentum + rmsprop_alpha.
        let mut hp = default_hyper_params();
        hp.optimizer = OptimizerChoice::RMSprop;
        hp.optimizer_params.rmsprop_momentum = Some(0.1);
        hp.optimizer_params.rmsprop_alpha = Some(0.95);
        match hp.optimizer_config() {
            OptimizerConfig::RMSprop(c) => {
                assert!((c.momentum - 0.1).abs() < f64::EPSILON);
                assert!((c.alpha - 0.95).abs() < f64::EPSILON);
            }
            other => panic!("expected RMSprop, got {other:?}"),
        }
    }

    // -- MLP build tests ----------------------------------------------------

    #[test]
    fn build_mlp_real_produces_real_variant_with_expected_shape() {
        let hp = default_hyper_params();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let model = hp.build_mlp(4, 2, vb.pp("model")).unwrap();
        match model {
            MlpModel::Real(_) => {}
            MlpModel::Complex(_) => panic!("expected Real variant"),
        }
    }

    #[test]
    fn build_mlp_complex_produces_complex_variant() {
        // default_complex_hyper_params already uses CReLU.
        let hp = default_complex_hyper_params();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let model = hp.build_mlp(2, 1, vb.pp("model")).unwrap();
        match model {
            MlpModel::Complex(_) => {}
            MlpModel::Real(_) => panic!("expected Complex variant"),
        }
    }

    #[test]
    fn build_mlp_complex_modrelu_builds_without_error() {
        let mut hp = default_complex_hyper_params();
        if let ModelKind::Complex {
            ref mut activation, ..
        } = hp.kind
        {
            *activation = sparam_models::ComplexActivation::ModReLU;
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        // Should build successfully — modrelu is a learnable complex
        // activation and must be accepted by `build_mlp`.
        hp.build_mlp(2, 1, vb.pp("model")).unwrap();
    }

    // -- Trainer config tests -----------------------------------------------

    #[test]
    fn trainer_config_from_hyper_params() {
        let mut hp = default_hyper_params();
        if let ModelKind::Real {
            grad_clip_norm,
            input_noise_std,
            ..
        } = &mut hp.kind
        {
            *grad_clip_norm = GradClipChoice::Clip10;
            *input_noise_std = 0.05;
        }
        let tc = hp.trainer_config(100, 15, 5);
        assert_eq!(tc.max_epochs, 100);
        assert_eq!(tc.max_grad_norm, Some(1.0));
        assert_eq!(tc.input_noise_std, Some(0.05));
        let es = tc.early_stopping.as_ref().unwrap();
        assert_eq!(es.patience, 15);
        assert_eq!(es.warmup_epochs, 5);
    }

    /// Complex trials structurally cannot carry grad-clip / input-noise
    /// — `trainer_config` resolves both to None regardless of what the
    /// trainer-builder arguments say.
    #[test]
    fn trainer_config_for_complex_has_no_regularization() {
        let hp = default_complex_hyper_params();
        let tc = hp.trainer_config(100, 15, 5);
        assert_eq!(tc.max_grad_norm, None);
        assert_eq!(tc.input_noise_std, None);
    }

    // -- Scheduler tests ----------------------------------------------------

    #[test]
    fn lr_scheduler_maps_each_choice_to_instance() {
        // None → returns None.
        let hp = default_hyper_params();
        assert!(hp.lr_scheduler(100).unwrap().is_none());

        // Cosine → returns Some(CosineAnnealing) AND scheduler_config()
        // carries t_max + eta_min through.
        let mut hp = default_hyper_params();
        hp.scheduler = SchedulerChoice::Cosine;
        hp.scheduler_params.cosine_eta_min = Some(1e-5);
        assert_eq!(
            hp.scheduler_config(75),
            SchedulerConfig::CosineAnnealing(CosineAnnealingConfig {
                t_max: 75,
                eta_min: 1e-5,
            }),
        );
        assert!(matches!(
            hp.lr_scheduler(100).unwrap(),
            Some(LRScheduler::CosineAnnealing(_))
        ));

        // Plateau → returns Some(ReduceOnPlateau).
        let mut hp = default_hyper_params();
        hp.scheduler = SchedulerChoice::Plateau;
        hp.scheduler_params.plateau_factor = Some(0.5);
        hp.scheduler_params.plateau_patience = Some(5);
        assert!(matches!(
            hp.lr_scheduler(100).unwrap(),
            Some(LRScheduler::ReduceOnPlateau(_))
        ));
    }

    #[test]
    fn trainer_builder_from_hyper_params_carries_scheduler() {
        let mut hp = default_hyper_params();
        hp.scheduler = SchedulerChoice::Cosine;
        hp.scheduler_params.cosine_eta_min = Some(1e-5);

        let builder = hp.trainer_builder(100, 10, 0).unwrap();
        assert_eq!(builder.config().max_epochs, 100);
        assert!(matches!(
            builder.scheduler_ref(),
            Some(LRScheduler::CosineAnnealing(_))
        ));
    }

    #[test]
    fn real_packed_loss_uses_relative_error() {
        let device = Device::Cpu;
        let target = Tensor::from_vec(vec![2.0f64, 4.0, 4.0, 8.0], (2, 2), &device).unwrap();
        let pred = Tensor::from_vec(vec![3.0f64, 2.0, 2.0, 16.0], (2, 2), &device).unwrap();
        let scaler = fit_standard_scaler(&target);
        let pred_scaled = scaler.transform(&pred).unwrap();
        let target_scaled = scaler.transform(&target).unwrap();

        let mut hp = default_hyper_params();
        hp.loss = LossChoice::EpsRelMse;
        let loss = packed_loss_factory(
            None,
            hp.loss,
            DEFAULT_PHYSICS_LAMBDA,
            None,
            Some(ScalerRef::from(&scaler)),
            None,
        );
        // EpsRelMse ignores the `input` arg — pass `pred_scaled` as a dummy.
        let actual = loss(&pred_scaled, &pred_scaled, &target_scaled)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        // Build expected with the same architectural softplus clamp the
        // factory now applies before the loss kernel runs.
        let pred_clamped = apply_physical_softplus_clamp(
            &pred_scaled,
            ScalerRef::from(&scaler),
            /*is_complex=*/ false,
        )
        .unwrap();
        let expected = RelativeErrorLoss::new(&scaler)
            .forward(&pred_clamped, &target_scaled)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();

        assert!((actual - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn real_forward_matches_magnitude_based_relative_error() {
        // Regression test for the Python `eps_rel_mse` parity fix: the
        // Real-path `forward` must treat `(col0, col1)` as the (Re, Im)
        // of a complex number and divide by the complex magnitude,
        // not by each column independently. The earlier element-wise
        // implementation exploded the loss whenever ε'' (col1) was
        // near zero — the exact regime of the training dataset.
        let device = Device::Cpu;
        // col1 (ε'') is two orders of magnitude smaller than col0.
        // With the old element-wise denominator this sample drove the
        // loss to ~1e10; with the magnitude denominator the sample is
        // well-scaled because |ε| ≈ ε'.
        let target = Tensor::from_vec(vec![100.0f64, 0.01, 50.0, 2.0], (2, 2), &device).unwrap();
        let pred = Tensor::from_vec(vec![100.5f64, 0.02, 49.0, 2.1], (2, 2), &device).unwrap();
        let scaler = fit_standard_scaler(&target);
        let pred_scaled = scaler.transform(&pred).unwrap();
        let target_scaled = scaler.transform(&target).unwrap();

        let actual = RelativeErrorLoss::new(&scaler)
            .forward(&pred_scaled, &target_scaled)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();

        // Expected: build the complex tensor by hand and run the same
        // complex relative-error path that the fixed `forward` uses.
        let target_complex = unpack_complex_features(&target, 1).unwrap();
        let pred_complex = unpack_complex_features(&pred, 1).unwrap();
        let rel_err = complex_relative_error_elements(&pred_complex, &target_complex).unwrap();
        let expected = rel_err
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();

        assert!((actual - expected).abs() < 1e-12);
        // Loss must stay O(1) — the old elementwise form blew up to
        // ~1e6 for this input.
        assert!(actual < 1.0, "loss should stay bounded, got {actual}");
    }

    #[test]
    fn complex_packed_loss_uses_relative_error() {
        // Production ComplexMLP shape: 1 complex permittivity output
        // (= 2 real columns after packing). Matches the always-on
        // softplus clamp's expected layout.
        let device = Device::Cpu;
        let target = ComplexTensor::new(
            Tensor::from_vec(vec![3.0f64, 5.0], (2, 1), &device).unwrap(),
            Tensor::from_vec(vec![4.0f64, 1.0], (2, 1), &device).unwrap(),
        )
        .unwrap();
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![6.0f64, 4.0], (2, 1), &device).unwrap(),
            Tensor::from_vec(vec![8.0f64, 2.0], (2, 1), &device).unwrap(),
        )
        .unwrap();
        let target_packed = pack_complex_features(&target).unwrap();
        let pred_packed = pack_complex_features(&pred).unwrap();
        let scaler = fit_standard_scaler(&target_packed);
        let pred_scaled = scaler.transform(&pred_packed).unwrap();
        let target_scaled = scaler.transform(&target_packed).unwrap();

        // Build a Complex MLP with output_size=1 so `output_size()`
        // yields the production-shape complex-feature-count.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let complex_model = sparam_models::ComplexMLPRegressor::new(
            vb.pp("model"),
            &sparam_models::ComplexMLPConfig::new(2, 8, 1, sparam_models::ComplexActivation::CReLU),
        )
        .unwrap();
        let model = MlpModel::Complex(complex_model);

        let mut hp = default_complex_hyper_params();
        hp.loss = LossChoice::EpsRelMse;
        let loss = mlp_model_packed_loss(&model, &hp, None, Some(ScalerRef::from(&scaler)), None);
        let actual = loss(&pred_scaled, &pred_scaled, &target_scaled)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        // Build expected with the same architectural softplus clamp
        // (Complex sign convention: Im(ε) ≤ 0).
        let pred_scaled_clamped = apply_physical_softplus_clamp(
            &pred_scaled,
            ScalerRef::from(&scaler),
            /*is_complex=*/ true,
        )
        .unwrap();
        let pred_clamped_complex = unpack_complex_features(&pred_scaled_clamped, 1).unwrap();
        let target_scaled_complex = unpack_complex_features(&target_scaled, 1).unwrap();
        let expected = RelativeErrorLoss::new(&scaler)
            .forward_complex(&pred_clamped_complex, &target_scaled_complex)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();

        assert!((actual - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn real_mlp_model_loss_matches_scalar_mse() {
        let builder = UnifiedMlpBuilder::new(4, 2, Device::Cpu, ModelType::Real);
        let hp = default_hyper_params();
        let (model, _varmap) = builder.build(&hp).unwrap();
        let pred = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let target = Tensor::from_vec(vec![0.0f64, 0.0, 1.0, 1.0], (2, 2), &Device::Cpu).unwrap();

        let actual = mlp_model_packed_loss(&model, &hp, None, None, None)(&pred, &pred, &target)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        let expected = mse_loss(&pred, &target, Reduction::Mean)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();

        assert!((actual - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn complex_mlp_model_loss_matches_complex_smooth_l1() {
        let builder = UnifiedMlpBuilder::new(2, 1, Device::Cpu, ModelType::Complex);
        let mut hp = default_complex_hyper_params();
        hp.loss = LossChoice::SmoothL1;
        let (model, _varmap) = builder.build(&hp).unwrap();
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![2.0f64, 4.0], (2, 1), &Device::Cpu).unwrap(),
            Tensor::from_vec(vec![1.0f64, 3.0], (2, 1), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let target = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 1.0], (2, 1), &Device::Cpu).unwrap(),
            Tensor::from_vec(vec![0.0f64, 1.0], (2, 1), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let pred_packed = pack_complex_features(&pred).unwrap();
        let target_packed = pack_complex_features(&target).unwrap();

        let actual = mlp_model_packed_loss(&model, &hp, None, None, None)(
            &pred_packed,
            &pred_packed,
            &target_packed,
        )
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
        let expected = complex_smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();

        assert!((actual - expected).abs() < f64::EPSILON);
    }

    // -- Model builder tests ------------------------------------------------

    #[test]
    fn mlp_model_builder_produces_model_with_hp_shape_and_params() {
        let builder = MlpModelBuilder::new(4, 2, Device::Cpu);
        let hp = default_hyper_params();
        let (model, _varmap) = builder.build(&hp).unwrap();
        // Shape comes from the builder + hyperparams; param count is
        // derived from shape. Asserting both in one test keeps the
        // fixture shared.
        assert_eq!(model.config().input_size, 4);
        assert_eq!(model.config().output_size, 2);
        assert_eq!(model.config().hidden_size, 32);
        assert!(model.parameter_count() > 0);
    }

    #[test]
    fn unified_builder_produces_real_model() {
        let builder = UnifiedMlpBuilder::new(4, 2, Device::Cpu, ModelType::Real);
        let hp = default_hyper_params();
        let (model, _varmap) = builder.build(&hp).unwrap();
        match model {
            MlpModel::Real(model) => {
                assert_eq!(model.config().input_size, 4);
                assert_eq!(model.config().output_size, 2);
            }
            MlpModel::Complex(_) => panic!("expected real model"),
        }
    }

    #[test]
    fn unified_builder_produces_complex_model() {
        let builder = UnifiedMlpBuilder::new(2, 1, Device::Cpu, ModelType::Complex);
        let hp = default_complex_hyper_params();
        let (model, _varmap) = builder.build(&hp).unwrap();
        match model {
            MlpModel::Real(_) => panic!("expected complex model"),
            MlpModel::Complex(model) => {
                assert_eq!(model.config().input_size, 2);
                assert_eq!(model.config().output_size, 1);
            }
        }
    }

    #[test]
    fn complex_tensor_data_source_packs_features() {
        let device = Device::Cpu;
        let train_x = rand_complex(10, 2, &device);
        let train_y = rand_complex(10, 1, &device);
        let val_x = rand_complex(4, 2, &device);
        let val_y = rand_complex(4, 1, &device);

        let data =
            ComplexTensorDataSource::new(&train_x, &train_y, &val_x, &val_y).with_shuffle(false);
        let hp = default_hyper_params();
        let (train_loader, val_loader) = data.loaders(&hp).unwrap();

        assert_eq!(train_loader.feature_count(), 4);
        assert_eq!(train_loader.target_count(), 2);
        assert_eq!(val_loader.feature_count(), 4);
        assert_eq!(val_loader.target_count(), 2);
    }

    #[test]
    fn unified_complex_model_forward_uses_packed_io() {
        let builder = UnifiedMlpBuilder::new(2, 1, Device::Cpu, ModelType::Complex);
        let hp = default_complex_hyper_params();
        let (model, _varmap) = builder.build(&hp).unwrap();
        let xs = Tensor::randn(0f64, 1.0, (8, 4), &Device::Cpu).unwrap();
        let ys = model.forward_t(&xs, true).unwrap();
        assert_eq!(ys.dims(), &[8, 2]);
    }

    // -- End-to-end integration tests (small model, few epochs) -------------

    #[test]
    fn regression_evaluator_single_objective() {
        let device = Device::Cpu;
        let train_x = Tensor::randn(0f64, 1.0, (50, 4), &device).unwrap();
        let train_y = Tensor::randn(0f64, 1.0, (50, 2), &device).unwrap();
        let val_x = Tensor::randn(0f64, 1.0, (20, 4), &device).unwrap();
        let val_y = Tensor::randn(0f64, 1.0, (20, 2), &device).unwrap();

        let data = TensorDataSource::new(&train_x, &train_y, &val_x, &val_y).with_shuffle(false);
        let builder = MlpModelBuilder::new(4, 2, device);
        let evaluator = RegressionEvaluator::new(&data, 5);

        let hp = default_hyper_params();
        let (model, mut varmap) = builder.build(&hp).unwrap();
        let outcome = evaluator.evaluate(&model, &mut varmap, &hp).unwrap();

        assert_eq!(outcome.status, TrialStatus::Completed);
        assert_eq!(outcome.objective_values().len(), 1);
        assert!(outcome.metrics.best_val_loss.is_finite());
        assert!(outcome.metrics.param_count > 0);
    }

    #[test]
    fn regression_evaluator_multi_objective() {
        let device = Device::Cpu;
        let train_x = Tensor::randn(0f64, 1.0, (50, 4), &device).unwrap();
        let train_y = Tensor::randn(0f64, 1.0, (50, 2), &device).unwrap();
        let val_x = Tensor::randn(0f64, 1.0, (20, 4), &device).unwrap();
        let val_y = Tensor::randn(0f64, 1.0, (20, 2), &device).unwrap();

        let data = TensorDataSource::new(&train_x, &train_y, &val_x, &val_y).with_shuffle(false);
        let builder = MlpModelBuilder::new(4, 2, device);
        let evaluator = RegressionEvaluator::new(&data, 3).with_multi_objective(true);

        let hp = default_hyper_params();
        let (model, mut varmap) = builder.build(&hp).unwrap();
        let outcome = evaluator.evaluate(&model, &mut varmap, &hp).unwrap();

        assert_eq!(outcome.objective_values().len(), 3);
        // Objective layout `[ok_at_1pct, param_count, max_error]` —
        // param_count (trained weights) replaces the sampled
        // hidden_size, and max_error is now a first-class objective
        // instead of a hard feasibility constraint.
        let triple = outcome
            .objective_triple()
            .expect("multi-objective outcome must produce a triple");
        assert_eq!(triple[0], outcome.metrics.ok_at_1pct);
        assert_eq!(triple[1], outcome.metrics.param_count as f64);
        assert_eq!(triple[2], outcome.metrics.max_error);
        assert!(outcome.metrics.param_count > 0);
        assert!(outcome.metrics.ok_at_1pct.is_finite());
        assert!(outcome.metrics.max_error.is_finite());
    }

    #[test]
    fn trial_runner_runs_end_to_end() {
        let device = Device::Cpu;
        let train_x = Tensor::randn(0f64, 1.0, (50, 4), &device).unwrap();
        let train_y = Tensor::randn(0f64, 1.0, (50, 2), &device).unwrap();
        let val_x = Tensor::randn(0f64, 1.0, (20, 4), &device).unwrap();
        let val_y = Tensor::randn(0f64, 1.0, (20, 2), &device).unwrap();

        let data = TensorDataSource::new(&train_x, &train_y, &val_x, &val_y).with_shuffle(false);
        let builder = MlpModelBuilder::new(4, 2, device);
        let evaluator = RegressionEvaluator::new(&data, 3);

        use crate::evaluation::TrialRunner;
        let runner = TrialRunner::new(&builder, &evaluator);

        let hp = default_hyper_params();
        let outcome = runner.run(&hp);

        assert_eq!(outcome.status, TrialStatus::Completed);
        assert!(outcome.metrics.best_val_loss.is_finite());
    }

    #[test]
    fn regression_evaluator_supports_unified_complex_model() {
        let device = Device::Cpu;
        let train_x = rand_complex(40, 2, &device);
        let train_y = rand_complex(40, 1, &device);
        let val_x = rand_complex(16, 2, &device);
        let val_y = rand_complex(16, 1, &device);

        let data =
            ComplexTensorDataSource::new(&train_x, &train_y, &val_x, &val_y).with_shuffle(false);
        let builder = UnifiedMlpBuilder::new(2, 1, device, ModelType::Complex);
        let evaluator = RegressionEvaluator::new(&data, 3);

        let hp = default_complex_hyper_params();
        let (model, mut varmap) = builder.build(&hp).unwrap();
        let outcome = evaluator.evaluate(&model, &mut varmap, &hp).unwrap();

        assert_eq!(outcome.status, TrialStatus::Completed);
        assert_eq!(outcome.objective_values().len(), 1);
        assert!(outcome.metrics.best_val_loss.is_finite());
        assert!(outcome.metrics.param_count > 0);
    }

    /// The HPO post-training `(OK@1%, max_error)` path in
    /// `compute_validation_objectives` MUST produce bit-for-bit
    /// identical numbers to
    /// `sparam_core::metrics::relative_error_metrics_from_components`
    /// given the same (predictions, targets) pair — otherwise a
    /// "retrain from trial" would report a different `max_error`
    /// than the original HPO trial even with bit-identical weights.
    ///
    /// Before this fix, the HPO path used the tensor-op
    /// `compute_relative_error` which wraps
    /// `sqrt(r² + i² + COMPLEX_STABILITY_EPS)` (1e-12 added for
    /// gradient stability inside `ComplexTensor::mag`), producing
    /// a tiny but real divergence vs. `f64::hypot`. This test
    /// locks in the unified scalar path.
    #[test]
    fn compute_validation_objectives_matches_scalar_hypot_path() {
        use candle_core::{Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use sparam_core::metrics::{classify_predictions, relative_error_metrics_from_components};
        use sparam_data::loader::{BatchSize, DataLoader};

        let device = Device::Cpu;
        // Hand-built packed-complex tensors: 4 samples, 2 complex
        // features (→ 4 real columns packed as [r0, r1, i0, i1]).
        // Targets are a fixed grid; "predictions" are targets + a
        // known offset so the rel-error answer is easy to reason
        // about.
        let targets = Tensor::from_slice(
            &[
                1.0, 2.0, 0.1, -0.2, 3.0, 4.0, 0.3, 0.4, 5.0, 6.0, -0.5, 0.6, 7.0, 8.0, 0.7, -0.8,
            ],
            (4, 4),
            &device,
        )
        .unwrap();
        let predictions = Tensor::from_slice(
            &[
                1.01, 2.02, 0.11, -0.19, 3.05, 3.95, 0.28, 0.42, 4.98, 6.03, -0.48, 0.59, 7.10,
                7.85, 0.72, -0.79,
            ],
            (4, 4),
            &device,
        )
        .unwrap();

        // Identity scaler path: treat predictions/targets as already
        // in original scale so we exercise ONLY the relative-error
        // arithmetic. Wrap the tensors in a fake DataLoader + model
        // that yields predictions verbatim.
        //
        // Direct call to the components-based fn:
        let c_feat = packed_complex_feature_count(&targets, "test").unwrap();
        let t_complex = unpack_complex_features(&targets, c_feat).unwrap();
        let p_complex = unpack_complex_features(&predictions, c_feat).unwrap();
        let tr = t_complex
            .real
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let ti = t_complex
            .imag
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let pr = p_complex
            .real
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let pi = p_complex
            .imag
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let components_rel = relative_error_metrics_from_components(&tr, &ti, &pr, &pi).unwrap();
        let expected_ok = classify_predictions(&components_rel.errors, 1.0).ok_percent;
        let expected_max_err = components_rel.max_error;

        // Now run the HPO path via `compute_validation_objectives`.
        // We need a DataLoader that yields (predictions, targets)
        // and a passthrough "model" that returns `predictions`.
        // Easiest: feed the predictions AS the input features to an
        // identity model. Use BatchSize::All so one batch = all
        // samples, no shuffling.
        struct IdentityModel(Tensor);
        impl candle_nn::ModuleT for IdentityModel {
            fn forward_t(&self, _x: &Tensor, _train: bool) -> Result<Tensor> {
                Ok(self.0.clone())
            }
        }
        let _ = VarBuilder::from_varmap(&VarMap::new(), DType::F64, &device);
        let model = IdentityModel(predictions.clone());
        // Use `targets` as both features and targets on the loader —
        // the IdentityModel ignores the features and returns
        // `predictions` directly, so we get (predictions, targets)
        // from `collect_validation_predictions`.
        let loader = DataLoader::new(targets.clone(), targets.clone(), BatchSize::All).unwrap();

        let (hpo_ok, hpo_max_err) =
            compute_validation_objectives(&model, &loader, None, /*is_complex=*/ false).unwrap();

        // Bit-for-bit equality — both paths call the same
        // `relative_error_metrics_from_components` after identical
        // pre-processing.
        assert_eq!(
            hpo_max_err.to_bits(),
            expected_max_err.to_bits(),
            "HPO max_error {hpo_max_err} must match scalar-hypot max_error \
             {expected_max_err} bit-for-bit after unifying the eval path",
        );
        assert_eq!(
            hpo_ok.to_bits(),
            expected_ok.to_bits(),
            "HPO OK@1% must match scalar-hypot OK@1% bit-for-bit",
        );
    }
}
