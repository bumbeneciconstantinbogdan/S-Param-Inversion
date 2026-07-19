//! Training-specific workflow helpers shared by train and study workflows.

use candle_core::{Result, Tensor};

use sparam_data::physical_constraint::apply_physical_softplus_clamp;
use sparam_data::scaling::ScalerRef;
use sparam_data::tensor_bridges::unpack_complex_features;
use sparam_hpo::RelativeErrorLoss;
use sparam_models::MlpModel;
use sparam_physics::PhysicsContext;
use sparam_training::losses::{
    MlpLoss, Reduction, complex_mse_loss, complex_smooth_l1_loss, mse_loss,
    physics_forward_loss, smooth_l1_loss,
};

/// Methodology-volume escape hatch for the M08 side-experiment
/// (Complex pipeline, ``unsupervised'' pure-physics loss).
/// Setting `SPARAM_PINN_UNSUPERVISED=1` drops the MSE-on-$\varepsilon$
/// term from PhysicsForward; the loss becomes
/// $\lambda \cdot \mathrm{MSE}(\mathrm{NRW}(\hat\varepsilon), S)$
/// alone — the model is supervised only by the NRW reconstruction
/// constraint, never by the target $\varepsilon$.  Production
/// paths leave this unset.
fn unsupervised_pinn() -> bool {
    std::env::var("SPARAM_PINN_UNSUPERVISED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Default weight of the physics term in the hybrid PhysicsForward
/// loss when no per-trial value is supplied. HPO retrains pass the
/// trial's sampled `physics_lambda` directly; the `/train` form
/// without an explicit override falls back here.
///
/// Original 3-seed sweep across λ ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
/// 1e-1} on the default WR-90 dataset: λ=1e-3 gave the lowest average
/// val_loss while staying stable across seeds. The HPO Complex
/// search now samples this in `log [1e-3, 1]`, so studies typically
/// override the default — kept here for the `/train`-without-form
/// path and for legacy-trial retrains where `loss_params.physics_lambda`
/// is `None`.
pub(crate) const DEFAULT_PHYSICS_LAMBDA: f64 = 0.001;

/// Packed-tensor loss closure shared by the `/train` workflow and
/// the HPO retrain path. `target_scaler` is required for any
/// scale-sensitive loss (`EpsRelMse`, `PhysicsForward`) — both
/// inverse-scale their inputs before the residual computation, the
/// former to normalise by `|target|`, the latter to feed ε in
/// physical units to the NRW forward model. `feature_scaler` is used
/// by `PhysicsForward` to recover the batch's raw S-parameters from
/// the model input so no second NRW evaluation on `target_eps` is
/// needed. `physics` supplies the captured waveguide geometry +
/// frequency grid. Missing scaler or physics context degrades the
/// respective loss to MSE, matching HPO's `packed_loss_factory`
/// fallback policy.
pub(crate) fn packed_loss_fn<'a>(
    model: &'a MlpModel,
    loss: MlpLoss,
    physics_lambda: f64,
    feature_scaler: Option<ScalerRef<'a>>,
    target_scaler: Option<ScalerRef<'a>>,
    physics: Option<&'a PhysicsContext>,
) -> impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + 'a {
    // Hoist the `RelativeErrorLoss` out of the per-batch hot path.
    let rel = loss
        .needs_unscaled_targets()
        .then(|| target_scaler.map(RelativeErrorLoss::new))
        .flatten();
    let is_complex = matches!(model, MlpModel::Complex(_));
    move |input, pred, target| {
        // Always-on architectural softplus clamp on the model output:
        // ε′ ≥ 1, ε″ in the physical half-line (sign per Real/Complex
        // convention). Done in physical space via the target scaler,
        // then re-scaled before the loss kernels see it. Falls back to
        // the raw `pred` only when no scaler was wired through (test
        // fixtures); production /train and HPO loops always pass one.
        let pred_owned: Tensor;
        let pred: &Tensor = match target_scaler {
            Some(scaler) => {
                pred_owned = apply_physical_softplus_clamp(pred, scaler, is_complex)?;
                &pred_owned
            }
            None => pred,
        };
        match model {
        MlpModel::Real(_) => match loss {
            MlpLoss::Mse => mse_loss(pred, target, Reduction::Mean),
            MlpLoss::SmoothL1 { beta } => smooth_l1_loss(pred, target, beta, Reduction::Mean),
            MlpLoss::EpsRelMse => match &rel {
                Some(rel) => rel.forward(pred, target),
                None => mse_loss(pred, target, Reduction::Mean),
            },
            // Hybrid PINN loss (Real path — hit only by programmatic
            // callers; the UI routes PhysicsForward exclusively
            // through Complex):
            //   L = MSE(ε̂, ε)  +  λ · physics_forward(NRW(ε̂), S_true)
            // λ = 0.1. Pure physics_forward was unstable (basin-swap
            // collapses mid-training) because NRW is non-injective
            // in the lossy regime — the ε-MSE term pins the
            // prediction to the correct basin, the physics term adds
            // the S-reconstruction constraint as a regulariser.  This
            // empirical choice was driven by side-by-side runs of
            // `--loss hybrid` vs pure physics_forward in the now-
            // retired `debug-train` diagnostic binary.
            MlpLoss::PhysicsForward => match (&rel, feature_scaler, physics) {
                (Some(rel), Some(feat), Some(ctx)) => {
                    let mse = mse_loss(pred, target, Reduction::Mean)?;
                    let pred_unscaled = rel.scaler_ref().inverse_transform(pred)?;
                    // For Real MLPs: pred_unscaled has shape [batch, 2] with [eps_prime, eps_double_prime]
                    // For the NRW forward model to work with physical convention (eps = eps' - j*eps''),
                    // we need to negate the imaginary part before unpacking into ComplexTensor
                    let eps_prime = pred_unscaled.narrow(1, 0, 1)?;
                    let eps_double_prime_neg = pred_unscaled.narrow(1, 1, 1)?.neg()?;
                    let pred_unscaled_corrected = Tensor::cat(&[&eps_prime, &eps_double_prime_neg], 1)?;
                    let pred_complex = unpack_complex_features(&pred_unscaled_corrected, 1)?;
                    let s_raw = feat.inverse_transform(input)?;
                    let s_target = unpack_complex_features(&s_raw, 2)?;
                    let phys = physics_forward_loss(
                        &pred_complex,
                        &s_target,
                        &ctx.waveguide,
                        &ctx.frequencies,
                        Reduction::Mean,
                    )?;
                    let phys_scaled = phys.affine(physics_lambda, 0.0)?;
                    if unsupervised_pinn() {
                        let _ = mse;
                        Ok(phys_scaled)
                    } else {
                        mse.broadcast_add(&phys_scaled)
                    }
                }
                _ => mse_loss(pred, target, Reduction::Mean),
            },
        },
        MlpModel::Complex(complex) => {
            let n = complex.output_size();
            match loss {
                // PhysicsForward uses the packed `pred` directly —
                // avoid the unpack→pack roundtrip that the other
                // variants incur. `pack_complex_features` is a `cat`
                // (memory copy); skipping it saves a per-batch alloc.
                // Hybrid PINN loss:
                //   L = complex_mse(ε̂, ε)  +  λ · physics_forward(NRW(ε̂), S_true)
                // λ = `physics_lambda` (HPO trial knob; defaults to
                // `DEFAULT_PHYSICS_LAMBDA` from `/train`). See the Real arm
                // above for the rationale — same idea, unpacked once
                // on the Complex path so both MSE and physics can
                // share the single `unpack_complex_features` call.
                MlpLoss::PhysicsForward => match (&rel, feature_scaler, physics) {
                    (Some(rel), Some(feat), Some(ctx)) => {
                        let pred_complex = unpack_complex_features(pred, n)?;
                        let target_complex = unpack_complex_features(target, n)?;
                        let mse = complex_mse_loss(
                            &pred_complex,
                            &target_complex,
                            Reduction::Mean,
                        )?;
                        let pred_unscaled = rel.scaler_ref().inverse_transform(pred)?;
                        let pred_physical = unpack_complex_features(&pred_unscaled, n)?;
                        let s_raw = feat.inverse_transform(input)?;
                        let s_target = unpack_complex_features(&s_raw, 2)?;
                        let phys = physics_forward_loss(
                            &pred_physical,
                            &s_target,
                            &ctx.waveguide,
                            &ctx.frequencies,
                            Reduction::Mean,
                        )?;
                        let phys_scaled = phys.affine(physics_lambda, 0.0)?;
                        if unsupervised_pinn() {
                            let _ = mse;
                            Ok(phys_scaled)
                        } else {
                            mse.broadcast_add(&phys_scaled)
                        }
                    }
                    _ => {
                        let pred = unpack_complex_features(pred, n)?;
                        let target = unpack_complex_features(target, n)?;
                        complex_mse_loss(&pred, &target, Reduction::Mean)
                    }
                },
                _ => {
                    let pred = unpack_complex_features(pred, n)?;
                    let target = unpack_complex_features(target, n)?;
                    match loss {
                        MlpLoss::Mse => complex_mse_loss(&pred, &target, Reduction::Mean),
                        MlpLoss::SmoothL1 { beta } => {
                            complex_smooth_l1_loss(&pred, &target, beta, Reduction::Mean)
                        }
                        MlpLoss::EpsRelMse => match &rel {
                            Some(rel) => rel.forward_complex(&pred, &target),
                            None => complex_mse_loss(&pred, &target, Reduction::Mean),
                        },
                        MlpLoss::PhysicsForward => unreachable!(
                            "PhysicsForward handled by the outer arm"
                        ),
                    }
                }
            }
        }
        }
    }
}
