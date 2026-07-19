//! End-to-end HPO workflow.

use std::time::Instant;

use candle_core::Result;
use serde::{Deserialize, Serialize};

use sparam_data::generation::PermittivitySample;
use sparam_data::scaling::{Scaler, StandardScaler};
use sparam_data::tensor_bridges::samples_to_tensors;
use sparam_models::ModelType;
use sparam_hpo::{
    CompletedTrial, HpoConfig, HpoStudyBuilder, SummaryConfig, SummaryInput,
    SummaryMeta, TrialRunner, print_summary,
};
use sparam_hpo::mlp_evaluator::{RegressionEvaluator, TensorDataSource, UnifiedMlpBuilder};
use sparam_training::logger::{LogMessage, LogSender};

#[derive(Debug, Clone)]
pub struct HpoWorkflowConfig {
    /// `Arc<[_]>` lets "both"-model runs share one backing allocation.
    pub train_samples: std::sync::Arc<[PermittivitySample]>,
    pub val_samples: std::sync::Arc<[PermittivitySample]>,
    pub test_samples: std::sync::Arc<[PermittivitySample]>,
    /// "real", "complex", or "both".
    pub model_type: String,
    pub n_trials: usize,
    /// 0 = auto-detect.
    pub n_jobs: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub warmup_epochs: usize,
    pub study_name: String,
    pub seed: u64,
    /// `None` = default search space.
    pub search_space_json: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HpoSummaryOutput {
    pub study_name: String,
    pub model_type: String,
    pub n_trials: usize,
    pub feasible_trials: usize,
    pub pareto_front_size: usize,
    pub total_time_secs: f64,
    pub best_configs: Vec<BestConfigEntry>,
    /// Legacy single importance map; populated with `ok_at_1pct`.
    pub param_importance: Vec<(String, f64)>,
    /// Per-objective importance, keyed on `ok_at_1pct` / `param_count`
    /// / `max_error`. Sorted descending.
    #[serde(default)]
    pub param_importance_per_objective:
        std::collections::HashMap<String, Vec<(String, f64)>>,
    /// fANOVA aggregate interaction fraction per objective. Empty
    /// when Spearman fallback was used.
    #[serde(default)]
    pub param_interactions_per_objective:
        std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BestConfigEntry {
    pub trial: usize,
    pub ok_at_1pct: f64,
    pub hidden_size: i64,
    pub param_count: usize,
    pub max_error: f64,
    pub lr: f64,
}

#[derive(Debug, Clone)]
pub struct HpoRunResult {
    pub summaries: Vec<HpoSummaryOutput>,
    pub total_time_secs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HpoTrialEvent {
    pub trial_number: usize,
    pub total_trials: usize,
    pub model_type: String,
    pub status: String,
    pub val_loss: f64,
    pub ok_at_1pct: f64,
    pub hidden_size: i64,
    pub param_count: usize,
    pub max_error: f64,
    pub lr: f64,
    pub training_time_secs: f64,
    pub is_feasible: bool,
    pub params_json: String,
}

pub fn run_hpo(config: &HpoWorkflowConfig) -> Result<HpoRunResult> {
    run_hpo_with_callback(config, |_| {})
}

pub fn run_hpo_with_callback<F>(
    config: &HpoWorkflowConfig,
    on_trial: F,
) -> Result<HpoRunResult>
where
    F: Fn(HpoTrialEvent) + Send + Sync,
{
    sparam_core::rng::set_global_seed(config.seed);

    // `with_context` consumes the base sender so no stale `LogSender`
    // lingers — `_worker.join()` waits for every clone to drop.
    let (log, worker) = LogSender::new();
    let _worker = worker;
    let log = log.with_context(format!("hpo:{}", config.study_name));

    let train_samples = std::sync::Arc::clone(&config.train_samples);
    let val_samples = std::sync::Arc::clone(&config.val_samples);
    let test_samples = std::sync::Arc::clone(&config.test_samples);

    if train_samples.is_empty() || val_samples.is_empty() {
        return Err(candle_core::Error::Msg(
            "HPO requires non-empty train/val samples".into(),
        ));
    }
    if test_samples.is_empty() {
        return Err(candle_core::Error::Msg(
            "HPO requires a non-empty test split (objectives are computed on it)".into(),
        ));
    }

    log.send(LogMessage::Info(format!(
        "  Train: {} samples, Val: {} samples, Test: {} samples",
        train_samples.len(),
        val_samples.len(),
        test_samples.len(),
    )));

    let model_types: Vec<ModelType> = parse_model_types(&config.model_type)?;
    let start = Instant::now();
    let mut summaries = Vec::new();

    // Shared counter so Complex trial numbers continue from where Real
    // left off in "both" mode, and each trial gets a distinct seed offset.
    let trial_id_counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    for model_type in &model_types {
        let type_label = model_type.as_str();
        let full_study_name = if model_types.len() > 1 {
            format!("{}_{type_label}", config.study_name)
        } else {
            config.study_name.clone()
        };

        log.send(LogMessage::Info(format!(
            "\n--- HPO: {full_study_name} ({type_label}) ---"
        )));
        log.send(LogMessage::Info(format!(
            "  Trials: {}, Jobs: {}, Epochs/trial: {}",
            config.n_trials,
            if config.n_jobs == 0 {
                "auto".to_string()
            } else {
                config.n_jobs.to_string()
            },
            config.max_epochs,
        )));

        let summary = run_hpo_for_type(
            &train_samples,
            &val_samples,
            &test_samples,
            *model_type,
            &full_study_name,
            config,
            &log,
            &on_trial,
            std::sync::Arc::clone(&trial_id_counter),
        )?;
        summaries.push(summary);
    }

    let elapsed = start.elapsed();
    log.send(LogMessage::Info(format!(
        "\nTotal HPO time: {:.1}s",
        elapsed.as_secs_f64()
    )));

    Ok(HpoRunResult {
        summaries,
        total_time_secs: elapsed.as_secs_f64(),
    })
}

fn run_hpo_for_type<F>(
    train_samples: &[PermittivitySample],
    val_samples: &[PermittivitySample],
    test_samples: &[PermittivitySample],
    model_type: ModelType,
    study_name: &str,
    config: &HpoWorkflowConfig,
    log: &LogSender,
    on_trial: &F,
    trial_id_counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> Result<HpoSummaryOutput>
where
    F: Fn(HpoTrialEvent) + Send + Sync,
{
    let device = candle_core::Device::Cpu;

    let train_ds = samples_to_tensors(model_type, train_samples, false)?;
    let val_ds = samples_to_tensors(model_type, val_samples, false)?;
    let test_ds = samples_to_tensors(model_type, test_samples, false)?;

    let mut feature_scaler = StandardScaler::new();
    feature_scaler.fit(&train_ds.features)?;
    let mut target_scaler = StandardScaler::new();
    target_scaler.fit(&train_ds.targets)?;

    let data_source = TensorDataSource::new(
        &train_ds.features,
        &train_ds.targets,
        &val_ds.features,
        &val_ds.targets,
    )
        .with_test(&test_ds.features, &test_ds.targets)
        .with_feature_scaler(&feature_scaler)
        .with_target_scaler(&target_scaler)
        .with_shuffle(true)
        .with_seed(config.seed);

    let (input_size, output_size) = model_type.io_shape();
    let builder = UnifiedMlpBuilder::new(input_size, output_size, device.clone(), model_type);
    let physics_ctx = sparam_physics::PhysicsContext::default_wr90(&device)?;
    let evaluator =
        RegressionEvaluator::new(&data_source, config.max_epochs)
            .with_patience(config.patience)
            .with_warmup_epochs(config.warmup_epochs)
            .with_multi_objective(true)
            .with_log(log.clone())
            .with_physics_context(&physics_ctx);
    let runner = TrialRunner::new(&builder, &evaluator).with_log(log.clone());

    let hpo_config = HpoConfig::new(study_name)
        .multi_objective()
        .with_n_trials(config.n_trials)
        .with_seed(config.seed)
        .with_n_jobs(config.n_jobs);

    let custom_space = if let Some(ref json) = config.search_space_json {
        let ss_config: SearchSpaceConfig = serde_json::from_str(json)
            .map_err(|e| candle_core::Error::Msg(format!("Invalid search space JSON: {e}")))?;
        build_custom_search_space(&ss_config, model_type)?
    } else {
        match model_type {
            ModelType::Real => sparam_hpo::search_space::MlpSearchSpace::for_real(),
            ModelType::Complex => sparam_hpo::search_space::MlpSearchSpace::for_complex(),
        }
    };

    let hpo_builder = HpoStudyBuilder::new(hpo_config)
        .map_err(|e| candle_core::Error::Msg(format!("{e:?}")))?
        .with_log(log.clone())
        .with_search_space(custom_space)
        .with_trial_id_counter(trial_id_counter);

    let start = Instant::now();
    let n_trials = config.n_trials;
    let type_label_str = model_type.as_str();
    let log_progress = log.clone();
    let progress = move |completed: usize, total: usize| {
        log_progress.send(LogMessage::Info(format!("\r  Trial {completed}/{total}")));
    };
    let trial_callback = |ct: &CompletedTrial| {
        let params_json = serde_json::to_string(&ct.params).unwrap_or_else(|_| "{}".to_string());
        on_trial(HpoTrialEvent {
            trial_number: ct.trial_number,
            total_trials: n_trials,
            model_type: type_label_str.to_string(),
            status: if ct.is_successful() { "completed".into() } else { "failed".into() },
            val_loss: ct.metrics.best_val_loss,
            ok_at_1pct: ct.metrics.ok_at_1pct,
            hidden_size: ct.params.hidden_size,
            param_count: ct.metrics.param_count,
            max_error: ct.metrics.max_error,
            lr: ct.params.lr,
            training_time_secs: ct.metrics.training_time_secs,
            is_feasible: ct.is_feasible(),
            params_json,
        });
    };

    let (results, importance_studies) = hpo_builder
        .optimize_multi_parallel_with_trial_callback(&runner, &progress, &trial_callback)
        .map_err(|e| candle_core::Error::Msg(format!("HPO failed: {e:?}")))?;
    let elapsed = start.elapsed();
    log.send(LogMessage::Info(String::new()));

    // fANOVA is O(n_trees · n_trials · n_params²); falls back to
    // Spearman past the cap.
    const FANOVA_TRIAL_CAP: usize = 2000;
    let n_trials_completed = results.trials.len();
    let use_spearman = n_trials_completed > FANOVA_TRIAL_CAP;

    let fanova_results = if use_spearman {
        log.send(LogMessage::Info(format!(
            "  Parameter importance (Spearman per objective — fANOVA skipped: \
             {n_trials_completed} trials > cap {FANOVA_TRIAL_CAP})",
        )));
        None
    } else {
        Some(importance_studies.fanova())
    };
    let spearman_results = importance_studies.param_importance();

    let mut importance_per_objective: std::collections::HashMap<String, Vec<(String, f64)>> =
        std::collections::HashMap::with_capacity(3);
    let mut interactions_per_objective: std::collections::HashMap<String, f64> =
        std::collections::HashMap::with_capacity(3);
    for (idx, &obj_name) in sparam_hpo::study::OBJECTIVE_NAMES.iter().enumerate() {
        // `interaction_share`: fANOVA only; closes the `Σ main + Σ
        // interactions = 1.0` budget. Spearman → `None`.
        let (imp, interaction_share, method_label) = match &fanova_results {
            Some(arr) if arr[idx].is_ok() => {
                let result = arr[idx].as_ref().unwrap();
                let interactions: f64 =
                    result.interactions.iter().map(|(_, v)| *v).sum();
                (result.main_effects.clone(), Some(interactions), "fANOVA")
            }
            Some(_) => (spearman_results[idx].clone(), None, "Spearman — fANOVA failed"),
            None => (spearman_results[idx].clone(), None, "Spearman"),
        };

        log.send(LogMessage::Info(format!(
            "  Parameter importance ({method_label} — {obj_name}):"
        )));
        if imp.is_empty() {
            log.send(LogMessage::MetricRow {
                label: "(no completed trials)".into(),
                value: "-".into(),
            });
        } else {
            for (name, score) in &imp {
                log.send(LogMessage::MetricRow {
                    label: name.clone(),
                    value: format!("{:.1}%", score * 100.0),
                });
            }
            if let Some(interactions) = interaction_share
                && interactions > 1e-4
            {
                log.send(LogMessage::MetricRow {
                    label: "(interactions)".into(),
                    value: format!("{:.1}%", interactions * 100.0),
                });
            }
        }
        importance_per_objective.insert((*obj_name).to_string(), imp);
        if let Some(interactions) = interaction_share {
            interactions_per_objective.insert((*obj_name).to_string(), interactions);
        }
    }

    // Legacy field: keep older UIs rendering without a schema bump.
    let importance = importance_per_objective
        .get(sparam_hpo::study::OBJECTIVE_NAMES[0])
        .cloned()
        .unwrap_or_default();

    let feasible: Vec<&CompletedTrial> = results.feasible_trials();
    let pareto: Vec<&CompletedTrial> = results.pareto_front();

    log.send(LogMessage::Info(format!(
        "  Feasible: {}/{} ({:.1}%)",
        feasible.len(),
        results.trials.len(),
        feasible.len() as f64 / results.trials.len() as f64 * 100.0
    )));
    log.send(LogMessage::Info(format!(
        "  Pareto front: {} configs",
        pareto.len()
    )));

    let summary_meta = SummaryMeta::new(study_name).with_total_duration(elapsed);
    let summary_cfg = SummaryConfig {
        top_n: 10,
        use_unicode: true,
        show_metrics: true,
        ..Default::default()
    };
    print_summary(
        log,
        SummaryInput::MultiObjective {
            meta: summary_meta,
            results: &results,
        },
        &summary_cfg,
    );

    let type_label = model_type.as_str();

    let best_configs: Vec<BestConfigEntry> = pareto
        .iter()
        .map(|t| BestConfigEntry {
            trial: t.trial_number,
            ok_at_1pct: t.metrics.ok_at_1pct,
            hidden_size: t.params.hidden_size,
            param_count: t.metrics.param_count,
            max_error: t.metrics.max_error,
            lr: t.params.lr,
        })
        .collect();

    let summary_output = HpoSummaryOutput {
        study_name: study_name.to_string(),
        model_type: type_label.to_string(),
        n_trials: results.trials.len(),
        feasible_trials: feasible.len(),
        pareto_front_size: pareto.len(),
        total_time_secs: elapsed.as_secs_f64(),
        best_configs,
        param_importance: importance,
        param_importance_per_objective: importance_per_objective,
        param_interactions_per_objective: interactions_per_objective,
    };

    Ok(summary_output)
}

/// Web UI's search-space JSON shape.
///
/// Real lists (`hidden_sizes`, `activations`) and Complex lists
/// (`complex_*`) are **not** cross-applied; numeric ranges and enum
/// subsets fall back across types when no Complex override is set.
/// Unknown fields are ignored by serde for older-UI compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchSpaceConfig {
    pub hidden_sizes: Option<Vec<i64>>,
    pub activations: Option<Vec<String>>,
    pub lr_min: Option<f64>,
    pub lr_max: Option<f64>,
    pub weight_decay_min: Option<f64>,
    pub weight_decay_max: Option<f64>,
    pub dropout_enabled: Option<bool>,
    pub dropout_min: Option<f64>,
    pub dropout_max: Option<f64>,
    pub norms: Option<Vec<String>>,
    pub noise_enabled: Option<bool>,

    /// `None` = sample every variant; a non-empty list subsets.
    /// Unknown names are dropped silently.
    pub optimizers: Option<Vec<String>>,
    pub batch_sizes: Option<Vec<String>>,
    pub losses: Option<Vec<String>>,
    pub schedulers: Option<Vec<String>>,
    /// Single-sided UI toggle: `["none"]` = no clipping, absent =
    /// sample every variant. No `complex_grad_clips`.
    pub grad_clips: Option<Vec<String>>,

    pub complex_activations: Option<Vec<String>>,
    pub complex_hidden_sizes: Option<Vec<i64>>,
    pub complex_lr_min: Option<f64>,
    pub complex_lr_max: Option<f64>,
    pub complex_weight_decay_min: Option<f64>,
    pub complex_weight_decay_max: Option<f64>,
    pub complex_optimizers: Option<Vec<String>>,
    pub complex_batch_sizes: Option<Vec<String>>,
    pub complex_losses: Option<Vec<String>>,
    pub complex_schedulers: Option<Vec<String>>,
    pub complex_dropout_enabled: Option<bool>,
    pub complex_dropout_min: Option<f64>,
    pub complex_dropout_max: Option<f64>,
    pub complex_grad_clips: Option<Vec<String>>,
    pub complex_noise_enabled: Option<bool>,
    /// Legacy PI-MAPE β range; ignored by the live physics-forward path.
    pub pi_mape_beta_min: Option<f64>,
    pub pi_mape_beta_max: Option<f64>,
}

impl SearchSpaceConfig {
    fn effective(&self, model_type: ModelType) -> EffectiveSearchSpace<'_> {
        match model_type {
            ModelType::Real => EffectiveSearchSpace {
                hidden_sizes: self.hidden_sizes.as_ref(),
                activations: self.activations.as_ref(),
                lr_min: self.lr_min,
                lr_max: self.lr_max,
                weight_decay_min: self.weight_decay_min,
                weight_decay_max: self.weight_decay_max,
                dropout_enabled: self.dropout_enabled,
                dropout_min: self.dropout_min,
                dropout_max: self.dropout_max,
                norms: self.norms.as_ref(),
                noise_enabled: self.noise_enabled,
                optimizers: self.optimizers.as_ref(),
                batch_sizes: self.batch_sizes.as_ref(),
                losses: self.losses.as_ref(),
                schedulers: self.schedulers.as_ref(),
                grad_clips: self.grad_clips.as_ref(),
                pi_mape_beta_min: None,
                pi_mape_beta_max: None,
            },
            ModelType::Complex => EffectiveSearchSpace {
                hidden_sizes: self.complex_hidden_sizes.as_ref(),
                activations: self.complex_activations.as_ref(),
                lr_min: self.complex_lr_min.or(self.lr_min),
                lr_max: self.complex_lr_max.or(self.lr_max),
                weight_decay_min: self.complex_weight_decay_min.or(self.weight_decay_min),
                weight_decay_max: self.complex_weight_decay_max.or(self.weight_decay_max),
                dropout_enabled: self.complex_dropout_enabled.or(self.dropout_enabled),
                dropout_min: self.complex_dropout_min.or(self.dropout_min),
                dropout_max: self.complex_dropout_max.or(self.dropout_max),
                // Real-only by design; ignored on Complex.
                norms: self.norms.as_ref(),
                noise_enabled: self.complex_noise_enabled.or(self.noise_enabled),
                optimizers: self.complex_optimizers.as_ref().or(self.optimizers.as_ref()),
                batch_sizes: self
                    .complex_batch_sizes
                    .as_ref()
                    .or(self.batch_sizes.as_ref()),
                losses: self.complex_losses.as_ref().or(self.losses.as_ref()),
                schedulers: self
                    .complex_schedulers
                    .as_ref()
                    .or(self.schedulers.as_ref()),
                grad_clips: self
                    .complex_grad_clips
                    .as_ref()
                    .or(self.grad_clips.as_ref()),
                pi_mape_beta_min: self.pi_mape_beta_min,
                pi_mape_beta_max: self.pi_mape_beta_max,
            },
        }
    }
}

struct EffectiveSearchSpace<'a> {
    hidden_sizes: Option<&'a Vec<i64>>,
    activations: Option<&'a Vec<String>>,
    lr_min: Option<f64>,
    lr_max: Option<f64>,
    weight_decay_min: Option<f64>,
    weight_decay_max: Option<f64>,
    dropout_enabled: Option<bool>,
    dropout_min: Option<f64>,
    dropout_max: Option<f64>,
    norms: Option<&'a Vec<String>>,
    noise_enabled: Option<bool>,
    optimizers: Option<&'a Vec<String>>,
    batch_sizes: Option<&'a Vec<String>>,
    losses: Option<&'a Vec<String>>,
    schedulers: Option<&'a Vec<String>>,
    grad_clips: Option<&'a Vec<String>>,
    pi_mape_beta_min: Option<f64>,
    pi_mape_beta_max: Option<f64>,
}

fn build_custom_search_space(
    config: &SearchSpaceConfig,
    model_type: ModelType,
) -> Result<sparam_hpo::search_space::MlpSearchSpace> {
    use optimizer::prelude::*;
    use sparam_hpo::search_space::*;

    let eff = config.effective(model_type);

    let mut space = match model_type {
        ModelType::Real => MlpSearchSpace::for_real(),
        ModelType::Complex => MlpSearchSpace::for_complex(),
    };

    if let Some(sizes) = eff.hidden_sizes {
        if !sizes.is_empty() {
            space.hidden_size = CategoricalParam::new(sizes.clone()).name("hidden_size");
        }
    }

    if let Some(acts) = eff.activations {
        // Filter to the family's valid names; unknown drop silently.
        let filtered: Vec<String> = acts
            .iter()
            .filter(|name| match model_type {
                ModelType::Real => sparam_models::activation::Activation::from_name(name).is_ok(),
                ModelType::Complex => {
                    sparam_models::complex_activation::ComplexActivation::from_name(name).is_ok()
                }
            })
            .cloned()
            .collect();
        if !filtered.is_empty() {
            space.activation = CategoricalParam::new(filtered).name("activation");
        }
    }

    if let (Some(min), Some(max)) = (eff.lr_min, eff.lr_max) {
        if min > 0.0 && max > min {
            space.lr = FloatParam::new(min, max).log_scale().name("lr");
        }
    }

    if let (Some(min), Some(max)) = (eff.weight_decay_min, eff.weight_decay_max) {
        if min > 0.0 && max > min {
            space.weight_decay = FloatParam::new(min, max).log_scale().name("weight_decay");
        }
    }

    let _ = (eff.pi_mape_beta_min, eff.pi_mape_beta_max);

    // Enum-variant subsetting: parse via `from_alias`, drop unknowns,
    // only swap when the filter produces at least one valid variant.
    if let Some(labels) = eff.norms
        && !labels.is_empty()
    {
        let choices: Vec<NormChoice> = labels.iter().filter_map(|s| NormChoice::from_alias(s)).collect();
        if !choices.is_empty() {
            space.norm = CategoricalParam::new(choices).name("norm");
        }
    }

    if let Some(names) = eff.optimizers {
        let choices: Vec<OptimizerChoice> =
            names.iter().filter_map(|s| OptimizerChoice::from_alias(s)).collect();
        if !choices.is_empty() {
            space.optimizer = CategoricalParam::new(choices).name("optimizer");
        }
    }

    if let Some(names) = eff.batch_sizes {
        let choices: Vec<BatchSizeChoice> =
            names.iter().filter_map(|s| BatchSizeChoice::from_alias(s)).collect();
        if !choices.is_empty() {
            space.train_batch_size = CategoricalParam::new(choices).name("train_batch_size");
        }
    }

    if let Some(names) = eff.losses {
        let choices: Vec<LossChoice> = names
            .iter()
            .filter_map(|s| LossChoice::from_alias(s))
            .filter(|loss| match model_type {
                ModelType::Real => loss.is_real_compatible(),
                ModelType::Complex => loss.is_complex_compatible(),
            })
            .collect();
        if !choices.is_empty() {
            space.loss = CategoricalParam::new(choices).name("loss");
        }
    }

    if let Some(names) = eff.schedulers {
        let choices: Vec<SchedulerChoice> =
            names.iter().filter_map(|s| SchedulerChoice::from_alias(s)).collect();
        if !choices.is_empty() {
            space.scheduler = CategoricalParam::new(choices).name("scheduler");
        }
    }

    if let Some(names) = eff.grad_clips {
        let choices: Vec<GradClipChoice> =
            names.iter().filter_map(|s| GradClipChoice::from_alias(s)).collect();
        if !choices.is_empty() {
            space.grad_clip_norm = CategoricalParam::new(choices).name("grad_clip_norm");
        }
    }

    if eff.dropout_enabled == Some(false) {
        space.dropout_p = FloatParam::new(0.0, 1e-15).name("dropout_p");
    } else if let Some(max) = eff.dropout_max {
        // Clamp both bounds into [0, 1]; fall back to [0, max] on an
        // inverted range instead of erroring.
        let raw_min = eff.dropout_min.unwrap_or(0.0).max(0.0);
        let raw_max = max.min(1.0);
        let (min, max) = if raw_min <= raw_max {
            (raw_min, raw_max)
        } else {
            (0.0, raw_max)
        };
        space.dropout_p = FloatParam::new(min, max).name("dropout_p");
    }

    if eff.noise_enabled == Some(false) {
        space.input_noise_std = FloatParam::new(0.0, 1e-15).name("input_noise_std");
    }

    Ok(space)
}

fn parse_model_types(s: &str) -> Result<Vec<ModelType>> {
    match s.trim().to_ascii_lowercase().as_str() {
        "both" => Ok(vec![ModelType::Real, ModelType::Complex]),
        other => other
            .parse::<ModelType>()
            .map(|m| vec![m])
            .map_err(|e| candle_core::Error::Msg(format!("{e}, or 'both'"))),
    }
}

#[cfg(test)]
mod tests {
    use super::{HpoSummaryOutput, ModelType, SearchSpaceConfig, build_custom_search_space, parse_model_types};
    use optimizer::Direction;
    use optimizer::multi_objective::MultiObjectiveStudy;
    use optimizer::parameter::{FloatParam, IntParam, Parameter};
    use optimizer::sampler::nsga2::Nsga2Sampler;

    #[test]
    fn parse_model_types_handles_real_complex_both_and_rejects_unknown() {
        assert_eq!(parse_model_types("real").unwrap(), vec![ModelType::Real]);
        assert_eq!(parse_model_types("complex").unwrap(), vec![ModelType::Complex]);
        assert_eq!(
            parse_model_types("both").unwrap(),
            vec![ModelType::Real, ModelType::Complex],
        );

        assert_eq!(parse_model_types("  REAL ").unwrap(), vec![ModelType::Real]);
        assert_eq!(parse_model_types("Both").unwrap(), vec![ModelType::Real, ModelType::Complex]);

        let err = parse_model_types("elephant").unwrap_err().to_string();
        assert!(err.contains("or 'both'"), "got: {err}");
    }

    #[test]
    fn complex_overrides_separate_hidden_size_from_real() {
        let cfg = SearchSpaceConfig {
            hidden_sizes: Some(vec![8, 16, 32]),
            complex_hidden_sizes: Some(vec![4, 8, 12]),
            ..Default::default()
        };

        let real = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();

        let real_repr = format!("{:?}", real.hidden_size);
        let complex_repr = format!("{:?}", complex.hidden_size);
        assert_ne!(real_repr, complex_repr);
        assert!(real_repr.contains('8') && real_repr.contains("16") && real_repr.contains("32"));
        assert!(complex_repr.contains('4') && complex_repr.contains('8') && complex_repr.contains("12"));
    }

    #[test]
    fn complex_does_not_inherit_base_hidden_sizes_or_activations() {
        let cfg = SearchSpaceConfig {
            hidden_sizes: Some(vec![64, 128]),
            activations: Some(vec!["relu".into(), "tanh".into()]),
            // Intentionally no `complex_hidden_sizes` / `complex_activations`.
            ..Default::default()
        };

        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();

        let sizes_repr = format!("{:?}", complex.hidden_size);
        // Complex defaults from `for_complex()`: [6, 12, 24, 48].
        assert!(sizes_repr.contains('6') && sizes_repr.contains("12"));
        // The Real-specific sizes 64 and 128 must NOT leak in.
        assert!(
            !sizes_repr.contains("64") && !sizes_repr.contains("128"),
            "Real hidden sizes leaked into Complex: {sizes_repr}"
        );

        // Activation defaults from `for_complex()`: crelu, cgelu, …
        // Real activations "relu" / "tanh" must NOT be there.
        let acts_repr = format!("{:?}", complex.activation);
        assert!(acts_repr.contains("crelu") || acts_repr.contains("cgelu"));
        assert!(
            !acts_repr.contains("\"relu\"") && !acts_repr.contains("tanh"),
            "Real activations leaked into Complex: {acts_repr}"
        );
    }

    #[test]
    fn complex_inherits_base_lr_range_when_no_override() {
        let cfg = SearchSpaceConfig {
            lr_min: Some(2e-4),
            lr_max: Some(1e-2),
            ..Default::default()
        };

        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();
        let lr_repr = format!("{:?}", complex.lr);
        assert!(lr_repr.contains("0.0002") || lr_repr.contains("2e-4"));
    }

    #[test]
    fn complex_inherits_base_weight_decay_range_when_no_override() {
        let cfg = SearchSpaceConfig {
            weight_decay_min: Some(5e-10),
            weight_decay_max: Some(3e-3),
            ..Default::default()
        };

        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();
        let wd_repr = format!("{:?}", complex.weight_decay);
        assert!(
            wd_repr.contains("5e-10") || wd_repr.contains("0.0000000005"),
            "Complex must inherit base weight_decay_min: {wd_repr}"
        );
        assert!(
            wd_repr.contains("3e-3") || wd_repr.contains("0.003"),
            "Complex must inherit base weight_decay_max: {wd_repr}"
        );
    }

    #[test]
    fn complex_weight_decay_override_wins_over_base() {
        let cfg = SearchSpaceConfig {
            weight_decay_min: Some(5e-10),
            weight_decay_max: Some(3e-3),
            complex_weight_decay_min: Some(7e-7),
            complex_weight_decay_max: Some(4e-4),
            ..Default::default()
        };

        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();
        let wd_repr = format!("{:?}", complex.weight_decay);
        assert!(
            wd_repr.contains("7e-7") || wd_repr.contains("0.0000007"),
            "Complex override min must take precedence: {wd_repr}"
        );
        assert!(
            wd_repr.contains("4e-4") || wd_repr.contains("0.0004"),
            "Complex override max must take precedence: {wd_repr}"
        );
    }

    #[test]
    fn activations_are_filtered_to_model_types_whitelist() {
        let cfg_real = SearchSpaceConfig {
            activations: Some(vec![
                "relu".into(),
                "tanh".into(),
                "crelu".into(), // Complex — must be dropped.
                "cardioid".into(), // Complex — must be dropped.
            ]),
            ..Default::default()
        };
        let real = build_custom_search_space(&cfg_real, ModelType::Real).unwrap();
        let repr = format!("{:?}", real.activation);
        assert!(repr.contains("relu") && repr.contains("tanh"));
        assert!(
            !repr.contains("crelu") && !repr.contains("cardioid"),
            "Complex activations leaked into Real: {repr}"
        );

        // Complex study gets a mixed list including Real-only names.
        let cfg_complex = SearchSpaceConfig {
            complex_activations: Some(vec![
                "crelu".into(),
                "cardioid".into(),
                "relu".into(), // Real — must be dropped.
                "gelu".into(), // Real — must be dropped.
            ]),
            ..Default::default()
        };
        let complex =
            build_custom_search_space(&cfg_complex, ModelType::Complex).unwrap();
        let repr = format!("{:?}", complex.activation);
        assert!(repr.contains("crelu") && repr.contains("cardioid"));
        // `"relu"` is a substring of `"crelu"` so match only the bare
        // token. Same for `"gelu"` vs `"cgelu"`: match the exact string
        // with surrounding quotes as serialized in the Debug output.
        assert!(
            !repr.contains("\"relu\"") && !repr.contains("\"gelu\""),
            "Real activations leaked into Complex: {repr}"
        );
    }

    #[test]
    fn empty_filtered_activations_falls_back_to_type_defaults() {
        let cfg = SearchSpaceConfig {
            activations: Some(vec!["crelu".into(), "cardioid".into()]),
            ..Default::default()
        };
        let real = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let repr = format!("{:?}", real.activation);
        // Should contain Real defaults, not be empty.
        assert!(repr.contains("relu") || repr.contains("tanh"));
    }

    #[test]
    fn enum_subset_toggles_restrict_sampler() {
        let cfg = SearchSpaceConfig {
            optimizers: Some(vec!["adam".into()]),
            batch_sizes: Some(vec!["32".into(), "64".into()]),
            losses: Some(vec!["smooth_l1".into()]),
            schedulers: Some(vec!["cosine".into()]),
            grad_clips: Some(vec!["none".into()]),
            ..Default::default()
        };
        let real = build_custom_search_space(&cfg, ModelType::Real).unwrap();

        let opt_repr = format!("{:?}", real.optimizer);
        assert!(opt_repr.contains("Adam"));
        assert!(
            !opt_repr.contains("AdamW")
                && !opt_repr.contains("SGD")
                && !opt_repr.contains("RMSprop"),
            "optimizer subset not applied: {opt_repr}"
        );

        let bs_repr = format!("{:?}", real.train_batch_size);
        assert!(bs_repr.contains("B32") && bs_repr.contains("B64"));
        assert!(
            !bs_repr.contains("B8")
                && !bs_repr.contains("B128")
                && !bs_repr.contains("B256")
                && !bs_repr.contains("All"),
            "batch_sizes subset not applied: {bs_repr}"
        );

        let loss_repr = format!("{:?}", real.loss);
        assert!(loss_repr.contains("SmoothL1"));
        assert!(
            !loss_repr.contains("Mse") && !loss_repr.contains("EpsRelMse"),
            "losses subset not applied: {loss_repr}"
        );

        let sched_repr = format!("{:?}", real.scheduler);
        assert!(sched_repr.contains("Cosine"));
        assert!(
            !sched_repr.contains("Plateau"),
            "schedulers subset not applied: {sched_repr}"
        );

        let gc_repr = format!("{:?}", real.grad_clip_norm);
        assert!(gc_repr.contains("None"));
        assert!(
            !gc_repr.contains("Clip05") && !gc_repr.contains("Clip50"),
            "grad_clips subset not applied: {gc_repr}"
        );
    }

    #[test]
    fn complex_inherits_base_enum_subsets_when_no_override() {
        let cfg = SearchSpaceConfig {
            optimizers: Some(vec!["adam".into()]),
            batch_sizes: Some(vec!["64".into()]),
            schedulers: Some(vec!["cosine".into()]),
            ..Default::default()
        };
        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();

        assert!(format!("{:?}", complex.optimizer).contains("Adam"));
        assert!(!format!("{:?}", complex.optimizer).contains("SGD"));
        assert!(format!("{:?}", complex.train_batch_size).contains("B64"));
        assert!(format!("{:?}", complex.scheduler).contains("Cosine"));
        assert!(!format!("{:?}", complex.scheduler).contains("Plateau"));
    }

    /// Per-family compatibility: Complex now accepts `mse` /
    /// `smooth_l1` (they dispatch to complex kernels via the
    /// model-aware loss factory), but `eps_rel_mse` is filtered out.
    #[test]
    fn eps_rel_mse_is_rejected_for_complex_model_type() {
        let cfg = SearchSpaceConfig {
            losses: Some(vec!["mse".into(), "smooth_l1".into(), "eps_rel_mse".into()]),
            ..Default::default()
        };
        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();
        let loss_repr = format!("{:?}", complex.loss);
        assert!(loss_repr.contains("Mse"));
        assert!(loss_repr.contains("SmoothL1"));
        assert!(!loss_repr.contains("EpsRelMse"));
    }

    /// Mirror of the above: the complex-only loss on a Real study is
    /// filtered out — accepts both the new name and the legacy alias.
    #[test]
    fn complex_only_loss_is_rejected_for_real_model_type() {
        let cfg = SearchSpaceConfig {
            losses: Some(vec!["mse".into(), "physics_forward".into()]),
            ..Default::default()
        };
        let real = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let loss_repr = format!("{:?}", real.loss);
        assert!(loss_repr.contains("Mse"));
        assert!(!loss_repr.contains("PhysicsForward"));
    }

    /// Complex-side override wins over the base field when both are
    /// set — e.g. user wants Real to sample all optimizers but
    /// Complex to only try Adam.
    #[test]
    fn complex_enum_subset_override_wins_over_base() {
        let cfg = SearchSpaceConfig {
            optimizers: Some(vec!["adam".into(), "sgd".into()]),
            complex_optimizers: Some(vec!["adam".into()]),
            ..Default::default()
        };
        let real = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();

        let real_repr = format!("{:?}", real.optimizer);
        assert!(real_repr.contains("Adam") && real_repr.contains("SGD"));

        let complex_repr = format!("{:?}", complex.optimizer);
        assert!(complex_repr.contains("Adam"));
        assert!(
            !complex_repr.contains("SGD"),
            "complex override did not take precedence: {complex_repr}"
        );
    }

    #[test]
    fn hpo_summary_output_serializes_per_objective_importance() {
        use std::collections::HashMap;

        let mut per_obj: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        per_obj.insert(
            "ok_at_1pct".to_string(),
            vec![("lr".into(), 0.5), ("activation".into(), 0.3)],
        );
        per_obj.insert(
            "param_count".to_string(),
            vec![("hidden_size".into(), 0.9)],
        );
        per_obj.insert(
            "max_error".to_string(),
            vec![("lr".into(), 0.4), ("dropout_p".into(), 0.2)],
        );

        let summary = HpoSummaryOutput {
            study_name: "t".into(),
            model_type: "real".into(),
            n_trials: 10,
            feasible_trials: 10,
            pareto_front_size: 3,
            total_time_secs: 1.0,
            best_configs: vec![],
            param_importance: vec![("lr".into(), 0.5)],
            param_importance_per_objective: per_obj,
            param_interactions_per_objective: std::collections::HashMap::new(),
        };

        let json = serde_json::to_value(&summary).expect("serialize");
        let obj_map = json
            .get("param_importance_per_objective")
            .and_then(|v| v.as_object())
            .expect("per-objective field present");

        assert!(obj_map.contains_key("ok_at_1pct"));
        assert!(obj_map.contains_key("param_count"));
        assert!(obj_map.contains_key("max_error"));

        let legacy = json
            .get("param_importance")
            .and_then(|v| v.as_array())
            .expect("legacy param_importance field present");
        assert_eq!(legacy.len(), 1);
    }

    #[test]
    #[ignore = "Requires full dataset generation; covered by integration runs"]
    fn param_importance_per_objective_populates_all_three_keys() {}

    #[test]
    fn enum_subset_unknown_names_are_ignored_and_empty_falls_back() {
        let cfg = SearchSpaceConfig {
            optimizers: Some(vec![
                "adam".into(),
                "lbfgs".into(),
                "nonsense".into(),
            ]),
            ..Default::default()
        };
        let space = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let repr = format!("{:?}", space.optimizer);
        assert!(repr.contains("Adam"));
        assert!(!repr.contains("SGD"));

        let cfg_empty = SearchSpaceConfig {
            optimizers: Some(vec!["nonsense".into(), "xyz".into()]),
            ..Default::default()
        };
        let fallback = build_custom_search_space(&cfg_empty, ModelType::Real).unwrap();
        let fallback_repr = format!("{:?}", fallback.optimizer);
        assert!(fallback_repr.contains("Adam") && fallback_repr.contains("SGD"));
    }

    #[test]
    fn dropout_min_is_honoured_when_supplied() {
        let cfg = SearchSpaceConfig {
            dropout_enabled: Some(true),
            dropout_min: Some(0.1),
            dropout_max: Some(0.5),
            ..Default::default()
        };
        let space = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let repr = format!("{:?}", space.dropout_p);
        assert!(repr.contains("0.1"));
        assert!(repr.contains("0.5"));
    }

    #[test]
    fn dropout_bounds_are_clamped_and_fallback_on_inverted_range() {
        let cfg = SearchSpaceConfig {
            dropout_enabled: Some(true),
            dropout_min: Some(0.9),
            dropout_max: Some(0.3),
            ..Default::default()
        };
        let space = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let repr = format!("{:?}", space.dropout_p);
        assert!(repr.contains("0.3"));
        assert!(!repr.contains("0.9"));
    }

    #[test]
    fn norm_subset_is_applied() {
        let cfg = SearchSpaceConfig {
            norms: Some(vec!["none".into(), "layernorm".into()]),
            ..Default::default()
        };
        let space = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let repr = format!("{:?}", space.norm);
        assert!(repr.contains("None"));
        assert!(repr.contains("LayerNorm"));
        assert!(!repr.contains("BatchNorm"));
    }

    #[test]
    fn every_complex_override_field_is_routed() {
        let cfg = SearchSpaceConfig {
            hidden_sizes: Some(vec![8]),
            complex_hidden_sizes: Some(vec![100]),
            lr_min: Some(1e-4),
            lr_max: Some(1e-2),
            complex_lr_min: Some(1e-3),
            complex_lr_max: Some(5e-2),
            ..Default::default()
        };

        let real = build_custom_search_space(&cfg, ModelType::Real).unwrap();
        let complex = build_custom_search_space(&cfg, ModelType::Complex).unwrap();

        assert_ne!(format!("{:?}", real.hidden_size), format!("{:?}", complex.hidden_size));
        assert_ne!(format!("{:?}", real.lr), format!("{:?}", complex.lr));
    }

    #[test]
    fn nsga2_diverges_from_ok_at_1pct_signal_alone() {
        let directions = vec![Direction::Maximize, Direction::Minimize];
        let n_trials: usize = 40;
        let base_seed: u64 = 123;

        let run_study = |optimum_h: f64| -> Vec<i64> {
            let study = MultiObjectiveStudy::with_sampler(
                directions.clone(),
                Nsga2Sampler::with_seed(base_seed),
            );
            let hidden = IntParam::new(4, 256).name("hidden_size");
            let lr = FloatParam::new(1e-4, 5e-2).log_scale().name("lr");

            let mut hidden_history: Vec<i64> = Vec::with_capacity(n_trials);
            for _ in 0..n_trials {
                let mut trial = study.ask();
                let h = hidden.suggest(&mut trial).unwrap();
                let _ = lr.suggest(&mut trial).unwrap();

                let ok_at_1pct = 100.0 - (h as f64 - optimum_h).abs();
                let param_count = h as f64;
                study
                    .tell(trial, Ok::<Vec<f64>, &str>(vec![ok_at_1pct, param_count]))
                    .unwrap();
                hidden_history.push(h);
            }
            hidden_history
        };

        let study_a = run_study(32.0);
        let study_b = run_study(200.0);
        assert_eq!(study_a.len(), study_b.len());

        let tail_start = (n_trials * 3) / 4;
        assert_ne!(&study_a[tail_start..], &study_b[tail_start..]);
    }

    #[test]
    fn nsga2_keeps_producing_valid_trials_after_tell_err_failures() {
        let directions = vec![Direction::Maximize, Direction::Minimize];
        let n_trials: usize = 40;
        let base_seed: u64 = 7;
        let fail_window = 12usize..24usize;

        let study = MultiObjectiveStudy::with_sampler(
            directions.clone(),
            Nsga2Sampler::with_seed(base_seed),
        );
        let hidden = IntParam::new(4, 256).name("hidden_size");
        let lr = FloatParam::new(1e-4, 5e-2).log_scale().name("lr");

        let mut succeeded_after_window = 0usize;
        for trial_idx in 0..n_trials {
            let mut trial = study.ask();
            let h = hidden.suggest(&mut trial).unwrap();
            let _ = lr.suggest(&mut trial).unwrap();

            if fail_window.contains(&trial_idx) {
                study
                    .tell(trial, Err::<Vec<f64>, &str>("synthetic failure"))
                    .unwrap();
                continue;
            }

            let ok = 100.0 - (h as f64 - 32.0).abs();
            let pc = h as f64;
            study
                .tell(trial, Ok::<Vec<f64>, &str>(vec![ok, pc]))
                .unwrap();
            if trial_idx >= fail_window.end {
                succeeded_after_window += 1;
            }
        }

        let post_window = n_trials - fail_window.end;
        assert_eq!(
            succeeded_after_window, post_window,
            "all {post_window} trials after the failure window should \
             still sample valid params; got {succeeded_after_window} — \
             the sampler has entered the lock-up failure mode",
        );
    }
}

