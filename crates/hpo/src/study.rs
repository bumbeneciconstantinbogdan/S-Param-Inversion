//! Study construction, constraint helpers, and parallel trial execution.
//!
//! [`HpoStudyBuilder`] is the main entry point: it validates an [`HpoConfig`],
//! constructs a single- or multi-objective study, and exposes sequential
//! and Rayon-parallel `optimize_*` methods.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::{HpoConfig, HpoConfigValidationError, PrunerChoice, SamplerChoice};
use crate::evaluation::{Evaluator, ModelBuilder, TrialRunner, TrialStatus};
use crate::pareto::{CompletedTrial, MultiObjectiveResults};
use crate::search_space::MlpSearchSpace;
use optimizer::FanovaResult;
use optimizer::parameter::{ParamId, ParamValue};
use optimizer::prelude::*;
use sparam_training::logger::{LogMessage, LogSender};
use std::collections::HashMap;

/// Names of the three NSGA-III objectives, in the same order as the
/// `[f64; 3]` objective vector carried on each [`CompletedTrial`] and
/// the three slots of [`ImportanceStudies`]. Exported so the app
/// workflow can label per-objective importance output consistently.
pub const OBJECTIVE_NAMES: [&str; 3] = ["ok_at_1pct", "param_count", "max_error"];

/// Three companion single-objective studies — one per NSGA-III
/// objective — used exclusively for post-hoc parameter-importance
/// analysis (fANOVA / Spearman). Each study is fed the *same* set of
/// completed-trial parameters via [`Study::enqueue`], but with a
/// **different outcome scalar**: the corresponding slot from the
/// trial's `[ok_at_1pct, param_count, max_error]` objective vector.
///
/// Prior to this type, a single companion study was fed
/// `outcome.metrics.best_val_loss` — a training-time metric that
/// isn't one of the three optimization objectives. On top of that,
/// the companion called `search_space.suggest(...)` on a fresh
/// `Trial` without `enqueue`-ing the real trial's params, so the
/// recorded (param, outcome) pairs were random params vs. real
/// outcomes — the importance numbers were essentially noise.
/// `ImportanceStudies::record` fixes both by enqueueing the exact
/// `HashMap<ParamId, ParamValue>` the real trial sampled, then
/// calling `ask → search_space.suggest → tell` for each of the three
/// objective slots.
//
// No `#[derive(Debug)]`: `optimizer::Study<V>` does not implement
// `Debug` (its internal sampler state is a trait object). Callers
// that need a readable print can format via fANOVA / Spearman
// output instead.
pub struct ImportanceStudies {
    // Array ordering matches [`OBJECTIVE_NAMES`] / the objective
    // vector slot order: [ok_at_1pct, param_count, max_error].
    // Direction is `Minimize` on all three — fANOVA / |Spearman| are
    // direction-agnostic (variance decomposition + absolute rank
    // correlation respectively), so the choice here doesn't affect
    // importance scores.
    studies: [Study<f64>; 3],
}

impl ImportanceStudies {
    #[must_use]
    pub fn new() -> Self {
        Self {
            studies: [
                Study::new(Direction::Minimize),
                Study::new(Direction::Minimize),
                Study::new(Direction::Minimize),
            ],
        }
    }

    /// Record a completed trial into all three companion studies so
    /// its params contribute to every per-objective importance map.
    ///
    /// Takes `params` by value so the final iteration can *move* the
    /// map into `Study::enqueue` instead of cloning it — shaves one
    /// `HashMap` clone off every trial. `params` MUST be a clone of
    /// `Trial::params()` from the real (multi-objective) trial,
    /// taken BEFORE `mo_study.tell(...)` consumes that trial.
    /// `objectives` is the trial's
    /// `[ok_at_1pct, param_count, max_error]` vector.
    ///
    /// `search_space` is the same [`MlpSearchSpace`] the multi-
    /// objective study uses — its `ParamId`s must match the keys in
    /// `params`, which is automatic because both studies share one
    /// search space.
    pub(crate) fn record(
        &self,
        mut params: HashMap<ParamId, ParamValue>,
        search_space: &MlpSearchSpace,
        objectives: [f64; 3],
    ) {
        let n = self.studies.len();
        for (idx, (study, objective_value)) in
            self.studies.iter().zip(objectives.iter()).enumerate()
        {
            // Inject the real trial's params as fixed values for the
            // next ask — under the hood `Study::create_trial()` calls
            // `trial.set_fixed_params(params)` on the dequeued entry,
            // which makes subsequent `Param::suggest(&mut trial)`
            // calls return the enqueued values verbatim instead of
            // sampling fresh random ones.
            //
            // The last iteration consumes `params` via `mem::take` to
            // avoid a final clone — for a 30-parameter HPO this saves
            // one HashMap<ParamId, ParamValue> copy per completed
            // trial (one of three total cycles). Earlier iterations
            // still need to clone since `study.enqueue` takes
            // ownership and we're not done with the map.
            let enqueued = if idx + 1 == n {
                std::mem::take(&mut params)
            } else {
                params.clone()
            };
            study.enqueue(enqueued);
            let mut so_trial = study.ask();
            // Walk the whole search space so every (ParamId, value)
            // from `params` is recorded on the trial — fANOVA needs
            // every sampled parameter to have a row per trial.
            let _ = search_space.suggest_best_effort(&mut so_trial);
            study.tell(so_trial, Ok::<_, &str>(*objective_value));
        }
    }

    /// Compute fANOVA main-effect importance for each of the three
    /// objectives. Returns `[ok_result, params_result, maxerr_result]`.
    /// Each slot is `Err(NoCompletedTrials)` when fewer than 2 trials
    /// have been recorded.
    #[must_use]
    pub fn fanova(&self) -> [optimizer::Result<FanovaResult>; 3] {
        [
            self.studies[0].fanova(),
            self.studies[1].fanova(),
            self.studies[2].fanova(),
        ]
    }

    /// Spearman-rank parameter importance for each of the three
    /// objectives. Used as a cheaper fallback when `fanova` would be
    /// too slow (above the trial-count cap).
    #[must_use]
    pub fn param_importance(&self) -> [Vec<(String, f64)>; 3] {
        [
            self.studies[0].param_importance(),
            self.studies[1].param_importance(),
            self.studies[2].param_importance(),
        ]
    }
}

impl Default for ImportanceStudies {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating and running HPO studies.
pub struct HpoStudyBuilder {
    config: HpoConfig,
    search_space: MlpSearchSpace,
    log: LogSender,
    /// Optional shared counter used to allocate the **display**
    /// `trial_number` (and the per-trial training RNG seed offset) via
    /// a single atomic `fetch_add`. When present, it replaces the
    /// study's internal `trial.id()` for these two purposes only —
    /// the optimizer's own per-study trial-ID bookkeeping is
    /// unchanged.
    ///
    /// Two use cases:
    /// 1. A Real + Complex sweep that runs sequentially wants
    ///    Complex's trial numbers to continue from where Real left
    ///    off (e.g. 0–49 → 50–99) so the combined trial table is
    ///    monotonic in the dashboard and every trial has a globally
    ///    unique ID. `run_hpo_with_callback` passes one counter to
    ///    both studies.
    /// 2. A future multi-study-in-parallel mode (e.g. two HPO jobs
    ///    launched concurrently from the web UI) can share one
    ///    counter so trial IDs never collide across studies.
    trial_id_counter: Option<Arc<AtomicU64>>,
}

impl std::fmt::Debug for HpoStudyBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HpoStudyBuilder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn with_optional_seed<T, F, G>(seed: Option<u64>, make_default: F, make_seeded: G) -> T
where
    F: FnOnce() -> T,
    G: FnOnce(u64) -> T,
{
    match seed {
        Some(seed) => make_seeded(seed),
        None => make_default(),
    }
}

impl HpoStudyBuilder {
    /// Create a new builder, validating the provided configuration.
    pub fn new(config: HpoConfig) -> std::result::Result<Self, HpoConfigValidationError> {
        config.validate()?;
        Ok(Self {
            config,
            search_space: MlpSearchSpace::new(),
            log: LogSender::null(),
            trial_id_counter: None,
        })
    }

    /// Override the default search space with a custom one.
    pub fn with_search_space(mut self, space: MlpSearchSpace) -> Self {
        self.search_space = space;
        self
    }

    /// Attach a logger for warnings and progress messages.
    pub fn with_log(mut self, log: LogSender) -> Self {
        self.log = log;
        self
    }

    /// Share a counter that allocates each trial's display
    /// `trial_number` and per-trial training-seed offset. See the
    /// `trial_id_counter` field doc on [`HpoStudyBuilder`] for the two
    /// motivating use cases (cross-type continuous numbering; future
    /// concurrent-study ID allocation).
    pub fn with_trial_id_counter(mut self, counter: Arc<AtomicU64>) -> Self {
        self.trial_id_counter = Some(counter);
        self
    }

    /// Allocate the next display trial number. Uses the shared counter
    /// when [`with_trial_id_counter`](Self::with_trial_id_counter) was
    /// called; otherwise falls back to the optimizer's internal
    /// per-study trial ID for backward-compatible numbering (0-based
    /// per study).
    fn next_trial_number(&self, optimizer_trial_id: u64) -> usize {
        match self.trial_id_counter.as_ref() {
            Some(counter) => counter.fetch_add(1, Ordering::SeqCst) as usize,
            None => optimizer_trial_id as usize,
        }
    }

    /// Create a single-objective study configured per `HpoConfig`.
    ///
    /// Studies run entirely in memory — no journal/persistence. Trial results
    /// are collected via callbacks and persisted by the caller (e.g. into the
    /// web app's SQLite DB).
    pub fn build_single_objective(&self) -> Study<f64> {
        let direction = self.config.directions[0];

        let mut study = Study::new(direction);

        // Apply sampler (overrides the default / placeholder set above).
        match self.config.sampler {
            SamplerChoice::Tpe => {
                study.set_sampler(with_optional_seed(
                    self.config.seed,
                    TpeSampler::new,
                    |seed| {
                        TpeSampler::builder()
                            .seed(seed)
                            .build()
                            .unwrap_or_else(|_| TpeSampler::new())
                    },
                ));
            }
            SamplerChoice::Random => {
                study.set_sampler(with_optional_seed(
                    self.config.seed,
                    RandomSampler::new,
                    RandomSampler::with_seed,
                ));
            }
            _ => {} // keep default sampler for multi-objective choices used in SO
        }

        // Apply pruner
        match self.config.pruner {
            PrunerChoice::None => {}
            PrunerChoice::Median => {
                study.set_pruner(MedianPruner::new(direction));
            }
            PrunerChoice::Percentile => {
                study.set_pruner(PercentilePruner::new(25.0, direction));
            }
            PrunerChoice::Hyperband => {
                study.set_pruner(HyperbandPruner::new());
            }
        }

        study
    }

    /// Create a multi-objective study configured per `HpoConfig`.
    pub fn build_multi_objective(&self) -> optimizer::multi_objective::MultiObjectiveStudy {
        let dirs = self.config.directions.clone();
        match self.config.sampler {
            SamplerChoice::NsgaII => optimizer::multi_objective::MultiObjectiveStudy::with_sampler(
                dirs,
                with_optional_seed(self.config.seed, Nsga2Sampler::new, Nsga2Sampler::with_seed),
            ),
            SamplerChoice::NsgaIII => {
                optimizer::multi_objective::MultiObjectiveStudy::with_sampler(
                    dirs,
                    with_optional_seed(
                        self.config.seed,
                        Nsga3Sampler::new,
                        Nsga3Sampler::with_seed,
                    ),
                )
            }
            _ => {
                // Fallback: use default (random multi-objective sampler).
                optimizer::multi_objective::MultiObjectiveStudy::new(dirs)
            }
        }
    }

    /// Reference to the search space for suggesting parameters.
    pub fn search_space(&self) -> &MlpSearchSpace {
        &self.search_space
    }

    /// Reference to the HPO config.
    pub fn config(&self) -> &HpoConfig {
        &self.config
    }

    /// Build a dedicated Rayon thread pool sized to [`effective_concurrency()`](HpoConfig::effective_concurrency).
    ///
    /// Using a dedicated pool avoids contention with other code that also uses
    /// the global Rayon pool.
    fn build_thread_pool(&self) -> optimizer::Result<rayon::ThreadPool> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.effective_concurrency())
            .stack_size(64 * 1024 * 1024) // 64 MB — candle's backward pass is deeply recursive
            .build()
            .map_err(|_| optimizer::Error::Internal("failed to build Rayon thread pool"))
    }

    /// Run a project-owned two-objective study and retain completed trial metadata.
    pub fn optimize_multi<B, E>(
        &self,
        runner: &TrialRunner<'_, B, E>,
    ) -> optimizer::Result<MultiObjectiveResults>
    where
        B: ModelBuilder,
        E: Evaluator<B::Model>,
    {
        let directions = match self.config.directions.as_slice() {
            [a, b, c] => [*a, *b, *c],
            _ => {
                return Err(optimizer::Error::ObjectiveDimensionMismatch {
                    expected: 3,
                    got: self.config.directions.len(),
                });
            }
        };

        let study = self.build_multi_objective();
        let mut completed_trials = Vec::with_capacity(self.config.n_trials);
        study.optimize(self.config.n_trials, |trial: &mut Trial| {
            let trial_number = self.next_trial_number(trial.id());
            let params = match self.search_space.suggest(trial) {
                Ok(p) => p,
                Err(e) => {
                    // Sampler crossover glitch. Return `Err` to the
                    // study — do NOT synthesise fake objectives. Feeding
                    // `Ok(worst)` here would put placeholder params in
                    // the history and NSGA-II's parent selection would
                    // propagate the poison across every later trial.
                    self.log.send(LogMessage::Warning(format!(
                        "Sampler error in sequential trial {trial_number}: {e:?}"
                    )));
                    return Err(e);
                }
            };
            let outcome = runner.run(&params);

            if outcome.status == TrialStatus::Completed {
                if let Some(objectives) = outcome.objective_triple() {
                    completed_trials.push(CompletedTrial {
                        trial_number,
                        params,
                        objectives,
                        num_objectives: 3,
                        metrics: outcome.metrics,
                        constraint_value: 0.0,
                        status: TrialStatus::Completed,
                    });
                    return Ok::<Vec<f64>, optimizer::Error>(outcome.objective_values().to_vec());
                }
            }

            completed_trials.push(CompletedTrial {
                trial_number,
                params,
                objectives: worst_objectives(directions),
                num_objectives: 3,
                metrics: outcome.metrics,
                constraint_value: 0.0,
                status: TrialStatus::Failed,
            });
            Err(optimizer::Error::Internal("multi-objective trial failed"))
        })?;

        Ok(MultiObjectiveResults {
            trials: completed_trials,
            directions,
            constraint_threshold: None,
        })
    }

    // -----------------------------------------------------------------------
    // Sequential single-objective
    // -----------------------------------------------------------------------

    /// Run a single-objective study sequentially via ask/tell, returning the
    /// configured [`Study`] so callers can query `best_trial()` etc.
    pub fn optimize_single<B, E>(
        &self,
        runner: &TrialRunner<'_, B, E>,
    ) -> optimizer::Result<Study<f64>>
    where
        B: ModelBuilder,
        E: Evaluator<B::Model>,
    {
        let study = self.build_single_objective();
        for _ in 0..self.config.n_trials {
            let mut trial = study.ask();
            let params = self.search_space.suggest(&mut trial)?;
            let outcome = runner.run(&params);
            if outcome.status == TrialStatus::Completed {
                if let Some(objective) = outcome.primary_objective() {
                    study.tell(trial, Ok::<_, &str>(objective));
                } else {
                    study.tell(trial, Err::<f64, _>("trial missing primary objective"));
                }
            } else {
                study.tell(trial, Err::<f64, _>("trial failed"));
            }
        }
        Ok(study)
    }

    // -----------------------------------------------------------------------
    // Parallel single-objective (Rayon batched ask/tell)
    // -----------------------------------------------------------------------

    /// Run a single-objective study with parallel trial evaluation via Rayon.
    ///
    /// Equivalent to [`optimize_single_parallel_with_progress`](Self::optimize_single_parallel_with_progress)
    /// with a no-op progress callback.
    pub fn optimize_single_parallel<B, E>(
        &self,
        runner: &TrialRunner<'_, B, E>,
    ) -> optimizer::Result<Study<f64>>
    where
        B: ModelBuilder + Sync,
        B::Model: Send,
        E: Evaluator<B::Model> + Sync,
    {
        self.optimize_single_parallel_with_progress(runner, &|_, _| {})
    }

    /// Run a single-objective study with parallel trial evaluation and
    /// per-trial progress reporting.
    ///
    /// `progress(completed, total)` is called from worker threads after each
    /// trial finishes evaluation. Use [`AtomicUsize`] or a channel inside the
    /// callback for thread-safe aggregation.
    pub fn optimize_single_parallel_with_progress<B, E, P>(
        &self,
        runner: &TrialRunner<'_, B, E>,
        progress: &P,
    ) -> optimizer::Result<Study<f64>>
    where
        B: ModelBuilder + Sync,
        B::Model: Send,
        E: Evaluator<B::Model> + Sync,
        P: Fn(usize, usize) + Sync,
    {
        let study = self.build_single_objective();
        let pool = self.build_thread_pool()?;
        let total = self.config.n_trials;

        for i in 0..total {
            let mut trial = study.ask();
            let params = match self.search_space.suggest(&mut trial) {
                Ok(p) => p,
                Err(e) => {
                    self.log.send(LogMessage::Warning(format!("Sampler error: {e:?}")));
                    study.tell(trial, Err::<f64, _>("sampler error"));
                    progress(i + 1, total);
                    continue;
                }
            };

            // Evaluate on rayon pool (large stack for candle backward pass)
            let outcome = pool.install(|| runner.run(&params));
            if outcome.status == TrialStatus::Completed {
                if let Some(objective) = outcome.primary_objective() {
                    study.tell(trial, Ok::<_, &str>(objective));
                } else {
                    study.tell(trial, Err::<f64, _>("missing primary objective"));
                }
            } else {
                study.tell(trial, Err::<f64, _>("trial failed"));
            }

            progress(i + 1, total);
        }

        Ok(study)
    }

    // -----------------------------------------------------------------------
    // Parallel multi-objective (Rayon batched ask/tell)
    // -----------------------------------------------------------------------

    /// Run a multi-objective study with parallel trial evaluation via Rayon.
    ///
    /// Equivalent to [`optimize_multi_parallel_with_progress`](Self::optimize_multi_parallel_with_progress)
    /// with a no-op progress callback.
    pub fn optimize_multi_parallel<B, E>(
        &self,
        runner: &TrialRunner<'_, B, E>,
    ) -> optimizer::Result<(MultiObjectiveResults, ImportanceStudies)>
    where
        B: ModelBuilder + Sync,
        B::Model: Send,
        E: Evaluator<B::Model> + Sync,
    {
        self.optimize_multi_parallel_with_progress(runner, &|_, _| {})
    }

    /// Run a multi-objective study with parallel trial evaluation and
    /// per-trial progress reporting.
    pub fn optimize_multi_parallel_with_progress<B, E, P>(
        &self,
        runner: &TrialRunner<'_, B, E>,
        progress: &P,
    ) -> optimizer::Result<(MultiObjectiveResults, ImportanceStudies)>
    where
        B: ModelBuilder + Sync,
        B::Model: Send,
        E: Evaluator<B::Model> + Sync,
        P: Fn(usize, usize) + Sync,
    {
        self.optimize_multi_parallel_with_trial_callback(runner, progress, &|_| {})
    }
    /// Like [`optimize_multi_parallel_with_progress`] but also calls `on_trial`
    /// after each trial completes, receiving the full [`CompletedTrial`] data.
    ///
    /// Uses the multi-objective NSGA-II (or NSGA-III) sampler configured by
    /// [`HpoConfig::sampler`]. The sampler receives the full
    /// `[ok_at_1pct, param_count]` objective vector and evolves the
    /// population toward the Pareto front. The second return value is a
    /// companion single-objective [`Study`] used only to carry
    /// parameter-importance (fANOVA/Spearman) for reporting.
    ///
    /// Each trial evaluation runs on a rayon thread pool with large stack
    /// (64 MB) for candle's recursive backward pass.
    pub fn optimize_multi_parallel_with_trial_callback<B, E, P, T>(
        &self,
        runner: &TrialRunner<'_, B, E>,
        progress: &P,
        on_trial: &T,
    ) -> optimizer::Result<(MultiObjectiveResults, ImportanceStudies)>
    where
        B: ModelBuilder + Sync,
        B::Model: Send,
        E: Evaluator<B::Model> + Sync,
        P: Fn(usize, usize) + Sync,
        T: Fn(&CompletedTrial) + Sync,
    {
        let directions = match self.config.directions.as_slice() {
            [a, b, c] => [*a, *b, *c],
            _ => {
                return Err(optimizer::Error::ObjectiveDimensionMismatch {
                    expected: 3,
                    got: self.config.directions.len(),
                });
            }
        };

        let mo_study = self.build_multi_objective();

        // Fallback study for when the main sampler hits the upstream
        // SBX-on-categoricals bug. `RandomMultiObjectiveSampler`
        // (default for `MultiObjectiveStudy::new`) can't hit it.
        let random_fallback_study =
            optimizer::multi_objective::MultiObjectiveStudy::new(self.config.directions.clone());

        // Companion single-objective studies for post-hoc fANOVA /
        // Spearman — one per NSGA-III objective. See
        // [`ImportanceStudies`] for the rationale (importance must be
        // reported per-objective and must be computed against the
        // real trial's params, not freshly-sampled ones).
        let importance_studies = ImportanceStudies::new();

        let pool = self.build_thread_pool()?;
        let total = self.config.n_trials;
        let mut completed_trials = Vec::with_capacity(total);

        let mut cumulative_ask = std::time::Duration::ZERO;
        let mut cumulative_train = std::time::Duration::ZERO;
        let mut cumulative_tell = std::time::Duration::ZERO;

        // Sampler-error recovery (upstream `optimizer-1.0.0` Nsga{2,3}
        // occasionally emits invalid `ParamValue::Categorical` during
        // SBX/polynomial crossover):
        //   1. `tell(Err)` the main study so the bad trial can't
        //      become a crossover parent.
        //   2. Draw a fresh trial from the random fallback and run the
        //      trainer with its valid params, recording as `Completed`.
        let mut i: usize = 0;
        while i < total {
            let t0 = std::time::Instant::now();

            let mut trial = mo_study.ask();

            let (attempted_params, sampler_err) =
                self.search_space.suggest_best_effort(&mut trial);

            if let Some(e) = sampler_err {
                let trial_number = self.next_trial_number(trial.id());
                self.log.send(LogMessage::Warning(format!(
                    "Sampler error on trial {trial_number} ({e:?}) — falling back to random-valid sampling",
                )));
                let _ = mo_study.tell(trial, Err::<Vec<f64>, _>("sampler error"));

                let mut rand_trial = random_fallback_study.ask();
                let fallback_params = match self.search_space.suggest(&mut rand_trial) {
                    Ok(p) => p,
                    Err(fallback_err) => {
                        self.log.send(LogMessage::Warning(format!(
                            "Random fallback also failed for trial {trial_number}: {fallback_err:?}",
                        )));
                        let _ = random_fallback_study
                            .tell(rand_trial, Err::<Vec<f64>, _>("fallback sampler error"));
                        let metrics = crate::evaluation::TrialMetrics {
                            best_val_loss: f64::NAN,
                            param_count: 0,
                            stopped_early: false,
                            best_epoch: 0,
                            final_epoch: 0,
                            training_time_secs: 0.0,
                            ok_at_1pct: 0.0,
                            max_error: f64::INFINITY,
                        };
                        completed_trials.push(CompletedTrial {
                            trial_number,
                            params: attempted_params,
                            objectives: worst_objectives(directions),
                            num_objectives: 3,
                            metrics,
                            constraint_value: 0.0,
                            status: TrialStatus::Failed,
                        });
                        on_trial(completed_trials.last().unwrap());
                        progress(i + 1, total);
                        i += 1;
                        continue;
                    }
                };
                cumulative_ask += t0.elapsed();

                if let Some(base_seed) = self.config.seed {
                    let trial_seed = base_seed.wrapping_add(trial_number as u64);
                    sparam_core::rng::set_global_seed(trial_seed);
                }
                let t1 = std::time::Instant::now();
                let outcome = pool.install(|| runner.run(&fallback_params));
                cumulative_train += t1.elapsed();

                let t2 = std::time::Instant::now();
                if outcome.status == TrialStatus::Completed {
                    let objectives = outcome.objective_triple().unwrap_or_else(|| {
                        [
                            outcome.metrics.ok_at_1pct,
                            outcome.metrics.param_count as f64,
                            outcome.metrics.max_error,
                        ]
                    });
                    // Snapshot BEFORE tell consumes the trial — the
                    // per-objective importance studies need the
                    // fallback trial's real params (not the failed
                    // mo_trial's, which we already rejected).
                    let params_snapshot = rand_trial.params().clone();
                    let _ = random_fallback_study
                        .tell(rand_trial, Ok::<Vec<f64>, &str>(objectives.to_vec()));
                    importance_studies.record(
                        params_snapshot,
                        &self.search_space,
                        objectives,
                    );
                    completed_trials.push(CompletedTrial {
                        trial_number,
                        params: fallback_params,
                        objectives,
                        num_objectives: 3,
                        metrics: outcome.metrics,
                        constraint_value: 0.0,
                        status: TrialStatus::Completed,
                    });
                } else {
                    let _ = random_fallback_study
                        .tell(rand_trial, Err::<Vec<f64>, _>("trial failed"));
                    completed_trials.push(CompletedTrial {
                        trial_number,
                        params: fallback_params,
                        objectives: worst_objectives(directions),
                        num_objectives: 3,
                        metrics: outcome.metrics,
                        constraint_value: 0.0,
                        status: TrialStatus::Failed,
                    });
                }
                cumulative_tell += t2.elapsed();

                on_trial(completed_trials.last().unwrap());
                progress(i + 1, total);
                i += 1;
                continue;
            }
            let params = attempted_params;
            cumulative_ask += t0.elapsed();

            // Per-trial seed: `base + trial_number` (or the optimizer's
            // internal id when no shared counter is attached). See
            // `next_trial_number`.
            let trial_number = self.next_trial_number(trial.id());
            if let Some(base_seed) = self.config.seed {
                let trial_seed = base_seed.wrapping_add(trial_number as u64);
                sparam_core::rng::set_global_seed(trial_seed);
            }

            let t1 = std::time::Instant::now();
            let outcome = pool.install(|| runner.run(&params));
            cumulative_train += t1.elapsed();

            let t2 = std::time::Instant::now();
            if outcome.status == TrialStatus::Completed {
                let objectives = outcome.objective_triple().unwrap_or_else(|| {
                    [
                        outcome.metrics.ok_at_1pct,
                        outcome.metrics.param_count as f64,
                        outcome.metrics.max_error,
                    ]
                });

                // Snapshot the mo_trial's recorded params BEFORE
                // `mo_study.tell` consumes the trial — the companion
                // studies need the exact (ParamId, ParamValue) pairs
                // to record a correct (params, objective_i) row.
                let params_snapshot = trial.params().clone();
                let _ = mo_study.tell(trial, Ok::<Vec<f64>, &str>(objectives.to_vec()));

                // Record the trial into the per-objective importance
                // studies. Each study sees the same params but a
                // different outcome scalar (objectives[0..3]), which
                // is how we get per-objective fANOVA / Spearman.
                importance_studies.record(
                    params_snapshot,
                    &self.search_space,
                    objectives,
                );

                completed_trials.push(CompletedTrial {
                    trial_number,
                    params,
                    objectives,
                    num_objectives: 3,
                    metrics: outcome.metrics,
                    constraint_value: 0.0,
                    status: TrialStatus::Completed,
                });
            } else {
                // Training failed (NaN etc.): `tell(Err)` to keep this
                // trial out of the parent pool.
                let _ = mo_study.tell(trial, Err::<Vec<f64>, _>("trial failed"));

                completed_trials.push(CompletedTrial {
                    trial_number,
                    params,
                    objectives: worst_objectives(directions),
                    num_objectives: 3,
                    metrics: outcome.metrics,
                    constraint_value: 0.0,
                    status: TrialStatus::Failed,
                });
            }
            cumulative_tell += t2.elapsed();

            on_trial(completed_trials.last().unwrap());
            progress(i + 1, total);

            // Log timing breakdown every 50 trials.
            if (i + 1) % 50 == 0 {
                self.log.send(LogMessage::Info(format!(
                    "[HPO timing @ {}] ask={:.1}ms  train={:.1}ms  tell={:.1}ms  per_trial={:.1}ms",
                    i + 1,
                    cumulative_ask.as_secs_f64() * 1000.0 / (i + 1) as f64,
                    cumulative_train.as_secs_f64() * 1000.0 / (i + 1) as f64,
                    cumulative_tell.as_secs_f64() * 1000.0 / (i + 1) as f64,
                    (cumulative_ask + cumulative_train + cumulative_tell).as_secs_f64() * 1000.0 / (i + 1) as f64,
                )));
            }
            i += 1;
        }

        Ok((MultiObjectiveResults {
            trials: completed_trials,
            directions,
            constraint_threshold: None,
        }, importance_studies))
    }
}

// ---------------------------------------------------------------------------
// Worst-objective helpers
// ---------------------------------------------------------------------------

// NOTE: the old `check_constraint` / `is_feasible` / `CONSTRAINT_BIG_M`
// helpers were removed when max_error moved from a feasibility
// constraint to the 3rd NSGA-III objective. CompletedTrial still carries
// a `constraint_value: f64` field for backwards-compat serde — it's
// always `0.0` on new runs, and `is_feasible()` on CompletedTrial now
// means "the trial completed successfully".

#[inline]
fn worst_objective_value(direction: Direction) -> f64 {
    match direction {
        Direction::Maximize => f64::NEG_INFINITY,
        Direction::Minimize => f64::INFINITY,
    }
}

#[inline]
fn worst_objectives(directions: [Direction; 3]) -> [f64; 3] {
    [
        worst_objective_value(directions[0]),
        worst_objective_value(directions[1]),
        worst_objective_value(directions[2]),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{TrialMetrics, TrialOutcome, TrialRunner, TrialStatus};
    use crate::search_space::HyperParams;
    use candle_core::Result;
    use candle_nn::VarMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_trial_types_are_thread_safe() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HyperParams>();
        assert_sync::<HyperParams>();
        assert_send::<TrialOutcome>();
        assert_send::<TrialMetrics>();
    }

    /// The companion `ImportanceStudies::record` path MUST feed each
    /// companion study the *real trial's* parameter values, not
    /// freshly-resampled random ones. Regression guard for the bug
    /// that made fANOVA importance essentially noise: before we
    /// switched to `Study::enqueue`, the companion trial called
    /// `search_space.suggest(...)` on an empty fresh `Trial` with no
    /// fixed params, so each companion trial ran the sampler from
    /// scratch and recorded RANDOM values paired with the real
    /// trial's outcomes — zero signal for fANOVA.
    ///
    /// This test fabricates a fixed `HashMap<ParamId, ParamValue>`
    /// for two `i64` params, enqueues it on a companion study, and
    /// asserts the trial's recorded `params()` after suggest match
    /// the enqueued values verbatim.
    #[test]
    fn importance_studies_enqueue_preserves_real_trial_params() {
        use optimizer::parameter::IntParam;

        // Two parameters with stable names. Would normally come from
        // `MlpSearchSpace` — here we use IntParams directly so we can
        // assert on exact values without hitting the full search space.
        let p_hidden = IntParam::new(1, 128).name("hidden_size");
        let p_bs = IntParam::new(1, 512).name("train_batch_size");

        // Fabricate a "real trial" param map: fixed, deterministic.
        let mut params = HashMap::new();
        params.insert(p_hidden.id(), ParamValue::Int(42));
        params.insert(p_bs.id(), ParamValue::Int(7));

        let companion: Study<f64> = Study::new(Direction::Minimize);

        // Enqueue the fabricated params → the next `ask()` trial is
        // pre-loaded with them as fixed values.
        companion.enqueue(params.clone());
        let mut t = companion.ask();

        // `param.suggest(&mut trial)` on a fixed-params trial returns
        // the enqueued value verbatim and records it on the trial.
        let h = p_hidden.suggest(&mut t).expect("suggest hidden");
        let b = p_bs.suggest(&mut t).expect("suggest batch_size");
        assert_eq!(h, 42, "enqueued hidden_size must be returned verbatim");
        assert_eq!(b, 7, "enqueued train_batch_size must be returned verbatim");

        // Trial's recorded params MUST exactly match the enqueued
        // map — this is what downstream fANOVA / Spearman consume.
        let recorded = t.params();
        assert_eq!(recorded.get(&p_hidden.id()), Some(&ParamValue::Int(42)));
        assert_eq!(recorded.get(&p_bs.id()), Some(&ParamValue::Int(7)));

        companion.tell(t, Ok::<_, &str>(0.25));
    }

    /// `ImportanceStudies::fanova` and `param_importance` return
    /// arrays of length 3, one slot per NSGA-III objective, ordered
    /// to match [`OBJECTIVE_NAMES`]. Both return `Err` / empty-vec
    /// respectively when the companion studies are empty — fANOVA
    /// needs ≥2 completed trials to produce a real result.
    #[test]
    fn importance_studies_return_per_objective_arrays() {
        let studies = ImportanceStudies::new();
        let fanova = studies.fanova();
        assert_eq!(fanova.len(), 3);
        for r in &fanova {
            assert!(r.is_err(), "empty study fanova must Err");
        }
        let spearman = studies.param_importance();
        assert_eq!(spearman.len(), 3);
        for v in &spearman {
            assert!(v.is_empty(), "empty study Spearman must be empty vec");
        }

        // OBJECTIVE_NAMES order is the contract consumers rely on.
        assert_eq!(
            OBJECTIVE_NAMES,
            ["ok_at_1pct", "param_count", "max_error"],
        );
    }

    struct DummyBuilder;

    impl ModelBuilder for DummyBuilder {
        type Model = usize;

        fn build(&self, params: &HyperParams) -> Result<(Self::Model, VarMap)> {
            Ok((params.hidden_size as usize, VarMap::new()))
        }
    }

    struct DummyEvaluator;

    impl Evaluator<usize> for DummyEvaluator {
        fn evaluate(
            &self,
            model: &usize,
            _varmap: &mut VarMap,
            params: &HyperParams,
        ) -> Result<TrialOutcome> {
            let ok_at_1pct = 100.0 - (params.hidden_size as f64 - 32.0).abs();
            let metrics = TrialMetrics {
                best_val_loss: 1.0 / (*model as f64 + 1.0),
                param_count: model * 10,
                stopped_early: false,
                best_epoch: 1,
                final_epoch: 1,
                training_time_secs: 0.0,
                ok_at_1pct,
                max_error: 5.0,
            };
            Ok(TrialOutcome::accuracy_complexity_error(
                ok_at_1pct,
                metrics.param_count,
                metrics.max_error,
                metrics,
                TrialStatus::Completed,
            ))
        }
    }

    #[test]
    fn builder_constructs_both_single_and_multi_objective_studies() {
        // Single-objective: direction threads through to the study.
        let cfg = HpoConfig::new("so_test")
            .single_objective(Direction::Minimize)
            .with_seed(42);
        let so = HpoStudyBuilder::new(cfg).unwrap().build_single_objective();
        assert_eq!(so.direction(), Direction::Minimize);

        // Multi-objective: `.multi_objective()` produces a 3-objective
        // study: [OK@1%, param_count, max_error].
        let cfg = HpoConfig::new("mo_test").multi_objective().with_seed(42);
        let mo = HpoStudyBuilder::new(cfg).unwrap().build_multi_objective();
        assert_eq!(mo.n_objectives(), 3);
    }

    #[test]
    fn single_objective_study_optimizes() {
        let cfg = HpoConfig::new("opt_test")
            .single_objective(Direction::Minimize)
            .with_seed(42);
        let builder = HpoStudyBuilder::new(cfg).unwrap();
        let study = builder.build_single_objective();
        let space = builder.search_space();

        for _ in 0..5 {
            let mut trial = study.ask();
            let params = space.suggest(&mut trial).unwrap();
            let score = (params.hidden_size as f64 - 32.0).powi(2) + params.lr * 100.0;
            study.tell(trial, Ok::<_, &str>(score));
        }

        assert!(study.n_trials() > 0);
    }

    #[test]
    fn multi_objective_study_produces_pareto_front() {
        let cfg = HpoConfig::new("pareto_test")
            .multi_objective()
            .with_seed(42);
        let builder = HpoStudyBuilder::new(cfg).unwrap();
        let study = builder.build_multi_objective();
        let space = builder.search_space();

        study
            .optimize(10, {
                move |trial: &mut Trial| {
                    let params = space.suggest(trial)?;
                    // 3 objectives: [accuracy (max), complexity (min),
                    // max_error (min)]. Synthetic values; the point is
                    // to exercise the 3-D Pareto path end-to-end.
                    let accuracy = 100.0 - (params.hidden_size as f64 - 32.0).abs();
                    let complexity = params.hidden_size as f64;
                    let synthetic_max_err = 1.0 / (accuracy.max(0.01) + 1.0);
                    Ok::<_, optimizer::Error>(vec![accuracy, complexity, synthetic_max_err])
                }
            })
            .unwrap();

        assert!(study.n_trials() > 0);
        let front = study.pareto_front();
        assert!(!front.is_empty());
    }

    #[test]
    fn builder_with_custom_search_space() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![16i64, 32]).name("hidden_size");

        let cfg = HpoConfig::new("custom_ss").single_objective(Direction::Minimize);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let study = builder.build_single_objective();
        let space = builder.search_space();

        for _ in 0..5 {
            let mut trial = study.ask();
            let params = space.suggest(&mut trial).unwrap();
            assert!([16, 32].contains(&params.hidden_size));
            study.tell(trial, Ok::<_, &str>(0.0));
        }
    }

    #[test]
    fn optimize_multi_returns_project_owned_pareto_results() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![8i64, 16, 32, 64]).name("hidden_size");

        let cfg = HpoConfig::new("optimize_multi")
            .multi_objective()
            .with_n_trials(6)
            .with_seed(7);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let trial_runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let results = builder.optimize_multi(&trial_runner).unwrap();

        assert!(!results.trials.is_empty());
        assert!(!results.pareto_front().is_empty());
        assert!(
            results
                .pareto_front()
                .iter()
                .all(|trial| trial.status == TrialStatus::Completed)
        );
    }

    #[test]
    fn builder_rejects_invalid_config() {
        let err = HpoStudyBuilder::new(
            HpoConfig::new("invalid")
                .with_n_trials(0), // zero trials is invalid
        )
        .unwrap_err();

        assert_eq!(err, HpoConfigValidationError::ZeroTrials);
    }

    // -----------------------------------------------------------------------
    // Sequential single-objective via optimize_single
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_single_returns_study_with_trials() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![8i64, 16, 32, 64]).name("hidden_size");

        let cfg = HpoConfig::new("single_test")
            .single_objective(Direction::Minimize)
            .with_n_trials(5)
            .with_seed(42);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let trial_runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let study = builder.optimize_single(&trial_runner).unwrap();
        assert_eq!(study.n_trials(), 5);
        let best = study.best_trial().unwrap();
        assert!(best.value.is_finite());
    }

    // -----------------------------------------------------------------------
    // Parallel single-objective
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_single_parallel_produces_same_trial_count() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![8i64, 16, 32, 64]).name("hidden_size");

        let cfg = HpoConfig::new("par_single")
            .single_objective(Direction::Minimize)
            .with_n_trials(8)
            .with_n_jobs(2)
            .with_seed(42);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let trial_runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let study = builder.optimize_single_parallel(&trial_runner).unwrap();
        assert_eq!(study.n_trials(), 8);
        let best = study.best_trial().unwrap();
        assert!(best.value.is_finite());
    }

    #[test]
    fn optimize_single_parallel_batch_larger_than_remaining() {
        // Concurrency > n_trials — should still complete all trials.
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![16i64]).name("hidden_size");

        let cfg = HpoConfig::new("par_big_batch")
            .single_objective(Direction::Minimize)
            .with_n_trials(3)
            .with_n_jobs(8)
            .with_seed(7);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let trial_runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let study = builder.optimize_single_parallel(&trial_runner).unwrap();
        assert_eq!(study.n_trials(), 3);
    }

    // -----------------------------------------------------------------------
    // Parallel multi-objective
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_multi_parallel_produces_pareto_results() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![8i64, 16, 32, 64]).name("hidden_size");

        let cfg = HpoConfig::new("par_multi")
            .multi_objective()
            .with_n_trials(8)
            .with_n_jobs(2)
            .with_seed(7);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let trial_runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let (results, _study) = builder.optimize_multi_parallel(&trial_runner).unwrap();

        assert!(!results.trials.is_empty());
        assert!(!results.pareto_front().is_empty());
        assert!(
            results
                .pareto_front()
                .iter()
                .all(|trial| trial.status == TrialStatus::Completed)
        );
    }

    #[test]
    fn optimize_multi_parallel_matches_sequential_trial_count() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![16i64, 32]).name("hidden_size");

        // Sequential
        let cfg_seq = HpoConfig::new("seq_mo")
            .multi_objective()
            .with_n_trials(6)
            .with_seed(42);
        let builder_seq = HpoStudyBuilder::new(cfg_seq)
            .unwrap()
            .with_search_space(space.clone());
        let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);
        let seq_results = builder_seq.optimize_multi(&runner).unwrap();

        // Parallel
        let cfg_par = HpoConfig::new("par_mo")
            .multi_objective()
            .with_n_trials(6)
            .with_n_jobs(3)
            .with_seed(42);
        let builder_par = HpoStudyBuilder::new(cfg_par)
            .unwrap()
            .with_search_space(space);
        let (par_results, _) = builder_par.optimize_multi_parallel(&runner).unwrap();

        assert_eq!(seq_results.trials.len(), par_results.trials.len());
    }

    #[test]
    fn optimize_multi_parallel_handles_failed_trials() {
        struct FailingBuilder;

        impl ModelBuilder for FailingBuilder {
            type Model = usize;

            fn build(&self, _params: &HyperParams) -> Result<(usize, VarMap)> {
                Err(candle_core::Error::Msg("intentional failure".to_string()))
            }
        }

        struct NeverEvaluator;

        impl Evaluator<usize> for NeverEvaluator {
            fn evaluate(
                &self,
                _model: &usize,
                _varmap: &mut VarMap,
                _params: &HyperParams,
            ) -> Result<TrialOutcome> {
                unreachable!("should never be called when build fails");
            }
        }

        let cfg = HpoConfig::new("fail_test")
            .multi_objective()
            .with_n_trials(4)
            .with_n_jobs(2)
            .with_seed(1);
        let builder = HpoStudyBuilder::new(cfg).unwrap();
        let runner = TrialRunner::new(&FailingBuilder, &NeverEvaluator);

        let (results, _study) = builder.optimize_multi_parallel(&runner).unwrap();

        // All trials failed -> all should be marked Failed
        assert_eq!(results.trials.len(), 4);
        assert!(
            results
                .trials
                .iter()
                .all(|t| t.status == TrialStatus::Failed)
        );
        assert!(results.pareto_front().is_empty());
    }

    // -----------------------------------------------------------------------
    // Progress callback
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_single_parallel_reports_progress() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![16i64, 32]).name("hidden_size");

        let cfg = HpoConfig::new("progress_so")
            .single_objective(Direction::Minimize)
            .with_n_trials(6)
            .with_n_jobs(2)
            .with_seed(42);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let max_seen = AtomicUsize::new(0);
        let total_seen = AtomicUsize::new(0);
        let study = builder
            .optimize_single_parallel_with_progress(&runner, &|completed, total| {
                max_seen.fetch_max(completed, Ordering::Relaxed);
                total_seen.store(total, Ordering::Relaxed);
            })
            .unwrap();

        assert_eq!(study.n_trials(), 6);
        assert_eq!(max_seen.load(Ordering::Relaxed), 6);
        assert_eq!(total_seen.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn optimize_multi_parallel_reports_progress() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size = CategoricalParam::new(vec![16i64, 32]).name("hidden_size");

        let cfg = HpoConfig::new("progress_mo")
            .multi_objective()
            .with_n_trials(6)
            .with_n_jobs(2)
            .with_seed(7);
        let builder = HpoStudyBuilder::new(cfg).unwrap().with_search_space(space);
        let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);

        let max_seen = AtomicUsize::new(0);
        let (results, _study) = builder
            .optimize_multi_parallel_with_progress(&runner, &|completed, _total| {
                max_seen.fetch_max(completed, Ordering::Relaxed);
            })
            .unwrap();

        assert_eq!(results.trials.len(), 6);
        assert_eq!(max_seen.load(Ordering::Relaxed), 6);
    }

    // -----------------------------------------------------------------------
    // Determinism: seeded parallel must produce same best trial
    // -----------------------------------------------------------------------

    #[test]
    fn parallel_single_is_deterministic_with_seed() {
        let space = MlpSearchSpace::new();

        let run = |seed: u64| -> f64 {
            let cfg = HpoConfig::new("det_so")
                .single_objective(Direction::Minimize)
                .with_n_trials(10)
                .with_n_jobs(2)
                .with_seed(seed);
            let builder = HpoStudyBuilder::new(cfg)
                .unwrap()
                .with_search_space(space.clone());
            let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);
            let study = builder.optimize_single_parallel(&runner).unwrap();
            study.best_trial().unwrap().value
        };

        let a = run(42);
        let b = run(42);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "same seed must produce same best value"
        );

        // Different seed may differ (not guaranteed, but highly likely)
        let c = run(99);
        // Just check both runs completed; we can't assert inequality.
        assert!(c.is_finite());
    }

    #[test]
    fn parallel_multi_is_deterministic_with_seed() {
        let space = MlpSearchSpace::new();

        let run = |seed: u64| -> Vec<[f64; 3]> {
            let cfg = HpoConfig::new("det_mo")
                .multi_objective()
                .with_n_trials(10)
                .with_n_jobs(2)
                .with_seed(seed);
            let builder = HpoStudyBuilder::new(cfg)
                .unwrap()
                .with_search_space(space.clone());
            let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);
            let (results, _study) = builder.optimize_multi_parallel(&runner).unwrap();
            let mut objs: Vec<[f64; 3]> = results.trials.iter().map(|t| t.objectives).collect();
            objs.sort_by(|a, b| {
                a[0].partial_cmp(&b[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
                    .then(a[2].partial_cmp(&b[2]).unwrap_or(std::cmp::Ordering::Equal))
            });
            objs
        };

        let a = run(42);
        let b = run(42);
        assert_eq!(a.len(), b.len());
        for (oa, ob) in a.iter().zip(b.iter()) {
            assert_eq!(oa[0].to_bits(), ob[0].to_bits());
            assert_eq!(oa[1].to_bits(), ob[1].to_bits());
            assert_eq!(oa[2].to_bits(), ob[2].to_bits());
        }
    }

    // -----------------------------------------------------------------------
    // Shared trial-id counter: cross-study continuous numbering + uniqueness
    // -----------------------------------------------------------------------

    /// Regression guard for the Real+Complex sweep requirement: when
    /// two studies share one `Arc<AtomicU64>`, their `trial_number`
    /// sequences concatenate instead of overlapping. Exercises both
    /// the cross-type-continuous-numbering and multi-concurrent-study
    /// use cases — the second fetch_add always returns a value past
    /// the first study's allocations.
    #[test]
    fn shared_trial_id_counter_gives_continuous_numbering_across_studies() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size =
            CategoricalParam::new(vec![8i64, 16, 32, 64]).name("hidden_size");

        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

        let run = |name: &str| -> Vec<usize> {
            let cfg = HpoConfig::new(name)
                .multi_objective()
                .with_n_trials(5)
                .with_seed(7);
            let builder = HpoStudyBuilder::new(cfg)
                .unwrap()
                .with_search_space(space.clone())
                .with_trial_id_counter(std::sync::Arc::clone(&counter));
            let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);
            let results = builder.optimize_multi(&runner).unwrap();
            results.trials.iter().map(|t| t.trial_number).collect()
        };

        let real_numbers = run("real");
        let complex_numbers = run("complex");

        // First study allocates 0..5.
        assert_eq!(real_numbers, vec![0, 1, 2, 3, 4]);
        // Second study continues from where the first left off.
        assert_eq!(complex_numbers, vec![5, 6, 7, 8, 9]);

        // Sanity: union is strictly increasing, no duplicates.
        let mut all = real_numbers;
        all.extend(complex_numbers);
        let sorted = {
            let mut s = all.clone();
            s.sort_unstable();
            s
        };
        assert_eq!(all, sorted);
        assert_eq!(all.len(), 10);
        assert_eq!(
            all.iter().collect::<std::collections::HashSet<_>>().len(),
            all.len(),
            "trial numbers across studies must be globally unique",
        );
    }

    /// Without a shared counter, each study keeps 0-based numbering —
    /// preserves backward compat for single-type runs.
    #[test]
    fn no_shared_counter_means_each_study_is_zero_based() {
        let mut space = MlpSearchSpace::new();
        space.hidden_size =
            CategoricalParam::new(vec![8i64, 16, 32, 64]).name("hidden_size");

        let run = || -> Vec<usize> {
            let cfg = HpoConfig::new("standalone")
                .multi_objective()
                .with_n_trials(4)
                .with_seed(7);
            let builder = HpoStudyBuilder::new(cfg)
                .unwrap()
                .with_search_space(space.clone());
            let runner = TrialRunner::new(&DummyBuilder, &DummyEvaluator);
            let results = builder.optimize_multi(&runner).unwrap();
            results.trials.iter().map(|t| t.trial_number).collect()
        };

        let first = run();
        let second = run();
        // Both restart at 0 — no global state shared.
        assert_eq!(first, vec![0, 1, 2, 3]);
        assert_eq!(second, vec![0, 1, 2, 3]);
    }

    /// Concurrent `fetch_add` from N threads on one counter yields
    /// exactly N unique IDs — validates the multi-job / parallel
    /// studies story at the counter level.
    #[test]
    fn shared_counter_allocates_unique_ids_under_concurrent_contention() {
        use std::collections::HashSet;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        use std::thread;

        let counter = Arc::new(AtomicU64::new(0));
        let n_threads = 8usize;
        let per_thread = 250usize;

        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                let mut ids = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    ids.push(c.fetch_add(1, Ordering::SeqCst));
                }
                ids
            }));
        }

        let mut all_ids: Vec<u64> = Vec::with_capacity(n_threads * per_thread);
        for h in handles {
            all_ids.extend(h.join().unwrap());
        }

        assert_eq!(all_ids.len(), n_threads * per_thread);
        assert_eq!(
            all_ids.iter().collect::<HashSet<_>>().len(),
            all_ids.len(),
            "atomic fetch_add must give every thread a unique ID",
        );
        // Final value equals the total number of allocations.
        assert_eq!(counter.load(Ordering::SeqCst), (n_threads * per_thread) as u64);
    }
}
