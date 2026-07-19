//! HPO study configuration: sampler, pruner, directions, and persistence.

use optimizer::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sampler / Pruner choice enums
// ---------------------------------------------------------------------------

/// Sampler algorithm for parameter suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerChoice {
    Random,
    #[default]
    Tpe,
    #[serde(rename = "nsga2")]
    NsgaII,
    #[serde(rename = "nsga3")]
    NsgaIII,
}

impl SamplerChoice {
    /// Whether this sampler is designed for multi-objective optimization.
    pub fn is_multi_objective(self) -> bool {
        matches!(self, Self::NsgaII | Self::NsgaIII)
    }
}

/// Pruner algorithm for early trial termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrunerChoice {
    #[default]
    None,
    Median,
    Percentile,
    Hyperband,
}

/// Format for exporting optimization results after a study completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageFormat {
    /// Single JSON file — human-readable, easy to parse with `jq` or Python.
    #[default]
    Json,
    /// SQLite database — compatible with Python analysis scripts.
    /// Requires the `storage-sqlite` feature.
    Sqlite,
}

/// Validation errors for HPO study configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpoConfigValidationError {
    EmptyDirections,
    ZeroTrials,
    IncompatibleSampler {
        sampler: SamplerChoice,
        is_multi_objective: bool,
    },
}

impl std::fmt::Display for HpoConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDirections => {
                write!(
                    f,
                    "HPO config must define at least one optimization direction"
                )
            }
            Self::ZeroTrials => write!(f, "HPO config requires n_trials > 0"),
            Self::IncompatibleSampler {
                sampler,
                is_multi_objective,
            } => {
                if *is_multi_objective {
                    write!(
                        f,
                        "multi-objective study requires NsgaII or NsgaIII sampler, got {:?}",
                        sampler
                    )
                } else {
                    write!(
                        f,
                        "single-objective study cannot use multi-objective sampler {:?}",
                        sampler
                    )
                }
            }
        }
    }
}

impl std::error::Error for HpoConfigValidationError {}

// ---------------------------------------------------------------------------
// HpoConfig
// ---------------------------------------------------------------------------

/// Top-level configuration for a hyperparameter optimization study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpoConfig {
    /// Study name for persistence and identification.
    pub study_name: String,

    /// Optimization direction(s).
    ///
    /// Single-objective: one direction (e.g. `Minimize`).
    /// Multi-objective: two directions (e.g. `[Maximize, Minimize]` for accuracy
    /// vs. complexity).
    pub directions: Vec<Direction>,

    /// Sampler algorithm.
    pub sampler: SamplerChoice,

    /// Pruner algorithm.
    pub pruner: PrunerChoice,

    /// Number of trials to run.
    pub n_trials: usize,

    /// Number of parallel workers (requires the `async` feature on `optimizer`).
    pub n_jobs: usize,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Export format for optimization results.
    pub storage_format: StorageFormat,

    /// If `true`, load an existing study from storage when it exists.
    pub load_if_exists: bool,
}

impl Default for HpoConfig {
    fn default() -> Self {
        Self {
            study_name: "mlp_study".to_string(),
            directions: vec![Direction::Minimize],
            sampler: SamplerChoice::default(),
            pruner: PrunerChoice::default(),
            n_trials: 50,
            n_jobs: 1,
            seed: None,
            storage_format: StorageFormat::default(),
            load_if_exists: true,
        }
    }
}

impl HpoConfig {
    /// Create a new configuration with the given study name and default settings.
    pub fn new(study_name: impl Into<String>) -> Self {
        Self {
            study_name: study_name.into(),
            ..Default::default()
        }
    }

    /// Configure for single-objective optimization.
    pub fn single_objective(mut self, direction: Direction) -> Self {
        self.directions = vec![direction];
        self.sampler = SamplerChoice::Tpe;
        self
    }

    /// Configure for multi-objective optimization over three axes:
    ///
    /// 1. **OK@1%** — maximise (accuracy metric).
    /// 2. **`param_count`** — minimise (architectural complexity).
    /// 3. **`max_error`** — minimise (worst-case test-grid error). Was
    ///    previously a hard feasibility constraint; is now a first-
    ///    class objective so NSGA-III's dominance gradient pushes away
    ///    from high-max-error regions without disqualifying trials
    ///    outright.
    ///
    /// Uses NSGA-III by default — better than NSGA-II at 3+ objectives
    /// because it replaces crowding distance with reference-point
    /// niching (Das–Dennis), which keeps the Pareto front well-spread
    /// as dimensionality grows.
    pub fn multi_objective(mut self) -> Self {
        self.directions = vec![
            Direction::Maximize, // OK@1%
            Direction::Minimize, // param_count
            Direction::Minimize, // max_error
        ];
        self.sampler = SamplerChoice::NsgaIII;
        self
    }

    /// Set the number of optimization trials.
    pub fn with_n_trials(mut self, n: usize) -> Self {
        self.n_trials = n;
        self
    }

    /// Set the sampler algorithm.
    pub fn with_sampler(mut self, sampler: SamplerChoice) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set the pruner algorithm.
    pub fn with_pruner(mut self, pruner: PrunerChoice) -> Self {
        self.pruner = pruner;
        self
    }

    /// Set the number of parallel workers.
    pub fn with_n_jobs(mut self, n: usize) -> Self {
        self.n_jobs = n;
        self
    }

    /// Set the random seed for reproducible sampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }


    /// Set the export format for optimization results.
    pub fn with_storage_format(mut self, format: StorageFormat) -> Self {
        self.storage_format = format;
        self
    }

    /// Returns `true` when more than one optimization direction is configured.
    pub fn is_multi_objective(&self) -> bool {
        self.directions.len() > 1
    }

    /// Effective number of parallel workers.
    ///
    /// - `n_jobs == 0` → auto-detect via [`rayon::current_num_threads()`].
    /// - `n_jobs == 1` → sequential (no parallelism).
    /// - `n_jobs > 1`  → use exactly that many workers.
    pub fn effective_concurrency(&self) -> usize {
        match self.n_jobs {
            0 => rayon::current_num_threads(),
            n => n,
        }
    }

    /// Whether parallel trial evaluation is enabled (`n_jobs != 1`).
    pub fn is_parallel(&self) -> bool {
        self.effective_concurrency() > 1
    }

    /// Validate all configuration fields, returning an error on invalid combinations.
    pub fn validate(&self) -> std::result::Result<(), HpoConfigValidationError> {
        if self.directions.is_empty() {
            return Err(HpoConfigValidationError::EmptyDirections);
        }

        if self.n_trials == 0 {
            return Err(HpoConfigValidationError::ZeroTrials);
        }

        // Sampler compatibility is no longer enforced — the study builder
        // uses TPE internally for multi-objective (with post-hoc Pareto analysis)
        // because NSGA-II's sampler has internal state corruption bugs.
        Ok(())
    }
}

impl std::fmt::Display for HpoConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HpoConfig {{ study={}, trials={}, sampler={:?}, directions={:?}, n_jobs={} }}",
            self.study_name, self.n_trials, self.sampler, self.directions, self.n_jobs,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_covers_all_user_visible_defaults() {
        let cfg = HpoConfig::default();
        // Core identity / trial budget / sampler / pruner defaults.
        assert_eq!(cfg.study_name, "mlp_study");
        assert_eq!(cfg.n_trials, 50);
        assert_eq!(cfg.sampler, SamplerChoice::Tpe);
        assert_eq!(cfg.pruner, PrunerChoice::None);
        assert!(!cfg.is_multi_objective());
        // Parallelism defaults to sequential (n_jobs = 1).
        assert_eq!(cfg.n_jobs, 1);
        assert!(!cfg.is_parallel());
    }

    #[test]
    fn single_objective_config() {
        let cfg = HpoConfig::new("test")
            .single_objective(Direction::Minimize)
            .with_n_trials(100)
            .with_seed(42);
        assert_eq!(cfg.directions, vec![Direction::Minimize]);
        assert_eq!(cfg.sampler, SamplerChoice::Tpe);
        assert_eq!(cfg.n_trials, 100);
        assert_eq!(cfg.seed, Some(42));
        assert!(!cfg.is_multi_objective());
    }

    #[test]
    fn multi_objective_config() {
        let cfg = HpoConfig::new("pareto").multi_objective();
        // Three objectives: [OK@1% (maximise), param_count (minimise),
        // max_error (minimise)].
        assert_eq!(cfg.directions, vec![
            Direction::Maximize, Direction::Minimize, Direction::Minimize,
        ]);
        // NSGA-III by default — better Pareto-front spread in 3-obj
        // regimes than NSGA-II's crowding-distance heuristic.
        assert_eq!(cfg.sampler, SamplerChoice::NsgaIII);
        assert!(cfg.is_multi_objective());
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = HpoConfig::new("test_serde")
            .single_objective(Direction::Maximize)
            .with_n_trials(200)
            .with_seed(7);

        let json = serde_json::to_string(&cfg).unwrap();
        let restored: HpoConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.study_name, "test_serde");
        assert_eq!(restored.n_trials, 200);
        assert_eq!(restored.seed, Some(7));
    }

    #[test]
    fn sampler_choice_multi_objective() {
        assert!(!SamplerChoice::Tpe.is_multi_objective());
        assert!(!SamplerChoice::Random.is_multi_objective());
        assert!(SamplerChoice::NsgaII.is_multi_objective());
        assert!(SamplerChoice::NsgaIII.is_multi_objective());
    }

    #[test]
    fn display_config() {
        let cfg = HpoConfig::new("display_test").with_n_trials(10);
        let s = format!("{cfg}");
        assert!(s.contains("display_test"));
        assert!(s.contains("10"));
    }

    #[test]
    fn effective_concurrency_and_is_parallel_cover_all_n_jobs_modes() {
        // n_jobs = 1: sequential — effective = 1, is_parallel = false.
        let seq = HpoConfig::new("seq").with_n_jobs(1);
        assert_eq!(seq.effective_concurrency(), 1);
        assert!(!seq.is_parallel());

        // n_jobs = N > 1: explicit parallelism — effective = N,
        // is_parallel = true.
        let par = HpoConfig::new("par").with_n_jobs(4);
        assert_eq!(par.effective_concurrency(), 4);
        assert!(par.is_parallel());

        // n_jobs = 0: auto-detect — effective >= 1 (system-dependent).
        let auto = HpoConfig::new("auto").with_n_jobs(0);
        assert!(auto.effective_concurrency() >= 1);
    }

    #[test]
    fn validate_rejects_zero_trials() {
        let err = HpoConfig::new("invalid")
            .with_n_trials(0)
            .validate()
            .unwrap_err();
        assert_eq!(err, HpoConfigValidationError::ZeroTrials);
    }

    #[test]
    fn validate_accepts_any_sampler_regardless_of_objective_mode() {
        // Sampler compatibility is no longer enforced at config level —
        // both single- and multi-objective configs accept any sampler
        // (the builder internally picks a compatible default if needed).
        assert!(HpoConfig::new("so_mismatched")
            .single_objective(Direction::Minimize)
            .with_sampler(SamplerChoice::NsgaII)
            .validate()
            .is_ok());
        assert!(HpoConfig::new("mo_mismatched")
            .multi_objective()
            .with_sampler(SamplerChoice::Tpe)
            .validate()
            .is_ok());
    }
}
