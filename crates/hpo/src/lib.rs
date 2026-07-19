//! Hyperparameter optimization with NSGA-II and TPE.
//!
//! Provides a configurable search space, study construction, and constraint
//! helpers for optimizing MLP hyperparameters using the `optimizer` crate.

pub mod config;
pub mod evaluation;
pub mod mlp_evaluator;
pub mod pareto;
pub mod search_space;
pub(crate) mod serde_helpers;
pub mod storage;
pub mod study;
pub mod summary;
#[cfg(test)]
pub(crate) mod test_fixtures;

pub use config::HpoConfig;
pub use evaluation::{TrialMetrics, TrialOutcome, TrialRunner, TrialStatus};
pub use mlp_evaluator::{
    ComplexTensorDataSource, MlpModelBuilder, RegressionEvaluator, RelativeErrorLoss,
    TensorDataSource, UnifiedMlpBuilder,
};
pub use pareto::{CompletedTrial, MultiObjectiveResults};
pub use search_space::MlpSearchSpace;
pub use search_space::{
    ActivationChoice, BatchSizeChoice, GradClipChoice, HyperParams, LossChoice, NormChoice,
    OptimizerChoice, OptimizerHyperParams, OptimizerView, SchedulerChoice,
    SchedulerHyperParams, SchedulerView,
};
pub use study::{HpoStudyBuilder, ImportanceStudies, OBJECTIVE_NAMES};
pub use summary::{SummaryConfig, SummaryInput, SummaryMeta, print_summary};

// Re-export common optimizer types for convenience.
pub use optimizer::multi_objective::MultiObjectiveStudy;
pub use optimizer::prelude::{Direction, Study, Trial};
