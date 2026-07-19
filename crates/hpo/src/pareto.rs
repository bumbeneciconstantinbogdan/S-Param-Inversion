//! Multi-objective Pareto front extraction and dominance utilities.
//!
//! After a multi-objective HPO study completes, this module identifies the
//! non-dominated (Pareto-optimal) trials and provides helpers for ranking
//! and selecting the best-compromise solution.

use std::cmp::Ordering;

use optimizer::prelude::Direction;

use crate::evaluation::{TrialMetrics, TrialStatus};
use crate::search_space::HyperParams;

/// Completed trial record retained by the project after an HPO study.
///
/// **Objectives layout** `[OK@1%, param_count, max_error]` matching
/// the NSGA-III study's 3-direction vector `[Maximize, Minimize,
/// Minimize]`. `num_objectives` remains so legacy 2-objective data can
/// still round-trip (slots 2+ are zeroed); new runs always use all 3.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompletedTrial {
    pub trial_number: usize,
    pub params: HyperParams,
    #[serde(with = "crate::serde_helpers::special_f64_array3")]
    pub objectives: [f64; 3],
    pub num_objectives: usize,
    pub metrics: TrialMetrics,
    /// Retained for backwards compatibility with older JSON snapshots;
    /// the 3-objective NSGA-III study no longer uses constraint-based
    /// feasibility (max_error is optimised directly as a third
    /// objective). New trials always store `0.0` here.
    #[serde(default)]
    pub constraint_value: f64,
    pub status: TrialStatus,
}

impl CompletedTrial {
    /// Returns `true` when the trial completed without error.
    #[inline]
    pub fn is_successful(&self) -> bool {
        self.status == TrialStatus::Completed
    }

    /// Every successfully completed trial is "feasible" under the
    /// 3-objective formulation — there is no longer a separate hard
    /// feasibility filter (the old `max_err_constraint` that this
    /// method used to track is gone; `max_error` is now NSGA-III's
    /// third objective). Kept as an alias so downstream code that
    /// counts feasible trials doesn't need to change.
    #[inline]
    pub fn is_feasible(&self) -> bool {
        self.is_successful()
    }

    /// Slice of objective values used in this trial (length = `num_objectives`).
    #[inline]
    pub fn objective_values(&self) -> &[f64] {
        &self.objectives[..self.num_objectives]
    }
}

/// Project-owned results of a completed multi-objective HPO study.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiObjectiveResults {
    pub trials: Vec<CompletedTrial>,
    pub directions: [Direction; 3],
    /// Legacy field — the 3-objective study no longer filters trials
    /// by a max-error threshold. Retained so older persisted
    /// `MultiObjectiveResults` JSON can still deserialise; always
    /// `None` for new runs.
    #[serde(default)]
    pub constraint_threshold: Option<f64>,
}

impl MultiObjectiveResults {
    /// Extract the Pareto-optimal (non-dominated) subset of all trials.
    #[must_use]
    pub fn pareto_front(&self) -> Vec<&CompletedTrial> {
        extract_pareto_front(&self.trials, &self.directions)
    }

    /// Trials whose constraint values are satisfied.
    #[must_use]
    pub fn feasible_trials(&self) -> Vec<&CompletedTrial> {
        self.trials
            .iter()
            .filter(|trial| trial.is_feasible())
            .collect()
    }

    /// Trials whose constraint values are violated.
    #[must_use]
    pub fn infeasible_trials(&self) -> Vec<&CompletedTrial> {
        self.trials
            .iter()
            .filter(|trial| !trial.is_feasible())
            .collect()
    }

    /// Select the best-compromise trial from the Pareto front using lexicographic ordering.
    /// Order of tie-breakers: objective 0 (accuracy), then objective 1
    /// (param_count — lower first since minimise), then objective 2
    /// (max_error — lower first), then trial number.
    #[must_use]
    pub fn best_compromise(&self) -> Option<&CompletedTrial> {
        let front = self.pareto_front();
        front.into_iter().max_by(|left, right| {
            compare_objective(left.objectives[0], right.objectives[0], self.directions[0])
                .then_with(|| {
                    compare_objective(right.objectives[1], left.objectives[1], self.directions[1])
                })
                .then_with(|| {
                    compare_objective(right.objectives[2], left.objectives[2], self.directions[2])
                })
                .then_with(|| right.trial_number.cmp(&left.trial_number))
        })
    }

    /// Pareto front sorted by the first objective, then by each
    /// subsequent objective as a tie-breaker.
    #[must_use]
    pub fn sorted_pareto_front(&self) -> Vec<&CompletedTrial> {
        let mut front = self.pareto_front();
        front.sort_by(|left, right| {
            compare_objective(left.objectives[0], right.objectives[0], self.directions[0])
                .reverse()
                .then_with(|| {
                    compare_objective(left.objectives[1], right.objectives[1], self.directions[1])
                        .reverse()
                })
                .then_with(|| {
                    compare_objective(left.objectives[2], right.objectives[2], self.directions[2])
                        .reverse()
                })
                .then_with(|| left.trial_number.cmp(&right.trial_number))
        });
        front
    }
}

#[inline]
fn compare_objective(left: f64, right: f64, direction: Direction) -> Ordering {
    match direction {
        Direction::Maximize => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
        Direction::Minimize => right.partial_cmp(&left).unwrap_or(Ordering::Equal),
    }
}

/// Returns `true` if `a` dominates `b` under the provided directions.
#[must_use]
pub fn dominates(a: &[f64], b: &[f64], directions: &[Direction]) -> bool {
    if a.len() != b.len() || a.len() != directions.len() {
        return false;
    }

    let mut strictly_better = false;
    for ((left, right), direction) in a.iter().zip(b.iter()).zip(directions.iter()) {
        let comparison = match direction {
            Direction::Maximize => left.partial_cmp(right),
            Direction::Minimize => right.partial_cmp(left),
        };

        match comparison {
            Some(Ordering::Less) => return false,
            Some(Ordering::Greater) => strictly_better = true,
            Some(Ordering::Equal) => {}
            None => return false,
        }
    }

    strictly_better
}

/// Fixed-size fast path for the three-objective NSGA-III case.
#[must_use]
pub fn dominates_triple(a: &[f64; 3], b: &[f64; 3], directions: &[Direction; 3]) -> bool {
    dominates(&a[..], &b[..], &directions[..])
}

/// Extract the non-dominated front using the optimizer crate's native
/// `pareto_front_indices` algorithm (fast non-dominated sort) across
/// all three objectives.
#[must_use]
pub fn extract_pareto_front<'a>(
    trials: &'a [CompletedTrial],
    directions: &[Direction; 3],
) -> Vec<&'a CompletedTrial> {
    let successful: Vec<_> = trials.iter().filter(|trial| trial.is_successful()).collect();
    if successful.is_empty() {
        return Vec::new();
    }

    // Convert to the format expected by optimizer::pareto::pareto_front_indices.
    // Uses only the `num_objectives` active slots so legacy 2-objective
    // trials round-trip correctly; new runs always use all 3.
    let solutions: Vec<Vec<f64>> = successful
        .iter()
        .map(|t| t.objective_values().to_vec())
        .collect();
    let dirs: Vec<Direction> = directions.to_vec();

    let indices = optimizer::pareto::pareto_front_indices(&solutions, &dirs);
    indices.into_iter()
        .filter_map(|i| successful.get(i).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::sample_real_params as sample_params;

    /// 3-objective directions used by every test below: maximise
    /// OK@1%, minimise param_count, minimise max_error.
    const DIRS: [Direction; 3] = [
        Direction::Maximize,
        Direction::Minimize,
        Direction::Minimize,
    ];

    fn completed_trial(
        trial_number: usize,
        ok_at_1pct: f64,
        param_count: i64,
        max_error: f64,
        status: TrialStatus,
    ) -> CompletedTrial {
        CompletedTrial {
            trial_number,
            params: sample_params(param_count),
            objectives: [ok_at_1pct, param_count as f64, max_error],
            num_objectives: 3,
            metrics: TrialMetrics {
                ok_at_1pct,
                max_error,
                ..TrialMetrics::default()
            },
            constraint_value: 0.0,
            status,
        }
    }

    #[test]
    fn dominates_correctly_identifies_dominated_trials() {
        // Triple dominance: A dominates B iff A ≥ B on all axes and
        // strictly better on at least one.
        assert!(dominates_triple(&[98.0, 16.0, 1.0], &[97.0, 32.0, 5.0], &DIRS));
        // Equal accuracy, smaller params + smaller max_error wins.
        assert!(dominates_triple(&[98.0, 16.0, 1.0], &[98.0, 32.0, 5.0], &DIRS));
        // Better accuracy but worse on both other axes — not dominated.
        assert!(!dominates_triple(&[99.0, 64.0, 10.0], &[98.0, 32.0, 5.0], &DIRS));
    }

    #[test]
    fn pareto_front_excludes_dominated_trials() {
        let trials = vec![
            completed_trial(0, 98.0, 16, 1.0, TrialStatus::Completed),
            // Strictly dominated by trial 0: worse accuracy AND larger params AND worse max_err.
            completed_trial(1, 97.0, 32, 5.0, TrialStatus::Completed),
            // Non-dominated: best accuracy (higher OK@1% trade-off).
            completed_trial(2, 99.0, 64, 3.0, TrialStatus::Completed),
        ];

        let front = extract_pareto_front(&trials, &DIRS);
        assert_eq!(front.len(), 2);
        assert!(front.iter().any(|trial| trial.trial_number == 0));
        assert!(front.iter().any(|trial| trial.trial_number == 2));
    }

    #[test]
    fn pareto_front_handles_equal_objectives() {
        let trials = vec![
            completed_trial(0, 98.0, 16, 1.0, TrialStatus::Completed),
            completed_trial(1, 98.0, 16, 1.0, TrialStatus::Completed),
        ];

        let front = extract_pareto_front(&trials, &DIRS);
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn pareto_front_excludes_failed_trials() {
        // Non-successful trials are dropped from the front regardless
        // of their objective values — the old "infeasible by
        // constraint" path is gone.
        let trials = vec![
            completed_trial(0, 99.0, 64, 12.0, TrialStatus::Failed),
            completed_trial(1, 97.0, 16, 3.0, TrialStatus::Completed),
        ];

        let front = extract_pareto_front(&trials, &DIRS);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].trial_number, 1);
    }

    #[test]
    fn best_compromise_selects_max_ok_at_1pct() {
        let results = MultiObjectiveResults {
            trials: vec![
                completed_trial(0, 97.0, 16, 2.0, TrialStatus::Completed),
                // Ties on accuracy with trial 2 but has smaller params
                // → wins on the param_count tie-breaker.
                completed_trial(1, 98.5, 32, 5.0, TrialStatus::Completed),
                completed_trial(2, 98.5, 64, 5.0, TrialStatus::Completed),
            ],
            directions: DIRS,
            constraint_threshold: None,
        };

        let best = results
            .best_compromise()
            .expect("expected compromise trial");
        assert_eq!(best.trial_number, 1);
    }
}
