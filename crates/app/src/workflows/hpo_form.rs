//! Web/CLI-agnostic HPO form schema.
//!
//! Mirrors [`super::train_form::TrainForm`] for the HPO launch path:
//! a single shared form struct + `HpoForm → HpoWorkflowConfig` mapping
//! so adding a field to the workflow config only requires updating
//! one place.

use serde::Deserialize;

use super::hpo::HpoWorkflowConfig;
use super::train_form::TrainSplits;

/// HPO submission form — same fields the web `/hpo` POST handler
/// accepts today, promoted out of the route so any future CLI / FFI
/// consumer can reuse the schema.
#[derive(Deserialize)]
pub struct HpoForm {
    pub dataset_id: i64,
    #[serde(default = "default_study_name")]
    pub study_name: String,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_n_trials")]
    pub n_trials: usize,
    #[serde(default = "default_max_epochs")]
    pub max_epochs: usize,
    #[serde(default)]
    pub n_jobs: usize,
    #[serde(default = "default_patience")]
    pub patience: usize,
    #[serde(default = "default_warmup_epochs")]
    pub warmup_epochs: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub search_space_json: String,
}

fn default_study_name() -> String { "hpo_study".into() }
fn default_model_type() -> String { "real".into() }
fn default_n_trials() -> usize { 50 }
fn default_max_epochs() -> usize { 25 }
// patience=10 tolerates the ~15-20% val_loss volatility from RNG-stream
// differences; patience=5 stopped trials prematurely on ~40% of configs.
fn default_patience() -> usize { 10 }
fn default_warmup_epochs() -> usize { 10 }
fn default_seed() -> u64 { 42 }

/// Samples loaded alongside the HPO form — DB rows for web, CSVs for
/// CLI. Reuses [`TrainSplits`] since the shape is identical.
pub type HpoSplits = TrainSplits;

/// Output of [`HpoForm::build`] — carries the workflow-ready config
/// plus the same payload as JSON so the caller can persist it
/// verbatim (e.g., `hpo_studies.config_json`).
pub struct BuiltHpoSubmission {
    /// `"real"`, `"complex"`, or `"both"` — passed through verbatim
    /// to the workflow so it dispatches to one or two studies.
    pub model_type: String,
    pub study_name: String,
    pub dataset_id: i64,
    pub n_trials: usize,
    /// `n_trials * 2` when `model_type == "both"`, else `n_trials`.
    pub total_trials: usize,
    pub config: HpoWorkflowConfig,
    pub config_json: String,
}

impl HpoForm {
    /// Validate the form and build the [`HpoWorkflowConfig`] + matching
    /// JSON for DB persistence. Single source of truth for the
    /// HpoForm → workflow mapping.
    pub fn build(
        self,
        splits: HpoSplits,
    ) -> Result<BuiltHpoSubmission, serde_json::Error> {
        let search_space_json = if self.search_space_json.is_empty() {
            None
        } else {
            Some(self.search_space_json)
        };

        let config_json = serde_json::to_string(&serde_json::json!({
            "model_type": self.model_type,
            "n_trials": self.n_trials,
            "max_epochs": self.max_epochs,
            "n_jobs": self.n_jobs,
            "patience": self.patience,
            "warmup_epochs": self.warmup_epochs,
            "seed": self.seed,
        }))?;

        let n_model_types = if self.model_type == "both" { 2 } else { 1 };
        let total_trials = self.n_trials * n_model_types;

        // `HpoWorkflowConfig.train_samples` takes ownership; move the
        // splits in without extra clones.
        let config = HpoWorkflowConfig {
            train_samples: splits.train,
            val_samples: splits.val,
            test_samples: splits.test,
            model_type: self.model_type.clone(),
            n_trials: self.n_trials,
            n_jobs: self.n_jobs,
            max_epochs: self.max_epochs,
            patience: self.patience,
            warmup_epochs: self.warmup_epochs,
            study_name: self.study_name.clone(),
            seed: self.seed,
            search_space_json,
        };

        Ok(BuiltHpoSubmission {
            model_type: self.model_type,
            study_name: self.study_name,
            dataset_id: self.dataset_id,
            n_trials: self.n_trials,
            total_trials,
            config,
            config_json,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparam_data::generation::PermittivitySample;

    #[test]
    fn form_deserialises_from_urlencoded_body() {
        let body = [
            ("dataset_id", "1"),
            ("study_name", "my_study"),
            ("model_type", "both"),
            ("n_trials", "100"),
            ("max_epochs", "50"),
            ("n_jobs", "4"),
            ("patience", "15"),
            ("warmup_epochs", "5"),
            ("seed", "123"),
            ("search_space_json", ""),
        ];
        let encoded = body
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let form: HpoForm = serde_urlencoded::from_str(&encoded).unwrap();
        assert_eq!(form.dataset_id, 1);
        assert_eq!(form.study_name, "my_study");
        assert_eq!(form.model_type, "both");
        assert_eq!(form.n_trials, 100);
    }

    #[test]
    fn build_doubles_trial_count_for_both_model_type() {
        let form = HpoForm {
            dataset_id: 1,
            study_name: "t".into(),
            model_type: "both".into(),
            n_trials: 20,
            max_epochs: 10,
            n_jobs: 1,
            patience: 5,
            warmup_epochs: 0,
            seed: 1,
            search_space_json: String::new(),
        };
        let splits = HpoSplits {
            train: Vec::<PermittivitySample>::new().into(),
            val: Vec::new().into(),
            test: Vec::new().into(),
        };
        let built = form.build(splits).unwrap();
        assert_eq!(built.total_trials, 40);
        assert_eq!(built.n_trials, 20);
    }

    #[test]
    fn build_keeps_single_trial_count_for_single_model_type() {
        let form = HpoForm {
            dataset_id: 1,
            study_name: "t".into(),
            model_type: "real".into(),
            n_trials: 30,
            max_epochs: 10,
            n_jobs: 1,
            patience: 5,
            warmup_epochs: 0,
            seed: 1,
            search_space_json: String::new(),
        };
        let splits = HpoSplits {
            train: Vec::<PermittivitySample>::new().into(),
            val: Vec::new().into(),
            test: Vec::new().into(),
        };
        let built = form.build(splits).unwrap();
        assert_eq!(built.total_trials, 30);
    }
}
