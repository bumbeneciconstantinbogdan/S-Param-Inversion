//! JSON exporter/loader for [`MultiObjectiveResults`].

use std::path::{Path, PathBuf};

use super::types::{StorageError, StorageResult};
use crate::pareto::MultiObjectiveResults;
use sparam_core::io::write_bytes_atomic;

// ---------------------------------------------------------------------------
// Serialisation envelope
// ---------------------------------------------------------------------------

/// Top-level JSON document wrapping study metadata and trial records.
#[derive(serde::Serialize, serde::Deserialize)]
struct Envelope {
    /// Schema version for forward compatibility.
    version: u32,
    /// Study metadata.
    metadata: Metadata,
    /// The full results payload.
    results: MultiObjectiveResults,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Metadata {
    /// ISO-8601 timestamp when the export was created (best-effort; empty if
    /// the system clock is unavailable).
    created_at: String,
}

fn now_iso8601() -> String {
    // std::time::SystemTime → seconds since epoch → basic ISO-like stamp.
    // No chrono dependency — keep it lightweight.
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("epoch:{}", d.as_secs()),
        Err(_) => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Exporter
// ---------------------------------------------------------------------------

pub struct JsonExporter {
    path: PathBuf,
}

impl JsonExporter {
    pub fn new(base_path: PathBuf) -> Self {
        let path = base_path.with_extension("json");
        Self { path }
    }

    pub fn export(&self, results: &MultiObjectiveResults) -> StorageResult<PathBuf> {
        let envelope = Envelope {
            version: 1,
            metadata: Metadata {
                created_at: now_iso8601(),
            },
            results: results.clone(),
        };

        let json = serde_json::to_vec_pretty(&envelope)?;
        write_bytes_atomic(&self.path, &json)?;

        Ok(self.path.clone())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

pub struct JsonLoader {
    path: PathBuf,
}

impl JsonLoader {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn load(&self) -> StorageResult<MultiObjectiveResults> {
        let file = std::fs::File::open(&self.path).map_err(StorageError::Io)?;
        let envelope: Envelope = serde_json::from_reader(file)?;
        Ok(envelope.results)
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto::CompletedTrial;
    use crate::evaluation::{TrialMetrics, TrialStatus};
    use crate::test_fixtures::sample_real_params as sample_params;
    use optimizer::prelude::Direction;

    fn sample_results() -> MultiObjectiveResults {
        MultiObjectiveResults {
            trials: vec![
                CompletedTrial {
                    trial_number: 0,
                    params: sample_params(32),
                    objectives: [95.0, 32.0, 2.0],
                    num_objectives: 3,
                    metrics: TrialMetrics {
                        best_val_loss: 0.001,
                        param_count: 320,
                        stopped_early: false,
                        best_epoch: 10,
                        final_epoch: 20,
                        training_time_secs: 1.5,
                        ok_at_1pct: 95.0,
                        max_error: 5.0,
                    },
                    constraint_value: -5.0,
                    status: TrialStatus::Completed,
                },
                CompletedTrial {
                    trial_number: 1,
                    params: sample_params(64),
                    objectives: [98.0, 64.0, 1.5],
                    num_objectives: 3,
                    metrics: TrialMetrics {
                        best_val_loss: 0.0005,
                        param_count: 640,
                        stopped_early: true,
                        best_epoch: 8,
                        final_epoch: 15,
                        training_time_secs: 2.3,
                        ok_at_1pct: 98.0,
                        max_error: 3.0,
                    },
                    constraint_value: -7.0,
                    status: TrialStatus::Completed,
                },
            ],
            directions: [Direction::Maximize, Direction::Minimize, Direction::Minimize],
            constraint_threshold: Some(10.0),
        }
    }

    #[test]
    fn json_roundtrip() {
        let dir = std::env::temp_dir().join("hpo_test_json_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let base = dir.join("study");
        let exporter = JsonExporter::new(base.clone());
        let original = sample_results();

        let written = exporter.export(&original).unwrap();
        assert!(written.exists());
        assert_eq!(written.extension().unwrap(), "json");

        let loader = JsonLoader::new(written);
        assert!(loader.exists());
        let loaded = loader.load().unwrap();

        assert_eq!(loaded.trials.len(), original.trials.len());
        assert_eq!(loaded.directions, original.directions);
        assert_eq!(loaded.constraint_threshold, original.constraint_threshold);

        for (orig, rest) in original.trials.iter().zip(loaded.trials.iter()) {
            assert_eq!(orig.trial_number, rest.trial_number);
            assert_eq!(orig.objectives, rest.objectives);
            assert_eq!(orig.status, rest.status);
            assert_eq!(orig.params.hidden_size, rest.params.hidden_size);
            assert_eq!(orig.params.lr.to_bits(), rest.params.lr.to_bits());
            assert_eq!(orig.metrics.param_count, rest.metrics.param_count);
            assert_eq!(
                orig.metrics.best_val_loss.to_bits(),
                rest.metrics.best_val_loss.to_bits()
            );
            assert_eq!(orig.metrics.stopped_early, rest.metrics.stopped_early);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_roundtrip_empty_trials() {
        let dir = std::env::temp_dir().join("hpo_test_json_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let base = dir.join("empty_study");
        let exporter = JsonExporter::new(base);
        let original = MultiObjectiveResults {
            trials: vec![],
            directions: [Direction::Maximize, Direction::Minimize, Direction::Minimize],
            constraint_threshold: None,
        };

        let path = exporter.export(&original).unwrap();
        let loaded = JsonLoader::new(path).load().unwrap();
        assert!(loaded.trials.is_empty());
        assert_eq!(loaded.constraint_threshold, None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_creates_parent_directories() {
        let dir = std::env::temp_dir().join("hpo_test_json_parents/nested/deep");
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("hpo_test_json_parents"));

        let base = dir.join("study");
        let exporter = JsonExporter::new(base);
        let results = sample_results();

        let path = exporter.export(&results).unwrap();
        assert!(path.exists());

        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("hpo_test_json_parents"));
    }

    #[test]
    fn json_loader_nonexistent_returns_error() {
        let loader = JsonLoader::new(PathBuf::from("/tmp/nonexistent_hpo_study_xyz.json"));
        assert!(!loader.exists());
        assert!(loader.load().is_err());
    }

    #[test]
    fn json_roundtrip_preserves_non_finite_values() {
        let dir = std::env::temp_dir().join("hpo_test_json_nan");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut results = sample_results();
        results.trials[0].status = TrialStatus::Failed;
        results.trials[0].objectives = [f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY];
        results.trials[0].metrics = TrialMetrics::default();

        let base = dir.join("nan_study");
        let exporter = JsonExporter::new(base);
        let path = exporter.export(&results).unwrap();
        let loaded = JsonLoader::new(path).load().unwrap();

        assert_eq!(loaded.trials[0].status, TrialStatus::Failed);
        assert!(loaded.trials[0].objectives[0].is_infinite());
        assert!(loaded.trials[0].objectives[0].is_sign_negative());
        assert!(loaded.trials[0].objectives[1].is_infinite());
        assert!(loaded.trials[0].objectives[1].is_sign_positive());
        assert!(loaded.trials[0].metrics.best_val_loss.is_infinite());
        assert!(loaded.trials[0].metrics.best_val_loss.is_sign_positive());
        assert!(loaded.trials[0].metrics.ok_at_1pct.is_nan());
        assert!(loaded.trials[0].metrics.max_error.is_infinite());
        assert!(loaded.trials[0].metrics.max_error.is_sign_positive());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
