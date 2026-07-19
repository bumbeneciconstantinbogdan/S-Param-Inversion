//! Optimization results storage: export and load.
//!
//! Uses enum-based dispatch (no `dyn` / dynamic dispatch). JSON is the
//! only supported format; `StorageFormat::Sqlite` is kept for
//! config-schema compatibility but returns an unsupported error.

mod json;
mod types;

use std::path::{Path, PathBuf};

use crate::config::StorageFormat;
use crate::pareto::MultiObjectiveResults;
pub use types::{StorageError, StorageResult};

// ---------------------------------------------------------------------------
// Enum-based exporter (no dyn dispatch)
// ---------------------------------------------------------------------------

/// Enum-dispatched results exporter.
///
/// Wraps the concrete format implementations, selected at runtime from
/// [`StorageFormat`].
pub enum ResultsExporter {
    Json(json::JsonExporter),
}

impl ResultsExporter {
    /// Create an exporter for the given format writing to `base_path`.
    pub fn new(format: StorageFormat, base_path: impl Into<PathBuf>) -> StorageResult<Self> {
        let base = base_path.into();
        match format {
            StorageFormat::Json => Ok(Self::Json(json::JsonExporter::new(base))),
            StorageFormat::Sqlite => Err(StorageError::UnsupportedFormat(format!("{format:?}"))),
        }
    }

    /// Write `results` to the configured path, creating parent dirs as needed.
    pub fn export(&self, results: &MultiObjectiveResults) -> StorageResult<PathBuf> {
        match self {
            Self::Json(e) => e.export(results),
        }
    }

    /// The path that will be (or was) written.
    pub fn path(&self) -> &Path {
        match self {
            Self::Json(e) => e.path(),
        }
    }
}

// ---------------------------------------------------------------------------
// Enum-based loader
// ---------------------------------------------------------------------------

/// Enum-dispatched results loader.
pub enum ResultsLoader {
    Json(json::JsonLoader),
}

impl ResultsLoader {
    /// Create a loader by detecting format from the file extension.
    pub fn from_path(path: impl Into<PathBuf>) -> StorageResult<Self> {
        let path = path.into();
        Ok(Self::Json(json::JsonLoader::new(path)))
    }

    /// Load results from the file.
    pub fn load(&self) -> StorageResult<MultiObjectiveResults> {
        match self {
            Self::Json(l) => l.load(),
        }
    }

    /// Whether the backing file exists.
    pub fn exists(&self) -> bool {
        match self {
            Self::Json(l) => l.exists(),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Export `results` using the format and base path from `config`.
///
/// Returns the path to the written file.
pub fn export_results(
    results: &MultiObjectiveResults,
    config: &crate::config::HpoConfig,
) -> StorageResult<PathBuf> {
    let base = PathBuf::from(&config.study_name);
    let exporter = ResultsExporter::new(config.storage_format, base)?;
    exporter.export(results)
}

/// Load previously exported results from `path` (format auto-detected from
/// file extension).
pub fn load_results(path: impl Into<PathBuf>) -> StorageResult<MultiObjectiveResults> {
    let loader = ResultsLoader::from_path(path)?;
    loader.load()
}
