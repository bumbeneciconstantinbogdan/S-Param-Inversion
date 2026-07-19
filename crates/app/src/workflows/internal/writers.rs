//! Typed workflow artifact writers.

use std::path::Path;

use sparam_core::error::{ErrorContext, Result};
use sparam_core::io::{ensure_dir, write_json, write_text_atomic};

pub(crate) fn write_json_artifact<T: serde::Serialize>(
    value: &T,
    path: &Path,
    description: &str,
) -> Result<()> {
    write_json(value, path)
        .context(&format!("failed to write {description} to {}", path.display()))
}

pub(crate) fn write_text_artifact(path: &Path, text: &str, description: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    write_text_atomic(path, text)
        .context(&format!("failed to write {description} to {}", path.display()))
}

