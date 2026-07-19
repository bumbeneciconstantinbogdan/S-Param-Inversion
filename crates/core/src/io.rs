//! File I/O utilities: atomic writes, directory creation, JSON helpers.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{ErrorContext, Result};

static ATOMIC_WRITE_ID: AtomicU64 = AtomicU64::new(0);

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn temp_path_for(path: &Path) -> PathBuf {
    let id = ATOMIC_WRITE_ID.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("atomic-write");
    path.with_file_name(format!(".{file_name}.tmp-{}-{id}", std::process::id()))
}

/// Atomically write bytes to `path` using temp-file + rename.
pub fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let tmp_path = temp_path_for(path);
    std::fs::write(&tmp_path, bytes)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Atomically write UTF-8 text to `path` using temp-file + rename.
pub fn write_text_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    write_bytes_atomic(path, text.as_bytes())
}

/// Ensure a directory exists, creating it and any parents if necessary.
pub fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)
            .context(&format!("failed to create directory {}", path.display()))?;
    }
    Ok(())
}

/// Atomically write a `serde::Serialize` value as pretty-printed JSON.
///
/// Creates parent directories automatically.
pub fn write_json<T: serde::Serialize>(value: &T, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    let json = serde_json::to_vec_pretty(value).context("JSON serialization failed")?;
    write_bytes_atomic(path, &json)
        .context(&format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sparam_core_{name}_{}_{}",
            std::process::id(),
            ATOMIC_WRITE_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn write_bytes_atomic_creates_parent_dirs_and_file() {
        let dir = temp_dir("write_bytes_atomic");
        let path = dir.join("nested/out.bin");
        write_bytes_atomic(&path, b"hello").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn write_text_atomic_overwrites_existing_file() {
        let dir = temp_dir("write_text_atomic");
        let path = dir.join("report.txt");
        write_text_atomic(&path, "first").unwrap();
        write_text_atomic(&path, "second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
        let _ = std::fs::remove_dir_all(dir);
    }
}
