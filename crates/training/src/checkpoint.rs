//! Binary model checkpoint save/load helpers backed by Candle safetensors.
//!
//! Candle already provides efficient binary serialization for model weights via
//! [`VarMap::save`] and [`VarMap::load`], which use the safetensors format.
//! This module wraps that support in a small public API that:
//! - creates parent directories on save,
//! - returns checkpoint metadata,
//! - emits clearer errors for missing files and empty variable maps, and
//! - works uniformly for real and complex-valued models because both register
//!   real-valued parameters in the same [`VarMap`].

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::Result;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use sparam_core::error::candle_msg;

/// Supported binary checkpoint formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CheckpointFormat {
    /// Hugging Face safetensors, backed by Candle's built-in `VarMap` support.
    #[default]
    #[serde(rename = "safetensors")]
    Safetensors,
}

impl CheckpointFormat {
    /// The canonical file extension for the format.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
        }
    }
}

impl fmt::Display for CheckpointFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safetensors => f.write_str("safetensors"),
        }
    }
}

/// Metadata returned after saving or loading a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointInfo {
    pub path: PathBuf,
    pub format: CheckpointFormat,
    pub tensor_count: usize,
    pub size_bytes: u64,
}

impl fmt::Display for CheckpointInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} ({} tensors, {} bytes)",
            self.format,
            self.path.display(),
            self.tensor_count,
            self.size_bytes
        )
    }
}

/// Reusable checkpoint handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCheckpoint {
    path: PathBuf,
    format: CheckpointFormat,
}

impl ModelCheckpoint {
    /// Create a checkpoint handle using the default binary format.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: CheckpointFormat::default(),
        }
    }

    /// Create a checkpoint handle with an explicit format.
    #[must_use]
    pub fn with_format(path: impl Into<PathBuf>, format: CheckpointFormat) -> Self {
        Self {
            path: path.into(),
            format,
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn format(&self) -> CheckpointFormat {
        self.format
    }

    /// Save all parameters currently registered in `varmap`.
    pub fn save(&self, varmap: &VarMap) -> Result<CheckpointInfo> {
        let tensor_count = ensure_non_empty(varmap, "saving a model checkpoint")?;
        ensure_parent_dir(&self.path)?;

        match self.format {
            CheckpointFormat::Safetensors => varmap.save(&self.path),
        }
        .map_err(|err| {
            candle_msg(format!(
                "failed to save {} checkpoint {}: {err}",
                self.format,
                self.path.display()
            ))
        })?;

        checkpoint_info(&self.path, self.format, tensor_count)
    }

    /// Load checkpoint values into the existing variables registered in `varmap`.
    pub fn load(&self, varmap: &mut VarMap) -> Result<CheckpointInfo> {
        let tensor_count = ensure_non_empty(varmap, "loading a model checkpoint")?;
        ensure_file_exists(&self.path)?;

        match self.format {
            CheckpointFormat::Safetensors => varmap.load(&self.path),
        }
        .map_err(|err| {
            candle_msg(format!(
                "failed to load {} checkpoint {}: {err}",
                self.format,
                self.path.display()
            ))
        })?;

        checkpoint_info(&self.path, self.format, tensor_count)
    }
}

/// Save a model checkpoint using Candle's binary safetensors support.
pub fn save_model_checkpoint<P: AsRef<Path>>(varmap: &VarMap, path: P) -> Result<CheckpointInfo> {
    ModelCheckpoint::new(path.as_ref().to_path_buf()).save(varmap)
}

/// Load a model checkpoint using Candle's binary safetensors support.
pub fn load_model_checkpoint<P: AsRef<Path>>(
    varmap: &mut VarMap,
    path: P,
) -> Result<CheckpointInfo> {
    ModelCheckpoint::new(path.as_ref().to_path_buf()).load(varmap)
}

/// Serialize `varmap` to an owned safetensors byte buffer — no
/// filesystem involvement. Used to persist model weights as a SQLite
/// BLOB instead of a `.safetensors` file on disk.
///
/// Output is byte-identical to what [`save_model_checkpoint`] would
/// write, so a checkpoint saved via this function can be loaded with
/// [`load_model_checkpoint`] after writing the bytes to a file (and
/// vice versa via [`load_model_checkpoint_bytes`]).
pub fn save_model_checkpoint_bytes(varmap: &VarMap) -> Result<Vec<u8>> {
    ensure_non_empty(varmap, "saving a model checkpoint")?;
    let guard = varmap
        .data()
        .lock()
        .map_err(|e| candle_msg(format!("VarMap mutex poisoned: {e}")))?;
    let entries = guard.iter().map(|(name, var)| (name, var.as_tensor()));
    safetensors::tensor::serialize(entries, None).map_err(|err| {
        candle_msg(format!(
            "failed to serialize VarMap to safetensors bytes: {err}"
        ))
    })
}

/// Inspect a safetensors byte buffer and list every tensor name it
/// contains, without actually deserialising the tensor data.
///
/// Used by the inference path to detect architecture variants that
/// the persisted `ModelConfig` doesn't (yet) describe — e.g. an old
/// Complex-MLP checkpoint that has `model.norm.*` weights from when
/// LayerNorm was always-on but the `ModelConfig::Complex.norm` field
/// hadn't been added to the JSON yet. Without this peek, the loader
/// would silently skip those tensors and run the trained input layer
/// straight into the activation, corrupting predictions.
pub fn peek_checkpoint_tensor_names(bytes: &[u8]) -> Result<Vec<String>> {
    if bytes.is_empty() {
        return Err(candle_msg(
            "cannot peek tensor names from empty byte buffer",
        ));
    }
    let view = safetensors::SafeTensors::deserialize(bytes).map_err(|err| {
        candle_msg(format!("failed to parse safetensors header: {err}"))
    })?;
    Ok(view.names().into_iter().map(|s| s.to_string()).collect())
}

/// Restore variables already registered on `varmap` from an in-memory
/// safetensors byte buffer. Mirror of [`load_model_checkpoint`] for
/// callers that store weights as BLOBs rather than files.
///
/// Uses Candle's [`candle_core::safetensors::load_buffer`] under the
/// hood, which validates each tensor's shape + dtype before assignment.
pub fn load_model_checkpoint_bytes(varmap: &mut VarMap, bytes: &[u8]) -> Result<()> {
    ensure_non_empty(varmap, "loading a model checkpoint")?;
    if bytes.is_empty() {
        return Err(candle_msg(
            "cannot load model checkpoint from empty byte buffer",
        ));
    }

    // All variables in this project share the same device; pick the
    // first one's device as the deserialization target.
    let device = {
        let guard = varmap
            .data()
            .lock()
            .map_err(|e| candle_msg(format!("VarMap mutex poisoned: {e}")))?;
        guard
            .values()
            .next()
            .map(|v| v.as_tensor().device().clone())
            .ok_or_else(|| candle_msg("VarMap has no variables"))?
    };

    let tensors = candle_core::safetensors::load_buffer(bytes, &device).map_err(|err| {
        candle_msg(format!("failed to deserialize safetensors bytes: {err}"))
    })?;
    for (name, tensor) in &tensors {
        varmap.set_one(name, tensor).map_err(|err| {
            candle_msg(format!("failed to restore variable `{name}`: {err}"))
        })?;
    }
    Ok(())
}

fn ensure_non_empty(varmap: &VarMap, context: &str) -> Result<usize> {
    let tensor_count = varmap.all_vars().len();
    if tensor_count == 0 {
        return Err(candle_msg(format!(
            "{context} requires a VarMap with at least one registered variable"
        )));
    }
    Ok(tensor_count)
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            candle_msg(format!(
                "failed to create checkpoint directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

fn ensure_file_exists(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Err(candle_msg(format!(
            "checkpoint file {} does not exist",
            path.display()
        )));
    }
    Ok(())
}

fn checkpoint_info(
    path: &Path,
    format: CheckpointFormat,
    tensor_count: usize,
) -> Result<CheckpointInfo> {
    let size_bytes = fs::metadata(path)
        .map_err(|err| {
            candle_msg(format!(
                "failed to read checkpoint metadata {}: {err}",
                path.display()
            ))
        })?
        .len();

    Ok(CheckpointInfo {
        path: path.to_path_buf(),
        format,
        tensor_count,
        size_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use sparam_models::{
        Activation, ComplexActivation, ComplexMLPConfig, ComplexMLPRegressor, MLPConfig,
        MLPRegressor,
    };
    use sparam_core::complex_tensor::ComplexTensor;

    static TEMP_CHECKPOINT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn make_real_model() -> (VarMap, MLPRegressor) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let model = MLPRegressor::new(vb, &MLPConfig::new(4, 8, 2, Activation::GELU)).unwrap();
        (varmap, model)
    }

    fn make_complex_model() -> (VarMap, ComplexMLPRegressor) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let model = ComplexMLPRegressor::new(
            vb,
            &ComplexMLPConfig::permittivity(4, ComplexActivation::ModReLU),
        )
        .unwrap();
        (varmap, model)
    }

    fn temp_checkpoint_root(label: &str) -> PathBuf {
        let id = TEMP_CHECKPOINT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "cvnn_checkpoint_test_{}_{}_{}",
            label,
            std::process::id(),
            id
        ))
    }

    fn perturb_varmap(varmap: &VarMap, offset: f64) -> Result<()> {
        for var in varmap.all_vars() {
            let updated = var.as_tensor().affine(1.0, offset)?;
            var.set(&updated)?;
        }
        Ok(())
    }

    fn max_abs_diff(actual: &[Vec<f64>], expected: &[Vec<f64>]) -> f64 {
        actual
            .iter()
            .zip(expected.iter())
            .flat_map(|(actual_row, expected_row)| actual_row.iter().zip(expected_row.iter()))
            .map(|(actual_value, expected_value)| (actual_value - expected_value).abs())
            .fold(0.0, f64::max)
    }

    fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                assert!(
                    (actual_value - expected_value).abs() <= tolerance,
                    "expected {expected_value}, got {actual_value}"
                );
            }
        }
    }

    #[test]
    fn checkpoint_format_is_config_friendly() {
        assert_eq!(CheckpointFormat::default(), CheckpointFormat::Safetensors);
        assert_eq!(CheckpointFormat::Safetensors.extension(), "safetensors");
        assert_eq!(CheckpointFormat::Safetensors.to_string(), "safetensors");

        let json = serde_json::to_string(&CheckpointFormat::Safetensors).unwrap();
        let restored: CheckpointFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, CheckpointFormat::Safetensors);
    }

    #[test]
    fn save_creates_parent_dirs_and_reports_metadata() {
        let (varmap, _model) = make_real_model();
        let root = temp_checkpoint_root("metadata");
        let path = root.join("nested/model.safetensors");

        let info = save_model_checkpoint(&varmap, &path).unwrap();

        assert!(path.exists());
        assert_eq!(info.path, path);
        assert_eq!(info.format, CheckpointFormat::Safetensors);
        assert_eq!(info.tensor_count, varmap.all_vars().len());
        assert!(info.size_bytes > 0);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn save_and_load_restore_real_model_outputs() {
        let (mut varmap, model) = make_real_model();
        let xs = Tensor::from_vec(
            vec![0.1f64, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
            (2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let baseline = model.forward(&xs).unwrap().to_vec2::<f64>().unwrap();

        let root = temp_checkpoint_root("real");
        let path = root.join("real-model.safetensors");
        let checkpoint = ModelCheckpoint::new(path.clone());
        checkpoint.save(&varmap).unwrap();

        perturb_varmap(&varmap, 0.25).unwrap();
        let mutated = model.forward(&xs).unwrap().to_vec2::<f64>().unwrap();
        assert!(max_abs_diff(&mutated, &baseline) > 1e-6);

        checkpoint.load(&mut varmap).unwrap();
        let restored = model.forward(&xs).unwrap().to_vec2::<f64>().unwrap();
        assert_matrix_close(&restored, &baseline, 1e-12);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn save_and_load_restore_complex_model_outputs() {
        let (mut varmap, model) = make_complex_model();
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![0.1f64, 0.2, -0.3, 0.4], (2, 2), &Device::Cpu).unwrap(),
            Tensor::from_vec(vec![0.05f64, -0.1, 0.15, -0.2], (2, 2), &Device::Cpu).unwrap(),
        )
        .unwrap();
        let baseline = model.forward(&xs).unwrap();
        let baseline_real = baseline.real.to_vec2::<f64>().unwrap();
        let baseline_imag = baseline.imag.to_vec2::<f64>().unwrap();

        let root = temp_checkpoint_root("complex");
        let path = root.join("complex-model.safetensors");
        let checkpoint = ModelCheckpoint::new(path.clone());
        checkpoint.save(&varmap).unwrap();

        perturb_varmap(&varmap, 0.1).unwrap();
        let mutated = model.forward(&xs).unwrap();
        assert!(max_abs_diff(&mutated.real.to_vec2::<f64>().unwrap(), &baseline_real) > 1e-6);

        checkpoint.load(&mut varmap).unwrap();
        let restored = model.forward(&xs).unwrap();
        assert_matrix_close(
            &restored.real.to_vec2::<f64>().unwrap(),
            &baseline_real,
            1e-12,
        );
        assert_matrix_close(
            &restored.imag.to_vec2::<f64>().unwrap(),
            &baseline_imag,
            1e-12,
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn save_and_load_bytes_round_trip_without_filesystem() {
        // Bytes API must produce the same restored-output as the file API
        // — weight blobs served from SQLite need to be interchangeable
        // with the historical `.safetensors` files on disk.
        let (mut varmap, model) = make_real_model();
        let xs = Tensor::from_vec(
            vec![0.1f64, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
            (2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let baseline = model.forward(&xs).unwrap().to_vec2::<f64>().unwrap();

        let bytes = save_model_checkpoint_bytes(&varmap).unwrap();
        assert!(!bytes.is_empty());

        perturb_varmap(&varmap, 0.3).unwrap();
        let mutated = model.forward(&xs).unwrap().to_vec2::<f64>().unwrap();
        assert!(max_abs_diff(&mutated, &baseline) > 1e-6);

        load_model_checkpoint_bytes(&mut varmap, &bytes).unwrap();
        let restored = model.forward(&xs).unwrap().to_vec2::<f64>().unwrap();
        assert_matrix_close(&restored, &baseline, 1e-12);
    }

    #[test]
    fn load_bytes_rejects_empty_buffer() {
        let (mut varmap, _model) = make_real_model();
        let err = load_model_checkpoint_bytes(&mut varmap, &[]).unwrap_err();
        assert!(err.to_string().contains("empty byte buffer"));
    }

    #[test]
    fn loading_missing_checkpoint_errors() {
        let (mut varmap, _model) = make_real_model();
        let root = temp_checkpoint_root("missing");
        let path = root.join("missing.safetensors");

        let error = load_model_checkpoint(&mut varmap, &path).unwrap_err();
        assert!(error.to_string().contains("does not exist"));
    }
}
