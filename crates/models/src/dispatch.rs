//! Unified model dispatch: wraps real and complex MLP variants behind a single
//! [`MlpModel`] enum so that training, evaluation, and HPO code can treat them
//! uniformly.

use std::str::FromStr;

use candle_core::{Result, Tensor};
use candle_nn::ModuleT;

use crate::complex_mlp::ComplexMLPRegressor;
use crate::mlp::MLPRegressor;

/// Model family selected for HPO evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Real,
    Complex,
}

impl ModelType {
    /// `(input_size, output_size)` of the packed-real tensors this
    /// model family consumes / produces.
    ///
    /// - Real: 4 real features (S₁₁.re, S₁₁.im, S₂₁.re, S₂₁.im) → 2
    ///   real outputs (ε′, ε″).
    /// - Complex: 2 complex features (S₁₁, S₂₁) packed into 4 reals →
    ///   1 complex output (ε) packed into 2 reals.
    #[inline]
    #[must_use]
    pub fn io_shape(self) -> (usize, usize) {
        match self {
            Self::Real => (4, 2),
            Self::Complex => (2, 1),
        }
    }

    /// Lowercase string label used in serialized configs and route
    /// form payloads (`"real"` / `"complex"`).
    #[inline]
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Complex => "complex",
        }
    }

    #[inline]
    #[must_use]
    pub fn is_complex(self) -> bool {
        matches!(self, Self::Complex)
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "real" => Ok(Self::Real),
            "complex" | "cvnn" => Ok(Self::Complex),
            other => Err(format!(
                "unknown model type '{other}' (expected 'real' or 'complex')"
            )),
        }
    }
}

/// Unified wrapper over the supported real and complex MLP variants.
#[derive(Clone, Debug)]
pub enum MlpModel {
    Real(MLPRegressor),
    Complex(ComplexMLPRegressor),
}

impl MlpModel {
    #[inline]
    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Real(_) => ModelType::Real,
            Self::Complex(_) => ModelType::Complex,
        }
    }

    #[inline]
    pub fn parameter_count(&self) -> usize {
        match self {
            Self::Real(model) => model.parameter_count(),
            Self::Complex(model) => model.parameter_count(),
        }
    }
}

impl ModuleT for MlpModel {
    #[inline]
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        match self {
            Self::Real(model) => model.forward_t(xs, train),
            Self::Complex(model) => model.forward_packed(xs, train),
        }
    }
}
