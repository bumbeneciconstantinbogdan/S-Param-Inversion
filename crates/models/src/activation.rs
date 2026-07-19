//! Real-valued activation functions.
//!
//! Wraps Candle's built-in activations (ReLU, GELU, Tanh, etc.) behind an
//! [`Activation`] enum for serializable configuration and `Module` dispatch.

use std::fmt;
use std::str::FromStr;

use candle_core::{Result, Tensor};
use candle_nn::ops;
use serde::{Deserialize, Serialize};

use sparam_core::config::normalize_config_name;
use sparam_core::error::candle_msg;

const DEFAULT_LEAKY_RELU_ALPHA: f64 = 0.01;
const DEFAULT_ELU_ALPHA: f64 = 1.0;
const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
const SELU_LAMBDA: f64 = 1.050_700_987_355_480_5;

/// Supported real-valued activation functions for scalar neural network models.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[serde(rename = "relu")]
    ReLU,
    #[serde(rename = "leaky_relu")]
    LeakyReLU { alpha: f64 },
    #[serde(rename = "elu")]
    ELU { alpha: f64 },
    #[serde(rename = "selu")]
    SELU,
    #[serde(rename = "tanh")]
    Tanh,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "gelu")]
    GELU,
    #[serde(rename = "silu", alias = "swish")]
    SiLU,
}

impl Default for Activation {
    fn default() -> Self {
        Self::ReLU
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReLU => write!(f, "relu"),
            Self::LeakyReLU { alpha } => write!(f, "leaky_relu(α={alpha})"),
            Self::ELU { alpha } => write!(f, "elu(α={alpha})"),
            Self::SELU => write!(f, "selu"),
            Self::Tanh => write!(f, "tanh"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::GELU => write!(f, "gelu"),
            Self::SiLU => write!(f, "silu"),
        }
    }
}

impl Activation {
    /// Canonical short name (`"relu"`, `"leaky_relu"`, etc.) used as
    /// the round-trip key for string ↔ enum conversion in HPO search
    /// spaces and persisted configs. Unlike [`Display`](fmt::Display),
    /// this strips the `α=...` parenthetical from LeakyReLU / ELU.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::ReLU => "relu",
            Self::LeakyReLU { .. } => "leaky_relu",
            Self::ELU { .. } => "elu",
            Self::SELU => "selu",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::GELU => "gelu",
            Self::SiLU => "silu",
        }
    }

    /// Create one of the supported activations from a configuration string.
    pub fn from_name(name: &str) -> Result<Self> {
        let normalized = normalize_config_name(name);
        match normalized.as_str() {
            "relu" => Ok(Self::ReLU),
            "leaky_relu" => Ok(Self::LeakyReLU {
                alpha: DEFAULT_LEAKY_RELU_ALPHA,
            }),
            "elu" => Ok(Self::ELU {
                alpha: DEFAULT_ELU_ALPHA,
            }),
            "selu" => Ok(Self::SELU),
            "tanh" => Ok(Self::Tanh),
            "sigmoid" => Ok(Self::Sigmoid),
            "gelu" => Ok(Self::GELU),
            "silu" | "swish" => Ok(Self::SiLU),
            _ => Err(candle_msg(format!("unknown activation: {name}"))),
        }
    }

    /// Apply the activation element-wise to a tensor.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.apply(xs)
    }

    fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::ReLU => xs.relu(),
            Self::LeakyReLU { alpha } => ops::leaky_relu(xs, *alpha),
            Self::ELU { alpha } => xs.elu(*alpha),
            Self::SELU => xs.elu(SELU_ALPHA)?.affine(SELU_LAMBDA, 0.0),
            Self::Tanh => xs.tanh(),
            Self::Sigmoid => ops::sigmoid(xs),
            Self::GELU => xs.gelu_erf(),
            Self::SiLU => xs.silu(),
        }
    }
}

impl candle_nn::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.apply(xs)
    }
}

impl FromStr for Activation {
    type Err = candle_core::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_name(s)
    }
}

/// Serde adaptor that represents an [`Activation`] as its canonical
/// short name (`"relu"`, `"gelu"`, …) in the serialized form. Use via
/// `#[serde(with = "sparam_models::activation::as_name")]` on struct
/// fields where activations need to round-trip through HPO / DB JSON
/// as flat strings — the default derive emits tagged objects for
/// `LeakyReLU` / `ELU`, which is noisy for configs that only use the
/// default alpha.
pub mod as_name {
    use super::Activation;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(a: &Activation, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(a.name())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Activation, D::Error> {
        let s = String::deserialize(d)?;
        Activation::from_name(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_nn::Module;

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_tensor_close(values: &[f64], expected: &[f64]) {
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }

    fn sample_input() -> Result<Tensor> {
        Tensor::from_vec(vec![-1.0f64, 0.0, 1.0], 3, &Device::Cpu)
    }

    fn json_result<T>(result: std::result::Result<T, serde_json::Error>) -> Result<T> {
        result.map_err(|error| candle_msg(format!("serde_json error: {error}")))
    }

    #[test]
    fn test_relu_family_activations_match_expected_values() -> Result<()> {
        let input = sample_input()?;

        let relu = Activation::ReLU.forward(&input)?.to_vec1::<f64>()?;
        let leaky_relu = Activation::LeakyReLU { alpha: 0.01 }
            .forward(&input)?
            .to_vec1::<f64>()?;
        let elu = Activation::ELU { alpha: 1.0 }
            .forward(&input)?
            .to_vec1::<f64>()?;
        let selu = Activation::SELU.forward(&input)?.to_vec1::<f64>()?;

        assert_tensor_close(&relu, &[0.0, 0.0, 1.0]);
        assert_tensor_close(&leaky_relu, &[-0.01, 0.0, 1.0]);
        assert_tensor_close(&elu, &[-0.632_120_558_828_557_7, 0.0, 1.0]);
        assert_tensor_close(
            &selu,
            &[-1.111_330_737_812_562_5, 0.0, 1.050_700_987_355_480_5],
        );

        Ok(())
    }

    #[test]
    fn test_sigmoid_family_and_modern_activations_match_expected_values() -> Result<()> {
        let input = sample_input()?;

        let tanh = Activation::Tanh.forward(&input)?.to_vec1::<f64>()?;
        let sigmoid = Activation::Sigmoid.forward(&input)?.to_vec1::<f64>()?;
        let gelu = Activation::GELU.forward(&input)?.to_vec1::<f64>()?;
        let silu = Activation::SiLU.forward(&input)?.to_vec1::<f64>()?;

        assert_tensor_close(
            &tanh,
            &[-0.761_594_155_955_764_9, 0.0, 0.761_594_155_955_764_9],
        );
        assert_tensor_close(
            &sigmoid,
            &[0.268_941_421_369_995_1, 0.5, 0.731_058_578_630_004_9],
        );
        assert_tensor_close(
            &gelu,
            &[-0.158_655_253_931_457_07, 0.0, 0.841_344_746_068_542_9],
        );
        assert_tensor_close(
            &silu,
            &[-0.268_941_421_369_995_1, 0.0, 0.731_058_578_630_004_9],
        );

        Ok(())
    }

    #[test]
    fn test_make_activation_maps_supported_names_and_aliases() -> Result<()> {
        assert_eq!(Activation::from_name("relu")?, Activation::ReLU);
        assert_eq!(
            Activation::from_name("leaky_relu")?,
            Activation::LeakyReLU { alpha: 0.01 }
        );
        assert_eq!(Activation::from_name("elu")?, Activation::ELU { alpha: 1.0 });
        assert_eq!(Activation::from_name("selu")?, Activation::SELU);
        assert_eq!(Activation::from_name("tanh")?, Activation::Tanh);
        assert_eq!(Activation::from_name("sigmoid")?, Activation::Sigmoid);
        assert_eq!(Activation::from_name("gelu")?, Activation::GELU);
        assert_eq!(Activation::from_name("silu")?, Activation::SiLU);
        assert_eq!(Activation::from_name("swish")?, Activation::SiLU);
        assert_eq!(
            Activation::from_name("Leaky-ReLU")?,
            Activation::LeakyReLU { alpha: 0.01 }
        );

        let error = Activation::from_name("unknown")
            .expect_err("unknown activations should be rejected");
        assert!(error.to_string().contains("unknown activation"));

        Ok(())
    }

    #[test]
    fn test_activation_implements_candle_module_trait() -> Result<()> {
        let input = sample_input()?;
        let output = Module::forward(&Activation::ReLU, &input)?.to_vec1::<f64>()?;

        assert_tensor_close(&output, &[0.0, 0.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_activation_default_and_display_are_config_friendly() {
        assert_eq!(Activation::default(), Activation::ReLU);
        assert_eq!(Activation::ReLU.to_string(), "relu");
        assert_eq!(
            Activation::LeakyReLU { alpha: 0.01 }.to_string(),
            "leaky_relu(α=0.01)"
        );
        assert_eq!(Activation::ELU { alpha: 1.0 }.to_string(), "elu(α=1)");
        assert_eq!(Activation::SELU.to_string(), "selu");
        assert_eq!(Activation::GELU.to_string(), "gelu");
        assert_eq!(Activation::SiLU.to_string(), "silu");
    }

    /// Every Real activation variant must produce a non-empty short
    /// name whose `from_name` parses back to the same variant. Adding
    /// a new variant to the `Activation` enum forces the `match` below
    /// to be extended (via exhaustiveness), which in turn forces the
    /// `all_variants` array to list the new name — this prevents
    /// silent drift between `name()`, `from_name`, and the serde
    /// `as_name` adaptor.
    #[test]
    fn every_activation_variant_has_round_trip_name() {
        let all_variants = [
            Activation::ReLU,
            Activation::LeakyReLU { alpha: DEFAULT_LEAKY_RELU_ALPHA },
            Activation::ELU { alpha: DEFAULT_ELU_ALPHA },
            Activation::SELU,
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::GELU,
            Activation::SiLU,
        ];
        for variant in &all_variants {
            let name = variant.name();
            assert!(!name.is_empty(), "empty name for {variant:?}");
            let reparsed = Activation::from_name(name)
                .unwrap_or_else(|e| panic!("re-parse '{name}' failed: {e}"));
            assert_eq!(
                &reparsed, variant,
                "round-trip mismatch: {variant:?} → '{name}' → {reparsed:?}",
            );
        }
        // Exhaustiveness trap: adding a new variant breaks this match
        // at compile time, forcing the author to update `all_variants`
        // above.
        fn _exhaustive_check(a: Activation) {
            match a {
                Activation::ReLU
                | Activation::LeakyReLU { .. }
                | Activation::ELU { .. }
                | Activation::SELU
                | Activation::Tanh
                | Activation::Sigmoid
                | Activation::GELU
                | Activation::SiLU => {}
            }
        }
    }

    #[test]
    fn test_activation_serde_round_trips_with_canonical_names() -> Result<()> {
        assert_eq!(
            json_result(serde_json::to_string(&Activation::ReLU))?,
            "\"relu\""
        );
        assert_eq!(
            json_result(serde_json::to_string(&Activation::LeakyReLU {
                alpha: 0.01
            }))?,
            r#"{"leaky_relu":{"alpha":0.01}}"#
        );
        assert_eq!(
            json_result(serde_json::from_str::<Activation>("\"sigmoid\""))?,
            Activation::Sigmoid
        );
        assert_eq!(
            json_result(serde_json::from_str::<Activation>("\"swish\""))?,
            Activation::SiLU
        );
        assert_eq!(
            json_result(serde_json::from_str::<Activation>(
                r#"{"elu":{"alpha":0.5}}"#
            ))?,
            Activation::ELU { alpha: 0.5 }
        );

        Ok(())
    }
}
