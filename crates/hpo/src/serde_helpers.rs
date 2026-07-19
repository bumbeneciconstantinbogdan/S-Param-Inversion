//! Serde helpers for persisted optimization results.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum SpecialFloat {
    NaN,
    Infinity,
    NegInfinity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
enum EncodedF64 {
    Finite(f64),
    Special(SpecialFloat),
}

impl EncodedF64 {
    fn into_f64(self) -> f64 {
        match self {
            Self::Finite(value) => value,
            Self::Special(SpecialFloat::NaN) => f64::NAN,
            Self::Special(SpecialFloat::Infinity) => f64::INFINITY,
            Self::Special(SpecialFloat::NegInfinity) => f64::NEG_INFINITY,
        }
    }
}

impl From<f64> for EncodedF64 {
    fn from(value: f64) -> Self {
        if value.is_nan() {
            Self::Special(SpecialFloat::NaN)
        } else if value.is_infinite() && value.is_sign_positive() {
            Self::Special(SpecialFloat::Infinity)
        } else if value.is_infinite() {
            Self::Special(SpecialFloat::NegInfinity)
        } else {
            Self::Finite(value)
        }
    }
}

pub mod special_f64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::EncodedF64;

    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        EncodedF64::from(*value).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(EncodedF64::deserialize(deserializer)?.into_f64())
    }
}

// Only referenced by this file's own tests (the 3-objective variant
// below is the one used in production `CompletedTrial` serde). Gated
// behind `#[cfg(test)]` to keep cargo warning-free.
#[cfg(test)]
pub mod special_f64_array2 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::EncodedF64;

    pub fn serialize<S>(values: &[f64; 2], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        [EncodedF64::from(values[0]), EncodedF64::from(values[1])].serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f64; 2], D::Error>
    where
        D: Deserializer<'de>,
    {
        let [left, right] = <[EncodedF64; 2]>::deserialize(deserializer)?;
        Ok([left.into_f64(), right.into_f64()])
    }
}


/// Same pattern as [`special_f64_array2`] but for the three-objective
/// layout `[OK@1%, param_count, max_error]` fed into NSGA-III.
pub mod special_f64_array3 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::EncodedF64;

    pub fn serialize<S>(values: &[f64; 3], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        [
            EncodedF64::from(values[0]),
            EncodedF64::from(values[1]),
            EncodedF64::from(values[2]),
        ]
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f64; 3], D::Error>
    where
        D: Deserializer<'de>,
    {
        let [a, b, c] = <[EncodedF64; 3]>::deserialize(deserializer)?;
        Ok([a.into_f64(), b.into_f64(), c.into_f64()])
    }
}


#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    // -----------------------------------------------------------------------
    // EncodedF64 conversion
    // -----------------------------------------------------------------------

    use super::EncodedF64;

    #[test]
    fn encoded_f64_finite_round_trips() {
        for v in [0.0, 1.0, -1.0, 3.14, f64::MIN, f64::MAX, f64::EPSILON] {
            let encoded = EncodedF64::from(v);
            assert_eq!(encoded.into_f64(), v);
        }
    }

    #[test]
    fn encoded_f64_nan_round_trips() {
        let encoded = EncodedF64::from(f64::NAN);
        assert!(encoded.into_f64().is_nan());
    }

    #[test]
    fn encoded_f64_infinity_round_trips() {
        let pos = EncodedF64::from(f64::INFINITY);
        assert_eq!(pos.into_f64(), f64::INFINITY);

        let neg = EncodedF64::from(f64::NEG_INFINITY);
        assert_eq!(neg.into_f64(), f64::NEG_INFINITY);
    }

    // -----------------------------------------------------------------------
    // special_f64 serde module
    // -----------------------------------------------------------------------

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Wrapper {
        #[serde(with = "super::special_f64")]
        value: f64,
    }

    #[test]
    fn special_f64_serde_finite() {
        let w = Wrapper { value: 42.5 };
        let json = serde_json::to_string(&w).unwrap();
        let recovered: Wrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered, w);
    }

    #[test]
    fn special_f64_serde_nan() {
        let w = Wrapper { value: f64::NAN };
        let json = serde_json::to_string(&w).unwrap();
        let recovered: Wrapper = serde_json::from_str(&json).unwrap();
        assert!(recovered.value.is_nan());
    }

    #[test]
    fn special_f64_serde_infinity() {
        for v in [f64::INFINITY, f64::NEG_INFINITY] {
            let w = Wrapper { value: v };
            let json = serde_json::to_string(&w).unwrap();
            let recovered: Wrapper = serde_json::from_str(&json).unwrap();
            assert_eq!(recovered.value, v);
        }
    }

    // -----------------------------------------------------------------------
    // special_f64_array2 serde module
    // -----------------------------------------------------------------------

    #[derive(Debug, Serialize, Deserialize)]
    struct ArrayWrapper {
        #[serde(with = "super::special_f64_array2")]
        values: [f64; 2],
    }

    #[test]
    fn special_f64_array2_serde_finite() {
        let w = ArrayWrapper { values: [1.0, 2.0] };
        let json = serde_json::to_string(&w).unwrap();
        let recovered: ArrayWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.values, [1.0, 2.0]);
    }

    #[test]
    fn special_f64_array2_serde_mixed_special_values() {
        let w = ArrayWrapper {
            values: [f64::NAN, f64::INFINITY],
        };
        let json = serde_json::to_string(&w).unwrap();
        let recovered: ArrayWrapper = serde_json::from_str(&json).unwrap();
        assert!(recovered.values[0].is_nan());
        assert_eq!(recovered.values[1], f64::INFINITY);
    }

    #[test]
    fn special_f64_array2_serde_both_neg_infinity() {
        let w = ArrayWrapper {
            values: [f64::NEG_INFINITY, f64::NEG_INFINITY],
        };
        let json = serde_json::to_string(&w).unwrap();
        let recovered: ArrayWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.values, [f64::NEG_INFINITY, f64::NEG_INFINITY]);
    }
}
