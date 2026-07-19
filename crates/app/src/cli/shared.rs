use serde::de::DeserializeOwned;
use std::fmt;
use std::path::Path;

/// Output format for results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => f.write_str("text"),
            Self::Json => f.write_str("json"),
            Self::Markdown => f.write_str("markdown"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = candle_core::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_name(s)
    }
}

impl OutputFormat {
    pub fn from_name(name: &str) -> Result<Self, candle_core::Error> {
        match name.to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "markdown" | "md" => Ok(Self::Markdown),
            other => Err(candle_core::Error::Msg(format!(
                "unknown output format '{other}', expected: text, json, markdown"
            ))),
        }
    }
}

/// Parse a comma-separated pair of floats into `(f64, f64)`.
pub fn parse_range(s: &str) -> Result<(f64, f64), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("expected two comma-separated values, got '{s}'"));
    }
    let lo: f64 = parts[0]
        .trim()
        .parse()
        .map_err(|e| format!("bad lower bound: {e}"))?;
    let hi: f64 = parts[1]
        .trim()
        .parse()
        .map_err(|e| format!("bad upper bound: {e}"))?;
    if lo > hi {
        return Err(format!("lower bound {lo} exceeds upper bound {hi}"));
    }
    Ok((lo, hi))
}

/// Load an optional TOML config file as an overlay for CLI defaults.
///
/// If `config_path` is `Some`, reads the file and deserialises it into `T`.
/// If `None`, returns `T::default()`.  The `cmd_name` is used in error messages.
pub fn load_toml_overlay<T: DeserializeOwned + Default>(
    config_path: Option<&Path>,
    cmd_name: &str,
) -> Result<T, candle_core::Error> {
    match config_path {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "failed to read {cmd_name} config {}: {e}",
                    path.display()
                ))
            })?;
            toml::from_str(&text)
                .map_err(|e| candle_core::Error::Msg(format!("invalid {cmd_name} TOML: {e}")))
        }
        None => Ok(T::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_default_is_text() {
        assert_eq!(OutputFormat::default(), OutputFormat::Text);
    }

    /// Canonical name round-trips: every `(input, variant)` pair must
    /// parse to the expected variant AND `variant.to_string()` must
    /// equal the canonical name. Aliases (`"md"`, `"MARKDOWN"`,
    /// `"TEXT"`) only test the parse direction; only the canonical
    /// spelling is exercised for Display.
    #[test]
    fn output_format_valid_names_parse_and_display() {
        for (input, variant, display) in [
            ("text", OutputFormat::Text, Some("text")),
            ("TEXT", OutputFormat::Text, None),
            ("json", OutputFormat::Json, Some("json")),
            ("JSON", OutputFormat::Json, None),
            ("markdown", OutputFormat::Markdown, Some("markdown")),
            ("Markdown", OutputFormat::Markdown, None),
            ("md", OutputFormat::Markdown, None),
        ] {
            assert_eq!(OutputFormat::from_name(input).unwrap(), variant, "input: {input:?}");
            if let Some(expected) = display {
                assert_eq!(variant.to_string(), expected);
            }
        }
    }

    #[test]
    fn output_format_from_name_rejects_unknown() {
        let err = OutputFormat::from_name("csv").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown output format"), "got: {msg}");
    }

    #[test]
    fn parse_range_accepts_valid_inputs() {
        for (input, expected) in [
            ("1.0,200.0", (1.0, 200.0)),
            (" 0.5 , 3.5 ", (0.5, 3.5)),   // whitespace trimmed
            ("5.0,5.0", (5.0, 5.0)),         // equal bounds allowed
        ] {
            assert_eq!(parse_range(input).unwrap(), expected, "input: {input:?}");
        }
    }

    #[test]
    fn parse_range_rejects_invalid_inputs_with_expected_messages() {
        for (input, expected_substring) in [
            ("10.0,1.0", "exceeds"),         // reversed bounds
            ("42.0", "two comma-separated"), // single value
            ("1,2,3", "two comma-separated"), // three values
            ("abc,2.0", "bad lower bound"),  // non-numeric lower
        ] {
            let err = parse_range(input).unwrap_err();
            assert!(
                err.contains(expected_substring),
                "input {input:?}: expected err containing {expected_substring:?}, got: {err}",
            );
        }
    }

}
