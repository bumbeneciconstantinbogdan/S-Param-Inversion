//! Unified error types and context helpers.
//!
//! Every module in the workspace wraps domain-specific messages into
//! [`CoreError`]. This module centralizes that pattern so callers can write
//! `msg_error("...")` or `.context("...")?` instead of maintaining per-module
//! error helpers.

/// Core error type for the sparam workspace.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// A free-form error message.
    #[error("{0}")]
    Msg(String),

    /// An I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A Candle tensor error.
    #[cfg(feature = "tensor")]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

/// Convenience alias for results using [`CoreError`].
pub type Result<T> = std::result::Result<T, CoreError>;

/// Construct a [`CoreError::Msg`] from anything that converts to `String`.
#[inline]
pub fn msg_error(message: impl Into<String>) -> CoreError {
    CoreError::Msg(message.into())
}

/// Construct a `candle_core::Error::Msg` from anything that converts to `String`.
///
/// Single helper shared by every module that returns `candle_core::Result`,
/// replacing the per-module `fn *_error(...)` clones.
#[cfg(feature = "tensor")]
#[inline]
pub fn candle_msg(message: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(message.into())
}

/// Extension trait that attaches a static context string to any error type
/// that implements [`std::fmt::Display`], converting it into a [`CoreError::Msg`].
///
/// ```ignore
/// std::fs::write(path, data).context("failed to write report")?;
/// ```
pub trait ErrorContext<T> {
    /// Map the error through `"{context}: {original_error}"`.
    fn context(self, context: &str) -> Result<T>;
}

impl<T, E: std::fmt::Display> ErrorContext<T> for std::result::Result<T, E> {
    #[inline]
    fn context(self, context: &str) -> Result<T> {
        self.map_err(|e| msg_error(format!("{context}: {e}")))
    }
}

#[cfg(feature = "tensor")]
impl From<CoreError> for candle_core::Error {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::Candle(e) => e,
            other => candle_core::Error::Msg(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msg_error_stores_message() {
        let err = msg_error("something went wrong");
        assert!(err.to_string().contains("something went wrong"));
    }

    #[test]
    fn context_wraps_io_error() {
        let io_err: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file missing",
        ));
        let candle_err = io_err.context("save checkpoint").unwrap_err();
        let msg = candle_err.to_string();
        assert!(msg.contains("save checkpoint"), "got: {msg}");
        assert!(msg.contains("file missing"), "got: {msg}");
    }

    #[test]
    fn context_preserves_ok_values() {
        let ok: std::result::Result<i32, String> = Ok(42);
        assert_eq!(ok.context("irrelevant").unwrap(), 42);
    }
}
