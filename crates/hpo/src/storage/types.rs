/// Errors that can occur during results I/O.
#[derive(Debug)]
pub enum StorageError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// The requested format requires a feature that was not compiled in.
    UnsupportedFormat(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "storage I/O error: {e}"),
            Self::Json(e) => write!(f, "storage JSON error: {e}"),
            Self::UnsupportedFormat(fmt) => {
                write!(
                    f,
                    "storage format {fmt:?} requires a feature not compiled in"
                )
            }
        }
    }
}

impl std::error::Error for StorageError {}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub type StorageResult<T> = std::result::Result<T, StorageError>;
