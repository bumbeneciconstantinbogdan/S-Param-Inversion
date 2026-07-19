//! Web-layer error type with `IntoResponse` implementation.

use askama::Template;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};

/// Unified error type for web handlers.
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Computation error: {0}")]
    Computation(String),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Template error: {0}")]
    Template(#[from] askama::Error),
}

impl From<candle_core::Error> for WebError {
    fn from(err: candle_core::Error) -> Self {
        Self::Computation(err.to_string())
    }
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Computation(_) | Self::Database(_) | Self::Json(_) | Self::Template(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };

        let body = format!(
            r#"<div style="background:#fef2f2;border:1px solid #fecaca;color:#991b1b;border-radius:0.5rem;padding:1rem;margin:1rem 0;">
                <p style="font-weight:600">Error</p>
                <p style="font-size:0.875rem;margin-top:0.25rem">{}</p>
               </div>"#,
            self
        );

        (status, Html(body)).into_response()
    }
}

pub type WebResult<T> = Result<T, WebError>;

/// Render an Askama template into an HTML response.
pub fn render<T: Template>(tmpl: T) -> WebResult<Html<String>> {
    let html = tmpl.render()?;
    Ok(Html(html))
}
