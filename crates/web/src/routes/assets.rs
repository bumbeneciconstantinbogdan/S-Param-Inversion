//! Embedded static asset serving.

use axum::body::Body;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "static/"]
#[prefix = ""]
struct StaticAssets;

/// Serve vendored static files (HTMX, Alpine, Chart.js, CSS, app.js).
pub async fn serve_static(Path(path): Path<String>) -> impl IntoResponse {
    match StaticAssets::get(&path) {
        Some(file) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream();
            Response::builder()
                .header("Content-Type", mime.as_ref())
                .header("Cache-Control", "public, max-age=31536000, immutable")
                .body(Body::from(file.data.to_vec()))
                .unwrap()
        }
        None => (StatusCode::NOT_FOUND, "Static file not found").into_response(),
    }
}

// NOTE: the old `serve_artifact` handler for `/artifacts/*path` was
// removed — every artifact (charts, metrics, weights) now lives in
// SQLite, and the handler built `data_dir().join(url_path)` without
// sanitising `..` segments. Keeping the route live would let any
// page loaded in the same browser read arbitrary files via e.g.
// `fetch('/artifacts/../../etc/passwd')`. Dead route, now deleted.
