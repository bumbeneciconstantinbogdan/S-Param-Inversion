//! Route tree assembly.

use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;

use crate::state::AppState;

pub mod assets;
pub mod dashboard;
pub mod evaluate;
pub mod generate;
pub mod hpo;
pub mod infer;
pub mod theory;
pub mod train;

/// Build the complete route tree.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        // Dashboard
        .route("/", get(dashboard::index))
        // Data generation
        .route("/generate", get(generate::form))
        .route("/generate", post(generate::run))
        // Dataset management
        .route("/datasets/{id}/preview", get(generate::preview))
        .route("/datasets/{id}", delete(dashboard::delete_dataset))
        // Training run management
        .route("/runs/{id}", delete(dashboard::delete_run))
        // Training
        .route("/train", get(train::form))
        .route("/train", post(train::launch))
        .route("/train/{run_id}/sse", get(train::sse_stream))
        .route("/train/{run_id}/result", get(train::result))
        .route("/train/{run_id}", get(train::detail))
        // Evaluation
        .route("/evaluate", get(evaluate::form))
        .route("/evaluate", post(evaluate::run))
        // Analytical NRW round-trip — no trained model required.
        .route("/evaluate/nrw", get(evaluate::nrw_form))
        .route("/evaluate/nrw", post(evaluate::nrw_run))
        // HPO
        .route("/hpo", get(hpo::dashboard))
        .route("/hpo/new", get(hpo::form))
        .route("/hpo", post(hpo::launch))
        .route("/hpo/{id}", get(hpo::detail))
        .route("/hpo/{id}/sse", get(hpo::sse_stream))
        .route("/hpo/{id}", delete(hpo::delete))
        // Inference
        .route("/infer", get(infer::form))
        .route("/infer", post(infer::predict))
        // Theory / architecture wiki
        .route("/theory", get(theory::page))
        // Static assets
        .route("/static/{*path}", get(assets::serve_static))
}
