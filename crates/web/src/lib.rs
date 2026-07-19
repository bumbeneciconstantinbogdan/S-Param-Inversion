//! S-Parameter Inversion UI — web UI.
//!
//! Wraps the `sparam-app` workflow API in an Axum + HTMX server with
//! live training progress via SSE.

pub mod db;
pub mod error;
pub mod json_config;
pub mod routes;
pub mod server;
pub mod sse;
pub mod state;
pub mod study_common;
pub mod tasks;
pub mod templates;
