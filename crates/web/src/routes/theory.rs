//! Static wiki page explaining the Real MLP and Complex MLP (CVNN)
//! architectures used by this project, with SVG diagrams.
//!
//! Purely presentational — no DB access, no form handling.

use askama::Template;
use axum::response::Html;

use crate::error::{WebResult, render};

#[derive(Template)]
#[template(path = "theory.html")]
pub struct TheoryTemplate;

pub async fn page() -> WebResult<Html<String>> {
    render(TheoryTemplate)
}
