//! Dashboard: unified view of datasets and training runs + delete handlers.

use std::sync::Arc;

use askama::Template;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};

use crate::db::{self, DatasetRow, TrainingRunRow};
use crate::error::{WebResult, render};
use crate::state::AppState;

#[derive(Template)]
#[template(path = "dashboard.html")]
pub struct DashboardTemplate {
    pub datasets: Vec<DatasetRow>,
    pub runs: Vec<TrainingRunRow>,
}

pub async fn index(State(state): State<Arc<AppState>>) -> WebResult<Html<String>> {
    let (datasets, runs) = {
        let conn = state.db.lock().await;
        (db::list_datasets(&conn)?, db::list_training_runs(&conn)?)
    };
    render(DashboardTemplate { datasets, runs })
}

pub async fn delete_dataset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> WebResult<Response> {
    let conn = state.db.lock().await;
    let data_dir = db::delete_dataset(&conn, id)?;
    drop(conn);
    let _ = tokio::fs::remove_dir_all(&data_dir).await;
    Ok((StatusCode::OK, [("HX-Redirect", "/")], "").into_response())
}

pub async fn delete_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> WebResult<Response> {
    let conn = state.db.lock().await;
    let output_dir = db::delete_training_run(&conn, id)?;
    drop(conn);
    let _ = tokio::fs::remove_dir_all(&output_dir).await;
    Ok((StatusCode::OK, [("HX-Redirect", "/")], "").into_response())
}
