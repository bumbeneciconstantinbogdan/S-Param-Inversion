//! Axum application builder and server startup.

use std::path::Path;
use std::sync::Arc;

use crate::db;
use crate::routes;
use crate::state::AppState;

/// Build and start the web server.
pub async fn serve(addr: &str, data_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Only the data dir itself — model weights live in the DB as a
    // `weights_blob` column; no on-disk `checkpoints/` directory is
    // created.
    tokio::fs::create_dir_all(data_dir).await?;

    // Open database.
    let db_path = data_dir.join("sparam.db");
    let conn = db::open_database(&db_path)?;

    let state = Arc::new(AppState::new(conn, data_dir.to_path_buf()));

    let app = routes::router().with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let display_addr = addr.replace("0.0.0.0", "localhost").replace("127.0.0.1", "localhost");
    eprintln!("S-Parameter Inversion UI listening on http://{display_addr}");
    eprintln!("Data directory: {}", data_dir.display());

    axum::serve(listener, app).await?;
    Ok(())
}
