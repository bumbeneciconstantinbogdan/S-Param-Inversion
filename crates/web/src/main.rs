//! S-Parameter Inversion UI — web server entry point.

use std::path::PathBuf;

fn main() {
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".into());
    let data_dir = std::env::var("DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("sparam_data"));
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let addr = format!("{host}:{port}");

    // 16 MiB blocking-thread stack absorbs the recursive drop at the end
    // of a 20k-trial HPO study (optimizer + candle graphs); default 2 MiB
    // overflows.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024)
        .build()
        .expect("failed to build tokio runtime");

    runtime.block_on(async move {
        if let Err(e) = sparam_web::server::serve(&addr, &data_dir).await {
            eprintln!("Server error: {e}");
            std::process::exit(1);
        }
    });
}
