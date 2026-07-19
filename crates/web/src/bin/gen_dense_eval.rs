//! Generate a dataset with **standard 200×100 + 8×8 train** and a
//! **dense 2000×1000 + 80×80 test** split.  The whole point is to
//! stress-test a model trained on the standard grid against an
//! evaluation grid that's 100× denser in the bulk and 100× denser in
//! the near-vacuum patch — i.e. ten samples between each two training
//! samples.  Train / val are kept at the standard sizes (we don't
//! care about retraining for this comparison; the model under test
//! was already trained on the matching dataset).
//!
//! Why this can't go through the normal `/generate` form: the form's
//! data generator derives the test grid from the train grid via
//! `derive_holdout_grid_dims` (test ≈ 12.5% of train).  Hitting a
//! 2000×1000 = 2M test would require a 16M train, which is wasteful
//! and slow.  This binary calls `generate_non_magnetic_data` twice
//! (once for the train+val with normal sizes, once with the dense
//! sizes to harvest its train output as our test split) so each split
//! gets exactly the dims we want.
//!
//! Usage (from repo root after `cargo build --release -p sparam-web --bin gen_dense_eval`):
//!     target/release/gen_dense_eval [--db PATH]
//!
//! Output: prints the new dataset_id.  Open `/evaluate` and pick the
//! new dataset under "Evaluate on dataset" to run a model against the
//! dense grid.

use std::path::PathBuf;

use sparam_data::generation::{DataGenerationConfig, generate_non_magnetic_data};
use sparam_web::db;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let db_path = parse_arg(&args, "--db")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("DATA_DIR")
                .map(|d| PathBuf::from(d).join("sparam.db"))
                .unwrap_or_else(|_| PathBuf::from("sparam_data/sparam.db"))
        });

    eprintln!("Opening DB at {}", db_path.display());
    let mut conn = db::open_database(&db_path)?;

    // Step 1: standard train+val with the production 200×100 bulk +
    // 8×8 patch overlay.  Reuses `DataGenerationConfig::default()`'s
    // ε-ranges (eps' ∈ [1, 50], eps'' ∈ [0, 25]) and the standard
    // patch window so the resulting train matches Dataset #1 exactly,
    // ensuring scaler statistics will be identical at evaluate time.
    let standard_cfg = DataGenerationConfig {
        n_eps_prime_train: 200,
        n_eps_double_prime_train: 100,
        dense_patch_n_train: (8, 8),
        // Eval patch unchanged (5×5) — we only customise the *bulk*
        // and *patch* test grids in step 2 below.
        ..DataGenerationConfig::default()
    };
    eprintln!("Generating standard train + val (200×100 bulk + 8×8 patch)...");
    let standard = generate_non_magnetic_data(&standard_cfg)?;
    eprintln!("  train={}, val={}", standard.train.len(), standard.validation.len());

    // Step 2: dense eval grid.  We ask `generate_non_magnetic_data`
    // for a generation whose *train* size is 2000×1000 + 80×80 patch,
    // then *use that train output as our test split*.  The samples
    // arrive at offset (0, 0) — the same zero-offset we'd get if we
    // simply oversampled the bulk grid; for the purposes of stress-
    // testing model generalisation this is equivalent to a denser
    // test_offset grid (both cover the same physical ε domain, just
    // displaced by ε' ≈ 0.05 / ε'' ≈ 0.025).
    let dense_cfg = DataGenerationConfig {
        n_eps_prime_train: 2000,
        n_eps_double_prime_train: 1000,
        dense_patch_n_train: (80, 80),
        ..DataGenerationConfig::default()
    };
    eprintln!("Generating dense eval grid (2000×1000 bulk + 80×80 patch)...");
    eprintln!("  this will allocate ~{} samples and may take 30-60s",
              2000 * 1000 + 80 * 80);
    let dense = generate_non_magnetic_data(&dense_cfg)?;
    let test_samples = dense.train; // see comment above
    eprintln!("  test={}", test_samples.len());

    // Step 3: persist as a new dataset row + samples.  The
    // `data_dir` field is legacy (samples live in dataset_samples
    // since the move-to-DB migration), so we leave it empty.
    let config_json = serde_json::to_string(&serde_json::json!({
        "purpose": "dense eval grid for stress-testing models trained on standard 200×100 + 8×8",
        "train_grid": [200, 100],
        "train_patch": [8, 8],
        "val_grid_derived_from_train_via_holdout_dims": true,
        "test_grid_BULK": [2000, 1000],
        "test_patch": [80, 80],
        "eps_prime_range": [1.0, 50.0],
        "eps_double_prime_range": [0.0, 25.0],
        "dense_patch_eps_prime_range": [1.0, 2.5],
        "dense_patch_eps_double_prime_range": [0.0, 1.75],
        "note": "test samples live at offset (0,0); train at offset (0,0). Different sub-domains via patch offsets."
    }))?;

    let dataset_id = db::insert_dataset(
        &conn,
        standard.train.len(),
        standard.validation.len(),
        test_samples.len(),
        &config_json,
        "",
    )?;
    eprintln!("Inserted dataset row #{dataset_id}, writing samples...");

    db::insert_dataset_samples(&mut conn, dataset_id, "train", &standard.train)?;
    db::insert_dataset_samples(&mut conn, dataset_id, "val", &standard.validation)?;
    db::insert_dataset_samples(&mut conn, dataset_id, "test", &test_samples)?;

    println!("OK — dataset_id={dataset_id}");
    println!("  train: {} samples (200×100 bulk + 8×8 patch)", standard.train.len());
    println!("  val:   {} samples", standard.validation.len());
    println!("  test:  {} samples (2000×1000 bulk + 80×80 patch)", test_samples.len());
    println!("Now open /evaluate, pick a trained model, and select Dataset #{dataset_id}.");
    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            return iter.next().cloned();
        }
    }
    None
}
