//! Data generation binary for S-parameter inversion
//!
//! Generates training/validation/test data with:
//! - eps_prime in [1, 200]
//! - eps_double_prime in [0, 100]
//! - S-parameters computed using NRW direct method
//!
//! Usage: cargo run --release --bin generate_data

use std::path::Path;
use sparam_data::generation::{generate_non_magnetic_data, DataGenerationConfig};
use sparam_core::io::ensure_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("GENERATING TRAINING DATA");
    println!("eps_prime range: [1, 200]");
    println!("eps_double_prime range: [0, 100]");
    println!("{}", "=".repeat(80));

    // Configuration for data generation
    let config = DataGenerationConfig {
        // Physical parameters (WR-90 waveguide at 8.2 GHz)
        d: 0.0015,           // sample thickness in meters
        a: 0.02286,          // waveguide width in meters
        frequency: 8.2e9,    // frequency in Hz
        
        // Permittivity ranges
        eps_prime_range: (1.0, 200.0),
        eps_double_prime_range: (0.0, 100.0),
        
        // Grid resolution
        n_eps_prime_train: 200,
        n_eps_double_prime_train: 100,
        
        // Dataset splits
        train_ratio: 0.8,
        val_offset: (0.0, 0.0),
        test_offset: (0.0, 0.0),
        seed: 42,
        
        // Dense patch (disabled for now)
        dense_patch_eps_prime_range: (1.0, 2.5),
        dense_patch_eps_double_prime_range: (0.0, 1.75),
        dense_patch_n_train: (0, 0),
        dense_patch_n_eval: (0, 0),
        dense_patch_val_offset: (0.0, 0.0),
        dense_patch_test_offset: (0.0, 0.0),
    };

    println!("\nGenerating non-magnetic dataset...");
    println!("  eps_prime: [{}, {}]", config.eps_prime_range.0, config.eps_prime_range.1);
    println!("  eps_double_prime: [{}, {}]", config.eps_double_prime_range.0, config.eps_double_prime_range.1);
    println!("  Grid: {} x {} = {} points", 
             config.n_eps_prime_train, config.n_eps_double_prime_train,
             config.n_eps_prime_train * config.n_eps_double_prime_train);

    // Generate data
    let dataset = generate_non_magnetic_data(&config)?;
    
    println!("\nDataset generated:");
    println!("  Train: {} samples", dataset.train.len());
    println!("  Val:   {} samples", dataset.validation.len());
    println!("  Test:  {} samples", dataset.test.len());

    // Save to CSV files
    let output_dir = Path::new("data");
    ensure_dir(output_dir)?;
    
    dataset.save_to_directory(output_dir)?;
    
    println!("\nData saved to: data/");
    println!("  - data_train.csv");
    println!("  - data_val.csv");
    println!("  - data_test.csv");

    // Print some statistics
    println!("\nTraining data statistics:");
    let mut eps_prime_sum = 0.0;
    let mut eps_prime_min = f64::INFINITY;
    let mut eps_prime_max = f64::NEG_INFINITY;
    for sample in &dataset.train {
        eps_prime_sum += sample.eps_prime;
        eps_prime_min = eps_prime_min.min(sample.eps_prime);
        eps_prime_max = eps_prime_max.max(sample.eps_prime);
    }
    let eps_prime_mean = eps_prime_sum / dataset.train.len() as f64;
    println!("  eps_prime: min={:.2}, max={:.2}, mean={:.2}", 
             eps_prime_min, eps_prime_max, eps_prime_mean);

    Ok(())
}
