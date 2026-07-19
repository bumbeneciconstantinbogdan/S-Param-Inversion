//! Compute S-parameters for specific permittivity ranges using NRW direct method
//! 
//! Usage: cargo run --release --bin compute_sparams
//!
//! This script computes S11 and S21 for:
//! 1. Resonance region: er' = [148.8, 149.2], er'' = [0.0, 0.2]
//!    (includes the resonance points at er' = 149.15 and 149.1547)
//! 2. Near-vacuum region: er' = [1.0, 1.01], er'' = [0.0, 0.005]
//!    (very close to vacuum: er ≈ 1 - j0)
//!
//! Key findings:
//! - At resonance (er = 149.1547 - j0): S11 ≈ 0, S21 ≈ -1 (|S21| = 1)
//! - Near vacuum (er = 1.0 - j0): S11 ≈ 0, S21 ≈ 0.988 - j0.154 (|S21| ≈ 1)

use std::f64::consts::PI;

fn main() {
    // Waveguide configuration (WR-90 at 8.2 GHz)
    let d = 0.0015;      // sample thickness in meters
    let a = 0.02286;     // waveguide width in meters
    let freq = 8.2e9;    // frequency in Hz
    let c = 299_792_458.0; // speed of light in m/s

    let sep80 = "=".repeat(80);
    let sep95 = "-".repeat(95);

    println!("{}", sep80);
    println!("S-PARAMETERS COMPUTATION USING NRW DIRECT METHOD");
    println!("{}", sep80);
    println!("\nConfiguration:");
    println!("  Frequency:       {:.1} GHz", freq / 1e9);
    println!("  Waveguide width: {:.4} mm (WR-90)", a * 1000.0);
    println!("  Sample thickness: {:.4} mm", d * 1000.0);
    println!("  Material:        non-magnetic (mu_r = 1 + j0)");

    // ========================================================================
    // SECTION 1: RESONANCE REGION
    // ========================================================================
    // er' = [148.8, 149.2], er'' = [0.0, 0.2]
    // Includes critical point: er = 149.15 - j0 and er = 149.1547 - j0
    // At er = 149.1547, we expect resonance: beta_s * d = pi (n=1 half-wavelength)
    // This gives S11 ≈ 0 and |S21| ≈ 1
    // ========================================================================
    println!("\n{}", sep80);
    println!("SECTION 1: RESONANCE REGION");
    println!("  er' = [148.8, 149.2], er'' = [0.0, 0.2]");
    println!("  Includes: er = 149.15 - j0 and er = 149.1547 - j0 (resonance)");
    println!("{}", sep80);

    // Define the grid - include 149.15 explicitly as requested
    let eps_prime_resonance: Vec<f64> = vec![148.8, 148.9, 149.0, 149.1, 149.15, 149.1547, 149.2];
    let eps_double_prime_resonance: Vec<f64> = vec![0.0, 0.05, 0.1, 0.15, 0.2];

    // Pre-compute resonance info
    let k0_res = 2.0 * PI * freq / c;
    let beta_s_res = (k0_res.powi(2) * 149.1547 - (PI / a).powi(2)).sqrt();
    let beta_s_d_res = beta_s_res * d;

    println!("\nResonance condition: beta_s * d = n * pi (n=1)");
    println!("At er' = 149.1547, er'' = 0: beta_s * d = {:.6} rad (pi = {:.6} rad)", 
             beta_s_d_res, PI);
    
    println!("\nFormat: er' | er'' | S11 (real, imag) | S21 (real, imag) | |S11| | |S21| | Note");
    println!("{}", sep95);

    for &eps_prime in &eps_prime_resonance {
        for &eps_double_prime in &eps_double_prime_resonance {
            let (s11, s21) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
                d, a, freq, eps_prime, eps_double_prime
            );
            
            let marker = if (eps_prime - 149.15).abs() < 0.001 && (eps_double_prime - 0.0).abs() < 0.001 {
                " <-- 149.15 RESONANCE"
            } else if (eps_prime - 149.1547).abs() < 0.001 && (eps_double_prime - 0.0).abs() < 0.001 {
                " <-- 149.1547 RESONANCE"
            } else {
                ""
            };
            
            println!(
                "{:.4} | {:.4} | ({:+.6}, {:+.6}) | ({:+.6}, {:+.6}) | {:.6} | {:.6} |{}",
                eps_prime,
                eps_double_prime,
                s11.re,
                s11.im,
                s21.re,
                s21.im,
                s11.abs(),
                s21.abs(),
                marker
            );
        }
    }

    // ========================================================================
    // SECTION 2: NEAR-VACUUM REGION (very close to er = 1 - j0)
    // ========================================================================
    // er' = [1.0, 1.01], er'' = [0.0, 0.005]
    // This is a very tight range around vacuum permittivity
    // At er = 1.0 - j0: free space, S11 ≈ 0, S21 ≈ 0.988 - j0.154
    // ========================================================================
    println!("\n{}", sep80);
    println!("SECTION 2: NEAR-VACUUM REGION (very close to er = 1 - j0)");
    println!("  er' = [1.0, 1.01], er'' = [0.0, 0.005]");
    println!("{}", sep80);

    // Very fine grid near vacuum
    let eps_prime_vacuum: Vec<f64> = vec![1.0, 1.0025, 1.005, 1.0075, 1.01];
    let eps_double_prime_vacuum: Vec<f64> = vec![0.0, 0.00125, 0.0025, 0.00375, 0.005];

    println!("\nVacuum reference: er = 1.0 - j0.0 (free space)");
    println!("\nFormat: er' | er'' | S11 (real, imag) | S21 (real, imag) | |S11| | |S21| | Note");
    println!("{}", sep95);

    for &eps_prime in &eps_prime_vacuum {
        for &eps_double_prime in &eps_double_prime_vacuum {
            let (s11, s21) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
                d, a, freq, eps_prime, eps_double_prime
            );
            
            let marker = if (eps_prime - 1.0).abs() < 0.001 && (eps_double_prime - 0.0).abs() < 0.001 {
                " <-- VACUUM"
            } else {
                ""
            };
            
            println!(
                "{:.6} | {:.6} | ({:+.8}, {:+.8}) | ({:+.8}, {:+.8}) | {:.8} | {:.8} |{}",
                eps_prime,
                eps_double_prime,
                s11.re,
                s11.im,
                s21.re,
                s21.im,
                s11.abs(),
                s21.abs(),
                marker
            );
        }
    }

    // ========================================================================
    // SECTION 3: CRITICAL POINTS SUMMARY
    // ========================================================================
    println!("\n{}", sep80);
    println!("SECTION 3: CRITICAL POINTS SUMMARY");
    println!("{}", sep80);

    // Resonance point at 149.1547
    let (s11_res_1547, s21_res_1547) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
        d, a, freq, 149.1547, 0.0
    );
    
    // Resonance point at 149.15
    let (s11_res_15, s21_res_15) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
        d, a, freq, 149.15, 0.0
    );
    
    // Vacuum point
    let (s11_vac, s21_vac) = sparam_physics::nrw::nrw_direct_scalar_non_magnetic(
        d, a, freq, 1.0, 0.0
    );

    println!("\n--- RESONANCE POINTS (mu = 1) ---");
    println!("\nPoint 1: er = 149.1547 - j0.0 (exact n=1 half-wavelength resonance):");
    println!("  S11 = {:+.8} + j{:+.8}  | |S11| = {:.8e}", s11_res_1547.re, s11_res_1547.im, s11_res_1547.abs());
    println!("  S21 = {:+.8} + j{:+.8}  | |S21| = {:.8}", s21_res_1547.re, s21_res_1547.im, s21_res_1547.abs());
    println!("  -> |S11| ≈ 0, |S21| ≈ 1: This is the problematic case for NRW!");
    println!("     NRW method will FAIL here due to division by zero in reflection coefficient calculation.");
    
    println!("\nPoint 2: er = 149.15 - j0.0 (close to resonance):");
    println!("  S11 = {:+.8} + j{:+.8}  | |S11| = {:.8e}", s11_res_15.re, s11_res_15.im, s11_res_15.abs());
    println!("  S21 = {:+.8} + j{:+.8}  | |S21| = {:.8}", s21_res_15.re, s21_res_15.im, s21_res_15.abs());

    println!("\n--- VACUUM POINT (mu = 1) ---");
    println!("\nPoint 3: er = 1.0 - j0.0 (free space/vacuum):");
    println!("  S11 = {:+.8} + j{:+.8}  | |S11| = {:.8e}", s11_vac.re, s11_vac.im, s11_vac.abs());
    println!("  S21 = {:+.8} + j{:+.8}  | |S21| = {:.8}", s21_vac.re, s21_vac.im, s21_vac.abs());
    println!("  -> |S11| ≈ 0 but non-zero, |S21| ≈ 1: NRW should work here (barely).");

    // ========================================================================
    // SECTION 4: RESONANCE VERIFICATION
    // ========================================================================
    println!("\n{}", sep80);
    println!("SECTION 4: RESONANCE VERIFICATION");
    println!("{}", sep80);

    let k0 = 2.0 * PI * freq / c;
    let kt = PI / a;
    
    // For 149.1547
    let beta_s_sq_1547 = k0.powi(2) * 149.1547 - kt.powi(2);
    let beta_s_1547 = beta_s_sq_1547.sqrt();
    let beta_s_d_1547 = beta_s_1547 * d;
    
    // For 149.15
    let beta_s_sq_15 = k0.powi(2) * 149.15 - kt.powi(2);
    let beta_s_15 = beta_s_sq_15.sqrt();
    let beta_s_d_15 = beta_s_15 * d;

    println!("\nResonance condition: beta_s * d = n * pi (for n=1 half-wavelength)");
    println!("\nFor er' = 149.1547:");
    println!("  beta_s * d = {:.10} rad", beta_s_d_1547);
    println!("  pi         = {:.10} rad", PI);
    println!("  Difference = {:.10e} rad", beta_s_d_1547 - PI);
    println!("  Match? {}", (beta_s_d_1547 - PI).abs() < 1e-3);
    
    println!("\nFor er' = 149.15:");
    println!("  beta_s * d = {:.10} rad", beta_s_d_15);
    println!("  pi         = {:.10} rad", PI);
    println!("  Difference = {:.10e} rad", beta_s_d_15 - PI);
    println!("  Match? {}", (beta_s_d_15 - PI).abs() < 1e-3);

    // ========================================================================
    // SECTION 5: LOCUS POINTS ANALYSIS (mu = 1)
    // ========================================================================
    println!("\n{}", sep80);
    println!("SECTION 5: LOCUS POINTS ANALYSIS (mu = 1)");
    println!("{}", sep80);
    
    println!("\nFor mu = 1, the locus of S-parameters as epsilon varies:");
    println!("\nResonance region (er' ~ 149.15):");
    println!("  - S11 traces a small circle near origin (|S11| << 1)");
    println!("  - S21 traces a circle near |S21| = 1");
    println!("  - At exact resonance (er' = 149.1547, er'' = 0): S11 = 0, S21 = -1");
    
    println!("\nVacuum region (er' ~ 1.0):");
    println!("  - S11 traces a very small circle near origin (|S11| << 1)");
    println!("  - S21 traces a circle near |S21| = 1");
    println!("  - At vacuum (er' = 1.0, er'' = 0): S11 = 0, |S21| ≈ 1");
    
    println!("\nNumerical setup for mu = 1:");
    println!("  - Sample thickness d = {:.4} mm", d * 1000.0);
    println!("  - Waveguide width a = {:.4} mm", a * 1000.0);
    println!("  - Frequency f = {:.1} GHz", freq / 1e9);
    println!("  - WR-90 waveguide, TE10 mode");
    println!("  - Locus points for epsilon are computed at these fixed parameters");
}
