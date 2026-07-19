//! Shared helpers for benchmark binaries.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use sparam_models::{Activation, MLPConfig, MLPRegressor, Normalization};
use sparam_core::complex_tensor::ComplexTensor;

/// Standard benchmark sample size.
pub const BENCH_SAMPLE_SIZE: usize = 100;
/// Standard benchmark measurement time in seconds.
pub const BENCH_MEASUREMENT_SECS: u64 = 5;
/// Standard benchmark warm-up time in seconds.
pub const BENCH_WARMUP_SECS: u64 = 2;

/// Device selection for benchmarks (GPU when the `cuda-bench` feature is active).
pub fn get_benchmark_device() -> Device {
    #[cfg(feature = "cuda-bench")]
    if candle_core::utils::cuda_is_available() {
        return Device::new_cuda(0).unwrap_or(Device::Cpu);
    }
    Device::Cpu
}

/// Standard batch sizes used across NRW and model benchmarks.
pub const BATCH_SIZES: &[usize] = &[1, 16, 64, 256, 1024, 4096, 16384];

/// Hidden layer widths to sweep in model benchmarks.
pub const HIDDEN_SIZES: &[usize] = &[16, 32, 64, 128];

/// Standard waveguide parameters matching the Python reference implementation.
pub const WAVEGUIDE_D: f64 = 1.5e-3;
pub const WAVEGUIDE_A: f64 = 22.86e-3;
pub const FREQUENCY: f64 = 8.2e9;

/// Input/output dimensions for the real-valued MLP.
pub const REAL_INPUT_DIM: usize = 4;
pub const REAL_OUTPUT_DIM: usize = 2;

/// Input dimension for the complex-valued MLP.
pub const COMPLEX_INPUT_DIM: usize = 2;

/// Generate realistic complex permittivity tensors for NRW benchmarks.
///
/// Intermediate `Vec` allocations are intentional — this is bench setup code
/// outside the measured region.
pub fn generate_test_permittivity(n: usize, device: &Device) -> ComplexTensor {
    // bench-setup: allocation is outside the measured region
    let eps_real: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 % 99.0)).collect();
    let eps_imag: Vec<f64> = (0..n).map(|i| -(i as f64 % 50.0)).collect();
    ComplexTensor::new(
        Tensor::from_vec(eps_real, n, device).unwrap(),
        Tensor::from_vec(eps_imag, n, device).unwrap(),
    )
    .unwrap()
}

/// Generate realistic S-parameter tensors for NRW inverse benchmarks.
///
/// Intermediate `Vec` allocations are intentional — bench setup only.
pub fn generate_test_s_params(n: usize, device: &Device) -> (ComplexTensor, ComplexTensor) {
    // bench-setup: allocation is outside the measured region
    let s11_real: Vec<f64> = (0..n).map(|i| 0.3 * ((i as f64) * 0.01).cos()).collect();
    let s11_imag: Vec<f64> = (0..n).map(|i| 0.3 * ((i as f64) * 0.01).sin()).collect();
    let s21_real: Vec<f64> = (0..n).map(|i| 0.7 * ((i as f64) * 0.02).cos()).collect();
    let s21_imag: Vec<f64> = (0..n).map(|i| 0.7 * ((i as f64) * 0.02).sin()).collect();

    let s11 = ComplexTensor::new(
        Tensor::from_vec(s11_real, n, device).unwrap(),
        Tensor::from_vec(s11_imag, n, device).unwrap(),
    )
    .unwrap();
    let s21 = ComplexTensor::new(
        Tensor::from_vec(s21_real, n, device).unwrap(),
        Tensor::from_vec(s21_imag, n, device).unwrap(),
    )
    .unwrap();

    (s11, s21)
}

fn generate_f32_tensor(
    batch_size: usize,
    dim: usize,
    phase: f64,
    offset: f64,
    device: &Device,
) -> Tensor {
    let values: Vec<f32> = (0..batch_size * dim)
        .map(|i| ((i as f64) * phase + offset).sin() as f32)
        .collect();
    Tensor::from_vec(values, (batch_size, dim), device).unwrap()
}

/// Generate a deterministic real-valued input batch for MLP benchmarks.
pub fn generate_real_input(batch_size: usize, input_dim: usize, device: &Device) -> Tensor {
    generate_f32_tensor(batch_size, input_dim, 0.017, 0.0, device)
}

/// Generate a deterministic real-valued target batch for training benchmarks.
pub fn generate_real_targets(batch_size: usize, output_dim: usize, device: &Device) -> Tensor {
    generate_f32_tensor(batch_size, output_dim, 0.031, 0.7, device)
}

/// Generate a complex-valued input batch for complex MLP benchmarks.
///
/// Intermediate `Vec` allocations are intentional — bench setup only.
pub fn generate_complex_input(
    batch_size: usize,
    input_dim: usize,
    device: &Device,
) -> ComplexTensor {
    // bench-setup: allocation is outside the measured region
    let real: Vec<f64> = (0..batch_size * input_dim)
        .map(|i| ((i as f64) * 0.013).sin())
        .collect();
    let imag: Vec<f64> = (0..batch_size * input_dim)
        .map(|i| ((i as f64) * 0.019).cos())
        .collect();
    ComplexTensor::new(
        Tensor::from_vec(real, (batch_size, input_dim), device).unwrap(),
        Tensor::from_vec(imag, (batch_size, input_dim), device).unwrap(),
    )
    .unwrap()
}

/// Generate frequency tensors filled with `FREQUENCY` for NRW benchmarks.
pub fn generate_frequencies(n: usize, device: &Device) -> Tensor {
    Tensor::from_vec(vec![FREQUENCY; n], n, device).unwrap()
}

/// Generate unit permeability (`μ_r = 1`) for non-magnetic NRW direct benchmarks.
pub fn generate_unit_mu(n: usize, device: &Device) -> ComplexTensor {
    ComplexTensor::new(
        Tensor::ones(n, DType::F64, device).unwrap(),
        Tensor::zeros(n, DType::F64, device).unwrap(),
    )
    .unwrap()
}

/// Build a real-valued MLP with GELU activation, LayerNorm, and dropout.
///
/// Returns the `VarMap` (needed for optimizer setup) and the model. Common
/// setup used across training step and epoch benchmarks.
pub fn build_real_mlp(hidden: usize, device: &Device) -> (VarMap, MLPRegressor) {
    let config = MLPConfig::permittivity(hidden, Activation::GELU)
        .with_norm(Normalization::FusedLayerNorm)
        .with_dropout(0.1);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MLPRegressor::new(vb, &config).unwrap();
    (varmap, model)
}
