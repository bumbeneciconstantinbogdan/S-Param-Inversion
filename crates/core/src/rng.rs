//! Deterministic random number generation with global seed management.
//!
//! # Thread Safety
//!
//! - [`set_global_seed`] should be called once at program startup, before spawning threads.
//! - [`with_seeded_rng`] is safe to call from any thread, including rayon workers.
//! - Each thread gets a unique RNG stream derived from the global seed.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// CLI argument struct for seed configuration.
///
/// Can be embedded into the main application's CLI struct using `#[command(flatten)]`.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, clap::Args, Default)]
pub struct SeedArgs {
    /// Random seed for global initialization.
    #[arg(long, env = "CVNN_SEED", default_value_t = 42)]
    pub seed: u64,
}

static GLOBAL_SEED: AtomicU64 = AtomicU64::new(42);
static SEED_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Initializes all random number generators to ensure reproducible results.
///
/// Sets the internal seed for `candle_core` (weight initialization and tensor
/// operations) and stores the global seed for deriving local RNGs.
pub fn set_global_seed(seed: u64) {
    GLOBAL_SEED.store(seed, Ordering::SeqCst);
    SEED_GENERATION.fetch_add(1, Ordering::SeqCst);

    #[cfg(feature = "tensor")]
    {
        use candle_core::Device;
        let _ = Device::Cpu.set_seed(seed);

        if candle_core::utils::cuda_is_available() {
            if let Ok(device) = Device::new_cuda(0) {
                let _ = device.set_seed(seed);
            }
        }

        if candle_core::utils::metal_is_available() {
            if let Ok(device) = Device::new_metal(0) {
                let _ = device.set_seed(seed);
            }
        }
    }
}

/// Retrieves the current global seed.
pub fn get_global_seed() -> u64 {
    GLOBAL_SEED.load(Ordering::SeqCst)
}

/// Creates a new deterministically seeded RNG for sequential operations.
pub fn get_seeded_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(get_global_seed())
}

/// Derives a deterministic sub-seed based on the global seed and an index.
///
/// Useful for parallel operations (e.g., `rayon` iterators) where each
/// thread/task needs its own seeded RNG.
pub fn derive_sub_rng(index: u64) -> ChaCha8Rng {
    let global_seed = get_global_seed();
    let mixed_seed = global_seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    ChaCha8Rng::seed_from_u64(mixed_seed)
}

thread_local! {
    static SEEDED_RNG: RefCell<(u64, ChaCha8Rng)> = {
        let generation = SEED_GENERATION.load(Ordering::Relaxed);
        let seed = GLOBAL_SEED.load(Ordering::Relaxed);
        RefCell::new((generation, ChaCha8Rng::seed_from_u64(seed)))
    };
}

/// Thread-local ChaCha8 seeded from the current global seed (no rayon
/// thread-index mixing). Reseeds lazily on any `set_global_seed` bump,
/// so every worker that picks up work after the bump starts from the
/// same state — what HPO needs to stay reproducible under rayon
/// scheduling. Assumes only one trial is in-flight per process.
pub fn with_seeded_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut ChaCha8Rng) -> R,
{
    SEEDED_RNG.with(|cell| {
        let mut pair = cell.borrow_mut();
        let current_gen = SEED_GENERATION.load(Ordering::Relaxed);
        if pair.0 != current_gen {
            let seed = GLOBAL_SEED.load(Ordering::Relaxed);
            pair.0 = current_gen;
            pair.1 = ChaCha8Rng::seed_from_u64(seed);
        }
        f(&mut pair.1)
    })
}

// Seeded tensor helpers — drop-in replacements for `Tensor::{rand,randn,
// rand_like,randn_like}` that draw from `with_seeded_rng` instead of
// Candle's non-seedable CPU RNG.

use rand::RngExt as _;

/// Box-Muller standard normal on `rng`. The `1e-300` floor dodges
/// `ln(0) = -inf`. Public so [`crate::determinism`] can draw its
/// complex-weight init samples from the same kernel.
pub fn standard_normal_via(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random_range(1e-300_f64..1.0_f64);
    let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

/// Gaussian noise tensor drawn from the seeded thread-local ChaCha8.
/// Use instead of `Tensor::randn`.
///
/// Dtype-specialised:
/// - `F64`: fill a `Vec<f64>`, hand it to `from_vec` directly (zero-copy
///   on CPU).
/// - `F32`: fill a `Vec<f32>` by casting each sample in the generation
///   loop (`as f32` is IEEE round-to-nearest, matching what the
///   `to_dtype(F32)` kernel would do element-wise). Halves peak memory
///   vs. the F64-then-cast path and avoids one full-tensor kernel pass.
/// - Other (F16/BF16): fall back to the F64-generate-then-cast path;
///   unused in practice but kept for completeness.
///
/// RNG stream is unchanged: `standard_normal_via(rng)` is still called
/// once per element with the same sequence of `random_range` draws.
#[cfg(feature = "tensor")]
pub fn seeded_randn(
    shape: &candle_core::Shape,
    mean: f64,
    stdev: f64,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> candle_core::Result<candle_core::Tensor> {
    let n = shape.elem_count();
    match dtype {
        candle_core::DType::F64 => {
            let values: Vec<f64> = with_seeded_rng(|rng| {
                (0..n).map(|_| mean + stdev * standard_normal_via(rng)).collect()
            });
            candle_core::Tensor::from_vec(values, shape.clone(), device)
        }
        candle_core::DType::F32 => {
            let values: Vec<f32> = with_seeded_rng(|rng| {
                (0..n)
                    .map(|_| (mean + stdev * standard_normal_via(rng)) as f32)
                    .collect()
            });
            candle_core::Tensor::from_vec(values, shape.clone(), device)
        }
        _ => {
            let values: Vec<f64> = with_seeded_rng(|rng| {
                (0..n).map(|_| mean + stdev * standard_normal_via(rng)).collect()
            });
            candle_core::Tensor::from_vec(values, shape.clone(), device)?.to_dtype(dtype)
        }
    }
}

/// Matching-shape variant of [`seeded_randn`].
#[cfg(feature = "tensor")]
pub fn seeded_randn_like(
    src: &candle_core::Tensor,
    mean: f64,
    stdev: f64,
) -> candle_core::Result<candle_core::Tensor> {
    seeded_randn(src.shape(), mean, stdev, src.dtype(), src.device())
}

/// Uniform `[lo, hi)` tensor drawn from the seeded thread-local ChaCha8.
/// Use instead of `Tensor::rand`. See [`seeded_randn`] for the dtype
/// specialisation rationale — same treatment (F64 direct, F32 fused
/// cast, others via F64-then-`to_dtype`).
#[cfg(feature = "tensor")]
pub fn seeded_rand(
    shape: &candle_core::Shape,
    lo: f64,
    hi: f64,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> candle_core::Result<candle_core::Tensor> {
    let n = shape.elem_count();
    match dtype {
        candle_core::DType::F64 => {
            let values: Vec<f64> = with_seeded_rng(|rng| {
                (0..n).map(|_| rng.random_range(lo..hi)).collect()
            });
            candle_core::Tensor::from_vec(values, shape.clone(), device)
        }
        candle_core::DType::F32 => {
            let values: Vec<f32> = with_seeded_rng(|rng| {
                (0..n).map(|_| rng.random_range(lo..hi) as f32).collect()
            });
            candle_core::Tensor::from_vec(values, shape.clone(), device)
        }
        _ => {
            let values: Vec<f64> = with_seeded_rng(|rng| {
                (0..n).map(|_| rng.random_range(lo..hi)).collect()
            });
            candle_core::Tensor::from_vec(values, shape.clone(), device)?.to_dtype(dtype)
        }
    }
}

/// Matching-shape variant of [`seeded_rand`].
#[cfg(feature = "tensor")]
pub fn seeded_rand_like(
    src: &candle_core::Tensor,
    lo: f64,
    hi: f64,
) -> candle_core::Result<candle_core::Tensor> {
    seeded_rand(src.shape(), lo, hi, src.dtype(), src.device())
}

/// Shared mutex for tests that mutate the process-global seed. Acquire
/// it before calling `set_global_seed` so parallel tests don't clobber
/// each other's RNG state. Gated behind the `testing` feature so the
/// lock and its `LazyLock` storage don't ship in release binaries.
#[cfg(any(test, feature = "testing"))]
pub fn test_seed_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{LazyLock, Mutex};
    static LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    &LOCK
}

/// Resets seed state for testing. Not intended for production use.
#[cfg(test)]
pub fn reset_for_testing(seed: u64) {
    set_global_seed(seed);
    SEEDED_RNG.with(|cell| {
        let mut pair = cell.borrow_mut();
        pair.0 = u64::MAX;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;
    use rand::seq::SliceRandom;
    use std::sync::{LazyLock, Mutex, MutexGuard};

    static RNG_TEST_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn test_guard() -> MutexGuard<'static, ()> {
        // Ignore prior test panics that poison the mutex — each test
        // resets the seed state anyway via `reset_for_testing`, so the
        // contents under the lock are already invalidated.
        match RNG_TEST_MUTEX.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[test]
    fn reproducibility() {
        let _guard = test_guard();
        reset_for_testing(12345);
        let mut rng1 = get_seeded_rng();
        let seq1: Vec<u32> = (0..10).map(|_| rng1.random()).collect();

        reset_for_testing(12345);
        let mut rng2 = get_seeded_rng();
        let seq2: Vec<u32> = (0..10).map(|_| rng2.random()).collect();

        assert_eq!(seq1, seq2);
    }

    #[test]
    fn data_shuffling_reproducibility() {
        let _guard = test_guard();
        reset_for_testing(42);
        let mut rng1 = get_seeded_rng();
        let mut data1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        data1.shuffle(&mut rng1);

        reset_for_testing(42);
        let mut rng2 = get_seeded_rng();
        let mut data2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        data2.shuffle(&mut rng2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn sub_rng_parallel_reproducibility() {
        let _guard = test_guard();
        reset_for_testing(999);
        let mut rng_0a = derive_sub_rng(0);
        let mut rng_1a = derive_sub_rng(1);
        let val_0a: u32 = rng_0a.random();
        let val_1a: u32 = rng_1a.random();

        reset_for_testing(999);
        let mut rng_0b = derive_sub_rng(0);
        let mut rng_1b = derive_sub_rng(1);
        let val_0b: u32 = rng_0b.random();
        let val_1b: u32 = rng_1b.random();

        assert_eq!(val_0a, val_0b);
        assert_eq!(val_1a, val_1b);
        assert_ne!(val_0a, val_1a);
    }

}
