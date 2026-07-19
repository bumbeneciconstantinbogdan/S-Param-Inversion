//! Reproducible training primitives.
//!
//! Candle's CPU RNG can't be seeded, so this module supplies:
//!
//! - [`deterministic_reinit_varmap`] — overwrites every variable with
//!   values drawn from a seeded ChaCha8, dispatched by name suffix so
//!   each role (linear weight, norm γ/β, ModReLU bias, complex He
//!   weights) keeps its original init scheme.
//! - [`stable_all_vars`] — sorted-by-name walk to replace
//!   `VarMap::all_vars()` in any cross-var reduction (otherwise
//!   HashMap iteration order is process-random and FP sums drift).
//! - [`clip_grad_norm_stable`] — in-place gradient 2-norm clip over a
//!   caller-supplied stable-ordered var slice. Single source of truth
//!   used by the trainer and the determinism stress tests.
//! - [`flatten_vars_f64`] — flatten a `VarMap` to `Vec<f64>` in
//!   sorted-name order, for bit-for-bit comparison in tests.
//! - Re-exported [`seeded_randn`], [`seeded_rand`] (and `_like` variants)
//!   from `sparam_core::rng` — drop-in replacements for Candle's
//!   non-seedable `Tensor::randn` / `Tensor::rand`.
//!
//! Init references: Trabelsi et al. 2018 (complex He), Arjovsky et al.
//! 2016 (ModReLU bias = −0.1).

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{Result, Shape, Tensor, Var};
use candle_nn::VarMap;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Re-export from sparam-core so both this crate and sparam-models can
// pull the same helpers from one path.
pub use crate::rng::{seeded_rand, seeded_rand_like, seeded_randn, seeded_randn_like};

/// ModReLU default bias, kept in sync with
/// `sparam_models::complex_activation::DEFAULT_MOD_RELU_BIAS`
/// (duplicated to avoid a reverse crate dependency).
const MOD_RELU_DEFAULT_BIAS: f64 = -0.1;

/// Re-initialise every variable in `varmap` from a ChaCha8 seeded with
/// `seed`, walked in sorted-by-name order. See the role table in the
/// classifier below for the per-suffix init scheme.
pub fn deterministic_reinit_varmap(varmap: &VarMap, seed: u64) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Collect (name, Var) under the VarMap lock, then drop the guard so
    // we can call `Var::set` without re-entering the same mutex.
    let entries: Vec<(String, Var)> = {
        let guard = varmap.data().lock().expect("VarMap mutex poisoned");
        let mut v: Vec<(String, Var)> = guard
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };

    // Lookup of every variable's shape so we can resolve "what's the
    // sibling `.weight` of this `.bias`?" for fan-in inference.
    let shape_of: HashMap<String, Shape> = entries
        .iter()
        .map(|(n, v)| (n.clone(), v.as_tensor().shape().clone()))
        .collect();

    for (name, var) in &entries {
        let tensor = var.as_tensor();
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();
        let device = tensor.device().clone();
        let elem_count = shape.elem_count();

        if elem_count == 0 {
            continue;
        }

        let role = classify(name, &shape, &shape_of);
        let values: Vec<f64> = match role {
            Role::LinearWeight { fan_in } => {
                // candle-nn Linear default: Kaiming-uniform, bound = 1/√fan_in.
                let bound = (1.0f64 / fan_in.max(1) as f64).sqrt();
                (0..elem_count).map(|_| rng.random_range(-bound..bound)).collect()
            }
            Role::ComplexLinearWeight { fan_in } => {
                // Trabelsi He: each part drawn from 𝒩(0, σ²=1/fan_in).
                let sigma = (1.0f64 / fan_in.max(1) as f64).sqrt();
                (0..elem_count).map(|_| standard_normal(&mut rng) * sigma).collect()
            }
            Role::LinearBias { fan_in } => {
                // Uniform(±1/√fan_in) — candle-nn Linear default bias init.
                let bound = (1.0f64 / fan_in.max(1) as f64).sqrt();
                (0..elem_count).map(|_| rng.random_range(-bound..bound)).collect()
            }
            Role::NormGamma => vec![1.0; elem_count],
            Role::NormBeta => vec![0.0; elem_count],
            Role::ModReLUBias => vec![MOD_RELU_DEFAULT_BIAS; elem_count],
            Role::Unknown => {
                // Fallback: keep zeros. Logging this would require a
                // dependency this crate doesn't have, so we stay silent.
                vec![0.0; elem_count]
            }
        };

        let init = Tensor::from_vec(values, elem_count, &device)?
            .to_dtype(dtype)?
            .reshape(&shape)?;
        var.set(&init)?;
    }
    Ok(())
}

/// Return every variable in `varmap` sorted by name. Drop-in
/// replacement for `VarMap::all_vars()` that pins the walk order so
/// cross-var reductions (grad-norm clipping, snapshot/restore) produce
/// the same floating-point result every run.
pub fn stable_all_vars(varmap: &VarMap) -> Vec<candle_core::Var> {
    let guard = varmap.data().lock().expect("VarMap mutex poisoned");
    let mut entries: Vec<(String, candle_core::Var)> = guard
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.into_iter().map(|(_, v)| v).collect()
}

/// Clip gradient 2-norm in place, walking `vars` in the given order.
/// Caller must pass a stable-ordered slice (use [`stable_all_vars`]) so
/// the cross-var sum reduces in the same order across runs. Returns the
/// pre-clip norm and errors if any grad is non-finite.
pub fn clip_grad_norm_stable(
    grads: &mut GradStore,
    vars: &[Var],
    max_norm: f64,
) -> Result<f64> {
    let mut total_norm_sq: Option<Tensor> = None;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let sq = g.sqr()?.sum_all()?;
            total_norm_sq = Some(match total_norm_sq {
                None => sq,
                Some(acc) => (acc + sq)?,
            });
        }
    }

    let total_norm = match total_norm_sq {
        None => 0.0,
        Some(t) => scalar_f64(&t)?.sqrt(),
    };
    if !total_norm.is_finite() {
        return Err(candle_core::Error::Msg(format!(
            "non-finite gradient norm encountered: {total_norm}"
        )));
    }

    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for v in vars {
            if let Some(g) = grads.remove(v.as_tensor()) {
                grads.insert(v.as_tensor(), g.affine(scale, 0.0)?);
            }
        }
    }

    Ok(total_norm)
}

/// 64-bit SipHash of every variable's f64 bits in sorted-name
/// order. Same input ⇒ same digest; bit-level divergence across
/// runs shows up as a hash mismatch. Not persisted — stable only
/// within one Rust version.
pub fn hash_varmap(varmap: &VarMap) -> u64 {
    use std::hash::{Hash, Hasher};

    let flat = flatten_vars_f64(varmap);
    let mut hasher = std::hash::DefaultHasher::new();
    flat.len().hash(&mut hasher);
    for value in flat {
        // bit pattern so −0.0 ≠ 0.0 and NaN payloads compare distinctly.
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// `SPARAM_DEBUG_HASH_VARMAP` env var check, cached.
pub fn debug_hash_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0); // 0=unchecked, 1=off, 2=on
    match CACHED.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let enabled = std::env::var_os("SPARAM_DEBUG_HASH_VARMAP")
                .is_some_and(|v| !v.is_empty());
            CACHED.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
            enabled
        }
    }
}

/// Print `[varmap hash] <tag> = 0x…` to stderr when the debug env
/// var is set. Bypasses `LogSender` so it fires even from silent
/// paths (`/train` builds with `log_progress: false`).
pub fn emit_varmap_hash(tag: std::fmt::Arguments<'_>, varmap: &VarMap) {
    if !debug_hash_enabled() {
        return;
    }
    eprintln!("[varmap hash] {tag} = 0x{:016x}", hash_varmap(varmap));
}

/// Flatten every variable in `varmap` to `f64`, walked in sorted-name
/// order. Used by determinism tests to compare full parameter vectors
/// element-for-element across runs. Panics on tensor errors — tests
/// relying on this helper would fail anyway if that ever occurred.
pub fn flatten_vars_f64(varmap: &VarMap) -> Vec<f64> {
    let vars = stable_all_vars(varmap);
    let total: usize = vars.iter().map(|v| v.as_tensor().elem_count()).sum();
    let mut flat = Vec::with_capacity(total);
    for v in &vars {
        let t = v
            .as_tensor()
            .flatten_all()
            .expect("flatten_all failed on VarMap tensor");
        flat.extend(
            t.to_vec1::<f64>()
                .expect("to_vec1::<f64> failed — VarMap tensor was not F64"),
        );
    }
    flat
}

/// Extract a scalar from a rank-0 tensor as `f64`, accepting both F32 and F64.
fn scalar_f64(t: &Tensor) -> Result<f64> {
    use candle_core::DType;
    match t.dtype() {
        DType::F32 => t.to_scalar::<f32>().map(f64::from),
        DType::F64 => t.to_scalar::<f64>(),
        dtype => Err(candle_core::Error::Msg(format!(
            "expected F32 or F64 scalar, got {dtype:?}"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Role {
    /// `*.weight`, rank 2, real linear layer weight.
    LinearWeight { fan_in: usize },
    /// `*.weight_re` / `*.weight_im`, rank 2, complex linear layer weight.
    ComplexLinearWeight { fan_in: usize },
    /// `*.bias`, rank 1, sibling `weight` is rank 2. Also used for
    /// `*.bias_re` / `*.bias_im` with their rank-2 complex siblings.
    LinearBias { fan_in: usize },
    /// `*.weight`, rank 1, γ scale of a Norm layer.
    NormGamma,
    /// `*.bias`, rank 1, sibling `weight` is rank 1 (Norm β shift).
    NormBeta,
    /// `*.bias`, rank 1, no sibling weight → assumed ModReLU learnable bias.
    ModReLUBias,
    /// Unrecognised suffix / layout. Initialised to zeros.
    Unknown,
}

fn classify(name: &str, shape: &Shape, shape_of: &HashMap<String, Shape>) -> Role {
    let (prefix, suffix) = split_last_dot(name);
    let rank = shape.rank();
    let last_dim = shape.dims().last().copied().unwrap_or(1);

    match suffix {
        "weight" => {
            if rank >= 2 {
                Role::LinearWeight { fan_in: last_dim }
            } else {
                Role::NormGamma
            }
        }
        "weight_re" | "weight_im" => {
            if rank >= 2 {
                Role::ComplexLinearWeight { fan_in: last_dim }
            } else {
                Role::Unknown
            }
        }
        "bias" => {
            // Look up sibling `{prefix}.weight`. If it's rank-2 we're a
            // linear-layer bias; rank-1 we're a norm β; missing we're a
            // ModReLU learnable bias (standalone).
            let sibling = format!("{prefix}.weight");
            match shape_of.get(&sibling) {
                Some(w_shape) if w_shape.rank() >= 2 => {
                    let fan_in = w_shape.dims().last().copied().unwrap_or(1);
                    Role::LinearBias { fan_in }
                }
                Some(_) => Role::NormBeta,
                None => Role::ModReLUBias,
            }
        }
        "bias_re" | "bias_im" => {
            // Complex-linear bias: pair with `weight_re` / `weight_im`.
            let partner = if suffix == "bias_re" {
                format!("{prefix}.weight_re")
            } else {
                format!("{prefix}.weight_im")
            };
            match shape_of.get(&partner) {
                Some(w_shape) if w_shape.rank() >= 2 => {
                    let fan_in = w_shape.dims().last().copied().unwrap_or(1);
                    Role::LinearBias { fan_in }
                }
                _ => Role::Unknown,
            }
        }
        _ => Role::Unknown,
    }
}

/// Split `"a.b.c"` into `("a.b", "c")`. Returns `("", name)` if there's no dot.
fn split_last_dot(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(i) => (&name[..i], &name[i + 1..]),
        None => ("", name),
    }
}

use crate::rng::standard_normal_via as standard_normal;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_var(shape: (usize, usize), dev: &Device) -> Var {
        Var::randn(0.0f64, 1.0, shape, dev).unwrap()
    }
    fn make_var_1d(features: usize, dev: &Device) -> Var {
        Var::ones(features, DType::F64, dev).unwrap()
    }

    #[test]
    fn seeded_randn_produces_same_values_for_same_seed() {
        use crate::rng::{set_global_seed, test_seed_lock};
        let _g = test_seed_lock().lock().unwrap_or_else(|e| e.into_inner());
        let dev = Device::Cpu;

        set_global_seed(42);
        let a = seeded_randn(&(4, 4).into(), 0.0, 1.0, DType::F64, &dev).unwrap()
            .to_vec2::<f64>().unwrap();

        set_global_seed(42);
        let b = seeded_randn(&(4, 4).into(), 0.0, 1.0, DType::F64, &dev).unwrap()
            .to_vec2::<f64>().unwrap();

        assert_eq!(a, b, "seeded_randn must be deterministic for same seed");

        set_global_seed(99);
        let c = seeded_randn(&(4, 4).into(), 0.0, 1.0, DType::F64, &dev).unwrap()
            .to_vec2::<f64>().unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn seeded_rand_produces_same_values_for_same_seed() {
        use crate::rng::{set_global_seed, test_seed_lock};
        let _g = test_seed_lock().lock().unwrap_or_else(|e| e.into_inner());
        let dev = Device::Cpu;

        set_global_seed(7);
        let a = seeded_rand(&(3, 3).into(), 0.0, 1.0, DType::F64, &dev).unwrap()
            .to_vec2::<f64>().unwrap();

        set_global_seed(7);
        let b = seeded_rand(&(3, 3).into(), 0.0, 1.0, DType::F64, &dev).unwrap()
            .to_vec2::<f64>().unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn reinit_is_deterministic_for_same_seed() {
        let dev = Device::Cpu;
        let vm1 = VarMap::new();
        let vm2 = VarMap::new();
        for vm in [&vm1, &vm2] {
            vm.data().lock().unwrap().insert("layer.weight".into(), make_var((4, 3), &dev));
            vm.data().lock().unwrap().insert("layer.bias".into(), make_var_1d(4, &dev));
        }

        deterministic_reinit_varmap(&vm1, 42).unwrap();
        deterministic_reinit_varmap(&vm2, 42).unwrap();

        let w1 = vm1.data().lock().unwrap()["layer.weight"]
            .as_tensor().to_vec2::<f64>().unwrap();
        let w2 = vm2.data().lock().unwrap()["layer.weight"]
            .as_tensor().to_vec2::<f64>().unwrap();
        assert_eq!(w1, w2);
    }

    #[test]
    fn reinit_differs_for_different_seeds() {
        let dev = Device::Cpu;
        let vm1 = VarMap::new();
        let vm2 = VarMap::new();
        for vm in [&vm1, &vm2] {
            vm.data().lock().unwrap().insert("layer.weight".into(), make_var((4, 3), &dev));
        }
        deterministic_reinit_varmap(&vm1, 42).unwrap();
        deterministic_reinit_varmap(&vm2, 43).unwrap();

        let w1 = vm1.data().lock().unwrap()["layer.weight"]
            .as_tensor().to_vec2::<f64>().unwrap();
        let w2 = vm2.data().lock().unwrap()["layer.weight"]
            .as_tensor().to_vec2::<f64>().unwrap();
        assert_ne!(w1, w2);
    }

    #[test]
    fn real_linear_weight_uses_kaiming_uniform() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // fan_in = 100, so bound = 1/√100 = 0.1
        vm.data().lock().unwrap().insert("fc.weight".into(), make_var((50, 100), &dev));
        deterministic_reinit_varmap(&vm, 42).unwrap();

        let w = vm.data().lock().unwrap()["fc.weight"]
            .as_tensor().to_vec2::<f64>().unwrap();
        let bound = 0.1_f64;
        for row in &w {
            for &v in row {
                assert!(v.abs() <= bound + 1e-12, "value {v} outside Kaiming bound");
            }
        }
        // Empirical variance ≈ bound²/3 = 1e-2/3.
        let n = (w.len() * w[0].len()) as f64;
        let mean: f64 = w.iter().flatten().sum::<f64>() / n;
        let var: f64 = w.iter().flatten().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let expected = bound * bound / 3.0;
        assert!(
            (var - expected).abs() < 0.2 * expected,
            "empirical var {var}, expected ≈ {expected}"
        );
    }

    #[test]
    fn complex_linear_weights_use_trabelsi_he_gaussian() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // fan_in = 128; each part ~ 𝒩(0, 1/fan_in).
        vm.data().lock().unwrap().insert("cl.weight_re".into(), make_var((64, 128), &dev));
        vm.data().lock().unwrap().insert("cl.weight_im".into(), make_var((64, 128), &dev));
        deterministic_reinit_varmap(&vm, 42).unwrap();

        for key in ["cl.weight_re", "cl.weight_im"] {
            let w = vm.data().lock().unwrap()[key]
                .as_tensor().to_vec2::<f64>().unwrap();
            let n = (w.len() * w[0].len()) as f64;
            let mean: f64 = w.iter().flatten().sum::<f64>() / n;
            let var: f64 = w.iter().flatten().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let target = 1.0 / 128.0;
            assert!(
                (var - target).abs() < 0.25 * target,
                "{key} empirical var {var}, target {target}"
            );
        }
    }

    #[test]
    fn norm_gamma_is_ones_and_beta_is_zeros() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // LayerNorm: weight (rank 1) is γ, bias (rank 1) is β.
        vm.data().lock().unwrap().insert("ln.weight".into(), make_var_1d(8, &dev));
        vm.data().lock().unwrap().insert("ln.bias".into(), make_var_1d(8, &dev));
        deterministic_reinit_varmap(&vm, 42).unwrap();

        let g = vm.data().lock().unwrap()["ln.weight"]
            .as_tensor().to_vec1::<f64>().unwrap();
        let b = vm.data().lock().unwrap()["ln.bias"]
            .as_tensor().to_vec1::<f64>().unwrap();
        assert_eq!(g, vec![1.0; 8]);
        assert_eq!(b, vec![0.0; 8]);
    }

    #[test]
    fn modrelu_bias_defaults_to_negative_zero_point_one() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // A standalone `bias` with no sibling `weight` → ModReLU learnable bias.
        vm.data().lock().unwrap().insert("modrelu.bias".into(), make_var_1d(5, &dev));
        deterministic_reinit_varmap(&vm, 42).unwrap();

        let b = vm.data().lock().unwrap()["modrelu.bias"]
            .as_tensor().to_vec1::<f64>().unwrap();
        assert_eq!(b, vec![MOD_RELU_DEFAULT_BIAS; 5]);
    }

    #[test]
    fn linear_bias_uses_sibling_weight_fan_in() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // fan_in = 100 → bias bound = 1/√100 = 0.1.
        vm.data().lock().unwrap().insert("fc.weight".into(), make_var((4, 100), &dev));
        vm.data().lock().unwrap().insert("fc.bias".into(), make_var_1d(4, &dev));
        deterministic_reinit_varmap(&vm, 42).unwrap();

        let b = vm.data().lock().unwrap()["fc.bias"]
            .as_tensor().to_vec1::<f64>().unwrap();
        for v in &b {
            assert!(v.abs() <= 0.1 + 1e-12);
        }
    }

    #[test]
    fn complex_linear_bias_uses_sibling_weight_fan_in() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        vm.data().lock().unwrap().insert("cl.weight_re".into(), make_var((4, 64), &dev));
        vm.data().lock().unwrap().insert("cl.weight_im".into(), make_var((4, 64), &dev));
        vm.data().lock().unwrap().insert("cl.bias_re".into(), make_var_1d(4, &dev));
        vm.data().lock().unwrap().insert("cl.bias_im".into(), make_var_1d(4, &dev));
        deterministic_reinit_varmap(&vm, 42).unwrap();

        let bound = 1.0f64 / 64.0f64.sqrt();
        for k in ["cl.bias_re", "cl.bias_im"] {
            let b = vm.data().lock().unwrap()[k]
                .as_tensor().to_vec1::<f64>().unwrap();
            for v in &b {
                assert!(v.abs() <= bound + 1e-12);
            }
        }
    }
}
