//! Complex-valued linear layer.
//!
//! Implements the standard complex linear transform:
//! `y = W·x + b` where `W`, `x`, `b`, and `y` are all complex-valued,
//! realized as pairs of real Candle tensors.

use std::fmt;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::error::candle_msg;

/// Complex weight initialization following Trabelsi et al. (2018),
/// *Deep Complex Networks*.
///
/// For a complex weight `W = Re(W) + i·Im(W)`, both parts are drawn from
/// `N(0, σ²)` where `σ² = Var(W) / 2` (the factor of 2 splits the total
/// variance equally between real and imaginary components).
///
/// The total variance `Var(W)` depends on the scheme:
///
/// - **He** (`2 / fan_in`): default for ReLU-family and magnitude-gated
///   activations (CReLU, CGELU, Cardioid, ModReLU, LPMA, CSwishPhase,
///   HybridCardioidGelu).
/// - **Glorot** (`2 / (fan_in + fan_out)`): for tanh/sigmoid-style
///   bounded activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexInit {
    /// Complex He (Kaiming) — σ² = 1 / fan_in per part.
    He,
    /// Complex Glorot (Xavier) — σ² = 1 / (fan_in + fan_out) per part.
    Glorot,
}

impl ComplexInit {
    /// Return the candle [`Init`] for one component (real or imaginary).
    pub fn to_candle_init(self, fan_in: usize, fan_out: usize) -> candle_nn::Init {
        let sigma = match self {
            // Var(W) = 2/fan_in  →  per-part σ² = 1/fan_in
            Self::He => 1.0 / (fan_in as f64).sqrt(),
            // Var(W) = 2/(fan_in+fan_out)  →  per-part σ² = 1/(fan_in+fan_out)
            Self::Glorot => 1.0 / ((fan_in + fan_out) as f64).sqrt(),
        };
        candle_nn::Init::Randn { mean: 0.0, stdev: sigma }
    }
}

impl Default for ComplexInit {
    fn default() -> Self {
        Self::He
    }
}

/// A complex-valued linear layer: $y = Wx + b$ where $W, x, b, y \in \mathbb{C}$.
///
/// Stores four real tensors internally:
/// - `weight_re`, `weight_im`: shape `(out_features, in_features)`
/// - `bias_re`, `bias_im`: shape `(out_features,)` (optional)
///
/// The forward pass computes:
/// $$y_r = W_r x_r - W_i x_i + b_r$$
/// $$y_i = W_r x_i + W_i x_r + b_i$$
///
/// Backpropagation through Candle's autograd on the decomposed real operations
/// yields the correct Wirtinger gradients for a real-valued loss.
#[derive(Clone, Debug)]
pub struct ComplexLinear {
    weight_re: Tensor,
    weight_im: Tensor,
    bias_re: Option<Tensor>,
    bias_im: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl ComplexLinear {
    #[cfg(any(test, doctest))]
    /// Construct from raw tensors (used internally and for deterministic test injection).
    pub fn from_parts(
        weight_re: Tensor,
        weight_im: Tensor,
        bias_re: Option<Tensor>,
        bias_im: Option<Tensor>,
    ) -> Result<Self> {
        let dims = weight_re.dims2()?;
        let out_features = dims.0;
        let in_features = dims.1;

        if weight_im.dims2()? != (out_features, in_features) {
            return Err(candle_msg(format!(
                "weight_im shape {:?} must match weight_re shape {:?}",
                weight_im.dims(),
                weight_re.dims()
            )));
        }

        match (&bias_re, &bias_im) {
            (Some(br), Some(bi)) => {
                let br_len = br.dims1()?;
                let bi_len = bi.dims1()?;
                if br_len != out_features || bi_len != out_features {
                    return Err(candle_msg(format!(
                        "bias shapes ({br_len}, {bi_len}) must both equal out_features ({out_features})"
                    )));
                }
            }
            (None, None) => {}
            _ => {
                return Err(candle_msg(
                    "bias_re and bias_im must either both be present or both be absent",
                ));
            }
        }

        Ok(Self {
            weight_re,
            weight_im,
            bias_re,
            bias_im,
            in_features,
            out_features,
        })
    }

    /// Construct a complex linear layer via [`VarBuilder`] using the default
    /// Complex He initialization (appropriate for ReLU-family activations).
    ///
    /// The VarBuilder prefix should scope the four parameter tensors:
    /// `weight_re`, `weight_im`, `bias_re`, `bias_im`.
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_init(in_features, out_features, ComplexInit::He, vb)
    }

    /// Construct a complex linear layer with explicit [`ComplexInit`] scheme.
    ///
    /// Use [`ComplexInit::He`] for ReLU-family activations (CReLU, zReLU,
    /// ModReLU, etc.) and [`ComplexInit::Glorot`] for Tanh/Sigmoid/linear.
    pub fn new_with_init(
        in_features: usize,
        out_features: usize,
        init: ComplexInit,
        vb: VarBuilder,
    ) -> Result<Self> {
        let init_ws = init.to_candle_init(in_features, out_features);
        let bound = 1.0 / (in_features as f64).sqrt();
        let init_bs = candle_nn::Init::Uniform {
            lo: -bound,
            up: bound,
        };

        let weight_re = vb.get_with_hints((out_features, in_features), "weight_re", init_ws)?;
        let weight_im = vb.get_with_hints((out_features, in_features), "weight_im", init_ws)?;
        let bias_re = vb.get_with_hints(out_features, "bias_re", init_bs)?;
        let bias_im = vb.get_with_hints(out_features, "bias_im", init_bs)?;

        Ok(Self {
            weight_re,
            weight_im,
            bias_re: Some(bias_re),
            bias_im: Some(bias_im),
            in_features,
            out_features,
        })
    }

    #[cfg(any(test, doctest))]
    /// Construct a complex linear layer **without bias** via [`VarBuilder`].
    pub fn new_no_bias(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let init_ws = ComplexInit::He.to_candle_init(in_features, out_features);

        let weight_re = vb.get_with_hints((out_features, in_features), "weight_re", init_ws)?;
        let weight_im = vb.get_with_hints((out_features, in_features), "weight_im", init_ws)?;

        Ok(Self {
            weight_re,
            weight_im,
            bias_re: None,
            bias_im: None,
            in_features,
            out_features,
        })
    }

    /// Run the complex linear forward pass.
    ///
    /// Input `xs` must have real and imaginary parts shaped `(batch, in_features)`.
    /// Output has real and imaginary parts shaped `(batch, out_features)`.
    ///
    /// Fuses the 4 real matmuls of the naïve complex product into one
    /// `(B, 2F) × (2F, 2H)` matmul by building the block weight
    /// `W_fused = [[W_re, -W_im], [W_im, W_re]]` at call time. BLAS
    /// amortizes setup cost over the larger kernel and SIMD lanes fill
    /// better at 2× width; the cat overhead is O(HF) and dwarfed by the
    /// matmul savings on CPU. Autograd routes through cat / matmul /
    /// narrow natively — the Wirtinger gradients come out identical to
    /// the four-matmul form.
    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        self.validate_input(xs)?;

        // Build fused weight: [[W_re, -W_im], [W_im, W_re]] — shape (2H, 2F).
        let neg_w_im = self.weight_im.neg()?;
        let top = Tensor::cat(&[&self.weight_re, &neg_w_im], 1)?;
        let bot = Tensor::cat(&[&self.weight_im, &self.weight_re], 1)?;
        let w_fused = Tensor::cat(&[&top, &bot], 0)?;

        // Pack input [x_re | x_im] along features — shape (B, 2F).
        let x_packed = Tensor::cat(&[&xs.real, &xs.imag], 1)?;

        // Single matmul: (B, 2F) × (2F, 2H) = (B, 2H).
        let y_packed = x_packed.matmul(&w_fused.t()?)?;

        let y_packed = match (&self.bias_re, &self.bias_im) {
            (Some(br), Some(bi)) => {
                let b_fused = Tensor::cat(&[br, bi], 0)?;
                y_packed.broadcast_add(&b_fused)?
            }
            (None, None) => y_packed,
            _ => {
                return Err(candle_msg(
                    "ComplexLinear bias_re/bias_im must both be present or both absent",
                ));
            }
        };

        // Zero-copy view into the packed output for the re/im split.
        let h = self.out_features;
        let y_re = y_packed.narrow(1, 0, h)?;
        let y_im = y_packed.narrow(1, h, h)?;

        Ok(ComplexTensor::new_unchecked(y_re, y_im))
    }

    #[cfg(any(test, doctest))]
    /// Number of learnable (real) parameters.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let weight_params = 2 * self.out_features * self.in_features;
        let bias_params = if self.bias_re.is_some() {
            2 * self.out_features
        } else {
            0
        };
        weight_params + bias_params
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Per-batch shape check, gated behind `debug_assertions`. Called
    /// twice per `ComplexMLPRegressor` forward (input layer + output
    /// layer); skipping it in release trims a handful of `usize`
    /// comparisons per batch. Release-mode shape bugs surface via
    /// Candle's matmul error on the dimension mismatch.
    fn validate_input(&self, xs: &ComplexTensor) -> Result<()> {
        #[cfg(debug_assertions)]
        {
            let r_dims = xs.real.dims();
            if r_dims.len() != 2 {
                return Err(candle_msg(format!(
                    "ComplexLinear expects 2D input (batch, {}), got shape {:?}",
                    self.in_features, r_dims
                )));
            }
            if r_dims[1] != self.in_features {
                return Err(candle_msg(format!(
                    "ComplexLinear expected {} input features, got {}",
                    self.in_features, r_dims[1]
                )));
            }
        }
        let _ = xs;
        Ok(())
    }
}

impl fmt::Display for ComplexLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComplexLinear({} -> {}, bias={})",
            self.in_features,
            self.out_features,
            self.bias_re.is_some()
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor, Var};
    use candle_nn::{VarBuilder, VarMap};

    use super::*;

    fn tensor_map(entries: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
        entries
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect()
    }

    // ── Shape & parameter-count tests ───────────────────────────────────

    #[test]
    fn test_complex_linear_forward_produces_expected_output_shape() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new(4, 3, vb)?;

        let xs = ComplexTensor::new(
            Tensor::zeros((5, 4), DType::F64, &Device::Cpu)?,
            Tensor::zeros((5, 4), DType::F64, &Device::Cpu)?,
        )?;

        let ys = layer.forward(&xs)?;

        assert_eq!(ys.real.dims(), &[5, 3]);
        assert_eq!(ys.imag.dims(), &[5, 3]);
        assert_eq!(layer.parameter_count(), 2 * 4 * 3 + 2 * 3); // 30
        assert_eq!(layer.in_features(), 4);
        assert_eq!(layer.out_features(), 3);

        Ok(())
    }

    #[test]
    fn test_complex_linear_display_is_log_friendly() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);

        let with_bias = ComplexLinear::new(4, 3, vb.pp("with_bias"))?;
        let without_bias = ComplexLinear::new_no_bias(4, 3, vb.pp("without_bias"))?;

        assert_eq!(with_bias.to_string(), "ComplexLinear(4 -> 3, bias=true)");
        assert_eq!(
            without_bias.to_string(),
            "ComplexLinear(4 -> 3, bias=false)"
        );

        Ok(())
    }

    #[test]
    fn test_complex_linear_no_bias_forward_and_param_count() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new_no_bias(4, 3, vb)?;

        let xs = ComplexTensor::new(
            Tensor::ones((2, 4), DType::F64, &Device::Cpu)?,
            Tensor::ones((2, 4), DType::F64, &Device::Cpu)?,
        )?;
        let ys = layer.forward(&xs)?;

        assert_eq!(ys.real.dims(), &[2, 3]);
        assert_eq!(ys.imag.dims(), &[2, 3]);
        assert_eq!(layer.parameter_count(), 2 * 4 * 3); // 24

        Ok(())
    }

    // ── Input validation tests ──────────────────────────────────────────

    #[test]
    fn test_complex_linear_rejects_wrong_rank_and_feature_count() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new(4, 3, vb)?;

        let wrong_rank = ComplexTensor::new(
            Tensor::zeros(4, DType::F64, &Device::Cpu)?,
            Tensor::zeros(4, DType::F64, &Device::Cpu)?,
        )?;
        let wrong_features = ComplexTensor::new(
            Tensor::zeros((2, 5), DType::F64, &Device::Cpu)?,
            Tensor::zeros((2, 5), DType::F64, &Device::Cpu)?,
        )?;

        assert!(layer.forward(&wrong_rank).is_err());
        assert!(layer.forward(&wrong_features).is_err());

        Ok(())
    }

    // ── Forward correctness (manual golden values) ──────────────────────

    #[test]
    fn test_complex_linear_forward_matches_hand_calculated_reference() -> Result<()> {
        // Layer: 2 -> 2, with bias.
        //
        //   W_re = [[1, 2],    W_im = [[0, -1],
        //           [3, 4]]            [1,  0]]
        //
        //   b_re = [0.5, -0.5],  b_im = [0.1, 0.2]
        //
        // Input (1 sample):
        //   x_re = [1, 0],  x_im = [0, 1]
        //
        // y_re = x_re @ W_re^T - x_im @ W_im^T + b_re
        //      = [1,0]@[[1,3],[2,4]] - [0,1]@[[0,1],[-1,0]] + [0.5,-0.5]
        //      = [1, 3] - [-1, 0] + [0.5, -0.5]
        //      = [2.5, 2.5]
        //
        // y_im = x_re @ W_im^T + x_im @ W_re^T + b_im
        //      = [1,0]@[[0,1],[-1,0]] + [0,1]@[[1,3],[2,4]] + [0.1, 0.2]
        //      = [0, 1] + [2, 4] + [0.1, 0.2]
        //      = [2.1, 5.2]

        let tensors = tensor_map(vec![
            (
                "weight_re",
                Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?,
            ),
            (
                "weight_im",
                Tensor::from_vec(vec![0.0f64, -1.0, 1.0, 0.0], (2, 2), &Device::Cpu)?,
            ),
            (
                "bias_re",
                Tensor::from_vec(vec![0.5f64, -0.5], 2, &Device::Cpu)?,
            ),
            (
                "bias_im",
                Tensor::from_vec(vec![0.1f64, 0.2], 2, &Device::Cpu)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new(2, 2, vb)?;

        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 0.0], (1, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![0.0f64, 1.0], (1, 2), &Device::Cpu)?,
        )?;

        let ys = layer.forward(&xs)?;
        let y_re = ys.real.to_vec2::<f64>()?;
        let y_im = ys.imag.to_vec2::<f64>()?;

        assert_eq!(y_re, vec![vec![2.5, 2.5]]);
        assert_eq!(y_im, vec![vec![2.1, 5.2]]);

        Ok(())
    }

    #[test]
    fn test_complex_linear_no_bias_forward_matches_hand_calculated_reference() -> Result<()> {
        // Same weights as above, no bias, batched input (2 samples).
        let layer = ComplexLinear::from_parts(
            Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![0.0f64, -1.0, 1.0, 0.0], (2, 2), &Device::Cpu)?,
            None,
            None,
        )?;

        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![0.0f64, 1.0, 1.0, 0.0], (2, 2), &Device::Cpu)?,
        )?;

        // Sample 0: same as biased test but without bias
        // y_re = [1,3] - [-1,0] = [2, 3]
        // y_im = [0,1] + [2,4]  = [2, 5]
        //
        // Sample 1: x_re=[0,1], x_im=[1,0]
        // y_re = [0,1]@W_re^T - [1,0]@W_im^T = [2,4] - [0,1] = [2, 3]
        // y_im = [0,1]@W_im^T + [1,0]@W_re^T = [-1,0] + [1,3] = [0, 3]

        let y_re = layer.forward(&xs)?.real.to_vec2::<f64>()?;
        let y_im = layer.forward(&xs)?.imag.to_vec2::<f64>()?;

        assert_eq!(y_re, vec![vec![2.0, 3.0], vec![2.0, 3.0]]);
        assert_eq!(y_im, vec![vec![2.0, 5.0], vec![0.0, 3.0]]);

        Ok(())
    }

    // ── Gradient / Wirtinger correctness via finite differences ──────────

    /// Compute a simple real-valued loss from complex output: L = sum(y_re^2 + y_im^2).
    fn sum_mag_sq_loss(layer: &ComplexLinear, xs: &ComplexTensor) -> Result<Tensor> {
        let ys = layer.forward(xs)?;
        let loss = ys.real.sqr()?.add(&ys.imag.sqr()?)?.sum_all()?;
        Ok(loss)
    }

    /// Build a small ComplexLinear from Vars (so Candle tracks gradients),
    /// run a forward+backward pass, and return the analytical gradient for
    /// the named variable alongside the variable's current value.
    fn analytical_grad(
        in_f: usize,
        out_f: usize,
        xs: &ComplexTensor,
        wre: &[f64],
        wim: &[f64],
        bre: &[f64],
        bim: &[f64],
        param_name: &str,
    ) -> Result<(Tensor, Tensor)> {
        let dev = Device::Cpu;
        let var_wre = Var::from_tensor(&Tensor::from_vec(wre.to_vec(), (out_f, in_f), &dev)?)?;
        let var_wim = Var::from_tensor(&Tensor::from_vec(wim.to_vec(), (out_f, in_f), &dev)?)?;
        let var_bre = Var::from_tensor(&Tensor::from_vec(bre.to_vec(), out_f, &dev)?)?;
        let var_bim = Var::from_tensor(&Tensor::from_vec(bim.to_vec(), out_f, &dev)?)?;

        let layer = ComplexLinear::from_parts(
            var_wre.as_tensor().clone(),
            var_wim.as_tensor().clone(),
            Some(var_bre.as_tensor().clone()),
            Some(var_bim.as_tensor().clone()),
        )?;

        let loss = sum_mag_sq_loss(&layer, xs)?;
        let grads = loss.backward()?;

        let (var, val) = match param_name {
            "weight_re" => (&var_wre, var_wre.as_tensor()),
            "weight_im" => (&var_wim, var_wim.as_tensor()),
            "bias_re" => (&var_bre, var_bre.as_tensor()),
            "bias_im" => (&var_bim, var_bim.as_tensor()),
            _ => return Err(candle_msg(format!("unknown param: {param_name}"))),
        };

        let grad = grads
            .get(var)
            .ok_or_else(|| candle_msg(format!("no gradient found for {param_name}")))?;

        Ok((grad.clone(), val.clone()))
    }

    /// Run backward with the inputs as Vars and return the analytical input gradient.
    fn analytical_input_grad(
        batch: usize,
        in_f: usize,
        out_f: usize,
        xre: &[f64],
        xim: &[f64],
        wre: &[f64],
        wim: &[f64],
        bre: &[f64],
        bim: &[f64],
        input_part: &str,
    ) -> Result<Tensor> {
        let dev = Device::Cpu;
        let var_xre = Var::from_tensor(&Tensor::from_vec(xre.to_vec(), (batch, in_f), &dev)?)?;
        let var_xim = Var::from_tensor(&Tensor::from_vec(xim.to_vec(), (batch, in_f), &dev)?)?;

        let layer = ComplexLinear::from_parts(
            Tensor::from_vec(wre.to_vec(), (out_f, in_f), &dev)?,
            Tensor::from_vec(wim.to_vec(), (out_f, in_f), &dev)?,
            Some(Tensor::from_vec(bre.to_vec(), out_f, &dev)?),
            Some(Tensor::from_vec(bim.to_vec(), out_f, &dev)?),
        )?;
        let xs = ComplexTensor::new(var_xre.as_tensor().clone(), var_xim.as_tensor().clone())?;

        let loss = sum_mag_sq_loss(&layer, &xs)?;
        let grads = loss.backward()?;
        let var = match input_part {
            "real" => &var_xre,
            "imag" => &var_xim,
            _ => return Err(candle_msg(format!("unknown input part: {input_part}"))),
        };

        grads
            .get(var)
            .cloned()
            .ok_or_else(|| candle_msg(format!("no gradient found for input_{input_part}")))
    }

    /// Compute the loss with one element of a parameter perturbed by `eps`.
    fn perturbed_loss(
        in_f: usize,
        out_f: usize,
        xs: &ComplexTensor,
        wre: &[f64],
        wim: &[f64],
        bre: &[f64],
        bim: &[f64],
        param_name: &str,
        flat_idx: usize,
        eps: f64,
    ) -> Result<f64> {
        let dev = Device::Cpu;

        // Clone the slices, perturb the target element.
        let mut wre = wre.to_vec();
        let mut wim = wim.to_vec();
        let mut bre = bre.to_vec();
        let mut bim = bim.to_vec();

        match param_name {
            "weight_re" => wre[flat_idx] += eps,
            "weight_im" => wim[flat_idx] += eps,
            "bias_re" => bre[flat_idx] += eps,
            "bias_im" => bim[flat_idx] += eps,
            _ => panic!("unknown param: {param_name}"),
        }

        let layer = ComplexLinear::from_parts(
            Tensor::from_vec(wre, (out_f, in_f), &dev)?,
            Tensor::from_vec(wim, (out_f, in_f), &dev)?,
            Some(Tensor::from_vec(bre, out_f, &dev)?),
            Some(Tensor::from_vec(bim, out_f, &dev)?),
        )?;

        sum_mag_sq_loss(&layer, xs)?.to_scalar::<f64>()
    }

    /// Compute the loss with one element of the input perturbed by `eps`.
    fn perturbed_input_loss(
        batch: usize,
        in_f: usize,
        out_f: usize,
        xre: &[f64],
        xim: &[f64],
        wre: &[f64],
        wim: &[f64],
        bre: &[f64],
        bim: &[f64],
        input_part: &str,
        flat_idx: usize,
        eps: f64,
    ) -> Result<f64> {
        let dev = Device::Cpu;

        let mut xre = xre.to_vec();
        let mut xim = xim.to_vec();
        match input_part {
            "real" => xre[flat_idx] += eps,
            "imag" => xim[flat_idx] += eps,
            _ => panic!("unknown input part: {input_part}"),
        }

        let xs = ComplexTensor::new(
            Tensor::from_vec(xre, (batch, in_f), &dev)?,
            Tensor::from_vec(xim, (batch, in_f), &dev)?,
        )?;
        let layer = ComplexLinear::from_parts(
            Tensor::from_vec(wre.to_vec(), (out_f, in_f), &dev)?,
            Tensor::from_vec(wim.to_vec(), (out_f, in_f), &dev)?,
            Some(Tensor::from_vec(bre.to_vec(), out_f, &dev)?),
            Some(Tensor::from_vec(bim.to_vec(), out_f, &dev)?),
        )?;

        sum_mag_sq_loss(&layer, &xs)?.to_scalar::<f64>()
    }

    #[test]
    fn test_complex_linear_wirtinger_gradients_match_finite_differences() -> Result<()> {
        let in_f = 2;
        let out_f = 2;
        let eps = 1e-6;
        let tol = 1e-5;

        // Fixed weights
        let wre = vec![0.3, -0.1, 0.5, 0.2];
        let wim = vec![-0.2, 0.4, 0.1, -0.3];
        let bre = vec![0.05, -0.1];
        let bim = vec![0.1, -0.05];

        // Input
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, -0.5, 0.3, 0.7], (2, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![-0.3f64, 0.8, 0.6, -0.4], (2, 2), &Device::Cpu)?,
        )?;

        for param_name in &["weight_re", "weight_im", "bias_re", "bias_im"] {
            let (analytical_grad, _param_val) =
                analytical_grad(in_f, out_f, &xs, &wre, &wim, &bre, &bim, param_name)?;
            let grad_flat = analytical_grad.flatten_all()?.to_vec1::<f64>()?;
            let n = grad_flat.len();

            for idx in 0..n {
                let l_plus = perturbed_loss(
                    in_f, out_f, &xs, &wre, &wim, &bre, &bim, param_name, idx, eps,
                )?;
                let l_minus = perturbed_loss(
                    in_f, out_f, &xs, &wre, &wim, &bre, &bim, param_name, idx, -eps,
                )?;
                let numerical = (l_plus - l_minus) / (2.0 * eps);
                let analytical = grad_flat[idx];

                let abs_err = (analytical - numerical).abs();
                let scale = analytical.abs().max(numerical.abs()).max(1e-8);
                let rel_err = abs_err / scale;

                assert!(
                    rel_err < tol,
                    "gradient mismatch for {param_name}[{idx}]: \
                     analytical={analytical:.8}, numerical={numerical:.8}, rel_err={rel_err:.2e}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_complex_linear_input_gradients_match_finite_differences() -> Result<()> {
        let batch = 2;
        let in_f = 2;
        let out_f = 2;
        let eps = 1e-6;
        let tol = 1e-5;

        let wre = vec![0.3, -0.1, 0.5, 0.2];
        let wim = vec![-0.2, 0.4, 0.1, -0.3];
        let bre = vec![0.05, -0.1];
        let bim = vec![0.1, -0.05];
        let xre = vec![1.0, -0.5, 0.3, 0.7];
        let xim = vec![-0.3, 0.8, 0.6, -0.4];

        for input_part in &["real", "imag"] {
            let analytical_grad = analytical_input_grad(
                batch, in_f, out_f, &xre, &xim, &wre, &wim, &bre, &bim, input_part,
            )?;
            let grad_flat = analytical_grad.flatten_all()?.to_vec1::<f64>()?;

            for idx in 0..grad_flat.len() {
                let l_plus = perturbed_input_loss(
                    batch, in_f, out_f, &xre, &xim, &wre, &wim, &bre, &bim, input_part, idx, eps,
                )?;
                let l_minus = perturbed_input_loss(
                    batch, in_f, out_f, &xre, &xim, &wre, &wim, &bre, &bim, input_part, idx, -eps,
                )?;
                let numerical = (l_plus - l_minus) / (2.0 * eps);
                let analytical = grad_flat[idx];

                let abs_err = (analytical - numerical).abs();
                let scale = analytical.abs().max(numerical.abs()).max(1e-8);
                let rel_err = abs_err / scale;

                assert!(
                    rel_err < tol,
                    "input gradient mismatch for {input_part}[{idx}]: analytical={analytical:.8}, numerical={numerical:.8}, rel_err={rel_err:.2e}"
                );
            }
        }

        Ok(())
    }

    // ── from_parts validation ───────────────────────────────────────────

    #[test]
    fn test_complex_linear_from_parts_rejects_mismatched_shapes() -> Result<()> {
        let dev = Device::Cpu;

        let err = ComplexLinear::from_parts(
            Tensor::zeros((3, 2), DType::F64, &dev)?,
            Tensor::zeros((2, 2), DType::F64, &dev)?, // wrong out_features
            None,
            None,
        );
        assert!(err.is_err());

        let err = ComplexLinear::from_parts(
            Tensor::zeros((3, 2), DType::F64, &dev)?,
            Tensor::zeros((3, 2), DType::F64, &dev)?,
            Some(Tensor::zeros(2, DType::F64, &dev)?), // wrong bias dim
            Some(Tensor::zeros(3, DType::F64, &dev)?),
        );
        assert!(err.is_err());

        let err = ComplexLinear::from_parts(
            Tensor::zeros((3, 2), DType::F64, &dev)?,
            Tensor::zeros((3, 2), DType::F64, &dev)?,
            Some(Tensor::zeros(3, DType::F64, &dev)?),
            None,
        );
        assert!(err.is_err());

        Ok(())
    }

    // ── PyTorch parity test ─────────────────────────────────────────────

    #[test]
    fn test_complex_linear_forward_matches_pytorch_reference() -> Result<()> {
        // PyTorch reference (torch 2.x):
        //
        // import torch
        // W_re = torch.tensor([[0.2, -0.1, 0.3, 0.5],
        //                      [-0.4, 0.6, 0.1, -0.2],
        //                      [0.7, 0.2, -0.5, 0.4]], dtype=torch.float64)
        // W_im = torch.tensor([[0.1, 0.3, -0.2, 0.4],
        //                      [0.5, -0.1, 0.2, -0.3],
        //                      [-0.3, 0.2, 0.6, 0.1]], dtype=torch.float64)
        // b_re = torch.tensor([0.05, -0.02, 0.1], dtype=torch.float64)
        // b_im = torch.tensor([-0.03, 0.04, -0.01], dtype=torch.float64)
        //
        // x_re = torch.tensor([[0.25, -0.5, 1.0, 0.75],
        //                      [-1.2, 0.3, 0.5, -0.8]], dtype=torch.float64)
        // x_im = torch.tensor([[0.1, -0.3, 0.6, -0.2],
        //                      [0.4, 0.7, -0.1, 0.5]], dtype=torch.float64)
        //
        // y_re = x_re @ W_re.T - x_im @ W_im.T + b_re
        // y_im = x_re @ W_im.T + x_im @ W_re.T + b_im
        //
        // print(y_re)
        // print(y_im)
        //
        // y_re = [[ 1.105, -0.73 , -0.275],
        //         [-0.94 ,  0.89 , -1.26 ]]
        // y_im = [[ 0.075,  0.07 ,  0.12 ],
        //         [-0.25 , -0.1  ,  1.3  ]]

        let tensors = tensor_map(vec![
            (
                "weight_re",
                Tensor::from_vec(
                    vec![
                        0.2f64, -0.1, 0.3, 0.5, -0.4, 0.6, 0.1, -0.2, 0.7, 0.2, -0.5, 0.4,
                    ],
                    (3, 4),
                    &Device::Cpu,
                )?,
            ),
            (
                "weight_im",
                Tensor::from_vec(
                    vec![
                        0.1f64, 0.3, -0.2, 0.4, 0.5, -0.1, 0.2, -0.3, -0.3, 0.2, 0.6, 0.1,
                    ],
                    (3, 4),
                    &Device::Cpu,
                )?,
            ),
            (
                "bias_re",
                Tensor::from_vec(vec![0.05f64, -0.02, 0.1], 3, &Device::Cpu)?,
            ),
            (
                "bias_im",
                Tensor::from_vec(vec![-0.03f64, 0.04, -0.01], 3, &Device::Cpu)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new(4, 3, vb)?;

        let xs = ComplexTensor::new(
            Tensor::from_vec(
                vec![0.25f64, -0.5, 1.0, 0.75, -1.2, 0.3, 0.5, -0.8],
                (2, 4),
                &Device::Cpu,
            )?,
            Tensor::from_vec(
                vec![0.1f64, -0.3, 0.6, -0.2, 0.4, 0.7, -0.1, 0.5],
                (2, 4),
                &Device::Cpu,
            )?,
        )?;

        let ys = layer.forward(&xs)?;
        let y_re = ys.real.to_vec2::<f64>()?;
        let y_im = ys.imag.to_vec2::<f64>()?;

        let tol = 1e-12;
        for (row_a, row_e) in y_re
            .iter()
            .zip(vec![vec![1.105, -0.73, -0.275], vec![-0.94, 0.89, -1.26]].iter())
        {
            for (a, e) in row_a.iter().zip(row_e.iter()) {
                assert!((a - e).abs() < tol, "y_re mismatch: got {a}, expected {e}");
            }
        }
        for (row_a, row_e) in y_im
            .iter()
            .zip(vec![vec![0.075, 0.07, 0.12], vec![-0.25, -0.1, 1.3]].iter())
        {
            for (a, e) in row_a.iter().zip(row_e.iter()) {
                assert!((a - e).abs() < tol, "y_im mismatch: got {a}, expected {e}");
            }
        }

        Ok(())
    }

    // ── ComplexInit variance tests (Trabelsi et al. 2018) ──────────────

    /// Helper: compute empirical variance of a tensor's elements.
    fn empirical_variance(t: &Tensor) -> f64 {
        let flat: Vec<f64> = t.flatten_all().unwrap().to_vec1().unwrap();
        let n = flat.len() as f64;
        let mean = flat.iter().sum::<f64>() / n;
        flat.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
    }

    #[test]
    fn complex_he_init_has_correct_variance() -> Result<()> {
        let fan_in: usize = 128;
        let fan_out: usize = 64;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new_with_init(fan_in, fan_out, ComplexInit::He, vb)?;

        // Expected per-part variance: σ² = 1 / fan_in
        let expected = 1.0 / fan_in as f64;
        let var_re = empirical_variance(&layer.weight_re);
        let var_im = empirical_variance(&layer.weight_im);

        // With 128×64 = 8192 elements, relative tolerance of 15% is safe.
        let tol = 0.15;
        assert!(
            (var_re - expected).abs() / expected < tol,
            "He weight_re variance {var_re:.6} too far from expected {expected:.6}"
        );
        assert!(
            (var_im - expected).abs() / expected < tol,
            "He weight_im variance {var_im:.6} too far from expected {expected:.6}"
        );
        Ok(())
    }

    #[test]
    fn complex_glorot_init_has_correct_variance() -> Result<()> {
        let fan_in: usize = 128;
        let fan_out: usize = 64;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let layer = ComplexLinear::new_with_init(fan_in, fan_out, ComplexInit::Glorot, vb)?;

        // Expected per-part variance: σ² = 1 / (fan_in + fan_out)
        let expected = 1.0 / (fan_in + fan_out) as f64;
        let var_re = empirical_variance(&layer.weight_re);
        let var_im = empirical_variance(&layer.weight_im);

        let tol = 0.15;
        assert!(
            (var_re - expected).abs() / expected < tol,
            "Glorot weight_re variance {var_re:.6} too far from expected {expected:.6}"
        );
        assert!(
            (var_im - expected).abs() / expected < tol,
            "Glorot weight_im variance {var_im:.6} too far from expected {expected:.6}"
        );
        Ok(())
    }

    #[test]
    fn complex_init_default_is_he() {
        assert_eq!(ComplexInit::default(), ComplexInit::He);
    }
}
