//! Single-point S-parameter scalar bundle.
//!
//! The four-tuple `(s11_real, s11_imag, s21_real, s21_imag)` shows up
//! as a data clump everywhere a single sample crosses a function or
//! HTTP boundary: the web `/infer` form fields, the
//! `NeuralNetInferRequest`, the inverse-NRW helper, the evaluate-grid
//! click handler, etc.  This struct gives that group a name without
//! adding any runtime cost (`Copy`, all `f64`).
//!
//! Lives in `sparam-core` so every higher-level crate (`physics`,
//! `app`, `web`) can use it without circular deps.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S11S21 {
    pub s11_real: f64,
    pub s11_imag: f64,
    pub s21_real: f64,
    pub s21_imag: f64,
}

impl S11S21 {
    pub const fn new(s11_real: f64, s11_imag: f64, s21_real: f64, s21_imag: f64) -> Self {
        Self { s11_real, s11_imag, s21_real, s21_imag }
    }

    /// Pack as `(Re(S₁₁), Im(S₁₁), Re(S₂₁), Im(S₂₁))` — the canonical
    /// row order the Real-MLP feature tensor uses.
    pub const fn as_real_row(&self) -> [f64; 4] {
        [self.s11_real, self.s11_imag, self.s21_real, self.s21_imag]
    }

    /// Pack as `(Re(S₁₁), Re(S₂₁), Im(S₁₁), Im(S₂₁))` — the row order
    /// the Complex-MLP packed feature tensor uses (real half then
    /// imaginary half), matching `pack_complex_features`.
    pub const fn as_complex_row(&self) -> [f64; 4] {
        [self.s11_real, self.s21_real, self.s11_imag, self.s21_imag]
    }
}
