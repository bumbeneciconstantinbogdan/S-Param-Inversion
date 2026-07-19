//! Nicolson-Ross-Weir (NRW) forward and inverse models for waveguide measurements.

use candle_core::{Device, Result, Tensor};

pub mod constants;
pub mod nrw;

pub use constants::{
    DEFAULT_OPERATING_FREQUENCY_HZ, DEFAULT_SAMPLE_THICKNESS_M, DEFAULT_WAVEGUIDE_WIDTH_M,
};
pub use nrw::WaveguideConfig;

/// Captured context required to compute physics-informed loss
/// inside a per-batch closure. Built once per training run and
/// passed to loss functions, which reuse the same waveguide
/// geometry and frequency tensor for every batch.
#[derive(Debug)]
pub struct PhysicsContext {
    pub waveguide: WaveguideConfig,
    pub frequencies: Tensor,
}

impl PhysicsContext {
    /// WR-90 @ 8.2 GHz — the defaults that
    /// [`sparam_data::generation::DataGenerationConfig::default`]
    /// uses for synthetic dataset generation. A training run on a
    /// default-generated dataset therefore reconstructs the same
    /// S-parameters the dataset was built from.
    pub fn default_wr90(device: &Device) -> Result<Self> {
        let waveguide = WaveguideConfig::new(
            DEFAULT_SAMPLE_THICKNESS_M,
            DEFAULT_WAVEGUIDE_WIDTH_M,
        )?;
        let frequencies = Tensor::from_vec(
            vec![DEFAULT_OPERATING_FREQUENCY_HZ],
            (1,),
            device,
        )?;
        Ok(Self {
            waveguide,
            frequencies,
        })
    }
}
