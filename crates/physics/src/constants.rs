//! Physical constants for WR-90 waveguide and NRW forward model.

/// WR-90 sample thickness (m). Matches
/// [`sparam_data::generation::DataGenerationConfig::default`] so a
/// train/HPO run on a default-generated dataset round-trips the same
/// physics constants into the NRW forward model.
pub const DEFAULT_SAMPLE_THICKNESS_M: f64 = 1.5e-3;

/// WR-90 waveguide width (m) — matches the generator default.
pub const DEFAULT_WAVEGUIDE_WIDTH_M: f64 = 22.86e-3;

/// Operating frequency (Hz) used to evaluate the NRW forward model.
/// Matches the generator default; all dataset samples share this single
/// frequency by construction.
pub const DEFAULT_OPERATING_FREQUENCY_HZ: f64 = 8.2e9;
