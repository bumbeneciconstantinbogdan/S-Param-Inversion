//! Physical constants used by the NRW algorithm and related diagnostics.

use std::f64::consts::PI;

/// Vacuum permittivity (F/m).
pub const EPSILON_0: f64 = 8.854_187_82e-12;

/// Vacuum permeability (H/m).
pub const MU_0: f64 = 4.0 * PI * 1e-7;
