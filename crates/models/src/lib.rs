//! Neural network architectures for S-parameter inversion.

pub mod activation;
pub mod complex_activation;
pub mod complex_linear;
pub mod complex_mlp;
pub mod dispatch;
pub mod mlp;

pub use activation::Activation;
pub use complex_activation::{
    CSwishPhase, ComplexActivation, HybridCardioidGelu, Lpma, ModReLU, Worelu,
};
pub use complex_linear::{ComplexInit, ComplexLinear};
pub use complex_mlp::{ComplexMLPConfig, ComplexMLPRegressor, ComplexNormChoice};
pub use dispatch::{MlpModel, ModelType};
pub use mlp::{MLPConfig, MLPRegressor, Normalization};
