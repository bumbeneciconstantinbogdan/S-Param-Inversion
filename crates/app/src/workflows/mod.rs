//! End-to-end application workflows (train, evaluate, HPO, benchmark).

pub mod benchmark;
pub mod config;
pub mod evaluate;
pub mod generate;
pub mod hpo;
pub mod hpo_form;
pub mod infer;
pub mod internal;
pub mod nrw_diagnostics;
pub mod stacked_training;
pub mod train;
pub mod train_form;

// No re-export hub. Each type lives at one canonical path:
// `sparam_app::workflows::<submodule>::<Item>`. See submodule modules
// above for the surface each one publishes.

