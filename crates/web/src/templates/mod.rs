//! Askama template structs and custom filters.
//!
//! Template HTML files live in `crates/web/templates/` and are compiled
//! at build time by Askama. Each route module defines its own template
//! structs alongside its handlers.

pub mod filters;
