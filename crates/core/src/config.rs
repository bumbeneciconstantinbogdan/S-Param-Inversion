//! Lightweight string normalization helpers for config parsing.

/// Normalizes a configuration name: trim, lowercase, replace `-` and spaces with `_`.
#[must_use]
pub fn normalize_config_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_mixed_input() {
        assert_eq!(normalize_config_name("  Cosine-Annealing "), "cosine_annealing");
        assert_eq!(normalize_config_name("AdamW"), "adamw");
        assert_eq!(normalize_config_name("smooth-l1"), "smooth_l1");
    }
}
