//! Grid structure detection for 2D permittivity domains.

use std::collections::HashSet;

/// A detected regular grid in (ε', ε'') space.
#[derive(Debug, Clone, PartialEq)]
pub struct GridStructure {
    /// Unique ε' axis values.
    pub eps_prime_values: Vec<f64>,
    /// Unique ε'' axis values.
    pub eps_double_prime_values: Vec<f64>,
    /// Grid dimensions (nx, ny).
    pub dims: (usize, usize),
}

/// Detect whether data forms a regular grid in (ε', ε'') space.
#[must_use]
pub fn detect_grid_structure(eps_prime: &[f64], eps_double_prime: &[f64]) -> Option<GridStructure> {
    if eps_prime.len() != eps_double_prime.len() || eps_prime.is_empty() {
        return None;
    }
    if eps_prime.iter().any(|value| !value.is_finite())
        || eps_double_prime.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let unique_real = unique_sorted(eps_prime);
    let unique_imag = unique_sorted(eps_double_prime);
    if unique_real.len() * unique_imag.len() != eps_prime.len() {
        return None;
    }

    let pair_set: HashSet<(u64, u64)> = eps_prime
        .iter()
        .zip(eps_double_prime.iter())
        .map(|(real, imag)| (real.to_bits(), imag.to_bits()))
        .collect();
    if pair_set.len() != eps_prime.len() {
        return None;
    }

    let is_complete_grid = unique_imag.iter().all(|imag| {
        unique_real
            .iter()
            .all(|real| pair_set.contains(&(real.to_bits(), imag.to_bits())))
    });
    if !is_complete_grid {
        return None;
    }

    Some(GridStructure {
        dims: (unique_real.len(), unique_imag.len()),
        eps_prime_values: unique_real,
        eps_double_prime_values: unique_imag,
    })
}

fn unique_sorted(values: &[f64]) -> Vec<f64> {
    let mut unique = values.to_vec();
    unique.sort_by(f64::total_cmp);
    unique.dedup_by(|left, right| left.total_cmp(right).is_eq());
    unique
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_2x3_grid() {
        let eps_prime = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let eps_double = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
        let grid = detect_grid_structure(&eps_prime, &eps_double).unwrap();
        assert_eq!(grid.dims, (2, 3));
    }

    #[test]
    fn rejects_non_grid() {
        let eps_prime = vec![1.0, 2.0, 3.0];
        let eps_double = vec![0.1, 0.2, 0.3];
        // 3 unique real * 3 unique imag = 9 != 3, so not a grid
        assert!(detect_grid_structure(&eps_prime, &eps_double).is_none());
    }

    #[test]
    fn rejects_empty() {
        assert!(detect_grid_structure(&[], &[]).is_none());
    }

    #[test]
    fn rejects_nan() {
        assert!(detect_grid_structure(&[f64::NAN], &[1.0]).is_none());
    }
}
