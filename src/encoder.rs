//! Feature encoding.
//!
//! The only stateful encoding is quantile binning of continuous features. The
//! bin *edges* are computed from a set of values (in cross-validation, that set
//! is always the **training fold**, which is what prevents leakage) and a value
//! is then mapped to an integer bin id.
//!
//! Categorical features are already integer ids and need no fitting, so they do
//! not appear here — the tree routes them by direct id equality.
//!
//! Bin convention: for `E = edges.len()` internal cut points there are `E + 1`
//! bins numbered `0..=E`. The bin id of a value is the number of edges that are
//! `<= value`. Equivalently value `v` lands in bin `b` iff
//! `edges[b-1] <= v < edges[b]` (with `-inf`/`+inf` for the open ends).

/// Sentinel bin id returned for missing (non-finite) continuous values.
pub const MISSING_BIN: u32 = u32::MAX;

/// Compute up to `n_bins - 1` quantile cut points from `values`.
///
/// Non-finite values are ignored. The returned edges are sorted ascending and
/// may contain duplicates when the underlying distribution is degenerate; that
/// is fine — duplicated edges simply leave some bins unreachable, which the
/// cross-validation engine handles by counting empty children as unsuccessful
/// folds.
pub fn quantile_edges(values: &[f64], n_bins: usize) -> Vec<f64> {
    if n_bins <= 1 {
        return Vec::new();
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Vec::new();
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len();
    let mut edges = Vec::with_capacity(n_bins - 1);
    for i in 1..n_bins {
        // Approximate the i/n_bins quantile by index; equal-frequency binning.
        let idx = ((i * m) / n_bins).min(m - 1);
        edges.push(v[idx]);
    }
    edges
}

/// Map a value to its bin id given sorted `edges`.
///
/// Returns [`MISSING_BIN`] for non-finite values.
pub fn bin_of(edges: &[f64], value: f64) -> u32 {
    if !value.is_finite() {
        return MISSING_BIN;
    }
    // Number of edges that are <= value. `edges` is sorted ascending, so the
    // predicate is true for a prefix and `partition_point` is valid.
    edges.partition_point(|&e| e <= value) as u32
}

/// Inclusive-lower / exclusive-upper interval for bin `b` given `edges`.
///
/// `None` bounds denote the open ends (`-inf` for the lowest bin, `+inf` for
/// the highest). Returns `None` if `b` is out of range for these edges.
pub fn bin_bounds(edges: &[f64], b: usize) -> Option<(Option<f64>, Option<f64>)> {
    let e = edges.len();
    if b > e {
        return None;
    }
    let lower = if b == 0 { None } else { Some(edges[b - 1]) };
    let upper = if b == e { None } else { Some(edges[b]) };
    Some((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_sorted_and_sized() {
        let vals: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let edges = quantile_edges(&vals, 4);
        assert_eq!(edges.len(), 3);
        assert!(edges.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn bin_of_is_monotonic() {
        let edges = vec![10.0, 20.0];
        assert_eq!(bin_of(&edges, 5.0), 0);
        assert_eq!(bin_of(&edges, 10.0), 1); // edge is <= value -> next bin
        assert_eq!(bin_of(&edges, 15.0), 1);
        assert_eq!(bin_of(&edges, 20.0), 2);
        assert_eq!(bin_of(&edges, 99.0), 2);
    }

    #[test]
    fn missing_values_are_flagged() {
        let edges = vec![1.0, 2.0];
        assert_eq!(bin_of(&edges, f64::NAN), MISSING_BIN);
        assert_eq!(bin_of(&edges, f64::INFINITY), MISSING_BIN);
    }

    #[test]
    fn bounds_match_bin_of() {
        let vals: Vec<f64> = (0..40).map(|x| x as f64).collect();
        let edges = quantile_edges(&vals, 5);
        for &v in &vals {
            let b = bin_of(&edges, v) as usize;
            let (lo, hi) = bin_bounds(&edges, b).unwrap();
            assert!(lo.map_or(true, |l| v >= l));
            assert!(hi.map_or(true, |h| v < h));
        }
    }

    #[test]
    fn single_bin_has_no_edges() {
        assert!(quantile_edges(&[1.0, 2.0, 3.0], 1).is_empty());
    }

    #[test]
    fn empty_input_yields_no_edges() {
        assert!(quantile_edges(&[], 4).is_empty());
    }
}
