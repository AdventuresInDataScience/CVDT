//! Impurity criteria.
//!
//! This is the only module that knows how to turn a set of targets into an
//! impurity number. Everything above it (cross-validation, aggregation, the
//! tree) treats a criterion as an opaque `impurity(&targets) -> f64`.
//!
//! A single [`Criterion`] trait covers both tasks via an associated target
//! type: classification criteria operate on class-id `usize` targets,
//! regression criteria on `f64` targets. The trait is object safe, so the tree
//! can hold a `Box<dyn Criterion<Target = _>>` and swap criteria at runtime.

/// Something that scores the impurity of a slice of targets.
///
/// Lower impurity means a purer (better) node.
pub trait Criterion: Send + Sync {
    /// Target type: `usize` class ids for classification, `f64` for regression.
    type Target: Copy;

    /// Impurity of the given targets. An empty slice has impurity `0`.
    fn impurity(&self, targets: &[Self::Target]) -> f64;

    /// Human-readable name, useful for debugging and reporting.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Classification criteria
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Statistic-based impurity helpers
// ---------------------------------------------------------------------------
//
// These pure functions compute impurity directly from sufficient statistics
// (class counts, or count/sum/sum-of-squares moments) rather than from a slice
// of targets. They are the single source of truth shared by the exact path
// (which materialises target slices) and the histogram "fast" path (which
// accumulates statistics without ever building child slices).

/// Gini impurity `1 - Σ pᵢ²` from raw class `counts` with denominator `n`.
pub fn gini_from_counts(counts: &[u64], n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut sum_sq = 0.0;
    for &c in counts {
        let p = c as f64 / nf;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

/// Shannon entropy in bits `-Σ pᵢ log₂ pᵢ` from raw class `counts`.
pub fn entropy_from_counts(counts: &[u64], n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut ent = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / nf;
            ent -= p * p.log2();
        }
    }
    ent
}

/// Population variance from the moments `n`, `Σx` and `Σx²`.
///
/// Uses `Var = E[x²] - E[x]²`, clamped at zero to absorb tiny negative values
/// produced by floating-point cancellation on near-constant data.
pub fn variance_from_moments(n: f64, sum: f64, sumsq: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let mean = sum / n;
    (sumsq / n - mean * mean).max(0.0)
}

fn class_counts(targets: &[usize], n_classes: usize) -> (Vec<u64>, f64) {
    let mut counts = vec![0u64; n_classes];
    for &t in targets {
        // Out-of-range labels are ignored rather than panicking; callers are
        // expected to pass ids in `0..n_classes`.
        if t < counts.len() {
            counts[t] += 1;
        }
    }
    (counts, targets.len() as f64)
}

/// Gini impurity: `1 - Σ pᵢ²`.
#[derive(Clone, Debug)]
pub struct Gini {
    /// Number of classes.
    pub n_classes: usize,
}

impl Gini {
    /// Create a Gini criterion for `n_classes` classes.
    pub fn new(n_classes: usize) -> Self {
        Gini { n_classes }
    }
}

impl Criterion for Gini {
    type Target = usize;

    fn impurity(&self, targets: &[usize]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let (counts, n) = class_counts(targets, self.n_classes);
        gini_from_counts(&counts, n as u64)
    }

    fn name(&self) -> &'static str {
        "gini"
    }
}

/// Shannon entropy in bits: `-Σ pᵢ log₂ pᵢ`.
#[derive(Clone, Debug)]
pub struct Entropy {
    /// Number of classes.
    pub n_classes: usize,
}

impl Entropy {
    /// Create an entropy criterion for `n_classes` classes.
    pub fn new(n_classes: usize) -> Self {
        Entropy { n_classes }
    }
}

impl Criterion for Entropy {
    type Target = usize;

    fn impurity(&self, targets: &[usize]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let (counts, n) = class_counts(targets, self.n_classes);
        entropy_from_counts(&counts, n as u64)
    }

    fn name(&self) -> &'static str {
        "entropy"
    }
}

// ---------------------------------------------------------------------------
// Regression criteria
// ---------------------------------------------------------------------------

/// Variance around the mean (population variance).
///
/// This equals the mean squared error of predicting the node mean, so
/// [`Variance`] and [`Mse`] are numerically identical; both names are provided
/// for clarity of intent.
#[derive(Clone, Debug, Default)]
pub struct Variance;

impl Criterion for Variance {
    type Target = f64;

    fn impurity(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let n = targets.len() as f64;
        let sum: f64 = targets.iter().sum();
        let sumsq: f64 = targets.iter().map(|x| x * x).sum();
        variance_from_moments(n, sum, sumsq)
    }

    fn name(&self) -> &'static str {
        "variance"
    }
}

/// Mean squared error of predicting the node mean (identical to [`Variance`]).
#[derive(Clone, Debug, Default)]
pub struct Mse;

impl Criterion for Mse {
    type Target = f64;

    fn impurity(&self, targets: &[f64]) -> f64 {
        Variance.impurity(targets)
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}

/// Mean absolute error of predicting the node median.
#[derive(Clone, Debug, Default)]
pub struct Mae;

impl Criterion for Mae {
    type Target = f64;

    fn impurity(&self, targets: &[f64]) -> f64 {
        let n = targets.len();
        if n == 0 {
            return 0.0;
        }
        let mut v = targets.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let median = if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        };
        targets.iter().map(|x| (x - median).abs()).sum::<f64>() / n as f64
    }

    fn name(&self) -> &'static str {
        "mae"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn gini_pure_is_zero() {
        assert!(approx(Gini::new(2).impurity(&[0, 0, 0]), 0.0));
    }

    #[test]
    fn gini_balanced_binary_is_half() {
        assert!(approx(Gini::new(2).impurity(&[0, 1, 0, 1]), 0.5));
    }

    #[test]
    fn entropy_balanced_binary_is_one_bit() {
        assert!(approx(Entropy::new(2).impurity(&[0, 1, 0, 1]), 1.0));
    }

    #[test]
    fn entropy_pure_is_zero() {
        assert!(approx(Entropy::new(3).impurity(&[2, 2, 2, 2]), 0.0));
    }

    #[test]
    fn variance_matches_mse() {
        let v = vec![0.0, 2.0, 4.0, 6.0];
        assert!(approx(Variance.impurity(&v), Mse.impurity(&v)));
    }

    #[test]
    fn variance_known_value() {
        // mean = 1, deviations 1 and 1 -> variance 1
        assert!(approx(Variance.impurity(&[0.0, 2.0]), 1.0));
    }

    #[test]
    fn mae_known_value() {
        // median = 1, abs deviations 1 and 1 -> mae 1
        assert!(approx(Mae.impurity(&[0.0, 2.0]), 1.0));
    }

    #[test]
    fn empty_is_zero_everywhere() {
        assert!(approx(Gini::new(2).impurity(&[]), 0.0));
        assert!(approx(Variance.impurity(&[]), 0.0));
        assert!(approx(Mae.impurity(&[]), 0.0));
    }
}
