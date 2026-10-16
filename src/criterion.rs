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

    /// Whether [`Criterion::heldout_loss`] is implemented. Criteria that
    /// return `false` fall back to in-sample scoring even when
    /// [`crate::cross_validation::Scoring::Honest`] is requested.
    fn supports_heldout(&self) -> bool {
        false
    }

    /// **Total** loss over the `val` targets of the model this criterion fits
    /// to the `train` targets (class proportions, mean, or median).
    ///
    /// This is the out-of-sample counterpart of `impurity`: when `train` and
    /// `val` are the same sample, `heldout_loss(t, t) == t.len() * impurity(t)`
    /// (exactly for Gini/Brier and variance/MSE; for entropy up to the
    /// log-loss smoothing). Unlike in-sample impurity decrease, a split scored
    /// this way **can lose** — which is what makes `score <= 0` a meaningful
    /// stopping rule. `train` must be non-empty (callers fall back to the
    /// parent's training targets for an empty child). Returns `None` when the
    /// criterion does not support held-out scoring.
    fn heldout_loss(&self, train: &[Self::Target], val: &[Self::Target]) -> Option<f64> {
        let _ = (train, val);
        None
    }

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

/// Additive smoothing applied to training-fold class counts when scoring
/// held-out log-loss (Jeffreys prior), so a class unseen in the training fold
/// costs a large but finite amount instead of infinity.
pub const LOGLOSS_ALPHA: f64 = 0.5;

/// Total Brier loss `Σ_val Σ_j (1[j = y] − p_j)²` of predicting the class
/// proportions of the `train` counts for samples with class counts `val`.
///
/// This is the held-out analogue of Gini: with `train == val` it equals
/// `n · gini`. Returns `NaN` when `train` is empty (callers must fall back).
pub fn brier_loss_counts(train: &[u64], val: &[u64]) -> f64 {
    let nv: u64 = val.iter().sum();
    if nv == 0 {
        return 0.0;
    }
    let nt: u64 = train.iter().sum();
    if nt == 0 {
        return f64::NAN;
    }
    let ntf = nt as f64;
    let mut sum_sq = 0.0;
    let mut cross = 0.0;
    for (c, &t) in train.iter().enumerate() {
        let p = t as f64 / ntf;
        sum_sq += p * p;
        let v = val.get(c).copied().unwrap_or(0) as f64;
        cross += v * p;
    }
    nv as f64 * (1.0 + sum_sq) - 2.0 * cross
}

/// Total log-loss in bits `−Σ_val log₂ p_y` of predicting the
/// `alpha`-smoothed class proportions of `train` for samples with class
/// counts `val`. The held-out analogue of entropy (`n · entropy` in-sample
/// when `alpha == 0`).
pub fn log_loss_counts(train: &[u64], val: &[u64], alpha: f64) -> f64 {
    let nv: u64 = val.iter().sum();
    if nv == 0 {
        return 0.0;
    }
    let c = train.len().max(val.len());
    let nt: u64 = train.iter().sum();
    let denom = nt as f64 + alpha * c as f64;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let mut loss = 0.0;
    for cl in 0..c {
        let v = val.get(cl).copied().unwrap_or(0);
        if v > 0 {
            let t = train.get(cl).copied().unwrap_or(0) as f64;
            let p = (t + alpha) / denom;
            if p <= 0.0 {
                return f64::INFINITY;
            }
            loss -= v as f64 * p.log2();
        }
    }
    loss
}

/// Total squared error `Σ (y − m)²` of predicting `m` for samples with
/// moments `n`, `Σy`, `Σy²`. Clamped at zero against cancellation.
pub fn sq_loss_moments(n: f64, sum: f64, sumsq: f64, m: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    (sumsq - 2.0 * m * sum + n * m * m).max(0.0)
}

fn median_of(targets: &[f64]) -> f64 {
    let n = targets.len();
    let mut v = targets.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
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

    fn supports_heldout(&self) -> bool {
        true
    }

    fn heldout_loss(&self, train: &[usize], val: &[usize]) -> Option<f64> {
        let (tc, _) = class_counts(train, self.n_classes);
        let (vc, _) = class_counts(val, self.n_classes);
        Some(brier_loss_counts(&tc, &vc))
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

    fn supports_heldout(&self) -> bool {
        true
    }

    fn heldout_loss(&self, train: &[usize], val: &[usize]) -> Option<f64> {
        let (tc, _) = class_counts(train, self.n_classes);
        let (vc, _) = class_counts(val, self.n_classes);
        Some(log_loss_counts(&tc, &vc, LOGLOSS_ALPHA))
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

    fn supports_heldout(&self) -> bool {
        true
    }

    fn heldout_loss(&self, train: &[f64], val: &[f64]) -> Option<f64> {
        if train.is_empty() {
            return Some(f64::NAN);
        }
        let m = train.iter().sum::<f64>() / train.len() as f64;
        Some(val.iter().map(|y| (y - m) * (y - m)).sum())
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

    fn supports_heldout(&self) -> bool {
        true
    }

    fn heldout_loss(&self, train: &[f64], val: &[f64]) -> Option<f64> {
        Variance.heldout_loss(train, val)
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
        let median = median_of(targets);
        targets.iter().map(|x| (x - median).abs()).sum::<f64>() / n as f64
    }

    fn supports_heldout(&self) -> bool {
        true
    }

    fn heldout_loss(&self, train: &[f64], val: &[f64]) -> Option<f64> {
        if train.is_empty() {
            return Some(f64::NAN);
        }
        let m = median_of(train);
        Some(val.iter().map(|y| (y - m).abs()).sum())
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
    fn heldout_in_sample_matches_n_times_impurity() {
        let t = vec![0usize, 0, 1, 1, 1, 2, 0];
        let n = t.len() as f64;
        let g = Gini::new(3);
        assert!(approx(g.heldout_loss(&t, &t).unwrap(), n * g.impurity(&t)));
        let y = vec![0.5, 2.0, -1.0, 3.5];
        let ny = y.len() as f64;
        assert!(approx(Variance.heldout_loss(&y, &y).unwrap(), ny * Variance.impurity(&y)));
        assert!(approx(Mae.heldout_loss(&y, &y).unwrap(), ny * Mae.impurity(&y)));
    }

    #[test]
    fn brier_counts_known_value() {
        // train p = (0.5, 0.5); each val sample costs 0.25 + 0.25 = 0.5.
        assert!(approx(brier_loss_counts(&[2, 2], &[3, 1]), 2.0));
        // In-sample equals n * gini.
        let c = [7u64, 3, 5];
        assert!(approx(brier_loss_counts(&c, &c), 15.0 * gini_from_counts(&c, 15)));
        assert!(brier_loss_counts(&[0, 0], &[1, 0]).is_nan());
        assert!(approx(brier_loss_counts(&[0, 0], &[0, 0]), 0.0));
    }

    #[test]
    fn log_loss_counts_unsmoothed_matches_entropy() {
        let c = [7u64, 3, 5];
        assert!(approx(log_loss_counts(&c, &c, 0.0), 15.0 * entropy_from_counts(&c, 15)));
        // Smoothing keeps unseen classes finite.
        assert!(log_loss_counts(&[4, 0], &[0, 1], LOGLOSS_ALPHA).is_finite());
    }

    #[test]
    fn heldout_can_be_worse_than_parent() {
        // Train says class 0, val is all class 1: child loss exceeds parent loss.
        let parent = brier_loss_counts(&[2, 2], &[0, 2]);
        let child = brier_loss_counts(&[2, 0], &[0, 2]);
        assert!(child > parent);
    }

    #[test]
    fn empty_is_zero_everywhere() {
        assert!(approx(Gini::new(2).impurity(&[]), 0.0));
        assert!(approx(Variance.impurity(&[]), 0.0));
        assert!(approx(Mae.impurity(&[]), 0.0));
    }
}
