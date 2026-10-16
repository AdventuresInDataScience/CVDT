//! Score aggregation.
//!
//! Converts a candidate's [`FoldStats`] into a single ranking number where
//! **higher is better**. This is the knob that lets the same tree be optimistic
//! (plain mean), robust to outlier folds (median / trimmed mean), or explicitly
//! risk-averse (`mean - λ·std`, a signal-to-noise ratio, a lower quantile, or
//! their variance-moderated versions that pool spread across the candidates
//! at a node).
//!
//! Fold scores share training data, so they are correlated: `std` describes
//! fold-to-fold spread but understates the uncertainty of the mean, and none
//! of these aggregators is a calibrated confidence bound.
//!
//! Aggregation is completely independent of how the fold scores were produced.

use crate::cross_validation::FoldStats;

/// Turns per-fold statistics into one comparable score (higher = better).
pub trait Aggregator: Send + Sync {
    /// Aggregate the statistics. Must return [`f64::NEG_INFINITY`] (or another
    /// clearly-losing value) when there are no successful folds.
    fn aggregate(&self, stats: &FoldStats) -> f64;

    /// Aggregate **all** candidates of one node at once.
    ///
    /// The tree calls this after every candidate at a node has been scored,
    /// so aggregators can borrow strength across candidates (see
    /// [`ModeratedSignalToNoise`]). The default scores each candidate
    /// independently with [`Aggregator::aggregate`].
    fn aggregate_node(&self, stats: &[&FoldStats]) -> Vec<f64> {
        stats.iter().map(|s| self.aggregate(s)).collect()
    }

    /// Human-readable name.
    fn name(&self) -> &'static str;
}

/// Linear-interpolated `q`-quantile of `v` (which is sorted in place).
pub(crate) fn quantile_of(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = q.max(0.0).min(1.0);
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let w = pos - lo as f64;
    v[lo] * (1.0 - w) + v[hi] * w
}

/// Arithmetic mean of the fold scores.
#[derive(Clone, Debug, Default)]
pub struct Mean;

impl Aggregator for Mean {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        if stats.n_success == 0 {
            f64::NEG_INFINITY
        } else {
            stats.mean
        }
    }
    fn name(&self) -> &'static str {
        "mean"
    }
}

/// Median of the fold scores — robust to a single bad fold.
#[derive(Clone, Debug, Default)]
pub struct Median;

impl Aggregator for Median {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        if stats.n_success == 0 {
            f64::NEG_INFINITY
        } else {
            stats.median
        }
    }
    fn name(&self) -> &'static str {
        "median"
    }
}

/// Symmetric trimmed mean: drop a fraction of the highest and lowest scores.
#[derive(Clone, Debug)]
pub struct TrimmedMean {
    /// Fraction to trim from *each* end, in `[0, 0.5)`.
    pub frac: f64,
}

impl Aggregator for TrimmedMean {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        if stats.scores.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut v = stats.scores.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        let len = v.len();
        // Never trim away everything: cap k so at least one element remains.
        let mut k = ((len as f64) * self.frac).floor() as usize;
        k = k.min((len.saturating_sub(1)) / 2);
        let slice = &v[k..len - k];
        if slice.is_empty() {
            return f64::NEG_INFINITY;
        }
        slice.iter().sum::<f64>() / slice.len() as f64
    }
    fn name(&self) -> &'static str {
        "trimmed_mean"
    }
}

/// Signal-to-noise ratio `mean / (std + eps)` — rewards consistent gains.
#[derive(Clone, Debug)]
pub struct SignalToNoise {
    /// Small constant to avoid division by zero.
    pub eps: f64,
}

impl Default for SignalToNoise {
    fn default() -> Self {
        SignalToNoise { eps: 1e-12 }
    }
}

impl Aggregator for SignalToNoise {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        if stats.n_success == 0 {
            f64::NEG_INFINITY
        } else {
            stats.mean / (stats.std + self.eps)
        }
    }
    fn name(&self) -> &'static str {
        "signal_to_noise"
    }
}

/// Risk-adjusted score `mean - λ·std` — penalises volatile splits.
#[derive(Clone, Debug)]
pub struct MeanMinusLambdaStd {
    /// Penalty weight on the standard deviation.
    pub lambda: f64,
}

impl Aggregator for MeanMinusLambdaStd {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        if stats.n_success == 0 {
            f64::NEG_INFINITY
        } else {
            stats.mean - self.lambda * stats.std
        }
    }
    fn name(&self) -> &'static str {
        "mean_minus_lambda_std"
    }
}

/// Lower (or any) quantile of the fold scores — a pessimistic, risk-averse
/// score: "the gain this split achieves in its worse folds". It is a quantile
/// of the per-fold score distribution, **not** a confidence bound on the
/// split's mean gain: K-fold scores share training data and are correlated,
/// and more folds (or `TreeParams::cv_repeats`) make the quantile more stable
/// without making it a calibrated bound.
#[derive(Clone, Debug)]
pub struct Quantile {
    /// Quantile in `[0, 1]`; e.g. `0.25` for the lower quartile.
    pub q: f64,
}

impl Aggregator for Quantile {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        let mut v = stats.scores.clone();
        quantile_of(&mut v, self.q)
    }
    fn name(&self) -> &'static str {
        "quantile"
    }
}

/// Pooled prior variance for the moderated aggregators: the median sample
/// variance of every candidate at the node with at least two folds.
fn pooled_variance(stats: &[&FoldStats]) -> Option<f64> {
    let mut vars: Vec<f64> = stats
        .iter()
        .filter(|s| s.n_success >= 2)
        .map(|s| s.var())
        .filter(|v| v.is_finite())
        .collect();
    if vars.is_empty() {
        None
    } else {
        Some(quantile_of(&mut vars, 0.5))
    }
}

/// Empirical-Bayes moderated variance (limma-style):
/// `(d0·s0² + (n−1)·s²) / (d0 + n − 1)`. With one fold it returns the prior;
/// with no prior it returns the raw variance (`NaN` for n < 2).
fn moderated_var(st: &FoldStats, prior: Option<f64>, d0: f64) -> f64 {
    let n = st.n_success as f64;
    let raw_ss = if st.n_success >= 2 {
        (n - 1.0) * st.var()
    } else {
        0.0
    };
    match prior {
        Some(s0) if d0 > 0.0 => (d0 * s0 + raw_ss) / (d0 + n - 1.0),
        _ => st.var(),
    }
}

/// Moderated signal-to-noise `mean / sqrt(moderated var)`.
///
/// Five fold scores give a very noisy std, so plain [`SignalToNoise`] lets a
/// candidate win just because its few folds happened to agree. Here each
/// candidate's variance is shrunk toward the median variance of all
/// candidates at the node, weighted as `d0` pseudo-folds — the moderated
/// t-statistic idea from limma. Larger `d0` = stronger shrinkage.
#[derive(Clone, Debug)]
pub struct ModeratedSignalToNoise {
    /// Prior weight, in fold-equivalents.
    pub d0: f64,
    /// Small constant to avoid division by zero.
    pub eps: f64,
}

impl Default for ModeratedSignalToNoise {
    fn default() -> Self {
        ModeratedSignalToNoise { d0: 4.0, eps: 1e-12 }
    }
}

impl Aggregator for ModeratedSignalToNoise {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        // Without node context there is no prior: plain signal-to-noise.
        self.aggregate_node(&[stats])[0]
    }
    fn aggregate_node(&self, stats: &[&FoldStats]) -> Vec<f64> {
        let prior = pooled_variance(stats);
        stats
            .iter()
            .map(|st| {
                if st.n_success == 0 {
                    return f64::NEG_INFINITY;
                }
                st.mean / (moderated_var(st, prior, self.d0).sqrt() + self.eps)
            })
            .collect()
    }
    fn name(&self) -> &'static str {
        "moderated_signal_to_noise"
    }
}

/// Moderated risk-adjusted score `mean − λ·sqrt(moderated var)`; see
/// [`ModeratedSignalToNoise`] for the shrinkage.
#[derive(Clone, Debug)]
pub struct ModeratedMeanMinusLambdaStd {
    /// Penalty weight on the (moderated) standard deviation.
    pub lambda: f64,
    /// Prior weight, in fold-equivalents.
    pub d0: f64,
}

impl Aggregator for ModeratedMeanMinusLambdaStd {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        self.aggregate_node(&[stats])[0]
    }
    fn aggregate_node(&self, stats: &[&FoldStats]) -> Vec<f64> {
        let prior = pooled_variance(stats);
        stats
            .iter()
            .map(|st| {
                if st.n_success == 0 {
                    return f64::NEG_INFINITY;
                }
                st.mean - self.lambda * moderated_var(st, prior, self.d0).sqrt()
            })
            .collect()
    }
    fn name(&self) -> &'static str {
        "moderated_mean_minus_lambda_std"
    }
}

/// User-supplied aggregation strategy.
pub struct Custom {
    /// The scoring closure.
    pub f: Box<dyn Fn(&FoldStats) -> f64 + Send + Sync>,
}

impl Custom {
    /// Wrap a closure as an aggregator.
    pub fn new(f: impl Fn(&FoldStats) -> f64 + Send + Sync + 'static) -> Self {
        Custom { f: Box::new(f) }
    }
}

impl Aggregator for Custom {
    fn aggregate(&self, stats: &FoldStats) -> f64 {
        (self.f)(stats)
    }
    fn name(&self) -> &'static str {
        "custom"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(scores: Vec<f64>) -> FoldStats {
        FoldStats::from_scores(scores, 5)
    }

    #[test]
    fn mean_and_median() {
        let s = stats(vec![1.0, 2.0, 9.0]);
        assert!((Mean.aggregate(&s) - 4.0).abs() < 1e-9);
        assert!((Median.aggregate(&s) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn trimmed_mean_drops_extremes() {
        let s = stats(vec![0.0, 5.0, 5.0, 5.0, 100.0]);
        // trim 20% from each end -> drop 0.0 and 100.0 -> mean of the middle
        let tm = TrimmedMean { frac: 0.2 };
        assert!((tm.aggregate(&s) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn signal_to_noise_prefers_consistency() {
        let consistent = stats(vec![1.0, 1.0, 1.0]);
        let volatile = stats(vec![0.0, 1.0, 2.0]); // same mean, higher std
        let stn = SignalToNoise::default();
        assert!(stn.aggregate(&consistent) > stn.aggregate(&volatile));
    }

    #[test]
    fn mean_minus_lambda_std_penalises_volatility() {
        let s = stats(vec![0.0, 2.0]); // mean 1, sample std sqrt(2)
        let a = MeanMinusLambdaStd { lambda: 0.5 };
        assert!((a.aggregate(&s) - (1.0 - 0.5 * 2f64.sqrt())).abs() < 1e-9);
    }

    #[test]
    fn single_fold_is_not_infinitely_consistent() {
        // One successful fold used to give std = 0 and an ~infinite SNR.
        let s = stats(vec![0.3]);
        assert!(!SignalToNoise::default().aggregate(&s).is_finite());
        assert!(!MeanMinusLambdaStd { lambda: 1.0 }.aggregate(&s).is_finite());
        // Mean is unaffected.
        assert!((Mean.aggregate(&s) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn quantile_interpolates() {
        let s = stats(vec![4.0, 1.0, 3.0, 2.0, 5.0]);
        assert!((Quantile { q: 0.0 }.aggregate(&s) - 1.0).abs() < 1e-9);
        assert!((Quantile { q: 0.5 }.aggregate(&s) - 3.0).abs() < 1e-9);
        assert!((Quantile { q: 0.25 }.aggregate(&s) - 2.0).abs() < 1e-9);
        assert!((Quantile { q: 0.1 }.aggregate(&s) - 1.4).abs() < 1e-9);
        assert_eq!(Quantile { q: 0.5 }.aggregate(&stats(vec![])), f64::NEG_INFINITY);
    }

    #[test]
    fn moderation_tames_lucky_agreement() {
        // `lucky` has a slightly lower mean but near-zero spread by chance;
        // the others have typical spread. Plain SNR crowns `lucky` by a mile;
        // moderation shrinks its variance toward the pooled one.
        let lucky = stats(vec![0.100, 0.101, 0.099]);
        let a = stats(vec![0.0, 0.2, 0.4]);
        let b = stats(vec![0.1, 0.3, 0.5]);
        let c = stats(vec![-0.1, 0.1, 0.3]);
        let all = [&lucky, &a, &b, &c];
        let plain = SignalToNoise::default();
        let modr = ModeratedSignalToNoise { d0: 4.0, eps: 1e-12 };
        let p = plain.aggregate_node(&all);
        let m = modr.aggregate_node(&all);
        assert!(p[0] / p[2] > 50.0);
        assert!(m[0] / m[2] < 2.0);
        // A single-fold candidate gets the pooled variance, not zero.
        let one = stats(vec![0.2]);
        let m1 = modr.aggregate_node(&[&one, &a, &b]);
        assert!(m1[0].is_finite());
    }

    #[test]
    fn custom_closure_runs() {
        let a = Custom::new(|st| st.n_success as f64);
        assert!((a.aggregate(&stats(vec![1.0, 2.0])) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn empty_stats_lose() {
        let s = stats(vec![]);
        assert_eq!(Mean.aggregate(&s), f64::NEG_INFINITY);
        assert_eq!(Median.aggregate(&s), f64::NEG_INFINITY);
        assert_eq!(SignalToNoise::default().aggregate(&s), f64::NEG_INFINITY);
    }
}
