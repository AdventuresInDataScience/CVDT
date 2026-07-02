//! Score aggregation.
//!
//! Converts a candidate's [`FoldStats`] into a single ranking number where
//! **higher is better**. This is the knob that lets the same tree be optimistic
//! (plain mean), robust to outlier folds (median / trimmed mean), or explicitly
//! risk-averse (`mean - λ·std`, or a signal-to-noise ratio).
//!
//! Aggregation is completely independent of how the fold scores were produced.

use crate::cross_validation::FoldStats;

/// Turns per-fold statistics into one comparable score (higher = better).
pub trait Aggregator: Send + Sync {
    /// Aggregate the statistics. Must return [`f64::NEG_INFINITY`] (or another
    /// clearly-losing value) when there are no successful folds.
    fn aggregate(&self, stats: &FoldStats) -> f64;

    /// Human-readable name.
    fn name(&self) -> &'static str;
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
        let s = stats(vec![0.0, 2.0]); // mean 1, std 1
        let a = MeanMinusLambdaStd { lambda: 0.5 };
        assert!((a.aggregate(&s) - 0.5).abs() < 1e-9);
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
