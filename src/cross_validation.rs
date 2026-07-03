//! Cross-validation engine.
//!
//! Two responsibilities:
//! 1. Deterministically partition a node's samples into K folds.
//! 2. Score candidate splits for one feature across those folds.
//!
//! The engine contains **no impurity logic** — it delegates every impurity
//! computation to a [`Criterion`]. For each fold it fits the (continuous) bin
//! edges on the *training* samples, applies them to the *validation* samples,
//! and asks the criterion for the impurity decrease on the validation labels.
//! This is what makes the score an out-of-sample estimate of split quality.

use crate::criterion::Criterion;
use crate::data::{Column, SampleId};
use crate::encoder::{bin_of, quantile_edges};

/// Small, fast, self-contained PRNG (SplitMix64) for reproducible shuffling.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next pseudo-random 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` (bound must be > 0).
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// A single train/validation split of sample indices.
#[derive(Clone, Debug)]
pub struct Fold {
    /// Training sample indices (into the global columns).
    pub train: Vec<usize>,
    /// Validation sample indices (into the global columns).
    pub val: Vec<usize>,
}

/// K-fold cross-validation configuration.
#[derive(Clone, Debug)]
pub struct KFold {
    /// Number of folds requested (clamped to `[2, n_samples]` per node).
    pub k: usize,
    /// Seed for shuffling.
    pub seed: u64,
    /// Whether to shuffle before partitioning.
    pub shuffle: bool,
}

impl KFold {
    /// Convenience constructor.
    pub fn new(k: usize, seed: u64) -> Self {
        KFold {
            k,
            seed,
            shuffle: true,
        }
    }

    /// Produce folds for the given sample indices.
    ///
    /// Deterministic: the same `indices`, `k`, `seed` and `shuffle` always yield
    /// the same folds.
    pub fn folds(&self, indices: &[usize]) -> Vec<Fold> {
        let n = indices.len();
        if n == 0 {
            return Vec::new();
        }
        let k = self.k.max(2).min(n);

        let mut idx: Vec<usize> = indices.to_vec();
        if self.shuffle {
            // Mix the node size into the seed so different nodes decorrelate,
            // while remaining fully reproducible for identical inputs.
            let mut rng =
                SplitMix64::new(self.seed ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for i in (1..idx.len()).rev() {
                let j = rng.below((i as u64) + 1) as usize;
                idx.swap(i, j);
            }
        }

        let base = n / k;
        let rem = n % k;
        let mut folds = Vec::with_capacity(k);
        let mut start = 0;
        for f in 0..k {
            let size = base + if f < rem { 1 } else { 0 };
            let val: Vec<usize> = idx[start..start + size].to_vec();
            let train: Vec<usize> = idx[..start]
                .iter()
                .chain(idx[start + size..].iter())
                .copied()
                .collect();
            folds.push(Fold { train, val });
            start += size;
        }
        folds
    }

    /// Compact fold assignment for the histogram ("fast") path.
    ///
    /// Returns the node samples in shuffled order, the validation-fold id of
    /// each (aligned with that order), and the effective fold count `k` (clamped
    /// to `[2, min(n, 255)]`). This is the representation the fast scorer
    /// scatters into per-fold histograms in a single pass; it is derived from
    /// exactly the same shuffle as [`KFold::folds`], so the two agree.
    pub fn assign(&self, indices: &[usize]) -> (Vec<SampleId>, Vec<u8>, usize) {
        let n = indices.len();
        if n == 0 {
            return (Vec::new(), Vec::new(), 0);
        }
        // u8 fold ids cap the fast path at 255 folds, which is far beyond any
        // sensible CV setting.
        let k = self.k.max(2).min(n).min(255);

        let mut idx: Vec<usize> = indices.to_vec();
        if self.shuffle {
            let mut rng =
                SplitMix64::new(self.seed ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for i in (1..idx.len()).rev() {
                let j = rng.below((i as u64) + 1) as usize;
                idx.swap(i, j);
            }
        }

        let base = n / k;
        let rem = n % k;
        let mut order = Vec::with_capacity(n);
        let mut val_fold = vec![0u8; n];
        let mut pos = 0;
        for f in 0..k {
            let size = base + if f < rem { 1 } else { 0 };
            for _ in 0..size {
                order.push(idx[pos] as SampleId);
                val_fold[pos] = f as u8;
                pos += 1;
            }
        }
        (order, val_fold, k)
    }
}

/// Per-fold score summary for one candidate split.
#[derive(Clone, Debug)]
pub struct FoldStats {
    /// Impurity-decrease score for each *successful* fold.
    pub scores: Vec<f64>,
    /// Mean of `scores` (`-inf` when there are no successful folds).
    pub mean: f64,
    /// Median of `scores` (`-inf` when there are no successful folds).
    pub median: f64,
    /// Population standard deviation of `scores` (`0` when empty).
    pub std: f64,
    /// Number of folds that produced a valid (non-degenerate) split.
    pub n_success: usize,
    /// Total number of folds attempted.
    pub n_total: usize,
}

impl FoldStats {
    /// Summarise a list of successful-fold scores.
    pub fn from_scores(scores: Vec<f64>, n_total: usize) -> Self {
        let n = scores.len();
        if n == 0 {
            return FoldStats {
                scores,
                mean: f64::NEG_INFINITY,
                median: f64::NEG_INFINITY,
                std: 0.0,
                n_success: 0,
                n_total,
            };
        }
        let mean = scores.iter().sum::<f64>() / n as f64;
        let var = scores
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std = var.sqrt();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        FoldStats {
            scores,
            mean,
            median,
            std,
            n_success: n,
            n_total,
        }
    }
}

/// Codes (bin id or category id) for the validation samples of one fold.
fn fold_val_codes(column: &Column, fold: &Fold, n_bins: usize) -> Vec<u32> {
    match column {
        Column::Continuous(v) => {
            // Fit edges on the TRAINING fold only — this is the leakage guard.
            let train_vals: Vec<f64> = fold.train.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&train_vals, n_bins);
            fold.val.iter().map(|&i| bin_of(&edges, v[i])).collect()
        }
        Column::Categorical(c) => fold.val.iter().map(|&i| c[i]).collect(),
    }
}

/// Bin codes for **both** the training and validation samples of a fold, using
/// the same edges (fit on the training fold). Used by objective-driven scoring,
/// which needs the training-side class distribution to assign each child a
/// predicted class before measuring the metric on validation.
pub(crate) fn fold_train_val_codes(
    column: &Column,
    fold: &Fold,
    n_bins: usize,
) -> (Vec<u32>, Vec<u32>) {
    match column {
        Column::Continuous(v) => {
            let train_vals: Vec<f64> = fold.train.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&train_vals, n_bins);
            let tc = fold.train.iter().map(|&i| bin_of(&edges, v[i])).collect();
            let vc = fold.val.iter().map(|&i| bin_of(&edges, v[i])).collect();
            (tc, vc)
        }
        Column::Categorical(c) => {
            let tc = fold.train.iter().map(|&i| c[i]).collect();
            let vc = fold.val.iter().map(|&i| c[i]).collect();
            (tc, vc)
        }
    }
}

/// Evaluate every candidate `state` of one feature across all folds.
///
/// Returns one [`FoldStats`] per state, aligned with `states`. The score for a
/// fold is the impurity decrease on the validation labels:
/// `parent - weighted_child`. A fold is skipped for a state when either child
/// has fewer than `min_child` validation samples.
#[allow(clippy::too_many_arguments)]
pub fn eval_feature<T: Copy>(
    columns: &[Column],
    targets: &[T],
    feature: usize,
    states: &[u32],
    folds: &[Fold],
    n_bins: usize,
    criterion: &dyn Criterion<Target = T>,
    min_child: usize,
    prefix: bool,
) -> Vec<FoldStats> {
    let mut per_state: Vec<Vec<f64>> = vec![Vec::with_capacity(folds.len()); states.len()];
    let min_child = min_child.max(1);

    for fold in folds {
        if fold.val.is_empty() {
            continue;
        }
        let codes = fold_val_codes(&columns[feature], fold, n_bins);
        let val_targets: Vec<T> = fold.val.iter().map(|&i| targets[i]).collect();
        let total = val_targets.len();
        let parent = criterion.impurity(&val_targets);

        for (si, &state) in states.iter().enumerate() {
            let mut left: Vec<T> = Vec::new();
            let mut right: Vec<T> = Vec::new();
            for (k, &code) in codes.iter().enumerate() {
                // Threshold: left is `code <= state` (i.e. `x < edge`). Missing
                // codes are MISSING_BIN (u32::MAX) and never fall left, so they
                // route right in both styles. Single-bin: left is `code == state`.
                let goes_left = if prefix {
                    code != crate::encoder::MISSING_BIN && code <= state
                } else {
                    code == state
                };
                if goes_left {
                    left.push(val_targets[k]);
                } else {
                    right.push(val_targets[k]);
                }
            }
            if left.len() < min_child || right.len() < min_child {
                continue; // degenerate split on this fold
            }
            let il = criterion.impurity(&left);
            let ir = criterion.impurity(&right);
            let child = (left.len() as f64 * il + right.len() as f64 * ir) / total as f64;
            per_state[si].push(parent - child);
        }
    }

    per_state
        .into_iter()
        .map(|s| FoldStats::from_scores(s, folds.len()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criterion::Gini;
    use std::collections::HashSet;

    #[test]
    fn folds_cover_all_and_are_disjoint() {
        let idx: Vec<usize> = (0..10).collect();
        let kf = KFold::new(5, 7);
        let folds = kf.folds(&idx);
        assert_eq!(folds.len(), 5);
        let mut seen: Vec<usize> = folds.iter().flat_map(|f| f.val.iter().copied()).collect();
        seen.sort_unstable();
        assert_eq!(seen, idx);
        for f in &folds {
            let vset: HashSet<usize> = f.val.iter().copied().collect();
            assert!(f.train.iter().all(|i| !vset.contains(i)));
            assert_eq!(f.train.len() + f.val.len(), idx.len());
        }
    }

    #[test]
    fn folds_are_deterministic() {
        let idx: Vec<usize> = (0..23).collect();
        let a = KFold::new(4, 99).folds(&idx);
        let b = KFold::new(4, 99).folds(&idx);
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa.val, fb.val);
            assert_eq!(fa.train, fb.train);
        }
    }

    #[test]
    fn k_is_clamped_to_sample_count() {
        let idx: Vec<usize> = (0..3).collect();
        let folds = KFold::new(10, 1).folds(&idx);
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn foldstats_summary_is_correct() {
        let s = FoldStats::from_scores(vec![1.0, 3.0], 2);
        assert_eq!(s.n_success, 2);
        assert!((s.mean - 2.0).abs() < 1e-9);
        assert!((s.median - 2.0).abs() < 1e-9);
        assert!((s.std - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_scores_sink_to_neg_infinity() {
        let s = FoldStats::from_scores(vec![], 3);
        assert_eq!(s.n_success, 0);
        assert_eq!(s.mean, f64::NEG_INFINITY);
    }

    #[test]
    fn perfectly_separating_split_has_positive_gain() {
        // Feature 0 (categorical) perfectly predicts the class.
        let columns = vec![Column::Categorical(vec![0, 0, 1, 1, 0, 0, 1, 1])];
        let targets = vec![0usize, 0, 1, 1, 0, 0, 1, 1];
        let idx: Vec<usize> = (0..8).collect();
        let folds = KFold::new(4, 5).folds(&idx);
        let gini = Gini::new(2);
        let stats = eval_feature(&columns, &targets, 0, &[0, 1], &folds, 2, &gini, 1, false);
        // At least one state should show a strictly positive mean gain.
        assert!(stats.iter().any(|s| s.n_success > 0 && s.mean > 0.0));
    }
}
