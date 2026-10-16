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
//!
//! Two scoring modes exist (see [`Scoring`]). The legacy
//! [`Scoring::InSample`] mode computes `parent − weighted child` impurity
//! entirely on the validation labels; because impurity is concave that is
//! **never negative**, even for pure-noise splits, so the training fold only
//! contributes bin edges. [`Scoring::Honest`] (the default) instead fits the
//! parent and child models (class proportions / means / medians) on the
//! training fold and measures their loss on the validation fold, so a split
//! that does not generalise scores below zero *in expectation*. (The best of
//! many such candidates can still score above zero by chance; see
//! `TreeParams::null_permutations`.)

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
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// How a candidate split is scored on a fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scoring {
    /// Out-of-sample: models fit on the training fold, loss measured on the
    /// validation fold (Brier for Gini, log-loss for entropy, squared /
    /// absolute error for regression). Per-fold gains can be negative, and
    /// their expectation is ≤ 0 for splits carrying no signal. This removes
    /// the per-candidate bias, not the selection effect of keeping the best
    /// candidate. Default.
    Honest,
    /// Legacy (≤ 0.7) behaviour: impurity decrease computed on the
    /// validation labels alone. Always ≥ 0 and biased upward on small nodes.
    InSample,
}

impl Default for Scoring {
    fn default() -> Self {
        Scoring::Honest
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
    /// Sample standard deviation of `scores` (Bessel-corrected, `n − 1`).
    /// `NaN` when fewer than two folds succeeded: the spread is undefined, so
    /// variance-penalising aggregators score such candidates as non-finite
    /// (excluded) rather than treating them as perfectly consistent.
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
                std: f64::NAN,
                n_success: 0,
                n_total,
            };
        }
        let mean = scores.iter().sum::<f64>() / n as f64;
        let std = if n < 2 {
            f64::NAN
        } else {
            let ss = scores
                .iter()
                .map(|x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f64>();
            (ss / (n - 1) as f64).sqrt()
        };
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

impl FoldStats {
    /// Sample variance (`std²`), `NaN` when undefined.
    pub fn var(&self) -> f64 {
        self.std * self.std
    }

    /// Merge the scores of several fold sets (e.g. repeated K-fold) into one.
    pub fn merged(parts: &[&FoldStats]) -> FoldStats {
        let mut scores = Vec::new();
        let mut n_total = 0;
        for p in parts {
            scores.extend_from_slice(&p.scores);
            n_total += p.n_total;
        }
        FoldStats::from_scores(scores, n_total)
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

#[inline]
fn goes_left(code: u32, state: u32, prefix: bool) -> bool {
    // Threshold: left is `code <= state` (i.e. `x < edge`). Missing codes are
    // MISSING_BIN (u32::MAX) and never fall left, so they route right in both
    // styles. Single-bin: left is `code == state`.
    if prefix {
        code != crate::encoder::MISSING_BIN && code <= state
    } else {
        code == state
    }
}

/// Evaluate every candidate `state` of one feature across all folds.
///
/// Returns one [`FoldStats`] per state, aligned with `states`.
///
/// * [`Scoring::Honest`] (used when the criterion supports it): the per-fold
///   score is `(L_parent − L_left − L_right) / n_val`, where each `L` is the
///   total validation loss of the model fit on the matching *training-fold*
///   samples ([`Criterion::heldout_loss`]). A child that is empty on the
///   training side predicts with the training parent (contributing zero
///   gain). A fold is skipped only when its training or validation side is
///   empty.
/// * [`Scoring::InSample`]: the legacy impurity decrease on the validation
///   labels; a fold is skipped when either child has fewer than `min_child`
///   validation samples.
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
    scoring: Scoring,
) -> Vec<FoldStats> {
    if scoring == Scoring::Honest && criterion.supports_heldout() {
        return eval_feature_honest(columns, targets, feature, states, folds, n_bins, criterion, prefix);
    }
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
                if goes_left(code, state, prefix) {
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

/// Honest (train-fit, validation-scored) variant of [`eval_feature`].
#[allow(clippy::too_many_arguments)]
fn eval_feature_honest<T: Copy>(
    columns: &[Column],
    targets: &[T],
    feature: usize,
    states: &[u32],
    folds: &[Fold],
    n_bins: usize,
    criterion: &dyn Criterion<Target = T>,
    prefix: bool,
) -> Vec<FoldStats> {
    let mut per_state: Vec<Vec<f64>> = vec![Vec::with_capacity(folds.len()); states.len()];

    for fold in folds {
        if fold.val.is_empty() || fold.train.is_empty() {
            continue;
        }
        // Edges fit on the training fold, applied to both sides.
        let (tcodes, vcodes) = fold_train_val_codes(&columns[feature], fold, n_bins);
        let tr: Vec<T> = fold.train.iter().map(|&i| targets[i]).collect();
        let va: Vec<T> = fold.val.iter().map(|&i| targets[i]).collect();
        let n_val = va.len() as f64;
        let parent = match criterion.heldout_loss(&tr, &va) {
            Some(l) if l.is_finite() => l,
            _ => continue,
        };

        let mut tl: Vec<T> = Vec::with_capacity(tr.len());
        let mut trr: Vec<T> = Vec::with_capacity(tr.len());
        let mut vl: Vec<T> = Vec::with_capacity(va.len());
        let mut vr: Vec<T> = Vec::with_capacity(va.len());
        for (si, &state) in states.iter().enumerate() {
            tl.clear();
            trr.clear();
            vl.clear();
            vr.clear();
            for (k, &code) in tcodes.iter().enumerate() {
                if goes_left(code, state, prefix) {
                    tl.push(tr[k]);
                } else {
                    trr.push(tr[k]);
                }
            }
            for (k, &code) in vcodes.iter().enumerate() {
                if goes_left(code, state, prefix) {
                    vl.push(va[k]);
                } else {
                    vr.push(va[k]);
                }
            }
            let child_loss = |t: &[T], v: &[T]| -> f64 {
                if v.is_empty() {
                    return 0.0;
                }
                // Empty training child: predict with the training parent.
                let fit_on = if t.is_empty() { &tr[..] } else { t };
                criterion.heldout_loss(fit_on, v).unwrap_or(f64::NAN)
            };
            let ll = child_loss(&tl[..], &vl[..]);
            let lr = child_loss(&trr[..], &vr[..]);
            let gain = (parent - ll - lr) / n_val;
            if gain.is_finite() {
                per_state[si].push(gain);
            }
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
        // Sample std of {1, 3} = sqrt(2).
        assert!((s.std - 2f64.sqrt()).abs() < 1e-9);
        // A single fold has undefined spread.
        assert!(FoldStats::from_scores(vec![1.0], 2).std.is_nan());
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
        for scoring in [Scoring::Honest, Scoring::InSample] {
            let stats =
                eval_feature(&columns, &targets, 0, &[0, 1], &folds, 2, &gini, 1, false, scoring);
            // At least one state should show a strictly positive mean gain.
            assert!(stats.iter().any(|s| s.n_success > 0 && s.mean > 0.0));
        }
    }

    #[test]
    fn honest_scores_noise_below_in_sample() {
        // Deterministic pseudo-noise: label independent of the feature.
        let mut rng = SplitMix64::new(11);
        let n = 400;
        let x: Vec<f64> = (0..n).map(|_| (rng.next_u64() % 1000) as f64).collect();
        let y: Vec<usize> = (0..n).map(|_| (rng.next_u64() % 2) as usize).collect();
        let columns = vec![Column::Continuous(x)];
        let idx: Vec<usize> = (0..n).collect();
        let folds = KFold::new(5, 3).folds(&idx);
        let gini = Gini::new(2);
        let states: Vec<u32> = (0..7).collect();
        let ins = eval_feature(&columns, &y, 0, &states, &folds, 8, &gini, 1, true, Scoring::InSample);
        let hon = eval_feature(&columns, &y, 0, &states, &folds, 8, &gini, 1, true, Scoring::Honest);
        // In-sample gains are never negative; honest gains are smaller on noise.
        assert!(ins.iter().all(|s| s.scores.iter().all(|&g| g >= -1e-12)));
        let m_ins: f64 = ins.iter().map(|s| s.mean).sum::<f64>() / ins.len() as f64;
        let m_hon: f64 = hon.iter().map(|s| s.mean).sum::<f64>() / hon.len() as f64;
        assert!(m_hon < m_ins);
    }
}
