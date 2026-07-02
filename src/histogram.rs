//! Histogram-based fast split scoring.
//!
//! This is the performance core of the fast path. For one feature it makes a
//! **single** pass over the node's samples, scattering each into the histogram
//! of the validation fold it belongs to. Because the K validation folds are
//! disjoint and cover the node, that one O(n) pass yields all K validation
//! histograms at once — the K-fold cross-validation no longer multiplies the
//! expensive linear work.
//!
//! Each candidate split (`feature == bin`) is then scored from sufficient
//! statistics in O(1) per fold: the "in-bin" child is a single histogram cell
//! and the "rest" child is `fold_total − cell` (the subtraction trick), so no
//! child sample slices are ever materialised. Gini/entropy come from class
//! counts; variance/MSE from `(count, Σy, Σy²)` moments.
//!
//! MAE has no additive sufficient statistic (the median does not decompose), so
//! it is not available on the fast path; the tree rejects it up front.

use crate::aggregation::Aggregator;
use crate::candidate::Candidate;
use crate::criterion::{entropy_from_counts, gini_from_counts, variance_from_moments};
use crate::cross_validation::FoldStats;
use crate::data::{Bin, SampleId, MISSING_BIN_CODE};
use crate::selector::ScoredCandidate;

/// Which classification impurity the fast path computes from class counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassImpurityKind {
    /// Gini impurity.
    Gini,
    /// Shannon entropy.
    Entropy,
}

/// Reusable scratch buffers so per-node scoring rarely allocates.
///
/// Buffers grow to the largest size seen and are zeroed (not reallocated) on
/// reuse. In serial scoring a single instance is reused across all features of
/// a node; parallel workers each hold their own.
#[derive(Default)]
pub struct FastScratch {
    // classification: per (fold, bin, class) counts and per (fold, class) totals
    hist: Vec<u32>,
    tot: Vec<u32>,
    // shared: per (fold, bin) sample counts
    ncnt: Vec<u32>,
    // regression: per (fold, bin) moments and per-fold totals
    rsum: Vec<f64>,
    rsq: Vec<f64>,
    rtot_sum: Vec<f64>,
    rtot_sq: Vec<f64>,
}

impl FastScratch {
    /// A fresh scratch with no allocation yet.
    pub fn new() -> Self {
        FastScratch::default()
    }
}

fn reset_u32(v: &mut Vec<u32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0);
    }
    for x in v[..n].iter_mut() {
        *x = 0;
    }
}

fn reset_f64(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    for x in v[..n].iter_mut() {
        *x = 0.0;
    }
}

fn imp_counts(kind: ClassImpurityKind, counts: &[u64], n: u64) -> f64 {
    match kind {
        ClassImpurityKind::Gini => gini_from_counts(counts, n),
        ClassImpurityKind::Entropy => entropy_from_counts(counts, n),
    }
}

/// Score every candidate `feature == bin` split for a **classification** node.
///
/// `codes` are this feature's global bin codes; `labels` the global class ids;
/// `order`/`val_fold` the node samples and their validation-fold ids (from
/// [`crate::cross_validation::KFold::assign`]); `k` the fold count.
#[allow(clippy::too_many_arguments)]
pub fn score_classif(
    codes: &[Bin],
    labels: &[usize],
    order: &[SampleId],
    val_fold: &[u8],
    k: usize,
    max_bin: usize,
    n_classes: usize,
    kind: ClassImpurityKind,
    aggregator: &dyn Aggregator,
    feature: usize,
    scratch: &mut FastScratch,
) -> Vec<ScoredCandidate> {
    let bins = max_bin;
    let c = n_classes;
    if bins == 0 || c == 0 || k == 0 {
        return Vec::new();
    }

    reset_u32(&mut scratch.hist, k * bins * c);
    reset_u32(&mut scratch.tot, k * c);
    reset_u32(&mut scratch.ncnt, k * bins);

    // Single scatter pass: each sample lands in its validation fold's histogram.
    for (pos, &sid) in order.iter().enumerate() {
        let s = sid as usize;
        let b = codes[s];
        if b == MISSING_BIN_CODE {
            continue;
        }
        let b = b as usize;
        if b >= bins {
            continue;
        }
        let lab = labels[s];
        if lab >= c {
            continue;
        }
        let f = val_fold[pos] as usize;
        scratch.hist[f * bins * c + b * c + lab] += 1;
        scratch.tot[f * c + lab] += 1;
        scratch.ncnt[f * bins + b] += 1;
    }

    // Per-fold sample totals.
    let mut fold_n = vec![0u64; k];
    for (f, fn_) in fold_n.iter_mut().enumerate() {
        let mut s = 0u64;
        for cl in 0..c {
            s += scratch.tot[f * c + cl] as u64;
        }
        *fn_ = s;
    }

    let mut parent = vec![0u64; c];
    let mut left = vec![0u64; c];
    let mut right = vec![0u64; c];
    let mut out = Vec::new();

    for b in 0..bins {
        // Skip bins absent from the node entirely.
        let mut present = 0u64;
        for f in 0..k {
            present += scratch.ncnt[f * bins + b] as u64;
        }
        if present == 0 {
            continue;
        }

        let mut scores: Vec<f64> = Vec::with_capacity(k);
        for f in 0..k {
            let nf = fold_n[f];
            if nf == 0 {
                continue;
            }
            let base = f * bins * c + b * c;
            let mut nl = 0u64;
            for cl in 0..c {
                let lc = scratch.hist[base + cl] as u64;
                let tc = scratch.tot[f * c + cl] as u64;
                parent[cl] = tc;
                left[cl] = lc;
                right[cl] = tc - lc;
                nl += lc;
            }
            let nr = nf - nl;
            if nl == 0 || nr == 0 {
                continue; // degenerate split on this fold
            }
            let ip = imp_counts(kind, &parent, nf);
            let il = imp_counts(kind, &left, nl);
            let ir = imp_counts(kind, &right, nr);
            let child = (nl as f64 * il + nr as f64 * ir) / nf as f64;
            scores.push(ip - child);
        }

        let stats = FoldStats::from_scores(scores, k);
        let score = aggregator.aggregate(&stats);
        out.push(ScoredCandidate {
            candidate: Candidate {
                feature,
                state: b as u32,
            },
            score,
            stats,
        });
    }
    out
}

/// Score every candidate `feature == bin` split for a **regression** node
/// using variance / MSE (moment-based).
#[allow(clippy::too_many_arguments)]
pub fn score_regr(
    codes: &[Bin],
    targets: &[f64],
    order: &[SampleId],
    val_fold: &[u8],
    k: usize,
    max_bin: usize,
    aggregator: &dyn Aggregator,
    feature: usize,
    scratch: &mut FastScratch,
) -> Vec<ScoredCandidate> {
    let bins = max_bin;
    if bins == 0 || k == 0 {
        return Vec::new();
    }

    reset_u32(&mut scratch.ncnt, k * bins);
    reset_f64(&mut scratch.rsum, k * bins);
    reset_f64(&mut scratch.rsq, k * bins);
    reset_f64(&mut scratch.rtot_sum, k);
    reset_f64(&mut scratch.rtot_sq, k);

    for (pos, &sid) in order.iter().enumerate() {
        let s = sid as usize;
        let b = codes[s];
        if b == MISSING_BIN_CODE {
            continue;
        }
        let b = b as usize;
        if b >= bins {
            continue;
        }
        let y = targets[s];
        let f = val_fold[pos] as usize;
        scratch.ncnt[f * bins + b] += 1;
        scratch.rsum[f * bins + b] += y;
        scratch.rsq[f * bins + b] += y * y;
        scratch.rtot_sum[f] += y;
        scratch.rtot_sq[f] += y * y;
    }

    let mut fold_n = vec![0u64; k];
    for (f, fn_) in fold_n.iter_mut().enumerate() {
        let mut s = 0u64;
        for b in 0..bins {
            s += scratch.ncnt[f * bins + b] as u64;
        }
        *fn_ = s;
    }

    let mut out = Vec::new();
    for b in 0..bins {
        let mut present = 0u64;
        for f in 0..k {
            present += scratch.ncnt[f * bins + b] as u64;
        }
        if present == 0 {
            continue;
        }

        let mut scores: Vec<f64> = Vec::with_capacity(k);
        for f in 0..k {
            let nf = fold_n[f];
            if nf == 0 {
                continue;
            }
            let nl = scratch.ncnt[f * bins + b] as u64;
            let nr = nf - nl;
            if nl == 0 || nr == 0 {
                continue;
            }
            let sl = scratch.rsum[f * bins + b];
            let ql = scratch.rsq[f * bins + b];
            let stot = scratch.rtot_sum[f];
            let qtot = scratch.rtot_sq[f];
            let sr = stot - sl;
            let qr = qtot - ql;
            let ip = variance_from_moments(nf as f64, stot, qtot);
            let il = variance_from_moments(nl as f64, sl, ql);
            let ir = variance_from_moments(nr as f64, sr, qr);
            let child = (nl as f64 * il + nr as f64 * ir) / nf as f64;
            scores.push(ip - child);
        }

        let stats = FoldStats::from_scores(scores, k);
        let score = aggregator.aggregate(&stats);
        out.push(ScoredCandidate {
            candidate: Candidate {
                feature,
                state: b as u32,
            },
            score,
            stats,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregation::Mean;

    #[test]
    fn separating_bin_scores_positive() {
        // Bin perfectly predicts the class: code 0 -> class 0, code 1 -> class 1.
        let codes: Vec<Bin> = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let labels: Vec<usize> = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let order: Vec<SampleId> = (0..8).collect();
        // Two folds: first half / second half.
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let mut sc = FastScratch::new();
        let out = score_classif(
            &codes,
            &labels,
            &order,
            &val_fold,
            2,
            2,
            2,
            ClassImpurityKind::Gini,
            &Mean,
            0,
            &mut sc,
        );
        assert!(out.iter().any(|c| c.stats.n_success > 0 && c.score > 0.0));
    }

    #[test]
    fn regression_separating_bin_scores_positive() {
        let codes: Vec<Bin> = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let y: Vec<f64> = vec![1.0, 1.0, 9.0, 9.0, 1.0, 1.0, 9.0, 9.0];
        let order: Vec<SampleId> = (0..8).collect();
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let mut sc = FastScratch::new();
        let out = score_regr(&codes, &y, &order, &val_fold, 2, 2, &Mean, 0, &mut sc);
        assert!(out.iter().any(|c| c.stats.n_success > 0 && c.score > 0.0));
    }

    #[test]
    fn scratch_reuse_is_consistent() {
        let codes: Vec<Bin> = vec![0, 1, 0, 1];
        let labels: Vec<usize> = vec![0, 1, 0, 1];
        let order: Vec<SampleId> = (0..4).collect();
        let val_fold: Vec<u8> = vec![0, 0, 1, 1];
        let mut sc = FastScratch::new();
        let a = score_classif(
            &codes,
            &labels,
            &order,
            &val_fold,
            2,
            2,
            2,
            ClassImpurityKind::Gini,
            &Mean,
            0,
            &mut sc,
        );
        let b = score_classif(
            &codes,
            &labels,
            &order,
            &val_fold,
            2,
            2,
            2,
            ClassImpurityKind::Gini,
            &Mean,
            0,
            &mut sc,
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.candidate, y.candidate);
            assert_eq!(x.score.is_finite(), y.score.is_finite());
        }
    }
}
