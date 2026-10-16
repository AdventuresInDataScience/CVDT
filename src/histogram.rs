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
//! With [`Scoring::Honest`] the training side of every fold is recovered by
//! subtraction (`node total − fold`), and the score is the validation loss of
//! training-fit models (Brier / log-loss / squared error), which can be
//! negative for splits that do not generalise.
//!
//! Missing (non-finite) feature values count toward each fold's totals but no
//! bin, so they are scored in the right child, matching how they are routed
//! and how the strict path scores them. For threshold sweeps, the cut after the
//! last bin ("x is present": all present values left, missing right) is
//! offered only when the node has missing values, as on the strict path.
//! More generally, a cut that sends every node sample to the same side is
//! never offered (global bins can leave a node with only some bins
//! populated, e.g. below an earlier cut on the same feature).
//!
//! MAE has no additive sufficient statistic (the median does not decompose), so
//! it is not available on the fast path; the tree rejects it up front.

use crate::aggregation::Aggregator;
use crate::candidate::Candidate;
use crate::criterion::{
    brier_loss_counts, entropy_from_counts, gini_from_counts, log_loss_counts, sq_loss_moments,
    variance_from_moments, LOGLOSS_ALPHA,
};
use crate::cross_validation::{FoldStats, Scoring};
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

/// Held-out loss of a class model fit on `train` (or `fallback` when `train`
/// is empty) for validation counts `val`.
fn heldout_class(kind: ClassImpurityKind, train: &[u64], fallback: &[u64], val: &[u64]) -> f64 {
    let nv: u64 = val.iter().sum();
    if nv == 0 {
        return 0.0;
    }
    let fit_on = if train.iter().any(|&t| t > 0) {
        train
    } else {
        fallback
    };
    match kind {
        ClassImpurityKind::Gini => brier_loss_counts(fit_on, val),
        ClassImpurityKind::Entropy => log_loss_counts(fit_on, val, LOGLOSS_ALPHA),
    }
}

/// Per-call scratch for the classification fold scorer.
struct ClassBufs {
    v_right: Vec<u64>,
    t_parent: Vec<u64>,
    t_left: Vec<u64>,
    t_right: Vec<u64>,
}

impl ClassBufs {
    fn new(c: usize) -> Self {
        ClassBufs {
            v_right: vec![0; c],
            t_parent: vec![0; c],
            t_left: vec![0; c],
            t_right: vec![0; c],
        }
    }
}

/// Score one candidate on one fold from class counts.
///
/// `v_parent`/`v_left` are this fold's validation counts; `node_parent` /
/// `node_left` the same summed over **all** folds, so the training side is
/// `node − fold` (the subtraction trick). Returns `None` for a skipped fold.
#[allow(clippy::too_many_arguments)]
fn class_fold_score(
    kind: ClassImpurityKind,
    scoring: Scoring,
    v_parent: &[u64],
    v_left: &[u64],
    node_parent: &[u64],
    node_left: &[u64],
    b: &mut ClassBufs,
) -> Option<f64> {
    let c = v_parent.len();
    let nf: u64 = v_parent.iter().sum();
    if nf == 0 {
        return None;
    }
    let mut nl = 0u64;
    for cl in 0..c {
        b.v_right[cl] = v_parent[cl] - v_left[cl];
        nl += v_left[cl];
    }
    match scoring {
        Scoring::InSample => {
            let nr = nf - nl;
            if nl == 0 || nr == 0 {
                return None; // degenerate split on this fold
            }
            let ip = imp_counts(kind, v_parent, nf);
            let il = imp_counts(kind, v_left, nl);
            let ir = imp_counts(kind, &b.v_right, nr);
            Some(ip - (nl as f64 * il + nr as f64 * ir) / nf as f64)
        }
        Scoring::Honest => {
            let mut nt = 0u64;
            for cl in 0..c {
                b.t_parent[cl] = node_parent[cl] - v_parent[cl];
                b.t_left[cl] = node_left[cl] - v_left[cl];
                b.t_right[cl] = b.t_parent[cl] - b.t_left[cl];
                nt += b.t_parent[cl];
            }
            if nt == 0 {
                return None;
            }
            let lp = heldout_class(kind, &b.t_parent, &b.t_parent, v_parent);
            let ll = heldout_class(kind, &b.t_left, &b.t_parent, v_left);
            let lr = heldout_class(kind, &b.t_right, &b.t_parent, &b.v_right);
            let g = (lp - ll - lr) / nf as f64;
            if g.is_finite() {
                Some(g)
            } else {
                None
            }
        }
    }
}

/// Score every candidate `feature == bin` split for a **classification** node.
///
/// `codes` are this feature's global bin codes; `labels` the global class ids;
/// `order`/`val_fold` the node samples and their validation-fold ids (from
/// [`crate::cross_validation::KFold::assign`]); `k` the fold count. With
/// [`Scoring::Honest`] the training-fold counts are recovered as
/// `node total − fold`, so honest scoring costs no extra data pass.
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
    prefix: bool,
    scoring: Scoring,
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
    // Missing values count toward the fold totals but no bin: every left child
    // is built from bins only and the right child is `total − left`, so missing
    // samples are scored in the right child — exactly where they are routed.
    let mut n_missing = 0u64;
    for (pos, &sid) in order.iter().enumerate() {
        let s = sid as usize;
        let lab = labels[s];
        if lab >= c {
            continue;
        }
        let f = val_fold[pos] as usize;
        let b = codes[s];
        if b == MISSING_BIN_CODE {
            scratch.tot[f * c + lab] += 1;
            n_missing += 1;
            continue;
        }
        let b = b as usize;
        if b >= bins {
            continue;
        }
        scratch.hist[f * bins * c + b * c + lab] += 1;
        scratch.tot[f * c + lab] += 1;
        scratch.ncnt[f * bins + b] += 1;
    }

    // Node-level class totals (all folds).
    let mut node_tot = vec![0u64; c];
    for f in 0..k {
        for cl in 0..c {
            node_tot[cl] += scratch.tot[f * c + cl] as u64;
        }
    }
    let n_node: u64 = node_tot.iter().sum();

    let mut v_parent = vec![0u64; c];
    let mut v_left = vec![0u64; c];
    let mut node_left = vec![0u64; c];
    let mut bufs = ClassBufs::new(c);
    let mut out = Vec::new();

    // Running per-fold prefix for the threshold sweep (cut after bin b ->
    // left child is bins 0..=b).
    let mut pref = vec![0u64; k * c];

    for b in 0..bins {
        let mut present = 0u64;
        for f in 0..k {
            present += scratch.ncnt[f * bins + b] as u64;
        }
        if prefix {
            for f in 0..k {
                let base = f * bins * c + b * c;
                for cl in 0..c {
                    pref[f * c + cl] += scratch.hist[base + cl] as u64;
                }
            }
            // A cut after an empty bin duplicates the previous one. The final
            // cut (after the last bin) is "x is present": it is only a real
            // split when the node has missing values to send right.
            if present == 0 || (b + 1 >= bins && n_missing == 0) {
                continue;
            }
        } else if present == 0 {
            continue; // bin absent from the node entirely
        }

        // Node-level left counts for this candidate.
        for x in node_left.iter_mut() {
            *x = 0;
        }
        for f in 0..k {
            for cl in 0..c {
                node_left[cl] += if prefix {
                    pref[f * c + cl]
                } else {
                    scratch.hist[f * bins * c + b * c + cl] as u64
                };
            }
        }
        // A cut that sends every node sample the same way cannot partition
        // the node (e.g. the cut after the last non-empty global bin in a
        // child of an earlier split on this feature). Honest scoring would
        // give it exactly 0, which can outrank real splits when
        // `min_impurity_decrease < 0` and then fail the partition check.
        let n_left_node: u64 = node_left.iter().sum();
        if n_left_node == 0 || n_left_node == n_node {
            continue;
        }

        let mut scores: Vec<f64> = Vec::with_capacity(k);
        for f in 0..k {
            for cl in 0..c {
                v_parent[cl] = scratch.tot[f * c + cl] as u64;
                v_left[cl] = if prefix {
                    pref[f * c + cl]
                } else {
                    scratch.hist[f * bins * c + b * c + cl] as u64
                };
            }
            if let Some(g) = class_fold_score(
                kind, scoring, &v_parent, &v_left, &node_tot, &node_left, &mut bufs,
            ) {
                scores.push(g);
            }
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

/// Moments `(n, Σy, Σy²)` of a sample.
#[derive(Clone, Copy, Default)]
struct Mom {
    n: f64,
    s: f64,
    q: f64,
}

impl Mom {
    fn sub(self, o: Mom) -> Mom {
        Mom {
            n: self.n - o.n,
            s: self.s - o.s,
            q: self.q - o.q,
        }
    }
}

/// Score one regression candidate on one fold from moments. `v_*` are the
/// fold's validation moments, `node_*` the all-fold moments.
fn regr_fold_score(
    scoring: Scoring,
    v_parent: Mom,
    v_left: Mom,
    node_parent: Mom,
    node_left: Mom,
) -> Option<f64> {
    if v_parent.n <= 0.0 {
        return None;
    }
    let v_right = v_parent.sub(v_left);
    match scoring {
        Scoring::InSample => {
            if v_left.n <= 0.0 || v_right.n <= 0.0 {
                return None;
            }
            let ip = variance_from_moments(v_parent.n, v_parent.s, v_parent.q);
            let il = variance_from_moments(v_left.n, v_left.s, v_left.q);
            let ir = variance_from_moments(v_right.n, v_right.s, v_right.q);
            Some(ip - (v_left.n * il + v_right.n * ir) / v_parent.n)
        }
        Scoring::Honest => {
            let t_parent = node_parent.sub(v_parent);
            if t_parent.n < 0.5 {
                return None;
            }
            let t_left = node_left.sub(v_left);
            let t_right = t_parent.sub(t_left);
            let mp = t_parent.s / t_parent.n;
            let ml = if t_left.n >= 0.5 { t_left.s / t_left.n } else { mp };
            let mr = if t_right.n >= 0.5 { t_right.s / t_right.n } else { mp };
            let lp = sq_loss_moments(v_parent.n, v_parent.s, v_parent.q, mp);
            let ll = sq_loss_moments(v_left.n, v_left.s, v_left.q, ml);
            let lr = sq_loss_moments(v_right.n, v_right.s, v_right.q, mr);
            let g = (lp - ll - lr) / v_parent.n;
            if g.is_finite() {
                Some(g)
            } else {
                None
            }
        }
    }
}

/// Score every candidate `feature == bin` split for a **regression** node
/// using variance / MSE (moment-based). Honest scoring predicts with
/// training-fold means and measures squared error on the validation fold.
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
    prefix: bool,
    scoring: Scoring,
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

    // Accumulate moments of `y − shift`, with `shift` the mean of a few
    // (shuffled) node samples. Squared-error gains are shift-invariant, but the
    // `Σy²` formulas below cancel catastrophically when the targets' offset
    // dwarfs their spread (y ≈ 1e9 ± 1 leaves no significant digits); after
    // the shift the moments are O(spread²). Costs at most 64 extra reads.
    let probe = order.len().min(64);
    let shift = if probe == 0 {
        0.0
    } else {
        order[..probe].iter().map(|&s| targets[s as usize]).sum::<f64>() / probe as f64
    };
    let shift = if shift.is_finite() { shift } else { 0.0 };

    // Missing values join the fold totals (and hence every right child, where
    // they are routed) but no bin; see `score_classif`.
    let mut fold_n = vec![0u64; k];
    let mut n_missing = 0u64;
    for (pos, &sid) in order.iter().enumerate() {
        let s = sid as usize;
        let b = codes[s];
        let b = if b == MISSING_BIN_CODE {
            n_missing += 1;
            None
        } else if (b as usize) < bins {
            Some(b as usize)
        } else {
            continue;
        };
        let y = targets[s] - shift;
        let f = val_fold[pos] as usize;
        fold_n[f] += 1;
        scratch.rtot_sum[f] += y;
        scratch.rtot_sq[f] += y * y;
        if let Some(b) = b {
            scratch.ncnt[f * bins + b] += 1;
            scratch.rsum[f * bins + b] += y;
            scratch.rsq[f * bins + b] += y * y;
        }
    }

    // Per-fold and node totals.
    let mut fold_tot = vec![Mom::default(); k];
    let mut node_tot = Mom::default();
    for f in 0..k {
        fold_tot[f] = Mom {
            n: fold_n[f] as f64,
            s: scratch.rtot_sum[f],
            q: scratch.rtot_sq[f],
        };
        node_tot.n += fold_tot[f].n;
        node_tot.s += fold_tot[f].s;
        node_tot.q += fold_tot[f].q;
    }

    let cell = |f: usize, b: usize| Mom {
        n: scratch.ncnt[f * bins + b] as f64,
        s: scratch.rsum[f * bins + b],
        q: scratch.rsq[f * bins + b],
    };

    let mut out = Vec::new();
    let mut pref = vec![Mom::default(); k];

    for b in 0..bins {
        let mut present = 0.0;
        for f in 0..k {
            let m = cell(f, b);
            present += m.n;
            if prefix {
                pref[f].n += m.n;
                pref[f].s += m.s;
                pref[f].q += m.q;
            }
        }
        if prefix {
            // Same candidate rules as `score_classif`.
            if present <= 0.0 || (b + 1 >= bins && n_missing == 0) {
                continue;
            }
        } else if present <= 0.0 {
            continue;
        }

        let left_of = |f: usize| if prefix { pref[f] } else { cell(f, b) };
        let mut node_left = Mom::default();
        for f in 0..k {
            let m = left_of(f);
            node_left.n += m.n;
            node_left.s += m.s;
            node_left.q += m.q;
        }
        // One-sided cut (every node sample on the same side): not a split.
        // Counts are sums of small integers, so the comparison is exact.
        if node_left.n <= 0.0 || node_left.n >= node_tot.n {
            continue;
        }

        let mut scores: Vec<f64> = Vec::with_capacity(k);
        for f in 0..k {
            if let Some(g) = regr_fold_score(scoring, fold_tot[f], left_of(f), node_tot, node_left)
            {
                scores.push(g);
            }
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
            false,
            Scoring::Honest,
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
        let out = score_regr(
            &codes,
            &y,
            &order,
            &val_fold,
            2,
            2,
            &Mean,
            0,
            &mut sc,
            false,
            Scoring::Honest,
        );
        assert!(out.iter().any(|c| c.stats.n_success > 0 && c.score > 0.0));
    }

    #[test]
    fn in_sample_mode_never_negative_honest_can_be() {
        // Bin/label association flips between folds, so the training-fold
        // model mispredicts the held-out fold (honest gain -0.15625 on folds
        // 0 and 2), while in-sample gain stays >= 0 by concavity.
        let codes: Vec<Bin> = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let labels: Vec<usize> = vec![0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let order: Vec<SampleId> = (0..12).collect();
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let mut sc = FastScratch::new();
        let run = |scoring: Scoring, sc: &mut FastScratch| {
            score_classif(
                &codes,
                &labels,
                &order,
                &val_fold,
                3,
                2,
                2,
                ClassImpurityKind::Gini,
                &Mean,
                0,
                sc,
                false,
                scoring,
            )
        };
        let ins = run(Scoring::InSample, &mut sc);
        assert!(ins.iter().all(|c| c.stats.scores.iter().all(|&g| g >= -1e-12)));
        let hon = run(Scoring::Honest, &mut sc);
        assert!(hon.iter().any(|c| c.stats.scores.iter().any(|&g| g < 0.0)));
    }

    /// Missing values must score exactly like values in an extra top bin that
    /// never joins a left child (that is how they are routed).
    #[test]
    fn missing_scores_as_always_right_bin() {
        let m = MISSING_BIN_CODE;
        let with_missing: Vec<Bin> = vec![0, 1, m, 0, m, 1, 0, m, 1, 1, m, 0];
        // Same data, missing recoded as bin 2.
        let as_top_bin: Vec<Bin> = with_missing
            .iter()
            .map(|&b| if b == m { 2 } else { b })
            .collect();
        let labels: Vec<usize> = vec![0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let y: Vec<f64> = vec![0.5, 2.0, 3.5, 0.1, 4.0, 1.8, 0.3, 3.9, 1.1, 2.2, 3.1, 0.4];
        let order: Vec<SampleId> = (0..12).collect();
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let mut sc = FastScratch::new();
        for &scoring in &[Scoring::Honest, Scoring::InSample] {
            for &prefix in &[true, false] {
                let ca = score_classif(
                    &with_missing, &labels, &order, &val_fold, 3, 2, 2,
                    ClassImpurityKind::Gini, &Mean, 0, &mut sc, prefix, scoring,
                );
                let cb = score_classif(
                    &as_top_bin, &labels, &order, &val_fold, 3, 3, 2,
                    ClassImpurityKind::Gini, &Mean, 0, &mut sc, prefix, scoring,
                );
                let ra = score_regr(
                    &with_missing, &y, &order, &val_fold, 3, 2, &Mean, 0, &mut sc, prefix,
                    scoring,
                );
                let rb = score_regr(
                    &as_top_bin, &y, &order, &val_fold, 3, 3, &Mean, 0, &mut sc, prefix,
                    scoring,
                );
                for (a_list, b_list) in [(&ca, &cb), (&ra, &rb)] {
                    assert!(!a_list.is_empty());
                    for a in a_list.iter() {
                        let b = b_list
                            .iter()
                            .find(|b| b.candidate == a.candidate)
                            .expect("candidate present in both");
                        assert_eq!(a.stats.scores.len(), b.stats.scores.len());
                        for (x, z) in a.stats.scores.iter().zip(b.stats.scores.iter()) {
                            assert!((x - z).abs() < 1e-12, "{x} vs {z}");
                        }
                    }
                }
            }
        }
    }

    /// The threshold sweep's last cut ("x is present") is a real split only
    /// when the node has missing values; it must not be offered otherwise.
    #[test]
    fn presence_cut_offered_only_with_missing() {
        let labels: Vec<usize> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let y: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let order: Vec<SampleId> = (0..8).collect();
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let full: Vec<Bin> = vec![0, 1, 2, 0, 1, 2, 0, 1];
        let mut gappy = full.clone();
        gappy[3] = MISSING_BIN_CODE;
        gappy[6] = MISSING_BIN_CODE;
        let top = Candidate {
            feature: 0,
            state: 2,
        };
        let mut sc = FastScratch::new();
        for &scoring in &[Scoring::Honest, Scoring::InSample] {
            for (codes, want) in [(&full, false), (&gappy, true)] {
                let c = score_classif(
                    codes, &labels, &order, &val_fold, 2, 3, 2,
                    ClassImpurityKind::Gini, &Mean, 0, &mut sc, true, scoring,
                );
                let r = score_regr(
                    codes, &y, &order, &val_fold, 2, 3, &Mean, 0, &mut sc, true, scoring,
                );
                assert_eq!(c.iter().any(|x| x.candidate == top), want);
                assert_eq!(r.iter().any(|x| x.candidate == top), want);
            }
        }
    }

    /// Squared-error gains are shift-invariant. Accumulating raw `Σy²` at
    /// y ≈ 1e9 ± 10 cancels away every significant digit; the node shift in
    /// `score_regr` must keep the scores equal to the unshifted ones.
    #[test]
    fn regression_scores_survive_large_target_offsets() {
        let codes: Vec<Bin> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let y: Vec<f64> = vec![0.5, 2.0, 9.5, 1.1, 3.9, 8.0, 0.2, 2.7, 9.9, 1.4, 3.1, 7.6];
        let big: Vec<f64> = y.iter().map(|v| v + 1e9).collect();
        let order: Vec<SampleId> = (0..12).collect();
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let mut sc = FastScratch::new();
        for &scoring in &[Scoring::Honest, Scoring::InSample] {
            for &prefix in &[true, false] {
                let a = score_regr(
                    &codes, &y, &order, &val_fold, 3, 3, &Mean, 0, &mut sc, prefix, scoring,
                );
                let b = score_regr(
                    &codes, &big, &order, &val_fold, 3, 3, &Mean, 0, &mut sc, prefix, scoring,
                );
                assert_eq!(a.len(), b.len());
                for (p, q) in a.iter().zip(b.iter()) {
                    assert_eq!(p.candidate, q.candidate);
                    assert_eq!(p.stats.scores.len(), q.stats.scores.len());
                    for (x, z) in p.stats.scores.iter().zip(q.stats.scores.iter()) {
                        assert!((x - z).abs() < 1e-6 * (1.0 + x.abs()), "{x} vs {z}");
                    }
                }
            }
        }
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
            false,
            Scoring::Honest,
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
            false,
            Scoring::Honest,
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.candidate, y.candidate);
            assert_eq!(x.score.is_finite(), y.score.is_finite());
        }
    }

    /// A cut that puts every node sample on one side is not a split and must
    /// not be offered: under honest scoring it would score exactly 0.
    #[test]
    fn one_sided_cuts_are_not_offered() {
        // Global feature has 4 bins but this node only populates bins 0 and 1
        // (as in the left child of an earlier `x < edge` cut on it).
        let codes: Vec<Bin> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let labels: Vec<usize> = vec![0, 1, 1, 0, 0, 1, 1, 0];
        let y: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let order: Vec<SampleId> = (0..8).collect();
        let val_fold: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let mut sc = FastScratch::new();
        for &scoring in &[Scoring::Honest, Scoring::InSample] {
            let c = score_classif(
                &codes, &labels, &order, &val_fold, 2, 4, 2,
                ClassImpurityKind::Gini, &Mean, 0, &mut sc, true, scoring,
            );
            let r = score_regr(
                &codes, &y, &order, &val_fold, 2, 4, &Mean, 0, &mut sc, true, scoring,
            );
            // Threshold sweep: only the cut after bin 0 partitions the node.
            let cs: Vec<u32> = c.iter().map(|x| x.candidate.state).collect();
            let rs: Vec<u32> = r.iter().map(|x| x.candidate.state).collect();
            assert_eq!(cs, vec![0]);
            assert_eq!(rs, vec![0]);
            // Single-bin style on a node that populates one bin only.
            let one: Vec<Bin> = vec![2; 8];
            let c1 = score_classif(
                &one, &labels, &order, &val_fold, 2, 4, 2,
                ClassImpurityKind::Gini, &Mean, 0, &mut sc, false, scoring,
            );
            let r1 = score_regr(
                &one, &y, &order, &val_fold, 2, 4, &Mean, 0, &mut sc, false, scoring,
            );
            assert!(c1.is_empty() && r1.is_empty());
        }
    }
}
