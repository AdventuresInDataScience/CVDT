//! Objective-driven split scoring for classification.
//!
//! Standard trees score a split by an *impurity* proxy (Gini, entropy). This
//! module lets the tree instead greedily optimise the **metric you actually
//! care about** — precision, recall, F1, Fβ, or accuracy — evaluated on the
//! held-out folds of the cross-validation engine. Rather than searching many
//! hyper-parameter combinations until a proxy happens to land on a good metric,
//! each split is chosen because it improves the objective on validation data.
//!
//! ## How a split is scored against a metric
//! A metric like F1 is a property of *predictions vs. labels*, so scoring a
//! split needs a predicted class per child. To stay leakage-honest we mirror
//! the rest of the framework:
//!
//! 1. On the **training** fold, each child is assigned the class that is the
//!    majority of its training samples (the class a leaf there would predict).
//! 2. On the **validation** fold, samples are routed to the children, each
//!    child predicts its assigned class, and a confusion matrix is built.
//! 3. The objective is computed from that confusion matrix, and the per-fold
//!    score is the *improvement* over the parent (which predicts its own
//!    training-majority class to everyone):
//!    `gain = objective(split) − objective(parent)`.
//!
//! The confusion matrix is a sufficient statistic, so on the fast path it is
//! read straight out of the per-fold class histograms: validation counts are
//! the histogram cells, and the training counts needed for the class
//! assignment come from `total − fold` (the same subtraction trick the
//! histogram scorer already uses). Only two classes are ever predicted per
//! split, so scoring stays O(n_classes) per candidate per fold.
//!
//! Because the score is an *improvement over doing nothing*, objective mode is
//! naturally self-stopping: when no split improves the objective, the node
//! becomes a leaf. This tends to produce shallower, objective-tuned trees; set
//! `min_impurity_decrease` below 0 if you want to allow non-improving splits.

use crate::aggregation::Aggregator;
use crate::candidate::{present_states, Candidate};
use crate::cross_validation::{fold_train_val_codes, Fold, FoldStats};
use crate::data::{Bin, Column, SampleId, MISSING_BIN_CODE};
use crate::histogram::FastScratch;
use crate::selector::ScoredCandidate;

// ---------------------------------------------------------------------------
// Averaging + confusion matrix
// ---------------------------------------------------------------------------

/// How a per-class metric is reduced to a single number for multiclass data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Average {
    /// Score only `pos_label` (the usual binary setting).
    Binary {
        /// The class treated as "positive".
        pos_label: usize,
    },
    /// Pool TP/FP/FN across all classes, then compute the metric once.
    Micro,
    /// Unweighted mean of the per-class metric.
    Macro,
    /// Mean of the per-class metric weighted by class support.
    Weighted,
}

/// Per-class confusion counts accumulated on a validation fold.
///
/// Each "group" (a child, or the whole parent) predicts a single class to all
/// of its members; [`Confusion::add_group`] folds that into the matrix.
#[derive(Clone, Debug)]
pub struct Confusion {
    /// Number of classes.
    pub n_classes: usize,
    /// True positives per class.
    pub tp: Vec<u64>,
    /// False positives per class.
    pub fp: Vec<u64>,
    /// False negatives per class.
    pub fn_: Vec<u64>,
    /// Support (actual count) per class.
    pub support: Vec<u64>,
    /// Total samples counted.
    pub total: u64,
}

impl Confusion {
    /// A zeroed matrix for `n_classes` classes.
    pub fn zero(n_classes: usize) -> Self {
        Confusion {
            n_classes,
            tp: vec![0; n_classes],
            fp: vec![0; n_classes],
            fn_: vec![0; n_classes],
            support: vec![0; n_classes],
            total: 0,
        }
    }

    /// Fold in a group that predicts class `pred` to all its members, given the
    /// group's validation true-class counts.
    pub fn add_group(&mut self, pred: usize, val_counts: &[u64]) {
        for c in 0..self.n_classes {
            let cnt = val_counts[c];
            if cnt == 0 {
                continue;
            }
            self.support[c] += cnt;
            self.total += cnt;
            if c == pred {
                self.tp[pred] += cnt;
            } else {
                self.fn_[c] += cnt;
                if pred < self.n_classes {
                    self.fp[pred] += cnt;
                }
            }
        }
    }
}

fn safe_div(n: f64, d: f64) -> f64 {
    if d > 0.0 {
        n / d
    } else {
        0.0
    }
}

/// Argmax with a deterministic lowest-index tie-break.
fn argmax(counts: &[u64]) -> usize {
    let mut idx = 0usize;
    let mut best = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        if c > best {
            best = c;
            idx = i;
        }
    }
    idx
}

/// Reduce a per-class `(tp, fp, fn)` metric over the confusion matrix.
fn averaged(cm: &Confusion, avg: Average, f: &dyn Fn(u64, u64, u64) -> f64) -> f64 {
    match avg {
        Average::Binary { pos_label } => {
            if pos_label >= cm.n_classes {
                0.0
            } else {
                f(cm.tp[pos_label], cm.fp[pos_label], cm.fn_[pos_label])
            }
        }
        Average::Macro => {
            if cm.n_classes == 0 {
                return 0.0;
            }
            let mut s = 0.0;
            for k in 0..cm.n_classes {
                s += f(cm.tp[k], cm.fp[k], cm.fn_[k]);
            }
            s / cm.n_classes as f64
        }
        Average::Weighted => {
            if cm.total == 0 {
                return 0.0;
            }
            let mut s = 0.0;
            for k in 0..cm.n_classes {
                s += cm.support[k] as f64 * f(cm.tp[k], cm.fp[k], cm.fn_[k]);
            }
            s / cm.total as f64
        }
        Average::Micro => {
            let tp: u64 = cm.tp.iter().sum();
            let fp: u64 = cm.fp.iter().sum();
            let fnn: u64 = cm.fn_.iter().sum();
            f(tp, fp, fnn)
        }
    }
}

// ---------------------------------------------------------------------------
// The objective trait + built-in metrics
// ---------------------------------------------------------------------------

/// A classification objective: turn a validation confusion matrix into a score
/// (higher is better). Implement this to plug in a custom metric.
pub trait ClassObjective: Send + Sync {
    /// Score the confusion matrix (higher is better).
    fn score(&self, cm: &Confusion) -> f64;
    /// A short name (for diagnostics).
    fn name(&self) -> &'static str;
}

/// Precision, `TP / (TP + FP)`.
pub struct Precision {
    /// Averaging strategy.
    pub average: Average,
}
impl ClassObjective for Precision {
    fn score(&self, cm: &Confusion) -> f64 {
        averaged(cm, self.average, &|tp, fp, _fn| {
            safe_div(tp as f64, (tp + fp) as f64)
        })
    }
    fn name(&self) -> &'static str {
        "precision"
    }
}

/// Recall, `TP / (TP + FN)`.
pub struct Recall {
    /// Averaging strategy.
    pub average: Average,
}
impl ClassObjective for Recall {
    fn score(&self, cm: &Confusion) -> f64 {
        averaged(cm, self.average, &|tp, _fp, fnn| {
            safe_div(tp as f64, (tp + fnn) as f64)
        })
    }
    fn name(&self) -> &'static str {
        "recall"
    }
}

/// F1, the harmonic mean of precision and recall.
pub struct F1 {
    /// Averaging strategy.
    pub average: Average,
}
impl ClassObjective for F1 {
    fn score(&self, cm: &Confusion) -> f64 {
        averaged(cm, self.average, &|tp, fp, fnn| {
            safe_div(2.0 * tp as f64, (2 * tp + fp + fnn) as f64)
        })
    }
    fn name(&self) -> &'static str {
        "f1"
    }
}

/// Fβ: weights recall β² times as much as precision.
pub struct FBeta {
    /// The β weight.
    pub beta: f64,
    /// Averaging strategy.
    pub average: Average,
}
impl ClassObjective for FBeta {
    fn score(&self, cm: &Confusion) -> f64 {
        let b2 = self.beta * self.beta;
        averaged(cm, self.average, &|tp, fp, fnn| {
            let num = (1.0 + b2) * tp as f64;
            let den = (1.0 + b2) * tp as f64 + b2 * fnn as f64 + fp as f64;
            safe_div(num, den)
        })
    }
    fn name(&self) -> &'static str {
        "fbeta"
    }
}

/// Overall accuracy, `correct / total`.
pub struct Accuracy;
impl ClassObjective for Accuracy {
    fn score(&self, cm: &Confusion) -> f64 {
        let tp: u64 = cm.tp.iter().sum();
        safe_div(tp as f64, cm.total as f64)
    }
    fn name(&self) -> &'static str {
        "accuracy"
    }
}

// ---------------------------------------------------------------------------
// Fast (histogram) path
// ---------------------------------------------------------------------------

/// Score every `feature == bin` split for a classification node against a
/// metric objective, using the per-fold class histograms.
///
/// Mirrors [`crate::histogram::score_classif`] but produces objective gain
/// instead of impurity gain. `scratch` is accepted for interface parity with
/// the impurity fast path; this scorer manages its own buffers.
#[allow(clippy::too_many_arguments)]
pub fn score_classif_objective(
    codes: &[Bin],
    labels: &[usize],
    order: &[SampleId],
    val_fold: &[u8],
    k: usize,
    max_bin: usize,
    n_classes: usize,
    objective: &dyn ClassObjective,
    aggregator: &dyn Aggregator,
    feature: usize,
    _scratch: &mut FastScratch,
) -> Vec<ScoredCandidate> {
    let bins = max_bin;
    let c = n_classes;
    if bins == 0 || c == 0 || k == 0 {
        return Vec::new();
    }

    // Single scatter pass: per-fold validation class histograms.
    let mut hist = vec![0u64; k * bins * c]; // hist[f][b][cl]
    let mut tot = vec![0u64; k * c]; // per (fold, class) validation totals
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
        hist[f * bins * c + b * c + lab] += 1;
        tot[f * c + lab] += 1;
    }

    // Node-wide (all folds) counts, for deriving training counts by subtraction.
    let mut hbc = vec![0u64; bins * c]; // H[b][cl] = sum_f hist
    let mut htot = vec![0u64; c]; // Htot[cl]  = sum_f tot
    for f in 0..k {
        for b in 0..bins {
            for cl in 0..c {
                let v = hist[f * bins * c + b * c + cl];
                hbc[b * c + cl] += v;
                htot[cl] += v;
            }
        }
    }

    let mut vc_in = vec![0u64; c];
    let mut vc_rest = vec![0u64; c];
    let mut tc_in = vec![0u64; c];
    let mut tc_rest = vec![0u64; c];
    let mut par_train = vec![0u64; c];
    let mut par_val = vec![0u64; c];
    let mut out = Vec::new();

    for b in 0..bins {
        // Skip bins with no samples anywhere in the node.
        let mut present = 0u64;
        for f in 0..k {
            for cl in 0..c {
                present += hist[f * bins * c + b * c + cl];
            }
        }
        if present == 0 {
            continue;
        }

        let mut scores: Vec<f64> = Vec::with_capacity(k);
        for f in 0..k {
            let mut nf = 0u64;
            for cl in 0..c {
                nf += tot[f * c + cl];
            }
            if nf == 0 {
                continue;
            }

            let mut nl = 0u64;
            let mut nr = 0u64;
            for cl in 0..c {
                let hin = hist[f * bins * c + b * c + cl];
                let vrest = tot[f * c + cl] - hin;
                vc_in[cl] = hin;
                vc_rest[cl] = vrest;
                nl += hin;
                nr += vrest;

                let ptrain = htot[cl] - tot[f * c + cl]; // all training of fold f
                par_train[cl] = ptrain;
                par_val[cl] = tot[f * c + cl];
                let tin = hbc[b * c + cl] - hin; // training in-state
                tc_in[cl] = tin;
                tc_rest[cl] = ptrain - tin; // training rest
            }
            if nl == 0 || nr == 0 {
                continue; // degenerate on validation
            }
            let tin_tot: u64 = tc_in.iter().sum();
            let trest_tot: u64 = tc_rest.iter().sum();
            if tin_tot == 0 || trest_tot == 0 {
                continue; // no training basis to assign a child's class
            }

            let p_in = argmax(&tc_in);
            let p_rest = argmax(&tc_rest);
            let p_par = argmax(&par_train);

            let mut split_cm = Confusion::zero(c);
            split_cm.add_group(p_in, &vc_in);
            split_cm.add_group(p_rest, &vc_rest);

            let mut par_cm = Confusion::zero(c);
            par_cm.add_group(p_par, &par_val);

            scores.push(objective.score(&split_cm) - objective.score(&par_cm));
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

// ---------------------------------------------------------------------------
// Strict path
// ---------------------------------------------------------------------------

/// Score every candidate `state` of one feature against a metric objective on
/// the strict path (per-fold bin edges), returning one scored candidate per
/// present state.
#[allow(clippy::too_many_arguments)]
pub fn eval_feature_objective(
    columns: &[Column],
    labels: &[usize],
    feature: usize,
    indices: &[usize],
    folds: &[Fold],
    n_bins: usize,
    n_classes: usize,
    objective: &dyn ClassObjective,
    aggregator: &dyn Aggregator,
) -> Vec<ScoredCandidate> {
    let states = present_states(&columns[feature], indices, n_bins);
    if states.is_empty() {
        return Vec::new();
    }
    let mut per_state: Vec<Vec<f64>> = vec![Vec::with_capacity(folds.len()); states.len()];

    for fold in folds {
        if fold.val.is_empty() || fold.train.is_empty() {
            continue;
        }
        let (tcodes, vcodes) = fold_train_val_codes(&columns[feature], fold, n_bins);
        let ttargets: Vec<usize> = fold.train.iter().map(|&i| labels[i]).collect();
        let vtargets: Vec<usize> = fold.val.iter().map(|&i| labels[i]).collect();

        let mut train_total = vec![0u64; n_classes];
        for &t in &ttargets {
            if t < n_classes {
                train_total[t] += 1;
            }
        }
        let mut val_total = vec![0u64; n_classes];
        for &t in &vtargets {
            if t < n_classes {
                val_total[t] += 1;
            }
        }
        let val_n: u64 = val_total.iter().sum();

        let p_par = argmax(&train_total);
        let mut par_cm = Confusion::zero(n_classes);
        par_cm.add_group(p_par, &val_total);
        let par_score = objective.score(&par_cm);

        for (si, &state) in states.iter().enumerate() {
            let mut tin = vec![0u64; n_classes];
            for (kk, &code) in tcodes.iter().enumerate() {
                if code == state {
                    let t = ttargets[kk];
                    if t < n_classes {
                        tin[t] += 1;
                    }
                }
            }
            let mut vin = vec![0u64; n_classes];
            for (kk, &code) in vcodes.iter().enumerate() {
                if code == state {
                    let t = vtargets[kk];
                    if t < n_classes {
                        vin[t] += 1;
                    }
                }
            }
            let nl: u64 = vin.iter().sum();
            let nr = val_n - nl;
            if nl == 0 || nr == 0 {
                continue;
            }
            let tin_tot: u64 = tin.iter().sum();
            let ttrain: u64 = train_total.iter().sum();
            if tin_tot == 0 || ttrain - tin_tot == 0 {
                continue;
            }
            let trest: Vec<u64> = (0..n_classes).map(|c| train_total[c] - tin[c]).collect();
            let vrest: Vec<u64> = (0..n_classes).map(|c| val_total[c] - vin[c]).collect();
            let p_in = argmax(&tin);
            let p_rest = argmax(&trest);

            let mut split_cm = Confusion::zero(n_classes);
            split_cm.add_group(p_in, &vin);
            split_cm.add_group(p_rest, &vrest);
            per_state[si].push(objective.score(&split_cm) - par_score);
        }
    }

    states
        .iter()
        .zip(per_state)
        .map(|(&state, s)| {
            let stats = FoldStats::from_scores(s, folds.len());
            let score = aggregator.aggregate(&stats);
            ScoredCandidate {
                candidate: Candidate { feature, state },
                score,
                stats,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_recall_f1_from_confusion() {
        // 2 classes. Predict class 1 to a group that is 8x class1, 2x class0.
        let mut cm = Confusion::zero(2);
        cm.add_group(1, &[2, 8]); // predicts 1: tp1=8, fp1=2
        cm.add_group(0, &[5, 1]); // predicts 0: tp0=5, fp0=1, fn1 += 1
                                  // class1: tp=8, fp=2, fn=1  -> P=0.8, R=8/9
        let p = Precision {
            average: Average::Binary { pos_label: 1 },
        };
        let r = Recall {
            average: Average::Binary { pos_label: 1 },
        };
        assert!((p.score(&cm) - 0.8).abs() < 1e-9);
        assert!((r.score(&cm) - 8.0 / 9.0).abs() < 1e-9);
        let f1 = F1 {
            average: Average::Binary { pos_label: 1 },
        };
        let expect = 2.0 * 0.8 * (8.0 / 9.0) / (0.8 + 8.0 / 9.0);
        assert!((f1.score(&cm) - expect).abs() < 1e-9);
    }

    #[test]
    fn accuracy_counts_all_correct() {
        let mut cm = Confusion::zero(2);
        cm.add_group(0, &[7, 3]); // 7 correct
        cm.add_group(1, &[1, 9]); // 9 correct
        let a = Accuracy;
        assert!((a.score(&cm) - 16.0 / 20.0).abs() < 1e-9);
    }

    #[test]
    fn zero_division_is_zero() {
        let cm = Confusion::zero(2); // nothing predicted
        let p = Precision {
            average: Average::Macro,
        };
        assert_eq!(p.score(&cm), 0.0);
    }
}
