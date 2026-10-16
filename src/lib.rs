//! # Cross-Validated Decision Tree (CVDT)
//!
//! A from-scratch, modular decision tree whose novelty is *how splits are
//! scored*: instead of estimating impurity once on the in-sample node data,
//! every candidate split is evaluated with **K-fold cross-validation** and the
//! per-fold scores are aggregated into a single robust ranking value.
//!
//! ## Pipeline
//! ```text
//! data -> encoder -> candidate generator -> cross-validation -> criterion
//!      -> aggregation -> split selector -> tree builder -> prediction
//! ```
//!
//! Each layer depends only on the layer beneath it and is independently unit
//! tested. The tree builder merely *orchestrates* these components; it contains
//! no impurity, cross-validation or aggregation logic of its own.
//!
//! ## The split interface
//! Candidates are enumerated over quantile bins as `feature == state`, and
//! [`SplitStyle`] decides how a continuous `state` becomes a partition:
//! * continuous features are quantile-binned (edges fit **only on the training
//!   fold** to avoid leakage), so `state` is a bin id. With
//!   [`SplitStyle::Threshold`] (default) the cut is CART-style ordered
//!   (`x < edge`, left child = bins `0..=state`), plus an "x is present" cut
//!   when the node has missing values; with [`SplitStyle::BinMembership`] it
//!   is one bin vs. the rest;
//! * categorical features use their category id directly, so `state` is a
//!   category id (always tested by equality).
//!
//! ## Per-fold score
//! By default ([`Scoring::Honest`]) the parent and child models are fit on the
//! **training** fold and scored on the **validation** fold:
//! ```text
//! gain = (L_val(parent fit on train) - L_val(children fit on train)) / n_val
//! ```
//! with `L` the Brier score (Gini), log-loss (entropy), or squared / absolute
//! error (regression). This is the out-of-sample counterpart of impurity
//! decrease: identical units, but its expectation is `≤ 0` for a split that
//! carries no signal, removing the upward bias of in-sample scoring.
//! [`Scoring::InSample`] keeps the legacy `parent_impurity(val) −
//! weighted_child_impurity(val)`, which concavity makes `≥ 0` for any split.
//!
//! Honest scoring alone does **not** make `min_impurity_decrease = 0` a
//! reliable stopping rule: the tree keeps the *best* of many candidates, and
//! the maximum of many noisy near-zero scores is usually positive, so pure
//! noise still splits, most of all at small nodes. The permutation-null test
//! (`TreeParams::null_permutations`) accounts for that search and is the
//! stopping rule to use when spurious splits matter.
//! Aggregators such as mean, a lower quantile, or (moderated) `mean − λ·std`
//! collapse the per-fold gains into one ranking number.
//!
//! See the module docs for details and the `tests/` directory for end-to-end
//! examples.
//!
//! ## Python
//! An optional, scikit-learn-compatible Python binding lives behind the
//! `python` cargo feature (PyO3 + the numpy crate). It is off by default, so
//! the core crate stays zero-dependency; see `PYTHON.md`.
// The core forbids unsafe entirely. The `python` feature enables PyO3, whose
// macros expand to unsafe code, so the lint is relaxed only for that build.
#![cfg_attr(not(feature = "python"), forbid(unsafe_code))]

pub mod aggregation;
pub mod binning;
pub mod candidate;
pub mod criterion;
pub mod cross_validation;
pub mod data;
pub mod encoder;
pub mod histogram;
pub mod objective;
pub mod parallel;
pub mod selector;
pub mod tree;

/// scikit-learn-compatible Python binding (only with `--features python`).
#[cfg(feature = "python")]
mod python;

pub use aggregation::{
    Aggregator, Custom, Mean, MeanMinusLambdaStd, Median, ModeratedMeanMinusLambdaStd,
    ModeratedSignalToNoise, Quantile, SignalToNoise, TrimmedMean,
};
pub use binning::{BinnedData, FeatureBins};
pub use candidate::{generate_candidates, present_states, split_states, Candidate};
pub use criterion::{
    brier_loss_counts, log_loss_counts, sq_loss_moments, Criterion, Entropy, Gini, Mae, Mse,
    Variance, LOGLOSS_ALPHA,
};
pub use cross_validation::{eval_feature, Fold, FoldStats, KFold, Scoring, SplitMix64};
pub use data::{Bin, CatId, Column, Dataset, FeatureValue, SampleId, UNKNOWN_CAT};
pub use encoder::{bin_of, quantile_edges, MISSING_BIN};
pub use histogram::{score_classif, score_regr, ClassImpurityKind, FastScratch};
pub use objective::{
    eval_feature_objective, score_classif_objective, Accuracy, Average, ClassObjective, Confusion,
    FBeta, Precision, Recall, F1,
};
pub use selector::{select_best, ScoredCandidate};
pub use tree::{
    ClassPrediction, Classification, DecisionTree, ExportedNode, LeafInfo, Node,
    ObjectiveClassification, Regression, SplitMode, SplitRule, SplitStyle, Task, TreeParams,
};
