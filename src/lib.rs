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
//!   (`x < edge`, left child = bins `0..=state`); with
//!   [`SplitStyle::BinMembership`] it is one bin vs. the rest;
//! * categorical features use their category id directly, so `state` is a
//!   category id (always tested by equality).
//!
//! ## Per-fold score
//! For a candidate on a validation fold the score is the *impurity decrease*
//! ```text
//! gain = parent_impurity(val) - weighted_child_impurity(val)
//! ```
//! computed on the validation labels, where the partition of the validation
//! samples is produced by an encoder fit on the training fold. Higher is
//! better. Aggregators such as mean, median, or `mean - λ·std` collapse the
//! per-fold gains into one ranking number.
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
    Aggregator, Custom, Mean, MeanMinusLambdaStd, Median, SignalToNoise, TrimmedMean,
};
pub use binning::{BinnedData, FeatureBins};
pub use candidate::{generate_candidates, present_states, Candidate};
pub use criterion::{Criterion, Entropy, Gini, Mae, Mse, Variance};
pub use cross_validation::{eval_feature, Fold, FoldStats, KFold};
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
