//! Tree construction and prediction — the orchestration layer.
//!
//! This module wires the other components together and owns *only* the concerns
//! the design doc assigns to the tree builder: recursion, stopping criteria,
//! node creation, and traversal at prediction time. It contains no impurity,
//! cross-validation, aggregation or selection logic; each of those is delegated
//! to the module that owns it.
//!
//! Classification vs. regression is expressed with the [`Task`] trait rather
//! than an inheritance hierarchy: `Task` supplies the target type, the leaf
//! prediction, and the impurity criterion, and the generic [`DecisionTree`]
//! works for any task. This is the Rust-idiomatic rendering of the
//! `DecisionTreeBase -> {Classifier, Regressor}` design.

use crate::aggregation::{quantile_of, Aggregator, Mean};
use crate::binning::{BinnedData, FeatureBins};
use crate::candidate::{split_states, Candidate};
use crate::criterion::{Criterion, Entropy, Gini, Mae, Mse, Variance};
use crate::cross_validation::{eval_feature, Fold, FoldStats, KFold, Scoring, SplitMix64};
use crate::data::{Bin, Column, FeatureValue, SampleId, UNKNOWN_CAT};
use crate::encoder::{bin_bounds, bin_of, quantile_edges};
use crate::histogram::{score_classif, score_regr, ClassImpurityKind, FastScratch};
use crate::objective::{
    eval_feature_objective, score_classif_objective, Accuracy, Average, ClassObjective, FBeta,
    Precision, Recall, F1,
};
use crate::parallel::par_map;
use crate::selector::{select_best, ScoredCandidate};

// ---------------------------------------------------------------------------
// Split rules and nodes
// ---------------------------------------------------------------------------

/// A concrete routing rule stored in an internal node.
///
/// "Left" always means *matches the split state* (`feature == state`), matching
/// the convention used when the split was scored. Missing / non-finite values
/// never match, so they route right.
#[derive(Clone, Debug, PartialEq)]
pub enum SplitRule {
    /// Continuous feature falls in the half-open bin: lower inclusive, upper exclusive.
    /// `None` bounds are open ends (`-inf` / `+inf`). With both bounds open the
    /// rule is "x is present": every finite value routes left and missing /
    /// non-finite values route right (the threshold sweep's top-bin cut).
    ContinuousBin {
        /// Feature index.
        feature: usize,
        /// Inclusive lower bound, or `None` for `-inf`.
        lower: Option<f64>,
        /// Exclusive upper bound, or `None` for `+inf`.
        upper: Option<f64>,
    },
    /// Categorical feature equals a specific category id.
    Category {
        /// Feature index.
        feature: usize,
        /// Category id the feature is tested against.
        category: u32,
    },
}

impl SplitRule {
    /// Feature index this rule tests.
    pub fn feature(&self) -> usize {
        match self {
            SplitRule::ContinuousBin { feature, .. } => *feature,
            SplitRule::Category { feature, .. } => *feature,
        }
    }

    /// Whether training sample `i` routes left, reading directly from columns.
    pub fn route_left_train(&self, columns: &[Column], i: usize) -> bool {
        match self {
            SplitRule::ContinuousBin {
                feature,
                lower,
                upper,
            } => {
                if let Column::Continuous(v) = &columns[*feature] {
                    let x = v[i];
                    if !x.is_finite() {
                        return false;
                    }
                    lower.map_or(true, |l| x >= l) && upper.map_or(true, |u| x < u)
                } else {
                    false
                }
            }
            SplitRule::Category { feature, category } => {
                if let Column::Categorical(c) = &columns[*feature] {
                    c[i] == *category
                } else {
                    false
                }
            }
        }
    }

    /// Whether a prediction-time sample routes left.
    pub fn route_left_sample(&self, sample: &[FeatureValue]) -> bool {
        match self {
            SplitRule::ContinuousBin {
                feature,
                lower,
                upper,
            } => match sample[*feature] {
                FeatureValue::Continuous(x) => {
                    if !x.is_finite() {
                        return false;
                    }
                    lower.map_or(true, |l| x >= l) && upper.map_or(true, |u| x < u)
                }
                _ => false,
            },
            SplitRule::Category { feature, category } => match sample[*feature] {
                FeatureValue::Categorical(c) => c == *category,
                _ => false,
            },
        }
    }
}

/// A node of the fitted tree, generic over the prediction payload `P`.
#[derive(Clone, Debug)]
pub enum Node<P> {
    /// Terminal node carrying a prediction and the number of training samples
    /// that reached it.
    Leaf {
        /// The stored prediction (class + probabilities, or a regression value).
        prediction: P,
        /// Training samples that reached this leaf.
        n_samples: usize,
    },
    /// Decision node with a routing rule and two children.
    Internal {
        /// The routing rule.
        rule: SplitRule,
        /// Subtree for samples that match the rule (`feature == state`).
        left: Box<Node<P>>,
        /// Subtree for the rest.
        right: Box<Node<P>>,
    },
}

fn node_depth<P>(node: &Node<P>) -> usize {
    match node {
        Node::Leaf { .. } => 0,
        Node::Internal { left, right, .. } => 1 + node_depth(left).max(node_depth(right)),
    }
}

fn node_leaves<P>(node: &Node<P>) -> usize {
    match node {
        Node::Leaf { .. } => 1,
        Node::Internal { left, right, .. } => node_leaves(left) + node_leaves(right),
    }
}

fn predict_node<P: Clone>(node: &Node<P>, sample: &[FeatureValue]) -> P {
    match node {
        Node::Leaf { prediction, .. } => prediction.clone(),
        Node::Internal { rule, left, right } => {
            if rule.route_left_sample(sample) {
                predict_node(left, sample)
            } else {
                predict_node(right, sample)
            }
        }
    }
}

/// A task-agnostic view of a leaf's prediction, used for tree export.
#[derive(Clone, Debug)]
pub struct LeafInfo {
    /// Predicted class id (classification), else `None`.
    pub class: Option<usize>,
    /// Class probabilities (classification), else empty.
    pub proba: Vec<f64>,
    /// Predicted value (regression), else `None`.
    pub value: Option<f64>,
}

/// One node of a fitted tree, flattened for export/visualisation.
///
/// Children are given as indices into the returned `Vec` (`-1` for leaves).
/// "Left" is the child that *matches* the rule (condition true). Continuous
/// bounds use `-inf`/`+inf` for open ends; unused numeric fields are `NaN`/`-1`.
#[derive(Clone, Debug)]
pub struct ExportedNode {
    /// Node id (its index in the returned vector).
    pub id: usize,
    /// Whether this node is a leaf.
    pub is_leaf: bool,
    /// Index of the "condition true" child, or `-1` for a leaf.
    pub left: i64,
    /// Index of the "condition false" child, or `-1` for a leaf.
    pub right: i64,
    /// Feature tested (internal nodes), else `-1`.
    pub feature: i64,
    /// Whether the split is categorical.
    pub is_categorical: bool,
    /// Inclusive lower bound for a continuous split (`-inf` if open).
    pub lower: f64,
    /// Exclusive upper bound for a continuous split (`+inf` if open).
    pub upper: f64,
    /// Category id for a categorical split, else `-1`.
    pub category: i64,
    /// Training samples reaching this node.
    pub n_samples: usize,
    /// Predicted class at a leaf, else `-1`.
    pub class: i64,
    /// Predicted value at a regression leaf, else `NaN`.
    pub value: f64,
    /// Class probabilities at a classification leaf, else empty.
    pub proba: Vec<f64>,
}

impl ExportedNode {
    fn placeholder(id: usize) -> Self {
        ExportedNode {
            id,
            is_leaf: false,
            left: -1,
            right: -1,
            feature: -1,
            is_categorical: false,
            lower: f64::NAN,
            upper: f64::NAN,
            category: -1,
            n_samples: 0,
            class: -1,
            value: f64::NAN,
            proba: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tasks: classification and regression
// ---------------------------------------------------------------------------

/// A learning task: the target type, how a leaf summarises targets, and the
/// impurity criterion to score splits with.
///
/// `Send + Sync` is required so the tree can evaluate features in parallel.
pub trait Task: Send + Sync {
    /// Per-sample target type (`usize` class ids, or `f64` responses).
    type Target: Copy + Send + Sync;
    /// Prediction produced at a leaf.
    type Prediction: Clone;

    /// Summarise the targets that reached a leaf into a prediction.
    fn leaf(&self, targets: &[Self::Target]) -> Self::Prediction;

    /// Render a leaf prediction into a task-agnostic form for tree export.
    fn describe_leaf(&self, pred: &Self::Prediction) -> LeafInfo;

    /// The impurity criterion used to score candidate splits.
    fn criterion(&self) -> &dyn Criterion<Target = Self::Target>;

    /// Whether this task supports the histogram ("fast") path. Defaults to
    /// `true`; tasks whose criterion has no additive sufficient statistic
    /// (e.g. MAE) override this to `false`.
    fn fast_supported(&self) -> bool {
        true
    }

    /// Score every `feature == bin` candidate for one feature on the fast path.
    ///
    /// Implementations delegate to [`crate::histogram`]. `codes` are the
    /// feature's global bin codes; `order`/`val_fold` are the node samples and
    /// their validation-fold ids; `k` the fold count; `max_bin` the number of
    /// valid bins for the feature. `scratch` is reused to avoid per-call
    /// allocation.
    #[allow(clippy::too_many_arguments)]
    fn score_feature_fast(
        &self,
        codes: &[Bin],
        targets: &[Self::Target],
        order: &[SampleId],
        val_fold: &[u8],
        k: usize,
        max_bin: usize,
        feature: usize,
        aggregator: &dyn Aggregator,
        scratch: &mut FastScratch,
        prefix: bool,
        scoring: Scoring,
    ) -> Vec<ScoredCandidate>;

    /// Score every candidate `state` of one feature on the **strict** path.
    ///
    /// The default computes impurity gain via [`eval_feature`] and this task's
    /// [`Task::criterion`] — the standard behaviour. Tasks that select splits by
    /// something other than impurity (e.g. an objective metric) override this.
    #[allow(clippy::too_many_arguments)]
    fn score_feature_strict(
        &self,
        columns: &[Column],
        targets: &[Self::Target],
        feature: usize,
        indices: &[usize],
        folds: &[Fold],
        n_bins: usize,
        aggregator: &dyn Aggregator,
        prefix: bool,
        scoring: Scoring,
    ) -> Vec<ScoredCandidate> {
        let states = split_states(&columns[feature], indices, n_bins, prefix);
        if states.is_empty() {
            return Vec::new();
        }
        // min_child = 1 here; min_samples_leaf applies to the node-level
        // partition and is enforced by the tree (`retain_admissible`), not
        // per fold.
        let stats = eval_feature(
            columns,
            targets,
            feature,
            &states,
            folds,
            n_bins,
            self.criterion(),
            1,
            prefix,
            scoring,
        );
        states
            .iter()
            .zip(stats)
            .map(|(&state, st)| {
                let score = aggregator.aggregate(&st);
                ScoredCandidate {
                    candidate: Candidate { feature, state },
                    score,
                    stats: st,
                }
            })
            .collect()
    }

    /// Hierarchical shrinkage step (Agarwal et al., 2022): the shrunk
    /// prediction of a child given its parent's shrunk and raw predictions,
    ///
    /// `shrunk_child = shrunk_parent + (raw_child − raw_parent) / (1 + λ / n_parent)`.
    ///
    /// Applied root-to-leaf, a deep split on few samples only nudges the
    /// prediction instead of committing to it. The default returns `raw`
    /// unchanged (no shrinkage).
    fn shrink(
        &self,
        parent_shrunk: &Self::Prediction,
        parent_raw: &Self::Prediction,
        raw: &Self::Prediction,
        parent_n: usize,
        lambda: f64,
    ) -> Self::Prediction {
        let _ = (parent_shrunk, parent_raw, parent_n, lambda);
        raw.clone()
    }
}

/// Shrink class probabilities (see [`Task::shrink`]); clamps at zero,
/// renormalises and recomputes the argmax class (lowest id on ties).
fn shrink_class(
    parent_shrunk: &ClassPrediction,
    parent_raw: &ClassPrediction,
    raw: &ClassPrediction,
    parent_n: usize,
    lambda: f64,
) -> ClassPrediction {
    let w = 1.0 / (1.0 + lambda / parent_n.max(1) as f64);
    let mut proba: Vec<f64> = raw
        .proba
        .iter()
        .enumerate()
        .map(|(j, &p)| {
            let ps = parent_shrunk.proba.get(j).copied().unwrap_or(0.0);
            let pr = parent_raw.proba.get(j).copied().unwrap_or(0.0);
            (ps + w * (p - pr)).max(0.0)
        })
        .collect();
    let total: f64 = proba.iter().sum();
    if total > 0.0 {
        for p in proba.iter_mut() {
            *p /= total;
        }
    } else {
        proba = parent_shrunk.proba.clone();
    }
    let mut class = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &p) in proba.iter().enumerate() {
        if p > best {
            best = p;
            class = i;
        }
    }
    ClassPrediction { class, proba }
}

/// Prediction returned by a classifier: the chosen class and the class
/// probabilities (fraction of leaf training samples per class).
#[derive(Clone, Debug, PartialEq)]
pub struct ClassPrediction {
    /// Predicted class id (argmax of `proba`, lowest id on ties).
    pub class: usize,
    /// Probability per class, indexed by class id.
    pub proba: Vec<f64>,
}

/// Build a leaf prediction (class probabilities + argmax class) from the class
/// ids that reached the leaf. Shared by every classification task.
fn class_leaf(n_classes: usize, targets: &[usize]) -> ClassPrediction {
    let mut counts = vec![0u64; n_classes];
    for &t in targets {
        if t < counts.len() {
            counts[t] += 1;
        }
    }
    let total: u64 = counts.iter().sum();
    let proba: Vec<f64> = if total == 0 {
        vec![0.0; n_classes]
    } else {
        counts.iter().map(|&c| c as f64 / total as f64).collect()
    };
    // Argmax with lowest-id tie-break (explicit loop, deterministic).
    let mut class = 0usize;
    let mut best = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        if c > best {
            best = c;
            class = i;
        }
    }
    ClassPrediction { class, proba }
}

/// Classification task.
pub struct Classification {
    /// Number of classes.
    pub n_classes: usize,
    /// Impurity criterion over class ids.
    pub criterion: Box<dyn Criterion<Target = usize>>,
    /// Which impurity the fast path computes from class counts.
    pub kind: ClassImpurityKind,
}

impl Classification {
    /// Classifier using Gini impurity.
    pub fn gini(n_classes: usize) -> Self {
        Classification {
            n_classes,
            criterion: Box::new(Gini::new(n_classes)),
            kind: ClassImpurityKind::Gini,
        }
    }

    /// Classifier using Shannon entropy.
    pub fn entropy(n_classes: usize) -> Self {
        Classification {
            n_classes,
            criterion: Box::new(Entropy::new(n_classes)),
            kind: ClassImpurityKind::Entropy,
        }
    }
}

impl Task for Classification {
    type Target = usize;
    type Prediction = ClassPrediction;

    fn leaf(&self, targets: &[usize]) -> ClassPrediction {
        class_leaf(self.n_classes, targets)
    }

    fn describe_leaf(&self, pred: &ClassPrediction) -> LeafInfo {
        LeafInfo {
            class: Some(pred.class),
            proba: pred.proba.clone(),
            value: None,
        }
    }

    fn criterion(&self) -> &dyn Criterion<Target = usize> {
        self.criterion.as_ref()
    }

    fn score_feature_fast(
        &self,
        codes: &[Bin],
        targets: &[usize],
        order: &[SampleId],
        val_fold: &[u8],
        k: usize,
        max_bin: usize,
        feature: usize,
        aggregator: &dyn Aggregator,
        scratch: &mut FastScratch,
        prefix: bool,
        scoring: Scoring,
    ) -> Vec<ScoredCandidate> {
        score_classif(
            codes,
            targets,
            order,
            val_fold,
            k,
            max_bin,
            self.n_classes,
            self.kind,
            aggregator,
            feature,
            scratch,
            prefix,
            scoring,
        )
    }

    fn shrink(
        &self,
        parent_shrunk: &ClassPrediction,
        parent_raw: &ClassPrediction,
        raw: &ClassPrediction,
        parent_n: usize,
        lambda: f64,
    ) -> ClassPrediction {
        shrink_class(parent_shrunk, parent_raw, raw, parent_n, lambda)
    }
}

/// Classification task that selects splits by a **metric objective**
/// (precision, recall, F1, Fβ, accuracy) evaluated on held-out folds, instead
/// of an impurity proxy. See [`crate::objective`].
///
/// Splits are accepted only when they *improve* the objective over making the
/// node a leaf, so trees tend to be shallower and tuned to the target metric.
pub struct ObjectiveClassification {
    /// Number of classes.
    pub n_classes: usize,
    /// The objective to greedily optimise.
    pub objective: Box<dyn ClassObjective>,
    // Kept only for the pure-node stopping check and the `criterion()` contract;
    // it plays no part in split *scoring*.
    gini: Gini,
}

impl ObjectiveClassification {
    /// Build from any [`ClassObjective`].
    pub fn new(n_classes: usize, objective: Box<dyn ClassObjective>) -> Self {
        ObjectiveClassification {
            n_classes,
            objective,
            gini: Gini::new(n_classes),
        }
    }

    /// Optimise F1.
    pub fn f1(n_classes: usize, average: Average) -> Self {
        Self::new(n_classes, Box::new(F1 { average }))
    }
    /// Optimise precision.
    pub fn precision(n_classes: usize, average: Average) -> Self {
        Self::new(n_classes, Box::new(Precision { average }))
    }
    /// Optimise recall.
    pub fn recall(n_classes: usize, average: Average) -> Self {
        Self::new(n_classes, Box::new(Recall { average }))
    }
    /// Optimise Fβ.
    pub fn fbeta(n_classes: usize, beta: f64, average: Average) -> Self {
        Self::new(n_classes, Box::new(FBeta { beta, average }))
    }
    /// Optimise accuracy.
    pub fn accuracy(n_classes: usize) -> Self {
        Self::new(n_classes, Box::new(Accuracy))
    }
}

impl Task for ObjectiveClassification {
    type Target = usize;
    type Prediction = ClassPrediction;

    fn leaf(&self, targets: &[usize]) -> ClassPrediction {
        class_leaf(self.n_classes, targets)
    }

    fn describe_leaf(&self, pred: &ClassPrediction) -> LeafInfo {
        LeafInfo {
            class: Some(pred.class),
            proba: pred.proba.clone(),
            value: None,
        }
    }

    fn criterion(&self) -> &dyn Criterion<Target = usize> {
        &self.gini
    }

    fn fast_supported(&self) -> bool {
        true
    }

    fn score_feature_fast(
        &self,
        codes: &[Bin],
        targets: &[usize],
        order: &[SampleId],
        val_fold: &[u8],
        k: usize,
        max_bin: usize,
        feature: usize,
        aggregator: &dyn Aggregator,
        scratch: &mut FastScratch,
        prefix: bool,
        _scoring: Scoring,
    ) -> Vec<ScoredCandidate> {
        // Objective scoring is already honest: child classes are assigned
        // from training-fold counts and the metric is measured on validation.
        score_classif_objective(
            codes,
            targets,
            order,
            val_fold,
            k,
            max_bin,
            self.n_classes,
            self.objective.as_ref(),
            aggregator,
            feature,
            scratch,
            prefix,
        )
    }

    fn score_feature_strict(
        &self,
        columns: &[Column],
        targets: &[usize],
        feature: usize,
        indices: &[usize],
        folds: &[Fold],
        n_bins: usize,
        aggregator: &dyn Aggregator,
        prefix: bool,
        _scoring: Scoring,
    ) -> Vec<ScoredCandidate> {
        eval_feature_objective(
            columns,
            targets,
            feature,
            indices,
            folds,
            n_bins,
            self.n_classes,
            self.objective.as_ref(),
            aggregator,
            prefix,
        )
    }

    fn shrink(
        &self,
        parent_shrunk: &ClassPrediction,
        parent_raw: &ClassPrediction,
        raw: &ClassPrediction,
        parent_n: usize,
        lambda: f64,
    ) -> ClassPrediction {
        shrink_class(parent_shrunk, parent_raw, raw, parent_n, lambda)
    }
}

/// Regression task.
pub struct Regression {
    /// Impurity criterion over responses.
    pub criterion: Box<dyn Criterion<Target = f64>>,
}

impl Regression {
    /// Regressor using mean squared error.
    pub fn mse() -> Self {
        Regression {
            criterion: Box::new(Mse),
        }
    }

    /// Regressor using variance (numerically identical to MSE).
    pub fn variance() -> Self {
        Regression {
            criterion: Box::new(Variance),
        }
    }

    /// Regressor using mean absolute error.
    pub fn mae() -> Self {
        Regression {
            criterion: Box::new(Mae),
        }
    }
}

impl Task for Regression {
    type Target = f64;
    type Prediction = f64;

    fn leaf(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        targets.iter().sum::<f64>() / targets.len() as f64
    }

    fn describe_leaf(&self, pred: &f64) -> LeafInfo {
        LeafInfo {
            class: None,
            proba: Vec::new(),
            value: Some(*pred),
        }
    }

    fn criterion(&self) -> &dyn Criterion<Target = f64> {
        self.criterion.as_ref()
    }

    fn fast_supported(&self) -> bool {
        // MAE's median has no additive sufficient statistic; only the
        // variance/MSE moment form works on the histogram path.
        self.criterion.name() != "mae"
    }

    fn score_feature_fast(
        &self,
        codes: &[Bin],
        targets: &[f64],
        order: &[SampleId],
        val_fold: &[u8],
        k: usize,
        max_bin: usize,
        feature: usize,
        aggregator: &dyn Aggregator,
        scratch: &mut FastScratch,
        prefix: bool,
        scoring: Scoring,
    ) -> Vec<ScoredCandidate> {
        score_regr(
            codes, targets, order, val_fold, k, max_bin, aggregator, feature, scratch, prefix,
            scoring,
        )
    }

    fn shrink(
        &self,
        parent_shrunk: &f64,
        parent_raw: &f64,
        raw: &f64,
        parent_n: usize,
        lambda: f64,
    ) -> f64 {
        let w = 1.0 / (1.0 + lambda / parent_n.max(1) as f64);
        parent_shrunk + w * (raw - parent_raw)
    }
}

// ---------------------------------------------------------------------------
// Hyper-parameters
// ---------------------------------------------------------------------------

/// Split-evaluation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    /// Exact path: continuous bin edges are refit on each training fold (the
    /// strongest leakage guard), child impurities computed from materialised
    /// target slices. This is the default and matches v1 behaviour.
    Strict,
    /// Histogram path: features are globally binned once, all K validation
    /// histograms are built in a single pass per node, and splits are scored
    /// from sufficient statistics. Much faster; trades away per-fold edge
    /// refitting (a feature-only leak — impurity is still measured on held-out
    /// labels). MAE regression is not supported here.
    Fast,
}

/// How continuous features are partitioned at a split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitStyle {
    /// CART-style ordered threshold: left child is `x < edge`, right is the rest
    /// (a contiguous, order-respecting partition). Candidate cuts are the
    /// quantile bin boundaries; the cross-validated score is still computed on
    /// held-out folds. This is the default.
    Threshold,
    /// One quantile bin vs. all others (`x in [lo, hi)`), the original CVDT
    /// behaviour. Rarely better for ordered features; kept for comparison.
    BinMembership,
}

/// Tree hyper-parameters.
#[derive(Clone, Debug)]
pub struct TreeParams {
    /// Maximum number of split levels; `None` for unlimited.
    pub max_depth: Option<usize>,
    /// Minimum samples a node must have to be eligible to split.
    pub min_samples_split: usize,
    /// Minimum samples each child must receive. Candidates that would leave a
    /// smaller child are excluded before the best split is chosen, so this
    /// restricts *which* split is taken rather than vetoing the winner.
    /// Defaults to 30 (the usual n ≥ 30 rule of thumb for a stable mean or
    /// proportion), which also stops honest scoring from chasing noise at
    /// tiny nodes; lower it for small datasets.
    pub min_samples_leaf: usize,
    /// Minimum aggregated score for a split to be accepted (must be exceeded).
    pub min_impurity_decrease: f64,
    /// Number of quantile bins for continuous features.
    pub n_bins: usize,
    /// Cross-validation configuration used to score splits.
    pub cv: KFold,
    /// Split-evaluation strategy.
    pub mode: SplitMode,
    /// How continuous features are partitioned (threshold vs single-bin).
    pub split_style: SplitStyle,
    /// Whether to evaluate features in parallel.
    pub parallel: bool,
    /// Worker-thread count when `parallel` is set.
    pub n_threads: usize,
    /// Only parallelise nodes with at least this many samples (spawning threads
    /// for tiny nodes costs more than it saves).
    pub parallel_min_samples: usize,
    /// How a candidate is scored on a fold. [`Scoring::Honest`] (default)
    /// measures training-fit loss on held-out data, so a split with no signal
    /// scores `≤ 0` in expectation. The *best* of many such candidates is
    /// still usually positive, so on its own this does not stop noise splits
    /// (least of all at small nodes); combine with `null_permutations` and/or
    /// `min_samples_leaf` for that.
    /// [`Scoring::InSample`] reproduces ≤ 0.7 behaviour.
    pub scoring: Scoring,
    /// Number of independently shuffled K-fold rounds per node (repeated
    /// K-fold). Each candidate's fold scores are pooled across rounds, giving
    /// `k · cv_repeats` scores and a more stable spread estimate for
    /// variance-aware aggregators (repeats reuse the same samples, so the
    /// pooled scores are correlated, not independent). `1` = plain K-fold.
    /// Cost scales linearly.
    pub cv_repeats: usize,
    /// A candidate must produce a valid score on at least this many folds
    /// (capped at the number attempted) or it is discarded.
    pub min_valid_folds: usize,
    /// Permutation-null stopping test: if `> 0`, the node's targets are
    /// permuted this many times, every candidate is rescored, and the split
    /// is accepted only if the real best score exceeds the `null_quantile`
    /// quantile of the permuted best scores. Adapts automatically to node
    /// size and to how many candidates were searched (the max over candidates
    /// is what overfits), which makes it the stopping rule that accounts for
    /// the search; 19 permutations with `null_quantile = 0.95` is a reasonable
    /// start. `0` = off. Costs `null_permutations` extra node scorings (only at
    /// nodes whose best score already passes `min_impurity_decrease`) plus an
    /// O(n_total) target copy each.
    pub null_permutations: usize,
    /// Quantile of the permutation null the best score must beat (e.g. 0.95).
    pub null_quantile: f64,
    /// Hierarchical-shrinkage strength λ (Agarwal et al., 2022) applied to
    /// leaf predictions; `0` = off. Typical values 1–100, tune by CV.
    pub shrinkage: f64,
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams {
            max_depth: Some(8),
            min_samples_split: 2,
            min_samples_leaf: 30,
            min_impurity_decrease: 0.0,
            n_bins: 8,
            cv: KFold {
                k: 5,
                seed: 42,
                shuffle: true,
            },
            mode: SplitMode::Strict,
            split_style: SplitStyle::Threshold,
            parallel: false,
            n_threads: 1,
            parallel_min_samples: 512,
            scoring: Scoring::Honest,
            cv_repeats: 1,
            min_valid_folds: 2,
            null_permutations: 0,
            null_quantile: 0.95,
            shrinkage: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

/// Turn a winning `(feature, state)` into a concrete routing rule using
/// node-level encoding. Returns `None` when the state cannot be realised
/// (unknown category, or bin id out of range for the node's edges).
fn make_rule(
    col: &Column,
    feature: usize,
    state: u32,
    indices: &[usize],
    n_bins: usize,
    prefix: bool,
) -> Option<SplitRule> {
    match col {
        Column::Continuous(v) => {
            let vals: Vec<f64> = indices.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&vals, n_bins);
            let (lower, upper) = bin_bounds(&edges, state as usize)?;
            if prefix {
                // Threshold split: left child is `x < edge`. The top bin has no
                // finite upper edge; its cut is "x is present" (both bounds
                // open: every finite value left, missing right), which
                // `split_states` only offers when the node has missing values.
                Some(SplitRule::ContinuousBin {
                    feature,
                    lower: None,
                    upper,
                })
            } else {
                Some(SplitRule::ContinuousBin {
                    feature,
                    lower,
                    upper,
                })
            }
        }
        Column::Categorical(_) => {
            if state == UNKNOWN_CAT {
                None
            } else {
                Some(SplitRule::Category {
                    feature,
                    category: state,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Node scoring shared by both build paths
// ---------------------------------------------------------------------------

/// Ancestor state threaded down the recursion for hierarchical shrinkage.
struct Ancestor<P> {
    shrunk: P,
    raw: P,
    n: usize,
}

fn shrunk_prediction<K: Task>(
    task: &K,
    params: &TreeParams,
    anc: Option<&Ancestor<K::Prediction>>,
    raw: &K::Prediction,
) -> K::Prediction {
    match anc {
        Some(a) if params.shrinkage > 0.0 => {
            task.shrink(&a.shrunk, &a.raw, raw, a.n, params.shrinkage)
        }
        _ => raw.clone(),
    }
}

/// The K-fold configuration for repeat `rep`. Repeat 0 is the configured one,
/// so `cv_repeats = 1` reproduces single K-fold exactly.
fn repeat_cv(cv: &KFold, rep: usize) -> KFold {
    let mut c = cv.clone();
    if rep > 0 {
        c.seed = cv.seed ^ (rep as u64).wrapping_mul(0xA24B_AED4_963E_E407);
    }
    c
}

/// Fold another repeat's candidates into `acc`, concatenating fold scores of
/// matching `(feature, state)` candidates.
fn merge_repeat(acc: &mut Vec<ScoredCandidate>, more: Vec<ScoredCandidate>) {
    use std::collections::BTreeMap;
    let mut pos: BTreeMap<(usize, u32), usize> = BTreeMap::new();
    for (i, c) in acc.iter().enumerate() {
        pos.insert((c.candidate.feature, c.candidate.state), i);
    }
    for c in more {
        let key = (c.candidate.feature, c.candidate.state);
        match pos.get(&key) {
            Some(&i) => {
                let merged = FoldStats::merged(&[&acc[i].stats, &c.stats]);
                acc[i].stats = merged;
            }
            None => {
                pos.insert(key, acc.len());
                acc.push(c);
            }
        }
    }
}

/// Final node-level aggregation: lets the aggregator see every candidate at
/// once (for variance moderation) and enforces `min_valid_folds`.
fn finalize_scores(aggregator: &dyn Aggregator, params: &TreeParams, scored: &mut [ScoredCandidate]) {
    let scores = {
        let refs: Vec<&FoldStats> = scored.iter().map(|c| &c.stats).collect();
        aggregator.aggregate_node(&refs)
    };
    for (c, sc) in scored.iter_mut().zip(scores) {
        let need = params.min_valid_folds.min(c.stats.n_total).max(1);
        c.score = if c.stats.n_success < need {
            f64::NEG_INFINITY
        } else {
            sc
        };
    }
}

/// Copy of `targets` with the values at `node` randomly permuted among
/// themselves (the permutation null for one node).
fn permuted_targets<T: Copy>(targets: &[T], node: &[usize], seed: u64) -> Vec<T> {
    let mut out = targets.to_vec();
    let mut vals: Vec<T> = node.iter().map(|&i| targets[i]).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..vals.len()).rev() {
        let j = rng.below((i as u64) + 1) as usize;
        vals.swap(i, j);
    }
    for (&i, v) in node.iter().zip(vals) {
        out[i] = v;
    }
    out
}

fn null_seed(params: &TreeParams, n: usize, p: usize) -> u64 {
    params.cv.seed
        ^ 0xD1B5_4A32_D192_ED03
        ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((p as u64) + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

fn best_score(scored: &[ScoredCandidate]) -> f64 {
    select_best(scored).map_or(f64::NEG_INFINITY, |b| b.score)
}

/// Run the permutation null: the `null_quantile` quantile of the best score
/// achievable on permuted targets. `score_fn` rescores the node.
fn null_threshold<T: Copy, F: Fn(&[T]) -> Vec<ScoredCandidate>>(
    params: &TreeParams,
    targets: &[T],
    node: &[usize],
    score_fn: F,
) -> f64 {
    let mut maxima: Vec<f64> = (0..params.null_permutations)
        .map(|p| {
            let t2 = permuted_targets(targets, node, null_seed(params, node.len(), p));
            best_score(&score_fn(&t2[..]))
        })
        .collect();
    quantile_of(&mut maxima, params.null_quantile)
}

/// Drop candidates whose node-level partition would leave a child with fewer
/// than `min_leaf` samples, so `min_samples_leaf` restricts which split is
/// *chosen*. (Checking only the winner would turn the node into a leaf even
/// when an admissible split exists.) `codes(f)` gives the node samples' codes
/// for feature `f` in the space of candidate states, exactly as the node will
/// be partitioned; `prefix(f)` says whether `f` is cut as a threshold
/// (`code <= state`) rather than by equality. Missing codes exceed every
/// state, so they count toward the right child, as they are routed.
fn retain_admissible<P, C>(
    all: &mut Vec<ScoredCandidate>,
    n: usize,
    min_leaf: usize,
    prefix: P,
    codes: C,
) where
    P: Fn(usize) -> bool,
    C: Fn(usize) -> Vec<u32>,
{
    use std::collections::BTreeMap;
    if min_leaf <= 1 {
        // Empty children are already excluded by the candidate generators.
        return;
    }
    let mut sorted: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    all.retain(|c| {
        let f = c.candidate.feature;
        let s = c.candidate.state;
        let v = sorted.entry(f).or_insert_with(|| {
            let mut v = codes(f);
            v.sort_unstable();
            v
        });
        let upto = v.partition_point(|&x| x <= s);
        let left = if prefix(f) {
            upto
        } else {
            upto - v.partition_point(|&x| x < s)
        };
        left >= min_leaf && n - left >= min_leaf
    });
}

/// Node-level codes of `indices` for one column, as [`make_rule`] partitions
/// them: continuous values binned on the node's own quantile edges (so
/// `x < edges[state]` is `code <= state`), categorical values as their ids.
fn node_codes(col: &Column, indices: &[usize], n_bins: usize) -> Vec<u32> {
    match col {
        Column::Continuous(v) => {
            let vals: Vec<f64> = indices.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&vals, n_bins);
            vals.iter().map(|&x| bin_of(&edges, x)).collect()
        }
        Column::Categorical(c) => indices.iter().map(|&i| c[i]).collect(),
    }
}

/// Score every candidate at a strict-path node (all features, all repeats).
fn score_node_strict<K: Task>(
    task: &K,
    params: &TreeParams,
    aggregator: &dyn Aggregator,
    columns: &[Column],
    targets: &[K::Target],
    indices: &[usize],
) -> Vec<ScoredCandidate> {
    let feature_ids: Vec<usize> = (0..columns.len()).collect();
    let mut all: Vec<ScoredCandidate> = Vec::new();
    for rep in 0..params.cv_repeats.max(1) {
        let folds = repeat_cv(&params.cv, rep).folds(indices);
        let eval_one = |f: &usize| -> Vec<ScoredCandidate> {
            let prefix = params.split_style == SplitStyle::Threshold
                && matches!(columns[*f], Column::Continuous(_));
            task.score_feature_strict(
                columns,
                targets,
                *f,
                indices,
                &folds,
                params.n_bins,
                aggregator,
                prefix,
                params.scoring,
            )
        };
        let scored: Vec<ScoredCandidate> = if params.parallel && params.n_threads > 1 {
            par_map(&feature_ids, params.n_threads, eval_one)
                .into_iter()
                .flatten()
                .collect()
        } else {
            feature_ids.iter().flat_map(|f| eval_one(f)).collect()
        };
        if rep == 0 {
            all = scored;
        } else {
            merge_repeat(&mut all, scored);
        }
    }
    retain_admissible(
        &mut all,
        indices.len(),
        params.min_samples_leaf,
        |f| {
            params.split_style == SplitStyle::Threshold
                && matches!(columns[f], Column::Continuous(_))
        },
        |f| node_codes(&columns[f], indices, params.n_bins),
    );
    finalize_scores(aggregator, params, &mut all);
    all
}

/// Score every candidate at a fast-path node (all features, all repeats).
fn score_node_fast<K: Task>(
    task: &K,
    params: &TreeParams,
    aggregator: &dyn Aggregator,
    binned: &BinnedData,
    targets: &[K::Target],
    samples: &[SampleId],
) -> Vec<ScoredCandidate> {
    let n = samples.len();
    let idx_usize: Vec<usize> = samples.iter().map(|&s| s as usize).collect();
    let feature_ids: Vec<usize> = (0..binned.n_features()).collect();
    let mut all: Vec<ScoredCandidate> = Vec::new();
    for rep in 0..params.cv_repeats.max(1) {
        let (order, val_fold, k) = repeat_cv(&params.cv, rep).assign(&idx_usize);
        if k == 0 {
            return Vec::new();
        }
        let eval_with = |feature: usize, scratch: &mut FastScratch| -> Vec<ScoredCandidate> {
            let prefix =
                params.split_style == SplitStyle::Threshold && binned.bins[feature].is_continuous;
            task.score_feature_fast(
                &binned.codes[feature],
                targets,
                &order,
                &val_fold,
                k,
                binned.bins[feature].max_bin,
                feature,
                aggregator,
                scratch,
                prefix,
                params.scoring,
            )
        };
        let scored: Vec<ScoredCandidate> =
            if params.parallel && params.n_threads > 1 && n >= params.parallel_min_samples {
                par_map(&feature_ids, params.n_threads, |f: &usize| {
                    let mut scratch = FastScratch::new();
                    eval_with(*f, &mut scratch)
                })
                .into_iter()
                .flatten()
                .collect()
            } else {
                // Serial: reuse a single scratch across features.
                let mut scratch = FastScratch::new();
                let mut v = Vec::new();
                for &f in &feature_ids {
                    v.extend(eval_with(f, &mut scratch));
                }
                v
            };
        if rep == 0 {
            all = scored;
        } else {
            merge_repeat(&mut all, scored);
        }
    }
    retain_admissible(
        &mut all,
        n,
        params.min_samples_leaf,
        |f| params.split_style == SplitStyle::Threshold && binned.bins[f].is_continuous,
        |f| samples.iter().map(|&s| binned.codes[f][s as usize] as u32).collect(),
    );
    finalize_scores(aggregator, params, &mut all);
    all
}

/// Smallest node that can be split at all: below `2 · min_samples_leaf` every
/// partition leaves a child too small, so the node is a leaf without scoring.
fn min_splittable(params: &TreeParams) -> usize {
    params
        .min_samples_split
        .max(2)
        .max(params.min_samples_leaf.max(1).saturating_mul(2))
}

#[allow(clippy::too_many_arguments)]
fn build_node<K: Task>(
    task: &K,
    params: &TreeParams,
    aggregator: &dyn Aggregator,
    columns: &[Column],
    targets: &[K::Target],
    indices: &[usize],
    depth: usize,
    anc: Option<&Ancestor<K::Prediction>>,
) -> Node<K::Prediction> {
    let node_targets: Vec<K::Target> = indices.iter().map(|&i| targets[i]).collect();
    let raw = task.leaf(&node_targets);
    let prediction = shrunk_prediction(task, params, anc, &raw);
    let leaf = |pred: K::Prediction| Node::Leaf {
        prediction: pred,
        n_samples: indices.len(),
    };

    // --- cheap stopping criteria -------------------------------------------
    if indices.len() < min_splittable(params) {
        return leaf(prediction);
    }
    if let Some(md) = params.max_depth {
        if depth >= md {
            return leaf(prediction);
        }
    }
    if task.criterion().impurity(&node_targets) <= 0.0 {
        // Pure (or degenerate) node: nothing to gain.
        return leaf(prediction);
    }

    // --- score every candidate split via cross-validation ------------------
    let scored = score_node_strict(task, params, aggregator, columns, targets, indices);

    // --- select and validate the winner ------------------------------------
    let best = match select_best(&scored) {
        Some(b) => b,
        None => return leaf(prediction),
    };
    if best.score <= params.min_impurity_decrease {
        return leaf(prediction);
    }
    if params.null_permutations > 0 {
        let thr = null_threshold(params, targets, indices, |t: &[K::Target]| {
            score_node_strict(task, params, aggregator, columns, t, indices)
        });
        if best.score <= thr {
            return leaf(prediction);
        }
    }

    let rule = match make_rule(
        &columns[best.candidate.feature],
        best.candidate.feature,
        best.candidate.state,
        indices,
        params.n_bins,
        params.split_style == SplitStyle::Threshold
            && matches!(columns[best.candidate.feature], Column::Continuous(_)),
    ) {
        Some(r) => r,
        None => return leaf(prediction),
    };

    // --- partition (min_samples_leaf was applied to the candidates; this ----
    // --- check is a safety net) ---------------------------------------------
    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for &i in indices {
        if rule.route_left_train(columns, i) {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    if left_idx.len() < params.min_samples_leaf.max(1)
        || right_idx.len() < params.min_samples_leaf.max(1)
    {
        return leaf(prediction);
    }

    // --- recurse (tree growth is sequential by design) ---------------------
    let me = Ancestor {
        shrunk: prediction,
        raw,
        n: indices.len(),
    };
    let left = build_node(
        task,
        params,
        aggregator,
        columns,
        targets,
        &left_idx,
        depth + 1,
        Some(&me),
    );
    let right = build_node(
        task,
        params,
        aggregator,
        columns,
        targets,
        &right_idx,
        depth + 1,
        Some(&me),
    );
    Node::Internal {
        rule,
        left: Box::new(left),
        right: Box::new(right),
    }
}

// ---------------------------------------------------------------------------
// Fast (histogram) build path
// ---------------------------------------------------------------------------

/// Turn a winning `(feature, bin)` on the fast path into a routing rule using
/// the global bins. Continuous bins become half-open bounds; categorical dense
/// ids map back to the original category id.
fn make_rule_fast(fb: &FeatureBins, feature: usize, state: u32, prefix: bool) -> Option<SplitRule> {
    if fb.is_continuous {
        let (lower, upper) = fb.cont_bounds(state as usize)?;
        if prefix {
            // `upper == None` is the top bin: the "x is present" cut, which
            // the fast scorers only offer when the node has missing values.
            Some(SplitRule::ContinuousBin {
                feature,
                lower: None,
                upper,
            })
        } else {
            Some(SplitRule::ContinuousBin {
                feature,
                lower,
                upper,
            })
        }
    } else {
        let category = fb.category(state as usize)?;
        Some(SplitRule::Category { feature, category })
    }
}

/// In-place Lomuto-style partition: move all samples satisfying `pred` to the
/// front and return the boundary index. The shared buffer is reordered so the
/// two children are contiguous sub-slices, avoiding a fresh allocation per node.
fn partition_in_place<F: FnMut(SampleId) -> bool>(samples: &mut [SampleId], mut pred: F) -> usize {
    let mut i = 0;
    for j in 0..samples.len() {
        if pred(samples[j]) {
            samples.swap(i, j);
            i += 1;
        }
    }
    i
}

#[allow(clippy::too_many_arguments)]
fn build_node_fast<K: Task>(
    task: &K,
    params: &TreeParams,
    aggregator: &dyn Aggregator,
    binned: &BinnedData,
    targets: &[K::Target],
    samples: &mut [SampleId],
    depth: usize,
    anc: Option<&Ancestor<K::Prediction>>,
) -> Node<K::Prediction> {
    let node_targets: Vec<K::Target> = samples.iter().map(|&s| targets[s as usize]).collect();
    let raw = task.leaf(&node_targets);
    let prediction = shrunk_prediction(task, params, anc, &raw);
    let n = samples.len();
    let leaf = |pred: K::Prediction| Node::Leaf {
        prediction: pred,
        n_samples: n,
    };

    if n < min_splittable(params) {
        return leaf(prediction);
    }
    if let Some(md) = params.max_depth {
        if depth >= md {
            return leaf(prediction);
        }
    }
    if task.criterion().impurity(&node_targets) <= 0.0 {
        return leaf(prediction);
    }

    let scored = score_node_fast(task, params, aggregator, binned, targets, &*samples);
    let best = match select_best(&scored) {
        Some(b) => b,
        None => return leaf(prediction),
    };
    if best.score <= params.min_impurity_decrease {
        return leaf(prediction);
    }
    if params.null_permutations > 0 {
        let node_idx: Vec<usize> = samples.iter().map(|&s| s as usize).collect();
        let samples_ro: &[SampleId] = &*samples;
        let thr = null_threshold(params, targets, &node_idx, |t: &[K::Target]| {
            score_node_fast(task, params, aggregator, binned, t, samples_ro)
        });
        if best.score <= thr {
            return leaf(prediction);
        }
    }
    let feature = best.candidate.feature;
    let state = best.candidate.state;
    let prefix = params.split_style == SplitStyle::Threshold && binned.bins[feature].is_continuous;

    let rule = match make_rule_fast(&binned.bins[feature], feature, state, prefix) {
        Some(r) => r,
        None => return leaf(prediction),
    };

    // Route by global code, exactly how the split was scored: prefix (`code <=
    // state`, i.e. `x < edge`) for threshold splits, equality for single-bin /
    // categorical. Missing codes (u16::MAX) exceed any state, so they route
    // right in both cases, consistent with the stored rule.
    let code_state = state as Bin;
    let codes = &binned.codes[feature];
    let p = if prefix {
        partition_in_place(samples, |s| codes[s as usize] <= code_state)
    } else {
        partition_in_place(samples, |s| codes[s as usize] == code_state)
    };

    let min_leaf = params.min_samples_leaf.max(1);
    if p < min_leaf || n - p < min_leaf {
        return leaf(prediction);
    }

    let me = Ancestor {
        shrunk: prediction,
        raw,
        n,
    };
    let (left_s, right_s) = samples.split_at_mut(p);
    let left = build_node_fast(
        task,
        params,
        aggregator,
        binned,
        targets,
        left_s,
        depth + 1,
        Some(&me),
    );
    let right = build_node_fast(
        task,
        params,
        aggregator,
        binned,
        targets,
        right_s,
        depth + 1,
        Some(&me),
    );
    Node::Internal {
        rule,
        left: Box::new(left),
        right: Box::new(right),
    }
}

// ---------------------------------------------------------------------------
// Public estimator
// ---------------------------------------------------------------------------

/// A cross-validated decision tree, generic over the [`Task`].
pub struct DecisionTree<K: Task> {
    /// The learning task (criterion + leaf/prediction behaviour).
    pub task: K,
    /// Hyper-parameters.
    pub params: TreeParams,
    /// Fold-score aggregation strategy.
    pub aggregator: Box<dyn Aggregator>,
    root: Option<Node<K::Prediction>>,
}

impl<K: Task> DecisionTree<K> {
    /// Build an estimator from an explicit task, params and aggregator.
    pub fn new(task: K, params: TreeParams, aggregator: Box<dyn Aggregator>) -> Self {
        DecisionTree {
            task,
            params,
            aggregator,
            root: None,
        }
    }

    /// Fit the tree. `columns` must be equal length and match `targets`.
    pub fn fit(&mut self, columns: &[Column], targets: &[K::Target]) -> Result<(), String> {
        if columns.is_empty() {
            return Err("need at least one feature column".to_string());
        }
        let n = columns[0].len();
        for (i, c) in columns.iter().enumerate() {
            if c.len() != n {
                return Err(format!(
                    "column {i} has length {} but expected {n}",
                    c.len()
                ));
            }
        }
        if targets.len() != n {
            return Err(format!(
                "targets length {} does not match {n} samples",
                targets.len()
            ));
        }
        if n == 0 {
            return Err("need at least one sample".to_string());
        }
        match self.params.mode {
            SplitMode::Strict => {
                let indices: Vec<usize> = (0..n).collect();
                self.root = Some(build_node(
                    &self.task,
                    &self.params,
                    self.aggregator.as_ref(),
                    columns,
                    targets,
                    &indices,
                    0,
                    None,
                ));
            }
            SplitMode::Fast => {
                if !self.task.fast_supported() {
                    return Err(
                        "this task (e.g. MAE) is not supported in SplitMode::Fast; use Strict"
                            .to_string(),
                    );
                }
                let binned = BinnedData::fit(columns, self.params.n_bins)?;
                let mut samples: Vec<SampleId> = (0..n as SampleId).collect();
                self.root = Some(build_node_fast(
                    &self.task,
                    &self.params,
                    self.aggregator.as_ref(),
                    &binned,
                    targets,
                    &mut samples,
                    0,
                    None,
                ));
            }
        }
        Ok(())
    }

    /// Install a pre-built tree as the fitted model.
    ///
    /// Prediction reads only the node tree (see [`predict_node`]), so a tree
    /// rebuilt from an [`ExportedNode`] dump predicts identically to the original
    /// regardless of the task/params/aggregator the estimator was created with.
    /// Used to reconstruct a fitted estimator when unpickling.
    pub fn set_fitted_root(&mut self, root: Node<K::Prediction>) {
        self.root = Some(root);
    }

    /// Predict a single sample. Panics if the tree has not been fitted.
    pub fn predict_one(&self, sample: &[FeatureValue]) -> K::Prediction {
        let root = self.root.as_ref().expect("model has not been fitted");
        predict_node(root, sample)
    }

    /// Predict a batch of samples.
    pub fn predict(&self, samples: &[Vec<FeatureValue>]) -> Vec<K::Prediction> {
        samples.iter().map(|s| self.predict_one(s)).collect()
    }

    /// Number of split levels (a lone leaf has depth 0).
    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |r| node_depth(r))
    }

    /// Number of leaves in the fitted tree.
    pub fn n_leaves(&self) -> usize {
        self.root.as_ref().map_or(0, |r| node_leaves(r))
    }

    /// Flatten the fitted tree into a preorder list of [`ExportedNode`]s for
    /// inspection or visualisation. Empty if the tree has not been fitted.
    pub fn export_nodes(&self) -> Vec<ExportedNode> {
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            self.walk_export(root, &mut out);
        }
        out
    }

    fn walk_export(&self, node: &Node<K::Prediction>, out: &mut Vec<ExportedNode>) -> usize {
        let id = out.len();
        out.push(ExportedNode::placeholder(id));
        match node {
            Node::Leaf {
                prediction,
                n_samples,
            } => {
                let info = self.task.describe_leaf(prediction);
                let e = &mut out[id];
                e.is_leaf = true;
                e.n_samples = *n_samples;
                e.class = info.class.map_or(-1, |c| c as i64);
                e.value = info.value.unwrap_or(f64::NAN);
                e.proba = info.proba;
            }
            Node::Internal { rule, left, right } => {
                let l = self.walk_export(left, out);
                let r = self.walk_export(right, out);
                let ns = out[l as usize].n_samples + out[r as usize].n_samples;
                let e = &mut out[id];
                e.is_leaf = false;
                e.left = l as i64;
                e.right = r as i64;
                e.feature = rule.feature() as i64;
                match rule {
                    SplitRule::ContinuousBin { lower, upper, .. } => {
                        e.is_categorical = false;
                        e.lower = lower.unwrap_or(f64::NEG_INFINITY);
                        e.upper = upper.unwrap_or(f64::INFINITY);
                        e.category = -1;
                    }
                    SplitRule::Category { category, .. } => {
                        e.is_categorical = true;
                        e.category = *category as i64;
                        e.lower = f64::NAN;
                        e.upper = f64::NAN;
                    }
                }
                e.n_samples = ns;
            }
        }
        id
    }
}

impl DecisionTree<Classification> {
    /// A Gini classifier with default params and mean aggregation.
    pub fn classifier(n_classes: usize) -> Self {
        DecisionTree::new(
            Classification::gini(n_classes),
            TreeParams::default(),
            Box::new(Mean),
        )
    }
}

impl DecisionTree<Regression> {
    /// An MSE regressor with default params and mean aggregation.
    pub fn regressor() -> Self {
        DecisionTree::new(Regression::mse(), TreeParams::default(), Box::new(Mean))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_depth_zero_is_a_single_leaf() {
        let cols = vec![Column::Categorical(vec![0, 1, 0, 1])];
        let y = vec![0usize, 1, 0, 1];
        let mut params = TreeParams::default();
        params.max_depth = Some(0);
        let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        assert_eq!(t.depth(), 0);
        assert_eq!(t.n_leaves(), 1);
    }

    #[test]
    fn pure_node_does_not_split() {
        let cols = vec![Column::Categorical(vec![0, 1, 0, 1])];
        let y = vec![7usize, 7, 7, 7]; // all one class
        let mut t = DecisionTree::classifier(8);
        t.fit(&cols, &y).unwrap();
        assert_eq!(t.n_leaves(), 1);
        assert_eq!(t.predict_one(&[FeatureValue::cat(0)]).class, 7);
    }

    #[test]
    fn separable_categorical_classifier() {
        // Category perfectly predicts the label.
        let cols = vec![Column::Categorical(vec![0, 0, 1, 1, 0, 0, 1, 1])];
        let y = vec![0usize, 0, 1, 1, 0, 0, 1, 1];
        let mut params = TreeParams::default();
        params.cv = KFold::new(2, 1);
        params.min_samples_leaf = 1; // toy data, smaller than the default leaf size
        let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        assert!(t.n_leaves() >= 2);
        assert_eq!(t.predict_one(&[FeatureValue::cat(0)]).class, 0);
        assert_eq!(t.predict_one(&[FeatureValue::cat(1)]).class, 1);
    }

    #[test]
    fn regression_predicts_group_means() {
        // Two categories with distinct response levels.
        let cols = vec![Column::Categorical(vec![0, 0, 0, 1, 1, 1])];
        let y = vec![1.0, 1.0, 1.0, 9.0, 9.0, 9.0];
        let mut params = TreeParams::default();
        params.cv = KFold::new(2, 3);
        params.min_samples_leaf = 1; // toy data, smaller than the default leaf size
        let mut t = DecisionTree::new(Regression::mse(), params, Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        assert!((t.predict_one(&[FeatureValue::cat(0)]) - 1.0).abs() < 1e-9);
        assert!((t.predict_one(&[FeatureValue::cat(1)]) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn class_prediction_probabilities_sum_to_one() {
        let cols = vec![Column::Categorical(vec![0, 0, 1, 1])];
        let y = vec![0usize, 1, 0, 1];
        let mut t = DecisionTree::classifier(2);
        t.fit(&cols, &y).unwrap();
        let p = t.predict_one(&[FeatureValue::cat(0)]);
        let s: f64 = p.proba.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    /// Pure-noise binary data: `n` samples, `p` continuous features.
    fn noise_data(n: usize, p: usize, seed: u64) -> (Vec<Column>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let cols = (0..p)
            .map(|_| Column::Continuous((0..n).map(|_| (rng.next_u64() % 10_000) as f64).collect()))
            .collect();
        let y = (0..n).map(|_| (rng.next_u64() % 2) as usize).collect();
        (cols, y)
    }

    #[test]
    fn null_test_stops_noise_splits() {
        // Honest scoring alone does not reliably shrink trees on pure noise:
        // the best of many near-zero candidates is still positive, most of all
        // at tiny nodes. The permutation null accounts for that search.
        let (cols, y) = noise_data(400, 3, 9);
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            // Leaf size 1 isolates the effect of the null test itself.
            let mut p_ins = TreeParams::default();
            p_ins.mode = mode;
            p_ins.min_samples_leaf = 1;
            p_ins.scoring = Scoring::InSample;
            let mut p_null = p_ins.clone();
            p_null.scoring = Scoring::Honest;
            p_null.null_permutations = 19;
            let mut a = DecisionTree::new(Classification::gini(2), p_ins, Box::new(Mean));
            let mut b = DecisionTree::new(Classification::gini(2), p_null, Box::new(Mean));
            a.fit(&cols, &y).unwrap();
            b.fit(&cols, &y).unwrap();
            assert!(
                b.n_leaves() <= 3 && b.n_leaves() < a.n_leaves(),
                "{mode:?}: honest+null {} vs in-sample {} leaves",
                b.n_leaves(),
                a.n_leaves()
            );
        }
    }

    #[test]
    fn permutation_null_only_prunes() {
        // The null test can only block splits, so it never adds leaves.
        let (cols, y) = noise_data(300, 2, 4);
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            let mut p0 = TreeParams::default();
            p0.mode = mode;
            let mut p1 = p0.clone();
            p1.null_permutations = 9;
            let mut a = DecisionTree::new(Classification::gini(2), p0, Box::new(Mean));
            let mut b = DecisionTree::new(Classification::gini(2), p1, Box::new(Mean));
            a.fit(&cols, &y).unwrap();
            b.fit(&cols, &y).unwrap();
            assert!(b.n_leaves() <= a.n_leaves());
        }
    }

    #[test]
    fn permutation_null_keeps_real_signal() {
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let y: Vec<usize> = (0..200).map(|i| usize::from(i >= 100)).collect();
        let mut params = TreeParams::default();
        params.null_permutations = 19;
        let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
        t.fit(&[Column::Continuous(x)], &y).unwrap();
        assert!(t.n_leaves() >= 2);
        assert_eq!(t.predict_one(&[FeatureValue::cont(10.0)]).class, 0);
        assert_eq!(t.predict_one(&[FeatureValue::cont(190.0)]).class, 1);
    }

    #[test]
    fn repeated_cv_is_deterministic_and_learns() {
        let x: Vec<f64> = (0..120).map(|i| i as f64).collect();
        let y: Vec<usize> = (0..120).map(|i| usize::from(i >= 60)).collect();
        let cols = vec![Column::Continuous(x)];
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            let mut params = TreeParams::default();
            params.mode = mode;
            params.cv_repeats = 3;
            let mut a = DecisionTree::new(Classification::gini(2), params.clone(), Box::new(Mean));
            let mut b = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
            a.fit(&cols, &y).unwrap();
            b.fit(&cols, &y).unwrap();
            assert_eq!(a.n_leaves(), b.n_leaves());
            assert_eq!(a.export_nodes().len(), b.export_nodes().len());
            assert_eq!(a.predict_one(&[FeatureValue::cont(5.0)]).class, 0);
            assert_eq!(a.predict_one(&[FeatureValue::cont(115.0)]).class, 1);
        }
    }

    #[test]
    fn shrinkage_pulls_leaves_toward_root() {
        let cols = vec![Column::Categorical(vec![0, 0, 0, 0, 1, 1, 1, 1])];
        let y = vec![1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0];
        let mut p0 = TreeParams::default();
        p0.cv = KFold::new(2, 1);
        p0.min_samples_leaf = 1; // toy data, smaller than the default leaf size
        let mut p1 = p0.clone();
        p1.shrinkage = 8.0; // parent n = 8 -> child deviation weighted 1/2
        let mut a = DecisionTree::new(Regression::mse(), p0, Box::new(Mean));
        let mut b = DecisionTree::new(Regression::mse(), p1, Box::new(Mean));
        a.fit(&cols, &y).unwrap();
        b.fit(&cols, &y).unwrap();
        assert_eq!(a.n_leaves(), b.n_leaves()); // structure unchanged
        assert_eq!(a.n_leaves(), 2);
        assert!((a.predict_one(&[FeatureValue::cat(0)]) - 1.0).abs() < 1e-9);
        // root 5 + (1 - 5) / (1 + 8/8) = 3, and symmetrically 7.
        assert!((b.predict_one(&[FeatureValue::cat(0)]) - 3.0).abs() < 1e-9);
        assert!((b.predict_one(&[FeatureValue::cat(1)]) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn class_shrinkage_keeps_valid_probabilities() {
        let ps = ClassPrediction { class: 0, proba: vec![0.9, 0.1] };
        let pr = ClassPrediction { class: 1, proba: vec![0.2, 0.8] };
        let raw = ClassPrediction { class: 0, proba: vec![1.0, 0.0] };
        let out = shrink_class(&ps, &pr, &raw, 10, 1.0);
        let total: f64 = out.proba.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(out.proba.iter().all(|&p| p >= 0.0));
        assert_eq!(out.class, 0);
    }

    #[test]
    fn mostly_missing_feature_does_not_win_root() {
        // x0: 90% missing, perfectly separates the 10% present.
        // x1: complete, 80% accurate. Scored per *node* sample (missing values
        // in the right child, as routed), x1 is far better (~0.18 vs ~0.03
        // honest Brier gain). Before the fast-path fix, missing values were
        // dropped from scoring and x0 won with ~0.50 per *present* sample.
        let n = 400usize;
        let y: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let x0: Vec<f64> = (0..n)
            .map(|i| if i % 20 < 2 { y[i] as f64 } else { f64::NAN })
            .collect();
        let x1: Vec<f64> = (0..n)
            .map(|i| {
                let ye = if i % 5 == 0 { 1 - y[i] } else { y[i] };
                ye as f64 + i as f64 * 1e-4
            })
            .collect();
        let cols = vec![Column::Continuous(x0), Column::Continuous(x1)];
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            let mut params = TreeParams::default();
            params.mode = mode;
            params.max_depth = Some(1);
            let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
            t.fit(&cols, &y).unwrap();
            let root = &t.export_nodes()[0];
            assert!(!root.is_leaf, "{mode:?}: expected a split");
            assert_eq!(root.feature, 1, "{mode:?}: mostly-missing feature won");
        }
    }

    #[test]
    fn informative_missingness_splits_on_presence() {
        // y = 1 exactly when x is missing; present values carry no signal, so
        // the best cut is "x is present". Strict mode used to score that cut
        // but could not build a rule for it, turning the root into a leaf.
        let n = 200usize;
        let x: Vec<f64> = (0..n)
            .map(|i| if i % 4 == 0 { f64::NAN } else { i as f64 })
            .collect();
        let y: Vec<usize> = (0..n).map(|i| usize::from(i % 4 == 0)).collect();
        let cols = vec![Column::Continuous(x)];
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            let mut params = TreeParams::default();
            params.mode = mode;
            let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
            t.fit(&cols, &y).unwrap();
            let root = &t.export_nodes()[0];
            assert!(!root.is_leaf, "{mode:?}: root should split");
            assert_eq!(root.lower, f64::NEG_INFINITY, "{mode:?}");
            assert_eq!(root.upper, f64::INFINITY, "{mode:?}: expected the presence rule");
            assert_eq!(t.n_leaves(), 2, "{mode:?}");
            assert_eq!(t.predict_one(&[FeatureValue::cont(f64::NAN)]).class, 1);
            assert_eq!(t.predict_one(&[FeatureValue::cont(50.0)]).class, 0);
            // Present values outside the training range still count as present.
            assert_eq!(t.predict_one(&[FeatureValue::cont(1e9)]).class, 0);
        }
    }

    #[test]
    fn no_presence_rule_without_missing_values() {
        let (cols, y) = noise_data(300, 2, 5);
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            let mut params = TreeParams::default();
            params.mode = mode;
            params.max_depth = Some(4);
            // Accept non-improving splits so many cuts get realised.
            params.min_impurity_decrease = -1.0;
            let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
            t.fit(&cols, &y).unwrap();
            assert!(t.n_leaves() > 1, "{mode:?}");
            for e in t.export_nodes().iter().filter(|e| !e.is_leaf) {
                assert!(e.upper.is_finite(), "{mode:?}: presence rule with no missing values");
            }
        }
    }

    #[test]
    fn min_samples_leaf_filters_candidates_before_selection() {
        // One threshold feature; node codes: 10 samples in bin 0, 40 in bin 1,
        // 50 in bin 2. The best-scoring cut (after bin 0) is inadmissible.
        let mut codes: Vec<u32> = vec![0; 10];
        codes.extend(vec![1; 40]);
        codes.extend(vec![2; 50]);
        let cand = |state: u32, score: f64| ScoredCandidate {
            candidate: Candidate { feature: 0, state },
            score,
            stats: FoldStats::from_scores(vec![score, score], 2),
        };
        let mut all = vec![cand(0, 0.9), cand(1, 0.5), cand(2, 0.1)];
        retain_admissible(&mut all, 100, 30, |_| true, |_| codes.clone());
        let kept: Vec<u32> = all.iter().map(|c| c.candidate.state).collect();
        assert_eq!(kept, vec![1]); // 10|90 and 100|0 dropped, 50|50 kept
        // Equality (single-bin) style: bin 1 alone is 40|60, admissible.
        let mut all = vec![cand(0, 0.9), cand(1, 0.5), cand(2, 0.1)];
        retain_admissible(&mut all, 100, 30, |_| false, |_| codes.clone());
        let kept: Vec<u32> = all.iter().map(|c| c.candidate.state).collect();
        assert_eq!(kept, vec![1, 2]);
    }

    #[test]
    fn inadmissible_best_split_does_not_stop_growth() {
        // The purest cut isolates the 20 positives at the low end, which is
        // below the default leaf size. Vetoing only the winner used to make
        // the root a leaf; an admissible split (50 | 150) exists and is taken.
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let y: Vec<usize> = (0..200).map(|i| usize::from(i < 20)).collect();
        for mode in [SplitMode::Strict, SplitMode::Fast] {
            let mut params = TreeParams::default();
            params.mode = mode;
            assert_eq!(params.min_samples_leaf, 30);
            let mut t = DecisionTree::new(Classification::gini(2), params, Box::new(Mean));
            t.fit(&[Column::Continuous(x.clone())], &y).unwrap();
            assert_eq!(t.n_leaves(), 2, "{mode:?}");
            for e in t.export_nodes().iter().filter(|e| e.is_leaf) {
                assert!(e.n_samples >= 30, "{mode:?}: leaf of {}", e.n_samples);
            }
        }
    }

    #[test]
    fn fit_rejects_mismatched_lengths() {
        let cols = vec![Column::Categorical(vec![0, 1, 0])];
        let y = vec![0usize, 1];
        let mut t = DecisionTree::classifier(2);
        assert!(t.fit(&cols, &y).is_err());
    }

    #[test]
    fn continuous_bin_routing_matches_bounds() {
        let rule = SplitRule::ContinuousBin {
            feature: 0,
            lower: Some(1.0),
            upper: Some(3.0),
        };
        assert!(!rule.route_left_sample(&[FeatureValue::cont(0.5)]));
        assert!(rule.route_left_sample(&[FeatureValue::cont(1.0)]));
        assert!(rule.route_left_sample(&[FeatureValue::cont(2.9)]));
        assert!(!rule.route_left_sample(&[FeatureValue::cont(3.0)]));
        assert!(!rule.route_left_sample(&[FeatureValue::cont(f64::NAN)]));
    }
}
