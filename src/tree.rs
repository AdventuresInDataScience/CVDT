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

use crate::aggregation::{Aggregator, Mean};
use crate::binning::{BinnedData, FeatureBins};
use crate::candidate::{present_states, Candidate};
use crate::criterion::{Criterion, Entropy, Gini, Mae, Mse, Variance};
use crate::cross_validation::{eval_feature, Fold, KFold};
use crate::data::{Bin, Column, FeatureValue, SampleId, UNKNOWN_CAT};
use crate::encoder::{bin_bounds, quantile_edges};
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
    /// `None` bounds are open ends (`-inf` / `+inf`).
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
    ) -> Vec<ScoredCandidate>;

    /// Score every candidate `state` of one feature on the **strict** path.
    ///
    /// The default computes impurity gain via [`eval_feature`] and this task's
    /// [`Task::criterion`] — the standard behaviour. Tasks that select splits by
    /// something other than impurity (e.g. an objective metric) override this.
    fn score_feature_strict(
        &self,
        columns: &[Column],
        targets: &[Self::Target],
        feature: usize,
        indices: &[usize],
        folds: &[Fold],
        n_bins: usize,
        aggregator: &dyn Aggregator,
    ) -> Vec<ScoredCandidate> {
        let states = present_states(&columns[feature], indices, n_bins);
        if states.is_empty() {
            return Vec::new();
        }
        // min_child = 1 here; real min_samples_leaf is enforced on the actual
        // node partition, not per-fold.
        let stats = eval_feature(
            columns,
            targets,
            feature,
            &states,
            folds,
            n_bins,
            self.criterion(),
            1,
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
        )
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
    ) -> Vec<ScoredCandidate> {
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
        )
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
    ) -> Vec<ScoredCandidate> {
        score_regr(
            codes, targets, order, val_fold, k, max_bin, aggregator, feature, scratch,
        )
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

/// Tree hyper-parameters.
#[derive(Clone, Debug)]
pub struct TreeParams {
    /// Maximum number of split levels; `None` for unlimited.
    pub max_depth: Option<usize>,
    /// Minimum samples a node must have to be eligible to split.
    pub min_samples_split: usize,
    /// Minimum samples each child must receive for a split to be accepted.
    pub min_samples_leaf: usize,
    /// Minimum aggregated score for a split to be accepted (must be exceeded).
    pub min_impurity_decrease: f64,
    /// Number of quantile bins for continuous features.
    pub n_bins: usize,
    /// Cross-validation configuration used to score splits.
    pub cv: KFold,
    /// Split-evaluation strategy.
    pub mode: SplitMode,
    /// Whether to evaluate features in parallel.
    pub parallel: bool,
    /// Worker-thread count when `parallel` is set.
    pub n_threads: usize,
    /// Only parallelise nodes with at least this many samples (spawning threads
    /// for tiny nodes costs more than it saves).
    pub parallel_min_samples: usize,
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams {
            max_depth: Some(8),
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            n_bins: 8,
            cv: KFold {
                k: 5,
                seed: 42,
                shuffle: true,
            },
            mode: SplitMode::Strict,
            parallel: false,
            n_threads: 1,
            parallel_min_samples: 512,
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
) -> Option<SplitRule> {
    match col {
        Column::Continuous(v) => {
            let vals: Vec<f64> = indices.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&vals, n_bins);
            let (lower, upper) = bin_bounds(&edges, state as usize)?;
            Some(SplitRule::ContinuousBin {
                feature,
                lower,
                upper,
            })
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

#[allow(clippy::too_many_arguments)]
fn build_node<K: Task>(
    task: &K,
    params: &TreeParams,
    aggregator: &dyn Aggregator,
    columns: &[Column],
    targets: &[K::Target],
    indices: &[usize],
    depth: usize,
) -> Node<K::Prediction> {
    let node_targets: Vec<K::Target> = indices.iter().map(|&i| targets[i]).collect();
    let prediction = task.leaf(&node_targets);
    let leaf = |pred: K::Prediction| Node::Leaf {
        prediction: pred,
        n_samples: indices.len(),
    };

    // --- cheap stopping criteria -------------------------------------------
    if indices.len() < params.min_samples_split.max(2) {
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
    let folds = params.cv.folds(indices);

    // Per-feature scoring closure, delegated to the task so alternative
    // strict-path scorers (e.g. objective metrics) plug in transparently.
    let eval_one = |f: &usize| -> Vec<ScoredCandidate> {
        task.score_feature_strict(
            columns,
            targets,
            *f,
            indices,
            &folds,
            params.n_bins,
            aggregator,
        )
    };

    let feature_ids: Vec<usize> = (0..columns.len()).collect();
    let scored: Vec<ScoredCandidate> = if params.parallel && params.n_threads > 1 {
        par_map(&feature_ids, params.n_threads, eval_one)
            .into_iter()
            .flatten()
            .collect()
    } else {
        feature_ids.iter().flat_map(|f| eval_one(f)).collect()
    };

    // --- select and validate the winner ------------------------------------
    let best = match select_best(&scored) {
        Some(b) => b,
        None => return leaf(prediction),
    };
    if best.score <= params.min_impurity_decrease {
        return leaf(prediction);
    }

    let rule = match make_rule(
        &columns[best.candidate.feature],
        best.candidate.feature,
        best.candidate.state,
        indices,
        params.n_bins,
    ) {
        Some(r) => r,
        None => return leaf(prediction),
    };

    // --- partition and enforce min_samples_leaf ----------------------------
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
    let left = build_node(
        task,
        params,
        aggregator,
        columns,
        targets,
        &left_idx,
        depth + 1,
    );
    let right = build_node(
        task,
        params,
        aggregator,
        columns,
        targets,
        &right_idx,
        depth + 1,
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
fn make_rule_fast(fb: &FeatureBins, feature: usize, state: u32) -> Option<SplitRule> {
    if fb.is_continuous {
        let (lower, upper) = fb.cont_bounds(state as usize)?;
        Some(SplitRule::ContinuousBin {
            feature,
            lower,
            upper,
        })
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
) -> Node<K::Prediction> {
    let node_targets: Vec<K::Target> = samples.iter().map(|&s| targets[s as usize]).collect();
    let prediction = task.leaf(&node_targets);
    let n = samples.len();
    let leaf = |pred: K::Prediction| Node::Leaf {
        prediction: pred,
        n_samples: n,
    };

    if n < params.min_samples_split.max(2) {
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

    // One compact fold assignment for the whole node.
    let idx_usize: Vec<usize> = samples.iter().map(|&s| s as usize).collect();
    let (order, val_fold, k) = params.cv.assign(&idx_usize);
    if k == 0 {
        return leaf(prediction);
    }

    let n_features = binned.n_features();
    let eval_one = |f: &usize| -> Vec<ScoredCandidate> {
        let feature = *f;
        let mut scratch = FastScratch::new();
        task.score_feature_fast(
            &binned.codes[feature],
            targets,
            &order,
            &val_fold,
            k,
            binned.bins[feature].max_bin,
            feature,
            aggregator,
            &mut scratch,
        )
    };

    let feature_ids: Vec<usize> = (0..n_features).collect();
    let scored: Vec<ScoredCandidate> =
        if params.parallel && params.n_threads > 1 && n >= params.parallel_min_samples {
            par_map(&feature_ids, params.n_threads, eval_one)
                .into_iter()
                .flatten()
                .collect()
        } else {
            // Serial: reuse a single scratch across features.
            let mut scratch = FastScratch::new();
            let mut all = Vec::new();
            for &feature in &feature_ids {
                all.extend(task.score_feature_fast(
                    &binned.codes[feature],
                    targets,
                    &order,
                    &val_fold,
                    k,
                    binned.bins[feature].max_bin,
                    feature,
                    aggregator,
                    &mut scratch,
                ));
            }
            all
        };

    let best = match select_best(&scored) {
        Some(b) => b,
        None => return leaf(prediction),
    };
    if best.score <= params.min_impurity_decrease {
        return leaf(prediction);
    }
    let feature = best.candidate.feature;
    let state = best.candidate.state;

    let rule = match make_rule_fast(&binned.bins[feature], feature, state) {
        Some(r) => r,
        None => return leaf(prediction),
    };

    // Route by global-code equality, which is exactly how the split was scored
    // (in-bin == matches state), and is consistent with the stored rule's
    // predict-time bounds.
    let code_state = state as Bin;
    let codes = &binned.codes[feature];
    let p = partition_in_place(samples, |s| codes[s as usize] == code_state);

    let min_leaf = params.min_samples_leaf.max(1);
    if p < min_leaf || n - p < min_leaf {
        return leaf(prediction);
    }

    let (left_s, right_s) = samples.split_at_mut(p);
    let left = build_node_fast(task, params, aggregator, binned, targets, left_s, depth + 1);
    let right = build_node_fast(
        task,
        params,
        aggregator,
        binned,
        targets,
        right_s,
        depth + 1,
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
