# CVDT API Reference

Crate: `cvdt` v0.8.0 · Rust 2021 · MSRV 1.63 · zero dependencies

This document describes every public item, organised by module. All listed
types and functions are re-exported at the crate root, so `use cvdt::Gini;`
works as well as `use cvdt::criterion::Gini;`.

> **v2 adds a histogram "fast" build path** alongside the exact "strict" path;
> items are marked *(v2)*. **v3 adds CART-style ordered threshold splits** for
> continuous features, marked *(v3)*. Note the v3 default *changes behaviour*:
> `TreeParams.split_style` defaults to `SplitStyle::Threshold`, so continuous
> features are now cut as `x < edge` rather than one-bin-vs-rest. Set
> `split_style: SplitStyle::BinMembership` to recover the v1/v2 semantics.
> `mode` still defaults to `SplitMode::Strict`.
>
> **0.8 changes how splits are scored**, marked *(0.8)*. The default
> `TreeParams.scoring` is `Scoring::Honest` (training-fit models scored on the
> held-out fold), which removes the per-candidate upward bias but is not by
> itself a stopping rule (see `null_permutations`); set
> `Scoring::InSample` to reproduce 0.7 scores. `FoldStats.std` is now the sample
> std (`NaN` below two folds), and `eval_feature`, `score_classif`,
> `score_regr` and the `Task::score_feature_*` methods take a trailing
> `scoring: Scoring` argument. 0.8 also adds repeated K-fold, a
> permutation-null stopping test, hierarchical shrinkage, new aggregators, and
> the "x is present" threshold cut for nodes with missing values.

Convention used below: **Target** is `usize` (class ids) for classification and
`f64` for regression.

---

## Quick start

```rust
use cvdt::{Classification, Column, DecisionTree, FeatureValue, Mean, TreeParams};

let columns = vec![Column::Continuous(vec![0.1, 0.2, 0.9, 1.0])];
let y: Vec<usize> = vec![0, 0, 1, 1];

let mut tree = DecisionTree::new(
    Classification::gini(2),
    TreeParams::default(),
    Box::new(Mean),
);
tree.fit(&columns, &y).unwrap();

let p = tree.predict_one(&[FeatureValue::cont(0.95)]);
assert_eq!(p.class, 1);
```

---

## Module `data`

Core column-major data representation.

### `type CatId = u32`
Integer id representing a categorical level.

### `const UNKNOWN_CAT: CatId = u32::MAX`
Sentinel for a categorical value never seen while fitting. Routes to the
"not-in-state" child.

### `type SampleId = u32` *(v2)*
Compact sample index used by the fast path's index buffers.

### `type Bin = u16` *(v2)*
Compact per-sample bin code used by the fast path.

### `const MISSING_BIN_CODE: Bin = u16::MAX` *(v2)*
Reserved fast-path code for a missing / unknown value.

### `enum Column`
A single feature column.
- `Column::Continuous(Vec<f64>)` — continuous values; non-finite entries are
  treated as missing.
- `Column::Categorical(Vec<CatId>)` — pre-encoded integer category ids.

Methods:
- `fn len(&self) -> usize`
- `fn is_empty(&self) -> bool`
- `fn is_continuous(&self) -> bool`
- `fn is_categorical(&self) -> bool`

### `struct Dataset`
Validated collection of equal-length columns.
- Fields: `columns: Vec<Column>`, `n_samples: usize`
- `fn new(columns: Vec<Column>) -> Result<Dataset, String>` — errors if empty or
  columns are ragged.
- `fn n_features(&self) -> usize`

`Dataset` is a convenience/validation helper; `DecisionTree::fit` accepts
`&[Column]` directly and performs the same checks.

### `enum FeatureValue`
A single value used at prediction time.
- `FeatureValue::Continuous(f64)`
- `FeatureValue::Categorical(CatId)`
- `fn cont(x: f64) -> FeatureValue`
- `fn cat(c: CatId) -> FeatureValue`

---

## Module `encoder`

Quantile binning of continuous features.

### `const MISSING_BIN: u32 = u32::MAX`
Bin id returned for non-finite values.

### `fn quantile_edges(values: &[f64], n_bins: usize) -> Vec<f64>`
Up to `n_bins - 1` equal-frequency cut points. Non-finite values ignored;
result sorted ascending. Returns empty when `n_bins <= 1` or no finite values.

### `fn bin_of(edges: &[f64], value: f64) -> u32`
Bin id of `value` = number of edges `<= value`; `MISSING_BIN` for non-finite.
For `E` edges there are `E + 1` bins (`0..=E`).

### `fn bin_bounds(edges: &[f64], b: usize) -> Option<(Option<f64>, Option<f64>)>`
Half-open interval `[lower, upper)` for bin `b`. `None` bounds denote open ends
(`-inf` / `+inf`). Returns `None` if `b > edges.len()`.

---

## Module `criterion`

Impurity measures. The tree never calls a concrete criterion directly; it holds
a trait object.

### `trait Criterion: Send + Sync`
- `type Target: Copy`
- `fn impurity(&self, targets: &[Self::Target]) -> f64` — empty slice ⇒ `0.0`.
- `fn supports_heldout(&self) -> bool` *(0.8)* — default `false`; criteria that
  return `false` fall back to in-sample scoring under `Scoring::Honest`.
- `fn heldout_loss(&self, train: &[Self::Target], val: &[Self::Target]) ->
  Option<f64>` *(0.8)* — **total** loss over `val` of the model this criterion
  fits to `train` (class proportions, mean, or median); `None` when
  unsupported. With `train == val` it equals `val.len() · impurity` exactly for
  Gini/Brier, variance/MSE and MAE, and approximately for entropy (log-loss
  smoothing). `train` must be non-empty.
- `fn name(&self) -> &'static str`

Object-safe: used as `Box<dyn Criterion<Target = usize>>` /
`Box<dyn Criterion<Target = f64>>`.

### Classification criteria (`Target = usize`)
- `struct Gini { pub n_classes: usize }` · `Gini::new(n_classes)` — `1 - Σ pᵢ²`.
- `struct Entropy { pub n_classes: usize }` · `Entropy::new(n_classes)` —
  `-Σ pᵢ log₂ pᵢ` (bits).

Out-of-range labels are ignored rather than causing a panic.

### Regression criteria (`Target = f64`)
- `struct Variance` — population variance about the mean.
- `struct Mse` — identical to `Variance`; distinct name for intent.
- `struct Mae` — mean absolute error about the median.

All three implement `Default`.

### Statistic-based helpers *(v2)*
Pure functions computing impurity directly from sufficient statistics — the
single source of truth shared by the exact and fast paths.
- `fn gini_from_counts(counts: &[u64], n: u64) -> f64`
- `fn entropy_from_counts(counts: &[u64], n: u64) -> f64` (bits)
- `fn variance_from_moments(n: f64, sum: f64, sumsq: f64) -> f64` — `Σx²/n −
  (Σx/n)²`, clamped at 0.
- `fn brier_loss_counts(train: &[u64], val: &[u64]) -> f64` *(0.8)* — total
  Brier loss on `val` counts of the class proportions of `train`; `NaN` if
  `train` is empty.
- `fn log_loss_counts(train: &[u64], val: &[u64], alpha: f64) -> f64` *(0.8)* —
  total log-loss (bits) on `val` of the `alpha`-smoothed `train` proportions.
- `const LOGLOSS_ALPHA: f64 = 0.5` *(0.8)* — Jeffreys smoothing used for
  held-out log-loss, so a class unseen in training costs a finite amount.
- `fn sq_loss_moments(n: f64, sum: f64, sumsq: f64, m: f64) -> f64` *(0.8)* —
  total squared error of predicting `m` for samples with those moments.

---

## Module `candidate`

Enumerate `feature == state` splits present at a node. No scoring.

### `struct Candidate`
- `feature: usize`, `state: u32`
- Derives `Clone, Copy, Debug, PartialEq, Eq`.

### `fn present_states(col: &Column, indices: &[usize], n_bins: usize) -> Vec<u32>`
Distinct states (bin ids or category ids) occurring in the given samples, sorted
ascending. Missing bins and unknown categories excluded. For continuous columns,
node-level quantile edges are fit to enumerate which bins occur.

### `fn split_states(col: &Column, indices: &[usize], n_bins: usize, prefix: bool) -> Vec<u32>` *(0.8)*
The states the tree actually scores. Same as `present_states`, except for a
threshold (`prefix = true`) split of a continuous column: the top bin's cut
("x is present": every present value left, missing right) is kept only when
the node has missing values. The fast path applies the same rule.

### `fn generate_candidates(columns: &[Column], indices: &[usize], n_bins: usize) -> Vec<Candidate>`
All candidates across all features for a node.

---

## Module `cross_validation`

Fold generation and per-fold split scoring. Contains no impurity logic.

### `struct SplitMix64`
Self-contained deterministic PRNG.
- `SplitMix64::new(seed: u64)`
- `fn next_u64(&mut self) -> u64`
- `fn below(&mut self, bound: u64) -> u64` — uniform in `0..bound`.

### `enum Scoring` *(0.8)*
How a candidate is scored on one fold. Derives `Clone, Copy, Debug, PartialEq,
Eq`; `Default` is `Honest`.
- `Scoring::Honest` — parent and child models fit on the training fold, loss
  (Brier / log-loss / squared / absolute error) measured on the validation fold:
  `gain = (L_parent − L_left − L_right) / n_val`. A split with no signal scores
  `≤ 0` in expectation. This removes the per-candidate upward bias; it does not
  correct for keeping the best of many candidates (see
  `TreeParams::null_permutations`).
- `Scoring::InSample` — the ≤ 0.7 score, impurity decrease on the validation
  labels alone; never negative.

### `struct Fold`
- `train: Vec<usize>`, `val: Vec<usize>` (global sample indices).

### `struct KFold`
- Fields: `k: usize`, `seed: u64`, `shuffle: bool`.
- `KFold::new(k, seed) -> KFold` (`shuffle = true`).
- `fn folds(&self, indices: &[usize]) -> Vec<Fold>` — deterministic; `k` clamped
  to `[2, indices.len()]`; the node size is mixed into the seed so sibling nodes
  decorrelate while remaining reproducible.
- `fn assign(&self, indices: &[usize]) -> (Vec<SampleId>, Vec<u8>, usize)` *(v2)*
  — compact fold representation for the fast path: the node samples in shuffled
  order, each one's validation-fold id, and the effective `k` (clamped to
  `[2, min(n, 255)]`). Derived from the same shuffle as `folds`.

### `struct FoldStats`
Per-candidate summary across folds.
- `scores: Vec<f64>` (successful folds only), `mean`, `median`, `std`,
  `n_success: usize`, `n_total: usize`.
- `mean`/`median` are `-inf` when there are no successful folds.
- `std` is the sample std (Bessel-corrected, `n − 1`) *(0.8)*, and `NaN` with
  fewer than two successful folds, so variance-penalising aggregators score
  such candidates as non-finite (excluded).
- `fn from_scores(scores: Vec<f64>, n_total: usize) -> FoldStats`.
- `fn var(&self) -> f64` *(0.8)* — `std²` (`NaN` when undefined).
- `fn merged(parts: &[&FoldStats]) -> FoldStats` *(0.8)* — pool several fold
  sets (repeated K-fold) into one.

### `fn eval_feature<T: Copy>(...) -> Vec<FoldStats>`
```rust
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
) -> Vec<FoldStats>
```
One `FoldStats` per state (aligned with `states`). For each fold the continuous
bin edges are fit on the **training** samples only and applied to both sides
(the leakage guard). *(0.8)* With `Scoring::Honest` (when the criterion
`supports_heldout`) the per-fold score is the honest gain above; a child that is
empty on the training side predicts with the training parent, and a fold is
skipped only when its training or validation side is empty (`min_child` is not
used). With `Scoring::InSample` it is the impurity decrease on the validation
labels, `parent - weighted_child`, and a fold is skipped for a state when either
child has fewer than `min_child` validation samples. *(v3)* `prefix = true`
scores a threshold cut (left child is `code <= state`, i.e. `x < edge`; missing
codes never go left); `false` scores single-bin membership (`code == state`).

---

## Module `binning` *(v2)*

Global one-time feature binning for the fast path. Continuous features are
quantile-binned; categorical features are densely re-indexed. Missing / unknown
values map to `MISSING_BIN_CODE`.

### `struct FeatureBins`
Per-feature metadata.
- `is_continuous: bool`, `edges: Vec<f64>` (continuous), `categories: Vec<CatId>`
  (dense id → original id, sorted), `max_bin: usize` (valid codes `0..max_bin`).
- `fn code(&self, value: FeatureValue) -> Bin` — code a raw value (predict-side).
- `fn cont_bounds(&self, b: usize) -> Option<(Option<f64>, Option<f64>)>`.
- `fn category(&self, b: usize) -> Option<CatId>`.

### `struct BinnedData`
- `codes: Vec<Vec<Bin>>` (`codes[feature][sample]`), `bins: Vec<FeatureBins>`,
  `n_samples: usize`.
- `fn fit(columns: &[Column], n_bins: usize) -> Result<BinnedData, String>` —
  errors if a feature needs more slots than `Bin` can represent.
- `fn n_features(&self) -> usize`.

---

## Module `histogram` *(v2)*

Histogram-based fast split scoring: one scatter pass builds all K validation-fold
histograms; each candidate is scored from sufficient statistics with the "rest"
child obtained by `fold_total − cell`. *(0.8)* Under `Scoring::Honest` the
training side of each fold is `node_total − fold`, so honest scoring needs no
extra pass. Missing values count toward fold totals but no bin, so they are
scored in the right child, where they are routed. In a threshold sweep the cut
after the last bin ("x is present") is offered only when the node has missing
values.

### `enum ClassImpurityKind`
`Gini` | `Entropy` — which impurity to compute from class counts. Derives
`Clone, Copy, Debug, PartialEq, Eq`.

### `struct FastScratch`
Reusable scratch buffers (grow-and-zero on reuse) so per-node scoring rarely
allocates. `FastScratch::new()`; implements `Default`.

### `fn score_classif(...) -> Vec<ScoredCandidate>`
```rust
pub fn score_classif(
    codes: &[Bin], labels: &[usize],
    order: &[SampleId], val_fold: &[u8], k: usize,
    max_bin: usize, n_classes: usize, kind: ClassImpurityKind,
    aggregator: &dyn Aggregator, feature: usize, scratch: &mut FastScratch,
    prefix: bool, scoring: Scoring,
) -> Vec<ScoredCandidate>
```
One `ScoredCandidate` per candidate cut, scored by CV gain on validation labels
(honest Brier / log-loss gain, or in-sample impurity decrease, per `scoring`).
*(v3)* `prefix = true` sweeps threshold cuts (left child is bins `0..=state`, via
a running per-fold class prefix); `false` scores single-bin membership
(`feature == bin`).

### `fn score_regr(...) -> Vec<ScoredCandidate>`
```rust
pub fn score_regr(
    codes: &[Bin], targets: &[f64],
    order: &[SampleId], val_fold: &[u8], k: usize,
    max_bin: usize, aggregator: &dyn Aggregator,
    feature: usize, scratch: &mut FastScratch,
    prefix: bool, scoring: Scoring,
) -> Vec<ScoredCandidate>
```
Variance/MSE via `(count, Σy, Σy²)` moments. (MAE is unsupported on this path.)
*(0.8)* Moments are accumulated for `y − c`, with `c` the mean of up to 64
(shuffled) node samples; gains are shift-invariant, and the shift avoids
catastrophic cancellation when targets have a large offset relative to their
spread.
*(v3)* `prefix` selects threshold-sweep vs single-bin as in `score_classif`.

---

## Module `aggregation`

Collapse `FoldStats` into one ranking number where **higher is better**.

### `trait Aggregator: Send + Sync`
- `fn aggregate(&self, stats: &FoldStats) -> f64`
- `fn aggregate_node(&self, stats: &[&FoldStats]) -> Vec<f64>` *(0.8)* — score
  every candidate of one node at once (lets aggregators pool information across
  candidates). Default: `aggregate` per candidate. The tree calls this after all
  candidates (and all `cv_repeats`) are scored.
- `fn name(&self) -> &'static str`

Implementations must return `f64::NEG_INFINITY` when `stats.n_success == 0`.
Fold scores share training data and are correlated, so `std` describes
fold-to-fold spread but understates the uncertainty of the mean; no aggregator
here is a calibrated confidence bound.

### Provided aggregators
- `struct Mean` — arithmetic mean (`Default`).
- `struct Median` — median; robust to one bad fold (`Default`).
- `struct TrimmedMean { pub frac: f64 }` — drop `frac` from each end (`[0, 0.5)`).
- `struct SignalToNoise { pub eps: f64 }` — `mean / (std + eps)`; `Default` uses
  `eps = 1e-12`.
- `struct MeanMinusLambdaStd { pub lambda: f64 }` — `mean - λ·std`.
- `struct Quantile { pub q: f64 }` *(0.8)* — linear-interpolated `q`-quantile of
  the fold scores (risk-averse for small `q`).
- `struct ModeratedSignalToNoise { pub d0: f64, pub eps: f64 }` *(0.8)* —
  `mean / sqrt(moderated var)`, where each candidate's variance is shrunk toward
  the node's median candidate variance with weight `d0` fold-equivalents
  (limma-style): `(d0·s0² + (n−1)·s²) / (d0 + n − 1)`. `Default`: `d0 = 4`,
  `eps = 1e-12`.
- `struct ModeratedMeanMinusLambdaStd { pub lambda: f64, pub d0: f64 }` *(0.8)*
  — `mean − λ·sqrt(moderated var)`.
- `struct Custom { pub f: Box<dyn Fn(&FoldStats) -> f64 + Send + Sync> }` ·
  `Custom::new(closure)`.

---

## Module `selector`

Rank scored candidates and pick the winner.

### `struct ScoredCandidate`
- `candidate: Candidate`, `score: f64`, `stats: FoldStats`. Derives `Clone`.

### `fn select_best(scored: &[ScoredCandidate]) -> Option<&ScoredCandidate>`
Highest finite score wins; non-finite scores ignored; ties broken
deterministically by `(feature, state)` ascending. `None` when no candidate has
a finite score.

---

## Module `parallel`

Dependency-free parallelism helper.

### `fn par_map<T, R, F>(items: &[T], n_threads: usize, f: F) -> Vec<R>`
```rust
where T: Sync, R: Send, F: Fn(&T) -> R + Sync
```
Order-preserving parallel map over contiguous chunks via
`std::thread::scope`. Falls back to a serial map when `n_threads <= 1` or there
is at most one item. Output is identical to `items.iter().map(f).collect()`.

---

## Module `tree`

Orchestration, tasks, hyper-parameters and prediction.

### `enum SplitRule`
Routing rule stored in an internal node. "Left" means *matches the state*;
missing / non-finite values route right.
- `SplitRule::ContinuousBin { feature: usize, lower: Option<f64>, upper: Option<f64> }`
  — `lower <= x < upper`, `None` = open end. With both bounds `None` it is the
  "x is present" rule *(0.8)*: every finite value left, missing right.
- `SplitRule::Category { feature: usize, category: u32 }`
- `fn feature(&self) -> usize`
- `fn route_left_train(&self, columns: &[Column], i: usize) -> bool`
- `fn route_left_sample(&self, sample: &[FeatureValue]) -> bool`

### `enum Node<P>`
- `Node::Leaf { prediction: P, n_samples: usize }`
- `Node::Internal { rule: SplitRule, left: Box<Node<P>>, right: Box<Node<P>> }`
- Derives `Clone, Debug`.

### `enum SplitMode` *(v2)*
- `SplitMode::Strict` — exact path; continuous edges refit per training fold
  (default, matches v1).
- `SplitMode::Fast` — histogram path; global one-time binning + sufficient-stat
  scoring. Faster; drops per-fold edge refitting (a feature-only relaxation).
  MAE is not supported.
Derives `Clone, Copy, Debug, PartialEq, Eq`.

### `enum SplitStyle` *(v3)*
How a continuous feature is partitioned at a split (categorical features are
always tested by category equality; this enum has no effect on them).
- `SplitStyle::Threshold` — CART-style ordered cut: left child is `x < edge`
  where `edge` is a quantile boundary (default). Candidate cuts are the
  `n_bins - 1` boundaries; the score is still cross-validated on held-out folds.
- `SplitStyle::BinMembership` — one quantile bin vs. the rest
  (`lower <= x < upper`), the original v1/v2 behaviour.
Derives `Clone, Copy, Debug, PartialEq, Eq`.

### `trait Task: Send + Sync`
- `type Target: Copy + Send + Sync`
- `type Prediction: Clone`
- `fn leaf(&self, targets: &[Self::Target]) -> Self::Prediction`
- `fn criterion(&self) -> &dyn Criterion<Target = Self::Target>`
- `fn fast_supported(&self) -> bool` *(v2)* — default `true`; `false` for tasks
  with no additive statistic (MAE).
- `fn score_feature_fast(&self, codes: &[Bin], targets: &[Self::Target], order:
  &[SampleId], val_fold: &[u8], k: usize, max_bin: usize, feature: usize,
  aggregator: &dyn Aggregator, scratch: &mut FastScratch, prefix: bool,
  scoring: Scoring) -> Vec<ScoredCandidate>` *(v2; `prefix` added v3, `scoring`
  added 0.8)* — fast-path scorer for one feature; implementations delegate to
  `histogram`. `prefix = true` scores CART-style threshold cuts
  (`code <= state`), `false` single-bin membership.
- `fn score_feature_strict(&self, columns: &[Column], targets: &[Self::Target],
  feature: usize, indices: &[usize], folds: &[Fold], n_bins: usize, aggregator:
  &dyn Aggregator, prefix: bool, scoring: Scoring) -> Vec<ScoredCandidate>` —
  strict-path scorer (default: `split_states` + `eval_feature`); `prefix`
  selects threshold vs single-bin as above.
- `fn shrink(&self, parent_shrunk: &Self::Prediction, parent_raw:
  &Self::Prediction, raw: &Self::Prediction, parent_n: usize, lambda: f64) ->
  Self::Prediction` *(0.8)* — hierarchical-shrinkage step
  `shrunk_parent + (raw − raw_parent) / (1 + λ / n_parent)`; default returns
  `raw` unchanged. Classification shrinks the probability vector and recomputes
  the argmax class; regression shrinks the mean.

### `struct ClassPrediction`
- `class: usize` (argmax; lowest id on ties), `proba: Vec<f64>` (per-class
  fraction of leaf training samples). Derives `Clone, Debug, PartialEq`.

### `struct Classification`
- Fields: `n_classes: usize`, `criterion: Box<dyn Criterion<Target = usize>>`,
  `kind: ClassImpurityKind` *(v2)*.
- `Classification::gini(n_classes)`, `Classification::entropy(n_classes)` — set
  `kind` accordingly.
- `Task::Target = usize`, `Task::Prediction = ClassPrediction`.

### `struct Regression`
- Field: `criterion: Box<dyn Criterion<Target = f64>>`.
- `Regression::mse()`, `Regression::variance()`, `Regression::mae()`.
- `fast_supported()` is `false` for MAE.
- `Task::Target = f64`, `Task::Prediction = f64` (leaf mean).

### `struct TreeParams`
| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_depth` | `Option<usize>` | `Some(8)` | max split levels; `None` = unlimited |
| `min_samples_split` | `usize` | `2` | min samples for a node to be eligible to split |
| `min_samples_leaf` | `usize` | `30` | min samples each child must receive; candidates violating it are excluded before selection *(0.8: default was 1, and only the winner was checked)* |
| `min_impurity_decrease` | `f64` | `0.0` | winning score must strictly exceed this |
| `n_bins` | `usize` | `8` | quantile bins for continuous features |
| `cv` | `KFold` | `KFold{k:5,seed:42,shuffle:true}` | CV configuration |
| `mode` | `SplitMode` | `Strict` | *(v2)* exact vs histogram path |
| `split_style` | `SplitStyle` | `Threshold` | *(v3)* continuous split geometry: ordered threshold vs single-bin |
| `parallel` | `bool` | `false` | evaluate features in parallel |
| `n_threads` | `usize` | `1` | worker threads when `parallel` |
| `parallel_min_samples` | `usize` | `512` | *(v2)* min node size to parallelise |
| `scoring` | `Scoring` | `Honest` | *(0.8)* per-fold score; see `Scoring` |
| `cv_repeats` | `usize` | `1` | *(0.8)* repeated K-fold rounds per node; scores pooled per candidate |
| `min_valid_folds` | `usize` | `2` | *(0.8)* candidates scoring on fewer folds (capped at the number attempted) are discarded |
| `null_permutations` | `usize` | `0` | *(0.8)* permutation-null stopping test; `0` = off |
| `null_quantile` | `f64` | `0.95` | *(0.8)* quantile of the permuted best scores the real best must exceed |
| `shrinkage` | `f64` | `0.0` | *(0.8)* hierarchical-shrinkage λ for leaf predictions; `0` = off |

Implements `Default` and `Clone`.

### `struct DecisionTree<K: Task>`
- Fields: `task: K`, `params: TreeParams`, `aggregator: Box<dyn Aggregator>`
  (the fitted tree is held privately).
- `fn new(task: K, params: TreeParams, aggregator: Box<dyn Aggregator>) -> Self`
- `fn fit(&mut self, columns: &[Column], targets: &[K::Target]) -> Result<(), String>`
  — validates equal column lengths and `targets.len()`. In `SplitMode::Fast`,
  also errors if the task is unsupported (MAE) or a feature exceeds the `Bin`
  capacity. *(v2)*
- `fn predict_one(&self, sample: &[FeatureValue]) -> K::Prediction` — panics if
  unfitted.
- `fn predict(&self, samples: &[Vec<FeatureValue>]) -> Vec<K::Prediction>`
- `fn depth(&self) -> usize` — a lone leaf has depth `0`.
- `fn n_leaves(&self) -> usize`

Convenience constructors (default params + `Mean` aggregator):
- `DecisionTree::<Classification>::classifier(n_classes)`
- `DecisionTree::<Regression>::regressor()`

---

## Behavioural notes

- **Leakage guard:** continuous bin edges used for *scoring* are refit on each
  training fold in `Strict` mode; only the *enumeration* of candidate states
  uses node-level edges.
- **Fast mode trade-off:** *(v2)* `SplitMode::Fast` bins globally once, so edges
  are shared across folds (a feature-only relaxation of the guard). Impurity is
  still measured on held-out validation labels. MAE is unavailable; predictions
  and determinism are otherwise equivalent, and parallel still equals serial.
- **Determinism:** identical data + params ⇒ identical tree ⇒ identical
  predictions. Tie-breaks never depend on iteration or thread order.
- **Parallel = serial:** `predict` output and tree shape are independent of
  `parallel` / `n_threads`.
- **Split acceptance:** a node becomes a leaf when it is pure, hits a stopping
  criterion, has no candidate with a finite score (candidates below
  `min_valid_folds` count as non-finite), the best aggregated score is
  `<= min_impurity_decrease`, the best score does not beat the permutation null
  (when `null_permutations > 0`). *(0.8)* Candidates that would leave a child
  below `min_samples_leaf` are dropped before selection, and a node with fewer
  than `2 · min_samples_leaf` samples is not scored at all.
- **Stopping on noise:** *(0.8)* honest scoring makes each noise split score
  `≤ 0` in expectation, but the best of many candidates is usually positive, so
  `min_impurity_decrease = 0` alone still lets noise split, most of all at
  small nodes. The permutation null accounts for the search; `min_samples_leaf`
  also helps.
- **Missing values:** non-finite continuous values and unknown categories never
  match a split state and therefore route to the right child. *(0.8)* With
  threshold splits, a node with missing values can also split on "x is present",
  so informative missingness is usable on both paths.
- **Shrinkage:** *(0.8)* changes leaf predictions only; tree structure is
  identical with and without it.
