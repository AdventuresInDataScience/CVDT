# CVDT — Cross-Validated Decision Tree

CVDT is a decision tree whose novelty is **how splits are scored**. A classical
tree picks the split that most reduces impurity on the node's own data — an
in-sample estimate that is easy to overfit. CVDT instead scores every candidate
split by **K-fold cross-validation**: it measures the impurity reduction each
split achieves on *held-out* folds and aggregates the per-fold scores into a
single robust ranking value. The result is a tree that selects splits which are
not just good on average but *reliably* good across resamples.

**What you get:**

- **Better-generalising splits out of the box** — the anti-overfit behaviour is
  baked into split selection, not bolted on via post-hoc pruning.
- **A tunable risk profile** — swap the fold-score aggregator (`mean`,
  `median`, `trimmed_mean`, `signal_to_noise`, `mean - λ·std`, or your own) to
  reward consistency or penalise volatile splits.
- **Metric-driven trees** — optionally optimise splits for the metric you
  actually care about (precision, recall, F1, Fβ, accuracy) instead of an
  impurity proxy.
- **Full explainability** — a fit produces a single, inspectable tree. Dump it
  as readable rules, render it with Graphviz, or pull the raw node arrays out as
  a dict for custom plots (see [Explainability](#explainability--inspect-the-fitted-tree)).
- **scikit-learn compatible** — `CVDTClassifier` / `CVDTRegressor` drop into
  `Pipeline`, `GridSearchCV`, `cross_val_score`, etc.
- **Fast, portable core** — the engine is a from-scratch Rust crate that uses
  **only the standard library** (no `rayon`, no `PyO3`, no external crates), so
  it builds and tests fully offline; a histogram fast path and std-only
  parallelism keep it quick.

The tree structure itself is deliberately ordinary; the cross-validated
split-evaluation framework is the point.

## The idea

The key move is **decoupling how a split's fold scores are combined from the
scoring itself**. Cross-validation gives each candidate a *vector* of per-fold
gains; a pluggable aggregator collapses that vector into the one number used for
ranking. Choosing the aggregator lets you express what "a good split" means:
`mean` for average gain, `mean - λ·std` to penalise splits whose benefit is
volatile across folds, or `signal_to_noise` to reward consistency directly. So
beyond simply resisting overfitting, CVDT lets you tune the *risk profile* of
split selection.

## Explainability — inspect the fitted tree

Because a fit is a single decision tree, it is fully transparent: you can read
off exactly why any prediction was made. The scikit-learn estimators expose the
fitted structure three ways.

```python
from cvdt import CVDTClassifier

clf = CVDTClassifier(max_depth=3).fit(X, y)

# 1. Readable, rule-style dump.
print(clf.export_text(feature_names=["age", "bmi", "bp"],
                      class_names=["healthy", "at-risk"]))
```

```
if 22.5 <= bmi < 30.1:
  if age < 47:
    class=healthy n=180 proba=[0.911, 0.089]
  else:  # not (age < 47)
    class=at-risk n=95 proba=[0.221, 0.779]
else:  # not (22.5 <= bmi < 30.1)
  class=at-risk n=140 proba=[0.107, 0.893]
```

```python
# 2. Graphviz DOT — render with graphviz, pydot, or dtreeviz.
dot = clf.export_graphviz(feature_names=["age", "bmi", "bp"])

# 3. Raw structure as a dict of parallel arrays, for custom plots or analysis.
tree = clf.get_tree()
tree["feature"]         # split feature per node
tree["lower"], tree["upper"]   # continuous split interval [lo, hi)
tree["children_left"], tree["children_right"]
tree["is_leaf"], tree["n_samples"], tree["proba"]   # ("value" for regressors)

clf.get_depth(), clf.get_n_leaves()   # sklearn-style size accessors
```

One thing to note when reading the output: CVDT splits are **membership tests**,
not CART-style thresholds. A continuous split routes a sample down the *true*
branch when its value falls in a half-open interval `[lo, hi)` (shown as a
one-sided inequality when one end is open); a categorical split routes true when
the feature equals a specific category. The "true" branch is always the one that
matches — missing / non-finite values route to "false".

## Architecture

The code mirrors the pipeline one stage per module; each layer depends only on
the one beneath it and is independently unit tested.

```
data      core column-major representation (continuous / categorical)
encoder   quantile binning of continuous features (edges fit per training fold)
binning   global one-time binning into compact u16 codes (fast path)
candidate enumerate every `feature == state` split present at a node
criterion impurity: Gini, Entropy / Variance, MSE, MAE
cross_validation   K-fold engine: fold generation + per-fold gain scoring
histogram          fast path: single-pass fold histograms + sufficient-stat scoring
aggregation        fold scores -> one number (mean, median, trimmed, StN, mean-λ·std, custom)
selector           rank scored candidates, deterministic tie-break
parallel           std-only order-preserving parallel map
tree               orchestration: recursion, stopping, prediction
```

The tree builder contains no impurity, cross-validation, aggregation or
selection logic — it only orchestrates the components, exactly as the design
requires.

## The unified split interface

Every candidate split is a membership test, `feature == state`:

- **Continuous** features are quantile-binned, so `state` is a bin id. The bin
  *edges* are refit on the **training fold only** during cross-validation, which
  is what prevents leakage. There are `E + 1` bins for `E` cut points; a value's
  bin is the number of edges `<= value`.
- **Categorical** features use their integer category id directly, so `state` is
  a category id. This is conceptually one-hot without the memory cost.

Missing / non-finite continuous values and unknown categories never match a
state, so they route to the "not-in-state" child.

## The per-fold score

For a candidate on a validation fold the score is the impurity **decrease**

```
gain = parent_impurity(val) - weighted_child_impurity(val)
```

computed on the validation labels, where the validation samples are partitioned
using an encoder fit on the *training* fold. Higher is better. A fold is skipped
for a candidate when either child is empty; a candidate with no successful folds
loses. The aggregator turns the vector of per-fold gains into the final ranking
number.

## Usage

```rust
use cvdt::{Classification, Column, DecisionTree, FeatureValue, Mean, TreeParams};

// One continuous feature, two classes.
let x: Vec<f64> = /* ... */;
let y: Vec<usize> = /* class ids ... */;
let columns = vec![Column::Continuous(x)];

let mut tree = DecisionTree::new(
    Classification::gini(2),   // task + criterion
    TreeParams::default(),     // max_depth 8, 8 bins, 5-fold CV, ...
    Box::new(Mean),            // fold-score aggregator
);
tree.fit(&columns, &y).unwrap();

let pred = tree.predict_one(&[FeatureValue::cont(0.42)]);
println!("class {}, proba {:?}", pred.class, pred.proba);
```

Convenience constructors `DecisionTree::classifier(n_classes)` and
`DecisionTree::regressor()` use sensible defaults. Swap the aggregator to change
the risk profile:

```rust
use cvdt::{MeanMinusLambdaStd, SignalToNoise};

Box::new(MeanMinusLambdaStd { lambda: 1.0 }); // penalise volatile splits
Box::new(SignalToNoise::default());           // reward consistent gains
```

## Strict vs Fast mode

`params.mode` selects how splits are evaluated.

**`SplitMode::Strict`** (default) is the exact path. Continuous bin edges are
refit on each training fold and child impurities are computed from materialised
target slices. This is the strongest leakage guard and is unchanged from v1.

**`SplitMode::Fast`** is the histogram path, built for speed. The whole dataset
is quantile-binned **once** into compact `u16` codes; at each node a single pass
over the samples scatters them into per-fold histograms (the K disjoint
validation folds are all built in that one pass), and every candidate split is
then scored from sufficient statistics — class counts for Gini/entropy, and
`(count, Σy, Σy²)` moments for variance/MSE — with the "rest" child obtained by
subtracting the in-bin cell from the fold total. No child slices are ever
allocated.

The trade-off is deliberate: fast mode gives up **per-fold edge refitting**.
Because bins are fixed for the node, the edges have "seen" validation feature
values — but only feature values, never labels; impurity is still measured on
held-out labels, which is the property that gives cross-validated scoring its
anti-overfit behaviour. In practice this is a weak leak for a large speedup.

Notes:
- MAE is unavailable in fast mode (the median has no additive statistic);
  `fit` returns an error suggesting `Strict`.
- Categorical features are densely re-indexed; a feature with more distinct
  levels than a `u16` can hold is rejected at fit time.
- Fast mode is deterministic and `parallel`-safe, exactly like strict mode.

```rust
let mut params = TreeParams::default();
params.mode = SplitMode::Fast;   // histogram path
```

Other speed-oriented details on this path: `u32` sample indices and in-place
Lomuto partitioning on a single shared buffer (no per-node index allocation),
reused histogram scratch buffers, and a `parallel_min_samples` threshold so
tiny nodes are not parallelised.

## Parallelism

Feature evaluation at each node is embarrassingly parallel, so setting
`params.parallel = true` and `params.n_threads = N` fans the per-feature scoring
out across `std::thread::scope` workers. Results are reassembled in feature order
and selection is deterministic, so **parallel output is identical to serial
output** — verified by an integration test (for both strict and fast modes).
Tree growth itself remains sequential, as in the design. Nodes smaller than
`params.parallel_min_samples` are scored serially to avoid thread-spawn overhead.

## Determinism

Fold shuffling uses a self-contained SplitMix64 PRNG seeded from `KFold.seed`
mixed with the node size, so a fit is fully reproducible: same data + params →
same tree → same predictions. Tie-breaks in split selection are resolved by
`(feature, state)` order, never by iteration or thread timing.

## Build and test

```sh
cargo build --release
cargo test           # unit tests beside each module + integration suite in tests/
```

No network access or external crates are needed for the core. The test suite is
comprehensive: inline unit tests in every `src/*.rs`, plus themed integration
files in `tests/` covering pure functions (exact values), objective metrics,
classification, regression, objective-driven trees, determinism and
**serial == parallel** equivalence, and edge cases. The scikit-learn binding has
its own pytest suite in `tests/test_sklearn.py`.

Full compile, test and usage instructions for both the Rust core and the Python
package are in **[BUILD.md](BUILD.md)**; the Python API reference is in
**[PYTHON.md](PYTHON.md)**.

## Design decisions and rationale

- **Equality (`feature == state`) split interface**, not thresholds. It unifies
  continuous and categorical handling behind one abstraction and matches the
  spec's one-vs-rest framing. Threshold splits are a straightforward future
  extension (a `SplitRule` already stores bin bounds).
- **Per-fold gain on validation labels, with edges fit on the training fold.**
  This is the crux of the leakage guard: enumeration of candidate states uses
  node-level encoding, but the *score* only ever sees training-fold-derived bin
  boundaries applied to held-out labels.
- **One `Criterion` trait with an associated `Target` type** (`usize` for
  classification, `f64` for regression) keeps the tree agnostic to the task and
  lets it hold a `Box<dyn Criterion<Target = _>>`.
- **`Task` trait instead of inheritance.** The `DecisionTreeBase → {Classifier,
  Regressor}` hierarchy from the design becomes a generic `DecisionTree<K: Task>`
  where `Task` supplies the target type, the leaf prediction and the criterion —
  the idiomatic Rust rendering of shared-base + specialised-subclass.
- **Aggregation decoupled from cross-validation.** The CV engine only produces
  fold statistics; how they become a ranking number is a separate, swappable
  concern (including a user-supplied `Custom` closure).
- **Zero dependencies.** Determinism, parallelism and everything else are built
  on `std` alone, which keeps the crate auditable and trivially buildable
  offline.

## Objective-driven splits (precision / recall / F1 ...)

Beyond impurity, CVDT can select splits by directly optimising the **metric you
care about**. Instead of tuning hyper-parameters until an impurity proxy happens
to yield good F1, `ObjectiveClassification` scores each candidate by how much it
*improves the objective on the held-out folds*:

```rust
use cvdt::{Average, DecisionTree, ObjectiveClassification, Mean, TreeParams};

let task = ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 });
let mut tree = DecisionTree::new(task, TreeParams::default(), Box::new(Mean));
```

Built-in objectives are precision, recall, F1, Fβ and accuracy, each with
binary / micro / macro / weighted averaging; custom metrics implement the
`ClassObjective` trait. Scoring works in both modes: a split's confusion matrix
is a sufficient statistic, so on the fast path it is read straight from the
per-fold class histograms (validation counts score it; training counts for the
child→class assignment come from `total − fold`). Because the per-fold score is
the improvement over making the node a leaf, objective mode is self-stopping and
tends to produce shallower, metric-tuned trees.

This is a fully modular addition: it slots in through the same `Task`/`Aggregator`
seams as everything else and changes no existing behaviour.

## Python (scikit-learn compatible)

A scikit-learn-compatible binding (v3) is available behind the `python` cargo
feature: `cvdt.CVDTClassifier` and `cvdt.CVDTRegressor` are full
`BaseEstimator`s that work with `Pipeline`, `cross_val_score`, `GridSearchCV`,
`clone` and `get_params`/`set_params`. The binding is a thin PyO3 layer over the
same Rust core; **the core stays zero-dependency** because PyO3 and the `numpy`
crate are pulled in *only* under that feature. Build with
`maturin develop --release --features python`. See `PYTHON.md` for install,
parameters and examples.

## Status and future work

The histogram fast path (v2) implements the largest per-node wins: single-pass
fold histograms, sufficient-statistic scoring with total−cell subtraction,
compact `u16` codes, `u32` indices, in-place partitioning, scratch reuse and a
parallelism threshold. A few further optimisations remain deliberately un-done:
caching a parent node's histogram to derive the smaller child by subtraction
across the recursion, and an audited `unsafe` inner-loop kernel to drop bounds
checks (the core is `#![forbid(unsafe_code)]`; the lint is relaxed only for the
PyO3 build, whose macros expand to unsafe).

The remaining documented extensions stay straightforward thanks to the component
boundaries: additional criteria and aggregators (already trait-based),
alternative CV strategies, threshold splits, sample weighting, richer
missing-value handling, feature importance, pruning, and forest / boosting
variants.
