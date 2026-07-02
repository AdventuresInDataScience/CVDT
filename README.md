# CVDT — Cross-Validated Decision Tree

A from-scratch, zero-dependency decision tree in Rust whose novelty is **how
splits are scored**. Instead of estimating impurity once on the in-sample node
data, every candidate split is evaluated with **K-fold cross-validation** and the
per-fold scores are aggregated into a single robust ranking value. The tree
itself is deliberately ordinary; the cross-validated split-evaluation framework
is the point.

The crate uses only the standard library — no `rayon`, no `PyO3`, no external
crates at all — so it builds and tests fully offline.

## The idea

A classical tree asks: *of all splits, which most reduces impurity on the data
at this node?* That estimate is in-sample and easy to overfit. CVDT instead
asks: *which split most reduces impurity on held-out data, consistently across
folds?* Each candidate is scored by cross-validation and the fold scores are
collapsed by a pluggable aggregator, so you can select splits that are not just
good on average but *reliably* good (e.g. `mean - λ·std`, or a signal-to-noise
ratio).

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
