# CVDT for Python (scikit-learn compatible)

`cvdt` ships two estimators that follow the scikit-learn API and are backed by
the zero-dependency Rust core:

- `cvdt.CVDTClassifier` — `ClassifierMixin`, `predict` / `predict_proba` /
  `predict_log_proba`
- `cvdt.CVDTRegressor` — `RegressorMixin`, `predict`

Both are full `BaseEstimator`s: they work with `Pipeline`, `cross_val_score`,
`GridSearchCV`, `clone`, and `get_params` / `set_params`.

## How the binding is layered

```
scikit-learn API  ──  python/cvdt/_estimator.py   (validation, tags, classes_, label encoding)
                          │  numpy arrays
native extension  ──  cvdt._cvdt  (src/python.rs, PyO3)   (numpy → flat Columns → tree)
                          │
Rust core         ──  DecisionTree<Task>  (zero-dependency crate)
```

All sklearn semantics live in Python; the Rust side stays a thin, fast core.
**The core crate remains zero-dependency** — PyO3 and the `numpy` crate are
pulled in *only* under the `python` cargo feature. A plain `cargo build` /
`cargo test` still compiles nothing but `std`.

## Building

Requires a Rust toolchain and [maturin](https://www.maturin.rs/).

```sh
pip install maturin numpy scikit-learn
# dev build into the current environment:
maturin develop --release --features python
# or build a wheel:
maturin build --release --features python
pip install target/wheels/cvdt-*.whl
```

Wheels are built with PyO3's `abi3-py38`, so a single wheel works across
CPython 3.8+.

## Quickstart

```python
import numpy as np
from cvdt import CVDTClassifier, CVDTRegressor

clf = CVDTClassifier(criterion="gini", cv_folds=5, max_depth=6)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_test, y_test)

reg = CVDTRegressor(criterion="mse", mode="fast", n_bins=32)
reg.fit(X_train, y_train)
```

Everything composes with sklearn:

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    CVDTClassifier(),
    {"max_depth": [4, 8], "aggregator": ["mean", "mean_minus_lambda_std"], "mode": ["strict", "fast"]},
    cv=5,
).fit(X, y)
```

See `examples/sklearn_usage.py`.

## Parameters

Common to both estimators:

| Parameter | Default | Meaning |
|---|---|---|
| `max_depth` | `8` | Max split levels; `None` for unlimited. |
| `min_samples_split` | `2` | Min samples for a node to be splittable. |
| `min_samples_leaf` | `1` | Min samples each child must receive. |
| `min_impurity_decrease` | `0.0` | Min aggregated gain to accept a split. |
| `n_bins` | `8` | Quantile bins for continuous features. |
| `cv_folds` | `5` | K in the K-fold split evaluation. |
| `cv_seed` | `42` | Fold-shuffle seed (determinism). |
| `cv_shuffle` | `True` | Shuffle before folding. |
| `mode` | `"strict"` | `"strict"` (per-fold edges) or `"fast"` (histogram path). |
| `aggregator` | `"mean"` | `mean`, `median`, `trimmed_mean`, `signal_to_noise`, `mean_minus_lambda_std`. |
| `agg_frac` | `0.1` | Trim fraction for `trimmed_mean`. |
| `agg_eps` | `1e-12` | Stabiliser for `signal_to_noise`. |
| `agg_lambda` | `1.0` | Std penalty for `mean_minus_lambda_std`. |
| `parallel` | `False` | Evaluate features in parallel. |
| `n_threads` | `1` | Worker threads when `parallel`. |
| `parallel_min_samples` | `512` | Only parallelise nodes at least this large. |
| `categorical_features` | `None` | Column indices to treat as categorical. |

Estimator-specific:

- `CVDTClassifier.criterion`: `"gini"` (default) or `"entropy"`.
- `CVDTRegressor.criterion`: `"mse"` (default), `"variance"`, or `"mae"`.

### Objective-driven classification

Instead of an impurity proxy, `CVDTClassifier` can select splits to directly
optimise a target metric on the held-out folds:

| Parameter | Default | Meaning |
|---|---|---|
| `objective` | `None` | `None` (use `criterion`), or `precision`/`recall`/`f1`/`fbeta`/`accuracy`. |
| `average` | `"binary"` | `binary`, `micro`, `macro`, or `weighted`. |
| `pos_label` | `1` | Positive class (index into sorted `classes_`) when `average="binary"`. |
| `beta` | `1.0` | β for `objective="fbeta"`. |

```python
# Optimise recall of the positive class on an imbalanced problem:
clf = CVDTClassifier(objective="recall", average="binary", pos_label=1).fit(X, y)

# Macro-F1 for multiclass:
clf = CVDTClassifier(objective="f1", average="macro").fit(X, y)
```

When `objective` is set, `criterion` is ignored. Splits are accepted only when
they improve the metric over making the node a leaf, so objective-mode trees are
usually shallower and tuned to the metric; set `min_impurity_decrease` below 0
to allow non-improving splits.

## Notes and caveats

- **Interpretability**: a fitted `CVDTClassifier`/`CVDTRegressor` is a single
  tree, so `est.get_depth()` and `est.get_n_leaves()` (sklearn-style accessors)
  report its size; total nodes of the binary tree is `2 * get_n_leaves() - 1`.
  The tree can be inspected and visualised:
  - `est.export_text(feature_names=..., class_names=...)` — readable rule dump.
  - `est.export_graphviz(...)` — Graphviz DOT string (render with graphviz,
    pydot, or dtreeviz).
  - `est.get_tree()` — the tree as a dict of parallel arrays for custom plots.

  Note CVDT splits are membership tests, so continuous branches read as
  intervals (`lo <= x[f] < hi`) rather than single thresholds, and the "true"
  branch is the one that matches (missing values route to "false").

- **`categorical_features`** columns are read as integer category ids. Feed
  already-integer-encoded values (e.g. from `OrdinalEncoder`); values are
  rounded to the nearest non-negative integer, and negatives / non-finite are
  treated as an unknown category (routed to the "not-in-state" child).
- **Missing values**: NaN / non-finite continuous values are accepted and
  routed right, so `allow_nan` is set in the estimator tags.
- **`mode="fast"` + `criterion="mae"`** is unsupported (the median has no
  additive sufficient statistic for the histogram path) and raises on `fit`.
  Use `mode="strict"` for MAE.
- **`mode="fast"`** relaxes the leakage guard slightly: bins are fit once per
  node instead of per training fold, so bin *edges* see validation feature
  values — but impurity is still measured on held-out *labels*. It is much
  faster; use `"strict"` when the stronger guard matters.

## Tests

`tests/test_sklearn.py` covers learning, `predict_proba`, `clone` /
`get_params`, fast vs. strict, NaN and categorical handling, pipelines, and a
full `sklearn.utils.estimator_checks.check_estimator` sweep. Build the
extension first, then run `pytest`.
