# Building, testing and using CVDT

CVDT has two layers:

- a **zero-dependency Rust core** (the library and its `cargo test` suite), and
- an optional **scikit-learn-compatible Python package** built with maturin,
  behind the `python` cargo feature.

You can use either on its own.

---

## 1. Prerequisites

**Rust core**
- A Rust toolchain (`rustc` + `cargo`), edition 2021, **MSRV 1.63**. Install via
  [rustup](https://rustup.rs). Nothing else — the core pulls no crates and
  builds fully offline.

**Python package (optional)**
- Python 3.8+ and the Rust toolchain above.
- `pip install maturin numpy scikit-learn` (add `pytest` to run the tests).
- Building the extension compiles PyO3, which needs your platform's C toolchain
  and Python development headers.

---

## 2. Build and test the Rust core

```sh
cargo build                 # debug build of the core (std only)
cargo build --release       # optimised build
cargo test                  # run unit + integration tests (see below)
cargo test --release        # same, optimised
cargo fmt                   # format
cargo clippy                # lints
```

`cargo test` runs both layers of the Rust suite:

- **Unit tests** — inline `#[cfg(test)]` modules in every `src/*.rs`, covering
  each component in isolation (criteria, encoder, aggregators, candidate
  generation, cross-validation folds/RNG, histogram scoring, objective metrics,
  parallel map, selector, tree).
- **Integration tests** — the files in `tests/`, exercising the public API
  end-to-end:
  - `pure_functions.rs` — exact-value checks for criteria, quantile encoding,
    aggregators and split selection.
  - `objective_metrics.rs` — confusion matrix + precision/recall/F1/Fβ/accuracy.
  - `classification.rs` / `regression.rs` — learning, probability invariants,
    stopping rules, missing values, categorical features, fast vs. strict.
  - `objective_tree.rs` — objective-driven trees (all metrics, both modes,
    multiclass, self-stopping).
  - `determinism_parallel.rs` — reproducibility and **serial == parallel**
    equivalence for classifier, regressor and objective tasks.
  - `edge_cases.rs` — tiny data, degenerate/all-missing features, clamped folds,
    error paths.
  - `integration.rs` — the original broad end-to-end suite.

Run a subset:

```sh
cargo test --test determinism_parallel      # one integration file
cargo test gini                             # tests whose name contains "gini"
cargo test --lib                            # unit tests only
```

> Note: `cargo test` builds the **core only** and needs no network or Python.
> The Python tests (`tests/test_sklearn.py`) are pytest, not cargo, and require
> the built extension (next section). Running `cargo test --features python`
> would additionally compile the PyO3 binding, which needs Python dev headers.

---

## 3. Build and test the Python package

From the project root:

```sh
pip install maturin numpy scikit-learn pytest

# Compile the Rust extension into your active environment:
maturin develop --release --features python

# Run the Python test suite (sklearn conformance + behaviour):
pytest tests/test_sklearn.py -q
```

To build a redistributable wheel instead:

```sh
maturin build --release --features python
pip install target/wheels/cvdt-*.whl
```

Wheels use PyO3's `abi3-py38`, so one wheel works on CPython 3.8+.

See `PYTHON.md` for the full parameter reference and sklearn integration notes.

---

## 4. Using the Rust core

```rust
use cvdt::{Classification, Column, DecisionTree, FeatureValue, Mean, TreeParams};

// One continuous feature, two classes.
let cols = vec![Column::Continuous(vec![0.0, 0.1, 3.0, 3.1])];
let y = vec![0usize, 0, 1, 1];

let mut tree = DecisionTree::new(Classification::gini(2), TreeParams::default(), Box::new(Mean));
tree.fit(&cols, &y).unwrap();

let pred = tree.predict(&[vec![FeatureValue::cont(3.05)]]);
println!("class = {}, proba = {:?}", pred[0].class, pred[0].proba);
```

Regression and the objective-driven classifier follow the same shape:

```rust
use cvdt::{Average, DecisionTree, Mean, ObjectiveClassification, Regression, TreeParams};

let mut reg = DecisionTree::new(Regression::mse(), TreeParams::default(), Box::new(Mean));

// Choose splits that greedily maximise F1 of the positive class:
let task = ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 });
let mut clf = DecisionTree::new(task, TreeParams::default(), Box::new(Mean));
```

Tune behaviour through `TreeParams` (max_depth, min_samples_*, n_bins, the
`KFold` config, `SplitMode::{Strict, Fast}`, parallelism) and the aggregator
(`Mean`, `Median`, `TrimmedMean`, `SignalToNoise`, `MeanMinusLambdaStd`).

---

## 5. Using the Python package

```python
from cvdt import CVDTClassifier, CVDTRegressor

clf = CVDTClassifier(criterion="gini", cv_folds=5, max_depth=6).fit(X, y)
clf.predict(X_test)
clf.predict_proba(X_test)

# Objective-driven: optimise the metric you care about directly.
clf = CVDTClassifier(objective="recall", average="binary", pos_label=1).fit(X, y)

reg = CVDTRegressor(criterion="mse", mode="fast", n_bins=32).fit(X, y)
```

It is a first-class sklearn estimator, so it composes with the ecosystem:

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    CVDTClassifier(),
    {"max_depth": [4, 8], "objective": [None, "f1"], "mode": ["strict", "fast"]},
    cv=5,
).fit(X, y)
```

---

## 6. Troubleshooting

- **`maturin` can't find Python headers** — install your distro's
  `python3-dev` / `python3-devel`, or use a virtualenv/conda env.
- **numpy/PyO3 version errors** — the binding is written against **PyO3 0.22 +
  numpy 0.22** (pinned in `Cargo.toml`). If you bump to 0.23, the numpy
  `into_pyarray_bound` calls drop their `_bound` suffix.
- **Core won't build offline** — it shouldn't need to fetch anything; only the
  `python` feature has dependencies. Confirm you're not passing
  `--features python` for a core-only build.
