# Changelog

All notable changes to CVDT are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-04

### Added
- **CART-style ordered threshold splits for continuous features**, selected via
  the new `split_style` parameter (`SplitStyle` in the Rust core). A continuous
  split is now a single ordered cut `x < edge` — where `edge` is one of the
  quantile bin boundaries — instead of a one-bin-vs-rest membership test. The
  cut is still scored by K-fold cross-validation on held-out folds; only the
  partition geometry changes. This respects the natural ordering of continuous
  variables and generally generalises better. Available on both the `strict`
  and `fast` (histogram) paths and for impurity- and objective-driven splits.
  - Python: `CVDTClassifier(split_style="threshold")` /
    `CVDTRegressor(split_style="threshold")`. Accepts `"threshold"` (default),
    `"bin"`, or the alias `"bin_membership"`.
  - Rust: `TreeParams.split_style: SplitStyle::{Threshold, BinMembership}`.
  - `n_bins` now doubles as the threshold resolution: it sets the number of
    candidate cut points (`n_bins - 1`) for a continuous feature.

### Changed
- **Default behaviour changed:** `split_style` defaults to `"threshold"`
  (`SplitStyle::Threshold`), so continuous features are now cut as `x < edge`
  out of the box. To reproduce pre-0.7.0 (v1/v2) behaviour exactly, pass
  `split_style="bin"` (Python) / `SplitStyle::BinMembership` (Rust).
- Tree exports render continuous threshold branches as `x[f] < edge` (open
  lower bound / `lower = -inf`); single-bin branches still render as the
  half-open interval `lo <= x[f] < hi`.
- Documentation (`README.md`, `PYTHON.md`, `API_REFERENCE.md`, crate-level docs)
  updated to describe both split styles.

### Fixed
- `python/cvdt/__init__.py` `__version__` was stale at `0.5.0`; it is now kept
  in step with the packaged version.

[0.7.0]: https://github.com/AdventuresInDataScience/CVDT/releases/tag/v0.7.0
