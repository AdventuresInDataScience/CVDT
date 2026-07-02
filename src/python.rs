//! PyO3 binding — the *low-level* native core exposed to Python.
//!
//! This module deliberately stays thin: it converts numpy arrays into the
//! crate's flat [`Column`] buffers, drives [`DecisionTree`], and hands results
//! back as numpy arrays. All scikit-learn semantics (parameter validation,
//! `get_params`/`set_params`, `classes_`, label encoding, tags) live in the
//! Python layer in `python/cvdt/_estimator.py`, which wraps these classes.
//!
//! Built only under `--features python`.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::aggregation::{
    Aggregator, Mean, MeanMinusLambdaStd, Median, SignalToNoise, TrimmedMean,
};
use crate::data::{CatId, Column, FeatureValue, UNKNOWN_CAT};
use crate::objective::{Accuracy, Average, ClassObjective, FBeta, Precision, Recall, F1};
use crate::tree::{
    ClassPrediction, Classification, DecisionTree, ExportedNode, Node, ObjectiveClassification,
    Regression, SplitMode, SplitRule, TreeParams,
};

// Pack a flattened tree into a Python dict of parallel arrays. `with_classes`
// controls whether classification-only fields (predicted class + proba) are
// included; `n_classes` is added for classifiers.
fn tree_to_dict<'py>(
    py: Python<'py>,
    nodes: &[ExportedNode],
    with_classes: bool,
    n_classes: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = nodes.len();
    let mut is_leaf = Vec::with_capacity(n);
    let mut children_left = Vec::with_capacity(n);
    let mut children_right = Vec::with_capacity(n);
    let mut feature = Vec::with_capacity(n);
    let mut is_categorical = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    let mut category = Vec::with_capacity(n);
    let mut n_samples = Vec::with_capacity(n);
    for e in nodes {
        is_leaf.push(e.is_leaf);
        children_left.push(e.left);
        children_right.push(e.right);
        feature.push(e.feature);
        is_categorical.push(e.is_categorical);
        lower.push(e.lower);
        upper.push(e.upper);
        category.push(e.category);
        n_samples.push(e.n_samples as u64);
    }
    let d = PyDict::new_bound(py);
    d.set_item("node_count", n)?;
    d.set_item("is_leaf", is_leaf)?;
    d.set_item("children_left", children_left)?;
    d.set_item("children_right", children_right)?;
    d.set_item("feature", feature)?;
    d.set_item("is_categorical", is_categorical)?;
    d.set_item("lower", lower)?;
    d.set_item("upper", upper)?;
    d.set_item("category", category)?;
    d.set_item("n_samples", n_samples)?;
    if with_classes {
        let predicted_class: Vec<i64> = nodes.iter().map(|e| e.class).collect();
        let proba: Vec<Vec<f64>> = nodes.iter().map(|e| e.proba.clone()).collect();
        d.set_item("predicted_class", predicted_class)?;
        d.set_item("proba", proba)?;
        d.set_item("n_classes", n_classes)?;
    } else {
        let value: Vec<f64> = nodes.iter().map(|e| e.value).collect();
        d.set_item("value", value)?;
    }
    Ok(d)
}

// --------------------------------------------------------------------------
// Tree reconstruction (for unpickling a fitted estimator)
//
// The inverse of `tree_to_dict`: rebuild the node tree from the parallel arrays
// produced by `export_tree`. Prediction reads only the node tree, so a model
// rebuilt this way predicts identically to the original.
// --------------------------------------------------------------------------

fn get_vec<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Vec<T>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("tree state missing key {key:?}")))?
        .extract()
}

/// The flat node arrays shared by classification and regression exports.
struct FlatTree {
    is_leaf: Vec<bool>,
    left: Vec<i64>,
    right: Vec<i64>,
    feature: Vec<i64>,
    is_categorical: Vec<bool>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    category: Vec<i64>,
    n_samples: Vec<u64>,
}

impl FlatTree {
    fn from_dict(d: &Bound<PyDict>) -> PyResult<Self> {
        let ft = FlatTree {
            is_leaf: get_vec(d, "is_leaf")?,
            left: get_vec(d, "children_left")?,
            right: get_vec(d, "children_right")?,
            feature: get_vec(d, "feature")?,
            is_categorical: get_vec(d, "is_categorical")?,
            lower: get_vec(d, "lower")?,
            upper: get_vec(d, "upper")?,
            category: get_vec(d, "category")?,
            n_samples: get_vec(d, "n_samples")?,
        };
        if ft.is_leaf.is_empty() {
            return Err(PyValueError::new_err("tree state has no nodes"));
        }
        Ok(ft)
    }

    /// Reconstruct the routing rule for internal node `id`. `export_tree` writes
    /// open bin ends as +/-inf, so those map back to `None` bounds.
    fn rule_at(&self, id: usize) -> SplitRule {
        if self.is_categorical[id] {
            SplitRule::Category {
                feature: self.feature[id] as usize,
                category: self.category[id] as u32,
            }
        } else {
            let lo = self.lower[id];
            let hi = self.upper[id];
            SplitRule::ContinuousBin {
                feature: self.feature[id] as usize,
                lower: (!(lo.is_infinite() && lo.is_sign_negative())).then_some(lo),
                upper: (!(hi.is_infinite() && hi.is_sign_positive())).then_some(hi),
            }
        }
    }
}

fn build_class_node(
    ft: &FlatTree,
    class: &[i64],
    proba: &[Vec<f64>],
    id: usize,
) -> Node<ClassPrediction> {
    if ft.is_leaf[id] {
        Node::Leaf {
            prediction: ClassPrediction {
                class: class[id].max(0) as usize,
                proba: proba[id].clone(),
            },
            n_samples: ft.n_samples[id] as usize,
        }
    } else {
        Node::Internal {
            rule: ft.rule_at(id),
            left: Box::new(build_class_node(ft, class, proba, ft.left[id] as usize)),
            right: Box::new(build_class_node(ft, class, proba, ft.right[id] as usize)),
        }
    }
}

fn build_regr_node(ft: &FlatTree, value: &[f64], id: usize) -> Node<f64> {
    if ft.is_leaf[id] {
        Node::Leaf {
            prediction: value[id],
            n_samples: ft.n_samples[id] as usize,
        }
    } else {
        Node::Internal {
            rule: ft.rule_at(id),
            left: Box::new(build_regr_node(ft, value, ft.left[id] as usize)),
            right: Box::new(build_regr_node(ft, value, ft.right[id] as usize)),
        }
    }
}

fn sorted_categorical(categorical: Vec<usize>) -> Vec<usize> {
    let mut c = categorical;
    c.sort_unstable();
    c.dedup();
    c
}

// --------------------------------------------------------------------------
// Shared helpers
// --------------------------------------------------------------------------

fn is_categorical(j: usize, categorical: &[usize]) -> bool {
    // `categorical` is kept sorted by the caller (the Python layer sorts it).
    categorical.binary_search(&j).is_ok()
}

fn to_cat(val: f64) -> CatId {
    if val.is_finite() && val >= 0.0 {
        val.round() as CatId
    } else {
        UNKNOWN_CAT
    }
}

/// Build column-major [`Column`]s from a row-major numpy view.
fn build_columns(x: ArrayView2<f64>, categorical: &[usize]) -> Vec<Column> {
    let n = x.nrows();
    let d = x.ncols();
    let mut cols = Vec::with_capacity(d);
    for j in 0..d {
        if is_categorical(j, categorical) {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(to_cat(x[[i, j]]));
            }
            cols.push(Column::Categorical(v));
        } else {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(x[[i, j]]);
            }
            cols.push(Column::Continuous(v));
        }
    }
    cols
}

/// Build per-sample feature rows for prediction.
fn build_rows(x: ArrayView2<f64>, categorical: &[usize]) -> Vec<Vec<FeatureValue>> {
    let n = x.nrows();
    let d = x.ncols();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(d);
        for j in 0..d {
            let val = x[[i, j]];
            if is_categorical(j, categorical) {
                row.push(FeatureValue::cat(to_cat(val)));
            } else {
                row.push(FeatureValue::cont(val));
            }
        }
        rows.push(row);
    }
    rows
}

#[allow(clippy::too_many_arguments)]
fn build_params(
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_impurity_decrease: f64,
    n_bins: usize,
    cv_folds: usize,
    cv_seed: u64,
    cv_shuffle: bool,
    mode: &str,
    parallel: bool,
    n_threads: usize,
    parallel_min_samples: usize,
) -> PyResult<TreeParams> {
    let mode = match mode {
        "strict" => SplitMode::Strict,
        "fast" => SplitMode::Fast,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown mode {other:?} (expected \"strict\" or \"fast\")"
            )))
        }
    };
    Ok(TreeParams {
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        n_bins,
        cv: crate::cross_validation::KFold {
            k: cv_folds,
            seed: cv_seed,
            shuffle: cv_shuffle,
        },
        mode,
        parallel,
        n_threads,
        parallel_min_samples,
    })
}

fn build_aggregator(name: &str, frac: f64, eps: f64, lambda: f64) -> PyResult<Box<dyn Aggregator>> {
    Ok(match name {
        "mean" => Box::new(Mean),
        "median" => Box::new(Median),
        "trimmed_mean" => Box::new(TrimmedMean { frac }),
        "signal_to_noise" => Box::new(SignalToNoise { eps }),
        "mean_minus_lambda_std" => Box::new(MeanMinusLambdaStd { lambda }),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown aggregator {other:?}"
            )))
        }
    })
}

fn build_average(average: &str, pos_label: usize) -> PyResult<Average> {
    Ok(match average {
        "binary" => Average::Binary { pos_label },
        "micro" => Average::Micro,
        "macro" => Average::Macro,
        "weighted" => Average::Weighted,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown average {other:?} (expected binary/micro/macro/weighted)"
            )))
        }
    })
}

fn build_objective(
    name: &str,
    average: &str,
    pos_label: usize,
    beta: f64,
) -> PyResult<Box<dyn ClassObjective>> {
    let avg = build_average(average, pos_label)?;
    Ok(match name {
        "precision" => Box::new(Precision { average: avg }),
        "recall" => Box::new(Recall { average: avg }),
        "f1" => Box::new(F1 { average: avg }),
        "fbeta" => Box::new(FBeta { beta, average: avg }),
        "accuracy" => Box::new(Accuracy),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown objective {other:?} (expected precision/recall/f1/fbeta/accuracy)"
            )))
        }
    })
}

// --------------------------------------------------------------------------
// Classifier
// --------------------------------------------------------------------------

/// Native classification tree. Targets are already label-encoded to `0..n`
/// by the Python layer; class recovery happens there via `classes_`.
#[pyclass]
pub struct RawClassifier {
    inner: DecisionTree<Classification>,
    n_classes: usize,
    categorical: Vec<usize>,
    fitted: bool,
}

#[pymethods]
impl RawClassifier {
    #[new]
    #[pyo3(signature = (
        n_classes, criterion, max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, n_bins, cv_folds, cv_seed, cv_shuffle, mode,
        aggregator, agg_frac, agg_eps, agg_lambda, parallel, n_threads,
        parallel_min_samples, categorical
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_classes: usize,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        n_bins: usize,
        cv_folds: usize,
        cv_seed: u64,
        cv_shuffle: bool,
        mode: &str,
        aggregator: &str,
        agg_frac: f64,
        agg_eps: f64,
        agg_lambda: f64,
        parallel: bool,
        n_threads: usize,
        parallel_min_samples: usize,
        categorical: Vec<usize>,
    ) -> PyResult<Self> {
        let task = match criterion {
            "gini" => Classification::gini(n_classes),
            "entropy" => Classification::entropy(n_classes),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown classification criterion {other:?} (expected \"gini\" or \"entropy\")"
                )))
            }
        };
        let params = build_params(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            n_bins,
            cv_folds,
            cv_seed,
            cv_shuffle,
            mode,
            parallel,
            n_threads,
            parallel_min_samples,
        )?;
        let agg = build_aggregator(aggregator, agg_frac, agg_eps, agg_lambda)?;
        let mut categorical = categorical;
        categorical.sort_unstable();
        categorical.dedup();
        Ok(Self {
            inner: DecisionTree::new(task, params, agg),
            n_classes,
            categorical,
            fitted: false,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i64>) -> PyResult<()> {
        let cols = build_columns(x.as_array(), &self.categorical);
        let y = y.as_array();
        let targets: Vec<usize> = y.iter().map(|&v| v.max(0) as usize).collect();
        self.inner
            .fit(&cols, &targets)
            .map_err(PyValueError::new_err)?;
        self.fitted = true;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        let rows = build_rows(x.as_array(), &self.categorical);
        let preds = self.inner.predict(&rows);
        let out: Vec<i64> = preds.iter().map(|p| p.class as i64).collect();
        Ok(out.into_pyarray_bound(py))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        let rows = build_rows(x.as_array(), &self.categorical);
        let preds = self.inner.predict(&rows);
        let n = preds.len();
        let c = self.n_classes.max(1);
        let mut flat = vec![0.0f64; n * c];
        for (i, p) in preds.iter().enumerate() {
            for (j, &pr) in p.proba.iter().enumerate() {
                if j < c {
                    flat[i * c + j] = pr;
                }
            }
        }
        let arr = Array2::from_shape_vec((n, c), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn n_leaves(&self) -> usize {
        self.inner.n_leaves()
    }

    fn export_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        tree_to_dict(py, &self.inner.export_nodes(), true, self.n_classes)
    }

    /// Rebuild a fitted classifier from an `export_tree` dump (used to unpickle).
    #[staticmethod]
    fn from_tree(
        tree: &Bound<PyDict>,
        n_classes: usize,
        categorical: Vec<usize>,
    ) -> PyResult<Self> {
        let ft = FlatTree::from_dict(tree)?;
        let class: Vec<i64> = get_vec(tree, "predicted_class")?;
        let proba: Vec<Vec<f64>> = get_vec(tree, "proba")?;
        let root = build_class_node(&ft, &class, &proba, 0);
        let mut inner = DecisionTree::new(
            Classification::gini(n_classes),
            TreeParams::default(),
            Box::new(Mean),
        );
        inner.set_fitted_root(root);
        Ok(Self {
            inner,
            n_classes,
            categorical: sorted_categorical(categorical),
            fitted: true,
        })
    }
}

// --------------------------------------------------------------------------
// Objective-driven classifier
// --------------------------------------------------------------------------

/// Native classification tree that selects splits by a metric objective
/// (precision/recall/F1/Fβ/accuracy) instead of impurity.
#[pyclass]
pub struct RawObjectiveClassifier {
    inner: DecisionTree<ObjectiveClassification>,
    n_classes: usize,
    categorical: Vec<usize>,
    fitted: bool,
}

#[pymethods]
impl RawObjectiveClassifier {
    #[new]
    #[pyo3(signature = (
        n_classes, objective, average, pos_label, beta, max_depth,
        min_samples_split, min_samples_leaf, min_impurity_decrease, n_bins,
        cv_folds, cv_seed, cv_shuffle, mode, aggregator, agg_frac, agg_eps,
        agg_lambda, parallel, n_threads, parallel_min_samples, categorical
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_classes: usize,
        objective: &str,
        average: &str,
        pos_label: usize,
        beta: f64,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        n_bins: usize,
        cv_folds: usize,
        cv_seed: u64,
        cv_shuffle: bool,
        mode: &str,
        aggregator: &str,
        agg_frac: f64,
        agg_eps: f64,
        agg_lambda: f64,
        parallel: bool,
        n_threads: usize,
        parallel_min_samples: usize,
        categorical: Vec<usize>,
    ) -> PyResult<Self> {
        let obj = build_objective(objective, average, pos_label, beta)?;
        let task = ObjectiveClassification::new(n_classes, obj);
        let params = build_params(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            n_bins,
            cv_folds,
            cv_seed,
            cv_shuffle,
            mode,
            parallel,
            n_threads,
            parallel_min_samples,
        )?;
        let agg = build_aggregator(aggregator, agg_frac, agg_eps, agg_lambda)?;
        let mut categorical = categorical;
        categorical.sort_unstable();
        categorical.dedup();
        Ok(Self {
            inner: DecisionTree::new(task, params, agg),
            n_classes,
            categorical,
            fitted: false,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i64>) -> PyResult<()> {
        let cols = build_columns(x.as_array(), &self.categorical);
        let y = y.as_array();
        let targets: Vec<usize> = y.iter().map(|&v| v.max(0) as usize).collect();
        self.inner
            .fit(&cols, &targets)
            .map_err(PyValueError::new_err)?;
        self.fitted = true;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        let rows = build_rows(x.as_array(), &self.categorical);
        let preds = self.inner.predict(&rows);
        let out: Vec<i64> = preds.iter().map(|p| p.class as i64).collect();
        Ok(out.into_pyarray_bound(py))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        let rows = build_rows(x.as_array(), &self.categorical);
        let preds = self.inner.predict(&rows);
        let n = preds.len();
        let c = self.n_classes.max(1);
        let mut flat = vec![0.0f64; n * c];
        for (i, p) in preds.iter().enumerate() {
            for (j, &pr) in p.proba.iter().enumerate() {
                if j < c {
                    flat[i * c + j] = pr;
                }
            }
        }
        let arr = Array2::from_shape_vec((n, c), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn n_leaves(&self) -> usize {
        self.inner.n_leaves()
    }

    fn export_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        tree_to_dict(py, &self.inner.export_nodes(), true, self.n_classes)
    }

    /// Rebuild a fitted objective classifier from an `export_tree` dump. The
    /// objective plays no part in prediction, so a placeholder task is used.
    #[staticmethod]
    fn from_tree(
        tree: &Bound<PyDict>,
        n_classes: usize,
        categorical: Vec<usize>,
    ) -> PyResult<Self> {
        let ft = FlatTree::from_dict(tree)?;
        let class: Vec<i64> = get_vec(tree, "predicted_class")?;
        let proba: Vec<Vec<f64>> = get_vec(tree, "proba")?;
        let root = build_class_node(&ft, &class, &proba, 0);
        let mut inner = DecisionTree::new(
            ObjectiveClassification::accuracy(n_classes),
            TreeParams::default(),
            Box::new(Mean),
        );
        inner.set_fitted_root(root);
        Ok(Self {
            inner,
            n_classes,
            categorical: sorted_categorical(categorical),
            fitted: true,
        })
    }
}

// --------------------------------------------------------------------------
// Regressor
// --------------------------------------------------------------------------

/// Native regression tree.
#[pyclass]
pub struct RawRegressor {
    inner: DecisionTree<Regression>,
    categorical: Vec<usize>,
    fitted: bool,
}

#[pymethods]
impl RawRegressor {
    #[new]
    #[pyo3(signature = (
        criterion, max_depth, min_samples_split, min_samples_leaf,
        min_impurity_decrease, n_bins, cv_folds, cv_seed, cv_shuffle, mode,
        aggregator, agg_frac, agg_eps, agg_lambda, parallel, n_threads,
        parallel_min_samples, categorical
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_impurity_decrease: f64,
        n_bins: usize,
        cv_folds: usize,
        cv_seed: u64,
        cv_shuffle: bool,
        mode: &str,
        aggregator: &str,
        agg_frac: f64,
        agg_eps: f64,
        agg_lambda: f64,
        parallel: bool,
        n_threads: usize,
        parallel_min_samples: usize,
        categorical: Vec<usize>,
    ) -> PyResult<Self> {
        let task = match criterion {
            "mse" => Regression::mse(),
            "variance" => Regression::variance(),
            "mae" => Regression::mae(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown regression criterion {other:?} (expected \"mse\", \"variance\" or \"mae\")"
                )))
            }
        };
        let params = build_params(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            n_bins,
            cv_folds,
            cv_seed,
            cv_shuffle,
            mode,
            parallel,
            n_threads,
            parallel_min_samples,
        )?;
        let agg = build_aggregator(aggregator, agg_frac, agg_eps, agg_lambda)?;
        let mut categorical = categorical;
        categorical.sort_unstable();
        categorical.dedup();
        Ok(Self {
            inner: DecisionTree::new(task, params, agg),
            categorical,
            fitted: false,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let cols = build_columns(x.as_array(), &self.categorical);
        let y = y.as_array();
        let targets: Vec<f64> = y.iter().copied().collect();
        self.inner
            .fit(&cols, &targets)
            .map_err(PyValueError::new_err)?;
        self.fitted = true;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        let rows = build_rows(x.as_array(), &self.categorical);
        let preds = self.inner.predict(&rows);
        Ok(preds.into_pyarray_bound(py))
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn n_leaves(&self) -> usize {
        self.inner.n_leaves()
    }

    fn export_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.fitted {
            return Err(PyValueError::new_err("estimator is not fitted"));
        }
        tree_to_dict(py, &self.inner.export_nodes(), false, 0)
    }

    /// Rebuild a fitted regressor from an `export_tree` dump (used to unpickle).
    #[staticmethod]
    fn from_tree(tree: &Bound<PyDict>, categorical: Vec<usize>) -> PyResult<Self> {
        let ft = FlatTree::from_dict(tree)?;
        let value: Vec<f64> = get_vec(tree, "value")?;
        let root = build_regr_node(&ft, &value, 0);
        let mut inner = DecisionTree::new(Regression::mse(), TreeParams::default(), Box::new(Mean));
        inner.set_fitted_root(root);
        Ok(Self {
            inner,
            categorical: sorted_categorical(categorical),
            fitted: true,
        })
    }
}

// --------------------------------------------------------------------------
// Module
// --------------------------------------------------------------------------

/// The native extension module `cvdt._cvdt`.
#[pymodule]
fn _cvdt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawClassifier>()?;
    m.add_class::<RawObjectiveClassifier>()?;
    m.add_class::<RawRegressor>()?;
    m.add(
        "__doc__",
        "Native CVDT core (Rust/PyO3). Use cvdt.CVDTClassifier / cvdt.CVDTRegressor.",
    )?;
    Ok(())
}
