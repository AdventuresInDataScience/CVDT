"""CVDT — Cross-Validated Decision Tree.

scikit-learn-compatible estimators backed by a zero-dependency Rust core.

The split-selection novelty (K-fold cross-validated impurity gain, with a
pluggable fold-score aggregator) lives in the Rust crate; this package is a
thin, fully sklearn-conformant wrapper around it.

Example
-------
>>> from cvdt import CVDTClassifier
>>> clf = CVDTClassifier(criterion="gini", cv_folds=5).fit(X, y)
>>> clf.predict(X_test)
"""

from ._estimator import CVDTClassifier, CVDTRegressor

__all__ = ["CVDTClassifier", "CVDTRegressor"]
__version__ = "0.7.0"
