//! Determinism and serial/parallel equivalence — core correctness guarantees
//! of the cross-validation engine and the parallel feature scorer.

use cvdt::{
    Average, Classification, Column, DecisionTree, FeatureValue, KFold, Mean,
    ObjectiveClassification, Regression, SplitMode, TreeParams,
};

/// A small multi-feature classification problem (so several features compete,
/// exercising the parallel scorer).
fn multi_feature_clf() -> (Vec<Column>, Vec<usize>) {
    let n = 200;
    let mut f0 = Vec::new();
    let mut f1 = Vec::new();
    let mut f2 = Vec::new();
    let mut y = Vec::new();
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..n {
        let label = i % 2;
        // f0 is informative, f1/f2 are noisy.
        f0.push(label as f64 + (next() % 1000) as f64 / 5000.0);
        f1.push((next() % 1000) as f64 / 100.0);
        f2.push((next() % 1000) as f64 / 100.0);
        y.push(label);
    }
    (
        vec![
            Column::Continuous(f0),
            Column::Continuous(f1),
            Column::Continuous(f2),
        ],
        y,
    )
}

fn test_samples(cols: &[Column], n: usize) -> Vec<Vec<FeatureValue>> {
    (0..n)
        .map(|i| {
            cols.iter()
                .map(|c| match c {
                    Column::Continuous(v) => FeatureValue::cont(v[i]),
                    Column::Categorical(v) => FeatureValue::cat(v[i]),
                })
                .collect()
        })
        .collect()
}

fn base_params() -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(5, 123);
    p
}

fn parallel_params(mode: SplitMode) -> TreeParams {
    let mut p = base_params();
    p.mode = mode;
    p.n_bins = 16;
    p.parallel = true;
    p.n_threads = 4;
    p.parallel_min_samples = 1; // force parallelism even on small nodes
    p
}

fn serial_params(mode: SplitMode) -> TreeParams {
    let mut p = base_params();
    p.mode = mode;
    p.n_bins = 16;
    p.parallel = false;
    p
}

#[test]
fn classifier_is_deterministic_across_refits() {
    let (cols, y) = multi_feature_clf();
    let samples = test_samples(&cols, y.len());
    let mut a = DecisionTree::new(Classification::gini(2), base_params(), Box::new(Mean));
    a.fit(&cols, &y).unwrap();
    let mut b = DecisionTree::new(Classification::gini(2), base_params(), Box::new(Mean));
    b.fit(&cols, &y).unwrap();
    assert_eq!(a.predict(&samples), b.predict(&samples));
    assert_eq!(a.depth(), b.depth());
    assert_eq!(a.n_leaves(), b.n_leaves());
}

#[test]
fn classifier_serial_equals_parallel_strict() {
    let (cols, y) = multi_feature_clf();
    let samples = test_samples(&cols, y.len());
    let mut s = DecisionTree::new(
        Classification::gini(2),
        serial_params(SplitMode::Strict),
        Box::new(Mean),
    );
    s.fit(&cols, &y).unwrap();
    let mut p = DecisionTree::new(
        Classification::gini(2),
        parallel_params(SplitMode::Strict),
        Box::new(Mean),
    );
    p.fit(&cols, &y).unwrap();
    assert_eq!(s.predict(&samples), p.predict(&samples));
}

#[test]
fn classifier_serial_equals_parallel_fast() {
    let (cols, y) = multi_feature_clf();
    let samples = test_samples(&cols, y.len());
    let mut s = DecisionTree::new(
        Classification::gini(2),
        serial_params(SplitMode::Fast),
        Box::new(Mean),
    );
    s.fit(&cols, &y).unwrap();
    let mut p = DecisionTree::new(
        Classification::gini(2),
        parallel_params(SplitMode::Fast),
        Box::new(Mean),
    );
    p.fit(&cols, &y).unwrap();
    assert_eq!(s.predict(&samples), p.predict(&samples));
}

#[test]
fn regressor_serial_equals_parallel() {
    let (cols, y0) = multi_feature_clf();
    let y: Vec<f64> = y0.iter().map(|&c| c as f64 * 5.0).collect();
    let samples = test_samples(&cols, y.len());
    let mut s = DecisionTree::new(
        Regression::mse(),
        serial_params(SplitMode::Strict),
        Box::new(Mean),
    );
    s.fit(&cols, &y).unwrap();
    let mut p = DecisionTree::new(
        Regression::mse(),
        parallel_params(SplitMode::Strict),
        Box::new(Mean),
    );
    p.fit(&cols, &y).unwrap();
    assert_eq!(s.predict(&samples), p.predict(&samples));
}

#[test]
fn objective_serial_equals_parallel_both_modes() {
    let (cols, y) = multi_feature_clf();
    let samples = test_samples(&cols, y.len());
    for mode in [SplitMode::Strict, SplitMode::Fast] {
        let mut s = DecisionTree::new(
            ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
            serial_params(mode),
            Box::new(Mean),
        );
        s.fit(&cols, &y).unwrap();
        let mut p = DecisionTree::new(
            ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
            parallel_params(mode),
            Box::new(Mean),
        );
        p.fit(&cols, &y).unwrap();
        assert_eq!(s.predict(&samples), p.predict(&samples));
    }
}
