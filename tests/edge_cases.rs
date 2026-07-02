//! Edge cases and robustness: tiny data, degenerate features, clamped folds.

use cvdt::{Classification, Column, DecisionTree, FeatureValue, KFold, Mean, TreeParams};

fn params(k: usize, seed: u64) -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(k, seed);
    p
}

#[test]
fn k_larger_than_samples_is_clamped() {
    let cols = vec![Column::Continuous(vec![0.0, 1.0, 2.0, 3.0])];
    let y = vec![0usize, 0, 1, 1];
    // k = 10 with only 4 samples must be clamped internally, not panic.
    let mut t = DecisionTree::new(Classification::gini(2), params(10, 5), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(t.n_leaves() >= 1);
}

#[test]
fn single_sample_is_a_leaf() {
    let cols = vec![Column::Continuous(vec![1.0])];
    let y = vec![0usize];
    let mut t = DecisionTree::new(Classification::gini(2), params(3, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert_eq!(t.n_leaves(), 1);
    assert_eq!(t.depth(), 0);
}

#[test]
fn all_missing_feature_yields_leaf() {
    // A feature that is entirely non-finite offers no usable split.
    let cols = vec![Column::Continuous(vec![f64::NAN; 20])];
    let mut y = vec![0usize; 10];
    y.extend(vec![1usize; 10]);
    let mut t = DecisionTree::new(Classification::gini(2), params(5, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert_eq!(t.n_leaves(), 1);
}

#[test]
fn unknown_category_at_predict_routes_without_panic() {
    let c: Vec<u32> = (0..40).map(|i| (i % 2) as u32).collect();
    let y: Vec<usize> = c.iter().map(|&x| x as usize).collect();
    let cols = vec![Column::Categorical(c)];
    let mut t = DecisionTree::new(Classification::gini(2), params(5, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    let preds = t.predict(&[vec![FeatureValue::cat(12345)]]);
    assert_eq!(preds.len(), 1);
    assert!(preds[0].class < 2);
}

#[test]
fn uninformative_extra_feature_is_tolerated() {
    // One informative continuous feature plus a constant feature.
    let mut a = Vec::new();
    let b = vec![7.0; 80];
    let mut y = Vec::new();
    for i in 0..40 {
        a.push(i as f64 * 0.01);
        y.push(0usize);
    }
    for i in 0..40 {
        a.push(2.0 + i as f64 * 0.01);
        y.push(1usize);
    }
    let cols = vec![Column::Continuous(a), Column::Continuous(b)];
    let mut t = DecisionTree::new(Classification::gini(2), params(5, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    let p = t.predict(&[
        vec![FeatureValue::cont(0.1), FeatureValue::cont(7.0)],
        vec![FeatureValue::cont(2.3), FeatureValue::cont(7.0)],
    ]);
    assert_eq!(p[0].class, 0);
    assert_eq!(p[1].class, 1);
}

#[test]
fn mismatched_lengths_error() {
    let cols = vec![Column::Continuous(vec![0.0, 1.0, 2.0])];
    let y = vec![0usize, 1]; // wrong length
    let mut t = DecisionTree::new(Classification::gini(2), params(3, 1), Box::new(Mean));
    assert!(t.fit(&cols, &y).is_err());
}
