//! End-to-end regression tests through the public tree API.

use cvdt::{Column, DecisionTree, FeatureValue, KFold, Mean, Regression, SplitMode, TreeParams};

fn params(k: usize, seed: u64) -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(k, seed);
    p
}

/// Step function: low x -> 0, high x -> 10, with a wide gap between clusters.
fn step() -> (Vec<Column>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        x.push(i as f64 * 0.01);
        y.push(0.0);
    }
    for i in 0..60 {
        x.push(3.0 + i as f64 * 0.01);
        y.push(10.0);
    }
    (vec![Column::Continuous(x)], y)
}

fn predict1(tree: &DecisionTree<Regression>, x: f64) -> f64 {
    tree.predict(&[vec![FeatureValue::cont(x)]])[0]
}

#[test]
fn mse_and_variance_learn_step() {
    for task in [Regression::mse(), Regression::variance()] {
        let mut t = DecisionTree::new(task, params(5, 7), Box::new(Mean));
        t.fit(&step().0, &step().1).unwrap();
        assert!(predict1(&t, 0.2) < 1.0);
        assert!(predict1(&t, 3.3) > 9.0);
    }
}

#[test]
fn mae_learns_step_strict() {
    let (cols, y) = step();
    let mut t = DecisionTree::new(Regression::mae(), params(5, 7), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(predict1(&t, 0.2) < 1.0);
    assert!(predict1(&t, 3.3) > 9.0);
}

#[test]
fn fast_mse_learns_step() {
    let (cols, y) = step();
    let mut p = params(5, 7);
    p.mode = SplitMode::Fast;
    p.n_bins = 16;
    let mut t = DecisionTree::new(Regression::mse(), p, Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(predict1(&t, 0.2) < 1.0);
    assert!(predict1(&t, 3.3) > 9.0);
}

#[test]
fn fast_mode_rejects_mae() {
    let (cols, y) = step();
    let mut p = params(3, 1);
    p.mode = SplitMode::Fast;
    let mut t = DecisionTree::new(Regression::mae(), p, Box::new(Mean));
    assert!(t.fit(&cols, &y).is_err());
}

#[test]
fn constant_target_is_a_single_leaf() {
    let cols = vec![Column::Continuous(vec![0.0, 1.0, 2.0, 3.0, 4.0])];
    let y = vec![42.0; 5];
    let mut t = DecisionTree::new(Regression::mse(), params(3, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert_eq!(t.n_leaves(), 1);
    assert_eq!(t.depth(), 0);
    assert!((predict1(&t, 100.0) - 42.0).abs() < 1e-9);
}

#[test]
fn convenience_regressor_constructor_works() {
    let (cols, y) = step();
    let mut t = DecisionTree::regressor();
    t.fit(&cols, &y).unwrap();
    assert!(predict1(&t, 0.2) < 1.0);
    assert!(predict1(&t, 3.3) > 9.0);
}
