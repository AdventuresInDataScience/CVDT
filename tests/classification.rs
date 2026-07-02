//! End-to-end classification tests through the public tree API.

use cvdt::{
    Classification, Column, DecisionTree, FeatureValue, KFold, Mean, SplitMode, TreeParams,
};

fn params(k: usize, seed: u64) -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(k, seed);
    p
}

fn fast_params(k: usize, seed: u64) -> TreeParams {
    let mut p = params(k, seed);
    p.mode = SplitMode::Fast;
    p.n_bins = 16;
    p
}

/// Two well-separated clusters on one continuous feature.
fn separable() -> (Vec<Column>, Vec<usize>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        x.push(i as f64 * 0.01); // 0.00..0.59 -> class 0
        y.push(0usize);
    }
    for i in 0..60 {
        x.push(3.0 + i as f64 * 0.01); // 3.00..3.59 -> class 1
        y.push(1usize);
    }
    (vec![Column::Continuous(x)], y)
}

fn train_accuracy(tree: &DecisionTree<Classification>, cols: &[Column], y: &[usize]) -> f64 {
    let n = y.len();
    let samples: Vec<Vec<FeatureValue>> = (0..n)
        .map(|i| {
            cols.iter()
                .map(|c| match c {
                    Column::Continuous(v) => FeatureValue::cont(v[i]),
                    Column::Categorical(v) => FeatureValue::cat(v[i]),
                })
                .collect()
        })
        .collect();
    let preds = tree.predict(&samples);
    let correct = preds.iter().zip(y).filter(|(p, &t)| p.class == t).count();
    correct as f64 / n as f64
}

#[test]
fn gini_and_entropy_learn_strict() {
    let (cols, y) = separable();
    for task in [Classification::gini(2), Classification::entropy(2)] {
        let mut t = DecisionTree::new(task, params(5, 7), Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        assert!(train_accuracy(&t, &cols, &y) > 0.95);
        assert!(t.depth() >= 1);
    }
}

#[test]
fn fast_mode_learns() {
    let (cols, y) = separable();
    let mut t = DecisionTree::new(Classification::gini(2), fast_params(5, 7), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(train_accuracy(&t, &cols, &y) > 0.95);
}

#[test]
fn probabilities_are_valid() {
    let (cols, y) = separable();
    let mut t = DecisionTree::new(Classification::gini(2), params(5, 7), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    let preds = t.predict(&[vec![FeatureValue::cont(0.1)], vec![FeatureValue::cont(3.3)]]);
    for p in &preds {
        assert_eq!(p.proba.len(), 2);
        let s: f64 = p.proba.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert!(p.proba.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert!(p.class < 2);
    }
    assert_eq!(preds[0].class, 0);
    assert_eq!(preds[1].class, 1);
}

#[test]
fn multiclass_three_way_separation() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (cluster, base) in [0.0, 3.0, 6.0].iter().enumerate() {
        for i in 0..40 {
            x.push(base + i as f64 * 0.01);
            y.push(cluster);
        }
    }
    let cols = vec![Column::Continuous(x)];
    let mut t = DecisionTree::new(Classification::gini(3), params(5, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(train_accuracy(&t, &cols, &y) > 0.9);
    let preds = t.predict(&[
        vec![FeatureValue::cont(0.2)],
        vec![FeatureValue::cont(3.2)],
        vec![FeatureValue::cont(6.2)],
    ]);
    assert_eq!(preds[0].class, 0);
    assert_eq!(preds[1].class, 1);
    assert_eq!(preds[2].class, 2);
}

#[test]
fn pure_node_becomes_single_leaf() {
    let cols = vec![Column::Continuous(vec![0.0, 1.0, 2.0, 3.0, 4.0])];
    let y = vec![1usize; 5]; // all one class
    let mut t = DecisionTree::new(Classification::gini(2), params(3, 1), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert_eq!(t.depth(), 0);
    assert_eq!(t.n_leaves(), 1);
}

#[test]
fn max_depth_is_respected() {
    let (cols, y) = separable();
    let mut p = params(5, 7);
    p.max_depth = Some(1);
    let mut t = DecisionTree::new(Classification::gini(2), p, Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(t.depth() <= 1);
}

#[test]
fn min_samples_split_blocks_splitting() {
    let (cols, y) = separable();
    let mut p = params(5, 7);
    p.min_samples_split = 10_000; // larger than the dataset
    let mut t = DecisionTree::new(Classification::gini(2), p, Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert_eq!(t.n_leaves(), 1);
}

#[test]
fn missing_values_do_not_panic_and_route() {
    let (mut cols, y) = separable();
    if let Column::Continuous(v) = &mut cols[0] {
        v[0] = f64::NAN; // a training missing value
    }
    let mut t = DecisionTree::new(Classification::gini(2), params(5, 7), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    // Non-finite at predict routes right; just must not panic.
    let preds = t.predict(&[vec![FeatureValue::cont(f64::NAN)]]);
    assert_eq!(preds.len(), 1);
    assert!(preds[0].class < 2);
}

#[test]
fn categorical_feature_determines_class() {
    // category 1 -> class 1, everything else -> class 0
    let mut c = Vec::new();
    let mut y = Vec::new();
    for i in 0..120 {
        let cat = (i % 3) as u32;
        c.push(cat);
        y.push(if cat == 1 { 1usize } else { 0usize });
    }
    let cols = vec![Column::Categorical(c)];
    let mut t = DecisionTree::new(Classification::gini(2), params(5, 3), Box::new(Mean));
    t.fit(&cols, &y).unwrap();
    assert!(train_accuracy(&t, &cols, &y) > 0.95);
    // An unknown category at predict routes to the "not in state" child; no panic.
    let preds = t.predict(&[vec![FeatureValue::cat(999)]]);
    assert_eq!(preds.len(), 1);
}

#[test]
fn convenience_constructor_works() {
    let (cols, y) = separable();
    let mut t = DecisionTree::classifier(2);
    t.fit(&cols, &y).unwrap();
    assert!(train_accuracy(&t, &cols, &y) > 0.9);
}
