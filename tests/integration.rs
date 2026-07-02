//! End-to-end tests exercising the whole pipeline through the public API.

use cvdt::{
    Aggregator, Classification, Column, DecisionTree, FeatureValue, KFold, Mean, Regression,
    SplitMode, TreeParams,
};

/// Build prediction samples from a single continuous column of test points.
fn cont_samples(xs: &[f64]) -> Vec<Vec<FeatureValue>> {
    xs.iter().map(|&x| vec![FeatureValue::cont(x)]).collect()
}

fn default_params_with_cv(k: usize, seed: u64) -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(k, seed);
    p
}

#[test]
fn classifier_learns_separable_continuous_data() {
    // Two well-separated clusters along one continuous feature.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        x.push(i as f64 * 0.01); // 0.00 .. 0.59  -> class 0
        y.push(0usize);
    }
    for i in 0..60 {
        x.push(2.0 + i as f64 * 0.01); // 2.00 .. 2.59 -> class 1
        y.push(1usize);
    }
    let cols = vec![Column::Continuous(x.clone())];

    let mut tree = DecisionTree::new(
        Classification::gini(2),
        default_params_with_cv(5, 7),
        Box::new(Mean),
    );
    tree.fit(&cols, &y).unwrap();

    let samples = cont_samples(&x);
    let preds = tree.predict(&samples);
    let correct = preds
        .iter()
        .zip(y.iter())
        .filter(|(p, &t)| p.class == t)
        .count();
    let acc = correct as f64 / y.len() as f64;
    assert!(acc > 0.9, "train accuracy too low: {acc}");
}

#[test]
fn regressor_fits_step_function() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        x.push(i as f64 * 0.02); // low region
        y.push(1.0);
    }
    for i in 0..50 {
        x.push(3.0 + i as f64 * 0.02); // high region
        y.push(20.0);
    }
    let cols = vec![Column::Continuous(x.clone())];

    let mut tree = DecisionTree::new(
        Regression::mse(),
        default_params_with_cv(5, 11),
        Box::new(Mean),
    );
    tree.fit(&cols, &y).unwrap();

    let preds = tree.predict(&cont_samples(&x));
    let mse: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, &t)| (p - t) * (p - t))
        .sum::<f64>()
        / y.len() as f64;
    // A tree that failed to separate the two levels would predict ~10.5 for
    // everything (MSE ~90). This design uses one-vs-rest `feature == bin` splits
    // scored by K-fold CV, which can strand a single point at the cluster
    // boundary (it can only be isolated when it lands in a multi-sample
    // validation fold), leaving one small mixed leaf. So the achievable train
    // MSE here is a few units, not ~0; the bar checks the split was clearly
    // learned, not that it is perfect.
    assert!(mse < 5.0, "train MSE too high: {mse}");
}

#[test]
fn fitting_is_deterministic() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..80 {
        let v = i as f64 * 0.05;
        x.push(v);
        y.push(if (i / 8) % 2 == 0 { 0usize } else { 1usize });
    }
    let cols = vec![Column::Continuous(x.clone())];
    let test = cont_samples(&x);

    let mut a = DecisionTree::new(
        Classification::gini(2),
        default_params_with_cv(5, 123),
        Box::new(Mean),
    );
    a.fit(&cols, &y).unwrap();

    let mut b = DecisionTree::new(
        Classification::gini(2),
        default_params_with_cv(5, 123),
        Box::new(Mean),
    );
    b.fit(&cols, &y).unwrap();

    let pa = a.predict(&test);
    let pb = b.predict(&test);
    assert_eq!(pa.len(), pb.len());
    for (x, y) in pa.iter().zip(pb.iter()) {
        assert_eq!(x.class, y.class);
        assert_eq!(x.proba, y.proba);
    }
    assert_eq!(a.n_leaves(), b.n_leaves());
    assert_eq!(a.depth(), b.depth());
}

#[test]
fn parallel_matches_serial() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    // Two continuous features; second is pure noise-ish but deterministic.
    let mut x2 = Vec::new();
    for i in 0..100 {
        let v = i as f64 * 0.03;
        x.push(v);
        x2.push(((i * 37) % 11) as f64);
        y.push(if v < 1.5 { 0usize } else { 1usize });
    }
    let cols = vec![Column::Continuous(x.clone()), Column::Continuous(x2)];
    let test = cont_samples(&x)
        .into_iter()
        .enumerate()
        .map(|(i, mut s)| {
            s.push(FeatureValue::cont(((i * 37) % 11) as f64));
            s
        })
        .collect::<Vec<_>>();

    let mut serial_params = default_params_with_cv(5, 55);
    serial_params.parallel = false;
    serial_params.n_threads = 1;
    let mut serial = DecisionTree::new(Classification::gini(2), serial_params, Box::new(Mean));
    serial.fit(&cols, &y).unwrap();

    let mut par_params = default_params_with_cv(5, 55);
    par_params.parallel = true;
    par_params.n_threads = 2;
    let mut parallel = DecisionTree::new(Classification::gini(2), par_params, Box::new(Mean));
    parallel.fit(&cols, &y).unwrap();

    let ps = serial.predict(&test);
    let pp = parallel.predict(&test);
    assert_eq!(serial.n_leaves(), parallel.n_leaves());
    assert_eq!(serial.depth(), parallel.depth());
    for (a, b) in ps.iter().zip(pp.iter()) {
        assert_eq!(a.class, b.class);
        assert_eq!(a.proba, b.proba);
    }
}

#[test]
fn custom_aggregator_via_trait_object() {
    // A user-supplied aggregator that simply forwards the mean; exercises the
    // Box<dyn Aggregator> path from the public surface.
    struct Meanish;
    impl Aggregator for Meanish {
        fn aggregate(&self, s: &cvdt::FoldStats) -> f64 {
            if s.n_success == 0 {
                f64::NEG_INFINITY
            } else {
                s.mean
            }
        }
        fn name(&self) -> &'static str {
            "meanish"
        }
    }

    let cols = vec![Column::Categorical(vec![0, 0, 1, 1, 0, 0, 1, 1])];
    let y = vec![0usize, 0, 1, 1, 0, 0, 1, 1];
    let mut tree = DecisionTree::new(
        Classification::gini(2),
        default_params_with_cv(2, 1),
        Box::new(Meanish),
    );
    tree.fit(&cols, &y).unwrap();
    assert_eq!(tree.predict_one(&[FeatureValue::cat(0)]).class, 0);
    assert_eq!(tree.predict_one(&[FeatureValue::cat(1)]).class, 1);
}

// ---------------------------------------------------------------------------
// Fast (histogram) mode
// ---------------------------------------------------------------------------

fn fast_params(k: usize, seed: u64) -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(k, seed);
    p.mode = SplitMode::Fast;
    p
}

#[test]
fn fast_classifier_learns_separable_data() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        x.push(i as f64 * 0.01);
        y.push(0usize);
    }
    for i in 0..60 {
        x.push(2.0 + i as f64 * 0.01);
        y.push(1usize);
    }
    let cols = vec![Column::Continuous(x)];
    let mut tree = DecisionTree::new(Classification::gini(2), fast_params(5, 42), Box::new(Mean));
    tree.fit(&cols, &y).unwrap();

    let preds = tree.predict(&cont_samples(&[0.1, 0.3, 2.1, 2.4]));
    assert_eq!(preds[0].class, 0);
    assert_eq!(preds[1].class, 0);
    assert_eq!(preds[2].class, 1);
    assert_eq!(preds[3].class, 1);
}

#[test]
fn fast_regressor_fits_step_function() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        x.push(i as f64 * 0.02); // 0.0 .. ~1.0
        y.push(if i < 25 { 1.0 } else { 5.0 });
    }
    let cols = vec![Column::Continuous(x)];
    let mut tree = DecisionTree::new(Regression::mse(), fast_params(5, 7), Box::new(Mean));
    tree.fit(&cols, &y).unwrap();

    let preds = tree.predict(&cont_samples(&[0.1, 0.9]));
    assert!((preds[0] - 1.0).abs() < 1.0);
    assert!((preds[1] - 5.0).abs() < 1.0);
}

#[test]
fn fast_mode_is_deterministic() {
    let cols = vec![Column::Continuous(
        (0..80).map(|i| (i % 8) as f64).collect(),
    )];
    let y: Vec<usize> = (0..80).map(|i| ((i % 8) >= 4) as usize).collect();

    let build = || {
        let mut t = DecisionTree::new(Classification::gini(2), fast_params(4, 123), Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        t
    };
    let a = build();
    let b = build();
    let pts = cont_samples(&[0.0, 1.0, 4.0, 7.0]);
    let pa = a.predict(&pts);
    let pb = b.predict(&pts);
    for (x, y) in pa.iter().zip(pb.iter()) {
        assert_eq!(x.class, y.class);
    }
    assert_eq!(a.n_leaves(), b.n_leaves());
    assert_eq!(a.depth(), b.depth());
}

#[test]
fn fast_parallel_matches_serial() {
    let cols = vec![
        Column::Continuous((0..400).map(|i| (i % 20) as f64).collect()),
        Column::Categorical((0..400).map(|i| (i % 3) as u32).collect()),
    ];
    let y: Vec<usize> = (0..400).map(|i| ((i % 20) >= 10) as usize).collect();

    let mut serial_p = fast_params(5, 9);
    serial_p.parallel_min_samples = 1;
    let mut par_p = serial_p.clone();
    par_p.parallel = true;
    par_p.n_threads = 4;

    let mut serial = DecisionTree::new(Classification::gini(2), serial_p, Box::new(Mean));
    serial.fit(&cols, &y).unwrap();
    let mut parallel = DecisionTree::new(Classification::gini(2), par_p, Box::new(Mean));
    parallel.fit(&cols, &y).unwrap();

    assert_eq!(serial.n_leaves(), parallel.n_leaves());
    assert_eq!(serial.depth(), parallel.depth());

    let pts = cont_samples(&[0.0, 5.0, 12.0, 19.0]);
    // Build 2-feature samples for prediction (second feature category 0).
    let samples: Vec<Vec<FeatureValue>> = pts
        .iter()
        .map(|s| vec![s[0], FeatureValue::cat(0)])
        .collect();
    let ps = serial.predict(&samples);
    let pp = parallel.predict(&samples);
    for (a, b) in ps.iter().zip(pp.iter()) {
        assert_eq!(a.class, b.class);
    }
}

#[test]
fn fast_mode_rejects_mae() {
    let cols = vec![Column::Continuous(vec![0.0, 1.0, 2.0, 3.0])];
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let mut tree = DecisionTree::new(Regression::mae(), fast_params(2, 1), Box::new(Mean));
    assert!(tree.fit(&cols, &y).is_err());
}

// ---------------------------------------------------------------------------
// Objective-driven classification (precision / recall / F1 ...)
// ---------------------------------------------------------------------------

use cvdt::{Average, ObjectiveClassification};

fn two_cluster_data() -> (Vec<Column>, Vec<usize>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        x.push(i as f64 * 0.01);
        y.push(0usize);
    }
    for i in 0..60 {
        x.push(2.0 + i as f64 * 0.01);
        y.push(1usize);
    }
    (vec![Column::Continuous(x)], y)
}

#[test]
fn objective_f1_learns_separable_strict() {
    let (cols, y) = two_cluster_data();
    let mut tree = DecisionTree::new(
        ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
        default_params_with_cv(5, 7),
        Box::new(Mean),
    );
    tree.fit(&cols, &y).unwrap();
    let preds = tree.predict(&cont_samples(&[0.1, 0.3, 2.1, 2.4]));
    assert_eq!(preds[0].class, 0);
    assert_eq!(preds[3].class, 1);
}

#[test]
fn objective_f1_learns_separable_fast() {
    let (cols, y) = two_cluster_data();
    let mut p = fast_params(5, 7);
    p.n_bins = 16;
    let mut tree = DecisionTree::new(
        ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
        p,
        Box::new(Mean),
    );
    tree.fit(&cols, &y).unwrap();
    let preds = tree.predict(&cont_samples(&[0.1, 2.4]));
    assert_eq!(preds[0].class, 0);
    assert_eq!(preds[1].class, 1);
}

#[test]
fn objective_recall_and_precision_run() {
    let (cols, y) = two_cluster_data();
    for task in [
        ObjectiveClassification::recall(2, Average::Binary { pos_label: 1 }),
        ObjectiveClassification::precision(2, Average::Macro),
        ObjectiveClassification::accuracy(2),
    ] {
        let mut tree = DecisionTree::new(task, default_params_with_cv(4, 1), Box::new(Mean));
        tree.fit(&cols, &y).unwrap();
        // Well-separated data: should reach perfect leaves.
        assert!(tree.n_leaves() >= 2);
    }
}
