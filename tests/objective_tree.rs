//! End-to-end tests for objective-driven classification trees.

use cvdt::{
    Average, Column, DecisionTree, FeatureValue, KFold, Mean, ObjectiveClassification, SplitMode,
    TreeParams,
};

fn params(k: usize, seed: u64) -> TreeParams {
    let mut p = TreeParams::default();
    p.cv = KFold::new(k, seed);
    p
}

fn separable() -> (Vec<Column>, Vec<usize>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        x.push(i as f64 * 0.01);
        y.push(0usize);
    }
    for i in 0..60 {
        x.push(3.0 + i as f64 * 0.01);
        y.push(1usize);
    }
    (vec![Column::Continuous(x)], y)
}

fn class_at(tree: &DecisionTree<ObjectiveClassification>, x: f64) -> usize {
    tree.predict(&[vec![FeatureValue::cont(x)]])[0].class
}

#[test]
fn each_objective_learns_separable_strict() {
    let (cols, y) = separable();
    let tasks = [
        ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
        ObjectiveClassification::precision(2, Average::Macro),
        ObjectiveClassification::recall(2, Average::Binary { pos_label: 1 }),
        ObjectiveClassification::accuracy(2),
        ObjectiveClassification::fbeta(2, 2.0, Average::Binary { pos_label: 1 }),
    ];
    for task in tasks {
        let mut t = DecisionTree::new(task, params(5, 7), Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        assert!(t.n_leaves() >= 2, "objective tree should split separable data");
        assert_eq!(class_at(&t, 0.2), 0);
        assert_eq!(class_at(&t, 3.3), 1);
    }
}

#[test]
fn objective_learns_separable_fast() {
    let (cols, y) = separable();
    let mut p = params(5, 7);
    p.mode = SplitMode::Fast;
    p.n_bins = 16;
    let mut t = DecisionTree::new(
        ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
        p,
        Box::new(Mean),
    );
    t.fit(&cols, &y).unwrap();
    assert_eq!(class_at(&t, 0.2), 0);
    assert_eq!(class_at(&t, 3.3), 1);
}

#[test]
fn objective_multiclass_macro_f1() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (cluster, base) in [0.0, 3.0, 6.0].iter().enumerate() {
        for i in 0..40 {
            x.push(base + i as f64 * 0.01);
            y.push(cluster);
        }
    }
    let cols = vec![Column::Continuous(x)];
    let mut t = DecisionTree::new(
        ObjectiveClassification::f1(3, Average::Macro),
        params(5, 1),
        Box::new(Mean),
    );
    t.fit(&cols, &y).unwrap();
    assert_eq!(class_at(&t, 0.2), 0);
    assert_eq!(class_at(&t, 6.2), 2);
}

#[test]
fn objective_single_class_is_one_leaf() {
    let cols = vec![Column::Continuous(vec![0.0, 1.0, 2.0, 3.0, 4.0])];
    let y = vec![0usize; 5];
    let mut t = DecisionTree::new(
        ObjectiveClassification::f1(2, Average::Binary { pos_label: 1 }),
        params(3, 1),
        Box::new(Mean),
    );
    t.fit(&cols, &y).unwrap();
    assert_eq!(t.n_leaves(), 1);
}

#[test]
fn negative_min_impurity_allows_more_growth() {
    // Objective mode is self-stopping; a negative threshold permits
    // non-improving splits, so the tree can be at least as large.
    let (cols, y) = separable();
    let base = {
        let mut t = DecisionTree::new(
            ObjectiveClassification::accuracy(2),
            params(5, 7),
            Box::new(Mean),
        );
        t.fit(&cols, &y).unwrap();
        t.n_leaves()
    };
    let grown = {
        let mut p = params(5, 7);
        p.min_impurity_decrease = -1.0;
        let mut t = DecisionTree::new(ObjectiveClassification::accuracy(2), p, Box::new(Mean));
        t.fit(&cols, &y).unwrap();
        t.n_leaves()
    };
    assert!(grown >= base);
}
