//! Core data representation.
//!
//! Data is stored **column-major**. Each feature is either a dense vector of
//! `f64` (continuous) or a dense vector of integer category ids (categorical,
//! conceptually one-hot but without the memory blow-up). Working column-major
//! lets the encoder and cross-validation engine touch one feature at a time,
//! which is both cache friendly and the natural unit of parallelism.

/// Integer id used to represent a categorical level.
pub type CatId = u32;

/// Compact sample index used by the histogram ("fast") build path.
///
/// Trees shuffle and partition index buffers constantly; using `u32` instead of
/// `usize` halves the memory traffic of those buffers on 64-bit targets.
pub type SampleId = u32;

/// Compact per-sample bin code used by the histogram ("fast") build path.
///
/// Continuous features are globally quantile-binned and categorical features
/// densely re-indexed, so a `u16` (up to 65 535 slots per feature) is ample and
/// four times smaller than an `f64` value — the compact representation is one of
/// the larger real-world speedups for tree building.
pub type Bin = u16;

/// Reserved bin code for a missing / unknown value on the fast path.
pub const MISSING_BIN_CODE: Bin = u16::MAX;

/// Sentinel for a categorical value that was never observed while fitting.
pub const UNKNOWN_CAT: CatId = u32::MAX;

/// A single feature column.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// Continuous values (may contain non-finite entries, treated as missing).
    Continuous(Vec<f64>),
    /// Pre-encoded categorical values as integer ids.
    Categorical(Vec<CatId>),
}

impl Column {
    /// Number of samples in the column.
    pub fn len(&self) -> usize {
        match self {
            Column::Continuous(v) => v.len(),
            Column::Categorical(v) => v.len(),
        }
    }

    /// Whether the column has zero samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this is a continuous column.
    pub fn is_continuous(&self) -> bool {
        matches!(self, Column::Continuous(_))
    }

    /// Whether this is a categorical column.
    pub fn is_categorical(&self) -> bool {
        matches!(self, Column::Categorical(_))
    }
}

/// A validated collection of equal-length feature columns.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Feature columns.
    pub columns: Vec<Column>,
    /// Number of samples (rows).
    pub n_samples: usize,
}

impl Dataset {
    /// Build a dataset, validating that all columns share the same length.
    pub fn new(columns: Vec<Column>) -> Result<Self, String> {
        if columns.is_empty() {
            return Err("dataset needs at least one column".to_string());
        }
        let n = columns[0].len();
        for (i, c) in columns.iter().enumerate() {
            if c.len() != n {
                return Err(format!(
                    "column {i} has length {} but expected {n}",
                    c.len()
                ));
            }
        }
        Ok(Dataset {
            columns,
            n_samples: n,
        })
    }

    /// Number of feature columns.
    pub fn n_features(&self) -> usize {
        self.columns.len()
    }
}

/// A single feature value used when predicting on unseen samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureValue {
    /// Continuous value.
    Continuous(f64),
    /// Categorical id.
    Categorical(CatId),
}

impl FeatureValue {
    /// Convenience constructor for a continuous value.
    pub fn cont(x: f64) -> Self {
        FeatureValue::Continuous(x)
    }

    /// Convenience constructor for a categorical value.
    pub fn cat(c: CatId) -> Self {
        FeatureValue::Categorical(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_rejects_ragged_columns() {
        let cols = vec![
            Column::Continuous(vec![1.0, 2.0, 3.0]),
            Column::Categorical(vec![0, 1]),
        ];
        assert!(Dataset::new(cols).is_err());
    }

    #[test]
    fn dataset_accepts_aligned_columns() {
        let cols = vec![
            Column::Continuous(vec![1.0, 2.0, 3.0]),
            Column::Categorical(vec![0, 1, 0]),
        ];
        let ds = Dataset::new(cols).unwrap();
        assert_eq!(ds.n_samples, 3);
        assert_eq!(ds.n_features(), 2);
    }
}
