//! Minimal, dependency-free parallelism.
//!
//! The design doc calls for parallel feature/fold evaluation, but the crate is
//! intentionally zero-dependency, so instead of pulling in a work-stealing pool
//! we provide a small order-preserving parallel map built on
//! [`std::thread::scope`]. Work is split into contiguous chunks, one per
//! worker thread; each worker writes into its own `Vec`, and the results are
//! concatenated back in the original order.
//!
//! Because the mapped function is applied to independent items and the results
//! are reassembled deterministically, `par_map` returns exactly what a serial
//! `items.iter().map(f).collect()` would — this is what guarantees the
//! "parallel == serial" property the tree relies on. When `n_threads <= 1` (or
//! there is at most one item) it falls back to a plain serial map.

use std::thread;

/// Apply `f` to every item, potentially across several threads, preserving the
/// input order in the output.
///
/// `f` must be `Sync` (it is shared by reference across threads) and its output
/// `Send` (results move back to the caller). The items are borrowed for the
/// duration, so nothing needs to be cloned.
pub fn par_map<T, R, F>(items: &[T], n_threads: usize, f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let n = items.len();
    if n_threads <= 1 || n <= 1 {
        return items.iter().map(|it| f(it)).collect();
    }

    let workers = n_threads.min(n);
    // Ceil-divide so every item is covered by exactly one contiguous chunk.
    let chunk = (n + workers - 1) / workers;
    let f_ref = &f;

    // Each thread returns (chunk_start, results_for_that_chunk); we then stitch
    // the chunks back together in ascending start order.
    let mut pieces: Vec<(usize, Vec<R>)> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut start = 0;
        while start < n {
            let end = (start + chunk).min(n);
            let slice = &items[start..end];
            let handle = scope.spawn(move || {
                let local: Vec<R> = slice.iter().map(|it| f_ref(it)).collect();
                (start, local)
            });
            handles.push(handle);
            start = end;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });

    pieces.sort_by_key(|(start, _)| *start);
    let mut out = Vec::with_capacity(n);
    for (_, mut part) in pieces {
        out.append(&mut part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_path_matches_map() {
        let items: Vec<usize> = (0..10).collect();
        let got = par_map(&items, 1, |&x| x * x);
        let want: Vec<usize> = items.iter().map(|&x| x * x).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn parallel_preserves_order() {
        let items: Vec<usize> = (0..1000).collect();
        let got = par_map(&items, 4, |&x| x * 2);
        let want: Vec<usize> = items.iter().map(|&x| x * 2).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn parallel_equals_serial() {
        let items: Vec<i64> = (-50..50).collect();
        let serial = par_map(&items, 1, |&x| x.pow(2) - x);
        let parallel = par_map(&items, 8, |&x| x.pow(2) - x);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn more_threads_than_items_is_fine() {
        let items = vec![1, 2, 3];
        let got = par_map(&items, 16, |&x| x + 1);
        assert_eq!(got, vec![2, 3, 4]);
    }

    #[test]
    fn empty_input() {
        let items: Vec<usize> = Vec::new();
        let got = par_map(&items, 4, |&x| x);
        assert!(got.is_empty());
    }
}
