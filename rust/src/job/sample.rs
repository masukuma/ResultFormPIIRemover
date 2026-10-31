//! How many pages are looked at, and which (HARNESS.md §4.3, §4.5, §8).
//!
//! The leak-rate bound: pages of a group below the risk threshold are
//! sampled at random until the leaks that could remain are bounded below
//! 0.5% at 95% confidence. No leak in 600 pages bounds the rate at
//! 3.00/600; one leak needs 950 pages (4.74/950), two 1,260 (6.30/1,260);
//! three are a pattern. A group of 600 pages or fewer is reviewed whole.
//! Samples are spread over a group's folders in proportion to their size,
//! and drawn with Python's `random.sample` from a recorded seed, so anyone
//! can draw the same ones.

use std::collections::BTreeMap;

use crate::pyrand::PyRandom;

/// The leak rate a group's sample must bound.
pub const LEAK_RATE: f64 = 0.005;

/// The upper 95% bound on the expected number of leaks, after finding 0,
/// 1 or 2 (Poisson).
pub const UPPER_95: [f64; 3] = [2.9957, 4.7439, 6.2958];

/// Pages reviewed at random after finding 0, 1 or 2 leaks.
pub const SAMPLE: [usize; 3] = [600, 950, 1260];

/// Confirmed leaks of one kind in a group that make a pattern.
pub const PATTERN: usize = 3;

/// Pages of a group of `n` to review at random, after `leaks` leaks found
/// in its sample: all of them if the group has 600 or fewer, else 600, 950
/// or 1,260. None once the leaks are a pattern.
pub fn sample_size(n: usize, leaks: usize) -> Option<usize> {
    let k = *SAMPLE.get(leaks)?;
    Some(if n <= SAMPLE[0] { n } else { k.min(n) })
}

/// The bound on the leak rate that `reviewed` random pages with `leaks`
/// leaks give; None for a pattern.
pub fn bound(reviewed: usize, leaks: usize) -> Option<f64> {
    Some(UPPER_95.get(leaks)? / reviewed.max(1) as f64)
}

/// Whether reviewing `reviewed` of `n` pages at random, `leaks` of them
/// leaking (each fixed), bounds the group's leak rate: every page was
/// looked at, or the bound is below 0.5%.
pub fn bounded(n: usize, reviewed: usize, leaks: usize) -> bool {
    leaks < PATTERN && (reviewed >= n || bound(reviewed, leaks).is_some_and(|b| b < LEAK_RATE))
}

/// Pilot pages per group of the sizes given: 2% of each, at least 20 (or
/// all of a smaller group), at most 500 in all (shared in proportion).
pub fn pilot_sizes(groups: &[usize]) -> Vec<usize> {
    let want: Vec<usize> = groups.iter().map(|&n| (n as f64 * 0.02).ceil().max(20.0).min(n as f64) as usize).collect();
    let total: usize = want.iter().sum();
    if total <= 500 {
        return want;
    }
    want.iter().map(|&w| (w * 500 / total).max(1)).collect()
}

/// Pages of a group of `n` re-run from their manifests: 1%, at least 5 (or
/// all of a smaller group).
pub fn rerun_size(n: usize) -> usize {
    ((n as f64 * 0.01).ceil() as usize).max(5).min(n)
}

/// A seed for one draw: the job's, mixed with what is drawn and from where.
pub fn seed(job: u64, what: &str) -> u64 {
    let h = crate::proof::sha256(format!("{}|{}", job, what).as_bytes());
    u64::from_str_radix(&h[..15], 16).unwrap_or(0)
}

/// `k` of `items` (each with its folder) at random, spread over the folders
/// in proportion to their size (largest remainders); in the order given.
pub fn spread<T: Clone>(items: &[(String, T)], k: usize, seed: u64) -> Vec<T> {
    let n = items.len();
    if k >= n {
        return items.iter().map(|(_, t)| t.clone()).collect();
    }
    let mut folders: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, (f, _)) in items.iter().enumerate() {
        folders.entry(f.as_str()).or_default().push(i);
    }
    let mut quota: Vec<(&str, usize, f64)> = folders.iter().map(|(f, v)| {
        let exact = k as f64 * v.len() as f64 / n as f64;
        (*f, exact.floor() as usize, exact - exact.floor())
    }).collect();
    let short = k - quota.iter().map(|q| q.1).sum::<usize>();
    let mut order: Vec<usize> = (0..quota.len()).collect();
    order.sort_by(|&a, &b| quota[b].2.partial_cmp(&quota[a].2).unwrap().then(a.cmp(&b)));
    for &i in order.iter().take(short) {
        quota[i].1 += 1;
    }
    let mut picked: Vec<usize> = Vec::new();
    for (f, q, _) in quota {
        let idx = &folders[f];
        let mut rng = PyRandom::new(self::seed(seed, f));
        picked.extend(rng.sample_range(idx.len(), q.min(idx.len())).into_iter().map(|j| idx[j]));
    }
    picked.sort_unstable();
    picked.into_iter().map(|i| items[i].1.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sample_sizes_bound_the_leak_rate_below_half_a_percent() {
        // the rule of three, and the Poisson bounds for one and two leaks
        for (leaks, &n) in SAMPLE.iter().enumerate() {
            assert!(bound(n, leaks).unwrap() < LEAK_RATE, "{} leaks in {}", leaks, n);
            assert_eq!(sample_size(50_000, leaks), Some(n));
        }
        assert!(bound(599, 0).unwrap() > LEAK_RATE && bound(1259, 2).unwrap() > LEAK_RATE);
        assert!((bound(600, 0).unwrap() - 3.0 / 600.0).abs() < 1e-5);
        // three leaks are a pattern: no sample size bounds that
        assert_eq!(sample_size(50_000, 3), None);
        assert!(!bounded(50_000, 50_000, 3));
        // a group of 600 pages or fewer is reviewed whole
        assert_eq!(sample_size(600, 0), Some(600));
        assert_eq!(sample_size(450, 2), Some(450));
        assert!(bounded(450, 450, 2) && !bounded(450, 449, 0));
        assert_eq!(sample_size(700, 2), Some(700));
        assert!(bounded(6608, 600, 0) && !bounded(6608, 600, 1) && bounded(6608, 950, 1));
    }

    #[test]
    fn pilots_and_reruns_have_floors_and_a_ceiling() {
        assert_eq!(pilot_sizes(&[6984]), vec![140]);
        assert_eq!(pilot_sizes(&[10, 100]), vec![10, 20]);
        let big = pilot_sizes(&[50_000, 20_000]);
        assert!(big.iter().sum::<usize>() <= 500 && big[0] > big[1]);
        assert_eq!((rerun_size(3), rerun_size(100), rerun_size(6608)), (3, 5, 67));
    }

    #[test]
    fn samples_are_spread_over_folders_and_drawn_again_alike() {
        let items: Vec<(String, usize)> = (0..100).map(|i| (if i < 75 { "A" } else { "B" }.to_string(), i)).collect();
        let s = spread(&items, 20, 7);
        assert_eq!(s.len(), 20);
        assert_eq!((s.iter().filter(|&&i| i < 75).count(), s.iter().filter(|&&i| i >= 75).count()), (15, 5));
        assert_eq!(s, spread(&items, 20, 7));
        assert_ne!(s, spread(&items, 20, 8));
        assert_eq!(spread(&items, 200, 7).len(), 100);
    }
}
