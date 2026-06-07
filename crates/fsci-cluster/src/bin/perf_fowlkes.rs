//! Same-process A/B bench + isomorphism harness for fowlkes_mallows_score.
//!
//! `old_fm` is the original O(n^2) pair-counting loop; `new_fm` is the
//! O(n + k^2) contingency-table form. We prove byte-identical f64 output across
//! a randomized suite and time both. Run: `cargo run --release -p fsci-cluster --bin perf_fowlkes`.

use std::time::Instant;

/// Deterministic LCG so the suite is reproducible without an RNG dependency.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, m: usize) -> usize {
        (self.next_u64() % m as u64) as usize
    }
}

fn gen_labels(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut r = Lcg(seed);
    (0..n).map(|_| r.below(k)).collect()
}

/// Original O(n^2) implementation (verbatim from the pre-optimization source).
fn old_fm(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n == 1 {
        return 1.0;
    }
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    for i in 0..n {
        for j in i + 1..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labels_pred[i] == labels_pred[j];
            match (same_true, same_pred) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                _ => {}
            }
        }
    }
    if tp == 0 {
        return 0.0;
    }
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    (precision * recall).sqrt()
}

fn comb2(x: usize) -> u64 {
    let x = x as u64;
    x * (x - 1) / 2
}

/// New O(n + k^2) contingency-table implementation (mirrors the library change).
fn new_fm(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    let n = labels_true.len();
    if n == 1 {
        return 1.0;
    }
    let kt = labels_true.iter().copied().max().map_or(0, |m| m + 1);
    let kp = labels_pred.iter().copied().max().map_or(0, |m| m + 1);
    let mut table = vec![0usize; kt * kp];
    let mut row = vec![0usize; kt];
    let mut col = vec![0usize; kp];
    for i in 0..n {
        let t = labels_true[i];
        let p = labels_pred[i];
        table[t * kp + p] += 1;
        row[t] += 1;
        col[p] += 1;
    }
    let tp: u64 = table.iter().map(|&m| comb2(m)).sum();
    if tp == 0 {
        return 0.0;
    }
    let tp_fp: u64 = col.iter().map(|&m| comb2(m)).sum(); // pairs sharing predicted cluster
    let tp_fn: u64 = row.iter().map(|&m| comb2(m)).sum(); // pairs sharing true cluster
    let precision = if tp_fp > 0 {
        tp as f64 / tp_fp as f64
    } else {
        0.0
    };
    let recall = if tp_fn > 0 {
        tp as f64 / tp_fn as f64
    } else {
        0.0
    };
    (precision * recall).sqrt()
}

fn main() {
    // ---- Isomorphism: byte-identical f64 across a randomized suite ----
    let mut mismatches = 0usize;
    let mut payload = String::new();
    let mut seed = 0x9E3779B97F4A7C15u64;
    for &n in &[1usize, 2, 7, 31, 64, 257, 1000, 4096] {
        for &(kt, kp) in &[(2usize, 2usize), (3, 5), (7, 11), (1, 4), (50, 50)] {
            for rep in 0..6 {
                seed = seed
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493);
                let lt = gen_labels(n, kt, seed ^ 0xABCD);
                let lp = gen_labels(n, kp, seed ^ 0x1234);
                let a = old_fm(&lt, &lp);
                let b = new_fm(&lt, &lp);
                // Also exercise the real library function (must equal old_fm bit-for-bit).
                let lib = fsci_cluster::fowlkes_mallows_score(&lt, &lp).unwrap();
                if a.to_bits() != b.to_bits() || a.to_bits() != lib.to_bits() {
                    mismatches += 1;
                }
                payload.push_str(&format!(
                    "n={n} kt={kt} kp={kp} rep={rep} old_bits={:016x} new_bits={:016x}\n",
                    a.to_bits(),
                    b.to_bits()
                ));
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches (0 == byte-identical)");

    // ---- Timing: large n where O(n^2) dominates ----
    for &n in &[2000usize, 8000, 20000] {
        let lt = gen_labels(n, 20, 0x111 + n as u64);
        let lp = gen_labels(n, 25, 0x222 + n as u64);

        let mut acc = 0.0f64;
        let t0 = Instant::now();
        for _ in 0..5 {
            acc += old_fm(&lt, &lp);
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..5 {
            acc += new_fm(&lt, &lp);
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.3})",
            old_t / 5,
            new_t / 5
        );
    }
}
