//! Same-process A/B + isomorphism harness for `theilslopes`.
//!
//! `old_theilslopes` is a verbatim copy of the original full-sort version. The
//! library now partitions the O(n^2) slopes to only the two CI order statistics
//! via select_nth (the median already uses a partial select), which is
//! bit-identical: select_nth(k) yields sorted[k]. We assert 0 mismatches on all
//! four result fields across sizes / tie densities / alphas, and time the win.
//! Run: `cargo run --release -p fsci-stats --bin perf_theilslopes`.

use fsci_stats::{Normal, TheilslopesResult, find_repeats, median, theil_sen, theilslopes};
use std::time::Instant;

/// Verbatim copy of the original full-sort theilslopes.
fn old_theilslopes(x: &[f64], y: &[f64], alpha: f64) -> TheilslopesResult {
    let n = x.len();
    if n < 2 || n != y.len() {
        return TheilslopesResult {
            slope: f64::NAN,
            intercept: f64::NAN,
            low_slope: f64::NAN,
            high_slope: f64::NAN,
        };
    }
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            if dx.abs() > 1e-15 {
                slopes.push((y[i] - y[j]) / dx);
            }
        }
    }
    if slopes.is_empty() {
        return TheilslopesResult {
            slope: 0.0,
            intercept: median(y),
            low_slope: f64::NAN,
            high_slope: f64::NAN,
        };
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let medslope = median(&slopes);
    let medinter = median(y) - medslope * median(x);
    let alpha_adj = if alpha > 0.5 { 1.0 - alpha } else { alpha };
    let z = Normal::new(0.0, 1.0).ppf(alpha_adj / 2.0);
    let x_reps = find_repeats(x);
    let y_reps = find_repeats(y);
    let nt = slopes.len() as f64;
    let ny = n as f64;
    let mut sigsq = ny * (ny - 1.0) * (2.0 * ny + 5.0) / 18.0;
    for &k in &x_reps.counts {
        let kf = k as f64;
        sigsq -= kf * (kf - 1.0) * (2.0 * kf + 5.0) / 18.0;
    }
    for &k in &y_reps.counts {
        let kf = k as f64;
        sigsq -= kf * (kf - 1.0) * (2.0 * kf + 5.0) / 18.0;
    }
    let sigma = sigsq.sqrt();
    let ru = ((nt - z * sigma) / 2.0).round() as usize;
    let rl = ((nt + z * sigma) / 2.0).round() as usize;
    let low_slope = if rl > 0 && rl <= slopes.len() {
        slopes[rl - 1]
    } else {
        f64::NAN
    };
    let high_slope = if ru < slopes.len() {
        slopes[ru]
    } else {
        f64::NAN
    };
    TheilslopesResult {
        slope: medslope,
        intercept: medinter,
        low_slope,
        high_slope,
    }
}

fn old_theil_sen(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 || n != y.len() {
        return (f64::NAN, f64::NAN);
    }
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            let dx = x[j] - x[i];
            if dx.abs() > 1e-15 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }
    if slopes.is_empty() {
        return (0.0, median(y));
    }
    let slope = median(&slopes);
    let intercepts: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| yi - slope * xi)
        .collect();
    let intercept = median(&intercepts);
    (slope, intercept)
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn make_xy(n: usize, grid: u64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xv = if grid == 0 {
            i as f64 + rng.next_f64()
        } else {
            (rng.next_f64() * grid as f64).floor()
        };
        x.push(xv);
        y.push(if grid == 0 {
            2.0 * xv + rng.next_f64() * 5.0
        } else {
            (rng.next_f64() * grid as f64).floor()
        });
    }
    (x, y)
}

/// Numerically identical (treats +0.0 == -0.0 and NaN == NaN).
fn eq_num(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Bit-identical EXCEPT a benign +0.0/-0.0 sign on a zero value.
fn bit_diff_is_only_zero_sign(a: f64, b: f64) -> bool {
    a.to_bits() != b.to_bits() && a == 0.0 && b == 0.0
}

fn main() {
    let mut num_mismatches = 0usize;
    let mut zero_sign_diffs = 0usize;
    let mut total = 0usize;
    let mut payload = String::new();
    for &n in &[2usize, 3, 8, 33, 120] {
        for &grid in &[0u64, 4, 12] {
            for &alpha in &[0.95f64, 0.90, 0.99] {
                for seed in 0..4u64 {
                    let (x, y) = make_xy(n, grid, seed * 2657 + 1);
                    let got = theilslopes(&x, &y, alpha);
                    let want = old_theilslopes(&x, &y, alpha);
                    let got_sen = theil_sen(&x, &y);
                    let want_sen = old_theil_sen(&x, &y);
                    total += 1;
                    let fields = [
                        (got.slope, want.slope),
                        (got.intercept, want.intercept),
                        (got.low_slope, want.low_slope),
                        (got.high_slope, want.high_slope),
                        (got_sen.0, want_sen.0),
                        (got_sen.1, want_sen.1),
                    ];
                    for &(g, w) in &fields {
                        if !eq_num(g, w) {
                            num_mismatches += 1;
                            if payload.len() < 1500 {
                                payload.push_str(&format!(
                                    "NUM-MISMATCH n={n} grid={grid} a={alpha} seed={seed} got={g} want={w}\n"
                                ));
                            }
                        } else if bit_diff_is_only_zero_sign(g, w) {
                            zero_sign_diffs += 1;
                        }
                    }
                    if payload.len() < 1500 {
                        payload.push_str(&format!(
                            "n={n} grid={grid} a={alpha} seed={seed} s={:016x} lo={:016x} hi={:016x} sen_s={:016x} sen_i={:016x}\n",
                            got.slope.to_bits(),
                            got.low_slope.to_bits(),
                            got.high_slope.to_bits(),
                            got_sen.0.to_bits(),
                            got_sen.1.to_bits()
                        ));
                    }
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "parity: {num_mismatches} numeric mismatches / {total} cases ({} fields); {zero_sign_diffs} benign +0.0/-0.0-sign-only diffs (numerically equal)",
        total * 6
    );

    // ---- Timing: large n (O(n^2) slopes; old sorts them, new partitions) ----
    for &n in &[1000usize, 2000, 3000] {
        let (x, y) = make_xy(n, 0, 7);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_theilslopes(&x, &y, 0.95).low_slope;
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += theilslopes(&x, &y, 0.95).low_slope;
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>5}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>6.1}x  (acc={acc:.4})",
            old_t / 3,
            new_t / 3
        );
    }
}
