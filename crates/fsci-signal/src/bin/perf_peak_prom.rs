//! Same-process A/B + isomorphism harness for peak_prominences.
//!
//! `naive` is the original O(peaks·N) outward scan; the library now switches to
//! an O(N log N + peaks) sparse-table form for large workloads. We prove
//! byte-identical output (prominences via `.to_bits()`, bases exact) across
//! randomized signals — including heavy ties, signed zeros, plateaus, and the
//! adversarial increasing-height case — then time the worst case.
//! Run: `cargo run --release -p fsci-signal --bin perf_peak_prom`.

use fsci_signal::peak_prominences;
use std::time::Instant;

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
        (self.next_u64() >> 11) as usize % m
    }
}

/// Verbatim original outward-scan prominence (the reference).
fn naive(x: &[f64], peaks: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut prom = Vec::with_capacity(peaks.len());
    let mut lb = Vec::with_capacity(peaks.len());
    let mut rb = Vec::with_capacity(peaks.len());
    for &pk in peaks {
        if pk >= x.len() {
            prom.push(0.0);
            lb.push(pk);
            rb.push(pk);
            continue;
        }
        let peak_val = x[pk];
        let mut left_min = peak_val;
        let mut left_base = pk;
        for i in (0..pk).rev() {
            if x[i] < left_min {
                left_min = x[i];
                left_base = i;
            }
            if x[i] > peak_val {
                break;
            }
        }
        let mut right_min = peak_val;
        let mut right_base = pk;
        for (i, &value) in x.iter().enumerate().skip(pk + 1) {
            if value < right_min {
                right_min = value;
                right_base = i;
            }
            if value > peak_val {
                break;
            }
        }
        prom.push(peak_val - left_min.max(right_min));
        lb.push(left_base);
        rb.push(right_base);
    }
    (prom, lb, rb)
}

/// All indices i (1<=i<=n-2) that are strict local maxima of `x`.
fn local_maxima(x: &[f64]) -> Vec<usize> {
    (1..x.len().saturating_sub(1))
        .filter(|&i| x[i] > x[i - 1] && x[i] > x[i + 1])
        .collect()
}

fn main() {
    let mut r = Lcg(0xfeed_face_dead_beef);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..400 {
        let n = 256 + r.below(900);
        // Mix value alphabets to stress ties / plateaus / signed zeros.
        let alphabet = trial % 4;
        let x: Vec<f64> = (0..n)
            .map(|_| match alphabet {
                0 => r.below(7) as f64,                      // heavy ties / plateaus
                1 => (r.below(2001) as f64 - 1000.0) / 13.0, // fine grained
                2 => {
                    let v = r.below(5) as f64 - 2.0;
                    if v == 0.0 && r.below(2) == 0 { -0.0 } else { v } // signed zeros
                }
                _ => r.below(40) as f64,
            })
            .collect();
        // Adversarial: every ~8th trial, make a monotonically increasing
        // sawtooth so every peak scans to the boundary.
        let x = if trial % 8 == 0 {
            (0..n)
                .map(|i| if i % 2 == 1 { (i / 2 + 1) as f64 } else { 0.0 })
                .collect()
        } else {
            x
        };

        let mut peaks = local_maxima(&x);
        // Also throw in some arbitrary (non-maxima) peak indices + an OOB one.
        peaks.push(0);
        peaks.push(n - 1);
        peaks.push(n + 3);
        if peaks.len() < 32 {
            continue; // below the fast-path gate; skip (naive == naive)
        }

        let (pa, la, ra) = naive(&x, &peaks);
        let (pb, lb, rb) = peak_prominences(&x, &peaks);
        let mut bad = 0usize;
        for i in 0..peaks.len() {
            total += 1;
            if pa[i].to_bits() != pb[i].to_bits() || la[i] != lb[i] || ra[i] != rb[i] {
                mismatches += 1;
                bad += 1;
            }
        }
        if bad > 0 && payload.len() < 4000 {
            payload.push_str(&format!("MISMATCH trial={trial} n={n} bad={bad}\n"));
        }
        let digest: u64 =
            pb.iter()
                .zip(&lb)
                .zip(&rb)
                .fold(1469598103934665603u64, |h, ((p, &l), &rr)| {
                    ((h ^ p.to_bits()).wrapping_mul(1099511628211) ^ l as u64)
                        .wrapping_mul(1099511628211)
                        ^ rr as u64
                });
        payload.push_str(&format!(
            "trial={trial} npeaks={} digest={digest:016x}\n",
            peaks.len()
        ));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} peak-results (0 == byte-identical)");

    // ---- Timing: adversarial increasing-height sawtooth, all peaks scan far ----
    for &n in &[4096usize, 16384, 65536] {
        let x: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 1 { (i / 2 + 1) as f64 } else { 0.0 })
            .collect();
        let peaks = local_maxima(&x);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += naive(&x, &peaks).0.iter().sum::<f64>();
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += peak_prominences(&x, &peaks).0.iter().sum::<f64>();
        }
        let fast_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / fast_t.as_secs_f64();
        println!(
            "n={n:>6} peaks={:>6}  naive={:>10.3?}  rmq={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.0})",
            peaks.len(),
            naive_t / 3,
            fast_t / 3
        );
    }
}
