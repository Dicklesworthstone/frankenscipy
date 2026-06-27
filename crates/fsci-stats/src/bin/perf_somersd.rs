use fsci_stats::*;
use std::time::Instant;
fn brute(x: &[f64], y: &[f64]) -> (f64, f64) {
    // statistic and a_term-based z, O(n^2), all-distinct
    let n = x.len();
    // ranks
    let mut xi: Vec<usize> = (0..n).collect();
    xi.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut yi: Vec<usize> = (0..n).collect();
    yi.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
    let mut xr = vec![0usize; n];
    for (r, &i) in xi.iter().enumerate() {
        xr[i] = r;
    }
    let mut yr = vec![0usize; n];
    for (r, &i) in yi.iter().enumerate() {
        yr[i] = r;
    }
    let mut p = 0i64;
    let mut q = 0i64;
    let mut a = 0i64;
    for k in 0..n {
        let (i, j) = (xr[k] as i64, yr[k] as i64);
        // A_p: points with (xr<i,yr<j) or (xr>i,yr>j)
        let mut ap = 0i64;
        let mut dp = 0i64;
        for m in 0..n {
            if m == k {
                continue;
            }
            let (ii, jj) = (xr[m] as i64, yr[m] as i64);
            if (ii < i && jj < j) || (ii > i && jj > j) {
                ap += 1;
            }
            if (ii > i && jj < j) || (ii < i && jj > j) {
                dp += 1;
            }
        }
        p += ap;
        q += dp;
        let d = ap - dp;
        a += d * d;
    }
    let total = n as f64;
    let sri2 = n as f64;
    let stat = (p as f64 - q as f64) / (total * total - sri2);
    let s = a as f64 - (p as f64 - q as f64).powi(2) / total;
    let z = if s > 0.0 {
        (p as f64 - q as f64) / (4.0 * s).sqrt()
    } else if p > q {
        f64::INFINITY
    } else if p < q {
        f64::NEG_INFINITY
    } else {
        0.0
    };
    (stat, z)
}
fn main() {
    let mut seed = 6u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    // correctness vs brute force (small distinct)
    let mut worst = 0.0f64;
    for _ in 0..20 {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|_| r()).collect();
        let y: Vec<f64> = (0..n).map(|_| r()).collect();
        let res = somersd(SomersDInput::Rankings(&x, &y), None).unwrap();
        let (bstat, _bz) = brute(&x, &y);
        worst = worst.max((res.statistic - bstat).abs());
    }
    println!("CORRECTNESS worst |stat-brute| = {worst:.3e}");
    // perf
    for &n in &[400usize, 800] {
        let x: Vec<f64> = (0..n).map(|_| r()).collect();
        let y: Vec<f64> = (0..n).map(|_| r()).collect();
        let t = Instant::now();
        for _ in 0..5 {
            let _ = somersd(SomersDInput::Rankings(&x, &y), None).unwrap();
        }
        println!(
            "somersd n={n}: {:.2} ms",
            t.elapsed().as_secs_f64() / 5.0 * 1000.0
        );
    }
}
