// Verify the SPD-solve fast-path gate change (line 783, solve_flat_min) shipped
// in 31a206e9: is cholesky_solve_blocked at n in [128,1000) a win or regression
// vs the portfolio path? assume_a = PositiveDefinite.
use fsci_linalg::{MatrixAssumption, SOLVE_FLAT_MIN_OVERRIDE, SolveOptions, solve};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut seed = 11u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[128usize, 256, 384, 512, 768] {
        // SPD matrix A = M Mᵀ/n + n I
        let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += m[i][k] * m[j][k];
                }
                a[i][j] = s / n as f64 + if i == j { n as f64 } else { 0.0 };
            }
        }
        let b: Vec<f64> = (0..n).map(|_| r()).collect();
        let opts = || SolveOptions {
            assume_a: Some(MatrixAssumption::PositiveDefinite),
            ..Default::default()
        };
        SOLVE_FLAT_MIN_OVERRIDE.store(1000, Ordering::Relaxed);
        let xp = solve(&a, &b, opts()).unwrap().x;
        SOLVE_FLAT_MIN_OVERRIDE.store(1, Ordering::Relaxed);
        let xf = solve(&a, &b, opts()).unwrap().x;
        let maxdiff = xp
            .iter()
            .zip(&xf)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let reps = 6;
        let bench = |ov: usize| {
            SOLVE_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            let _ = black_box(solve(&a, &b, opts()).unwrap());
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(solve(black_box(&a), black_box(&b), opts()).unwrap());
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let portfolio = bench(1000).min(bench(1000));
        let fast = bench(1).min(bench(1));
        println!(
            "spd_solve n={n:4}: portfolio {portfolio:8.2}ms -> fast {fast:8.2}ms = {:.2}x  maxdiff={maxdiff:.1e}",
            portfolio / fast
        );
    }
    SOLVE_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
}
