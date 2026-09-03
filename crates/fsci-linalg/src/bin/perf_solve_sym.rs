use fsci_linalg::{MatrixAssumption, SolveOptions, solve};
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let mut seed = 7u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[256usize, 384, 512] {
        let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        // PD symmetric: A = (M+M^T)/2 + n*I (diagonally dominant => PD => Cholesky path)
        let mut pd = vec![vec![0.0; n]; n];
        // indefinite symmetric: (M+M^T)/2 + small shift (invertible but not PD => LU fallback)
        let mut indef = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let s = (m[i][j] + m[j][i]) * 0.5;
                pd[i][j] = s + if i == j { n as f64 } else { 0.0 };
                indef[i][j] = s + if i == j { 0.3 } else { 0.0 };
            }
        }
        let b: Vec<f64> = (0..n).map(|_| r()).collect();
        let sym = SolveOptions {
            assume_a: Some(MatrixAssumption::Symmetric),
            ..Default::default()
        };

        for (name, a) in [("PD", &pd), ("INDEF", &indef)] {
            let xg = solve(a, &b, SolveOptions::default()).unwrap().x;
            let xs = solve(a, &b, sym).unwrap().x;
            let maxdiff = xg
                .iter()
                .zip(&xs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let bench = |o: SolveOptions| {
                let _ = black_box(solve(a, &b, o).unwrap());
                let reps = 6;
                let t = Instant::now();
                for _ in 0..reps {
                    let _ = black_box(solve(black_box(a), black_box(&b), o).unwrap());
                }
                t.elapsed().as_secs_f64() / reps as f64 * 1000.0
            };
            let (mut g, mut s) = (f64::MAX, f64::MAX);
            for _ in 0..4 {
                g = g.min(bench(SolveOptions::default()));
                s = s.min(bench(sym));
            }
            println!(
                "n={n} {name}: General {g:.2}ms  Symmetric {s:.2}ms  ratio(G/S)={:.2}x  maxdiff={maxdiff:.1e}",
                g / s
            );
        }
    }
}
