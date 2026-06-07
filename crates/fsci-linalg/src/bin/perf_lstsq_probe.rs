use fsci_linalg::{LstsqOptions, MatrixAssumption, PinvOptions, SolveOptions, lstsq, pinv, solve};
use std::hint::black_box;
use std::time::Instant;
fn mk(m: usize, n: usize, s: f64) -> Vec<Vec<f64>> {
    (0..m)
        .map(|i| {
            (0..n)
                .map(|j| ((i as f64 * 0.013 + j as f64 * 0.007 + s).sin()) + 0.5)
                .collect()
        })
        .collect()
}
fn spd(n: usize, s: f64) -> Vec<Vec<f64>> {
    // A = M^T M + n I  (symmetric positive definite)
    let mm = mk(n, n, s);
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut t = 0.0;
            for row in mm.iter().take(n) {
                t += row[i] * row[j];
            }
            a[i][j] = t + if i == j { n as f64 } else { 0.0 };
        }
    }
    a
}
fn t(label: &str, mut f: impl FnMut()) {
    f();
    let st = Instant::now();
    f();
    println!("{label}: {:.1} ms", st.elapsed().as_secs_f64() * 1e3);
}
fn main() {
    for n in [1000usize, 1500] {
        let m = 2 * n;
        let a = mk(m, n, 0.3);
        let b: Vec<f64> = (0..m).map(|i| (i as f64 * 0.01).cos()).collect();
        t(&format!("lstsq m={m} n={n}"), || {
            black_box(lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
        t(&format!("pinv  m={m} n={n}"), || {
            black_box(pinv(&a, PinvOptions::default()).unwrap());
        });
    }
    for n in [1024usize, 2048] {
        let a = spd(n, 0.7);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
        let o = SolveOptions {
            assume_a: Some(MatrixAssumption::PositiveDefinite),
            ..SolveOptions::default()
        };
        t(&format!("spd-solve n={n}"), || {
            black_box(solve(&a, &b, o).unwrap());
        });
    }
}
