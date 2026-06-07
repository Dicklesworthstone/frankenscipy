use fsci_linalg::{InvOptions, SolveOptions, inv, matmul, solve};
use std::hint::black_box;
use std::time::Instant;
fn mk(n: usize, s: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let v = (i as f64 * 0.013 + j as f64 * 0.007 + s).sin();
                    if i == j { v + n as f64 } else { v * 0.1 } // diagonally dominant
                })
                .collect()
        })
        .collect()
}
fn t(label: &str, mut f: impl FnMut()) {
    f();
    let st = Instant::now();
    f();
    println!("{label}: {:.1} ms", st.elapsed().as_secs_f64() * 1e3);
}
fn main() {
    for n in [1024usize, 2048] {
        let a = mk(n, 0.3);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
        let bb = mk(n, 1.1);
        t(&format!("matmul   n={n}"), || {
            black_box(matmul(&a, &bb).unwrap());
        });
        t(&format!("solve    n={n}"), || {
            black_box(solve(&a, &b, SolveOptions::default()).unwrap());
        });
        t(&format!("inv      n={n}"), || {
            black_box(inv(&a, InvOptions::default()).unwrap());
        });
    }
}
