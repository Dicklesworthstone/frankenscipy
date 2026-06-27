use fsci_linalg::*;
use std::time::Instant;
fn main() {
    let mut seed = 17u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[20usize, 40, 80] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            q[i][i] = 1.0;
        }
        macro_rules! bn {
            ($nm:expr,$e:expr) => {{
                let _ = $e;
                let t = Instant::now();
                for _ in 0..3 {
                    let _ = $e;
                }
                println!(
                    "n={} {:<14} {:.1} ms",
                    n,
                    $nm,
                    t.elapsed().as_secs_f64() / 3.0 * 1000.0
                );
            }};
        }
        bn!(
            "disc_lyapunov",
            solve_discrete_lyapunov(&a, &q, DecompOptions::default())
        );
        bn!(
            "cont_are",
            solve_continuous_are(&a, &q, &q, &q, DecompOptions::default())
        );
        bn!(
            "disc_are",
            solve_discrete_are(&a, &q, &q, &q, DecompOptions::default())
        );
    }
}
