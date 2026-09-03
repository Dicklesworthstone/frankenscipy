use fsci_linalg::*;
use std::time::Instant;
fn main() {
    let mut seed = 11u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[40usize, 80, 160] {
        let mk = |rr: &mut dyn FnMut() -> f64| {
            let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| rr()).collect()).collect();
            m
        };
        let a = mk(&mut r);
        let b = mk(&mut r);
        let c = mk(&mut r);
        let mut as_ = a.clone();
        for i in 0..n {
            as_[i][i] -= n as f64;
        }
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
                    "n={} {:<16} {:.1} ms",
                    n,
                    $nm,
                    t.elapsed().as_secs_f64() / 3.0 * 1000.0
                );
            }};
        }
        bn!(
            "sylvester",
            solve_sylvester(&as_, &b, &c, DecompOptions::default())
        );
        bn!(
            "cont_lyapunov",
            solve_continuous_lyapunov(&as_, &q, DecompOptions::default())
        );
        bn!("pinv", pinv(&a, PinvOptions::default()));
    }
}
