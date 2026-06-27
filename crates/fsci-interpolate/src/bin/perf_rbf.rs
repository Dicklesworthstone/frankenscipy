use fsci_interpolate::{RbfInterpolator, RbfKernel};
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let mut seed = 3u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for &n in &[150usize, 300, 600] {
        let pts: Vec<Vec<f64>> = (0..n).map(|_| vec![r(), r(), r()]).collect();
        let vals: Vec<f64> = (0..n).map(|_| r()).collect();
        let _ = RbfInterpolator::new(&pts, &vals, RbfKernel::Multiquadric, 1.0);
        let t = Instant::now();
        let reps = 3;
        let mut acc = 0.0;
        for _ in 0..reps {
            let rbf = RbfInterpolator::new(
                black_box(&pts),
                black_box(&vals),
                RbfKernel::Multiquadric,
                1.0,
            )
            .unwrap();
            acc += rbf.eval(&[0.5, 0.5, 0.5]);
        }
        println!(
            "n={n}: rbf_build {:.1} ms (acc={acc:.3})",
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        );
    }
}
