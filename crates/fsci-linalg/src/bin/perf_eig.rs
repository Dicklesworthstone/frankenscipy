use fsci_linalg::*;
use std::time::Instant;
fn main() {
    let mut seed = 42u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[100usize, 150, 200, 300, 400] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let _ = eig(&a, DecompOptions::default());
        let t = Instant::now();
        let reps = 2;
        for _ in 0..reps {
            let _ = eig(&a, DecompOptions::default());
        }
        let ms = t.elapsed().as_secs_f64() / reps as f64 * 1000.0;
        println!("eig n={n}: {ms:.1} ms");
    }
}
