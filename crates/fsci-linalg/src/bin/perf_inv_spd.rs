use fsci_linalg::{InvOptions, MatrixAssumption, inv};
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
    for &n in &[256usize, 384, 512, 768] {
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
        let pos = InvOptions {
            assume_a: Some(MatrixAssumption::PositiveDefinite),
            ..Default::default()
        };
        let ig = inv(&a, InvOptions::default()).unwrap().inverse;
        let ip = inv(&a, pos).unwrap().inverse;
        let maxdiff = ig
            .iter()
            .zip(&ip)
            .flat_map(|(p, q)| p.iter().zip(q))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let t = |o: InvOptions| {
            let _ = black_box(inv(&a, o).unwrap());
            let reps = 4;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(inv(black_box(&a), o).unwrap());
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let g = t(InvOptions::default());
        let p = t(pos);
        println!(
            "n={n:4}: inv(General) {g:6.2}ms | inv(pos)NEW {p:6.2}ms ({:.2}x vs General) maxdiff={maxdiff:.1e}",
            g / p
        );
    }
}
