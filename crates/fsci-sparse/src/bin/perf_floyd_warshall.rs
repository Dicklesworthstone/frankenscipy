//! Timing + tolerance-parity harness for `floyd_warshall`.
//!
//! The naive O(n³) relaxation sweeps the whole matrix n times. The tiled version
//! processes B×B blocks so a tile stays cache-resident across all B of its
//! pivots — same shortest-path distances up to float reassociation. This dumps a
//! reachability count (must match EXACTLY) + a distance checksum (tolerance) and
//! a bit digest (exact on the n<128 reference path), then times the large-n win.
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_floyd_warshall`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, floyd_warshall};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

fn graph(n: usize, deg: usize, seed: u64) -> fsci_sparse::CsrMatrix {
    let mut s = seed;
    let (mut data, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        for _ in 0..deg {
            let j = (lcg(&mut s) >> 11) as usize % n;
            if j == i {
                continue;
            }
            let w = ((lcg(&mut s) >> 11) as f64 / (1u64 << 53) as f64) * 10.0 + 0.1;
            rows.push(i);
            cols.push(j);
            data.push(w);
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, true)
        .expect("coo")
        .to_csr()
        .expect("csr")
}

fn summary(d: &[Vec<f64>]) -> (usize, f64, u64) {
    let (mut reach, mut sum, mut h) = (0usize, 0.0f64, 1469598103934665603u64);
    for row in d {
        for &v in row {
            h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
            if v.is_finite() {
                reach += 1;
                sum += v;
            }
        }
    }
    (reach, sum, h)
}

fn main() {
    println!("===PARITY_PAYLOAD_BEGIN===");
    for &n in &[64usize, 127, 300, 600] {
        let (reach, sum, h) = summary(&floyd_warshall(&graph(n, 6, 7), true));
        println!("n={n} reach={reach} checksum={sum:.10e} digest={h:016x}");
    }
    println!("===PARITY_PAYLOAD_END===");

    for &n in &[512usize, 1024, 1600] {
        let g = graph(n, 8, 7);
        let reps = 3;
        let _ = floyd_warshall(&g, true);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let d = floyd_warshall(black_box(&g), true);
            acc += d[n / 2][n / 3];
        }
        println!("n={n}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
