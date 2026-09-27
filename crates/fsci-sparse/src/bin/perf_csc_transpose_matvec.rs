//! A/B for byte-identical parallel Aᵀ·x via a cached CSC column-gather vs the
//! serial CSR scatter (`csr_matvec_transpose`). The serial scatter accumulates
//! result[col] in increasing-row order; a CSC stores each column's entries in
//! increasing-row order, so the column-gather sums in the SAME order ⇒
//! byte-identical, and each output column is independent ⇒ parallel.
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_csc_transpose_matvec`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{CscMatrix, CsrMatrix, FormatConvertible, Shape2D, random};

// Serial CSR scatter (verbatim csr_matvec_transpose).
fn csr_t(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let shape = a.shape();
    let (indptr, indices, data) = (a.indptr(), a.indices(), a.data());
    let mut result = vec![0.0; shape.cols];
    for i in 0..shape.rows {
        for idx in indptr[i]..indptr[i + 1] {
            result[indices[idx]] += data[idx] * x[i];
        }
    }
    result
}

// Parallel CSC column-gather. result[c] = Σ data[idx]*x[rows[idx]] over column c.
fn csc_gather(csc: &CscMatrix, x: &[f64], nthreads: usize) -> Vec<f64> {
    let n = csc.shape().cols;
    let (indptr, indices, data) = (csc.indptr(), csc.indices(), csc.data());
    let mut result = vec![0.0; n];
    if nthreads <= 1 {
        for c in 0..n {
            let mut s = 0.0;
            for idx in indptr[c]..indptr[c + 1] {
                s += data[idx] * x[indices[idx]];
            }
            result[c] = s;
        }
        return result;
    }
    let chunk = n.div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (t, slot) in result.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            scope.spawn(move || {
                for (r, out) in slot.iter_mut().enumerate() {
                    let c = base + r;
                    let mut s = 0.0;
                    for idx in indptr[c]..indptr[c + 1] {
                        s += data[idx] * x[indices[idx]];
                    }
                    *out = s;
                }
            });
        }
    });
    result
}

fn main() {
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1);
    println!("cores={cores}");
    for &(m, n, density) in &[
        (60_000usize, 40_000usize, 0.00015f64),
        (200_000, 150_000, 0.00005),
        (400_000, 300_000, 0.00003),
    ] {
        let a = random(Shape2D::new(m, n), density, 0xA5A5 ^ m as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let csc = a.to_csc().unwrap();
        let nnz = a.data().len();
        let mut x = vec![0.0; m];
        let mut s = 0x9e37u64;
        for xi in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *xi = (s >> 11) as f64 / (1u64 << 53) as f64;
        }

        let ser = csr_t(&a, &x);
        let par = csc_gather(&csc, &x, cores);
        let identical = ser.len() == par.len()
            && ser
                .iter()
                .zip(&par)
                .all(|(a, b)| a.to_bits() == b.to_bits());

        let reps = 200;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += csr_t(black_box(&a), black_box(&x))[0];
        }
        let ts = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..reps {
            acc += csc_gather(black_box(&csc), black_box(&x), cores)[0];
        }
        let tp = t1.elapsed();
        println!(
            "m={m:>7} n={n:>7} nnz={nnz:>9} identical={identical}  csr_scatter={:>9.3?}  csc_parallel={:>9.3?}  ratio={:>5.1}x (acc={acc:.3})",
            ts / reps,
            tp / reps,
            ts.as_secs_f64() / tp.as_secs_f64()
        );
    }
    println!("\n--- end-to-end (lib lsqr + svds, use cached-CSC Aᵀ) ---");
    end_to_end();
}

// End-to-end: lib lsqr + svds (both alternate A and Aᵀ each iteration). Run after
// stashing the lib change to get the before/after. Uses the lib's cached-CSC Aᵀ.
fn end_to_end() {
    use fsci_sparse::{EigsOptions, IterativeSolveOptions, lsqr, svds};
    for &(m, n, density) in &[(200_000usize, 150_000usize, 0.00006f64)] {
        let a = random(Shape2D::new(m, n), density, 0xD00D ^ m as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let nnz = a.data().len();
        let mut b = vec![0.0; m];
        let mut s = 0x55u64;
        for bi in b.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *bi = (s >> 11) as f64 / (1u64 << 53) as f64;
        }
        let opt = IterativeSolveOptions {
            max_iter: Some(60),
            ..Default::default()
        };
        let t0 = Instant::now();
        let r = lsqr(&a, &b, opt);
        println!(
            "lsqr  m={m} n={n} nnz={nnz} ok={} in {:?}",
            r.is_ok(),
            t0.elapsed()
        );

        let sq = random(Shape2D::new(120_000, 120_000), 0.00008, 0xBEEF)
            .unwrap()
            .to_csr()
            .unwrap();
        let t1 = Instant::now();
        let rs = svds(
            &sq,
            2,
            EigsOptions {
                tol: 1e-7,
                max_iter: 60,
                ..EigsOptions::default()
            },
        );
        println!(
            "svds  n=120000 nnz={} ok={} in {:?}",
            sq.data().len(),
            rs.is_ok(),
            t1.elapsed()
        );
    }
}
