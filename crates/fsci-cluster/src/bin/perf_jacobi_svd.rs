// jacobi_svd robustness/throughput on the exactly-rank-deficient inputs that stall the
// implicit-shift svd() (~32 s, bead 9xrce). Times jacobi_svd and checks reconstruction; the
// implicit-shift svd() numbers come from the probe_svd_stall repro (not re-run here — it hangs).
use fsci_linalg::{DecompOptions, jacobi_svd};
use std::hint::black_box;
use std::time::Instant;

fn lowrank(m: usize, n: usize, r: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut st = seed;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let b: Vec<Vec<f64>> = (0..m).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let g: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng()).collect()).collect();
    (0..m)
        .map(|i| {
            (0..n)
                .map(|j| (0..r).map(|t| b[i][t] * g[t][j]).sum())
                .collect()
        })
        .collect()
}

fn recon_err(a: &[Vec<f64>], res: &fsci_linalg::SvdResult) -> f64 {
    let m = a.len();
    let n = a[0].len();
    let k = res.s.len();
    let mut e = 0.0f64;
    for (i, a_i) in a.iter().enumerate().take(m) {
        for (j, &aij) in a_i.iter().enumerate().take(n) {
            let v: f64 = (0..k).map(|t| res.u[i][t] * res.s[t] * res.vt[t][j]).sum();
            e = e.max((v - aij).abs());
        }
    }
    e
}

fn main() {
    for &(m, n, r, tag) in &[
        (
            800usize,
            600usize,
            30usize,
            "800x600 rank-30 EXACT (svd() stalls ~35s)",
        ),
        (600, 600, 30, "600x600 rank-30 EXACT (svd() stalls ~28s)"),
        (800, 600, 600, "800x600 FULL rank"),
    ] {
        let a = lowrank(m, n, r.min(n), 0x1234 + m as u64);
        let t = Instant::now();
        let res = jacobi_svd(&a, DecompOptions::default()).expect("jacobi_svd");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let nz = res.s.iter().filter(|&&x| x > 1e-9).count();
        black_box(&res);
        println!(
            "jacobi_svd {tag}: {ms:.1} ms | reconstruction={:.2e} | nonzero_sv={nz}",
            recon_err(&a, &res)
        );
    }
}
