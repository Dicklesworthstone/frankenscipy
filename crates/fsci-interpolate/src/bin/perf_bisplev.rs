use fsci_interpolate::{BISPLEV_COMPACT_DISABLE, bisplev};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn clamped_knots(nc: usize, k: usize) -> Vec<f64> {
    // len = nc + k + 1; first/last k+1 clamped to 0/1, interior uniform.
    let n_int = nc - k; // number of intervals
    let mut t = vec![0.0; k + 1];
    for i in 1..n_int {
        t.push(i as f64 / n_int as f64);
    }
    t.extend(std::iter::repeat_n(1.0, k + 1));
    t
}
fn main() {
    let mut s = 7u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let (kx, ky) = (3usize, 3usize);
    for &(nc, gpts) in &[(30usize, 300usize), (60, 500), (100, 500)] {
        let tx = clamped_knots(nc, kx);
        let ty = clamped_knots(nc, ky);
        let nx_c = tx.len() - kx - 1;
        let ny_c = ty.len() - ky - 1;
        let c: Vec<f64> = (0..nx_c * ny_c).map(|_| r() * 2.0 - 1.0).collect();
        let tck = (tx, ty, c, kx, ky);
        let xg: Vec<f64> = (0..gpts).map(|i| i as f64 / (gpts - 1) as f64).collect();
        let yg = xg.clone();
        let run = |dis: bool| {
            BISPLEV_COMPACT_DISABLE.store(dis, Ordering::Relaxed);
            bisplev(&xg, &yg, &tck).unwrap()
        };
        let cpt = run(false);
        let full = run(true);
        let md = cpt
            .iter()
            .flatten()
            .zip(full.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let bench = |dis: bool| {
            BISPLEV_COMPACT_DISABLE.store(dis, Ordering::Relaxed);
            let f = || bisplev(&xg, &yg, &tck).unwrap().len();
            let _ = black_box(f());
            let reps = 4;
            let t = Instant::now();
            for _ in 0..reps {
                black_box(f());
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let (mut a, mut b) = (f64::MAX, f64::MAX);
        for _ in 0..3 {
            b = b.min(bench(true));
            a = a.min(bench(false));
        }
        println!(
            "nc={nc} grid={gpts}²: full {b:.2}ms compact {a:.2}ms {:.2}x maxdiff={md:.1e}",
            b / a
        );
    }
}
