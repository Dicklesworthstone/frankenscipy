// De-risk for parallelizing matmul_toeplitz_fft over columns. The shared
// fft(emb) is computed once; each column's fft(xpad)/ifft(prod) is INDEPENDENT,
// so distributing columns across threads is BYTE-IDENTICAL (same per-column
// arithmetic, same fhat). This bench measures serial vs thread::scope-parallel
// column sweeps at various (n, k) to find where parallel amortizes thread spawn.
use std::time::Instant;

fn next_pow2(mut v: usize) -> usize {
    if v <= 1 {
        return 1;
    }
    v -= 1;
    let mut p = 1usize;
    while p < v + 1 {
        p <<= 1;
    }
    p
}

fn build_emb_fhat(c: &[f64], row: &[f64], m: usize, n: usize, l: usize) -> Vec<(f64, f64)> {
    let opts = fsci_fft::FftOptions::default();
    let mut emb = vec![(0.0_f64, 0.0_f64); l];
    for i in 0..m {
        emb[i] = (c[i], 0.0);
    }
    for s in 0..n - 1 {
        emb[l - (n - 1) + s] = (row[n - 1 - s], 0.0);
    }
    fsci_fft::fft(&emb, &opts).expect("emb fft")
}

fn one_col(fhat: &[(f64, f64)], xcol: &[f64], m: usize, n: usize, l: usize) -> Vec<f64> {
    let opts = fsci_fft::FftOptions::default();
    let mut xpad = vec![(0.0_f64, 0.0_f64); l];
    for i in 0..n {
        xpad[i] = (xcol[i], 0.0);
    }
    let xhat = fsci_fft::fft(&xpad, &opts).expect("x fft");
    let mut prod = Vec::with_capacity(l);
    for (&(fr, fi), &(xr, xi)) in fhat.iter().zip(xhat.iter()) {
        prod.push((fr * xr - fi * xi, fr * xi + fi * xr));
    }
    let yfull = fsci_fft::ifft(&prod, &opts).expect("ifft");
    (0..m).map(|i| yfull[i].0).collect()
}

// columns of x stored column-major here: x_cols[col] = the length-n column.
fn serial(fhat: &[(f64, f64)], x_cols: &[Vec<f64>], m: usize, n: usize, l: usize) -> Vec<Vec<f64>> {
    x_cols.iter().map(|xc| one_col(fhat, xc, m, n, l)).collect()
}

fn parallel(
    fhat: &[(f64, f64)],
    x_cols: &[Vec<f64>],
    m: usize,
    n: usize,
    l: usize,
) -> Vec<Vec<f64>> {
    let k = x_cols.len();
    let cores = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(16);
    let nthreads = cores.min(k).max(1);
    let chunk = k.div_ceil(nthreads);
    let mut out: Vec<(usize, Vec<f64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .map(|t| {
                let start = t * chunk;
                let end = ((t + 1) * chunk).min(k);
                scope.spawn(move || {
                    (start..end)
                        .map(|col| (col, one_col(fhat, &x_cols[col], m, n, l)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut acc: Vec<(usize, Vec<f64>)> = Vec::with_capacity(k);
        for h in handles {
            acc.extend(h.join().expect("thread"));
        }
        acc
    })
    .into_iter()
    .collect::<Vec<_>>();
    out.sort_by_key(|(c, _)| *c);
    out.into_iter().map(|(_, v)| v).collect()
}

fn build(n: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let mut s: u64 = 0x51ed_270b_2e07_6a13;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let c: Vec<f64> = (0..n).map(|_| u()).collect();
    let mut r: Vec<f64> = (0..n).map(|_| u()).collect();
    r[0] = c[0];
    // column-major x: k columns each length n
    let x_cols: Vec<Vec<f64>> = (0..k).map(|_| (0..n).map(|_| u()).collect()).collect();
    (c, r, x_cols)
}

fn best_of(
    reps: usize,
    mut f: impl FnMut() -> Vec<Vec<f64>>,
) -> (std::time::Duration, Vec<Vec<f64>>) {
    let mut best = std::time::Duration::MAX;
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        out = f();
        std::hint::black_box(out.len());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, out)
}

fn main() {
    println!(
        "{:>6} {:>5} {:>12} {:>12} {:>9}  exact",
        "n", "k", "serial_us", "par_us", "speedup"
    );
    for &(n, k) in &[
        (512usize, 4usize),
        (512, 16),
        (512, 64),
        (1024, 8),
        (1024, 32),
        (1024, 64),
        (2048, 16),
        (2048, 64),
        (4096, 64),
    ] {
        let (c, r, x_cols) = build(n, k);
        let m = n;
        let l = next_pow2(m + n - 1);
        let fhat = build_emb_fhat(&c, &r, m, n, l);
        let reps = if n >= 2048 { 6 } else { 20 };
        let (ts, ys) = best_of(reps, || serial(&fhat, &x_cols, m, n, l));
        let (tp, yp) = best_of(reps, || parallel(&fhat, &x_cols, m, n, l));
        let exact = ys == yp;
        let su = ts.as_secs_f64() * 1e6;
        let pu = tp.as_secs_f64() * 1e6;
        println!(
            "{n:>6} {k:>5} {su:>12.2} {pu:>12.2} {:>8.2}x  {}",
            su / pu,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
