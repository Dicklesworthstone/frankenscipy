//! Median-null-gated A/B for `sparse::sparse_add`: ORIG serial per-row BTreeMap merge vs
//! gather-then-concat across contiguous row-blocks. BYTE-IDENTICAL (same merge/filter, rows
//! concatenated in ascending order). Toggled by `SPARSE_ADD_FORCE_SERIAL`.
//! Args: rows [nnz_per_row] [iters].
use fsci_sparse::{CsrMatrix, SPARSE_ADD_FORCE_SERIAL, Shape2D, sparse_add};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 }
}

fn build(seed: u64, rows: usize, nnz_per: usize, cols: usize) -> CsrMatrix {
    let mut s = seed;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let nnz = rows * nnz_per;
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    // deterministic ascending distinct column indices per row (canonical), spread across cols
    let stride = (cols / nnz_per.max(1)).max(1);
    for i in 0..rows {
        indptr.push(i * nnz_per);
        let off = (i * 7) % stride;
        for j in 0..nnz_per {
            indices.push((off + j * stride).min(cols - 1));
            data.push(r());
        }
    }
    indptr.push(nnz);
    CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false).expect("csr")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let nnz_per: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
    let cols = (nnz_per.max(1) * 8).max(16);

    let a = build(0x1234_5678, rows, nnz_per, cols);
    let b = build(0x9abc_def0, rows, nnz_per, cols);

    SPARSE_ADD_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let s_out = sparse_add(&a, &b);
    SPARSE_ADD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let p_out = sparse_add(&a, &b);
    let dbit = s_out
        .data()
        .iter()
        .zip(p_out.data())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let ibit = s_out
        .indices()
        .iter()
        .zip(p_out.indices())
        .filter(|(x, y)| x != y)
        .count();
    let pbit = s_out
        .indptr()
        .iter()
        .zip(p_out.indptr())
        .filter(|(x, y)| x != y)
        .count();
    let lenmis = (s_out.data().len() != p_out.data().len()) as usize;
    let bitmism = dbit + ibit + pbit + lenmis;
    println!(
        "# sparse::sparse_add rows={rows} out_nnz={} dbit={dbit} ibit={ibit} pbit={pbit} lenmis={lenmis} bitmism={bitmism}",
        s_out.data().len()
    );

    // Dispatch proof (frankenscipy-vacuous-perf-toggles-qcuyy). A two-arm A/B
    // measures something only if the library actually consults the switch. When
    // it never does, both arms run identical code and the ratio is noise over
    // noise — so refuse to print one rather than emit a confident number.
    if !SPARSE_ADD_FORCE_SERIAL.dispatch_observed(|| {
        let _ = black_box(sparse_add(black_box(&a), black_box(&b)));
    }) {
        eprintln!(
            "VACUOUS sparse_add: SPARSE_ADD_FORCE_SERIAL is never read by fsci-sparse, so \
             both arms are the same code path. No ratio is reportable until the parallel \
             route is restored. See frankenscipy-vacuous-perf-toggles-qcuyy."
        );
        std::process::exit(2);
    }

    let bench = |serial: bool| -> f64 {
        SPARSE_ADD_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(sparse_add(black_box(&a), black_box(&b)));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(sparse_add(black_box(&a), black_box(&b)));
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv, mut nr, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        nr.push(o / o2);
        cr.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    SPARSE_ADD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} sparse_add serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
