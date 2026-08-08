//! Median-null-gated A/B for `sparse::sparse_submatrix`: ORIG serial per-row col-range extract vs
//! gather-then-concat across contiguous row-blocks. BYTE-IDENTICAL (same extract/shift, rows/entries
//! concatenated in stored order). Toggled by `SPARSE_SUBMATRIX_FORCE_SERIAL`.
//! Args: rows [nnz_per_row] [iters].
use fsci_sparse::{CsrMatrix, SPARSE_SUBMATRIX_FORCE_SERIAL, Shape2D, sparse_submatrix};
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let nnz_per: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
    let cols = nnz_per.max(1) * 4;

    let mut s = 0x6c8e_9cf5u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let nnz = rows * nnz_per;
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    for i in 0..rows {
        indptr.push(i * nnz_per);
        for j in 0..nnz_per {
            indices.push(j); // ascending distinct cols 0..nnz_per
            let bits = rnd();
            data.push((bits >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0);
        }
    }
    indptr.push(nnz);
    let mat = CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false)
        .expect("csr");

    // Extract all rows, keep the lower half of the stored column band -> processes all nnz.
    let (r0, r1, c0, c1) = (0usize, rows, 0usize, nnz_per / 2 + 1);

    SPARSE_SUBMATRIX_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let sa = sparse_submatrix(&mat, r0, r1, c0, c1);
    SPARSE_SUBMATRIX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let pa = sparse_submatrix(&mat, r0, r1, c0, c1);
    let dbit = sa
        .data()
        .iter()
        .zip(pa.data())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let ibit = sa
        .indices()
        .iter()
        .zip(pa.indices())
        .filter(|(x, y)| x != y)
        .count();
    let pbit = sa
        .indptr()
        .iter()
        .zip(pa.indptr())
        .filter(|(x, y)| x != y)
        .count();
    let lenmis = (sa.data().len() != pa.data().len()) as usize;
    let bitmism = dbit + ibit + pbit + lenmis;
    println!(
        "# sparse::sparse_submatrix rows={rows} in_nnz={nnz} out_nnz={} dbit={dbit} ibit={ibit} pbit={pbit} lenmis={lenmis} bitmism={bitmism}",
        sa.data().len()
    );

    // Dispatch proof (frankenscipy-vacuous-perf-toggles-qcuyy). A two-arm A/B
    // measures something only if the library actually consults the switch. When
    // it never does, both arms run identical code and the ratio is noise over
    // noise — so refuse to print one rather than emit a confident number.
    if !SPARSE_SUBMATRIX_FORCE_SERIAL.dispatch_observed(|| {
        let _ = black_box(sparse_submatrix(black_box(&mat), r0, r1, c0, c1));
    }) {
        eprintln!(
            "VACUOUS sparse_submatrix: SPARSE_SUBMATRIX_FORCE_SERIAL is never read by \
             fsci-sparse, so both arms are the same code path. No ratio is reportable until \
             the parallel route is restored. See frankenscipy-vacuous-perf-toggles-qcuyy."
        );
        std::process::exit(2);
    }

    let bench = |serial: bool| -> f64 {
        SPARSE_SUBMATRIX_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(sparse_submatrix(black_box(&mat), r0, r1, c0, c1));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(sparse_submatrix(black_box(&mat), r0, r1, c0, c1));
        }
        t.elapsed().as_secs_f64() / 5.0 * 1e3
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
    SPARSE_SUBMATRIX_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} sparse_submatrix serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
