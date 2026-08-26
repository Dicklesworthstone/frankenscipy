//! Emit fsci's `griddata` linear values for the fixture written by `perf_griddata_live`, so the
//! hull-classification mismatch it detects can be attributed to specific query points.
//!
//! Exists because the live harness's parity gate found 4 of 2000 queries where fsci and SciPy
//! disagree on inside-vs-outside the convex hull, and "4 mismatches" is not yet a finding until
//! you know WHICH points and how far they sit from the boundary.
use fsci_interpolate::{GriddataMethod, griddata};
use std::io::Read;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/tmp/fsci_griddata_in.bin".to_string());
    let mut buf = Vec::new();
    std::fs::File::open(&path)
        .expect("fixture")
        .read_to_end(&mut buf)
        .expect("read");

    let rd = |o: usize| f64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
    let np = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let nq = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    let mut off = 16;
    let mut pts = Vec::with_capacity(np);
    for _ in 0..np {
        pts.push(vec![rd(off), rd(off + 8)]);
        off += 16;
    }
    let mut vals = Vec::with_capacity(np);
    for _ in 0..np {
        vals.push(rd(off));
        off += 8;
    }
    let mut xi = Vec::with_capacity(nq);
    for _ in 0..nq {
        xi.push(vec![rd(off), rd(off + 8)]);
        off += 16;
    }

    let out = griddata(&pts, &vals, &xi, GriddataMethod::Linear).expect("griddata");
    let body: Vec<String> = out
        .iter()
        .map(|v| {
            if v.is_nan() {
                "nan".to_string()
            } else {
                format!("{v:.17e}")
            }
        })
        .collect();
    println!("{}", body.join(","));
}
