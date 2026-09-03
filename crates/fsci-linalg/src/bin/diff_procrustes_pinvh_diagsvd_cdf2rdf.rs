//! Differential oracle probe for four entry points that had NO differential coverage at all
//! (frankenscipy-ivxx6): `orthogonal_procrustes`, `pinvh`, `diagsvd`, `cdf2rdf`.
//!
//! Each exists in `scipy.linalg` under the same name, and each was verified to return zero
//! referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! before this file was written. That check is the one that matters: the audit which found them
//! initially had its corpus wrong, so every name here was re-confirmed against the full surface.
//!
//! Lines: `name,r,c,value`. Inputs must match the python comparator
//! `crates/fsci-linalg/python/diff_procrustes_pinvh_diagsvd_cdf2rdf.py`.
use fsci_linalg::{cdf2rdf, diagsvd, orthogonal_procrustes, pinvh};

fn dump(name: &str, m: &[Vec<f64>]) {
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            println!("{name},{r},{c},{v:.17e}");
        }
    }
}

fn main() {
    // ---- orthogonal_procrustes -------------------------------------------------------------
    // A deliberately non-symmetric, non-square-friendly pair so the SVD has no accidental
    // structure to exploit; the answer is the orthogonal R minimising ||A R - B||_F.
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
        vec![-1.0, 0.5, 2.0],
    ];
    let b = vec![
        vec![2.0, 1.0, 3.5],
        vec![5.0, 4.5, 6.5],
        vec![8.0, 7.5, 9.0],
        vec![0.0, -0.5, 1.5],
    ];
    if let Ok((r, scale)) = orthogonal_procrustes(&a, &b) {
        dump("procrustes_r", &r);
        println!("procrustes_scale,0,0,{scale:.17e}");
    }

    // ---- pinvh ------------------------------------------------------------------------------
    // Symmetric indefinite, and separately a rank-deficient symmetric case, because the
    // pseudo-inverse of a singular matrix is where conventions diverge.
    let sym = vec![
        vec![4.0, 1.0, -2.0],
        vec![1.0, 3.0, 0.5],
        vec![-2.0, 0.5, 5.0],
    ];
    if let Ok(p) = pinvh(&sym, None, None) {
        dump("pinvh_sym", &p);
    }
    // Rank 2 of 3: third row/col is the sum of the first two.
    let rank_deficient = vec![
        vec![2.0, 1.0, 3.0],
        vec![1.0, 2.0, 3.0],
        vec![3.0, 3.0, 6.0],
    ];
    if let Ok(p) = pinvh(&rank_deficient, None, None) {
        dump("pinvh_rank2", &p);
    }

    // ---- diagsvd ----------------------------------------------------------------------------
    // Both non-square orientations, since diagsvd's whole job is the zero padding.
    if let Ok(d) = diagsvd(&[3.0, 2.0, 1.0], 5, 3) {
        dump("diagsvd_5x3", &d);
    }
    if let Ok(d) = diagsvd(&[3.0, 2.0, 1.0], 3, 5) {
        dump("diagsvd_3x5", &d);
    }

    // ---- cdf2rdf ----------------------------------------------------------------------------
    // One conjugate pair plus one real eigenvalue: the case the function exists for. Values are
    // the eigendecomposition of [[0, -1, 0], [1, 0, 0], [0, 0, 2]].
    let w = vec![(0.0, 1.0), (0.0, -1.0), (2.0, 0.0)];
    let v = vec![
        vec![
            (0.0, -0.707_106_781_186_547_5),
            (0.0, 0.707_106_781_186_547_5),
            (0.0, 0.0),
        ],
        vec![
            (0.707_106_781_186_547_5, 0.0),
            (0.707_106_781_186_547_5, 0.0),
            (0.0, 0.0),
        ],
        vec![(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
    ];
    if let Ok((wr, vr)) = cdf2rdf(&w, &v) {
        dump("cdf2rdf_w", &wr);
        dump("cdf2rdf_v", &vr);
    }
}
