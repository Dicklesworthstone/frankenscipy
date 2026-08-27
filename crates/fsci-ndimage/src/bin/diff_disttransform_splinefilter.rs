//! Differential oracle probe for three ndimage entry points with NO differential coverage
//! (frankenscipy-ivxx6): `distance_transform_bf`, `distance_transform_cdt`, `spline_filter1d`.
//!
//! All three are timed by existing perf bins -- `perf_bf_edt.rs`, `perf_cdt_chessboard.rs`,
//! `perf_cdt_taxicab.rs`, `perf_spline_filter1d.rs` -- so a vs-SciPy ratio could be taken on them
//! today without anyone having shown the two libraries agree.
//!
//! Each name was confirmed to return zero referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! across the FULL corpus -- both locations, since differential tests are split between them.
//!
//! FIXTURE NOTES. The binary image is deliberately NOT symmetric and has two disconnected
//! foreground blobs plus a single isolated pixel: distance transforms are easy to get right on a
//! filled rectangle and wrong on disconnected components. Anisotropic `sampling` is exercised
//! because that is where a metric implementation most often diverges. For `spline_filter1d` every
//! boundary mode is swept, since the prefilter's boundary handling is the part that differs
//! between implementations.
//!
//! Lines: `name|v;v;v`. Inputs must match `python/diff_disttransform_splinefilter.py`.
use fsci_ndimage::{
    BoundaryMode, DistanceMetric, NdArray, binary_dilation_with_structure, distance_transform_bf,
    distance_transform_cdt, spline_filter1d,
};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}

/// 6x7 binary image: two disconnected blobs and one isolated pixel.
fn binary_image() -> NdArray {
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    NdArray::new(data, vec![6, 7]).unwrap()
}

fn ramp() -> NdArray {
    let data: Vec<f64> = (0..12)
        .map(|i| {
            let x = i as f64;
            (0.7 * x).sin() * 3.0 + 0.25 * x
        })
        .collect();
    NdArray::new(data, vec![12]).unwrap()
}

fn main() {
    let img = binary_image();

    // ---- binary_dilation_with_structure, ASYMMETRIC footprint --------------------------------
    // Also uncovered. An asymmetric structuring element is the case that distinguishes a correct
    // dilation from one that forgets SciPy reflects the footprint, and it is the fixture an
    // in-tree test currently asserts DIFFERENT values for.
    #[rustfmt::skip]
    let dense_input = NdArray::new(vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
    ], vec![8, 9]).unwrap();
    #[rustfmt::skip]
    let dense_structure = NdArray::new(vec![
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
    ], vec![3, 5]).unwrap();
    if let Ok(out) = binary_dilation_with_structure(&dense_input, &dense_structure, 1) {
        dump("dilate_asym", &out.data);
    }

    // ---- distance_transform_bf -------------------------------------------------------------
    for (mname, metric) in [
        ("euclidean", DistanceMetric::Euclidean),
        ("taxicab", DistanceMetric::Taxicab),
        ("chessboard", DistanceMetric::Chessboard),
    ] {
        if let Ok(out) = distance_transform_bf(&img, metric, None) {
            dump(&format!("bf_{mname}"), &out.data);
        }
    }
    // Anisotropic sampling: the axes are weighted differently, which is where a metric
    // implementation most often diverges from SciPy.
    if let Ok(out) = distance_transform_bf(&img, DistanceMetric::Euclidean, Some(&[2.0, 0.5])) {
        dump("bf_euclidean_sampling", &out.data);
    }

    // ---- distance_transform_cdt ------------------------------------------------------------
    for (mname, metric) in [
        ("taxicab", DistanceMetric::Taxicab),
        ("chessboard", DistanceMetric::Chessboard),
    ] {
        if let Ok(out) = distance_transform_cdt(&img, metric) {
            dump(&format!("cdt_{mname}"), &out.data);
        }
    }

    // ---- spline_filter1d --------------------------------------------------------------------
    let r = ramp();
    for order in [2usize, 3, 4, 5] {
        for (bname, mode) in [
            ("reflect", BoundaryMode::Reflect),
            ("nearest", BoundaryMode::Nearest),
            ("wrap", BoundaryMode::Wrap),
            ("mirror", BoundaryMode::Mirror),
            ("constant", BoundaryMode::Constant),
        ] {
            if let Ok(out) = spline_filter1d(&r, order, 0, mode) {
                dump(&format!("spf1d_o{order}_{bname}"), &out.data);
            }
        }
    }
}
