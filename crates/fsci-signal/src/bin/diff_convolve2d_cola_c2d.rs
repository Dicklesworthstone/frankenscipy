//! Differential oracle probe for four `scipy.signal` entry points with NO differential coverage
//! (frankenscipy-ivxx6): `convolve2d`, `check_COLA`, `check_NOLA`, `cont2discrete`.
//!
//! Each was confirmed to return zero referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! across the FULL corpus -- both locations, since differential tests are split between them.
//!
//! FIXTURE NOTES.
//!   * `convolve2d` uses a NON-SYMMETRIC kernel and a non-square image. A symmetric kernel hides
//!     any transposition or reflection error, which is the commonest way a 2-D convolution
//!     diverges; convolution reflects the kernel where correlation does not.
//!   * every `mode` is swept, and every `boundary` on the boundary-aware entry point, since the
//!     edge treatment is where implementations differ once the interior agrees.
//!   * `check_COLA` / `check_NOLA` are swept over windows and hop sizes that must give BOTH
//!     answers. A predicate probed only where it returns true is indistinguishable from one that
//!     always returns true.
//!
//! Lines: `name|v;v;v`. Inputs must match `python/diff_convolve2d_cola_c2d.py`.
use fsci_signal::{
    Boundary2d, ConvolveMode, check_COLA, check_NOLA, cont2discrete, convolve2d,
    convolve2d_with_boundary,
};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}

fn main() {
    // 4x5 image, 3x2 NON-SYMMETRIC kernel.
    #[rustfmt::skip]
    let a: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 1.0,
        2.0, 4.0, 6.0, 8.0, 3.0,
        5.0, 7.0, 9.0, 2.0, 4.0,
    ];
    let a_shape = (4usize, 5usize);
    let v: Vec<f64> = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5];
    let v_shape = (3usize, 2usize);

    for (mname, mode) in [
        ("full", ConvolveMode::Full),
        ("same", ConvolveMode::Same),
        ("valid", ConvolveMode::Valid),
    ] {
        if let Ok(out) = convolve2d(&a, a_shape, &v, v_shape, mode) {
            dump(&format!("conv2d_{mname}"), &out);
        }
    }

    for (bname, boundary) in [
        ("fill", Boundary2d::Fill),
        ("wrap", Boundary2d::Wrap),
        ("symm", Boundary2d::Symm),
    ] {
        if let Ok(out) =
            convolve2d_with_boundary(&a, a_shape, &v, v_shape, ConvolveMode::Same, boundary, 0.0)
        {
            dump(&format!("conv2d_same_{bname}"), &out);
        }
    }

    // ---- check_COLA / check_NOLA -------------------------------------------------------------
    // Hann at 50% overlap satisfies COLA; the same window at an awkward hop does not. Both arms
    // are emitted so the predicate is pinned in both directions.
    let n = 16usize;
    let hann: Vec<f64> = (0..n)
        .map(|i| {
            let x = std::f64::consts::PI * i as f64 / n as f64;
            x.sin().powi(2)
        })
        .collect();
    let boxcar: Vec<f64> = vec![1.0; n];
    let mut cola = Vec::new();
    let mut nola = Vec::new();
    for (w, _wname) in [(&hann, "hann"), (&boxcar, "boxcar")] {
        for noverlap in [0usize, 4, 8, 12, 13] {
            cola.push(match check_COLA(w, n, noverlap) {
                Ok(true) => 1.0,
                Ok(false) => 0.0,
                Err(_) => -1.0,
            });
            nola.push(match check_NOLA(w, n, noverlap) {
                Ok(true) => 1.0,
                Ok(false) => 0.0,
                Err(_) => -1.0,
            });
        }
    }
    dump("check_cola", &cola);
    dump("check_nola", &nola);

    // ---- cont2discrete -----------------------------------------------------------------------
    // Second-order lowpass, several discretisation methods.
    let num = vec![1.0];
    let den = vec![1.0, 0.7, 1.0];
    for method in ["zoh", "bilinear", "euler", "backward_diff"] {
        if let Ok((nd, dd)) = cont2discrete(&num, &den, 0.1, method, None) {
            dump(&format!("c2d_{method}_num"), &nd);
            dump(&format!("c2d_{method}_den"), &dd);
        }
    }
    if let Ok((nd, dd)) = cont2discrete(&num, &den, 0.1, "gbt", Some(0.3)) {
        dump("c2d_gbt03_num", &nd);
        dump("c2d_gbt03_den", &dd);
    }
}
