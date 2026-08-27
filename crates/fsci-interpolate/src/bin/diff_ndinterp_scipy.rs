//! Differential oracle probe: `LinearNDInterpolator` and `CloughTocher2DInterpolator` against
//! `scipy.interpolate` (frankenscipy-ivxx6).
//!
//! WHY THESE TWO FIRST. Of the entry points with no differential coverage, these are the only ones
//! exercised by a harness that has a LIVE SciPy arm -- `perf_griddata_scipy.rs`. That harness can
//! take a competitive ratio against SciPy on interpolators nobody has shown compute the same
//! thing, which is exactly the situation that stopped the RBF row (frankenscipy-icozs).
//!
//! FIXTURE DESIGN, because a naive scattered-data comparison proves very little:
//!   * LINEAR control -- on data from a linear function EVERY valid triangulation produces the
//!     same interpolant, so agreement here tests the barycentric arithmetic and NOT the
//!     triangulation. If this arm disagrees, the basic math is wrong.
//!   * NONLINEAR case in GENERAL POSITION -- coordinates are chosen irrational-ish so no four
//!     points are cocircular, which makes the Delaunay triangulation UNIQUE. Agreement is then
//!     genuinely required rather than lucky; on a degenerate fixture the two libraries could pick
//!     different legal triangulations and disagree without either being wrong.
//!   * OUTSIDE THE HULL -- must be NaN, SciPy's default. This is the must-miss arm: an
//!     interpolator that extrapolates silently would pass every interior check.
//!
//! Lines: `name,r,c,value`. Inputs must match `python/diff_ndinterp_scipy.py`.
use fsci_interpolate::{CloughTocher2DInterpolator, LinearNDInterpolator};

/// Twelve sites in general position on the unit square. Deliberately not on a grid and not
/// symmetric, so the Delaunay triangulation is unique.
fn sites() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![0.318_309_886_183_790_7, 0.15915494309189535],
        vec![0.693_147_180_559_945_3, 0.43429448190325176],
        vec![0.577_215_664_901_532_9, 0.866_025_403_784_438_6],
        vec![0.135_335_283_236_612_7, 0.606_530_659_712_633_4],
        vec![0.866_025_403_784_438_6, 0.20787957635076193],
        vec![0.41421356237309515, 0.732_050_807_568_877_3],
        vec![0.22360679774997896, 0.33166247903553998],
        vec![0.785_398_163_397_448_3, 0.618_033_988_749_894_8],
    ]
}

fn queries() -> Vec<Vec<f64>> {
    vec![
        // interior
        vec![0.25, 0.25],
        vec![0.5, 0.5],
        vec![0.75, 0.25],
        vec![0.4, 0.8],
        vec![0.6, 0.15],
        vec![0.15, 0.85],
        // outside the convex hull -- must be NaN
        vec![-0.25, 0.5],
        vec![1.4, 0.5],
        vec![0.5, -0.3],
    ]
}

fn linear(p: &[f64]) -> f64 {
    2.0 * p[0] - 3.0 * p[1] + 1.0
}

fn nonlinear(p: &[f64]) -> f64 {
    (3.0 * p[0]).sin() * (2.0 * p[1]).cos() + 0.5 * p[0]
}

fn emit(name: &str, vals: &[f64]) {
    for (i, &v) in vals.iter().enumerate() {
        // NaN is printed as `nan` so the comparator can match it exactly rather than by value.
        if v.is_nan() {
            println!("{name},{i},0,nan");
        } else {
            println!("{name},{i},0,{v:.17e}");
        }
    }
}

fn main() {
    let pts = sites();
    let qs = queries();

    for (label, f) in [
        ("linear", linear as fn(&[f64]) -> f64),
        ("nonlinear", nonlinear),
    ] {
        let vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();

        if let Ok(it) = LinearNDInterpolator::new(&pts, &vals) {
            let out: Vec<f64> = qs.iter().map(|q| it.eval(q).unwrap_or(f64::NAN)).collect();
            emit(&format!("linearnd_{label}"), &out);
        }

        if let Ok(it) = CloughTocher2DInterpolator::new(&pts, &vals) {
            let out: Vec<f64> = qs.iter().map(|q| it.eval(q).unwrap_or(f64::NAN)).collect();
            emit(&format!("clough_{label}"), &out);
        }
    }
}
