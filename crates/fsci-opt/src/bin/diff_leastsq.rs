//! leastsq (MINPACK-style interface) probe vs scipy.optimize.leastsq.
//! Lines: `name,key,values...`. Inputs must match the python comparator.
use fsci_opt::curvefit::{LeastSquaresOptions, leastsq};

fn dump(name: &str, x: &[f64], ier: i32) {
    let xs: Vec<String> = x.iter().map(|v| format!("{v:.15e}")).collect();
    println!("{name},ier,{ier},x,{}", xs.join(","));
}

fn main() {
    let opts = LeastSquaresOptions::default();

    // 1) Rosenbrock residuals r = [10(x1 - x0^2), 1 - x0]; minimum at (1, 1).
    let r1 = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
    if let Ok(res) = leastsq(r1, &[-1.2, 1.0], opts.clone()) {
        dump("rosen", &res.x, res.ier);
    }

    // 2) Exponential decay fit: residuals a*exp(-b*t) + c - y, truth [2.5, 1.3, 0.5].
    let ts: Vec<f64> = (0..25).map(|i| i as f64 * 0.2).collect();
    let truth = [2.5_f64, 1.3, 0.5];
    let ys: Vec<f64> = ts
        .iter()
        .map(|&t| truth[0] * (-truth[1] * t).exp() + truth[2])
        .collect();
    let r2 = move |p: &[f64]| -> Vec<f64> {
        ts.iter()
            .zip(ys.iter())
            .map(|(&t, &y)| p[0] * (-p[1] * t).exp() + p[2] - y)
            .collect()
    };
    if let Ok(res) = leastsq(r2, &[1.0, 1.0, 1.0], opts.clone()) {
        dump("expfit", &res.x, res.ier);
    }

    // 3) Overdetermined linear residuals r_i = a*u_i + b - v_i.
    let us = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let vs = [0.1_f64, 0.9, 2.1, 2.9, 4.2, 4.8];
    let r3 = move |p: &[f64]| -> Vec<f64> {
        us.iter()
            .zip(vs.iter())
            .map(|(&u, &v)| p[0] * u + p[1] - v)
            .collect()
    };
    if let Ok(res) = leastsq(r3, &[0.0, 0.0], opts) {
        dump("linfit", &res.x, res.ier);
    }
}
