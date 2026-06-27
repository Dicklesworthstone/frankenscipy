//! complex_ode probe vs scipy.integrate.complex_ode (dopri5).
//! Lines: `case,j,re,im`. The python comparator integrates the same systems.
use fsci_integrate::complex::{Complex64, complex_ode};

fn cmul(a: Complex64, b: Complex64) -> Complex64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn dump(case: &str, y: &[Complex64]) {
    for (j, c) in y.iter().enumerate() {
        println!("{case},{j},{:.17e},{:.17e}", c.0, c.1);
    }
}

fn main() {
    let rtol = 1e-9;
    let atol = 1e-12;

    // 1) y' = i*y, y(0)=1 -> exp(i t); evaluate at t=2.
    let f1 = |_t: f64, y: &[Complex64]| vec![cmul((0.0, 1.0), y[0])];
    let r1 = complex_ode(f1, &[(1.0, 0.0)], (0.0, 2.0), Some(&[2.0]), rtol, atol).unwrap();
    dump("expi", r1.y.last().unwrap());

    // 2) y' = (-1+2i)*y, y(0)=1 -> exp((-1+2i) t); evaluate at t=1.5.
    let f2 = |_t: f64, y: &[Complex64]| vec![cmul((-1.0, 2.0), y[0])];
    let r2 = complex_ode(f2, &[(1.0, 0.0)], (0.0, 1.5), Some(&[1.5]), rtol, atol).unwrap();
    dump("decay", r2.y.last().unwrap());

    // 3) Coupled 2-D system: y0' = i*y1, y1' = i*y0; y(0) = (1, 0). Evaluate at t=1.
    let f3 = |_t: f64, y: &[Complex64]| vec![cmul((0.0, 1.0), y[1]), cmul((0.0, 1.0), y[0])];
    let r3 = complex_ode(
        f3,
        &[(1.0, 0.0), (0.0, 0.0)],
        (0.0, 1.0),
        Some(&[1.0]),
        rtol,
        atol,
    )
    .unwrap();
    dump("coupled", r3.y.last().unwrap());
}
