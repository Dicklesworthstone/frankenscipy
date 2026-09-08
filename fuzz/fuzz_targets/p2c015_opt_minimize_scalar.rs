#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::{golden, minimize_scalar, MinimizeScalarOptions};
use libfuzzer_sys::fuzz_target;

const TOL: f64 = 1e-6;

#[derive(Debug, Arbitrary)]
struct MinInput {
    a: f64,
    b: f64,
    func_type: u8,
    coeffs: [f64; 3],
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-100.0, 100.0)
    } else {
        0.0
    }
}

fn eval_func(func_type: u8, coeffs: &[f64; 3], x: f64) -> f64 {
    match func_type % 4 {
        0 => {
            let c = coeffs.map(sanitize);
            c[0] + c[1] * x + c[2] * x * x
        }
        1 => x.powi(4) - 4.0 * x.powi(2) + x,
        2 => (x - 1.0).powi(2) + 0.5,
        3 => x.sin() + 0.1 * x.powi(2),
        _ => x.powi(2),
    }
}

fuzz_target!(|input: MinInput| {
    let mut a = sanitize(input.a);
    let mut b = sanitize(input.b);

    if (a - b).abs() < 1e-8 {
        return;
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let f = |x: f64| eval_func(input.func_type, &input.coeffs, x);

    let fa = f(a);
    let fb = f(b);
    if !fa.is_finite() || !fb.is_finite() {
        return;
    }

    let bracket = (a, b);
    let opts = MinimizeScalarOptions::default();

    if let Ok(result) = minimize_scalar(f, bracket, opts) {
        let x_min = result.x;
        let f_min = result.fun;

        if x_min < a - TOL || x_min > b + TOL {
            panic!(
                "minimize_scalar: x_min {} outside bracket [{}, {}]",
                x_min, a, b
            );
        }

        let computed_fmin = f(x_min);
        if (computed_fmin - f_min).abs() > TOL * f_min.abs().max(1.0) {
            panic!(
                "minimize_scalar: f(x_min)={} != fun={} (diff={})",
                computed_fmin, f_min, (computed_fmin - f_min).abs()
            );
        }

        if f_min > fa + TOL && f_min > fb + TOL {
            panic!(
                "minimize_scalar: f_min={} > f(a)={} and f(b)={} — not a minimum",
                f_min, fa, fb
            );
        }
    }

    let (golden_x, golden_f) = golden(f, a, b, 1e-6, 500);

    if golden_x < a - TOL || golden_x > b + TOL {
        panic!(
            "golden: x_min {} outside bracket [{}, {}]",
            golden_x, a, b
        );
    }

    let computed_golden_f = f(golden_x);
    if (computed_golden_f - golden_f).abs() > TOL * golden_f.abs().max(1.0) {
        panic!(
            "golden: f(x_min)={} != returned f_min={} (diff={})",
            computed_golden_f, golden_f, (computed_golden_f - golden_f).abs()
        );
    }
});
