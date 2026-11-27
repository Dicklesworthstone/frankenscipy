#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::{MinimizeScalarOptions, golden, minimize_scalar, minimize_scalar_bounded};
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

        // The two seeds may be expanded downhill, including beyond [a, b].
        if result.success {
            assert!(x_min.is_finite());
            assert!(!f_min.is_nan());
            assert!(result.nfev >= result.nit + 3);
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

    // A hard box belongs to the explicitly bounded solver. Check both its
    // evaluation domain and the point/value pair, including boundary minima.
    let (bounded_x, bounded_f) = minimize_scalar_bounded(
        |x| {
            assert!(x >= a - TOL && x <= b + TOL);
            f(x)
        },
        bracket,
        TOL,
        500,
    );
    assert!(bounded_x >= a - TOL && bounded_x <= b + TOL);
    assert_eq!(bounded_f, f(bounded_x));

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
