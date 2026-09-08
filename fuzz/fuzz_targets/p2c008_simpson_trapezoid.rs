#![no_main]

use arbitrary::Arbitrary;
use fsci_integrate::{cumulative_trapezoid, simpson, trapezoid};
use libfuzzer_sys::fuzz_target;

// Composite-quadrature property oracle for [frankenscipy-vgofo]:
//
// Verifies four invariants on trapezoid/simpson over random samples:
//
//   1. Finite output for finite input.
//   2. trapezoid(const, x) ≈ const · (x[-1] − x[0]).
//   3. cumulative_trapezoid last value equals trapezoid total.
//   4. simpson and trapezoid agree on linear data (Newton-Cotes
//      formulas are exact for degree-1 polynomials).
//
// Catches regressions in the per-segment Δx accumulation, the
// even-segment requirement of Simpson's rule, and the running-sum
// invariant of cumulative_trapezoid.

const MAX_N: usize = 32;

#[derive(Debug, Arbitrary)]
struct CompositeInput {
    raw_x: Vec<f64>,
    raw_y: Vec<f64>,
    constant_value: f64,
    linear_slope: f64,
    linear_offset: f64,
}

fn sanitize(value: f64, bound: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-bound, bound)
    } else {
        0.0
    }
}

fn build_strictly_increasing_x(raw: &[f64]) -> Option<Vec<f64>> {
    let mut xs: Vec<f64> = raw
        .iter()
        .take(MAX_N)
        .map(|&v| sanitize(v, 1e3))
        .collect();
    if xs.len() < 2 {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for i in 1..xs.len() {
        if xs[i] <= xs[i - 1] {
            xs[i] = xs[i - 1] + 1e-3;
        }
    }
    Some(xs)
}

fuzz_target!(|input: CompositeInput| {
    let Some(xs) = build_strictly_increasing_x(&input.raw_x) else {
        return;
    };
    let n = xs.len();

    // Property 1: finite output for finite input on a generic y.
    let ys: Vec<f64> = input
        .raw_y
        .iter()
        .take(n)
        .map(|&v| sanitize(v, 1e6))
        .chain(std::iter::repeat(0.0))
        .take(n)
        .collect();
    if ys.len() != n {
        return;
    }

    if let Ok(t) = trapezoid(&ys, &xs)
        && !t.integral.is_finite()
    {
        panic!(
            "trapezoid produced non-finite output {} for n={n}",
            t.integral
        );
    }

    // Property 2: trapezoid of a constant series.
    let c = sanitize(input.constant_value, 1e3);
    let const_ys = vec![c; n];
    if let Ok(t) = trapezoid(&const_ys, &xs) {
        let expected = c * (xs[n - 1] - xs[0]);
        let scale = expected.abs().max(1.0);
        if (t.integral - expected).abs() > 1e-9 * scale {
            panic!(
                "trapezoid(const={c}) = {} != const · (x[n-1] − x[0]) = {expected}",
                t.integral
            );
        }
    }

    // Property 3: cumulative_trapezoid last == trapezoid total.
    if n >= 2
        && let Ok(t) = trapezoid(&const_ys, &xs)
        && let Ok(cum) = cumulative_trapezoid(&const_ys, &xs)
    {
        if cum.is_empty() {
            return;
        }
        let last = *cum.last().unwrap();
        let scale = t.integral.abs().max(1.0);
        if (last - t.integral).abs() > 1e-9 * scale {
            panic!(
                "cumulative_trapezoid.last={last} != trapezoid total={}",
                t.integral
            );
        }
    }

    // Property 4: simpson and trapezoid agree on linear data.
    // Both Newton-Cotes are exact for degree-1 polynomials, so
    // simpson(linear) ≡ trapezoid(linear) up to f64 round-off.
    // Simpson requires an odd number of samples (even number of
    // segments).
    if n >= 3 && n % 2 == 1 {
        let slope = sanitize(input.linear_slope, 100.0);
        let offset = sanitize(input.linear_offset, 1e3);
        let lin_ys: Vec<f64> = xs.iter().map(|&x| slope * x + offset).collect();

        if let (Ok(t), Ok(s)) = (trapezoid(&lin_ys, &xs), simpson(&lin_ys, &xs)) {
            let scale = t.integral.abs().max(s.integral.abs()).max(1.0);
            if (t.integral - s.integral).abs() > 1e-9 * scale {
                panic!(
                    "trapezoid={} ≠ simpson={} on linear data \
                     (slope={slope}, offset={offset}, n={n})",
                    t.integral, s.integral
                );
            }
        }
    }
});
