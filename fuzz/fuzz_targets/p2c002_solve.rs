#![no_main]

use arbitrary::Arbitrary;
use fsci_linalg::{SolveOptions, solve};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 64;

#[derive(Debug, Arbitrary)]
struct SolveInput {
    rows: u8,
    cols: u8,
    hardened: bool,
    check_finite: bool,
    lower: bool,
    transposed: bool,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

fn build_matrix(rows: usize, cols: usize, values: &[f64]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; cols]; rows];
    for (idx, value) in values.iter().copied().take(rows * cols).enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        matrix[r][c] = value;
    }
    matrix
}

fn build_vector(len: usize, values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (idx, value) in values.iter().copied().take(len).enumerate() {
        out[idx] = value;
    }
    out
}

fn clamp_dimension(raw: u8) -> usize {
    usize::from(raw) % (MAX_DIM + 1)
}

/// Residual oracle: for a successful solve, ||Ax - b||_inf must be
/// small relative to ||A||_inf * ||x||_inf + ||b||_inf (scale-aware).
/// This is strength-4 (metamorphic) — see br-66c2. Failures here
/// indicate solve returned an `Ok` with a garbage x.
fn check_residual(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> bool {
    if a.is_empty() || x.is_empty() || b.is_empty() {
        return true;
    }
    let n = a.len();
    let mut resid = 0.0_f64;
    let mut ax_scale = 0.0_f64;
    let mut b_scale = 0.0_f64;
    for i in 0..n {
        if a[i].len() != x.len() {
            return true; // shape handled elsewhere
        }
        let mut ax_i = 0.0_f64;
        for (j, &xj) in x.iter().enumerate() {
            ax_i += a[i][j] * xj;
        }
        if !ax_i.is_finite() {
            return true; // fp overflow — oracle cannot adjudicate
        }
        let bi = b[i];
        ax_scale = ax_scale.max(ax_i.abs());
        b_scale = b_scale.max(bi.abs());
        resid = resid.max((ax_i - bi).abs());
    }
    let scale = ax_scale.max(b_scale).max(1.0);
    // Loose tolerance: a conforming solve usually achieves < 1e-8
    // relative residual; we guard against orders-of-magnitude failures
    // rather than tight accuracy. Tight oracle lives in conformance.
    resid / scale < 1e-4
}

fuzz_target!(|input: SolveInput| {
    let rows = clamp_dimension(input.rows);
    let cols = clamp_dimension(input.cols);
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let a = build_matrix(rows, cols, &input.values);
    let b = build_vector(rows, &input.rhs);
    let options = SolveOptions {
        mode,
        check_finite: input.check_finite,
        assume_a: None,
        lower: input.lower,
        transposed: input.transposed,
        ..SolveOptions::default()
    };
    if let Ok(result) = solve(&a, &b, options)
        && rows == cols
        && rows > 0
        && !input.transposed
        && result.x.iter().all(|v| v.is_finite())
        && a.iter().flatten().all(|v| v.is_finite())
        && b.iter().all(|v| v.is_finite())
    {
        assert!(
            check_residual(&a, &result.x, &b),
            "solve returned Ok but ||Ax - b|| exceeded sanity bound"
        );
    }
});
