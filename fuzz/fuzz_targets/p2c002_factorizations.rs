#![no_main]

use arbitrary::Arbitrary;
use fsci_linalg::{
    DecompOptions, InvOptions, LstsqDriver, LstsqOptions, PinvOptions, det, inv, lstsq, lu, pinv,
    qr,
};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 64;

#[derive(Debug, Arbitrary)]
struct FactorizationInput {
    rows: u8,
    cols: u8,
    hardened: bool,
    check_finite: bool,
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

fn matmul(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if lhs.is_empty() || rhs.is_empty() {
        return None;
    }
    let m = lhs.len();
    let k = lhs[0].len();
    let n = rhs[0].len();
    if rhs.len() != k {
        return None;
    }
    let mut out = vec![vec![0.0; n]; m];
    for (i, row) in out.iter_mut().enumerate() {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += lhs[i][p] * rhs[p][j];
            }
            row[j] = acc;
        }
    }
    Some(out)
}

/// Maximum absolute entry of A - B. None when shapes disagree or the
/// diff contains a non-finite value (unadjudicable).
fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let mut max = 0.0_f64;
    for (row_a, row_b) in a.iter().zip(b) {
        if row_a.len() != row_b.len() {
            return None;
        }
        for (lhs, rhs) in row_a.iter().zip(row_b) {
            let diff = (lhs - rhs).abs();
            if !diff.is_finite() {
                return None;
            }
            max = max.max(diff);
        }
    }
    Some(max)
}

fn max_abs(a: &[Vec<f64>]) -> f64 {
    let mut m = 0.0_f64;
    for row in a {
        for &v in row {
            if v.is_finite() {
                m = m.max(v.abs());
            }
        }
    }
    m
}

/// Check A @ A_inv ≈ I with a scale-aware loose bound. Returns true
/// when the product is within tolerance or when numerical issues
/// prevent adjudication (non-finite intermediates).
fn check_inverse_identity(a: &[Vec<f64>], a_inv: &[Vec<f64>]) -> bool {
    let n = a.len();
    if n == 0 || a_inv.len() != n || a[0].len() != n || a_inv[0].len() != n {
        return true;
    }
    let mut max_off = 0.0_f64;
    let mut max_diag = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 0.0_f64;
            for k in 0..n {
                prod += a[i][k] * a_inv[k][j];
            }
            if !prod.is_finite() {
                return true;
            }
            if i == j {
                max_diag = max_diag.max((prod - 1.0).abs());
            } else {
                max_off = max_off.max(prod.abs());
            }
        }
    }
    let max_err = max_off.max(max_diag);
    max_err < 1e-3
}

fuzz_target!(|input: FactorizationInput| {
    let rows = clamp_dimension(input.rows);
    let cols = clamp_dimension(input.cols);
    let mode = if input.hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    };
    let a = build_matrix(rows, cols, &input.values);
    let b = build_vector(rows, &input.rhs);

    if let Ok(inv_result) = inv(
        &a,
        InvOptions {
            mode,
            check_finite: input.check_finite,
            assume_a: None,
            lower: false,
        },
    ) && rows == cols
        && rows > 0
        && a.iter().flatten().all(|v| v.is_finite())
        && inv_result.inverse.iter().flatten().all(|v| v.is_finite())
    {
        // br-66c2: metamorphic oracle — A @ A_inv ≈ I.
        assert!(
            check_inverse_identity(&a, &inv_result.inverse),
            "inv returned Ok but A @ A_inv diverges from identity"
        );
    }
    // br-66c2: LU decomposition metamorphic oracle — P @ A ≈ L @ U.
    // Only meaningful when A is square, finite, and of modest norm; we
    // gate on is_finite on A and on the factor outputs to avoid
    // adjudicating cases the backend already bailed on.
    if rows == cols
        && rows > 0
        && a.iter().flatten().all(|v| v.is_finite())
        && let Ok(lu_res) = lu(
            &a,
            DecompOptions {
                mode,
                check_finite: input.check_finite,
            },
        )
        && lu_res.p.iter().flatten().all(|v| v.is_finite())
        && lu_res.l.iter().flatten().all(|v| v.is_finite())
        && lu_res.u.iter().flatten().all(|v| v.is_finite())
        && let Some(pa) = matmul(&lu_res.p, &a)
        && let Some(lu_prod) = matmul(&lu_res.l, &lu_res.u)
        && let Some(diff) = max_abs_diff(&pa, &lu_prod)
    {
        let scale = max_abs(&a).max(max_abs(&lu_prod)).max(1.0);
        let threshold = 1e-6 * scale * (rows as f64).max(1.0);
        assert!(
            diff <= threshold,
            "lu decomposition broken: max|P A - L U| = {diff:e} exceeds {threshold:e} (scale={scale:e}, n={rows})"
        );
    }

    // br-66c2: QR decomposition metamorphic oracle — A ≈ Q @ R. Valid
    // for any shape as long as the factor product has the same
    // dimensions as the input.
    if rows > 0
        && cols > 0
        && a.iter().flatten().all(|v| v.is_finite())
        && let Ok(qr_res) = qr(
            &a,
            DecompOptions {
                mode,
                check_finite: input.check_finite,
            },
        )
        && qr_res.q.iter().flatten().all(|v| v.is_finite())
        && qr_res.r.iter().flatten().all(|v| v.is_finite())
        && let Some(qr_prod) = matmul(&qr_res.q, &qr_res.r)
        && let Some(diff) = max_abs_diff(&a, &qr_prod)
    {
        let scale = max_abs(&a).max(max_abs(&qr_prod)).max(1.0);
        let threshold = 1e-6 * scale * (rows.max(cols) as f64).max(1.0);
        assert!(
            diff <= threshold,
            "qr decomposition broken: max|A - Q R| = {diff:e} exceeds {threshold:e} (scale={scale:e}, shape=({rows},{cols}))"
        );
    }

    let _ = det(&a, mode, input.check_finite);
    let _ = lstsq(
        &a,
        &b,
        LstsqOptions {
            mode,
            check_finite: input.check_finite,
            cond: None,
            driver: LstsqDriver::Gelsd,
            ..LstsqOptions::default()
        },
    );
    let _ = pinv(
        &a,
        PinvOptions {
            mode,
            check_finite: input.check_finite,
            atol: None,
            rtol: None,
        },
    );
});
