#![forbid(unsafe_code)]
//! Analytic-derivative parity for fsci_opt::numerical_gradient,
//! numerical_jacobian, numerical_hessian (single-step finite differences).
//!
//! Resolves [frankenscipy-pi79i]. 1e-5 abs (one-shot FD is less accurate
//! than the adaptive variants tested in diff_opt_jacobian_hessian).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{numerical_gradient, numerical_hessian, numerical_jacobian};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-4;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create num_grad diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize num_grad diff log");
    fs::write(path, json).expect("write num_grad diff log");
}

fn frob_max_vec(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn frob_max_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    if b.len() != m || b.first().map_or(0, Vec::len) != n {
        return f64::INFINITY;
    }
    let mut max = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

/// Row-major flattening, so the ledger can see a NaN that `frob_max_mat`'s `f64::max` would drop
/// and an element-count mismatch (e.g. the empty Jacobian fsci returns on bad input).
fn flat(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

#[test]
fn diff_opt_numerical_grad_jac_hess() {
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let eps = 1e-5_f64;
    let mut ledger = CompareLedger::new(
        "diff_opt_numerical_grad_jac_hess",
        &[
            "numerical_gradient",
            "numerical_jacobian",
            "numerical_hessian",
        ],
    );

    // === numerical_gradient ===
    // f(x, y) = x² + 3xy + 2y² + 5 at (1, 2)
    // ∇f = (2x + 3y, 3x + 4y) = (8, 11)
    {
        let x = vec![1.0_f64, 2.0_f64];
        let g = numerical_gradient(
            |v: &[f64]| v[0] * v[0] + 3.0 * v[0] * v[1] + 2.0 * v[1] * v[1] + 5.0,
            &x,
            eps,
        );
        let expected = vec![8.0_f64, 11.0_f64];
        if let Some((expected, g)) = ledger.slices(
            "numerical_gradient",
            "grad_quad_2d",
            Some(expected.as_slice()),
            Some(g.as_slice()),
        ) {
            let d = frob_max_vec(g, expected);
            max_overall = max_overall.max(d);
            ledger.compared("numerical_gradient", "grad_quad_2d", d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: "grad_quad_2d".into(),
                op: "numerical_gradient".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }
    // f(x, y, z) = x² + y² + z² at (1, 2, 3): ∇f = (2, 4, 6)
    {
        let x = vec![1.0_f64, 2.0, 3.0];
        let g = numerical_gradient(|v: &[f64]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2], &x, eps);
        let expected = vec![2.0_f64, 4.0, 6.0];
        if let Some((expected, g)) = ledger.slices(
            "numerical_gradient",
            "grad_sum_sq_3d",
            Some(expected.as_slice()),
            Some(g.as_slice()),
        ) {
            let d = frob_max_vec(g, expected);
            max_overall = max_overall.max(d);
            ledger.compared("numerical_gradient", "grad_sum_sq_3d", d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: "grad_sum_sq_3d".into(),
                op: "numerical_gradient".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    // === numerical_jacobian ===
    // f: R^2 -> R^2, f(x, y) = (x² + y, x*y); at (1.5, 2): J = [[3, 1], [2, 1.5]]
    {
        let x = vec![1.5_f64, 2.0_f64];
        let j = numerical_jacobian(|v: &[f64]| vec![v[0] * v[0] + v[1], v[0] * v[1]], &x, eps);
        let expected = vec![vec![2.0 * x[0], 1.0], vec![x[1], x[0]]];
        let (expected_flat, j_flat) = (flat(&expected), flat(&j));
        if ledger
            .slices(
                "numerical_jacobian",
                "jac_2x2_quad_prod",
                Some(expected_flat.as_slice()),
                Some(j_flat.as_slice()),
            )
            .is_some()
        {
            let d = frob_max_mat(&j, &expected);
            max_overall = max_overall.max(d);
            ledger.compared("numerical_jacobian", "jac_2x2_quad_prod", d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: "jac_2x2_quad_prod".into(),
                op: "numerical_jacobian".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    // === numerical_hessian ===
    // f(x, y) = x² + 3xy + 2y² + 5: H = [[2, 3], [3, 4]]
    {
        let x = vec![1.0_f64, 2.0_f64];
        let h = numerical_hessian(
            |v: &[f64]| v[0] * v[0] + 3.0 * v[0] * v[1] + 2.0 * v[1] * v[1] + 5.0,
            &x,
            eps,
        );
        let expected = vec![vec![2.0, 3.0], vec![3.0, 4.0]];
        let (expected_flat, h_flat) = (flat(&expected), flat(&h));
        if ledger
            .slices(
                "numerical_hessian",
                "hess_quad_2d",
                Some(expected_flat.as_slice()),
                Some(h_flat.as_slice()),
            )
            .is_some()
        {
            let d = frob_max_mat(&h, &expected);
            max_overall = max_overall.max(d);
            ledger.compared("numerical_hessian", "hess_quad_2d", d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: "hess_quad_2d".into(),
                op: "numerical_hessian".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }
    // f(x, y, z) = x² + y² + z² + xy: H = [[2, 1, 0], [1, 2, 0], [0, 0, 2]]
    {
        let x = vec![1.0_f64, 1.0, 1.0];
        let h = numerical_hessian(
            |v: &[f64]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[0] * v[1],
            &x,
            eps,
        );
        let expected = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let (expected_flat, h_flat) = (flat(&expected), flat(&h));
        if ledger
            .slices(
                "numerical_hessian",
                "hess_quad_3d",
                Some(expected_flat.as_slice()),
                Some(h_flat.as_slice()),
            )
            .is_some()
        {
            let d = frob_max_mat(&h, &expected);
            max_overall = max_overall.max(d);
            ledger.compared("numerical_hessian", "hess_quad_3d", d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: "hess_quad_3d".into(),
                op: "numerical_hessian".into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_numerical_grad_jac_hess".into(),
        category: "fsci_opt numerical_gradient/jacobian/hessian vs analytic".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "numerical_grad_jac_hess conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // The smallest arm: numerical_jacobian has one fixture, gradient and hessian two each.
    ledger.finish(1);
}
