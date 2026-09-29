#![forbid(unsafe_code)]
//! Analytic-derivative parity for fsci_opt::jacobian and hessian.
//!
//! Resolves [frankenscipy-advc1]. Tests vs hand-derived ground truth on
//! several closed-form models. 1e-7 abs (jacobian/hessian finite-
//! difference accuracy floor for 8th-order central differences).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{DifferentiateOptions, hessian, jacobian};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-7;

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
    fs::create_dir_all(output_dir()).expect("create jac_hess diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize jac_hess diff log");
    fs::write(path, json).expect("write jac_hess diff log");
}

fn frob_max(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
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

#[test]
fn diff_opt_jacobian_hessian() {
    let opts = DifferentiateOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    // (case id, op, analytic derivative, fsci's derivative or None when the call failed)
    let mut cases: Vec<(&str, &str, Vec<Vec<f64>>, Option<Vec<Vec<f64>>>)> = Vec::new();

    // === Jacobian tests ===
    // Model A: f: R^2 -> R^2, f(x, y) = (x² + y, x*y)
    //   J = [[2x, 1], [y, x]]
    {
        let x = vec![1.5_f64, 2.0_f64];
        let res = jacobian(|v: &[f64]| vec![v[0] * v[0] + v[1], v[0] * v[1]], &x, opts).ok();
        let expected = vec![vec![2.0 * x[0], 1.0], vec![x[1], x[0]]];
        cases.push((
            "jac_2x2_quad_prod",
            "jacobian",
            expected,
            res.map(|res| res.df),
        ));
    }

    // Model B: f: R^3 -> R^2, f = (x + y + z, sin(x) + cos(y) + z²)
    //   J = [[1, 1, 1], [cos(x), -sin(y), 2z]]
    {
        let x = vec![0.5_f64, 1.0_f64, 1.5_f64];
        let res = jacobian(
            |v: &[f64]| vec![v[0] + v[1] + v[2], v[0].sin() + v[1].cos() + v[2] * v[2]],
            &x,
            opts,
        )
        .ok();
        let expected = vec![
            vec![1.0, 1.0, 1.0],
            vec![x[0].cos(), -x[1].sin(), 2.0 * x[2]],
        ];
        cases.push(("jac_2x3_mixed", "jacobian", expected, res.map(|res| res.df)));
    }

    // === Hessian tests ===
    // Model C: f(x, y) = x² + 3xy + 2y² + 5
    //   H = [[2, 3], [3, 4]]
    {
        let x = vec![1.0_f64, 2.0_f64];
        let res = hessian(
            |v: &[f64]| v[0] * v[0] + 3.0 * v[0] * v[1] + 2.0 * v[1] * v[1] + 5.0,
            &x,
            opts,
        )
        .ok();
        let expected = vec![vec![2.0, 3.0], vec![3.0, 4.0]];
        cases.push(("hess_quad_2x2", "hessian", expected, res.map(|res| res.ddf)));
    }

    // Model D: f(x, y, z) = x² + y² + z² + xy
    //   H = [[2, 1, 0], [1, 2, 0], [0, 0, 2]]
    {
        let x = vec![1.0_f64, 1.0, 1.0];
        let res = hessian(
            |v: &[f64]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[0] * v[1],
            &x,
            opts,
        )
        .ok();
        let expected = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        cases.push(("hess_quad_3x3", "hessian", expected, res.map(|res| res.ddf)));
    }

    // Model E: f(x, y) = sin(x*y), at x=1, y=0.5
    //   df/dx = y cos(xy), df/dy = x cos(xy)
    //   d²/dx² = -y² sin(xy), d²/dy² = -x² sin(xy)
    //   d²/dxdy = cos(xy) - xy sin(xy)
    {
        let xv = 1.0_f64;
        let yv = 0.5_f64;
        let x = vec![xv, yv];
        let res = hessian(|v: &[f64]| (v[0] * v[1]).sin(), &x, opts).ok();
        let s = (xv * yv).sin();
        let c = (xv * yv).cos();
        let expected = vec![
            vec![-yv * yv * s, c - xv * yv * s],
            vec![c - xv * yv * s, -xv * xv * s],
        ];
        cases.push(("hess_sin_prod", "hessian", expected, res.map(|res| res.ddf)));
    }

    let mut ledger = CompareLedger::new("diff_opt_jacobian_hessian", &["jacobian", "hessian"]);
    for (case_id, op, expected, fsci) in &cases {
        // Every entry goes through the ledger, flattened, so a NaN derivative cannot vanish in
        // `frob_max`'s max fold; `frob_max` still rejects a same-size matrix of the wrong shape.
        let expected_flat = expected.concat();
        let fsci_flat = fsci.as_ref().map(|m| m.concat());
        let (Some(_), Some(fsci)) = (
            ledger.slices(
                op,
                case_id,
                Some(expected_flat.as_slice()),
                fsci_flat.as_deref(),
            ),
            fsci,
        ) else {
            continue;
        };
        let d = frob_max(fsci, expected);
        max_overall = max_overall.max(d);
        ledger.compared(op, case_id, d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: (*case_id).into(),
            op: (*op).into(),
            abs_diff: d,
            pass: d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_jacobian_hessian".into(),
        category: "fsci_opt::jacobian + hessian vs analytic".into(),
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
        "jacobian_hessian conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(cases.iter().filter(|c| c.1 == "jacobian").count());
}
