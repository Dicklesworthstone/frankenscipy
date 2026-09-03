#![forbid(unsafe_code)]
//! Property-based + scipy parity for fsci_opt::lbfgsb.
//!
//! Resolves [frankenscipy-pcc1j]. L-BFGS-B is bound-constrained
//! quasi-Newton optimization. Verifies:
//!   * Unconstrained quadratic f(x) = (x-target)² finds target
//!   * Bounded quadratic clips to the bound when target is outside
//!   * Smooth multivariate problem (sum of squares) converges to origin
//!   * scipy parity on a Rosenbrock-style problem (converged solutions
//!     match closely)

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::types::Bound;
use fsci_opt::{MinimizeOptions, lbfgsb};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create lbfgsb diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct ScipyMinimum {
    converged: bool,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
}

fn scipy_oracle_rosen_lbfgsb_or_skip() -> Option<ScipyMinimum> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

res = minimize(rosen, [-1.2, 1.0], method='L-BFGS-B', tol=1e-8,
               options={'maxiter': 2000})
out = {
    "converged": bool(res.success),
    "x": [float(v) for v in res.x] if res.success else None,
    "fun": float(res.fun) if res.success else None,
}
print(json.dumps(out))
"#;
    let mut child = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "python3 spawn failed: {e}"
            );
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if stdin.write_all(b"").is_err() {
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_opt_lbfgsb_minimize() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    let opts = MinimizeOptions {
        tol: Some(1.0e-10),
        maxiter: Some(2000),
        ..MinimizeOptions::default()
    };

    // === 1. Unconstrained scalar quadratic: minimize (x - 3)² → x = 3 ===
    {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);
        let r = lbfgsb(&f, &[0.0], opts, None).expect("scalar quadratic");
        check(
            "unconstrained_scalar_quadratic_finds_3",
            (r.x[0] - 3.0).abs() < ABS_TOL,
            format!("x={:?}", r.x),
        );
    }

    // === 2. Bounded scalar: minimize (x - 5)² with x ∈ [0, 2] → x = 2 ===
    {
        let f = |x: &[f64]| (x[0] - 5.0).powi(2);
        let bounds: [Bound; 1] = [(Some(0.0), Some(2.0))];
        let r = lbfgsb(&f, &[1.0], opts, Some(&bounds)).expect("bounded scalar");
        check(
            "bounded_clipped_to_upper",
            (r.x[0] - 2.0).abs() < ABS_TOL,
            format!("x={:?}", r.x),
        );
    }

    // === 3. Bounded scalar: minimize (x + 5)² with x ∈ [0, 2] → x = 0 ===
    {
        let f = |x: &[f64]| (x[0] + 5.0).powi(2);
        let bounds: [Bound; 1] = [(Some(0.0), Some(2.0))];
        let r = lbfgsb(&f, &[1.0], opts, Some(&bounds)).expect("bounded scalar");
        check(
            "bounded_clipped_to_lower",
            (r.x[0]).abs() < ABS_TOL,
            format!("x={:?}", r.x),
        );
    }

    // === 4. Multivariate sum of squares: minimize Σ x_i² → x = 0 ===
    {
        let f = |x: &[f64]| x.iter().map(|v| v.powi(2)).sum::<f64>();
        let r = lbfgsb(&f, &[1.0, -2.0, 3.0, -0.5], opts, None).expect("sum sq");
        let max_abs = r.x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        check(
            "multivariate_sum_of_squares_to_origin",
            max_abs < ABS_TOL,
            format!("x={:?}", r.x),
        );
    }

    // === 5. scipy parity: Rosenbrock at standard init point ===
    {
        let rosen = |x: &[f64]| -> f64 {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum()
        };
        let fsci_r = lbfgsb(&rosen, &[-1.2, 1.0], opts, None).expect("rosen");
        let fsci_close = (fsci_r.x[0] - 1.0).abs() < 1.0e-3 && (fsci_r.x[1] - 1.0).abs() < 1.0e-3;
        check(
            "rosenbrock_fsci_converges_near_one",
            fsci_close,
            format!("x={:?} fun={:?}", fsci_r.x, fsci_r.fun),
        );

        if let Some(scipy) = scipy_oracle_rosen_lbfgsb_or_skip()
            && scipy.converged
            && let Some(scipy_x) = scipy.x.as_ref()
        {
            let close_to_scipy = (fsci_r.x[0] - scipy_x[0]).abs() < 1.0e-3
                && (fsci_r.x[1] - scipy_x[1]).abs() < 1.0e-3;
            check(
                "rosenbrock_close_to_scipy",
                close_to_scipy,
                format!("fsci={:?} scipy={:?}", fsci_r.x, scipy_x),
            );
        }
    }

    // === 6. Bounds-with-x0-outside: x0 = -1.0 outside bounds [0, 5] should be projected ===
    {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2);
        let bounds: [Bound; 1] = [(Some(0.0), Some(5.0))];
        let r = lbfgsb(&f, &[-1.0], opts, Some(&bounds)).expect("projected x0");
        check(
            "x0_outside_bounds_projected_then_solved",
            (r.x[0] - 2.0).abs() < ABS_TOL,
            format!("x={:?}", r.x),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_lbfgsb_minimize".into(),
        category: "fsci_opt::lbfgsb (L-BFGS-B) coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("lbfgsb mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(all_pass, "lbfgsb coverage failed: {} cases", diffs.len());
}
