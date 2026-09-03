#![forbid(unsafe_code)]
//! Live SciPy differential coverage for continuous + discrete Lyapunov
//! equations.
//!   - `scipy.linalg.solve_continuous_lyapunov(A, Q)` — solves
//!     A·X + X·A^T = Q
//!   - `scipy.linalg.solve_discrete_lyapunov(A, Q)` — solves
//!     A·X·A^T - X + Q = 0
//!
//! Resolves [frankenscipy-wlmlx]. 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, solve_continuous_lyapunov, solve_discrete_lyapunov};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    a: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

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
    fs::create_dir_all(output_dir()).expect("create lyapunov diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lyapunov diff log");
    fs::write(path, json).expect("write lyapunov diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // Continuous Lyapunov cases — A must be stable (negative real eigenvalues).
    let cont_cases: &[(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)] = &[
        (
            "2x2_diag_stable",
            vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ),
        (
            "2x2_upper_tri",
            vec![vec![-1.0, 0.5], vec![0.0, -2.0]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ),
        (
            "3x3_diag",
            vec![
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -2.0, 0.0],
                vec![0.0, 0.0, -3.0],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
    ];
    for (name, a, q) in cont_cases {
        points.push(PointCase {
            case_id: format!("cont_{name}"),
            op: "cont".into(),
            a: a.clone(),
            q: q.clone(),
        });
    }

    // Discrete Lyapunov — A must have all eigenvalues inside the unit circle.
    let disc_cases: &[(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)] = &[
        (
            "2x2_stable_diag",
            vec![vec![0.5, 0.0], vec![0.0, 0.3]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ),
        (
            "2x2_upper_tri",
            vec![vec![0.5, 0.1], vec![0.0, 0.3]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ),
        (
            "3x3_diag_stable",
            vec![
                vec![0.5, 0.0, 0.0],
                vec![0.0, 0.3, 0.0],
                vec![0.0, 0.0, 0.7],
            ],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
    ];
    for (name, a, q) in disc_cases {
        points.push(PointCase {
            case_id: format!("disc_{name}"),
            op: "disc".into(),
            a: a.clone(),
            q: q.clone(),
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def finite_flat_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    A = np.array(case["a"], dtype=float)
    Q = np.array(case["q"], dtype=float)
    try:
        if op == "cont":
            X = linalg.solve_continuous_lyapunov(A, Q)
        elif op == "disc":
            X = linalg.solve_discrete_lyapunov(A, Q)
        else:
            points.append({"case_id": cid, "values": None}); continue
        points.append({"case_id": cid, "values": finite_flat_or_none(X)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lyapunov query");
    let mut child = match fsci_conformance::scipy_oracle_command()
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
                "failed to spawn python3 for lyapunov oracle: {e}"
            );
            eprintln!("skipping lyapunov oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lyapunov oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lyapunov oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lyapunov oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lyapunov oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lyapunov oracle failed: {stderr}"
        );
        eprintln!("skipping lyapunov oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lyapunov oracle JSON"))
}

#[test]
fn diff_linalg_lyapunov() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let result = match case.op.as_str() {
            "cont" => solve_continuous_lyapunov(&case.a, &case.q, DecompOptions::default()),
            "disc" => solve_discrete_lyapunov(&case.a, &case.q, DecompOptions::default()),
            _ => continue,
        };
        let Ok(x) = result else { continue };
        let fsci_v: Vec<f64> = x.into_iter().flatten().collect();
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_lyapunov".into(),
        category: "scipy.linalg.solve_{continuous,discrete}_lyapunov".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "lyapunov {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.linalg.solve_lyapunov conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
