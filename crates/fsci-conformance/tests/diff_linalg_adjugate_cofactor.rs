#![forbid(unsafe_code)]
//! Live SciPy/numpy differential coverage for fsci_linalg::adjugate and
//! cofactor. Neither has a direct scipy/numpy counterpart, so we use
//! the identity adj(A) = det(A) * inv(A) for nonsingular A, computed
//! via numpy as the oracle. cofactor(A) = adj(A).T.
//!
//! Resolves [frankenscipy-yeu5m]. 1e-9 abs covers the LU/inv roundoff
//! on small probe matrices.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{adjugate, cofactor};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    matrix: Vec<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create adj_cof diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize adj_cof diff log");
    fs::write(path, json).expect("write adj_cof diff log");
}

fn generate_query() -> OracleQuery {
    let matrices: &[(&str, Vec<Vec<f64>>)] = &[
        ("2x2_simple", vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
        ("2x2_signs", vec![vec![3.0, -1.0], vec![2.0, 5.0]]),
        (
            "3x3_dense",
            vec![
                vec![1.0, 2.0, 0.5],
                vec![-1.0, 3.0, 1.0],
                vec![2.0, -1.0, 4.0],
            ],
        ),
        (
            "3x3_tridiag",
            vec![
                vec![4.0, 1.0, 0.0],
                vec![1.0, 4.0, 1.0],
                vec![0.0, 1.0, 4.0],
            ],
        ),
        (
            "4x4_diag",
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 5.0],
            ],
        ),
    ];
    let mut points = Vec::new();
    for (name, m) in matrices {
        for op in ["adjugate", "cofactor"] {
            points.push(PointCase {
                case_id: format!("{op}_{name}"),
                op: op.into(),
                matrix: m.clone(),
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

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
    m = np.array(case["matrix"], dtype=float)
    try:
        d = float(np.linalg.det(m))
        if abs(d) < 1e-12:
            points.append({"case_id": cid, "values": None}); continue
        inv = np.linalg.inv(m)
        adj = d * inv
        if op == "adjugate":
            r = adj
        elif op == "cofactor":
            r = adj.T
        else:
            r = None
        points.append({"case_id": cid, "values": finite_flat_or_none(r) if r is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize adj_cof query");
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
                "failed to spawn python3 for adj_cof oracle: {e}"
            );
            eprintln!("skipping adj_cof oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open adj_cof oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "adj_cof oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping adj_cof oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for adj_cof oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "adj_cof oracle failed: {stderr}"
        );
        eprintln!("skipping adj_cof oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse adj_cof oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let m = match case.op.as_str() {
        "adjugate" => adjugate(&case.matrix),
        "cofactor" => cofactor(&case.matrix),
        _ => return None,
    };
    Some(m.into_iter().flatten().collect())
}

#[test]
fn diff_linalg_adjugate_cofactor() {
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
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
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
        test_id: "diff_linalg_adjugate_cofactor".into(),
        category: "fsci.adjugate/cofactor vs det(A)*inv(A)".into(),
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
                "adj_cof {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "fsci adjugate/cofactor conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
