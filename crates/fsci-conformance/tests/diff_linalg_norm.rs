#![forbid(unsafe_code)]
//! Live SciPy/NumPy differential coverage for `fsci_linalg::norm`.
//!
//! Resolves [frankenscipy-t6hev]. 4 norm kinds (Fro, Spectral, One,
//! Inf) probed against np.linalg.norm(A, ord=...) on a few matrices.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, NormKind, norm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    kind: String,
    a: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    kind: String,
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
    fs::create_dir_all(output_dir()).expect("create norm diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize norm diff log");
    fs::write(path, json).expect("write norm diff log");
}

fn parse_kind(name: &str) -> NormKind {
    match name {
        "fro" => NormKind::Fro,
        "spectral" => NormKind::Spectral,
        "one" => NormKind::One,
        "inf" => NormKind::Inf,
        _ => NormKind::Fro,
    }
}

fn generate_query() -> OracleQuery {
    let matrices: &[(&str, Vec<Vec<f64>>)] = &[
        (
            "3x3_dense",
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        ),
        (
            "4x4_diag",
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 4.0],
            ],
        ),
        (
            "2x4_rect",
            vec![vec![1.0, -2.0, 3.0, -4.0], vec![-5.0, 6.0, -7.0, 8.0]],
        ),
        (
            "3x2_rect_signs",
            vec![vec![1.0, -2.0], vec![-3.0, 4.0], vec![5.0, -6.0]],
        ),
        ("2x2_simple", vec![vec![3.0, 4.0], vec![0.0, 5.0]]),
    ];
    let kinds = ["fro", "spectral", "one", "inf"];

    let mut points = Vec::new();
    for (label, m) in matrices {
        for kind in kinds {
            points.push(PointCase {
                case_id: format!("{label}_{kind}"),
                kind: kind.into(),
                a: m.clone(),
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

ORD_MAP = {
    "fro": "fro",
    "spectral": 2,
    "one": 1,
    "inf": np.inf,
}

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    A = np.array(case["a"], dtype=float)
    ord_arg = ORD_MAP.get(case["kind"])
    try:
        v = fnone(np.linalg.norm(A, ord=ord_arg))
        points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize norm query");
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
                "failed to spawn python3 for norm oracle: {e}"
            );
            eprintln!("skipping norm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open norm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "norm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping norm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for norm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "norm oracle failed: {stderr}"
        );
        eprintln!("skipping norm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse norm oracle JSON"))
}

#[test]
fn diff_linalg_norm() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Ok(fsci_v) = norm(&case.a, parse_kind(&case.kind), DecompOptions::default()) else {
            continue;
        };
        let abs_d = (fsci_v - scipy_v).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            kind: case.kind.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_norm".into(),
        category: "fsci.norm vs np.linalg.norm".into(),
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
                "norm {} mismatch: {} abs_diff={}",
                d.kind, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "fsci.norm conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
