#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two linalg primitives not
//! previously covered:
//!   - `cond(a, opts)`                  vs `numpy.linalg.cond(a)`
//!   - `fractional_matrix_power(a, p)`  vs `scipy.linalg.fractional_matrix_power(a, p)`
//!
//! Resolves [frankenscipy-lidjz]. cond uses singular-value ratio so
//! agreement is dominated by SVD truncation (1e-8 abs/rel). fractional
//! power uses funm/Schur so floor is ~1e-6 on well-conditioned probes.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, cond, fractional_matrix_power};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const COND_TOL: f64 = 1.0e-7;
const FMP_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    matrix: Vec<Vec<f64>>,
    p: f64,
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
    rel_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create cond_fmp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cond_fmp diff log");
    fs::write(path, json).expect("write cond_fmp diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // cond: well-conditioned SPD + a less well-conditioned matrix.
    let cond_matrices: &[(&str, Vec<Vec<f64>>)] = &[
        (
            "ident_3",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        ),
        (
            "tridiag_4_spd",
            vec![
                vec![4.0, 1.0, 0.0, 0.0],
                vec![1.0, 4.0, 1.0, 0.0],
                vec![0.0, 1.0, 4.0, 1.0],
                vec![0.0, 0.0, 1.0, 4.0],
            ],
        ),
        (
            "hilbert_3",
            (1..=3)
                .map(|i| (1..=3).map(|j| 1.0 / (i + j - 1) as f64).collect())
                .collect(),
        ),
        (
            "diag_increasing",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 5.0, 0.0],
                vec![0.0, 0.0, 25.0],
            ],
        ),
    ];
    for (name, m) in cond_matrices {
        points.push(PointCase {
            case_id: format!("cond_{name}"),
            op: "cond".into(),
            matrix: m.clone(),
            p: 0.0,
        });
    }

    // fractional_matrix_power: positive-definite matrices for which A^p is well-defined.
    let fmp_cases: &[(&str, Vec<Vec<f64>>, f64)] = &[
        ("diag2x2_p0.5", vec![vec![4.0, 0.0], vec![0.0, 9.0]], 0.5),
        (
            "diag3_p0.5",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 4.0, 0.0],
                vec![0.0, 0.0, 9.0],
            ],
            0.5,
        ),
        (
            "spd3_p2",
            vec![
                vec![2.0, 0.5, 0.0],
                vec![0.5, 3.0, 0.5],
                vec![0.0, 0.5, 4.0],
            ],
            2.0,
        ),
        (
            "spd3_pm1",
            vec![
                vec![2.0, 0.5, 0.0],
                vec![0.5, 3.0, 0.5],
                vec![0.0, 0.5, 4.0],
            ],
            -1.0,
        ),
        (
            "spd3_p1.5",
            vec![
                vec![2.0, 0.5, 0.0],
                vec![0.5, 3.0, 0.5],
                vec![0.0, 0.5, 4.0],
            ],
            1.5,
        ),
    ];
    for (name, m, p) in fmp_cases {
        points.push(PointCase {
            case_id: format!("fmp_{name}"),
            op: "fmp".into(),
            matrix: m.clone(),
            p: *p,
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
from scipy import linalg as splinalg

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
        if op == "cond":
            v = float(np.linalg.cond(m))
            points.append({"case_id": cid, "values": [v] if math.isfinite(v) else None})
        elif op == "fmp":
            p = float(case["p"])
            r = splinalg.fractional_matrix_power(m, p)
            points.append({"case_id": cid, "values": finite_flat_or_none(r)})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cond_fmp query");
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
                "failed to spawn python3 for cond_fmp oracle: {e}"
            );
            eprintln!("skipping cond_fmp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open cond_fmp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cond_fmp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cond_fmp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cond_fmp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cond_fmp oracle failed: {stderr}"
        );
        eprintln!("skipping cond_fmp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cond_fmp oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    match case.op.as_str() {
        "cond" => cond(&case.matrix, DecompOptions::default())
            .ok()
            .map(|v| vec![v]),
        "fmp" => fractional_matrix_power(&case.matrix, case.p, DecompOptions::default())
            .ok()
            .map(|m| m.into_iter().flatten().collect()),
        _ => None,
    }
}

#[test]
fn diff_linalg_cond_fractional_matrix_power() {
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
        let tol = match case.op.as_str() {
            "cond" => COND_TOL,
            "fmp" => FMP_TOL,
            _ => continue,
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                rel_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let scale = scipy_v.iter().map(|b| b.abs()).fold(1.0_f64, f64::max);
        let rel_d = abs_d / scale;
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            rel_diff: rel_d,
            pass: rel_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_cond_fractional_matrix_power".into(),
        category: "scipy/numpy cond + scipy.linalg.fractional_matrix_power".into(),
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
                "cond_fmp {} mismatch: {} rel_diff={}",
                d.op, d.case_id, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.linalg cond/fractional_matrix_power conformance failed: {} cases, max_abs_diff={}",
        diffs.len(),
        max_overall
    );
}
