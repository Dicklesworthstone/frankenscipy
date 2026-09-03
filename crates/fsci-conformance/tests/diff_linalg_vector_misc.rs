#![forbid(unsafe_code)]
//! Live numpy/scipy parity for fsci_linalg vector & misc utilities:
//! vector_norm, vdot, vnorm, hadamard_product, rot2d, antidiag.
//!
//! Resolves [frankenscipy-ceof2]. All deterministic; 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{antidiag, hadamard_product, rot2d, vdot, vector_norm, vnorm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    /// vector_norm / vnorm / vdot
    v: Vec<f64>,
    w: Vec<f64>, // for vdot
    ord: f64,
    /// hadamard_product
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    /// rot2d
    theta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Scalar wrapped as len-1 vec, or flattened matrix.
    values: Option<Vec<f64>>,
    #[allow(dead_code)]
    out_rows: Option<usize>,
    #[allow(dead_code)]
    out_cols: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create vector_misc diff dir");
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

fn flatten_matrix(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flat_map(|r| r.iter().copied()).collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let v1 = vec![3.0_f64, -4.0, 0.0];
    let v2 = vec![1.0_f64, 2.0, -2.0, 4.0, -1.0];
    let v3 = vec![0.5_f64; 8];
    let w1 = vec![1.0_f64, -1.0, 2.0];
    let w2 = vec![0.5_f64, 1.5, -2.0, 0.25, -1.0];

    // vector_norm probes: a few orders
    for (label, v) in [("v1", &v1), ("v2", &v2), ("v3", &v3)] {
        // 0, 1, 2, 3, inf, -inf
        for ord in [0.0_f64, 1.0, 2.0, 3.0, f64::INFINITY, f64::NEG_INFINITY] {
            let ord_label = if ord.is_infinite() {
                if ord > 0.0 {
                    "inf".to_string()
                } else {
                    "neginf".to_string()
                }
            } else {
                format!("{ord}")
            };
            points.push(Case {
                case_id: format!("vector_norm_{label}_ord{ord_label}"),
                op: "vector_norm".into(),
                v: v.clone(),
                w: vec![],
                ord,
                a: vec![],
                b: vec![],
                theta: 0.0,
            });
        }
    }

    // vnorm (Euclidean)
    for (label, v) in [("v1", &v1), ("v2", &v2), ("v3", &v3)] {
        points.push(Case {
            case_id: format!("vnorm_{label}"),
            op: "vnorm".into(),
            v: v.clone(),
            w: vec![],
            ord: 0.0,
            a: vec![],
            b: vec![],
            theta: 0.0,
        });
    }

    // vdot
    for (label, v, w) in [("v1w1", &v1, &w1), ("v2w2", &v2, &w2)] {
        points.push(Case {
            case_id: format!("vdot_{label}"),
            op: "vdot".into(),
            v: v.clone(),
            w: w.clone(),
            ord: 0.0,
            a: vec![],
            b: vec![],
            theta: 0.0,
        });
    }

    // hadamard_product
    let m_a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let m_b = vec![vec![-1.0, 0.5, 2.0], vec![3.0, -0.5, 1.5]];
    let m_a2 = vec![vec![1.0_f64; 4]; 3];
    let m_b2: Vec<Vec<f64>> = (0..3)
        .map(|i| (0..4).map(|j| (i + j) as f64 * 0.3 - 0.7).collect())
        .collect();

    for (label, a, b) in [("2x3", &m_a, &m_b), ("3x4", &m_a2, &m_b2)] {
        points.push(Case {
            case_id: format!("hadamard_{label}"),
            op: "hadamard".into(),
            v: vec![],
            w: vec![],
            ord: 0.0,
            a: a.clone(),
            b: b.clone(),
            theta: 0.0,
        });
    }

    // rot2d
    for theta in [
        0.0_f64,
        std::f64::consts::FRAC_PI_6,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        -std::f64::consts::FRAC_PI_3,
    ] {
        points.push(Case {
            case_id: format!("rot2d_t{theta}"),
            op: "rot2d".into(),
            v: vec![],
            w: vec![],
            ord: 0.0,
            a: vec![],
            b: vec![],
            theta,
        });
    }

    // antidiag
    for v in [
        vec![1.0_f64, 2.0, 3.0],
        vec![5.0_f64, -2.0, 3.5, 0.0, 1.0],
        vec![7.0_f64; 4],
    ] {
        points.push(Case {
            case_id: format!("antidiag_n{}", v.len()),
            op: "antidiag".into(),
            v,
            w: vec![],
            ord: 0.0,
            a: vec![],
            b: vec![],
            theta: 0.0,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "vector_norm":
            v = np.array(case["v"], dtype=float)
            ord_v = case["ord"]
            if not math.isfinite(ord_v):
                # +/- infinity sent as JSON number — read sign
                pass
            v_ord = ord_v
            res = float(np.linalg.norm(v, ord=v_ord))
            points.append({"case_id": cid, "values": [res], "out_rows": None, "out_cols": None})
        elif op == "vnorm":
            v = np.array(case["v"], dtype=float)
            res = float(np.linalg.norm(v))
            points.append({"case_id": cid, "values": [res], "out_rows": None, "out_cols": None})
        elif op == "vdot":
            v = np.array(case["v"], dtype=float)
            w = np.array(case["w"], dtype=float)
            res = float(np.dot(v, w))
            points.append({"case_id": cid, "values": [res], "out_rows": None, "out_cols": None})
        elif op == "hadamard":
            a = np.array(case["a"], dtype=float)
            b = np.array(case["b"], dtype=float)
            r = a * b
            flat = [float(v) for v in r.flatten().tolist()]
            points.append({"case_id": cid, "values": flat, "out_rows": int(r.shape[0]), "out_cols": int(r.shape[1])})
        elif op == "rot2d":
            t = float(case["theta"])
            r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
            flat = [float(v) for v in r.flatten().tolist()]
            points.append({"case_id": cid, "values": flat, "out_rows": 2, "out_cols": 2})
        elif op == "antidiag":
            v = np.array(case["v"], dtype=float)
            r = np.fliplr(np.diag(v))
            flat = [float(v) for v in r.flatten().tolist()]
            points.append({"case_id": cid, "values": flat, "out_rows": int(r.shape[0]), "out_cols": int(r.shape[1])})
        else:
            points.append({"case_id": cid, "values": None, "out_rows": None, "out_cols": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None, "out_rows": None, "out_cols": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for vector_misc oracle: {e}"
            );
            eprintln!("skipping vector_misc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "vector_misc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping vector_misc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for vector_misc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "vector_misc oracle failed: {stderr}"
        );
        eprintln!("skipping vector_misc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse vector_misc oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_linalg_vector_misc() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let actual: Vec<f64> = match case.op.as_str() {
            "vector_norm" => vec![vector_norm(&case.v, case.ord)],
            "vnorm" => vec![vnorm(&case.v)],
            "vdot" => vec![vdot(&case.v, &case.w)],
            "hadamard" => flatten_matrix(&hadamard_product(&case.a, &case.b)),
            "rot2d" => flatten_matrix(&rot2d(case.theta)),
            "antidiag" => flatten_matrix(&antidiag(&case.v)),
            _ => continue,
        };
        let abs_d = vec_max_diff(&actual, expected);
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
        test_id: "diff_linalg_vector_misc".into(),
        category: "fsci_linalg vector & misc utilities vs numpy".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "vector_misc conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
