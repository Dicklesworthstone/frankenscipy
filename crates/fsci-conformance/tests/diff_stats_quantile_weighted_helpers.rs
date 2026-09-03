#![forbid(unsafe_code)]
//! Live numpy parity for fsci_stats::{quantile, weighted_mean, weighted_var}.
//!
//! Resolves [frankenscipy-b3rxr]. fsci's quantile uses linear
//! interpolation between order statistics (same as np.quantile(method=
//! "linear")). weighted_mean matches np.average(values, weights=...).
//! weighted_var is the population variance (weights/sum-of-weights),
//! computable directly from numpy.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{quantile, weighted_mean, weighted_var};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-12;
const ABS_TOL: f64 = 1.0e-14;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "quantile" | "weighted_mean" | "weighted_var"
    op: String,
    data: Vec<f64>,
    /// Quantile probabilities (for quantile only)
    q: Vec<f64>,
    /// Weights (for weighted_mean/_var)
    weights: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    /// For quantile: vector. For weighted_mean/_var: single-element vector.
    out: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    max_abs_diff: f64,
    max_rel_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create quantile diff dir");
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

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // quantile sweeps
    let datasets: &[(&str, Vec<f64>)] = &[
        ("uniform_10", (1..=10).map(|i| i as f64).collect()),
        (
            "sorted_floats",
            vec![0.1, 0.2, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0],
        ),
        (
            "unsorted",
            vec![5.0, 3.0, 1.0, 8.0, 4.0, 9.0, 2.0, 6.0, 7.0, 0.0],
        ),
        (
            "with_duplicates",
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        ),
    ];
    let q_grid = vec![0.0_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    for (label, data) in datasets {
        pts.push(CasePoint {
            case_id: format!("quantile_{label}"),
            op: "quantile".into(),
            data: data.clone(),
            q: q_grid.clone(),
            weights: Vec::new(),
        });
    }

    // weighted_mean / weighted_var sweeps
    let wpairs: &[(&str, Vec<f64>, Vec<f64>)] = &[
        ("equal_weights", vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1.0; 5]),
        (
            "linear_weights",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ),
        (
            "mixed",
            vec![10.0, 5.0, 3.0, 8.0, 2.0],
            vec![0.1, 0.3, 0.4, 0.15, 0.05],
        ),
        (
            "small_values",
            vec![0.001, 0.002, 0.005, 0.01],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
    ];
    for (label, values, weights) in wpairs {
        for op in ["weighted_mean", "weighted_var"] {
            pts.push(CasePoint {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                data: values.clone(),
                q: Vec::new(),
                weights: weights.clone(),
            });
        }
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    op = c["op"]
    try:
        data = np.array(c["data"], dtype=float)
        if op == "quantile":
            qv = np.array(c["q"], dtype=float)
            r = np.quantile(data, qv, method="linear")
            r = [float(v) for v in r]
        elif op == "weighted_mean":
            w = np.array(c["weights"], dtype=float)
            r = [float(np.average(data, weights=w))]
        elif op == "weighted_var":
            w = np.array(c["weights"], dtype=float)
            mean = np.average(data, weights=w)
            var = float(np.sum(w * (data - mean) ** 2) / np.sum(w))
            r = [var]
        else:
            r = None
        if r is None or not all(math.isfinite(v) for v in r):
            out.append({"case_id": cid, "out": None})
        else:
            out.append({"case_id": cid, "out": r})
    except Exception:
        out.append({"case_id": cid, "out": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping quantile oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping quantile oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping quantile oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_stats_quantile_weighted_helpers() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let Some(expected) = o.out.as_ref() else {
            continue;
        };

        let actual: Vec<f64> = match case.op.as_str() {
            "quantile" => quantile(&case.data, &case.q),
            "weighted_mean" => vec![weighted_mean(&case.data, &case.weights)],
            "weighted_var" => vec![weighted_var(&case.data, &case.weights)],
            other => panic!("unknown op {other}"),
        };
        if actual.len() != expected.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                max_abs_diff: f64::INFINITY,
                max_rel_diff: f64::INFINITY,
                pass: false,
                note: format!(
                    "length mismatch: fsci={} numpy={}",
                    actual.len(),
                    expected.len()
                ),
            });
            continue;
        }

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (a, e) in actual.iter().zip(expected.iter()) {
            let abs_d = (a - e).abs();
            let denom = e.abs().max(1.0e-300);
            max_abs = max_abs.max(abs_d);
            max_rel = max_rel.max(abs_d / denom);
        }
        let pass = max_rel <= REL_TOL || max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_stats_quantile_weighted_helpers".into(),
        category: "fsci_stats::{quantile, weighted_mean, weighted_var} vs numpy".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "stats_helpers mismatch: {} ({}) max_rel={} max_abs={} note={}",
                d.case_id, d.op, d.max_rel_diff, d.max_abs_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "quantile/weighted parity failed: {} cases",
        diffs.len()
    );
}
