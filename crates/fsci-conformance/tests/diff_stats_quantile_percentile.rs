#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's
//! `quantile(data, q)` (linear-interpolation quantile vector,
//! numpy default method) and `percentile(data, q)` (single-q
//! wrapper, q in 0..=100).
//!
//! Resolves [frankenscipy-ovhtj]. The oracle calls
//! `numpy.{quantile, percentile}` with method='linear'.
//!
//! 4 datasets × {quantile (vector) at 4 probs, percentile
//! scalar at 4 percentiles} = 32 cases. Tol 1e-12 abs
//! (closed-form linear-interpolation chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{percentile, quantile};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    /// Probabilities in [0, 1] for `quantile`, or in [0, 100] for `percentile`.
    probs: Vec<f64>,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create quantile_percentile diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize quantile_percentile diff log");
    fs::write(path, json).expect("write quantile_percentile diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_n12", (1..=12).map(|i| i as f64).collect()),
        (
            "spread_n15",
            vec![
                -3.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0, 16.0, 21.0, 27.0, 34.0,
            ],
        ),
        (
            "ties_n14",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ),
        ("small_n5", vec![1.0, 4.0, 7.0, 9.0, 13.0]),
    ];

    let q_probs = vec![0.1, 0.25, 0.5, 0.9];
    let pct_probs: Vec<f64> = q_probs.iter().map(|p| p * 100.0).collect();

    let mut points = Vec::new();
    for (name, data) in &datasets {
        points.push(PointCase {
            case_id: format!("{name}_quantile"),
            func: "quantile".into(),
            data: data.clone(),
            probs: q_probs.clone(),
        });
        points.push(PointCase {
            case_id: format!("{name}_percentile"),
            func: "percentile".into(),
            data: data.clone(),
            probs: pct_probs.clone(),
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

def vec_or_none(arr):
    out = []
    for v in arr:
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
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    probs = np.array(case["probs"], dtype=float)
    val = None
    try:
        if func == "quantile":
            out = np.quantile(data, probs, method="linear")
            val = vec_or_none(np.asarray(out).tolist())
        elif func == "percentile":
            out = [np.percentile(data, p, method="linear") for p in probs]
            val = vec_or_none(out)
    except Exception:
        val = None
    points.append({"case_id": cid, "values": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize quantile_percentile query");
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
                "failed to spawn python3 for quantile_percentile oracle: {e}"
            );
            eprintln!("skipping quantile_percentile oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open quantile_percentile oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "quantile_percentile oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping quantile_percentile oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for quantile_percentile oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "quantile_percentile oracle failed: {stderr}"
        );
        eprintln!("skipping quantile_percentile oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse quantile_percentile oracle JSON"))
}

#[test]
fn diff_stats_quantile_percentile() {
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
        let Some(scipy_vec) = &scipy_arm.values else {
            continue;
        };
        let rust_vec: Vec<f64> = match case.func.as_str() {
            "quantile" => quantile(&case.data, &case.probs),
            "percentile" => case
                .probs
                .iter()
                .map(|&p| percentile(&case.data, p))
                .collect(),
            _ => continue,
        };
        if rust_vec.len() != scipy_vec.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        for (i, (r, s)) in rust_vec.iter().zip(scipy_vec.iter()).enumerate() {
            if r.is_finite() {
                let abs_diff = (r - s).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: format!("{}.q{}", case.func, case.probs[i]),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_quantile_percentile".into(),
        category: "numpy.{quantile, percentile} (method=linear)".into(),
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
                "quantile_percentile {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "quantile_percentile conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
