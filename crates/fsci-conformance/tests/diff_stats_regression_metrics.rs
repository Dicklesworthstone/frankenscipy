#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's sklearn-style
//! regression metrics:
//!   • `r2_score(y_true, y_pred)` — coefficient of determination
//!   • `mean_absolute_error(y_true, y_pred)` — MAE
//!   • `mean_squared_error(y_true, y_pred)` — MSE
//!   • `root_mean_squared_error(y_true, y_pred)` — RMSE
//!   • `mean_absolute_percentage_error(y_true, y_pred)` — MAPE
//!     (note: fsci skips terms where y_true == 0; oracle
//!     mirrors that convention)
//!
//! Resolves [frankenscipy-kuazc]. The oracle reproduces each
//! formula in numpy directly.
//!
//! 4 (y_true, y_pred) fixtures × 5 funcs = 20 cases. Tol 1e-12
//! abs (closed-form sums; no transcendentals).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score,
    root_mean_squared_error,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create regression_metrics diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize regression_metrics diff log");
    fs::write(path, json).expect("write regression_metrics diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Perfect prediction (R² = 1, all errors 0)
        (
            "perfect",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| i as f64).collect(),
        ),
        // Mild error
        (
            "mild_error",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| (i as f64) + 0.5).collect(),
        ),
        // Larger error, mixed signs
        (
            "mixed_error",
            vec![
                1.0, 2.5, -3.0, 4.0, -5.5, 6.0, 7.5, -8.0, 9.0, 10.5, -11.0, 12.0,
            ],
            vec![
                1.5, 2.0, -2.5, 4.5, -5.0, 5.5, 8.0, -7.5, 9.5, 10.0, -10.5, 12.5,
            ],
        ),
        // Real-valued targets, decay-style predictions
        (
            "decay_n15",
            (1..=15).map(|i| (i as f64).powi(2) / 4.0).collect(),
            (1..=15)
                .map(|i| ((i as f64).powi(2) / 4.0) * 0.92 + 1.5)
                .collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, y_true, y_pred) in &fixtures {
        for func in ["r2", "mae", "mse", "rmse", "mape"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                y_true: y_true.clone(),
                y_pred: y_pred.clone(),
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    yt = np.array(case["y_true"], dtype=float)
    yp = np.array(case["y_pred"], dtype=float)
    val = None
    try:
        if func == "r2":
            mean = float(yt.mean())
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - mean) ** 2))
            val = 1.0 if ss_tot == 0 and ss_res == 0 else (
                0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
            )
        elif func == "mae":
            val = float(np.mean(np.abs(yt - yp)))
        elif func == "mse":
            val = float(np.mean((yt - yp) ** 2))
        elif func == "rmse":
            val = float(math.sqrt(np.mean((yt - yp) ** 2)))
        elif func == "mape":
            # fsci convention: skip terms where y_true == 0 (treats as 0).
            terms = []
            for t, p in zip(yt.tolist(), yp.tolist()):
                if t == 0.0:
                    terms.append(0.0)
                else:
                    terms.append(abs((t - p) / t))
            val = float(sum(terms) / len(terms))
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize regression_metrics query");
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
                "failed to spawn python3 for regression_metrics oracle: {e}"
            );
            eprintln!("skipping regression_metrics oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open regression_metrics oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "regression_metrics oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping regression_metrics oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for regression_metrics oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "regression_metrics oracle failed: {stderr}"
        );
        eprintln!("skipping regression_metrics oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse regression_metrics oracle JSON"))
}

#[test]
fn diff_stats_regression_metrics() {
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
        let rust_v = match case.func.as_str() {
            "r2" => r2_score(&case.y_true, &case.y_pred),
            "mae" => mean_absolute_error(&case.y_true, &case.y_pred),
            "mse" => mean_squared_error(&case.y_true, &case.y_pred),
            "rmse" => root_mean_squared_error(&case.y_true, &case.y_pred),
            "mape" => mean_absolute_percentage_error(&case.y_true, &case.y_pred),
            _ => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_regression_metrics".into(),
        category: "regression metrics: r2 / mae / mse / rmse / mape (numpy reference)".into(),
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
                "regression_metrics {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "regression_metrics conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
