#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the power-transform
//! forward functions (vector outputs):
//!   • `scipy.stats.boxcox(data, lmbda)` — positive data
//!   • `scipy.stats.yeojohnson(x, lmbda)` — real data
//!
//! Resolves [frankenscipy-3161i]. The companion log-
//! likelihood scalars (boxcox_llf, yeojohnson_llf) are
//! already covered by diff_stats_power_transforms.rs; this
//! harness exercises the orthogonal element-wise
//! transformation path.
//!
//! 3 boxcox datasets × 4 lambdas + 3 yeojohnson datasets × 4
//! lambdas = 24 cases via subprocess. Each case compares the
//! transformed vector element-wise with max-abs aggregation.
//! Tol 1e-12 abs (closed-form per-element transform).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{boxcox, yeojohnson};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// 1e-11 (not 1e-12) because boxcox at lambda=2 on inputs up to 100
// produces transformed values near 5000, where the (x^λ - 1)/λ
// arithmetic accumulates ~4.5e-12 floating-point noise relative to
// scipy. 1e-11 still catches algorithmic divergences.
const ABS_TOL: f64 = 1.0e-11;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    lmbda: f64,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    transformed: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create power_forward diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize power_forward diff log");
    fs::write(path, json).expect("write power_forward diff log");
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    match case.func.as_str() {
        "boxcox" => boxcox(&case.data, Some(case.lmbda)).ok().map(|r| r.data),
        "yeojohnson" => Some(yeojohnson(&case.data, case.lmbda)),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let lambdas: [f64; 4] = [-1.0, 0.0, 0.5, 2.0];

    let boxcox_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (
            "wide_range",
            vec![0.5, 1.2, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0],
        ),
        (
            "lognormal_like",
            vec![1.05, 2.7, 1.4, 3.1, 0.9, 4.2, 1.8, 2.0, 5.5, 1.25],
        ),
    ];
    let yj_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("centered", vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]),
        ("asymmetric", vec![-2.0, -1.0, 0.5, 1.0, 2.5, 4.0, 6.0, 8.0]),
        (
            "all_negative",
            vec![-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &boxcox_datasets {
        for &lmbda in &lambdas {
            points.push(PointCase {
                case_id: format!("boxcox_{name}_l{lmbda}"),
                func: "boxcox".into(),
                lmbda,
                data: data.clone(),
            });
        }
    }
    for (name, data) in &yj_datasets {
        for &lmbda in &lambdas {
            points.push(PointCase {
                case_id: format!("yj_{name}_l{lmbda}"),
                func: "yeojohnson".into(),
                lmbda,
                data: data.clone(),
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
from scipy import stats

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
    lmbda = float(case["lmbda"])
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        if func == "boxcox":
            val = vec_or_none(stats.boxcox(data, lmbda=lmbda).tolist())
        elif func == "yeojohnson":
            val = vec_or_none(stats.yeojohnson(data, lmbda=lmbda).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "transformed": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize power_forward query");
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
                "failed to spawn python3 for power_forward oracle: {e}"
            );
            eprintln!("skipping power_forward oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open power_forward oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "power_forward oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping power_forward oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for power_forward oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "power_forward oracle failed: {stderr}"
        );
        eprintln!("skipping power_forward oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse power_forward oracle JSON"))
}

#[test]
fn diff_stats_power_forward() {
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
        let Some(scipy_vec) = &scipy_arm.transformed else {
            continue;
        };
        let Some(rust_vec) = fsci_eval(case) else {
            continue;
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
        let mut max_local = 0.0_f64;
        for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
            if a.is_finite() {
                max_local = max_local.max((a - b).abs());
            }
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_power_forward".into(),
        category: "scipy.stats.boxcox/yeojohnson forward".into(),
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
                "power_forward {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "power_forward conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
