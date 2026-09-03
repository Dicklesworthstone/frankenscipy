#![forbid(unsafe_code)]
//! Live SciPy differential coverage for closed-form central
//! moments and the biased skew / excess-kurtosis ratios:
//!   • `moment(data, k)` — k-th central moment, k ∈ 2..=5
//!   • `skew(data)` — biased skewness m3/m2^1.5
//!   • `kurtosis(data)` — biased excess kurtosis m4/m2² − 3
//!
//! Resolves [frankenscipy-mj4c5]. The oracle calls
//! `scipy.stats.{moment, skew, kurtosis}` with the bias=True
//! convention that fsci uses by default.
//!
//! 3 datasets × 4 moment orders (k=2..5) + 3 datasets ×
//! {skew, kurtosis} = 18 cases. Tol 1e-12 abs (closed-form
//! polynomial in (x − mean)).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{kurtosis, moment, skew};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    /// Only used for the `moment` arm.
    k: u32,
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
    fs::create_dir_all(output_dir()).expect("create central_moments diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize central_moments diff log");
    fs::write(path, json).expect("write central_moments diff log");
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
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for k in 2u32..=5 {
            points.push(PointCase {
                case_id: format!("{name}_moment_k{k}"),
                func: "moment".into(),
                data: data.clone(),
                k,
            });
        }
        points.push(PointCase {
            case_id: format!("{name}_skew"),
            func: "skew".into(),
            data: data.clone(),
            k: 0,
        });
        points.push(PointCase {
            case_id: format!("{name}_kurtosis"),
            func: "kurtosis".into(),
            data: data.clone(),
            k: 0,
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
from scipy import stats

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
    data = np.array(case["data"], dtype=float)
    k = int(case["k"])
    val = None
    try:
        if func == "moment":
            val = stats.moment(data, moment=k)
        elif func == "skew":
            # bias=True matches fsci's biased m3/m2^1.5 default.
            val = stats.skew(data, bias=True)
        elif func == "kurtosis":
            # fisher=True (excess), bias=True (m4/m2^2 - 3).
            val = stats.kurtosis(data, fisher=True, bias=True)
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize central_moments query");
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
                "failed to spawn python3 for central_moments oracle: {e}"
            );
            eprintln!("skipping central_moments oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open central_moments oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "central_moments oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping central_moments oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for central_moments oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "central_moments oracle failed: {stderr}"
        );
        eprintln!("skipping central_moments oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse central_moments oracle JSON"))
}

#[test]
fn diff_stats_central_moments() {
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
            "moment" => moment(&case.data, case.k),
            "skew" => skew(&case.data),
            "kurtosis" => kurtosis(&case.data),
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
        test_id: "diff_stats_central_moments".into(),
        category: "scipy.stats.{moment, skew, kurtosis}".into(),
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
                "central_moments {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "central_moments conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
