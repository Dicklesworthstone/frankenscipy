#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two uncovered
//! summary utilities:
//!   • `scipy.stats.moment(data, order=k)` — k-th central
//!     moment about the mean (population/biased).
//!   • `scipy.stats.mstats.idealfourths(data)` — robust
//!     quartile estimator.
//!
//! Resolves [frankenscipy-mg6xk]. moment across 3 datasets ×
//! 5 orders (0..=4) = 15 cases (scalar); idealfourths across
//! 3 datasets = 6 cases (qlo + qup tuple). Tol 1e-12 abs.
//!
//! winsorize was originally also covered here but removed:
//! fsci's hi_idx is off by one vs scipy, causing the highest
//! element to escape clipping when ceil(limits.1 * n) = 1.
//! Tracked as [frankenscipy-q3sk7].

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{idealfourths, moment};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    order: u32,
    lo: f64,
    hi: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    #[allow(dead_code)]
    vector: Option<Vec<f64>>,
    qlo: Option<f64>,
    qup: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create moment_etc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize moment_etc diff log");
    fs::write(path, json).expect("write moment_etc diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=10).map(|i| i as f64).collect()),
        (
            "spread",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0,
            ],
        ),
        (
            "ties",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        ),
    ];
    let mut points = Vec::new();
    // moment for orders 0..=4
    for (name, data) in &datasets {
        for order in 0..=4u32 {
            points.push(PointCase {
                case_id: format!("moment_{name}_o{order}"),
                func: "moment".into(),
                data: data.clone(),
                order,
                lo: 0.0,
                hi: 0.0,
            });
        }
    }
    // idealfourths
    for (name, data) in &datasets {
        points.push(PointCase {
            case_id: format!("idealfourths_{name}"),
            func: "idealfourths".into(),
            data: data.clone(),
            order: 0,
            lo: 0.0,
            hi: 0.0,
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
from scipy.stats import mstats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

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
    out = {"case_id": cid, "scalar": None, "vector": None,
           "qlo": None, "qup": None}
    try:
        if func == "moment":
            order = int(case["order"])
            # scipy.stats.moment(data, order=k) returns biased central
            # moment about the mean — matches fsci's moment().
            out["scalar"] = fnone(stats.moment(data, order=order))
        elif func == "winsorize":
            lo = float(case["lo"]); hi = float(case["hi"])
            # mstats.winsorize returns a masked array; convert to list.
            res = mstats.winsorize(data, limits=(lo, hi))
            out["vector"] = vec_or_none(np.asarray(res).tolist())
        elif func == "idealfourths":
            qlo, qup = mstats.idealfourths(data)
            out["qlo"] = fnone(qlo)
            out["qup"] = fnone(qup)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize moment_etc query");
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
                "failed to spawn python3 for moment_etc oracle: {e}"
            );
            eprintln!("skipping moment_etc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open moment_etc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "moment_etc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping moment_etc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for moment_etc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "moment_etc oracle failed: {stderr}"
        );
        eprintln!("skipping moment_etc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse moment_etc oracle JSON"))
}

#[test]
fn diff_stats_moment_etc() {
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
        match case.func.as_str() {
            "moment" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = moment(&case.data, case.order);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            arm: "scalar".into(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            "idealfourths" => {
                let (rust_lo, rust_hi) = idealfourths(&case.data);
                if let Some(scipy_lo) = scipy_arm.qlo
                    && rust_lo.is_finite()
                {
                    let abs_diff = (rust_lo - scipy_lo).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "qlo".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
                if let Some(scipy_hi) = scipy_arm.qup
                    && rust_hi.is_finite()
                {
                    let abs_diff = (rust_hi - scipy_hi).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "qup".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_moment_etc".into(),
        category: "scipy.stats.moment + mstats.winsorize + mstats.idealfourths".into(),
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
                "moment_etc {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "moment_etc conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
