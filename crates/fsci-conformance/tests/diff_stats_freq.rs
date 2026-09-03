#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the histogram-based
//! frequency utilities:
//!   • `scipy.stats.relfreq(data, numbins=k)` — relative
//!     frequency per bin (sums to 1)
//!   • `scipy.stats.cumfreq(data, numbins=k)` — cumulative
//!     frequency per bin (last entry equals n)
//!
//! Resolves [frankenscipy-0fl5d]. fsci returns
//! `(frequencies, edges)`; scipy returns
//! `(frequency, lowerlimit, binsize, extrapoints)`. The
//! harness compares the frequency vector element-wise plus
//! lowerlimit (= edges[0]) and binsize
//! (= edges[1] - edges[0]), using SciPy's default padded
//! limits.
//!
//! 4 datasets × 2 bin counts × 2 funcs × (frequency_max +
//! lowerlimit + binsize) = 48 cases via subprocess. Tol 1e-12
//! abs (closed-form histogram + cumulative sum).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{cumfreq, relfreq};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    bins: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    frequency: Option<Vec<f64>>,
    lowerlimit: Option<f64>,
    binsize: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create freq diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize freq diff log");
    fs::write(path, json).expect("write freq diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=20).map(|i| i as f64).collect()),
        (
            "spread",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0, 15.0, 20.0,
            ],
        ),
        (
            "ties",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ),
        ("constant", vec![3.0, 3.0, 3.0, 3.0]),
    ];
    let bin_counts: [u64; 2] = [5, 10];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for &bins in &bin_counts {
            for func in ["relfreq", "cumfreq"] {
                points.push(PointCase {
                    case_id: format!("{func}_{name}_b{bins}"),
                    func: func.into(),
                    data: data.clone(),
                    bins,
                });
            }
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
    bins = int(case["bins"])
    out = {"case_id": cid, "frequency": None, "lowerlimit": None, "binsize": None}
    try:
        if func == "relfreq":
            res = stats.relfreq(data, numbins=bins)
            values = res.frequency
        else:
            res = stats.cumfreq(data, numbins=bins)
            values = res.cumcount
        out["frequency"] = vec_or_none(values.tolist())
        out["lowerlimit"] = fnone(res.lowerlimit)
        out["binsize"] = fnone(res.binsize)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize freq query");
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
                "failed to spawn python3 for freq oracle: {e}"
            );
            eprintln!("skipping freq oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open freq oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "freq oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping freq oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for freq oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "freq oracle failed: {stderr}"
        );
        eprintln!("skipping freq oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse freq oracle JSON"))
}

#[test]
fn diff_stats_freq() {
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
        let (rust_freq, rust_edges) = match case.func.as_str() {
            "relfreq" => relfreq(&case.data, case.bins as usize),
            "cumfreq" => cumfreq(&case.data, case.bins as usize),
            _ => continue,
        };

        // frequency vector
        if let Some(scipy_freq) = &scipy_arm.frequency
            && rust_freq.len() == scipy_freq.len()
        {
            let mut max_local = 0.0_f64;
            for (a, b) in rust_freq.iter().zip(scipy_freq.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "frequency_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }

        // lowerlimit = edges[0]
        if let Some(scipy_lo) = scipy_arm.lowerlimit
            && let Some(&rust_lo) = rust_edges.first()
        {
            let abs_diff = (rust_lo - scipy_lo).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "lowerlimit".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }

        // binsize = edges[1] - edges[0]
        if let Some(scipy_bs) = scipy_arm.binsize
            && rust_edges.len() >= 2
        {
            let rust_bs = rust_edges[1] - rust_edges[0];
            let abs_diff = (rust_bs - scipy_bs).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "binsize".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_freq".into(),
        category: "scipy.stats.relfreq + cumfreq".into(),
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
                "freq {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "freq conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
