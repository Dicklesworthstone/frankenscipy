#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the trimmed-stat
//! family in scipy.stats:
//!   • `tmean(data, limits, inclusive)`
//!   • `tvar (data, limits, inclusive, ddof=1)`
//!   • `tstd (data, limits, inclusive, ddof=1)`
//!   • `tsem (data, limits, inclusive, ddof=1)`
//!   • `tmin (data, lowerlimit, inclusive)`
//!   • `tmax (data, upperlimit, inclusive)`
//!
//! Resolves [frankenscipy-pn2lk]. Each function trims the
//! input by limits then computes the corresponding summary;
//! the harness exercises representative fixtures × limit
//! configs across all six.
//!
//! 3 datasets × 5 configs × 6 functions ≈ 30+ cases via
//! subprocess. Tol 1e-12 abs (closed-form filter + sum/var).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{tmax, tmean, tmin, tsem, tstd, tvar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    lo: f64,
    hi: f64,
    inc_lo: bool,
    inc_hi: bool,
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
    fs::create_dir_all(output_dir()).expect("create trimmed diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize trimmed diff log");
    fs::write(path, json).expect("write trimmed diff log");
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let limits = (case.lo, case.hi);
    let inclusive = (case.inc_lo, case.inc_hi);
    let v = match case.func.as_str() {
        "tmean" => tmean(&case.data, limits, inclusive),
        "tvar" => tvar(&case.data, limits, inclusive, 1),
        "tstd" => tstd(&case.data, limits, inclusive, 1),
        "tsem" => tsem(&case.data, limits, inclusive, 1),
        "tmin" => tmin(&case.data, case.lo, case.inc_lo),
        "tmax" => tmax(&case.data, case.hi, case.inc_hi),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
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
    ];
    // (lo, hi, inc_lo, inc_hi)
    let configs: &[(&str, f64, f64, bool, bool)] = &[
        ("inner", 5.0, 15.0, true, true),
        ("excl", 5.0, 15.0, false, false),
        ("low_excl", 5.0, 15.0, false, true),
        ("hi_excl", 5.0, 15.0, true, false),
        ("wide", -100.0, 100.0, true, true),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for (cname, lo, hi, ilo, ihi) in configs {
            for func in ["tmean", "tvar", "tstd", "tsem", "tmin", "tmax"] {
                points.push(PointCase {
                    case_id: format!("{func}_{name}_{cname}"),
                    func: func.into(),
                    data: data.clone(),
                    lo: *lo,
                    hi: *hi,
                    inc_lo: *ilo,
                    inc_hi: *ihi,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    lo = float(case["lo"]); hi = float(case["hi"])
    ilo = bool(case["inc_lo"]); ihi = bool(case["inc_hi"])
    val = None
    try:
        if func == "tmean":
            val = float(stats.tmean(data, (lo, hi), (ilo, ihi)))
        elif func == "tvar":
            val = float(stats.tvar(data, (lo, hi), (ilo, ihi), ddof=1))
        elif func == "tstd":
            val = float(stats.tstd(data, (lo, hi), (ilo, ihi), ddof=1))
        elif func == "tsem":
            val = float(stats.tsem(data, (lo, hi), (ilo, ihi), ddof=1))
        elif func == "tmin":
            val = float(stats.tmin(data, lowerlimit=lo, inclusive=ilo))
        elif func == "tmax":
            val = float(stats.tmax(data, upperlimit=hi, inclusive=ihi))
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize trimmed query");
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
                "failed to spawn python3 for trimmed oracle: {e}"
            );
            eprintln!("skipping trimmed oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open trimmed oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trimmed oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trimmed oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for trimmed oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trimmed oracle failed: {stderr}"
        );
        eprintln!("skipping trimmed oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trimmed oracle JSON"))
}

#[test]
fn diff_stats_trimmed() {
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
        if let Some(scipy_v) = scipy_arm.value
            && let Some(rust_v) = fsci_eval(case)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_trimmed".into(),
        category: "scipy.stats trimmed family".into(),
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
                "trimmed {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "trimmed conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
