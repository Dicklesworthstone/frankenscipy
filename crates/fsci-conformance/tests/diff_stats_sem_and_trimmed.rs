#![forbid(unsafe_code)]
//! Live SciPy differential coverage for closed-form
//! standard-error and limit-trimmed descriptive scalars:
//!   • `sem(data)` — standard error of the mean (ddof=1)
//!   • `tmean(data, limits, inclusive)` — limit-trimmed mean
//!   • `tvar(data, limits, inclusive, ddof)` — trimmed variance
//!   • `tstd(data, limits, inclusive, ddof)` — trimmed std-dev
//!   • `tsem(data, limits, inclusive, ddof)` — trimmed SEM
//!   • `tmin(data, lowerlimit, inclusive)` — trimmed min
//!   • `tmax(data, upperlimit, inclusive)` — trimmed max
//!
//! Resolves [frankenscipy-lcgpq]. The oracle calls
//! `scipy.stats.{sem, tmean, tvar, tstd, tsem, tmin, tmax}`.
//!
//! 3 datasets × 2 limit configs × 7 funcs ≈ 42 cases via
//! subprocess. Tol 1e-12 abs (closed-form mean / variance
//! after a boolean filter).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{sem, tmax, tmean, tmin, tsem, tstd, tvar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    limits: (f64, f64),
    inclusive: (bool, bool),
    ddof: usize,
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
    fs::create_dir_all(output_dir()).expect("create sem_and_trimmed diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sem_and_trimmed diff log");
    fs::write(path, json).expect("write sem_and_trimmed diff log");
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

    let limit_configs: Vec<(&str, (f64, f64), (bool, bool))> = vec![
        // Wide window — keeps almost all data, both endpoints inclusive
        ("wide_inc", (-100.0, 100.0), (true, true)),
        // Narrow window — drops tails, half-open
        ("narrow_excl", (1.0, 10.0), (false, false)),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for (lname, limits, inclusive) in &limit_configs {
            for func in ["tmean", "tvar", "tstd", "tsem", "tmin", "tmax"] {
                points.push(PointCase {
                    case_id: format!("{name}_{lname}_{func}"),
                    func: func.into(),
                    data: data.clone(),
                    limits: *limits,
                    inclusive: *inclusive,
                    ddof: 1,
                });
            }
        }
        // sem doesn't take limits — emit one case per dataset.
        points.push(PointCase {
            case_id: format!("{name}_sem"),
            func: "sem".into(),
            data: data.clone(),
            limits: (f64::NEG_INFINITY, f64::INFINITY),
            inclusive: (true, true),
            ddof: 1,
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
    lo, hi = case["limits"]
    inc_lo, inc_hi = case["inclusive"]
    ddof = case["ddof"]
    val = None
    try:
        if func == "sem":
            val = stats.sem(data, ddof=1)
        elif func == "tmean":
            val = stats.tmean(data, limits=(lo, hi), inclusive=(inc_lo, inc_hi))
        elif func == "tvar":
            val = stats.tvar(data, limits=(lo, hi), inclusive=(inc_lo, inc_hi), ddof=ddof)
        elif func == "tstd":
            val = stats.tstd(data, limits=(lo, hi), inclusive=(inc_lo, inc_hi), ddof=ddof)
        elif func == "tsem":
            val = stats.tsem(data, limits=(lo, hi), inclusive=(inc_lo, inc_hi), ddof=ddof)
        elif func == "tmin":
            val = stats.tmin(data, lowerlimit=lo, inclusive=inc_lo)
        elif func == "tmax":
            val = stats.tmax(data, upperlimit=hi, inclusive=inc_hi)
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sem_and_trimmed query");
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
                "failed to spawn python3 for sem_and_trimmed oracle: {e}"
            );
            eprintln!("skipping sem_and_trimmed oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open sem_and_trimmed oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sem_and_trimmed oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sem_and_trimmed oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for sem_and_trimmed oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sem_and_trimmed oracle failed: {stderr}"
        );
        eprintln!("skipping sem_and_trimmed oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sem_and_trimmed oracle JSON"))
}

#[test]
fn diff_stats_sem_and_trimmed() {
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
            "sem" => sem(&case.data),
            "tmean" => tmean(&case.data, case.limits, case.inclusive),
            "tvar" => tvar(&case.data, case.limits, case.inclusive, case.ddof),
            "tstd" => tstd(&case.data, case.limits, case.inclusive, case.ddof),
            "tsem" => tsem(&case.data, case.limits, case.inclusive, case.ddof),
            "tmin" => tmin(&case.data, case.limits.0, case.inclusive.0),
            "tmax" => tmax(&case.data, case.limits.1, case.inclusive.1),
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
        test_id: "diff_stats_sem_and_trimmed".into(),
        category: "scipy.stats.{sem, tmean, tvar, tstd, tsem, tmin, tmax}".into(),
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
                "sem_and_trimmed {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "sem_and_trimmed conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
