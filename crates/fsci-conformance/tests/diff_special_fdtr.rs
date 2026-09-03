#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the F-distribution
//! scipy-compat wrappers
//! `scipy.special.fdtr/fdtrc/fdtri`.
//!
//! Resolves [frankenscipy-qnh7y]. Verifies the scipy-compat
//! cdf/sf/ppf wrappers directly; complements the diff_stats_f
//! harness which exercises the same kernel indirectly via
//! `FDistribution`. fdtr = cdf, fdtrc = sf, fdtri = ppf.
//!
//! 6 (dfn, dfd) pairs × 7 x or q = 84 cases × 3 funcs cap.
//! Tolerances: 1e-12 abs cdf/sf (regularized incomplete beta),
//! 1e-9 rel ppf.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{fdtr, fdtrc, fdtri};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    dfn: f64,
    dfd: f64,
    arg: f64,
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
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create fdtr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fdtr diff log");
    fs::write(path, json).expect("write fdtr diff log");
}

fn fsci_eval(func: &str, dfn: f64, dfd: f64, arg: f64) -> Option<f64> {
    let v = match func {
        "fdtr" => fdtr(dfn, dfd, arg),
        "fdtrc" => fdtrc(dfn, dfd, arg),
        "fdtri" => fdtri(dfn, dfd, arg),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let pairs = [
        (1.0_f64, 1.0),
        (2.0, 5.0),
        (3.0, 10.0),
        (5.0, 5.0),
        (10.0, 30.0),
        (50.0, 100.0),
    ];
    let xs = [0.01_f64, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let mut points = Vec::new();
    for &(dfn, dfd) in &pairs {
        for &x in &xs {
            for func in ["fdtr", "fdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_dfn{dfn}_dfd{dfd}_x{x}"),
                    func: func.to_string(),
                    dfn,
                    dfd,
                    arg: x,
                });
            }
        }
        for &q in &qs {
            points.push(PointCase {
                case_id: format!("fdtri_dfn{dfn}_dfd{dfd}_q{q}"),
                func: "fdtri".into(),
                dfn,
                dfd,
                arg: q,
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
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]
    dfn = float(case["dfn"]); dfd = float(case["dfd"]); arg = float(case["arg"])
    try:
        if func == "fdtr":   value = special.fdtr(dfn, dfd, arg)
        elif func == "fdtrc":value = special.fdtrc(dfn, dfd, arg)
        elif func == "fdtri":value = special.fdtri(dfn, dfd, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize fdtr query");
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
                "failed to spawn python3 for fdtr oracle: {e}"
            );
            eprintln!("skipping fdtr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fdtr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fdtr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fdtr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for fdtr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fdtr oracle failed: {stderr}"
        );
        eprintln!("skipping fdtr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fdtr oracle JSON"))
}

#[test]
fn diff_special_fdtr() {
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
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.dfn, case.dfd, case.arg)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);

            let pass = match case.func.as_str() {
                "fdtr" | "fdtrc" => abs_diff <= CDF_TOL,
                "fdtri" => abs_diff <= PPF_TOL_REL * scale,
                _ => false,
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_fdtr".into(),
        category: "scipy.special.fdtr/fdtrc/fdtri".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "fdtr {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special fdtr conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
