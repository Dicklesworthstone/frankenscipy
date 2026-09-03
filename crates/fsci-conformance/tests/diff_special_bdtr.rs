#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the binomial and
//! negative-binomial scipy-compat wrappers
//! `scipy.special.bdtr/bdtrc/bdtri/nbdtr/nbdtrc/nbdtri`.
//!
//! Resolves [frankenscipy-mqguh]. Verifies the cdf/sf/ppf
//! family wrappers directly; complements diff_stats_binom and
//! diff_stats_nbinom which exercise the same kernel indirectly.
//!
//! Tolerances: 1e-12 abs cdf/sf, 1e-9 rel ppf.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{bdtr, bdtrc, bdtri, nbdtr, nbdtrc, nbdtri};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    k: f64,
    n: f64,
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
    fs::create_dir_all(output_dir()).expect("create bdtr/nbdtr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bdtr/nbdtr diff log");
    fs::write(path, json).expect("write bdtr/nbdtr diff log");
}

fn fsci_eval(func: &str, k: f64, n: f64, arg: f64) -> Option<f64> {
    let v = match func {
        "bdtr" => bdtr(k, n, arg),
        "bdtrc" => bdtrc(k, n, arg),
        "bdtri" => bdtri(k, n, arg),
        "nbdtr" => nbdtr(k, n, arg),
        "nbdtrc" => nbdtrc(k, n, arg),
        "nbdtri" => nbdtri(k, n, arg),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // Binomial: (n, p) trials, k successes.
    let bdtr_cases = [(10.0_f64, 0.3), (20.0, 0.5), (50.0, 0.1), (100.0, 0.95)];
    let ks_bdtr = [0_u32, 2, 5, 10, 20];
    // Negative binomial: (n, p) waiting for n successes, k failures.
    let nbdtr_cases = [(1.0_f64, 0.5), (5.0, 0.5), (10.0, 0.3), (3.0, 0.9)];
    let ks_nbdtr = [0_u32, 1, 5, 10, 20];
    // y/p in (0, 1).
    let ys = [0.05_f64, 0.25, 0.5, 0.75, 0.95];

    let mut points = Vec::new();
    for &(n, p) in &bdtr_cases {
        for &k in &ks_bdtr {
            let kf = k as f64;
            if kf > n {
                continue;
            }
            for func in ["bdtr", "bdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_n{n}_p{p}_k{k}"),
                    func: func.to_string(),
                    k: kf,
                    n,
                    arg: p,
                });
            }
        }
        // bdtri: k, n fixed; arg=y in (0,1) returns p.
        for &k in &ks_bdtr {
            let kf = k as f64;
            if kf > n - 1.0 {
                continue;
            }
            for &y in &ys {
                points.push(PointCase {
                    case_id: format!("bdtri_n{n}_y{y}_k{k}"),
                    func: "bdtri".into(),
                    k: kf,
                    n,
                    arg: y,
                });
            }
        }
    }
    for &(nn, p) in &nbdtr_cases {
        for &k in &ks_nbdtr {
            let kf = k as f64;
            for func in ["nbdtr", "nbdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_n{nn}_p{p}_k{k}"),
                    func: func.to_string(),
                    k: kf,
                    n: nn,
                    arg: p,
                });
            }
        }
        for &k in &ks_nbdtr {
            let kf = k as f64;
            for &y in &ys {
                points.push(PointCase {
                    case_id: format!("nbdtri_n{nn}_y{y}_k{k}"),
                    func: "nbdtri".into(),
                    k: kf,
                    n: nn,
                    arg: y,
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
    cid = case["case_id"]; func = case["func"]
    k = float(case["k"]); n = float(case["n"]); arg = float(case["arg"])
    try:
        if func == "bdtr":   value = special.bdtr(k, n, arg)
        elif func == "bdtrc":value = special.bdtrc(k, n, arg)
        elif func == "bdtri":value = special.bdtri(k, n, arg)
        elif func == "nbdtr":value = special.nbdtr(k, n, arg)
        elif func == "nbdtrc":value = special.nbdtrc(k, n, arg)
        elif func == "nbdtri":value = special.nbdtri(k, n, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize bdtr/nbdtr query");
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
                "failed to spawn python3 for bdtr/nbdtr oracle: {e}"
            );
            eprintln!("skipping bdtr/nbdtr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open bdtr/nbdtr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bdtr/nbdtr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bdtr/nbdtr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for bdtr/nbdtr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bdtr/nbdtr oracle failed: {stderr}"
        );
        eprintln!("skipping bdtr/nbdtr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bdtr/nbdtr oracle JSON"))
}

#[test]
fn diff_special_bdtr() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.k, case.n, case.arg)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);

            let pass = match case.func.as_str() {
                "bdtr" | "bdtrc" | "nbdtr" | "nbdtrc" => abs_diff <= CDF_TOL,
                "bdtri" | "nbdtri" => abs_diff <= PPF_TOL_REL * scale,
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
        test_id: "diff_special_bdtr".into(),
        category: "scipy.special.bdtr/bdtrc/bdtri/nbdtr/nbdtrc/nbdtri".into(),
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
                "bdtr/nbdtr {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special bdtr/nbdtr conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
