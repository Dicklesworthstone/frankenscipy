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
//!
//! Tail cases (frankenscipy-xzrpr): fdtri with q in the 1e-300 tail and within 1e-11 of 1,
//! held to the same 1e-9 relative to |SciPy| itself (the tail roots are far below 1). Each was
//! mpmath-checked (60 digits) with SciPy within 6e-16 of it; points where SciPy is NaN are
//! pinned to mpmath in fsci-special's unit tests instead.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_special::{fdtr, fdtrc, fdtri};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per function.
const ARMS: [&str; 3] = ["fdtr", "fdtrc", "fdtri"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    dfn: f64,
    dfd: f64,
    arg: f64,
    /// Compared relative to |SciPy| itself (see the module docs).
    tail: bool,
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
    compared: BTreeMap<String, ArmCounts>,
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
    // A non-finite value reaches the ledger, which records it as an fsci failure.
    Some(v)
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
                    tail: false,
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
                tail: false,
            });
        }
    }
    // frankenscipy-xzrpr. mpmath (60 digits): 6.598208468062813e-106, 5.917252068222744e-68,
    // 1.869629141547289e-28, 2.875419174435946e-25, 1.421218821030826e18, 3.729539123404430e20,
    // 4.303389216365210e16. Old fsci: 1.86e-29, 9.0e-48, 4.02e-19 and 2.35e-19 in the tail, and
    // inf for the three near 1, whose complement 1 − x it formed by subtraction.
    let tail_cases: [(f64, f64, f64); 7] = [
        (
            5.324849525451424,
            0.5324224027327918,
            5.864793095899144e-279,
        ),
        (
            3.3416874749375753,
            36.15028804913585,
            7.578048166151894e-113,
        ),
        (
            15.363963908108275,
            21.306223081330756,
            2.2377004571426287e-210,
        ),
        (
            15.677099968700094,
            23.498175505473483,
            1.0393981743088624e-189,
        ),
        (33.707123878628366, 1.208342874971375, 0.9999999999911681),
        (33.687472437582684, 1.0707827876516371, 0.9999999999922503),
        (26.55909782270653, 0.6721604835185495, 0.999998022105552),
    ];
    for (i, &(dfn, dfd, q)) in tail_cases.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("fdtri_tail{i}_dfn{dfn}_dfd{dfd}_q{q:e}"),
            func: "fdtri".into(),
            dfn,
            dfd,
            arg: q,
            tail: true,
        });
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
    let mut ledger = CompareLedger::new("diff_special_fdtr", &ARMS);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let Some((scipy_v, rust_v)) = ledger.pair(
            arm,
            &case.case_id,
            oracle.value,
            fsci_eval(&case.func, case.dfn, case.dfd, case.arg),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        let scale = if case.tail {
            scipy_v.abs()
        } else {
            scipy_v.abs().max(1.0)
        };
        let rel_diff = abs_diff / scale;
        max_abs_overall = max_abs_overall.max(abs_diff);
        max_rel_overall = max_rel_overall.max(rel_diff);

        let pass = match arm {
            "fdtr" | "fdtrc" => abs_diff <= CDF_TOL,
            "fdtri" => abs_diff <= PPF_TOL_REL * scale,
            _ => false,
        };
        ledger.compared(arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_fdtr".into(),
        category: "scipy.special.fdtr/fdtrc/fdtri".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
    // Each arm has its own case set; each must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
