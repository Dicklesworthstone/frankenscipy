#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the standard-normal
//! family `scipy.special.ndtr/ndtri/log_ndtr`.
//!
//! Resolves [frankenscipy-r4ner]. ndtr/ndtri/log_ndtr underpin
//! Normal/Lognormal/PowerNorm/PowerLognorm/SkewNorm/JohnsonSU/
//! JohnsonSB cdf and ppf paths in fsci-stats.
//!
//! Tolerances:
//!   ndtr     5e-13 abs (erfc helper lands ~1.24e-13 at |x|=5)
//!   log_ndtr 1e-6 abs (asymptotic series for x<-3 lands ~4.3e-7
//!                      off scipy at x=-5; tracked in
//!                      frankenscipy-0om9c)
//!   ndtri    1e-7 rel (Beasley-Springer-Moro rational; floor
//!                      widens near q∈{0, 1})
//!   ndtri_exp 2e-6 rel-scaled abs (Acklam log-tail inverse; finite
//!                         at y=-800 and y≈0 where exp(y) would fail)

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{log_ndtr, ndtr, ndtri, ndtri_exp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const NDTR_TOL: f64 = 5.0e-13;
const LOG_NDTR_TOL: f64 = 1.0e-6;
const NDTRI_TOL_REL: f64 = 1.0e-7;
const NDTRI_EXP_TOL_REL: f64 = 2.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: f64,
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
    fs::create_dir_all(output_dir()).expect("create ndtr family diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ndtr family diff log");
    fs::write(path, json).expect("write ndtr family diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "ndtr" => ndtr(&arg, RuntimeMode::Strict),
        "log_ndtr" => log_ndtr(&arg, RuntimeMode::Strict),
        "ndtri" => ndtri(&arg, RuntimeMode::Strict),
        "ndtri_exp" => ndtri_exp(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // ndtr/log_ndtr: x walks ±10σ to exercise both deep tails.
    let xs = [
        -10.0_f64, -5.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0, 10.0,
    ];
    // ndtri: q in (0, 1). Stop at 1e-9 from each edge — past
    // BSM's documented design domain.
    let qs = [
        1.0e-9_f64,
        1.0e-6,
        0.001,
        0.1,
        0.5,
        0.9,
        0.999,
        1.0 - 1.0e-6,
        1.0 - 1.0e-9,
    ];
    // ndtri_exp takes log-probabilities directly and must stay finite
    // in both tails where ndtri(exp(y)) loses information.
    let log_ps = [
        -1000.0_f64,
        -800.0,
        -100.0,
        -50.0,
        -10.0,
        -2.0,
        -1.0,
        -std::f64::consts::LN_2,
        -0.1,
        -1.0e-9,
        -1.0e-20,
    ];

    let mut points = Vec::new();
    for &x in &xs {
        for func in ["ndtr", "log_ndtr"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &q in &qs {
        points.push(PointCase {
            case_id: format!("ndtri_q{q}"),
            func: "ndtri".into(),
            x: q,
        });
    }
    for &log_p in &log_ps {
        points.push(PointCase {
            case_id: format!("ndtri_exp_logp{log_p}"),
            func: "ndtri_exp".into(),
            x: log_p,
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

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]; x = float(case["x"])
    try:
        if func == "ndtr":     value = special.ndtr(x)
        elif func == "log_ndtr":value = special.log_ndtr(x)
        elif func == "ndtri":  value = special.ndtri(x)
        elif func == "ndtri_exp": value = special.ndtri_exp(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize ndtr family query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for ndtr family oracle: {e}"
            );
            eprintln!("skipping ndtr family oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ndtr family oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ndtr family oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ndtr family oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ndtr family oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ndtr family oracle failed: {stderr}"
        );
        eprintln!("skipping ndtr family oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ndtr family oracle JSON"))
}

#[test]
fn diff_special_ndtr() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);

            let pass = match case.func.as_str() {
                "ndtr" => abs_diff <= NDTR_TOL,
                "log_ndtr" => abs_diff <= LOG_NDTR_TOL,
                "ndtri" => {
                    let scale = scipy_v.abs().max(1.0);
                    abs_diff <= NDTRI_TOL_REL * scale
                }
                "ndtri_exp" => {
                    let scale = scipy_v.abs().max(1.0);
                    abs_diff <= NDTRI_EXP_TOL_REL * scale
                }
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
        test_id: "diff_special_ndtr".into(),
        category: "scipy.special.ndtr/ndtri/ndtri_exp/log_ndtr".into(),
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
                "ndtr family {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special ndtr family conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
