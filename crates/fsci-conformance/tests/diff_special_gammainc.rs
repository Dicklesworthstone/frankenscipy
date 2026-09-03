#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the regularized
//! incomplete-gamma family
//! `scipy.special.gammainc/gammaincc/gammaincinv/gammainccinv`.
//!
//! Resolves [frankenscipy-5e55x]. These helpers are the backbone
//! of ChiSquared, Gamma, Erlang, Pearson3, Maxwell, Chi,
//! Nakagami, Rice, Poisson, NegBinomial, etc. cdf/ppf paths in
//! fsci-stats.
//!
//! Tolerances: 1e-12 abs for the regularized series (mature in
//! fsci); 1e-9 rel for the inverse.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{gammainc, gammaincc, gammainccinv, gammaincinv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const GAMMAINC_TOL: f64 = 1.0e-12;
const GAMMAINCINV_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    a: f64,
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
    fs::create_dir_all(output_dir()).expect("create gammainc family diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gammainc family diff log");
    fs::write(path, json).expect("write gammainc family diff log");
}

fn fsci_eval(func: &str, a: f64, x: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let px = SpecialTensor::RealScalar(x);
    let result = match func {
        "gammainc" => gammainc(&pa, &px, RuntimeMode::Strict),
        "gammaincc" => gammaincc(&pa, &px, RuntimeMode::Strict),
        "gammaincinv" => gammaincinv(&pa, &px, RuntimeMode::Strict),
        "gammainccinv" => gammainccinv(&pa, &px, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Walk shape parameter a across J-shaped (a<1), exponential
    // (a=1), bell-shape, and large-a regimes.
    let as_ = [0.5_f64, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0];
    // For gammainc/gammaincc the second argument is x ≥ 0.
    let xs = [0.001_f64, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0];
    // For gamma inverse functions the second argument is q in (0, 1).
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];

    let mut points = Vec::new();
    for &a in &as_ {
        for &x in &xs {
            for func in ["gammainc", "gammaincc"] {
                points.push(PointCase {
                    case_id: format!("{func}_a{a}_x{x}"),
                    func: func.to_string(),
                    a,
                    x,
                });
            }
        }
        for &q in &qs {
            for func in ["gammaincinv", "gammainccinv"] {
                points.push(PointCase {
                    case_id: format!("{func}_a{a}_q{q}"),
                    func: func.into(),
                    a,
                    x: q,
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

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]; a = float(case["a"]); x = float(case["x"])
    try:
        if func == "gammainc":      value = special.gammainc(a, x)
        elif func == "gammaincc":   value = special.gammaincc(a, x)
        elif func == "gammaincinv": value = special.gammaincinv(a, x)
        elif func == "gammainccinv": value = special.gammainccinv(a, x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize gammainc family query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for gammainc family oracle: {e}"
            );
            eprintln!("skipping gammainc family oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open gammainc family oracle script stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gammainc family oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gammainc family oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for gammainc family oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gammainc family oracle failed: {stderr}"
        );
        eprintln!("skipping gammainc family oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gammainc family oracle JSON"))
}

#[test]
fn diff_special_gammainc() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.a, case.x)
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
                "gammainc" | "gammaincc" => abs_diff <= GAMMAINC_TOL,
                "gammaincinv" | "gammainccinv" => {
                    let scale = scipy_v.abs().max(1.0);
                    abs_diff <= GAMMAINCINV_TOL_REL * scale
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
        test_id: "diff_special_gammainc".into(),
        category: "scipy.special.gammainc/gammaincc/gammaincinv/gammainccinv".into(),
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
                "gammainc family {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special gammainc family conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
