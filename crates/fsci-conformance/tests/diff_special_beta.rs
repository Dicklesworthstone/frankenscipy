#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Beta-function family
//! `scipy.special.beta/betaln/betainc`.
//!
//! Resolves [frankenscipy-10k8v]. Beta/betaln are used by Beta,
//! BetaPrime, F, Mielke, Burr, etc. distributions; betainc is
//! the regularized incomplete beta — backbone of all Beta- and
//! F-related cdfs in fsci-stats. No dedicated diff harness
//! existed.
//!
//! Tolerances: beta/betaln 1e-10 abs OR 1e-12 rel; betainc
//! 1e-12 abs (the canonical regularized series is mature in
//! fsci — used by StudentT, F, BetaDist cdf paths).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{beta, betainc, betaln};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const BETA_TOL_ABS: f64 = 1.0e-10;
const BETA_TOL_REL: f64 = 1.0e-12;
const BETAINC_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct TwoArgCase {
    case_id: String,
    func: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ThreeArgCase {
    case_id: String,
    a: f64,
    b: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    two_arg: Vec<TwoArgCase>,
    three_arg: Vec<ThreeArgCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    two_arg: Vec<OracleArm>,
    three_arg: Vec<OracleArm>,
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
    fs::create_dir_all(output_dir()).expect("create beta family diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize beta family diff log");
    fs::write(path, json).expect("write beta family diff log");
}

fn fsci_two_arg(func: &str, a: f64, b: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let result = match func {
        "beta" => beta(&pa, &pb, RuntimeMode::Strict),
        "betaln" => betaln(&pa, &pb, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn fsci_three_arg(a: f64, b: f64, x: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let px = SpecialTensor::RealScalar(x);
    let result = betainc(&pa, &pb, &px, RuntimeMode::Strict);
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let pairs = [
        (0.5_f64, 0.5),
        (1.0, 1.0),
        (2.0, 5.0),
        (5.0, 2.0),
        (3.0, 3.0),
        (10.0, 10.0),
        (0.3, 7.0),
        (50.0, 50.0),
    ];
    let xs = [0.001_f64, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999];

    let mut two_arg = Vec::new();
    for &(a, b) in &pairs {
        for func in ["beta", "betaln"] {
            two_arg.push(TwoArgCase {
                case_id: format!("{func}_a{a}_b{b}"),
                func: func.into(),
                a,
                b,
            });
        }
    }
    let mut three_arg = Vec::new();
    for &(a, b) in &pairs {
        for &x in &xs {
            three_arg.push(ThreeArgCase {
                case_id: format!("betainc_a{a}_b{b}_x{x}"),
                a,
                b,
                x,
            });
        }
    }
    OracleQuery { two_arg, three_arg }
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
two_arg = []
for case in q["two_arg"]:
    cid = case["case_id"]
    func = case["func"]; a = float(case["a"]); b = float(case["b"])
    try:
        if func == "beta":     value = special.beta(a, b)
        elif func == "betaln": value = special.betaln(a, b)
        else: value = None
        two_arg.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        two_arg.append({"case_id": cid, "value": None})
three_arg = []
for case in q["three_arg"]:
    cid = case["case_id"]
    a = float(case["a"]); b = float(case["b"]); x = float(case["x"])
    try:
        value = special.betainc(a, b, x)
        three_arg.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        three_arg.append({"case_id": cid, "value": None})
print(json.dumps({"two_arg": two_arg, "three_arg": three_arg}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize beta family query");
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
                "failed to spawn python3 for beta family oracle: {e}"
            );
            eprintln!("skipping beta family oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open beta family oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "beta family oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping beta family oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for beta family oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "beta family oracle failed: {stderr}"
        );
        eprintln!("skipping beta family oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse beta family oracle JSON"))
}

#[test]
fn diff_special_beta() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.two_arg.len(), query.two_arg.len());
    assert_eq!(oracle.three_arg.len(), query.three_arg.len());

    let two_map: HashMap<String, OracleArm> = oracle
        .two_arg
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let three_map: HashMap<String, OracleArm> = oracle
        .three_arg
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &query.two_arg {
        let oracle = two_map.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_two_arg(&case.func, case.a, case.b)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let scale = scipy_v.abs().max(1.0);
            let pass = abs_diff <= BETA_TOL_ABS || rel_diff <= BETA_TOL_REL * scale;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    for case in &query.three_arg {
        let oracle = three_map.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_three_arg(case.a, case.b, case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: "betainc".into(),
                abs_diff,
                rel_diff,
                pass: abs_diff <= BETAINC_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_beta".into(),
        category: "scipy.special.beta/betaln/betainc".into(),
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
                "beta family {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special beta family conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
