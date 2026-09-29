#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the inverse regularized
//! incomplete beta `scipy.special.betaincinv`.
//!
//! Resolves [frankenscipy-b2v5f]. Companion to
//! `diff_special_beta` (which covers betainc itself); the
//! inverse is the backbone of StudentT/F/Beta ppf paths in
//! fsci-stats.
//!
//! 8 (a, b) pairs × 7 q-values = 56 cases via subprocess.
//! Tolerances: 1e-9 rel against scipy_v with scale =
//! max(|scipy|, 1).
//!
//! Plus tail cases in the 1e-300 tail (frankenscipy-xzrpr), held to the same 1e-9 relative to
//! |SciPy| itself, since the roots are far below 1. Each was mpmath-checked (60 digits) with
//! SciPy within 2e-16 of it; points where SciPy is NaN or wrong are pinned to mpmath in
//! fsci-special's unit tests instead.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::betaincinv;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    b: f64,
    q: f64,
    /// Compared relative to |SciPy| (see the module docs).
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
    fs::create_dir_all(output_dir()).expect("create betaincinv diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize betaincinv diff log");
    fs::write(path, json).expect("write betaincinv diff log");
}

fn fsci_eval(a: f64, b: f64, q: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let pq = SpecialTensor::RealScalar(q);
    match betaincinv(&pa, &pb, &pq, RuntimeMode::Strict) {
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
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let mut points = Vec::new();
    for &(a, b) in &pairs {
        for &q in &qs {
            points.push(PointCase {
                case_id: format!("a{a}_b{b}_q{q}"),
                a,
                b,
                q,
                tail: false,
            });
        }
    }
    // frankenscipy-xzrpr: roots in the 1e-300 tail. Old fsci: 1.38e-99 for 1.06e-195,
    // 2.01e-33 for 3.59e-52, 2.10e-18 for 4.28e-26, 2.22e-14 for 2.2185e-14 (1.6e-3 off).
    let tail_cases: [(f64, f64, f64); 4] = [
        (
            1.165665503003491,
            14.55107334677996,
            1.1163949326526923e-226,
        ),
        (
            2.2050358848923453,
            21.82666602000194,
            1.4245691623898521e-111,
        ),
        (
            9.192797921606235,
            11.741000276999168,
            1.0440467452774665e-228,
        ),
        (17.227600317199048, 8.984617046148863, 6.90302558253208e-230),
    ];
    for (i, &(a, b, q)) in tail_cases.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("tail{i}_a{a}_b{b}_q{q:e}"),
            a,
            b,
            q,
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
    a = float(case["a"]); b = float(case["b"]); qv = float(case["q"])
    try:
        value = special.betaincinv(a, b, qv)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize betaincinv query");
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
                "failed to spawn python3 for betaincinv oracle: {e}"
            );
            eprintln!("skipping betaincinv oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open betaincinv oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "betaincinv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping betaincinv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for betaincinv oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "betaincinv oracle failed: {stderr}"
        );
        eprintln!("skipping betaincinv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse betaincinv oracle JSON"))
}

#[test]
fn diff_special_betaincinv() {
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
    let mut ledger = CompareLedger::new("diff_special_betaincinv", &["betaincinv"]);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let Some((scipy_v, rust_v)) = ledger.pair(
            "betaincinv",
            &case.case_id,
            oracle.value,
            fsci_eval(case.a, case.b, case.q),
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
        ledger.compared("betaincinv", &case.case_id, abs_diff <= TOL_REL * scale);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff,
            rel_diff,
            pass: abs_diff <= TOL_REL * scale,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_betaincinv".into(),
        category: "scipy.special.betaincinv".into(),
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
                "betaincinv mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special.betaincinv conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
    ledger.finish(query.points.len());
}
