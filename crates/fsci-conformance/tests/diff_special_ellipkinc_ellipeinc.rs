#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the incomplete elliptic
//! integrals `scipy.special.ellipkinc` (F(φ, m)) and
//! `scipy.special.ellipeinc` (E(φ, m)).
//!
//! Resolves [frankenscipy-ecqls]. Companion to
//! `diff_special_ellipk_ellipe` (complete forms) and
//! `diff_special_carlson` (Carlson symmetric form).
//!
//! 6 φ × 5 m = 30 cases × 2 functions = 60 cases via subprocess.
//! Tolerances: 1e-12 abs (canonical Carlson-based composition).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{ellipeinc, ellipkinc};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    phi: f64,
    m: f64,
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
    fs::create_dir_all(output_dir()).expect("create incomplete-elliptic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize incomplete-elliptic diff log");
    fs::write(path, json).expect("write incomplete-elliptic diff log");
}

fn fsci_eval(func: &str, phi: f64, m: f64) -> Option<f64> {
    let pphi = SpecialTensor::RealScalar(phi);
    let pm = SpecialTensor::RealScalar(m);
    let result = match func {
        "ellipkinc" => ellipkinc(&pphi, &pm, RuntimeMode::Strict),
        "ellipeinc" => ellipeinc(&pphi, &pm, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // φ ∈ (0, π/2] (canonical incomplete elliptic range);
    // m ∈ [0, 1) (parameter; ellipkinc diverges as m → 1 and
    // φ → π/2 simultaneously).
    let phis = [0.1_f64, 0.3, 0.5, 0.8, 1.0, PI / 2.0];
    let ms = [0.0_f64, 0.1, 0.3, 0.5, 0.7];
    let mut points = Vec::new();
    for &phi in &phis {
        for &m in &ms {
            for func in ["ellipkinc", "ellipeinc"] {
                points.push(PointCase {
                    case_id: format!("{func}_phi{phi:.3}_m{m}"),
                    func: func.to_string(),
                    phi,
                    m,
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
    cid = case["case_id"]
    func = case["func"]; phi = float(case["phi"]); m = float(case["m"])
    try:
        if func == "ellipkinc":   value = special.ellipkinc(phi, m)
        elif func == "ellipeinc": value = special.ellipeinc(phi, m)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize incomplete-elliptic query");
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
                "failed to spawn python3 for incomplete-elliptic oracle: {e}"
            );
            eprintln!("skipping incomplete-elliptic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open incomplete-elliptic oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "incomplete-elliptic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping incomplete-elliptic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for incomplete-elliptic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "incomplete-elliptic oracle failed: {stderr}"
        );
        eprintln!("skipping incomplete-elliptic oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse incomplete-elliptic oracle JSON"))
}

#[test]
fn diff_special_ellipkinc_ellipeinc() {
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
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.phi, case.m)
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
        test_id: "diff_special_ellipkinc_ellipeinc".into(),
        category: "scipy.special.ellipkinc/ellipeinc".into(),
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
                "incomplete-elliptic {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special incomplete-elliptic conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
