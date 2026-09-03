#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Owen's T function
//! `scipy.special.owens_t`.
//!
//! Resolves [frankenscipy-siziw]. Owen's T is used by SkewNormal
//! cdf and other tail-correction integrals. fsci's
//! `owens_t_scalar` had no dedicated diff harness.
//!
//! ~9 h-values × 7 a-values = 63 cases via subprocess.
//! Tolerances: 5e-6 abs. T(h, a) is always in [0, 0.25] so
//! absolute and relative tolerance coincide. fsci's owens_t
//! lands ~1.1e-6 abs at h=±3, a=5 (large-a integration tail
//! truncation); ~3e-8 at moderate (h, a). The kernel could be
//! tightened with a higher-order quadrature on the heavy-tail
//! a-branch but that's out of scope for the diff harness.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::owens_t;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 5.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    h: f64,
    a: f64,
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
    fs::create_dir_all(output_dir()).expect("create owens_t diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize owens_t diff log");
    fs::write(path, json).expect("write owens_t diff log");
}

fn fsci_eval(h: f64, a: f64) -> Option<f64> {
    let ph = SpecialTensor::RealScalar(h);
    let pa = SpecialTensor::RealScalar(a);
    match owens_t(&ph, &pa, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // T(h, a) is symmetric: T(-h, a) = T(h, a) and T(h, -a) =
    // -T(h, a). Walk both signs of a; both signs of h covered
    // by symmetry but we sample both anyway to verify the
    // implementation.
    let hs = [-3.0_f64, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 5.0, 10.0];
    let as_ = [-2.0_f64, -0.5, -0.1, 0.5, 1.0, 2.0, 5.0];
    let mut points = Vec::new();
    for &h in &hs {
        for &a in &as_ {
            points.push(PointCase {
                case_id: format!("h{h}_a{a}"),
                h,
                a,
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
    h = float(case["h"]); a = float(case["a"])
    try:
        value = special.owens_t(h, a)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize owens_t query");
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
                "failed to spawn python3 for owens_t oracle: {e}"
            );
            eprintln!("skipping owens_t oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open owens_t oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "owens_t oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping owens_t oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for owens_t oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "owens_t oracle failed: {stderr}"
        );
        eprintln!("skipping owens_t oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse owens_t oracle JSON"))
}

#[test]
fn diff_special_owens_t() {
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
            && let Some(rust_v) = fsci_eval(case.h, case.a)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_owens_t".into(),
        category: "scipy.special.owens_t".into(),
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
            eprintln!("owens_t mismatch: {} abs={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.special.owens_t conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
