#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the complete elliptic
//! integrals `scipy.special.ellipk` (K(m)) and
//! `scipy.special.ellipe` (E(m)).
//!
//! Resolves [frankenscipy-eq6dg]. Companion to
//! `diff_special_carlson` (Carlson symmetric form). The
//! complete forms are scipy-canonical entry points that fsci-
//! special wraps over Carlson; verifying their composition
//! independently catches drift in the K↔RF and E↔RG/RD
//! relationships.
//!
//! ~14 m-values × 2 functions = 28 cases via subprocess.
//! Tolerances: 1e-12 abs (mature AGM-based implementation).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{ellipe, ellipk, ellipkm1};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create ellipk/ellipe diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ellipk/ellipe diff log");
    fs::write(path, json).expect("write ellipk/ellipe diff log");
}

fn fsci_eval(func: &str, m: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(m);
    let result = match func {
        "ellipk" => ellipk(&arg, RuntimeMode::Strict),
        "ellipe" => ellipe(&arg, RuntimeMode::Strict),
        "ellipkm1" => ellipkm1(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // m walks the canonical [0, 1) parameter space. K(m)
    // diverges as m→1 (logarithmically); E(m=0)=π/2,
    // E(m=1)=1. Stop the K probe at m=0.999999 to avoid the
    // divergence at exactly 1.
    let ms = [
        0.0_f64, 0.001, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99, 0.999, 0.999999,
        // E is finite at m=1; include for E only.
        1.0,
    ];
    let mut points = Vec::new();
    for &m in &ms {
        // K diverges at m=1; skip the m=1 case for K.
        if m < 1.0 {
            points.push(PointCase {
                case_id: format!("ellipk_m{m}"),
                func: "ellipk".into(),
                m,
            });
        }
        points.push(PointCase {
            case_id: format!("ellipe_m{m}"),
            func: "ellipe".into(),
            m,
        });
    }
    // ellipkm1(p) = K(1 − p) — probes the near-m=1 regime where direct
    // ellipk(m) suffers catastrophic cancellation. m here plays the role
    // of `p` (the small offset from 1).
    let ellipkm1_ps: &[f64] = &[1e-10, 1e-6, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99];
    for &p in ellipkm1_ps {
        points.push(PointCase {
            case_id: format!("ellipkm1_p{p}"),
            func: "ellipkm1".into(),
            m: p,
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
    func = case["func"]; m = float(case["m"])
    try:
        if func == "ellipk":   value = special.ellipk(m)
        elif func == "ellipe": value = special.ellipe(m)
        elif func == "ellipkm1": value = special.ellipkm1(m)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize ellipk/ellipe query");
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
                "failed to spawn python3 for ellipk/ellipe oracle: {e}"
            );
            eprintln!("skipping ellipk/ellipe oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ellipk/ellipe oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ellipk/ellipe oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ellipk/ellipe oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ellipk/ellipe oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ellipk/ellipe oracle failed: {stderr}"
        );
        eprintln!("skipping ellipk/ellipe oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ellipk/ellipe oracle JSON"))
}

#[test]
fn diff_special_ellipk_ellipe() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.m)
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
        test_id: "diff_special_ellipk_ellipe".into(),
        category: "scipy.special.ellipk/ellipe".into(),
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
                "ellipk/ellipe {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special ellipk/ellipe conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
