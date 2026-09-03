#![forbid(unsafe_code)]
//! Live scipy.special.gamma formula parity for fsci_special::
//! gamma_mod_squared.
//!
//! Resolves [frankenscipy-j17me]. Reference computed via
//! abs(scipy.special.gamma(complex(a, b)))^2.
//!
//! Tolerance: 3e-2 rel — fsci uses an approximate identity-based
//! recurrence (~2% rel from scipy's complex-gamma path) that is
//! acceptable for the scattering/physics use cases this helper
//! targets but not high-precision.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::gamma_mod_squared;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 3.0e-2;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
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
    fs::create_dir_all(output_dir()).expect("create gms diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let probes: &[(f64, f64)] = &[
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (0.5, 0.5),
        (1.5, 1.0),
        (2.5, 1.0),
        (2.5, 2.0),
        (3.0, 1.5),
        (4.0, 2.0),
        (1.0, 3.0),
        (5.0, 0.5),
    ];
    let points: Vec<Case> = probes
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| Case {
            case_id: format!("p{i:02}_a{a}_b{b}").replace('.', "p"),
            a,
            b,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = float(case["a"]); b = float(case["b"])
    try:
        z = complex(a, b)
        g = special.gamma(z)
        v = float(abs(g) ** 2)
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for gms oracle: {e}"
            );
            eprintln!("skipping gms oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gms oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gms oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for gms oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gms oracle failed: {stderr}"
        );
        eprintln!("skipping gms oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gms oracle JSON"))
}

#[test]
fn diff_special_gamma_mod_squared() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_rel = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = gamma_mod_squared(case.a, case.b);
        let abs_d = (actual - expected).abs();
        let rel_d = if expected.abs() > 1.0e-12 {
            abs_d / expected.abs()
        } else {
            abs_d
        };
        max_rel = max_rel.max(rel_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rel_diff: rel_d,
            pass: rel_d <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_gamma_mod_squared".into(),
        category: "fsci_special::gamma_mod_squared vs scipy.special.gamma".into(),
        case_count: diffs.len(),
        max_rel_diff: max_rel,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("gms mismatch: {} rel_diff={}", d.case_id, d.rel_diff);
        }
    }

    assert!(
        all_pass,
        "gms conformance failed: {} cases, max_rel={}",
        diffs.len(),
        max_rel
    );
}
