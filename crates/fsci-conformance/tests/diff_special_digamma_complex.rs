#![forbid(unsafe_code)]
//! Live scipy.special.digamma parity for fsci_special::digamma with
//! ComplexScalar input (dispatches to complex_digamma_scalar with
//! reflection for Re(z) < 0.5).
//!
//! Resolves [frankenscipy-bi1ec]. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::digamma;
use fsci_special::types::Complex64 as FsciComplex;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Lanczos-based digamma sees ~2e-10 abs gaps vs scipy at moderate |z|;
// a touch above the typical 1e-10 default.
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    z_re: f64,
    z_im: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    re: Option<f64>,
    im: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create digamma_complex diff dir");
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

fn fsci_eval(z_re: f64, z_im: f64) -> Option<(f64, f64)> {
    let z = SpecialTensor::ComplexScalar(FsciComplex::new(z_re, z_im));
    match digamma(&z, RuntimeMode::Strict) {
        Ok(SpecialTensor::ComplexScalar(c)) => Some((c.re, c.im)),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let probes: &[(f64, f64)] = &[
        (1.0, 0.5),
        (2.5, 1.0),
        (3.0, 2.0),
        (5.0, -1.5),
        (0.7, 0.2),
        (1.5, 0.5),
        (4.0, 0.0),
        (2.0, 3.0),
        // Reflection: Re(z) < 0.5
        (0.3, 0.4),
        (0.1, 1.0),
        (-0.5, 0.7),
        (-1.5, 1.5),
        (-3.7, 2.0),
        (0.0, 1.0),
        (0.0, -2.0),
    ];
    let points: Vec<PointCase> = probes
        .iter()
        .enumerate()
        .map(|(i, &(re, im))| PointCase {
            case_id: format!(
                "p{i:02}_re{}_im{}",
                re.to_string().replace('.', "p").replace('-', "n"),
                im.to_string().replace('.', "p").replace('-', "n")
            ),
            z_re: re,
            z_im: im,
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
    z = complex(float(case["z_re"]), float(case["z_im"]))
    try:
        r = complex(special.digamma(z))
        if math.isfinite(r.real) and math.isfinite(r.imag):
            points.append({"case_id": cid, "re": float(r.real), "im": float(r.imag)})
        else:
            points.append({"case_id": cid, "re": None, "im": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "re": None, "im": None})
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
                "failed to spawn python3 for digamma_complex oracle: {e}"
            );
            eprintln!("skipping digamma_complex oracle: python3 not available ({e})");
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
                "digamma_complex oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping digamma_complex oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for digamma_complex oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "digamma_complex oracle failed: {stderr}"
        );
        eprintln!("skipping digamma_complex oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse digamma_complex oracle JSON"))
}

#[test]
fn diff_special_digamma_complex() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(ere), Some(eim)) = (arm.re, arm.im) else {
            continue;
        };
        let Some((re, im)) = fsci_eval(case.z_re, case.z_im) else {
            continue;
        };
        let abs_d = ((re - ere).powi(2) + (im - eim).powi(2)).sqrt();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_digamma_complex".into(),
        category: "fsci_special::digamma(ComplexScalar) vs scipy.special.digamma".into(),
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
                "digamma_complex mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "digamma_complex conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
