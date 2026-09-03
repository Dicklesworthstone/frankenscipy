#![forbid(unsafe_code)]
//! Live scipy parity for fsci_special::sph_harm_y.
//!
//! Resolves [frankenscipy-ibt3u]. New SciPy convention:
//! sph_harm_y(n, m, theta, phi) where theta is polar (colatitude)
//! and phi is azimuth. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::sph_harm_y;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    n: u32,
    m: i32,
    theta: f64,
    phi: f64,
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
    fs::create_dir_all(output_dir()).expect("create sph_harm_y diff dir");
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
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    // Covers positive, zero, and negative orders m. Negative m exercises
    // the Y_n^{-m} = (-1)^m·conj(Y_n^m) convention (frankenscipy-gojfp /
    // frankenscipy-39e3y, fixed).
    let probes: &[(u32, i32, f64, f64)] = &[
        (0, 0, FRAC_PI_2, 0.0),
        (0, 0, FRAC_PI_4, PI / 3.0),
        (1, 0, FRAC_PI_2, 0.0),
        (1, 1, FRAC_PI_2, 0.0),
        (1, 0, FRAC_PI_4, 0.5),
        (1, 1, FRAC_PI_4, 0.5),
        (1, -1, FRAC_PI_4, 0.5),
        (2, 0, FRAC_PI_4, 0.7),
        (2, 1, FRAC_PI_4, 0.7),
        (2, 2, FRAC_PI_4, 0.7),
        (2, -1, FRAC_PI_4, 0.7),
        (2, -2, FRAC_PI_4, 0.7),
        (3, 0, FRAC_PI_2, 1.0),
        (3, 2, FRAC_PI_2, 1.0),
        (3, 3, 0.3, 0.4),
        (3, -1, FRAC_PI_2, 1.0),
        (3, -2, 0.3, 0.4),
        (3, -3, 0.3, 0.4),
        (4, 0, 0.6, 0.8),
        (4, 2, 0.6, 0.8),
        (4, -3, 0.6, 0.8),
        (5, 3, 0.9, 1.2),
        (5, -4, 0.9, 1.2),
        (6, 4, 1.2, 0.5),
        (6, -5, 1.2, 0.5),
    ];
    let points: Vec<PointCase> = probes
        .iter()
        .enumerate()
        .map(|(i, &(n, m, theta, phi))| PointCase {
            case_id: format!("p{i:02}_n{n}_m{m}"),
            n,
            m,
            theta,
            phi,
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
    n = int(case["n"]); m = int(case["m"])
    theta = float(case["theta"]); phi = float(case["phi"])
    try:
        r = complex(special.sph_harm_y(n, m, theta, phi))
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
                "failed to spawn python3 for sph_harm_y oracle: {e}"
            );
            eprintln!("skipping sph_harm_y oracle: python3 not available ({e})");
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
                "sph_harm_y oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sph_harm_y oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for sph_harm_y oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sph_harm_y oracle failed: {stderr}"
        );
        eprintln!("skipping sph_harm_y oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sph_harm_y oracle JSON"))
}

#[test]
fn diff_special_sph_harm_y() {
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
        let actual = sph_harm_y(case.n, case.m, case.theta, case.phi);
        let abs_d = ((actual.re - ere).powi(2) + (actual.im - eim).powi(2)).sqrt();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_sph_harm_y".into(),
        category: "fsci_special::sph_harm_y vs scipy.special.sph_harm_y".into(),
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
            eprintln!("sph_harm_y mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "sph_harm_y conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
