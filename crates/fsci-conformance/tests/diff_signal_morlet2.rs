#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_signal::morlet2.
//!
//! Resolves [frankenscipy-myxfs]. scipy.signal.morlet2 was removed in
//! scipy 1.13+, so the oracle reproduces the canonical formula:
//!
//!   ψ(t) = π^(-1/4) / sqrt(s) · exp(i·w·t/s) · exp(-t²/(2s²))
//!   t    = arange(M) - (M-1)/2
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::morlet2;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Morlet2Case {
    case_id: String,
    m: usize,
    s: f64,
    w: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Morlet2Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Packed (re0, im0, re1, im1, …).
    packed: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create morlet2 diff dir");
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
    let probes: &[(usize, f64, f64)] = &[
        (16, 1.0, 5.0),
        (16, 2.0, 5.0),
        (32, 1.0, 5.0),
        (32, 4.0, 5.0),
        (32, 4.0, 6.0),
        (64, 8.0, 5.0),
        (64, 8.0, 10.0),
        (128, 16.0, 5.0),
    ];
    let points: Vec<Morlet2Case> = probes
        .iter()
        .enumerate()
        .map(|(i, &(m, s, w))| Morlet2Case {
            case_id: format!("p{i:02}_m{m}_s{s}_w{w}").replace('.', "p"),
            m,
            s,
            w,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def morlet2(M, s, w):
    x = np.arange(M, dtype=float) - (M - 1) / 2.0
    out = np.exp(1j * w * x / s) * np.exp(-0.5 * (x / s) ** 2)
    out *= np.pi ** (-0.25) / np.sqrt(s)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    M = int(case["m"]); s = float(case["s"]); w = float(case["w"])
    try:
        y = morlet2(M, s, w)
        packed = []
        for v in y.tolist():
            packed.append(float(v.real)); packed.append(float(v.imag))
        if all(math.isfinite(v) for v in packed):
            points.append({"case_id": cid, "packed": packed})
        else:
            points.append({"case_id": cid, "packed": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "packed": None})
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
                "failed to spawn python3 for morlet2 oracle: {e}"
            );
            eprintln!("skipping morlet2 oracle: python3 not available ({e})");
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
                "morlet2 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping morlet2 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for morlet2 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "morlet2 oracle failed: {stderr}"
        );
        eprintln!("skipping morlet2 oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse morlet2 oracle JSON"))
}

#[test]
fn diff_signal_morlet2() {
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
        let Some(expected) = arm.packed.as_ref() else {
            continue;
        };
        let out = morlet2(case.m, case.s, case.w);
        let mut packed = Vec::with_capacity(out.len() * 2);
        for &(re, im) in &out {
            packed.push(re);
            packed.push(im);
        }
        let abs_d = if packed.len() != expected.len() {
            f64::INFINITY
        } else {
            packed
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_morlet2".into(),
        category: "fsci_signal::morlet2 vs numpy formula (scipy.signal.morlet2 removed)".into(),
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
            eprintln!("morlet2 mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "morlet2 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
