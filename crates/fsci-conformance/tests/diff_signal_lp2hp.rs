#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.lp2hp(b, a, wo)`.
//!
//! Resolves [frankenscipy-dhrog]. The lp2hp port shipped in
//! 2863292 has 4 closed-form anchor tests (identity, 1st-order,
//! 2nd-order, validation) but no live scipy oracle. This harness
//! drives 8 (b, a, wo) cases through scipy via subprocess and
//! asserts byte-stable agreement at tol 1e-12. Skips cleanly if
//! scipy/python3 is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::lp2hp;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Lp2hpCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    wo: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    b: Option<Vec<f64>>,
    a: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_b_diff: f64,
    max_a_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    abs_tol: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create lp2hp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lp2hp diff log");
    fs::write(path, json).expect("write lp2hp diff log");
}

fn generate_cases() -> Vec<Lp2hpCase> {
    vec![
        Lp2hpCase {
            case_id: "identity_first_order".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            wo: 1.0,
        },
        Lp2hpCase {
            case_id: "first_order_compress".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            wo: 2.0,
        },
        Lp2hpCase {
            case_id: "first_order_stretch".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            wo: 0.5,
        },
        Lp2hpCase {
            case_id: "second_order_canonical".into(),
            b: vec![1.0, 2.0],
            a: vec![1.0, 3.0, 4.0],
            wo: 2.0,
        },
        Lp2hpCase {
            case_id: "third_order_butter".into(),
            b: vec![1.0],
            a: vec![1.0, 2.0, 2.0, 1.0],
            wo: 3.0,
        },
        Lp2hpCase {
            case_id: "high_order_b_equal_a".into(),
            b: vec![0.5, 1.0, 1.5, 2.0, 2.5],
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            wo: 1.5,
        },
        Lp2hpCase {
            case_id: "high_order_b_shorter".into(),
            b: vec![1.0, 2.0],
            a: vec![1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            wo: 0.7,
        },
        Lp2hpCase {
            case_id: "wo_large".into(),
            b: vec![1.0],
            a: vec![1.0, 2.0, 1.0],
            wo: 100.0,
        },
    ]
}

fn scipy_oracle_or_skip(cases: &[Lp2hpCase]) -> Vec<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    try:
        b, a = signal.lp2hp(c["b"], c["a"], wo=float(c["wo"]))
        results.append({
            "case_id": cid,
            "b": [float(v) for v in b],
            "a": [float(v) for v in a],
        })
    except Exception:
        results.append({"case_id": cid, "b": None, "a": None})
print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize lp2hp cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for lp2hp oracle: {e}"
            );
            eprintln!("skipping lp2hp oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open lp2hp oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lp2hp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lp2hp oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for lp2hp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lp2hp oracle failed: {stderr}"
        );
        eprintln!("skipping lp2hp oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse lp2hp oracle JSON")
}

#[test]
fn diff_signal_lp2hp() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);
    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy lp2hp oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &cases {
        let oracle = oracle_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        let (Some(scipy_b), Some(scipy_a)) = (&oracle.b, &oracle.a) else {
            continue;
        };
        let (rust_b, rust_a) = lp2hp(&case.b, &case.a, case.wo).expect("lp2hp");
        assert_eq!(
            rust_b.len(),
            scipy_b.len(),
            "{}: b length mismatch (rust={}, scipy={})",
            case.case_id,
            rust_b.len(),
            scipy_b.len()
        );
        assert_eq!(
            rust_a.len(),
            scipy_a.len(),
            "{}: a length mismatch (rust={}, scipy={})",
            case.case_id,
            rust_a.len(),
            scipy_a.len()
        );

        let mut max_b_diff = 0.0_f64;
        for (rb, sb) in rust_b.iter().zip(scipy_b.iter()) {
            max_b_diff = max_b_diff.max((rb - sb).abs());
        }
        let mut max_a_diff = 0.0_f64;
        for (ra, sa) in rust_a.iter().zip(scipy_a.iter()) {
            max_a_diff = max_a_diff.max((ra - sa).abs());
        }
        let pass = max_b_diff <= ABS_TOL && max_a_diff <= ABS_TOL;
        max_overall = max_overall.max(max_b_diff).max(max_a_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_b_diff,
            max_a_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_lp2hp".into(),
        category: "scipy.signal.lp2hp".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        abs_tol: ABS_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "lp2hp mismatch: {} b_diff={} a_diff={}",
                d.case_id, d.max_b_diff, d.max_a_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.lp2hp conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
