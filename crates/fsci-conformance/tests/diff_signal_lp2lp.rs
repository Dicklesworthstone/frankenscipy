#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.lp2lp(b, a, wo)`.
//!
//! Resolves [frankenscipy-5rg2r]. The lp2lp port shipped in
//! 25aea8e has 4 closed-form anchor tests (identity, 1st-order,
//! 2nd-order, validation) but no live scipy oracle. This harness
//! drives 8 (b, a, wo) cases through scipy via subprocess and
//! asserts byte-stable agreement at tol 1e-12. Skips cleanly if
//! scipy/python3 is unavailable.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::lp2lp;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Lp2lpCase {
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create lp2lp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lp2lp diff log");
    fs::write(path, json).expect("write lp2lp diff log");
}

fn generate_cases() -> Vec<Lp2lpCase> {
    vec![
        Lp2lpCase {
            case_id: "identity_first_order".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            wo: 1.0,
        },
        Lp2lpCase {
            case_id: "first_order_compress".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            wo: 2.0,
        },
        Lp2lpCase {
            case_id: "first_order_stretch".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            wo: 0.5,
        },
        Lp2lpCase {
            case_id: "second_order_canonical".into(),
            b: vec![1.0, 2.0],
            a: vec![1.0, 3.0, 4.0],
            wo: 2.0,
        },
        Lp2lpCase {
            case_id: "third_order_butter".into(),
            b: vec![1.0],
            a: vec![1.0, 2.0, 2.0, 1.0],
            wo: 3.0,
        },
        Lp2lpCase {
            case_id: "high_order_b_equal_a".into(),
            b: vec![0.5, 1.0, 1.5, 2.0, 2.5],
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            wo: 1.5,
        },
        Lp2lpCase {
            case_id: "high_order_b_shorter".into(),
            b: vec![1.0, 2.0],
            a: vec![1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            wo: 0.7,
        },
        Lp2lpCase {
            case_id: "wo_large".into(),
            b: vec![1.0],
            a: vec![1.0, 2.0, 1.0],
            wo: 100.0,
        },
    ]
}

fn scipy_oracle_or_skip(cases: &[Lp2lpCase]) -> Vec<OracleResult> {
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
        b, a = signal.lp2lp(c["b"], c["a"], wo=float(c["wo"]))
        results.append({
            "case_id": cid,
            "b": [float(v) for v in b],
            "a": [float(v) for v in a],
        })
    except Exception:
        results.append({"case_id": cid, "b": None, "a": None})
print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize lp2lp cases");

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
                "failed to spawn python3 for lp2lp oracle: {e}"
            );
            eprintln!("skipping lp2lp oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open lp2lp oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lp2lp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lp2lp oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for lp2lp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lp2lp oracle failed: {stderr}"
        );
        eprintln!("skipping lp2lp oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse lp2lp oracle JSON")
}

/// b then a in one vector, so the ledger's slice check sees a NaN or a length mismatch in either.
fn packed(b: &[f64], a: &[f64]) -> Vec<f64> {
    b.iter().chain(a).copied().collect()
}

#[test]
fn diff_signal_lp2lp() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);
    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy lp2lp oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_signal_lp2lp", &["lp2lp"]);

    for case in &cases {
        let oracle = oracle_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        let scipy_ba = match (&oracle.b, &oracle.a) {
            (Some(b), Some(a)) => Some((b, a)),
            _ => None,
        };
        let rust_ba = lp2lp(&case.b, &case.a, case.wo).ok();
        let scipy_v = scipy_ba.map(|(b, a)| packed(b, a));
        let rust_v = rust_ba.as_ref().map(|(b, a)| packed(b, a));
        let Some((scipy_v, rust_v)) = ledger.slices(
            "lp2lp",
            &case.case_id,
            scipy_v.as_deref(),
            rust_v.as_deref(),
        ) else {
            continue;
        };
        // The packed lengths agree; the b/a split must agree too.
        let nb = scipy_ba.map_or(0, |(b, _)| b.len());
        let split_matches = rust_ba.as_ref().map(|(b, _)| b.len()) == Some(nb);
        let (mut max_b_diff, mut max_a_diff) = (f64::INFINITY, f64::INFINITY);
        if split_matches {
            max_b_diff = 0.0_f64;
            for (rb, sb) in rust_v[..nb].iter().zip(&scipy_v[..nb]) {
                max_b_diff = max_b_diff.max((rb - sb).abs());
            }
            max_a_diff = 0.0_f64;
            for (ra, sa) in rust_v[nb..].iter().zip(&scipy_v[nb..]) {
                max_a_diff = max_a_diff.max((ra - sa).abs());
            }
        }
        let pass = max_b_diff <= ABS_TOL && max_a_diff <= ABS_TOL;
        max_overall = max_overall.max(max_b_diff).max(max_a_diff);
        ledger.compared("lp2lp", &case.case_id, pass);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_b_diff,
            max_a_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_lp2lp".into(),
        category: "scipy.signal.lp2lp".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
                "lp2lp mismatch: {} b_diff={} a_diff={}",
                d.case_id, d.max_b_diff, d.max_a_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.lp2lp conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(cases.len());
}
