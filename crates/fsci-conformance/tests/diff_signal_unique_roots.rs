#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.signal.unique_roots.
//!
//! Resolves [frankenscipy-mvj7j]. The unique_roots port (0fe245d)
//! has deterministic anchor tests but no live scipy comparison.
//! This harness drives curated real-valued root vectors through
//! scipy.signal.unique_roots and diffs the unique roots and
//! multiplicities. Skips cleanly if scipy/python3 is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::unique_roots;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct UniqueRootsCase {
    case_id: String,
    p: Vec<f64>,
    tol: f64,
    rtype: String,
}

#[derive(Debug, Clone, Deserialize)]
struct UniqueRootsOracleResult {
    case_id: String,
    roots: Option<Vec<f64>>,
    mult: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    rtype: String,
    max_root_diff: f64,
    mult_match: bool,
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
    fs::create_dir_all(output_dir()).expect("create unique_roots diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize unique_roots diff log");
    fs::write(path, json).expect("write unique_roots diff log");
}

fn generate_cases() -> Vec<UniqueRootsCase> {
    let mut cases = Vec::new();

    let inputs: &[(&str, Vec<f64>, f64)] = &[
        ("all_distinct_default_tol", vec![1.0, 5.0, 10.0], 1e-3),
        ("close_cluster", vec![1.0, 1.0001, 1.0002, 5.0], 1e-3),
        ("triple_pair", vec![1.0, 1.0, 2.0, 2.0, 3.0], 0.0),
        ("wide_cluster", vec![0.0, 0.5, 1.0, 2.5, 5.0], 1.0),
        ("single", vec![3.14], 1e-3),
        ("two_close", vec![1.0, 1.0001], 1e-3),
        ("scattered", vec![0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 0.4),
    ];

    for (label, p, tol) in inputs {
        for rtype in ["min", "max", "avg"] {
            cases.push(UniqueRootsCase {
                case_id: format!("{label}_{rtype}"),
                p: p.clone(),
                tol: *tol,
                rtype: rtype.into(),
            });
        }
    }

    cases
}

fn scipy_oracle_or_skip(cases: &[UniqueRootsCase]) -> Vec<UniqueRootsOracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    p = np.asarray(c["p"], dtype=float)
    tol = float(c["tol"])
    rtype = c["rtype"]
    try:
        roots, mult = signal.unique_roots(p, tol=tol, rtype=rtype)
        roots_list = [float(v.real) if hasattr(v, "real") else float(v) for v in roots]
        mult_list = [int(v) for v in mult]
        results.append({
            "case_id": cid,
            "roots": roots_list,
            "mult": mult_list,
        })
    except Exception:
        results.append({"case_id": cid, "roots": None, "mult": None})

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize unique_roots cases");

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
                "failed to spawn python3 for unique_roots oracle: {e}"
            );
            eprintln!("skipping unique_roots oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open unique_roots oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "unique_roots oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping unique_roots oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for unique_roots oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "unique_roots oracle failed: {stderr}"
        );
        eprintln!("skipping unique_roots oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse unique_roots oracle JSON")
}

#[test]
fn diff_signal_unique_roots() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy unique_roots oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, UniqueRootsOracleResult> = oracle_results
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
        let (Some(scipy_roots), Some(scipy_mult)) = (&oracle.roots, &oracle.mult) else {
            continue;
        };

        let (rust_roots, rust_mult) = unique_roots(&case.p, case.tol, &case.rtype);

        // Sort both sides by root value for comparison (scipy returns
        // groups in greedy order; we return in sorted order; for unique
        // group representatives the multiset is order-independent).
        let mut rust_pairs: Vec<(f64, usize)> = rust_roots
            .iter()
            .copied()
            .zip(rust_mult.iter().copied())
            .collect();
        let mut scipy_pairs: Vec<(f64, usize)> = scipy_roots
            .iter()
            .copied()
            .zip(scipy_mult.iter().copied())
            .collect();
        rust_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scipy_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut max_root_diff = 0.0_f64;
        let mut mult_match = rust_pairs.len() == scipy_pairs.len();
        if mult_match {
            for (rp, sp) in rust_pairs.iter().zip(scipy_pairs.iter()) {
                let diff = (rp.0 - sp.0).abs();
                max_root_diff = max_root_diff.max(diff);
                if rp.1 != sp.1 {
                    mult_match = false;
                }
            }
        }

        let pass = mult_match && max_root_diff <= ABS_TOL;
        max_overall = max_overall.max(max_root_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rtype: case.rtype.clone(),
            max_root_diff,
            mult_match,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_unique_roots".into(),
        category: "scipy.signal.unique_roots".into(),
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
                "unique_roots mismatch: {} (rtype={}) max_diff={} mult_match={}",
                d.case_id, d.rtype, d.max_root_diff, d.mult_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.unique_roots conformance failed: {} cases, max_root_diff={}",
        diffs.len(),
        max_overall
    );
}
