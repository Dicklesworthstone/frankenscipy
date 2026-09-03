#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.correlation_lags`.
//!
//! Resolves [frankenscipy-1469f]. The correlation_lags port shipped
//! in 28244b7 has 6 closed-form anchor cases and a fuzz harness
//! (f22c80e) but no live scipy oracle. This harness drives 12
//! (in1, in2) size pairs × 3 modes = 36 cases through scipy via
//! subprocess and asserts byte-for-byte equality (the result is
//! Vec<i64>, no tolerance needed). Skips cleanly if scipy/python3
//! is unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{CorrelationMode, correlation_lags};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct LagCase {
    case_id: String,
    in1_len: usize,
    in2_len: usize,
    mode: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    lags: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    rust_len: usize,
    scipy_len: usize,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create correlation_lags diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize correlation_lags diff log");
    fs::write(path, json).expect("write correlation_lags diff log");
}

fn parse_mode(label: &str) -> CorrelationMode {
    match label {
        "full" => CorrelationMode::Full,
        "same" => CorrelationMode::Same,
        "valid" => CorrelationMode::Valid,
        _ => unreachable!("invalid mode label: {label}"),
    }
}

fn generate_cases() -> Vec<LagCase> {
    let pairs: &[(usize, usize)] = &[
        (1, 1),
        (3, 3),
        (4, 4),
        (5, 5),
        (5, 3),
        (3, 5),
        (10, 1),
        (1, 10),
        (10, 7),
        (7, 10),
        (47, 815),
        (1024, 512),
    ];
    let mut cases = Vec::new();
    for (n1, n2) in pairs {
        for mode in ["full", "same", "valid"] {
            cases.push(LagCase {
                case_id: format!("{n1}x{n2}_{mode}"),
                in1_len: *n1,
                in2_len: *n2,
                mode: mode.to_string(),
            });
        }
    }
    cases
}

fn scipy_oracle_or_skip(cases: &[LagCase]) -> Vec<OracleResult> {
    let script = r#"
import json
import sys
from scipy.signal import correlation_lags

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    try:
        lags = correlation_lags(int(c["in1_len"]), int(c["in2_len"]), c["mode"])
        results.append({"case_id": cid, "lags": [int(v) for v in lags]})
    except Exception:
        results.append({"case_id": cid, "lags": None})
print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize correlation_lags cases");

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
                "failed to spawn python3 for correlation_lags oracle: {e}"
            );
            eprintln!("skipping correlation_lags oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open correlation_lags oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "correlation_lags oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping correlation_lags oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for correlation_lags oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "correlation_lags oracle failed: {stderr}"
        );
        eprintln!("skipping correlation_lags oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse correlation_lags oracle JSON")
}

#[test]
fn diff_signal_correlation_lags() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);
    if oracle_results.is_empty() {
        return;
    }
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy correlation_lags oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, OracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();

    for case in &cases {
        let oracle = oracle_map
            .get(&case.case_id)
            .expect("validated complete oracle map");
        let Some(scipy_lags) = &oracle.lags else {
            continue;
        };
        let rust_lags = correlation_lags(case.in1_len, case.in2_len, parse_mode(&case.mode));
        let pass = &rust_lags == scipy_lags;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rust_len: rust_lags.len(),
            scipy_len: scipy_lags.len(),
            pass,
        });
        if !pass {
            eprintln!(
                "correlation_lags mismatch [{}]: rust={rust_lags:?} scipy={scipy_lags:?}",
                case.case_id
            );
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_correlation_lags".into(),
        category: "scipy.signal.correlation_lags".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);

    assert!(all_pass, "scipy.signal.correlation_lags conformance failed");
}
