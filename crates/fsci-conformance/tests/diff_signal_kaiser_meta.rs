#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Kaiser metadata
//! helpers `scipy.signal.kaiser_atten` and `scipy.signal.kaiser_beta`.
//!
//! Resolves [frankenscipy-y5u2q]. Both helpers were ported in
//! 93fe366 with closed-form anchor tests, then patched in
//! bbc1621 (kaiser_atten(0, w) boundary). Neither had a live
//! scipy oracle. This harness drives:
//!   - kaiser_atten: 6 (numtaps, width) cases × 4 widths = 24
//!   - kaiser_beta : 8 attenuation values spanning the three
//!     branches plus boundaries
//!
//! through scipy via subprocess oracle and asserts byte-stable
//! agreement at tol 1e-12. Skips cleanly if scipy/python3 is
//! unavailable.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{kaiser_atten, kaiser_beta};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct AttenCase {
    case_id: String,
    numtaps: usize,
    width: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BetaCase {
    case_id: String,
    a: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    atten_cases: Vec<AttenCase>,
    beta_cases: Vec<BetaCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct AttenOracleResult {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct BetaOracleResult {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    atten: Vec<AttenOracleResult>,
    beta: Vec<BetaOracleResult>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    family: String,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create kaiser_meta diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize kaiser_meta diff log");
    fs::write(path, json).expect("write kaiser_meta diff log");
}

fn generate_query() -> OracleQuery {
    let mut atten_cases = Vec::new();
    let numtaps_set: &[(&str, usize)] = &[
        ("zero_boundary", 0),
        ("single", 1),
        ("five", 5),
        ("twentyone", 21),
        ("hundred_one", 101),
        ("five_hundred", 500),
    ];
    let widths: &[(f64, &str)] = &[(0.01, "w001"), (0.05, "w005"), (0.1, "w010"), (0.4, "w040")];
    for (label, n) in numtaps_set {
        for (w, wlabel) in widths {
            atten_cases.push(AttenCase {
                case_id: format!("atten_{label}_{wlabel}"),
                numtaps: *n,
                width: *w,
            });
        }
    }

    // Beta: sweep across all three branches plus boundaries.
    let beta_cases: Vec<BetaCase> = [
        ("low_zero", 0.0),
        ("low_below_21", 15.0),
        ("low_just_under_21", 20.99),
        ("middle_at_21", 21.0),
        ("middle_30", 30.0),
        ("middle_50", 50.0),
        ("high_just_above_50", 50.0001),
        ("high_100", 100.0),
    ]
    .iter()
    .map(|(label, a)| BetaCase {
        case_id: format!("beta_{label}"),
        a: *a,
    })
    .collect();

    OracleQuery {
        atten_cases,
        beta_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy import signal

q = json.load(sys.stdin)
atten_results = []
for c in q["atten_cases"]:
    cid = c["case_id"]
    try:
        v = float(signal.kaiser_atten(int(c["numtaps"]), float(c["width"])))
    except Exception:
        v = None
    atten_results.append({"case_id": cid, "value": v})
beta_results = []
for c in q["beta_cases"]:
    cid = c["case_id"]
    try:
        v = float(signal.kaiser_beta(float(c["a"])))
    except Exception:
        v = None
    beta_results.append({"case_id": cid, "value": v})
print(json.dumps({"atten": atten_results, "beta": beta_results}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize kaiser_meta query");

    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for kaiser_meta oracle: {e}"
            );
            eprintln!("skipping kaiser_meta oracle: python3 not available ({e})");
            return None;
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open kaiser_meta oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "kaiser_meta oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kaiser_meta oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for kaiser_meta oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "kaiser_meta oracle failed: {stderr}"
        );
        eprintln!("skipping kaiser_meta oracle: scipy not available\n{stderr}");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse kaiser_meta oracle JSON"))
}

#[test]
fn diff_signal_kaiser_meta() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    assert_eq!(
        oracle.atten.len(),
        query.atten_cases.len(),
        "scipy kaiser_atten oracle returned partial coverage"
    );
    assert_eq!(
        oracle.beta.len(),
        query.beta_cases.len(),
        "scipy kaiser_beta oracle returned partial coverage"
    );

    let atten_oracle: HashMap<String, AttenOracleResult> = oracle
        .atten
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let beta_oracle: HashMap<String, BetaOracleResult> = oracle
        .beta
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.atten_cases {
        let scipy_value = atten_oracle
            .get(&case.case_id)
            .and_then(|r| r.value)
            .expect("scipy kaiser_atten produced a value for every case");
        let rust_value = kaiser_atten(case.numtaps, case.width);
        let diff = (rust_value - scipy_value).abs();
        let pass = diff <= ABS_TOL;
        max_overall = max_overall.max(diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            family: "kaiser_atten".into(),
            abs_diff: diff,
            pass,
        });
    }

    for case in &query.beta_cases {
        let scipy_value = beta_oracle
            .get(&case.case_id)
            .and_then(|r| r.value)
            .expect("scipy kaiser_beta produced a value for every case");
        let rust_value = kaiser_beta(case.a);
        let diff = (rust_value - scipy_value).abs();
        let pass = diff <= ABS_TOL;
        max_overall = max_overall.max(diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            family: "kaiser_beta".into(),
            abs_diff: diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_kaiser_meta".into(),
        category: "scipy.signal.kaiser_atten+kaiser_beta".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.kaiser_atten/kaiser_beta conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
