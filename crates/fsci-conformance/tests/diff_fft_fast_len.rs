#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.fft.next_fast_len` and
//! `prev_fast_len` (5-smooth real-FFT-friendly lengths).
//!
//! Resolves [frankenscipy-wiz6v]. fsci uses {2, 3, 5} factors; scipy's
//! `real=True` flag selects the same scheme.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{next_fast_len, prev_fast_len};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    target: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    target: usize,
    fsci: usize,
    scipy: usize,
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
    fs::create_dir_all(output_dir()).expect("create fast_len diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fast_len diff log");
    fs::write(path, json).expect("write fast_len diff log");
}

fn generate_query() -> OracleQuery {
    let targets: &[usize] = &[
        1, 2, 3, 5, 7, 8, 11, 13, 17, 23, 31, 64, 100, 127, 257, 500, 1000, 1023, 10000,
    ];
    let mut points = Vec::new();
    for &t in targets {
        for op in ["next", "prev"] {
            points.push(PointCase {
                case_id: format!("{op}_{t}"),
                op: op.into(),
                target: t,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy import fft

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    op = case["op"]
    t = int(case["target"])
    try:
        if op == "next":
            v = int(fft.next_fast_len(t, real=True))
        elif op == "prev":
            v = int(fft.prev_fast_len(t, real=True))
        else:
            v = None
        points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fast_len query");
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
                "failed to spawn python3 for fast_len oracle: {e}"
            );
            eprintln!("skipping fast_len oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fast_len oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fast_len oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fast_len oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for fast_len oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fast_len oracle failed: {stderr}"
        );
        eprintln!("skipping fast_len oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fast_len oracle JSON"))
}

#[test]
fn diff_fft_fast_len() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "next" => next_fast_len(case.target, true),
            "prev" => prev_fast_len(case.target, true),
            _ => continue,
        };
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            target: case.target,
            fsci: fsci_v,
            scipy: scipy_v,
            pass: fsci_v == scipy_v,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_fast_len".into(),
        category: "scipy.fft.next_fast_len + prev_fast_len".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "fast_len {} mismatch: target={} fsci={} scipy={}",
                d.op, d.target, d.fsci, d.scipy
            );
        }
    }

    assert!(
        all_pass,
        "scipy.fft.next_fast_len/prev_fast_len conformance failed: {} cases",
        diffs.len()
    );
}
