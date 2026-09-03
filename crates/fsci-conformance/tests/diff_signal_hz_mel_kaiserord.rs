#![forbid(unsafe_code)]
//! Live scipy/librosa parity for fsci_signal::{hz_to_mel, mel_to_hz,
//! kaiserord}.
//!
//! Resolves [frankenscipy-8m6nm].
//!
//! - `hz_to_mel(hz)` / `mel_to_hz(mel)`: HTK formula
//!   mel = 2595 · log10(1 + hz/700)  /  hz = 700 · (10^(mel/2595) - 1)
//!   Compare against numpy formula at 1e-12 abs.
//! - `kaiserord(ripple, width)`: returns (numtaps, beta) — Kaiser's
//!   empirical FIR-design formula. Compare against scipy.signal.kaiserord.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{hz_to_mel, kaiserord, mel_to_hz};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "hz2mel" | "mel2hz" | "kord"
    x: f64,
    /// kord
    ripple: f64,
    width: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// scalar f64 (hz2mel/mel2hz) or [numtaps_as_f64, beta]
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create hz_mel_kord diff dir");
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
    let mut points = Vec::new();
    // hz_to_mel
    for &hz in &[0.0_f64, 100.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0] {
        points.push(Case {
            case_id: format!("hz2mel_{hz}"),
            op: "hz2mel".into(),
            x: hz,
            ripple: 0.0,
            width: 0.0,
        });
    }
    // mel_to_hz
    for &mel in &[0.0_f64, 100.0, 500.0, 1500.0, 2595.0, 3000.0, 4000.0] {
        points.push(Case {
            case_id: format!("mel2hz_{mel}"),
            op: "mel2hz".into(),
            x: mel,
            ripple: 0.0,
            width: 0.0,
        });
    }
    // kaiserord — ripple in dB, width as fraction of Nyquist
    for &ripple in &[30.0_f64, 40.0, 50.0, 60.0, 80.0, 100.0] {
        for &width in &[0.05_f64, 0.1, 0.2, 0.3] {
            points.push(Case {
                case_id: format!("kord_r{ripple}_w{width}"),
                op: "kord".into(),
                x: 0.0,
                ripple,
                width,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.signal import kaiserord

def hz2mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel2hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "hz2mel":
            v = hz2mel(float(case["x"]))
            points.append({"case_id": cid, "values": [v]})
        elif op == "mel2hz":
            v = mel2hz(float(case["x"]))
            points.append({"case_id": cid, "values": [v]})
        elif op == "kord":
            r = float(case["ripple"]); w = float(case["width"])
            numtaps, beta = kaiserord(r, w)
            points.append({"case_id": cid, "values": [float(numtaps), float(beta)]})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for hz_mel_kord oracle: {e}"
            );
            eprintln!("skipping hz_mel_kord oracle: python3 not available ({e})");
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
                "hz_mel_kord oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hz_mel_kord oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for hz_mel_kord oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hz_mel_kord oracle failed: {stderr}"
        );
        eprintln!("skipping hz_mel_kord oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hz_mel_kord oracle JSON"))
}

#[test]
fn diff_signal_hz_mel_kaiserord() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let abs_d = match case.op.as_str() {
            "hz2mel" => (hz_to_mel(case.x) - expected[0]).abs(),
            "mel2hz" => (mel_to_hz(case.x) - expected[0]).abs(),
            "kord" => {
                let Ok((numtaps, beta)) = kaiserord(case.ripple, case.width) else {
                    continue;
                };
                let d_taps = (numtaps as f64 - expected[0]).abs();
                let d_beta = (beta - expected[1]).abs();
                d_taps.max(d_beta)
            }
            _ => continue,
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_hz_mel_kaiserord".into(),
        category: "fsci_signal::{hz_to_mel, mel_to_hz, kaiserord} vs scipy/numpy".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "hz_mel_kaiserord conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
