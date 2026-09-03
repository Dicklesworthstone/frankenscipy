#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_signal time-domain summary
//! statistics: rms, signal_energy, signal_power, crest_factor,
//! peak_to_peak, zero_crossing_rate.
//!
//! Resolves [frankenscipy-j0uvi]. 1e-12 abs (exact float arithmetic).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{
    crest_factor, peak_to_peak, rms, signal_energy, signal_power, zero_crossing_rate,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create signal_stats diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize signal_stats diff log");
    fs::write(path, json).expect("write signal_stats diff log");
}

fn generate_query() -> OracleQuery {
    let signals: &[(&str, Vec<f64>)] = &[
        (
            "sine_64",
            (0..64).map(|i| ((i as f64) * 0.1).sin()).collect(),
        ),
        ("ramp_30", (1..=30).map(|i| i as f64).collect()),
        (
            "noisy_50",
            (0..50)
                .map(|i| ((i as f64) * 0.3).sin() + ((i as f64) * 1.7).cos() * 0.5)
                .collect(),
        ),
        (
            "alt_20",
            (0..20)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
        (
            "decay_40",
            (0..40).map(|i| (-(i as f64) / 8.0).exp()).collect(),
        ),
        ("constant_10", vec![3.5; 10]),
    ];

    let mut points = Vec::new();
    for (label, x) in signals {
        for op in [
            "rms",
            "signal_energy",
            "signal_power",
            "crest_factor",
            "peak_to_peak",
            "zero_crossing_rate",
        ] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                x: x.clone(),
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
import numpy as np

def zero_crossing_rate(x):
    # Number of sign-or-zero transitions divided by (len-1).
    if len(x) < 2:
        return 0.0
    s = (x >= 0).astype(int)
    crossings = int(np.sum(s[1:] != s[:-1]))
    return crossings / (len(x) - 1)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = np.array(case["x"], dtype=float)
    try:
        if op == "rms":
            v = float(np.sqrt(np.mean(x**2)))
        elif op == "signal_energy":
            v = float(np.sum(x**2))
        elif op == "signal_power":
            v = float(np.sum(x**2) / len(x)) if len(x) > 0 else 0.0
        elif op == "crest_factor":
            r = float(np.sqrt(np.mean(x**2))) if len(x) > 0 else 0.0
            peak = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
            v = peak / r if r != 0.0 else 0.0
        elif op == "peak_to_peak":
            v = float(np.max(x) - np.min(x)) if len(x) > 0 else 0.0
        elif op == "zero_crossing_rate":
            v = zero_crossing_rate(x)
        else:
            v = None
        if v is None or not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": float(v)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize signal_stats query");
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
                "failed to spawn python3 for signal_stats oracle: {e}"
            );
            eprintln!("skipping signal_stats oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open signal_stats oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "signal_stats oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping signal_stats oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for signal_stats oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "signal_stats oracle failed: {stderr}"
        );
        eprintln!("skipping signal_stats oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse signal_stats oracle JSON"))
}

#[test]
fn diff_signal_stats_summaries() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "rms" => rms(&case.x),
            "signal_energy" => signal_energy(&case.x),
            "signal_power" => signal_power(&case.x),
            "crest_factor" => crest_factor(&case.x),
            "peak_to_peak" => peak_to_peak(&case.x),
            "zero_crossing_rate" => zero_crossing_rate(&case.x),
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
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
        test_id: "diff_signal_stats_summaries".into(),
        category: "fsci_signal time-domain summary stats vs numpy".into(),
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
        "signal_stats_summaries conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
