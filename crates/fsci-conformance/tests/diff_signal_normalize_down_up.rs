#![forbid(unsafe_code)]
//! Live numpy parity for fsci_signal::{normalize_signal, normalize_minmax,
//! downsample, upsample}.
//!
//! Resolves [frankenscipy-eilzb]. All formulas direct.
//!   normalize_signal(x): (x - mean) / std   (ddof=0)
//!   normalize_minmax(x): (x - min) / (max - min)
//!   downsample(x, factor): x[::factor]
//!   upsample(x, factor): zero-insert factor-1 zeros between samples
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{downsample, normalize_minmax, normalize_signal, upsample};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    x: Vec<f64>,
    factor: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create normalize_du diff dir");
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
    let s8: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let s_sin: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.4).sin() + 1.0).collect();
    let s_mixed: Vec<f64> = vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -2.0];

    let mut points = Vec::new();
    for (label, data) in [("s8", &s8), ("sin16", &s_sin), ("mixed", &s_mixed)] {
        points.push(Case {
            case_id: format!("norm_signal_{label}"),
            op: "normalize_signal".into(),
            x: data.clone(),
            factor: 0,
        });
        points.push(Case {
            case_id: format!("norm_minmax_{label}"),
            op: "normalize_minmax".into(),
            x: data.clone(),
            factor: 0,
        });
    }
    for (label, data) in [("s8", &s8), ("sin16", &s_sin)] {
        for &factor in &[2_usize, 3, 4] {
            points.push(Case {
                case_id: format!("downsample_{label}_f{factor}"),
                op: "downsample".into(),
                x: data.clone(),
                factor,
            });
            points.push(Case {
                case_id: format!("upsample_{label}_f{factor}"),
                op: "upsample".into(),
                x: data.clone(),
                factor,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = np.array(case["x"], dtype=float)
    factor = int(case["factor"])
    try:
        if op == "normalize_signal":
            mean = float(np.mean(x))
            std = float(np.std(x))  # ddof=0
            if std == 0.0:
                v = (x - mean).tolist()  # zero std → zero result
            else:
                v = ((x - mean) / std).tolist()
        elif op == "normalize_minmax":
            mn = float(np.min(x)); mx = float(np.max(x))
            rng = mx - mn
            if rng == 0.0:
                v = [0.0] * len(x)
            else:
                v = ((x - mn) / rng).tolist()
        elif op == "downsample":
            v = x[::factor].tolist()
        elif op == "upsample":
            n = len(x)
            out = np.zeros(n * factor)
            out[::factor] = x
            v = out.tolist()
        else:
            v = None
        if v is None or any(not math.isfinite(t) for t in v):
            points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": [float(t) for t in v]})
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
                "failed to spawn python3 for normalize_du oracle: {e}"
            );
            eprintln!("skipping normalize_du oracle: python3 not available ({e})");
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
                "normalize_du oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping normalize_du oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for normalize_du oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "normalize_du oracle failed: {stderr}"
        );
        eprintln!("skipping normalize_du oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse normalize_du oracle JSON"))
}

#[test]
fn diff_signal_normalize_down_up() {
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
        let actual: Vec<f64> = match case.op.as_str() {
            "normalize_signal" => normalize_signal(&case.x),
            "normalize_minmax" => normalize_minmax(&case.x),
            "downsample" => downsample(&case.x, case.factor),
            "upsample" => upsample(&case.x, case.factor),
            _ => continue,
        };
        let abs_d = if actual.len() != expected.len() {
            f64::INFINITY
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
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
        test_id: "diff_signal_normalize_down_up".into(),
        category: "fsci_signal normalize_signal/normalize_minmax/downsample/upsample vs numpy"
            .into(),
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
        "normalize_du conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
