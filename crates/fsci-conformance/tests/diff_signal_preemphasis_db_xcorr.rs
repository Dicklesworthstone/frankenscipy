#![forbid(unsafe_code)]
//! Live formula-derived numpy parity for fsci_signal helpers:
//! preemphasis, deemphasis, power_db, xcorr_coefficient.
//!
//! Resolves [frankenscipy-e2tnv]. All formulas are well-known and
//! direct; tolerance: 1e-12 abs.
//!
//!   preemphasis(x, α): y[0] = x[0]; y[i] = x[i] - α·x[i-1]
//!   deemphasis(x, α):  y[0] = x[0]; y[i] = x[i] + α·y[i-1]
//!   power_db(x, ref):  10·log10(mean(x²) / ref)
//!   xcorr_coefficient(x, y): Pearson correlation coefficient

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{deemphasis, power_db, preemphasis, xcorr_coefficient};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    x: Vec<f64>,
    y: Vec<f64>,
    coeff: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create preemphasis diff dir");
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
    let s_sin: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.4).sin()).collect();
    let s_mixed: Vec<f64> = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let s_pair_x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let s_pair_y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // perfect correlation
    let s_anti_y: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // anti-correlated
    let s_uncorrelated: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0];

    let mut points = Vec::new();
    for &coeff in &[0.0, 0.5, 0.95, 0.97] {
        for (label, data) in [("s8", &s8), ("sin16", &s_sin), ("mixed", &s_mixed)] {
            points.push(Case {
                case_id: format!("preemphasis_{label}_a{coeff}").replace('.', "p"),
                op: "preemphasis".into(),
                x: data.clone(),
                y: vec![],
                coeff,
            });
            points.push(Case {
                case_id: format!("deemphasis_{label}_a{coeff}").replace('.', "p"),
                op: "deemphasis".into(),
                x: data.clone(),
                y: vec![],
                coeff,
            });
        }
    }

    for &ref_p in &[1.0_f64, 0.5, 0.1] {
        for (label, data) in [("s8", &s8), ("sin16", &s_sin), ("mixed", &s_mixed)] {
            points.push(Case {
                case_id: format!("power_db_{label}_r{ref_p}").replace('.', "p"),
                op: "power_db".into(),
                x: data.clone(),
                y: vec![],
                coeff: ref_p,
            });
        }
    }

    points.push(Case {
        case_id: "xcorr_perfect".into(),
        op: "xcorr".into(),
        x: s_pair_x.clone(),
        y: s_pair_y,
        coeff: 0.0,
    });
    points.push(Case {
        case_id: "xcorr_anti".into(),
        op: "xcorr".into(),
        x: s_pair_x.clone(),
        y: s_anti_y,
        coeff: 0.0,
    });
    points.push(Case {
        case_id: "xcorr_uncorrelated".into(),
        op: "xcorr".into(),
        x: s_pair_x,
        y: s_uncorrelated,
        coeff: 0.0,
    });

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
    coeff = float(case["coeff"])
    try:
        if op == "preemphasis":
            y = np.zeros_like(x); y[0] = x[0]
            for i in range(1, len(x)):
                y[i] = x[i] - coeff * x[i-1]
            points.append({"case_id": cid, "scalar": None, "values": [float(v) for v in y]})
        elif op == "deemphasis":
            y = np.zeros_like(x); y[0] = x[0]
            for i in range(1, len(x)):
                y[i] = x[i] + coeff * y[i-1]
            points.append({"case_id": cid, "scalar": None, "values": [float(v) for v in y]})
        elif op == "power_db":
            ref_p = coeff
            p = float(np.mean(x ** 2))
            if p <= 0.0 or ref_p <= 0.0:
                points.append({"case_id": cid, "scalar": None, "values": None})
            else:
                v = 10.0 * math.log10(p / ref_p)
                points.append({"case_id": cid, "scalar": float(v), "values": None})
        elif op == "xcorr":
            y = np.array(case["y"], dtype=float)
            # Pearson correlation
            mx = x.mean(); my = y.mean()
            num = float(((x - mx) * (y - my)).sum())
            denom = float(math.sqrt(((x - mx)**2).sum() * ((y - my)**2).sum()))
            if denom == 0.0:
                points.append({"case_id": cid, "scalar": 0.0, "values": None})
            else:
                points.append({"case_id": cid, "scalar": num / denom, "values": None})
        else:
            points.append({"case_id": cid, "scalar": None, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "scalar": None, "values": None})
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
                "failed to spawn python3 for preemphasis oracle: {e}"
            );
            eprintln!("skipping preemphasis oracle: python3 not available ({e})");
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
                "preemphasis oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping preemphasis oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for preemphasis oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "preemphasis oracle failed: {stderr}"
        );
        eprintln!("skipping preemphasis oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse preemphasis oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_signal_preemphasis_db_xcorr() {
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

        let abs_d = match case.op.as_str() {
            "preemphasis" => {
                let Some(expected) = arm.values.as_ref() else {
                    continue;
                };
                let actual = preemphasis(&case.x, case.coeff);
                vec_max_diff(&actual, expected)
            }
            "deemphasis" => {
                let Some(expected) = arm.values.as_ref() else {
                    continue;
                };
                let actual = deemphasis(&case.x, case.coeff);
                vec_max_diff(&actual, expected)
            }
            "power_db" => {
                let Some(expected) = arm.scalar else {
                    continue;
                };
                let actual = power_db(&case.x, case.coeff);
                (actual - expected).abs()
            }
            "xcorr" => {
                let Some(expected) = arm.scalar else {
                    continue;
                };
                let actual = xcorr_coefficient(&case.x, &case.y);
                (actual - expected).abs()
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
        test_id: "diff_signal_preemphasis_db_xcorr".into(),
        category: "fsci_signal preemphasis/deemphasis/power_db/xcorr_coefficient vs numpy formula"
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
        "preemphasis_db_xcorr conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
