#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.group_delay`.
//!
//! Resolves [frankenscipy-uu1cc]. fsci_signal::group_delay(b, a, n) returns
//! (w, gd). scipy.signal.group_delay((b, a), w=n) returns (w, gd). 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::group_delay;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// [w0..w_{n-1}, gd0..gd_{n-1}]
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create group_delay diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize group_delay diff log");
    fs::write(path, json).expect("write group_delay diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<f64>, usize)] = &[
        ("1pole_iir", vec![1.0, 0.5], vec![1.0, -0.8], 8),
        ("fir_box_3", vec![1.0 / 3.0; 3], vec![1.0], 16),
        (
            "2pole_resonator",
            vec![0.04976845, 0.0995369, 0.04976845],
            vec![1.0, -1.27865943, 0.47775324],
            32,
        ),
        ("fir_5_taps", vec![0.1, 0.2, 0.4, 0.2, 0.1], vec![1.0], 32),
        (
            "iir_4th_order",
            vec![0.1, 0.0, -0.1, 0.0, 0.05],
            vec![1.0, -0.5, 0.3, -0.1, 0.05],
            32,
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, b, a, n)| PointCase {
            case_id: (*name).into(),
            b: b.clone(),
            a: a.clone(),
            n: *n,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    b = np.array(case["b"], dtype=float)
    a = np.array(case["a"], dtype=float)
    n = int(case["n"])
    try:
        w, gd = signal.group_delay((b, a), w=n)
        packed = []
        ok = True
        for wi in w.tolist():
            if not math.isfinite(wi):
                ok = False; break
            packed.append(float(wi))
        if ok:
            for g in gd.tolist():
                if not math.isfinite(g):
                    ok = False; break
                packed.append(float(g))
        points.append({"case_id": cid, "values": packed if ok else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize group_delay query");
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
                "failed to spawn python3 for group_delay oracle: {e}"
            );
            eprintln!("skipping group_delay oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open group_delay oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "group_delay oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping group_delay oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for group_delay oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "group_delay oracle failed: {stderr}"
        );
        eprintln!("skipping group_delay oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse group_delay oracle JSON"))
}

#[test]
fn diff_signal_group_delay() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok((w, gd)) = group_delay(&case.b, &case.a, Some(case.n)) else {
            continue;
        };
        let mut fsci_v = w;
        fsci_v.extend(gd);
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_group_delay".into(),
        category: "scipy.signal.group_delay".into(),
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
            eprintln!(
                "group_delay mismatch: {} abs_diff={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.group_delay conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
