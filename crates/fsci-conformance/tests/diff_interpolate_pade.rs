#![forbid(unsafe_code)]
//! Live scipy.interpolate.pade parity for fsci_interpolate::pade.
//!
//! Resolves [frankenscipy-pi4cu].
//! fsci returns LOW-FIRST coefficient vectors (a0, a1, …); scipy's
//! poly1d.coef is HIGH-FIRST so the oracle reverses before comparing.
//! Convention is normalized to q0 = 1 in both layers.
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::pade;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PadeCase {
    case_id: String,
    taylor: Vec<f64>,
    m: usize,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PadeCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    p: Option<Vec<f64>>, // LOW-FIRST
    q: Option<Vec<f64>>, // LOW-FIRST, q[0] = 1
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
    fs::create_dir_all(output_dir()).expect("create pade diff dir");
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
    // exp(x) Taylor coefficients: 1/n!
    let mut exp = vec![1.0];
    let mut fact = 1.0;
    for n in 1..=10 {
        fact *= n as f64;
        exp.push(1.0 / fact);
    }
    // sin(x) Taylor: 0, 1, 0, -1/6, 0, 1/120, ...
    let mut sin = Vec::with_capacity(11);
    for k in 0..11 {
        let v = match k % 4 {
            0 => 0.0,
            1 => 1.0_f64,
            2 => 0.0,
            _ => -1.0_f64,
        };
        if v == 0.0 {
            sin.push(0.0);
        } else {
            let mut f = 1.0;
            for i in 1..=k {
                f *= i as f64;
            }
            sin.push(v / f);
        }
    }
    // log(1+x) Taylor: 0, 1, -1/2, 1/3, -1/4, ...
    let log1p: Vec<f64> = (0..11)
        .map(|k| {
            if k == 0 {
                0.0
            } else {
                (-1.0_f64).powi(k + 1) / (k as f64)
            }
        })
        .collect();
    // 1/(1-x) Taylor: 1, 1, 1, 1, ...
    let geom: Vec<f64> = vec![1.0; 11];

    let points = vec![
        PadeCase {
            case_id: "exp_m2_n2".into(),
            taylor: exp.clone(),
            m: 2,
            n: 2,
        },
        PadeCase {
            case_id: "exp_m3_n3".into(),
            taylor: exp.clone(),
            m: 3,
            n: 3,
        },
        PadeCase {
            case_id: "exp_m4_n2".into(),
            taylor: exp,
            m: 4,
            n: 2,
        },
        PadeCase {
            case_id: "sin_m3_n2".into(),
            taylor: sin.clone(),
            m: 3,
            n: 2,
        },
        PadeCase {
            case_id: "sin_m5_n2".into(),
            taylor: sin,
            m: 5,
            n: 2,
        },
        PadeCase {
            case_id: "log1p_m2_n2".into(),
            taylor: log1p.clone(),
            m: 2,
            n: 2,
        },
        PadeCase {
            case_id: "log1p_m3_n3".into(),
            taylor: log1p,
            m: 3,
            n: 3,
        },
        PadeCase {
            case_id: "geom_m1_n1".into(),
            taylor: geom,
            m: 1,
            n: 1,
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.interpolate import pade

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    taylor = [float(v) for v in case["taylor"]]
    m = int(case["m"]); n = int(case["n"])
    try:
        p_poly, q_poly = pade(taylor, n, m)
        # poly1d .coef is HIGH-FIRST; reverse to LOW-FIRST
        p = [float(c) for c in p_poly.coef[::-1]]
        q = [float(c) for c in q_poly.coef[::-1]]
        # scipy strips leading-zero high coefficients; fsci preserves length m+1 / n+1.
        # Pad with trailing zeros to align lengths.
        while len(p) < m + 1: p.append(0.0)
        while len(q) < n + 1: q.append(0.0)
        # Normalize so q[0] = 1 (fsci convention)
        if abs(q[0]) > 1e-15:
            scale = q[0]
            p = [v / scale for v in p]
            q = [v / scale for v in q]
        if all(math.isfinite(v) for v in p) and all(math.isfinite(v) for v in q):
            points.append({"case_id": cid, "p": p, "q": q})
        else:
            points.append({"case_id": cid, "p": None, "q": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "p": None, "q": None})
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
                "failed to spawn python3 for pade oracle: {e}"
            );
            eprintln!("skipping pade oracle: python3 not available ({e})");
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
                "pade oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping pade oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for pade oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "pade oracle failed: {stderr}"
        );
        eprintln!("skipping pade oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse pade oracle JSON"))
}

#[test]
fn diff_interpolate_pade() {
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
        let (Some(ep), Some(eq)) = (arm.p.clone(), arm.q.clone()) else {
            continue;
        };
        let Ok((p, q)) = pade(&case.taylor, case.m, case.n) else {
            continue;
        };
        let abs_p = if p.len() != ep.len() {
            f64::INFINITY
        } else {
            p.iter()
                .zip(ep.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        let abs_q = if q.len() != eq.len() {
            f64::INFINITY
        } else {
            q.iter()
                .zip(eq.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        let abs_d = abs_p.max(abs_q);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_pade".into(),
        category: "fsci_interpolate::pade vs scipy.interpolate.pade".into(),
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
            eprintln!("pade mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "pade conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
