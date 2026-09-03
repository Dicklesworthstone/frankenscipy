#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.zpk2tf`.
//!
//! Resolves [frankenscipy-aq0mh]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{ZpkCoeffs, zpk2tf};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    zeros_re: Vec<f64>,
    zeros_im: Vec<f64>,
    poles_re: Vec<f64>,
    poles_im: Vec<f64>,
    gain: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    n_b: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create zpk2tf diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize zpk2tf diff log");
    fs::write(path, json).expect("write zpk2tf diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64)] = &[
        (
            "real_zeros_complex_poles",
            vec![1.0, -1.0],
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.5, -0.5],
            2.0,
        ),
        (
            "all_real",
            vec![0.5, -0.3],
            vec![0.0, 0.0],
            vec![0.2, 0.6],
            vec![0.0, 0.0],
            1.5,
        ),
        (
            "no_zeros",
            vec![],
            vec![],
            vec![0.8, 0.6],
            vec![0.0, 0.0],
            0.04,
        ),
        (
            "3pairs_complex",
            vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.2],
            vec![0.5, -0.5, 0.0],
            1.0,
        ),
        (
            "biquad_lp_zpk",
            vec![-1.0, -1.0],
            vec![0.0, 0.0],
            vec![0.5715, 0.5715],
            vec![0.2916, -0.2916],
            0.0675,
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, zr, zi, pr, pi, k)| PointCase {
            case_id: (*name).into(),
            zeros_re: zr.clone(),
            zeros_im: zi.clone(),
            poles_re: pr.clone(),
            poles_im: pi.clone(),
            gain: *k,
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

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    zr = case["zeros_re"]; zi = case["zeros_im"]
    pr = case["poles_re"]; pi = case["poles_im"]
    k = float(case["gain"])
    z = np.array([complex(r, i) for r, i in zip(zr, zi)], dtype=complex)
    p = np.array([complex(r, i) for r, i in zip(pr, pi)], dtype=complex)
    try:
        b, a = signal.zpk2tf(z, p, k)
        # b/a may have a tiny imaginary part from numerical noise — take real.
        bv = finite_vec_or_none(np.asarray(b).real)
        av = finite_vec_or_none(np.asarray(a).real)
        if bv is None or av is None:
            points.append({"case_id": cid, "values": None, "n_b": None})
        else:
            points.append({"case_id": cid, "values": bv + av, "n_b": len(bv)})
    except Exception:
        points.append({"case_id": cid, "values": None, "n_b": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize zpk2tf query");
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
                "failed to spawn python3 for zpk2tf oracle: {e}"
            );
            eprintln!("skipping zpk2tf oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open zpk2tf oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "zpk2tf oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping zpk2tf oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for zpk2tf oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "zpk2tf oracle failed: {stderr}"
        );
        eprintln!("skipping zpk2tf oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse zpk2tf oracle JSON"))
}

#[test]
fn diff_signal_zpk2tf() {
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
        let Some(n_b) = scipy_arm.n_b else { continue };
        let zpk = ZpkCoeffs {
            zeros_re: case.zeros_re.clone(),
            zeros_im: case.zeros_im.clone(),
            poles_re: case.poles_re.clone(),
            poles_im: case.poles_im.clone(),
            gain: case.gain,
        };
        let coeffs = zpk2tf(&zpk);
        let mut fsci_v = coeffs.b.clone();
        fsci_v.extend(coeffs.a.iter().copied());
        if fsci_v.len() != scipy_v.len() || coeffs.b.len() != n_b {
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
        test_id: "diff_signal_zpk2tf".into(),
        category: "scipy.signal.zpk2tf".into(),
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
            eprintln!("zpk2tf mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.zpk2tf conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
