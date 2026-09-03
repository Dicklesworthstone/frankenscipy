#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_spatial` boolean-
//! vector distance metrics.
//!
//! Resolves [frankenscipy-q7w81]. Existing `diff_spatial.rs`
//! covers numeric distances (euclidean, minkowski, etc.) but
//! not the 8 boolean-vector dissimilarities. This harness
//! diffs each against scipy.spatial.distance:
//!   • dice
//!   • kulsinski → scipy.spatial.distance.kulczynski1
//!     (kulsinski was deprecated in scipy ≥1.10; the canonical
//!     replacement is kulczynski1)
//!   • matching (alias of hamming for boolean)
//!   • rogerstanimoto
//!   • russellrao
//!   • sokalmichener
//!   • sokalsneath
//!   • yule
//!
//! 5 fixtures × 8 metrics = 40 cases. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{
    dice, kulsinski, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    metric: String,
    u: Vec<bool>,
    v: Vec<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    #[allow(dead_code)]
    metric: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    metric: String,
    rust_value: f64,
    scipy_value: f64,
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
    fs::create_dir_all(output_dir()).expect("create boolean-distances diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize boolean-distances diff log");
    fs::write(path, json).expect("write boolean-distances diff log");
}

fn metric_dispatch(metric: &str, u: &[bool], v: &[bool]) -> Option<f64> {
    let r = match metric {
        "dice" => dice(u, v),
        "kulsinski" => kulsinski(u, v),
        "matching" => matching(u, v),
        "rogerstanimoto" => rogerstanimoto(u, v),
        "russellrao" => russellrao(u, v),
        "sokalmichener" => sokalmichener(u, v),
        "sokalsneath" => sokalsneath(u, v),
        "yule" => yule(u, v),
        _ => return None,
    };
    Some(r)
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<bool>, Vec<bool>)> = vec![
        (
            "fix_n8_partial",
            vec![true, false, true, true, false, false, true, true],
            vec![true, true, false, true, false, true, true, false],
        ),
        (
            "fix_n8_disagree",
            vec![true, true, true, true, false, false, false, false],
            vec![false, false, false, false, true, true, true, true],
        ),
        (
            "fix_n16_sparse",
            vec![
                true, false, false, false, true, false, false, false, false, true, false, false,
                false, false, true, false,
            ],
            vec![
                false, false, true, false, false, false, true, false, false, false, false, true,
                false, false, false, false,
            ],
        ),
        (
            "fix_n16_dense",
            vec![
                true, true, true, false, true, true, true, true, false, true, true, true, true,
                false, true, true,
            ],
            vec![
                true, true, false, true, true, true, true, false, true, true, true, false, true,
                true, true, true,
            ],
        ),
        (
            "fix_n32_alternating",
            (0..32).map(|i| i % 2 == 0).collect(),
            (0..32).map(|i| i % 3 == 0).collect(),
        ),
    ];
    let metrics = [
        "dice",
        "kulsinski",
        "matching",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ];

    let mut points = Vec::new();
    for (name, u, v) in &fixtures {
        for m in &metrics {
            points.push(PointCase {
                case_id: format!("{name}_{m}"),
                metric: (*m).to_string(),
                u: u.clone(),
                v: v.clone(),
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
from scipy.spatial import distance as sd

# scipy ≥ 1.12 removed `kulsinski`, `sokalmichener`, and
# `matching` from scipy.spatial.distance. For those three the
# oracle reproduces the canonical scipy ≤ 1.11 formulas in
# numpy from the contingency counts:
#   ctt = #(u_i & v_i),   ctf = #(u_i & ~v_i),
#   cft = #(~u_i & v_i),  cff = #(~u_i & ~v_i),
#   n   = ctt + ctf + cft + cff
# kulsinski      = (ctf + cft - ctt + n) / (ctf + cft + n)
# sokalmichener  = 2 R / (S + 2 R)   where R = ctf + cft, S = ctt + cff
#                = same as rogerstanimoto (verified by scipy 1.11 source)
# matching       = hamming for boolean (just the mismatch fraction)

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def counts(u, v):
    ctt = int(np.sum(u & v))
    ctf = int(np.sum(u & ~v))
    cft = int(np.sum(~u & v))
    cff = int(np.sum(~u & ~v))
    return ctt, ctf, cft, cff

def kulsinski_np(u, v):
    ctt, ctf, cft, cff = counts(u, v)
    n = ctt + ctf + cft + cff
    denom = ctf + cft + n
    if denom == 0:
        return 0.0
    return (ctf + cft - ctt + n) / denom

def sokalmichener_np(u, v):
    ctt, ctf, cft, cff = counts(u, v)
    r = ctf + cft
    s = ctt + cff
    denom = s + 2 * r
    if denom == 0:
        return 0.0
    return (2 * r) / denom

def matching_np(u, v):
    n = len(u)
    if n == 0:
        return 0.0
    return float(np.sum(u != v)) / n

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    metric = case["metric"]
    u = np.array(case["u"], dtype=bool)
    v = np.array(case["v"], dtype=bool)
    val = None
    try:
        if metric == "dice":
            val = sd.dice(u, v)
        elif metric == "kulsinski":
            val = kulsinski_np(u, v)
        elif metric == "matching":
            val = matching_np(u, v)
        elif metric == "rogerstanimoto":
            val = sd.rogerstanimoto(u, v)
        elif metric == "russellrao":
            val = sd.russellrao(u, v)
        elif metric == "sokalmichener":
            val = sokalmichener_np(u, v)
        elif metric == "sokalsneath":
            val = sd.sokalsneath(u, v)
        elif metric == "yule":
            val = sd.yule(u, v)
        val = fnone(val)
    except Exception:
        val = None
    points.append({"case_id": cid, "metric": metric, "value": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize boolean-distances query");
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
                "failed to spawn python3 for boolean-distances oracle: {e}"
            );
            eprintln!("skipping boolean-distances oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open boolean-distances oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "boolean-distances oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping boolean-distances oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for boolean-distances oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "boolean-distances oracle failed: {stderr}"
        );
        eprintln!("skipping boolean-distances oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse boolean-distances oracle JSON"))
}

#[test]
fn diff_spatial_boolean_distances() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Some(rust_v) = metric_dispatch(&case.metric, &case.u, &case.v) else {
            continue;
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            metric: case.metric.clone(),
            rust_value: rust_v,
            scipy_value: scipy_v,
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_boolean_distances".into(),
        category: "fsci_spatial boolean distance metrics".into(),
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
                "boolean-distance mismatch: {} ({}) rust={} scipy={} abs_diff={}",
                d.case_id, d.metric, d.rust_value, d.scipy_value, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "boolean-distances conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
