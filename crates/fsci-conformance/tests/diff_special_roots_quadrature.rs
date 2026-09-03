#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Gauss quadrature root/weight
//! functions: roots_legendre, roots_chebyt, roots_hermite,
//! roots_hermitenorm, roots_laguerre.
//!
//! Resolves [frankenscipy-9r4on]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{
    roots_chebyt, roots_hermite, roots_hermitenorm, roots_laguerre, roots_legendre,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    family: String,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    nodes: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
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
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create roots_quadrature diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize roots_quadrature diff log");
    fs::write(path, json).expect("write roots_quadrature diff log");
}

fn generate_query() -> OracleQuery {
    let families = [
        "roots_legendre",
        "roots_chebyt",
        "roots_hermite",
        "roots_hermitenorm",
        "roots_laguerre",
    ];
    let degrees = [3_usize, 5, 8];
    let mut points = Vec::new();
    for family in &families {
        for &n in &degrees {
            points.push(PointCase {
                case_id: format!("{family}_n{n}"),
                family: (*family).into(),
                n,
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
from scipy import special

def finite_vec_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; family = case["family"]; n = int(case["n"])
    try:
        fn = getattr(special, family)
        nodes, weights = fn(n)
        # sort by node value to give a permutation-stable comparison
        order = np.argsort(np.asarray(nodes, dtype=float))
        nodes = np.asarray(nodes, dtype=float)[order]
        weights = np.asarray(weights, dtype=float)[order]
        points.append({
            "case_id": cid,
            "nodes": finite_vec_or_none(nodes),
            "weights": finite_vec_or_none(weights),
        })
    except Exception:
        points.append({"case_id": cid, "nodes": None, "weights": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize roots_quadrature query");
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
                "failed to spawn python3 for roots_quadrature oracle: {e}"
            );
            eprintln!("skipping roots_quadrature oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open roots_quadrature oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "roots_quadrature oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping roots_quadrature oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for roots_quadrature oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "roots_quadrature oracle failed: {stderr}"
        );
        eprintln!("skipping roots_quadrature oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse roots_quadrature oracle JSON"))
}

fn sort_pairs(nodes: Vec<f64>, weights: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = nodes.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        nodes[a]
            .partial_cmp(&nodes[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_nodes: Vec<f64> = idx.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = idx.iter().map(|&i| weights[i]).collect();
    (sorted_nodes, sorted_weights)
}

#[test]
fn diff_special_roots_quadrature() {
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
        let Some(nodes_exp) = scipy_arm.nodes.as_ref() else {
            continue;
        };
        let Some(weights_exp) = scipy_arm.weights.as_ref() else {
            continue;
        };
        let (n_raw, w_raw) = match case.family.as_str() {
            "roots_legendre" => roots_legendre(case.n),
            "roots_chebyt" => roots_chebyt(case.n),
            "roots_hermite" => roots_hermite(case.n),
            "roots_hermitenorm" => roots_hermitenorm(case.n),
            "roots_laguerre" => roots_laguerre(case.n),
            _ => continue,
        };
        let (nodes, weights) = sort_pairs(n_raw, w_raw);
        if nodes.len() != nodes_exp.len() || weights.len() != weights_exp.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: case.family.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let d_nodes = nodes
            .iter()
            .zip(nodes_exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let d_weights = weights
            .iter()
            .zip(weights_exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let abs_d = d_nodes.max(d_weights);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            family: case.family.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_roots_quadrature".into(),
        category: "scipy.special.roots_* (Gauss-quadrature nodes & weights)".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "roots_quadrature conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
