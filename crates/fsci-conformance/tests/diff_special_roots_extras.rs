#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Gauss-quadrature root/weight
//! variants not covered by diff_special_roots_quadrature:
//! roots_chebyu, roots_chebyc, roots_chebys, roots_sh_legendre,
//! roots_sh_chebyt, roots_sh_chebyu, roots_sh_jacobi, roots_genlaguerre,
//! roots_gegenbauer, roots_jacobi, and SciPy's legacy *_roots aliases.
//!
//! Resolves [frankenscipy-lhrxe]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{
    c_roots, cg_roots, h_roots, he_roots, j_roots, js_roots, l_roots, la_roots, p_roots, ps_roots,
    roots_chebyc, roots_chebys, roots_chebyu, roots_gegenbauer, roots_genlaguerre, roots_jacobi,
    roots_sh_chebyt, roots_sh_chebyu, roots_sh_jacobi, roots_sh_legendre, s_roots, t_roots,
    ts_roots, u_roots, us_roots,
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
    /// Meaningful for genlaguerre, gegenbauer, jacobi; stores p for sh_jacobi.
    alpha: f64,
    /// Meaningful for jacobi; stores q for sh_jacobi.
    beta: f64,
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
    fs::create_dir_all(output_dir()).expect("create roots_extras diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize roots_extras diff log");
    fs::write(path, json).expect("write roots_extras diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // 0-arg families that match scipy at the default tolerance,
    // including the scaled Chebyshev variants (frankenscipy-s2qfz fixed).
    for family in [
        "roots_chebyu",
        "roots_chebyc",
        "roots_chebys",
        "roots_sh_legendre",
        "roots_sh_chebyt",
        "roots_sh_chebyu",
    ] {
        for n in [3_usize, 5, 8] {
            points.push(PointCase {
                case_id: format!("{family}_n{n}"),
                family: family.into(),
                n,
                alpha: 0.0,
                beta: 0.0,
            });
        }
    }

    // genlaguerre(n, alpha)
    for n in [3_usize, 5, 8] {
        for alpha in [0.0_f64, 0.5, 1.5] {
            points.push(PointCase {
                case_id: format!("roots_genlaguerre_n{n}_a{alpha}"),
                family: "roots_genlaguerre".into(),
                n,
                alpha,
                beta: 0.0,
            });
        }
    }

    // gegenbauer(n, alpha) — alpha > -0.5 and not 0
    for n in [3_usize, 5, 8] {
        for alpha in [0.5_f64, 1.0, 2.0] {
            points.push(PointCase {
                case_id: format!("roots_gegenbauer_n{n}_a{alpha}"),
                family: "roots_gegenbauer".into(),
                n,
                alpha,
                beta: 0.0,
            });
        }
    }

    // jacobi(n, alpha, beta) — alpha, beta > -1
    for n in [3_usize, 5, 8] {
        for (alpha, beta) in [(0.0_f64, 0.0), (0.5, 0.3), (1.5, -0.5)] {
            points.push(PointCase {
                case_id: format!("roots_jacobi_n{n}_a{alpha}_b{beta}"),
                family: "roots_jacobi".into(),
                n,
                alpha,
                beta,
            });
        }
    }

    // shifted jacobi(n, p, q) — p-q > -1 and q > 0
    for n in [3_usize, 5, 8] {
        for (p, q) in [(0.5_f64, 1.25), (1.5, 0.75), (2.0, 2.0)] {
            points.push(PointCase {
                case_id: format!("roots_sh_jacobi_n{n}_p{p}_q{q}"),
                family: "roots_sh_jacobi".into(),
                n,
                alpha: p,
                beta: q,
            });
        }
    }

    // Legacy alias names SciPy keeps at scipy.special top level.
    for family in [
        "p_roots", "ps_roots", "t_roots", "ts_roots", "u_roots", "us_roots", "c_roots", "s_roots",
        "h_roots", "he_roots", "l_roots",
    ] {
        for n in [3_usize, 5, 8] {
            points.push(PointCase {
                case_id: format!("{family}_n{n}"),
                family: family.into(),
                n,
                alpha: 0.0,
                beta: 0.0,
            });
        }
    }
    for n in [3_usize, 5, 8] {
        for alpha in [0.0_f64, 0.5, 1.5] {
            points.push(PointCase {
                case_id: format!("la_roots_n{n}_a{alpha}"),
                family: "la_roots".into(),
                n,
                alpha,
                beta: 0.0,
            });
        }
    }
    for n in [3_usize, 5, 8] {
        for alpha in [0.5_f64, 1.0, 2.0] {
            points.push(PointCase {
                case_id: format!("cg_roots_n{n}_a{alpha}"),
                family: "cg_roots".into(),
                n,
                alpha,
                beta: 0.0,
            });
        }
    }
    for n in [3_usize, 5, 8] {
        for (alpha, beta) in [(0.0_f64, 0.0), (0.5, 0.3), (1.5, -0.5)] {
            points.push(PointCase {
                case_id: format!("j_roots_n{n}_a{alpha}_b{beta}"),
                family: "j_roots".into(),
                n,
                alpha,
                beta,
            });
        }
    }
    for n in [3_usize, 5, 8] {
        for (p, q) in [(0.5_f64, 1.25), (1.5, 0.75), (2.0, 2.0)] {
            points.push(PointCase {
                case_id: format!("js_roots_n{n}_p{p}_q{q}"),
                family: "js_roots".into(),
                n,
                alpha: p,
                beta: q,
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

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]; family = case["family"]
    n = int(case["n"])
    alpha = float(case["alpha"]); beta = float(case["beta"])
    try:
        fn = getattr(special, family)
        if family in ("roots_jacobi", "roots_sh_jacobi", "j_roots", "js_roots"):
            nodes, weights = fn(n, alpha, beta)
        elif family in ("roots_genlaguerre", "roots_gegenbauer", "la_roots", "cg_roots"):
            nodes, weights = fn(n, alpha)
        else:
            nodes, weights = fn(n)
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
    let query_json = serde_json::to_string(query).expect("serialize roots_extras query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for roots_extras oracle: {e}"
            );
            eprintln!("skipping roots_extras oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open roots_extras oracle script stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "roots_extras oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping roots_extras oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for roots_extras oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "roots_extras oracle failed: {stderr}"
        );
        eprintln!("skipping roots_extras oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse roots_extras oracle JSON"))
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
fn diff_special_roots_extras() {
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
            "roots_chebyu" => roots_chebyu(case.n),
            "roots_chebyc" => roots_chebyc(case.n),
            "roots_chebys" => roots_chebys(case.n),
            "roots_sh_legendre" => roots_sh_legendre(case.n),
            "roots_sh_chebyt" => roots_sh_chebyt(case.n),
            "roots_sh_chebyu" => roots_sh_chebyu(case.n),
            "roots_genlaguerre" => roots_genlaguerre(case.n, case.alpha),
            "roots_gegenbauer" => roots_gegenbauer(case.n, case.alpha),
            "roots_jacobi" => roots_jacobi(case.n, case.alpha, case.beta),
            "roots_sh_jacobi" => roots_sh_jacobi(case.n, case.alpha, case.beta),
            "p_roots" => p_roots(case.n),
            "ps_roots" => ps_roots(case.n),
            "t_roots" => t_roots(case.n),
            "ts_roots" => ts_roots(case.n),
            "u_roots" => u_roots(case.n),
            "us_roots" => us_roots(case.n),
            "c_roots" => c_roots(case.n),
            "s_roots" => s_roots(case.n),
            "h_roots" => h_roots(case.n),
            "he_roots" => he_roots(case.n),
            "l_roots" => l_roots(case.n),
            "la_roots" => la_roots(case.n, case.alpha),
            "cg_roots" => cg_roots(case.n, case.alpha),
            "j_roots" => j_roots(case.n, case.alpha, case.beta),
            "js_roots" => js_roots(case.n, case.alpha, case.beta),
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
        test_id: "diff_special_roots_extras".into(),
        category: "scipy.special.roots_* (extras)".into(),
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
        "roots_extras conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
