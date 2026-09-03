#![forbid(unsafe_code)]
//! Live scipy parity for fsci_opt::rosen_hess and rosen_hess_prod.
//!
//! Resolves [frankenscipy-czl9x]. fsci's rosen / rosen_der already
//! ship a conformance harness (diff_opt_rosen). The Hessian helpers
//! rosen_hess(x) → n×n matrix and rosen_hess_prod(x, p) → n vector
//! had no dedicated coverage; this harness fills that gap. Both have
//! closed-form analytic expressions so the parity tolerance is tight
//! (rel 1e-12, abs 1e-14).

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{rosen_hess, rosen_hess_prod};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-12;
const ABS_TOL: f64 = 1.0e-14;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    x: Vec<f64>,
    /// Probe vector for hess_prod; ignored for hess
    p: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    /// scipy.optimize.rosen_hess(x) flattened row-major
    hess: Option<Vec<f64>>,
    /// scipy.optimize.rosen_hess_prod(x, p)
    hess_prod: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    n: usize,
    max_abs_diff_hess: f64,
    max_rel_diff_hess: f64,
    max_abs_diff_prod: f64,
    max_rel_diff_prod: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create rosen_hess diff dir");
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

fn build_query() -> OracleQuery {
    vec![
        ("ones_n2", vec![1.0, 1.0], vec![1.0, -1.0]),
        ("ones_n3", vec![1.0, 1.0, 1.0], vec![1.0, 0.5, -1.0]),
        ("ones_n5", vec![1.0; 5], vec![1.0, -1.0, 1.0, -1.0, 1.0]),
        ("standard_init_n2", vec![-1.2, 1.0], vec![0.5, 0.3]),
        (
            "standard_init_n5",
            vec![-1.2, 1.0, -1.2, 1.0, -1.2],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        ("near_zero_n3", vec![0.1, 0.05, -0.02], vec![1.0, 1.0, 1.0]),
        ("large_negative_n2", vec![-3.0, -2.0], vec![-0.5, 2.0]),
        (
            "mixed_sign_n4",
            vec![1.5, -0.5, 0.25, -1.0],
            vec![0.7, -0.3, 1.2, 0.8],
        ),
        ("zero_vec_n3", vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]),
        (
            "near_minimum_n4",
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
    ]
    .into_iter()
    .map(|(case_id, x, p)| CasePoint {
        case_id: case_id.into(),
        x,
        p,
    })
    .collect::<Vec<_>>()
    .into_iter()
    .fold(OracleQuery { points: Vec::new() }, |mut q, c| {
        q.points.push(c);
        q
    })
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np
from scipy.optimize import rosen_hess, rosen_hess_prod

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        x = np.array(c["x"], dtype=float)
        p = np.array(c["p"], dtype=float)
        h = rosen_hess(x)
        hp = rosen_hess_prod(x, p)
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(hp)):
            out.append({"case_id": cid, "hess": None, "hess_prod": None})
        else:
            out.append({
                "case_id": cid,
                "hess": [float(v) for v in h.flatten()],
                "hess_prod": [float(v) for v in hp],
            })
    except Exception:
        out.append({"case_id": cid, "hess": None, "hess_prod": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping rosen_hess oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rosen_hess oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping rosen_hess oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

fn compare_pair(actual: f64, expected: f64) -> (f64, f64) {
    let abs_d = (actual - expected).abs();
    let denom = expected.abs().max(1.0e-300);
    (abs_d, abs_d / denom)
}

#[test]
fn diff_opt_rosen_hess_and_prod() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let (Some(exp_hess), Some(exp_prod)) = (o.hess.as_ref(), o.hess_prod.as_ref()) else {
            continue;
        };

        let n = case.x.len();
        let h_fsci = rosen_hess(&case.x);
        let p_fsci = rosen_hess_prod(&case.x, &case.p);

        if h_fsci.len() != n || h_fsci.iter().any(|row| row.len() != n) {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                n,
                max_abs_diff_hess: f64::INFINITY,
                max_rel_diff_hess: f64::INFINITY,
                max_abs_diff_prod: f64::INFINITY,
                max_rel_diff_prod: f64::INFINITY,
                pass: false,
                note: format!(
                    "hess shape mismatch: fsci {}x{}",
                    h_fsci.len(),
                    h_fsci.first().map_or(0, |r| r.len())
                ),
            });
            continue;
        }
        if exp_hess.len() != n * n {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                n,
                max_abs_diff_hess: f64::INFINITY,
                max_rel_diff_hess: f64::INFINITY,
                max_abs_diff_prod: f64::INFINITY,
                max_rel_diff_prod: f64::INFINITY,
                pass: false,
                note: format!("scipy hess flat length {} != n*n {}", exp_hess.len(), n * n),
            });
            continue;
        }

        let mut max_abs_h = 0.0_f64;
        let mut max_rel_h = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let (abs_d, rel_d) = compare_pair(h_fsci[i][j], exp_hess[i * n + j]);
                max_abs_h = max_abs_h.max(abs_d);
                max_rel_h = max_rel_h.max(rel_d);
            }
        }

        let mut max_abs_p = 0.0_f64;
        let mut max_rel_p = 0.0_f64;
        if p_fsci.len() != exp_prod.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                n,
                max_abs_diff_hess: max_abs_h,
                max_rel_diff_hess: max_rel_h,
                max_abs_diff_prod: f64::INFINITY,
                max_rel_diff_prod: f64::INFINITY,
                pass: false,
                note: format!(
                    "hess_prod length mismatch: fsci={} scipy={}",
                    p_fsci.len(),
                    exp_prod.len()
                ),
            });
            continue;
        }
        for (a, e) in p_fsci.iter().zip(exp_prod.iter()) {
            let (abs_d, rel_d) = compare_pair(*a, *e);
            max_abs_p = max_abs_p.max(abs_d);
            max_rel_p = max_rel_p.max(rel_d);
        }

        let hess_pass = max_rel_h <= REL_TOL || max_abs_h <= ABS_TOL;
        let prod_pass = max_rel_p <= REL_TOL || max_abs_p <= ABS_TOL;
        let pass = hess_pass && prod_pass;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            n,
            max_abs_diff_hess: max_abs_h,
            max_rel_diff_hess: max_rel_h,
            max_abs_diff_prod: max_abs_p,
            max_rel_diff_prod: max_rel_p,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_rosen_hess_and_prod".into(),
        category: "fsci_opt::{rosen_hess, rosen_hess_prod} vs scipy.optimize".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "rosen_hess mismatch: {} hess_rel={} hess_abs={} prod_rel={} prod_abs={} note={}",
                d.case_id,
                d.max_rel_diff_hess,
                d.max_abs_diff_hess,
                d.max_rel_diff_prod,
                d.max_abs_diff_prod,
                d.note
            );
        }
    }

    assert!(
        all_pass,
        "rosen_hess/rosen_hess_prod parity failed: {} cases",
        diffs.len()
    );
}
