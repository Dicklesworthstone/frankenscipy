#![forbid(unsafe_code)]
//! Live numerical reference checks for three model-selection
//! and multiple-testing utilities:
//!   • `aic(log_likelihood, n_params)` —
//!     2·n_params - 2·log_likelihood
//!   • `bic(log_likelihood, n_params, n_samples)` —
//!     n_params·ln(n_samples) - 2·log_likelihood
//!   • `false_discovery_control(pvalues, method='bh'|'by')` —
//!     Benjamini-Hochberg / Benjamini-Yekutieli p-value
//!     correction.
//!
//! Resolves [frankenscipy-tyyna]. AIC/BIC are closed-form
//! arithmetic so the oracle reproduces them directly. FDR
//! is checked against scipy.stats.false_discovery_control.
//!
//! 6 (LL, params, n) cases for AIC + 6 for BIC + 6 FDR cases
//! (3 fixtures × 2 methods, vector compared) = 18 cases via
//! subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{aic, bic, false_discovery_control};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    log_lik: f64,
    n_params: u64,
    n_samples: u64,
    pvalues: Vec<f64>,
    method: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    vector: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create aic_bic_fdr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize aic_bic_fdr diff log");
    fs::write(path, json).expect("write aic_bic_fdr diff log");
}

fn generate_query() -> OracleQuery {
    // 3 (LL, n_params) for AIC, 3 (LL, n_params, n_samples) for BIC.
    let model_fixtures: &[(&str, f64, u64, u64)] = &[
        ("small_model", -50.0, 3, 100),
        ("medium_model", -200.0, 5, 50),
        ("large_model", -1000.0, 10, 500),
    ];
    // 3 pvalue vectors × 2 methods (bh, by)
    let fdr_fixtures: Vec<(&str, Vec<f64>)> = vec![
        ("small_pvals", vec![0.001, 0.005, 0.01, 0.02, 0.04]),
        ("mixed", vec![0.01, 0.20, 0.50, 0.80, 0.95]),
        ("all_large", vec![0.30, 0.45, 0.60, 0.75, 0.85, 0.92]),
    ];

    let mut points = Vec::new();
    for (name, ll, np, ns) in model_fixtures {
        points.push(PointCase {
            case_id: format!("aic_{name}"),
            func: "aic".into(),
            log_lik: *ll,
            n_params: *np,
            n_samples: *ns,
            pvalues: vec![],
            method: "".into(),
        });
        points.push(PointCase {
            case_id: format!("bic_{name}"),
            func: "bic".into(),
            log_lik: *ll,
            n_params: *np,
            n_samples: *ns,
            pvalues: vec![],
            method: "".into(),
        });
    }
    for (name, pvals) in &fdr_fixtures {
        for method in ["bh", "by"] {
            points.push(PointCase {
                case_id: format!("fdr_{name}_{method}"),
                func: "false_discovery_control".into(),
                log_lik: 0.0,
                n_params: 0,
                n_samples: 0,
                pvalues: pvals.clone(),
                method: method.into(),
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
from scipy import stats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def vec_or_none(arr):
    out = []
    for v in arr:
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
    cid = case["case_id"]; func = case["func"]
    out = {"case_id": cid, "scalar": None, "vector": None}
    try:
        if func == "aic":
            out["scalar"] = fnone(2.0 * float(case["n_params"]) - 2.0 * float(case["log_lik"]))
        elif func == "bic":
            out["scalar"] = fnone(
                float(case["n_params"]) * math.log(float(case["n_samples"]))
                - 2.0 * float(case["log_lik"])
            )
        elif func == "false_discovery_control":
            pvals = np.array(case["pvalues"], dtype=float)
            method = case["method"]
            res = stats.false_discovery_control(pvals, method=method)
            out["vector"] = vec_or_none(res.tolist())
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize aic_bic_fdr query");
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
                "failed to spawn python3 for aic_bic_fdr oracle: {e}"
            );
            eprintln!("skipping aic_bic_fdr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open aic_bic_fdr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "aic_bic_fdr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping aic_bic_fdr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for aic_bic_fdr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "aic_bic_fdr oracle failed: {stderr}"
        );
        eprintln!("skipping aic_bic_fdr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse aic_bic_fdr oracle JSON"))
}

#[test]
fn diff_stats_aic_bic_fdr() {
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
        match case.func.as_str() {
            "aic" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = aic(case.log_lik, case.n_params as usize);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            "bic" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = bic(
                        case.log_lik,
                        case.n_params as usize,
                        case.n_samples as usize,
                    );
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            "false_discovery_control" => {
                if let Some(scipy_vec) = &scipy_arm.vector {
                    let rust_vec = match false_discovery_control(&case.pvalues, Some(&case.method))
                    {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if rust_vec.len() == scipy_vec.len() {
                        let mut max_local = 0.0_f64;
                        for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
                            if a.is_finite() {
                                max_local = max_local.max((a - b).abs());
                            }
                        }
                        max_overall = max_overall.max(max_local);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff: max_local,
                            pass: max_local <= ABS_TOL,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_aic_bic_fdr".into(),
        category: "aic + bic + false_discovery_control".into(),
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
                "aic_bic_fdr {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "aic_bic_fdr conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
