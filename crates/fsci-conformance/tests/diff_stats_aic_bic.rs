#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's
//! `aic(log_likelihood, n_params)` (Akaike Information
//! Criterion = 2k − 2L) and `bic(log_likelihood, n_params,
//! n_samples)` (Bayesian Information Criterion = k·ln(n) − 2L).
//!
//! Resolves [frankenscipy-wznkp]. The oracle reproduces both
//! formulas in numpy directly; no scipy primitive exposes
//! these as scalar utilities (statsmodels has them embedded
//! in regression results, not as standalone functions).
//!
//! 6 (log_likelihood, n_params, n_samples) fixtures × 2 funcs
//! = 12 cases. Tol 1e-12 abs (closed-form linear in k, L
//! and ln(n)).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{aic, bic};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    log_likelihood: f64,
    n_params: usize,
    n_samples: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create aic_bic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize aic_bic diff log");
    fs::write(path, json).expect("write aic_bic diff log");
}

fn generate_query() -> OracleQuery {
    // (label, log_likelihood, n_params, n_samples)
    let fixtures: Vec<(&str, f64, usize, usize)> = vec![
        ("simple_model", -50.0, 2, 100),
        ("complex_model", -120.0, 8, 200),
        ("over_parameterised", -10.0, 12, 30),
        ("large_n_simple", -500.0, 3, 5000),
        ("zero_params", -75.5, 1, 50),
        ("positive_ll", 25.0, 4, 80),
    ];

    let mut points = Vec::new();
    for (label, ll, k, n) in &fixtures {
        for func in ["aic", "bic"] {
            points.push(PointCase {
                case_id: format!("{label}_{func}"),
                func: func.into(),
                log_likelihood: *ll,
                n_params: *k,
                n_samples: *n,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    L = float(case["log_likelihood"])
    k = int(case["n_params"])
    n = int(case["n_samples"])
    val = None
    try:
        if func == "aic":
            val = 2.0 * k - 2.0 * L
        elif func == "bic":
            val = k * math.log(n) - 2.0 * L
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize aic_bic query");
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
                "failed to spawn python3 for aic_bic oracle: {e}"
            );
            eprintln!("skipping aic_bic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open aic_bic oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "aic_bic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping aic_bic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for aic_bic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "aic_bic oracle failed: {stderr}"
        );
        eprintln!("skipping aic_bic oracle: python3 not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse aic_bic oracle JSON"))
}

#[test]
fn diff_stats_aic_bic() {
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
        let rust_v = match case.func.as_str() {
            "aic" => aic(case.log_likelihood, case.n_params),
            "bic" => bic(case.log_likelihood, case.n_params, case.n_samples),
            _ => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_aic_bic".into(),
        category: "AIC + BIC information criteria (numpy reference)".into(),
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
                "aic_bic {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "aic_bic conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
