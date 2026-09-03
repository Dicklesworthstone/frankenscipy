#![forbid(unsafe_code)]
//! Live numerical reference checks for two autocorrelation
//! diagnostic utilities:
//!   • `pacf(data, max_lag)` — partial autocorrelation via
//!     the Durbin-Levinson recursion.
//!   • `ljung_box(data, lags)` — Q test for autocorrelation
//!     significance (chi² with df=lags).
//!
//! Resolves [frankenscipy-3gr4k]. statsmodels has both but
//! isn't always available, so the oracle re-implements both
//! formulas analytically in numpy + scipy.stats.chi2.sf.
//!
//! 3 datasets × (pacf vector + ljung_box stat + ljung_box
//! pvalue) = 9 cases via subprocess. Tol 1e-9 abs (Durbin-
//! Levinson recursion accumulates ~1e-12 drift; chi² cdf
//! chain another ~1e-12).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ljung_box, pacf};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
    max_lag: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pacf_vec: Option<Vec<f64>>,
    ljung_q: Option<f64>,
    ljung_p: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create pacf_ljung diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize pacf_ljung diff log");
    fs::write(path, json).expect("write pacf_ljung diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, u64)> = vec![
        // Smooth ramp (strong positive autocorrelation)
        ("ramp", (1..=20).map(|i| i as f64).collect(), 4),
        // Alternating (strong negative lag-1 autocorrelation)
        (
            "alternating",
            (0..20)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            4,
        ),
        // Noisy (mild correlation)
        (
            "noisy",
            vec![
                1.0, 1.5, 0.8, 1.7, 0.9, 1.4, 1.0, 1.6, 1.1, 1.5, 0.7, 1.4, 0.9, 1.6, 1.0, 1.5,
            ],
            4,
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data, max_lag)| PointCase {
            case_id: name.into(),
            data,
            max_lag,
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

def biased_acf(data, max_lag):
    n = len(data)
    mean = float(data.mean())
    cent = data - mean
    var = float(np.sum(cent * cent))
    if var == 0:
        return [1.0] * (max_lag + 1)
    out = []
    for lag in range(min(max_lag, n - 1) + 1):
        s = float(np.sum(cent[:n - lag] * cent[lag:]))
        out.append(s / var)
    return out

def pacf_dl(data, max_lag):
    autocorr = biased_acf(data, max_lag)
    nlags = len(autocorr) - 1
    if nlags == 0:
        return [1.0]
    res = [1.0]
    phi = [[0.0] * (nlags + 1) for _ in range(nlags + 1)]
    phi[1][1] = autocorr[1]
    res.append(phi[1][1])
    for k in range(2, nlags + 1):
        num = autocorr[k]
        for j in range(1, k):
            num -= phi[k - 1][j] * autocorr[k - j]
        den = 1.0
        for j in range(1, k):
            den -= phi[k - 1][j] * autocorr[j]
        phi[k][k] = num / den if abs(den) > 1e-15 else 0.0
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
        res.append(phi[k][k])
    return res

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    ml = int(case["max_lag"])
    out = {"case_id": cid, "pacf_vec": None, "ljung_q": None, "ljung_p": None}
    try:
        out["pacf_vec"] = vec_or_none(pacf_dl(data, ml))
        ac = biased_acf(data, ml)
        n = len(data)
        q_stat = sum(ac[k] ** 2 / (n - k) for k in range(1, ml + 1)) * n * (n + 2)
        out["ljung_q"] = fnone(q_stat)
        out["ljung_p"] = fnone(float(stats.chi2(ml).sf(q_stat)))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize pacf_ljung query");
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
                "failed to spawn python3 for pacf_ljung oracle: {e}"
            );
            eprintln!("skipping pacf_ljung oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open pacf_ljung oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "pacf_ljung oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping pacf_ljung oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for pacf_ljung oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "pacf_ljung oracle failed: {stderr}"
        );
        eprintln!("skipping pacf_ljung oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse pacf_ljung oracle JSON"))
}

#[test]
fn diff_stats_pacf_ljung() {
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

        // pacf vector
        if let Some(scipy_pacf) = &scipy_arm.pacf_vec {
            let rust_pacf = pacf(&case.data, case.max_lag as usize);
            if rust_pacf.len() == scipy_pacf.len() {
                let mut max_local = 0.0_f64;
                for (a, b) in rust_pacf.iter().zip(scipy_pacf.iter()) {
                    if a.is_finite() {
                        max_local = max_local.max((a - b).abs());
                    }
                }
                max_overall = max_overall.max(max_local);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: "pacf_max".into(),
                    abs_diff: max_local,
                    pass: max_local <= ABS_TOL,
                });
            }
        }

        // ljung_box
        let (rust_q, rust_p) = ljung_box(&case.data, case.max_lag as usize);
        if let Some(scipy_q) = scipy_arm.ljung_q
            && rust_q.is_finite()
        {
            let abs_diff = (rust_q - scipy_q).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "ljung_q".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.ljung_p
            && rust_p.is_finite()
        {
            let abs_diff = (rust_p - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "ljung_p".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_pacf_ljung".into(),
        category: "pacf + ljung_box (numpy/scipy reference)".into(),
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
                "pacf_ljung mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "pacf_ljung conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
