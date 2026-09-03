#![forbid(unsafe_code)]
//! Live SciPy / numpy differential coverage for two
//! summary-style functions:
//!   • `pmean(data, p)` — generalised (Hölder) power mean
//!     M_p = ((1/n) Σ xᵢᵖ)^(1/p), with p=0 falling back to
//!     gmean and negative p with any zero element returning 0
//!   • `norm_loglikelihood(data, mu, sigma)` — closed-form
//!     Gaussian log-likelihood Σ ln φ((x−μ)/σ) − n·ln σ
//!
//! Resolves [frankenscipy-7cu75]. The oracle calls
//! `scipy.stats.pmean(data, p)` and reproduces the Gaussian
//! log-likelihood directly (closed form) — no scipy primitive
//! exposes the bare scalar.
//!
//! 4 datasets × 4 power values + 4 (data, mu, sigma) fixtures
//! × norm_loglikelihood = 20 cases. Tol 1e-12 abs (closed-form
//! power-sum / Gaussian log).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{norm_loglikelihood, pmean};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    /// p for pmean.
    p: f64,
    /// (mu, sigma) for norm_loglikelihood.
    mu: f64,
    sigma: f64,
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
    fs::create_dir_all(output_dir()).expect("create pmean_loglik diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize pmean_loglik diff log");
    fs::write(path, json).expect("write pmean_loglik diff log");
}

fn generate_query() -> OracleQuery {
    // pmean is restricted to non-negative data (fsci returns NaN otherwise).
    let pmean_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_positive_n10", (1..=10).map(|i| i as f64).collect()),
        (
            "spread_positive_n12",
            vec![
                0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0,
            ],
        ),
        (
            "near_one_n8",
            vec![0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3],
        ),
        (
            "wide_range_n15",
            (1..=15).map(|i| (i as f64).powi(2) / 3.0).collect(),
        ),
    ];
    // p in {-1, 1, 2, 3}; p=0 falls through to gmean which is covered elsewhere.
    let powers = [-1.0_f64, 1.0, 2.0, 3.0];

    // norm_loglikelihood: include realistic distributions
    let nll_fixtures: Vec<(&str, Vec<f64>, f64, f64)> = vec![
        (
            "centered_unit",
            vec![-1.0, -0.5, 0.0, 0.5, 1.0, -0.2, 0.3, 0.1, -0.4, 0.6],
            0.0,
            1.0,
        ),
        ("shifted", (1..=12).map(|i| i as f64).collect(), 6.5, 3.5),
        (
            "tight_sigma",
            vec![5.0, 4.95, 5.05, 5.0, 4.98, 5.02, 4.97, 5.03, 5.0, 4.99],
            5.0,
            0.05,
        ),
        (
            "wide_sigma_n14",
            (1..=14).map(|i| (i as f64) * 2.0).collect(),
            15.0,
            8.0,
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &pmean_datasets {
        for &p in &powers {
            let pname = if p < 0.0 {
                format!("neg{}", p.abs())
            } else {
                format!("p{p}")
            };
            points.push(PointCase {
                case_id: format!("{name}_pmean_{pname}"),
                func: "pmean".into(),
                data: data.clone(),
                p,
                mu: 0.0,
                sigma: 1.0,
            });
        }
    }
    for (name, data, mu, sigma) in &nll_fixtures {
        points.push(PointCase {
            case_id: format!("{name}_norm_loglik"),
            func: "norm_loglikelihood".into(),
            data: data.clone(),
            p: 0.0,
            mu: *mu,
            sigma: *sigma,
        });
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        if func == "pmean":
            val = float(stats.pmean(data, float(case["p"])))
        elif func == "norm_loglikelihood":
            mu = float(case["mu"]); sigma = float(case["sigma"])
            n = data.size
            log_sigma = math.log(sigma)
            two_s2 = 2.0 * sigma * sigma
            ll = (
                -n / 2.0 * math.log(2.0 * math.pi)
                - n * log_sigma
                - float(np.sum((data - mu) ** 2 / two_s2))
            )
            val = ll
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize pmean_loglik query");
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
                "failed to spawn python3 for pmean_loglik oracle: {e}"
            );
            eprintln!("skipping pmean_loglik oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open pmean_loglik oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "pmean_loglik oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping pmean_loglik oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for pmean_loglik oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "pmean_loglik oracle failed: {stderr}"
        );
        eprintln!("skipping pmean_loglik oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse pmean_loglik oracle JSON"))
}

#[test]
fn diff_stats_pmean_loglik() {
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
            "pmean" => pmean(&case.data, case.p),
            "norm_loglikelihood" => norm_loglikelihood(&case.data, case.mu, case.sigma),
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
        test_id: "diff_stats_pmean_loglik".into(),
        category: "scipy.stats.pmean + Gaussian log-likelihood (numpy reference)".into(),
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
                "pmean_loglik {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "pmean_loglik conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
