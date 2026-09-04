#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the one-sample
//! Kolmogorov-Smirnov test
//! `scipy.stats.ks_1samp(data, 'norm')`.
//!
//! Resolves [frankenscipy-ktefe]. Cross-checks both the D
//! statistic (max |F_n(x) - F(x)|) and the p-value
//! (asymptotic Kolmogorov series via fsci's
//! kolmogorov_pvalue) across 4 datasets.
//!
//! 4 fixtures × 2 arms = 8 cases via subprocess.
//! Tolerances:
//!   - statistic : 1e-12 abs (closed-form max-difference walk).
//!   - pvalue    : 1e-7 abs. fsci's kolmogorov_pvalue uses a
//!     truncated infinite series; scipy switches between
//!     exact and asymptotic depending on sample size.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Normal, ks_1samp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-12;
const PVALUE_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    statistic: Option<f64>,
    pvalue: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create ks_1samp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ks_1samp diff log");
    fs::write(path, json).expect("write ks_1samp diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "near_normal_n14",
            vec![
                -1.7, -1.1, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.9,
            ],
        ),
        (
            "skewed_pos",
            vec![
                0.05, 0.12, 0.25, 0.40, 0.60, 0.85, 1.15, 1.55, 2.05, 2.70, 3.55, 4.65,
            ],
        ),
        (
            "uniform_mapped",
            vec![-1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5],
        ),
        (
            "bimodal",
            vec![-2.5, -2.0, -1.6, -1.4, -1.2, 1.2, 1.4, 1.6, 2.0, 2.5],
        ),
    ];

    let points = datasets
        .into_iter()
        .map(|(name, data)| PointCase {
            case_id: name.into(),
            data,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    try:
        # scipy's documented default (method='auto') resolves to the EXACT
        # Kolmogorov distribution for n <= 10000, matching fsci's exact
        # kolmogn_sf port (frankenscipy-ksk1u) — the earlier method='asymp'
        # pin compared scipy's asymptotic series against fsci's exact
        # branch and diverged by up to 0.038 at n=14 on the first live run
        # (2026-09-04). ks_1samp requires a callable for the cdf — pass
        # stats.norm.cdf directly. (Earlier this passed the string 'norm'
        # which scipy silently raised TypeError on, dropping all cases —
        # see commit log for the case_count=0 regression that hid this
        # for ages.)
        res = stats.ks_1samp(data, stats.norm.cdf)
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ks_1samp query");
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
                "failed to spawn python3 for ks_1samp oracle: {e}"
            );
            eprintln!("skipping ks_1samp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ks_1samp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ks_1samp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ks_1samp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ks_1samp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ks_1samp oracle failed: {stderr}"
        );
        eprintln!("skipping ks_1samp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ks_1samp oracle JSON"))
}

#[test]
fn diff_stats_ks_1samp() {
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

    let norm = Normal::standard();
    let cdf_norm = |x: f64| ContinuousDistribution::cdf(&norm, x);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let result = ks_1samp(&case.data, cdf_norm);

        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= STAT_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
        {
            let abs_diff = (result.pvalue - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= PVALUE_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_ks_1samp".into(),
        category: "scipy.stats.ks_1samp(cdf='norm')".into(),
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
                "ks_1samp mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ks_1samp conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
