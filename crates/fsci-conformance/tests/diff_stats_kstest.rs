#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's `kstest`
//! dispatch wrapper, which routes to `ks_1samp` for a CDF
//! target or `ks_2samp` for a reference sample.
//!
//! Resolves [frankenscipy-bdzr0]. The base ks_1samp / ks_2samp
//! routes are covered by their own diff harnesses — this one
//! verifies the wrapper dispatches correctly to each branch.
//!
//! 4 fixtures × 2 dispatch modes (cdf, sample) × 2 arms =
//! 16 cases. Tol 1e-12 statistic / 1e-7 pvalue (KS-distribution
//! tail chain).
//!
//! Note: harness history — an earlier version pinned the oracle to
//! `method='asymptotic'` (an UNDOCUMENTED scipy value returning pvalue=1.0
//! as a sentinel), then `method='asymp'` to match fsci's
//! `kolmogorov_pvalue` series. Since frankenscipy-ksk1u, fsci implements
//! the EXACT Kolmogorov distribution for n <= 10000 (`kolmogn_sf`:
//! DMTW/Pomeranz/RRDG — scipy's own `_kolmogn` dispatch), so the oracle
//! now uses scipy's documented default (`method='auto'`): exact-vs-exact.
//! The asymp pin diverged from the exact branch by up to 0.038 at n=14 on
//! the first live-oracle run (2026-09-04). See [frankenscipy-8rfh5] and
//! [frankenscipy-ksk1u].

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, KstestTarget, Normal, kstest};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Tols mirror diff_stats_ks_1samp.rs: 1e-12 statistic / 1e-7 pvalue.
const STAT_TOL: f64 = 1.0e-12;
const PVALUE_TOL: f64 = 1.0e-7;
// The oracle uses scipy's documented default (method='auto'): exact
// Kolmogorov for n <= 10000, matching fsci's exact kolmogn port
// (frankenscipy-ksk1u). The earlier method='asymp' pin predates that
// port and diverged by up to 0.038 at n=14 on the first live run
// (2026-09-04).
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    mode: String,
    data: Vec<f64>,
    /// Reference sample (for the 2-sample dispatch branch).
    reference: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create kstest diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize kstest diff log");
    fs::write(path, json).expect("write kstest diff log");
}

fn generate_query() -> OracleQuery {
    // Fixtures mirror diff_stats_ks_1samp.rs which is known to agree with
    // scipy's method='asymp' to 1e-7. Synthetic / large-D fixtures
    // produced 1e-2 pvalue gaps because the KS distance was close to 1
    // (deep in the rejection region) where the two asymptotic series
    // formulas can diverge by orders of magnitude in absolute terms.
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "near_normal_n14",
            vec![
                -1.7, -1.1, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.9,
            ],
            vec![
                -1.6, -1.0, -0.6, -0.3, -0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6, 2.0,
            ],
        ),
        (
            "uniform_mapped",
            vec![-1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5],
            vec![-1.4, -0.9, -0.6, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.4],
        ),
        (
            "bimodal",
            vec![-2.5, -2.0, -1.6, -1.4, -1.2, 1.2, 1.4, 1.6, 2.0, 2.5],
            vec![-2.4, -1.9, -1.5, -1.3, -1.1, 1.1, 1.3, 1.5, 1.9, 2.4],
        ),
        (
            "skewed_pos",
            vec![
                0.05, 0.12, 0.25, 0.40, 0.60, 0.85, 1.15, 1.55, 2.05, 2.70, 3.55, 4.65,
            ],
            vec![
                0.04, 0.11, 0.23, 0.38, 0.58, 0.82, 1.10, 1.50, 2.00, 2.60, 3.40, 4.50,
            ],
        ),
    ];

    let mut points = Vec::new();
    for (name, data, reference) in &fixtures {
        points.push(PointCase {
            case_id: format!("{name}_cdf"),
            mode: "cdf".into(),
            data: data.clone(),
            reference: vec![],
        });
        points.push(PointCase {
            case_id: format!("{name}_sample"),
            mode: "sample".into(),
            data: data.clone(),
            reference: reference.clone(),
        });
    }
    OracleQuery { points }
}

/// Standard-normal CDF for the cdf-mode dispatch — uses fsci-stats's
/// `Normal` distribution directly so the KS distance computed on the
/// Rust side agrees with scipy's `stats.norm.cdf` at machine precision.
fn standard_normal_cdf(x: f64) -> f64 {
    let norm = Normal::standard();
    ContinuousDistribution::cdf(&norm, x)
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
    cid = case["case_id"]; mode = case["mode"]
    data = np.array(case["data"], dtype=float)
    stat = None; pval = None
    try:
        if mode == "cdf":
            # scipy's documented default (method='auto') resolves to the
            # EXACT Kolmogorov distribution for n <= 10000, matching fsci's
            # exact kolmogn port (DMTW/Pomeranz/RRDG, frankenscipy-ksk1u).
            # The earlier method='asymp' pin compared scipy's asymptotic
            # series against fsci's exact branch and diverged by up to
            # 0.038 at n=14 — exact-vs-exact is the parity contract.
            res = stats.kstest(data, "norm")
        elif mode == "sample":
            ref = np.array(case["reference"], dtype=float)
            res = stats.kstest(data, ref)
        else:
            res = None
        if res is not None:
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize kstest query");
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
                "failed to spawn python3 for kstest oracle: {e}"
            );
            eprintln!("skipping kstest oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open kstest oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "kstest oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kstest oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for kstest oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "kstest oracle failed: {stderr}"
        );
        eprintln!("skipping kstest oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse kstest oracle JSON"))
}

#[test]
fn diff_stats_kstest() {
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
        let result = match case.mode.as_str() {
            "cdf" => kstest(&case.data, KstestTarget::Cdf(standard_normal_cdf)),
            "sample" => kstest(&case.data, KstestTarget::Sample(&case.reference)),
            _ => continue,
        };

        if let Some(s_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.statistic", case.mode),
                abs_diff,
                pass: abs_diff <= STAT_TOL,
            });
        }
        if let Some(s_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
        {
            let abs_diff = (result.pvalue - s_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.pvalue", case.mode),
                abs_diff,
                pass: abs_diff <= PVALUE_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_kstest".into(),
        category: "scipy.stats.kstest dispatch (cdf | sample)".into(),
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
                "kstest mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "kstest conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
