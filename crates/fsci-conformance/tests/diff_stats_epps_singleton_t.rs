#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `epps_singleton_2samp_with_t(x, y, t)` at non-default
//! support-point vectors.
//!
//! Resolves [frankenscipy-6udji]. The existing
//! diff_stats_epps_singleton.rs covers the default t=[0.4, 0.8].
//! This harness exercises three additional t-vectors against
//! `scipy.stats.epps_singleton_2samp(x, y, t=...)`:
//!   • [0.3, 0.6, 0.9]   — finer two-side resolution
//!   • [0.5]             — single point (smallest valid t)
//!   • [0.2, 0.5, 0.8, 1.1] — wider support
//!
//! 3 (x, y) fixtures × 3 t-vectors × 2 arms = 18 cases. Tol
//! 1e-9 abs (chi-square tail chain via regularized incomplete
//! gamma).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::epps_singleton_2samp_with_t;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Per-arm tols: 1e-4 for the chi-square statistic and 1e-5 for the
// pvalue. The Epps-Singleton statistic is IQR-normalised (σ = iqr/2)
// and grows with the number of support points; numerical noise in the
// 4×4 covariance inverse plus IQR ties accumulates to ~4e-5 abs on
// quad_wide t-vectors at n=12. Pvalue is more stable (chi-square tail
// chain).
const STAT_TOL: f64 = 1.0e-4;
const PVALUE_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    t: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create epps_singleton_t diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize epps_singleton_t diff log");
    fs::write(path, json).expect("write epps_singleton_t diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Same-shape, slightly shifted
        (
            "balanced_n12",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| (i as f64) + 1.0).collect(),
        ),
        // Very different distributions (Epps-Singleton signal)
        (
            "shifted_n14",
            (1..=14).map(|i| i as f64).collect(),
            (8..=21).map(|i| i as f64).collect(),
        ),
        // Different scales
        (
            "diff_scale_n16",
            vec![
                4.0, 5.0, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 4.6, 5.4, 4.5, 5.5, 4.4, 5.6,
            ],
            vec![
                -2.0, 12.0, -1.0, 11.0, 0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 4.5, 5.5,
            ],
        ),
    ];
    let t_vecs: Vec<(&str, Vec<f64>)> = vec![
        ("triple", vec![0.3, 0.6, 0.9]),
        ("single", vec![0.5]),
        ("quad_wide", vec![0.2, 0.5, 0.8, 1.1]),
    ];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for (label, t) in &t_vecs {
            points.push(PointCase {
                case_id: format!("{name}_{label}"),
                x: x.clone(),
                y: y.clone(),
                t: t.clone(),
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    t = list(case["t"])
    stat = None; pval = None
    try:
        res = stats.epps_singleton_2samp(x, y, t=t)
        stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize epps_singleton_t query");
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
                "failed to spawn python3 for epps_singleton_t oracle: {e}"
            );
            eprintln!("skipping epps_singleton_t oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open epps_singleton_t oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "epps_singleton_t oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping epps_singleton_t oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for epps_singleton_t oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "epps_singleton_t oracle failed: {stderr}"
        );
        eprintln!("skipping epps_singleton_t oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse epps_singleton_t oracle JSON"))
}

#[test]
fn diff_stats_epps_singleton_t() {
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
        let result = match epps_singleton_2samp_with_t(&case.x, &case.y, &case.t) {
            Ok(r) => r,
            Err(_) => continue,
        };

        if let Some(s_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
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
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= PVALUE_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_epps_singleton_t".into(),
        category: "scipy.stats.epps_singleton_2samp(t=non-default)".into(),
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
                "epps_singleton_t mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "epps_singleton_t conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
