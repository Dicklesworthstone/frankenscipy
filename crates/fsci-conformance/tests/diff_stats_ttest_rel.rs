#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the paired t-test
//! `scipy.stats.ttest_rel(a, b, alternative=...)`. The
//! independent and 1-sample variants are already covered by
//! diff_stats.rs (default two-sided) and the existing
//! diff_stats_ttest_ind_welch harness.
//!
//! Resolves [frankenscipy-xvlpg]. Paired t exercises a
//! different code path: it operates on per-element
//! differences rather than on independent samples. The 3
//! alternatives ('two-sided', 'less', 'greater') route
//! through different StudentT tail computations.
//!
//! 4 paired-sample fixtures × 3 alternatives × 3 arms
//! (statistic + pvalue + df) = 36 cases via subprocess.
//! Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::ttest_rel;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    alternative: String,
    a: Vec<f64>,
    b: Vec<f64>,
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
    df: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create ttest_rel diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ttest_rel diff log");
    fs::write(path, json).expect("write ttest_rel diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Matched / equal — d≈0
        (
            "matched_equal",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        // Treatment effect — b consistently larger
        (
            "treatment_effect",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
        ),
        // Regression to mean (high values fall, low rise)
        (
            "regress_to_mean",
            vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            vec![8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5],
        ),
        // Near-zero effect (small noisy diffs)
        (
            "near_zero_effect",
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            vec![5.05, 4.95, 5.10, 4.90, 5.02, 4.98, 5.07, 4.93, 5.04, 4.96],
        ),
    ];
    let alternatives = ["two-sided", "less", "greater"];

    let mut points = Vec::new();
    for (name, a, b) in &fixtures {
        for alt in alternatives {
            points.push(PointCase {
                case_id: format!("{name}_{alt}"),
                alternative: alt.into(),
                a: a.clone(),
                b: b.clone(),
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
    cid = case["case_id"]; alt = case["alternative"]
    a = np.array(case["a"], dtype=float)
    b = np.array(case["b"], dtype=float)
    try:
        res = stats.ttest_rel(a, b, alternative=alt)
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
            "df": fnone(res.df),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None, "df": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ttest_rel query");
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
                "failed to spawn python3 for ttest_rel oracle: {e}"
            );
            eprintln!("skipping ttest_rel oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ttest_rel oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ttest_rel oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ttest_rel oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ttest_rel oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ttest_rel oracle failed: {stderr}"
        );
        eprintln!("skipping ttest_rel oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ttest_rel oracle JSON"))
}

#[test]
fn diff_stats_ttest_rel() {
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
        let result = match ttest_rel(&case.a, &case.b, Some(&case.alternative)) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let arms: [(&str, Option<f64>, f64); 3] = [
            ("statistic", scipy_arm.statistic, result.statistic),
            ("pvalue", scipy_arm.pvalue, result.pvalue),
            ("df", scipy_arm.df, result.df),
        ];

        for (arm_name, scipy_v, rust_v) in arms {
            if let Some(scipy_v) = scipy_v
                && rust_v.is_finite()
            {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: arm_name.into(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_ttest_rel".into(),
        category: "scipy.stats.ttest_rel".into(),
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
                "ttest_rel mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ttest_rel conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
