#![forbid(unsafe_code)]
//! Live SciPy differential coverage for three classical
//! goodness-of-fit / variance tests:
//!   • `levene(groups)` — Brown-Forsythe variant (median-
//!     centered absolute deviations, F-tail).
//!   • `bartlett(groups)` — log-pooled-variance χ² test
//!     (sensitive to non-normality; complements levene).
//!   • `jarque_bera(data)` — D'Agostino-style skew + excess-
//!     kurtosis K² normality test.
//!
//! Resolves [frankenscipy-qst52]. The oracle calls
//! `scipy.stats.{levene, bartlett, jarque_bera}`.
//!
//! 4 group-fixtures × 2 (levene + bartlett) × 2 arms +
//! 4 datasets × jarque_bera × 2 arms = 24 cases. Tol 1e-9 abs
//! (chi-square / F-tail chain via regularized incomplete
//! gamma / beta).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{bartlett, jarque_bera, levene};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    /// Used for levene / bartlett (variance-of-groups family).
    groups: Vec<Vec<f64>>,
    /// Used for jarque_bera.
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
    fs::create_dir_all(output_dir()).expect("create variance_normality_tests diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json =
        serde_json::to_string_pretty(log).expect("serialize variance_normality_tests diff log");
    fs::write(path, json).expect("write variance_normality_tests diff log");
}

fn generate_query() -> OracleQuery {
    let group_fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // Two equal-variance groups
        (
            "two_equal_var",
            vec![
                (1..=10).map(|i| i as f64).collect(),
                (11..=20).map(|i| i as f64).collect(),
            ],
        ),
        // Two unequal-variance groups
        (
            "two_diff_var",
            vec![
                vec![5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0],
                vec![1.0, 5.0, 9.0, 0.5, 4.5, 8.5, 1.5, 5.5, 9.5, 2.0],
            ],
        ),
        // Three groups
        (
            "three_groups",
            vec![
                (1..=8).map(|i| i as f64).collect(),
                (1..=8).map(|i| (i as f64) * 2.0).collect(),
                (1..=8).map(|i| (i as f64).powi(2)).collect(),
            ],
        ),
        // Four small groups
        (
            "four_small",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                vec![3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                vec![1.0, 1.5, 4.0, 8.0, 12.0, 16.0],
            ],
        ),
    ];

    let normality_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("uniform_n20", (1..=20).map(|i| (i as f64) - 10.5).collect()),
        // Near-normal cubic-probit
        (
            "near_normal_n30",
            (0..30)
                .map(|i| {
                    let p = (i as f64 + 0.5) / 30.0;
                    let q = p - 0.5;
                    2.5 * (q + 4.0 * q * q * q)
                })
                .collect(),
        ),
        (
            "exp_like_n25",
            (1..=25).map(|i| ((i as f64) / 5.0).exp() - 1.0).collect(),
        ),
        (
            "heavy_tail_n40",
            (0..40)
                .map(|i| {
                    let q = (i as f64 + 0.5) / 40.0 - 0.5;
                    q * q * q * 12.0
                })
                .collect(),
        ),
    ];

    let empty = Vec::<f64>::new();
    let empty_groups: Vec<Vec<f64>> = Vec::new();
    let mut points = Vec::new();
    for (name, groups) in &group_fixtures {
        for func in ["levene", "bartlett"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                groups: groups.clone(),
                data: empty.clone(),
            });
        }
    }
    for (name, data) in &normality_datasets {
        points.push(PointCase {
            case_id: format!("{name}_jarque_bera"),
            func: "jarque_bera".into(),
            groups: empty_groups.clone(),
            data: data.clone(),
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
    stat = None; pval = None
    try:
        if func == "levene":
            groups = [np.array(g, dtype=float) for g in case["groups"]]
            # Default scipy center is 'median' = Brown-Forsythe (matches fsci).
            res = stats.levene(*groups)
        elif func == "bartlett":
            groups = [np.array(g, dtype=float) for g in case["groups"]]
            res = stats.bartlett(*groups)
        elif func == "jarque_bera":
            data = np.array(case["data"], dtype=float)
            res = stats.jarque_bera(data)
        else:
            res = None
        if res is not None:
            stat = fnone(res.statistic)
            pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize variance_normality query");
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
                "failed to spawn python3 for variance_normality oracle: {e}"
            );
            eprintln!("skipping variance_normality oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open variance_normality oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "variance_normality oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping variance_normality oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for variance_normality oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "variance_normality oracle failed: {stderr}"
        );
        eprintln!("skipping variance_normality oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse variance_normality oracle JSON"))
}

#[test]
fn diff_stats_variance_normality_tests() {
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
        let (rust_stat, rust_p) = match case.func.as_str() {
            "levene" => {
                let refs: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
                let r = levene(&refs);
                (r.statistic, r.pvalue)
            }
            "bartlett" => {
                let refs: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
                let r = bartlett(&refs);
                (r.statistic, r.pvalue)
            }
            "jarque_bera" => {
                let r = jarque_bera(&case.data);
                (r.statistic, r.pvalue)
            }
            _ => continue,
        };

        if let Some(s_stat) = scipy_arm.statistic
            && rust_stat.is_finite()
        {
            let abs_diff = (rust_stat - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.statistic", case.func),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_p) = scipy_arm.pvalue
            && rust_p.is_finite()
        {
            let abs_diff = (rust_p - s_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.pvalue", case.func),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_variance_normality_tests".into(),
        category: "scipy.stats.{levene, bartlett, jarque_bera}".into(),
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
                "variance_normality_tests mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "variance_normality_tests conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
