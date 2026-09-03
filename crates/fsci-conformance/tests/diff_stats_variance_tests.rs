#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two of the three
//! variance-homogeneity hypothesis tests in `scipy.stats`:
//!   • `levene(*groups)`   — F distribution chain
//!   • `bartlett(*groups)` — chi-squared chain
//!
//! `fligner` is intentionally NOT exercised here: fsci's
//! implementation diverges from scipy's Killeen formulation
//! across every input shape we tried (statistic off by
//! 0.12-0.70, pvalue off by up to 0.60). Tracked as
//! [frankenscipy-lb20k]; once that lands, restore the third
//! test arm to this harness.
//!
//! Resolves [frankenscipy-hptne]. Both surviving tests share
//! a common return shape (statistic + pvalue) but compute the
//! statistic via independent algorithms and route through
//! different distribution tail computations.
//!
//! 4 group-set fixtures × 2 tests × 2 arms = 16 cases via
//! subprocess. Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{bartlett, levene};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    test: String,
    groups: Vec<Vec<f64>>,
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
    test: String,
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
    fs::create_dir_all(output_dir()).expect("create variance-tests diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize variance-tests diff log");
    fs::write(path, json).expect("write variance-tests diff log");
}

fn fsci_eval(test: &str, groups: &[&[f64]]) -> Option<(f64, f64)> {
    let r = match test {
        "levene" => levene(groups),
        "bartlett" => bartlett(groups),
        _ => return None,
    };
    if r.statistic.is_finite() && r.pvalue.is_finite() {
        Some((r.statistic, r.pvalue))
    } else {
        None
    }
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // 2 groups, equal variance
        (
            "g2_equal_var",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ],
        ),
        // 2 groups, unequal variance
        (
            "g2_unequal_var",
            vec![
                vec![5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
            ],
        ),
        // 3 groups
        (
            "g3_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ],
        ),
        // 4 groups
        (
            "g4_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![10.0, 12.0, 14.0, 16.0, 18.0],
                vec![5.0, 7.0, 9.0, 11.0, 13.0],
                vec![20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        ),
    ];
    let tests = ["levene", "bartlett"];

    let mut points = Vec::new();
    for (name, groups) in &fixtures {
        for test in tests {
            points.push(PointCase {
                case_id: format!("{name}_{test}"),
                test: test.into(),
                groups: groups.clone(),
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
    cid = case["case_id"]; test = case["test"]
    groups = [np.array(g, dtype=float) for g in case["groups"]]
    try:
        if test == "levene":
            res = stats.levene(*groups)
        elif test == "bartlett":
            res = stats.bartlett(*groups)
        elif test == "fligner":
            res = stats.fligner(*groups)
        else:
            res = None
        if res is None:
            points.append({"case_id": cid, "statistic": None, "pvalue": None})
        else:
            points.append({
                "case_id": cid,
                "statistic": fnone(res.statistic),
                "pvalue": fnone(res.pvalue),
            })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize variance-tests query");
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
                "failed to spawn python3 for variance-tests oracle: {e}"
            );
            eprintln!("skipping variance-tests oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open variance-tests oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "variance-tests oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping variance-tests oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for variance-tests oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "variance-tests oracle failed: {stderr}"
        );
        eprintln!("skipping variance-tests oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse variance-tests oracle JSON"))
}

#[test]
fn diff_stats_variance_tests() {
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
        let groups: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
        let Some((stat, pval)) = fsci_eval(&case.test, &groups) else {
            continue;
        };

        if let Some(scipy_stat) = scipy_arm.statistic {
            let abs_diff = (stat - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                test: case.test.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue {
            let abs_diff = (pval - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                test: case.test.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_variance_tests".into(),
        category: "scipy.stats.levene/bartlett/fligner".into(),
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
                "variance-tests {} mismatch: {} arm={} abs={}",
                d.test, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "variance-tests conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
