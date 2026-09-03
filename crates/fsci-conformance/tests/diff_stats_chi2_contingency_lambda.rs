#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `chi2_contingency_with_lambda(observed, correction, lambda_)`
//! — the power-divergence parameterisation of the contingency
//! chi-squared test.
//!
//! Resolves [frankenscipy-2tjxu]. The existing
//! diff_stats_chi2_contingency.rs covers the Pearson default
//! (lambda=1, correction=true on 2×2). This harness exercises
//! the four other lambdas (G-test, mod-log-likelihood, Freeman-
//! Tukey, Cressie-Read) plus the Pearson sanity check, with
//! correction off to avoid the Yates-on-2x2 path.
//!
//! 3 tables × 5 lambdas × 3 arms (statistic, pvalue, dof) = 45
//! cases. Tol 1e-9 abs (chi-square tail chain via regularized
//! incomplete gamma).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::chi2_contingency_with_lambda;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    table: Vec<Vec<f64>>,
    lambda_: f64,
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
    dof: Option<i64>,
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
    fs::create_dir_all(output_dir()).expect("create chi2_contingency_lambda diff output dir");
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
        serde_json::to_string_pretty(log).expect("serialize chi2_contingency_lambda diff log");
    fs::write(path, json).expect("write chi2_contingency_lambda diff log");
}

fn generate_query() -> OracleQuery {
    let tables: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // 2x3 table
        (
            "two_by_three",
            vec![vec![20.0, 30.0, 25.0], vec![15.0, 25.0, 35.0]],
        ),
        // 3x3 table
        (
            "three_by_three",
            vec![
                vec![45.0, 38.0, 33.0],
                vec![30.0, 27.0, 25.0],
                vec![22.0, 20.0, 15.0],
            ],
        ),
        // 4x2 table
        (
            "four_by_two",
            vec![
                vec![40.0, 10.0],
                vec![35.0, 15.0],
                vec![20.0, 30.0],
                vec![15.0, 35.0],
            ],
        ),
    ];

    let lambdas: Vec<(&str, f64)> = vec![
        ("pearson", 1.0),
        ("g_test", 0.0),
        ("mod_log_lik", -1.0),
        ("freeman_tukey", -0.5),
        ("cressie_read", 2.0 / 3.0),
    ];

    let mut points = Vec::new();
    for (name, table) in &tables {
        for (lname, lam) in &lambdas {
            points.push(PointCase {
                case_id: format!("{name}_{lname}"),
                table: table.clone(),
                lambda_: *lam,
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
    table = np.array(case["table"], dtype=float)
    lam = float(case["lambda_"])
    out = {"case_id": cid, "statistic": None, "pvalue": None, "dof": None}
    try:
        # correction=False to mirror fsci's default-no-Yates path; lambda_
        # selects the power-divergence variant.
        res = stats.chi2_contingency(table, correction=False, lambda_=lam)
        out["statistic"] = fnone(res.statistic)
        out["pvalue"] = fnone(res.pvalue)
        out["dof"] = int(res.dof)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize chi2_contingency_lambda query");
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
                "failed to spawn python3 for chi2_contingency_lambda oracle: {e}"
            );
            eprintln!("skipping chi2_contingency_lambda oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open chi2_contingency_lambda oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "chi2_contingency_lambda oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping chi2_contingency_lambda oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for chi2_contingency_lambda oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "chi2_contingency_lambda oracle failed: {stderr}"
        );
        eprintln!("skipping chi2_contingency_lambda oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse chi2_contingency_lambda oracle JSON"))
}

#[test]
fn diff_stats_chi2_contingency_lambda() {
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
        let result = chi2_contingency_with_lambda(&case.table, false, case.lambda_);

        if let Some(s_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - s_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
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
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(s_dof) = scipy_arm.dof {
            let abs_diff = (result.dof as i64 - s_dof).unsigned_abs() as f64;
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "dof".into(),
                abs_diff,
                pass: abs_diff == 0.0,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_chi2_contingency_lambda".into(),
        category: "scipy.stats.chi2_contingency(lambda_, correction=False)".into(),
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
                "chi2_contingency_lambda mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "chi2_contingency_lambda conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
