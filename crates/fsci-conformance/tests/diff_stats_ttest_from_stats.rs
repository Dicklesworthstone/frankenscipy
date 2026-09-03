#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2,
//! equal_var) → TtestResult` — independent two-sample t-test
//! computed from summary statistics only (no raw data needed).
//!
//! Resolves [frankenscipy-ji78s]. The oracle calls
//! `scipy.stats.ttest_ind_from_stats(..., equal_var=...)`.
//!
//! 4 summary fixtures × 2 equal_var modes (pooled / Welch) × 3
//! arms (statistic, pvalue, df) = 24 cases. Tol 1e-12 stat /
//! 1e-9 pvalue (Student-t pvalue chain via betainc rational
//! approximation).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::ttest_ind_from_stats;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-12;
const PVALUE_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    mean1: f64,
    std1: f64,
    n1: usize,
    mean2: f64,
    std2: f64,
    n2: usize,
    equal_var: bool,
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
    fs::create_dir_all(output_dir()).expect("create ttest_from_stats diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ttest_from_stats diff log");
    fs::write(path, json).expect("write ttest_from_stats diff log");
}

fn generate_query() -> OracleQuery {
    // (label, mean1, std1, n1, mean2, std2, n2)
    let summaries: Vec<(&str, f64, f64, usize, f64, f64, usize)> = vec![
        ("balanced_close", 5.0, 2.0, 30, 5.5, 2.1, 30),
        ("unbalanced_diff", 10.0, 1.5, 20, 8.0, 3.0, 50),
        ("tight_vs_wide", 100.0, 0.5, 12, 100.0, 5.0, 12),
        ("large_n_significant", 50.0, 4.0, 200, 52.0, 4.5, 250),
    ];

    let mut points = Vec::new();
    for (label, m1, s1, n1, m2, s2, n2) in &summaries {
        for &equal_var in &[true, false] {
            let suffix = if equal_var { "pooled" } else { "welch" };
            points.push(PointCase {
                case_id: format!("{label}_{suffix}"),
                mean1: *m1,
                std1: *s1,
                n1: *n1,
                mean2: *m2,
                std2: *s2,
                n2: *n2,
                equal_var,
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
    out = {"case_id": cid, "statistic": None, "pvalue": None, "df": None}
    try:
        # scipy 1.13+ ttest_ind_from_stats returns a TtestResult with
        # statistic/pvalue/df fields. Pre-1.13 only returns (statistic, pvalue);
        # fall back gracefully and compute df manually.
        res = stats.ttest_ind_from_stats(
            mean1=case["mean1"], std1=case["std1"], nobs1=case["n1"],
            mean2=case["mean2"], std2=case["std2"], nobs2=case["n2"],
            equal_var=case["equal_var"],
        )
        out["statistic"] = fnone(res.statistic)
        out["pvalue"] = fnone(res.pvalue)
        df = getattr(res, "df", None)
        if df is None:
            n1 = case["n1"]; n2 = case["n2"]
            if case["equal_var"]:
                df = n1 + n2 - 2
            else:
                v1 = (case["std1"] ** 2) / n1
                v2 = (case["std2"] ** 2) / n2
                df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
        out["df"] = fnone(df)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ttest_from_stats query");
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
                "failed to spawn python3 for ttest_from_stats oracle: {e}"
            );
            eprintln!("skipping ttest_from_stats oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ttest_from_stats oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ttest_from_stats oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ttest_from_stats oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ttest_from_stats oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ttest_from_stats oracle failed: {stderr}"
        );
        eprintln!("skipping ttest_from_stats oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ttest_from_stats oracle JSON"))
}

#[test]
fn diff_stats_ttest_from_stats() {
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
        let result = ttest_ind_from_stats(
            case.mean1,
            case.std1,
            case.n1,
            case.mean2,
            case.std2,
            case.n2,
            case.equal_var,
        );

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
        if let Some(s_df) = scipy_arm.df
            && result.df.is_finite()
        {
            let abs_diff = (result.df - s_df).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "df".into(),
                abs_diff,
                pass: abs_diff <= STAT_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_ttest_from_stats".into(),
        category: "scipy.stats.ttest_ind_from_stats".into(),
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
                "ttest_from_stats mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ttest_from_stats conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
