#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Page's L trend test
//! `scipy.stats.page_trend_test(data, predicted_ranks=..., method=...)`.
//!
//! Resolves [frankenscipy-78chv]. Page's test is a non-parametric repeated-measures test
//! where conditions are pre-ordered along a hypothesized monotonic trend. Within each subject
//! (row) ranks are assigned to the k conditions; L is the sum of column rank sums weighted by
//! their predicted rank.
//!
//! Every case runs all three methods. The `auto` arm also checks which method SciPy picked;
//! for small tables it is the exact distribution. Until frankenscipy-fmyg3 this harness pinned
//! SciPy to `method='asymptotic'` because fsci only had the normal approximation, so fsci's
//! default answer went unchecked. The fixtures cover:
//! - ties;
//! - predicted ranks;
//! - 7 columns (exact over 7! orderings per row);
//! - 22 rows (`auto` switches to asymptotic);
//! - a 30 × 9 trend whose asymptotic p-value is 1.97e-54, which `1 - cdf` loses entirely.
//!
//! L is compared to `ABS_TOL`, p-values to `P_REL_TOL` relative.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{PageTrendMethod, page_trend_test};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const P_REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ARMS: [&str; 4] = ["statistic", "auto", "exact", "asymptotic"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<Vec<f64>>,
    predicted_ranks: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct MethodArm {
    pvalue: Option<f64>,
    method: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    statistic: Option<f64>,
    auto: MethodArm,
    exact: MethodArm,
    asymptotic: MethodArm,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_statistic_abs_diff: f64,
    max_pvalue_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create page_trend diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize page_trend diff log");
    fs::write(path, json).expect("write page_trend diff log");
}

/// `rows × cols` values with a drift of `trend` per column and deterministic noise, rounded to
/// 0.1 so that ties occur.
fn drifting(rows: usize, cols: usize, trend: f64, salt: u64) -> Vec<Vec<f64>> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|c| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                    ((trend * c as f64 + 2.0 * noise) * 10.0).round() / 10.0
                })
                .collect()
        })
        .collect()
}

fn generate_query() -> OracleQuery {
    // n×k repeated-measures matrices. Conditions (columns) are
    // pre-ordered along the hypothesized increasing trend.
    let fixtures: Vec<(&str, Vec<Vec<f64>>, Option<Vec<usize>>)> = vec![
        // Strong increasing trend: each row is monotone non-decreasing
        (
            "monotone_n6_k4",
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2.0, 3.0, 4.0, 5.0],
                vec![1.5, 2.5, 3.5, 4.5],
                vec![1.0, 2.0, 4.0, 5.0],
                vec![2.0, 3.0, 4.0, 6.0],
                vec![1.0, 3.0, 4.0, 5.0],
            ],
            None,
        ),
        // Mild increasing trend with noise
        (
            "noisy_n8_k3",
            vec![
                vec![3.0, 4.0, 5.0],
                vec![4.0, 5.0, 6.0],
                vec![3.0, 5.0, 4.0],
                vec![5.0, 6.0, 7.0],
                vec![4.0, 6.0, 5.0],
                vec![5.0, 7.0, 8.0],
                vec![6.0, 7.0, 9.0],
                vec![5.0, 8.0, 9.0],
            ],
            None,
        ),
        // No clear trend (negative test)
        (
            "no_trend_n5_k4",
            vec![
                vec![3.0, 1.0, 4.0, 2.0],
                vec![2.0, 4.0, 1.0, 3.0],
                vec![1.0, 3.0, 2.0, 4.0],
                vec![4.0, 2.0, 3.0, 1.0],
                vec![3.0, 4.0, 1.0, 2.0],
            ],
            None,
        ),
        // 5 conditions, 7 subjects, increasing trend
        (
            "increasing_n7_k5",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![2.0, 3.0, 4.0, 5.0, 6.0],
                vec![1.5, 2.5, 3.5, 4.5, 5.5],
                vec![3.0, 4.0, 5.0, 6.0, 7.0],
                vec![2.5, 3.5, 4.5, 5.5, 6.5],
                vec![4.0, 5.0, 6.0, 7.0, 8.0],
                vec![3.5, 4.5, 5.5, 6.5, 7.5],
            ],
            None,
        ),
        ("ties_n9_k4", drifting(9, 4, 0.3, 1), None),
        (
            "predicted_n6_k5",
            drifting(6, 5, 0.4, 2),
            Some(vec![2, 4, 1, 5, 3]),
        ),
        ("wide_n5_k7", drifting(5, 7, 0.25, 3), None),
        ("tall_n22_k4", drifting(22, 4, 0.2, 4), None),
        ("strong_n30_k9", drifting(30, 9, 3.0, 5), None),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data, predicted_ranks)| PointCase {
            case_id: name.into(),
            data,
            predicted_ranks,
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
    data = np.array(case["data"], dtype=float)
    out = {"case_id": case["case_id"], "statistic": None}
    for method in ("auto", "exact", "asymptotic"):
        try:
            res = stats.page_trend_test(data, predicted_ranks=case["predicted_ranks"],
                                        method=method)
            out["statistic"] = fnone(res.statistic)
            out[method] = {"pvalue": fnone(res.pvalue), "method": res.method}
        except Exception:
            out[method] = {"pvalue": None, "method": None}
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize page_trend query");
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
                "failed to spawn python3 for page_trend oracle: {e}"
            );
            eprintln!("skipping page_trend oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open page_trend oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "page_trend oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping page_trend oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for page_trend oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "page_trend oracle failed: {stderr}"
        );
        eprintln!("skipping page_trend oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse page_trend oracle JSON"))
}

fn method_name(method: PageTrendMethod) -> &'static str {
    match method {
        PageTrendMethod::Auto => "auto",
        PageTrendMethod::Exact => "exact",
        PageTrendMethod::Asymptotic => "asymptotic",
    }
}

#[test]
fn diff_stats_page_trend() {
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
    let (mut max_stat, mut max_p) = (0.0_f64, 0.0_f64);
    let mut ledger = CompareLedger::new("diff_stats_page_trend", &ARMS);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let data: Vec<&[f64]> = case.data.iter().map(|r| r.as_slice()).collect();
        let run =
            |method| page_trend_test(&data, false, case.predicted_ranks.as_deref(), method).ok();

        let auto = run(PageTrendMethod::Auto);
        if let Some((s, f)) = ledger.pair(
            "statistic",
            &case.case_id,
            scipy_arm.statistic,
            auto.as_ref().map(|r| r.statistic),
        ) {
            let diff = (f - s).abs();
            max_stat = max_stat.max(diff);
            ledger.compared("statistic", &case.case_id, diff <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                diff,
                pass: diff <= ABS_TOL,
            });
        }

        let methods = [
            ("auto", &scipy_arm.auto, auto),
            ("exact", &scipy_arm.exact, run(PageTrendMethod::Exact)),
            (
                "asymptotic",
                &scipy_arm.asymptotic,
                run(PageTrendMethod::Asymptotic),
            ),
        ];
        for (arm, scipy, fsci) in methods {
            let Some((s, f)) = ledger.pair(
                arm,
                &case.case_id,
                scipy.pvalue,
                fsci.as_ref().map(|r| r.pvalue),
            ) else {
                continue;
            };
            let diff = if f == s { 0.0 } else { (f - s).abs() / s.abs() };
            let same_method =
                fsci.as_ref().map(|r| method_name(r.method)) == scipy.method.as_deref();
            let pass = diff <= P_REL_TOL && same_method;
            max_p = max_p.max(diff);
            ledger.compared(arm, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: arm.into(),
                diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_page_trend".into(),
        category: "scipy.stats.page_trend_test".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_statistic_abs_diff: max_stat,
        max_pvalue_rel_diff: max_p,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "page_trend mismatch: {} arm={} diff={}",
                d.case_id, d.arm, d.diff
            );
        }
    }

    assert!(
        all_pass,
        "page_trend conformance failed: {} cases, max statistic diff {max_stat}, max p rel diff \
         {max_p}",
        diffs.len()
    );
    ledger.finish(query.points.len());
}
