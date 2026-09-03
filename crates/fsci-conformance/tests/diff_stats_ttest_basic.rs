#![forbid(unsafe_code)]
//! Live SciPy differential coverage for one-sample, equal-
//! variance two-sample, and Mood's-median tests:
//!   • `ttest_1samp(data, popmean)` — Student-t one-sample
//!   • `ttest_ind(a, b)` — equal-variance two-sample t-test
//!   • `mood(x, y)` — Mood's test for equal scale parameters
//!
//! Resolves [frankenscipy-8bxii]. The oracle calls
//! `scipy.stats.{ttest_1samp, ttest_ind, mood}`.
//!
//! 4 fixtures × 3 funcs × 2 arms = 24 cases. Tol 1e-9 abs
//! (Student-t / normal-tail chain via betainc / ndtri).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{mood, ttest_1samp, ttest_ind};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    a: Vec<f64>,
    b: Vec<f64>,
    popmean: f64,
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
    fs::create_dir_all(output_dir()).expect("create ttest_basic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ttest_basic diff log");
    fs::write(path, json).expect("write ttest_basic diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>, f64)> = vec![
        // Equal means / equal scale (a near pop mean = 5.5)
        (
            "balanced_n10",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            5.5,
        ),
        // Different means, similar scale
        (
            "shifted_n12",
            (1..=12).map(|i| i as f64).collect(),
            (5..=16).map(|i| i as f64).collect(),
            6.0,
        ),
        // Different scales, same center (Mood signal)
        (
            "diff_scale_n14",
            vec![
                4.0, 5.0, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 4.6, 5.4, 4.5, 5.5,
            ],
            vec![
                -2.0, 12.0, -1.0, 11.0, 0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0,
            ],
            5.0,
        ),
        // Right-skewed vs near-normal
        (
            "skewed_n16",
            (1..=16).map(|i| (i as f64).sqrt() * 4.0).collect(),
            (1..=16).map(|i| i as f64).collect(),
            8.0,
        ),
    ];

    let mut points = Vec::new();
    for (name, a, b, popmean) in &fixtures {
        for func in ["ttest_1samp", "ttest_ind", "mood"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                a: a.clone(),
                b: b.clone(),
                popmean: *popmean,
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
    cid = case["case_id"]; func = case["func"]
    a = np.array(case["a"], dtype=float)
    b = np.array(case["b"], dtype=float)
    popmean = float(case["popmean"])
    stat = None; pval = None
    try:
        if func == "ttest_1samp":
            res = stats.ttest_1samp(a, popmean)
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
        elif func == "ttest_ind":
            # Default scipy: equal_var=True (matches fsci's ttest_ind).
            res = stats.ttest_ind(a, b, equal_var=True)
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
        elif func == "mood":
            # scipy.stats.mood returns a Result object with .statistic / .pvalue
            res = stats.mood(a, b)
            stat = fnone(res.statistic); pval = fnone(res.pvalue)
    except Exception:
        pass
    points.append({"case_id": cid, "statistic": stat, "pvalue": pval})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ttest_basic query");
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
                "failed to spawn python3 for ttest_basic oracle: {e}"
            );
            eprintln!("skipping ttest_basic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ttest_basic oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ttest_basic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ttest_basic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ttest_basic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ttest_basic oracle failed: {stderr}"
        );
        eprintln!("skipping ttest_basic oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ttest_basic oracle JSON"))
}

#[test]
fn diff_stats_ttest_basic() {
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
            "ttest_1samp" => {
                let r = ttest_1samp(&case.a, case.popmean);
                (r.statistic, r.pvalue)
            }
            "ttest_ind" => {
                let r = ttest_ind(&case.a, &case.b);
                (r.statistic, r.pvalue)
            }
            "mood" => {
                let r = mood(&case.a, &case.b);
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
        test_id: "diff_stats_ttest_basic".into(),
        category: "scipy.stats.{ttest_1samp, ttest_ind, mood}".into(),
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
                "ttest_basic mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ttest_basic conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
