#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the two-sample
//! Kolmogorov-Smirnov test `scipy.stats.ks_2samp` at its defaults
//! (`alternative='two-sided'`, `method='auto'`).
//!
//! Resolves [frankenscipy-c4ke3]; widened by [frankenscipy-qwa3t]. Cross-checks the D
//! statistic (max |F1(x) - F2(x)|, snapped to `h / lcm(n1, n2)` where SciPy snaps it) and the
//! p-value through every branch of SciPy's 'auto':
//!   - exact, equal sizes: the reflection series (identical, shifted, ties, n = 30);
//!   - exact, equal sizes, series out of [0, 1]: n = 60 with h = 2, where SciPy falls back to
//!     `kstwo.sf(d, round(en))`;
//!   - exact, unequal sizes: Viehmann's `1 − p` recursion, including three tail cases with
//!     p < 1e-19 that the `1 − inside` sweep it replaced returned as ~1e-16 or 0;
//!   - asymptotic, max(n1, n2) > 10000: `kstwo.sf(d, round(en))`, including n1 = n2 = 10001,
//!     where en = 5000.5 must round half to even (size 5000).
//!
//! Tolerances, from the measured agreement (every case bit-identical on scipy 1.17.1,
//! numpy 2.4.3, glibc 2.43): statistic exactly equal; p-value 1e-13 relative, headroom over
//! bit-identity for a libm that rounds a last bit differently. The previous 1e-7 absolute
//! could not see a wrong tail at all: 2.2e-16 against SciPy's 4.0e-35 passes it.
//!
//! MUST-HIT control: at least `MIN_TAIL_CASES` fixtures have a SciPy p-value below 1e-15,
//! where `1 − inside` cannot represent the answer; without them the relative tolerance has
//! nothing to catch.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::ks_2samp;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// The statistic is compared for equality; it has no tolerance const because the tolerance lint
// rejects a zero one.
const PVALUE_REL_TOL: f64 = 1.0e-13;
/// Fixtures whose SciPy p-value is below 1e-15 (the unequal-size tail cases).
const MIN_TAIL_CASES: usize = 3;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data1: Vec<f64>,
    data2: Vec<f64>,
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
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    max_pvalue_rel_diff: f64,
    tail_cases: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create ks_2samp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ks_2samp diff log");
    fs::write(path, json).expect("write ks_2samp diff log");
}

/// `i/n + shift` for i in 0..n: exactly representable arithmetic, so both sides see the same
/// samples.
fn grid(n: usize, shift: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 / n as f64 + shift).collect()
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Identical samples — D should be small, pvalue large.
        (
            "identical",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        // Mean-shifted — D moderate, pvalue small.
        (
            "mean_shift",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ),
        // Scale-shifted (different std) — D moderate.
        (
            "scale_shift",
            vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ),
        // Small-sample asymmetric — exact, unequal sizes.
        ("small_asym", vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]),
        // n = 30 each — still exact (SciPy's cap is max(n1, n2) <= 10000).
        (
            "larger",
            (0..30).map(|i| i as f64 / 10.0).collect(),
            (0..30).map(|i| 0.3 + i as f64 / 10.0).collect(),
        ),
        (
            "ties",
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            vec![2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        ),
        ("coprime_7_11", grid(7, 0.0), grid(11, 0.2)),
        // Equal sizes, h = 2: the series returns 1.0000000000000002, SciPy rejects it and
        // evaluates kstwo.sf(1/30, 30).
        (
            "n60_series_out_of_range",
            (0..60).map(f64::from).collect(),
            (2..62).map(f64::from).collect(),
        ),
        // Unequal-size tails (Viehmann recursion).
        ("tail_1000_37", grid(1000, 0.0), grid(37, 0.9)),
        ("tail_400_600", grid(400, 0.0), grid(600, 0.3)),
        ("tail_50_120", grid(50, 0.0), grid(120, 0.8)),
        ("unequal_600_700", grid(600, 0.0), grid(700, 0.04)),
        // max(n1, n2) > 10000: 'asymp', kstwo.sf(d, round(en)).
        ("asymp_20000_15000", grid(20_000, 0.0), grid(15_000, 0.012)),
        ("asymp_10001_10001", grid(10_001, 0.0), grid(10_001, 0.015)),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data1, data2)| PointCase {
            case_id: name.into(),
            data1,
            data2,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import warnings
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
    data1 = np.array(case["data1"], dtype=float)
    data2 = np.array(case["data2"], dtype=float)
    try:
        with warnings.catch_warnings():
            # 'auto' warns when its exact attempt fails and it switches to 'asymp'.
            warnings.simplefilter("ignore", RuntimeWarning)
            res = stats.ks_2samp(data1, data2, alternative='two-sided', method='auto')
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ks_2samp query");
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
                "failed to spawn python3 for ks_2samp oracle: {e}"
            );
            eprintln!("skipping ks_2samp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ks_2samp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ks_2samp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ks_2samp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ks_2samp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ks_2samp oracle failed: {stderr}"
        );
        eprintln!("skipping ks_2samp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ks_2samp oracle JSON"))
}

#[test]
fn diff_stats_ks_2samp() {
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
    let mut max_pvalue_rel = 0.0_f64;
    let mut tail_cases = 0_usize;
    let mut ledger = CompareLedger::new("diff_stats_ks_2samp", &["statistic", "pvalue"]);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let result = ks_2samp(&case.data1, &case.data2);
        if scipy_arm.pvalue.is_some_and(|p| p < 1e-15) {
            tail_cases += 1;
        }
        let arms = [
            ("statistic", scipy_arm.statistic, result.statistic),
            ("pvalue", scipy_arm.pvalue, result.pvalue),
        ];
        for (arm, scipy, fsci) in arms {
            let Some((s, f)) = ledger.pair(arm, &case.case_id, scipy, Some(fsci)) else {
                continue;
            };
            let abs_diff = (f - s).abs();
            let rel_diff = relative_diff(f, s);
            let pass = if arm == "pvalue" {
                max_pvalue_rel = max_pvalue_rel.max(rel_diff);
                rel_diff <= PVALUE_REL_TOL
            } else {
                f == s
            };
            max_overall = max_overall.max(abs_diff);
            ledger.compared(arm, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: arm.into(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_ks_2samp".into(),
        category: "scipy.stats.ks_2samp(method='auto')".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        max_pvalue_rel_diff: max_pvalue_rel,
        tail_cases,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "ks_2samp mismatch: {} arm={} abs={} rel={}",
                d.case_id, d.arm, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "ks_2samp conformance failed: {} cases, max_abs={}, max pvalue rel={}",
        diffs.len(),
        max_overall,
        max_pvalue_rel
    );
    assert!(
        tail_cases >= MIN_TAIL_CASES,
        "only {tail_cases} fixtures have a SciPy p-value below 1e-15 (need {MIN_TAIL_CASES}); \
         the relative p-value tolerance no longer has a tail to check"
    );
    ledger.finish(query.points.len());
}

/// |f − s| / |s|, with a zero SciPy value matched only exactly.
fn relative_diff(f: f64, s: f64) -> f64 {
    if f == s {
        0.0
    } else if s == 0.0 {
        f64::INFINITY
    } else {
        (f - s).abs() / s.abs()
    }
}
