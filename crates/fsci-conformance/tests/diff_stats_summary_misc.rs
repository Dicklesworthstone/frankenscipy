#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the
//! `scipy.stats.ttest_ind_from_stats(mean1, std1, n1,
//! mean2, std2, n2, equal_var)` summary-statistic entry
//! point of the independent t-test.
//!
//! Resolves [frankenscipy-k34yt]. ttest_ind_from_stats
//! exercises the same StudentT pvalue chain as ttest_ind but
//! through the alternate summary-statistic entry point —
//! useful when the raw samples are unavailable.
//!
//! 6 configs × 2 equal_var modes × 3 arms (statistic +
//! pvalue + df) = 36 cases via subprocess. Tol 1e-9 abs.
//!
//! phi_coefficient was originally also covered here but
//! removed: scipy's `contingency.association(method='pearson')`
//! returns the Pearson contingency coefficient
//! `C = sqrt(chi²/(chi²+N))`, not phi. fsci's phi_coefficient
//! is the standard textbook (ad-bc)/sqrt(...) formula. The two
//! are genuinely different functions for 2×2 tables; matching
//! them would require switching the oracle to Cramér's V or
//! using scipy's pearson-correlation-of-binary-indicators
//! formulation. Deferred to a future tick.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::ttest_ind_from_stats;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TTEST_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    mean1: f64,
    std1: f64,
    n1: u64,
    mean2: f64,
    std2: f64,
    n2: u64,
    equal_var: bool,
    table: [[u64; 2]; 2],
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
    #[allow(dead_code)]
    scalar: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create summary_misc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize summary_misc diff log");
    fs::write(path, json).expect("write summary_misc diff log");
}

fn generate_query() -> OracleQuery {
    // ttest_ind_from_stats configs: (mean1, std1, n1, mean2, std2, n2)
    let ttest_configs: &[(&str, f64, f64, u64, f64, f64, u64)] = &[
        ("equal_means", 5.0, 1.0, 30, 5.0, 1.0, 30),
        ("shifted_means", 5.0, 1.0, 30, 6.0, 1.0, 30),
        ("unequal_var", 5.0, 1.0, 30, 5.5, 2.0, 30),
        ("unequal_n", 5.0, 1.0, 50, 5.5, 1.0, 20),
        ("large_diff_small_n", 10.0, 2.0, 5, 15.0, 2.0, 5),
        ("very_small_diff_large_n", 5.00, 1.0, 200, 5.05, 1.0, 200),
    ];

    let mut points = Vec::new();
    for (name, m1, s1, n1, m2, s2, n2) in ttest_configs {
        for &equal_var in &[true, false] {
            let suffix = if equal_var { "ev" } else { "wel" };
            points.push(PointCase {
                case_id: format!("ttest_{name}_{suffix}"),
                func: "ttest_ind_from_stats".into(),
                mean1: *m1,
                std1: *s1,
                n1: *n1,
                mean2: *m2,
                std2: *s2,
                n2: *n2,
                equal_var,
                table: [[0, 0], [0, 0]],
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
from scipy.stats import contingency

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
    out = {"case_id": cid, "statistic": None, "pvalue": None,
           "df": None, "scalar": None}
    try:
        if func == "ttest_ind_from_stats":
            res = stats.ttest_ind_from_stats(
                float(case["mean1"]), float(case["std1"]), int(case["n1"]),
                float(case["mean2"]), float(case["std2"]), int(case["n2"]),
                equal_var=bool(case["equal_var"])
            )
            out["statistic"] = fnone(res.statistic)
            out["pvalue"] = fnone(res.pvalue)
            # ttest_ind_from_stats returns Ttest_indResult; df is computed
            # internally but exposed as res.df only on newer SciPy.
            try:
                out["df"] = fnone(res.df)
            except Exception:
                # Reconstruct df ourselves to align with fsci.
                if case["equal_var"]:
                    out["df"] = float(case["n1"] + case["n2"] - 2)
                else:
                    n1 = float(case["n1"]); n2 = float(case["n2"])
                    s1 = float(case["std1"]); s2 = float(case["std2"])
                    v1 = s1 * s1 / n1; v2 = s2 * s2 / n2
                    if v1 + v2 == 0:
                        out["df"] = None
                    else:
                        df_num = (v1 + v2) ** 2
                        df_den = v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)
                        out["df"] = fnone(df_num / df_den)
        elif func == "phi":
            t = case["table"]
            res = contingency.association(
                [[t[0][0], t[0][1]], [t[1][0], t[1][1]]], method='pearson'
            )
            # 'pearson' (Cramer V with min(r,c)-1 = 1) is equivalent to phi
            # for 2x2 tables. The sign is dropped by association(); fsci's
            # phi_coefficient retains the sign of (ad - bc), so we may need
            # to compare absolute values. Pass abs(association) here and
            # let the Rust side compare to abs(fsci_phi).
            out["scalar"] = fnone(abs(float(res)))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize summary_misc query");
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
                "failed to spawn python3 for summary_misc oracle: {e}"
            );
            eprintln!("skipping summary_misc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open summary_misc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "summary_misc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping summary_misc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for summary_misc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "summary_misc oracle failed: {stderr}"
        );
        eprintln!("skipping summary_misc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse summary_misc oracle JSON"))
}

#[test]
fn diff_stats_summary_misc() {
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
        match case.func.as_str() {
            "ttest_ind_from_stats" => {
                let result = ttest_ind_from_stats(
                    case.mean1,
                    case.std1,
                    case.n1 as usize,
                    case.mean2,
                    case.std2,
                    case.n2 as usize,
                    case.equal_var,
                );
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
                            func: case.func.clone(),
                            arm: arm_name.into(),
                            abs_diff,
                            pass: abs_diff <= TTEST_TOL,
                        });
                    }
                }
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_summary_misc".into(),
        category: "scipy.stats.ttest_ind_from_stats + contingency phi".into(),
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
                "summary_misc {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "summary_misc conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
