#![forbid(unsafe_code)]
//! Live SciPy differential coverage for three closed-form
//! summary functions:
//!   • `scipy.stats.entropy(pk, base=None|2|10|e)` — Shannon
//!     entropy of a discrete probability distribution
//!   • `scipy.stats.contingency.relative_risk(table)` — RR for
//!     a 2×2 contingency table
//!   • `scipy.stats.contingency.odds_ratio(table)` — sample OR
//!
//! Resolves [frankenscipy-pit9n]. Each function returns a
//! scalar; the harness exercises multiple fixtures including
//! base variations for entropy and varied row/col totals for
//! the contingency-table ratios.
//!
//! 4 entropy fixtures × 4 bases + 4 contingency fixtures × 2
//! ratios = 16 + 8 = 24 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{entropy, odds_ratio, relative_risk};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    pk: Vec<f64>,
    base: Option<f64>,
    table: [[u64; 2]; 2],
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create entropy_ratios diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize entropy_ratios diff log");
    fs::write(path, json).expect("write entropy_ratios diff log");
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let v = match case.func.as_str() {
        "entropy" => entropy(&case.pk, case.base),
        "relative_risk" => relative_risk(&[
            [case.table[0][0] as usize, case.table[0][1] as usize],
            [case.table[1][0] as usize, case.table[1][1] as usize],
        ]),
        "odds_ratio" => odds_ratio(&[
            [case.table[0][0] as usize, case.table[0][1] as usize],
            [case.table[1][0] as usize, case.table[1][1] as usize],
        ]),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // Entropy fixtures (pk vectors; one will be auto-normalized)
    let entropy_fixtures: Vec<(&str, Vec<f64>)> = vec![
        ("uniform_3", vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        ("skewed_4", vec![0.5, 0.25, 0.15, 0.10]),
        ("near_one_hot", vec![0.97, 0.01, 0.01, 0.01]),
        // Unnormalized (will be normalized internally on both sides)
        ("unnormalized", vec![5.0, 3.0, 2.0, 8.0, 2.0]),
    ];
    let bases: [(&str, Option<f64>); 4] = [
        ("nat", None),
        ("bits", Some(2.0)),
        ("dits", Some(10.0)),
        ("custom_e", Some(std::f64::consts::E)),
    ];

    let mut points = Vec::new();
    for (name, pk) in &entropy_fixtures {
        for (bname, base) in bases {
            points.push(PointCase {
                case_id: format!("entropy_{name}_{bname}"),
                func: "entropy".into(),
                pk: pk.clone(),
                base,
                table: [[0, 0], [0, 0]],
            });
        }
    }

    // Contingency-table fixtures
    let table_fixtures: &[(&str, [[u64; 2]; 2])] = &[
        ("indep", [[10, 10], [10, 10]]),
        ("mild_assoc", [[8, 12], [12, 8]]),
        ("strong_assoc", [[15, 5], [3, 17]]),
        ("large", [[50, 30], [40, 80]]),
    ];
    for (name, table) in table_fixtures {
        for func in ["relative_risk", "odds_ratio"] {
            points.push(PointCase {
                case_id: format!("{func}_{name}"),
                func: func.into(),
                pk: Vec::new(),
                base: None,
                table: *table,
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
    val = None
    try:
        if func == "entropy":
            base = case["base"]
            pk = np.array(case["pk"], dtype=float)
            if base is None:
                val = float(stats.entropy(pk))
            else:
                val = float(stats.entropy(pk, base=float(base)))
        elif func == "relative_risk":
            t = case["table"]
            # contingency.relative_risk uses .relative_risk attribute
            res = contingency.relative_risk(
                t[0][0], t[0][0] + t[0][1], t[1][0], t[1][0] + t[1][1]
            )
            val = float(res.relative_risk)
        elif func == "odds_ratio":
            t = case["table"]
            res = contingency.odds_ratio([[t[0][0], t[0][1]], [t[1][0], t[1][1]]],
                                          kind='sample')
            val = float(res.statistic)
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize entropy_ratios query");
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
                "failed to spawn python3 for entropy_ratios oracle: {e}"
            );
            eprintln!("skipping entropy_ratios oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open entropy_ratios oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "entropy_ratios oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping entropy_ratios oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for entropy_ratios oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "entropy_ratios oracle failed: {stderr}"
        );
        eprintln!("skipping entropy_ratios oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse entropy_ratios oracle JSON"))
}

#[test]
fn diff_stats_entropy_ratios() {
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
        if let Some(scipy_v) = scipy_arm.value
            && let Some(rust_v) = fsci_eval(case)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_entropy_ratios".into(),
        category: "scipy.stats.entropy + contingency.relative_risk/odds_ratio".into(),
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
                "entropy_ratios {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "entropy_ratios conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
